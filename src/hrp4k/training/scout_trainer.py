"""PyTorch Region Scout Training and Validation Pipeline for AdaPoth-Lite.

Implements:
1. Scout Dataset loader with on-the-fly Elliptical Gaussian heatmap generation.
2. Training loop with ScoutLoss (Focal Loss + lambda_cov * Coverage Loss).
3. Validation pipeline prioritizing Region Recall (>= 97%) for checkpoint selection.
4. Cloud synchronization to Hugging Face Hub.
5. Standalone Scout model evaluation utility.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..data.coco import load_split
from ..data.paths import image_path
from ..infra.environment import environment_snapshot
from ..infra.upload import BackgroundHFSyncer, ensure_weights
from ..models.scout import (
    CandidateGenerator,
    MobileNetV3Scout,
    ScoutLoss,
    evaluate_scout_regions,
    generate_scout_heatmap_gt,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    TORCH_AVAILABLE = False


class ScoutDataset(Dataset):
    """Dataset for Scout heatmap training."""
    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        img_size: tuple[int, int] = (540, 960),  # (H, W)
        heat_size: tuple[int, int] = (34, 60),  # (H, W)
        limit: int | None = None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_h, self.img_w = img_size
        self.heat_h, self.heat_w = heat_size
        self.augment = augment

        coco = load_split(self.data_dir, split)
        self.images = [
            im for im in coco.get("images", [])
            if image_path(self.data_dir, split, im["file_name"]).is_file()
        ]
        if limit is not None:
            self.images = self.images[:limit]

        # Index annotations by image_id
        self.img_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            img_id = int(ann["image_id"])
            self.img_to_anns.setdefault(img_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        im_info = self.images[idx]
        img_id = int(im_info["id"])
        file_path = image_path(self.data_dir, self.split, im_info["file_name"])

        # Load image
        import cv2
        img = cv2.imread(str(file_path))
        if img is None:
            img = np.zeros((2160, 3840, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]
        anns = self.img_to_anns.get(img_id, [])
        boxes = [list(map(float, a["bbox"])) for a in anns]

        # Horizontal flip augmentation
        if self.augment and random.random() < 0.5:
            img = cv2.flip(img, 1)
            flipped_boxes = []
            for x, y, w, h in boxes:
                flipped_boxes.append([orig_w - (x + w), y, w, h])
            boxes = flipped_boxes

        # Resize image to scout input size (960x540)
        img_resized = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        # Normalize with standard ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Generate Ground-Truth heatmap (60x34)
        gt_heatmap = generate_scout_heatmap_gt(
            boxes,
            img_w=orig_w,
            img_h=orig_h,
            heat_w=self.heat_w,
            heat_h=self.heat_h,
            sigma_x_scale=0.35,
            sigma_y_scale=0.50,
            expand_ratio=0.25,
        )
        gt_tensor = torch.from_numpy(gt_heatmap).unsqueeze(0).float()  # (1, heat_H, heat_W)

        # Collect GT centers in heatmap coordinates for coverage loss
        gt_centers = []
        for x, y, w, h in boxes:
            cx = (x + w * 0.5) * (self.heat_w / float(orig_w))
            cy = (y + h * 0.5) * (self.heat_h / float(orig_h))
            gt_centers.append((float(cy), float(cx)))

        return {
            "image": img_tensor,
            "heatmap": gt_tensor,
            "image_id": img_id,
            "orig_boxes": boxes,
            "orig_size": (orig_w, orig_h),
            "gt_centers": gt_centers,
        }


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    heatmaps = torch.stack([b["heatmap"] for b in batch], dim=0)
    image_ids = [b["image_id"] for b in batch]
    orig_boxes = [b["orig_boxes"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    gt_centers = [b["gt_centers"] for b in batch]
    return {
        "image": images,
        "heatmap": heatmaps,
        "image_ids": image_ids,
        "orig_boxes": orig_boxes,
        "orig_sizes": orig_sizes,
        "gt_centers": gt_centers,
    }


def train_scout(
    data_dir: Path | str,
    output_dir: Path | str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    img_size: tuple[int, int] = (540, 960),
    lambda_cov: float = 2.0,
    device: str | None = None,
    smoke: bool = False,
    resume: bool = False,
    seed: int = 42,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
) -> dict[str, Any]:
    """Train MobileNetV3-Small Region Scout model on HRP4K dataset."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training the Scout model")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device is None or device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"\n[Scout Training] Launching on device: {dev}, epochs: {1 if smoke else epochs}, batch: {batch_size}")

    # Initialize Model, Loss, Optimizer
    model = MobileNetV3Scout().to(dev)
    with torch.no_grad():
        dummy_out = model(torch.zeros(1, 3, img_size[0], img_size[1], device=dev))
        heat_size = (dummy_out.shape[2], dummy_out.shape[3])

    # Initialize Datasets
    train_limit = 4 if smoke else None
    val_limit = 2 if smoke else None
    actual_epochs = 1 if smoke else epochs

    train_ds = ScoutDataset(data_dir, split="train", img_size=img_size, heat_size=heat_size, limit=train_limit, augment=True)
    val_ds = ScoutDataset(data_dir, split="valid", img_size=img_size, heat_size=heat_size, limit=val_limit, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size if not smoke else 2, shuffle=True, collate_fn=_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size if not smoke else 2, shuffle=False, collate_fn=_collate_fn, num_workers=0)

    criterion = ScoutLoss(lambda_cov=lambda_cov)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_epochs, eta_min=1e-5)

    start_epoch = 0
    best_recall = 0.0
    best_loss = float("inf")

    last_ckpt = weights_dir / "scout_last.pt"
    best_ckpt = weights_dir / "scout_best.pt"

    if resume and last_ckpt.is_file():
        ckpt = torch.load(str(last_ckpt), map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_recall = ckpt.get("best_recall", 0.0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[Resume] Loaded Scout checkpoint from Epoch {start_epoch}, best recall: {best_recall:.4f}")

    # Cloud Syncer
    syncer = BackgroundHFSyncer(repo_id=hf_repo, token=hf_token, path_in_repo=output_dir.name, enabled=hf_sync and not smoke)

    candidate_gen = CandidateGenerator(threshold=0.30, context_margin=0.20, k_max=4)
    history = []

    for epoch in range(start_epoch, actual_epochs):
        model.train()
        train_loss_list = []
        focal_loss_list = []
        cov_loss_list = []

        t0 = time.time()
        for batch in train_loader:
            imgs = batch["image"].to(dev)
            targets = batch["heatmap"].to(dev)
            gt_centers = batch["gt_centers"]

            optimizer.zero_grad()
            preds = model(imgs)
            loss_dict = criterion(preds, targets, gt_centers)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            train_loss_list.append(float(loss.item()))
            focal_loss_list.append(float(loss_dict["focal_loss"].item()))
            cov_loss_list.append(float(loss_dict["coverage_loss"].item()))

        scheduler.step()
        epoch_time = time.time() - t0

        # Validation phase: Evaluate Region Recall
        model.eval()
        val_loss_list = []
        val_recalls = []
        val_coverages = []
        val_false_rates = []
        val_k_crops = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev)
                targets = batch["heatmap"].to(dev)
                gt_centers = batch["gt_centers"]

                preds = model(imgs)
                loss_dict = criterion(preds, targets, gt_centers)
                val_loss_list.append(float(loss_dict["loss"].item()))

                # Evaluate candidate regions for each sample
                pred_np = preds.cpu().numpy()
                for i in range(len(batch["image_ids"])):
                    hmap = pred_np[i, 0]
                    orig_w, orig_h = batch["orig_sizes"][i]
                    orig_boxes = batch["orig_boxes"][i]

                    candidates = candidate_gen.generate(hmap, source_width=orig_w, source_height=orig_h)
                    res = evaluate_scout_regions(orig_boxes, candidates)
                    val_recalls.append(res["region_recall"])
                    val_coverages.append(res["gt_coverage_ratio"])
                    val_false_rates.append(res["false_region_rate"])
                    val_k_crops.append(res["k_crops"])

        mean_train_loss = float(np.mean(train_loss_list)) if train_loss_list else 0.0
        mean_val_loss = float(np.mean(val_loss_list)) if val_loss_list else 0.0
        mean_recall = float(np.mean(val_recalls)) if val_recalls else 0.0
        mean_cov = float(np.mean(val_coverages)) if val_coverages else 0.0
        mean_false_rate = float(np.mean(val_false_rates)) if val_false_rates else 0.0
        mean_k = float(np.mean(val_k_crops)) if val_k_crops else 0.0

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "region_recall": mean_recall,
            "gt_coverage": mean_cov,
            "false_region_rate": mean_false_rate,
            "avg_k": mean_k,
            "epoch_time_s": epoch_time,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch+1}/{actual_epochs}] - "
            f"Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f} | "
            f"🎯 Region Recall: {mean_recall*100:.2f}%, GT Cov: {mean_cov*100:.2f}%, "
            f"False Rate: {mean_false_rate*100:.1f}%, Avg K: {mean_k:.2f} ({epoch_time:.1f}s)"
        )

        # Save last checkpoint
        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_recall": best_recall,
            "best_loss": best_loss,
            "metrics": epoch_record,
        }
        torch.save(ckpt_state, str(last_ckpt))

        # Checkpoint Selection: Prioritize Region Recall >= 97% first
        is_best = False
        if mean_recall > best_recall:
            is_best = True
            best_recall = mean_recall
        elif mean_recall >= 0.97 and mean_val_loss < best_loss:
            is_best = True
            best_loss = mean_val_loss

        if is_best or epoch == 0:
            torch.save(ckpt_state, str(best_ckpt))
            print(f"  ⭐ Saved new best Scout checkpoint: Region Recall = {mean_recall*100:.2f}%")

        if syncer.enabled:
            syncer.sync_epoch(
                epoch=epoch + 1,
                weights_dir=weights_dir,
                extra_files=[output_dir / "metrics.json"],
                path_in_repo=output_dir.name,
            )

    # Save metrics JSON and configuration
    final_payload = {
        "model_name": "MobileNetV3Scout",
        "parameters": model.count_parameters(),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "best_region_recall": best_recall,
        "final_epoch": actual_epochs,
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": list(img_size),
            "heat_size": list(heat_size),
            "lambda_cov": lambda_cov,
            "seed": seed,
        },
        "environment": environment_snapshot(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    if syncer.enabled:
        syncer.sync_epoch(
            epoch=actual_epochs,
            weights_dir=weights_dir,
            extra_files=[output_dir / "metrics.json"],
            path_in_repo=output_dir.name,
        )
        syncer.wait_until_done(timeout=30.0)
        syncer.shutdown(wait=True)

    return final_payload


def evaluate_scout_model(
    data_dir: Path | str,
    weights_path: Path | str,
    split: str = "valid",
    output_path: Path | str | None = None,
    threshold: float = 0.30,
    context_margin: float = 0.20,
    k_max: int = 4,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate Scout Model candidate generation quality on specified split."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate Scout model")

    data_dir = Path(data_dir)
    weights_path = ensure_weights(weights_path)
    
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MobileNetV3Scout().to(dev)

    if Path(weights_path).is_file():
        ckpt = torch.load(str(weights_path), map_location=dev, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    ds = ScoutDataset(data_dir, split=split, limit=limit, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    candidate_gen = CandidateGenerator(threshold=threshold, context_margin=context_margin, k_max=k_max)

    recalls = []
    coverages = []
    false_rates = []
    k_crops_list = []
    per_image_results = []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(dev)
            img_id = batch["image_ids"][0]
            orig_w, orig_h = batch["orig_sizes"][0]
            orig_boxes = batch["orig_boxes"][0]

            pred_heatmap = model(img).cpu().numpy()[0, 0]
            candidates = candidate_gen.generate(pred_heatmap, source_width=orig_w, source_height=orig_h)
            res = evaluate_scout_regions(orig_boxes, candidates)

            recalls.append(res["region_recall"])
            coverages.append(res["gt_coverage_ratio"])
            false_rates.append(res["false_region_rate"])
            k_crops_list.append(res["k_crops"])

            per_image_results.append({
                "image_id": img_id,
                "candidates": [c.xyxy for c in candidates],
                "scores": [c.score for c in candidates],
                **res,
            })

    summary = {
        "split": split,
        "weights": str(weights_path),
        "total_images": len(ds),
        "mean_region_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_gt_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_false_region_rate": float(np.mean(false_rates)) if false_rates else 0.0,
        "mean_k": float(np.mean(k_crops_list)) if k_crops_list else 0.0,
        "max_k": int(np.max(k_crops_list)) if k_crops_list else 0,
        "threshold": threshold,
        "context_margin": context_margin,
        "k_max": k_max,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({"summary": summary, "per_image": per_image_results}, indent=2), encoding="utf-8")

    return summary
