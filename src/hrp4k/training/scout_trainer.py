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


def _load_and_resize_img(file_path: Path | str, target_size: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Load image from path and resize to target_size (W, H). Returns (img_bgr, orig_h, orig_w)."""
    target_w, target_h = target_size
    try:
        import cv2
        raw = cv2.imread(str(file_path))
        if raw is not None:
            orig_h, orig_w = raw.shape[:2]
            resized = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized, orig_h, orig_w
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(file_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            orig_w, orig_h = pil_img.size
            resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            rgb_arr = np.array(resized)
            bgr_arr = rgb_arr[:, :, ::-1].copy()
            return bgr_arr, orig_h, orig_w
    except Exception:
        pass
    return np.zeros((target_h, target_w, 3), dtype=np.uint8), 2160, 3840


def ensure_scout_cache(
    data_dir: Path | str,
    split: str,
    img_size: tuple[int, int] = (540, 960),
    max_workers: int = 16,
) -> Path:
    """Pre-resizes dataset images to thumbnail resolution in parallel for 20x faster dataloading."""
    data_dir = Path(data_dir)
    cache_dir = data_dir / f".cache_scout_{img_size[1]}x{img_size[0]}" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    coco = load_split(data_dir, split)
    images = coco.get("images", [])

    missing = []
    for im in images:
        cf = cache_dir / Path(im["file_name"]).name
        if not cf.is_file():
            missing.append(im)

    if missing:
        from concurrent.futures import ThreadPoolExecutor

        def _process(im: dict[str, Any]) -> None:
            src = image_path(data_dir, split, im["file_name"])
            dst = cache_dir / Path(im["file_name"]).name
            if src.is_file() and not dst.is_file():
                try:
                    import cv2
                    img = cv2.imread(str(src))
                    if img is not None:
                        thumb = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(str(dst), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        return
                except Exception:
                    pass
                try:
                    from PIL import Image
                    with Image.open(src) as pimg:
                        pimg = pimg.convert("RGB")
                        thumb = pimg.resize((img_size[1], img_size[0]), Image.Resampling.BILINEAR)
                        thumb.save(dst, "JPEG", quality=90)
                except Exception:
                    pass

        print(f"[Scout Cache] Pre-caching {len(missing)} thumbnails for '{split}' ({max_workers} threads)...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_process, missing))
        print(f"[Scout Cache] Finished pre-caching for '{split}' at {cache_dir}!")
    return cache_dir


class ScoutDataset(Dataset):
    """Dataset for Scout heatmap training with thumbnail cache acceleration."""
    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        img_size: tuple[int, int] = (540, 960),  # (H, W)
        heat_size: tuple[int, int] = (34, 60),  # (H, W)
        limit: int | None = None,
        augment: bool = True,
        cache_dir: Path | str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_h, self.img_w = img_size
        self.heat_h, self.heat_w = heat_size
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None

        coco = load_split(self.data_dir, split)
        self.images = [
            im for im in coco.get("images", [])
            if (self.cache_dir and (self.cache_dir / Path(im["file_name"]).name).is_file())
            or image_path(self.data_dir, split, im["file_name"]).is_file()
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
        file_name = Path(im_info["file_name"]).name
        orig_w = int(im_info.get("width", 3840))
        orig_h = int(im_info.get("height", 2160))

        img = None
        if self.cache_dir:
            cache_p = self.cache_dir / file_name
            if cache_p.is_file():
                img, _, _ = _load_and_resize_img(cache_p, (self.img_w, self.img_h))

        if img is None:
            file_path = image_path(self.data_dir, self.split, im_info["file_name"])
            img, orig_h, orig_w = _load_and_resize_img(file_path, (self.img_w, self.img_h))

        anns = self.img_to_anns.get(img_id, [])
        boxes = [list(map(float, a["bbox"])) for a in anns]

        # Horizontal flip augmentation
        if self.augment and random.random() < 0.5:
            img = np.fliplr(img).copy()
            flipped_boxes = []
            for x, y, w, h in boxes:
                flipped_boxes.append([orig_w - (x + w), y, w, h])
            boxes = flipped_boxes

        # Convert to tensor and normalize (3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

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
    batch_size: int = 32,
    lr: float = 1e-3,
    img_size: tuple[int, int] = (540, 960),
    lambda_cov: float = 3.0,
    device: str | None = None,
    smoke: bool = False,
    resume: bool = False,
    seed: int = 42,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    num_workers: int | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Train MobileNetV3-Small Region Scout model on HRP4K dataset with AMP and multi-worker caching."""
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
    elif str(device).isdigit():
        dev = torch.device(f"cuda:{device}")
    else:
        dev = torch.device(device)

    print(f"\n[Scout Training] Launching on device: {dev}, epochs: {1 if smoke else epochs}, batch: {batch_size}")

    # Initialize Model, Loss, Optimizer
    model = MobileNetV3Scout().to(dev)
    with torch.no_grad():
        dummy_out = model(torch.zeros(1, 3, img_size[0], img_size[1], device=dev))
        heat_size = (dummy_out.shape[2], dummy_out.shape[3])

    # Pre-cache thumbnails if enabled and not smoke
    train_cache = None
    val_cache = None
    if use_cache and not smoke:
        try:
            train_cache = ensure_scout_cache(data_dir, "train", img_size=img_size)
            val_cache = ensure_scout_cache(data_dir, "valid", img_size=img_size)
        except Exception as exc:
            print(f"[Scout Cache Warning] Pre-caching failed ({exc}), falling back to direct load")

    # Initialize Datasets
    train_limit = 4 if smoke else None
    val_limit = 2 if smoke else None
    actual_epochs = 1 if smoke else epochs

    train_ds = ScoutDataset(data_dir, split="train", img_size=img_size, heat_size=heat_size, limit=train_limit, augment=True, cache_dir=train_cache)
    val_ds = ScoutDataset(data_dir, split="valid", img_size=img_size, heat_size=heat_size, limit=val_limit, augment=False, cache_dir=val_cache)

    actual_workers = 0 if smoke else (num_workers if num_workers is not None else (8 if dev.type == "cuda" else 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size if not smoke else 2,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=actual_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(actual_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size if not smoke else 2,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=actual_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(actual_workers > 0),
    )

    criterion = ScoutLoss(lambda_cov=lambda_cov)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = min(3, max(1, actual_epochs // 10))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, actual_epochs - warmup_epochs), eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))

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
    target_repo_path = f"checkpoints/{output_dir.name}"
    syncer = BackgroundHFSyncer(repo_id=hf_repo, token=hf_token, path_in_repo=target_repo_path, enabled=hf_sync and not smoke)

    candidate_gen = CandidateGenerator(threshold=0.05, context_margin=0.30, k_max=4)
    history = []

    for epoch in range(start_epoch, actual_epochs):
        model.train()
        train_loss_list = []
        focal_loss_list = []
        cov_loss_list = []

        t0 = time.time()
        for batch in train_loader:
            imgs = batch["image"].to(dev, non_blocking=True)
            targets = batch["heatmap"].to(dev, non_blocking=True)
            gt_centers = batch["gt_centers"]
            orig_boxes = batch["orig_boxes"]
            orig_sizes = batch["orig_sizes"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                preds = model(imgs)
            
            # Compute loss in FP32 for numerical stability (prevent NaN/overflow)
            loss_dict = criterion(preds.float(), targets.float(), gt_boxes=orig_boxes, orig_sizes=orig_sizes, gt_centers=gt_centers)
            loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_list.append(float(loss.item()))
            focal_loss_list.append(float(loss_dict["focal_loss"].item()))
            cov_loss_list.append(float(loss_dict["coverage_loss"].item()))

        scheduler.step()
        epoch_time = time.time() - t0

        # Validation Phase
        model.eval()
        val_loss_list = []
        val_recalls = []
        val_coverages = []
        val_false_rates = []
        val_k_crops = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev, non_blocking=True)
                targets = batch["heatmap"].to(dev, non_blocking=True)
                gt_centers = batch["gt_centers"]
                orig_boxes = batch["orig_boxes"]
                orig_sizes = batch["orig_sizes"]

                with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                    preds = model(imgs)
                
                loss_dict = criterion(preds.float(), targets.float(), gt_boxes=orig_boxes, orig_sizes=orig_sizes, gt_centers=gt_centers)
                val_loss_list.append(float(loss_dict["loss"].item()))

                # Evaluate candidate regions for each sample
                pred_np = preds.float().cpu().numpy()
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
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_s": epoch_time,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch+1}/{actual_epochs}] - "
            f"Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f} | "
            f"🎯 Region Recall: {mean_recall*100:.2f}%, GT Cov: {mean_cov*100:.2f}%, "
            f"False Rate: {mean_false_rate*100:.1f}%, Avg K: {mean_k:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f} ({epoch_time:.1f}s)"
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

        # Checkpoint Selection: Prioritize Region Recall
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
                path_in_repo=f"checkpoints/{output_dir.name}",
            )

    # Threshold & K Sweep Evaluation (if not smoke)
    sweep_results = None
    if not smoke and best_ckpt.is_file():
        try:
            print("\n[Scout Evaluation] Running Threshold & K-Sweep on Valid and Test splits...")
            sweep_results = sweep_scout_eval(
                data_dir=data_dir,
                weights_path=best_ckpt,
                splits=("valid", "test"),
                device=str(dev),
            )
            sweep_path = output_dir / "sweep_results.json"
            with sweep_path.open("w", encoding="utf-8") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"[Scout Evaluation] Saved sweep results to {sweep_path}")
        except Exception as exc:
            print(f"[Scout Evaluation Warning] Sweep evaluation failed: {exc}")

    # Save metrics JSON and configuration
    final_payload = {
        "model_name": "MobileNetV3Scout",
        "parameters": model.count_parameters(),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "best_region_recall": best_recall,
        "final_epoch": actual_epochs,
        "history": history,
        "sweep_results": sweep_results,
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
            path_in_repo=f"checkpoints/{output_dir.name}",
        )
        syncer.wait_until_done(timeout=30.0)
        syncer.shutdown(wait=True)

    return final_payload


def evaluate_scout_model(
    data_dir: Path | str,
    weights_path: Path | str,
    split: str = "valid",
    output_path: Path | str | None = None,
    threshold: float = 0.05,
    context_margin: float = 0.30,
    k_max: int = 4,
    device: str | None = None,
    limit: int | None = None,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = False,
) -> dict[str, Any]:
    """Evaluate Scout Model candidate generation quality on specified split."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate Scout model")

    data_dir = Path(data_dir)
    weights_path = ensure_weights(weights_path)
    
    if device is None or device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    elif str(device).isdigit():
        dev = torch.device(f"cuda:{device}")
    else:
        dev = torch.device(device)
    model = MobileNetV3Scout().to(dev)

    if Path(weights_path).is_file():
        ckpt = torch.load(str(weights_path), map_location=dev, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    ds = ScoutDataset(data_dir, split=split, limit=limit, augment=False)
    eval_batch_size = 32 if dev.type == "cuda" else 8
    loader = DataLoader(
        ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    candidate_gen = CandidateGenerator(threshold=threshold, context_margin=context_margin, k_max=k_max)

    recalls = []
    coverages = []
    false_rates = []
    k_crops_list = []
    per_image_results = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                preds = model(imgs)
            pred_nps = preds.float().cpu().numpy()

            for i in range(len(batch["image_ids"])):
                img_id = batch["image_ids"][i]
                orig_w, orig_h = batch["orig_sizes"][i]
                orig_boxes = batch["orig_boxes"][i]
                hmap = pred_nps[i, 0]

                candidates = candidate_gen.generate(hmap, source_width=orig_w, source_height=orig_h)
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

        if hf_sync:
            from ..infra.upload import upload_to_hf, get_hf_credentials
            token, repo, rtype = get_hf_credentials(hf_token, hf_repo)
            if token:
                try:
                    upload_to_hf(
                        repo_id=repo,
                        local_path=out_p,
                        token=token,
                        repo_type=rtype,
                        path_in_repo=f"metrics/{out_p.name}",
                    )
                except Exception as e:
                    print(f"[Cloud Warning] Failed to upload scout evaluation report to HF: {e}")
    return summary


def sweep_scout_eval(
    data_dir: Path | str,
    weights_path: Path | str,
    splits: tuple[str, ...] = ("valid", "test"),
    thresholds: list[float] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    k_values: list[int] = [1, 2, 3, 4, 6],
    context_margin: float = 0.30,
    device: str | None = None,
) -> dict[str, Any]:
    """Sweep thresholds and K budgets on validation/test splits to construct Recall(K) curves."""
    results: dict[str, Any] = {}
    for split in splits:
        results[split] = {
            "threshold_sweep": {},
            "k_sweep": {},
        }
        # 1. Sweep thresholds at fixed k_max=4
        for th in thresholds:
            rep = evaluate_scout_model(data_dir, split=split, weights_path=weights_path, threshold=th, context_margin=context_margin, k_max=4, device=device)
            results[split]["threshold_sweep"][f"tau_{th:.2f}"] = {
                "region_recall": rep["mean_region_recall"],
                "gt_coverage": rep["mean_gt_coverage"],
                "false_region_rate": rep["mean_false_region_rate"],
                "avg_k": rep["mean_k"],
            }
        # 2. Sweep K at fixed threshold=0.05
        for k in k_values:
            rep = evaluate_scout_model(data_dir, split=split, weights_path=weights_path, threshold=0.05, context_margin=context_margin, k_max=k, device=device)
            results[split]["k_sweep"][f"k_{k}"] = {
                "region_recall": rep["mean_region_recall"],
                "gt_coverage": rep["mean_gt_coverage"],
                "false_region_rate": rep["mean_false_region_rate"],
                "avg_k": rep["mean_k"],
            }
    return results
