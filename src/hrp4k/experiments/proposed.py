"""Proposed Method — Frozen RT-DETR-L with Lightweight Dense P2 Auxiliary Detector and NMS Fusion.

Architecture:
    Backbone (HGNetv2)
    ├── C2 (Stride 4) ──► P2 Adapter (1x1 + 3x3 Conv) ──► Lightweight Dense P2 Head ──┐
    │                                                                                  │
    ├── P3 (Stride 8)  ──┐                                                             │
    ├── P4 (Stride 16) ──┼──► Native RT-DETR (Frozen: AIFI + CCFM + Decoder) ─────────┤
    └── P5 (Stride 32) ──┘                                                             │
                                                                                       ▼
                                                                          Concatenate + NMS Fusion
                                                                                       │
                                                                                       ▼
                                                                                Final Predictions

Training: Fully frozen RT-DETR baseline + lightweight dense P2 head optimized with L_P2.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRDetectionModel

from .registry import ExperimentConfig
from ..detectors.base import Detection, DetectorAdapter
from ..models.p2_branch import find_c2_backbone_stage, extract_c2_backbone, P2Adapter, P2Branch
from ..models.p2_head import (
    DenseP2Loss,
    LightweightP2Head,
    P2DenseHead,
    P2QueryHead,
    P2HeadLoss,
    decode_dense_p2_predictions,
    RTDETRP2Model,
)
from ..inference.p2_fusion import fuse_native_and_p2_predictions, fuse_prediction_tensors
from ..infra.environment import environment_snapshot
from ..infra.hf_storage import ExperimentStorage
from ..infra.upload import BackgroundHFSyncer, ensure_weights


# ---------------------------------------------------------------------------
# Compatibility / Abstract Structures
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A region of interest in the source image."""
    x: int
    y: int
    width: int
    height: int
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ProposedMethodPipeline:
    """Modular pipeline interface wrapper."""
    def __init__(self, scout: Any = None, region_selector: Any = None, budget: int = 5, **kwargs: Any) -> None:
        self.scout = scout
        self.region_selector = region_selector
        self.budget = budget

    def run(self, image: np.ndarray, confidence: float = 0.25) -> dict[str, Any]:
        regions = self.scout.identify_regions(image) if self.scout else [Region(0, 0, image.shape[1], image.shape[0])]
        selected = self.region_selector.select(regions, self.budget) if self.region_selector else regions[:self.budget]
        return {
            "num_regions_scouted": len(regions),
            "num_regions_selected": len(selected),
            "pipeline_status": "active",
        }


# ---------------------------------------------------------------------------
# Detector Adapter for Frozen RT-DETR-L + Dense P2 Head
# ---------------------------------------------------------------------------

class RTDETRP2Adapter(DetectorAdapter):
    """Detector adapter wrapping Frozen RT-DETR-L + Lightweight Dense P2 Head with NMS Fusion."""

    def __init__(
        self,
        weights: Path | str,
        category_id: int = 0,
        device: str | None = None,
        name: str = "rtdetr-l-p2",
        precision: str = "fp32",
        fusion_iou_threshold: float = 0.5,
        p2_checkpoint: Path | str | None = None,
        mode: str = "fused",
    ) -> None:
        self.weights = Path(weights) if isinstance(weights, str) else weights
        self.category_id = category_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.precision = precision
        self.fusion_iou_threshold = fusion_iou_threshold
        self.p2_checkpoint = p2_checkpoint
        self.mode = mode
        self.is_coco_base = False

        self._init_model()

    def _init_model(self) -> None:
        weights_str = str(self.weights)
        native = RTDETR(weights_str).model if Path(weights_str).is_file() else RTDETR("rtdetr-l.pt").model
        native_nc = getattr(native, "yaml", {}).get("nc", 80)
        self.is_coco_base = (native_nc is not None and native_nc > 1)
        if self.is_coco_base:
            print(f"[RTDETRP2Adapter Notice] Native checkpoint is COCO ({native_nc} classes).")
            print(f"[RTDETRP2Adapter Notice] Mode: {self.mode}. Native queries will be filtered to class {self.category_id}.")

        self.model = RTDETRP2Model(native_model=native, nc=1, freeze_native=True)

        # Load P2 head weights if provided
        ckpt_path = self.p2_checkpoint or (self.weights if str(self.weights).endswith("best_p2.pt") else None)
        if ckpt_path and Path(ckpt_path).is_file():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "p2_state_dict" in ckpt:
                    self.model.p2_head.load_state_dict(ckpt["p2_state_dict"])
                if "p2_adapter_state_dict" in ckpt:
                    self.model.p2_branch.load_state_dict(ckpt["p2_adapter_state_dict"])
                elif "p2_model" in ckpt:
                    self.model.load_state_dict(ckpt["p2_model"], strict=False)

        self.model.to(self.device)
        self.model.eval()
        if self.precision == "fp16" and self.device != "cpu":
            self.model.half()

    def warmup(self, image: np.ndarray, image_size: int) -> None:
        self.predict(image, image_size, 0.01)

    def predict(self, image: np.ndarray, image_size: int, confidence: float) -> list[Detection]:
        h_orig, w_orig = image.shape[:2]
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.ops import scale_boxes

        letterbox = LetterBox(image_size, auto=True, stride=32)
        lb_img = letterbox(image=image)
        h_lb, w_lb = lb_img.shape[:2]

        tensor = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        if self.precision == "fp16" and self.device != "cpu":
            tensor = tensor.half()

        with torch.no_grad():
            out = self.model(tensor)

        native_preds = out["native_preds"][0]  # (300, 6) [x1, y1, x2, y2, score, cls]
        p2_preds = out["p2_preds"][0]          # (300, 6) [x1, y1, x2, y2, score, cls]

        def to_detections(tensor_preds: torch.Tensor, is_native: bool = False) -> list[Detection]:
            dets: list[Detection] = []
            mask = tensor_preds[:, 4] >= confidence
            if is_native and self.is_coco_base:
                mask = mask & (tensor_preds[:, 5].long() == self.category_id)
            if not mask.any():
                return dets

            filtered = tensor_preds[mask]
            # Unscale boxes from letterbox canvas to original image canvas
            boxes_lb = filtered[:, :4].clone()
            unscaled_boxes = scale_boxes((h_lb, w_lb), boxes_lb, (h_orig, w_orig)).cpu().numpy()
            scores = filtered[:, 4].cpu().numpy()

            for i in range(len(scores)):
                x1, y1, x2, y2 = unscaled_boxes[i]
                scaled_xyxy = (
                    float(np.clip(x1, 0, w_orig)),
                    float(np.clip(y1, 0, h_orig)),
                    float(np.clip(x2, 0, w_orig)),
                    float(np.clip(y2, 0, h_orig)),
                )
                dets.append(Detection(scaled_xyxy, float(scores[i]), self.category_id))
            return dets

        p2_dets = to_detections(p2_preds, is_native=False)
        if self.mode == "p2_only":
            return p2_dets

        native_dets = to_detections(native_preds, is_native=True)
        if self.mode == "native_only":
            return native_dets

        # Pure Concatenation + NMS Fusion
        fused = fuse_native_and_p2_predictions(
            native_dets,
            p2_dets,
            iou_threshold=self.fusion_iou_threshold,
        )
        return fused

    def predict_batch(self, images: list[np.ndarray], image_size: int, confidence: float) -> list[list[Detection]]:
        return [self.predict(img, image_size, confidence) for img in images]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "Frozen-RT-DETR-L-Dense-P2",
            "framework": "ultralytics + hrp4k",
            "weights": str(self.weights),
            "device": self.device,
            "precision": self.precision,
            "fusion_iou_threshold": self.fusion_iou_threshold,
        }


# ---------------------------------------------------------------------------
# Standalone Training Pipeline for Dense P2 Head
# ---------------------------------------------------------------------------

def train_rtdetr_p2(
    dataset_yaml: Path,
    weights: Path | str,
    run_dir: Path,
    smoke: bool = False,
    epochs: int = 150,
    image_size: int | tuple[int, int] | str = 1920,
    batch: int = 16,
    accumulation: int = 1,
    patience: int = 10,
    device: str | None = None,
    allow_full: bool = False,
    experiment: dict[str, Any] | None = None,
    seed: int = 42,
    eval_confidence: float = 0.001,
    resume: bool = False,
    p2_checkpoint: Path | str | None = None,
    rect: bool = True,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    path_in_repo: str | None = None,
) -> dict[str, Any]:
    """Execute dedicated training for Frozen RT-DETR-L + Lightweight Dense P2 Head."""
    if not smoke and not allow_full and not resume:
        raise ValueError("Full training requires explicit --allow-full; use --smoke for local verification")

    run_dir = run_dir.resolve()
    dataset_yaml = dataset_yaml.resolve()
    manifest_path = dataset_yaml.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_dir.mkdir(parents=True, exist_ok=True)

    actual_epochs = min(1, epochs) if smoke else epochs
    resolved_imgsz = 3840 if str(image_size).strip().lower() in {"original", "4k", "native"} else image_size
    actual_imgsz = min(320, resolved_imgsz) if (smoke and isinstance(resolved_imgsz, int)) else resolved_imgsz

    target_repo_path = path_in_repo or run_dir.name
    syncer = BackgroundHFSyncer(
        repo_id=hf_repo,
        token=hf_token,
        path_in_repo=target_repo_path,
        enabled=hf_sync and not smoke,
    )

    resolved_weights = ensure_weights(weights, repo_id=hf_repo, token=hf_token)

    # 1. Load Frozen RT-DETR-L Baseline
    rtdetr = RTDETR(str(resolved_weights))
    native_model = rtdetr.model
    if not isinstance(native_model, RTDETRDetectionModel):
        native_model.__class__ = RTDETRDetectionModel
    native_model.nc = 1

    p2_model = RTDETRP2Model(
        native_model=native_model,
        nc=1,
        input_size=(actual_imgsz, actual_imgsz) if isinstance(actual_imgsz, int) else actual_imgsz,
        freeze_native=True,
    )
    p2_model.to(target_device)

    # 2. Verify Parameter Freezing
    frozen_params = sum(p.numel() for p in p2_model.native_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in p2_model.parameters() if p.requires_grad)
    p2_params_count = sum(p.numel() for p in p2_model.p2_branch.parameters()) + sum(p.numel() for p in p2_model.p2_head.parameters())
    print(f"[Architecture] Base RT-DETR Model:       {resolved_weights}")
    print(f"[Architecture] Base Model Classes:       {getattr(native_model, 'nc', 1)} (Pothole)")
    print(f"[Architecture] Frozen Base Parameters:   {frozen_params:,} (100% FROZEN)")
    print(f"[Architecture] Trainable P2 Parameters: {trainable_params:,} (P2 ONLY: {p2_params_count:,})")

    # 3. Setup Optimizer ONLY for P2 parameters
    p2_params = list(p2_model.p2_branch.parameters()) + list(p2_model.p2_head.parameters())
    optimizer = torch.optim.AdamW(p2_params, lr=0.0005, weight_decay=0.0001)

    print(f"\n[Proposed Engine] Launching Frozen RT-DETR-L + Dense P2 Head (Epochs: {actual_epochs}, Imgsz: {actual_imgsz}, Device: {target_device})")

    # 4. Build Dataset & DataLoader using Ultralytics Data Loader or custom loader
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = str(dataset_yaml)
    cfg.imgsz = actual_imgsz
    cfg.batch = batch
    cfg.rect = rect
    cfg.workers = 0 if smoke else 8

    # Load dataset info
    import yaml
    with open(dataset_yaml, "r") as f:
        data_info = yaml.safe_load(f)

    train_path = Path(dataset_yaml).parent / data_info.get("train", "train")
    if not train_path.exists():
        train_path = Path(data_info.get("train", "train"))

    train_dataset = build_yolo_dataset(cfg, str(train_path), batch, data_info, mode="train", rect=rect, stride=32)
    train_loader = build_dataloader(train_dataset, batch, cfg.workers, shuffle=True, rank=-1)

    # Training Loop & Resume State
    best_loss = float("inf")
    start_epoch = 0
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_p2_path = weights_dir / "best_p2.pt"
    last_p2_path = weights_dir / "last_p2.pt"
    patience_counter = 0

    # If resume is requested and P2 checkpoint provided, restore P2 weights
    if resume and p2_checkpoint and Path(p2_checkpoint).is_file():
        try:
            ckpt_data = torch.load(p2_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "p2_state_dict" in ckpt_data:
                print(f"[Resume P2] Restoring P2 head checkpoint from {p2_checkpoint}...")
                p2_model.p2_head.load_state_dict(ckpt_data["p2_state_dict"])
                if "p2_adapter_state_dict" in ckpt_data:
                    p2_model.p2_branch.load_state_dict(ckpt_data["p2_adapter_state_dict"])
                if "optimizer_state_dict" in ckpt_data:
                    try:
                        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                    except Exception as exc:
                        print(f"[Resume Warning] Could not restore optimizer state: {exc}")
                start_epoch = int(ckpt_data.get("epoch", 0))
                best_loss = float(ckpt_data.get("mean_p2_loss", float("inf")))
                print(f"[Resume P2] Resuming training from epoch {start_epoch + 1}/{actual_epochs} (best loss: {best_loss:.4f})")
        except Exception as exc:
            print(f"[Resume Warning] Failed loading resume checkpoint: {exc}")
    else:
        print(f"[P2 Initialization] Initialized FRESH Lightweight P2 Head & Adapter (Training P2 from scratch)")

    for epoch in range(start_epoch, actual_epochs):
        p2_model.p2_branch.train()
        p2_model.p2_head.train()
        epoch_losses: list[float] = []

        for batch_idx, batch_data in enumerate(train_loader):
            img = batch_data["img"].to(target_device).float() / 255.0
            bs = img.shape[0]
            b_idx = batch_data["batch_idx"].to(target_device)
            gt_groups = [(b_idx == i).sum().item() for i in range(bs)]
            targets = {
                "cls": batch_data["cls"].to(target_device, dtype=torch.long).view(-1),
                "bboxes": batch_data["bboxes"].to(device=target_device),
                "batch_idx": b_idx.view(-1),
                "gt_groups": gt_groups,
            }

            # Forward P2 with Frozen Backbone
            with torch.no_grad():
                c2_feat = extract_c2_backbone(p2_model.native_model, img, c2_layer_idx=p2_model.c2_layer_idx)

            p2_feat = p2_model.p2_branch(c2_feat)
            cls_logits, box_offsets = p2_model.p2_head(p2_feat)

            loss_dict = p2_model.p2_head.compute_loss(
                cls_logits=cls_logits,
                box_offsets=box_offsets,
                targets=targets,
                img_size=(img.shape[-2], img.shape[-1]),
            )
            loss = loss_dict["loss_p2_total"]
            scaled_loss = loss / max(1, accumulation)
            scaled_loss.backward()

            if (batch_idx + 1) % max(1, accumulation) == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())

            if smoke and batch_idx >= 1:
                break

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"Epoch {epoch + 1}/{actual_epochs} - Mean P2 Loss: {mean_loss:.4f}")

        # Construct checkpoint payload
        payload = {
            "p2_state_dict": p2_model.p2_head.state_dict(),
            "p2_adapter_state_dict": p2_model.p2_branch.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "mean_p2_loss": mean_loss,
            "base_checkpoint": str(resolved_weights),
            "image_size": actual_imgsz,
            "architecture": "frozen_rtdetr_l_p2",
        }

        # Save last checkpoint (both last_p2.pt and last.pt for HF syncer)
        torch.save(payload, str(last_p2_path))
        torch.save(payload, str(weights_dir / "last.pt"))

        # Save best checkpoint & track early stopping patience
        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            patience_counter = 0
            torch.save(payload, str(best_p2_path))
            torch.save(payload, str(weights_dir / "best.pt"))
        else:
            patience_counter += 1
            if not smoke and patience > 0 and patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in loss for {patience} consecutive epochs (best loss: {best_loss:.4f}).")
                print(f"[Early Stopping] Stopping training early at epoch {epoch + 1}/{actual_epochs}.")
                break

    val_metrics = {"p2_loss": best_loss, "epochs": actual_epochs}
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    # Post-train evaluation on test set
    test_metrics: dict[str, Any] = {}
    try:
        from ..inference.runner import predict_detector
        from ..evaluation.coco import evaluate_files
        from ..data.paths import resolve_data_dir

        raw_data_dir = resolve_data_dir(Path(dataset_yaml).parent if dataset_yaml else "HRP4K")
        test_gt = raw_data_dir / "test.json"
        if not test_gt.is_file():
            for candidate in (Path("HRP4K/test.json"), Path("test.json"), Path("../HRP4K/test.json")):
                if candidate.is_file():
                    test_gt = candidate
                    raw_data_dir = candidate.parent
                    break

        test_pred_path = run_dir / "test_predictions.json"
        adapter = RTDETRP2Adapter(
            weights=resolved_weights,
            category_id=0,
            device=target_device,
            p2_checkpoint=best_p2_path,
        )

        if test_gt.is_file():
            print(f"\n[Proposed Evaluation] Evaluating Frozen RT-DETR-L + Dense P2 Head on test set...")
            pred_summary = predict_detector(
                data_dir=raw_data_dir,
                split="test",
                detector=adapter,
                output_path=test_pred_path,
                method="resize",
                image_size=actual_imgsz if isinstance(actual_imgsz, int) else actual_imgsz[0],
                confidence=eval_confidence,
            )

            test_metrics_path = run_dir / "test_metrics.json"
            test_metrics = evaluate_files(
                gt_path=test_gt,
                prediction_path=test_pred_path,
                output_path=test_metrics_path,
                confidence=eval_confidence,
            )
            test_metrics["mean_latency_ms"] = pred_summary.get("mean_end_to_end_latency_ms", 0.0)
            test_metrics["eval_confidence"] = eval_confidence
            test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            (run_dir / "predictions.json").write_text(test_pred_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[Proposed Evaluation Complete] AP50: {test_metrics.get('AP50', 0)*100:.2f}%, AP50-95: {test_metrics.get('AP50_95', 0)*100:.2f}%, FPPI: {test_metrics.get('FPPI_official', 0.0):.4f}")
    except Exception as exc:
        print(f"[Proposed Evaluation Warning] Post-train evaluation issue: {exc}")
        test_metrics = {"error": str(exc)}

    # Save resolved config & environment
    config_dict = {
        "dataset": str(dataset_yaml.resolve()),
        "weights": str(resolved_weights),
        "smoke": smoke,
        "epochs": actual_epochs,
        "image_size": actual_imgsz,
        "batch": batch,
        "rect": rect,
        "optimizer": "AdamW",
        "lr": 0.0005,
        "architecture": "frozen_rtdetr_l_p2",
        "experiment": experiment,
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2), encoding="utf-8")

    if syncer.enabled:
        print(f"\n[Cloud Sync] Training and test evaluation complete. Syncing all final artifacts to Hugging Face ({hf_repo})...")
        final_files = [
            run_dir / "val_metrics.json",
            run_dir / "test_metrics.json",
            run_dir / "test_predictions.json",
            run_dir / "predictions.json",
            run_dir / "resolved_config.json",
            run_dir / "environment.json",
        ]
        syncer.sync_epoch(
            epoch=actual_epochs,
            weights_dir=weights_dir,
            extra_files=[f for f in final_files if f.is_file()],
            path_in_repo=target_repo_path,
        )
        syncer.wait_until_done(timeout=120.0)
        syncer.shutdown(wait=True)
        print(f"[Cloud Sync] All checkpoints and evaluation outputs successfully uploaded to Hugging Face!")

    return {
        "run_dir": str(run_dir),
        "best": str(best_p2_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "eval_confidence": eval_confidence,
    }


# ---------------------------------------------------------------------------
# Full Experiment Orchestrator
# ---------------------------------------------------------------------------

def run_proposed_experiment(
    config: ExperimentConfig,
    dataset_yaml: Path,
    output_dir: Path,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute a full proposed method feasibility experiment."""
    experiment_name = config.name
    exp_id = config.experiment_id
    run_dir = output_dir / experiment_name

    print(f"\n{'='*60}")
    print(f"[Proposed Experiment] {experiment_name}")
    print(f"[Architecture]        Frozen {config.detector} + Lightweight Dense P2 Head")
    print(f"[Resolution]          {config.resolution} ({config.imgsz}px)")
    print(f"[Batch]               {config.batch} × {config.accumulation}x accum = {config.effective_batch}")
    print(f"[Fusion Strategy]     Concatenate + Class-Aware NMS")
    print(f"[Exp ID]              {exp_id}")
    print(f"{'='*60}\n")

    if dry_run:
        return {"experiment": experiment_name, "experiment_id": exp_id, "status": "dry_run", "config": config.to_dict()}

    storage = ExperimentStorage(exp_id, repo_id=hf_repo, token=hf_token)
    state = storage.check_experiment_exists()

    # 1. Ensure Base Fine-Tuned RT-DETR Model is Available Locally
    base_weights_path = Path(config.weights)
    if not base_weights_path.is_file():
        print(f"\n[Base Model Download] Fine-tuned model '{config.weights}' not found locally.")
        print(f"[Base Model Download] Downloading from Hugging Face repository '{hf_repo or 'Cuong2004/HRP4K'}'...")
        try:
            from huggingface_hub import hf_hub_download
            target_rel = str(config.weights).replace("\\", "/")
            downloaded = hf_hub_download(
                repo_id=hf_repo or "Cuong2004/HRP4K",
                filename=target_rel,
                repo_type="dataset",
                token=hf_token or os.environ.get("HF_TOKEN"),
            )
            if downloaded and Path(downloaded).is_file():
                base_weights_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(downloaded, str(base_weights_path))
                print(f"[Base Model Download] Successfully downloaded fine-tuned baseline to: {base_weights_path}")
        except Exception as exc:
            print(f"[Base Model Download Warning] Could not download from HF: {exc}")
            if not base_weights_path.is_file():
                print(f"[Base Model Download Warning] Falling back to 'rtdetr-l.pt'")
                base_weights_path = Path("rtdetr-l.pt")

    # 2. Check P2 Resume State (Only if resume=True requested)
    p2_checkpoint = None
    if resume:
        storage = ExperimentStorage(exp_id, repo_id=hf_repo, token=hf_token)
        state = storage.check_experiment_exists()
        if state.exists and state.checkpoint_path:
            print(f"[Resume P2] Found existing P2 experiment on HF (epoch {state.latest_epoch}). Downloading checkpoint...")
            local_ckpt = storage.download_checkpoint(state.latest_epoch)
            if local_ckpt:
                p2_checkpoint = str(local_ckpt)
        if not p2_checkpoint:
            for local_cand in [
                run_dir / "weights" / "best_p2.pt",
                run_dir / "weights" / "last.pt",
                run_dir / "weights" / "best.pt",
            ]:
                if local_cand.is_file():
                    p2_checkpoint = str(local_cand)
                    print(f"[Resume P2] Resuming from local checkpoint: {p2_checkpoint}")
                    break
    else:
        print(f"\n[Proposed Mode] Starting FRESH training of P2 Head on top of Frozen Fine-Tuned RT-DETR: {base_weights_path}")

    storage = ExperimentStorage(exp_id, repo_id=hf_repo, token=hf_token)
    storage.upload_config(config.to_dict())
    storage.upload_manifest({
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "detector": config.detector,
        "phase": config.phase,
        "resolution": config.resolution,
        "status": "training",
        "base_model": str(base_weights_path),
        "environment": environment_snapshot(),
    })

    train_result = train_rtdetr_p2(
        dataset_yaml=dataset_yaml,
        weights=base_weights_path,
        run_dir=run_dir,
        smoke=False,
        epochs=config.epochs,
        image_size=config.imgsz,
        batch=config.batch,
        accumulation=config.accumulation,
        patience=config.patience,
        device=None,
        allow_full=True,
        experiment={"name": experiment_name, "id": exp_id},
        seed=config.seed,
        eval_confidence=config.confidence,
        resume=resume,
        p2_checkpoint=p2_checkpoint,
        rect=config.rect,
        hf_repo=hf_repo,
        hf_token=hf_token,
        hf_sync=hf_sync,
        path_in_repo=f"experiments/{exp_id}",
    )

    val_path = run_dir / "val_metrics.json"
    test_path = run_dir / "test_metrics.json"
    best_p2_file = run_dir / "weights" / "best_p2.pt"
    if best_p2_file.is_file():
        storage.upload_file(best_p2_file, "weights/best_p2.pt", "Upload final best_p2.pt weights")
        storage.upload_file(best_p2_file, "checkpoints/best.pt", "Upload final best.pt checkpoint")
    storage.upload_final_results(
        val_metrics_path=val_path if val_path.exists() else None,
        test_metrics_path=test_path if test_path.exists() else None,
    )

    storage.upload_manifest({
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "detector": config.detector,
        "phase": config.phase,
        "resolution": config.resolution,
        "status": "completed",
        "val_metrics": train_result.get("val_metrics", {}),
        "test_metrics": train_result.get("test_metrics", {}),
        "best_checkpoint": train_result.get("best", ""),
    })

    try:
        from ..reports.report import update_experiment_final
        update_experiment_final(config, train_result)
    except Exception as exc:
        print(f"[Report Warning] Could not update Experiment_Final.md: {exc}")

    return {
        "experiment": experiment_name,
        "experiment_id": exp_id,
        "status": "completed",
        "run_dir": str(run_dir),
        "val_metrics": train_result.get("val_metrics", {}),
        "test_metrics": train_result.get("test_metrics", {}),
    }


# ---------------------------------------------------------------------------
# Local Feasibility Smoke Test
# ---------------------------------------------------------------------------

def run_proposed_smoke() -> dict[str, Any]:
    """Execute end-to-end local feasibility test:
    1. Dynamic C2 discovery from RT-DETR-L runtime graph
    2. Dynamic P2Adapter & LightweightP2Head construction
    3. Forward pass verification
    4. Gradient backward flow verification (only P2 receiving gradients, native frozen)
    5. Concatenation + NMS prediction fusion verification
    """
    native = RTDETR("rtdetr-l.pt").model
    c2_idx, c2_channels = find_c2_backbone_stage(native, input_size=(640, 640))

    model = RTDETRP2Model(native_model=native, nc=1, freeze_native=True)
    model.eval()

    # 1. Forward Pass in Eval
    dummy_input = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        eval_out = model(dummy_input)

    native_shape = list(eval_out["native_preds"].shape)
    p2_shape = list(eval_out["p2_preds"].shape)

    # 2. Prediction Fusion
    fused_tensor = fuse_prediction_tensors(eval_out["native_preds"], eval_out["p2_preds"], iou_threshold=0.5)

    # 3. Dedicated 1-batch Train Flow & Shape Validations
    model.p2_branch.train()
    model.p2_head.train()
    train_input = torch.randn(2, 3, 640, 640)

    # Verify C2 extraction
    with torch.no_grad():
        c2_feat = extract_c2_backbone(model.native_model, train_input, c2_layer_idx=c2_idx)
    assert c2_feat.shape == (2, c2_channels, 160, 160), f"Unexpected C2 shape: {c2_feat.shape}"

    # Verify P2 feature & dense head output shapes
    p2_feat = model.p2_branch(c2_feat)
    assert p2_feat.shape == (2, 256, 160, 160), f"Unexpected P2 shape: {p2_feat.shape}"
    cls_logits, box_offsets = model.p2_head(p2_feat)
    assert cls_logits.shape == (2, 1, 160, 160), f"Unexpected cls_logits shape: {cls_logits.shape}"
    assert box_offsets.shape == (2, 4, 160, 160), f"Unexpected box_offsets shape: {box_offsets.shape}"

    batch = {
        "img": train_input,
        "cls": torch.tensor([0, 0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.3, 0.1, 0.2]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    total_loss, loss_dict = model.loss(batch)
    assert torch.isfinite(total_loss), "Loss must be finite!"

    # 4. Backward Pass & Gradient Verification
    p2_params = list(model.p2_branch.parameters()) + list(model.p2_head.parameters())
    optimizer = torch.optim.AdamW(p2_params, lr=0.0005, weight_decay=0.0001)
    optimizer.zero_grad()
    total_loss.backward()

    # Verify native backbone received NO gradients (frozen)
    native_grads = [p.grad for p in model.native_model.parameters() if p.grad is not None]
    assert len(native_grads) == 0, "Native RT-DETR parameters must be strictly frozen!"
    assert model.p2_branch.adapter.conv1x1.weight.grad is not None, "P2 adapter must receive gradients!"
    assert model.p2_head.cls_conv[-1].weight.grad is not None, "P2 head must receive gradients!"

    # Verify optimizer step completes without issue
    optimizer.step()

    # 5. Verify Letterbox & Unscaling Inference
    adapter = RTDETRP2Adapter(weights="rtdetr-l.pt", category_id=0, device="cpu")
    non_square_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dets = adapter.predict(non_square_img, image_size=640, confidence=0.001)
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        assert 0.0 <= x1 <= 1920.0 and 0.0 <= x2 <= 1920.0, f"Box x-bounds invalid: {d.xyxy}"
        assert 0.0 <= y1 <= 1080.0 and 0.0 <= y2 <= 1080.0, f"Box y-bounds invalid: {d.xyxy}"

    return {
        "status": "pass",
        "architecture": "frozen_rtdetr_l_p2_dense",
        "c2_layer_index": c2_idx,
        "c2_channels": c2_channels,
        "c2_shape": list(c2_feat.shape),
        "p2_shape": list(p2_feat.shape),
        "cls_shape": list(cls_logits.shape),
        "box_shape": list(box_offsets.shape),
        "native_eval_shape": native_shape,
        "p2_eval_shape": p2_shape,
        "fused_batch_size": len(fused_tensor),
        "train_loss": {k: float(v.detach()) for k, v in loss_dict.items()},
        "native_frozen_verified": True,
        "gradient_check": "passed",
        "optimizer_step": "passed",
        "letterbox_unscaling": "passed",
    }
