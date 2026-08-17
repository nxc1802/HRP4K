from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..infra.environment import environment_snapshot
from ..infra.upload import BackgroundHFSyncer


def train_yolo(
    dataset_yaml: Path,
    weights: Path | str,
    run_dir: Path,
    smoke: bool = False,
    epochs: int = 150,
    image_size: int | tuple[int, int] | str = 640,
    batch: int = 16,
    device: str | None = None,
    allow_full: bool = False,
    experiment: dict[str, Any] | None = None,
    seed: int = 42,
    eval_confidence: float = 0.001,
    resume: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    path_in_repo: str | None = None,
) -> dict[str, Any]:
    """Execute YOLO baseline training with optional background Hugging Face synchronization."""
    if not smoke and not allow_full:
        raise ValueError("Full training requires explicit --allow-full; use --smoke for local verification")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Training requires the 'vision' dependencies") from exc

    run_dir = run_dir.resolve()
    dataset_yaml = dataset_yaml.resolve()
    manifest_path = dataset_yaml.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if not smoke and (not manifest or not manifest.get("official_dataset_identity") or not manifest.get("official_dataset_view")):
        raise ValueError(
            "Official training requires the verified single official dataset view. Run `hrp4k prepare-dataset` without limits."
        )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run_dir.mkdir(parents=True, exist_ok=True)

    actual_epochs = min(2, epochs) if smoke else epochs
    resolved_imgsz = (2176, 3840) if str(image_size).strip().lower() in {"original", "4k", "native"} else image_size
    actual_imgsz = (
        min(640, resolved_imgsz)
        if (smoke and isinstance(resolved_imgsz, int))
        else ((320, 640) if (smoke and isinstance(resolved_imgsz, (tuple, list))) else resolved_imgsz)
    )
    is_rect = isinstance(actual_imgsz, (tuple, list)) or (isinstance(actual_imgsz, int) and actual_imgsz >= 1280)

    # Initialize Cloud Syncer (Background thread uploading checkpoints to HF)
    target_repo_path = path_in_repo or run_dir.name
    syncer = BackgroundHFSyncer(
        repo_id=hf_repo,
        token=hf_token,
        path_in_repo=target_repo_path,
        enabled=hf_sync and not smoke,
    )

    config = {
        "dataset": str(dataset_yaml.resolve()),
        "weights": str(weights),
        "smoke": smoke,
        "epochs": actual_epochs,
        "image_size": actual_imgsz,
        "batch": batch,
        "rect": is_rect,
        "amp": True,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "fliplr": 0.5,
        "device": device,
        "seed": seed,
        "eval_confidence": eval_confidence,
        "resume": resume,
        "hf_sync_enabled": syncer.enabled,
        "hf_repo": syncer.repo_id if syncer.enabled else None,
        "dataset_manifest": manifest,
        "benchmark_label": "smoke" if smoke else (manifest or {}).get("benchmark_label", "unverified"),
        "experiment": experiment,
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2), encoding="utf-8")

    model = YOLO(str(weights))

    # Register background epoch-end callback for Ultralytics
    def on_fit_epoch_end_callback(trainer: Any) -> None:
        if not syncer.enabled:
            return
        try:
            current_epoch = getattr(trainer, "epoch", 0) + 1
            save_dir = Path(getattr(trainer, "save_dir", run_dir))
            weights_dir = save_dir / "weights"
            results_csv = save_dir / "results.csv"
            args_yaml = save_dir / "args.yaml"
            syncer.sync_epoch(
                epoch=current_epoch,
                weights_dir=weights_dir,
                extra_files=[results_csv, args_yaml],
                path_in_repo=target_repo_path,
            )
        except Exception as cb_exc:
            print(f"[Cloud Sync Warning] Failed to trigger epoch sync: {cb_exc}")

    if syncer.enabled:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

    result = model.train(
        data=str(dataset_yaml),
        epochs=actual_epochs,
        imgsz=actual_imgsz,
        batch=batch,
        rect=is_rect,
        amp=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        mosaic=1.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        seed=seed,
        deterministic=True,
        workers=0 if smoke else 2,
        cache=False,
        plots=not smoke,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        device=device,
        verbose=True,
        resume=resume,
    )

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    eval_model_path = best if best.exists() else last

    val_metrics = {str(key): float(value) for key, value in getattr(result, "results_dict", {}).items()}
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    # Evaluate on test split with benchmark confidence (default: 0.001)
    test_metrics: dict[str, Any] = {}
    if eval_model_path.exists():
        try:
            eval_model = YOLO(str(eval_model_path))
            test_res = eval_model.val(
                data=str(dataset_yaml),
                split="test",
                imgsz=actual_imgsz,
                batch=batch,
                device=device,
                plots=not smoke,
                verbose=True,
                conf=eval_confidence,
            )
            test_metrics = {str(key): float(value) for key, value in getattr(test_res, "results_dict", {}).items()}
            test_metrics["eval_confidence"] = eval_confidence
            (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        except Exception as exc:
            test_metrics = {"error": str(exc)}

    # Perform final sync of weights, test metrics, and results
    if syncer.enabled:
        final_files = [
            run_dir / "val_metrics.json",
            run_dir / "test_metrics.json",
            run_dir / "results.csv",
            run_dir / "args.yaml",
            run_dir / "resolved_config.json",
        ]
        syncer.sync_epoch(
            epoch=actual_epochs,
            weights_dir=run_dir / "weights",
            extra_files=final_files,
            path_in_repo=target_repo_path,
        )
        print("[Cloud Sync] Finalizing upload of best.pt, last.pt and metric artifacts...")
        syncer.wait_until_done(timeout=60.0)
        syncer.shutdown(wait=True)

    return {
        "run_dir": str(run_dir),
        "best": str(eval_model_path),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metrics": val_metrics,
        "eval_confidence": eval_confidence,
    }
