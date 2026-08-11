from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        snapshot.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                         "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()})
    except ImportError: pass
    try:
        import ultralytics
        snapshot["ultralytics"] = ultralytics.__version__
    except ImportError: pass
    try:
        snapshot["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError): snapshot["git_commit"] = None
    try:
        snapshot["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (subprocess.SubprocessError, FileNotFoundError): snapshot["pip_freeze"] = []
    return snapshot


def train_yolo(
    dataset_yaml: Path, weights: Path | str, run_dir: Path, smoke: bool = False,
    epochs: int = 150, image_size: int = 640, batch: int = 16, device: str | None = None,
    allow_full: bool = False, allow_incomplete_train: bool = False,
) -> dict[str, Any]:
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
    if not smoke and (not manifest or not manifest.get("official_training_complete")) and not allow_incomplete_train:
        raise ValueError(
            "Official training requires all 4,203 train images. Pass --allow-incomplete-train only for a clearly labelled local-available run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    actual_epochs = min(2, epochs) if smoke else epochs
    actual_imgsz = min(640, image_size) if smoke else image_size
    config = {
        "dataset": str(dataset_yaml.resolve()), "weights": str(weights), "smoke": smoke,
        "epochs": actual_epochs, "image_size": actual_imgsz, "batch": batch,
        "amp": True, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "mosaic": 1.0, "mixup": 0.0, "fliplr": 0.5, "device": device, "seed": 42,
        "dataset_manifest": manifest, "benchmark_label": "smoke" if smoke else (manifest or {}).get("benchmark_label", "unverified"),
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2), encoding="utf-8")
    model = YOLO(str(weights))
    result = model.train(
        data=str(dataset_yaml), epochs=actual_epochs, imgsz=actual_imgsz, batch=batch,
        amp=True, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        mosaic=1.0, mixup=0.0, degrees=0.0, translate=0.1, scale=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5,
        seed=42, deterministic=True, workers=0 if smoke else 8, cache=False, plots=not smoke,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True, device=device, verbose=True,
    )
    best = run_dir / "weights" / "best.pt"; last = run_dir / "weights" / "last.pt"
    metrics = getattr(result, "results_dict", {})
    return {"run_dir": str(run_dir), "best": str(best if best.exists() else last),
            "metrics": {str(key): float(value) for key, value in metrics.items()}}
