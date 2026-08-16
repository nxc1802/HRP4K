from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..detectors.registry import BASELINE_PRESETS, get_baseline_preset
from ..training.runner import train_yolo

OFFICIAL_MODELS = [
    "yolo11m",
    "yolov8m",
    "yolov5m-compat",
    "yolov5m-official",
    "rt-detr-v1",
    "rt-detr-v2",
]


def run_phase_1(
    dataset_yaml: Path,
    weights: Path | str | None = None,
    output_dir: Path | None = None,
    smoke: bool = False,
    epochs: int = 150,
    image_size: int | str = 1280,
    batch: int = 16,
    device: str | None = None,
    allow_full: bool = False,
    preset: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Phase 1 baseline detector training with single-model or all-models support."""
    target_preset = preset or ("all" if weights == "all" else None)
    resolved_imgsz = 3840 if str(image_size).strip().lower() in {"original", "4k", "native"} else image_size
    
    if target_preset == "all":
        base_output = output_dir or Path("outputs/phase1_runs")
        base_output.mkdir(parents=True, exist_ok=True)
        results = []
        for model_name in OFFICIAL_MODELS:
            preset_dict = get_baseline_preset(model_name)
            model_weights = Path(preset_dict["weights"])
            model_output = base_output / model_name
            run_result = train_yolo(
                dataset_yaml=dataset_yaml,
                weights=model_weights,
                run_dir=model_output,
                smoke=smoke,
                epochs=epochs,
                image_size=resolved_imgsz,
                batch=batch,
                device=device,
                allow_full=allow_full,
                experiment=preset_dict,
                seed=seed,
            )
            results.append({"model": model_name, **run_result})
        
        summary_path = base_output / "phase1_benchmark_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return {"preset": "all", "image_size": resolved_imgsz, "summary_path": str(summary_path), "runs": results}

    preset_dict = get_baseline_preset(target_preset) if target_preset else None
    resolved_weights = weights or Path(preset_dict["weights"] if preset_dict else "yolo11m.pt")
    resolved_output = output_dir or Path("outputs/runs") / (target_preset or Path(resolved_weights).stem)
    return train_yolo(
        dataset_yaml=dataset_yaml,
        weights=resolved_weights,
        run_dir=resolved_output,
        smoke=smoke,
        epochs=epochs,
        image_size=resolved_imgsz,
        batch=batch,
        device=device,
        allow_full=allow_full,
        experiment=preset_dict,
        seed=seed,
    )
