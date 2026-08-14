from __future__ import annotations

from pathlib import Path
from typing import Any

from ..detectors.registry import get_baseline_preset
from ..training.runner import train_yolo


def run_phase_1(
    dataset_yaml: Path, weights: Path | str | None, output_dir: Path | None,
    smoke: bool = False, epochs: int = 150, image_size: int = 640,
    batch: int = 16, device: str | None = None, allow_full: bool = False,
    preset: str | None = None,
) -> dict[str, Any]:
    """Execute Phase 1 baseline detector training with safeguards."""
    preset_dict = get_baseline_preset(preset) if preset else None
    resolved_weights = weights or Path(preset_dict["weights"] if preset_dict else "yolo11m.pt")
    resolved_output = output_dir or Path("outputs/runs") / (preset or Path(resolved_weights).stem)
    return train_yolo(
        dataset_yaml=dataset_yaml, weights=resolved_weights, run_dir=resolved_output,
        smoke=smoke, epochs=epochs, image_size=image_size, batch=batch,
        device=device, allow_full=allow_full, experiment=preset_dict,
    )
