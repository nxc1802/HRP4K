from __future__ import annotations

from pathlib import Path
from typing import Any

from .phase_0 import run_phase_0
from .phase_1 import run_phase_1
from .phase_2 import run_phase_2
from .phase_3 import run_phase_3
from ..data.views import prepare_dataset_view


def run_smoke_pipeline(
    data_dir: Path, output_dir: Path, weights: Path | str = Path("yolo11n.pt"),
    train_limit: int = 2, eval_limit: int = 1, image_size: int = 256,
    device: str | None = "cpu",
) -> dict[str, Any]:
    """Execute Phase 0–3 end-to-end smoke pipeline with absolute minimal compute footprint (2 train, 1 eval)."""
    root = output_dir
    dataset_dir = root / "dataset"
    actual_device = device or "cpu"

    run_phase_0(data_dir, root / "phase0", quality_samples=1)
    prepare_dataset_view(data_dir, dataset_dir, train_limit, 1, eval_limit, seed=42)
    training = run_phase_1(
        dataset_yaml=dataset_dir / "dataset.yaml", weights=weights,
        output_dir=root / "runs" / "yolo11n", smoke=True, epochs=1,
        image_size=image_size, batch=1, device=actual_device,
    )
    prediction_paths = []
    for method in ("resize", "sliced-nms", "perspective-grid", "sahi"):
        prediction_path = root / "predictions" / f"{method}.json"
        try:
            run_phase_2(
                data_dir=dataset_dir, split="test", weights=training["best"],
                output_path=prediction_path, method=method, limit=eval_limit,
                image_size=image_size, confidence=0.01, device=actual_device, warmup=0,
                evaluate_after=True, ground_truth=dataset_dir / "test.json", eval_confidence=0.25,
            )
            prediction_paths.append(prediction_path)
        except RuntimeError as err:
            print(f"[Smoke Warning] Skipped {method} due to optional dependency: {err}")
    result = run_phase_3(dataset_dir / "test.json", prediction_paths, root / "phase3")
    return {
        "status": "complete", "output": str(root), "training": training,
        "diagnostics": {
            "ground_truth": result["ground_truth"],
            "methods": {
                name: {
                    "evaluation_status": value["evaluation_status"],
                    "AP50_95": value["metrics"].get("AP50_95"),
                    "compute_amplification_nominal_canvas": value["processing"].get("compute_amplification_nominal_canvas"),
                }
                for name, value in result["methods"].items()
            },
            "report": str(root / "phase3" / "phase3_report.md"),
        },
    }
