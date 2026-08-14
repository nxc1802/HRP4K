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
    train_limit: int = 24, eval_limit: int = 8, image_size: int = 320,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute Phase 0–3 end-to-end smoke pipeline."""
    root = output_dir
    dataset_dir = root / "dataset"
    run_phase_0(data_dir, root / "phase0", quality_samples=4)
    prepare_dataset_view(data_dir, dataset_dir, train_limit, max(eval_limit, 4), eval_limit, seed=42)
    training = run_phase_1(
        dataset_yaml=dataset_dir / "dataset.yaml", weights=weights,
        output_dir=root / "runs" / "yolo11n", smoke=True, epochs=1,
        image_size=image_size, batch=4, device=device,
    )
    prediction_paths = []
    for method in ("resize", "sliced-nms", "perspective-grid"):
        prediction_path = root / "predictions" / f"{method}.json"
        run_phase_2(
            data_dir=dataset_dir, split="test", weights=training["best"],
            output_path=prediction_path, method=method, limit=eval_limit,
            image_size=image_size, confidence=0.01, device=device, warmup=1,
            evaluate_after=True, ground_truth=dataset_dir / "test.json", eval_confidence=0.25,
        )
        prediction_paths.append(prediction_path)
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
