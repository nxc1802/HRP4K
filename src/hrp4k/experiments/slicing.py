"""Slicing experiment pipeline.

Executes: Frozen 640 checkpoint → 4K test images → Spatial method → Predictions
         → Coordinate remap → NMS → Evaluation → HF sync → Report update

Slicing experiments do NOT retrain the detector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import ExperimentConfig
from ..infra.hf_storage import ExperimentStorage


def run_slicing_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    output_dir: Path,
    frozen_checkpoint: Path | str | None = None,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a slicing experiment using a frozen 640 detector checkpoint."""
    experiment_name = config.name
    exp_id = config.experiment_id
    run_dir = output_dir / experiment_name

    print(f"\n{'='*60}")
    print(f"[Slicing Experiment] {experiment_name}")
    print(f"[Detector]   {config.detector}")
    print(f"[Method]     {config.method}")
    print(f"[Checkpoint] {frozen_checkpoint or config.frozen_checkpoint or config.weights}")
    print(f"[Exp ID]     {exp_id}")
    print(f"{'='*60}\n")

    if dry_run:
        return {"experiment": experiment_name, "experiment_id": exp_id, "status": "dry_run", "config": config.to_dict()}

    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve frozen checkpoint
    weights = frozen_checkpoint or config.frozen_checkpoint or config.weights
    from ..infra.upload import ensure_weights
    weights = ensure_weights(weights, repo_id=hf_repo, token=hf_token)

    # Storage
    storage = ExperimentStorage(exp_id, repo_id=hf_repo, token=hf_token)
    storage.upload_config(config.to_dict())

    # Run prediction with the specified slicing method
    from ..phases.phase_2 import run_phase_2

    predictions_path = run_dir / f"{experiment_name}_predictions.json"

    result = run_phase_2(
        data_dir=data_dir,
        split="test",
        weights=weights,
        output_path=predictions_path,
        method=config.method,
        image_size=640,
        confidence=config.confidence,
        tile_size=config.tile_size,
        overlap=config.overlap,
        detector_name=config.detector,
        evaluate_after=True,
        ground_truth=data_dir / "test.json",
        hf_repo=hf_repo,
        hf_token=hf_token,
        hf_sync=storage.enabled,
    )

    # Upload results
    metrics = result.get("metrics", {})

    metrics_path = run_dir / f"{experiment_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    storage.upload_final_results(
        test_metrics_path=metrics_path,
        test_predictions_path=predictions_path if predictions_path.exists() else None,
    )

    # Update manifest
    storage.upload_manifest({
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "detector": config.detector,
        "phase": "slicing",
        "method": config.method,
        "status": "completed",
        "metrics": metrics,
        "frozen_checkpoint": str(weights),
    })

    # Update report
    try:
        from ..reports.report import update_experiment_final
        update_experiment_final(config, {"metrics": metrics, "test_metrics": metrics})
    except Exception as exc:
        print(f"[Report Warning] Could not update Experiment_Final.md: {exc}")

    return {
        "experiment": experiment_name,
        "experiment_id": exp_id,
        "status": "completed",
        "method": config.method,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
    }
