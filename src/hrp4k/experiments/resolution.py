"""Resolution experiment pipeline.

Executes: Dataset → Train → Checkpoint → Val → Test → Scale eval → HF sync → Report update
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import ExperimentConfig
from ..infra.hf_storage import ExperimentStorage
from ..infra.environment import environment_snapshot


def run_resolution_experiment(
    config: ExperimentConfig,
    dataset_yaml: Path,
    output_dir: Path,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a full resolution experiment pipeline."""
    experiment_name = config.name
    exp_id = config.experiment_id
    run_dir = output_dir / experiment_name

    print(f"\n{'='*60}")
    print(f"[Experiment] {experiment_name}")
    print(f"[Detector]   {config.detector}")
    print(f"[Resolution] {config.resolution} ({config.imgsz}px)")
    print(f"[Batch]      {config.batch} × {config.accumulation}x accum = {config.effective_batch}")
    print(f"[Optimizer]  {config.optimizer} lr0={config.lr0}")
    print(f"[Epochs]     {config.epochs} (patience={config.patience})")
    print(f"[AMP]        {config.amp}")
    print(f"[Exp ID]     {exp_id}")
    print(f"{'='*60}\n")

    if dry_run:
        return {"experiment": experiment_name, "experiment_id": exp_id, "status": "dry_run", "config": config.to_dict()}

    # 1. Check HF for existing experiment state
    storage = ExperimentStorage(exp_id, repo_id=hf_repo, token=hf_token)
    state = storage.check_experiment_exists()

    resume = False
    weights = config.weights
    if state.exists and state.checkpoint_path:
        print(f"[Resume] Found existing experiment on HF (epoch {state.latest_epoch}). Downloading checkpoint...")
        local_ckpt = storage.download_checkpoint(state.latest_epoch)
        if local_ckpt:
            weights = str(local_ckpt)
            resume = True
            print(f"[Resume] Will resume from epoch {state.latest_epoch}")
    elif state.exists:
        print(f"[Info] Experiment exists on HF but no checkpoint found. Starting fresh.")

    # 2. Upload config and manifest
    storage.upload_config(config.to_dict())
    storage.upload_manifest({
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "detector": config.detector,
        "phase": config.phase,
        "resolution": config.resolution,
        "status": "training",
        "environment": environment_snapshot(),
    })

    # 3. Train
    from ..training.runner import train_yolo

    train_result = train_yolo(
        dataset_yaml=dataset_yaml,
        weights=weights,
        run_dir=run_dir,
        smoke=False,
        epochs=config.epochs,
        image_size=config.imgsz,
        batch=config.batch,
        device=None,
        allow_full=True,
        experiment={"name": experiment_name, "id": exp_id},
        seed=config.seed,
        eval_confidence=config.confidence,
        resume=resume,
        rect=config.rect,
        hf_repo=hf_repo,
        hf_token=hf_token,
        hf_sync=hf_sync,
        path_in_repo=f"experiments/{exp_id}",
    )

    # 4. Upload final results
    val_path = run_dir / "val_metrics.json"
    test_path = run_dir / "test_metrics.json"
    storage.upload_final_results(
        val_metrics_path=val_path if val_path.exists() else None,
        test_metrics_path=test_path if test_path.exists() else None,
    )

    # 5. Update manifest with final status
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

    # 6. Update Experiment Final report
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
