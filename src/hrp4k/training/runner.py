from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..infra.environment import environment_snapshot
from ..infra.upload import BackgroundHFSyncer, ensure_weights


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
    rect: bool = True,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_sync: bool = True,
    path_in_repo: str | None = None,
) -> dict[str, Any]:
    """Execute YOLO baseline training with optional background Hugging Face synchronization."""
    if not smoke and not allow_full and not resume:
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

    actual_epochs = min(1, epochs) if smoke else epochs
    resolved_imgsz = 3840 if str(image_size).strip().lower() in {"original", "4k", "native"} else image_size
    actual_imgsz = min(320, resolved_imgsz) if (smoke and isinstance(resolved_imgsz, int)) else resolved_imgsz
    is_rect = rect

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

    resolved_weights = ensure_weights(weights, repo_id=hf_repo, token=hf_token)
    if not Path(resolved_weights).is_file() and str(resolved_weights) not in {
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
        "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
        "rtdetr-l.pt", "rtdetr-x.pt",
    }:
        if resume:
            raise FileNotFoundError(
                f"Checkpoint '{weights}' not found locally or on Hugging Face ({hf_repo or 'Cuong2004/HRP4K'}).\n"
                f"👉 Cannot resume because no previous checkpoint exists yet for '{run_dir.name}'.\n"
                f"👉 To start training from scratch, run Lựa chọn A (bỏ '--resume' và '--weights'):\n"
                f"   hrp4k phase1 --model {manifest.get('model', 'yolo11m') if manifest else 'yolo11m'} --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output {run_dir}"
            )
        raise FileNotFoundError(
            f"Weight file '{weights}' not found locally or on Hugging Face ({hf_repo or 'Cuong2004/HRP4K'}). "
            f"Please verify the file path."
        )

    # Inspect resume checkpoint to synchronize exact embedded configuration
    if resume and Path(resolved_weights).is_file():
        try:
            import torch
            ckpt_dict = torch.load(str(resolved_weights), map_location="cpu", weights_only=False)
            ckpt_args = ckpt_dict.get("train_args", {})
            ckpt_imgsz = ckpt_args.get("imgsz")
            ckpt_epoch = ckpt_dict.get("epoch", 0)
            if ckpt_imgsz:
                if str(image_size).strip().lower() in {"original", "4k", "native"}:
                    actual_imgsz = 3840
                elif str(image_size) in {"1280", "None", ""} and (ckpt_imgsz == 3840 or ckpt_imgsz == [2176, 3840] or ckpt_imgsz == (2176, 3840)):
                    actual_imgsz = 3840
                else:
                    actual_imgsz = 3840 if (isinstance(ckpt_imgsz, (list, tuple)) and 3840 in ckpt_imgsz) else ckpt_imgsz
                print(f"[Resume Auto-Sync] Native checkpoint resolution: imgsz={actual_imgsz} (Resuming from Epoch {ckpt_epoch + 1})")
        except Exception as exc:
            print(f"[Resume Warning] Could not inspect checkpoint metadata: {exc}")
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

    # Build gradient accumulation candidates: target batch falls back as 16 -> 8 -> 4 -> 2 -> 1 on OOM
    target_batch = max(1, batch)
    candidate_batches: list[int] = []
    for b in [target_batch, 16, 8, 4, 2, 1]:
        if b <= target_batch and b not in candidate_batches:
            candidate_batches.append(b)

    result = None
    actual_batch = candidate_batches[0]

    for current_batch in candidate_batches:
        accumulate_steps = max(1, round(target_batch / current_batch))
        effective_batch = current_batch * accumulate_steps
        print(f"\n[Training Engine] Launching batch={current_batch} (Target: {target_batch}, Gradient Accumulation: {accumulate_steps}x -> Effective Batch: {effective_batch})")

        try:
            model = YOLO(str(resolved_weights))
            if syncer.enabled:
                model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

            result = model.train(
                data=str(dataset_yaml),
                epochs=actual_epochs,
                imgsz=actual_imgsz,
                batch=current_batch,
                nbs=target_batch,
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
            actual_batch = current_batch
            break
        except (RuntimeError, Exception) as exc:
            err_msg = str(exc).lower()
            if "out of memory" in err_msg or "cuda out of memory" in err_msg or "cuda oom" in err_msg:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                next_batches = [b for b in candidate_batches if b < current_batch]
                if next_batches:
                    next_batch = next_batches[0]
                    next_accumulate = max(1, round(target_batch / next_batch))
                    print(f"\n⚠️ [CUDA OOM Detected] Batch size {current_batch} exceeded GPU VRAM.")
                    print(f"🔄 Auto-Fallback: Reducing batch size to {next_batch} with gradient accumulation {next_accumulate}x (Effective batch preserved at {target_batch})...\n")
                    continue
                else:
                    raise RuntimeError(f"CUDA Out Of Memory occurred even with batch=1. Please reduce image resolution or free GPU memory.") from exc
            else:
                raise exc

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
                batch=actual_batch,
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

    accumulate_steps = max(1, round(target_batch / actual_batch))
    config["target_batch"] = target_batch
    config["actual_batch"] = actual_batch
    config["gradient_accumulation_steps"] = accumulate_steps
    config["effective_batch"] = actual_batch * accumulate_steps
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

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
        "target_batch": target_batch,
        "actual_batch": actual_batch,
        "gradient_accumulation_steps": accumulate_steps,
        "effective_batch": actual_batch * accumulate_steps,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metrics": val_metrics,
        "eval_confidence": eval_confidence,
    }
