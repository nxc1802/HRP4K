from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .detectors.registry import BASELINE_PRESETS, DETECTOR_STATUS, get_baseline_preset
from .experiments.registry import EXPERIMENT_MATRIX, resolve_experiment, list_experiments


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_imgsz(value: Any) -> int | tuple[int, int]:
    val = str(value).strip().lower()
    if val in {"original", "4k", "native"}:
        return 3840  # Native 4K UHD max dimension with rect=True (3840x2176)
    if "," in val:
        parts = [int(p.strip()) for p in val.split(",")]
        return (parts[0], parts[1])
    if "x" in val:
        parts = [int(p.strip()) for p in val.split("x")]
        return (parts[0], parts[1])
    return int(val)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrp4k", description="HRP4K Research Experiment CLI")
    parser.add_argument("--version", action="version", version=f"hrp4k {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    # -----------------------------------------------------------------------
    # setup — One-command environment setup
    # -----------------------------------------------------------------------
    setup = commands.add_parser("setup", help="One-command setup: dependencies, dataset, environment, HF credentials")
    setup.add_argument("--data", type=_path, default=Path("HRP4K"))
    setup.add_argument("--skip-dataset", action="store_true", help="Skip dataset download")

    # -----------------------------------------------------------------------
    # experiment — One-command experiment execution
    # -----------------------------------------------------------------------
    experiment_choices = sorted(EXPERIMENT_MATRIX.keys())
    exp = commands.add_parser("experiment", help="Run an official experiment by name")
    exp.add_argument("name", choices=experiment_choices + ["list"],
                     help="Experiment name (e.g., yolo11m-resolution-4k) or 'list' to show all")
    exp.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml"))
    exp.add_argument("--data", type=_path, default=Path("HRP4K"), help="Raw dataset directory")
    exp.add_argument("--output", type=_path, default=Path("outputs/experiments"))
    exp.add_argument("--device", help="CUDA device index or 'cpu'")
    exp.add_argument("--batch", type=int, help="Override physical batch size (e.g. --batch 16 on 100GB GPU)")
    exp.add_argument("--accumulation", type=int, help="Override gradient accumulation steps (default: auto)")
    exp.add_argument("--epochs", type=int, help="Override number of training epochs (e.g. --epochs 50)")
    exp.add_argument("--patience", type=int, help="Override early stopping patience (e.g. --patience 10)")
    exp.add_argument("--dry-run", action="store_true", help="Show resolved config without executing")
    exp.add_argument("--frozen-checkpoint", type=_path, help="Frozen checkpoint for slicing experiments")
    exp.add_argument("--hf-repo", help="Target Hugging Face repository")
    exp.add_argument("--hf-token", help="Hugging Face write access token")
    exp.add_argument("--no-hf-sync", action="store_true", help="Disable background Hugging Face checkpoint syncing during training")

    # -----------------------------------------------------------------------
    # setup-data — Dataset-only setup
    # -----------------------------------------------------------------------
    setup_data = commands.add_parser("setup-data", help="Auto-link dataset from Kaggle input or download from Hugging Face")
    setup_data.add_argument("--data", type=_path, default=Path("HRP4K"))

    # -----------------------------------------------------------------------
    # push-hf — Upload to Hugging Face
    # -----------------------------------------------------------------------
    for cmd_name in ("push-hf", "upload-hf"):
        push_hf = commands.add_parser(cmd_name, help="Push checkpoints and evaluation outputs to Hugging Face Hub")
        push_hf.add_argument("--token", required=True, help="Hugging Face write access token")
        push_hf.add_argument("--repo", default="Cuong2004/HRP4K", help="Target Hugging Face repository ID")
        push_hf.add_argument("--path", type=_path, default=Path("outputs"), help="Local directory or file path to upload")
        push_hf.add_argument("--repo-type", choices=["dataset", "model", "space"], default="dataset")
        push_hf.add_argument("--path-in-repo", help="Target folder path inside repository")

    # -----------------------------------------------------------------------
    # prepare-smoke / prepare-dataset — Dataset views
    # -----------------------------------------------------------------------
    for name, help_text, default_output, defaults in (
        ("prepare-smoke", "Create a deterministic smoke dataset view using symlinks", Path("outputs/smoke/dataset"), (24, 12, 12)),
        ("prepare-dataset", "Create an all-available dataset view using symlinks", Path("outputs/full_dataset"), (None, None, None)),
    ):
        prepare = commands.add_parser(name, help=help_text)
        prepare.add_argument("--data", type=_path, default=Path("HRP4K"))
        prepare.add_argument("--output", type=_path, default=default_output)
        prepare.add_argument("--train-limit", type=int, default=defaults[0])
        prepare.add_argument("--valid-limit", type=int, default=defaults[1])
        prepare.add_argument("--test-limit", type=int, default=defaults[2])
        prepare.add_argument("--seed", type=int, default=42)

    # -----------------------------------------------------------------------
    # train — Direct training (internal, used by experiment command)
    # -----------------------------------------------------------------------
    model_choices = list(BASELINE_PRESETS) + ["all"]
    train = commands.add_parser("train", help="Direct baseline training (prefer 'experiment' command)")
    train.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml"))
    train.add_argument("--weights", type=_path, help="Path to weights or preset name")
    train.add_argument("--model", "--preset", dest="model", choices=model_choices, default="yolo11m")
    train.add_argument("--output", type=_path, default=Path("outputs/phase1_runs"))
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--allow-full", action="store_true")
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--imgsz", type=parse_imgsz, default=1280)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", help="CUDA device index or 'cpu'")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--confidence", type=float, default=0.001)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--hf-repo", help="Target Hugging Face repository")
    train.add_argument("--hf-token", help="Hugging Face write access token")
    train.add_argument("--no-hf-sync", action="store_true")

    # -----------------------------------------------------------------------
    # predict — Direct inference (internal, used by experiment command)
    # -----------------------------------------------------------------------
    from .methods.base import METHOD_REGISTRY
    method_choices = list(METHOD_REGISTRY) + ["all"]
    predict = commands.add_parser("predict", help="Run inference with spatial methods (prefer 'experiment' command)")
    predict.add_argument("--data", type=_path, default=Path("HRP4K"))
    predict.add_argument("--split", choices=["train", "valid", "test"], default="test")
    predict.add_argument("--detector", "--model", dest="detector", choices=model_choices, default="yolo11m")
    predict.add_argument("--weights", type=_path, help="Path to trained checkpoint")
    predict.add_argument("--output", type=_path, default=Path("outputs/predictions.json"))
    predict.add_argument("--method", choices=method_choices, default="resize")
    predict.add_argument("--limit", type=int)
    predict.add_argument("--imgsz", type=parse_imgsz, default=640)
    predict.add_argument("--confidence", type=float, default=0.05)
    predict.add_argument("--tile-size", type=int, default=960)
    predict.add_argument("--overlap", type=float, default=0.2)
    predict.add_argument("--device", help="CUDA device or 'cpu'")
    predict.add_argument("--warmup", type=int, default=20)
    predict.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    predict.add_argument("--evaluate", action="store_true", default=True)
    predict.add_argument("--ground-truth", type=_path)
    predict.add_argument("--hf-repo")
    predict.add_argument("--hf-token")
    predict.add_argument("--hf-sync", action="store_true")

    # -----------------------------------------------------------------------
    # evaluate — Standalone evaluation
    # -----------------------------------------------------------------------
    evaluate = commands.add_parser("evaluate", help="Unified COCO/scale/FPPI evaluator")
    evaluate.add_argument("--ground-truth", type=_path, required=True)
    evaluate.add_argument("--predictions", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True)
    evaluate.add_argument("--confidence", type=float, default=0.25)

    # -----------------------------------------------------------------------
    # status — Show available experiments and detector status
    # -----------------------------------------------------------------------
    commands.add_parser("status", help="Show registered experiments and detector status")

    # -----------------------------------------------------------------------
    # preflight — Environment validation
    # -----------------------------------------------------------------------
    check = commands.add_parser("preflight", help="Verify dataset identity, integrity and runtime dependencies")
    check.add_argument("--data", type=_path, default=Path("HRP4K"))
    check.add_argument("--output", type=_path)
    check.add_argument("--weights", type=_path)
    check.add_argument("--device")

    # -----------------------------------------------------------------------
    # run-smoke — End-to-end smoke test
    # -----------------------------------------------------------------------
    smoke = commands.add_parser("run-smoke", help="Run end-to-end smoke pipeline")
    smoke.add_argument("--data", type=_path, default=Path("HRP4K"))
    smoke.add_argument("--output", type=_path, default=Path("outputs/smoke"))
    smoke.add_argument("--weights", type=_path, default=Path("yolo11m.pt"))
    smoke.add_argument("--train-limit", type=int, default=2)
    smoke.add_argument("--eval-limit", type=int, default=1)
    smoke.add_argument("--imgsz", type=int, default=256)
    smoke.add_argument("--device", default="cpu")

    # -----------------------------------------------------------------------
    # smoke-proposed — Proposed P2 Feasibility smoke test
    # -----------------------------------------------------------------------
    commands.add_parser("smoke-proposed", help="Run local RT-DETR-L + P2 Feasibility smoke test (shape check, loss, fusion)")

    # -----------------------------------------------------------------------
    # inspect-checkpoint — View metadata and summary of any checkpoint
    # -----------------------------------------------------------------------
    insp = commands.add_parser("inspect-checkpoint", help="Inspect checkpoint contents and metadata")
    insp.add_argument("path", type=_path, help="Path to checkpoint file (.pt)")

    # -----------------------------------------------------------------------
    # eval-proposed — Evaluate a trained P2 checkpoint
    # -----------------------------------------------------------------------
    eval_p2 = commands.add_parser("eval-proposed", help="Evaluate trained Frozen RT-DETR-L + P2 checkpoint on test set")
    eval_p2.add_argument("--checkpoint", type=_path, required=True, help="Path to best_p2.pt checkpoint")
    eval_p2.add_argument("--weights", type=_path, default=Path("rtdetr-l.pt"), help="Base RT-DETR-L weights")
    eval_p2.add_argument("--data", type=_path, default=Path("HRP4K"), help="Raw dataset directory")
    eval_p2.add_argument("--imgsz", type=parse_imgsz, default=1920, help="Evaluation image size")
    eval_p2.add_argument("--confidence", type=float, default=0.001, help="Evaluation confidence threshold")
    eval_p2.add_argument("--output", type=_path, help="Directory to save test_metrics.json")
    eval_p2.add_argument("--device", help="CUDA device index or 'cpu'")
    eval_p2.add_argument("--hf-repo", default="Cuong2004/HRP4K", help="Hugging Face repository")
    eval_p2.add_argument("--hf-token", help="Hugging Face write token")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ===================================================================
    # setup
    # ===================================================================
    if args.command == "setup":
        from .data.paths import ensure_dataset
        from .data.views import prepare_dataset_view
        from .protocol.gates import preflight

        print("=" * 60)
        print("HRP4K Setup")
        print("=" * 60)

        # Step 1: Check Python/runtime
        print("\n[1/5] Checking runtime environment...")
        from .infra.environment import environment_snapshot
        env = environment_snapshot()
        _print({"python": env.get("python_version"), "cuda": env.get("cuda_available")})

        # Step 2: Check HF credentials
        print("\n[2/5] Checking Hugging Face credentials...")
        from .infra.upload import get_hf_credentials
        token, repo, rtype = get_hf_credentials()
        hf_ok = bool(token)
        print(f"  HF Token: {'✅ Found' if hf_ok else '❌ Not found (set HF_TOKEN in .env)'}")
        print(f"  HF Repo:  {repo}")

        # Step 3: Dataset
        if not args.skip_dataset:
            print("\n[3/5] Preparing dataset...")
            path, source = ensure_dataset(args.data, auto_download=True)
            print(f"  Dataset: {path} (source: {source})")

            # Step 4: Create dataset view
            print("\n[4/5] Creating dataset view...")
            view_out = Path("outputs/full_dataset")
            prepare_dataset_view(path, view_out)
            print(f"  Dataset view: {view_out}")
        else:
            print("\n[3/5] Skipping dataset download (--skip-dataset)")
            print("[4/5] Skipped")

        # Step 5: Preflight
        print("\n[5/5] Running preflight checks...")
        try:
            result = preflight(args.data)
            status = result.get("status", "unknown")
            print(f"  Preflight: {'✅ Pass' if status == 'pass' else f'⚠️ {status}'}")
        except Exception as exc:
            print(f"  Preflight: ⚠️ {exc}")

        print("\n" + "=" * 60)
        print("Setup complete! Run experiments with:")
        print("  hrp4k experiment yolo11m-resolution-640")
        print("  hrp4k experiment list")
        print("=" * 60)

    # ===================================================================
    # experiment
    # ===================================================================
    elif args.command == "experiment":
        if args.name == "list":
            experiments = list_experiments()
            print(f"\n{'Name':<40} {'Phase':<12} {'Detector':<12} {'Resolution':<6} {'Method'}")
            print("-" * 90)
            for exp in experiments:
                method = exp.get("method") or "—"
                print(f"{exp['name']:<40} {exp['phase']:<12} {exp['detector']:<12} {exp['resolution']:<6} {method}")
            return 0

        config = resolve_experiment(args.name)
        if getattr(args, "batch", None):
            config.batch = args.batch
            if getattr(args, "accumulation", None):
                config.accumulation = args.accumulation
            else:
                config.accumulation = max(1, config.effective_batch // config.batch)
            config.effective_batch = config.batch * config.accumulation
        if getattr(args, "epochs", None) is not None:
            config.epochs = args.epochs
        if getattr(args, "patience", None) is not None:
            config.patience = args.patience

        # Ensure dataset is ready
        dataset_yaml = args.dataset
        if not dataset_yaml.exists():
            from .data.paths import ensure_dataset
            from .data.views import prepare_dataset_view
            data_path, _ = ensure_dataset(args.data, auto_download=True)
            view_out = dataset_yaml.parent if dataset_yaml.parent != Path(".") else Path("outputs/full_dataset")
            prepare_dataset_view(data_path, view_out)
            dataset_yaml = view_out / "dataset.yaml"

        if config.phase == "resolution":
            from .experiments.resolution import run_resolution_experiment
            result = run_resolution_experiment(
                config=config,
                dataset_yaml=dataset_yaml,
                output_dir=args.output,
                hf_repo=args.hf_repo,
                hf_token=args.hf_token,
                hf_sync=not args.no_hf_sync,
                dry_run=args.dry_run,
            )
            _print(result)

        elif config.phase == "slicing":
            from .experiments.slicing import run_slicing_experiment
            result = run_slicing_experiment(
                config=config,
                data_dir=args.data,
                output_dir=args.output,
                frozen_checkpoint=args.frozen_checkpoint,
                hf_repo=args.hf_repo,
                hf_token=args.hf_token,
                dry_run=args.dry_run,
            )
            _print(result)

        elif config.phase == "proposed":
            from .experiments.proposed import run_proposed_experiment
            result = run_proposed_experiment(
                config=config,
                dataset_yaml=dataset_yaml,
                output_dir=args.output,
                hf_repo=args.hf_repo,
                hf_token=args.hf_token,
                hf_sync=not args.no_hf_sync,
                dry_run=args.dry_run,
            )
            _print(result)

        else:
            print(f"Unknown phase: {config.phase}")
            return 1

    # ===================================================================
    # setup-data
    # ===================================================================
    elif args.command == "setup-data":
        from .data.paths import ensure_dataset
        path, source = ensure_dataset(args.data, auto_download=True)
        _print({"status": "ready", "dataset_path": str(path), "source": source})

    # ===================================================================
    # push-hf / upload-hf
    # ===================================================================
    elif args.command in {"push-hf", "upload-hf"}:
        from .infra.upload import upload_to_hf
        _print(upload_to_hf(
            repo_id=args.repo, local_path=args.path, token=args.token,
            repo_type=args.repo_type, path_in_repo=args.path_in_repo,
        ))

    # ===================================================================
    # prepare-smoke / prepare-dataset
    # ===================================================================
    elif args.command in {"prepare-smoke", "prepare-dataset"}:
        from .data.views import prepare_dataset_view
        _print(prepare_dataset_view(args.data, args.output, args.train_limit, args.valid_limit, args.test_limit, args.seed))

    # ===================================================================
    # train
    # ===================================================================
    elif args.command == "train":
        from .data.paths import ensure_dataset
        from .data.views import prepare_dataset_view
        from .phases.phase_1 import run_phase_1

        resolved_dataset = args.dataset
        if not resolved_dataset.exists():
            data_path, _ = ensure_dataset(auto_download=True)
            view_out = resolved_dataset.parent if resolved_dataset.parent != Path(".") else Path("outputs/full_dataset")
            prepare_dataset_view(data_path, view_out)
            resolved_dataset = view_out / "dataset.yaml"

        _print(run_phase_1(
            dataset_yaml=resolved_dataset, weights=args.weights, output_dir=args.output,
            smoke=args.smoke, epochs=args.epochs, image_size=args.imgsz, batch=args.batch,
            device=args.device, allow_full=args.allow_full, preset=args.model, seed=args.seed,
            confidence=args.confidence, resume=args.resume, rect=args.rect,
            hf_repo=args.hf_repo, hf_token=args.hf_token, hf_sync=not args.no_hf_sync,
        ))

    # ===================================================================
    # predict
    # ===================================================================
    elif args.command == "predict":
        from .phases.phase_2 import run_phase_2
        resolved_weights = args.weights
        if not resolved_weights:
            preset_dict = get_baseline_preset(args.detector) if args.detector != "all" else None
            resolved_weights = Path(preset_dict["weights"]) if preset_dict else Path("yolo11m.pt")

        _print(run_phase_2(
            data_dir=args.data, split=args.split, weights=resolved_weights,
            output_path=args.output, method=args.method, limit=args.limit,
            image_size=args.imgsz, confidence=args.confidence, tile_size=args.tile_size,
            overlap=args.overlap, device=args.device, warmup=args.warmup,
            detector_name=args.detector, precision=args.precision,
            evaluate_after=args.evaluate, ground_truth=args.ground_truth,
            hf_repo=getattr(args, "hf_repo", None), hf_token=getattr(args, "hf_token", None),
            hf_sync=getattr(args, "hf_sync", False),
        ))

    # ===================================================================
    # evaluate
    # ===================================================================
    elif args.command == "evaluate":
        from .evaluation.coco import evaluate_files
        _print(evaluate_files(args.ground_truth, args.predictions, args.output, args.confidence))

    # ===================================================================
    # status
    # ===================================================================
    elif args.command == "status":
        experiments = list_experiments()
        _print({
            "detectors": DETECTOR_STATUS,
            "experiments": experiments,
            "total_experiments": len(experiments),
        })

    # ===================================================================
    # preflight
    # ===================================================================
    elif args.command == "preflight":
        from .protocol.gates import preflight
        result = preflight(args.data, output_dir=args.output, weights=args.weights, device=args.device)
        _print(result)
        if result["status"] != "pass":
            return 2

    # ===================================================================
    # run-smoke
    # ===================================================================
    elif args.command == "run-smoke":
        from .phases.pipeline import run_smoke_pipeline
        _print(run_smoke_pipeline(
            data_dir=args.data, output_dir=args.output, weights=args.weights,
            train_limit=args.train_limit, eval_limit=args.eval_limit,
            image_size=args.imgsz, device=args.device,
        ))

    # ===================================================================
    # smoke-proposed
    # ===================================================================
    elif args.command == "smoke-proposed":
        from .experiments.proposed import run_proposed_smoke
        _print(run_proposed_smoke())

    # ===================================================================
    # inspect-checkpoint
    # ===================================================================
    elif args.command == "inspect-checkpoint":
        import torch
        ckpt_path = args.path.resolve()
        if not ckpt_path.is_file():
            print(f"Error: file not found: {ckpt_path}")
            return 1
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("=" * 60)
        print(f"Checkpoint Inspection: {ckpt_path}")
        print("=" * 60)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"  {k:<25}: dict with {len(v)} keys")
                elif hasattr(v, "shape"):
                    print(f"  {k:<25}: tensor of shape {v.shape}")
                else:
                    print(f"  {k:<25}: {v}")
        else:
            print(f"Type: {type(data)}")
        print("=" * 60)
        return 0

    # ===================================================================
    # eval-proposed
    # ===================================================================
    elif args.command == "eval-proposed":
        import os
        import torch
        from .data.paths import resolve_data_dir
        from .experiments.proposed import RTDETRP2Adapter
        from .inference.runner import predict_detector
        from .evaluation.coco import evaluate_files

        ckpt_path = args.checkpoint.resolve()
        if not ckpt_path.is_file():
            print(f"Error: checkpoint file not found at {ckpt_path}")
            return 1

        print("=" * 60)
        print(f"[Proposed Evaluation] Evaluating Checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "unknown")
        mean_loss = ckpt.get("mean_p2_loss", "unknown")
        base_weights = ckpt.get("base_checkpoint", str(args.weights))
        print(f"  Epoch Reached:   {epoch}")
        print(f"  Best Mean Loss:  {mean_loss}")
        print(f"  Base Weights:    {base_weights}")
        print(f"  Image Size:      {args.imgsz}")
        print("=" * 60)

        out_dir = args.output.resolve() if args.output else ckpt_path.parent.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "test_predictions.json"
        metrics_path = out_dir / "test_metrics.json"

        data_dir = resolve_data_dir(args.data)
        test_gt = data_dir / "test.json"
        if not test_gt.is_file():
            for c in (Path("HRP4K/test.json"), Path("test.json"), Path("../HRP4K/test.json")):
                if c.is_file():
                    test_gt = c
                    data_dir = c.parent
                    break

        if not test_gt.is_file():
            print(f"Error: test.json ground truth not found in {data_dir}")
            return 1

        target_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        adapter = RTDETRP2Adapter(
            weights=base_weights,
            category_id=0,
            device=target_device,
            p2_checkpoint=ckpt_path,
        )

        print(f"\n[Inference] Running test set prediction at {args.imgsz}px on {target_device}...")
        pred_summary = predict_detector(
            data_dir=data_dir,
            split="test",
            detector=adapter,
            output_path=pred_path,
            method="resize",
            image_size=args.imgsz if isinstance(args.imgsz, int) else args.imgsz[0],
            confidence=args.confidence,
        )

        print(f"\n[Evaluation] Evaluating against {test_gt}...")
        metrics = evaluate_files(
            gt_path=test_gt,
            prediction_path=pred_path,
            output_path=metrics_path,
            confidence=args.confidence,
        )
        metrics["mean_latency_ms"] = pred_summary.get("mean_end_to_end_latency_ms", 0.0)
        metrics["eval_confidence"] = args.confidence
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print("\n" + "=" * 60)
        print("OFFICIAL TEST EVALUATION RESULTS:")
        print(f"  AP50:          {metrics.get('AP50', 0.0)*100:.2f}%")
        print(f"  AP50-95:       {metrics.get('AP50_95', 0.0)*100:.2f}%")
        print(f"  FPPI Official: {metrics.get('FPPI_official', 0.0):.4f}")
        print(f"  Mean Latency:  {metrics.get('mean_latency_ms', 0.0):.2f} ms")
        print("=" * 60)
        print(f"Metrics saved:     {metrics_path}")
        print(f"Predictions saved: {pred_path}")

        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token:
            from .infra.upload import upload_to_hf
            print(f"\n[Upload] Syncing to Hugging Face repo {args.hf_repo}...")
            upload_to_hf(repo_id=args.hf_repo, local_path=metrics_path, token=token, path_in_repo="experiments/rtdetr-l-proposed-p2-2k/test_metrics.json")
            upload_to_hf(repo_id=args.hf_repo, local_path=pred_path, token=token, path_in_repo="experiments/rtdetr-l-proposed-p2-2k/test_predictions.json")
            upload_to_hf(repo_id=args.hf_repo, local_path=ckpt_path, token=token, path_in_repo="experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt")
            print("Sync complete!")

        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
