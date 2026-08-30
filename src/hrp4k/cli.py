from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config.resolver import resolve, to_dict
from .config.validation import validate
from .data.audit import analyze_dataset
from .data.paths import ensure_dataset
from .data.views import prepare_dataset_view
from .detectors.registry import BASELINE_PRESETS, DETECTOR_STATUS, get_baseline_preset
from .diagnostics.diagnostics import diagnose
from .evaluation.coco import evaluate_files
from .infra.hashing import experiment_id
from .methods.base import METHOD_REGISTRY, METHOD_STATUS
from .phases.phase_1 import run_phase_1, OFFICIAL_MODELS
from .phases.phase_2 import run_phase_2, RUNNABLE_METHODS
from .phases.pipeline import run_smoke_pipeline
from .protocol.gates import preflight


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
    parser = argparse.ArgumentParser(prog="hrp4k", description="HRP4K Phase 0–3 Unified Benchmark CLI")
    parser.add_argument("--version", action="version", version=f"hrp4k {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    # Data Setup: Auto-detect Kaggle input or download from HF
    setup = commands.add_parser("setup-data", help="Auto-link dataset from Kaggle input or download from Hugging Face")
    setup.add_argument("--data", type=_path, default=Path("HRP4K"))

    # Push to Hugging Face
    for cmd_name in ("push-hf", "upload-hf"):
        push_hf = commands.add_parser(cmd_name, help="Push checkpoints and evaluation outputs to Hugging Face Hub")
        push_hf.add_argument("--token", required=True, help="Hugging Face write access token")
        push_hf.add_argument("--repo", default="Cuong2004/HRP4K", help="Target Hugging Face repository ID (default: Cuong2004/HRP4K)")
        push_hf.add_argument("--path", type=_path, default=Path("outputs"), help="Local directory or file path to upload (default: outputs)")
        push_hf.add_argument("--repo-type", choices=["dataset", "model", "space"], default="dataset", help="Hugging Face repo type")
        push_hf.add_argument("--path-in-repo", help="Target folder path inside repository (default: same as folder name)")

    # Phase 0: Dataset Audit & Analysis
    for cmd_name in ("phase0", "analyze"):
        analyze = commands.add_parser(cmd_name, help="Phase 0 dataset integrity and statistics")
        analyze.add_argument("--data", type=_path, default=Path("HRP4K"))
        analyze.add_argument("--output", type=_path, default=Path("outputs/phase0"))
        analyze.add_argument("--quality-samples", type=int, default=12)

    for name, help_text, default_output, defaults in (
        ("prepare-smoke", "Create a deterministic smoke dataset view using symlinks", Path("outputs/smoke/dataset"), (24, 12, 12)),
        ("prepare-dataset", "Create an all-available dataset view using symlinks", Path("outputs/local_dataset"), (None, None, None)),
    ):
        prepare = commands.add_parser(name, help=help_text)
        prepare.add_argument("--data", type=_path, default=Path("HRP4K"))
        prepare.add_argument("--output", type=_path, default=default_output)
        prepare.add_argument("--train-limit", type=int, default=defaults[0])
        prepare.add_argument("--valid-limit", type=int, default=defaults[1])
        prepare.add_argument("--test-limit", type=int, default=defaults[2])
        prepare.add_argument("--seed", type=int, default=42)

    # Phase 1: Baseline Training (Large-Size 1280 or original 4K)
    model_choices = list(BASELINE_PRESETS) + ["all"]
    for cmd_name in ("phase1", "train"):
        train = commands.add_parser(cmd_name, help="Phase 1 Baseline training (supports large image sizes and multiple models)")
        train.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml"))
        train.add_argument("--weights", type=_path, help="Path to weights or preset name")
        train.add_argument("--model", "--preset", dest="model", choices=model_choices, default="yolo11m", help="Select model to train ('all' trains all 6 baseline models)")
        train.add_argument("--output", type=_path, default=Path("outputs/phase1_runs"))
        train.add_argument("--smoke", action="store_true", help="Fast smoke test (1-2 epochs)")
        train.add_argument("--allow-full", action="store_true", help="Explicitly authorize a full 150-epoch training run")
        train.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default: 150)")
        train.add_argument("--imgsz", type=parse_imgsz, default=1280, help="Input image size (e.g. 1280, 640, or 'original' for native 4K 3840)")
        train.add_argument("--batch", type=int, default=16, help="Training batch size")
        train.add_argument("--device", help="CUDA device index or 'cpu'")
        train.add_argument("--seed", type=int, default=42)
        train.add_argument("--confidence", type=float, default=0.001, help="Validation/test evaluation confidence threshold (default: 0.001)")
        train.add_argument("--resume", action="store_true", help="Resume training from last saved checkpoint")
        train.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True, help="Enable rectangular training to reduce useless padding VRAM (default: True)")
        train.add_argument("--hf-repo", help="Target Hugging Face repository for auto-syncing checkpoints (default: from .env or Cuong2004/HRP4K)")
        train.add_argument("--hf-token", help="Hugging Face write access token (default: from .env or HF_TOKEN)")
        train.add_argument("--no-hf-sync", action="store_true", help="Disable background Hugging Face checkpoint syncing")

    # Phase 2: High-Resolution Inference & Resolution Allocation
    method_choices = list(METHOD_REGISTRY) + ["all"]
    for cmd_name in ("phase2", "predict"):
        predict = commands.add_parser(cmd_name, help="Phase 2 High-Resolution Inference and Slicing")
        predict.add_argument("--data", type=_path, default=Path("HRP4K"))
        predict.add_argument("--split", choices=["train", "valid", "test"], default="test")
        predict.add_argument("--detector", "--model", dest="detector", choices=model_choices, default="yolo11m")
        predict.add_argument("--weights", type=_path, help="Path to trained checkpoint (e.g. best.pt)")
        predict.add_argument("--output", type=_path, default=Path("outputs/phase2_predictions.json"))
        predict.add_argument("--method", choices=method_choices, default="resize", help="Resolution method ('all' runs resize, sliced-nms, perspective-grid, sahi)")
        predict.add_argument("--limit", type=int, help="Optional image limit for fast evaluation")
        predict.add_argument("--imgsz", type=parse_imgsz, default=640, help="Detector input image size (e.g. 640, 1280, or 'original' for native 4K)")
        predict.add_argument("--confidence", type=float, default=0.05, help="Detection confidence threshold")
        predict.add_argument("--tile-size", type=int, default=960, help="Tile size for slicing methods (default: 960)")
        predict.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio for slicing (default: 0.2)")
        predict.add_argument("--device", help="CUDA device or 'cpu'")
        predict.add_argument("--warmup", type=int, default=20)
        predict.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
        predict.add_argument("--evaluate", action="store_true", default=True, help="Automatically evaluate COCO metrics after prediction")
        predict.add_argument("--ground-truth", type=_path, help="Optional path to ground truth JSON (default: <data>/<split>.json)")
        predict.add_argument("--hf-repo", help="Target Hugging Face repository for uploading predictions and metrics")
        predict.add_argument("--hf-token", help="Hugging Face write access token")
        predict.add_argument("--hf-sync", action="store_true", help="Auto-upload predictions and evaluated metrics to Hugging Face")

    # Phase 3 / Evaluation
    for cmd_name in ("phase3", "evaluate"):
        evaluate = commands.add_parser(cmd_name, help="Unified COCO/scale/FPPI evaluator")
        evaluate.add_argument("--ground-truth", type=_path, required=True)
        evaluate.add_argument("--predictions", type=_path, required=True)
        evaluate.add_argument("--output", type=_path, required=True)
        evaluate.add_argument("--confidence", type=float, default=0.25)
        evaluate.add_argument("--hf-repo", help="Target Hugging Face repository")
        evaluate.add_argument("--hf-token", help="Hugging Face write access token")
        evaluate.add_argument("--hf-sync", action="store_true", help="Auto-upload final benchmark artifacts to Hugging Face")

    diagnostic = commands.add_parser("diagnose", help="Phase 3 diagnostics from saved predictions")
    diagnostic.add_argument("--ground-truth", type=_path, required=True)
    diagnostic.add_argument("--predictions", type=_path, nargs="+", required=True)
    diagnostic.add_argument("--output", type=_path, default=Path("outputs/phase3"))
    diagnostic.add_argument("--hf-repo", help="Target Hugging Face repository")
    diagnostic.add_argument("--hf-token", help="Hugging Face write access token")
    diagnostic.add_argument("--hf-sync", action="store_true", help="Auto-upload diagnostic artifacts and report to Hugging Face")

    commands.add_parser("status", help="Show detector presets and Phase 2 reproduction status")

    check = commands.add_parser("preflight", help="Verify dataset identity, integrity and runtime dependencies")
    check.add_argument("--data", type=_path, default=Path("HRP4K"))
    check.add_argument("--output", type=_path)
    check.add_argument("--weights", type=_path)
    check.add_argument("--device")
    check.add_argument("--require-official", action="store_true")



    # Prepare Patch Dataset (Crop Before Training 640)
    patches = commands.add_parser("prepare-patches", help="Generate patch dataset from 4K images for Patch-Based training")
    patches.add_argument("--data", type=_path, default=Path("HRP4K"))
    patches.add_argument("--tile-size", type=int, default=640, help="Tile size for patches (default: 640)")
    patches.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio between patches (default: 0.2)")
    patches.add_argument("--bg-ratio", type=float, default=0.20, help="Background-to-positive patches ratio (default: 0.20)")
    patches.add_argument("--min-visibility", type=float, default=0.25, help="Minimum visible fraction to keep cropped box")
    patches.add_argument("--output", type=_path, default=Path("outputs/dataset_patches_640"))

    # Prepare Warped Dataset (ZoomDet Warped Training 640)
    warped = commands.add_parser("prepare-warped", help="Generate 2D continuous deformation warped dataset for ZoomDet training")
    warped.add_argument("--data", type=_path, default=Path("HRP4K"))
    warped.add_argument("--canvas-size", type=int, default=640, help="Canvas size for warped images (default: 640)")
    warped.add_argument("--horizon-ratio", type=float, default=0.40, help="Horizon boundary ratio for road expansion (default: 0.40)")
    warped.add_argument("--output", type=_path, default=Path("outputs/dataset_zoomdet_640"))

    # Config Subcommands (Upgrade 3.0)
    config_parser = commands.add_parser("config", help="Inspect and validate modular configurations")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)

    config_show = config_sub.add_parser("show", help="Show resolved configuration")
    config_show.add_argument("--config", type=_path)
    config_show.add_argument("--detector", choices=["yolov5m", "yolov5m_compat", "yolov5m_official", "yolov8m", "yolo11m", "rt_detr_v1", "rt_detr_v2", "rtdetr", "d_fine", "dfine"])
    config_show.add_argument("--method", choices=["resize", "sliced_nms", "sliced-nms", "sahi", "perspective_grid", "perspective-grid", "learned_tpp", "two-plane-prior"])
    config_show.add_argument("--profile", choices=["smoke", "research", "benchmark"])

    config_val = config_sub.add_parser("validate", help="Validate configuration before execution")
    config_val.add_argument("--config", type=_path)
    config_val.add_argument("--detector", choices=["yolov5m", "yolov5m_compat", "yolov5m_official", "yolov8m", "yolo11m", "rt_detr_v1", "rt_detr_v2", "rtdetr", "d_fine", "dfine"])
    config_val.add_argument("--method", choices=["resize", "sliced_nms", "sliced-nms", "sahi", "perspective_grid", "perspective-grid", "learned_tpp", "two-plane-prior"])
    config_val.add_argument("--profile", choices=["smoke", "research", "benchmark"])

    # Experiment Subcommand
    exp_parser = commands.add_parser("experiment", help="Experiment utilities")
    exp_sub = exp_parser.add_subparsers(dest="experiment_command", required=True)
    exp_id = exp_sub.add_parser("id", help="Compute deterministic experiment ID")
    exp_id.add_argument("--config", type=_path, required=True)

    # Config-driven Run
    run = commands.add_parser("run", help="Run an inference experiment from YAML/JSON config")
    run.add_argument("--config", type=_path, required=True)

    # End-to-end Smoke
    smoke = commands.add_parser("run-smoke", help="Run Phase 0–3 end-to-end smoke pipeline")
    smoke.add_argument("--data", type=_path, default=Path("HRP4K"))
    smoke.add_argument("--output", type=_path, default=Path("outputs/smoke"))
    smoke.add_argument("--weights", type=_path, default=Path("yolo11n.pt"))
    smoke.add_argument("--train-limit", type=int, default=2, help="Number of training samples for smoke test (default: 2)")
    smoke.add_argument("--eval-limit", type=int, default=1, help="Number of evaluation samples for smoke test (default: 1)")
    smoke.add_argument("--imgsz", type=int, default=256, help="Image size for smoke test (default: 256)")
    smoke.add_argument("--device", default="cpu", help="Device for smoke test (default: cpu)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup-data":
        path, source = ensure_dataset(args.data, auto_download=True)
        _print({"status": "ready", "dataset_path": str(path), "source": source})
    elif args.command in {"push-hf", "upload-hf"}:
        from .infra.upload import upload_to_hf
        _print(upload_to_hf(
            repo_id=args.repo,
            local_path=args.path,
            token=args.token,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo,
        ))
    elif args.command in {"phase0", "analyze"}:
        _print(analyze_dataset(args.data, args.output, args.quality_samples))
    elif args.command == "prepare-patches":
        from .data.patches import create_patch_dataset
        _print(create_patch_dataset(
            data_dir=args.data,
            output_dir=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            bg_ratio=args.bg_ratio,
            min_visibility=args.min_visibility,
        ))
    elif args.command == "prepare-warped":
        from .data.warped import create_warped_dataset
        _print(create_warped_dataset(
            data_dir=args.data,
            output_dir=args.output,
            canvas_size=args.canvas_size,
            horizon_ratio=args.horizon_ratio,
        ))
    elif args.command in {"prepare-smoke", "prepare-dataset"}:
        _print(prepare_dataset_view(args.data, args.output, args.train_limit, args.valid_limit, args.test_limit, args.seed))
    elif args.command in {"phase1", "train"}:
        resolved_dataset = args.dataset
        if not resolved_dataset.exists():
            data_path, _ = ensure_dataset(auto_download=True)
            view_out = resolved_dataset.parent if resolved_dataset.parent != Path(".") else Path("outputs/full_dataset")
            prepare_dataset_view(data_path, view_out)
            resolved_dataset = view_out / "dataset.yaml"

        _print(run_phase_1(
            dataset_yaml=resolved_dataset,
            weights=args.weights,
            output_dir=args.output,
            smoke=args.smoke,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch=args.batch,
            device=args.device,
            allow_full=args.allow_full,
            preset=args.model,
            seed=args.seed,
            confidence=args.confidence,
            resume=args.resume,
            rect=args.rect,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            hf_sync=not args.no_hf_sync,
        ))

    elif args.command in {"phase2", "predict"}:
        resolved_weights = args.weights
        if not resolved_weights:
            preset_dict = get_baseline_preset(args.detector) if args.detector != "all" else None
            resolved_weights = Path(preset_dict["weights"]) if preset_dict else Path("yolo11m.pt")
        
        _print(run_phase_2(
            data_dir=args.data,
            split=args.split,
            weights=resolved_weights,
            output_path=args.output,
            method=args.method,
            limit=args.limit,
            image_size=args.imgsz,
            confidence=args.confidence,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
            warmup=args.warmup,
            detector_name=args.detector,
            precision=args.precision,
            evaluate_after=args.evaluate,
            ground_truth=args.ground_truth,
            hf_repo=getattr(args, "hf_repo", None),
            hf_token=getattr(args, "hf_token", None),
            hf_sync=getattr(args, "hf_sync", False),
        ))
    elif args.command in {"phase3", "evaluate"}:
        res = evaluate_files(args.ground_truth, args.predictions, args.output, args.confidence)
        if getattr(args, "hf_sync", False):
            from .infra.upload import upload_to_hf, get_hf_credentials
            token, repo, rtype = get_hf_credentials(args.hf_token, args.hf_repo)
            if token and args.output.exists():
                try:
                    upload_to_hf(repo_id=repo, local_path=args.output, token=token, repo_type=rtype, path_in_repo=f"metrics/{args.output.name}")
                except Exception as e:
                    print(f"[Cloud Warning] Failed to upload Phase 3 metrics to HF: {e}")
        _print(res)
    elif args.command == "diagnose":
        res = diagnose(args.ground_truth, args.predictions, args.output)
        if getattr(args, "hf_sync", False):
            from .infra.upload import upload_to_hf, get_hf_credentials
            token, repo, rtype = get_hf_credentials(args.hf_token, args.hf_repo)
            if token and args.output.exists():
                try:
                    upload_to_hf(repo_id=repo, local_path=args.output, token=token, repo_type=rtype, path_in_repo=f"reports/{args.output.name}")
                except Exception as e:
                    print(f"[Cloud Warning] Failed to upload diagnostic report to HF: {e}")
        _print(res)
    elif args.command == "status":
        _print({
            "baseline_presets": BASELINE_PRESETS,
            "detectors": DETECTOR_STATUS,
            "methods": METHOD_REGISTRY,
            "method_summary": METHOD_STATUS,
        })
    elif args.command == "preflight":
        result = preflight(
            args.data, output_dir=args.output, weights=args.weights,
            device=args.device, require_official=args.require_official,
        )
        _print(result)
        if result["status"] != "pass":
            return 2
    elif args.command == "config":
        resolved = resolve(
            config_path=args.config, detector=args.detector,
            method=args.method, profile=args.profile,
        )
        if args.config_command == "show":
            _print(to_dict(resolved))
        elif args.config_command == "validate":
            errors = validate(resolved)
            if errors:
                _print({"status": "invalid", "errors": errors})
                return 1
            _print({"status": "valid", "schema_version": resolved.schema_version})
    elif args.command == "experiment":
        if args.experiment_command == "id":
            resolved = resolve(config_path=args.config)
            _print({"experiment_id": experiment_id(to_dict(resolved))})
    elif args.command == "run":
        resolved = resolve(config_path=args.config)
        errors = validate(resolved)
        if errors:
            raise ValueError("Invalid configuration:\n" + "\n".join(errors))
        detector = resolved.detector
        method = resolved.method
        runtime = resolved.runtime
        output = resolved.output
        if detector.name in {"d-fine", "dfine"}:
            raise RuntimeError(f"{detector.name} requires its official external runtime and canonical export contract")
        if method.name in METHOD_REGISTRY and METHOD_REGISTRY[method.name]["status"] == "external-required":
            raise RuntimeError(f"{method.name} requires its paper-faithful external training/runtime")
        if not output.predictions:
            raise ValueError("output.predictions must be specified in the configuration")
        _print(run_phase_2(
            data_dir=Path(resolved.dataset.root),
            split=resolved.dataset.split,
            weights=Path(detector.checkpoint),
            output_path=Path(output.predictions),
            method=method.name,
            limit=runtime.limit,
            image_size=int(detector.input_size),
            confidence=float(detector.confidence),
            tile_size=method.tile_size,
            overlap=method.overlap,
            device=runtime.device or detector.device,
            warmup=runtime.warmup,
            detector_name=detector.name,
            precision=runtime.precision,
        )["summary"])
    elif args.command == "run-smoke":
        _print(run_smoke_pipeline(
            data_dir=args.data, output_dir=args.output, weights=args.weights,
            train_limit=args.train_limit, eval_limit=args.eval_limit,
            image_size=args.imgsz, device=args.device,
        ))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
