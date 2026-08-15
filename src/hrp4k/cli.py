from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .config.resolver import resolve, to_dict
from .config.validation import validate
from .data.audit import analyze_dataset
from .data.views import prepare_dataset_view
from .detectors.registry import BASELINE_PRESETS, DETECTOR_STATUS, get_baseline_preset
from .diagnostics.diagnostics import diagnose
from .evaluation.coco import evaluate_files
from .infra.hashing import experiment_id
from .methods.base import METHOD_REGISTRY, METHOD_STATUS
from .phases.phase_1 import run_phase_1
from .phases.phase_2 import run_phase_2
from .phases.pipeline import run_smoke_pipeline
from .protocol.gates import preflight


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _print(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrp4k", description="HRP4K Phase 0–3 benchmark CLI")
    parser.add_argument("--version", action="version", version=f"hrp4k {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    # Phase 0
    analyze = commands.add_parser("analyze", help="Phase 0 dataset integrity and statistics")
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

    # Phase 1
    train = commands.add_parser("train", help="Phase 1 YOLO benchmark training")
    train.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml"))
    train.add_argument("--weights", type=_path)
    train.add_argument("--preset", choices=[name for name, value in BASELINE_PRESETS.items() if value.get("framework") == "ultralytics"])
    train.add_argument("--output", type=_path)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--allow-full", action="store_true", help="Explicitly authorize a non-smoke training run")
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device")

    # Phase 2
    predict = commands.add_parser("predict", help="Phase 1/2 inference and unified COCO export")
    predict.add_argument("--data", type=_path, default=Path("outputs/smoke/dataset"))
    predict.add_argument("--split", choices=["train", "valid", "test"], default="test")
    predict.add_argument("--detector", choices=["ultralytics", "yolov5m-compat", "yolov5m-official", "yolov8m", "yolo11m", "rt-detr-v1", "rt-detr-v2", "d-fine"], default="ultralytics")
    predict.add_argument("--weights", type=_path, required=True)
    predict.add_argument("--output", type=_path, required=True)
    predict.add_argument("--method", choices=list(METHOD_REGISTRY), default="resize")
    predict.add_argument("--limit", type=int)
    predict.add_argument("--imgsz", type=int, default=320)
    predict.add_argument("--confidence", type=float, default=0.05)
    predict.add_argument("--tile-size", type=int, default=960)
    predict.add_argument("--overlap", type=float, default=0.2)
    predict.add_argument("--device")
    predict.add_argument("--warmup", type=int, default=20)
    predict.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")

    # Phase 3 / Evaluation
    evaluate = commands.add_parser("evaluate", help="Unified COCO/scale/FPPI evaluator")
    evaluate.add_argument("--ground-truth", type=_path, required=True)
    evaluate.add_argument("--predictions", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True)
    evaluate.add_argument("--confidence", type=float, default=0.25)

    diagnostic = commands.add_parser("diagnose", help="Phase 3 diagnostics from saved predictions")
    diagnostic.add_argument("--ground-truth", type=_path, required=True)
    diagnostic.add_argument("--predictions", type=_path, nargs="+", required=True)
    diagnostic.add_argument("--output", type=_path, default=Path("outputs/phase3"))

    commands.add_parser("status", help="Show detector presets and Phase 2 reproduction status")

    check = commands.add_parser("preflight", help="Verify dataset identity, integrity and runtime dependencies")
    check.add_argument("--data", type=_path, default=Path("HRP4K"))
    check.add_argument("--output", type=_path)
    check.add_argument("--weights", type=_path)
    check.add_argument("--device")
    check.add_argument("--require-official", action="store_true")

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
    smoke.add_argument("--train-limit", type=int, default=24)
    smoke.add_argument("--eval-limit", type=int, default=8)
    smoke.add_argument("--imgsz", type=int, default=320)
    smoke.add_argument("--device")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        _print(analyze_dataset(args.data, args.output, args.quality_samples))
    elif args.command in {"prepare-smoke", "prepare-dataset"}:
        _print(prepare_dataset_view(args.data, args.output, args.train_limit, args.valid_limit, args.test_limit, args.seed))
    elif args.command == "train":
        _print(run_phase_1(args.dataset, args.weights, args.output, args.smoke, args.epochs, args.imgsz, args.batch, args.device, args.allow_full, args.preset))
    elif args.command == "predict":
        _print(run_phase_2(
            data_dir=args.data, split=args.split, weights=args.weights, output_path=args.output,
            method=args.method, limit=args.limit, image_size=args.imgsz, confidence=args.confidence,
            tile_size=args.tile_size, overlap=args.overlap, device=args.device, warmup=args.warmup,
            detector_name=args.detector, precision=args.precision,
        )["summary"])
    elif args.command == "evaluate":
        _print(evaluate_files(args.ground_truth, args.predictions, args.output, args.confidence))
    elif args.command == "diagnose":
        _print(diagnose(args.ground_truth, args.predictions, args.output))
    elif args.command == "status":
        _print({"baseline_presets": BASELINE_PRESETS, "detectors": DETECTOR_STATUS,
                "methods": METHOD_REGISTRY, "method_summary": METHOD_STATUS})
    elif args.command == "preflight":
        result = preflight(args.data, output_dir=args.output, weights=args.weights,
                           device=args.device, require_official=args.require_official)
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
            raise ValueError(f"Invalid configuration:\n" + "\n".join(errors))
        detector = resolved.detector
        method = resolved.method
        runtime = resolved.runtime
        output = resolved.output
        if detector.name in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine"}:
            raise RuntimeError(f"{detector.name} requires its official external runtime and canonical export contract")
        if method.name in METHOD_REGISTRY and METHOD_REGISTRY[method.name]["status"] == "external-required":
            raise RuntimeError(f"{method.name} requires its paper-faithful external training/runtime")
        if not output.predictions:
            raise ValueError("output.predictions must be specified in the configuration")
        _print(run_phase_2(
            data_dir=Path(resolved.dataset.root), split=resolved.dataset.split,
            weights=Path(detector.checkpoint), output_path=Path(output.predictions),
            method=method.name, limit=runtime.limit, image_size=int(detector.input_size),
            confidence=float(detector.confidence),
            tile_size=method.tile_size,
            overlap=method.overlap, device=runtime.device or detector.device,
            warmup=runtime.warmup, detector_name=detector.name,
            precision=runtime.precision,
        )["summary"])
    elif args.command == "run-smoke":
        _print(run_smoke_pipeline(
            data_dir=args.data, output_dir=args.output, weights=args.weights,
            train_limit=args.train_limit, eval_limit=args.eval_limit,
            image_size=args.imgsz, device=args.device,
        ))
    return 0
