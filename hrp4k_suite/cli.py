from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import BASELINE_PRESETS, get_baseline_preset
from .dataset import analyze_dataset, prepare_dataset_view
from .detectors import DETECTOR_STATUS
from .diagnostics import diagnose
from .evaluation import evaluate_files
from .processing import METHOD_STATUS, predict_yolo
from .preflight import preflight
from .registry import METHOD_REGISTRY
from .training import train_yolo
from . import __version__


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrp4k", description="HRP4K Phase 0–3 benchmark CLI")
    parser.add_argument("--version", action="version", version=f"hrp4k {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Phase 0 dataset integrity and statistics")
    analyze.add_argument("--data", type=_path, default=Path("HRP4K")); analyze.add_argument("--output", type=_path, default=Path("outputs/phase0"))
    analyze.add_argument("--quality-samples", type=int, default=12)

    for name, help_text, default_output, defaults in (
        ("prepare-smoke", "Create a deterministic smoke dataset view using symlinks", Path("outputs/smoke/dataset"), (24, 12, 12)),
        ("prepare-dataset", "Create an all-available dataset view using symlinks", Path("outputs/local_dataset"), (None, None, None)),
    ):
        prepare = commands.add_parser(name, help=help_text)
        prepare.add_argument("--data", type=_path, default=Path("HRP4K")); prepare.add_argument("--output", type=_path, default=default_output)
        prepare.add_argument("--train-limit", type=int, default=defaults[0]); prepare.add_argument("--valid-limit", type=int, default=defaults[1]); prepare.add_argument("--test-limit", type=int, default=defaults[2])
        prepare.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Phase 1 YOLO benchmark training")
    train.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml")); train.add_argument("--weights", type=_path)
    train.add_argument("--preset", choices=[name for name, value in BASELINE_PRESETS.items() if value.get("framework") == "ultralytics"])
    train.add_argument("--output", type=_path); train.add_argument("--smoke", action="store_true")
    train.add_argument("--allow-full", action="store_true", help="Explicitly authorize a non-smoke training run")
    train.add_argument("--epochs", type=int, default=150); train.add_argument("--imgsz", type=int, default=640); train.add_argument("--batch", type=int, default=16); train.add_argument("--device")

    predict = commands.add_parser("predict", help="Phase 1/2 inference and unified COCO export")
    predict.add_argument("--data", type=_path, default=Path("outputs/smoke/dataset")); predict.add_argument("--split", choices=["train", "valid", "test"], default="test")
    predict.add_argument("--detector", choices=["ultralytics", "yolov5m-compat", "yolov5m-official", "yolov8m", "yolo11m", "rt-detr-v1", "rt-detr-v2", "d-fine"], default="ultralytics")
    predict.add_argument("--weights", type=_path, required=True); predict.add_argument("--output", type=_path, required=True)
    predict.add_argument("--method", choices=list(METHOD_REGISTRY), default="resize")
    predict.add_argument("--limit", type=int); predict.add_argument("--imgsz", type=int, default=320); predict.add_argument("--confidence", type=float, default=0.05)
    predict.add_argument("--tile-size", type=int, default=960); predict.add_argument("--overlap", type=float, default=0.2); predict.add_argument("--device")
    predict.add_argument("--warmup", type=int, default=20); predict.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")

    evaluate = commands.add_parser("evaluate", help="Unified COCO/scale/FPPI evaluator")
    evaluate.add_argument("--ground-truth", type=_path, required=True); evaluate.add_argument("--predictions", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True); evaluate.add_argument("--confidence", type=float, default=0.25)

    diagnostic = commands.add_parser("diagnose", help="Phase 3 diagnostics from saved predictions")
    diagnostic.add_argument("--ground-truth", type=_path, required=True); diagnostic.add_argument("--predictions", type=_path, nargs="+", required=True)
    diagnostic.add_argument("--output", type=_path, default=Path("outputs/phase3"))

    commands.add_parser("status", help="Show detector presets and Phase 2 reproduction status")

    check = commands.add_parser("preflight", help="Verify dataset identity, integrity and runtime dependencies")
    check.add_argument("--data", type=_path, default=Path("HRP4K")); check.add_argument("--output", type=_path)
    check.add_argument("--weights", type=_path); check.add_argument("--device"); check.add_argument("--require-official", action="store_true")

    run = commands.add_parser("run", help="Run an inference experiment from YAML/JSON config")
    run.add_argument("--config", type=_path, required=True)

    smoke = commands.add_parser("run-smoke", help="Run Phase 0–3 end-to-end smoke pipeline")
    smoke.add_argument("--data", type=_path, default=Path("HRP4K")); smoke.add_argument("--output", type=_path, default=Path("outputs/smoke"))
    smoke.add_argument("--weights", type=_path, default=Path("yolo11n.pt")); smoke.add_argument("--train-limit", type=int, default=24)
    smoke.add_argument("--eval-limit", type=int, default=8); smoke.add_argument("--imgsz", type=int, default=320); smoke.add_argument("--device")
    return parser


def _print(payload):
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze": _print(analyze_dataset(args.data, args.output, args.quality_samples))
    elif args.command in {"prepare-smoke", "prepare-dataset"}: _print(prepare_dataset_view(args.data, args.output, args.train_limit, args.valid_limit, args.test_limit, args.seed))
    elif args.command == "train":
        preset = get_baseline_preset(args.preset) if args.preset else None
        weights = args.weights or Path(preset["weights"] if preset else "yolo11m.pt")
        output = args.output or Path("outputs/runs") / (args.preset or Path(weights).stem)
        _print(train_yolo(args.dataset, weights, output, args.smoke, args.epochs, args.imgsz, args.batch, args.device,
                          args.allow_full, preset))
    elif args.command == "predict":
        if args.detector in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine"}:
            location = "yolov5" if args.detector == "yolov5m-official" else "rtdetr" if args.detector.startswith("rt-detr") else "dfine"
            raise RuntimeError(f"{args.detector} requires its official external runtime; use canonical export contract in external/{location}")
        if METHOD_REGISTRY[args.method]["status"] == "external-required":
            raise RuntimeError(f"{args.method} requires its paper-faithful external training/runtime; no heuristic substitute is enabled")
        _print(predict_yolo(args.data, args.split, args.weights, args.output, args.method, args.limit, args.imgsz,
                            args.confidence, args.tile_size, args.overlap, args.device, args.warmup, args.detector, args.precision)["summary"])
    elif args.command == "evaluate": _print(evaluate_files(args.ground_truth, args.predictions, args.output, args.confidence))
    elif args.command == "diagnose": _print(diagnose(args.ground_truth, args.predictions, args.output))
    elif args.command == "status": _print({"baseline_presets": BASELINE_PRESETS, "detectors": DETECTOR_STATUS,
                                             "methods": METHOD_REGISTRY, "method_summary": METHOD_STATUS})
    elif args.command == "preflight":
        result = preflight(args.data, output_dir=args.output, weights=args.weights,
                           device=args.device, require_official=args.require_official)
        _print(result)
        if result["status"] != "pass": return 2
    elif args.command == "run":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Config runner requires PyYAML from the vision dependencies") from exc
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        detector = config["detector"]; method = config["method"]; runtime = config.get("runtime", {}); output = config["output"]
        if detector["name"] in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine"}:
            raise RuntimeError(f"{detector['name']} requires its official external runtime and canonical export contract")
        if METHOD_REGISTRY[method["name"]]["status"] == "external-required":
            raise RuntimeError(f"{method['name']} requires its paper-faithful external training/runtime")
        _print(predict_yolo(Path(config["dataset"]["root"]), config["dataset"].get("split", "test"),
                            Path(detector["checkpoint"]), Path(output["predictions"]), method["name"],
                            runtime.get("limit"), int(detector.get("input_size", 640)), float(detector.get("confidence", 0.05)),
                            int(method.get("slice_width", method.get("tile_size", 960))), float(method.get("overlap", 0.2)),
                            detector.get("device"), int(runtime.get("warmup_images", 20)), detector["name"],
                            runtime.get("precision", "fp32"))["summary"])
    elif args.command == "run-smoke":
        root = args.output; dataset_dir = root / "dataset"
        analyze_dataset(args.data, root / "phase0", quality_samples=4)
        prepare_dataset_view(args.data, dataset_dir, args.train_limit, max(args.eval_limit, 4), args.eval_limit, 42)
        training = train_yolo(dataset_dir / "dataset.yaml", args.weights, root / "runs" / "yolo11n", True, 1, args.imgsz, 4, args.device)
        prediction_paths = []
        for method in ("resize", "sliced-nms", "perspective-grid"):
            prediction_path = root / "predictions" / f"{method}.json"
            predict_yolo(dataset_dir, "test", training["best"], prediction_path, method, args.eval_limit, args.imgsz, 0.01, device=args.device, warmup=1)
            metrics_path = prediction_path.with_name(prediction_path.stem + "_metrics.json")
            evaluate_files(dataset_dir / "test.json", prediction_path, metrics_path, 0.25)
            prediction_paths.append(prediction_path)
        result = diagnose(dataset_dir / "test.json", prediction_paths, root / "phase3")
        _print({"status": "complete", "output": str(root), "training": training,
                "diagnostics": {"ground_truth": result["ground_truth"],
                                "methods": {name: {"evaluation_status": value["evaluation_status"],
                                                   "AP50_95": value["metrics"].get("AP50_95"),
                                                   "compute_amplification_nominal_canvas": value["processing"].get("compute_amplification_nominal_canvas")}
                                            for name, value in result["methods"].items()},
                                "report": str(root / "phase3" / "phase3_report.md")}})
    return 0
