from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import analyze_dataset, prepare_smoke_dataset
from .detectors import DETECTOR_STATUS
from .diagnostics import diagnose
from .evaluation import evaluate_files
from .processing import METHOD_STATUS, predict_yolo
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

    for name, help_text, default_output in (
        ("prepare-smoke", "Create a deterministic smoke dataset view using symlinks", Path("outputs/smoke/dataset")),
        ("prepare-dataset", "Create a deterministic local-available dataset view using symlinks", Path("outputs/local_dataset")),
    ):
        prepare = commands.add_parser(name, help=help_text)
        prepare.add_argument("--data", type=_path, default=Path("HRP4K")); prepare.add_argument("--output", type=_path, default=default_output)
        prepare.add_argument("--train-limit", type=int, default=24); prepare.add_argument("--valid-limit", type=int, default=12); prepare.add_argument("--test-limit", type=int, default=12)
        prepare.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Phase 1 YOLO benchmark training")
    train.add_argument("--dataset", type=_path, default=Path("outputs/full_dataset/dataset.yaml")); train.add_argument("--weights", type=_path, default=Path("yolo11m.pt"))
    train.add_argument("--output", type=_path, default=Path("outputs/runs/yolo11m")); train.add_argument("--smoke", action="store_true")
    train.add_argument("--allow-full", action="store_true", help="Explicitly authorize a non-smoke training run")
    train.add_argument("--allow-incomplete-train", action="store_true", help="Label and allow a non-official local-available training set")
    train.add_argument("--epochs", type=int, default=150); train.add_argument("--imgsz", type=int, default=640); train.add_argument("--batch", type=int, default=16); train.add_argument("--device")

    predict = commands.add_parser("predict", help="Phase 1/2 inference and unified COCO export")
    predict.add_argument("--data", type=_path, default=Path("outputs/smoke/dataset")); predict.add_argument("--split", choices=["train", "valid", "test"], default="test")
    predict.add_argument("--weights", type=_path, required=True); predict.add_argument("--output", type=_path, required=True)
    predict.add_argument("--method", choices=["resize", "uniform-2", "uniform-3", "sliced-nms", "perspective-grid", "sahi", "perspective-bands"], default="resize")
    predict.add_argument("--limit", type=int); predict.add_argument("--imgsz", type=int, default=320); predict.add_argument("--confidence", type=float, default=0.05)
    predict.add_argument("--tile-size", type=int, default=960); predict.add_argument("--overlap", type=float, default=0.2); predict.add_argument("--device")

    evaluate = commands.add_parser("evaluate", help="Unified COCO/scale/FPPI evaluator")
    evaluate.add_argument("--ground-truth", type=_path, required=True); evaluate.add_argument("--predictions", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True); evaluate.add_argument("--confidence", type=float, default=0.25)

    diagnostic = commands.add_parser("diagnose", help="Phase 3 diagnostics from saved predictions")
    diagnostic.add_argument("--ground-truth", type=_path, required=True); diagnostic.add_argument("--predictions", type=_path, nargs="+", required=True)
    diagnostic.add_argument("--output", type=_path, default=Path("outputs/phase3"))

    commands.add_parser("status", help="Show Phase 2 reproduction status")

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
    elif args.command in {"prepare-smoke", "prepare-dataset"}: _print(prepare_smoke_dataset(args.data, args.output, args.train_limit, args.valid_limit, args.test_limit, args.seed))
    elif args.command == "train": _print(train_yolo(args.dataset, args.weights, args.output, args.smoke, args.epochs, args.imgsz, args.batch, args.device, args.allow_full, args.allow_incomplete_train))
    elif args.command == "predict": _print(predict_yolo(args.data, args.split, args.weights, args.output, args.method, args.limit, args.imgsz, args.confidence, args.tile_size, args.overlap, args.device)["summary"])
    elif args.command == "evaluate": _print(evaluate_files(args.ground_truth, args.predictions, args.output, args.confidence))
    elif args.command == "diagnose": _print(diagnose(args.ground_truth, args.predictions, args.output))
    elif args.command == "status": _print({"detectors": DETECTOR_STATUS, "processors": METHOD_STATUS})
    elif args.command == "run-smoke":
        root = args.output; dataset_dir = root / "dataset"
        analyze_dataset(args.data, root / "phase0", quality_samples=4)
        prepare_smoke_dataset(args.data, dataset_dir, args.train_limit, max(args.eval_limit, 4), args.eval_limit, 42)
        training = train_yolo(dataset_dir / "dataset.yaml", args.weights, root / "runs" / "yolo11n", True, 1, args.imgsz, 4, args.device)
        prediction_paths = []
        for method in ("resize", "sliced-nms", "perspective-grid"):
            prediction_path = root / "predictions" / f"{method}.json"
            predict_yolo(dataset_dir, "test", training["best"], prediction_path, method, args.eval_limit, args.imgsz, 0.01, device=args.device)
            metrics_path = prediction_path.with_name(prediction_path.stem + "_metrics.json")
            evaluate_files(dataset_dir / "test.json", prediction_path, metrics_path, 0.25)
            prediction_paths.append(prediction_path)
        result = diagnose(dataset_dir / "test.json", prediction_paths, root / "phase3")
        _print({"status": "complete", "output": str(root), "training": training,
                "diagnostics": {"ground_truth": result["ground_truth"],
                                "methods": {name: {"evaluation_status": value["evaluation_status"],
                                                   "AP50_95": value["metrics"].get("AP50_95"),
                                                   "compute_amplification_input": value["processing"].get("compute_amplification_input")}
                                            for name, value in result["methods"].items()},
                                "report": str(root / "phase3" / "phase3_report.md")}})
    return 0
