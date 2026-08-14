from __future__ import annotations

from pathlib import Path
from typing import Any

from ..inference.runner import predict_yolo
from ..evaluation.coco import evaluate_files
from ..methods.base import METHOD_REGISTRY


def run_phase_2(
    data_dir: Path, split: str, weights: Path | str, output_path: Path,
    method: str = "resize", limit: int | None = None, image_size: int = 640,
    confidence: float = 0.05, tile_size: int = 960, overlap: float = 0.2,
    device: str | None = None, warmup: int = 20, detector_name: str = "ultralytics",
    precision: str = "fp32", evaluate_after: bool = False,
    ground_truth: Path | None = None, eval_confidence: float = 0.25,
) -> dict[str, Any]:
    """Execute Phase 2 resolution allocation and canonical COCO prediction."""
    if detector_name in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine"}:
        location = "yolov5" if detector_name == "yolov5m-official" else "rtdetr" if detector_name.startswith("rt-detr") else "dfine"
        raise RuntimeError(f"{detector_name} requires its official external runtime; use canonical export contract in external/{location}")
    if METHOD_REGISTRY[method]["status"] == "external-required":
        raise RuntimeError(f"{method} requires its paper-faithful external training/runtime; no heuristic substitute is enabled")
    payload = predict_yolo(
        data_dir=data_dir, split=split, weights=weights, output_path=output_path,
        method=method, limit=limit, image_size=image_size, confidence=confidence,
        tile_size=tile_size, overlap=overlap, device=device, warmup=warmup,
        detector_name=detector_name, precision=precision,
    )
    if evaluate_after and ground_truth is not None:
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        evaluate_files(ground_truth, output_path, metrics_path, confidence=eval_confidence)
    return payload
