from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset import image_path, load_split
from .detectors import UltralyticsAdapter


@dataclass
class ProcessedView:
    image: np.ndarray
    x0: int
    y0: int
    source_width: int
    source_height: int

    def map_box(self, xyxy: Iterable[float]) -> list[float]:
        x1, y1, x2, y2 = map(float, xyxy)
        return [x1 + self.x0, y1 + self.y0, x2 - x1, y2 - y1]


def _starts(length: int, window: int, overlap: float) -> list[int]:
    if window >= length:
        return [0]
    stride = max(1, round(window * (1 - overlap)))
    starts = list(range(0, max(1, length - window + 1), stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def make_views(image: np.ndarray, method: str, tile_size: int = 960, overlap: float = 0.2) -> list[ProcessedView]:
    height, width = image.shape[:2]
    if method == "sahi":
        warnings.warn("'sahi' is a compatibility alias; use 'sliced-nms' for the in-house implementation", DeprecationWarning)
        method = "sliced-nms"
    if method == "resize":
        return [ProcessedView(image, 0, 0, width, height)]
    if method.startswith("uniform"):
        grid = int(method.split("-", 1)[1]) if "-" in method else 2
        views = []
        for row in range(grid):
            y0, y1 = round(row * height / grid), round((row + 1) * height / grid)
            for col in range(grid):
                x0, x1 = round(col * width / grid), round((col + 1) * width / grid)
                views.append(ProcessedView(image[y0:y1, x0:x1], x0, y0, x1 - x0, y1 - y0))
        return views
    if method == "sliced-nms":
        views = []
        window_w = min(tile_size, width)
        window_h = min(max(1, round(tile_size * height / width)), height)
        for y0 in _starts(height, window_h, overlap):
            for x0 in _starts(width, window_w, overlap):
                views.append(ProcessedView(image[y0:y0 + window_h, x0:x0 + window_w], x0, y0, window_w, window_h))
        return views
    if method == "perspective-bands":
        warnings.warn("'perspective-bands' removes vertical context but does not magnify horizontally; use 'perspective-grid'", DeprecationWarning)
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        return [ProcessedView(image[y0:y1], 0, y0, width, y1 - y0) for y0, y1 in zip(boundaries, boundaries[1:])]
    if method == "perspective-grid":
        # Hand-designed ground-plane baseline. Far bands receive more horizontal crops and therefore more detector pixels.
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        columns_by_band = [4, 3, 2]
        views = []
        for (y0, y1), columns in zip(zip(boundaries, boundaries[1:]), columns_by_band):
            window_w = min(width, int(np.ceil(width / (columns - (columns - 1) * overlap))))
            starts = _starts(width, window_w, overlap)
            if len(starts) > columns:
                starts = np.linspace(0, width - window_w, columns, dtype=int).tolist()
            for x0 in starts:
                views.append(ProcessedView(image[y0:y1, x0:x0 + window_w], x0, y0, window_w, y1 - y0))
        return views
    raise ValueError(f"Unknown processing method: {method}")


def nms(predictions: list[dict[str, Any]], threshold: float = 0.5) -> tuple[list[dict[str, Any]], int]:
    if not predictions:
        return [], 0
    boxes = np.asarray([p["bbox"] for p in predictions], dtype=float)
    xyxy = np.column_stack((boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]))
    scores = np.asarray([p["score"] for p in predictions], dtype=float)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        current = int(order[0]); keep.append(current)
        if order.size == 1: break
        rest = order[1:]
        x1 = np.maximum(xyxy[current, 0], xyxy[rest, 0]); y1 = np.maximum(xyxy[current, 1], xyxy[rest, 1])
        x2 = np.minimum(xyxy[current, 2], xyxy[rest, 2]); y2 = np.minimum(xyxy[current, 3], xyxy[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = boxes[current, 2] * boxes[current, 3]
        area_rest = boxes[rest, 2] * boxes[rest, 3]
        overlaps = inter / np.maximum(area_current + area_rest - inter, 1e-12)
        order = rest[overlaps <= threshold]
    return [predictions[index] for index in keep], len(predictions) - len(keep)


def predict_yolo(
    data_dir: Path, split: str, weights: Path | str, output_path: Path, method: str = "resize",
    limit: int | None = None, image_size: int = 640, confidence: float = 0.05,
    tile_size: int = 960, overlap: float = 0.2, device: str | None = None,
) -> dict[str, Any]:
    if method == "sahi":
        warnings.warn("'sahi' is deprecated; recording this run as 'sliced-nms'", DeprecationWarning)
        method = "sliced-nms"
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Prediction requires the 'vision' dependencies") from exc
    coco = load_split(data_dir, split)
    category_ids = [int(category["id"]) for category in coco.get("categories", [])]
    if len(category_ids) != 1:
        raise ValueError(f"Expected exactly one HRP4K category, found {category_ids}")
    category_id = category_ids[0]
    detector = UltralyticsAdapter(weights, category_id, device)
    images = [im for im in coco.get("images", []) if image_path(data_dir, split, im["file_name"]).is_file()]
    if limit is not None: images = images[:limit]
    if images:
        warmup_image = cv2.imread(str(image_path(data_dir, split, images[0]["file_name"])))
        if warmup_image is not None:
            detector.predict(make_views(warmup_image, method, tile_size, overlap)[0].image, image_size, confidence)
    peak_vram_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    predictions: list[dict[str, Any]] = []; image_meta = []
    for im in images:
        end_to_end_started = time.perf_counter()
        path = image_path(data_dir, split, im["file_name"])
        decode_started = time.perf_counter()
        source = cv2.imread(str(path))
        if source is None: continue
        decode_latency_ms = (time.perf_counter() - decode_started) * 1000
        processor_started = time.perf_counter()
        views = make_views(source, method, tile_size, overlap)
        processor_latency_ms = (time.perf_counter() - processor_started) * 1000
        candidates = []; detector_latency_ms = 0.0
        for view in views:
            detector_started = time.perf_counter()
            for prediction in detector.predict(view.image, image_size, confidence):
                candidates.append({"image_id": int(im["id"]), "category_id": prediction["category_id"],
                                   "bbox": view.map_box(prediction["xyxy"]), "score": prediction["score"]})
            detector_latency_ms += (time.perf_counter() - detector_started) * 1000
        fusion_started = time.perf_counter()
        merged, suppressed = nms(candidates)
        fusion_latency_ms = (time.perf_counter() - fusion_started) * 1000
        end_to_end_latency_ms = (time.perf_counter() - end_to_end_started) * 1000
        predictions.extend(merged)
        source_pixels = sum(view.source_width * view.source_height for view in views)
        nominal_canvas_pixels = len(views) * image_size * image_size
        image_meta.append({"image_id": int(im["id"]), "method": method,
                           "decode_latency_ms": decode_latency_ms, "processor_latency_ms": processor_latency_ms,
                           "detector_latency_ms": detector_latency_ms, "fusion_latency_ms": fusion_latency_ms,
                           "end_to_end_latency_ms": end_to_end_latency_ms,
                           "detector_calls": len(views), "processed_source_pixels": source_pixels,
                           "processed_area_ratio": source_pixels / (source.shape[0] * source.shape[1]),
                           "nominal_detector_canvas_pixels": nominal_canvas_pixels,
                           "compute_amplification_nominal_canvas": nominal_canvas_pixels / (image_size * image_size),
                           "fusion_suppression_count": suppressed, "predictions": len(merged)})
    end_to_end_latencies = [x["end_to_end_latency_ms"] for x in image_meta]
    processor_latencies = [x["processor_latency_ms"] for x in image_meta]
    detector_latencies = [x["detector_latency_ms"] for x in image_meta]
    calls = [x["detector_calls"] for x in image_meta]
    payload = {
        "method": method, "weights": str(weights), "split": split,
        "settings": {"image_size": image_size, "confidence": confidence, "tile_size": tile_size, "overlap": overlap,
                     "warmup_iterations": 1 if images else 0,
                     "latency_protocol": "end_to_end includes decode, processor, detector, remapping and fusion"},
        "predictions": predictions, "image_metadata": image_meta,
        "summary": {
            "images": len(image_meta), "predictions": len(predictions),
            "mean_processor_latency_ms": float(np.mean(processor_latencies)) if image_meta else 0.0,
            "mean_detector_latency_ms": float(np.mean(detector_latencies)) if image_meta else 0.0,
            "mean_end_to_end_latency_ms": float(np.mean(end_to_end_latencies)) if image_meta else 0.0,
            "p50_end_to_end_latency_ms": float(np.percentile(end_to_end_latencies, 50)) if image_meta else 0.0,
            "p95_end_to_end_latency_ms": float(np.percentile(end_to_end_latencies, 95)) if image_meta else 0.0,
            "mean_detector_calls": float(np.mean(calls)) if image_meta else 0.0,
            "p95_detector_calls": float(np.percentile(calls, 95)) if image_meta else 0.0,
            "mean_processed_area_ratio": float(np.mean([x["processed_area_ratio"] for x in image_meta])) if image_meta else 0.0,
            "mean_nominal_detector_canvas_pixels": float(np.mean([x["nominal_detector_canvas_pixels"] for x in image_meta])) if image_meta else 0.0,
            "compute_amplification_nominal_canvas": float(np.mean([x["compute_amplification_nominal_canvas"] for x in image_meta])) if image_meta else 0.0,
        },
    }
    if torch is not None and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    payload["summary"]["peak_vram_mb"] = peak_vram_mb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


METHOD_STATUS = {
    "resize": "implemented",
    "uniform-2": "implemented",
    "uniform-3": "implemented",
    "sliced-nms": "implemented in-house sliced inference; not official SAHI",
    "sahi": "deprecated compatibility alias for sliced-nms",
    "perspective-bands": "deprecated context-removal baseline",
    "perspective-grid": "implemented hand-designed geometry allocation baseline; not learned TPP",
    "autofocus": "external reproduction required",
    "adazoom": "external RL reproduction required",
    "fovea": "external MMDetection reproduction required",
    "two-plane-prior": "external learned reproduction required",
    "zoomdet": "external learned reproduction required",
}
