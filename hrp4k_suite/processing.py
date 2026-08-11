from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset import image_path, load_split


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
    if method == "sahi":
        views = []
        window_w = min(tile_size, width)
        window_h = min(max(1, round(tile_size * height / width)), height)
        for y0 in _starts(height, window_h, overlap):
            for x0 in _starts(width, window_w, overlap):
                views.append(ProcessedView(image[y0:y0 + window_h, x0:x0 + window_w], x0, y0, window_w, window_h))
        return views
    if method == "perspective-bands":
        # A transparent geometry baseline, not a reproduction of learned TPP.
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        return [ProcessedView(image[y0:y1], 0, y0, width, y1 - y0) for y0, y1 in zip(boundaries, boundaries[1:])]
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
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Prediction requires the 'vision' dependencies") from exc
    model = YOLO(str(weights))
    coco = load_split(data_dir, split)
    images = [im for im in coco.get("images", []) if image_path(data_dir, split, im["file_name"]).is_file()]
    if limit is not None: images = images[:limit]
    predictions: list[dict[str, Any]] = []; image_meta = []
    for im in images:
        path = image_path(data_dir, split, im["file_name"])
        source = cv2.imread(str(path))
        if source is None: continue
        views = make_views(source, method, tile_size, overlap)
        started = time.perf_counter(); candidates = []
        for view in views:
            result = model.predict(view.image, imgsz=image_size, conf=confidence, verbose=False, device=device)[0]
            if result.boxes is None: continue
            for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                candidates.append({"image_id": int(im["id"]), "category_id": 0,
                                   "bbox": view.map_box(xyxy), "score": float(score)})
        merged, suppressed = nms(candidates)
        elapsed = (time.perf_counter() - started) * 1000
        predictions.extend(merged)
        source_pixels = sum(view.source_width * view.source_height for view in views)
        image_meta.append({"image_id": int(im["id"]), "method": method, "latency_ms": elapsed,
                           "detector_calls": len(views), "processed_source_pixels": source_pixels,
                           "processed_area_ratio": source_pixels / (source.shape[0] * source.shape[1]),
                           "fusion_suppression_count": suppressed, "predictions": len(merged)})
    payload = {
        "method": method, "weights": str(weights), "split": split,
        "settings": {"image_size": image_size, "confidence": confidence, "tile_size": tile_size, "overlap": overlap},
        "predictions": predictions, "image_metadata": image_meta,
        "summary": {
            "images": len(image_meta), "predictions": len(predictions),
            "mean_latency_ms": float(np.mean([x["latency_ms"] for x in image_meta])) if image_meta else 0.0,
            "mean_detector_calls": float(np.mean([x["detector_calls"] for x in image_meta])) if image_meta else 0.0,
            "mean_processed_area_ratio": float(np.mean([x["processed_area_ratio"] for x in image_meta])) if image_meta else 0.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


METHOD_STATUS = {
    "resize": "implemented",
    "uniform-2": "implemented",
    "uniform-3": "implemented",
    "sahi": "implemented (framework-independent sliced inference)",
    "perspective-bands": "implemented geometry baseline; not learned TPP",
    "autofocus": "external reproduction required",
    "adazoom": "external RL reproduction required",
    "fovea": "external MMDetection reproduction required",
    "two-plane-prior": "external learned reproduction required",
    "zoomdet": "external learned reproduction required",
}
