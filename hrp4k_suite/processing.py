from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import image_path, load_split
from .detectors import create_detector
from .registry import METHOD_REGISTRY, METHOD_STATUS
from .transforms import CropTransform, IdentityTransform, ProcessedView


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
        raise ValueError("Official SAHI is executed by the generic runner, not make_views()")
    if method == "resize":
        return [ProcessedView(image, IdentityTransform(), width, height)]
    if method.startswith("uniform"):
        grid = int(method.split("-", 1)[1]) if "-" in method else 2
        views = []
        for row in range(grid):
            y0, y1 = round(row * height / grid), round((row + 1) * height / grid)
            for col in range(grid):
                x0, x1 = round(col * width / grid), round((col + 1) * width / grid)
                views.append(ProcessedView(image[y0:y1, x0:x1], CropTransform(x0, y0), x1 - x0, y1 - y0,
                                           {"crop": [x0, y0, x1, y1]}))
        return views
    if method == "sliced-nms":
        views = []
        window_w = min(tile_size, width)
        window_h = min(max(1, round(tile_size * height / width)), height)
        for y0 in _starts(height, window_h, overlap):
            for x0 in _starts(width, window_w, overlap):
                views.append(ProcessedView(image[y0:y0 + window_h, x0:x0 + window_w], CropTransform(x0, y0), window_w, window_h,
                                           {"crop": [x0, y0, x0 + window_w, y0 + window_h]}))
        return views
    if method == "perspective-bands":
        warnings.warn("'perspective-bands' removes vertical context but does not magnify horizontally; use 'perspective-grid'", DeprecationWarning)
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        return [ProcessedView(image[y0:y1], CropTransform(0, y0), width, y1 - y0) for y0, y1 in zip(boundaries, boundaries[1:])]
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
                views.append(ProcessedView(image[y0:y1, x0:x0 + window_w], CropTransform(x0, y0), window_w, y1 - y0,
                                           {"crop": [x0, y0, x0 + window_w, y1]}))
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
    tile_size: int = 960, overlap: float = 0.2, device: str | None = None, warmup: int = 20,
    detector_name: str = "ultralytics", precision: str = "fp32",
) -> dict[str, Any]:
    coco = load_split(data_dir, split)
    category_ids = [int(category["id"]) for category in coco.get("categories", [])]
    if len(category_ids) != 1:
        raise ValueError(f"Expected exactly one HRP4K category, found {category_ids}")
    category_id = category_ids[0]
    detector = create_detector(detector_name, weights, category_id, device, precision)
    from .runner import predict_detector
    return predict_detector(data_dir, split, detector, output_path, method, limit=limit,
                            image_size=image_size, confidence=confidence, tile_size=tile_size,
                            overlap=overlap, warmup=warmup, precision=precision)
