from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class CoordinateTransform(Protocol):
    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray: ...
    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray: ...


class IdentityTransform:
    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return np.asarray(boxes_xyxy, dtype=float).copy()

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return np.asarray(boxes_xyxy, dtype=float).copy()


@dataclass(frozen=True)
class CropTransform:
    x0: float
    y0: float

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] -= self.x0; result[:, [1, 3]] -= self.y0
        return result

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] += self.x0; result[:, [1, 3]] += self.y0
        return result


@dataclass
class ProcessedView:
    image: np.ndarray
    transform: CoordinateTransform
    source_width: int
    source_height: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def map_box(self, xyxy) -> list[float]:
        source = self.transform.inverse_boxes(np.asarray([xyxy], dtype=float))[0]
        x1, y1, x2, y2 = source
        w = max(0.01, float(x2 - x1))
        h = max(0.01, float(y2 - y1))
        return [float(x1), float(y1), w, h]


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


def _starts(length: int, window: int, overlap: float) -> list[int]:
    if window >= length:
        return [0]
    stride = max(1, round(window * (1 - overlap)))
    starts = list(range(0, max(1, length - window + 1), stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


METHOD_REGISTRY = {
    "resize": {"type": "inference", "requires_training": False, "implementation": "native", "status": "ready"},
    "sliced-nms": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "sahi": {"type": "crop", "requires_training": False, "implementation": "official-library", "status": "optional-ready"},
    "perspective-grid": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
}

METHOD_STATUS = {
    name: f"{entry['status']} ({entry['implementation']})"
    for name, entry in METHOD_REGISTRY.items()
}


def make_views(
    image,
    method: str,
    tile_size: int = 960,
    overlap: float = 0.2,
    device: str | None = None,
) -> list[ProcessedView]:
    height, width = image.shape[:2]
    if method == "sahi":
        raise ValueError("Official SAHI is executed by the generic runner, not make_views()")
    if method == "resize":
        return [ProcessedView(image, IdentityTransform(), width, height)]
    if method == "sliced-nms":
        views = []
        window_w = min(tile_size, width)
        window_h = min(max(1, round(tile_size * height / width)), height)
        for y0 in _starts(height, window_h, overlap):
            for x0 in _starts(width, window_w, overlap):
                views.append(ProcessedView(image[y0:y0 + window_h, x0:x0 + window_w], CropTransform(x0, y0), window_w, window_h,
                                           {"crop": [x0, y0, x0 + window_w, y0 + window_h]}))
        return views
    if method == "perspective-grid":
        # Hand-designed ground-plane baseline with 2D (horizontal + vertical) overlap.
        # Far bands receive more horizontal crops and therefore more detector pixels.
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        columns_by_band = [4, 3, 2]
        views = []
        for idx, ((y0, y1), columns) in enumerate(zip(zip(boundaries, boundaries[1:]), columns_by_band)):
            band_h = y1 - y0
            pad_y = round(band_h * overlap * 0.5)
            y0_crop = max(0, y0 - pad_y) if idx > 0 else 0
            y1_crop = min(height, y1 + pad_y) if idx < len(columns_by_band) - 1 else height
            crop_h = y1_crop - y0_crop

            window_w = min(width, int(np.ceil(width / (columns - (columns - 1) * overlap))))
            starts = _starts(width, window_w, overlap)
            if len(starts) > columns:
                starts = np.linspace(0, width - window_w, columns, dtype=int).tolist()
            for x0 in starts:
                views.append(ProcessedView(image[y0_crop:y1_crop, x0:x0 + window_w],
                                           CropTransform(x0, y0_crop), window_w, crop_h,
                                           {"crop": [x0, y0_crop, x0 + window_w, y1_crop]}))
        return views
    raise ValueError(f"Unknown processing method: {method}")
