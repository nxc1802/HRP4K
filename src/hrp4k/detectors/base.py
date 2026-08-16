from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int


class DetectorAdapter(Protocol):
    """Framework boundary: detectors return boxes in the supplied view coordinates."""

    name: str
    def warmup(self, image: np.ndarray, image_size: int) -> None: ...
    def predict(self, image: np.ndarray, image_size: int, confidence: float) -> list[Detection]: ...
    def metadata(self) -> dict[str, Any]: ...


def create_detector(
    name: str, weights, category_id: int, device: str | None = None, precision: str = "fp32",
    allow_ultralytics: bool = False,
) -> DetectorAdapter:
    """Create a core detector or fail clearly for isolated official runtimes."""
    if name in {"ultralytics", "yolov5m-compat", "yolov8m", "yolo11m"}:
        from .ultralytics import UltralyticsAdapter
        return UltralyticsAdapter(weights, category_id, device, name, precision)
    if allow_ultralytics and name in {"yolov5m-official", "yolov5m", "rt-detr-v1", "rt-detr-v2", "rtdetr_v1", "rtdetr_v2", "rtdetr"}:
        from .ultralytics import UltralyticsAdapter
        return UltralyticsAdapter(weights, category_id, device, name, precision)
    if name in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine", "dfine"}:
        raise RuntimeError(f"{name} requires its official isolated runtime and canonical export contract")
    raise ValueError(f"Unknown detector: {name}")
