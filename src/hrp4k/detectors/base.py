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
    """Create a detector adapter. Only YOLO11m and RT-DETR-L are supported via Ultralytics."""
    if name in {"ultralytics", "yolo11m", "rtdetr-l", "rtdetr_l", "rtdetr"}:
        from .ultralytics import UltralyticsAdapter
        return UltralyticsAdapter(weights, category_id, device, name, precision)
    raise ValueError(f"Unknown detector: {name!r}; supported: yolo11m, rtdetr-l")
