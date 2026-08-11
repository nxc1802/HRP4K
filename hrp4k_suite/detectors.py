from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class DetectorAdapter(Protocol):
    """Framework boundary: detectors return boxes in the supplied view coordinates."""

    def predict(self, image: np.ndarray, image_size: int, confidence: float) -> list[dict[str, Any]]: ...


@dataclass
class UltralyticsAdapter:
    weights: Path | str
    category_id: int
    device: str | None = None

    def __post_init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics prediction requires the 'vision' dependencies") from exc
        self.model = YOLO(str(self.weights))

    def predict(self, image: np.ndarray, image_size: int, confidence: float) -> list[dict[str, Any]]:
        result = self.model.predict(image, imgsz=image_size, conf=confidence, verbose=False, device=self.device)[0]
        if result.boxes is None:
            return []
        return [
            {"xyxy": xyxy.tolist(), "score": float(score), "category_id": self.category_id}
            for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ]


DETECTOR_STATUS = {
    "yolov5m-compat": "Ultralytics compatibility preset configured; original-paper reproduction not claimed",
    "yolov8m": "Ultralytics medium preset configured; smoke-capable",
    "yolo11m": "Ultralytics medium preset configured; smoke verified",
    "rt-detr-v1": "official external adapter required",
    "rt-detr-v2": "official external adapter required",
    "d-fine": "official external adapter required",
}
