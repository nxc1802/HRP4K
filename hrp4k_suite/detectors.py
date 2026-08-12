from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class UltralyticsAdapter:
    weights: Path | str
    category_id: int
    device: str | None = None
    name: str = "ultralytics"
    precision: str = "fp32"

    def __post_init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics prediction requires the 'vision' dependencies") from exc
        self.model = YOLO(str(self.weights))

    def warmup(self, image: np.ndarray, image_size: int) -> None:
        self.predict(image, image_size, 0.01)

    def predict(self, image: np.ndarray, image_size: int, confidence: float) -> list[Detection]:
        result = self.model.predict(image, imgsz=image_size, conf=confidence, verbose=False, device=self.device,
                                    half=self.precision == "fp16")[0]
        if result.boxes is None:
            return []
        return [
            Detection(tuple(map(float, xyxy)), float(score), self.category_id)
            for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ]

    def metadata(self) -> dict[str, Any]:
        import ultralytics
        return {"name": self.name, "family": "YOLO", "framework": "ultralytics",
                "framework_version": ultralytics.__version__, "weights": str(self.weights),
                "device": self.device, "precision": self.precision}


def create_detector(
    name: str, weights: Path | str, category_id: int, device: str | None = None, precision: str = "fp32",
) -> DetectorAdapter:
    """Create a core detector or fail clearly for isolated official runtimes."""
    if name in {"ultralytics", "yolov5m-compat", "yolov8m", "yolo11m"}:
        return UltralyticsAdapter(weights, category_id, device, name, precision)
    if name in {"yolov5m-official", "rt-detr-v1", "rt-detr-v2", "d-fine"}:
        raise RuntimeError(f"{name} requires its official isolated runtime and canonical export contract")
    raise ValueError(f"Unknown detector: {name}")


DETECTOR_STATUS = {
    "yolov5m-compat": "Ultralytics compatibility preset configured; original-paper reproduction not claimed",
    "yolov5m-official": "original YOLOv5 medium reproduction requires isolated external runtime",
    "yolov8m": "Ultralytics medium preset configured; smoke-capable",
    "yolo11m": "Ultralytics medium preset configured; smoke verified",
    "rt-detr-v1": "official external adapter required",
    "rt-detr-v2": "official external adapter required",
    "d-fine": "official external adapter required",
}
