from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import Detection


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
        predict_kwargs = {"half": True} if self.precision == "fp16" else {}
        result = self.model.predict(image, imgsz=image_size, conf=confidence, verbose=False, device=self.device, **predict_kwargs)[0]
        if result.boxes is None:
            return []
        return [
            Detection(tuple(map(float, xyxy)), float(score), self.category_id)
            for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ]

    def predict_batch(self, images: list[np.ndarray], image_size: int, confidence: float) -> list[list[Detection]]:
        if not images:
            return []
        predict_kwargs = {"half": True} if self.precision == "fp16" else {}
        results = self.model.predict(images, imgsz=image_size, conf=confidence, verbose=False, device=self.device, batch=len(images), **predict_kwargs)
        batch_detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                batch_detections.append([])
            else:
                batch_detections.append([
                    Detection(tuple(map(float, xyxy)), float(score), self.category_id)
                    for xyxy, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                ])
        return batch_detections

    def metadata(self) -> dict[str, Any]:
        import ultralytics
        return {"name": self.name, "family": "YOLO", "framework": "ultralytics",
                "framework_version": ultralytics.__version__, "weights": str(self.weights),
                "device": self.device, "precision": self.precision}
