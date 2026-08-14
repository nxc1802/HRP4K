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
