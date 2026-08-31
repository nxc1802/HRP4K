from __future__ import annotations

from typing import Any


BASELINE_PRESETS: dict[str, dict[str, Any]] = {
    "yolo11m": {
        "family": "YOLOv11", "framework": "ultralytics", "weights": "yolo11m.pt", "size": "medium",
        "reproduction_scope": "Ultralytics implementation with project-resolved protocol",
        "status": "ready",
    },
    "rtdetr-l": {
        "family": "RT-DETR-L", "framework": "ultralytics", "weights": "rtdetr-l.pt", "size": "large",
        "reproduction_scope": "Ultralytics RT-DETR-L Transformer baseline detector (32.8M params)",
        "status": "ready",
    },
}


DETECTOR_STATUS = {
    "yolo11m": "Ultralytics YOLOv11 medium preset configured; ready",
    "rtdetr-l": "Ultralytics RT-DETR-L preset configured; ready",
}


def get_baseline_preset(name: str) -> dict[str, Any]:
    try:
        return {"name": name, **BASELINE_PRESETS[name]}
    except KeyError as exc:
        raise ValueError(f"Unknown baseline preset {name!r}; choose from {sorted(BASELINE_PRESETS) + ['all']}") from exc
