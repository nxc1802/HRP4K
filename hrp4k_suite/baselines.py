from __future__ import annotations

from typing import Any


BASELINE_PRESETS: dict[str, dict[str, Any]] = {
    "yolov5m-compat": {
        "family": "YOLOv5", "framework": "ultralytics", "weights": "yolov5mu.pt", "size": "medium",
        "reproduction_scope": "Ultralytics compatibility checkpoint; not the original YOLOv5 paper repository",
        "status": "smoke-capable; paper reproduction not claimed",
    },
    "yolov8m": {
        "family": "YOLOv8", "framework": "ultralytics", "weights": "yolov8m.pt", "size": "medium",
        "reproduction_scope": "Ultralytics implementation with project-resolved protocol",
        "status": "smoke-capable; full experiment pending",
    },
    "yolo11m": {
        "family": "YOLOv11", "framework": "ultralytics", "weights": "yolo11m.pt", "size": "medium",
        "reproduction_scope": "Ultralytics implementation with project-resolved protocol",
        "status": "smoke-capable; full experiment pending",
    },
    "rt-detr-v1": {"family": "RT-DETRv1", "framework": "official external", "status": "adapter required"},
    "rt-detr-v2": {"family": "RT-DETRv2", "framework": "official external", "status": "adapter required"},
    "d-fine": {"family": "D-FINE", "framework": "official external", "status": "adapter required"},
}


def get_baseline_preset(name: str) -> dict[str, Any]:
    try:
        return {"name": name, **BASELINE_PRESETS[name]}
    except KeyError as exc:
        raise ValueError(f"Unknown baseline preset {name!r}; choose from {sorted(BASELINE_PRESETS)}") from exc
