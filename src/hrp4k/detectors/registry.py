from __future__ import annotations

from typing import Any


BASELINE_PRESETS: dict[str, dict[str, Any]] = {
    "yolov5m-compat": {
        "family": "YOLOv5", "framework": "ultralytics", "weights": "yolov5mu.pt", "size": "medium",
        "reproduction_scope": "Ultralytics compatibility checkpoint; not the original YOLOv5 paper repository",
        "status": "smoke-capable; paper reproduction not claimed",
    },
    "yolov5m-official": {
        "family": "YOLOv5", "framework": "official external", "weights": "yolov5m.pt", "size": "medium",
        "reproduction_scope": "Original YOLOv5 repository in an isolated environment",
        "status": "external runner required",
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


DETECTOR_STATUS = {
    "yolov5m-compat": "Ultralytics compatibility preset configured; original-paper reproduction not claimed",
    "yolov5m-official": "original YOLOv5 medium reproduction requires isolated external runtime",
    "yolov8m": "Ultralytics medium preset configured; smoke-capable",
    "yolo11m": "Ultralytics medium preset configured; smoke verified",
    "rt-detr-v1": "official external adapter required",
    "rt-detr-v2": "official external adapter required",
    "d-fine": "official external adapter required",
}


def get_baseline_preset(name: str) -> dict[str, Any]:
    try:
        return {"name": name, **BASELINE_PRESETS[name]}
    except KeyError as exc:
        raise ValueError(f"Unknown baseline preset {name!r}; choose from {sorted(BASELINE_PRESETS)}") from exc
