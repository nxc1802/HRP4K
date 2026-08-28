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
    "rt-detr-v1": {
        "family": "RT-DETRv1", "framework": "ultralytics", "weights": "rtdetr-l.pt", "size": "large",
        "reproduction_scope": "Ultralytics RT-DETR-L baseline detector",
        "status": "smoke-capable; full experiment ready",
    },
    "rt-detr-v2": {
        "family": "RT-DETRv2", "framework": "ultralytics", "weights": "rtdetr-x.pt", "size": "extra-large",
        "reproduction_scope": "Ultralytics RT-DETR-X baseline detector",
        "status": "smoke-capable; full experiment ready",
    },
    "d-fine": {
        "family": "D-FINE", "framework": "ultralytics", "weights": "rtdetr-l.pt", "size": "medium",
        "reproduction_scope": "D-FINE/RT-DETR Fine-grained Distribution Refinement SOTA transformer baseline",
        "status": "smoke-capable; ready",
    },
    "yolo11n-p2": {
        "family": "YOLOv11", "framework": "ultralytics", "weights": "yolo11n.pt", "size": "nano-p2",
        "reproduction_scope": "YOLO11n with P2 ultra-fine feature map (stride 4) for AdaPoth-Lite shared detector",
        "status": "smoke-capable; ready",
    },
    "yolo11n-p2-lite": {
        "family": "YOLOv11", "framework": "ultralytics", "weights": "yolo11n.pt", "size": "nano-p2-lite",
        "reproduction_scope": "YOLO11n-P2-lite (~3.2M params) lightweight shared detector for AdaPoth-Lite",
        "status": "smoke-capable; ready",
    },
}


DETECTOR_STATUS = {
    "yolov5m-compat": "Ultralytics compatibility preset configured; ready",
    "yolov5m-official": "Ultralytics official YOLOv5m preset configured; ready",
    "yolov8m": "Ultralytics medium preset configured; ready",
    "yolo11m": "Ultralytics medium preset configured; ready",
    "rt-detr-v1": "Ultralytics RT-DETR-L preset configured; ready",
    "rt-detr-v2": "Ultralytics RT-DETR-X preset configured; ready",
    "d-fine": "Ultralytics D-FINE/RT-DETR-L preset configured; ready",
    "yolo11n-p2": "Ultralytics YOLO11n-P2 shared detector preset configured; ready",
    "yolo11n-p2-lite": "Ultralytics YOLO11n-P2-lite shared detector preset configured; ready",
}


def get_baseline_preset(name: str) -> dict[str, Any]:
    try:
        return {"name": name, **BASELINE_PRESETS[name]}
    except KeyError as exc:
        raise ValueError(f"Unknown baseline preset {name!r}; choose from {sorted(BASELINE_PRESETS) + ['all']}") from exc
