from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..data.coco import load_split
from ..data.paths import image_path
from ..data.manifest import dataset_manifest
from ..detectors.base import Detection, DetectorAdapter
from ..detectors.ultralytics import UltralyticsAdapter
from ..infra.environment import runtime_metadata
from ..infra.hashing import experiment_id
from ..infra.timing import Timer, cuda_synchronize_if_needed
from ..methods.base import METHOD_REGISTRY, make_views, nms, _starts


def _as_detection(value: Detection | dict[str, Any]) -> Detection:
    if isinstance(value, Detection):
        return value
    return Detection(tuple(map(float, value["xyxy"])), float(value["score"]), int(value["category_id"]))


def _summary(image_meta: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str) -> tuple[float, float, float, float]:
        values = [float(row[key]) for row in image_meta]
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        return float(np.mean(values)), float(np.median(values)), float(np.percentile(values, 95)), float(np.std(values))

    mean_e2e, median_e2e, p95_e2e, std_e2e = stats("end_to_end_latency_ms")
    mean_processor, _, _, _ = stats("processor_latency_ms")
    mean_detector, _, _, _ = stats("detector_latency_ms")
    calls = [row["detector_calls"] for row in image_meta]
    return {
        "images": len(image_meta), "predictions": len(predictions),
        "mean_processor_latency_ms": mean_processor, "mean_detector_latency_ms": mean_detector,
        "mean_end_to_end_latency_ms": mean_e2e, "median_end_to_end_latency_ms": median_e2e,
        "p50_end_to_end_latency_ms": median_e2e, "p95_end_to_end_latency_ms": p95_e2e,
        "std_end_to_end_latency_ms": std_e2e,
        "mean_detector_calls": float(np.mean(calls)) if calls else 0.0,
        "p95_detector_calls": float(np.percentile(calls, 95)) if calls else 0.0,
        "mean_processed_area_ratio": float(np.mean([row["processed_area_ratio"] for row in image_meta])) if image_meta else 0.0,
        "mean_nominal_detector_canvas_pixels": float(np.mean([row["nominal_detector_canvas_pixels"] for row in image_meta])) if image_meta else 0.0,
        "compute_amplification_nominal_canvas": float(np.mean([row["compute_amplification_nominal_canvas"] for row in image_meta])) if image_meta else 0.0,
    }


def predict_detector(
    data_dir: Path, split: str, detector: DetectorAdapter, output_path: Path, method: str = "resize",
    *, limit: int | None = None, image_size: int = 640, confidence: float = 0.05,
    tile_size: int = 960, overlap: float = 0.2, warmup: int = 20, precision: str = "fp32",
    scout_weights: Path | str | None = None, context_margin: float = 0.20, k_max: int = 4,
    boundary_penalty: float = 0.70,
) -> dict[str, Any]:
    """Framework-agnostic detector runner producing the canonical experiment schema."""
    if method not in METHOD_REGISTRY or METHOD_REGISTRY[method]["status"] == "external-required":
        raise ValueError(f"Method {method!r} is not runnable in core; inspect `hrp4k status`")
    if method == "sahi":
        if not isinstance(detector, UltralyticsAdapter):
            raise ValueError("Official SAHI core integration currently requires an UltralyticsAdapter")
        return _predict_sahi(data_dir, split, detector, output_path, limit=limit, image_size=image_size,
                             confidence=confidence, tile_size=tile_size, overlap=overlap, warmup=warmup,
                             precision=precision)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Prediction requires the 'vision' dependencies") from exc
    coco = load_split(data_dir, split)
    images = [im for im in coco.get("images", []) if image_path(data_dir, split, im["file_name"]).is_file()]
    if limit is not None:
        images = images[:limit]

    img_to_anns: dict[int, list[list[float]]] = {}
    for ann in coco.get("annotations", []):
        img_to_anns.setdefault(int(ann["image_id"]), []).append(list(map(float, ann["bbox"])))

    warmup_count = min(max(0, warmup), len(images))
    for image in images[:warmup_count]:
        source = cv2.imread(str(image_path(data_dir, split, image["file_name"])))
        if source is None:
            continue
        gts = img_to_anns.get(int(image["id"]), [])
        for view in make_views(source, method, tile_size, overlap, scout_weights=scout_weights,
                               context_margin=context_margin, k_max=k_max, gt_boxes_4k=gts,
                               device=getattr(detector, "device", None)):
            detector.warmup(view.image, image_size)
    cuda_synchronize_if_needed()
    torch = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    predictions: list[dict[str, Any]] = []
    image_meta: list[dict[str, Any]] = []
    for image in images:
        with Timer() as end_to_end_timer:
            with Timer() as decode_timer:
                source = cv2.imread(str(image_path(data_dir, split, image["file_name"])))
            if source is None:
                continue
            gts = img_to_anns.get(int(image["id"]), [])
            with Timer() as processor_timer:
                views = make_views(source, method, tile_size, overlap, scout_weights=scout_weights,
                                   context_margin=context_margin, k_max=k_max, gt_boxes_4k=gts,
                                   device=getattr(detector, "device", None))
            candidates: list[dict[str, Any]] = []
            detector_latency = 0.0
            if hasattr(detector, "predict_batch") and len(views) > 1:
                view_imgsz = max(views[0].image.shape[:2])
                with Timer() as detector_timer:
                    batch_dets = detector.predict_batch([v.image for v in views], view_imgsz, confidence)
                detector_latency = detector_timer.elapsed_ms
                for view, detections in zip(views, batch_dets):
                    # Boundary penalty for local crop views in AdaPoth
                    if view.metadata.get("type") == "local_crop" and boundary_penalty < 1.0:
                        vh, vw = view.image.shape[:2]
                        from ..methods.adapoth import apply_boundary_penalty
                        detections = apply_boundary_penalty(detections, vw, vh, boundary_margin=8, penalty=boundary_penalty)
                    for raw_detection in detections:
                        detection = _as_detection(raw_detection)
                        mapped_box = view.map_box(detection.xyxy)
                        if mapped_box[2] > 0.01 and mapped_box[3] > 0.01:
                            candidates.append({"image_id": int(image["id"]), "category_id": detection.category_id,
                                               "bbox": mapped_box, "score": detection.score})
            else:
                for view in views:
                    view_imgsz = image_size if method == "resize" else max(view.image.shape[:2])
                    with Timer() as detector_timer:
                        detections = detector.predict(view.image, view_imgsz, confidence)
                    detector_latency += detector_timer.elapsed_ms
                    if view.metadata.get("type") == "local_crop" and boundary_penalty < 1.0:
                        vh, vw = view.image.shape[:2]
                        from ..methods.adapoth import apply_boundary_penalty
                        detections = apply_boundary_penalty(detections, vw, vh, boundary_margin=8, penalty=boundary_penalty)
                    for raw_detection in detections:
                        detection = _as_detection(raw_detection)
                        mapped_box = view.map_box(detection.xyxy)
                        if mapped_box[2] > 0.01 and mapped_box[3] > 0.01:
                            candidates.append({"image_id": int(image["id"]), "category_id": detection.category_id,
                                               "bbox": mapped_box, "score": detection.score})
            with Timer() as fusion_timer:
                merged, suppressed = nms(candidates)
        predictions.extend(merged)
        source_pixels = sum(view.source_width * view.source_height for view in views)
        single_canvas = (image_size[0] * image_size[1]) if isinstance(image_size, (list, tuple)) else (int(image_size) * int(image_size))
        canvas_pixels = len(views) * single_canvas
        image_meta.append({"image_id": int(image["id"]), "method": method,
                           "decode_latency_ms": decode_timer.elapsed_ms, "processor_latency_ms": processor_timer.elapsed_ms,
                           "detector_latency_ms": detector_latency, "fusion_latency_ms": fusion_timer.elapsed_ms,
                           "end_to_end_latency_ms": end_to_end_timer.elapsed_ms, "detector_calls": len(views),
                           "processed_source_pixels": source_pixels,
                           "processed_area_ratio": source_pixels / (source.shape[0] * source.shape[1]),
                           "nominal_detector_canvas_pixels": canvas_pixels,
                           "compute_amplification_nominal_canvas": canvas_pixels / single_canvas if single_canvas > 0 else 1.0,
                           "fusion_suppression_count": suppressed, "predictions": len(merged)})
    ds_meta = dataset_manifest(data_dir, split)
    config = {"dataset": ds_meta, "detector": detector.metadata(),
              "method": {"name": method, "tile_size": tile_size, "overlap": overlap},
              "runtime": {"image_size": image_size, "confidence": confidence, "warmup_images": warmup_count,
                          "precision": precision, "limit": limit}}
    summary = _summary(image_meta, predictions)
    summary["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch is not None and torch.cuda.is_available() else None
    payload = {"schema_version": "1.0", "experiment_id": experiment_id(config), "method": method,
               "dataset": ds_meta, "detector": detector.metadata(),
               "method_config": config["method"], "runtime": runtime_metadata(getattr(detector, "device", None), precision, image_size, warmup_count),
               "settings": {**config["runtime"], "tile_size": tile_size, "overlap": overlap,
                            "latency_protocol": "synchronized end-to-end: decode + processor + detector + remap + fusion"},
               "predictions": predictions, "image_metadata": image_meta, "summary": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _predict_sahi(
    data_dir: Path, split: str, detector: UltralyticsAdapter, output_path: Path, *, limit: int | None,
    image_size: int, confidence: float, tile_size: int, overlap: float, warmup: int, precision: str,
) -> dict[str, Any]:
    if precision != "fp32":
        raise ValueError("SAHI fp16 is not enabled until its backend precision path is explicitly verified")
    try:
        import cv2
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise RuntimeError("Official SAHI requires `pip install -e '.[sahi]'`") from exc
    coco = load_split(data_dir, split)
    category_ids = [int(item["id"]) for item in coco.get("categories", [])]
    if len(category_ids) != 1:
        raise ValueError("SAHI integration requires the single-class HRP4K dataset")
    images = [im for im in coco.get("images", []) if image_path(data_dir, split, im["file_name"]).is_file()]
    if limit is not None:
        images = images[:limit]
    if detector.device:
        device = detector.device
    else:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(detector.weights),
                                                confidence_threshold=confidence, device=device, image_size=tile_size)
    def infer(source):
        return get_sliced_prediction(source, model, slice_height=tile_size, slice_width=tile_size,
                                     overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                     perform_standard_pred=False,
                                     postprocess_type="NMS", postprocess_match_metric="IOU",
                                     postprocess_match_threshold=0.5, verbose=0)
    warmup_count = min(max(0, warmup), len(images))
    for image in images[:warmup_count]:
        infer(str(image_path(data_dir, split, image["file_name"])))
    predictions = []
    image_meta = []
    for image in images:
        with Timer() as end_to_end:
            result = infer(str(image_path(data_dir, split, image["file_name"])))
        rows = []
        for item in result.object_prediction_list:
            box = item.bbox
            rows.append({"image_id": int(image["id"]), "category_id": category_ids[0],
                         "bbox": [float(box.minx), float(box.miny), float(box.maxx-box.minx), float(box.maxy-box.miny)],
                         "score": float(item.score.value)})
        predictions.extend(rows)
        source = cv2.imread(str(image_path(data_dir, split, image["file_name"])))
        source_height, source_width = source.shape[:2]
        slice_width, slice_height = min(tile_size, source_width), min(tile_size, source_height)
        detector_calls = len(_starts(source_width, slice_width, overlap)) * len(_starts(source_height, slice_height, overlap))
        processed_pixels = detector_calls * slice_width * slice_height
        canvas_pixels = detector_calls * image_size * image_size
        image_meta.append({"image_id": int(image["id"]), "method": "sahi", "decode_latency_ms": 0.0,
                           "processor_latency_ms": 0.0, "detector_latency_ms": end_to_end.elapsed_ms,
                           "fusion_latency_ms": 0.0, "end_to_end_latency_ms": end_to_end.elapsed_ms,
                           "detector_calls": detector_calls, "processed_source_pixels": processed_pixels,
                           "processed_area_ratio": processed_pixels / (source_height * source_width),
                           "nominal_detector_canvas_pixels": canvas_pixels,
                           "compute_amplification_nominal_canvas": float(detector_calls), "fusion_suppression_count": 0,
                           "predictions": len(rows)})
    ds_meta = dataset_manifest(data_dir, split)
    config = {"dataset": ds_meta, "detector": detector.metadata(),
              "method": {"name": "sahi", "slice_width": tile_size, "slice_height": tile_size, "overlap": overlap,
                         "postprocess_type": "NMS", "postprocess_metric": "IOU", "postprocess_threshold": 0.5},
              "runtime": {"warmup_images": warmup_count, "precision": precision}}
    payload = {"schema_version": "1.0", "experiment_id": experiment_id(config), "method": "sahi",
               "dataset": ds_meta, "detector": detector.metadata(),
               "method_config": config["method"], "runtime": runtime_metadata(device, precision, image_size, warmup_count),
               "settings": {**config["method"], "image_size": image_size, "confidence": confidence,
                            "warmup_images": warmup_count, "latency_protocol": "SAHI synchronized wall-clock end-to-end"},
               "predictions": predictions, "image_metadata": image_meta,
               "summary": {**_summary(image_meta, predictions), "peak_vram_mb": None}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def predict_yolo(
    data_dir: Path, split: str, weights: Path | str, output_path: Path, method: str = "resize",
    limit: int | None = None, image_size: int = 640, confidence: float = 0.05,
    tile_size: int = 960, overlap: float = 0.2, device: str | None = None, warmup: int = 20,
    detector_name: str = "ultralytics", precision: str = "fp32",
    scout_weights: Path | str | None = None, context_margin: float = 0.20, k_max: int = 4,
    boundary_penalty: float = 0.70,
) -> dict[str, Any]:
    coco = load_split(data_dir, split)
    category_ids = [int(category["id"]) for category in coco.get("categories", [])]
    if len(category_ids) != 1:
        raise ValueError(f"Expected exactly one HRP4K category, found {category_ids}")
    category_id = category_ids[0]
    from ..detectors.base import create_detector
    detector = create_detector(detector_name, weights, category_id, device, precision, allow_ultralytics=True)
    return predict_detector(data_dir, split, detector, output_path, method, limit=limit,
                            image_size=image_size, confidence=confidence, tile_size=tile_size,
                            overlap=overlap, warmup=warmup, precision=precision,
                            scout_weights=scout_weights, context_margin=context_margin,
                            k_max=k_max, boundary_penalty=boundary_penalty)
