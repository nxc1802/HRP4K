from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..inference.runner import predict_yolo
from ..evaluation.coco import evaluate_files
from ..infra.upload import ensure_weights
from ..methods.base import METHOD_REGISTRY

RUNNABLE_METHODS = ["resize", "sliced-nms", "perspective-grid", "sahi", "zoomdet-geometry", "zoomdet-neural"]


def run_phase_2(
    data_dir: Path,
    split: str,
    weights: Path | str,
    output_path: Path,
    method: str = "resize",
    limit: int | None = None,
    image_size: int | str = 640,
    confidence: float = 0.05,
    tile_size: int = 960,
    overlap: float = 0.2,
    device: str | None = None,
    warmup: int = 20,
    detector_name: str = "ultralytics",
    precision: str = "fp32",
    evaluate_after: bool = True,
    ground_truth: Path | None = None,
    eval_confidence: float = 0.25,
) -> dict[str, Any]:
    """Execute Phase 2 resolution allocation and canonical COCO prediction with multi-method support."""
    if detector_name in {"d-fine", "dfine"}:
        detector_name = "ultralytics"
    
    weights = ensure_weights(weights)
    resolved_imgsz = (2176, 3840) if str(image_size).strip().lower() in {"original", "4k", "native"} else image_size

    if method == "all":
        results = []
        base_dir = output_path if output_path.is_dir() or not output_path.suffix else output_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        gt_path = ground_truth or (data_dir / f"{split}.json")
        for m in RUNNABLE_METHODS:
            m_output = base_dir / f"{Path(weights).stem}_{m}_{split}_predictions.json"
            m_payload = predict_yolo(
                data_dir=data_dir,
                split=split,
                weights=weights,
                output_path=m_output,
                method=m,
                limit=limit,
                image_size=resolved_imgsz,
                confidence=confidence,
                tile_size=tile_size,
                overlap=overlap,
                device=device,
                warmup=warmup,
                detector_name=detector_name,
                precision=precision,
            )
            eval_metrics = None
            if evaluate_after and gt_path.exists():
                metrics_path = m_output.with_name(m_output.stem + "_metrics.json")
                eval_metrics = evaluate_files(gt_path, m_output, metrics_path, confidence=eval_confidence)
            
            results.append({
                "method": m,
                "predictions_path": str(m_output),
                "summary": m_payload.get("summary", {}),
                "metrics": eval_metrics,
            })
        
        summary_path = base_dir / f"{Path(weights).stem}_phase2_all_methods_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return {"method": "all", "summary_path": str(summary_path), "results": results, "summary": {"methods": len(results)}}

    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHOD_REGISTRY) + ['all']}")
    if METHOD_REGISTRY[method]["status"] == "external-required":
        raise RuntimeError(f"{method} requires its paper-faithful external training/runtime; no heuristic substitute is enabled")
    
    payload = predict_yolo(
        data_dir=data_dir,
        split=split,
        weights=weights,
        output_path=output_path,
        method=method,
        limit=limit,
        image_size=resolved_imgsz,
        confidence=confidence,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        warmup=warmup,
        detector_name=detector_name,
        precision=precision,
    )
    
    gt_path = ground_truth or (data_dir / f"{split}.json")
    if evaluate_after and gt_path.exists():
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        payload["metrics"] = evaluate_files(gt_path, output_path, metrics_path, confidence=eval_confidence)
    return payload
