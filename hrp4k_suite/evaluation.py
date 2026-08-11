from __future__ import annotations

import json
import contextlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import SCALE_ORDER, scale_class


def _xywh_to_xyxy(box):
    x, y, w, h = map(float, box); return np.array([x, y, x + w, y + h], dtype=float)


def iou(a, b) -> float:
    a, b = _xywh_to_xyxy(a), _xywh_to_xyxy(b)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _evaluate_at(gt: dict[str, Any], predictions: list[dict[str, Any]], threshold: float, scale: str | None = None):
    images = {int(x["id"]): x for x in gt["images"]}
    grouped_gt = defaultdict(list)
    ignored_gt = defaultdict(list)
    for ann in gt["annotations"]:
        im = images[int(ann["image_id"])]
        ratio = float(ann["bbox"][2]) * float(ann["bbox"][3]) / (float(im["width"]) * float(im["height"]))
        if scale is None or scale_class(ratio) == scale:
            grouped_gt[int(ann["image_id"])].append(ann)
        else:
            ignored_gt[int(ann["image_id"])].append(ann)
    matched = {image_id: set() for image_id in grouped_gt}
    records = []
    for pred in sorted(predictions, key=lambda x: float(x.get("score", 0)), reverse=True):
        image_id = int(pred["image_id"]); best_iou, best_idx = 0.0, -1
        for idx, ann in enumerate(grouped_gt.get(image_id, [])):
            if idx in matched[image_id]: continue
            overlap = iou(pred["bbox"], ann["bbox"])
            if overlap > best_iou: best_iou, best_idx = overlap, idx
        is_tp = best_idx >= 0 and best_iou >= threshold
        if is_tp: matched[image_id].add(best_idx)
        if not is_tp and scale is not None and any(iou(pred["bbox"], ann["bbox"]) >= threshold for ann in ignored_gt.get(image_id, [])):
            continue
        records.append((float(pred.get("score", 0)), int(is_tp), int(not is_tp)))
    positives = sum(len(v) for v in grouped_gt.values())
    if not records:
        return {"ap": 0.0, "recall": 0.0, "precision": 0.0, "positives": positives, "records": []}
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([r[2] for r in records])
    recall = tp / max(positives, 1); precision = tp / np.maximum(tp + fp, 1)
    interp = [float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0 for level in np.linspace(0, 1, 101)]
    return {"ap": float(np.mean(interp)), "recall": float(recall[-1]), "precision": float(precision[-1]), "positives": positives, "records": records}


def evaluate(gt: dict[str, Any], predictions: list[dict[str, Any]], confidence: float = 0.25) -> dict[str, Any]:
    thresholds = np.arange(0.5, 0.96, 0.05)
    evaluations = {f"{t:.2f}": _evaluate_at(gt, predictions, float(t)) for t in thresholds}
    active = [p for p in predictions if float(p.get("score", 0)) >= confidence]
    operating = _evaluate_at(gt, active, 0.5)
    tp = sum(r[1] for r in operating["records"]); fp = sum(r[2] for r in operating["records"])
    fn = operating["positives"] - tp
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    negative_ids = {int(im["id"]) for im in gt["images"]} - {int(a["image_id"]) for a in gt["annotations"]}
    negative_fp = sum(int(p["image_id"]) in negative_ids for p in active)
    official = _coco_metrics(gt, predictions)
    return {
        "protocol": {"iou": "COCO-style 0.50:0.95", "ap_interpolation_points": 101, "confidence_operating_point": confidence},
        "num_images": len(gt["images"]), "num_ground_truth": len(gt["annotations"]), "num_predictions": len(predictions),
        "AP50": official.get("AP50", evaluations["0.50"]["ap"]),
        "AP75": official.get("AP75", evaluations["0.75"]["ap"]),
        "AP50_95": official.get("AP50_95", float(np.mean([v["ap"] for v in evaluations.values()]))),
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "FPPI": fp / max(len(gt["images"]), 1), "negative_FPPI": negative_fp / max(len(negative_ids), 1),
        "tp": tp, "fp": fp, "fn": fn,
        "scale": {s: {"AP50": _evaluate_at(gt, predictions, 0.5, s)["ap"], "recall50": _evaluate_at(gt, predictions, 0.5, s)["recall"]} for s in SCALE_ORDER},
        "coco_evaluator": official,
    }


def _coco_metrics(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Use pycocotools when available; fall back to the transparent NumPy evaluator."""
    if not predictions:
        return {"AP50_95": 0.0, "AP50": 0.0, "AP75": 0.0, "backend": "empty"}
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO()
        coco_gt.dataset = {
            **gt,
            "info": gt.get("info") or {}, "licenses": gt.get("licenses") or [],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(predictions)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = [int(im["id"]) for im in gt.get("images", [])]
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        return {
            "AP50_95": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]), "AR100": float(evaluator.stats[8]),
            "backend": "pycocotools",
        }
    except (ImportError, KeyError, ValueError, IndexError) as exc:
        return {"backend": "numpy-fallback", "warning": str(exc)}


def match_diagnostics(gt: dict[str, Any], predictions: list[dict[str, Any]], confidence: float = 0.25) -> list[dict[str, Any]]:
    """Per-image TP/FP/FN/localization counts used by Phase 3 without re-inference."""
    grouped_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    grouped_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt.get("annotations", []): grouped_gt[int(ann["image_id"])].append(ann)
    for pred in predictions:
        if float(pred.get("score", 0)) >= confidence: grouped_pred[int(pred["image_id"])].append(pred)
    output = []
    for image in gt.get("images", []):
        image_id = int(image["id"]); annotations = grouped_gt[image_id]
        preds = sorted(grouped_pred[image_id], key=lambda item: float(item.get("score", 0)), reverse=True)
        used: set[int] = set(); tp = fp = localization = 0
        for pred in preds:
            overlaps = [(iou(pred["bbox"], ann["bbox"]), idx) for idx, ann in enumerate(annotations) if idx not in used]
            best, index = max(overlaps, default=(0.0, -1))
            if best >= 0.5:
                tp += 1; used.add(index)
            elif best >= 0.1:
                localization += 1
            else:
                fp += 1
        output.append({"image_id": image_id, "ground_truth": len(annotations), "predictions": len(preds),
                       "tp": tp, "fp": fp, "fn": len(annotations) - tp, "localization_errors": localization})
    return output


def evaluate_files(gt_path: Path, prediction_path: Path, output_path: Path, confidence: float = 0.25):
    gt = json.loads(gt_path.read_text(encoding="utf-8")); raw = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = raw.get("predictions", raw) if isinstance(raw, dict) else raw
    metrics = evaluate(gt, predictions, confidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    diagnostics_path = output_path.with_name(output_path.stem + "_per_image.json")
    diagnostics_path.write_text(json.dumps(match_diagnostics(gt, predictions, confidence), indent=2), encoding="utf-8")
    return metrics
