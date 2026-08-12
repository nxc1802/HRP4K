from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import scale_class
from .processing import METHOD_STATUS
from .predictions import validate_predictions


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    predictions = raw.get("predictions") if isinstance(raw, dict) else raw
    if not isinstance(predictions, list):
        raise ValueError("not a prediction document: missing predictions list")
    required = {"image_id", "category_id", "bbox", "score"}
    if any(not isinstance(item, dict) or not required.issubset(item) for item in predictions):
        raise ValueError("not a prediction document: records do not match canonical COCO prediction schema")
    payload = raw if isinstance(raw, dict) else {"predictions": raw}
    return predictions, payload


def _pareto_methods(methods: dict[str, Any]) -> list[str]:
    candidates = []
    for name, value in methods.items():
        if value["evaluation_status"] != "evaluated": continue
        ap = value["metrics"].get("AP50_95")
        cost = value["processing"].get("compute_amplification_nominal_canvas")
        if ap is not None and cost is not None: candidates.append((name, float(ap), float(cost)))
    return [name for name, ap, cost in candidates
            if not any(other_ap >= ap and other_cost <= cost and (other_ap > ap or other_cost < cost)
                       for other_name, other_ap, other_cost in candidates if other_name != name)]


def _format_metric(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def diagnose(gt_path: Path, prediction_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    """Build Phase 3 artifacts exclusively from saved predictions (no inference)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in gt.get("images", [])}
    effective_rows = []
    for ann in gt.get("annotations", []):
        im = images[int(ann["image_id"])]
        x, y, width, height = map(float, ann["bbox"])
        ratio = width * height / (float(im["width"]) * float(im["height"]))
        row: dict[str, Any] = {"image_id": int(ann["image_id"]), "annotation_id": int(ann["id"]),
                               "scale": scale_class(ratio), "original_width": width, "original_height": height,
                               "x_center": (x + width / 2) / im["width"], "y_center": (y + height / 2) / im["height"]}
        for resolution in (640, 960, 1280, 1920):
            factor = resolution / float(im["width"])
            row[f"width_at_{resolution}"] = width * factor
            row[f"height_at_{resolution}"] = height * factor
        effective_rows.append(row)
    if effective_rows:
        with (output_dir / "effective_object_sizes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(effective_rows[0])); writer.writeheader(); writer.writerows(effective_rows)

    methods: dict[str, Any] = {}; ignored_inputs = []
    for path in prediction_paths:
        try:
            predictions, payload = _load_predictions(path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            ignored_inputs.append({"path": str(path), "reason": str(exc)})
            continue
        predictions = validate_predictions(gt, predictions)
        method = str(payload.get("method") or path.stem)
        metrics_path = path.with_name(path.stem + "_metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        sidecar_path = metrics_path.with_name(metrics_path.stem + "_per_image.json")
        per_image = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else []
        scale = metrics.get("scale", {})
        ap_large = scale.get("large", {}).get("AP50")
        ap_uf = scale.get("ultra_fine", {}).get("AP50")
        methods[method] = {"prediction_file": str(path), "predictions": len(predictions),
                           "processing": payload.get("summary", {}), "metrics": metrics,
                           "evaluation_status": "evaluated" if metrics_path.exists() else "not_evaluated",
                           "per_image": per_image,
                           "scale_sensitivity_ap50": (float(ap_large) - float(ap_uf)) if ap_large is not None and ap_uf is not None else None,
                           "localization_gap_ap50_ap75": (float(metrics["AP50"]) - float(metrics["AP75"])) if "AP50" in metrics and "AP75" in metrics else None}
    paired = {}
    method_names = list(methods)
    for left_index, left_name in enumerate(method_names):
        left_rows = {int(row["image_id"]): row for row in methods[left_name]["per_image"]}
        for right_name in method_names[left_index + 1:]:
            right_rows = {int(row["image_id"]): row for row in methods[right_name]["per_image"]}
            common = sorted(set(left_rows) & set(right_rows))
            if not common: continue
            paired[f"{left_name}__vs__{right_name}"] = {
                "images": len(common),
                f"{left_name}_wins": sum(left_rows[i]["tp"] > right_rows[i]["tp"] for i in common),
                f"{right_name}_wins": sum(right_rows[i]["tp"] > left_rows[i]["tp"] for i in common),
                "ties": sum(left_rows[i]["tp"] == right_rows[i]["tp"] for i in common),
            }
    result = {
        "ground_truth": {"images": len(images), "annotations": len(gt.get("annotations", []))},
        "methods": methods, "method_reproduction_status": METHOD_STATUS,
        "effective_size": {
            str(res): {"median_width": float(np.median([r[f"width_at_{res}"] for r in effective_rows])) if effective_rows else 0,
                       "median_height": float(np.median([r[f"height_at_{res}"] for r in effective_rows])) if effective_rows else 0}
            for res in (640, 960, 1280, 1920)
        }, "paired_image_analysis": paired, "ignored_inputs": ignored_inputs,
    }
    result["accuracy_compute_pareto"] = _pareto_methods(methods)
    (output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Phase 3 — Diagnostic Report", "",
             "> Smoke outputs validate plumbing only; only fully trained, fully evaluated runs may be used as scientific benchmark results.", "",
             f"Ground truth subset: {len(images)} images / {len(gt.get('annotations', []))} instances.", "",
             "## Effective object size after global resize", "",
             "| Width canvas | Median bbox width | Median bbox height |", "|---:|---:|---:|"]
    for resolution, values in result["effective_size"].items():
        lines.append(f"| {resolution} | {values['median_width']:.2f} | {values['median_height']:.2f} |")
    lines.extend(["", "## Methods", "", "| Method | Status | Predictions | AP50 | AP50:95 | FPPI official | CAF nominal | P95 end-to-end (ms) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for method, value in methods.items():
        metric = value["metrics"]; processing = value["processing"]
        lines.append(f"| {method} | {value['evaluation_status']} | {value['predictions']} | {_format_metric(metric.get('AP50'))} | "
                     f"{_format_metric(metric.get('AP50_95'))} | {_format_metric(metric.get('FPPI_official'))} | "
                     f"{_format_metric(processing.get('compute_amplification_nominal_canvas'), 2)} | {_format_metric(processing.get('p95_end_to_end_latency_ms'), 2)} |")
    lines.extend(["", f"Accuracy–compute Pareto methods: `{', '.join(result['accuracy_compute_pareto']) or 'N/A'}`."])
    if ignored_inputs:
        lines.extend(["", "## Ignored non-prediction inputs", "", "```json", json.dumps(ignored_inputs, indent=2), "```"])
    if paired:
        lines.extend(["", "## Per-image paired analysis", "", "```json", json.dumps(paired, indent=2), "```"])
    lines.extend(["", "## Interpretation boundary", "",
                  "The smoke subset and one-epoch model are only sufficient to verify data loading, training, coordinate remapping, fusion and evaluation.",
                  "Material/city conclusions are unavailable because those metadata are absent from the COCO release.",
                  "Learned adaptive methods remain external reproductions and must not be compared against these smoke baselines.", ""])
    (output_dir / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
