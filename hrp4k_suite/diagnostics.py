from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import scale_class
from .processing import METHOD_STATUS


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict): return raw.get("predictions", []), raw
    return raw, {"predictions": raw}


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

    methods: dict[str, Any] = {}
    for path in prediction_paths:
        predictions, payload = _load_predictions(path)
        method = str(payload.get("method") or path.stem)
        metrics_path = path.with_name(path.stem + "_metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        methods[method] = {"prediction_file": str(path), "predictions": len(predictions),
                           "processing": payload.get("summary", {}), "metrics": metrics}
    result = {
        "ground_truth": {"images": len(images), "annotations": len(gt.get("annotations", []))},
        "methods": methods, "method_reproduction_status": METHOD_STATUS,
        "effective_size": {
            str(res): {"median_width": float(np.median([r[f"width_at_{res}"] for r in effective_rows])) if effective_rows else 0,
                       "median_height": float(np.median([r[f"height_at_{res}"] for r in effective_rows])) if effective_rows else 0}
            for res in (640, 960, 1280, 1920)
        },
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Phase 3 — Smoke Diagnostic Report", "",
             "> Smoke outputs validate plumbing only; they are not scientific benchmark results.", "",
             f"Ground truth subset: {len(images)} images / {len(gt.get('annotations', []))} instances.", "",
             "## Effective object size after global resize", "",
             "| Width canvas | Median bbox width | Median bbox height |", "|---:|---:|---:|"]
    for resolution, values in result["effective_size"].items():
        lines.append(f"| {resolution} | {values['median_width']:.2f} | {values['median_height']:.2f} |")
    lines.extend(["", "## Methods", "", "| Method | Predictions | AP50 | AP50:95 | Mean calls | Mean latency (ms) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for method, value in methods.items():
        metric = value["metrics"]; processing = value["processing"]
        lines.append(f"| {method} | {value['predictions']} | {metric.get('AP50', 0):.4f} | {metric.get('AP50_95', 0):.4f} | "
                     f"{processing.get('mean_detector_calls', 0):.2f} | {processing.get('mean_latency_ms', 0):.2f} |")
    lines.extend(["", "## Interpretation boundary", "",
                  "The smoke subset and one-epoch model are only sufficient to verify data loading, training, coordinate remapping, fusion and evaluation.",
                  "Material/city conclusions are unavailable because those metadata are absent from the COCO release.",
                  "Learned adaptive methods remain external reproductions and must not be compared against these smoke baselines.", ""])
    (output_dir / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
