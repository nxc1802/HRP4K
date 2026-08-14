from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .coco import SCALE_ORDER, iter_master_rows, load_split, scale_class
from .paths import SPLITS, image_path


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size), "min": float(arr.min()), "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std()),
    }


def _rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2: return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _js_divergence(left: list[float], right: list[float]) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12); midpoint = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
    return (kl(p, midpoint) + kl(q, midpoint)) / 2


def _ks_distance(left: list[float], right: list[float]) -> float:
    if not left or not right: return 0.0
    a, b = np.sort(left), np.sort(right); values = np.sort(np.concatenate((a, b)))
    return float(np.max(np.abs(np.searchsorted(a, values, side="right") / len(a) - np.searchsorted(b, values, side="right") / len(b))))


def analyze_dataset(data_dir: Path, output_dir: Path, quality_samples: int = 24) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_master_rows(data_dir))
    integrity: dict[str, Any] = {"dataset_dir": str(data_dir.resolve()), "splits": {}, "errors": [], "warnings": []}
    split_summary: dict[str, Any] = {}
    all_images = all_annotations = 0

    for split in SPLITS:
        data = load_split(data_dir, split)
        images, anns = data.get("images", []), data.get("annotations", [])
        image_ids = {int(im["id"]) for im in images}
        ann_image_ids = {int(a["image_id"]) for a in anns}
        available = sum(image_path(data_dir, split, im["file_name"]).is_file() for im in images)
        label_dir = data_dir / split / "labels"
        label_files = len(list(label_dir.glob("*.txt"))) if label_dir.is_dir() else 0
        invalid_boxes = 0
        image_by_id = {int(im["id"]): im for im in images}
        for ann in anns:
            im = image_by_id.get(int(ann["image_id"]))
            x, y, w, h = map(float, ann["bbox"])
            epsilon = 1e-6
            if not im or w <= 0 or h <= 0 or x < -epsilon or y < -epsilon or x + w > im["width"] + epsilon or y + h > im["height"] + epsilon:
                invalid_boxes += 1
        entry = {
            "declared_images": len(images), "available_images": available,
            "missing_images": len(images) - available, "label_files": label_files,
            "annotations": len(anns), "positive_images": len(ann_image_ids),
            "negative_images": len(image_ids - ann_image_ids), "orphan_annotation_image_ids": len(ann_image_ids - image_ids),
            "invalid_boxes": invalid_boxes,
        }
        integrity["splits"][split] = entry
        if entry["missing_images"]:
            integrity["warnings"].append(f"{split}: {entry['missing_images']} declared images are missing on disk")
        if invalid_boxes:
            integrity["errors"].append(f"{split}: {invalid_boxes} invalid bounding boxes")
        all_images += len(images); all_annotations += len(anns)
        split_rows = [r for r in rows if r["split"] == split]
        split_summary[split] = {
            **entry,
            "scale_counts": dict(Counter(r["scale_class"] for r in split_rows)),
            "box_width_px": _summary([r["width_px"] for r in split_rows]),
            "box_height_px": _summary([r["height_px"] for r in split_rows]),
            "area_ratio": _summary([r["area_ratio"] for r in split_rows]),
        }

    summary = {
        "images": all_images, "annotations": all_annotations,
        "positive_images": sum(v["positive_images"] for v in integrity["splits"].values()),
        "negative_images": sum(v["negative_images"] for v in integrity["splits"].values()),
        "resolution_counts": {}, "scale_counts": dict(Counter(r["scale_class"] for r in rows)),
        "box_width_px": _summary([r["width_px"] for r in rows]),
        "box_height_px": _summary([r["height_px"] for r in rows]),
        "area_ratio": _summary([r["area_ratio"] for r in rows]),
        "objects_per_positive_image": _summary(list(Counter((r["split"], r["image_id"]) for r in rows).values())),
        "paper_reference": {"images": 6003, "positive_images": 4003, "negative_images": 2000, "annotations": 7217, "resolution": "3840x2160"},
    }
    resolution_counts = Counter()
    for split in SPLITS:
        for im in load_split(data_dir, split).get("images", []):
            resolution_counts[f"{im['width']}x{im['height']}"] += 1
    summary["resolution_counts"] = dict(resolution_counts)

    spatial = {
        "x_center": _summary([r["x_center"] for r in rows]),
        "y_center": _summary([r["y_center"] for r in rows]),
        "y_bottom": _summary([r["y_bottom"] for r in rows]),
        "corr_y_bottom_log_area": float(np.corrcoef([r["y_bottom"] for r in rows], [r["log_area"] for r in rows])[0, 1]),
        "corr_y_center_log_area": float(np.corrcoef([r["y_center"] for r in rows], [r["log_area"] for r in rows])[0, 1]),
        "spearman_y_bottom_log_area": _spearman([r["y_bottom"] for r in rows], [r["log_area"] for r in rows]),
        "spearman_y_center_log_area": _spearman([r["y_center"] for r in rows], [r["log_area"] for r in rows]),
    }
    shape = {"width_px": summary["box_width_px"], "height_px": summary["box_height_px"], "aspect_ratio": _summary([r["aspect_ratio"] for r in rows])}
    density = {"objects_per_positive_image": summary["objects_per_positive_image"], "count_distribution": dict(Counter(r["objects_in_image"] for r in rows))}

    quality: list[dict[str, Any]] = []
    if quality_samples > 0:
        try:
            import cv2
            candidates = []
            for split in SPLITS:
                data = load_split(data_dir, split)
                positive_ids = {int(ann["image_id"]) for ann in data.get("annotations", [])}
                candidates.extend((split, im, int(im["id"]) in positive_ids) for im in data["images"] if image_path(data_dir, split, im["file_name"]).is_file())
            if candidates:
                indices = np.linspace(0, len(candidates) - 1, min(quality_samples, len(candidates)), dtype=int)
                for idx in indices:
                    split, im, positive = candidates[int(idx)]; path = image_path(data_dir, split, im["file_name"])
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is None: continue
                    quality.append({"split": split, "image_id": im["id"], "file_name": path.name, "positive": positive,
                                    "brightness": float(image.mean()), "contrast": float(image.std()),
                                    "sharpness_laplacian_var": float(cv2.Laplacian(image, cv2.CV_64F).var())})
        except ImportError:
            integrity["warnings"].append("OpenCV unavailable; image-quality sampling skipped")

    # Raw spatial grid and nearest-neighbour fields make Phase 3 joins reproducible.
    grid = np.zeros((8, 12), dtype=int)
    by_image: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gx = min(11, max(0, int(row["x_center"] * 12)))
        gy = min(7, max(0, int(row["y_center"] * 8)))
        grid[gy, gx] += 1
        by_image[(row["split"], row["image_id"])].append(row)
    for group in by_image.values():
        for row in group:
            distances = [math.hypot(row["x_center"] - other["x_center"], row["y_center"] - other["y_center"])
                         for other in group if other["annotation_id"] != row["annotation_id"]]
            row["nearest_object_distance"] = min(distances) if distances else None
            row["local_object_density"] = sum(distance <= 0.1 for distance in distances)
            row["difficulty_tags"] = "|".join(filter(None, [
                row["scale_class"], "far" if row["y_center"] < 0.5 else "near",
                "extreme_horizontal" if row["aspect_ratio"] >= 5 else "",
                "dense" if row["objects_in_image"] >= 5 else "isolated" if row["objects_in_image"] == 1 else "moderate",
            ]))
    spatial["grid_shape"] = [8, 12]
    spatial["center_grid"] = grid.tolist()
    y_bands = []
    for band in range(10):
        values = [row["area_ratio"] for row in rows if band / 10 <= row["y_center"] < (band + 1) / 10]
        y_bands.append({"band": f"{band / 10:.1f}-{(band + 1) / 10:.1f}", "area_ratio": _summary(values),
                        "conditional_variance": float(np.var(values)) if values else None})
    spatial["y_center_bands"] = y_bands

    split_shift = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            left_rows = [row for row in rows if row["split"] == left]; right_rows = [row for row in rows if row["split"] == right]
            left_counts = Counter(row["scale_class"] for row in left_rows); right_counts = Counter(row["scale_class"] for row in right_rows)
            split_shift[f"{left}_vs_{right}"] = {
                "scale_js_divergence": _js_divergence([left_counts[s] for s in SCALE_ORDER], [right_counts[s] for s in SCALE_ORDER]),
                "area_ratio_ks_distance": _ks_distance([row["area_ratio"] for row in left_rows], [row["area_ratio"] for row in right_rows]),
                "y_center_ks_distance": _ks_distance([row["y_center"] for row in left_rows], [row["y_center"] for row in right_rows]),
                "aspect_ratio_ks_distance": _ks_distance([row["aspect_ratio"] for row in left_rows], [row["aspect_ratio"] for row in right_rows]),
            }
    split_summary["distribution_shift"] = split_shift
    quality_by_label = {
        label: {metric: _summary([sample[metric] for sample in quality if sample["positive"] is positive])
                for metric in ("brightness", "contrast", "sharpness_laplacian_var")}
        for label, positive in (("positive", True), ("negative", False))
    }

    artifacts = {
        "dataset_integrity.json": integrity, "dataset_summary.json": summary,
        "scale_analysis.json": {"thresholds_area_ratio": [0.0005, 0.001, 0.0025], "counts": summary["scale_counts"]},
        "spatial_analysis.json": spatial, "position_scale_analysis.json": {
            s: {"count": len(v), "y_bottom": _summary([r["y_bottom"] for r in v]), "x_center": _summary([r["x_center"] for r in v])}
            for s in SCALE_ORDER for v in [[r for r in rows if r["scale_class"] == s]]},
        "shape_analysis.json": shape, "image_object_density.json": density,
        "split_analysis.json": split_summary,
        "domain_analysis.json": {"status": "unavailable", "reason": "COCO files do not contain city or pavement-material metadata"},
        "image_quality_analysis.json": {"sample_count": len(quality), "positive_vs_negative": quality_by_label, "samples": quality},
    }
    for name, payload in artifacts.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (output_dir / "difficulty_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (output_dir / "spatial_grid.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["y_bin", *[f"x{i}" for i in range(12)]])
            for idx, values in enumerate(grid.tolist()): writer.writerow([idx, *values])
    report = [
        "# HRP4K Dataset Analysis", "",
        f"- Declared images: {summary['images']:,}", f"- Declared annotations: {summary['annotations']:,}",
        f"- Positive / negative images: {summary['positive_images']:,} / {summary['negative_images']:,}",
        f"- Local available images: {sum(v['available_images'] for v in integrity['splits'].values()):,}",
        f"- Scale distribution: `{json.dumps(summary['scale_counts'], ensure_ascii=False)}`", "",
        f"- Spearman(y_center, log_area): {spatial['spearman_y_center_log_area']:.4f}",
        f"- Spearman(y_bottom, log_area): {spatial['spearman_y_bottom_log_area']:.4f}",
        f"- Split-shift diagnostics: `{json.dumps(split_shift, ensure_ascii=False)}`", "",
        "The official video-level train/valid/test split is preserved. Missing local train images are skipped, not re-split.",
        "City and pavement analysis cannot be reconstructed because those fields are absent from the released COCO JSON.",
    ]
    (output_dir / "dataset_analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"integrity": integrity, "summary": summary, "output_dir": str(output_dir)}
