from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hrp4k_suite.diagnostics import diagnose
from hrp4k_suite.processing import make_views
from hrp4k_suite.training import train_yolo


class ProcessingTests(unittest.TestCase):
    def test_full_training_requires_explicit_authorization(self):
        with self.assertRaises(ValueError):
            train_yolo(Path("missing.yaml"), "missing.pt", Path("missing-run"), smoke=False)

    def test_view_counts_and_geometry(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        self.assertEqual(len(make_views(image, "resize")), 1)
        self.assertEqual(len(make_views(image, "uniform-2")), 4)
        self.assertEqual(len(make_views(image, "uniform-3")), 9)
        self.assertEqual(len(make_views(image, "sliced-nms", 960, 0.2)), 25)
        perspective = make_views(image, "perspective-grid", 960, 0.2)
        self.assertEqual(len(perspective), 9)
        self.assertTrue(all(view.source_width < 3840 for view in perspective))

    def test_missing_metrics_are_not_rendered_as_zero(self):
        gt = {"categories": [{"id": 0, "name": "pothole"}], "images": [{"id": 1, "width": 100, "height": 100}], "annotations": []}
        prediction = {"method": "resize", "predictions": [], "summary": {"compute_amplification_nominal_canvas": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); gt_path = root / "gt.json"; pred_path = root / "resize.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8"); pred_path.write_text(json.dumps(prediction), encoding="utf-8")
            result = diagnose(gt_path, [pred_path], root / "report")
            self.assertEqual(result["methods"]["resize"]["evaluation_status"], "not_evaluated")
            report = (root / "report" / "phase3_report.md").read_text(encoding="utf-8")
            self.assertIn("| resize | not_evaluated | 0 | N/A | N/A |", report)

    def test_metrics_json_is_ignored_when_wildcard_is_too_broad(self):
        gt = {"categories": [{"id": 0, "name": "pothole"}], "images": [], "annotations": []}
        prediction = {"method": "resize", "predictions": []}
        metrics = {"AP50": 0.5, "AP50_95": 0.3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); gt_path = root / "gt.json"; pred_path = root / "resize.json"; metrics_path = root / "resize_metrics.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8"); pred_path.write_text(json.dumps(prediction), encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            result = diagnose(gt_path, [pred_path, metrics_path], root / "report")
            self.assertEqual(list(result["methods"]), ["resize"])
            self.assertEqual(result["ignored_inputs"][0]["path"], str(metrics_path))


    def test_paired_failure_transitions_and_rescue_analysis(self):
        gt = {
            "categories": [{"id": 0, "name": "pothole"}],
            "images": [{"id": 1, "width": 1000, "height": 1000}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [100, 100, 20, 20]},
            ],
        }
        pred_a = {
            "method": "method_a",
            "predictions": [{"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9}],
        }
        pred_b = {
            "method": "method_b",
            "predictions": [
                {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
                {"image_id": 1, "category_id": 0, "bbox": [100, 100, 20, 20], "score": 0.9},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_path = root / "gt.json"
            pred_a_path = root / "method_a.json"
            pred_b_path = root / "method_b.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            pred_a_path.write_text(json.dumps(pred_a), encoding="utf-8")
            pred_b_path.write_text(json.dumps(pred_b), encoding="utf-8")

            # write per_image sidecars
            sidecar_a = [{"image_id": 1, "tp": 1, "fp": 0, "fn": 1, "localization_errors": 0, "matched_annotation_ids": [1]}]
            sidecar_b = [{"image_id": 1, "tp": 2, "fp": 0, "fn": 0, "localization_errors": 0, "matched_annotation_ids": [1, 2]}]
            (root / "method_a_per_image.json").write_text(json.dumps(sidecar_a), encoding="utf-8")
            (root / "method_b_per_image.json").write_text(json.dumps(sidecar_b), encoding="utf-8")
            (root / "method_a_metrics.json").write_text(json.dumps({"AP50": 0.5, "AP50_95": 0.5}), encoding="utf-8")
            (root / "method_b_metrics.json").write_text(json.dumps({"AP50": 1.0, "AP50_95": 1.0}), encoding="utf-8")

            result = diagnose(gt_path, [pred_a_path, pred_b_path], root / "report")
            paired = result["paired_image_analysis"]["method_a__vs__method_b"]
            self.assertEqual(paired["delta_tp"], 1)
            self.assertEqual(paired["delta_fn"], -1)
            self.assertEqual(paired["object_rescue_transition"]["miss_to_detected"], 1)
            self.assertEqual(paired["object_rescue_transition"]["detected_to_miss"], 0)
            self.assertEqual(paired["object_rescue_transition"]["net_rescued_objects"], 1)


if __name__ == "__main__":
    unittest.main()

