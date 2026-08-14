from __future__ import annotations

import unittest

from hrp4k_suite.evaluation import evaluate


def ground_truth():
    return {
        "info": {}, "licenses": [], "categories": [{"id": 7, "name": "pothole"}],
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "positive.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "negative.jpg"},
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
    }


class EvaluationTests(unittest.TestCase):
    def test_perfect_prediction(self):
        metrics = evaluate(ground_truth(), [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 0.99}])
        self.assertGreater(metrics["AP50"], 0.999)
        self.assertGreater(metrics["AP50_95"], 0.999)

    def test_official_fppi_uses_negative_images(self):
        predictions = [
            {"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 0.99},
            {"image_id": 2, "category_id": 7, "bbox": [20, 20, 10, 10], "score": 0.90},
        ]
        metrics = evaluate(ground_truth(), predictions)
        self.assertEqual(metrics["FPPI_official"], 1.0)
        self.assertEqual(metrics["FPPI_all_images"], 0.5)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(ground_truth(), [{"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.99}])


    def test_scale_conditioned_metrics(self):
        gt = ground_truth()
        predictions = [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 0.99}]
        metrics = evaluate(gt, predictions)
        self.assertIn("scale", metrics)
        for s in ("ultra_fine", "fine", "medium", "large"):
            self.assertIn(s, metrics["scale"])
            self.assertIn("AP50", metrics["scale"][s])
            self.assertIn("AP75", metrics["scale"][s])
            self.assertIn("AP50_95", metrics["scale"][s])
            self.assertIn("recall50", metrics["scale"][s])


if __name__ == "__main__":
    unittest.main()

