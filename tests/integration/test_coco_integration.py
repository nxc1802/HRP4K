from __future__ import annotations

import os
import unittest

from hrp4k.evaluation.coco import evaluate
from tests.contracts.test_evaluation import ground_truth


class CocoIntegrationTests(unittest.TestCase):
    def test_official_backend(self):
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            if os.getenv("HRP4K_REQUIRE_PYCOCOTOOLS") == "1":
                self.fail("pycocotools is required by the evaluation integration job")
            self.skipTest("pycocotools is not installed")
        metrics = evaluate(
            ground_truth(),
            [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 0.99}],
        )
        self.assertEqual(metrics["coco_evaluator"]["backend"], "pycocotools")


if __name__ == "__main__":
    unittest.main()
