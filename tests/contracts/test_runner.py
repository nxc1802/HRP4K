from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hrp4k.detectors.base import Detection
from hrp4k.inference.schema import validate_predictions
from hrp4k.inference.runner import predict_detector


class MockDetector:
    name = "mock"
    device = "cpu"
    def warmup(self, image, image_size): pass
    def predict(self, image, image_size, confidence):
        return [Detection((1.0, 2.0, 6.0, 8.0), 0.9, 7)]
    def metadata(self): return {"name": self.name, "framework": "test", "weights": None}


class RunnerTests(unittest.TestCase):
    def test_generic_runner_writes_canonical_manifest(self):
        try: import cv2
        except ImportError: self.skipTest("OpenCV not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "test" / "images"; images.mkdir(parents=True)
            cv2.imwrite(str(images / "one.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
            gt = {"images": [{"id": 1, "file_name": "one.jpg", "width": 30, "height": 20}],
                  "annotations": [], "categories": [{"id": 7, "name": "pothole"}]}
            (root / "test.json").write_text(json.dumps(gt), encoding="utf-8")
            payload = predict_detector(root, "test", MockDetector(), root / "predictions.json",
                                       "sliced-nms", image_size=16, warmup=1)
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(len(payload["experiment_id"]), 12)
            validate_predictions(gt, payload["predictions"])


if __name__ == "__main__": unittest.main()
