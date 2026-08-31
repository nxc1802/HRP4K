import tempfile
import unittest
from pathlib import Path
from hrp4k.protocol.gates import preflight
from hrp4k.detectors.base import create_detector
from hrp4k.data.identity import EXPECTED_ANNOTATION_SHA256, verify_dataset_identity


class TestScientificGates(unittest.TestCase):
    def test_official_hashes_frozen(self):
        self.assertIn("train", EXPECTED_ANNOTATION_SHA256)
        self.assertIn("valid", EXPECTED_ANNOTATION_SHA256)
        self.assertIn("test", EXPECTED_ANNOTATION_SHA256)
        self.assertEqual(len(EXPECTED_ANNOTATION_SHA256["train"]), 64)

    def test_external_detectors_fail_fast_in_core(self):
        for name in ("rt-detr-v1", "rt-detr-v2", "d-fine", "yolov5m-official", "unknown-detector"):
            with self.assertRaises(ValueError):
                create_detector(name, "dummy.pt", category_id=0)

    def test_preflight_on_actual_dataset(self):
        data_dir = Path("HRP4K")
        if data_dir.is_dir():
            result = preflight(data_dir)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["official_dataset_identity"])


if __name__ == "__main__":
    unittest.main()
