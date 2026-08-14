import unittest
from pathlib import Path
from hrp4k.config.resolver import resolve, to_dict
from hrp4k.config.validation import validate
from hrp4k.infra.hashing import experiment_id


class TestConfigContract(unittest.TestCase):
    def test_resolve_default_base(self):
        config = resolve()
        self.assertEqual(config.schema_version, "hrp4k.config.v1")
        self.assertEqual(config.dataset.root, "HRP4K")
        self.assertEqual(config.detector.name, "ultralytics")
        self.assertEqual(config.method.name, "resize")

    def test_resolve_modular_layers(self):
        config = resolve(detector="yolo11m", method="sliced_nms", profile="smoke")
        self.assertEqual(config.detector.checkpoint, "yolo11m.pt")
        self.assertEqual(config.method.name, "sliced-nms")
        self.assertEqual(config.method.tile_size, 960)
        self.assertEqual(config.experiment.profile, "smoke")
        self.assertTrue(config.training.smoke)
        self.assertEqual(config.training.epochs, 1)

    def test_validation_valid_config(self):
        config = resolve(profile="smoke")
        errors = validate(config)
        self.assertEqual(errors, [])

    def test_validation_catches_invalid_detector_and_eval(self):
        config = resolve()
        config.detector.input_size = -100
        config.detector.confidence = 2.5
        config.runtime.precision = "invalid_precision"
        errors = validate(config)
        self.assertTrue(any("input_size" in err for err in errors))
        self.assertTrue(any("confidence" in err for err in errors))
        self.assertTrue(any("precision" in err for err in errors))

    def test_deterministic_experiment_id(self):
        config1 = resolve(detector="yolo11m", method="resize", profile="smoke")
        config2 = resolve(detector="yolo11m", method="resize", profile="smoke")
        config3 = resolve(detector="yolov8m", method="resize", profile="smoke")
        id1 = experiment_id(to_dict(config1))
        id2 = experiment_id(to_dict(config2))
        id3 = experiment_id(to_dict(config3))
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertEqual(len(id1), 12)


if __name__ == "__main__":
    unittest.main()
