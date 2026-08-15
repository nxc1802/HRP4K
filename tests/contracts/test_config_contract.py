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


    def test_unknown_fields_raise_error(self):
        with self.assertRaises(ValueError) as ctx:
            resolve(overrides={"runtime": {"invalid_field_xyz": 123}})
        self.assertIn("invalid_field_xyz", str(ctx.exception))

    def test_nested_method_parameters(self):
        config = resolve(overrides={
            "method": {
                "name": "sliced-nms",
                "parameters": {
                    "tile_size": 1280,
                    "overlap": 0.25,
                }
            }
        })
        self.assertEqual(config.method.name, "sliced-nms")
        self.assertEqual(config.method.tile_size, 1280)
        self.assertEqual(config.method.overlap, 0.25)

    def test_experiment_yaml_resolution(self):
        exp_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "yolo11m_resize_smoke.yaml"
        config = resolve(config_path=exp_path)
        errors = validate(config)
        self.assertEqual(errors, [])
        self.assertEqual(config.experiment.experiment_name, "yolo11m_resize_smoke")
        self.assertEqual(config.runtime.limit, 2)
        self.assertEqual(config.runtime.warmup, 1)

    def test_canonical_detectors_resolve(self):
        for det in ("yolo11m", "yolov8m", "yolov5m", "yolov5m_compat", "yolov5m_official", "rt_detr_v1", "rt_detr_v2", "d_fine"):
            cfg = resolve(detector=det)
            self.assertTrue(bool(cfg.detector.name))


if __name__ == "__main__":
    unittest.main()
