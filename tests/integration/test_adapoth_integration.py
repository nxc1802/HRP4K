import unittest
import numpy as np
from pathlib import Path

from hrp4k.cli import build_parser
from hrp4k.config.resolver import resolve, to_dict
from hrp4k.config.validation import validate
from hrp4k.methods.base import make_views, METHOD_REGISTRY
from hrp4k.detectors.registry import BASELINE_PRESETS, get_baseline_preset


class TestAdaPothIntegration(unittest.TestCase):
    def test_cli_parser_adapoth_commands(self):
        parser = build_parser()

        # Test train-scout CLI parsing
        args = parser.parse_args(["train-scout", "--data", "HRP4K", "--epochs", "10", "--batch", "8", "--smoke"])
        self.assertEqual(args.command, "train-scout")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch, 8)
        self.assertTrue(args.smoke)

        # Test eval-scout CLI parsing
        args = parser.parse_args(["eval-scout", "--data", "HRP4K", "--weights", "checkpoints/scout.pt", "--k-max", "4"])
        self.assertEqual(args.command, "eval-scout")
        self.assertEqual(args.k_max, 4)

        # Test prepare-adapoth-crops CLI parsing
        args = parser.parse_args(["prepare-adapoth-crops", "--data", "HRP4K", "--stage", "stage2", "--crop-size", "640"])
        self.assertEqual(args.command, "prepare-adapoth-crops")
        self.assertEqual(args.stage, "stage2")
        self.assertEqual(args.crop_size, 640)

        # Test phase2 --method adapoth CLI parsing
        args = parser.parse_args(["phase2", "--method", "adapoth", "--scout-weights", "scout.pt", "--k-max", "4"])
        self.assertEqual(args.command, "phase2")
        self.assertEqual(args.method, "adapoth")
        self.assertEqual(args.k_max, 4)

    def test_adapoth_presets_and_registry(self):
        self.assertIn("yolo11n-p2", BASELINE_PRESETS)
        self.assertIn("yolo11n-p2-lite", BASELINE_PRESETS)
        self.assertIn("adapoth", METHOD_REGISTRY)
        self.assertIn("adapoth-oracle", METHOD_REGISTRY)

        p2_preset = get_baseline_preset("yolo11n-p2-lite")
        self.assertEqual(p2_preset["name"], "yolo11n-p2-lite")

    def test_adapoth_config_layers(self):
        cfg = resolve(detector="yolo11n_p2_lite", method="adapoth", profile="smoke")
        errors = validate(cfg)
        self.assertEqual(errors, [])
        self.assertEqual(cfg.detector.name, "yolo11n-p2-lite")
        self.assertEqual(cfg.method.name, "adapoth")

    def test_make_adapoth_views(self):
        dummy_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        views = make_views(dummy_img, method="adapoth", k_max=4)
        # Should contain 1 global view + candidate views (at least safety view if 0 comps)
        self.assertGreaterEqual(len(views), 2)
        self.assertEqual(views[0].metadata.get("type"), "global")

    def test_make_adapoth_oracle_views(self):
        dummy_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        gt_boxes = [[1000.0, 500.0, 200.0, 100.0], [2500.0, 1200.0, 300.0, 150.0]]
        views = make_views(dummy_img, method="adapoth-oracle", gt_boxes_4k=gt_boxes, k_max=4)
        self.assertEqual(len(views), 3)  # 1 global + 2 oracle crops


if __name__ == "__main__":
    unittest.main()
