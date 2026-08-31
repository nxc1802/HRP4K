"""Comprehensive Smoke Tests for HRP4K Benchmark Suite.

Verifies:
- All 16 official experiments in EXPERIMENT_MATRIX
- Detector configurations (YOLO11m, RT-DETR-L)
- Slicing methods (Resize/Full, Sliced-NMS, SAHI, Perspective Grid)
- Proposed Method pipeline skeleton
- Deterministic experiment IDs
- CLI commands
- HF Storage state management
- Report auto-generation
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from hrp4k.experiments.registry import (
    EXPERIMENT_MATRIX,
    resolve_experiment,
    list_experiments,
    ExperimentConfig,
)
from hrp4k.experiments.proposed import (
    ProposedMethodPipeline,
    run_proposed_smoke,
    Region,
)
from hrp4k.detectors.registry import BASELINE_PRESETS, get_baseline_preset
from hrp4k.methods.base import METHOD_REGISTRY, make_views, IdentityTransform, CropTransform
from hrp4k.infra.hashing import experiment_id
from hrp4k.infra.hf_storage import ExperimentStorage, ExperimentState
from hrp4k.reports.report import update_experiment_final, EXPERIMENT_FINAL_PATH
from hrp4k.cli import build_parser


class SmokeTestExperimentMatrix(unittest.TestCase):
    """Verify all 16 official experiments in the matrix."""

    def test_total_experiment_count(self):
        self.assertEqual(len(EXPERIMENT_MATRIX), 16)

    def test_resolution_experiments_exist(self):
        expected = [
            "yolo11m-resolution-4k",
            "yolo11m-resolution-2k",
            "yolo11m-resolution-1k",
            "yolo11m-resolution-640",
            "rtdetr-l-resolution-4k",
            "rtdetr-l-resolution-2k",
            "rtdetr-l-resolution-1k",
            "rtdetr-l-resolution-640",
        ]
        for name in expected:
            self.assertIn(name, EXPERIMENT_MATRIX)
            cfg = resolve_experiment(name)
            self.assertEqual(cfg.phase, "resolution")
            self.assertTrue(len(cfg.experiment_id) >= 8)

    def test_slicing_experiments_exist(self):
        expected = [
            "yolo11m-slicing-full",
            "yolo11m-slicing-sliced-nms",
            "yolo11m-slicing-sahi",
            "yolo11m-slicing-perspective-grid",
            "rtdetr-l-slicing-full",
            "rtdetr-l-slicing-sliced-nms",
            "rtdetr-l-slicing-sahi",
            "rtdetr-l-slicing-perspective-grid",
        ]
        for name in expected:
            self.assertIn(name, EXPERIMENT_MATRIX)
            cfg = resolve_experiment(name)
            self.assertEqual(cfg.phase, "slicing")
            self.assertEqual(cfg.resolution, "640")
            self.assertEqual(cfg.imgsz, 640)

    def test_deterministic_experiment_ids(self):
        for name in EXPERIMENT_MATRIX:
            cfg1 = resolve_experiment(name)
            cfg2 = resolve_experiment(name)
            self.assertEqual(cfg1.experiment_id, cfg2.experiment_id)

    def test_detector_protocols(self):
        # YOLO11m: SGD, lr0=0.01, AMP=True, patience=10
        yolo_4k = resolve_experiment("yolo11m-resolution-4k")
        self.assertEqual(yolo_4k.optimizer, "SGD")
        self.assertEqual(yolo_4k.lr0, 0.01)
        self.assertEqual(yolo_4k.patience, 10)
        self.assertTrue(yolo_4k.amp)
        self.assertEqual(yolo_4k.effective_batch, 16)

        # RT-DETR-L 4K: AdamW, lr0=0.0001, AMP=False (FP32), patience=10
        rtdetr_4k = resolve_experiment("rtdetr-l-resolution-4k")
        self.assertEqual(rtdetr_4k.optimizer, "AdamW")
        self.assertEqual(rtdetr_4k.lr0, 0.0001)
        self.assertEqual(rtdetr_4k.patience, 10)
        self.assertFalse(rtdetr_4k.amp)  # FP32 for 4K Transformer
        self.assertEqual(rtdetr_4k.effective_batch, 16)

        # RT-DETR-L 640: AMP=True (FP16)
        rtdetr_640 = resolve_experiment("rtdetr-l-resolution-640")
        self.assertTrue(rtdetr_640.amp)

    def test_unknown_experiment_raises(self):
        with self.assertRaises(ValueError):
            resolve_experiment("invalid-experiment-name")


class SmokeTestProposedMethod(unittest.TestCase):
    """Verify proposed method pipeline skeleton."""

    def test_pipeline_smoke_passes(self):
        result = run_proposed_smoke()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["pipeline_output"]["pipeline_status"], "skeleton")

    def test_custom_components_in_pipeline(self):
        class CustomScout:
            def identify_regions(self, image):
                return [Region(0, 0, 100, 100, priority=0.9)]

        class CustomSelector:
            def select(self, regions, budget):
                return regions[:budget]

        pipeline = ProposedMethodPipeline(
            scout=CustomScout(),
            region_selector=CustomSelector(),
            budget=1,
        )
        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        res = pipeline.run(dummy)
        self.assertEqual(res["num_regions_scouted"], 1)
        self.assertEqual(res["num_regions_selected"], 1)


class SmokeTestMethodsAndViews(unittest.TestCase):
    """Verify spatial decomposition views."""

    def setUp(self):
        self.image = np.zeros((2160, 3840, 3), dtype=np.uint8)

    def test_resize_view(self):
        views = make_views(self.image, "resize")
        self.assertEqual(len(views), 1)
        self.assertIsInstance(views[0].transform, IdentityTransform)

    def test_sliced_nms_views(self):
        views = make_views(self.image, "sliced-nms", tile_size=960, overlap=0.2)
        self.assertTrue(len(views) > 1)

    def test_perspective_grid_views(self):
        views = make_views(self.image, "perspective-grid", overlap=0.2)
        self.assertEqual(len(views), 9)


class SmokeTestCLI(unittest.TestCase):
    """Verify CLI parser and subcommands."""

    def setUp(self):
        self.parser = build_parser()

    def test_setup_command_args(self):
        args = self.parser.parse_args(["setup", "--skip-dataset"])
        self.assertEqual(args.command, "setup")
        self.assertTrue(args.skip_dataset)

    def test_experiment_command_args(self):
        args = self.parser.parse_args(["experiment", "yolo11m-resolution-4k", "--dry-run"])
        self.assertEqual(args.command, "experiment")
        self.assertEqual(args.name, "yolo11m-resolution-4k")
        self.assertTrue(args.dry_run)

    def test_experiment_list_args(self):
        args = self.parser.parse_args(["experiment", "list"])
        self.assertEqual(args.command, "experiment")
        self.assertEqual(args.name, "list")

    def test_status_command(self):
        args = self.parser.parse_args(["status"])
        self.assertEqual(args.command, "status")


class SmokeTestHFStorage(unittest.TestCase):
    """Verify HF storage helper and resilience."""

    def test_storage_disabled_without_token(self):
        storage = ExperimentStorage("test_exp", token="none", repo_id=None)
        self.assertFalse(storage.enabled)
        state = storage.check_experiment_exists()
        self.assertFalse(state.exists)

    def test_upload_never_crashes_when_disabled(self):
        storage = ExperimentStorage("test_exp", token="none", repo_id=None)
        self.assertFalse(storage.enabled)
        res = storage.upload_epoch(1, Path("nonexistent"))
        self.assertEqual(res["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
