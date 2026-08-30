import unittest
import numpy as np

from hrp4k.models.scout import (
    generate_scout_heatmap_gt,
    CandidateGenerator,
    CandidateRegion,
    evaluate_scout_regions,
)
from hrp4k.methods.adapoth import (
    apply_boundary_penalty,
    ScaledCropTransform,
    GlobalScaleTransform,
    make_adapoth_views,
)
from hrp4k.detectors.base import Detection


class TestAdaPothScoutUnit(unittest.TestCase):
    def test_mobilenetv3_scout_fpn_forward(self):
        import torch
        from hrp4k.models.scout import MobileNetV3Scout
        model = MobileNetV3Scout()
        self.assertGreater(model.count_parameters(), 100000)
        self.assertLess(model.count_parameters(), 600000)

        # Input 960x540
        dummy = torch.randn(1, 3, 540, 960)
        with torch.no_grad():
            out = model(dummy)
        # Expected Stride-8 output: (1, 1, 68, 120)
        self.assertEqual(out.shape, (1, 1, 68, 120))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_generate_scout_heatmap_gt(self):
        boxes = [
            [1000.0, 500.0, 200.0, 100.0],  # Box 1
            [2500.0, 1200.0, 300.0, 150.0],  # Box 2
        ]
        heat = generate_scout_heatmap_gt(
            boxes,
            img_w=3840,
            img_h=2160,
            heat_w=120,
            heat_h=68,
            expand_ratio=0.30,
        )
        self.assertEqual(heat.shape, (68, 120))
        self.assertGreaterEqual(heat.min(), 0.0)
        self.assertLessEqual(heat.max(), 1.0)
        self.assertGreater(heat.max(), 0.8)

    def test_candidate_generator(self):
        # Create a synthetic heatmap with 2 distinct Gaussian peaks
        heat = np.zeros((34, 60), dtype=np.float32)
        heat[10:14, 15:20] = 0.85
        heat[20:25, 40:46] = 0.90

        gen = CandidateGenerator(threshold=0.05, context_margin=0.30, k_max=4)
        candidates = gen.generate(heat, source_width=3840, source_height=2160)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertLessEqual(len(candidates), 4)

        for c in candidates:
            self.assertIsInstance(c, CandidateRegion)
            self.assertGreaterEqual(c.x0, 0)
            self.assertGreaterEqual(c.y0, 0)
            self.assertLessEqual(c.x1, 3840)
            self.assertLessEqual(c.y1, 2160)
            self.assertGreater(c.width, 0)
            self.assertGreater(c.height, 0)

    def test_candidate_generator_empty(self):
        # Empty heatmap should return 0 crops (no safety distortion)
        heat = np.zeros((34, 60), dtype=np.float32)
        gen = CandidateGenerator(threshold=0.05, context_margin=0.30, k_max=4)
        candidates = gen.generate(heat, source_width=3840, source_height=2160)
        self.assertEqual(len(candidates), 0)

    def test_evaluate_scout_regions(self):
        gt_boxes = [
            [1000.0, 500.0, 200.0, 100.0],
            [2500.0, 1200.0, 300.0, 150.0],
        ]
        candidates = [
            CandidateRegion(x0=900, y0=450, x1=1300, y1=650, score=0.9, component_id=1),
            CandidateRegion(x0=2400, y0=1100, x1=2900, y1=1400, score=0.85, component_id=2),
        ]
        res = evaluate_scout_regions(gt_boxes, candidates)
        self.assertEqual(res["total_gts"], 2)
        self.assertEqual(res["covered_gts"], 2)
        self.assertAlmostEqual(res["region_recall"], 1.0)
        self.assertAlmostEqual(res["false_region_rate"], 0.0)

    def test_apply_boundary_penalty(self):
        dets = [
            Detection(xyxy=(2.0, 50.0, 100.0, 150.0), score=0.90, category_id=0),   # Touches left boundary (<= 8)
            Detection(xyxy=(50.0, 50.0, 150.0, 150.0), score=0.90, category_id=0), # Center detection
        ]
        penalized = apply_boundary_penalty(dets, view_width=640, view_height=640, boundary_margin=8, penalty=0.70)
        self.assertAlmostEqual(penalized[0].score, 0.90 * 0.70)
        self.assertAlmostEqual(penalized[1].score, 0.90)

    def test_scaled_crop_transform_roundtrip(self):
        transform = ScaledCropTransform(x0=1000.0, y0=500.0, crop_w=640.0, crop_h=480.0, view_w=640.0, view_h=640.0)
        orig_box = np.array([[1100.0, 600.0, 1200.0, 700.0]])
        fwd = transform.forward_boxes(orig_box)
        inv = transform.inverse_boxes(fwd)
        np.testing.assert_allclose(orig_box, inv, atol=1e-4)

    def test_global_scale_transform_roundtrip(self):
        transform = GlobalScaleTransform(src_w=3840.0, src_h=2160.0, dst_w=960.0, dst_h=544.0)
        orig_box = np.array([[1000.0, 500.0, 2000.0, 1500.0]])
        fwd = transform.forward_boxes(orig_box)
        inv = transform.inverse_boxes(fwd)
        np.testing.assert_allclose(orig_box, inv, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
