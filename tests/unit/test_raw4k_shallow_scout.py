import unittest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from hrp4k.training.raw4k_shallow_scout import (
    Raw4KShallowScout,
    generate_raw4k_scout_gt,
    Raw4KScoutLoss,
    Raw4KCandidateGenerator,
    CandidateRegion,
    evaluate_raw4k_scout_regions,
)


class TestRaw4KShallowScout(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_model_forward_pass(self):
        # Create model (unpretrained weights for fast unit testing)
        model = Raw4KShallowScout(pretrained=False)
        self.assertGreater(model.count_parameters(), 0)
        self.assertLess(model.count_parameters(), 10000)  # Should be ~2.5k params

        # Test dummy 4K tensor: (1, 3, 2160, 3840)
        dummy_input = torch.randn(1, 3, 2160, 3840)
        with torch.no_grad():
            out = model(dummy_input)
        
        # Expected output resolution at stride 4: (1, 1, 540, 960)
        self.assertEqual(out.shape, (1, 1, 540, 960))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_generate_raw4k_scout_gt(self):
        boxes = [
            [1000.0, 500.0, 200.0, 100.0],
            [2500.0, 1200.0, 300.0, 150.0],
        ]
        heat = generate_raw4k_scout_gt(
            boxes,
            img_w=3840,
            img_h=2160,
            heat_w=960,
            heat_h=540,
            expand_ratio=0.20,
        )
        self.assertEqual(heat.shape, (540, 960))
        self.assertGreaterEqual(heat.min(), 0.0)
        self.assertLessEqual(heat.max(), 1.0)
        self.assertGreater(heat.max(), 0.70)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_loss_computation(self):
        criterion = Raw4KScoutLoss(lambda_cov=2.0)
        pred = torch.full((1, 1, 54, 96), 0.5, requires_grad=True)
        target = torch.zeros((1, 1, 54, 96))
        target[0, 0, 10:20, 20:30] = 1.0

        loss_dict = criterion(pred, target)
        self.assertIn("loss", loss_dict)
        self.assertIn("focal_loss", loss_dict)
        self.assertIn("coverage_loss", loss_dict)
        self.assertFalse(torch.isnan(loss_dict["loss"]))

        # Check backward gradient
        loss_dict["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.isnan(pred.grad).any())

    def test_candidate_generator(self):
        # Synthetic heatmap with 2 peaks
        heat = np.zeros((540, 960), dtype=np.float32)
        heat[100:150, 200:300] = 0.85
        heat[300:380, 600:750] = 0.90

        gen = Raw4KCandidateGenerator(threshold=0.30, context_margin=0.20, k_max=4)
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

    def test_evaluate_raw4k_scout_regions(self):
        gt_boxes = [
            [1000.0, 500.0, 200.0, 100.0],
            [2500.0, 1200.0, 300.0, 150.0],
        ]
        candidates = [
            CandidateRegion(x0=900, y0=450, x1=1300, y1=650, score=0.95, component_id=1, area=400*200),
            CandidateRegion(x0=2400, y0=1100, x1=2900, y1=1400, score=0.90, component_id=2, area=500*300),
        ]
        res = evaluate_raw4k_scout_regions(gt_boxes, candidates, img_w=3840, img_h=2160)

        self.assertEqual(res["total_gts"], 2)
        self.assertAlmostEqual(res["recall_75"], 1.0)
        self.assertAlmostEqual(res["recall_50"], 1.0)
        self.assertAlmostEqual(res["recall_90"], 1.0)
        self.assertAlmostEqual(res["false_region_rate"], 0.0)
        self.assertGreater(res["processed_area_ratio"], 0.0)
        self.assertLess(res["processed_area_ratio"], 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_flops_calculation(self):
        model = Raw4KShallowScout(pretrained=False)
        gflops = model.compute_flops_4k(2160, 3840)
        # Should be ~2.5 - 3.5 GFLOPs on raw 4K
        self.assertGreater(gflops, 1.0)
        self.assertLess(gflops, 10.0)


if __name__ == "__main__":
    unittest.main()
