from __future__ import annotations
import unittest
import numpy as np

try:
    import torch
    from ultralytics import RTDETR
    from hrp4k.models.p2_branch import find_c2_backbone_stage, P2Adapter, P2Branch
    from hrp4k.models.p2_head import P2HungarianMatcher, P2HeadLoss, P2QueryHead, RTDETRP2Model
    from hrp4k.inference.p2_fusion import fuse_native_and_p2_predictions, fuse_prediction_tensors
    from hrp4k.experiments.proposed import RTDETRP2Adapter, run_proposed_smoke
    from hrp4k.detectors.base import Detection
    HAS_TORCH_RTDETR = True
except ImportError:
    HAS_TORCH_RTDETR = False


@unittest.skipUnless(HAS_TORCH_RTDETR, "torch and ultralytics required for P2 feasibility test")
class TestP2Feasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.native_rtdetr = RTDETR("rtdetr-l.pt").model
        except Exception:
            cls.native_rtdetr = None

    def test_dynamic_c2_discovery(self):
        """Verify runtime shape inspection dynamically discovers C2 stage (stride 4, 128 channels)."""
        if self.native_rtdetr is None:
            self.skipTest("rtdetr-l.pt not available locally")
        layer_idx, in_channels = find_c2_backbone_stage(self.native_rtdetr, input_size=(640, 640))
        self.assertEqual(layer_idx, 1)
        self.assertEqual(in_channels, 128)

    def test_p2_adapter(self):
        """Verify P2 adapter architecture: C2 -> 1x1 Conv -> 3x3 Conv -> P2 (stride 4 preserved)."""
        adapter = P2Adapter(in_channels=128, out_channels=256)
        c2_dummy = torch.randn(2, 128, 160, 160)
        p2_out = adapter(c2_dummy)
        self.assertEqual(p2_out.shape, (2, 256, 160, 160))

    def test_p2_hungarian_matcher_and_loss(self):
        """Verify Hungarian matcher and P2 head loss computation."""
        loss_module = P2HeadLoss(nc=1)
        bs = 2
        nq = 100
        pred_bboxes = torch.rand(bs, nq, 4)  # normalized cxcywh
        pred_scores = torch.randn(bs, nq, 1)
        targets = {
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.3, 0.1, 0.2]]),
            "gt_groups": [1, 1],
        }
        loss_dict = loss_module(pred_bboxes, pred_scores, targets)
        self.assertIn("loss_p2_total", loss_dict)
        self.assertIn("loss_p2_class", loss_dict)
        self.assertIn("loss_p2_bbox", loss_dict)
        self.assertIn("loss_p2_giou", loss_dict)
        self.assertGreater(loss_dict["loss_p2_total"].item(), 0)

    def test_p2_query_head(self):
        """Verify P2QueryHead outputs queries and bounding boxes properly."""
        head = P2QueryHead(in_channels=256, num_queries=150, nc=1, num_decoder_layers=2)
        p2_feat = torch.randn(2, 256, 40, 40)
        bboxes, scores = head(p2_feat)
        self.assertEqual(bboxes.shape, (2, 150, 4))
        self.assertEqual(scores.shape, (2, 150, 1))
        self.assertTrue((bboxes >= 0.0).all() and (bboxes <= 1.0).all())

    def test_rtdetr_p2_model_eval_and_train_forward(self):
        """Verify RTDETRP2Model forward pass in both eval mode and train mode."""
        if self.native_rtdetr is None:
            self.skipTest("rtdetr-l.pt not available locally")
        model = RTDETRP2Model(native_model=self.native_rtdetr, nc=1, lambda_p2=0.25)
        model.eval()
        dummy_input = torch.zeros(2, 3, 640, 640)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertIn("native_preds", out)
        self.assertIn("p2_preds", out)
        self.assertEqual(out["native_preds"].shape[0], 2)
        self.assertEqual(out["native_preds"].shape[2], 6)
        self.assertEqual(out["p2_preds"].shape[0], 2)
        self.assertEqual(out["p2_preds"].shape[2], 6)

        model.train()
        train_input = torch.randn(2, 3, 640, 640, requires_grad=True)
        batch = {
            "img": train_input,
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.3, 0.3, 0.1, 0.2]]),
            "batch_idx": torch.tensor([0, 1]),
        }
        total_loss, loss_dict = model.loss(batch)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("p2_loss", loss_dict)
        self.assertTrue(total_loss.requires_grad)

        total_loss.backward()
        self.assertIsNotNone(model.p2_branch.adapter.conv1x1.weight.grad)
        self.assertIsNotNone(model.p2_head.bbox_head[0].weight.grad)

    def test_prediction_fusion(self):
        """Verify Concatenation + NMS Prediction Fusion."""
        det_native = [
            Detection((100.0, 100.0, 200.0, 200.0), 0.90, category_id=0),
            Detection((300.0, 300.0, 400.0, 400.0), 0.80, category_id=0),
        ]
        det_p2 = [
            Detection((102.0, 101.0, 198.0, 201.0), 0.85, category_id=0),
            Detection((50.0, 50.0, 70.0, 70.0), 0.75, category_id=0),
        ]
        fused = fuse_native_and_p2_predictions(det_native, det_p2, iou_threshold=0.5)
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0].score, 0.90)

    def test_proposed_smoke_pipeline(self):
        """Verify one-call proposed smoke pipeline function."""
        if self.native_rtdetr is None:
            self.skipTest("rtdetr-l.pt not available locally")
        res = run_proposed_smoke()
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["c2_channels"], 128)
        self.assertEqual(res["gradient_check"], "passed")


if __name__ == "__main__":
    unittest.main()
