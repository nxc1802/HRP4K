from __future__ import annotations
import unittest
import numpy as np

try:
    import torch
    from ultralytics import RTDETR
    from hrp4k.models.p2_branch import find_c2_backbone_stage, extract_c2_backbone, P2Adapter, P2Branch
    from hrp4k.models.p2_head import (
        DenseP2Loss,
        LightweightP2Head,
        decode_dense_p2_predictions,
        RTDETRP2Model,
    )
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

    def test_dynamic_c2_discovery_and_extraction(self):
        """Verify runtime shape inspection and direct C2 extraction from backbone."""
        if self.native_rtdetr is None:
            self.skipTest("rtdetr-l.pt not available locally")
        layer_idx, in_channels = find_c2_backbone_stage(self.native_rtdetr, input_size=(640, 640))
        self.assertEqual(layer_idx, 1)
        self.assertEqual(in_channels, 128)

        x = torch.zeros(2, 3, 640, 640)
        c2 = extract_c2_backbone(self.native_rtdetr, x, c2_layer_idx=layer_idx)
        self.assertEqual(c2.shape, (2, 128, 160, 160))

    def test_p2_adapter(self):
        """Verify P2 adapter architecture: C2 -> 1x1 Conv -> 3x3 Conv -> P2 (stride 4 preserved)."""
        adapter = P2Adapter(in_channels=128, out_channels=256)
        c2_dummy = torch.randn(2, 128, 160, 160)
        p2_out = adapter(c2_dummy)
        self.assertEqual(p2_out.shape, (2, 256, 160, 160))

    def test_dense_p2_loss(self):
        """Verify Dense anchor-free loss computation for P2."""
        loss_module = DenseP2Loss(nc=1, stride=4)
        bs = 2
        cls_logits = torch.randn(bs, 1, 160, 160)
        box_offsets = torch.rand(bs, 4, 160, 160) * 20.0
        targets = {
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.3, 0.3, 0.02, 0.02]]),
            "gt_groups": [1, 1],
        }
        loss_dict = loss_module(cls_logits, box_offsets, targets, img_size=(640, 640))
        self.assertIn("loss_p2_total", loss_dict)
        self.assertIn("loss_p2_class", loss_dict)
        self.assertIn("loss_p2_bbox", loss_dict)
        self.assertIn("loss_p2_giou", loss_dict)
        self.assertGreater(loss_dict["loss_p2_total"].item(), 0)

    def test_lightweight_p2_head_and_decoding(self):
        """Verify LightweightP2Head forward and dense prediction decoding."""
        head = LightweightP2Head(in_channels=256, num_classes=1, stride=4)
        p2_feat = torch.randn(2, 256, 160, 160)
        cls_logits, box_offsets = head(p2_feat)
        self.assertEqual(cls_logits.shape, (2, 1, 160, 160))
        self.assertEqual(box_offsets.shape, (2, 4, 160, 160))

        decoded = decode_dense_p2_predictions(cls_logits, box_offsets, stride=4, topk=300)
        self.assertEqual(decoded.shape, (2, 300, 6))

    def test_rtdetr_p2_model_frozen_native_and_training(self):
        """Verify native RT-DETR is frozen and only P2 branch/head receives gradients."""
        if self.native_rtdetr is None:
            self.skipTest("rtdetr-l.pt not available locally")
        model = RTDETRP2Model(native_model=self.native_rtdetr, nc=1, freeze_native=True)
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

        # Training pass: verify gradient isolation
        model.p2_branch.train()
        model.p2_head.train()
        train_input = torch.randn(2, 3, 640, 640)
        batch = {
            "img": train_input,
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.3, 0.3, 0.02, 0.02]]),
            "batch_idx": torch.tensor([0, 1]),
        }
        total_loss, loss_dict = model.loss(batch)
        self.assertIn("loss_p2_total", loss_dict)
        self.assertTrue(total_loss.requires_grad)

        total_loss.backward()
        # Verify native model has NO gradients (strictly frozen)
        native_grads = [p.grad for p in model.native_model.parameters() if p.grad is not None]
        self.assertEqual(len(native_grads), 0, "Native RT-DETR parameters must be strictly frozen")
        # Verify P2 parameters received gradients
        self.assertIsNotNone(model.p2_branch.adapter.conv1x1.weight.grad)
        self.assertIsNotNone(model.p2_head.cls_conv[-1].weight.grad)

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
        self.assertTrue(res["native_frozen_verified"])
        self.assertEqual(res["gradient_check"], "passed")


if __name__ == "__main__":
    unittest.main()
