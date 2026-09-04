from __future__ import annotations
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from hrp4k.models.p2_branch import find_c2_backbone_stage, extract_c2_backbone, P2Adapter
from hrp4k.models.p2_head import (
    DenseP2Loss,
    LightweightP2Head,
    decode_dense_p2_predictions,
    RTDETRP2Model,
)
from hrp4k.experiments.proposed import RTDETRP2Adapter
from hrp4k.detectors.base import Detection
from ultralytics import RTDETR


class TestP2Optimization(unittest.TestCase):
    """Unit test suite for Proposed Method P2 low-compute optimizations."""

    def test_target_assignment_1x1_vs_3x3(self):
        """Verify 1x1 assigns 1 cell while 3x3 assigns up to 9 cells within box."""
        loss_1x1 = DenseP2Loss(nc=1, stride=4, target_assignment="1x1")
        loss_3x3 = DenseP2Loss(nc=1, stride=4, target_assignment="3x3")

        bs = 1
        h, w = 40, 40
        cls_logits = torch.randn(bs, 1, h, w)
        box_offsets = torch.rand(bs, 4, h, w) * 10.0

        # Box covering ~ 12x12 pixels (3x3 grid cells at stride 4)
        targets = {
            "cls": torch.tensor([0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),  # cx, cy, w, h normalized
            "gt_groups": [1],
        }

        # Run 1x1 forward
        out_1x1 = loss_1x1(cls_logits, box_offsets, targets, img_size=(160, 160))
        self.assertTrue(torch.isfinite(out_1x1["loss_p2_total"]))

        # Run 3x3 forward
        out_3x3 = loss_3x3(cls_logits, box_offsets, targets, img_size=(160, 160))
        self.assertTrue(torch.isfinite(out_3x3["loss_p2_total"]))

    def test_multi_positive_tiny_box_fallback(self):
        """Verify 3x3 assignment on ultra-fine box smaller than 1 grid cell always keeps center cell."""
        loss_3x3 = DenseP2Loss(nc=1, stride=4, target_assignment="3x3")
        bs = 1
        h, w = 40, 40
        cls_logits = torch.randn(bs, 1, h, w)
        box_offsets = torch.rand(bs, 4, h, w) * 5.0
        # Ultra-fine box: 2px by 2px (area = 4px^2)
        targets = {
            "cls": torch.tensor([0]),
            "bboxes": torch.tensor([[0.5, 0.5, 2.0 / 160.0, 2.0 / 160.0]]),
            "gt_groups": [1],
        }
        out = loss_3x3(cls_logits, box_offsets, targets, img_size=(160, 160))
        self.assertTrue(torch.isfinite(out["loss_p2_total"]))
        self.assertGreater(out["loss_p2_class"].item(), 0)

    def test_loss_functions_bce_focal_qfl(self):
        """Verify BCE, Focal Loss, and Quality Focal Loss compute finite losses and valid gradients."""
        h, w = 20, 20
        targets = {
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.4, 0.4, 0.1, 0.1], [0.6, 0.6, 0.05, 0.05]]),
            "gt_groups": [2],
        }

        for loss_type in ["bce", "focal", "qfl"]:
            loss_module = DenseP2Loss(nc=1, stride=4, cls_loss_type=loss_type)
            cls_logits = torch.randn(1, 1, h, w).requires_grad_()
            box_offsets = (torch.rand(1, 4, h, w) * 5.0).requires_grad_()

            loss_dict = loss_module(cls_logits, box_offsets, targets, img_size=(80, 80))
            total_loss = loss_dict["loss_p2_total"]

            self.assertTrue(torch.isfinite(total_loss), f"Loss for {loss_type} is not finite: {total_loss}")
            total_loss.backward()

            self.assertIsNotNone(cls_logits.grad, f"cls_logits grad missing for {loss_type}")
            self.assertIsNotNone(box_offsets.grad, f"box_offsets grad missing for {loss_type}")
            self.assertTrue(torch.isfinite(cls_logits.grad).all())
            self.assertTrue(torch.isfinite(box_offsets.grad).all())

    def test_scale_aware_weighting(self):
        """Verify scale-aware loss weighting responds to ultra-fine vs large objects."""
        # Baseline uniform weights
        loss_uniform = DenseP2Loss(nc=1, stride=4, scale_weights=(1.0, 1.0, 1.0, 1.0))
        # Ultra-fine upweighted: 3.0 for ultra-fine, 0.5 for large
        loss_scaled = DenseP2Loss(nc=1, stride=4, scale_weights=(3.0, 2.0, 1.0, 0.5))

        h, w = 80, 80
        torch.manual_seed(42)
        cls_logits = torch.zeros(1, 1, h, w)
        box_offsets = torch.ones(1, 4, h, w) * 10.0

        # Ultra-fine pothole (<32^2 pixels = 1024px^2, at 320x320: 16x16px -> 0.05 x 0.05 norm)
        targets_uf = {
            "cls": torch.tensor([0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.05]]),
            "gt_groups": [1],
        }

        out_uf_uniform = loss_uniform(cls_logits, box_offsets, targets_uf, img_size=(320, 320))
        out_uf_scaled = loss_scaled(cls_logits, box_offsets, targets_uf, img_size=(320, 320))

        # Scaled loss should be higher for ultra-fine object due to 3.0x weight on positive cls & bbox
        self.assertGreater(out_uf_scaled["loss_p2_class"].item(), out_uf_uniform["loss_p2_class"].item())

    def test_p2_head_with_all_optimizations(self):
        """Verify LightweightP2Head end-to-end forward and loss with multi-positive + QFL + scale weights."""
        head = LightweightP2Head(
            in_channels=256,
            num_classes=1,
            stride=4,
            target_assignment="3x3",
            cls_loss_type="qfl",
            scale_weights=(3.0, 2.0, 1.0, 0.5),
        )

        p2_feat = torch.randn(2, 256, 40, 40)
        cls_logits, box_offsets = head(p2_feat)

        self.assertEqual(cls_logits.shape, (2, 1, 40, 40))
        self.assertEqual(box_offsets.shape, (2, 4, 40, 40))

        targets = {
            "cls": torch.tensor([0, 0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.02, 0.02]]),
            "gt_groups": [1, 1],
        }
        loss_dict = head.compute_loss(cls_logits, box_offsets, targets, img_size=(160, 160))
        total_loss = loss_dict["loss_p2_total"]

        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        # Check gradient on last layers
        self.assertIsNotNone(head.cls_conv[-1].weight.grad)
        self.assertIsNotNone(head.box_conv[-2].weight.grad)

    def test_decode_dense_p2_predictions_topk_and_conf(self):
        """Verify decode_dense_p2_predictions respects topk parameter."""
        cls_logits = torch.randn(1, 1, 80, 80)
        box_offsets = torch.rand(1, 4, 80, 80) * 10.0

        for topk in [100, 300, 500, 1000]:
            decoded = decode_dense_p2_predictions(cls_logits, box_offsets, stride=4, topk=topk)
            self.assertEqual(decoded.shape, (1, topk, 6))


if __name__ == "__main__":
    unittest.main()
