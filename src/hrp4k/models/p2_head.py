from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from .p2_branch import find_c2_backbone_stage, extract_c2_backbone, P2Branch, _unwrap_sequential


# ---------------------------------------------------------------------------
# Dense Loss for Lightweight P2 Auxiliary Head
# ---------------------------------------------------------------------------

class DenseP2Loss(nn.Module):
    """Anchor-free dense loss for Lightweight P2 detector head.

    Computes:
    L_P2 = L_cls (BCE/Focal) + lambda_box * L_box (L1) + lambda_giou * L_giou (GIoU)
    """

    def __init__(
        self,
        nc: int = 1,
        stride: int = 4,
        loss_gain: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.stride = stride
        self.loss_gain = loss_gain or {"cls": 1.0, "box": 2.0, "giou": 2.0}

    def forward(
        self,
        cls_logits: torch.Tensor,    # (B, nc, H, W)
        box_offsets: torch.Tensor,   # (B, 4, H, W) in image pixels [l, t, r, b]
        targets: dict[str, Any],     # dict with cls, bboxes (normalized cxcywh), gt_groups
        img_size: tuple[int, int] = (640, 640),
    ) -> dict[str, torch.Tensor]:
        bs, _, h, w = cls_logits.shape
        device = cls_logits.device
        stride = self.stride

        # Grid centers in pixel space
        y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H, W)

        gt_cls = targets["cls"]
        gt_bboxes = targets["bboxes"]  # (total_gt, 4) in normalized cxcywh
        gt_groups = targets.get("gt_groups", [])

        target_cls = torch.zeros_like(cls_logits)
        pos_mask = torch.zeros((bs, h, w), dtype=torch.bool, device=device)
        target_boxes_ltrb = torch.zeros((bs, 4, h, w), device=device)

        pred_boxes_xyxy_list: list[torch.Tensor] = []
        target_boxes_xyxy_list: list[torch.Tensor] = []

        # Reconstruct predicted bounding boxes in xyxy
        pred_l = box_offsets[:, 0]
        pred_t = box_offsets[:, 1]
        pred_r = box_offsets[:, 2]
        pred_b = box_offsets[:, 3]

        pred_x1 = grid_x.unsqueeze(0) - pred_l
        pred_y1 = grid_y.unsqueeze(0) - pred_t
        pred_x2 = grid_x.unsqueeze(0) + pred_r
        pred_y2 = grid_y.unsqueeze(0) + pred_b

        img_h, img_w = float(img_size[0]), float(img_size[1])
        gt_start = 0

        for b in range(bs):
            num_gt = gt_groups[b] if b < len(gt_groups) else 0
            if num_gt == 0:
                continue

            b_gt_cls = gt_cls[gt_start:gt_start + num_gt]
            b_gt_bboxes = gt_bboxes[gt_start:gt_start + num_gt]
            gt_start += num_gt

            # Sort by area descending so smaller/ultra-fine potholes are assigned last
            # and overwrite overlapping grid centers
            areas = b_gt_bboxes[:, 2] * b_gt_bboxes[:, 3]
            sort_indices = torch.argsort(areas, descending=True)

            for idx in sort_indices:
                bw = float(b_gt_bboxes[idx, 2] * img_w)
                bh = float(b_gt_bboxes[idx, 3] * img_h)
                cx = float(b_gt_bboxes[idx, 0] * img_w)
                cy = float(b_gt_bboxes[idx, 1] * img_h)

                gx1 = cx - bw / 2.0
                gy1 = cy - bh / 2.0
                gx2 = cx + bw / 2.0
                gy2 = cy + bh / 2.0

                # Nearest P2 grid cell center
                cx_idx = int(torch.clamp(torch.round(torch.tensor(cx / stride - 0.5)), 0, w - 1).item())
                cy_idx = int(torch.clamp(torch.round(torch.tensor(cy / stride - 0.5)), 0, h - 1).item())

                c_idx = int(torch.clamp(b_gt_cls[idx], 0, self.nc - 1).item())
                target_cls[b, c_idx, cy_idx, cx_idx] = 1.0
                pos_mask[b, cy_idx, cx_idx] = True

                grid_cx = (cx_idx + 0.5) * stride
                grid_cy = (cy_idx + 0.5) * stride

                # Target ltrb distance offsets (non-negative)
                target_boxes_ltrb[b, 0, cy_idx, cx_idx] = max(0.0, grid_cx - gx1)
                target_boxes_ltrb[b, 1, cy_idx, cx_idx] = max(0.0, grid_cy - gy1)
                target_boxes_ltrb[b, 2, cy_idx, cx_idx] = max(0.0, gx2 - grid_cx)
                target_boxes_ltrb[b, 3, cy_idx, cx_idx] = max(0.0, gy2 - grid_cy)

                # Reconstruct valid positive box for GIoU
                px1_c = pred_x1[b, cy_idx, cx_idx]
                py1_c = pred_y1[b, cy_idx, cx_idx]
                px2_c = pred_x2[b, cy_idx, cx_idx]
                py2_c = pred_y2[b, cy_idx, cx_idx]

                p_box = torch.stack([
                    torch.min(px1_c, px2_c),
                    torch.min(py1_c, py2_c),
                    torch.max(px1_c, px2_c),
                    torch.max(py1_c, py2_c),
                ], dim=-1)
                t_box = torch.tensor([gx1, gy1, gx2, gy2], device=device, dtype=p_box.dtype)

                pred_boxes_xyxy_list.append(p_box.unsqueeze(0))
                target_boxes_xyxy_list.append(t_box.unsqueeze(0))

        num_pos = max(1.0, float(pos_mask.sum().item()))

        # 1. Classification Loss (BCE with Logits)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, target_cls, reduction="sum") / num_pos

        # 2. Box L1 and GIoU Loss
        if pos_mask.any() and pred_boxes_xyxy_list:
            pred_ltrb_pos = box_offsets.permute(0, 2, 3, 1)[pos_mask]
            target_ltrb_pos = target_boxes_ltrb.permute(0, 2, 3, 1)[pos_mask]
            loss_box = F.l1_loss(pred_ltrb_pos, target_ltrb_pos, reduction="sum") / num_pos

            p_xyxy = torch.cat(pred_boxes_xyxy_list, dim=0)
            t_xyxy = torch.cat(target_boxes_xyxy_list, dim=0)
            giou = ops.generalized_box_iou(p_xyxy, t_xyxy)
            loss_giou = (1.0 - torch.diag(giou)).sum() / num_pos
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (
            self.loss_gain["cls"] * loss_cls
            + self.loss_gain["box"] * loss_box
            + self.loss_gain["giou"] * loss_giou
        )

        return {
            "loss_p2_total": total_loss,
            "loss_p2_class": loss_cls,
            "loss_p2_bbox": loss_box,
            "loss_p2_giou": loss_giou,
        }


# ---------------------------------------------------------------------------
# Prediction Decoding for Dense P2 Head
# ---------------------------------------------------------------------------

def decode_dense_p2_predictions(
    cls_logits: torch.Tensor,
    box_offsets: torch.Tensor,
    stride: int = 4,
    topk: int = 300,
    conf_threshold: float = 0.001,
) -> torch.Tensor:
    """Decodes dense classification logits and box offsets into (B, topk, 6) detection tensor.

    Args:
        cls_logits: (B, nc, H, W) logits
        box_offsets: (B, 4, H, W) [l, t, r, b] in pixels
        stride: spatial stride (default: 4)
        topk: maximum number of predictions per image (default: 300)
        conf_threshold: confidence score cutoff

    Returns:
        torch.Tensor: (B, topk, 6) tensor containing [x1, y1, x2, y2, score, cls_id]
    """
    bs, nc, h, w = cls_logits.shape
    device = cls_logits.device

    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

    pred_l = box_offsets[:, 0]
    pred_t = box_offsets[:, 1]
    pred_r = box_offsets[:, 2]
    pred_b = box_offsets[:, 3]

    pred_x1_raw = grid_x.unsqueeze(0) - pred_l
    pred_y1_raw = grid_y.unsqueeze(0) - pred_t
    pred_x2_raw = grid_x.unsqueeze(0) + pred_r
    pred_y2_raw = grid_y.unsqueeze(0) + pred_b

    pred_x1 = torch.min(pred_x1_raw, pred_x2_raw)
    pred_y1 = torch.min(pred_y1_raw, pred_y2_raw)
    pred_x2 = torch.max(pred_x1_raw, pred_x2_raw)
    pred_y2 = torch.max(pred_y1_raw, pred_y2_raw)

    boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)  # (B, H, W, 4)
    boxes = boxes.view(bs, -1, 4)  # (B, H*W, 4)

    scores = cls_logits.sigmoid().view(bs, nc, -1)  # (B, nc, H*W)

    img_w_max = float(w * stride)
    img_h_max = float(h * stride)

    batch_preds: list[torch.Tensor] = []
    for b in range(bs):
        b_scores, b_classes = scores[b].max(dim=0)  # (H*W,), (H*W,)
        b_boxes = boxes[b]

        k = min(topk, b_scores.shape[0])
        top_scores, top_idx = torch.topk(b_scores, k=k)
        top_boxes = b_boxes[top_idx]
        top_classes = b_classes[top_idx].float()

        # Clamp box coordinates within canvas
        top_boxes[:, [0, 2]] = torch.clamp(top_boxes[:, [0, 2]], min=0.0, max=img_w_max)
        top_boxes[:, [1, 3]] = torch.clamp(top_boxes[:, [1, 3]], min=0.0, max=img_h_max)

        pred = torch.cat([top_boxes, top_scores.unsqueeze(-1), top_classes.unsqueeze(-1)], dim=-1)
        batch_preds.append(pred)

    return torch.stack(batch_preds, dim=0)


# ---------------------------------------------------------------------------
# Lightweight Dense P2 Head Module
# ---------------------------------------------------------------------------

class LightweightP2Head(nn.Module):
    """Lightweight Dense Anchor-Free Detection Head for P2 (Stride 4).

    Eliminates Transformer query decoders and O(N^2) cross-attention over large token grids.
    Memory complexity is strictly linear with spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        stride: int = 4,
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.nc = num_classes
        self.stride = stride

        # Classification convolutional branch
        cls_layers: list[nn.Module] = []
        for _ in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            ])
        cls_layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))
        self.cls_conv = nn.Sequential(*cls_layers)

        # Bounding box regression convolutional branch
        box_layers: list[nn.Module] = []
        for _ in range(num_convs):
            box_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(),
            ])
        box_layers.extend([
            nn.Conv2d(in_channels, 4, kernel_size=1),
            nn.ReLU(),  # Distance offsets must be non-negative
        ])
        self.box_conv = nn.Sequential(*box_layers)

        self.criterion = DenseP2Loss(nc=num_classes, stride=stride)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Focal loss prior initialization for classification head bias: -log((1-p)/p) with p=0.01 -> -4.59
        prior_prob = 0.01
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_conv[-1].bias, bias_val)

    def forward(self, p2_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: P2 features -> (cls_logits, box_offsets in pixels)."""
        cls_logits = self.cls_conv(p2_feat)
        box_offsets = self.box_conv(p2_feat) * self.stride
        return cls_logits, box_offsets

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        box_offsets: torch.Tensor,
        targets: dict[str, Any],
        img_size: tuple[int, int] = (640, 640),
    ) -> dict[str, torch.Tensor]:
        return self.criterion(cls_logits, box_offsets, targets, img_size=img_size)


# Backward-compatible alias
P2DenseHead = LightweightP2Head
P2QueryHead = LightweightP2Head
P2HeadLoss = DenseP2Loss


# ---------------------------------------------------------------------------
# Unified Frozen RT-DETR + P2 Model Container
# ---------------------------------------------------------------------------

class RTDETRP2Model(nn.Module):
    """Combines Frozen RT-DETR-L baseline with Lightweight Dense P2 Auxiliary Head."""

    def __init__(
        self,
        native_model: nn.Module,
        c2_layer_idx: int | None = None,
        c2_channels: int | None = None,
        p2_channels: int = 256,
        nc: int = 1,
        input_size: tuple[int, int] = (640, 640),
        freeze_native: bool = True,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.freeze_native = freeze_native

        det_model, sub_modules, _ = _unwrap_sequential(native_model)
        self.native_model = det_model

        if freeze_native:
            self.native_model.eval()
            for p in self.native_model.parameters():
                p.requires_grad = False

        if c2_layer_idx is None or c2_channels is None:
            c2_layer_idx, c2_channels = find_c2_backbone_stage(self.native_model, input_size=input_size)

        self.c2_layer_idx = c2_layer_idx
        self.c2_channels = c2_channels

        # Attach Lightweight P2 Branch & Dense Head
        self.p2_branch = P2Branch(c2_layer_idx=c2_layer_idx, in_channels=c2_channels, out_channels=p2_channels)
        self.p2_head = LightweightP2Head(in_channels=p2_channels, num_classes=nc, stride=4)

    @property
    def names(self) -> dict[int, str]:
        return getattr(self.native_model, "names", {i: str(i) for i in range(self.nc)})

    @names.setter
    def names(self, value: dict[int, str]) -> None:
        self.native_model.names = value

    @property
    def yaml(self) -> dict[str, Any]:
        return getattr(self.native_model, "yaml", {})

    @property
    def args(self) -> dict[str, Any]:
        return getattr(self.native_model, "args", {})

    def forward(
        self,
        x: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward pass for evaluation / inference."""
        # 1. Native RT-DETR forward (frozen)
        with torch.no_grad():
            native_out = self.native_model(x)

        # 2. Extract C2 and forward P2 Dense Head
        c2_feat = extract_c2_backbone(self.native_model, x, c2_layer_idx=self.c2_layer_idx)
        p2_feat = self.p2_branch(c2_feat)
        cls_logits, box_offsets = self.p2_head(p2_feat)

        # 3. Decode P2 predictions
        p2_formatted = decode_dense_p2_predictions(
            cls_logits=cls_logits,
            box_offsets=box_offsets,
            stride=self.p2_head.stride,
            topk=300,
        )

        if isinstance(native_out, (list, tuple)):
            native_formatted = native_out[0]
        else:
            native_formatted = native_out

        return {
            "native_preds": native_formatted,
            "p2_preds": p2_formatted,
            "p2_raw": (cls_logits, box_offsets),
            "native_raw": native_out,
        }

    def loss(
        self,
        batch: dict[str, Any],
        preds: Any = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute P2 loss: only P2 parameters receive gradients."""
        img = batch["img"]
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        with torch.no_grad():
            c2_feat = extract_c2_backbone(self.native_model, img, c2_layer_idx=self.c2_layer_idx)

        p2_feat = self.p2_branch(c2_feat)
        cls_logits, box_offsets = self.p2_head(p2_feat)

        loss_dict = self.p2_head.compute_loss(
            cls_logits=cls_logits,
            box_offsets=box_offsets,
            targets=targets,
            img_size=(img.shape[-2], img.shape[-1]),
        )
        total_p2_loss = loss_dict["loss_p2_total"]

        return total_p2_loss, loss_dict
