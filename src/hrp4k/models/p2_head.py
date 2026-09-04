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
# ---------------------------------------------------------------------------
# Dense Loss for Lightweight P2 Auxiliary Head
# ---------------------------------------------------------------------------

class DenseP2Loss(nn.Module):
    """Anchor-free dense loss for Lightweight P2 detector head.

    Supports:
    - Target Assignment: "1x1" (nearest center cell) vs "3x3" (center 3x3 region positive cells)
    - Classification Loss: "bce" (Binary Cross Entropy), "focal" (Sigmoid Focal Loss), "qfl" (Quality Focal Loss)
    - Scale-aware Loss Weighting: assigns instance weight based on bounding box pixel area
      (Ultra-fine: <32^2, Fine: 32^2-96^2, Medium: 96^2-144^2, Large: >=144^2)
    """

    def __init__(
        self,
        nc: int = 1,
        stride: int = 4,
        loss_gain: dict[str, float] | None = None,
        target_assignment: str = "1x1",
        cls_loss_type: str = "bce",
        scale_weights: dict[str, float] | tuple[float, float, float, float] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.stride = stride
        self.loss_gain = loss_gain or {"cls": 1.0, "box": 2.0, "giou": 2.0}
        self.target_assignment = str(target_assignment).lower()
        self.cls_loss_type = str(cls_loss_type).lower()

        if scale_weights is None:
            self.scale_weights = {"ultra_fine": 1.0, "fine": 1.0, "medium": 1.0, "large": 1.0}
        elif isinstance(scale_weights, (list, tuple)):
            self.scale_weights = {
                "ultra_fine": float(scale_weights[0]),
                "fine": float(scale_weights[1]),
                "medium": float(scale_weights[2]),
                "large": float(scale_weights[3]),
            }
        elif isinstance(scale_weights, dict):
            self.scale_weights = {
                "ultra_fine": float(scale_weights.get("ultra_fine", 1.0)),
                "fine": float(scale_weights.get("fine", 1.0)),
                "medium": float(scale_weights.get("medium", 1.0)),
                "large": float(scale_weights.get("large", 1.0)),
            }
        else:
            self.scale_weights = {"ultra_fine": 1.0, "fine": 1.0, "medium": 1.0, "large": 1.0}

    def _get_scale_weight(self, area_px: float) -> float:
        if area_px < 32.0 * 32.0:
            return self.scale_weights["ultra_fine"]
        elif area_px < 96.0 * 96.0:
            return self.scale_weights["fine"]
        elif area_px < 144.0 * 144.0:
            return self.scale_weights["medium"]
        else:
            return self.scale_weights["large"]

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
        pos_weights = torch.ones((bs, h, w), device=device, dtype=torch.float32)

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
                area_px = bw * bh
                scale_w = self._get_scale_weight(area_px)

                gx1 = cx - bw / 2.0
                gy1 = cy - bh / 2.0
                gx2 = cx + bw / 2.0
                gy2 = cy + bh / 2.0

                cx_idx = min(max(0, int(cx // stride)), w - 1)
                cy_idx = min(max(0, int(cy // stride)), h - 1)
                c_idx = int(torch.clamp(b_gt_cls[idx], 0, self.nc - 1).item())

                if self.target_assignment == "3x3":
                    # Multi-positive assignment: 3x3 window around center
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny = cy_idx + dy
                            nx = cx_idx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                cell_cx = (nx + 0.5) * stride
                                cell_cy = (ny + 0.5) * stride
                                # Include center cell always, and neighbor cells if inside GT box
                                if (dx == 0 and dy == 0) or (gx1 <= cell_cx <= gx2 and gy1 <= cell_cy <= gy2):
                                    target_cls[b, c_idx, ny, nx] = 1.0
                                    pos_mask[b, ny, nx] = True
                                    target_boxes_ltrb[b, 0, ny, nx] = max(0.0, cell_cx - gx1)
                                    target_boxes_ltrb[b, 1, ny, nx] = max(0.0, cell_cy - gy1)
                                    target_boxes_ltrb[b, 2, ny, nx] = max(0.0, gx2 - cell_cx)
                                    target_boxes_ltrb[b, 3, ny, nx] = max(0.0, gy2 - cell_cy)
                                    pos_weights[b, ny, nx] = scale_w
                else:
                    # 1x1 center assignment (baseline)
                    target_cls[b, c_idx, cy_idx, cx_idx] = 1.0
                    pos_mask[b, cy_idx, cx_idx] = True
                    grid_cx = (cx_idx + 0.5) * stride
                    grid_cy = (cy_idx + 0.5) * stride
                    target_boxes_ltrb[b, 0, cy_idx, cx_idx] = max(0.0, grid_cx - gx1)
                    target_boxes_ltrb[b, 1, cy_idx, cx_idx] = max(0.0, grid_cy - gy1)
                    target_boxes_ltrb[b, 2, cy_idx, cx_idx] = max(0.0, gx2 - grid_cx)
                    target_boxes_ltrb[b, 3, cy_idx, cx_idx] = max(0.0, gy2 - grid_cy)
                    pos_weights[b, cy_idx, cx_idx] = scale_w

        num_pos = max(1.0, float(pos_mask.sum().item()))

        # Box L1 and GIoU Loss (vectorized directly over assigned positive cells)
        if pos_mask.any():
            pred_ltrb_pos = box_offsets.permute(0, 2, 3, 1)[pos_mask]
            target_ltrb_pos = target_boxes_ltrb.permute(0, 2, 3, 1)[pos_mask]
            sample_w = pos_weights[pos_mask]  # (N_pos,)
            w_sum = max(1.0, float(sample_w.sum().item()))

            # L1 box regression loss with scale weights
            l1_raw = F.l1_loss(pred_ltrb_pos, target_ltrb_pos, reduction="none").sum(dim=-1)  # (N_pos,)
            loss_box = (l1_raw * sample_w).sum() / w_sum

            grid_x_pos = grid_x.unsqueeze(0).expand(bs, -1, -1)[pos_mask]
            grid_y_pos = grid_y.unsqueeze(0).expand(bs, -1, -1)[pos_mask]

            t_xyxy = torch.stack([
                grid_x_pos - target_ltrb_pos[:, 0],
                grid_y_pos - target_ltrb_pos[:, 1],
                grid_x_pos + target_ltrb_pos[:, 2],
                grid_y_pos + target_ltrb_pos[:, 3],
            ], dim=-1)

            px1_raw = grid_x_pos - pred_ltrb_pos[:, 0]
            py1_raw = grid_y_pos - pred_ltrb_pos[:, 1]
            px2_raw = grid_x_pos + pred_ltrb_pos[:, 2]
            py2_raw = grid_y_pos + pred_ltrb_pos[:, 3]

            p_xyxy = torch.stack([
                torch.min(px1_raw, px2_raw),
                torch.min(py1_raw, py2_raw),
                torch.max(px1_raw, px2_raw),
                torch.max(py1_raw, py2_raw),
            ], dim=-1)

            giou = ops.generalized_box_iou(p_xyxy, t_xyxy)
            diag_giou = torch.diag(giou)
            loss_giou = ((1.0 - diag_giou) * sample_w).sum() / w_sum

            # Compute IoU for Quality Focal Loss if needed
            if self.cls_loss_type == "qfl":
                iou = ops.box_iou(p_xyxy, t_xyxy)
                diag_iou = torch.diag(iou).detach().clamp(min=0.0, max=1.0)
            else:
                diag_iou = None
        else:
            loss_box = torch.tensor(0.0, device=device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
            diag_iou = None

        # Classification Loss (BCE / Focal / QFL with scale weighting)
        cls_sample_weights = torch.ones_like(cls_logits)
        if pos_mask.any():
            for c in range(self.nc):
                c_mask = (target_cls[:, c] > 0)
                if c_mask.any():
                    cls_sample_weights[:, c][c_mask] = pos_weights[c_mask]

        if self.cls_loss_type == "focal":
            # Sigmoid Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            prob = cls_logits.sigmoid()
            p_t = prob * target_cls + (1.0 - prob) * (1.0 - target_cls)
            alpha_factor = 0.25 * target_cls + 0.75 * (1.0 - target_cls)
            modulating_factor = (1.0 - p_t) ** 2.0
            fl = -alpha_factor * modulating_factor * torch.log(p_t.clamp(min=1e-6, max=1.0))
            loss_cls = (fl * cls_sample_weights).sum() / num_pos

        elif self.cls_loss_type == "qfl":
            # Quality Focal Loss: QFL(sigma, y) = -|y - sigma|^beta * ((1 - y)*log(1 - sigma) + y*log(sigma))
            target_quality = torch.zeros_like(cls_logits)
            if pos_mask.any() and diag_iou is not None:
                for c in range(self.nc):
                    target_quality[:, c][pos_mask] = diag_iou

            prob = cls_logits.sigmoid()
            modulating_factor = torch.abs(target_quality - prob) ** 2.0
            bce_continuous = -(
                target_quality * torch.log(prob.clamp(min=1e-6, max=1.0))
                + (1.0 - target_quality) * torch.log((1.0 - prob).clamp(min=1e-6, max=1.0))
            )
            qfl = modulating_factor * bce_continuous
            loss_cls = (qfl * cls_sample_weights).sum() / num_pos

        else:
            # Baseline BCE with Logits
            bce_loss = F.binary_cross_entropy_with_logits(cls_logits, target_cls, reduction="none")
            loss_cls = (bce_loss * cls_sample_weights).sum() / num_pos

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
        target_assignment: str = "1x1",
        cls_loss_type: str = "bce",
        scale_weights: dict[str, float] | tuple[float, float, float, float] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.nc = num_classes
        self.stride = stride
        self.target_assignment = target_assignment
        self.cls_loss_type = cls_loss_type
        self.scale_weights = scale_weights

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

        self.criterion = DenseP2Loss(
            nc=num_classes,
            stride=stride,
            target_assignment=target_assignment,
            cls_loss_type=cls_loss_type,
            scale_weights=scale_weights,
        )
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
        target_assignment: str = "1x1",
        cls_loss_type: str = "bce",
        scale_weights: dict[str, float] | tuple[float, float, float, float] | list[float] | None = None,
        topk: int = 300,
        conf_threshold: float = 0.001,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.freeze_native = freeze_native
        self.topk = topk
        self.conf_threshold = conf_threshold

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
        self.p2_head = LightweightP2Head(
            in_channels=p2_channels,
            num_classes=nc,
            stride=4,
            target_assignment=target_assignment,
            cls_loss_type=cls_loss_type,
            scale_weights=scale_weights,
        )

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
            topk=self.topk,
            conf_threshold=self.conf_threshold,
        )

        if isinstance(native_out, (list, tuple)):
            native_formatted = native_out[0]
        else:
            native_formatted = native_out

        # Convert native predictions from normalized cxcywh to pixel xyxy in canvas coordinates [0, W] x [0, H]
        h_canvas, w_canvas = float(x.shape[-2]), float(x.shape[-1])
        ncx = native_formatted[..., 0]
        ncy = native_formatted[..., 1]
        nw = native_formatted[..., 2]
        nh = native_formatted[..., 3]
        x1 = torch.clamp((ncx - nw / 2.0) * w_canvas, min=0.0, max=w_canvas)
        y1 = torch.clamp((ncy - nh / 2.0) * h_canvas, min=0.0, max=h_canvas)
        x2 = torch.clamp((ncx + nw / 2.0) * w_canvas, min=0.0, max=w_canvas)
        y2 = torch.clamp((ncy + nh / 2.0) * h_canvas, min=0.0, max=h_canvas)
        native_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        native_converted = torch.cat([native_boxes, native_formatted[..., 4:]], dim=-1)

        return {
            "native_preds": native_converted,
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

