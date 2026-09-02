from __future__ import annotations

import math
from typing import Any
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.rtdetr.train import RTDETRDetectionModel

from .p2_branch import find_c2_backbone_stage, P2Branch, _unwrap_sequential


# ---------------------------------------------------------------------------
# Hungarian Matcher for P2 Auxiliary Head
# ---------------------------------------------------------------------------

class P2HungarianMatcher(nn.Module):
    """Bipartite Hungarian Matcher for P2 query-based detector head."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: torch.Tensor,   # (B, Nq, 4) in [cx, cy, w, h] normalized
        pred_scores: torch.Tensor,   # (B, Nq, nc) logits or probabilities
        targets: dict[str, Any],     # dict with cls, bboxes (normalized xywh), gt_groups
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute bipartite matching between predictions and targets.

        Returns:
            list of (src_idx, tgt_idx) for each batch element.
        """
        bs, num_queries = pred_bboxes.shape[:2]
        gt_cls = targets["cls"]
        gt_bboxes = targets["bboxes"]  # (total_gt, 4) in xywh
        gt_groups = targets.get("gt_groups", [])

        if len(gt_groups) == 0:
            return [(torch.empty(0, dtype=torch.long, device=pred_bboxes.device),
                     torch.empty(0, dtype=torch.long, device=pred_bboxes.device)) for _ in range(bs)]

        pred_prob = pred_scores.sigmoid()
        pred_xyxy = ops.box_convert(pred_bboxes, "cxcywh", "xyxy")

        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        gt_start = 0

        for b in range(bs):
            num_gt = gt_groups[b] if b < len(gt_groups) else 0
            if num_gt == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=pred_bboxes.device),
                    torch.empty(0, dtype=torch.long, device=pred_bboxes.device),
                ))
                continue

            b_gt_cls = gt_cls[gt_start:gt_start + num_gt]
            b_gt_bboxes = gt_bboxes[gt_start:gt_start + num_gt]
            b_gt_xyxy = ops.box_convert(b_gt_bboxes, "cxcywh", "xyxy")
            gt_start += num_gt

            # 1. Classification cost
            b_prob = pred_prob[b]  # (Nq, nc)
            if b_prob.shape[-1] == 1:
                cost_class = -b_prob[:, 0:1].repeat(1, num_gt)
            else:
                c_idx = b_gt_cls.clamp(0, b_prob.shape[-1] - 1).long()
                cost_class = -b_prob[:, c_idx]

            # 2. L1 box cost
            cost_bbox = torch.cdist(pred_bboxes[b], b_gt_bboxes, p=1)  # (Nq, num_gt)

            # 3. GIoU cost
            giou = ops.generalized_box_iou(pred_xyxy[b], b_gt_xyxy)    # (Nq, num_gt)
            cost_giou = -giou

            # Total cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.cpu().numpy()
            C = np.nan_to_num(C, nan=100.0, posinf=100.0, neginf=-100.0)

            src_ind, tgt_ind = scipy.optimize.linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(src_ind, dtype=torch.long, device=pred_bboxes.device),
                torch.as_tensor(tgt_ind, dtype=torch.long, device=pred_bboxes.device),
            ))

        return indices


# ---------------------------------------------------------------------------
# Loss for P2 Auxiliary Head
# ---------------------------------------------------------------------------

class P2HeadLoss(nn.Module):
    """Computes Hungarian loss for P2 query predictions."""

    def __init__(
        self,
        nc: int = 1,
        loss_gain: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.matcher = P2HungarianMatcher()
        gains = loss_gain or {"class": 2.0, "bbox": 5.0, "giou": 2.0}
        self.loss_gain = gains

    def forward(
        self,
        pred_bboxes: torch.Tensor,   # (B, Nq, 4) normalized cxcywh
        pred_scores: torch.Tensor,   # (B, Nq, nc) logits
        targets: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        bs, num_queries = pred_bboxes.shape[:2]
        indices = self.matcher(pred_bboxes, pred_scores, targets)

        gt_cls = targets["cls"]
        gt_bboxes = targets["bboxes"]
        gt_groups = targets.get("gt_groups", [])

        num_boxes = sum(len(src) for src, _ in indices)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_bboxes.device)
        num_boxes_t = torch.clamp(num_boxes_t, min=1.0).item()

        target_classes = torch.zeros_like(pred_scores)
        src_bboxes_list = []
        tgt_bboxes_list = []
        gt_start = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            num_gt = gt_groups[b] if b < len(gt_groups) else 0
            if len(src_idx) > 0 and num_gt > 0:
                b_gt_cls = gt_cls[gt_start:gt_start + num_gt]
                b_gt_bboxes = gt_bboxes[gt_start:gt_start + num_gt]

                for s, t in zip(src_idx, tgt_idx):
                    c = b_gt_cls[t].clamp(0, self.nc - 1).long()
                    target_classes[b, s, c] = 1.0

                src_bboxes_list.append(pred_bboxes[b, src_idx])
                tgt_bboxes_list.append(b_gt_bboxes[tgt_idx])

            gt_start += num_gt

        # 1. Classification Loss
        loss_class = F.binary_cross_entropy_with_logits(pred_scores, target_classes, reduction="sum") / (num_boxes_t * num_queries)

        # 2. Box L1 and GIoU loss
        if src_bboxes_list:
            src_b = torch.cat(src_bboxes_list, dim=0)
            tgt_b = torch.cat(tgt_bboxes_list, dim=0)

            loss_bbox = F.l1_loss(src_b, tgt_b, reduction="sum") / num_boxes_t

            src_xyxy = ops.box_convert(src_b, "cxcywh", "xyxy")
            tgt_xyxy = ops.box_convert(tgt_b, "cxcywh", "xyxy")
            giou = ops.generalized_box_iou(src_xyxy, tgt_xyxy)
            loss_giou = (1.0 - torch.diag(giou)).sum() / num_boxes_t
        else:
            loss_bbox = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        total_loss = (
            self.loss_gain["class"] * loss_class
            + self.loss_gain["bbox"] * loss_bbox
            + self.loss_gain["giou"] * loss_giou
        )

        return {
            "loss_p2_total": total_loss,
            "loss_p2_class": loss_class,
            "loss_p2_bbox": loss_bbox,
            "loss_p2_giou": loss_giou,
        }


# ---------------------------------------------------------------------------
# P2 Query-Based Head Module
# ---------------------------------------------------------------------------

class P2QueryHead(nn.Module):
    """Query-based auxiliary detector head operating on P2 feature map."""

    def __init__(
        self,
        in_channels: int = 256,
        num_queries: int = 300,
        nc: int = 1,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_queries = num_queries
        self.nc = nc

        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, in_channels)
        self.query_pos = nn.Embedding(num_queries, in_channels)

        # Spatial projection for high-resolution P2
        self.feat_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Multi-layer Transformer Decoder for P2
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Prediction Heads
        self.class_head = nn.Linear(in_channels, nc)
        self.bbox_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 4),
            nn.Sigmoid(),  # Outputs normalized [cx, cy, w, h] in (0, 1)
        )

        self.criterion = P2HeadLoss(nc=nc)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.query_pos.weight, std=0.02)
        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_val)

    def forward(
        self,
        p2_feat: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """Forward pass over P2 features."""
        bs, c, h, w = p2_feat.shape

        feat_proj = self.feat_proj(p2_feat).flatten(2).permute(0, 2, 1)

        query_content = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(bs, -1, -1)
        tgt = query_content + query_pos

        hs = self.transformer_decoder(tgt=tgt, memory=feat_proj)

        pred_scores = self.class_head(hs)
        pred_bboxes = self.bbox_head(hs)

        return pred_bboxes, pred_scores

    def compute_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        targets: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return self.criterion(pred_bboxes, pred_scores, targets)


# ---------------------------------------------------------------------------
# Unified RT-DETR-L + P2 Model Container
# ---------------------------------------------------------------------------

class RTDETRP2Model(nn.Module):
    """Augments native RT-DETR-L with isolated P2 Auxiliary Detector Branch.

    - Backbone -> C2 -> P2 Adapter -> P2 Query Head
    - Backbone -> P3, P4, P5 -> Native RT-DETR
    - Loss: L_total = L_native + lambda_p2 * L_p2 (lambda_p2 = 0.25)
    """

    def __init__(
        self,
        native_model: nn.Module,
        c2_layer_idx: int | None = None,
        c2_channels: int | None = None,
        p2_channels: int = 256,
        num_p2_queries: int = 300,
        nc: int = 1,
        lambda_p2: float = 0.25,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        self.nc = nc
        self.lambda_p2 = lambda_p2

        # Configure native model
        det_model, sub_modules, _ = _unwrap_sequential(native_model)
        if not isinstance(det_model, RTDETRDetectionModel):
            det_model.__class__ = RTDETRDetectionModel
        if not hasattr(det_model, "nc") or det_model.nc is None:
            det_model.nc = getattr(sub_modules[-1], "nc", nc)

        self.native_model = det_model

        # Discover C2 stage if not explicitly passed
        if c2_layer_idx is None or c2_channels is None:
            c2_layer_idx, c2_channels = find_c2_backbone_stage(self.native_model, input_size=input_size)

        self.c2_layer_idx = c2_layer_idx
        self.c2_channels = c2_channels

        # Attach isolated P2 Branch & Query Head
        self.p2_branch = P2Branch(c2_layer_idx=c2_layer_idx, in_channels=c2_channels, out_channels=p2_channels)
        self.p2_head = P2QueryHead(in_channels=p2_channels, num_queries=num_p2_queries, nc=nc)

        # Hook storage for C2 extraction
        self._c2_tensor: torch.Tensor | None = None
        self._register_c2_hook()

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

    @property
    def stride(self) -> torch.Tensor | None:
        return getattr(self.native_model, "stride", None)

    def _register_c2_hook(self) -> None:
        det_model, sub_modules, _ = _unwrap_sequential(self.native_model)
        if self.c2_layer_idx < len(sub_modules):
            target_layer = sub_modules[self.c2_layer_idx]
            def hook_fn(module: nn.Module, inp: Any, outp: torch.Tensor) -> None:
                self._c2_tensor = outp
            target_layer.register_forward_hook(hook_fn)

    def forward(
        self,
        x: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> Any:
        """Forward pass through native RT-DETR and P2 auxiliary branch."""
        self._c2_tensor = None
        native_out = self.native_model(x)

        if self._c2_tensor is None:
            raise RuntimeError(
                f"C2 feature map was not captured at layer index {self.c2_layer_idx} during forward pass."
            )

        c2_feat = self._c2_tensor
        p2_feat = self.p2_branch(c2_feat)
        p2_bboxes, p2_scores = self.p2_head(p2_feat)

        if self.training:
            return native_out, (p2_bboxes, p2_scores)

        # Format P2 predictions for inference: (B, Nq, 6) [x1, y1, x2, y2, score, cls]
        bs, nq = p2_bboxes.shape[:2]
        img_h, img_w = x.shape[-2], x.shape[-1]

        p2_xyxy_norm = ops.box_convert(p2_bboxes, "cxcywh", "xyxy")
        p2_xyxy_pixel = p2_xyxy_norm * torch.tensor([img_w, img_h, img_w, img_h], device=x.device)

        p2_prob = p2_scores.sigmoid()
        p2_conf, p2_cls = p2_prob.max(dim=-1, keepdim=True)
        p2_formatted = torch.cat([p2_xyxy_pixel, p2_conf, p2_cls.float()], dim=-1)

        if isinstance(native_out, (list, tuple)):
            native_formatted = native_out[0]
        else:
            native_formatted = native_out

        return {
            "native_preds": native_formatted,
            "p2_preds": p2_formatted,
            "p2_raw": (p2_bboxes, p2_scores),
            "native_raw": native_out,
        }

    def loss(
        self,
        batch: dict[str, Any],
        preds: Any = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss: L_total = L_native + 0.25 * L_P2."""
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

        # 1. Native RT-DETR Loss
        det_model, _, _ = _unwrap_sequential(self.native_model)
        native_loss_val, native_loss_dict = det_model.loss(batch)

        # 2. P2 Auxiliary Loss
        if self._c2_tensor is None:
            _ = self.forward(img)

        c2_feat = self._c2_tensor
        p2_feat = self.p2_branch(c2_feat)
        p2_bboxes, p2_scores = self.p2_head(p2_feat)
        p2_loss_dict = self.p2_head.compute_loss(p2_bboxes, p2_scores, targets)
        p2_loss_val = p2_loss_dict["loss_p2_total"]

        total_loss = native_loss_val + self.lambda_p2 * p2_loss_val

        combined_loss_dict = {
            **native_loss_dict,
            "p2_loss": p2_loss_val.detach(),
            "p2_class": p2_loss_dict["loss_p2_class"].detach(),
            "p2_bbox": p2_loss_dict["loss_p2_bbox"].detach(),
            "p2_giou": p2_loss_dict["loss_p2_giou"].detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss, combined_loss_dict
