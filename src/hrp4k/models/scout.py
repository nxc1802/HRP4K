"""AdaPoth-Lite Region Scout Model and Candidate Generation.

Implements:
1. MobileNetV3-Small Scout architecture (~1-1.5M parameters) producing stride-16 heatmap.
2. Elliptical Gaussian Ground-Truth heatmap generator for pothole annotations.
3. ScoutLoss combining modified Focal Loss with Coverage Loss (lambda_cov = 2.0).
4. CandidateGenerator: Threshold -> Connected Components -> Context Expansion -> Region NMS -> Dynamic Top-K (K <= 4).
5. Scout Region Recall & Coverage evaluation metric utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


def _build_conv_bn_act(in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True),
    )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (3x3 depthwise + 1x1 pointwise)."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = _build_conv_bn_act(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c)
        self.pw = _build_conv_bn_act(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block with optional SE."""
    def __init__(self, in_c: int, exp_c: int, out_c: int, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.stride = stride
        self.use_res = (stride == 1 and in_c == out_c)
        layers = []
        if exp_c != in_c:
            layers.append(_build_conv_bn_act(in_c, exp_c, kernel_size=1, stride=1, padding=0))
        layers.append(_build_conv_bn_act(exp_c, exp_c, kernel_size=3, stride=stride, padding=1, groups=exp_c))
        if use_se:
            layers.append(SEBlock(exp_c))
        layers.append(nn.Sequential(
            nn.Conv2d(exp_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        ))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return (x + out) if self.use_res else out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


if TORCH_AVAILABLE:
    class MobileNetV3Scout(nn.Module):
        """MobileNetV3-Small based Region Scout Network (< 1.5M parameters).
        
        Processes low-resolution thumbnail (e.g. 960x540) and produces a single-channel
        heatmap (e.g. 60x34, stride 16) representing pothole presence likelihood.
        """
        def __init__(self, in_channels: int = 3, feature_dim: int = 576):
            super().__init__()

            # MobileNetV3-Small Backbone (Stride 16 output)
            self.stem = _build_conv_bn_act(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 1/2

            self.stages = nn.Sequential(
                InvertedResidual(16, 16, 16, stride=2, use_se=True),   # 1/4
                InvertedResidual(16, 72, 24, stride=2, use_se=False),  # 1/8
                InvertedResidual(24, 88, 24, stride=1, use_se=False),
                InvertedResidual(24, 96, 40, stride=2, use_se=True),   # 1/16
                InvertedResidual(40, 240, 40, stride=1, use_se=True),
                InvertedResidual(40, 240, 40, stride=1, use_se=True),
                InvertedResidual(40, 120, 48, stride=1, use_se=True),
                InvertedResidual(48, 144, 48, stride=1, use_se=True),
                InvertedResidual(48, 288, 96, stride=1, use_se=True),
            )
            
            self.conv_head = _build_conv_bn_act(96, feature_dim, kernel_size=1, stride=1, padding=0)

            # Lightweight Depthwise Separable Heatmap Prediction Head (~1.5M total model params)
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, groups=feature_dim, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(feature_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.
            
            Args:
                x: Tensor of shape (B, 3, H, W) normalized to [0, 1] or ImageNet stats.
            Returns:
                heatmap: Tensor of shape (B, 1, H//16, W//16) with values in [0, 1].
            """
            feat = self.stem(x)
            feat = self.stages(feat)
            feat = self.conv_head(feat)
            heatmap = self.heatmap_head(feat)
            return heatmap

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
else:
    class MobileNetV3Scout:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate MobileNetV3Scout")


def generate_scout_heatmap_gt(
    boxes_xywh: list[list[float]] | np.ndarray,
    img_w: int = 3840,
    img_h: int = 2160,
    heat_w: int = 60,
    heat_h: int = 34,
    expand_ratio: float = 0.25,
) -> np.ndarray:
    """Generate Coverage-Aware Soft Region Target heatmap for Region Scout training.
    
    Creates a dense rectangular region target (value 1.0 inside object box)
    with smooth falloff margin (25% expansion), providing direct supervision
    for region coverage instead of sharp center point.
    """
    heatmap = np.zeros((heat_h, heat_w), dtype=np.float32)
    if len(boxes_xywh) == 0:
        return heatmap

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    scale_x = float(heat_w) / float(img_w)
    scale_y = float(heat_h) / float(img_h)

    grid_y, grid_x = np.ogrid[:heat_h, :heat_w]

    for box in boxes:
        x, y, w, h = box[:4]
        if w <= 0 or h <= 0:
            continue

        u0 = x * scale_x
        v0 = y * scale_y
        u1 = (x + w) * scale_x
        v1 = (y + h) * scale_y

        bw = max(1.0, u1 - u0)
        bh = max(1.0, v1 - v0)

        sigma_x = max(1.0, bw * expand_ratio)
        sigma_y = max(1.0, bh * expand_ratio)

        # Distance from grid points to bounding box rectangle
        dx = np.maximum(0.0, np.maximum(u0 - grid_x, grid_x - u1))
        dy = np.maximum(0.0, np.maximum(v0 - grid_y, grid_y - v1))

        dist_sq = (dx / sigma_x) ** 2 + (dy / sigma_y) ** 2
        soft_region = np.exp(-0.5 * dist_sq).astype(np.float32)

        np.maximum(heatmap, soft_region, out=heatmap)

    return np.clip(heatmap, 0.0, 1.0)


if TORCH_AVAILABLE:
    class ScoutLoss(nn.Module):
        """Scout Loss combining Dense Region Focal Loss with True GT Region Coverage Loss."""
        def __init__(
            self,
            alpha: float = 2.0,
            beta: float = 4.0,
            pos_weight: float = 3.0,
            lambda_cov: float = 3.0,
            target_cov: float = 0.85,
            eps: float = 1e-4,
        ):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.pos_weight = pos_weight
            self.lambda_cov = lambda_cov
            self.target_cov = target_cov
            self.eps = eps

        def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            gt_boxes: list[list[list[float]]] | None = None,
            orig_sizes: list[tuple[int, int]] | None = None,
            gt_centers: list[list[tuple[float, float]]] | None = None,
        ) -> dict[str, torch.Tensor]:
            pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
            
            pos_mask = (target >= 0.70)
            neg_mask = (target < 0.70)

            # Balanced Dense Region Focal Loss
            pos_loss = -((1.0 - pred) ** self.alpha) * torch.log(pred) * pos_mask.float()
            neg_loss = -((1.0 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1.0 - pred) * neg_mask.float()

            num_pos = pos_mask.sum()
            if num_pos > 0:
                focal_loss = (self.pos_weight * pos_loss.sum() + neg_loss.sum()) / num_pos
            else:
                focal_loss = neg_loss.mean() * 100.0

            # True GT Region Coverage Loss: penalizes low average activation across the entire GT box
            coverage_loss = torch.tensor(0.0, device=pred.device)
            if gt_boxes is not None and orig_sizes is not None:
                cov_losses = []
                b, _, heat_h, heat_w = pred.shape
                import math
                for i, boxes in enumerate(gt_boxes):
                    if not boxes:
                        continue
                    orig_w, orig_h = orig_sizes[i]
                    scale_x = float(heat_w) / float(orig_w)
                    scale_y = float(heat_h) / float(orig_h)
                    for box in boxes:
                        x, y, w, h = box[:4]
                        if w <= 0 or h <= 0:
                            continue
                        u0 = int(np.clip(math.floor(x * scale_x), 0, heat_w - 1))
                        u1 = int(np.clip(math.ceil((x + w) * scale_x) + 1, u0 + 1, heat_w))
                        v0 = int(np.clip(math.floor(y * scale_y), 0, heat_h - 1))
                        v1 = int(np.clip(math.ceil((y + h) * scale_y) + 1, v0 + 1, heat_h))

                        roi_act = pred[i, 0, v0:v1, u0:u1]
                        if roi_act.numel() > 0:
                            mean_act = roi_act.mean()
                            cov_losses.append(F.relu(self.target_cov - mean_act))
                if cov_losses:
                    coverage_loss = torch.stack(cov_losses).mean()
            elif gt_centers is not None:
                cov_losses = []
                b, _, h, w = pred.shape
                for i, centers in enumerate(gt_centers):
                    if not centers:
                        continue
                    for cy, cx in centers:
                        iy = int(np.clip(round(cy), 0, h - 1))
                        ix = int(np.clip(round(cx), 0, w - 1))
                        p_val = pred[i, 0, iy, ix]
                        cov_losses.append(F.relu(self.target_cov - p_val))
                if cov_losses:
                    coverage_loss = torch.stack(cov_losses).mean()
            else:
                if pos_mask.any():
                    coverage_loss = F.relu(self.target_cov - pred[pos_mask].mean())

            total_loss = focal_loss + self.lambda_cov * coverage_loss
            return {
                "loss": total_loss,
                "focal_loss": focal_loss,
                "coverage_loss": coverage_loss,
            }
else:
    class ScoutLoss:
        def __init__(self, *args, **kwargs):
            pass


@dataclass
class CandidateRegion:
    """A generated candidate crop region mapped to 4K coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    component_id: int
    area: int = 0

    @property
    def width(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def xyxy(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def xywh(self) -> list[int]:
        return [self.x0, self.y0, self.width, self.height]


class CandidateGenerator:
    """Adaptive Dynamic Cluster Candidate Region Generator from Scout Heatmap (Module B).
    
    Pipeline:
      1. Adaptive Thresholding: tau_eff = max(tau_base, 0.15 * max(H))
      2. Connected Components Labeling (8-connectivity)
      3. Region Scoring: S_r = alpha * max(H_r) + (1 - alpha) * mean(H_r)
      4. Dynamic Cluster Box Scaling with Context Margin (default 30%)
      5. ROI Merging: Merge highly overlapping or adjacent candidate boxes
      6. Region NMS (IoU 0.35)
      7. Dynamic Top-K (K <= k_max, returns [] for empty images - NO distorted safety fallback)
    """
    def __init__(
        self,
        threshold: float = 0.05,
        min_crop_size: int = 320,
        max_crop_size: int = 640,
        alpha_score: float = 0.70,
        context_margin: float = 0.30,
        region_nms_iou: float = 0.35,
        merge_iou_thresh: float = 0.20,
        k_max: int = 4,
        min_region_size: int = 2,
    ):
        self.threshold = threshold
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.alpha_score = alpha_score
        self.context_margin = context_margin
        self.region_nms_iou = region_nms_iou
        self.merge_iou_thresh = merge_iou_thresh
        self.k_max = k_max
        self.min_region_size = min_region_size

    def generate(
        self,
        heatmap: np.ndarray,
        source_width: int = 3840,
        source_height: int = 2160,
    ) -> list[CandidateRegion]:
        """Generate candidate regions from a 2D float heatmap [heat_h, heat_w]."""
        if heatmap.ndim == 3:
            heatmap = heatmap[0]  # Remove channel dim if present
        
        heat_h, heat_w = heatmap.shape[:2]
        scale_x = float(source_width) / float(heat_w)
        scale_y = float(source_height) / float(heat_h)

        max_val = float(np.max(heatmap)) if heatmap.size > 0 else 0.0
        # If no significant activation above base threshold, cleanly return empty (K=0)
        if max_val < self.threshold:
            return []

        # 1. Adaptive Thresholding
        effective_thresh = max(self.threshold, 0.15 * max_val)
        binary_mask = (heatmap >= effective_thresh).astype(np.uint8)

        # 2. Connected Components Labeling
        try:
            import cv2
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            components = []
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < self.min_region_size:
                    continue
                u0 = stats[label, cv2.CC_STAT_LEFT]
                v0 = stats[label, cv2.CC_STAT_TOP]
                uw = stats[label, cv2.CC_STAT_WIDTH]
                vh = stats[label, cv2.CC_STAT_HEIGHT]
                u1 = u0 + uw
                v1 = v0 + vh

                comp_mask = (labels == label)
                comp_vals = heatmap[comp_mask]
                c_max = float(comp_vals.max()) if len(comp_vals) > 0 else 0.0
                c_mean = float(comp_vals.mean()) if len(comp_vals) > 0 else 0.0
                score = self.alpha_score * c_max + (1.0 - self.alpha_score) * c_mean

                components.append({
                    "u0": u0, "v0": v0, "u1": u1, "v1": v1,
                    "score": score, "label": label, "area": area,
                })
        except Exception:
            components = self._fallback_connected_components(binary_mask, heatmap)

        # If no components found, cleanly return empty list (K=0) without safety distortion
        if not components:
            return []

        # 3. Dynamic Cluster Box Scaling with Context Margin
        raw_candidates: list[CandidateRegion] = []
        for comp in components:
            u_center = (comp["u0"] + comp["u1"]) * 0.5
            v_center = (comp["v0"] + comp["v1"]) * 0.5
            cx = u_center * scale_x
            cy = v_center * scale_y

            w_raw = (comp["u1"] - comp["u0"]) * scale_x
            h_raw = (comp["v1"] - comp["v0"]) * scale_y

            exp_w = w_raw * (1.0 + 2.0 * self.context_margin)
            exp_h = h_raw * (1.0 + 2.0 * self.context_margin)

            box_side = max(float(self.min_crop_size), max(exp_w, exp_h))
            box_w = min(float(self.max_crop_size), box_side)
            box_h = box_w

            x0 = int(np.clip(cx - box_w * 0.5, 0, max(0, source_width - box_w)))
            y0 = int(np.clip(cy - box_h * 0.5, 0, max(0, source_height - box_h)))
            x1 = int(min(source_width, x0 + box_w))
            y1 = int(min(source_height, y0 + box_h))

            raw_candidates.append(CandidateRegion(
                x0=x0, y0=y0, x1=x1, y1=y1,
                score=comp["score"], component_id=comp["label"],
                area=(x1 - x0) * (y1 - y0),
            ))

        # 4. ROI Merging
        merged_candidates = self._merge_clusters(raw_candidates, self.merge_iou_thresh, max_size=self.max_crop_size)

        # 5. Region NMS (IoU threshold 0.35)
        kept_candidates = self._region_nms(merged_candidates, self.region_nms_iou)

        # 6. Dynamic Top-K (K <= k_max)
        return kept_candidates[:self.k_max]

    def _merge_clusters(self, regions: list[CandidateRegion], iou_threshold: float, max_size: int = 640) -> list[CandidateRegion]:
        if len(regions) <= 1:
            return regions
        sorted_regions = sorted(regions, key=lambda r: r.score, reverse=True)
        merged: list[CandidateRegion] = []
        used = [False] * len(sorted_regions)

        for i in range(len(sorted_regions)):
            if used[i]:
                continue
            r1 = sorted_regions[i]
            x0, y0, x1, y1 = r1.x0, r1.y0, r1.x1, r1.y1
            score = r1.score
            cid = r1.component_id
            used[i] = True

            for j in range(i + 1, len(sorted_regions)):
                if used[j]:
                    continue
                r2 = sorted_regions[j]
                # Check bounding box union size
                nx0 = min(x0, r2.x0)
                ny0 = min(y0, r2.y0)
                nx1 = max(x1, r2.x1)
                ny1 = max(y1, r2.y1)
                nw = nx1 - nx0
                nh = ny1 - ny0

                # Compute IoU between r1 and r2
                ix1 = max(x0, r2.x0)
                iy1 = max(y0, r2.y0)
                ix2 = min(x1, r2.x1)
                iy2 = min(y1, r2.y1)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (x1 - x0) * (y1 - y0) + (r2.x1 - r2.x0) * (r2.y1 - r2.y0) - inter
                iou = inter / union if union > 0 else 0.0

                if iou >= iou_threshold and nw <= max_size and nh <= max_size:
                    x0, y0, x1, y1 = nx0, ny0, nx1, ny1
                    score = max(score, r2.score)
                    used[j] = True

            merged.append(CandidateRegion(
                x0=x0, y0=y0, x1=x1, y1=y1,
                score=score, component_id=cid, area=(x1 - x0) * (y1 - y0),
            ))
        return merged

    def _region_nms(self, regions: list[CandidateRegion], iou_threshold: float) -> list[CandidateRegion]:
        if not regions:
            return []
        sorted_regions = sorted(regions, key=lambda r: r.score, reverse=True)
        keep = []
        while sorted_regions:
            current = sorted_regions.pop(0)
            keep.append(current)
            remaining = []
            for r in sorted_regions:
                ix1 = max(current.x0, r.x0)
                iy1 = max(current.y0, r.y0)
                ix2 = min(current.x1, r.x1)
                iy2 = min(current.y1, r.y1)
                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter = inter_w * inter_h
                union = current.area + r.area - inter
                iou = inter / union if union > 0 else 0.0
                if iou <= iou_threshold:
                    remaining.append(r)
            sorted_regions = remaining
        return keep

    def _fallback_connected_components(self, binary_mask: np.ndarray, heatmap: np.ndarray) -> list[dict[str, Any]]:
        """Pure numpy fallback for connected components when OpenCV is not installed."""
        h, w = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        components = []
        label = 1

        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] and not visited[y, x]:
                    queue = [(y, x)]
                    visited[y, x] = True
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    vals = []

                    while queue:
                        cy, cx = queue.pop(0)
                        vals.append(heatmap[cy, cx])
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)

                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                    if len(vals) >= self.min_region_size:
                        c_max = float(np.max(vals))
                        c_mean = float(np.mean(vals))
                        score = self.alpha_score * c_max + (1.0 - self.alpha_score) * c_mean
                        components.append({
                            "u0": min_x, "v0": min_y, "u1": max_x + 1, "v1": max_y + 1,
                            "score": score, "label": label, "area": len(vals),
                        })
                        label += 1
        return components


def evaluate_scout_regions(
    gt_boxes_4k: list[list[float]] | np.ndarray,
    candidate_regions_4k: list[CandidateRegion | list[int]],
    coverage_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate Scout Candidate Generation Quality (Module B / Exp 4).
    
    Metrics:
      - Region Recall: Fraction of GT pothole boxes covered by at least one candidate region (Target >= 97%)
      - GT Coverage Ratio: Average fraction of GT box area overlapping candidate regions
      - False Region Rate: Fraction of candidate regions that contain NO GT pothole
      - Average K: Number of candidate crops generated
    """
    if len(gt_boxes_4k) == 0:
        return {
            "total_gts": 0,
            "covered_gts": 0,
            "region_recall": 1.0,
            "gt_coverage_ratio": 1.0,
            "false_region_rate": 0.0 if len(candidate_regions_4k) == 0 else 1.0,
            "k_crops": len(candidate_regions_4k),
        }

    # Normalize candidate region coordinates
    c_boxes = []
    for r in candidate_regions_4k:
        if isinstance(r, CandidateRegion):
            c_boxes.append([r.x0, r.y0, r.x1, r.y1])
        else:
            c_boxes.append(list(map(int, r[:4])))

    gt_covered_count = 0
    gt_coverages = []

    for gt in gt_boxes_4k:
        gx, gy, gw, gh = gt[:4]
        gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
        gt_area = max(1e-6, gw * gh)
        gt_cx, gt_cy = gx + gw * 0.5, gy + gh * 0.5

        max_overlap_ratio = 0.0
        covered = False

        for cx1, cy1, cx2, cy2 in c_boxes:
            # Check center inclusion
            if cx1 <= gt_cx <= cx2 and cy1 <= gt_cy <= cy2:
                covered = True

            # Compute intersection area
            ix1 = max(gx1, cx1)
            iy1 = max(gy1, cy1)
            ix2 = min(gx2, cx2)
            iy2 = min(gy2, cy2)
            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            overlap_ratio = (inter_w * inter_h) / gt_area
            max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

            if overlap_ratio >= coverage_threshold:
                covered = True

        if covered:
            gt_covered_count += 1
        gt_coverages.append(max_overlap_ratio)

    # False Region Rate: candidates containing no GT boxes (overlap < 10% of GT)
    false_regions = 0
    for cx1, cy1, cx2, cy2 in c_boxes:
        has_gt = False
        for gt in gt_boxes_4k:
            gx, gy, gw, gh = gt[:4]
            gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
            gt_area = max(1e-6, gw * gh)
            ix1 = max(gx1, cx1)
            iy1 = max(gy1, cy1)
            ix2 = min(gx2, cx2)
            iy2 = min(gy2, cy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if (inter / gt_area) >= 0.10:
                    has_gt = True
                    break
        if not has_gt:
            false_regions += 1

    total_gts = len(gt_boxes_4k)
    return {
        "total_gts": total_gts,
        "covered_gts": gt_covered_count,
        "region_recall": float(gt_covered_count / total_gts),
        "gt_coverage_ratio": float(np.mean(gt_coverages)) if gt_coverages else 0.0,
        "false_region_rate": float(false_regions / len(c_boxes)) if c_boxes else 0.0,
        "k_crops": len(c_boxes),
    }
