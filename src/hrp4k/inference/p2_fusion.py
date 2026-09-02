from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torchvision.ops as ops

from ..detectors.base import Detection


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of xyxy boxes."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def fuse_native_and_p2_predictions(
    native_detections: list[Detection],
    p2_detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Pure Concatenation + NMS Prediction Fusion.

    Fuses predictions from Native RT-DETR and P2 Auxiliary Head by:
    1. Concatenating all detection candidates
    2. Applying standard Non-Maximum Suppression (NMS)

    Explicitly excludes score boosting, WBF, learned fusion, or scale-aware gating.

    Args:
        native_detections: List of Detection objects from native RT-DETR
        p2_detections: List of Detection objects from P2 auxiliary branch
        iou_threshold: IoU threshold for NMS suppression (default: 0.5)

    Returns:
        list[Detection]: Fused detections after NMS
    """
    if not native_detections and not p2_detections:
        return []
    if not native_detections:
        return sorted(p2_detections, key=lambda d: d.score, reverse=True)
    if not p2_detections:
        return sorted(native_detections, key=lambda d: d.score, reverse=True)

    combined: list[Detection] = list(native_detections) + list(p2_detections)

    # Separate by category for class-aware NMS
    by_category: dict[int, list[Detection]] = {}
    for det in combined:
        by_category.setdefault(det.category_id, []).append(det)

    fused_results: list[Detection] = []
    for cat_id, cat_dets in by_category.items():
        if len(cat_dets) <= 1:
            fused_results.extend(cat_dets)
            continue

        boxes = np.array([d.xyxy for d in cat_dets], dtype=float)
        scores = np.array([d.score for d in cat_dets], dtype=float)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = _compute_iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

        for idx in keep:
            fused_results.append(cat_dets[idx])

    # Sort final detections descending by score
    fused_results.sort(key=lambda d: d.score, reverse=True)
    return fused_results


def fuse_prediction_tensors(
    native_tensor: torch.Tensor,
    p2_tensor: torch.Tensor,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.001,
) -> list[torch.Tensor]:
    """Batched tensor fusion via concatenation + NMS.

    Args:
        native_tensor: (B, N1, 6) tensor [x1, y1, x2, y2, score, cls]
        p2_tensor: (B, N2, 6) tensor [x1, y1, x2, y2, score, cls]
        iou_threshold: IoU threshold for NMS
        conf_threshold: Pre-filter confidence threshold

    Returns:
        list[torch.Tensor]: Per-batch tensor of kept detections of shape (K, 6)
    """
    bs = native_tensor.shape[0]
    combined = torch.cat([native_tensor, p2_tensor], dim=1)  # (B, N1+N2, 6)
    batch_fused: list[torch.Tensor] = []

    for b in range(bs):
        dets = combined[b]
        # Filter by confidence
        mask = dets[:, 4] >= conf_threshold
        dets = dets[mask]

        if dets.shape[0] == 0:
            batch_fused.append(torch.empty((0, 6), device=dets.device, dtype=dets.dtype))
            continue

        boxes = dets[:, :4]
        scores = dets[:, 4]
        classes = dets[:, 5]

        # Batched NMS across classes
        keep = ops.batched_nms(boxes, scores, classes.long(), iou_threshold)
        batch_fused.append(dets[keep])

    return batch_fused
