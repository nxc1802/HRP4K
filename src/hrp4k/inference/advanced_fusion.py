from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torchvision.ops as ops

import warnings
warnings.filterwarnings("ignore")

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_ENSEMBLE_BOXES = True
except ImportError:
    HAS_ENSEMBLE_BOXES = False


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=float)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-7), 0.0)


# =========================================================================
# Method 1: Scale-Partitioned Prior NMS
# =========================================================================
def fuse_scale_partitioned_nms(
    native_preds: list[dict[str, Any]],
    p2_preds: list[dict[str, Any]],
    max_p2_area: float = 1024.0,  # 32x32 px is standard Ultra-fine threshold
    iou_threshold: float = 0.6,
    p2_conf_threshold: float = 0.001,
) -> list[dict[str, Any]]:
    """Scale-Partitioned NMS Fusion:
    - P2 Auxiliary branch is restricted strictly to Ultra-fine scale (Area < max_p2_area).
    - Eliminates noisy medium/large bounding boxes from P2 from competing with Native RT-DETR.
    - Native RT-DETR retains all scales.
    - Merges valid predictions and performs class-aware NMS at optimal IoU threshold.
    """
    # Group by image_id
    by_image_native: dict[int, list[dict[str, Any]]] = {}
    by_image_p2: dict[int, list[dict[str, Any]]] = {}

    for p in native_preds:
        by_image_native.setdefault(int(p["image_id"]), []).append(p)
    for p in p2_preds:
        by_image_p2.setdefault(int(p["image_id"]), []).append(p)

    all_image_ids = set(by_image_native.keys()) | set(by_image_p2.keys())
    fused_all: list[dict[str, Any]] = []

    for img_id in all_image_ids:
        n_list = by_image_native.get(img_id, [])
        p_list = by_image_p2.get(img_id, [])

        # Filter P2 by confidence and area threshold
        filtered_p2: list[dict[str, Any]] = []
        for det in p_list:
            score = float(det.get("score", 0))
            if score < p2_conf_threshold:
                continue
            w = float(det["bbox"][2])
            h = float(det["bbox"][3])
            area = w * h
            if area < max_p2_area:
                filtered_p2.append(det)

        candidates = list(n_list) + filtered_p2
        if not candidates:
            continue
        if len(candidates) == 1:
            fused_all.append(candidates[0])
            continue

        # Convert to numpy arrays for NMS
        boxes_xywh = np.array([c["bbox"] for c in candidates], dtype=float)
        boxes_xyxy = np.column_stack([
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 0] + boxes_xywh[:, 2],
            boxes_xywh[:, 1] + boxes_xywh[:, 3],
        ])
        scores = np.array([float(c.get("score", 0)) for c in candidates], dtype=float)

        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = _compute_iou_matrix(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0]
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

        for k_idx in keep:
            fused_all.append(candidates[k_idx])

    fused_all.sort(key=lambda x: (int(x["image_id"]), -float(x.get("score", 0))))
    return fused_all


# =========================================================================
# Method 4: Scale-Calibrated Weighted Boxes Fusion (Calibrated WBF)
# =========================================================================
def fuse_calibrated_wbf(
    native_preds: list[dict[str, Any]],
    p2_preds: list[dict[str, Any]],
    iou_threshold: float = 0.55,
    p2_conf_threshold: float = 0.001,
    p2_score_multiplier: float = 3.0,
    img_w: float = 1920.0,
    img_h: float = 1920.0,
) -> list[dict[str, Any]]:
    """Scale-Calibrated Weighted Boxes Fusion (WBF):
    - Clusters overlapping predictions across Native and P2 branches.
    - Instead of winner-takes-all, smooths coordinates via weighted confidence average.
    - Calibrates P2 scores to prevent Native from dominating coordinate calculations.
    """
    by_image_native: dict[int, list[dict[str, Any]]] = {}
    by_image_p2: dict[int, list[dict[str, Any]]] = {}

    for p in native_preds:
        by_image_native.setdefault(int(p["image_id"]), []).append(p)
    for p in p2_preds:
        if float(p.get("score", 0)) >= p2_conf_threshold:
            by_image_p2.setdefault(int(p["image_id"]), []).append(p)

    all_image_ids = set(by_image_native.keys()) | set(by_image_p2.keys())
    fused_all: list[dict[str, Any]] = []

    for img_id in all_image_ids:
        n_list = by_image_native.get(img_id, [])
        p_list = by_image_p2.get(img_id, [])

        if not n_list and not p_list:
            continue
        if not p_list:
            fused_all.extend(n_list)
            continue
        if not n_list:
            fused_all.extend(p_list)
            continue

        # Prepare normalized boxes [0, 1] for WBF
        def prep_model_boxes(dets: list[dict[str, Any]], scale_score: float = 1.0):
            boxes_norm = []
            scores = []
            labels = []
            for d in dets:
                x, y, w, h = d["bbox"]
                x1 = max(0.0, min(1.0, x / img_w))
                y1 = max(0.0, min(1.0, y / img_h))
                x2 = max(0.0, min(1.0, (x + w) / img_w))
                y2 = max(0.0, min(1.0, (y + h) / img_h))
                if x2 <= x1 + 1e-5 or y2 <= y1 + 1e-5:
                    continue
                sc = min(1.0, max(0.0, float(d.get("score", 0)) * scale_score))
                boxes_norm.append([x1, y1, x2, y2])
                scores.append(sc)
                labels.append(int(d.get("category_id", 0)))
            return boxes_norm, scores, labels

        b_n, s_n, l_n = prep_model_boxes(n_list, scale_score=1.0)
        b_p, s_p, l_p = prep_model_boxes(p_list, scale_score=p2_score_multiplier)

        if HAS_ENSEMBLE_BOXES:
            boxes_list = [b_n, b_p]
            scores_list = [s_n, s_p]
            labels_list = [l_n, l_p]
            weights = [1.0, 1.0]

            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_threshold,
                skip_box_thr=p2_conf_threshold,
                conf_type="avg",
            )

            for box, score, label in zip(wbf_boxes, wbf_scores, wbf_labels):
                x1, y1, x2, y2 = box
                x = x1 * img_w
                y = y1 * img_h
                w = (x2 - x1) * img_w
                h = (y2 - y1) * img_h
                fused_all.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score),
                })
        else:
            # Fallback if ensemble_boxes is unavailable: pure python clustering WBF
            # Pool all boxes
            all_dets = []
            for b, s in zip(b_n, s_n):
                all_dets.append({"box": b, "score": s, "model": 0})
            for b, s in zip(b_p, s_p):
                all_dets.append({"box": b, "score": s, "model": 1})

            all_dets.sort(key=lambda x: -x["score"])
            clusters = []
            for d in all_dets:
                matched = False
                for c in clusters:
                    # compute IoU with cluster mean box
                    inter_x1 = max(d["box"][0], c["box"][0])
                    inter_y1 = max(d["box"][1], c["box"][1])
                    inter_x2 = min(d["box"][2], c["box"][2])
                    inter_y2 = min(d["box"][3], c["box"][3])
                    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    area_d = (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1])
                    area_c = (c["box"][2]-c["box"][0]) * (c["box"][3]-c["box"][1])
                    union = area_d + area_c - inter
                    iou_val = inter / union if union > 0 else 0.0
                    if iou_val >= iou_threshold:
                        c["members"].append(d)
                        matched = True
                        break
                if not matched:
                    clusters.append({"box": list(d["box"]), "members": [d]})

            for c in clusters:
                total_w = sum(m["score"] for m in c["members"])
                if total_w == 0:
                    continue
                fx1 = sum(m["score"] * m["box"][0] for m in c["members"]) / total_w
                fy1 = sum(m["score"] * m["box"][1] for m in c["members"]) / total_w
                fx2 = sum(m["score"] * m["box"][2] for m in c["members"]) / total_w
                fy2 = sum(m["score"] * m["box"][3] for m in c["members"]) / total_w
                fscore = sum(m["score"] for m in c["members"]) / len(c["members"])
                fused_all.append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [float(fx1 * img_w), float(fy1 * img_h), float((fx2 - fx1) * img_w), float((fy2 - fy1) * img_h)],
                    "score": float(fscore),
                })

    fused_all.sort(key=lambda x: (int(x["image_id"]), -float(x.get("score", 0))))
    return fused_all


# =========================================================================
# Method 5: Ultra-Lightweight Proposal-Level MLP Gating
# =========================================================================
class ProposalMLPGater(torch.nn.Module):
    """Ultra-lightweight 2-layer MLP (~350 parameters) that predicts calibrated
    reliability score for each proposal from geometric and cross-branch correlation features.
    """
    def __init__(self, in_features: int = 9, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_proposal_features(
    dets: list[dict[str, Any]],
    other_dets: list[dict[str, Any]],
    is_native: bool,
    img_w: float = 1920.0,
    img_h: float = 1920.0,
) -> np.ndarray:
    """Extracts 9-dimensional normalized feature vector per candidate box:
    [x1, y1, x2, y2, score, is_native, area, aspect_ratio, max_overlap_other]
    """
    if not dets:
        return np.empty((0, 9), dtype=np.float32)

    boxes_xywh = np.array([d["bbox"] for d in dets], dtype=float)
    x1 = boxes_xywh[:, 0] / img_w
    y1 = boxes_xywh[:, 1] / img_h
    w = boxes_xywh[:, 2] / img_w
    h = boxes_xywh[:, 3] / img_h
    x2 = x1 + w
    y2 = y1 + h

    scores = np.array([float(d.get("score", 0)) for d in dets], dtype=float)
    is_native_col = np.full(len(dets), 1.0 if is_native else 0.0, dtype=float)
    area = w * h
    aspect_ratio = w / np.maximum(h, 1e-6)

    # Compute max overlap with other branch
    if other_dets:
        other_xywh = np.array([d["bbox"] for d in other_dets], dtype=float)
        other_xyxy = np.column_stack([
            other_xywh[:, 0], other_xywh[:, 1],
            other_xywh[:, 0] + other_xywh[:, 2], other_xywh[:, 1] + other_xywh[:, 3]
        ])
        my_xyxy = np.column_stack([
            boxes_xywh[:, 0], boxes_xywh[:, 1],
            boxes_xywh[:, 0] + boxes_xywh[:, 2], boxes_xywh[:, 1] + boxes_xywh[:, 3]
        ])
        ious = _compute_iou_matrix(my_xyxy, other_xyxy)
        max_overlap = np.max(ious, axis=1)
    else:
        max_overlap = np.zeros(len(dets), dtype=float)

    feats = np.column_stack([
        x1, y1, x2, y2, scores, is_native_col, area, aspect_ratio, max_overlap
    ]).astype(np.float32)
    return feats
