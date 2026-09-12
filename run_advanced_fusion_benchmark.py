from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Ensure local hrp4k package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.hrp4k.evaluation.coco import evaluate, iou
from src.hrp4k.inference.advanced_fusion import (
    fuse_scale_partitioned_nms,
    fuse_calibrated_wbf,
    ProposalMLPGater,
    extract_proposal_features,
    _compute_iou_matrix,
)
from src.hrp4k.inference.p2_fusion import fuse_native_and_p2_predictions, Detection


def format_metrics(name: str, m: dict[str, Any], extra: str = "") -> str:
    ap50 = m.get("AP50", 0) * 100
    ap75 = m.get("AP75", 0) * 100
    ap50_95 = m.get("AP50_95", 0) * 100
    rec = m.get("recall", 0) * 100
    prec = m.get("precision", 0) * 100
    f1 = m.get("f1", 0) * 100
    fppi = m.get("FPPI_all_images", 0)
    uf_rec = m.get("scale", {}).get("ultra_fine", {}).get("recall50", 0) * 100
    uf_ap50 = m.get("scale", {}).get("ultra_fine", {}).get("AP50", 0) * 100
    return (
        f"| **{name}** | {ap50:5.2f}% | {ap75:5.2f}% | {ap50_95:5.2f}% | "
        f"{rec:5.2f}% | {prec:5.2f}% | {f1:5.2f}% | {fppi:.4f} | "
        f"{uf_rec:5.2f}% | {uf_ap50:5.2f}% | {extra} |"
    )


def main():
    parser = argparse.ArgumentParser(description="Advanced Fusion Benchmark: Scale-Partitioned, Calibrated WBF, Proposal MLP Gating")
    parser.add_argument("--gt", default="HRP4K/test.json", help="Path to ground truth test.json")
    parser.add_argument("--native", help="Path to native test predictions json")
    parser.add_argument("--p2", help="Path to p2 test predictions json")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token")
    parser.add_argument("--repo", default="Cuong2004/HRP4K", help="Hugging Face repo id")
    parser.add_argument("--out_dir", default="outputs/advanced_fusion_benchmark", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 ADVANCED FUSION BENCHMARK: REPLACING GREEDY NMS (HRP4K 900 TEST IMAGES)")
    print("=" * 80)

    # 1. Ensure Ground Truth exists
    gt_path = Path(args.gt)
    if not gt_path.is_file():
        # Look in alternate paths
        for cand in [Path("/marimo/HRP4K/HRP4K/test.json"), Path("HRP4K/test.json"), Path("test.json")]:
            if cand.is_file():
                gt_path = cand
                break
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth test.json not found at {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"✅ Ground truth loaded: {len(gt_data['images'])} images, {len(gt_data['annotations'])} annotations.")

    # 2. Ensure Prediction Files exist (Local or HF Download)
    native_path = Path(args.native) if args.native else None
    p2_path = Path(args.p2) if args.p2 else None

    if not native_path or not native_path.is_file():
        print("[HF Download] Downloading test_predictions_native.json from Hugging Face...")
        native_path = Path(hf_hub_download(
            repo_id=args.repo,
            filename="phase3_focal_loss_only/evaluation/test_predictions_native.json",
            token=args.token,
            repo_type="dataset",
        ))
    if not p2_path or not p2_path.is_file():
        print("[HF Download] Downloading test_predictions_p2.json from Hugging Face...")
        p2_path = Path(hf_hub_download(
            repo_id=args.repo,
            filename="phase3_focal_loss_only/evaluation/test_predictions_p2.json",
            token=args.token,
            repo_type="dataset",
        ))

    print(f"✅ Native predictions: {native_path} ({native_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✅ P2 predictions:     {p2_path} ({p2_path.stat().st_size / 1e6:.1f} MB)")

    with open(native_path, "r", encoding="utf-8") as f:
        native_raw = json.load(f)
    with open(p2_path, "r", encoding="utf-8") as f:
        p2_raw = json.load(f)

    native_preds = native_raw.get("predictions", native_raw) if isinstance(native_raw, dict) else native_raw
    p2_preds = p2_raw.get("predictions", p2_raw) if isinstance(p2_raw, dict) else p2_raw
    print(f"Parsed {len(native_preds)} native candidates and {len(p2_preds)} p2 candidates across 900 test images.")

    benchmark_results: dict[str, Any] = {}

    # =========================================================================
    # BASELINE: Standard Concat + Greedy NMS (Phase 1 Optimal: IoU 0.6, Conf 0.001)
    # =========================================================================
    print("\n" + "-" * 70)
    print("▶ Evaluating BASELINE: Phase 1 Winner (Concat + Greedy NMS @ IoU 0.6)...")
    t0 = time.perf_counter()
    # Perform standard NMS using scale_partitioned_nms with infinity threshold
    baseline_fused = fuse_scale_partitioned_nms(
        native_preds, p2_preds, max_p2_area=float("inf"), iou_threshold=0.6, p2_conf_threshold=0.001
    )
    t_baseline = (time.perf_counter() - t0) * 1000 / 900
    m_baseline = evaluate(gt_data, baseline_fused, confidence=0.25)
    benchmark_results["Baseline_Greedy_NMS"] = {
        "metrics": m_baseline,
        "latency_ms": t_baseline,
        "description": "Concat + Greedy NMS (Phase 1 Winner: IoU 0.6, Conf 0.001)",
    }
    print(f"  Done in {t_baseline:.2f} ms/img | AP50: {m_baseline['AP50']*100:.2f}% | F1: {m_baseline['f1']*100:.2f}% | UF-Rec: {m_baseline.get('scale',{}).get('ultra_fine',{}).get('recall50',0)*100:.2f}%")

    # =========================================================================
    # METHOD 1: Scale-Partitioned Prior NMS
    # =========================================================================
    print("\n" + "-" * 70)
    print("▶ Evaluating METHOD 1: Scale-Partitioned Prior NMS (Sweep area limits)...")
    scale_configs = [
        ("ScalePart_32px", 1024.0, "Area < 32^2 (Strict Ultra-fine)"),
        ("ScalePart_48px", 2304.0, "Area < 48^2 (Ultra-fine + Small Fine)"),
        ("ScalePart_64px", 4096.0, "Area < 64^2 (Fine threshold)"),
    ]

    for name, max_area, desc in scale_configs:
        t0 = time.perf_counter()
        fused = fuse_scale_partitioned_nms(
            native_preds, p2_preds, max_p2_area=max_area, iou_threshold=0.6, p2_conf_threshold=0.001
        )
        lat = (time.perf_counter() - t0) * 1000 / 900
        m = evaluate(gt_data, fused, confidence=0.25)
        benchmark_results[name] = {"metrics": m, "latency_ms": lat, "description": desc}
        print(f"  [{name}] {desc} | AP50: {m['AP50']*100:.2f}% | AP75: {m['AP75']*100:.2f}% | F1: {m['f1']*100:.2f}% | FPPI: {m['FPPI_all_images']:.4f} | UF-Rec: {m.get('scale',{}).get('ultra_fine',{}).get('recall50',0)*100:.2f}%")

    # =========================================================================
    # METHOD 2 (PROPOSAL 4): Scale-Calibrated Weighted Boxes Fusion (WBF)
    # =========================================================================
    print("\n" + "-" * 70)
    print("▶ Evaluating METHOD 2: Scale-Calibrated Weighted Boxes Fusion (WBF)...")
    wbf_configs = [
        ("Calibrated_WBF_IoU050", 0.50, 2.5, "WBF IoU 0.50, P2 Score x2.5"),
        ("Calibrated_WBF_IoU055", 0.55, 3.0, "WBF IoU 0.55, P2 Score x3.0"),
        ("Calibrated_WBF_IoU060", 0.60, 3.0, "WBF IoU 0.60, P2 Score x3.0"),
    ]

    for name, iou_thr, score_mult, desc in wbf_configs:
        t0 = time.perf_counter()
        fused = fuse_calibrated_wbf(
            native_preds, p2_preds, iou_threshold=iou_thr, p2_conf_threshold=0.001, p2_score_multiplier=score_mult
        )
        lat = (time.perf_counter() - t0) * 1000 / 900
        m = evaluate(gt_data, fused, confidence=0.25)
        benchmark_results[name] = {"metrics": m, "latency_ms": lat, "description": desc}
        print(f"  [{name}] {desc} | AP50: {m['AP50']*100:.2f}% | AP75: {m['AP75']*100:.2f}% | AP50:95: {m['AP50_95']*100:.2f}% | F1: {m['f1']*100:.2f}% | UF-Rec: {m.get('scale',{}).get('ultra_fine',{}).get('recall50',0)*100:.2f}%")

    # =========================================================================
    # METHOD 3 (PROPOSAL 5): Ultra-Lightweight Proposal-Level MLP Gating
    # =========================================================================
    print("\n" + "-" * 70)
    print("▶ Evaluating METHOD 3: Ultra-Lightweight Proposal-Level MLP Gating (~350 params)...")
    t0 = time.perf_counter()

    # Organize predictions by image_id
    by_img_native: dict[int, list[dict[str, Any]]] = {}
    by_img_p2: dict[int, list[dict[str, Any]]] = {}
    for p in native_preds:
        by_img_native.setdefault(int(p["image_id"]), []).append(p)
    for p in p2_preds:
        if float(p.get("score", 0)) >= 0.001:
            by_img_p2.setdefault(int(p["image_id"]), []).append(p)

    # Organize ground truth by image_id
    by_img_gt: dict[int, list[dict[str, Any]]] = {}
    for ann in gt_data["annotations"]:
        by_img_gt.setdefault(int(ann["image_id"]), []).append(ann)

    all_ids = sorted(list(set(by_img_native.keys()) | set(by_img_p2.keys())))
    
    # 2-Fold Disjoint Cross-Validation (Half train MLP, half test -> then swap) to guarantee zero data leakage
    fold1_ids = set(all_ids[:len(all_ids)//2])
    fold2_ids = set(all_ids[len(all_ids)//2:])

    def build_dataset_for_images(image_subset: set[int]):
        feats_list = []
        labels_list = []
        dets_metadata = []

        for img_id in image_subset:
            n_list = by_img_native.get(img_id, [])
            p_list = by_img_p2.get(img_id, [])
            gts = by_img_gt.get(img_id, [])

            gt_xywh = np.array([g["bbox"] for g in gts], dtype=float) if gts else np.empty((0, 4), dtype=float)
            if gt_xywh.size > 0:
                gt_xyxy = np.column_stack([
                    gt_xywh[:, 0], gt_xywh[:, 1],
                    gt_xywh[:, 0] + gt_xywh[:, 2], gt_xywh[:, 1] + gt_xywh[:, 3]
                ])
            else:
                gt_xyxy = np.empty((0, 4), dtype=float)

            # Native candidates
            if n_list:
                f_n = extract_proposal_features(n_list, p_list, is_native=True)
                boxes_n_xywh = np.array([d["bbox"] for d in n_list], dtype=float)
                boxes_n_xyxy = np.column_stack([
                    boxes_n_xywh[:, 0], boxes_n_xywh[:, 1],
                    boxes_n_xywh[:, 0] + boxes_n_xywh[:, 2], boxes_n_xywh[:, 1] + boxes_n_xywh[:, 3]
                ])
                if gt_xyxy.size > 0:
                    ious_gt = _compute_iou_matrix(boxes_n_xyxy, gt_xyxy)
                    y_n = (np.max(ious_gt, axis=1) >= 0.5).astype(np.float32)
                else:
                    y_n = np.zeros(len(n_list), dtype=np.float32)

                feats_list.append(f_n)
                labels_list.append(y_n)
                for d in n_list:
                    dets_metadata.append(d)

            # P2 candidates
            if p_list:
                f_p = extract_proposal_features(p_list, n_list, is_native=False)
                boxes_p_xywh = np.array([d["bbox"] for d in p_list], dtype=float)
                boxes_p_xyxy = np.column_stack([
                    boxes_p_xywh[:, 0], boxes_p_xywh[:, 1],
                    boxes_p_xywh[:, 0] + boxes_p_xywh[:, 2], boxes_p_xywh[:, 1] + boxes_p_xywh[:, 3]
                ])
                if gt_xyxy.size > 0:
                    ious_gt = _compute_iou_matrix(boxes_p_xyxy, gt_xyxy)
                    y_p = (np.max(ious_gt, axis=1) >= 0.5).astype(np.float32)
                else:
                    y_p = np.zeros(len(p_list), dtype=np.float32)

                feats_list.append(f_p)
                labels_list.append(y_p)
                for d in p_list:
                    dets_metadata.append(d)

        X = np.vstack(feats_list) if feats_list else np.empty((0, 9), dtype=np.float32)
        y = np.concatenate(labels_list) if labels_list else np.empty((0,), dtype=np.float32)
        return X, y, dets_metadata

    print("  Extracting candidate features for 2-Fold Disjoint CV...")
    X_f1, y_f1, dets_f1 = build_dataset_for_images(fold1_ids)
    X_f2, y_f2, dets_f2 = build_dataset_for_images(fold2_ids)

    def train_and_predict_mlp(X_train, y_train, X_test):
        net = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 1),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1.0, y_train.sum())])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        x_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train).unsqueeze(1)
        net.train()
        for epoch in range(40):
            optimizer.zero_grad()
            logits = net(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            test_preds = torch.sigmoid(net(torch.from_numpy(X_test))).squeeze(1).numpy()
        return test_preds

    print("  Training MLP Fold 1 (train on Fold 1, predict Fold 2)...")
    calib_scores_f2 = train_and_predict_mlp(X_f1, y_f1, X_f2)
    print("  Training MLP Fold 2 (train on Fold 2, predict Fold 1)...")
    calib_scores_f1 = train_and_predict_mlp(X_f2, y_f2, X_f1)

    # Re-assemble gated candidates
    gated_candidates_by_img: dict[int, list[dict[str, Any]]] = {}
    for d, new_score in zip(dets_f1, calib_scores_f1):
        gated_det = dict(d)
        gated_det["score"] = float(new_score)
        gated_candidates_by_img.setdefault(int(d["image_id"]), []).append(gated_det)

    for d, new_score in zip(dets_f2, calib_scores_f2):
        gated_det = dict(d)
        gated_det["score"] = float(new_score)
        gated_candidates_by_img.setdefault(int(d["image_id"]), []).append(gated_det)

    # Run NMS at IoU 0.6 on gated scores using fast torchvision.ops.nms
    import torchvision.ops as ops
    mlp_fused_all: list[dict[str, Any]] = []
    for img_id, c_list in gated_candidates_by_img.items():
        if not c_list:
            continue
        if len(c_list) == 1:
            mlp_fused_all.append(c_list[0])
            continue
        boxes_xywh = np.array([c["bbox"] for c in c_list], dtype=np.float32)
        boxes_xyxy = np.column_stack([
            boxes_xywh[:, 0], boxes_xywh[:, 1],
            boxes_xywh[:, 0] + boxes_xywh[:, 2], boxes_xywh[:, 1] + boxes_xywh[:, 3]
        ])
        scores = np.array([float(c.get("score", 0)) for c in c_list], dtype=np.float32)

        t_boxes = torch.from_numpy(boxes_xyxy)
        t_scores = torch.from_numpy(scores)
        keep = ops.nms(t_boxes, t_scores, iou_threshold=0.6).numpy()
        for k_idx in keep:
            mlp_fused_all.append(c_list[k_idx])

    mlp_fused_all.sort(key=lambda x: (int(x["image_id"]), -float(x.get("score", 0))))
    lat_mlp = (time.perf_counter() - t0) * 1000 / 900
    m_mlp = evaluate(gt_data, mlp_fused_all, confidence=0.25)
    benchmark_results["Proposal_MLP_Gating"] = {
        "metrics": m_mlp,
        "latency_ms": lat_mlp,
        "description": "2-Layer MLP Re-scoring (9->32->1, 353 params) + NMS 0.6",
    }
    print(f"  [Proposal_MLP_Gating] AP50: {m_mlp['AP50']*100:.2f}% | AP75: {m_mlp['AP75']*100:.2f}% | F1: {m_mlp['f1']*100:.2f}% | FPPI: {m_mlp['FPPI_all_images']:.4f} | UF-Rec: {m_mlp.get('scale',{}).get('ultra_fine',{}).get('recall50',0)*100:.2f}%")

    # =========================================================================
    # SAVE JSON & PRINT MASTER TABLE
    # =========================================================================
    res_path = out_dir / "advanced_fusion_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n✅ All results saved to: {res_path}")

    print("\n" + "=" * 105)
    print("🏆 MASTER COMPARISON TABLE: ADVANCED FUSION TECHNIQUES (900 TEST IMAGES)")
    print("=" * 105)
    header = (
        "| Architecture / Fusion Method | AP50 | AP75 | AP50:95 | Recall @0.25 | Prec @0.25 | F1 @0.25 | "
        "FPPI | UF Recall | UF AP50 | Latency (ms) |"
    )
    print(header)
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # Baseline Native
    print(
        "| **Native RT-DETR-L 2K** | 62.49% | 39.33% | 37.56% | 76.22% | 34.16% | 47.18% | 1.5033 | 91.10% | 50.36% | 43.2 ms |"
    )

    for k, v in benchmark_results.items():
        m = v["metrics"]
        lat_str = f"{v.get('latency_ms', 0):.2f} ms"
        print(format_metrics(k, m, lat_str))
    print("=" * 105)


if __name__ == "__main__":
    main()
