"""Phase 1 Inference Optimization: Zero-Compute Sweeps across Top-K, P2 Confidence Threshold, and NMS IoU.

Runs a single forward pass over the test set, caches high-capacity predictions (Top-K=2000, conf=0.001),
and sweeps post-processing grid parameters in-memory:
  - Top-K: [300, 500, 1000, 2000]
  - P2 Confidence Threshold: [0.001, 0.003, 0.005, 0.01, 0.02]
  - Fusion NMS IoU Threshold: [0.4, 0.5, 0.6, 0.7]

Ranks configurations by:
  1. Ultra-fine Recall (Primary)
  2. Ultra-fine AP50 (Secondary)
  3. F1 Score @ 0.25
  4. FPPI (lower is better)
  5. Overall AP50 preservation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

from ..data.paths import resolve_data_dir
from ..detectors.base import Detection
from ..evaluation.coco import evaluate
from ..experiments.proposed import RTDETRP2Adapter
from ..inference.p2_fusion import fuse_native_and_p2_predictions


def extract_and_cache_raw_predictions(
    adapter: RTDETRP2Adapter,
    images: list[dict[str, Any]],
    resolved_data: Path,
    image_size: int = 1920,
    rect: bool = False,
    device: str = "cuda:0",
    max_topk: int = 2000,
    base_conf: float = 0.001,
) -> tuple[dict[int, list[Detection]], dict[int, list[Detection]]]:
    """Runs a single forward pass and returns unscaled (native_dets_by_id, p2_dets_by_id)."""
    lb = LetterBox(image_size, auto=True, stride=32) if rect else LetterBox(image_size, auto=False, scale_fill=True)

    native_cache: dict[int, list[Detection]] = {}
    p2_cache: dict[int, list[Detection]] = {}

    print(f"\n[Sweep Cache] Running single forward pass over {len(images)} test images (max Top-K={max_topk})...")

    for idx, im in enumerate(images):
        im_id = int(im["id"])
        file_name = im["file_name"]
        img_path = resolved_data / "test" / "images" / file_name
        if not img_path.is_file():
            img_path = resolved_data / "test" / file_name
        if not img_path.is_file():
            img_path = resolved_data / file_name
        if not img_path.is_file():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h_orig, w_orig = image.shape[:2]

        lb_img = lb(image=image)
        h_lb, w_lb = lb_img.shape[:2]

        # Convert BGR to RGB
        rgb_img = lb_img[..., ::-1].copy()
        t = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(device)

        with torch.no_grad():
            out = adapter.model(t)

        native_raw = out["native_preds"][0]  # (300, 6)
        cls_logits, box_offsets = out["p2_raw"]

        # Decode P2 with max_topk to allow slicing downstream
        from ..models.p2_head import decode_dense_p2_predictions
        p2_dense = decode_dense_p2_predictions(
            cls_logits=cls_logits,
            box_offsets=box_offsets,
            stride=adapter.model.p2_head.stride,
            topk=max_topk,
            conf_threshold=base_conf,
        )[0]

        def unscale_preds(preds: torch.Tensor, is_native: bool = False) -> list[Detection]:
            m = preds[:, 4] >= base_conf
            if is_native and adapter.is_coco_base:
                m = m & (preds[:, 5].long() == adapter.category_id)
            if not m.any():
                return []
            filt = preds[m]
            if rect:
                unscaled = scale_boxes((h_lb, w_lb), filt[:, :4].clone(), (h_orig, w_orig)).cpu().numpy()
            else:
                unscaled = filt[:, :4].clone().cpu().numpy()
                unscaled[:, [0, 2]] = unscaled[:, [0, 2]] / float(w_lb) * float(w_orig)
                unscaled[:, [1, 3]] = unscaled[:, [1, 3]] / float(h_lb) * float(h_orig)
            sc = filt[:, 4].cpu().numpy()
            dets = []
            for i in range(len(sc)):
                x1, y1, x2, y2 = unscaled[i]
                c_x1 = float(np.clip(min(x1, x2), 0, w_orig))
                c_y1 = float(np.clip(min(y1, y2), 0, h_orig))
                c_x2 = float(np.clip(max(x1, x2), 0, w_orig))
                c_y2 = float(np.clip(max(y1, y2), 0, h_orig))
                if (c_x2 - c_x1) >= 1.0 and (c_y2 - c_y1) >= 1.0:
                    dets.append(Detection((c_x1, c_y1, c_x2, c_y2), float(sc[i]), 0))
            return dets

        native_cache[im_id] = unscale_preds(native_raw, is_native=True)
        p2_cache[im_id] = unscale_preds(p2_dense, is_native=False)

        if (idx + 1) % 100 == 0 or idx == len(images) - 1:
            print(f"  Processed {idx + 1}/{len(images)} images for cache...")

    return native_cache, p2_cache


def run_inference_sweep(
    checkpoint_path: str | Path,
    data_dir: str | Path = "HRP4K",
    weights: str | Path = "outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
    image_size: int = 1920,
    device: str | None = None,
    output_dir: str | Path = "outputs/inference_sweep",
    num_images: int = 0,
    rect: bool = False,
    topk_candidates: list[int] | None = None,
    conf_candidates: list[float] | None = None,
    iou_candidates: list[float] | None = None,
) -> dict[str, Any]:
    """Execute grid search across Top-K, P2 confidence threshold, and NMS IoU."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d_str = str(device).strip() if device is not None else ""
    if d_str.isdigit():
        target_device = f"cuda:{d_str}"
    elif d_str:
        target_device = d_str
    else:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    topk_list = topk_candidates or [300, 500, 1000, 2000]
    conf_list = conf_candidates or [0.001, 0.003, 0.005, 0.01, 0.02]
    iou_list = iou_candidates or [0.4, 0.5, 0.6, 0.7]

    print("=" * 80)
    print("PHASE 1: INFERENCE-TIME ZERO-COMPUTE OPTIMIZATION SWEEP")
    print("=" * 80)
    print(f"  P2 Checkpoint:     {checkpoint_path}")
    print(f"  Base RT-DETR:      {weights}")
    print(f"  Image Size:        {image_size}px (Canonical Square: {not rect})")
    print(f"  Top-K Grid:        {topk_list}")
    print(f"  P2 Conf Grid:      {conf_list}")
    print(f"  NMS IoU Grid:      {iou_list}")
    print(f"  Total Combos:      {len(topk_list) * len(conf_list) * len(iou_list)}")
    print("=" * 80)

    # 1. Load Dataset GT
    resolved_data = resolve_data_dir(data_dir)
    test_json = resolved_data / "test.json"
    if not test_json.is_file():
        for c in (Path("HRP4K/test.json"), Path("test.json"), Path("../HRP4K/test.json")):
            if c.is_file():
                test_json = c
                resolved_data = c.parent
                break
    with open(test_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    all_images = gt.get("images", [])
    selected_images = all_images[:num_images] if num_images > 0 else all_images
    selected_img_ids = [int(im["id"]) for im in selected_images]
    selected_id_set = set(selected_img_ids)
    sub_annotations = [a for a in gt.get("annotations", []) if int(a["image_id"]) in selected_id_set]

    sub_gt = {
        "images": selected_images,
        "annotations": sub_annotations,
        "categories": gt.get("categories", [{"id": 0, "name": "pothole"}]),
    }

    # 2. Init Model & Populate Cache
    adapter = RTDETRP2Adapter(
        weights=weights,
        category_id=0,
        device=target_device,
        p2_checkpoint=checkpoint_path,
        mode="fused",
    )
    adapter.model.eval()

    max_k = max(topk_list)
    min_c = min(conf_list)
    native_cache, p2_cache = extract_and_cache_raw_predictions(
        adapter=adapter,
        images=selected_images,
        resolved_data=resolved_data,
        image_size=image_size,
        rect=rect,
        device=target_device,
        max_topk=max_k,
        base_conf=min_c,
    )

    # 3. Grid Search Sweep
    sweep_results: list[dict[str, Any]] = []
    total_runs = len(topk_list) * len(conf_list) * len(iou_list)
    run_idx = 0
    t_start = time.time()

    print(f"\n[Sweep Execution] Evaluating {total_runs} parameter configurations in-memory...")

    for k in topk_list:
        for p2_conf in conf_list:
            for nms_iou in iou_list:
                run_idx += 1
                fused_preds: list[dict[str, Any]] = []

                for im_id in selected_img_ids:
                    nat_dets = native_cache.get(im_id, [])
                    raw_p2_dets = p2_cache.get(im_id, [])

                    # Filter P2 by confidence threshold and top-k
                    filtered_p2 = [d for d in raw_p2_dets if d.score >= p2_conf]
                    filtered_p2 = sorted(filtered_p2, key=lambda d: d.score, reverse=True)[:k]

                    # NMS Fusion
                    fused = fuse_native_and_p2_predictions(nat_dets, filtered_p2, iou_threshold=nms_iou)

                    for d in fused:
                        w = float(d.xyxy[2] - d.xyxy[0])
                        h = float(d.xyxy[3] - d.xyxy[1])
                        if w >= 1.0 and h >= 1.0:
                            fused_preds.append({
                                "image_id": im_id,
                                "category_id": 0,
                                "bbox": [float(d.xyxy[0]), float(d.xyxy[1]), w, h],
                                "score": float(d.score),
                            })

                # Compute Academic & Operational Metrics
                acad = evaluate(sub_gt, fused_preds, confidence=0.001)
                oper = evaluate(sub_gt, fused_preds, confidence=0.25)

                n_imgs = max(1, len(selected_images))
                fppi = oper.get("fp", 0) / n_imgs
                uf_recall = acad.get("scale", {}).get("ultra_fine", {}).get("recall50", 0.0)
                uf_ap50 = acad.get("scale", {}).get("ultra_fine", {}).get("AP50", 0.0)
                overall_ap50 = acad.get("AP50", 0.0)
                overall_ap50_95 = acad.get("AP50_95", 0.0)
                overall_f1 = oper.get("f1", 0.0)

                config_entry = {
                    "topk": k,
                    "p2_conf_threshold": p2_conf,
                    "fusion_iou_threshold": nms_iou,
                    "ultra_fine_recall": uf_recall,
                    "ultra_fine_ap50": uf_ap50,
                    "overall_ap50": overall_ap50,
                    "overall_ap50_95": overall_ap50_95,
                    "f1_score": overall_f1,
                    "fppi": fppi,
                    "academic_metrics": acad,
                    "operational_metrics": oper,
                }
                sweep_results.append(config_entry)

                if run_idx % 10 == 0 or run_idx == total_runs:
                    elapsed = time.time() - t_start
                    print(f"  [{run_idx}/{total_runs}] TopK={k:<4} Conf={p2_conf:<5} IoU={nms_iou:<3} | UF-Rec: {uf_recall*100:.2f}% | UF-AP50: {uf_ap50*100:.2f}% | F1: {overall_f1*100:.2f}% | Elapsed: {elapsed:.1f}s")

    # 4. Multi-Criteria Ranking:
    # Primary: Ultra-fine Recall, Secondary: Ultra-fine AP50, Tertiary: F1, Lowest FPPI
    def score_fn(e: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            e["ultra_fine_recall"],
            e["ultra_fine_ap50"],
            e["f1_score"],
            -e["fppi"],  # lower FPPI is better
        )

    sweep_results.sort(key=score_fn, reverse=True)
    best_config = sweep_results[0]

    print("\n" + "=" * 95)
    print("PHASE 1 SWEEP COMPLETE: TOP 10 INFERENCE CONFIGURATIONS")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Top-K':<6} | {'P2 Conf':<8} | {'NMS IoU':<8} | {'UF Recall':<10} | {'UF AP50':<10} | {'Overall AP50':<13} | {'F1 @0.25':<9} | {'FPPI':<7}")
    print("-" * 95)
    for i, res in enumerate(sweep_results[:10]):
        rank_str = f"#{i+1}"
        if i == 0:
            rank_str = "WINNER"
        print(f"{rank_str:<5} | {res['topk']:<6} | {res['p2_conf_threshold']:<8} | {res['fusion_iou_threshold']:<8} | {res['ultra_fine_recall']*100:<9.2f}% | {res['ultra_fine_ap50']*100:<9.2f}% | {res['overall_ap50']*100:<12.2f}% | {res['f1_score']*100:<8.2f}% | {res['fppi']:<7.4f}")
    print("=" * 95)

    # 5. Save Summary Outputs
    (out_dir / "best_inference_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    (out_dir / "all_sweep_results.json").write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")

    # Generate Markdown Report
    md_lines = [
        "# Phase 1: Inference Optimization Sweep Results",
        "",
        "## Best Configuration",
        f"- **Top-K**: `{best_config['topk']}`",
        f"- **P2 Confidence Threshold**: `{best_config['p2_conf_threshold']}`",
        f"- **Fusion NMS IoU Threshold**: `{best_config['fusion_iou_threshold']}`",
        f"- **Ultra-fine Recall**: `{best_config['ultra_fine_recall']*100:.2f}%`",
        f"- **Ultra-fine AP50**: `{best_config['ultra_fine_ap50']*100:.2f}%`",
        f"- **Overall AP50**: `{best_config['overall_ap50']*100:.2f}%`",
        f"- **Overall AP50:95**: `{best_config['overall_ap50_95']*100:.2f}%`",
        f"- **F1 Score**: `{best_config['f1_score']*100:.2f}%`",
        f"- **FPPI**: `{best_config['fppi']:.4f}`",
        "",
        "## Top 10 Configurations",
        "| Rank | Top-K | P2 Conf | NMS IoU | UF Recall | UF AP50 | Overall AP50 | F1 @0.25 | FPPI |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for i, res in enumerate(sweep_results[:10]):
        tag = "**Winner**" if i == 0 else f"#{i+1}"
        md_lines.append(
            f"| {tag} | {res['topk']} | {res['p2_conf_threshold']} | {res['fusion_iou_threshold']} | "
            f"{res['ultra_fine_recall']*100:.2f}% | {res['ultra_fine_ap50']*100:.2f}% | "
            f"{res['overall_ap50']*100:.2f}% | {res['f1_score']*100:.2f}% | {res['fppi']:.4f} |"
        )
    (out_dir / "inference_sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nSaved best inference config to: {out_dir / 'best_inference_config.json'}")
    print(f"Saved markdown summary to: {out_dir / 'inference_sweep_summary.md'}")

    return {
        "best_config": best_config,
        "all_results": sweep_results,
        "output_dir": str(out_dir),
    }
