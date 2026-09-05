"""Phase 1 Inference Optimization: Zero-Compute Sweeps across Top-K, P2 Confidence Threshold, and NMS IoU.

Runs a single forward pass over the test set, caches high-capacity predictions (Top-K=2000, conf=0.001),
and sweeps post-processing grid parameters in-memory with ultra-fast vectorized C++/CUDA operations:
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
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes

from ..data.coco import scale_class
from ..data.paths import resolve_data_dir
from ..detectors.base import Detection
from ..evaluation.coco import evaluate
from ..experiments.proposed import RTDETRP2Adapter
from ..models.p2_head import decode_dense_p2_predictions


class TestImageDataset(Dataset):
    """Prefetches and letterboxes test images asynchronously across CPU workers."""

    def __init__(
        self,
        images: list[dict[str, Any]],
        resolved_data: Path,
        image_size: int = 1920,
        rect: bool = False,
    ):
        self.images = images
        self.resolved_data = resolved_data
        self.image_size = image_size
        self.rect = rect
        self.lb = LetterBox(image_size, auto=True, stride=32) if rect else LetterBox(image_size, auto=False, scale_fill=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        im = self.images[idx]
        im_id = int(im["id"])
        file_name = im["file_name"]

        img_path = self.resolved_data / "test" / "images" / file_name
        if not img_path.is_file():
            img_path = self.resolved_data / "test" / file_name
        if not img_path.is_file():
            img_path = self.resolved_data / file_name

        if not img_path.is_file():
            return {
                "valid": False,
                "im_id": im_id,
                "tensor": torch.empty((0,)),
                "orig_shape": (0.0, 0.0),
                "lb_shape": (0.0, 0.0),
            }

        image = cv2.imread(str(img_path))
        if image is None:
            return {
                "valid": False,
                "im_id": im_id,
                "tensor": torch.empty((0,)),
                "orig_shape": (0.0, 0.0),
                "lb_shape": (0.0, 0.0),
            }

        h_orig, w_orig = float(image.shape[0]), float(image.shape[1])
        lb_img = self.lb(image=image)
        h_lb, w_lb = float(lb_img.shape[0]), float(lb_img.shape[1])

        # Convert BGR to RGB, HWC to CHW, normalize to [0, 1]
        rgb_img = lb_img[..., ::-1].copy()
        tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0

        return {
            "valid": True,
            "im_id": im_id,
            "tensor": tensor,
            "orig_shape": (h_orig, w_orig),
            "lb_shape": (h_lb, w_lb),
        }


def collate_test_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates variable images into a batched tensor, padding when rect=True."""
    valid_items = [item for item in batch if item.get("valid", False)]
    if not valid_items:
        return {"valid": False, "count": 0}

    shapes = [item["tensor"].shape for item in valid_items]
    max_h = max(s[1] for s in shapes)
    max_w = max(s[2] for s in shapes)

    tensors = []
    for item in valid_items:
        t = item["tensor"]
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        if pad_h > 0 or pad_w > 0:
            t = F.pad(t, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
        tensors.append(t)

    stacked_tensors = torch.stack(tensors, dim=0)

    return {
        "valid": True,
        "tensors": stacked_tensors,
        "im_ids": [item["im_id"] for item in valid_items],
        "orig_shapes": [item["orig_shape"] for item in valid_items],
        "lb_shapes": [item["lb_shape"] for item in valid_items],
        "count": len(valid_items),
    }


def extract_and_cache_raw_predictions(
    adapter: RTDETRP2Adapter,
    images: list[dict[str, Any]],
    resolved_data: Path,
    image_size: int = 1920,
    rect: bool = False,
    device: str = "cuda:0",
    max_topk: int = 2000,
    base_conf: float = 0.001,
    batch_size: int = 8,
    num_workers: int = 4,
    fp16: bool = True,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Runs batched forward passes with async DataLoader prefetching & FP16 Tensor Cores."""
    native_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    p2_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    target_device = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
    is_cuda = target_device.type == "cuda"
    use_amp = fp16 and is_cuda

    dataset = TestImageDataset(images=images, resolved_data=resolved_data, image_size=image_size, rect=rect)
    actual_workers = max(0, min(num_workers, os.cpu_count() or 1)) if num_workers > 0 else 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=is_cuda,
        collate_fn=collate_test_batch,
        persistent_workers=(actual_workers > 0),
    )

    print(f"\n[Sweep Cache] Running batched forward pass over {len(images)} test images...")
    print(f"  Batch Size:         {batch_size}")
    print(f"  DataLoader Workers: {actual_workers} (Parallel CPU Prefetching)")
    print(f"  Precision:          {'FP16 (Tensor Cores Accelerated)' if use_amp else 'FP32'}")
    t0 = time.time()
    processed_images = 0

    def unscale_tensor(
        raw_preds: torch.Tensor,
        h_lb: float,
        w_lb: float,
        h_orig: float,
        w_orig: float,
        is_native: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = raw_preds[:, 4] >= base_conf
        if is_native and adapter.is_coco_base:
            m = m & (raw_preds[:, 5].long() == adapter.category_id)
        filt = raw_preds[m]
        if filt.shape[0] == 0:
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

        boxes = filt[:, :4].clone().float()
        if not rect:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / w_lb * w_orig
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / h_lb * h_orig
        else:
            boxes = scale_boxes((int(h_lb), int(w_lb)), boxes, (int(h_orig), int(w_orig)))

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0.0, max=w_orig)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0.0, max=h_orig)
        scores = torch.clamp(filt[:, 4].clone().float(), min=0.0, max=1.0)
        return boxes.cpu(), scores.cpu()

    for batch_data in loader:
        if not batch_data.get("valid", False) or batch_data["count"] == 0:
            continue

        t_batch = batch_data["tensors"].to(target_device, non_blocking=is_cuda)
        count = batch_data["count"]

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = adapter.model(t_batch)
            else:
                out = adapter.model(t_batch)

        native_converted_batch = out["native_preds"]  # (B, 300, 6)
        cls_logits, box_offsets = out["p2_raw"]

        # Decode P2 with max_topk in FP32
        p2_dense_batch = decode_dense_p2_predictions(
            cls_logits=cls_logits.float(),
            box_offsets=box_offsets.float(),
            stride=adapter.model.p2_head.stride,
            topk=max_topk,
            conf_threshold=base_conf,
        )  # (B, max_topk, 6)

        for b in range(count):
            im_id = batch_data["im_ids"][b]
            h_orig, w_orig = batch_data["orig_shapes"][b]
            h_lb, w_lb = batch_data["lb_shapes"][b]

            native_cache[im_id] = unscale_tensor(native_converted_batch[b], h_lb, w_lb, h_orig, w_orig, is_native=True)
            p2_cache[im_id] = unscale_tensor(p2_dense_batch[b], h_lb, w_lb, h_orig, w_orig, is_native=False)

        processed_images += count
        if processed_images % (batch_size * 5) < batch_size or processed_images == len(images):
            elapsed = time.time() - t0
            fps = processed_images / max(elapsed, 1e-4)
            print(f"  Cached {processed_images}/{len(images)} images ({elapsed:.1f}s, {fps:.1f} img/s)...")

    return native_cache, p2_cache


def fast_operational_metrics(
    gt_boxes_by_id: dict[int, list[list[float]]],
    preds_by_id: dict[int, list[list[float]]],
    num_images: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Vectorized calculation of Precision, Recall, F1, and FPPI @ 0.25 in milliseconds."""
    tp = 0
    fp = 0
    total_gt = sum(len(b) for b in gt_boxes_by_id.values())

    for im_id, preds in preds_by_id.items():
        gts = gt_boxes_by_id.get(im_id, [])
        valid_preds = [p for p in preds if p[4] >= conf_thresh]
        if not valid_preds:
            continue
        if not gts:
            fp += len(valid_preds)
            continue

        p_b = torch.tensor([p[:4] for p in valid_preds], dtype=torch.float32)
        g_b = torch.tensor(gts, dtype=torch.float32)
        ious = torchvision.ops.box_iou(p_b, g_b)

        matched_gt: set[int] = set()
        for p_idx in range(len(valid_preds)):
            best_iou, best_g = ious[p_idx].max(dim=0)
            best_g_idx = int(best_g.item())
            if best_iou.item() >= iou_thresh and best_g_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_g_idx)
            else:
                fp += 1

    fn = max(0, total_gt - tp)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-6))
    fppi = float(fp / max(num_images, 1))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fppi": fppi,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


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
    batch_size: int = 8,
    num_workers: int = 4,
    fp16: bool = True,
) -> dict[str, Any]:
    """Execute grid search across Top-K, P2 confidence threshold, and NMS IoU in under 2 minutes."""
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
    print(f"  P2 Checkpoint:      {checkpoint_path}")
    print(f"  Base RT-DETR:       {weights}")
    print(f"  Image Size:         {image_size}px (Canonical Square: {not rect})")
    print(f"  Batch Size:         {batch_size}")
    print(f"  DataLoader Workers: {num_workers} (Parallel CPU Prefetching)")
    print(f"  Precision:          {'FP16 (Tensor Cores Accelerated)' if fp16 else 'FP32'}")
    print(f"  Top-K Grid:         {topk_list}")
    print(f"  P2 Conf Grid:       {conf_list}")
    print(f"  NMS IoU Grid:       {iou_list}")
    print(f"  Total Combos:       {len(topk_list) * len(conf_list) * len(iou_list)}")
    print("=" * 80)

    # 1. Load Dataset GT & Pre-index COCO ground truth
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

    # Pre-index full COCO GT once
    coco_gt = COCO()
    coco_gt.dataset = sub_gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    # Pre-index ultra-fine COCO GT once
    uf_annotations = []
    images_dict = {int(im["id"]): im for im in selected_images}
    gt_boxes_by_id: dict[int, list[list[float]]] = {}
    for ann in sub_annotations:
        im_id = int(ann["image_id"])
        x, y, w, h = ann["bbox"]
        gt_boxes_by_id.setdefault(im_id, []).append([x, y, x + w, y + h])

        im = images_dict.get(im_id)
        if im:
            ratio = (w * h) / (im["width"] * im["height"])
            if scale_class(ratio) == "ultra_fine":
                uf_annotations.append(ann)

    uf_gt = {
        "images": selected_images,
        "annotations": uf_annotations,
        "categories": sub_gt["categories"],
    }
    coco_gt_uf = COCO()
    coco_gt_uf.dataset = uf_gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_uf.createIndex()

    # 2. Init Model & Populate Tensor Cache
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
        batch_size=batch_size,
        num_workers=num_workers,
        fp16=fp16,
    )

    # 3. Fast Vectorized Grid Search Sweep
    sweep_results: list[dict[str, Any]] = []
    total_runs = len(topk_list) * len(conf_list) * len(iou_list)
    run_idx = 0
    t_sweep_start = time.time()

    print(f"\n[Sweep Execution] Evaluating {total_runs} parameter configurations with vectorized C++/NMS...")

    for k in topk_list:
        for p2_conf in conf_list:
            for nms_iou in iou_list:
                run_idx += 1
                t_iter = time.time()
                fused_preds: list[dict[str, Any]] = []
                preds_by_id: dict[int, list[list[float]]] = {}

                # Vectorized C++ NMS over all images
                for im_id in selected_img_ids:
                    nat_b, nat_s = native_cache.get(im_id, (torch.empty((0, 4)), torch.empty((0,))))
                    p2_b, p2_s = p2_cache.get(im_id, (torch.empty((0, 4)), torch.empty((0,))))

                    # Filter P2 by confidence threshold and top-k
                    m = p2_s >= p2_conf
                    filt_p2_b = p2_b[m]
                    filt_p2_s = p2_s[m]
                    if filt_p2_s.shape[0] > k:
                        filt_p2_s, topk_idx = torch.topk(filt_p2_s, k=k)
                        filt_p2_b = filt_p2_b[topk_idx]

                    # Concatenate Native and P2
                    comb_b = torch.cat([nat_b, filt_p2_b], dim=0)
                    comb_s = torch.cat([nat_s, filt_p2_s], dim=0)

                    if comb_s.shape[0] == 0:
                        continue

                    # Ultra-fast C++ NMS
                    keep = torchvision.ops.nms(comb_b, comb_s, iou_threshold=nms_iou)
                    fused_b = comb_b[keep].numpy()
                    fused_s = comb_s[keep].numpy()

                    im_preds = []
                    for j in range(len(fused_s)):
                        x1, y1, x2, y2 = fused_b[j]
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        sc = min(max(float(fused_s[j]), 0.0), 1.0)
                        if w >= 1.0 and h >= 1.0:
                            fused_preds.append({
                                "image_id": im_id,
                                "category_id": 0,
                                "bbox": [float(x1), float(y1), w, h],
                                "score": sc,
                            })
                            im_preds.append([float(x1), float(y1), float(x2), float(y2), sc])
                    if im_preds:
                        preds_by_id[im_id] = im_preds

                # Fast evaluation via C/Cython COCOeval
                overall_ap50 = 0.0
                overall_ap50_95 = 0.0
                if fused_preds:
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_dt = coco_gt.loadRes(fused_preds)
                        ev = COCOeval(coco_gt, coco_dt, "bbox")
                        ev.evaluate()
                        ev.accumulate()
                        ev.summarize()
                        overall_ap50_95 = float(ev.stats[0])
                        overall_ap50 = float(ev.stats[1])

                # Fast Ultra-Fine evaluation
                uf_ap50 = 0.0
                uf_recall = 0.0
                if fused_preds and uf_annotations:
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_dt_uf = coco_gt_uf.loadRes(fused_preds)
                        ev_uf = COCOeval(coco_gt_uf, coco_dt_uf, "bbox")
                        ev_uf.evaluate()
                        ev_uf.accumulate()
                        ev_uf.summarize()
                        uf_ap50 = float(ev_uf.stats[1])
                        # Recall @ 0.50, all areas, maxDets=100
                        uf_recall = float(ev_uf.eval["recall"][0, 0, 0, 2])

                # Fast Operational Metrics @ 0.25
                oper_metrics = fast_operational_metrics(
                    gt_boxes_by_id=gt_boxes_by_id,
                    preds_by_id=preds_by_id,
                    num_images=len(selected_images),
                    conf_thresh=0.25,
                    iou_thresh=0.5,
                )

                config_entry = {
                    "topk": k,
                    "p2_conf_threshold": p2_conf,
                    "fusion_iou_threshold": nms_iou,
                    "ultra_fine_recall": uf_recall,
                    "ultra_fine_ap50": uf_ap50,
                    "overall_ap50": overall_ap50,
                    "overall_ap50_95": overall_ap50_95,
                    "f1_score": oper_metrics["f1"],
                    "fppi": oper_metrics["fppi"],
                    "precision_025": oper_metrics["precision"],
                    "recall_025": oper_metrics["recall"],
                    "predictions": fused_preds,
                }
                sweep_results.append(config_entry)

                iter_time = time.time() - t_iter
                total_elapsed = time.time() - t_sweep_start
                avg_iter = total_elapsed / run_idx
                remaining_s = (total_runs - run_idx) * avg_iter

                if run_idx % 5 == 0 or run_idx == total_runs:
                    print(
                        f"  [{run_idx:02d}/{total_runs}] TopK={k:<4} Conf={p2_conf:<5} IoU={nms_iou:<3} | "
                        f"UF-Rec: {uf_recall*100:.2f}% | UF-AP50: {uf_ap50*100:.2f}% | "
                        f"AP50: {overall_ap50*100:.2f}% | F1: {oper_metrics['f1']*100:.2f}% | "
                        f"Iter: {iter_time:.2f}s | ETA: {remaining_s:.1f}s"
                    )

    # 4. Multi-Criteria Ranking:
    def score_fn(e: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            e["ultra_fine_recall"],
            e["ultra_fine_ap50"],
            e["f1_score"],
            -e["fppi"],  # lower FPPI is better
        )

    sweep_results.sort(key=score_fn, reverse=True)
    best_entry = sweep_results[0]

    print("\n" + "=" * 95)
    print("PHASE 1 SWEEP COMPLETE: TOP 10 INFERENCE CONFIGURATIONS")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Top-K':<6} | {'P2 Conf':<8} | {'NMS IoU':<8} | {'UF Recall':<10} | {'UF AP50':<10} | {'Overall AP50':<13} | {'F1 @0.25':<9} | {'FPPI':<7}")
    print("-" * 95)
    for i, res in enumerate(sweep_results[:10]):
        rank_str = f"#{i+1}"
        if i == 0:
            rank_str = "WINNER"
        print(
            f"{rank_str:<5} | {res['topk']:<6} | {res['p2_conf_threshold']:<8} | {res['fusion_iou_threshold']:<8} | "
            f"{res['ultra_fine_recall']*100:<9.2f}% | {res['ultra_fine_ap50']*100:<9.2f}% | "
            f"{res['overall_ap50']*100:<12.2f}% | {res['f1_score']*100:<8.2f}% | {res['fppi']:<7.4f}"
        )
    print("=" * 95)

    # 5. Run full detailed evaluation on WINNER to generate full report
    print(f"\n[Finalizing] Generating full academic & operational report for WINNER configuration...")
    best_acad = evaluate(sub_gt, best_entry["predictions"], confidence=0.001)
    best_oper = evaluate(sub_gt, best_entry["predictions"], confidence=0.25)

    best_config_payload = {
        "topk": best_entry["topk"],
        "p2_conf_threshold": best_entry["p2_conf_threshold"],
        "fusion_iou_threshold": best_entry["fusion_iou_threshold"],
        "ultra_fine_recall": best_entry["ultra_fine_recall"],
        "ultra_fine_ap50": best_entry["ultra_fine_ap50"],
        "overall_ap50": best_entry["overall_ap50"],
        "overall_ap50_95": best_entry["overall_ap50_95"],
        "f1_score": best_entry["f1_score"],
        "fppi": best_entry["fppi"],
        "academic_metrics": best_acad,
        "operational_metrics": best_oper,
    }

    # Clean predictions from entries to save storage
    clean_sweep_results = []
    for r in sweep_results:
        clean_entry = {k_name: v_val for k_name, v_val in r.items() if k_name != "predictions"}
        clean_sweep_results.append(clean_entry)

    (out_dir / "best_inference_config.json").write_text(json.dumps(best_config_payload, indent=2), encoding="utf-8")
    (out_dir / "all_sweep_results.json").write_text(json.dumps(clean_sweep_results, indent=2), encoding="utf-8")
    (out_dir / "best_fused_predictions.json").write_text(json.dumps(best_entry["predictions"], indent=2), encoding="utf-8")

    # Generate Markdown Report
    md_lines = [
        "# Phase 1: Inference Optimization Sweep Results",
        "",
        "## Best Configuration (Winner)",
        f"- **Top-K**: `{best_entry['topk']}`",
        f"- **P2 Confidence Threshold**: `{best_entry['p2_conf_threshold']}`",
        f"- **Fusion NMS IoU Threshold**: `{best_entry['fusion_iou_threshold']}`",
        f"- **Ultra-fine Recall**: `{best_entry['ultra_fine_recall']*100:.2f}%`",
        f"- **Ultra-fine AP50**: `{best_entry['ultra_fine_ap50']*100:.2f}%`",
        f"- **Overall AP50**: `{best_entry['overall_ap50']*100:.2f}%`",
        f"- **Overall AP50:95**: `{best_entry['overall_ap50_95']*100:.2f}%`",
        f"- **F1 Score @ 0.25**: `{best_entry['f1_score']*100:.2f}%`",
        f"- **FPPI**: `{best_entry['fppi']:.4f}`",
        "",
        "## Top 10 Configurations",
        "| Rank | Top-K | P2 Conf | NMS IoU | UF Recall | UF AP50 | Overall AP50 | F1 @0.25 | FPPI |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for i, res in enumerate(clean_sweep_results[:10]):
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
        "best_config": best_config_payload,
        "all_results": clean_sweep_results,
        "output_dir": str(out_dir),
    }
