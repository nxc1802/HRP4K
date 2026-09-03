#!/usr/bin/env python3
"""HRP4K Evaluation Comparison: P2-Only vs Native vs Fused.

Runs test set inference and produces a side-by-side comparative evaluation of:
  1. P2-Only Head
  2. Native RT-DETR
  3. Fused (Native + P2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
import time

from hrp4k.data.paths import resolve_data_dir
from hrp4k.experiments.proposed import RTDETRP2Adapter
from hrp4k.detectors.base import Detection
from hrp4k.inference.p2_fusion import fuse_native_and_p2_predictions
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes


def evaluate_coco_subset(gt_dict: dict, pred_list: list[dict], img_ids: list[int]) -> dict[str, float]:
    """Run COCOeval and return AP50, AP50-95, AP75, AR100."""
    if not pred_list:
        return {"AP50": 0.0, "AP50_95": 0.0, "AP75": 0.0, "AR100": 0.0}
    try:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(pred_list)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.imgIds = img_ids
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return {
            "AP50_95": float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]),
            "AR100": float(ev.stats[8]),
        }
    except Exception as exc:
        print(f"Warning in COCOeval: {exc}")
        return {"AP50": 0.0, "AP50_95": 0.0, "AP75": 0.0, "AR100": 0.0}


def run_comparison(
    checkpoint_path: str | Path,
    data_dir: str | Path = "HRP4K",
    weights: str | Path = "outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
    image_size: int = 1920,
    confidence: float = 0.001,
    num_images: int = 0,  # 0 means all
    device: str | None = None,
    output_dir: str | Path = "outputs/evaluation_comparison",
    hf_upload: bool = False,
    hf_repo: str = "Cuong2004/HRP4K",
    hf_token: str | None = None,
) -> dict:
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    d_str = str(device).strip() if device is not None else ""
    if d_str.isdigit():
        target_device = f"cuda:{d_str}"
    elif d_str:
        target_device = d_str
    else:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HRP4K COMPREHENSIVE EVALUATION: P2-ONLY vs NATIVE vs FUSED")
    print("=" * 80)
    print(f"  P2 Checkpoint:  {ckpt_path}")
    print(f"  Base Weights:   {weights}")
    print(f"  Image Size:     {image_size}px")
    print(f"  Confidence:     {confidence}")
    print(f"  Device:         {target_device}")
    print("=" * 80)

    # 1. Load dataset & GT
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

    print(f"\n[Inference] Evaluating {len(selected_images)} images ({len(sub_annotations)} GT potholes)...")

    # 2. Init Adapter
    adapter = RTDETRP2Adapter(
        weights=weights,
        category_id=0,
        device=target_device,
        p2_checkpoint=ckpt_path,
        mode="fused",
    )
    adapter.model.eval()

    p2_predictions = []
    native_predictions = []
    fused_predictions = []
    latencies = []

    lb = LetterBox(image_size, auto=True, stride=32)

    for idx, im in enumerate(selected_images):
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

        # Convert BGR to RGB (matching Ultralytics YOLODataset Format transform)
        rgb_img = lb_img[..., ::-1].copy()
        t = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(target_device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = adapter.model(t)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        native_raw = out["native_preds"][0]  # (300, 6)
        p2_raw = out["p2_preds"][0]          # (300, 6)

        # Convert to detections
        def get_dets(preds: torch.Tensor, is_native: bool = False) -> list[Detection]:
            m = preds[:, 4] >= min(confidence, 0.001)
            if is_native and adapter.is_coco_base:
                m = m & (preds[:, 5].long() == adapter.category_id)
            if not m.any():
                return []
            filt = preds[m]
            unscaled = scale_boxes((h_lb, w_lb), filt[:, :4].clone(), (h_orig, w_orig)).cpu().numpy()
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

        p2_dets = get_dets(p2_raw, is_native=False)
        nat_dets = get_dets(native_raw, is_native=True)
        fused_dets = fuse_native_and_p2_predictions(nat_dets, p2_dets, iou_threshold=0.5)

        for d in p2_dets:
            w = float(d.xyxy[2] - d.xyxy[0])
            h = float(d.xyxy[3] - d.xyxy[1])
            if w >= 1.0 and h >= 1.0:
                p2_predictions.append({
                    "image_id": im_id, "category_id": 0,
                    "bbox": [float(d.xyxy[0]), float(d.xyxy[1]), w, h],
                    "score": float(d.score),
                })
        for d in nat_dets:
            w = float(d.xyxy[2] - d.xyxy[0])
            h = float(d.xyxy[3] - d.xyxy[1])
            if w >= 1.0 and h >= 1.0:
                native_predictions.append({
                    "image_id": im_id, "category_id": 0,
                    "bbox": [float(d.xyxy[0]), float(d.xyxy[1]), w, h],
                    "score": float(d.score),
                })
        for d in fused_dets:
            w = float(d.xyxy[2] - d.xyxy[0])
            h = float(d.xyxy[3] - d.xyxy[1])
            if w >= 1.0 and h >= 1.0:
                fused_predictions.append({
                    "image_id": im_id, "category_id": 0,
                    "bbox": [float(d.xyxy[0]), float(d.xyxy[1]), w, h],
                    "score": float(d.score),
                })

        if (idx + 1) % 50 == 0 or idx == len(selected_images) - 1:
            print(f"  Processed {idx + 1}/{len(selected_images)} images...")

    # 3. Save Raw Prediction Files Locally
    print("\n[Predictions] Saving raw predictions for P2-Only, Native, and Fused...")
    (out_dir / "test_predictions_p2.json").write_text(json.dumps(p2_predictions, indent=2), encoding="utf-8")
    (out_dir / "test_predictions_native.json").write_text(json.dumps(native_predictions, indent=2), encoding="utf-8")
    (out_dir / "test_predictions_fused.json").write_text(json.dumps(fused_predictions, indent=2), encoding="utf-8")

    # 4. Compute Metrics for BOTH Benchmarks:
    from hrp4k.evaluation.coco import evaluate

    print("\n[Metrics] 1. Computing Benchmark A: Academic Detection Protocol (COCO-style, conf=0.001)...")
    p2_acad = evaluate(sub_gt, p2_predictions, confidence=0.001)
    nat_acad = evaluate(sub_gt, native_predictions, confidence=0.001)
    fused_acad = evaluate(sub_gt, fused_predictions, confidence=0.001)

    print("[Metrics] 2. Computing Benchmark B: Operational Deployment Protocol (Standard, conf=0.25)...")
    p2_oper = evaluate(sub_gt, p2_predictions, confidence=0.25)
    nat_oper = evaluate(sub_gt, native_predictions, confidence=0.25)
    fused_oper = evaluate(sub_gt, fused_predictions, confidence=0.25)

    n_imgs = max(1, len(selected_images))
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    def delta_str(val_fused: float, val_base: float, is_percent: bool = True, higher_is_better: bool = True) -> str:
        diff = val_fused - val_base
        sign = "+" if diff > 0 else ""
        sym = "▲" if (diff > 0 if higher_is_better else diff < 0) else ("▼" if diff != 0 else "=")
        if is_percent:
            return f"{sign}{diff*100:+.2f}% {sym}"
        return f"{sign}{diff:+.4f} {sym}"

    # 5. Print BENCHMARK A: Academic / Scientific Detection Benchmark
    print("\n" + "=" * 95)
    print("BENCHMARK A: ACADEMIC / SCIENTIFIC DETECTION BENCHMARK (Protocol: COCO-style, conf=0.001)")
    print("=" * 95)
    print(f"{'Metric':<28} | {'Baseline (Native)':<18} | {'Proposed (Fused)':<18} | {'Δ (Gain)':<12} | {'P2-Only':<10}")
    print("-" * 95)
    ap50_95_b = nat_acad.get('AP50_95', 0)
    ap50_95_f = fused_acad.get('AP50_95', 0)
    print(f"{'mAP50:95':<28} | {ap50_95_b*100:<17.2f}% | {ap50_95_f*100:<17.2f}% | {delta_str(ap50_95_f, ap50_95_b):<12} | {p2_acad.get('AP50_95', 0)*100:<9.2f}%")

    ap50_b = nat_acad.get('AP50', 0)
    ap50_f = fused_acad.get('AP50', 0)
    print(f"{'AP50':<28} | {ap50_b*100:<17.2f}% | {ap50_f*100:<17.2f}% | {delta_str(ap50_f, ap50_b):<12} | {p2_acad.get('AP50', 0)*100:<9.2f}%")

    ap75_b = nat_acad.get('AP75', 0)
    ap75_f = fused_acad.get('AP75', 0)
    print(f"{'AP75':<28} | {ap75_b*100:<17.2f}% | {ap75_f*100:<17.2f}% | {delta_str(ap75_f, ap75_b):<12} | {p2_acad.get('AP75', 0)*100:<9.2f}%")

    rec_b = nat_acad.get('recall', 0)
    rec_f = fused_acad.get('recall', 0)
    print(f"{'Overall Recall':<28} | {rec_b*100:<17.2f}% | {rec_f*100:<17.2f}% | {delta_str(rec_f, rec_b):<12} | {p2_acad.get('recall', 0)*100:<9.2f}%")

    for sc, sc_label in [("ultra_fine", "AP_ultra_fine (<32²)"), ("fine", "AP_fine (32²-96²)"), ("medium", "AP_medium (96²-144²)"), ("large", "AP_large (≥144²)")]:
        b_val = nat_acad.get("scale", {}).get(sc, {}).get("AP50", 0.0)
        f_val = fused_acad.get("scale", {}).get(sc, {}).get("AP50", 0.0)
        p_val = p2_acad.get("scale", {}).get(sc, {}).get("AP50", 0.0)
        print(f"{sc_label:<28} | {b_val*100:<17.2f}% | {f_val*100:<17.2f}% | {delta_str(f_val, b_val):<12} | {p_val*100:<9.2f}%")

    rec_uf_b = nat_acad.get("scale", {}).get("ultra_fine", {}).get("recall50", 0.0)
    rec_uf_f = fused_acad.get("scale", {}).get("ultra_fine", {}).get("recall50", 0.0)
    rec_uf_p = p2_acad.get("scale", {}).get("ultra_fine", {}).get("recall50", 0.0)
    print(f"{'Recall_ultra_fine (<32²)':<28} | {rec_uf_b*100:<17.2f}% | {rec_uf_f*100:<17.2f}% | {delta_str(rec_uf_f, rec_uf_b):<12} | {rec_uf_p*100:<9.2f}%")
    print("=" * 95)

    # 6. Print BENCHMARK B: Operational / Deployment Benchmark
    print("\n" + "=" * 95)
    print("BENCHMARK B: OPERATIONAL / DEPLOYMENT BENCHMARK (Standard Operating Condition, conf=0.25)")
    print("=" * 95)
    print(f"{'Metric':<28} | {'Baseline (Native)':<18} | {'Proposed (Fused)':<18} | {'Δ (Gain)':<12} | {'P2-Only':<10}")
    print("-" * 95)
    prec_b = nat_oper.get('precision', 0)
    prec_f = fused_oper.get('precision', 0)
    print(f"{'Precision @0.25':<28} | {prec_b*100:<17.2f}% | {prec_f*100:<17.2f}% | {delta_str(prec_f, prec_b):<12} | {p2_oper.get('precision', 0)*100:<9.2f}%")

    rec_op_b = nat_oper.get('recall', 0)
    rec_op_f = fused_oper.get('recall', 0)
    print(f"{'Recall @0.25':<28} | {rec_op_b*100:<17.2f}% | {rec_op_f*100:<17.2f}% | {delta_str(rec_op_f, rec_op_b):<12} | {p2_oper.get('recall', 0)*100:<9.2f}%")

    f1_b = nat_oper.get('f1', 0)
    f1_f = fused_oper.get('f1', 0)
    print(f"{'F1 @0.25':<28} | {f1_b*100:<17.2f}% | {f1_f*100:<17.2f}% | {delta_str(f1_f, f1_b):<12} | {p2_oper.get('f1', 0)*100:<9.2f}%")

    fppi_b = nat_oper.get('fp', 0) / n_imgs
    fppi_f = fused_oper.get('fp', 0) / n_imgs
    fppi_p = p2_oper.get('fp', 0) / n_imgs
    print(f"{'FP / image (FPPI)':<28} | {fppi_b:<18.4f} | {fppi_f:<18.4f} | {delta_str(fppi_f, fppi_b, is_percent=False, higher_is_better=False):<12} | {fppi_p:<10.4f}")

    fn_b = nat_oper.get('fn', 0) / n_imgs
    fn_f = fused_oper.get('fn', 0) / n_imgs
    fn_p = p2_oper.get('fn', 0) / n_imgs
    print(f"{'FN / image':<28} | {fn_b:<18.4f} | {fn_f:<18.4f} | {delta_str(fn_f, fn_b, is_percent=False, higher_is_better=False):<12} | {fn_p:<10.4f}")

    for sc, sc_label in [("ultra_fine", "Recall_ultra_fine @0.25"), ("fine", "Recall_fine @0.25"), ("medium", "Recall_medium @0.25"), ("large", "Recall_large @0.25")]:
        b_val = nat_oper.get("scale", {}).get(sc, {}).get("recall50", 0.0)
        f_val = fused_oper.get("scale", {}).get(sc, {}).get("recall50", 0.0)
        p_val = p2_oper.get("scale", {}).get(sc, {}).get("recall50", 0.0)
        print(f"{sc_label:<28} | {b_val*100:<17.2f}% | {f_val*100:<17.2f}% | {delta_str(f_val, b_val):<12} | {p_val*100:<9.2f}%")

    print(f"{'Mean Latency (ms)':<28} | {mean_lat:<18.2f} | {mean_lat:<18.2f} | {'=':<12} | {mean_lat:<10.2f}")
    print(f"{'Inference FPS':<28} | {fps:<18.2f} | {fps:<18.2f} | {'=':<12} | {fps:<10.2f}")
    print("=" * 95)

    # 7. Save results locally
    results = {
        "benchmark_academic_conf_0_001": {
            "baseline_native": nat_acad,
            "proposed_fused": fused_acad,
            "p2_only": p2_acad,
        },
        "benchmark_operational_conf_0_25": {
            "baseline_native": {**nat_oper, "fppi": fppi_b, "fn_per_img": fn_b},
            "proposed_fused": {**fused_oper, "fppi": fppi_f, "fn_per_img": fn_f},
            "p2_only": {**p2_oper, "fppi": fppi_p, "fn_per_img": fn_p},
        },
        "settings": {
            "checkpoint": str(ckpt_path),
            "weights": str(weights),
            "image_size": image_size,
            "num_images": len(selected_images),
            "mean_latency_ms": mean_lat,
            "fps": fps,
        },
    }
    comp_path = out_dir / "test_metrics_comparison.json"
    comp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "test_metrics_academic.json").write_text(json.dumps(results["benchmark_academic_conf_0_001"], indent=2), encoding="utf-8")
    (out_dir / "test_metrics_operational.json").write_text(json.dumps(results["benchmark_operational_conf_0_25"], indent=2), encoding="utf-8")
    print(f"\nSaved comparison results to: {comp_path}")

    # 8. Upload to Hugging Face if requested
    if hf_upload:
        try:
            from huggingface_hub import HfApi
            token = hf_token or os.environ.get("HF_TOKEN")
            repo = hf_repo or "Cuong2004/HRP4K"
            print(f"\n[HF Upload] Uploading comprehensive metrics & predictions to Hugging Face repo: {repo}...")
            api = HfApi(token=token)
            files_to_upload = [
                (comp_path, "experiments/9b68a1164e96/test/test_metrics_comparison.json"),
                (out_dir / "test_metrics_academic.json", "experiments/9b68a1164e96/test/test_metrics_academic.json"),
                (out_dir / "test_metrics_operational.json", "experiments/9b68a1164e96/test/test_metrics_operational.json"),
                (out_dir / "test_predictions_p2.json", "experiments/9b68a1164e96/test/test_predictions_p2.json"),
                (out_dir / "test_predictions_native.json", "experiments/9b68a1164e96/test/test_predictions_native.json"),
                (out_dir / "test_predictions_fused.json", "experiments/9b68a1164e96/test/test_predictions_fused.json"),
            ]
            for local_p, remote_p in files_to_upload:
                if local_p.is_file():
                    api.upload_file(
                        path_or_fileobj=str(local_p),
                        path_in_repo=remote_p,
                        repo_id=repo,
                        repo_type="dataset",
                        commit_message=f"Upload {local_p.name} (Academic + Operational benchmarks)",
                    )
                    print(f"  [OK] Uploaded {local_p.name} -> {remote_p}")
            print("[HF Upload] All metrics and raw predictions successfully synced to Hugging Face!")
        except Exception as upload_exc:
            print(f"[HF Upload Warning] Could not upload to HF: {upload_exc}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="HRP4K Evaluation Comparison: P2-Only vs Native vs Fused")
    parser.add_argument("--checkpoint", default="outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt",
                        help="Path to best_p2.pt checkpoint")
    parser.add_argument("--weights", default="outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
                        help="Path to base RT-DETR weights (e.g. fine-tuned checkpoint)")
    parser.add_argument("--data", default="HRP4K", help="Path to HRP4K dataset")
    parser.add_argument("--imgsz", type=int, default=1920, help="Image size (default: 1920)")
    parser.add_argument("--confidence", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--num-images", type=int, default=0, help="Number of test images (0 = all 900 images)")
    parser.add_argument("--device", help="CUDA device index or 'cpu'")
    parser.add_argument("--output", default="outputs/evaluation_comparison", help="Output directory")
    parser.add_argument("--hf-upload", action="store_true", help="Upload comparison metrics to Hugging Face")
    parser.add_argument("--hf-repo", default="Cuong2004/HRP4K", help="Target HF repo")
    parser.add_argument("--hf-token", help="Hugging Face write access token")
    args = parser.parse_args()

    run_comparison(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        weights=args.weights,
        image_size=args.imgsz,
        confidence=args.confidence,
        num_images=args.num_images,
        device=args.device,
        output_dir=args.output,
        hf_upload=args.hf_upload,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
