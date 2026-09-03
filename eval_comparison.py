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

        t = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(target_device)

        with torch.no_grad():
            out = adapter.model(t)

        native_raw = out["native_preds"][0]  # (300, 6)
        p2_raw = out["p2_preds"][0]          # (300, 6)

        # Convert to detections
        def get_dets(preds: torch.Tensor, is_native: bool = False) -> list[Detection]:
            m = preds[:, 4] >= confidence
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
                b = (float(np.clip(x1, 0, w_orig)), float(np.clip(y1, 0, h_orig)),
                     float(np.clip(x2, 0, w_orig)), float(np.clip(y2, 0, h_orig)))
                dets.append(Detection(b, float(sc[i]), 0))
            return dets

        p2_dets = get_dets(p2_raw, is_native=False)
        nat_dets = get_dets(native_raw, is_native=True)
        fused_dets = fuse_native_and_p2_predictions(nat_dets, p2_dets, iou_threshold=0.5)

        for d in p2_dets:
            p2_predictions.append({
                "image_id": im_id, "category_id": 0,
                "bbox": [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]],
                "score": d.score,
            })
        for d in nat_dets:
            native_predictions.append({
                "image_id": im_id, "category_id": 0,
                "bbox": [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]],
                "score": d.score,
            })
        for d in fused_dets:
            fused_predictions.append({
                "image_id": im_id, "category_id": 0,
                "bbox": [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]],
                "score": d.score,
            })

        if (idx + 1) % 50 == 0 or idx == len(selected_images) - 1:
            print(f"  Processed {idx + 1}/{len(selected_images)} images...")

    # 3. Compute Metrics with Official Protocol & Scale Decomposition
    print("\n[Metrics] Computing full COCO evaluation with scale decomposition...")
    from hrp4k.evaluation.coco import evaluate

    p2_eval = evaluate(sub_gt, p2_predictions, confidence=confidence)
    nat_eval = evaluate(sub_gt, native_predictions, confidence=confidence)
    fused_eval = evaluate(sub_gt, fused_predictions, confidence=confidence)

    n_imgs = max(1, len(selected_images))
    p2_avg_dets = len(p2_predictions) / n_imgs
    nat_avg_dets = len(native_predictions) / n_imgs
    fused_avg_dets = len(fused_predictions) / n_imgs

    # 4. Print Comparison Tables
    print("\n" + "=" * 85)
    print("TABLE 1: OVERALL BENCHMARK COMPARISON (P2-ONLY vs NATIVE vs FUSED)")
    print("=" * 85)
    print(f"{'Metric':<25} | {'P2-Only Head':<16} | {'Native RT-DETR':<16} | {'Fused (Native+P2)':<18}")
    print("-" * 85)
    print(f"{'AP50':<25} | {p2_eval.get('AP50', 0)*100:<15.2f}% | {nat_eval.get('AP50', 0)*100:<15.2f}% | {fused_eval.get('AP50', 0)*100:<17.2f}%")
    print(f"{'AP50-95':<25} | {p2_eval.get('AP50_95', 0)*100:<15.2f}% | {nat_eval.get('AP50_95', 0)*100:<15.2f}% | {fused_eval.get('AP50_95', 0)*100:<17.2f}%")
    print(f"{'AP75':<25} | {p2_eval.get('AP75', 0)*100:<15.2f}% | {nat_eval.get('AP75', 0)*100:<15.2f}% | {fused_eval.get('AP75', 0)*100:<17.2f}%")
    print(f"{'Overall Recall':<25} | {p2_eval.get('recall', 0)*100:<15.2f}% | {nat_eval.get('recall', 0)*100:<15.2f}% | {fused_eval.get('recall', 0)*100:<17.2f}%")
    print(f"{'True Positives (TP)':<25} | {p2_eval.get('tp', 0):<16} | {nat_eval.get('tp', 0):<16} | {fused_eval.get('tp', 0):<18}")
    print(f"{'Avg Predictions / img':<25} | {p2_avg_dets:<16.2f} | {nat_avg_dets:<16.2f} | {fused_avg_dets:<18.2f}")
    print(f"{'Total Predictions':<25} | {len(p2_predictions):<16} | {len(native_predictions):<16} | {len(fused_predictions):<18}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("TABLE 2: SCALE BREAKDOWN COMPARISON (RECALL & AP50 BY POTHOLE SIZE)")
    print("=" * 85)
    print(f"{'Scale Category':<25} | {'P2-Only Recall':<16} | {'Native Recall':<16} | {'Fused Recall':<18}")
    print("-" * 85)
    for sc in ["ultra_fine", "fine", "medium", "large"]:
        p2_rec = p2_eval.get("scale", {}).get(sc, {}).get("recall50", 0.0) * 100
        nat_rec = nat_eval.get("scale", {}).get(sc, {}).get("recall50", 0.0) * 100
        fused_rec = fused_eval.get("scale", {}).get(sc, {}).get("recall50", 0.0) * 100
        num_pos = p2_eval.get("scale", {}).get(sc, {}).get("positives", 0)
        label = f"{sc.capitalize()} ({num_pos})"
        print(f"{label:<25} | {p2_rec:<15.2f}% | {nat_rec:<15.2f}% | {fused_rec:<17.2f}%")
    print("-" * 85)
    print(f"{'Scale Category':<25} | {'P2-Only AP50':<16} | {'Native AP50':<16} | {'Fused AP50':<18}")
    print("-" * 85)
    for sc in ["ultra_fine", "fine", "medium", "large"]:
        p2_ap = p2_eval.get("scale", {}).get(sc, {}).get("AP50", 0.0) * 100
        nat_ap = nat_eval.get("scale", {}).get(sc, {}).get("AP50", 0.0) * 100
        fused_ap = fused_eval.get("scale", {}).get(sc, {}).get("AP50", 0.0) * 100
        num_pos = p2_eval.get("scale", {}).get(sc, {}).get("positives", 0)
        label = f"{sc.capitalize()} ({num_pos})"
        print(f"{label:<25} | {p2_ap:<15.2f}% | {nat_ap:<15.2f}% | {fused_ap:<17.2f}%")
    print("=" * 85)

    # 5. Save results locally
    results = {
        "p2_only": {**p2_eval, "avg_predictions_per_image": p2_avg_dets, "total_predictions": len(p2_predictions)},
        "native_only": {**nat_eval, "avg_predictions_per_image": nat_avg_dets, "total_predictions": len(native_predictions)},
        "fused": {**fused_eval, "avg_predictions_per_image": fused_avg_dets, "total_predictions": len(fused_predictions)},
        "settings": {
            "checkpoint": str(ckpt_path),
            "weights": str(weights),
            "image_size": image_size,
            "confidence": confidence,
            "num_images": len(selected_images),
        },
    }
    comp_path = out_dir / "test_metrics_comparison.json"
    comp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "test_metrics_p2_only.json").write_text(json.dumps(p2_eval, indent=2), encoding="utf-8")
    (out_dir / "test_metrics_native_only.json").write_text(json.dumps(nat_eval, indent=2), encoding="utf-8")
    (out_dir / "test_metrics_fused.json").write_text(json.dumps(fused_eval, indent=2), encoding="utf-8")
    print(f"\nSaved comparison results to: {comp_path}")

    # 6. Upload to Hugging Face if requested
    if hf_upload:
        try:
            from huggingface_hub import HfApi
            token = hf_token or os.environ.get("HF_TOKEN")
            repo = hf_repo or "Cuong2004/HRP4K"
            print(f"\n[HF Upload] Uploading comparison metrics to Hugging Face repo: {repo}...")
            api = HfApi(token=token)
            files_to_upload = [
                (comp_path, "experiments/9b68a1164e96/test/test_metrics_comparison.json"),
                (out_dir / "test_metrics_p2_only.json", "experiments/9b68a1164e96/test/test_metrics_p2_only.json"),
                (out_dir / "test_metrics_native_only.json", "experiments/9b68a1164e96/test/test_metrics_native_only.json"),
                (out_dir / "test_metrics_fused.json", "experiments/9b68a1164e96/test/test_metrics_fused.json"),
            ]
            for local_p, remote_p in files_to_upload:
                if local_p.is_file():
                    api.upload_file(
                        path_or_fileobj=str(local_p),
                        path_in_repo=remote_p,
                        repo_id=repo,
                        repo_type="dataset",
                        commit_message=f"Add {local_p.name} comparative evaluation",
                    )
                    print(f"  [OK] Uploaded {local_p.name} -> {remote_p}")
            print("[HF Upload] All comparison metrics successfully synced to Hugging Face!")
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
