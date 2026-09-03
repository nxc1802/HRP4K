#!/usr/bin/env python3
"""HRP4K Unified Benchmark Evaluator: Academic (@conf=0.001) & Operational (@conf=0.25).

Evaluates any detector on the 900-image HRP4K test set across BOTH standard protocols:
  1. Benchmark A: Academic / Scientific Detection Protocol (COCO-style, conf=0.001)
     -> mAP50:95, AP50, AP75, Overall Recall, AP_ultra_fine, AP_fine, AP_medium, AP_large
  2. Benchmark B: Operational Deployment Protocol (Standard Operating Condition, conf=0.25)
     -> Precision @0.25, Recall @0.25, F1 @0.25, FP/image (FPPI), FN/image, Latency, FPS
"""

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
import cv2

from hrp4k.data.paths import resolve_data_dir
from hrp4k.evaluation.coco import evaluate


# Official Phase 1 Resolution Baseline Model Registry
PHASE1_MODELS = {
    "yolo11m-640": {
        "weights": "yolo11m_640/weights/best.pt",
        "alt_weights": "checkpoints/yolo11m_640/best.pt",
        "imgsz": 640,
        "family": "yolo",
    },
    "yolo11m-1k": {
        "weights": "outputs/experiments/yolo11m-resolution-1k/weights/best.pt",
        "alt_weights": "yolo11m-resolution-1k/weights/best.pt",
        "imgsz": 960,
        "family": "yolo",
    },
    "yolo11m-2k": {
        "weights": "outputs/experiments/yolo11m-resolution-2k/weights/best.pt",
        "alt_weights": "yolo11m-resolution-2k/weights/best.pt",
        "imgsz": 1920,
        "family": "yolo",
    },
    "yolo11m-4k": {
        "weights": "checkpoints/yolo11m_4k/best.pt",
        "alt_weights": "outputs/experiments/yolo11m_4k/weights/best.pt",
        "imgsz": 3840,
        "family": "yolo",
    },
    "rtdetr-l-640": {
        "weights": "checkpoints/dfine_640/best.pt",
        "alt_weights": "outputs/experiments/dfine_640/weights/best.pt",
        "imgsz": 640,
        "family": "rtdetr",
    },
    "rtdetr-l-1k": {
        "weights": "outputs/experiments/rtdetr-l-resolution-1k/weights/best.pt",
        "alt_weights": "rtdetr-l-resolution-1k/weights/best.pt",
        "imgsz": 960,
        "family": "rtdetr",
    },
    "rtdetr-l-2k": {
        "weights": "outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
        "alt_weights": "rtdetr-l-resolution-2k/weights/best.pt",
        "imgsz": 1920,
        "family": "rtdetr",
    },
    "rtdetr-l-4k": {
        "weights": "checkpoints/dfine_4k/best.pt",
        "alt_weights": "outputs/experiments/dfine_4k/weights/best.pt",
        "imgsz": 3840,
        "family": "rtdetr",
    },
}


def resolve_checkpoint(weights_path: str | Path, hf_repo: str = "Cuong2004/HRP4K") -> Path:
    p = Path(weights_path)
    if p.is_file():
        return p
    # Try downloading from HF Hub if not available locally
    try:
        from huggingface_hub import hf_hub_download
        print(f"  [Hub] Downloading {weights_path} from Hugging Face ({hf_repo})...")
        downloaded = hf_hub_download(repo_id=hf_repo, filename=str(weights_path), repo_type="dataset")
        return Path(downloaded)
    except Exception as exc:
        raise FileNotFoundError(f"Checkpoint not found locally or on Hugging Face: {weights_path} ({exc})")


def evaluate_single_model(
    model_name: str,
    weights_path: str | Path,
    image_size: int,
    data_dir: Path,
    device: str = "0",
    hf_upload: bool = False,
    hf_repo: str = "Cuong2004/HRP4K",
    hf_token: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run dual-mode evaluation (Academic & Operational) for a single model."""
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {model_name} (imgsz={image_size}px, device={device})")
    print(f"Weights: {weights_path}")
    print("=" * 80)

    # Format device
    target_device = device
    if target_device not in {"cpu", "mps"} and not target_device.startswith("cuda:"):
        target_device = f"cuda:{target_device}"

    # Load ground truth
    gt_file = data_dir / "test.json"
    if not gt_file.is_file():
        gt_file = data_dir / "test" / "test.json"
    if not gt_file.is_file():
        raise FileNotFoundError(f"Cannot find test.json in {data_dir}")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt = json.load(f)

    images = gt.get("images", [])
    print(f"Loaded {len(images)} test images ({len(gt.get('annotations', []))} annotations).")

    # Load model via Ultralytics
    weights_resolved = resolve_checkpoint(weights_path, hf_repo=hf_repo)
    w_str = str(weights_resolved).lower()

    if "rtdetr" in w_str or "dfine" in w_str or "d-fine" in w_str:
        from ultralytics import RTDETR
        model = RTDETR(str(weights_resolved))
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights_resolved))

    # Inference loop (gather predictions down to conf=0.001)
    predictions = []
    latencies = []

    print("[Inference] Running test set inference...")
    for idx, im in enumerate(images):
        im_id = int(im["id"])
        file_name = im["file_name"]
        img_path = data_dir / "test" / "images" / file_name
        if not img_path.is_file():
            img_path = data_dir / "test" / file_name
        if not img_path.is_file():
            img_path = data_dir / file_name
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        t0 = time.perf_counter()
        results = model.predict(
            img,
            imgsz=image_size,
            conf=0.001,
            verbose=False,
            device=target_device,
        )[0]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = b
                c_x1 = float(np.clip(min(x1, x2), 0, w_orig))
                c_y1 = float(np.clip(min(y1, y2), 0, h_orig))
                c_x2 = float(np.clip(max(x1, x2), 0, w_orig))
                c_y2 = float(np.clip(max(y1, y2), 0, h_orig))
                w = c_x2 - c_x1
                h = c_y2 - c_y1
                if w >= 1.0 and h >= 1.0:
                    predictions.append({
                        "image_id": im_id,
                        "category_id": 0,
                        "bbox": [c_x1, c_y1, w, h],
                        "score": float(s),
                    })

        if (idx + 1) % 100 == 0 or idx == len(images) - 1:
            print(f"  Processed {idx + 1}/{len(images)} images...")

    n_imgs = max(1, len(images))
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Evaluate BOTH benchmarks
    print("[Metrics] Computing Benchmark A (conf=0.001) & Benchmark B (conf=0.25)...")
    acad = evaluate(gt, predictions, confidence=0.001)
    oper = evaluate(gt, predictions, confidence=0.25)

    fppi = oper.get("fp", 0) / n_imgs
    fn_per_img = oper.get("fn", 0) / n_imgs

    # Print summary
    print("\n" + "-" * 75)
    print(f"RESULT SUMMARY: {model_name}")
    print("-" * 75)
    print(f"  [Benchmark A — Academic @ conf=0.001]")
    print(f"    mAP50:95:             {acad.get('AP50_95', 0)*100:.2f}%")
    print(f"    AP50:                 {acad.get('AP50', 0)*100:.2f}%")
    print(f"    AP75:                 {acad.get('AP75', 0)*100:.2f}%")
    print(f"    Overall Recall:       {acad.get('recall', 0)*100:.2f}%")
    print(f"    AP_ultra_fine:        {acad.get('scale', {}).get('ultra_fine', {}).get('AP50', 0)*100:.2f}%")
    print(f"    AP_fine:              {acad.get('scale', {}).get('fine', {}).get('AP50', 0)*100:.2f}%")
    print(f"    Recall_ultra_fine:    {acad.get('scale', {}).get('ultra_fine', {}).get('recall50', 0)*100:.2f}%")
    print(f"  [Benchmark B — Operational @ conf=0.25]")
    print(f"    Precision @0.25:      {oper.get('precision', 0)*100:.2f}%")
    print(f"    Recall @0.25:         {oper.get('recall', 0)*100:.2f}%")
    print(f"    F1 @0.25:             {oper.get('f1', 0)*100:.2f}%")
    print(f"    FP / image (FPPI):    {fppi:.4f}")
    print(f"    FN / image:           {fn_per_img:.4f}")
    print(f"    Mean Latency:         {mean_lat:.2f} ms")
    print(f"    Inference FPS:        {fps:.2f} FPS")
    print("-" * 75)

    # Output structure
    out = {
        "model_name": model_name,
        "weights": str(weights_resolved),
        "image_size": image_size,
        "mean_latency_ms": mean_lat,
        "fps": fps,
        "benchmark_academic_conf_0_001": acad,
        "benchmark_operational_conf_0_25": {**oper, "fppi": fppi, "fn_per_img": fn_per_img},
    }

    if output_dir is None:
        output_dir = Path("outputs/benchmark_results") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "test_metrics_dual.json"
    metrics_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (output_dir / "test_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Saved evaluation outputs to: {metrics_file}")

    if hf_upload:
        try:
            from huggingface_hub import HfApi
            token = hf_token or os.environ.get("HF_TOKEN")
            api = HfApi(token=token)
            remote_p = f"benchmark_results/{model_name}/test_metrics_dual.json"
            api.upload_file(
                path_or_fileobj=str(metrics_file),
                path_in_repo=remote_p,
                repo_id=hf_repo,
                repo_type="dataset",
                commit_message=f"Add {model_name} dual academic & operational benchmark metrics",
            )
            print(f"  [HF Upload OK] -> {remote_p}")
        except Exception as e:
            print(f"  [HF Upload Warning] {e}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="HRP4K Unified Benchmark Evaluator (Academic & Operational)")
    parser.add_argument("--model", help="Model name or preset (e.g. yolo11m-2k, rtdetr-l-2k)")
    parser.add_argument("--weights", help="Path to model weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=1920, help="Evaluation image size (e.g. 1920, 3840)")
    parser.add_argument("--data", default="HRP4K", help="Path to HRP4K dataset")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--all-phase1", action="store_true", help="Evaluate all 8 Phase 1 resolution baselines sequentially")
    parser.add_argument("--hf-upload", action="store_true", help="Upload results to Hugging Face Hub")
    parser.add_argument("--hf-repo", default="Cuong2004/HRP4K", help="Hugging Face repo ID")
    parser.add_argument("--hf-token", help="Hugging Face access token")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data)

    if args.all_phase1:
        print("\n" + "=" * 80)
        print("RUNNING ALL 8 PHASE 1 RESOLUTION BASELINE BENCHMARKS")
        print("=" * 80)
        all_results = {}
        for m_name, m_cfg in PHASE1_MODELS.items():
            w = m_cfg["weights"]
            if not Path(w).is_file() and Path(m_cfg["alt_weights"]).is_file():
                w = m_cfg["alt_weights"]
            try:
                res = evaluate_single_model(
                    model_name=m_name,
                    weights_path=w,
                    image_size=m_cfg["imgsz"],
                    data_dir=data_dir,
                    device=args.device,
                    hf_upload=args.hf_upload,
                    hf_repo=args.hf_repo,
                    hf_token=args.hf_token,
                )
                all_results[m_name] = res
            except Exception as exc:
                print(f"[ERROR] Failed evaluating {m_name}: {exc}")

        # Save combined summary
        summary_p = Path("outputs/benchmark_results/phase1_resolution_summary.json")
        summary_p.parent.mkdir(parents=True, exist_ok=True)
        summary_p.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n[ALL COMPLETE] Phase 1 Summary saved to: {summary_p}")
        return 0

    if args.model and args.model in PHASE1_MODELS:
        cfg = PHASE1_MODELS[args.model]
        w = args.weights or cfg["weights"]
        imgsz = args.imgsz if args.imgsz != 1920 else cfg["imgsz"]
        evaluate_single_model(
            model_name=args.model,
            weights_path=w,
            image_size=imgsz,
            data_dir=data_dir,
            device=args.device,
            hf_upload=args.hf_upload,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
        )
        return 0

    if args.weights:
        name = args.model or Path(args.weights).stem
        evaluate_single_model(
            model_name=name,
            weights_path=args.weights,
            image_size=args.imgsz,
            data_dir=data_dir,
            device=args.device,
            hf_upload=args.hf_upload,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
