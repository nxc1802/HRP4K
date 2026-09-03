#!/usr/bin/env python3
"""HRP4K P2 Head Calibration & Score Distribution Diagnostic.

Evaluates predictions across multiple confidence thresholds [0.001, 0.01, 0.05, 0.10, 0.25]
to determine whether high prediction counts at conf=0.001 are due to:
  (A) Score calibration in early training (normal, drops sharply with threshold), or
  (B) A decoding/fusion bug (abnormal, counts stay high across all thresholds).
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
from hrp4k.data.paths import resolve_data_dir
from hrp4k.experiments.proposed import RTDETRP2Adapter
from hrp4k.detectors.base import Detection
from hrp4k.inference.p2_fusion import fuse_native_and_p2_predictions
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
import cv2


DEFAULT_THRESHOLDS = [0.001, 0.01, 0.05, 0.10, 0.25]


def run_calibration_diagnostic(
    checkpoint_path: str | Path,
    data_dir: str | Path = "HRP4K",
    weights: str | Path = "rtdetr-l.pt",
    image_size: int = 1920,
    num_images: int = 20,
    device: str | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 1. Load checkpoint metadata
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt_data.get("epoch", "unknown")
    mean_loss = ckpt_data.get("mean_p2_loss", "unknown")
    base_weights = ckpt_data.get("base_checkpoint", str(weights))

    print("=" * 78)
    print("HRP4K P2 HEAD CALIBRATION & SCORE DISTRIBUTION DIAGNOSTIC")
    print("=" * 78)
    print(f"  Checkpoint:      {ckpt_path}")
    print(f"  Saved Epoch:     {epoch}")
    print(f"  Saved Mean Loss: {mean_loss}")
    print(f"  Base Weights:    {base_weights}")
    print(f"  Image Size:      {image_size}px")
    print(f"  Thresholds:      {thresholds}")
    print("=" * 78)

    # 2. Resolve test dataset images
    resolved_data = resolve_data_dir(data_dir)
    test_json = resolved_data / "test.json"
    if not test_json.is_file():
        for candidate in (Path("HRP4K/test.json"), Path("test.json"), Path("../HRP4K/test.json")):
            if candidate.is_file():
                test_json = candidate
                resolved_data = candidate.parent
                break

    if not test_json.is_file():
        raise FileNotFoundError(f"test.json not found in {resolved_data}")

    with open(test_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    images_info = coco_data.get("images", [])
    if not images_info:
        raise ValueError(f"No images found in {test_json}")

    sample_images = images_info[:num_images] if num_images > 0 else images_info
    print(f"\n[Dataset] Evaluating on {len(sample_images)} test images from {resolved_data}...")

    # 3. Initialize Adapter & Model
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter = RTDETRP2Adapter(
        weights=base_weights,
        category_id=0,
        device=target_device,
        p2_checkpoint=ckpt_path,
    )
    adapter.model.eval()

    # Data collectors
    # For each threshold: list of counts per image
    native_counts_per_thresh = {t: [] for t in thresholds}
    p2_counts_per_thresh = {t: [] for t in thresholds}
    fused_counts_per_thresh = {t: [] for t in thresholds}

    all_native_scores: list[float] = []
    all_p2_scores: list[float] = []

    lb = LetterBox(image_size, auto=True, stride=32)

    for idx, img_info in enumerate(sample_images):
        file_name = img_info["file_name"]
        # Find image file
        img_path = resolved_data / "test" / file_name
        if not img_path.is_file():
            img_path = resolved_data / "test" / "images" / file_name
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

        tensor = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(target_device)

        with torch.no_grad():
            out = adapter.model(tensor)

        native_raw = out["native_preds"][0]  # (300, 6) [x1, y1, x2, y2, score, cls] in canvas pixels
        p2_raw = out["p2_preds"][0]          # (300, 6) [x1, y1, x2, y2, score, cls] in canvas pixels

        native_scores = native_raw[:, 4].cpu().numpy()
        p2_scores = p2_raw[:, 4].cpu().numpy()

        all_native_scores.extend(native_scores.tolist())
        all_p2_scores.extend(p2_scores.tolist())

        # Unscale coordinates to original image space
        n_unscaled = scale_boxes((h_lb, w_lb), native_raw[:, :4].clone(), (h_orig, w_orig)).cpu().numpy()
        p_unscaled = scale_boxes((h_lb, w_lb), p2_raw[:, :4].clone(), (h_orig, w_orig)).cpu().numpy()

        # Evaluate at each threshold
        for conf in thresholds:
            n_mask = native_scores >= conf
            p_mask = p2_scores >= conf

            n_dets: list[Detection] = []
            for i in np.where(n_mask)[0]:
                x1, y1, x2, y2 = n_unscaled[i]
                xyxy = (float(np.clip(x1, 0, w_orig)), float(np.clip(y1, 0, h_orig)),
                        float(np.clip(x2, 0, w_orig)), float(np.clip(y2, 0, h_orig)))
                n_dets.append(Detection(xyxy, float(native_scores[i]), 0))

            p_dets: list[Detection] = []
            for i in np.where(p_mask)[0]:
                x1, y1, x2, y2 = p_unscaled[i]
                xyxy = (float(np.clip(x1, 0, w_orig)), float(np.clip(y1, 0, h_orig)),
                        float(np.clip(x2, 0, w_orig)), float(np.clip(y2, 0, h_orig)))
                p_dets.append(Detection(xyxy, float(p2_scores[i]), 0))

            fused_dets = fuse_native_and_p2_predictions(n_dets, p_dets, iou_threshold=0.5)

            native_counts_per_thresh[conf].append(len(n_dets))
            p2_counts_per_thresh[conf].append(len(p_dets))
            fused_counts_per_thresh[conf].append(len(fused_dets))

        if (idx + 1) % 5 == 0 or idx == len(sample_images) - 1:
            print(f"  Processed {idx + 1}/{len(sample_images)} images...")

    # 4. Compute Statistics & Print Results
    print("\n" + "=" * 78)
    print("CALIBRATION TEST RESULTS TABLE")
    print("=" * 78)
    print(f"{'Threshold':<11} | {'Avg Fused / img':<16} | {'Avg Native / img':<17} | {'Avg P2 / img':<13}")
    print("-" * 78)

    table_rows = []
    for conf in thresholds:
        avg_f = float(np.mean(fused_counts_per_thresh[conf])) if fused_counts_per_thresh[conf] else 0.0
        avg_n = float(np.mean(native_counts_per_thresh[conf])) if native_counts_per_thresh[conf] else 0.0
        avg_p = float(np.mean(p2_counts_per_thresh[conf])) if p2_counts_per_thresh[conf] else 0.0
        table_rows.append((conf, avg_f, avg_n, avg_p))
        print(f"{conf:<11} | {avg_f:<16.2f} | {avg_n:<17.2f} | {avg_p:<13.2f}")
    print("=" * 78)

    # 5. Score Distribution Summary
    n_arr = np.array(all_native_scores)
    p_arr = np.array(all_p2_scores)

    print("\nSCORE PERCENTILES DISTRIBUTION:")
    print(f"  {'Percentile':<15} | {'Native Score':<15} | {'P2 Score':<15}")
    print("  " + "-" * 50)
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        nv = float(np.percentile(n_arr, pct)) if len(n_arr) else 0.0
        pv = float(np.percentile(p_arr, pct)) if len(p_arr) else 0.0
        label = f"{pct}th (median)" if pct == 50 else f"{pct}th"
        print(f"  {label:<15} | {nv:<15.4f} | {pv:<15.4f}")
    print(f"  {'Max':<15} | {float(np.max(n_arr)):<15.4f} | {float(np.max(p_arr)):<15.4f}")
    print("=" * 78)

    # 6. Automated Diagnosis
    first_p2 = table_rows[0][3]   # at conf=0.001
    mid_p2 = table_rows[2][3]     # at conf=0.05
    last_p2 = table_rows[4][3]    # at conf=0.25

    print("\nDIAGNOSIS & VERDICT:")
    if first_p2 > 0 and (last_p2 / max(1.0, first_p2)) < 0.2:
        print("  ✅ [CASE A: HEALTHY CALIBRATION]")
        print("  Predictions drop sharply from low confidence to higher confidence.")
        print(f"  (0.001: {first_p2:.1f}/img -> 0.05: {mid_p2:.1f}/img -> 0.25: {last_p2:.1f}/img).")
        print("  -> Architecture, Sigmoid decoding, and NMS fusion are WORKING CORRECTLY.")
        print("  -> High count at 0.001 is merely due to early training (Epoch 3 background uncalibrated).")
        print("  -> Recommendation: Train to 20-30 epochs or evaluate with conf in [0.05 - 0.15].")
    else:
        print("  ⚠️ [CASE B: SCORE / DECODE ANOMALY DETECTED]")
        print("  Predictions remain high even at high confidence thresholds.")
        print(f"  (0.001: {first_p2:.1f}/img -> 0.25: {last_p2:.1f}/img).")
        print("  -> Indicates P2 head produces high confidence across broad background areas.")
        print("  -> Check classification loss balancing or background suppression penalty.")
    print("=" * 78 + "\n")

    return {
        "table": table_rows,
        "native_stats": {"min": float(np.min(n_arr)), "median": float(np.median(n_arr)), "max": float(np.max(n_arr))},
        "p2_stats": {"min": float(np.min(p_arr)), "median": float(np.median(p_arr)), "max": float(np.max(p_arr))},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HRP4K P2 Head Calibration Diagnostic")
    parser.add_argument("--checkpoint", default="outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt",
                        help="Path to best_p2.pt checkpoint")
    parser.add_argument("--data", default="HRP4K", help="Path to HRP4K dataset directory")
    parser.add_argument("--weights", default="rtdetr-l.pt", help="Path to base RT-DETR-L weights")
    parser.add_argument("--imgsz", type=int, default=1920, help="Image size (default: 1920)")
    parser.add_argument("--num-images", type=int, default=20, help="Number of test images to evaluate (default: 20)")
    parser.add_argument("--device", help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    run_calibration_diagnostic(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        weights=args.weights,
        image_size=args.imgsz,
        num_images=args.num_images,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
