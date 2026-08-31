#!/usr/bin/env python3
"""Run Warped ZoomDet 640 Evaluation on HRP4K 900 Test Images.

Loads D-FINE 640 checkpoint, executes 1-pass perspective continuous grid warping,
unwarps bounding boxes to native 4K UHD coordinates, and evaluates official COCO metrics.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hrp4k.methods.zoomdet import make_zoomdet_view


def main():
    test_json_path = "HRP4K/test.json"
    image_dir = "HRP4K/test/images"
    weights_path = "checkpoints/dfine_640/best.pt"
    output_pred_path = "outputs/predictions/dfine_zoomdet_geometry.json"
    output_metrics_path = "outputs/predictions/dfine_zoomdet_geometry_metrics.json"

    print("=== HRP4K OFFICIAL WARPED ZOOMDET 640 EVALUATION ===")
    print(f"Loading Test Annotations: {test_json_path}")
    with open(test_json_path) as f:
        coco_gt_dict = json.load(f)

    test_images = coco_gt_dict["images"]
    print(f"Total Test Images: {len(test_images)}")

    print(f"Loading Model: {weights_path}")
    model = RTDETR(weights_path)

    all_predictions = []
    latencies = []

    print("Starting Inference on 900 Test Images...")
    t_start = time.time()

    for idx, img_info in enumerate(test_images):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(image_dir, fname)

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        t0 = time.perf_counter()

        # 1. Perspective Warp to 640x640
        view = make_zoomdet_view(img_np, canvas_size=640, mode="geometry", horizon_ratio=0.40)

        # 2. Forward Predict on Warped Canvas
        res = model.predict(view.image, imgsz=640, conf=0.001, verbose=False)[0]

        # 3. Unwarp BBoxes to 4K Space
        image_preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            for xyxy, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                box_4k = view.map_box(tuple(xyxy))
                image_preds.append({
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [round(float(c), 4) for c in box_4k],
                    "score": round(float(score), 6)
                })

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        all_predictions.extend(image_preds)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
            elapsed = time.time() - t_start
            avg_lat = np.mean(latencies)
            print(f"[{idx+1:3d}/{len(test_images)}] Elapsed: {elapsed:.1f}s | Avg Latency: {avg_lat:.2f} ms/img | Predictions: {len(all_predictions)}")

    # Save Predictions Document
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "experiment_id": "dfine_zoomdet640_official",
        "method": "zoomdet-geometry",
        "dataset": "HRP4K",
        "detector": "dfine_640",
        "method_config": {
            "canvas_size": 640,
            "mode": "geometry",
            "horizon_ratio": 0.40,
            "road_expansion": 1.75
        },
        "runtime": {
            "framework": "ultralytics/rtdetr",
            "weights": weights_path,
            "num_images": len(test_images),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2)
        },
        "predictions": all_predictions
    }

    with open(output_pred_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved Predictions to: {output_pred_path} ({len(all_predictions)} predictions)")

    # Run Official COCO Evaluation
    print("\nRunning Official pycocotools COCOeval...")
    coco_gt = COCO(test_json_path)

    # Temporary pred file for pycocotools
    coco_dt = coco_gt.loadRes(all_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Scale breakdown
    # COCO area: < 32^2 (small), 32^2-96^2 (medium), > 96^2 (large)
    metrics_summary = {
        "mAP_50": round(float(coco_eval.stats[1] * 100), 2),
        "mAP_75": round(float(coco_eval.stats[2] * 100), 2),
        "mAP_50_95": round(float(coco_eval.stats[0] * 100), 2),
        "mAP_small": round(float(coco_eval.stats[3] * 100), 2),
        "mAP_medium": round(float(coco_eval.stats[4] * 100), 2),
        "mAP_large": round(float(coco_eval.stats[5] * 100), 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "total_test_images": len(test_images),
        "total_predictions": len(all_predictions)
    }

    # FPPI on 300 Negative Images
    # Positive images: 600, Negative images: 300
    gt_img_with_ann = set(a["image_id"] for a in coco_gt_dict["annotations"])
    neg_img_ids = set(img["id"] for img in test_images if img["id"] not in gt_img_with_ann)
    
    fp_neg_count = sum(1 for p in all_predictions if p["image_id"] in neg_img_ids and p["score"] >= 0.25)
    fppi_neg = fp_neg_count / max(1, len(neg_img_ids))
    metrics_summary["FPPI_negative"] = round(float(fppi_neg), 4)
    metrics_summary["num_negative_images"] = len(neg_img_ids)
    metrics_summary["fp_negative_count"] = fp_neg_count

    with open(output_metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved Official Metrics to: {output_metrics_path}")
    print("Metrics Summary:")
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
