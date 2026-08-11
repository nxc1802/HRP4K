#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/6] PHASE 0: DATASET INTEGRITY & ANALYSIS ==="
python -m hrp4k_suite analyze --data HRP4K --output outputs/phase0

echo "=== [2/6] DATASET PREPARATION: FULL AVAILABLE SPLIT ==="
python -m hrp4k_suite prepare-dataset \
  --data HRP4K \
  --output outputs/full_dataset \
  --train-limit 2286 \
  --valid-limit 900 \
  --test-limit 900

echo "=== [3/6] PHASE 1: LOCAL-AVAILABLE TRAINING (NOT OFFICIAL REPRODUCTION) ==="
python -m hrp4k_suite train \
  --dataset outputs/full_dataset/dataset.yaml \
  --weights yolo11m.pt \
  --output outputs/runs/yolo11m_full \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --allow-full \
  --allow-incomplete-train

echo "=== [4/6] PHASE 2: RESOLUTION ALLOCATION INFERENCES ==="
mkdir -p outputs/predictions

# Resize 640
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method resize --imgsz 640 \
  --output outputs/predictions/resize_640.json

# In-house sliced inference (not official SAHI)
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method sliced-nms --tile-size 960 --overlap 0.2 \
  --output outputs/predictions/sliced_nms_960.json

# Hand-designed perspective grid (not learned TPP)
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method perspective-grid \
  --output outputs/predictions/perspective_grid.json

echo "=== [5/6] METRICS EVALUATION ==="
python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/resize_640.json \
  --output outputs/predictions/resize_640_metrics.json

python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/sliced_nms_960.json \
  --output outputs/predictions/sliced_nms_960_metrics.json

python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/perspective_grid.json \
  --output outputs/predictions/perspective_grid_metrics.json

echo "=== [6/6] PHASE 3: DEEP DIAGNOSTICS REPORT ==="
python -m hrp4k_suite diagnose \
  --ground-truth outputs/full_dataset/test.json \
  --predictions \
    outputs/predictions/resize_640.json \
    outputs/predictions/sliced_nms_960.json \
    outputs/predictions/perspective_grid.json \
  --output outputs/phase3_report

echo "=== BENCHMARK PIPELINE COMPLETE! Output saved to outputs/phase3_report ==="
