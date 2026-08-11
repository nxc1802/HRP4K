#!/usr/bin/env bash
set -e

echo "=== [1/6] PHASE 0: DATASET INTEGRITY & ANALYSIS ==="
python -m hrp4k_suite analyze --data HRP4K --output outputs/phase0

echo "=== [2/6] DATASET PREPARATION: FULL AVAILABLE SPLIT ==="
python -m hrp4k_suite prepare-smoke \
  --data HRP4K \
  --output outputs/full_dataset \
  --train-limit 2286 \
  --valid-limit 900 \
  --test-limit 900

echo "=== [3/6] PHASE 1: DETECTOR BASELINE TRAINING (YOLO11m, 150 EPOCHS) ==="
python -m hrp4k_suite train \
  --dataset outputs/full_dataset/dataset.yaml \
  --weights yolo11m.pt \
  --output outputs/runs/yolo11m_full \
  --epochs 150 \
  --imgsz 640 \
  --batch 16

echo "=== [4/6] PHASE 2: RESOLUTION ALLOCATION INFERENCES ==="
mkdir -p outputs/predictions

# Resize 640
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method resize --imgsz 640 \
  --output outputs/predictions/resize_640.json

# SAHI Slicing
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method sahi --tile-size 960 --overlap 0.2 \
  --output outputs/predictions/sahi_960.json

# Perspective-Bands
python -m hrp4k_suite predict \
  --data outputs/full_dataset --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method perspective-bands \
  --output outputs/predictions/perspective_bands.json

echo "=== [5/6] METRICS EVALUATION ==="
python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/resize_640.json \
  --output outputs/predictions/resize_640_metrics.json

python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/sahi_960.json \
  --output outputs/predictions/sahi_960_metrics.json

echo "=== [6/6] PHASE 3: DEEP DIAGNOSTICS REPORT ==="
python -m hrp4k_suite diagnose \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/*.json \
  --output outputs/phase3_report

echo "=== BENCHMARK PIPELINE COMPLETE! Output saved to outputs/phase3_report ==="
