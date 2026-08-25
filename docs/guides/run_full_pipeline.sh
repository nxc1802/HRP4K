#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/5] SETUP DATASET (HUGGING FACE) ==="
hrp4k setup-data --data HRP4K

echo "=== [2/5] PHASE 0: DATASET AUDIT & INTEGRITY ==="
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12

echo "=== [3/5] PHASE 1: NATIVE 4K / BASELINE TRAINING ==="
# Huấn luyện YOLO11m trên 4K gốc (hoặc imgsz 640)
hrp4k phase1 \
  --model yolo11m \
  --imgsz original \
  --batch 16 \
  --epochs 150 \
  --allow-full \
  --confidence 0.001 \
  --rect \
  --output outputs/runs/yolo11m_4k

echo "=== [4/5] PHASE 2: HIGH-RESOLUTION INFERENCE BENCHMARK ==="
# Chạy toàn bộ các phương pháp Phase 2 (SAHI, Perspective-Grid, Sliced-NMS, ZoomDet, Resize)
hrp4k phase2 \
  --method all \
  --weights outputs/runs/yolo11m_4k/weights/best.pt \
  --output outputs/phase2_benchmark/

echo "=== [5/5] PHASE 3: DEEP DIAGNOSTICS REPORT ==="
hrp4k diagnose \
  --ground-truth HRP4K/test.json \
  --predictions outputs/phase2_benchmark/best_perspective-grid_test_predictions.json \
  --output outputs/diagnostics

echo "=== BENCHMARK PIPELINE COMPLETE! Kết quả lưu tại outputs/ ==="

