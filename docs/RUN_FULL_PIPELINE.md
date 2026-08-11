# HRP4K Benchmark: Full Pipeline Execution Guide (A to Z)

Tài liệu này cung cấp kịch bản thực thi hoàn chỉnh **chạy thật (Full Official Benchmark Run)** cho toàn bộ pipeline Phase 0–3 trên tập dữ liệu HRP4K. Các tham số được thiết lập chuẩn 100% theo bài báo khoa học chính thức HRP4K (*Scientific Data / Nature*, [s41597-026-07317-w.md](file:///Volumes/WorkSpace/Project/HRP4K/docs/s41597-026-07317-w.md)).

---

## 💡 Siêu tham số chuẩn theo Paper công bố (Table 2 & Technical Validation)

| Siêu tham số | Giá trị chuẩn trong Paper | Mô tả |
| :--- | :--- | :--- |
| **Model Variant** | `yolo11m.pt` (Medium - 20.1M params) | Biến thể Medium được bài báo lựa chọn đánh giá chuẩn cho tất cả 6 dòng mô hình. |
| **Epochs** | `150` epochs | Số epoch huấn luyện đồng nhất cho tất cả các mô hình. |
| **Canvas Size (`imgsz`)** | `640 × 640` px | Độ phân giải canvas đầu vào chuẩn cho detector. |
| **Batch Size** | `16` | Batch size tiêu chuẩn cho mỗi GPU L40S. |
| **Optimizer** | `SGD` (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`) | Siêu tham số tối ưu hóa chính thức của Ultralytics. |
| **Warmup** | `3.0` epochs | `warmup_momentum=0.8`, `warmup_bias_lr=0.1`. |
| **Augmentation** | `mosaic=1.0`, `mixup=0.0`, `translate=0.1`, `scale=0.5`, `fliplr=0.5` | Augmentation chuẩn theo công bố bài báo. |
| **Precision** | AMP FP16 / BF16 | Tự động bật mixed precision trên GPU L40S. |

---

## 1. Yêu cầu môi trường

Kích hoạt môi trường Python virtualenv trước khi chạy:

```bash
source venv/bin/activate
```

---

## 2. Kịch bản thực thi CLI từ A đến Z (Đã khớp 100% Hyperparameters Paper)

### Bước 1: Phase 0 — Phân tích toàn vẹn dữ liệu & Thống kê gốc
Kiểm tra số lượng ảnh khả dụng thực tế trên ổ cứng (2.286 train, 900 valid, 900 test), phát hiện lỗi bounding box và thống kê phân bố diện tích (`ultra_fine`, `fine`, `medium`, `large`):

```bash
python -m hrp4k_suite analyze \
  --data HRP4K \
  --output outputs/phase0
```

---

### Bước 2: Chuẩn bị Full Available Dataset (2.286 Train / 900 Valid / 900 Test)
Tạo cấu trúc YOLO/COCO dataset cho toàn bộ ảnh thực tế khả dụng. Sử dụng symlink để không nhân bản ảnh 4K gốc:

```bash
python -m hrp4k_suite prepare-smoke \
  --data HRP4K \
  --output outputs/full_dataset \
  --train-limit 2286 \
  --valid-limit 900 \
  --test-limit 900
```

---

### Bước 3: Phase 1 — Detector Baseline Training (YOLO11m Medium, 150 Epochs)
Huấn luyện mô hình YOLO11m (Medium variant - 20.1M params) trên canvas 640×640 với AMP FP16/BF16 đúng **150 epochs** và toàn bộ hyperparameter của paper:

```bash
python -m hrp4k_suite train \
  --dataset outputs/full_dataset/dataset.yaml \
  --weights yolo11m.pt \
  --output outputs/runs/yolo11m_full \
  --epochs 150 \
  --imgsz 640 \
  --batch 16
```

---

### Bước 4: Phase 2 — Inference theo các phương pháp phân bổ độ phân giải (Resolution Allocation)
Thực hiện suy luận trên toàn bộ **900 ảnh tập test** cho từng chiến lược và xuất kết quả chuẩn COCO JSON:

#### A. Resize Baseline (Canvas 640 - Phương pháp chuẩn Phase 1)
```bash
python -m hrp4k_suite predict \
  --data outputs/full_dataset \
  --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method resize \
  --imgsz 640 \
  --output outputs/predictions/resize_640.json
```

#### B. Uniform 2×2 Tiling
```bash
python -m hrp4k_suite predict \
  --data outputs/full_dataset \
  --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method uniform-2 \
  --imgsz 640 \
  --output outputs/predictions/uniform_2x2.json
```

#### C. SAHI (Sliced Inference với Tile 960, Overlap 0.2 & Global NMS)
```bash
python -m hrp4k_suite predict \
  --data outputs/full_dataset \
  --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method sahi \
  --tile-size 960 \
  --overlap 0.2 \
  --output outputs/predictions/sahi_960.json
```

#### D. Perspective-Bands (Geometry Baseline)
```bash
python -m hrp4k_suite predict \
  --data outputs/full_dataset \
  --split test \
  --weights outputs/runs/yolo11m_full/weights/best.pt \
  --method perspective-bands \
  --output outputs/predictions/perspective_bands.json
```

---

### Bước 5: Đánh giá chỉ số hợp nhất (Unified Metrics Evaluation)
Tính toán `AP50`, `AP75`, `mAP[50:95]`, `Precision`, `Recall`, `F1`, `FPPI` và phân rã chỉ số theo từng dải kích thước object:

```bash
# Đánh giá Resize Baseline
python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/resize_640.json \
  --output outputs/predictions/resize_640_metrics.json

# Đánh giá SAHI Slicing
python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/sahi_960.json \
  --output outputs/predictions/sahi_960_metrics.json
```

---

### Bước 6: Phase 3 — Xuất Báo cáo Chẩn đoán lỗi Chuyên sâu (Deep Diagnostics)
Đọc toàn bộ kết quả prediction đã lưu để tổng hợp bảng so sánh, biểu đồ phân bố lỗi và status matrix mà không cần chạy lại suy luận:

```bash
python -m hrp4k_suite diagnose \
  --ground-truth outputs/full_dataset/test.json \
  --predictions outputs/predictions/*.json \
  --output outputs/phase3_report
```

---

## 3. Bash Script tự động hoá một lần gõ (`run_full_pipeline.sh`)

```bash
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
```
