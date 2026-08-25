# HRP4K Benchmark: Local-Available Pipeline Guide

Tài liệu này mô tả pipeline Phase 0–3 trên **single official downloaded release**. Release hiện có 2.286 file train khả dụng; thiếu hụt bắt nguồn từ nguồn phát hành và nhóm sẽ liên hệ tác giả để xin archive đầy đủ. Smoke output vẫn chỉ kiểm tra plumbing, không phải kết quả paper.

---

## Cấu hình paper-aligned được project cố định

Paper công bố model medium và 150 epoch nhưng không đủ thông tin để tái tạo bit-for-bit toàn bộ optimizer/preprocessing/software stack. Các giá trị dưới đây là resolved protocol của project và phải được lưu cùng run.

| Siêu tham số | Giá trị project | Mô tả |
| :--- | :--- | :--- |
| **Model Variant** | `yolo11m.pt` (Medium - 20.1M params) | Biến thể Medium được bài báo lựa chọn đánh giá chuẩn cho tất cả 6 dòng mô hình. |
| **Epochs** | `150` epochs | Số epoch huấn luyện đồng nhất cho tất cả các mô hình. |
| **Canvas Size (`imgsz`)** | `640 × 640` px | Độ phân giải canvas đầu vào chuẩn cho detector. |
| **Batch Size** | `16` | Batch size tiêu chuẩn cho mỗi GPU L40S. |
| **Optimizer** | `SGD` (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`) | Được truyền explicit để tránh Ultralytics `optimizer=auto`; đây là project protocol. |
| **Warmup** | `3.0` epochs | `warmup_momentum=0.8`, `warmup_bias_lr=0.1`. |
| **Augmentation** | `mosaic=1.0`, `mixup=0.0`, `translate=0.1`, `scale=0.5`, `fliplr=0.5` | Resolved project configuration. |
| **Precision** | AMP FP16 / BF16 | Tự động bật mixed precision trên GPU L40S. |

---

## 1. Yêu cầu môi trường

Kích hoạt môi trường Python virtualenv trước khi chạy:

```bash
source venv/bin/activate
```

---

## 2. Kịch bản thực thi CLI từ A đến Z (`hrp4k`)

### Bước 1: Thiết lập Dữ liệu (Hugging Face) & Kiểm tra Tính Toàn vẹn (Phase 0)
Tải và chuẩn hóa dataset đầy đủ $6.003$ ảnh ($11.92\text{ GB}$) từ Hugging Face [`Cuong2004/HRP4K`](https://huggingface.co/datasets/Cuong2004/HRP4K) và thực hiện phân tích Scale Bins:

```bash
# 1. Tự động kiểm tra / tải dữ liệu từ Hugging Face
hrp4k setup-data --data HRP4K

# 2. Phase 0: Kiểm định toàn vẹn nhãn và phân tích 4 dải Scale Bins
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12
```

---

### Bước 2: Phase 1 — Huấn Luyện Detector Baseline (Native 4K & Canvas 640)
Huấn luyện các mô hình Baseline trên GPU Server (NVIDIA RTX PRO 6000 95GB VRAM) với $150$ Epochs:

```bash
# Lựa chọn A (KHUYÊN DÙNG): Huấn luyện YOLO11m Native 4K UHD (batch 16, --rect)
hrp4k phase1 --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# Lựa chọn B: Huấn luyện Baseline chuẩn 640x640
hrp4k phase1 --model yolo11m --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_640
```

---

### Bước 3: Phase 2 — Đánh Giá Các Phương Pháp Phân Bổ Độ Phân Giải (Resolution Allocation)
Thực hiện suy luận trên toàn bộ **$900$ ảnh Test độc lập** và xuất bảng Benchmark so sánh:

```bash
# Chạy đồng thời toàn bộ 5 phương pháp Phase 2 (SAHI, Perspective-Grid, Sliced-NMS, ZoomDet, Resize)
hrp4k phase2 --method all --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/phase2_benchmark/
```

Hoặc chạy từng phương pháp riêng lẻ:
```bash
# Perspective-Grid (Khuyên dùng, 9 calls)
hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# SAHI Slicing (15 calls)
hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_4k/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json
```

---

### Bước 4: Phase 3 — Chẩn Đoán Lỗi & Phân Tích Chuyên Sâu (Deep Diagnostics)
Đánh giá sai số phân loại, FPPI trên $300$ ảnh âm tính và phân rã 4 dải kích thước:

```bash
# Đánh giá độ chính xác chi tiết
hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/phase2_benchmark/best_perspective-grid_test_predictions.json --output outputs/metrics/perspective_grid_metrics.json

# Chẩn đoán phân loại lỗi (Localization, Background FP, Classification confusion)
hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/phase2_benchmark/best_perspective-grid_test_predictions.json --output outputs/diagnostics
```

---

## 3. Bash Script tự động hoá một lần gõ (`run_full_pipeline.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Dataset & Phase 0
hrp4k setup-data --data HRP4K
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12

# 2. Phase 1 Training (Native 4K)
hrp4k phase1 --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# 3. Phase 2 Multi-Method Inference Benchmark
hrp4k phase2 --method all --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/phase2_benchmark/

# 4. Phase 3 Diagnostics
hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/phase2_benchmark/best_perspective-grid_test_predictions.json --output outputs/diagnostics

echo "=== BENCHMARK PIPELINE COMPLETE! Kết quả lưu tại outputs/ ==="
```

