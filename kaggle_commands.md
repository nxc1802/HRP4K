# 🚀 HRP4K Kaggle Minimal CLI Cheatsheet

> **Lưu ý**: Chạy trực tiếp các cell lệnh dưới đây trên **Kaggle Notebooks (GPU P100 / T4)**. Mọi xử lý tìm dataset (ưu tiên Kaggle input path `/kaggle/input/...` $\to$ tự động tạo symlink $\to$ fallback tải từ Hugging Face) đã được tích hợp **100% tự động vào CLI**.

```bash
# 0. Clone Repo mới nhất hoặc Git Pull nếu đã có
![ -d "HRP4K/.git" ] && (cd HRP4K && git pull) || git clone https://github.com/nxc1802/HRP4K.git
%cd HRP4K
```

```bash
# 1. Cài đặt Framework & Dependencies
!pip install -q --upgrade pip
!pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"
!pip install -q -e .
!hrp4k --version
```

```bash
# 1.1 (KHUYÊN DÙNG) Thiết Lập HF_TOKEN trong .env để Tự Động Lưu Checkpoint Ngầm lên Cloud sau mỗi Epoch
# Token lấy tại: https://huggingface.co/settings/tokens (Write access)
!echo "HF_TOKEN=hf_your_write_token_here" > .env
!echo "HF_REPO=Cuong2004/HRP4K" >> .env
```

```bash
# 2. Tự Động Thiết Lập Dataset (Tự nhận diện Kaggle Input hoặc tải từ Hugging Face trong 1 lệnh)
!hrp4k setup-data
```

```bash
# 3. Phase 0: Kiểm Định Dataset & Phân Bố Scale Bins
!hrp4k phase0
```

```bash
# 4. Phase 1: Huấn luyện Baseline (Kích Thước Gốc 4K hoặc 1280, Auto-Sync HF & Conf 0.001)
#
# 📋 Danh sách các options cho parameter --model:
# -----------------------------------------------------------------------------------------
# | Option              | Mô tả kiến trúc           | Pretrained Weights | Batch rec (T4 GPU) |
# |---------------------|---------------------------|--------------------|--------------------|
# | yolo11m             | YOLOv11 Medium            | yolo11m.pt         | 4K: batch=1 / 1280: batch=16 |
# | yolov8m             | YOLOv8 Medium             | yolov8m.pt         | 4K: batch=1 / 1280: batch=16 |
# | yolov5m-compat      | YOLOv5m (Ultralytics)     | yolov5mu.pt        | 4K: batch=1 / 1280: batch=16 |
# | yolov5m-official    | YOLOv5m (Official)        | yolov5m.pt         | 4K: batch=1 / 1280: batch=16 |
# | rt-detr-v1          | RT-DETR-L (Transformer)   | rtdetr-l.pt        | 4K: batch=1 / 1280: batch=8  |
# | rt-detr-v2          | RT-DETR-X (Transformer)   | rtdetr-x.pt        | 4K: batch=1 / 1280: batch=8  |
# | all                 | Chạy TOÀN BỘ 6 models     | (tuần tự)          | 4K: batch=1 / 1280: batch=16 |
# -----------------------------------------------------------------------------------------
# 💡 TÍNH NĂNG TỰ ĐỘNG LƯU CLOUD & BENCHMARK CONFIDENCE:
# - Checkpoints (best.pt, last.pt, results.csv) tự động đồng bộ ngầm lên Hugging Face sau mỗi epoch (qua Background Worker).
# - Mặc định đánh giá --confidence 0.001 bám sát chuẩn COCO/PASCAL và Paper Benchmark.
# - Hỗ trợ cờ --resume để tiếp tục train nếu phiên Kaggle bị ngắt quãng.

# Lựa chọn A (KHUYÊN DÙNG): Huấn luyện YOLO11m với Size Lớn 1280x1280 (--imgsz 1280, batch 16, nhanh & không OOM)
!hrp4k phase1 --model yolo11m --imgsz 1280 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_1280

# Lựa chọn B: Tiếp tục Huấn luyện khi bị ngắt (Continuous Training với --resume)
!hrp4k phase1 --model yolo11m --weights outputs/runs/yolo11m_1280/weights/last.pt --resume --output outputs/runs/yolo11m_1280

# Lựa chọn C: Huấn luyện YOLOv5m với Size Lớn 1280x1280 (--imgsz 1280, batch 16)
!hrp4k phase1 --model yolov5m-compat --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolov5m_1280

# Lựa chọn D: Huấn luyện YOLO11m với KÍCH THƯỚC GỐC 4K (--imgsz original, bắt buộc --batch 1 trên T4 GPU)
!hrp4k phase1 --model yolo11m --imgsz original --batch 1 --epochs 150 --allow-full --output outputs/runs/yolo11m_4k

# Lựa chọn E: Huấn luyện RT-DETRv2 (Transformer SOTA) với kích thước gốc 4K (--batch 1)
!hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 1 --epochs 150 --allow-full --output outputs/runs/rtdetr_v2_4k

# Lựa chọn F: Huấn luyện TẤT CẢ 6 mô hình Baseline tự động (--model all, --imgsz 1280)
!hrp4k phase1 --model all --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/phase1_all_1280
```

```bash
# 5. Phase 2: High-Resolution Inference (Slicing 4K) & Chấm điểm COCO tự động
# Lựa chọn A: Chạy Sliced-NMS (Tile 960x960, overlap 20%)
!hrp4k phase2 --method sliced-nms --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_sliced_nms.json

# Lựa chọn B: Chạy Perspective-Grid (Bám sát dải mặt đường ở xa)
!hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# Lựa chọn C: Chạy SAHI Inference
!hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_1280/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json

# Lựa chọn D: Chạy TẤT CẢ các method Phase 2 cùng lúc
!hrp4k phase2 --method all --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/phase2_benchmark/
```

```bash
# 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi
!hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/metrics/yolo11m_sliced_nms_metrics.json
!hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/diagnostics
```

```bash
# 7. Đóng Gói Hoặc Đẩy Toàn Bộ Checkpoints & Kết Quả Lên Hugging Face Thủ Công (Nếu cần)
!hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token <YOUR_HF_WRITE_TOKEN>
```
