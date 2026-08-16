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
# 2. Tự Động Thiết Lập Dataset (Tự nhận diện Kaggle Input hoặc tải từ Hugging Face trong 1 lệnh)
!hrp4k setup-data
```

```bash
# 3. Phase 0: Kiểm Định Dataset & Phân Bố Scale Bins
!hrp4k phase0
```

```bash
# 4. Phase 1: Huấn luyện Baseline (Kích Thước Gốc 4K hoặc 1280)
#
# 📋 Danh sách các options cho parameter --model:
# -----------------------------------------------------------------------------------------
# | Option              | Mô tả kiến trúc           | Pretrained Weights | Batch rec (4K) |
# |---------------------|---------------------------|--------------------|----------------|
# | yolo11m             | YOLOv11 Medium            | yolo11m.pt         | batch=2-4      |
# | yolov8m             | YOLOv8 Medium             | yolov8m.pt         | batch=2-4      |
# | yolov5m-compat      | YOLOv5m (Ultralytics)     | yolov5mu.pt        | batch=2-4      |
# | yolov5m-official    | YOLOv5m (Official)        | yolov5m.pt         | batch=2-4      |
# | rt-detr-v1          | RT-DETR-L (Transformer)   | rtdetr-l.pt        | batch=2        |
# | rt-detr-v2          | RT-DETR-X (Transformer)   | rtdetr-x.pt        | batch=2        |
# | all                 | Chạy TOÀN BỘ 6 models     | (tuần tự)          | batch=2        |
# -----------------------------------------------------------------------------------------

# Lựa chọn A: Huấn luyện YOLO11m với KÍCH THƯỚC GỐC 4K 3840x2160 (--imgsz original, batch 2-4 trên Kaggle GPU)
!hrp4k phase1 --model yolo11m --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/runs/yolo11m_4k

# Lựa chọn B: Huấn luyện YOLOv8m với Kích Thước Gốc 4K
!hrp4k phase1 --model yolov8m --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/runs/yolov8m_4k

# Lựa chọn C: Huấn luyện RT-DETRv2 (Transformer SOTA) với kích thước gốc 4K
!hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/runs/rtdetr_v2_4k

# Lựa chọn D: Huấn luyện YOLO11m với Size 1280x1280 (Nhanh hơn, batch 16)
!hrp4k phase1 --model yolo11m --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolo11m_1280

# Lựa chọn E: Huấn luyện TẤT CẢ 6 mô hình Baseline tự động (--model all)
!hrp4k phase1 --model all --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/phase1_all_4k
```

```bash
# 5. Phase 2: High-Resolution Inference (Slicing 4K) & Chấm điểm COCO tự động

# Lựa chọn A: Chạy Sliced-NMS (Tile 960x960, overlap 20%)
!hrp4k phase2 --method sliced-nms --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_sliced_nms.json

# Lựa chọn B: Chạy Perspective-Grid (Bám sát dải mặt đường ở xa)
!hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# Lựa chọn C: Chạy SAHI Inference
!hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_4k/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json

# Lựa chọn D: Chạy TẤT CẢ các method Phase 2 cùng lúc
!hrp4k phase2 --method all --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/phase2_benchmark/
```

```bash
# 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi
!hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/metrics/yolo11m_sliced_nms_metrics.json
!hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/diagnostics
```

```bash
# 7. Đóng Gói Hoặc Đẩy Toàn Bộ Checkpoints & Kết Quả Lên Hugging Face
!tar -czvf hrp4k_kaggle_results.tar.gz outputs/
!hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token <YOUR_HF_WRITE_TOKEN>
```
