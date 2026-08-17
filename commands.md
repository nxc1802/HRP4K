# 🚀 HRP4K Benchmark Execution Commands

Tài liệu tổng hợp toàn bộ các lệnh chạy thực thi quy trình nghiên cứu, huấn luyện baseline và đánh giá độ phân giải cao cho tập dữ liệu **HRP4K (Phase 0 → Phase 3)**.

Tài liệu được chia thành **2 phân loại môi trường riêng biệt**:
- 🟦 **Loại 1: Kaggle Notebooks** (Dành cho giao diện Jupyter/Kaggle Notebooks có GPU P100 / T4, sử dụng tiền tố `!` và `%cd`).
- 🟩 **Loại 2: Native CLI / Terminal** (Dành cho Terminal Local, Cloud VM, GPU Server chạy Linux/macOS bằng shell bash thuần túy).

---

## 📋 Danh Sách & So Sánh Các Mô Hình Baseline (`--model`)

| Tùy chọn `--model` | Kiến trúc mô hình | Pretrained Weights | Khuyến nghị Batch Size (T4/P100 GPU) | Ghi chú hiệu năng |
| :--- | :--- | :--- | :--- | :--- |
| **`yolo11m`** *(Khuyên dùng)* | YOLOv11 Medium | `yolo11m.pt` | **4K:** `batch=1` \| **1280:** `batch=16` | SOTA mAP, hội tụ nhanh, tối ưu VRAM |
| **`yolov8m`** | YOLOv8 Medium | `yolov8m.pt` | **4K:** `batch=1` \| **1280:** `batch=16` | Baseline tiêu chuẩn cộng đồng |
| **`yolov5m-compat`** | YOLOv5m (Ultralytics) | `yolov5mu.pt` | **4K:** `batch=1` \| **1280:** `batch=16` | Tương thích hệ sinh thái mới |
| **`yolov5m-official`** | YOLOv5m (Official) | `yolov5m.pt` | **4K:** `batch=1` \| **1280:** `batch=16` | Đúng cấu hình bài báo gốc |
| **`rt-detr-v1`** | RT-DETR-L (Transformer) | `rtdetr-l.pt` | **4K:** `batch=1` \| **1280:** `batch=8` | Transformer Detector thời gian thực |
| **`rt-detr-v2`** | RT-DETR-X (Transformer) | `rtdetr-x.pt` | **4K:** `batch=1` \| **1280:** `batch=8` | SOTA Transformer độ chính xác cao |
| **`all`** | Chạy toàn bộ 6 models | *(Tự động tuần tự)* | **4K:** `batch=1` \| **1280:** `batch=16` | Benchmark toàn diện tự động |

> [!TIP]
> **Tính năng Tự Động Lưu Cloud & Chuẩn Đánh Giá Paper:**
> - Checkpoints (`best.pt`, `last.pt`, `results.csv`) tự động đồng bộ ngầm lên Hugging Face Hub (`Cuong2004/HRP4K`) sau mỗi epoch qua Background Thread Worker.
> - Mặc định đánh giá `--confidence 0.001` bám sát chuẩn COCO/PASCAL và Paper Benchmark.
> - Hỗ trợ cờ `--resume` để tiếp tục huấn luyện nếu phiên chạy bị ngắt giữa chừng.

---

# 🟦 LOẠI 1: KAGGLE NOTEBOOKS (GPU P100 / T4)

Chạy trực tiếp các cell lệnh dưới đây trong môi trường **Kaggle Notebooks**.

### 0. Clone Repo mới nhất hoặc Git Pull nếu đã có
```bash
![ -d "HRP4K/.git" ] && (cd HRP4K && git pull) || git clone https://github.com/nxc1802/HRP4K.git
%cd HRP4K
```

### 1. Cài đặt Framework & Dependencies
```bash
!pip install -q --upgrade pip
!pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"
!pip install -q -e .
!hrp4k --version
```

### 1.1 Thiết lập HF Token trong `.env` để Tự Động Lưu Checkpoint Ngầm lên Cloud
```bash
!echo "HF_TOKEN=<YOUR_HF_WRITE_TOKEN>" > .env
!echo "HF_REPO=Cuong2004/HRP4K" >> .env
```

### 2. Tự Động Thiết Lập Dataset (Nhận diện Kaggle Input hoặc tải từ Hugging Face)
```bash
!hrp4k setup-data
```

### 3. Phase 0: Kiểm Định Dataset & Phân Bố Scale Bins
```bash
!hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12
```

### 4. Phase 1: Huấn Luyện Baseline Detectors

```bash
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

### 5. Phase 2: High-Resolution Inference (Slicing 4K) & Chấm Điểm Tự Động

```bash
# Lựa chọn A (KHUYÊN DÙNG): Chạy Sliced-NMS (Tile 960x960, overlap 20%)
!hrp4k phase2 --method sliced-nms --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_sliced_nms.json

# Lựa chọn B: Chạy Perspective-Grid (Bám sát dải mặt đường ở xa)
!hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# Lựa chọn C: Chạy SAHI Inference (Tile 640x640, overlap 20%)
!hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_1280/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json

# Lựa chọn D: Chạy TẤT CẢ các method Phase 2 cùng lúc
!hrp4k phase2 --method all --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/phase2_benchmark/
```

### 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi

```bash
# Đánh giá độ chính xác COCO/FPPI chi tiết theo các dải kích thước (Tiny, Small, Medium, Large)
!hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/metrics/yolo11m_sliced_nms_metrics.json

# Chẩn đoán phân loại lỗi (Localization, Background FP, Classification confusion)
!hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/diagnostics
```

### 7. Đóng Gói Hoặc Đẩy Toàn Bộ Checkpoints & Kết Quả Lên Hugging Face Thủ Công
```bash
!hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token <YOUR_HF_WRITE_TOKEN>
```

---

# 🟩 LOẠI 2: NATIVE CLI / TERMINAL (LOCAL / CLOUD VM / SERVER)

Dành cho môi trường Terminal/Bash cục bộ hoặc trên máy chủ GPU.

### 0. Clone Repo hoặc Cập Nhật Mã Nguồn
```bash
[ -d "HRP4K/.git" ] && (cd HRP4K && git pull) || git clone https://github.com/nxc1802/HRP4K.git
cd HRP4K
```

### 1. Cài Đặt Môi Trường & Dependencies
```bash
# Nâng cấp pip và cài đặt dependencies cốt lõi
pip install -q --upgrade pip
pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"

# Cài đặt hrp4k CLI ở chế độ editable
pip install -q -e .

# Kiểm tra phiên bản CLI
hrp4k --version
```

### 1.1 Thiết Lập Biến Môi Trường & File `.env`
```bash
# Ghi cấu hình token và repo vào file .env
cat << 'EOF' > .env
HF_TOKEN=your_huggingface_write_token
HF_REPO=Cuong2004/HRP4K
EOF

# Hoặc export trực tiếp vào shell session
export HF_TOKEN="your_huggingface_write_token"
export HF_REPO="Cuong2004/HRP4K"
```

### 2. Tự Động Khởi Tạo & Kiểm Tra Dataset
```bash
# Lệnh sẽ kiểm tra thư mục HRP4K cục bộ hoặc tự động tải từ Hugging Face Hub (Cuong2004/HRP4K)
hrp4k setup-data --data HRP4K
```

### 3. Phase 0: Kiểm Định Dataset & Phân Tích Scale Bins
```bash
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12
```

### 4. Phase 1: Huấn Luyện Baseline Detectors

```bash
# Lựa chọn A (KHUYÊN DÙNG): Huấn luyện YOLO11m với Size Lớn 1280x1280
hrp4k phase1 --model yolo11m --imgsz 1280 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_1280

# Lựa chọn B: Tiếp tục Huấn luyện khi bị ngắt (Continuous Training với --resume)
hrp4k phase1 --model yolo11m --weights outputs/runs/yolo11m_1280/weights/last.pt --resume --output outputs/runs/yolo11m_1280

# Lựa chọn C: Huấn luyện YOLOv5m với Size 1280x1280
hrp4k phase1 --model yolov5m-compat --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolov5m_1280

# Lựa chọn D: Huấn luyện YOLO11m với KÍCH THƯỚC GỐC 4K UHD (3840x2176, batch 1)
hrp4k phase1 --model yolo11m --imgsz original --batch 1 --epochs 150 --allow-full --output outputs/runs/yolo11m_4k

# Lựa chọn E: Huấn luyện RT-DETRv2 (Transformer SOTA) với kích thước gốc 4K
hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 1 --epochs 150 --allow-full --output outputs/runs/rtdetr_v2_4k

# Lựa chọn F: Huấn luyện TOÀN BỘ 6 mô hình Baseline tự động
hrp4k phase1 --model all --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/phase1_all_1280
```

### 5. Phase 2: High-Resolution Inference (Slicing 4K) & Chấm Điểm Tự Động

```bash
# Lựa chọn A (KHUYÊN DÙNG): Chạy Sliced-NMS (Tile 960x960, overlap 20%)
hrp4k phase2 --method sliced-nms --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_sliced_nms.json

# Lựa chọn B: Chạy Perspective-Grid (Bám sát dải mặt đường ở xa)
hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# Lựa chọn C: Chạy SAHI Inference
hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_1280/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json

# Lựa chọn D: Chạy TẤT CẢ các method Phase 2 cùng lúc
hrp4k phase2 --method all --weights outputs/runs/yolo11m_1280/weights/best.pt --output outputs/phase2_benchmark/
```

### 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi

```bash
# Đánh giá COCO mAP theo các Scale Bins
hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/metrics/yolo11m_sliced_nms_metrics.json

# Chẩn đoán phân loại sai số phát hiện
hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_sliced_nms.json --output outputs/diagnostics
```

### 7. Đóng Gói Hoặc Đẩy Toàn Bộ Checkpoints & Kết Quả Lên Hugging Face Thủ Công
```bash
hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token <YOUR_HF_WRITE_TOKEN>
```
