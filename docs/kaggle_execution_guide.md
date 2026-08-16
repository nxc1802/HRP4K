# HRP4K Kaggle Execution Guide (Hướng Dẫn Chạy CLI Trên Kaggle)

Tài liệu này cung cấp toàn bộ các câu lệnh CLI tối giản, sẵn sàng sao chép và thực thi trực tiếp trên **Kaggle Notebooks (GPU T4 / P100 / dual T4)** với tiền tố `!`.

---

## 0. Clone Mã Nguồn Hoặc Cập Nhật (Git Clone / Git Pull)

Lệnh tự động kiểm tra: nếu chưa có thì `git clone`, nếu đã có thì `git pull` để lấy code mới nhất và di chuyển vào thư mục dự án:

```bash
# [Cell 0: Clone hoặc Cập nhật mã nguồn]
![ -d "HRP4K/.git" ] && (cd HRP4K && git pull) || git clone https://github.com/nxc1802/HRP4K.git
%cd HRP4K
```

---

## 1. Thiết Lập Môi Trường (Environment Setup)

Chạy cell sau trong Kaggle Notebook để cài đặt repository và các thư viện phụ thuộc:

```bash
# [Cell 1: Cài đặt Dependencies & HRP4K Framework]
!pip install -q --upgrade pip
!pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"
!pip install -q -e .
!nvidia-smi
!hrp4k --version
```

---

## 2. Thiết Lập Dataset Tự Động (Auto Data Setup)

Lệnh này tự động kiểm tra xem dataset đã được đính kèm ở `/kaggle/input/...` chưa để liên kết ngay trong $0.1\text{s}$. Nếu không có, lệnh sẽ tự động tải 11.92 GB từ Hugging Face:

```bash
# [Cell 2: Tự Động Thiết Lập Dataset]
!hrp4k setup-data
```

---

## 3. Phase 0: Kiểm Định Toàn Vẹn Dataset & Thống Kê Phân Bố

```bash
# [Cell 3: Phân tích toàn diện 4 scale bins, tỉ lệ âm tính và xuất báo cáo]
!hrp4k phase0
```

---

## 4. Phase 1: Huấn Luyện & Đánh Giá Baseline (Hỗ Trợ Kích Thước Gốc `--imgsz original` & `--imgsz 1280`)

### 📌 Lựa chọn 1: Huấn luyện với Kích Thước Gốc 4K 3840x2160 (`--imgsz original`)

```bash
# Huấn luyện YOLO11m với kích thước gốc 4K (Batch 2-4 trên Kaggle GPU để tránh OOM)
!hrp4k phase1 --model yolo11m --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/runs/yolo11m_4k

# Huấn luyện RT-DETRv2 (Transformer) với kích thước gốc 4K
!hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/runs/rtdetr_v2_4k
```

### 📌 Lựa chọn 2: Huấn luyện với Kích Thước Lớn 1280x1280 (`--imgsz 1280`)

```bash
# Huấn luyện YOLO11m với canvas 1280x1280 (Batch 16)
!hrp4k phase1 --model yolo11m --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolo11m_1280

# Huấn luyện YOLOv8m với canvas 1280x1280 (Batch 16)
!hrp4k phase1 --model yolov8m --imgsz 1280 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolov8m_1280
```

### 📌 Lựa chọn 3: Huấn luyện TOÀN BỘ 6 Model Baseline (`--model all`)

```bash
# Tự động chạy tuần tự toàn bộ 6 mô hình Baseline ở kích thước gốc 4K
!hrp4k phase1 --model all --imgsz original --batch 2 --epochs 150 --allow-full --output outputs/phase1_all_4k
```

---

## 5. Phase 2: High-Resolution Inference (Slicing, Perspective Grid, SAHI)

Áp dụng các chiến lược xử lý độ phân giải cao 4K trực tiếp trên checkpoint đã huấn luyện:

### 📌 Lựa chọn 1: Chạy một Phương Pháp Cụ Thể (Single Method)

```bash
# 1. Phương pháp Cắt lát chuẩn Sliced-NMS (Tile 960x960, overlap 20%)
!hrp4k phase2 --method sliced-nms \
    --weights outputs/runs/yolo11m_4k/weights/best.pt \
    --imgsz 640 --tile-size 960 --overlap 0.2 \
    --output outputs/predictions/yolo11m_sliced_nms.json

# 2. Phương pháp Chia lưới Perspective Grid (Tập trung vùng mặt đường ở xa)
!hrp4k phase2 --method perspective-grid \
    --weights outputs/runs/yolo11m_4k/weights/best.pt \
    --imgsz 640 \
    --output outputs/predictions/yolo11m_perspective_grid.json

# 3. Phương pháp SAHI (Sliced Aided Hyper Inference)
!hrp4k phase2 --method sahi \
    --weights outputs/runs/yolo11m_4k/weights/best.pt \
    --tile-size 640 --overlap 0.2 \
    --output outputs/predictions/yolo11m_sahi.json
```

### 📌 Lựa chọn 2: Chạy TOÀN BỘ các Phương Pháp Phase 2 (`--method all`)

```bash
# Tự động chạy cả 4 phương pháp (resize, sliced-nms, perspective-grid, sahi) và tự động chấm điểm COCO
!hrp4k phase2 --method all \
    --weights outputs/runs/yolo11m_4k/weights/best.pt \
    --output outputs/phase2_benchmark/
```

---

## 6. Phase 3: Đánh Giá Chuẩn Hóa COCO & Chẩn Đoán Lỗi (Diagnostics)

```bash
# [Cell 6.1: Đánh giá COCO AP, 4 scale bins và đo lường FPPI]
!hrp4k phase3 \
    --ground-truth HRP4K/test.json \
    --predictions outputs/predictions/yolo11m_sliced_nms.json \
    --output outputs/metrics/yolo11m_sliced_nms_metrics.json \
    --confidence 0.25

# [Cell 6.2: Chẩn đoán nguyên nhân lỗi: trôi box, missed small targets]
!hrp4k diagnose \
    --ground-truth HRP4K/test.json \
    --predictions outputs/predictions/yolo11m_sliced_nms.json outputs/predictions/yolo11m_perspective_grid.json \
    --output outputs/phase3_diagnostics
```

---

## 7. Đóng Gói & Tải Kết Quả Về Máy / Lưu Ra Kaggle Output

```bash
# Nén toàn bộ weights và báo cáo thành 1 file tar.gz duy nhất
!tar -czvf hrp4k_kaggle_results.tar.gz outputs/
```
