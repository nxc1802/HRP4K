# 🚀 HRP4K Benchmark Execution Commands (Marimo Lab / GPU Server / Local / Kaggle)

Tài liệu tổng hợp toàn bộ các lệnh chạy thực thi quy trình nghiên cứu, huấn luyện baseline, kiến trúc biến dạng hình học mặt đường **Warped ZoomDet 640** và đánh giá độ phân giải cao cho tập dữ liệu **HRP4K (Phase 0 → Phase 3)** trên **Marimo Lab & Native GPU Server (NVIDIA RTX PRO 6000 96GB / A100 / H100)**, **Kaggle Notebooks (Dual T4/P100)**, và **Local Machine (Minimal Smoke Mode)**.

---

## ⚡ 0. Keep-Alive Code (Giữ Server Luôn Chạy & Ổn Định)
Chạy cell code Python này đầu tiên trong Notebook để giữ server luôn hoạt động liên tục, ngăn ngừa timeout/sleep khi huấn luyện dài hạn:

```python
import time
while True:
    time.sleep(3600)
    print("Processing...")
```

---

## 🧪 0.1. Kiểm Tra Cục Bộ Siêu Nhẹ (Local Minimal Smoke Test - 10 Giây Trên CPU)
Khi kiểm thử tính đúng đắn của toàn bộ pipeline mã nguồn trên máy cá nhân mà không làm tốn tài nguyên hay đơ máy:

```bash
# Chạy toàn bộ pipeline Phase 0 -> Phase 1 -> Phase 2 (tất cả methods) -> Phase 3 trong 10 giây (CPU, 2 ảnh train, 1 epoch)
hrp4k run-smoke --data HRP4K --output outputs/smoke_minimal
```

---

## 📋 Danh Sách & So Sánh Các Mô Hình Baseline (`--model`)

| Tùy chọn `--model` | Kiến trúc mô hình | Pretrained Weights | Số lượng Tham số | Cấu hình Resolution & Mục tiêu |
| :--- | :--- | :--- | :--- | :--- |
| **`yolo11m`** | YOLOv11 Medium | `yolo11m.pt` | ~20.1M | **4K Gốc Baseline & 640:** `--imgsz original / 640 --batch 16 --rect` |
| **`d-fine`** | D-FINE / RT-DETR Fine Refined | `rtdetr-l.pt` | ~32.0M | **SOTA Fine-grained Distribution Transformer Baseline** |
| **`rt-detr-v1`** | RT-DETR-L (Transformer) | `rtdetr-l.pt` | ~32.0M | **4K Gốc Transformer Detector** |
| **`rt-detr-v2`** | RT-DETR-X (Transformer) | `rtdetr-x.pt` | ~65.0M | **SOTA Transformer Độ chính xác cao** |
| **`yolov8m`** | YOLOv8 Medium | `yolov8m.pt` | ~25.9M | **Dense CNN Baseline** |
| **`yolov5m-compat`** | YOLOv5m (Ultralytics) | `yolov5mu.pt` | ~21.2M | **Dense CNN Baseline** |
| **`yolov5m-official`** | YOLOv5m (Official) | `yolov5m.pt` | ~21.2M | **4K Gốc Baseline:** Đúng cấu hình bài báo gốc |
| **`all`** | Chạy toàn bộ các mô hình | *(Tự động tuần tự)* | Varies | Benchmark toàn diện tự động |

> [!TIP]
> **Các Cờ Mặc Định Đã Được Kích Hoạt Toàn Diện:**
> - `--allow-full`: Cấp quyền chạy trọn vẹn 150 Epochs publication-grade.
> - `--confidence 0.001`: Ngưỡng đánh giá chuẩn COCO / PASCAL VOC của bài báo gốc.
> - `--rect`: Mặc định bật Rectangular training (giảm 45% VRAM đệm đen thừa của ảnh 16:9).
> - `--resume`: Tự động tìm checkpoint cục bộ hoặc **tải tự động từ Hugging Face** nếu bị ngắt.
> - Checkpoints (`best.pt`, `last.pt`, `results.csv`) được đồng bộ ngầm lên Cloud `Cuong2004/HRP4K`.

---

## ⚙️ 1. Thiết Lập Môi Trường & Dataset (Khởi Tạo Ban Đầu)

```bash
# 1. Cập nhật mã nguồn
[ -d "HRP4K/.git" ] && (cd HRP4K && git pull) || git clone https://github.com/nxc1802/HRP4K.git
cd HRP4K

# 2. Cài đặt thư viện & hrp4k CLI
pip install -q --upgrade pip
pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"
pip install -q -e .

# 3. Cấu hình Hugging Face Token & Repo
cat << 'EOF' > .env
HF_TOKEN=${HF_TOKEN:-your_hf_token_here}
HF_REPO=Cuong2004/HRP4K
EOF
export HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"
export HF_REPO="Cuong2004/HRP4K"

# 4. Tự động kiểm tra / tải dataset từ Hugging Face
hrp4k setup-data --data HRP4K
```

---

## 🔬 2. Phase 0: Kiểm Định Dataset & Phân Tích Scale Bins
```bash
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12
```

---

## 🚀 3. Phase 1: QUY TRÌNH HUẤN LUYỆN BASELINES & PROPOSED WARPED MODEL

### 3.1. Chuẩn Bị Dữ Liệu Patch & Warped
```bash
# A. Sinh tập dữ liệu Patch 640x640 (cho Patch-based Training)
hrp4k prepare-patches --data HRP4K --output outputs/dataset_patches_640 --tile-size 640 --overlap 0.2

# B. Sinh tập dữ liệu Warped Continuous Deformation (cho ZoomDet Training)
hrp4k prepare-warped --data HRP4K --output outputs/dataset_zoomdet_640 --canvas-size 640 --horizon-ratio 0.40
```

### 3.2. Huấn Luyện Các Mô Hình Cốt Lõi (150 Epochs)
```bash
# 1. Native 4K UHD Baseline (YOLO11m)
hrp4k phase1 --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# 2. Native 4K UHD Baseline (D-FINE)
hrp4k phase1 --model d-fine --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/dfine_4k

# 3. Standard Resize 640 Baseline (YOLO11m & D-FINE)
hrp4k phase1 --model yolo11m --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_640
hrp4k phase1 --model d-fine --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/dfine_640

# 4. Patch-Train 640 (YOLO11m & D-FINE trên tập patch)
hrp4k phase1 --model yolo11m --dataset outputs/dataset_patches_640/dataset.yaml --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_patch640
hrp4k phase1 --model d-fine --dataset outputs/dataset_patches_640/dataset.yaml --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/dfine_patch640

# 5. Proposed Warped ZoomDet 640 (YOLO11m & D-FINE trên tập warped)
hrp4k phase1 --model yolo11m --dataset outputs/dataset_zoomdet_640/dataset.yaml --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_zoomdet640
hrp4k phase1 --model d-fine --dataset outputs/dataset_zoomdet_640/dataset.yaml --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/dfine_zoomdet640
```

---

## 🔍 4. Phase 2: SUY LUẬN & ĐÁNH GIÁ TRÊN 900 ẢNH TEST 4K ĐỘC LẬP

### 4.1. Chạy Tự Động Toàn Bộ Methods Trên Checkpoint
```bash
hrp4k phase2 --method all --weights outputs/runs/dfine_zoomdet640/weights/best.pt --output outputs/phase2_benchmark/
```

### 4.2. Chạy Từng Phương Pháp Riêng Lẻ:
```bash
# 1. Proposed 1-Pass Continuous Deformation Warp (ZoomDet 640)
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/dfine_zoomdet640/weights/best.pt --method zoomdet --imgsz 640 --output outputs/predictions/dfine_zoomdet640.json

# 2. Native 4K UHD (1 pass)
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/dfine_4k/weights/best.pt --method resize --imgsz original --output outputs/predictions/dfine_4k.json

# 3. Uniform Sliced NMS (25 calls, 960 tile size, 0.2 overlap)
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/dfine_patch640/weights/best.pt --method sliced-nms --tile-size 960 --overlap 0.2 --output outputs/predictions/dfine_sliced_nms.json

# 4. Perspective Grid (9 calls)
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/dfine_patch640/weights/best.pt --method perspective-grid --output outputs/predictions/dfine_perspective_grid.json

# 5. SAHI Multi-Scale (32 calls)
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/dfine_patch640/weights/best.pt --method sahi --tile-size 640 --overlap 0.2 --output outputs/predictions/dfine_sahi.json
```

---

## 📊 5. Phase 3: BÁO CÁO ĐỐI SÁNH & CHẨN ĐOÁN (DIAGNOSTICS)

```bash
hrp4k diagnose \
  --ground-truth HRP4K/test.json \
  --predictions \
    outputs/predictions/dfine_4k.json \
    outputs/predictions/dfine_zoomdet640.json \
    outputs/predictions/dfine_sliced_nms.json \
    outputs/predictions/dfine_perspective_grid.json \
    outputs/predictions/dfine_sahi.json \
  --output outputs/phase3_report
```
