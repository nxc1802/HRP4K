# 🚀 HRP4K Benchmark Execution Commands (Marimo Lab / GPU Server / Local / Kaggle)

Tài liệu tổng hợp toàn bộ các lệnh chạy thực thi quy trình nghiên cứu, huấn luyện baseline và đánh giá độ phân giải cao cho tập dữ liệu **HRP4K (Phase 0 → Phase 3)** trên **Marimo Lab & Native GPU Server (NVIDIA RTX PRO 6000 96GB)**, **Kaggle Notebooks**, và **Local Machine (Minimal Smoke Mode)**.

---

## ⚡ 0. Marimo Keep-Alive Code (Giữ Server Luôn Chạy & Ổn Định)
Chạy cell code Python này đầu tiên trong Marimo Notebook để giữ server luôn hoạt động liên tục, ngăn ngừa timeout/sleep khi huấn luyện dài hạn:

```python
import time
while True:
    time.sleep(3600)
    print("Processing")
```

---

## 🧪 0.1. Kiểm Tra Cục Bộ Siêu Nhẹ (Local Minimal Smoke Test - 10 Giây Trên CPU)
Khi kiểm thử tính đúng đắn của toàn bộ pipeline mã nguồn trên máy cá nhân mà không làm tốn tài nguyên hay đơ máy:

```bash
# Chạy toàn bộ pipeline Phase 0 -> Phase 1 -> Phase 2 (5 methods) -> Phase 3 trong 10 giây (CPU, 2 ảnh train, 1 epoch)
hrp4k run-smoke --data HRP4K --output outputs/smoke_minimal
```

---

## 📋 Danh Sách & So Sánh Các Mô Hình Baseline (`--model`)

| Tùy chọn `--model` | Kiến trúc mô hình | Pretrained Weights | Cấu hình Độ Phân Giải & Batch Size | Ghi chú hiệu năng |
| :--- | :--- | :--- | :--- | :--- |
| **`yolo11m`** *(Khuyên dùng)* | YOLOv11 Medium | `yolo11m.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | SOTA mAP (55.05%), hội tụ nhanh, tối ưu VRAM |
| **`yolov8m`** | YOLOv8 Medium | `yolov8m.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | Baseline tiêu chuẩn cộng đồng |
| **`yolov5m-compat`** | YOLOv5m (Ultralytics) | `yolov5mu.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | Tương thích hệ sinh thái mới |
| **`yolov5m-official`** | YOLOv5m (Official) | `yolov5m.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | Đúng cấu hình bài báo gốc |
| **`rt-detr-v1`** | RT-DETR-L (Transformer) | `rtdetr-l.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | Transformer Detector thời gian thực |
| **`rt-detr-v2`** | RT-DETR-X (Transformer) | `rtdetr-x.pt` | **4K Gốc:** `--imgsz original --batch 16 --rect` | SOTA Transformer độ chính xác cao |
| **`all`** | Chạy toàn bộ 6 models | *(Tự động tuần tự)* | **4K Gốc:** `--imgsz original --batch 16 --rect` | Benchmark toàn diện tự động |

> [!TIP]
> **Các Cờ Mặc Định Đã Được Kích Hoạt Toàn Diện:**
> - `--allow-full`: Cấp quyền chạy trọn vẹn 150 Epochs publication-grade.
> - `--confidence 0.001`: Ngưỡng đánh giá chuẩn COCO / PASCAL VOC của bài báo gốc.
> - `--rect`: Mặc định bật Rectangular training (giảm 45% VRAM đệm đen thừa của ảnh 16:9).
> - `--resume`: Tự động tìm checkpoint cục bộ hoặc **tải tự động từ Hugging Face** nếu bị ngắt.
> - Checkpoints (`best.pt`, `last.pt`, `results.csv`) được đồng bộ ngầm lên Cloud `Cuong2004/HRP4K`.

---

## 🖥️ 1. Quản Lý Đa Terminal (`tmux`) & Chia Màn Hình Ngang (Chạy Song Song)

Sử dụng 1 khối lệnh duy nhất dưới đây để cài đặt, bật chuột, tạo session và chia đôi màn hình ngang (trên / dưới) để chạy song song 2 model:

```bash
apt-get update && apt-get install -y tmux && echo "set -g mouse on" >> ~/.tmux.conf && tmux new-session -d -s benchmark && tmux split-window -v -t benchmark && tmux attach -t benchmark
```
*(Sau khi chạy lệnh trên, màn hình terminal sẽ tự động tách làm 2 nửa trên và dưới. Bạn có thể dùng chuột click vào ô trên để chạy Model 1, click vào ô dưới để chạy Model 2).*

---

## ⚙️ 2. Thiết Lập Môi Trường & Dataset (Khởi Tạo Ban Đầu)

Gộp toàn bộ các bước Clone/Pull repo, cài đặt dependencies, cấu hình Hugging Face Token và tải tập dữ liệu HRP4K vào **1 khối lệnh duy nhất**:

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

## 🔬 3. Phase 0: Kiểm Định Dataset & Phân Tích Scale Bins
```bash
hrp4k phase0 --data HRP4K --output outputs/phase0 --quality-samples 12
```

---

## 🏋️ 4. Phase 1: Huấn Luyện Baseline Detectors (Toàn Bộ 4K Gốc & Batch Size 16)

### 4.1. Huấn Luyện Trực Tiếp Trên 4K Gốc (Native 4K End-to-End)
```bash
# Lựa chọn A (KHUYÊN DÙNG): Huấn luyện YOLO11m với KÍCH THƯỚC GỐC 4K UHD (--imgsz original, batch 16, --rect)
hrp4k phase1 --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# Lựa chọn B: Tiếp tục Huấn luyện khi bị ngắt (Continuous Training với --resume)
hrp4k phase1 --model yolo11m --weights outputs/runs/yolo11m_4k/weights/last.pt --imgsz original --resume --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# Lựa chọn C: Huấn luyện YOLOv8m với KÍCH THƯỚC GỐC 4K (batch 16, --rect)
hrp4k phase1 --model yolov8m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolov8m_4k

# Lựa chọn D: Huấn luyện YOLOv5m với KÍCH THƯỚC GỐC 4K (batch 16, --rect)
hrp4k phase1 --model yolov5m-compat --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolov5m_4k

# Lựa chọn E: Huấn luyện RT-DETRv2 (Transformer SOTA) với KÍCH THƯỚC GỐC 4K (batch 16, --rect)
hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/rtdetr_v2_4k

# Lựa chọn G (D-FINE 4K FP32): Huấn luyện tiếp từ best.pt với độ chính xác FP32 thuần (Tự động tải best.pt từ HF)
hrp4k phase1 --model d-fine --weights checkpoints/dfine_4k/best.pt --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/dfine_4k

# Lựa chọn H: Huấn luyện TOÀN BỘ các mô hình Baseline tự động với kích thước gốc 4K
hrp4k phase1 --model all --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/phase1_all_4k
```

---

### 4.2. Huấn Luyện Dựa Trên Mảng Cắt Cục Bộ (Crop Before Training 640 / Patch-Train-640)
Cắt ảnh 4K thành các mảng patches $640 \times 640$ giữ nguyên mật độ điểm ảnh gốc để huấn luyện mô hình nhẹ:

```bash
# Bước 1: Tạo tập dữ liệu Patches 640x640 từ ảnh 4K (tự động tính tọa độ bbox và nhãn YOLO)
hrp4k prepare-patches --data HRP4K --tile-size 640 --overlap 0.2 --output outputs/dataset_patches_640

# Bước 2: Huấn luyện YOLO11m trên tập Patches 640 (150 Epochs, batch 16)
hrp4k phase1 --model yolo11m --dataset outputs/dataset_patches_640/dataset.yaml --imgsz 640 --batch 16 --epochs 150 --allow-full --confidence 0.001 --output outputs/runs/yolo11m_patch640

# Bước 3: Đánh giá mô hình Patch bằng SAHI trên tập Test 4K gốc
hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_patch640/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_patch_sahi.json
```

---

## 🔍 5. Phase 2: High-Resolution Inference & Resolution Allocation (900 Ảnh Test 4K)

### 5.1. Slicing & Warping Inference Trên Model 4K Gốc / ZoomDet
```bash
# Lựa chọn A1 (ZOOMDET NEURAL): Dùng Mạng Neural ConvNet nhẹ tự sinh lưới biến dạng
hrp4k phase2 --method zoomdet-neural --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_zoomdet_neural.json

# Lựa chọn A2 (ZOOMDET GEOMETRY PRIOR): Dùng hàm phối cảnh mặt đường
hrp4k phase2 --method zoomdet-geometry --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_zoomdet_geometry.json

# Lựa chọn B: Perspective-Grid (9 calls)
hrp4k phase2 --method perspective-grid --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_perspective_grid.json

# Lựa chọn C: SAHI Inference (15 calls)
hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_4k/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_sahi.json

# Lựa chọn D: Sliced-NMS (25 calls)
hrp4k phase2 --method sliced-nms --weights outputs/runs/yolo11m_4k/weights/best.pt --output outputs/predictions/yolo11m_sliced_nms.json
```

---

### 🧩 5.2. Slicing Inference Trên Patch-640 Models (3 Lệnh Cần Chạy Để Hoàn Tất Bảng IV)
Chạy suy luận các phương pháp Slicing trên $900$ ảnh Test 4K cho mô hình huấn luyện bằng Patch 640 (chuẩn tile-size 640x640):

```bash
# 1️⃣ YOLO11m Patch 640 + Sliced-NMS (Lưới đều 640x640):
hrp4k phase2 --data HRP4K --split test --weights checkpoints/yolo11m_patch640/best.pt --method sliced-nms --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_patch_sliced_nms.json

# 2️⃣ D-FINE Patch 640 + SAHI (SAHI đa cấp 640x640):
hrp4k phase2 --data HRP4K --split test --weights checkpoints/dfine_patch640/best.pt --method sahi --tile-size 640 --overlap 0.2 --output outputs/predictions/dfine_patch_sahi.json

# 3️⃣ D-FINE Patch 640 + Sliced-NMS (Lưới đều 640x640):
hrp4k phase2 --data HRP4K --split test --weights checkpoints/dfine_patch640/best.pt --method sliced-nms --tile-size 640 --overlap 0.2 --output outputs/predictions/dfine_patch_sliced_nms.json
```

---

## 📊 6. Phase 3: Đánh Giá Chi Tiết & Chẩn Đoán Lỗi

```bash
# Đánh giá độ chính xác COCO/FPPI chi tiết theo các dải kích thước (Tiny, Small, Medium, Large)
hrp4k phase3 --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_perspective_grid.json --output outputs/metrics/yolo11m_perspective_grid_metrics.json

# Chẩn đoán phân loại sai số (Localization, Background FP, Classification confusion)
hrp4k diagnose --ground-truth HRP4K/test.json --predictions outputs/predictions/yolo11m_perspective_grid.json --output outputs/diagnostics
```

---

## ☁️ 7. Đẩy Toàn Bộ Checkpoints & Kết Quả Lên Hugging Face Thủ Công
```bash
hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token ${HF_TOKEN}
```

---

## 🟦 8. Thực Thi Trên Kaggle Notebooks (GPU T4 / P100)

### Cell 1: Khởi Tạo Môi Trường & Kết Nối Hugging Face Trên Kaggle
```bash
!git clone https://github.com/nxc1802/HRP4K.git || (cd HRP4K && git pull)
%cd HRP4K
!pip install -q --upgrade pip
!pip install -q "ultralytics>=8.3.0" "pycocotools>=2.0.7" "sahi>=0.12.5" "opencv-python>=4.8" "pyyaml>=6.0" "huggingface_hub"
!pip install -q -e .

import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "your_hf_token_here")
os.environ["HF_REPO"] = "Cuong2004/HRP4K"

!hrp4k setup-data --data HRP4K
```

### Cell 2: Chạy Song Song 3 Lệnh Slicing Trên Dual GPU (GPU 0 & GPU 1) — Chỉ Mất ~2.5 Phút!
```bash
%%bash
# [GPU 0]: Chạy YOLO11m Patch 640 + Sliced-NMS (Tile 640x640)
(
  echo "🚀 [GPU 0] Bắt đầu YOLO11m Patch 640 + Sliced-NMS (Tile 640)..."
  CUDA_VISIBLE_DEVICES=0 hrp4k phase2 \
    --data HRP4K \
    --split test \
    --weights checkpoints/yolo11m_patch640/best.pt \
    --method sliced-nms \
    --tile-size 640 \
    --overlap 0.2 \
    --device 0 \
    --output outputs/predictions/yolo11m_patch_sliced_nms.json
  echo "✅ [GPU 0] Hoàn tất YOLO11m Patch 640 + Sliced-NMS!"
) &

# [GPU 1]: Chạy D-FINE Patch 640 + SAHI rồi đến D-FINE Patch 640 + Sliced-NMS (Tile 640x640)
(
  echo "🚀 [GPU 1] Bắt đầu (1/2): D-FINE Patch 640 + SAHI (Tile 640)..."
  CUDA_VISIBLE_DEVICES=1 hrp4k phase2 \
    --data HRP4K \
    --split test \
    --weights checkpoints/dfine_patch640/best.pt \
    --method sahi \
    --tile-size 640 \
    --overlap 0.2 \
    --device 0 \
    --output outputs/predictions/dfine_patch_sahi.json
  echo "✅ [GPU 1] Hoàn tất D-FINE SAHI! Bắt đầu (2/2): D-FINE Patch 640 + Sliced-NMS (Tile 640)..."
  
  CUDA_VISIBLE_DEVICES=1 hrp4k phase2 \
    --data HRP4K \
    --split test \
    --weights checkpoints/dfine_patch640/best.pt \
    --method sliced-nms \
    --tile-size 640 \
    --overlap 0.2 \
    --device 0 \
    --output outputs/predictions/dfine_patch_sliced_nms.json
  echo "✅ [GPU 1] Hoàn tất D-FINE Patch 640 + Sliced-NMS!"
) &

# Chờ cả 2 luồng GPU hoàn thành
wait
echo "🎉 TẤT CẢ 3 THÍ NGHIỆM ĐÃ HOÀN TẤT ĐỒNG THỜI TRÊN 2 GPU!"
```


### Cell 3: Chẩn Đoán Lỗi & Xem Bảng Kết Quả Tổng Hợp
```bash
!hrp4k diagnose --ground-truth HRP4K/test.json \
  --predictions \
    outputs/predictions/yolo11m_patch_sliced_nms.json \
    outputs/predictions/dfine_patch_sahi.json \
    outputs/predictions/dfine_patch_sliced_nms.json \
  --output outputs/diagnostics
```

### Cell 4: Nén Toàn Bộ Kết Quả & Tạo Link Tải Về Local (Kaggle / Server)
Chạy cell này để đóng gói toàn bộ thư mục `outputs/` thành file zip và hiện link click tải trực tiếp về máy tính:

```python
import shutil
import os
from IPython.display import FileLink, display

# 1. Đóng gói outputs thành file zip
output_zip = '/kaggle/working/hrp4k_results.zip' if os.path.exists('/kaggle/working') else 'outputs/hrp4k_results.zip'
source_dir = 'outputs'

print(f"📦 Đang nén thư mục {source_dir} thành {output_zip}...")
shutil.make_archive(output_zip.replace('.zip', ''), 'zip', source_dir)
print(f"✅ Đã nén thành công! Dung lượng: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")

# 2. Hiển thị link click tải trực tiếp
print("👉 Bấm vào link dưới đây để tải về máy local:")
display(FileLink(output_zip))
```

Hoặc dùng lệnh bash trực tiếp trên terminal:
```bash
zip -r /kaggle/working/hrp4k_results.zip outputs/
```

---

### Cell 5: Đồng Bộ Trực Tiếp Lên Hugging Face Repo
```bash
!hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token ${HF_TOKEN}
```



