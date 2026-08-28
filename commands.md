# 🚀 HRP4K Benchmark Execution Commands (Marimo Lab / GPU Server / Local / Kaggle)

Tài liệu tổng hợp toàn bộ các lệnh chạy thực thi quy trình nghiên cứu, huấn luyện baseline, kiến trúc mới **AdaPoth-Lite** (MobileNetV3-Small Scout + Shared YOLO11n-P2-lite) và đánh giá độ phân giải cao cho tập dữ liệu **HRP4K (Phase 0 → Phase 3)** trên **Marimo Lab & Native GPU Server (NVIDIA RTX PRO 6000 96GB / A100 / H100)**, **Kaggle Notebooks (Dual T4/P100)**, và **Local Machine (Minimal Smoke Mode)**.

---

## ⚡ 0. Marimo / Colab / Kaggle Keep-Alive Code (Giữ Server Luôn Chạy & Ổn Định)
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

## 📋 Danh Sách & So Sánh Các Mô Hình Baseline & AdaPoth-Lite (`--model`)

| Tùy chọn `--model` | Kiến trúc mô hình | Pretrained Weights | Số lượng Tham số | Cấu hình Resolution & Mục tiêu |
| :--- | :--- | :--- | :--- | :--- |
| **`yolo11n-p2-lite`** *(AdaPoth)* | YOLO11n-P2-lite (Shared Detector) | `yolo11n.pt` / `yolo11n_p2_lite.yaml` | **~3.2M** | **AdaPoth-Lite Shared View:** Stride 4 (P2), 8 (P3), 16 (P4) |
| **`yolo11n-p2`** | YOLO11n-P2 Full | `yolo11n.pt` | **~3.5M** | **Ultra-fine Pothole Detector:** Stride 4 (P2) |
| **`yolo11m`** | YOLOv11 Medium | `yolo11m.pt` | ~20.1M | **4K Gốc Baseline:** `--imgsz original --batch 16 --rect` |
| **`yolov8m`** | YOLOv8 Medium | `yolov8m.pt` | ~25.9M | **4K Gốc Baseline:** `--imgsz original --batch 16 --rect` |
| **`yolov5m-compat`** | YOLOv5m (Ultralytics) | `yolov5mu.pt` | ~21.2M | **4K Gốc Baseline:** `--imgsz original --batch 16 --rect` |
| **`yolov5m-official`** | YOLOv5m (Official) | `yolov5m.pt` | ~21.2M | **4K Gốc Baseline:** Đúng cấu hình bài báo gốc |
| **`rt-detr-v1`** | RT-DETR-L (Transformer) | `rtdetr-l.pt` | ~32.0M | **4K Gốc Transformer Detector** |
| **`rt-detr-v2`** | RT-DETR-X (Transformer) | `rtdetr-x.pt` | ~65.0M | **SOTA Transformer Độ chính xác cao** |
| **`d-fine`** | D-FINE / RT-DETR Fine Refined | `rtdetr-l.pt` | ~32.0M | **SOTA Fine-grained Distribution Transformer** |
| **`all`** | Chạy toàn bộ các mô hình | *(Tự động tuần tự)* | Varies | Benchmark toàn diện tự động |

> [!TIP]
> **Các Cờ Mặc Định Đã Được Kích Hoạt Toàn Diện:**
> - `--allow-full`: Cấp quyền chạy trọn vẹn 150 Epochs publication-grade.
> - `--confidence 0.001`: Ngưỡng đánh giá chuẩn COCO / PASCAL VOC của bài báo gốc.
> - `--rect`: Mặc định bật Rectangular training (giảm 45% VRAM đệm đen thừa của ảnh 16:9).
> - `--resume`: Tự động tìm checkpoint cục bộ hoặc **tải tự động từ Hugging Face** nếu bị ngắt.
> - Checkpoints (`best.pt`, `last.pt`, `results.csv`, `scout_best.pt`) được đồng bộ ngầm lên Cloud `Cuong2004/HRP4K`.

---

## 🖥️ 1. Quản Lý Đa Terminal (`tmux`) & Chia Màn Hình Ngang (Chạy Song Song)

Sử dụng 1 khối lệnh duy nhất dưới đây để cài đặt, bật chuột, tạo session và chia đôi màn hình ngang (trên / dưới) để chạy song song:

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

## 🚀 4. QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ ADAPOTH-LITE (Theo 3 Giai Đoạn Trong upgrade.md)

### 📌 4.1. Module A: Huấn Luyện & Đánh Giá Region Scout (`MobileNetV3-Small`, ~1.2M Params)
Mục tiêu số 1: **Region Recall $\ge 97\%$** trên heatmap $60 \times 34$ (stride 16) từ ảnh thumbnail $960 \times 540$.

```bash
# Bước 1: Huấn luyện MobileNetV3-Small Scout (50 Epochs, Focal Loss + 2.0 * Coverage Loss)
hrp4k train-scout \
  --data HRP4K \
  --output outputs/runs/scout \
  --epochs 50 \
  --batch 16 \
  --lr 0.001 \
  --lambda-cov 2.0 \
  --device 0

# Bước 2: Đánh giá Region Recall, GT Coverage và False Region Rate của Scout trên tập Validation
hrp4k eval-scout \
  --data HRP4K \
  --split valid \
  --weights outputs/runs/scout/weights/scout_best.pt \
  --threshold 0.30 \
  --context-margin 0.20 \
  --k-max 4 \
  --output outputs/metrics/scout_valid_eval.json

# Bước 3: Đánh giá Scout trên tập Test 900 ảnh 4K (Metric báo cáo trong paper)
hrp4k eval-scout \
  --data HRP4K \
  --split test \
  --weights outputs/runs/scout/weights/scout_best.pt \
  --threshold 0.30 \
  --context-margin 0.20 \
  --k-max 4 \
  --output outputs/metrics/scout_test_eval.json
```

---

### 📌 4.2. Stage 1: Huấn Luyện Full-Image Baseline (YOLO11n-P2-lite)
Huấn luyện mô hình YOLO11n-P2-lite (~3.2M parameters) trên toàn bộ ảnh thumbnail 960x544:

```bash
hrp4k phase1 \
  --model yolo11n-p2-lite \
  --imgsz 960 \
  --batch 16 \
  --epochs 150 \
  --allow-full \
  --confidence 0.001 \
  --rect \
  --output outputs/runs/yolo11n_p2_lite_stage1
```

---

### 📌 4.3. Stage 2: Huấn Luyện Local Crop Training (50% Jittered + 25% Hard Negatives + 25% Full)
Tạo tập dữ liệu local crops với jitter và các mảng hard negatives (vết nứt, vệt vá nhựa đường, bóng râm, vạch kẻ):

```bash
# Bước 1: Sinh tập dữ liệu Stage 2 Local Crops
hrp4k prepare-adapoth-crops \
  --data HRP4K \
  --stage stage2 \
  --crop-size 640 \
  --context-margin 0.20 \
  --output outputs/dataset_adapoth_stage2

# Bước 2: Huấn luyện YOLO11n-P2-lite trên tập Stage 2 (150 Epochs, batch 16)
hrp4k phase1 \
  --model yolo11n-p2-lite \
  --dataset outputs/dataset_adapoth_stage2/dataset.yaml \
  --imgsz 640 \
  --batch 16 \
  --epochs 150 \
  --allow-full \
  --confidence 0.001 \
  --rect \
  --output outputs/runs/yolo11n_p2_lite_stage2
```

---

### 📌 4.4. Stage 3: Scout-Generated Crop Fine-Tuning (60% Scout Crops + 40% GT/Full)
Đồng bộ phân phối huấn luyện với phân phối suy luận thực tế bằng cách dùng chính Scout model để sinh crop:

```bash
# Bước 1: Dùng Scout checkpoint tốt nhất để sinh tập dữ liệu Stage 3
hrp4k prepare-adapoth-crops \
  --data HRP4K \
  --stage stage3 \
  --scout-weights outputs/runs/scout/weights/scout_best.pt \
  --crop-size 640 \
  --context-margin 0.20 \
  --output outputs/dataset_adapoth_stage3

# Bước 2: Fine-tune YOLO11n-P2-lite từ Stage 2 checkpoint (100 Epochs, batch 16)
hrp4k phase1 \
  --model yolo11n-p2-lite \
  --weights outputs/runs/yolo11n_p2_lite_stage2/weights/best.pt \
  --dataset outputs/dataset_adapoth_stage3/dataset.yaml \
  --imgsz 640 \
  --batch 16 \
  --epochs 100 \
  --allow-full \
  --confidence 0.001 \
  --rect \
  --output outputs/runs/yolo11n_p2_lite_stage3
```

---

## 🔍 5. Phase 2: SUY LUẬN ADAPOTH-LITE & 9 THÍ NGHIỆM ABLATION (900 Ảnh Test 4K)

### 🏆 5.1. Mô Hình Chính: AdaPoth-Lite Dynamic Top-K ($K \le 4$)
Chạy suy luận trọn vẹn pipeline AdaPoth-Lite: Scout $\to$ Dynamic Top-K $\to$ Shared YOLO11n-P2-lite $\to$ Boundary Penalty $\to$ Fusion:

```bash
hrp4k phase2 \
  --data HRP4K \
  --split test \
  --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt \
  --method adapoth \
  --scout-weights outputs/runs/scout/weights/scout_best.pt \
  --k-max 4 \
  --context-margin 0.20 \
  --boundary-penalty 0.70 \
  --confidence 0.05 \
  --output outputs/predictions/adapoth_lite_dynamic_k4.json
```

---

### 🧪 5.2. Experiment 2 — Oracle Crop Upper Bound
Dùng GT boxes để sinh local crop lý tưởng nhằm xác định giới hạn trên của detector:

```bash
hrp4k phase2 \
  --data HRP4K \
  --split test \
  --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt \
  --method adapoth-oracle \
  --k-max 4 \
  --context-margin 0.20 \
  --boundary-penalty 0.70 \
  --output outputs/predictions/adapoth_oracle_k4.json
```

---

### ⚖️ 5.3. Experiment 3 — Fixed vs Adaptive Allocation Ablation
So sánh các chiến lược phân bổ vùng xử lý:

```bash
# 1️⃣ Global only (K = 0):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method resize --imgsz 960 --output outputs/predictions/ablation_global_only.json

# 2️⃣ Random crop (K = 2):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth-random --k-max 2 --output outputs/predictions/ablation_random_k2.json

# 3️⃣ Scout + Fixed K (K = 4):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth-fixed --scout-weights outputs/runs/scout/weights/scout_best.pt --k-max 4 --output outputs/predictions/ablation_fixed_k4.json

# 4️⃣ Scout + Dynamic Top-K (K <= 4 - Main Proposed):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --k-max 4 --output outputs/predictions/ablation_dynamic_k4.json
```

---

### 🔢 5.4. Experiment 5 — Số Lượng Candidate Tối Đa ($K_{max} = 2, 4, 6$)
```bash
# Kmax = 2 (Ultra-fast):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --k-max 2 --output outputs/predictions/ablation_kmax2.json

# Kmax = 4 (Main Configuration):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --k-max 4 --output outputs/predictions/ablation_kmax4.json

# Kmax = 6 (Max Coverage):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --k-max 6 --output outputs/predictions/ablation_kmax6.json
```

---

### 📏 5.5. Experiment 6 — Context Margin Ablation ($10\%, 20\%, 30\%$)
```bash
# Margin 10%:
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --context-margin 0.10 --output outputs/predictions/ablation_margin10.json

# Margin 20% (Main Configuration):
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --context-margin 0.20 --output outputs/predictions/ablation_margin20.json

# Margin 30%:
hrp4k phase2 --data HRP4K --split test --weights outputs/runs/yolo11n_p2_lite_stage3/weights/best.pt --method adapoth --scout-weights outputs/runs/scout/weights/scout_best.pt --context-margin 0.30 --output outputs/predictions/ablation_margin30.json
```

---

## 🏋️ 6. Huấn Luyện Các Baseline Detectors Khác (4K Gốc & Slicing)

### 6.1. Huấn Luyện Trực Tiếp Trên 4K Gốc
```bash
# Lựa chọn A (YOLO11m 4K Gốc):
hrp4k phase1 --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolo11m_4k

# Lựa chọn B (D-FINE 4K FP32):
hrp4k phase1 --model d-fine --weights checkpoints/dfine_4k/best.pt --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/dfine_4k
```

### 6.2. Slicing Inference Trên Mô Hình 4K Gốc / Patch 640
```bash
# YOLO11m Patch 640 + Sliced-NMS (25 calls):
hrp4k phase2 --data HRP4K --split test --weights checkpoints/yolo11m_patch640/best.pt --method sliced-nms --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_patch_sliced_nms.json

# D-FINE Patch 640 + SAHI (15 calls):
hrp4k phase2 --data HRP4K --split test --weights checkpoints/dfine_patch640/best.pt --method sahi --tile-size 640 --overlap 0.2 --output outputs/predictions/dfine_patch_sahi.json
```

---

## 📊 7. Phase 3: Đánh Giá Chi Tiết COCO / FPPI / Scale Bins & Chẩn Đoán Lỗi

```bash
# 1. Đánh giá toàn diện các chỉ số mAP50, mAP75, mAP50-95, AP-ultra-fine, AP-fine, AP-medium, AP-large:
hrp4k phase3 \
  --ground-truth HRP4K/test.json \
  --predictions outputs/predictions/adapoth_lite_dynamic_k4.json \
  --output outputs/metrics/adapoth_lite_metrics.json

# 2. Chẩn đoán phân loại sai số (Localization Error, Background FP, Scale Confusion):
hrp4k diagnose \
  --ground-truth HRP4K/test.json \
  --predictions \
    outputs/predictions/adapoth_lite_dynamic_k4.json \
    outputs/predictions/adapoth_oracle_k4.json \
    outputs/predictions/yolo11m_patch_sliced_nms.json \
  --output outputs/diagnostics
```

---

## ☁️ 8. Cơ Chế Đồng Bộ Hugging Face Toàn Diện (End-to-End Cloud Auto-Sync)

Hệ thống HRP4K được tích hợp cơ chế **Đồng bộ Đám mây Bất đồng bộ (Asynchronous Background Syncer)** hoàn chỉnh trên `Cuong2004/HRP4K`, chạy trên worker thread riêng biệt nên **hoàn toàn không làm chậm hay block GPU**:

### 🔄 8.1. Cơ Chế Đồng Bộ Tự Động Từng Phase:
1. **Scout Model Training (`hrp4k train-scout`):**
   * Tự động đồng bộ checkpoint tốt nhất `scout_best.pt` và `scout_last.pt` lên `checkpoints/scout/`.
   * Tự động đồng bộ `metrics.json` (chứa toàn bộ lịch sử loss, Region Recall, GT Coverage, False Region Rate, Avg K, cấu hình huấn luyện và môi trường runtime).
2. **Phase 1 Detector Training (Stage 1, 2, 3):**
   * Sau mỗi epoch, tự động đồng bộ `best.pt`, `last.pt`, `results.csv`, `args.yaml`.
   * Khi kết thúc huấn luyện, tự động đồng bộ trọn bộ `val_metrics.json`, `test_metrics.json`, `resolved_config.json` lên `checkpoints/{run_name}/`.
3. **Scout Evaluation (`hrp4k eval-scout --hf-sync`):**
   * Tự động upload báo cáo đánh giá `scout_eval.json` (chứa summary metrics và per-image candidate bounding boxes) lên `metrics/`.
4. **Phase 2 Inference (`hrp4k phase2 --hf-sync`):**
   * Tự động upload file dự đoán chuẩn canonical COCO `*_predictions.json` và metrics đánh giá `*_metrics.json` lên `predictions/` và `metrics/`.
5. **Phase 3 & Diagnostics (`hrp4k phase3 --hf-sync` / `hrp4k diagnose --hf-sync`):**
   * Tự động upload toàn bộ báo cáo phân tích lỗi `phase3_report.md`, biểu đồ Pareto, scale bins `scale_bins.json` lên `reports/` và `metrics/`.
6. **Auto-Download / Auto-Resume:**
   * Khi thực hiện Phase 2 hoặc `--resume` Phase 1, hàm `ensure_weights` sẽ tự động tìm kiếm và tải checkpoint tương ứng từ Hugging Face nếu máy cục bộ chưa có.

### 📤 8.2. Đẩy Thủ Công Toàn Bộ Thư Mục Outputs Lên Hugging Face
Nếu muốn đồng bộ thủ công toàn bộ cây thư mục `outputs/` lên Hugging Face bất kỳ lúc nào:

```bash
hrp4k push-hf --repo Cuong2004/HRP4K --path outputs/ --token ${HF_TOKEN}
```

---

## 🟦 9. Thực Thi Trên Kaggle Notebooks (GPU T4 / P100)

### Cell 1: Khởi Tạo Môi Trường & Dataset
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

### Cell 2: Huấn Luyện Scout & Detector Song Song Trên Dual GPU (GPU 0 & GPU 1)
```bash
%%bash
# [GPU 0]: Huấn luyện MobileNetV3-Small Scout (50 Epochs)
(
  echo "🚀 [GPU 0] Bắt đầu Huấn luyện Region Scout..."
  CUDA_VISIBLE_DEVICES=0 hrp4k train-scout \
    --data HRP4K \
    --output outputs/runs/scout \
    --epochs 50 \
    --batch 16 \
    --lr 0.001 \
    --lambda-cov 2.0 \
    --device 0
  echo "✅ [GPU 0] Hoàn tất Huấn luyện Region Scout!"
) &

# [GPU 1]: Huấn luyện YOLO11n-P2-lite Stage 1 (Full-Image Baseline)
(
  echo "🚀 [GPU 1] Bắt đầu Huấn luyện YOLO11n-P2-lite Stage 1..."
  CUDA_VISIBLE_DEVICES=1 hrp4k phase1 \
    --model yolo11n-p2-lite \
    --imgsz 960 \
    --batch 16 \
    --epochs 150 \
    --allow-full \
    --confidence 0.001 \
    --rect \
    --device 0 \
    --output outputs/runs/yolo11n_p2_lite_stage1
  echo "✅ [GPU 1] Hoàn tất Stage 1 Training!"
) &

wait
echo "🎉 HOÀN TẤT HUẤN LUYỆN SONG SONG TRÊN 2 GPU!"
```

### Cell 3: Chạy Toàn Bộ Suy Luận AdaPoth-Lite & Đánh Giá Metrics
```bash
!hrp4k phase2 --data HRP4K --split test \
  --weights outputs/runs/yolo11n_p2_lite_stage1/weights/best.pt \
  --method adapoth \
  --scout-weights outputs/runs/scout/weights/scout_best.pt \
  --k-max 4 \
  --output outputs/predictions/adapoth_lite_dynamic_k4.json

!hrp4k phase3 \
  --ground-truth HRP4K/test.json \
  --predictions outputs/predictions/adapoth_lite_dynamic_k4.json \
  --output outputs/metrics/adapoth_lite_metrics.json
```

### Cell 4: Nén Toàn Bộ Kết Quả & Tạo Link Tải Về Máy Tính
```python
import shutil
import os
from IPython.display import FileLink, display

output_zip = '/kaggle/working/hrp4k_results.zip' if os.path.exists('/kaggle/working') else 'outputs/hrp4k_results.zip'
shutil.make_archive(output_zip.replace('.zip', ''), 'zip', 'outputs')
print(f"✅ Đã nén thành công! Dung lượng: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
display(FileLink(output_zip))
```
