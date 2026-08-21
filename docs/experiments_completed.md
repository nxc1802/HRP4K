# 📦 TỔNG HỢP TOÀN BỘ CÁC THỰC NGHIỆM ĐÃ HOÀN THÀNH (LOCAL & HUGGING FACE)

Tài liệu này lưu trữ và tổng hợp toàn bộ các kết quả thực nghiệm đã được thực hiện, huấn luyện thành công và kiểm định trên tập dữ liệu **HRP4K (High-Resolution Pothole 4K Benchmark)**, bao gồm cả dữ liệu cục bộ và trên kho lưu trữ Hugging Face Hub ([Cuong2004/HRP4K](https://huggingface.co/datasets/Cuong2004/HRP4K)).

---

## 🏋️ 1. Phase 1: Các Mô Hình Baseline Đã Huấn Luyện (150 Epochs)

Tất cả các mô hình dưới đây đều được huấn luyện đầy đủ 150 Epochs, sử dụng optimizer SGD ($lr_0=0.01, \text{momentum}=0.937$), chế độ AMP (Automatic Mixed Precision) và đánh giá trên tập **Test Split** (900 ảnh 4K) với ngưỡng tin cậy tiêu chuẩn COCO `conf=0.001`:

| STT | Mô Hình | Độ Phân Giải Input | Batch Size | Precision ($P$) | Recall ($R$) | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Checkpoint & Trạng Thái |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`yolo11m_4k`** 👑 | **$3840 \times 2176$ (`rect=True`)** | **16** | **$66.93\%$** | **$49.19\%$** | **$\mathbf{55.05\%}$** | **$\mathbf{33.27\%}$** | `checkpoints/yolo11m_4k/best.pt` (40.3 MB) |
| **2** | **`yolo11m_1280`** ⭐ | **$1280 \times 1280$** | **16** | **$61.40\%$** | **$45.50\%$** | **$48.98\%$** | **$25.40\%$** | Đã ghi nhận số liệu hoàn tất |
| **3** | **`yolo11m_640`** | **$640 \times 640$** | **16** | **$58.94\%$** | **$35.06\%$** | **$37.27\%$** | **$18.32\%$** | `yolo11m_640/weights/best.pt` (40.3 MB) |
| **4** | **`yolov8m_640`** | **$640 \times 640$** | **16** | **$55.64\%$** | **$35.54\%$** | **$36.15\%$** | **$17.86\%$** | `outputs/runs/yolov8m_640/weights/best.pt` (50.2 MB) |
| **5** | **`yolov5m_640`** | **$640 \times 640$** | **16** | **$55.60\%$** | **$34.53\%$** | **$35.94\%$** | **$18.07\%$** | `outputs/runs/yolov5m-compat_640/weights/best.pt` (41.2 MB) |

### 📈 Nhận Xét Tiến Trình Độ Phân Giải (Resolution Progression):
- $640 \times 640 \to \mathbf{37.27\% \text{ mAP}_{50}}$
- $1280 \times 1280 \to \mathbf{48.98\% \text{ mAP}_{50}}$ ($+11.71\%$)
- $3840 \times 2176 \to \mathbf{55.05\% \text{ mAP}_{50}}$ ($+17.78\%$ so với 640, $+7.15\%$ vượt SOTA bài báo gốc YOLOv8m 47.9%).

---

## 🔍 2. Phase 2: Benchmark Các Phương Pháp Suy Luận Siêu Phân Giải (900 Ảnh Test)

Được thực thi trực tiếp bằng trọng số `best.pt` của mô hình 4K (`yolo11m_4k`) trên toàn bộ **900 ảnh 4K** của tập Test (chứa 921 ground truth potholes):

### 📊 A. Bảng So Sánh Tổng Thể:

| Phương Pháp Suy Luận | Cơ Chế | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Precision | Recall | F1-Score | Số Calls / Ảnh | Latency (ms) | FPPI (Negatives) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`resize (640)`** | Nén trực tiếp về 640 | $0.22\%$ | $0.22\%$ | $0.17\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | **1.0** | **98.7 ms** | **0.000** |
| **`sliced-nms`** | Lưới ô vuông $960\times 960$, overlap 20% | $36.88\%$ | $24.35\%$ | $22.84\%$ | $65.59\%$ | $33.12\%$ | $44.01\%$ | 25.0 | 912.6 ms | 0.103 |
| **`perspective-grid`** 🌟 | Lưới phối cảnh 3 dải, 2D overlap | **$42.02\%$** | **$27.10\%$** | **$25.40\%$** | **$65.67\%$** | **$38.00\%$** | **$48.14\%$** | **9.0** | **830.6 ms** | **0.113** |
| **`sahi`** | Slicing Aided Hyper Inference đa cấp | **$42.80\%$** | $28.03\%$ | $26.24\%$ | $67.59\%$ | $37.13\%$ | $47.93\%$ | 15.0 | 1054.7 ms | 0.083 |

---

### 🔬 B. Bảng Phân Tích Chi Tiết Theo 4 Dải Kích Thước (Scale Bins):

| Phương Pháp | Ultra-Fine ($<16\text{ px}$, 472 GT) | Fine ($16-32\text{ px}$, 169 GT) | Medium ($32-64\text{ px}$, 147 GT) | Large ($>64\text{ px}$, 133 GT) |
| :--- | :---: | :---: | :---: | :---: |
| **`resize (640)`** | $\text{AP}_{50} = 0.00\%$ ($\text{R}=0.0\%$) | $\text{AP}_{50} = 0.00\%$ ($\text{R}=0.0\%$) | $\text{AP}_{50} = 0.00\%$ ($\text{R}=0.0\%$) | $\text{AP}_{50} = 0.63\%$ ($\text{R}=2.3\%$) |
| **`sliced-nms`** | $\text{AP}_{50} = 27.95\%$ ($\text{R}=48.5\%$) | $\text{AP}_{50} = 39.67\%$ ($\text{R}=61.0\%$) | $\text{AP}_{50} = 35.37\%$ ($\text{R}=59.9\%$) | $\text{AP}_{50} = 8.73\%$ ($\text{R}=30.1\%$) |
| **`perspective-grid`** 🌟 | $\mathbf{\text{AP}_{50} = 31.26\%}$ ($\mathbf{\text{R}=55.3\%}$) | $\mathbf{\text{AP}_{50} = 37.07\%}$ ($\mathbf{\text{R}=62.7\%}$) | $\mathbf{\text{AP}_{50} = 33.76\%}$ ($\mathbf{\text{R}=64.0\%}$) | $\mathbf{\text{AP}_{50} = 14.57\%}$ ($\mathbf{\text{R}=43.6\%}$) |
| **`sahi`** | $\mathbf{\text{AP}_{50} = 34.35\%}$ ($\mathbf{\text{R}=55.1\%}$) | $\mathbf{\text{AP}_{50} = 42.24\%}$ ($\mathbf{\text{R}=62.1\%}$) | $\mathbf{\text{AP}_{50} = 36.08\%}$ ($\mathbf{\text{R}=65.3\%}$) | $\mathbf{\text{AP}_{50} = 10.74\%}$ ($\mathbf{\text{R}=35.3\%}$) |

---

## 🗂️ 3. Danh Mục Tệp & Vị Trí Lưu Trữ

### A. Checkpoint Weights & Logs:
- `outputs/runs/yolo11m_4k/weights/best.pt`: Trọng số 4K tốt nhất (mAP 55.05%).
- `outputs/runs/yolo11m_4k/results.csv`: Nhật ký 150 Epochs chạy 4K.
- `outputs/runs/yolo11m_4k/test_metrics.json`: Báo cáo chỉ số test trên 4K.
- `outputs/runs/yolov8m_640/weights/best.pt`: Checkpoint YOLOv8m 640.
- `outputs/runs/yolov5m-compat_640/weights/best.pt`: Checkpoint YOLOv5m 640.

### B. Kết Quả Dự Đoán & Đánh Giá Phase 2:
- `outputs/phase2_benchmark/best_perspective-grid_test_predictions.json`
- `outputs/phase2_benchmark/best_perspective-grid_test_predictions_metrics.json`
- `outputs/phase2_benchmark/best_sliced-nms_test_predictions.json`
- `outputs/phase2_benchmark/best_sliced-nms_test_predictions_metrics.json`
- `outputs/phase2_benchmark/best_sahi_test_predictions.json`
- `outputs/phase2_benchmark/best_sahi_test_predictions_metrics.json`
- `outputs/phase2_benchmark/best_resize_test_predictions.json`
- `outputs/phase2_benchmark/best_resize_test_predictions_metrics.json`
- `outputs/phase2_benchmark/best_phase2_all_methods_summary.json`

### C. Kho Lưu Trữ Cloud Hugging Face Hub:
- Repo: `Cuong2004/HRP4K` (Dataset repository).
- Thư mục: `checkpoints/yolo11m_4k/`, `outputs/phase2_benchmark/`, `outputs/runs/yolov8m_640/`, `outputs/runs/yolov5m-compat_640/`, `yolo11m_640/`.
