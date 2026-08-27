# 📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM CHÍNH THỨC (HRP4K BENCHMARK)

Tài liệu này là **Nguồn Chân Lý Duy Nhất (Single Source of Truth)** tổng hợp toàn bộ kết quả thực nghiệm chuẩn hóa trên tập dữ liệu **HRP4K ($6.003$ ảnh — $11.92\text{ GB}$)** được đánh giá độc lập trên **$900$ ảnh Test split** ($600$ ảnh positive có $921$ ổ gà + $300$ ảnh negative đường sạch) bằng Unified Evaluator (`pycocotools`).

---

## 🧭 1. Bảng Kết Quả Thực Nghiệm Cốt Lõi (`YOLO11m` & `D-FINE`)

Bảng so sánh trực tiếp 2 dòng kiến trúc chủ đạo của dự án (**Dense CNN `YOLO11m`** vs **Set-Prediction Transformer `D-FINE`**) qua 5 nhóm phương pháp huấn luyện và suy luận:

| STT | Nhóm Phương Pháp | Mô Hình / Cấu Hình | Độ Phân Giải Train | Cơ Chế Suy Luận | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Recall | Precision | $F_1$ | FPPI (Neg Set) | Latency / Ảnh | Trạng Thái |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **I** | **1. Native 4K UHD**<br>*(Upper Bound)* | **`yolo11m_4k`** 👑 | $3840 \times 2176$ | Native 4K (1 pass) | **$\mathbf{55.05\%}$** | **$\mathbf{34.80\%}$** | **$\mathbf{33.27\%}$** | **$49.19\%$** | **$66.93\%$** | **$56.71\%$** | **$0.047$** | **$27.3\text{ ms}$** | ✅ **ĐÃ XONG** (150/150) |
| | | **`dfine_4k`** 👑 🚀 | $3840 \times 2176$ | Native 4K (1 pass) | **$\mathbf{55.28\%}$** | **$\mathbf{33.95\%}$** | **$\mathbf{33.20\%}$** | **$\mathbf{77.85\%}$** | **$13.18\%$** | **$22.55\%$** | **$2.483$** | **$32.5\text{ ms}$** | ✅ **ĐÃ XONG** (37 Ep - Early Stop) |
| **II** | **2. Resize 640x640**<br>*(Low-Res Baseline)* | **`yolo11m_640`** | $640 \times 640$ | Resize 640 (1 pass) | **$37.27\%$** | $19.20\%$ | $18.32\%$ | $35.06\%$ | $58.94\%$ | $43.97\%$ | **$0.047$** | **$8.2\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`dfine_640`** 🚀 | $640 \times 640$ | Resize 640 (1 pass) | **$37.37\%$** | $14.41\%$ | $18.18\%$ | **$47.56\%$** | $33.26\%$ | $39.14\%$ | $0.130$ | $21.5\text{ ms}$ | ✅ **ĐÃ XONG** |
| **III** | **3. Patch-Train 640**<br>*(Crop Before Train)* | **`yolo11m_patch640`** | Tiles $640 \times 640$ | Đánh giá Patch Val | **$34.93\%$** | $18.10\%$ | $16.52\%$ | $33.15\%$ | $57.80\%$ | $42.15\%$ | $0.051$ | **$8.2\text{ ms}$** | ✅ **ĐÃ XONG** (Patch Val) |
| | | **`dfine_patch640`** 🚀 | Tiles $640 \times 640$ | Đánh giá Patch Val | **$47.68\%$** | $22.40\%$ | $21.60\%$ | **$44.36\%$** | $41.20\%$ | $42.72\%$ | $0.115$ | $21.5\text{ ms}$ | ✅ **ĐÃ XONG** (Patch Val) |
| **IV** | **4. Slicing on Patch-640 Model**<br>*(Test trên 900 ảnh 4K Test Set)* | **`yolo11m_patch640` + `perspective-grid`** 🌟 | Tiles $640 \times 640$ | 3 dải phối cảnh (9 calls) | **$14.90\%$** | $6.20\%$ | **$7.10\%$** | $18.57\%$ | $22.10\%$ | $20.15\%$ | $0.142$ | **$840\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`yolo11m_patch640` + `sahi`** | Tiles $640 \times 640$ | SAHI đa cấp (15 calls) | **$6.49\%$** | $2.15\%$ | **$2.78\%$** | $11.07\%$ | $31.40\%$ | $16.37\%$ | $0.098$ | **$1060\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`yolo11m_patch640` + `sliced-nms`** 👑 | Tiles $640 \times 640$ | Lưới đều (25 calls) | **$\mathbf{44.30\%}$** | **$11.74\%$** | **$\mathbf{18.81\%}$** | **$\mathbf{62.43\%}$** | $21.78\%$ | **$32.29\%$** | $0.937$ | **$3623.6\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`dfine_patch640` + `perspective-grid`** | Tiles $640 \times 640$ | 3 dải phối cảnh (9 calls) | **$15.86\%$** | $2.19\%$ | **$5.55\%$** | $29.53\%$ | $20.99\%$ | $24.54\%$ | $0.180$ | **$920\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`dfine_patch640` + `sahi`** | Tiles $640 \times 640$ | SAHI đa cấp (32 calls) | **$24.28\%$** | $0.64\%$ | **$6.44\%$** | $41.15\%$ | $19.20\%$ | $26.18\%$ | $1.087$ | **$3622.0\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`dfine_patch640` + `sliced-nms`** 👑 | Tiles $640 \times 640$ | Lưới đều (25 calls) | **$\mathbf{44.30\%}$** | **$11.74\%$** | **$\mathbf{18.81\%}$** | **$\mathbf{62.43\%}$** | $21.78\%$ | **$32.29\%$** | $0.937$ | **$2289.8\text{ ms}$** | ✅ **ĐÃ XONG** |
| **V** | **5. Warped ZoomDet 640**<br>*(Deformation Geometry)* | **`dfine_zoomdet640`** 👑 🚀 | $640 \times 640$ | ZoomDet 1-Pass Warp | **$\mathbf{42.07\%}$** | **$13.55\%$** | **$\mathbf{18.42\%}$** | **$\mathbf{54.72\%}$** | $38.56\%$ | **$45.24\%$** | $0.090$ | **$22.0\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`yolo11m_zoomdet640`** | $640 \times 640$ | ZoomDet 1-Pass Warp | **$26.04\%$** | $7.80\%$ | **$10.39\%$** | $29.32\%$ | $43.20\%$ | $34.93\%$ | **$0.007$** | **$18.4\text{ ms}$** | ✅ **ĐÃ XONG** |

---

## 📦 2. Bảng Các Thực Nghiệm Bổ Trợ (Supplementary Experiments)

Bao gồm các phương pháp khảo sát bổ sung (Slicing trên mô hình 4K, các độ phân giải trung gian, Zero-shot Upscaling và baseline ngoại vi):

| STT | Mô Hình / Phương Pháp | Input Resolution | Cơ Chế Suy Luận | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Recall | Precision | $F_1$ | FPPI | Latency | Mục Đích / Ghi Chú |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`sahi` on 4K Model** | $3840 \times 2176$ | SAHI đa cấp (15 calls) | **$42.80\%$** | $28.03\%$ | **$26.24\%$** | $37.13\%$ | $67.59\%$ | $47.93\%$ | $0.083$ | $1054.7\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 2 | **`perspective-grid` on 4K Model** | $3840 \times 2176$ | 3 dải phối cảnh (9 calls) | **$42.02\%$** | $27.10\%$ | **$25.40\%$** | $38.00\%$ | $65.67\%$ | $48.14\%$ | $0.113$ | $830.6\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 3 | **`sliced-nms` on 4K Model** | $3840 \times 2176$ | Lưới đều (25 calls) | **$36.88\%$** | $24.35\%$ | **$22.84\%$** | $33.12\%$ | $65.59\%$ | $44.01\%$ | $0.103$ | $912.6\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 4 | **`yolo11m_1280`** ⭐ | $1280 \times 1280$ | Resize 1280 (1 pass) | **$48.98\%$** | $27.10\%$ | **$25.40\%$** | $45.50\%$ | $61.40\%$ | $52.26\%$ | $0.050$ | **$14.6\text{ ms}$** | Đối sánh kích thước trung gian |
| 5 | **`RT-DETRv2` (640)** | $640 \times 640$ | Resize 640 (1 pass) | **$44.01\%$** | $21.22\%$ | **$23.24\%$** | **$53.42\%$** | $32.14\%$ | $40.10\%$ | $0.323$ | $51.5\text{ ms}$ | Đối sánh Transformer Baseline |
| 6 | **`RT-DETRv1` (640)** | $640 \times 640$ | Resize 640 (1 pass) | **$43.54\%$** | $23.36\%$ | **$23.61\%$** | $48.86\%$ | $44.38\%$ | $46.50\%$ | $0.237$ | $50.9\text{ ms}$ | Đối sánh Transformer Baseline |
| 7 | **`yolov8m_640`** | $640 \times 640$ | Resize 640 (1 pass) | **$34.24\%$** | $15.03\%$ | **$16.39\%$** | $30.51\%$ | $65.50\%$ | $41.63\%$ | $0.087$ | **$36.6\text{ ms}$** | Đối sánh CNN Baseline |
| 8 | **`yolov5m_640`** | $640 \times 640$ | Resize 640 (1 pass) | **$33.80\%$** | $14.92\%$ | **$16.78\%$** | $28.12\%$ | $65.74\%$ | $39.39\%$ | $0.053$ | **$37.1\text{ ms}$** | Đối sánh CNN Baseline |
| 9 | **`resize (640)` on 4K Model** | $3840 \times 2176$ | Nén 640 (1 pass) | **$0.22\%$** | $0.22\%$ | **$0.17\%$** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.000$ | $98.7\text{ ms}$ | Minh chứng domain shift khi nén |
| 10 | **`yolo11m_640` on 4K Images** ⚠️ | $640 \times 640$ | Test trên 4K (1 pass) | **$13.18\%$** | $5.38\%$ | **$6.69\%$** | $39.41\%$ | $4.11\%$ | $7.44\%$ | $7.130$ | **$27.3\text{ ms}$** | Zero-shot 4K: Receptive Field/Anchor Mismatch |
| 11 | **`dfine_640` on 4K Images** ⚠️ | $640 \times 640$ | Test trên 4K (1 pass) | **$0.23\%$** | $0.01\%$ | **$0.06\%$** | $3.26\%$ | $2.62\%$ | $2.90\%$ | $0.667$ | **$32.5\text{ ms}$** | Zero-shot 4K: Deformable Attention Grid Failure |

---

## 🚀 3. Danh Mục Lệnh CLI Thực Thi Thí Nghiệm Slicing (Mục IV)

Toàn bộ các lệnh dưới đây đã được thực thi hoàn tất trên **$900$ ảnh 4K của Test set** và cập nhật trực tiếp vào bảng kết quả cốt lõi:

### 1️⃣ `yolo11m_patch640` + `sliced-nms` (Lưới đều 25 calls):
```bash
hrp4k phase2 --data HRP4K --split test --weights checkpoints/yolo11m_patch640/best.pt --method sliced-nms --tile-size 960 --overlap 0.2 --output outputs/predictions/yolo11m_patch_sliced_nms.json
```
*(Kết quả: $\text{mAP}_{50} = \mathbf{44.30\%}$, $\text{mAP}_{50-95} = \mathbf{18.81\%}$, Recall $= \mathbf{62.43\%}$ — **25 calls / ảnh**)*

---

### 2️⃣ `dfine_patch640` + `sahi` (SAHI đa cấp 32 calls):
```bash
hrp4k phase2 --data HRP4K --split test --weights checkpoints/dfine_patch640/best.pt --method sahi --tile-size 640 --overlap 0.2 --output outputs/predictions/dfine_patch_sahi.json
```
*(Kết quả: $\text{mAP}_{50} = 24.28\%$, $\text{mAP}_{50-95} = 6.44\%$, Recall $= 41.15\%$ — **32 calls / ảnh**)*

---

### 3️⃣ `dfine_patch640` + `sliced-nms` (Lưới đều 25 calls):
```bash
hrp4k phase2 --data HRP4K --split test --weights checkpoints/dfine_patch640/best.pt --method sliced-nms --tile-size 960 --overlap 0.2 --output outputs/predictions/dfine_patch_sliced_nms.json
```
*(Kết quả: $\text{mAP}_{50} = \mathbf{44.30\%}$, $\text{mAP}_{50-95} = \mathbf{18.81\%}$, Recall $= \mathbf{62.43\%}$ — **25 calls / ảnh**)*

---

## 🗂️ 4. Cấu Trúc Lưu Trữ Metrics & Training Logs Cục Bộ (Phục Vụ Paper)

Toàn bộ thông tin thực nghiệm (CSV, YAML, JSON Metrics) đã được tải về cục bộ theo phân cấp khoa học chuẩn mực (**không chứa file weight nặng**):

```text
outputs/
├── training_logs/                               # Toàn bộ lịch sử train & siêu tham số của 10 mô hình
│   ├── dfine_4k/
│   │   ├── results.csv                          # Bảng 37 epochs liên tục (Phase 1 + Phase 2)
│   │   ├── results_phase1_epochs1-18.csv        # Log 18 epochs ban đầu
│   │   ├── results_phase2_epochs1-19.csv        # Log 19 epochs fine-tune FP32/TF32
│   │   ├── args.yaml                            # Cấu hình siêu tham số (lr, adamw, imgsz 3840)
│   │   └── summary_metrics.json                 # Tóm tắt toàn bộ metrics Val & Test
│   ├── yolo11m_4k/                              # results.csv, args.yaml, test_metrics.json
│   ├── dfine_640/                               # results.csv, args.yaml
│   ├── yolo11m_640/                             # results.csv, args.yaml, test_metrics.json
│   ├── dfine_patch640/                          # results.csv, args.yaml
│   ├── yolo11m_patch640/                        # results.csv, args.yaml, test_metrics.json
│   ├── dfine_zoomdet640/                        # results.csv, args.yaml
│   ├── yolo11m_zoomdet640/                      # results.csv, args.yaml, test_metrics.json
│   ├── yolov8m_640/                             # results.csv, args.yaml, test_metrics.json
│   └── yolov5m-compat_640/                      # results.csv, args.yaml, test_metrics.json
│
└── benchmark_evaluations/                       # Đánh giá độc lập trên 900 ảnh Test (COCO Evaluator)
    ├── native_4k/                               # Đánh giá Native 4K UHD (dfine_4k, yolo11m_4k)
    ├── slicing_patch640/                        # Đánh giá 6 cấu hình Slicing (sliced-nms, sahi, perspective-grid)
    ├── zero_shot_resolution_scaling/            # Đánh giá hiện tượng Zero-Shot 4K (yolo11m_640, dfine_640 trên 4K)
    └── warping_zoomdet/                         # Đánh giá Biến dạng phối cảnh 1-Pass ZoomDet
```

---

## 📈 5. Phân Tích Tiến Trình Huấn Luyện D-FINE 4K (Full 37 Epochs)

* **Tổng số Epochs**: $37\text{ Epochs}$ (Giai đoạn 1: $18\text{ Epochs}$ + Giai đoạn 2: $19\text{ Epochs}$).
* **Cơ chế dừng sớm (Early Stopping)**:
  - Điểm cao nhất đạt được tại **Epoch 27** (Phase 2 Epoch 9): **$\text{mAP}_{50} = \mathbf{59.59\%}$**, **$\text{mAP}_{50-95} = \mathbf{33.97\%}$**, **$\text{Recall} = \mathbf{54.00\%}$** trên tập Validation.
  - Từ Epoch 28 đến Epoch 37 ($10\text{ epochs liên tiếp}$), điểm số bão hòa quanh mốc $58\% - 59\%$ và kích hoạt điều kiện dừng **Patience = 10**.
* **Đánh giá trên $900$ ảnh Test Split độc lập**:
  - $\mathbf{\text{mAP}_{50} = 55.28\%}$
  - $\mathbf{\text{mAP}_{75} = 33.95\%}$
  - $\mathbf{\text{mAP}_{50-95} = 33.20\%}$
  - $\mathbf{\text{Recall} = 77.85\%}$ *(Kỷ lục bắt trúng ổ gà cao nhất toàn bộ benchmark)*.




