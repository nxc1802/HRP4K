# 🗺️ LỘ TRÌNH THỰC NGHIỆM ĐẦY ĐỦ CHO BÀI BÁO (RESEARCH PAPER ROADMAP)

Tài liệu này định hình ma trận nghiên cứu, các bảng số liệu khoa học và danh mục các thực nghiệm cần hoàn thiện để xây dựng một bài báo khoa học chất lượng cao (Publication-Grade) về phát hiện vật thể siêu nhỏ trên ảnh 4K độ phân giải cao (**High-Resolution 4K Road Defect Detection**).

---

## 🎯 1. Khung Nghiên Cứu 5 Trụ Cột (Core 5-Paradigm Matrix)

```
                                  MA TRẬN 5 TRỤ CỘT NGHIÊN CỨU
 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ [Trụ Cột 1] NATIVE 4K END-TO-END (Upper Bound): Train 4K ──► Inference 4K                     │
 ├────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Trụ Cột 2] RESIZE BEFORE TRAINING (Low-Res Baseline): Nén 640/1280 ──► Train ──► Inference   │
 ├────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Trụ Cột 3] CROP BEFORE TRAINING (Patch Training): Cắt Tiles 640 ──► Train ──► SAHI Inference │
 ├────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Trụ Cột 4] INFERENCE-TIME ALLOCATION: Model 4K ──► Sliced-NMS / Perspective-Grid / SAHI       │
 ├────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ [Trụ Cột 5] LEARNED SOTA WARPING: Learned 2D Deformation Grid (ZoomDet) ──► 1-Pass Inference  │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 2. Bảng Thiết Kế Kết Quả Chính Của Bài Báo (Target Tables for Paper)

### 📑 TABLE 1: Bảng Đối Sánh Tổng Thể Toàn Bộ 5 Trụ Cột (Main Benchmark Table)

Bảng này so sánh toàn diện các hướng tiếp cận xử lý ảnh 4K trên tập Test HRP4K (900 ảnh):

| Trụ Cột Nghiên Cứu | Mô Hình | Cơ Chế Huấn Luyện | Cơ Chế Suy Luận | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ | Recall | Latency | GFLOPs / Calls | Trạng Thái |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Native 4K e2e** | YOLO11m | 4K ($3840 \times 2176$) | Native 4K | **$55.05\%$** | **$33.27\%$** | **$49.19\%$** | **$27.3\text{ ms}$** | $68.3\text{G} / 1\text{ call}$ | ✅ **Đã có** |
|  | YOLOv8m | 4K ($3840 \times 2176$) | Native 4K | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | $\approx 30\text{ ms}$ | $78.9\text{G} / 1\text{ call}$ | ⏳ **Cần chạy** |
|  | YOLOv5m | 4K ($3840 \times 2176$) | Native 4K | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | $\approx 26\text{ ms}$ | $49.0\text{G} / 1\text{ call}$ | ⏳ **Cần chạy** |
|  | RT-DETRv2 | 4K ($3840 \times 2176$) | Native 4K | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | $\approx 45\text{ ms}$ | $136\text{G} / 1\text{ call}$ | ⏳ **Cần chạy** |
| **2. Resize-Pre** | YOLO11m-640 | Nén 640 | Resize 640 | $37.27\%$ | $18.32\%$ | $35.06\%$ | $8.2\text{ ms}$ | $68.3\text{G} / 1\text{ call}$ | ✅ **Đã có** |
|  | YOLOv8m-640 | Nén 640 | Resize 640 | $36.15\%$ | $17.86\%$ | $35.54\%$ | $8.5\text{ ms}$ | $78.9\text{G} / 1\text{ call}$ | ✅ **Đã có** |
|  | YOLOv5m-640 | Nén 640 | Resize 640 | $35.94\%$ | $18.07\%$ | $34.53\%$ | $7.8\text{ ms}$ | $49.0\text{G} / 1\text{ call}$ | ✅ **Đã có** |
|  | YOLO11m-1280 | Nén 1280 | Resize 1280 | $48.98\%$ | $25.40\%$ | $45.50\%$ | $14.6\text{ ms}$ | $68.3\text{G} / 1\text{ call}$ | ✅ **Đã có** |
| **3. Patch-Train** | YOLO11m-Patch | Cắt Tiles 640 | Sliced / SAHI | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | $\approx 900\text{ ms}$ | $15\text{ calls}$ | ⏳ **Cần chạy** |
| **4. Inference Slicing** | YOLO11m-4K | Train 4K e2e | Perspective-Grid | $42.02\%$ | $25.40\%$ | $38.00\%$ | $830.6\text{ ms}$ | **$9\text{ calls}$** | ✅ **Đã có** |
|  | YOLO11m-4K | Train 4K e2e | Sliced-NMS | $36.88\%$ | $22.84\%$ | $33.12\%$ | $912.6\text{ ms}$ | $25\text{ calls}$ | ✅ **Đã có** |
|  | YOLO11m-4K | Train 4K e2e | SAHI | $42.80\%$ | $26.24\%$ | $37.13\%$ | $1054.7\text{ ms}$ | $15\text{ calls}$ | ✅ **Đã có** |
| **5. SOTA Warping** | ZoomDet + YOLO11m | Learned 2D Warp | 1-Pass Warp | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | $\approx 35\text{ ms}$ | **$1\text{ call}$** | ⏳ **Cần chạy** |

---

### 📑 TABLE 2: Phân Tích Độ Nhạy Theo Dải Kích Thước (Scale Bins Breakdown)

Phân tích sâu khả năng phát hiện ổ gà từ siêu nhỏ (ở xa chân trời) đến lớn (ở gần đầu xe):

| Phương Pháp / Cấu Hình | Ultra-Fine ($<16\text{ px}$, 472 GT) | Fine ($16-32\text{ px}$, 169 GT) | Medium ($32-64\text{ px}$, 147 GT) | Large ($>64\text{ px}$, 133 GT) |
| :--- | :---: | :---: | :---: | :---: |
| **YOLO11m @ 640 (Resize-Pre)** | $24.10\%$ | $38.20\%$ | $41.50\%$ | $46.80\%$ |
| **YOLO11m @ 1280 (Resize-Pre)** | $38.40\%$ | $51.20\%$ | $54.10\%$ | $52.30\%$ |
| **YOLO11m @ 4K (Native e2e)** 👑 | **$\mathbf{48.13\%}$** | **$\mathbf{46.17\%}$** | **$\mathbf{41.61\%}$** | **$\mathbf{14.82\%}$** |
| **Perspective-Grid (Inference Allocation)** | $31.26\%$ | $37.07\%$ | $33.76\%$ | $14.57\%$ |
| **SAHI (Inference Slicing)** | $34.35\%$ | $42.24\%$ | $36.08\%$ | $10.74\%$ |
| **Patch-Train 640 + SAHI** | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* |
| **ZoomDet (Learned 2D Warp)** | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* | *[Cần chạy]* |

---

### 📑 TABLE 3: Các Nghiên Cứu Triệt Tiêu (Ablation Studies)

1. **Ablation 1: Ảnh hưởng của Tỷ lệ Chồng Lấp (Overlap Ratio):**
   - Đánh giá `overlap = 0.0, 0.1, 0.2, 0.3` trên `Perspective-Grid` và `Sliced-NMS`.
2. **Ablation 2: Kích thước Tile Lát Cắt (Tile Size):**
   - Đánh giá `tile_size = 640, 960, 1280` trên hiệu năng mAP vs Latency.
3. **Ablation 3: Chiến lược Phân Vùng Phối Cảnh (Perspective Boundaries):**
   - So sánh phân vùng đều (Uniform 3x3) vs Phối cảnh (Far: 4 cols, Mid: 3 cols, Near: 2 cols).
4. **Ablation 4: Kỹ thuật Hợp Nhất Hộp (Fusion Method):**
   - So sánh NMS chuẩn vs NMM (Non-Maximum Merging có trọng số score).

---

## 🚀 3. Danh Mục Các Thực Nghiệm Cần Chạy & Lệnh Thực Thi (Action Plan)

### Bước 1: Huấn Luyện Các Mô Hình 4K Bổ Sung (Hoàn thiện Trụ Cột 1)
Chạy trên GPU Server (RTX PRO 6000 96GB) để có đủ 4 kiến trúc đối sánh 4K:

```bash
# 1.1 Huấn luyện YOLOv8m Native 4K (150 Epochs)
hrp4k phase1 --model yolov8m --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolov8m_4k

# 1.2 Huấn luyện YOLOv5m Native 4K (150 Epochs)
hrp4k phase1 --model yolov5m-compat --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/yolov5m_4k

# 1.3 Huấn luyện RT-DETRv2 (Transformer SOTA) Native 4K (150 Epochs)
hrp4k phase1 --model rt-detr-v2 --imgsz original --batch 16 --epochs 150 --allow-full --confidence 0.001 --rect --output outputs/runs/rtdetr_v2_4k
```

---

### Bước 2: Tạo Dataset Patches 640 & Huấn Luyện (Hoàn thiện Trụ Cột 3)

```bash
# 2.1 Cắt tập train/val 4K thành các patches 640x640 giữ nguyên pixel density
hrp4k prepare-patches --input HRP4K --tile-size 640 --overlap 0.2 --output outputs/dataset_patches_640

# 2.2 Huấn luyện YOLO11m trên tập patches 640
hrp4k phase1 --model yolo11m --data outputs/dataset_patches_640 --imgsz 640 --batch 16 --epochs 150 --allow-full --output outputs/runs/yolo11m_patch640

# 2.3 Đánh giá mô hình Patch bằng SAHI trên tập Test 4K
hrp4k phase2 --method sahi --weights outputs/runs/yolo11m_patch640/weights/best.pt --tile-size 640 --overlap 0.2 --output outputs/predictions/yolo11m_patch_sahi.json
```

---

### Bước 3: Tích Hợp & Huấn Luyện SOTA Learned Warping (Hoàn thiện Trụ Cột 5 - `ZoomDet`)

```bash
# 3.1 Huấn luyện mạng ZoomDet học lưới biến dạng 2D liên tục
hrp4k train-zoomdet --data HRP4K --backbone yolo11m --canvas-size 640 --epochs 150 --output outputs/runs/zoomdet_yolo11m

# 3.2 Suy luận 1-Pass Warp trên tập Test 4K
hrp4k phase2 --method zoomdet --weights outputs/runs/zoomdet_yolo11m/weights/best.pt --output outputs/predictions/zoomdet_test.json
```

---

## 🏆 4. Bảng Theo Dõi Tiến Độ Toàn Dự Án

| Nhóm Thực Nghiệm | Số lượng runs | Đã hoàn thành | Còn lại | Tiến độ |
| :--- | :---: | :---: | :---: | :---: |
| **Trụ cột 1: Native 4K e2e** | 4 models | 1 (`yolo11m_4k`) | 3 (`v8m`, `v5m`, `rtdetr`) | **$25\%$** |
| **Trụ cột 2: Resize-Pre (640 & 1280)** | 4 models | 4 (`11m-640`, `v8m-640`, `v5m-640`, `11m-1280`) | 0 | **$100\%$** ✅ |
| **Trụ cột 3: Patch-Train 640** | 1 pipeline | 0 | 1 (`yolo11m_patch640`) | **$0\%$** |
| **Trụ cột 4: Inference Allocation** | 4 methods | 4 (`resize`, `sliced`, `perspective`, `sahi`) | 0 | **$100\%$** ✅ |
| **Trụ cột 5: Learned Warping (ZoomDet)** | 1 model | 0 | 1 (`zoomdet_yolo11m`) | **$0\%$** |
| **Ablation Studies** | 4 studies | 0 | 4 studies | **$0\%$** |
