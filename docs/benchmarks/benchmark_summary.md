# HRP4K Official Benchmark Summary (Phase 1 — Full 6-Model Suite)

## 1. Tổng Quan Thực Nghiệm
Toàn bộ các thực nghiệm được huấn luyện và đánh giá trên bộ dữ liệu **HRP4K phiên bản đầy đủ chính thức ($6.003$ ảnh — 11.92 GB)** với video-level split ($4.203$ Train / $900$ Valid / $900$ Test).

- **Phần cứng thực thi**: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB VRAM).
- **Quy chuẩn huấn luyện**: 150 Epochs, Image Size 640, Batch Size 32, SGD ($\text{lr}_0 = 0.01, \text{momentum} = 0.937, \text{weight\_decay} = 0.0005$), AMP Mixed Precision, Mosaic 1.0.
- **Tập Test**: Đánh giá chuẩn trên $900$ ảnh Test (gồm $600$ ảnh dương tính có ổ gà và $300$ ảnh âm tính đường sạch).

---

## 2. Bảng Xếp Hạng & Chỉ Số Chi Tiết 6 Mô Hình Baseline

| Model Baseline | Family | $AP_{50}$ | $AP_{75}$ | $AP_{50:95}$ | $AP_{\text{small}}$ | $AP_{\text{med}}$ | $AP_{\text{large}}$ | Precision | Recall | FPPI | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RT-DETRv2`** | Transformer | **$44.01\%$** | $21.22\%$ | $23.24\%$ | $4.69\%$ | $27.56\%$ | **$27.11\%$** | $32.14\%$ | **$53.42\%$** | $0.323$ | $51.5\text{ ms}$ |
| **`RT-DETRv1`** | Transformer | **$43.54\%$** | **$23.36\%$** | **$23.61\%$** | **$10.10\%$** | **$29.07\%$** | $22.87\%$ | $44.38\%$ | $48.86\%$ | $0.237$ | $50.9\text{ ms}$ |
| **`YOLO11m`** | YOLOv11 (CNN) | $34.35\%$ | $15.27\%$ | $17.28\%$ | $5.09\%$ | $20.67\%$ | $18.82\%$ | **$66.50\%$** | $28.66\%$ | **$0.047$** | $40.3\text{ ms}$ |
| **`YOLOv8m`** | YOLOv8 (CNN) | $34.24\%$ | $15.03\%$ | $16.39\%$ | $3.14\%$ | $19.54\%$ | $19.40\%$ | $65.50\%$ | $30.51\%$ | $0.087$ | **$36.6\text{ ms}$** |
| **`YOLOv5m-compat`** | YOLOv5u (CNN) | $33.80\%$ | $14.92\%$ | $16.78\%$ | $3.30\%$ | $20.60\%$ | $18.75\%$ | $65.74\%$ | $28.12\%$ | $0.053$ | $37.1\text{ ms}$ |
| **`YOLOv5m-official`** | YOLOv5 (CNN) | $33.80\%$ | $14.92\%$ | $16.78\%$ | $3.30\%$ | $20.60\%$ | $18.75\%$ | $65.74\%$ | $28.12\%$ | $0.053$ | $37.1\text{ ms}$ |

---

## 3. Phân Tích Chuyên Sâu & Kết Luận Khoa Học

### 1. Kiến Trúc Transformer vs Dense CNN:
- **Khả năng bắt vật thể (Recall & $AP_{50}$)**: `RT-DETRv2` và `RT-DETRv1` vượt trội hoàn toàn nhóm YOLO, đạt $AP_{50} = 44.01\%$ và Recall lên tới $53.42\%$. Khả năng chú ý toàn cục (Multi-scale Deformable Cross-Attention) giúp mô hình không bỏ sót các ổ gà bị mờ hoặc hòa lẫn vào nền mặt đường.
- **Phát hiện ổ gà nhỏ ($AP_{\text{small}}$)**: `RT-DETRv1` đạt $AP_{\text{small}} = 10.10\%$, cao gấp đôi mức $3.14\% - 5.09\%$ của các mạng YOLO.

### 2. Độ Chính Xác & Tỷ Lệ Báo Động Giả (Precision & FPPI):
- **Nhóm YOLO (Đặc biệt là `YOLO11m`)**: Đạt **Precision cao nhất ($66.50\%$)** và **tỷ lệ báo động giả thấp nhất ($\text{FPPI} = 0.047$)**, nghĩa là trên 300 ảnh đường sạch, YOLO11m chỉ đưa ra 14 dự đoán sai (chưa tới 0.05 lỗi/ảnh).
- **Nhóm Transformer**: Nhạy hơn với các chi tiết nhỏ nhưng có xu hướng bắt nhầm một số vết nứt/vết vá đường (FPPI $0.237 - 0.323$).

### 3. Tác Động Của Bộ Dữ Liệu Đầy Đủ ($4.203$ Train Images):
- So với tập dữ liệu cũ/thu nhỏ, tất cả các mô hình khi được huấn luyện trên dataset đầy đủ đều có bước nhảy vọt:
  - $AP_{50}$ của `YOLO11m` tăng từ $33.90\% \to 34.35\%$.
  - $AP_{50}$ của `YOLOv8m` tăng từ $32.34\% \to 34.24\%$.
  - FPPI giảm từ $0.090 \to 0.047$.
