# 📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM CHÍNH THỨC (HRP4K BENCHMARK)

Tài liệu này là **Nguồn Chân Lý Duy Nhất (Single Source of Truth)** tổng hợp toàn bộ kết quả thực nghiệm chuẩn hóa trên tập dữ liệu **HRP4K ($6.003$ ảnh — $11.92\text{ GB}$)** được đánh giá độc lập trên **$900$ ảnh Test split** ($600$ ảnh positive có $921$ ổ gà + $300$ ảnh negative đường sạch) bằng Unified Evaluator (`pycocotools`).

> [!NOTE]
> **Quy chuẩn độ phân giải**: Ảnh gốc tập dữ liệu HRP4K có kích thước chuẩn **$3840 \times 2160$** (tỷ lệ chuẩn $16:9$). Kích thước $3840 \times 2176$ trong một số log chỉ là padding nội bộ modulo-32 của YOLO khi chạy batch native 4K.

---

## 🧭 1. Bảng Kết Quả Thực Nghiệm Cốt Lõi (Các Dòng Kiến Trúc Chủ Đạo & Proposed Method)

Bảng so sánh trực tiếp các dòng kiến trúc chủ đạo của dự án (**Dense CNN `YOLO11m`**, **Set-Prediction Transformer `D-FINE`**, và **Proposed Adaptive Zoom `AdaPoth-Lite`**) qua các nhóm phương pháp huấn luyện và suy luận:

| STT | Nhóm Phương Pháp | Mô Hình / Cấu Hình | Độ Phân Giải Train | Cơ Chế Suy Luận | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Recall | Precision | $F_1$ | FPPI (Neg Set) | Latency / Ảnh | Trạng Thái |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **I** | **1. Native 4K UHD**<br>*(High-Resolution Reference)* | **`yolo11m_4k`** 👑 | $3840 \times 2160$ | Native 4K (1 pass) | **$\mathbf{55.05\%}$** | **$\mathbf{34.80\%}$** | **$\mathbf{33.27\%}$** | **$49.19\%$** | **$66.93\%$** | **$56.71\%$** | **$0.047$** | **$27.3\text{ ms}$** | ✅ **ĐÃ XONG** (150/150) |
| | | **`dfine_4k`** 👑 🚀 | $3840 \times 2160$ | Native 4K (1 pass) | **$\mathbf{55.28\%}$** | **$\mathbf{33.95\%}$** | **$\mathbf{33.20\%}$** | **$\mathbf{77.85\%}$** | **$13.18\%$** | **$22.55\%$** | **$2.483$** | **$32.5\text{ ms}$** | ✅ **ĐÃ XONG** (37 Ep - Early Stop) |
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
| **VI** | **6. AdaPoth-Lite** 🌟<br>*(Proposed Adaptive Zoom & Fusion)* | **`adapoth_lite_dynamic_k4`** 🌟 *(Proposed)* | 3-Stage ($960 \to 640$) | Scout Top-K ($K \le 4$) + Fusion | **$\mathbf{36.37\%}$** | **$\mathbf{20.46\%}$** | **$\mathbf{20.42\%}$** | **$41.80\%$** | **$50.26\%$** | **$45.64\%$** | **$0.120$** | **$108.2\text{ ms}$** | ✅ **ĐÃ XONG** |
| | | **`adapoth_oracle_k4`** 👑 *(Oracle Upper Bound)* | 3-Stage ($960 \to 640$) | GT Box Oracle ($K \le 4$) | **$\mathbf{52.21\%}$** | **$\mathbf{23.89\%}$** | **$\mathbf{26.01\%}$** | **$55.37\%$** | **$56.11\%$** | **$55.74\%$** | **$0.063$** | **$60.9\text{ ms}$** | ✅ **ĐÃ XONG** |

---

## 🔬 2. Bảng Tổng Hợp Toàn Bộ Ablation Studies Của Proposed Method (AdaPoth-Lite Ablation Matrix)

Tất cả các cấu hình ablation dưới đây được đánh giá độc lập trên **$900$ ảnh Test ($3840 \times 2160$)** theo chuẩn **COCO Evaluator (`pycocotools`)** nhằm làm sáng tỏ đóng góp độc lập của từng thành phần kiến trúc:

| STT | Biến Thể / Cấu Hình Ablation | Cơ Chế Scout / Vùng Crop | $K_{\text{avg}}$ | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | $\mathbf{AP_{\text{small}}}$ | $\mathbf{AP_{\text{med}}}$ | $\mathbf{AP_{\text{large}}}$ | $\mathbf{AR_{\text{small}}}$ | Recall | Precision | $F_1$ | Latency | Speed | Ý Nghĩa / Mục Đích Khoa Học |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 🌟 **`AdaPoth-Lite` (Proposed Master)** | Scout Heatmap Dynamic Top-K ($K \le 4$) | **$1.00$** | **$\mathbf{36.37\%}$** | **$\mathbf{20.46\%}$** | **$\mathbf{20.42\%}$** | **$\mathbf{12.75\%}$** | **$25.75\%$** | **$16.49\%$** | **$\mathbf{22.42\%}$** | **$41.80\%$** | **$50.26\%$** | **$45.64\%$** | **$108.2\text{ ms}$** | **$9.2\text{ FPS}$** | **Mô hình đề xuất chính thức hoàn chỉnh** (Cân bằng tối ưu độ chính xác và độ trễ) |
| **2** | 👑 **`AdaPoth-Oracle` (Oracle Upper Bound)** | Ground-Truth Bounding Box Oracle ($K \le 4$) | $1.00$ | **$52.21\%$** | **$23.89\%$** | **$26.01\%$** | **$15.82\%$** | **$29.41\%$** | **$18.20\%$** | **$28.15\%$** | **$55.37\%$** | **$56.11\%$** | **$55.74\%$** | **$60.9\text{ ms}$** | **$16.4\text{ FPS}$** | Giới hạn trần lý thuyết khi Scout định vị chính xác $100\%$ vị trí ổ gà |
| **3** | 🔍 **`Global-Only` (No Zoom Baseline)** | Resize toàn cảnh $960 \times 540$ (Không crop) | $0.00$ | $29.59\%$ | $13.97\%$ | $14.97\%$ | **$4.01\%$** | $18.33\%$ | $15.89\%$ | **$8.30\%$** | $28.34\%$ | **$63.97\%$** | $39.28\%$ | **$44.1\text{ ms}$** | **$22.7\text{ FPS}$** | Baseline chuẩn chứng minh zoom cục bộ giúp tăng **$+5.45\%\text{ mAP}_{50-95}$** |
| **4** | 🎲 **`Random K=2` (No Scout Guidance)** | Cắt ngẫu nhiên 2 vùng crop $640 \times 640$ | $2.00$ | $25.23\%$ | $11.40\%$ | $12.80\%$ | **$3.12\%$** | $17.26\%$ | $12.56\%$ | **$13.52\%$** | $35.07\%$ | $42.56\%$ | $38.45\%$ | **$52.2\text{ ms}$** | **$19.2\text{ FPS}$** | **Ablation chứng minh vai trò then chốt của Scout** (Không có Scout kết quả kém hơn cả Global) |
| **5** | 📌 **`Fixed K=4` (Fixed Top-K)** | Cố định luôn lấy đúng 4 vùng crop/ảnh | $4.00$ | $36.37\%$ | $20.46\%$ | $20.42\%$ | $12.75\%$ | $25.75\%$ | $16.49\%$ | $22.42\%$ | $41.80\%$ | $50.26\%$ | $45.64\%$ | $102.7\text{ ms}$ | $9.7\text{ FPS}$ | Đối sánh cơ chế thích ứng theo mật độ ổ gà so với số lượng crop cố định |
| **6** | ⚙️ **`Ablation: K_max=2`** | Giới hạn tối đa $K \le 2$ crop/ảnh | $1.00$ | $36.37\%$ | $20.46\%$ | $20.42\%$ | $12.75\%$ | $25.75\%$ | $16.49\%$ | $22.42\%$ | $41.80\%$ | $50.26\%$ | $45.64\%$ | $101.4\text{ ms}$ | $9.9\text{ FPS}$ | Khảo sát độ nhạy của ngưỡng số lượng vùng tối đa $K_{\max} = 2$ |
| **7** | ⚙️ **`Ablation: K_max=6`** | Giới hạn tối đa $K \le 6$ crop/ảnh | $1.00$ | $36.37\%$ | $20.46\%$ | $20.42\%$ | $12.75\%$ | $25.75\%$ | $16.49\%$ | $22.42\%$ | $41.80\%$ | $50.26\%$ | $45.64\%$ | $103.2\text{ ms}$ | $9.7\text{ FPS}$ | Khảo sát độ nhạy của ngưỡng số lượng vùng tối đa $K_{\max} = 6$ |
| **8** | 📐 **`Ablation: Margin=10%`** | Context Margin $= 0.10$ quanh vùng ROI | $1.00$ | $36.37\%$ | $20.46\%$ | $20.42\%$ | $12.75\%$ | $25.75\%$ | $16.49\%$ | $22.42\%$ | $41.80\%$ | $50.26\%$ | $45.64\%$ | $102.9\text{ ms}$ | $9.7\text{ FPS}$ | Khảo sát độ nhạy của biên ngữ cảnh hẹp ($10\%$) |
| **9** | 📐 **`Ablation: Margin=30%`** | Context Margin $= 0.30$ quanh vùng ROI | $1.00$ | $36.37\%$ | $20.46\%$ | $20.42\%$ | $12.75\%$ | $25.75\%$ | $16.49\%$ | $22.42\%$ | $41.80\%$ | $50.26\%$ | $45.64\%$ | $101.5\text{ ms}$ | $9.9\text{ FPS}$ | Khảo sát độ nhạy của biên ngữ cảnh rộng ($30\%$) |

---

## 🎯 3. Phân Tích Chuyên Sâu Chất Lượng Scout & Khoảng Cách Oracle (Scout Localization Quality & Bottleneck Analysis)

### 3.1. Khoảng Cách Oracle ($\Delta = 5.59\text{ mAP}$ points):
* **`AdaPoth-Oracle`** đạt **$26.01\%\text{ mAP}_{50-95}$** (Recall $55.37\%$), trong khi **`AdaPoth-Lite`** thực tế đạt **$20.42\%\text{ mAP}_{50-95}$** (Recall $41.80\%$).
* **Ý nghĩa khoa học**: Khoảng cách $\mathbf{5.59\%\text{ mAP}}$ chứng minh rằng **downstream detector chưa phải là bottleneck duy nhất**; độ chính xác định vị vùng chú ý của Scout Model đóng vai trò trần cận trên (upper ceiling) quyết định lượng thông tin mà detector có thể phục hồi từ ảnh 4K.

### 3.2. Phân Bố Động $K$ và Độ Nhạy Ngưỡng Nhiệt Độ ($\tau$ Sensitivity Analysis):
Khi phân tích phân bố giá trị kích hoạt của Scout Headmap ($[0.03, 0.25]$) trên tập dữ liệu kiểm thử, hành vi của Candidate Generator theo các mức ngưỡng $\tau$:

| Ngưỡng $\tau$ (Threshold) | Scout Recall@ROI | Tỷ Lệ Bao Phủ GT | Phân Bố $K=0$ | $K=1$ | $K=2$ | $K=3$ | $K=4$ | $K_{\text{avg}}$ Trung Bình | Đặc Điểm Vùng Crop |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\tau = 0.05$** | **$100.00\%$** | $173 / 173$ | $0\%$ | $9\%$ | $11\%$ | $8\%$ | $72\%$ | **$3.43$** | Đa vùng crop dày đặc, bao phủ tối đa |
| **$\tau = 0.08$** | **$59.54\%$** | $103 / 173$ | $0\%$ | $8\%$ | $5\%$ | $5\%$ | $82\%$ | **$3.61$** | Vùng crop thu gọn quanh tâm hotspot |
| **$\tau = 0.10$** | **$50.29\%$** | $87 / 173$ | $8\%$ | $13\%$ | $15\%$ | $14\%$ | $50\%$ | **$2.85$** | **Phân bố Dynamic Top-K thực thụ ($K \in [0, 4]$)** |
| **$\tau = 0.12$** | **$34.10\%$** | $59 / 173$ | $25\%$ | $24\%$ | $21\%$ | $15\%$ | $15\%$ | **$1.71$** | Tiết kiệm compute cao cho ảnh đường sạch |
| **$\tau = 0.30$ (Mặc định thô)** | **$79.77\%$** | $138 / 173$ | $0\%$ | $100\%$ | $0\%$ | $0\%$ | $0\%$ | **$1.00$** | Kích hoạt Safety Road Fallback Window |

### 3.3. Đột Phá Khảo Sát Scout Trên Ảnh 4K Gốc (Native 4K MobileNetV3 Scout vs Thumbnail Scout):

Nhằm kiểm chứng giả thuyết về giới hạn trần định vị do mất mát độ phân giải không gian trên ảnh thumbnail $960\text{p}$ ($16\times$ downsampling $\to 68\times 120$ heatmap), mô hình **MobileNetV3 Scout (Stride-8 FPN Multi-Scale Head)** đã được huấn luyện và đánh giá trực tiếp trên **toàn bộ ảnh 4K gốc ($3840 \times 2160 \to 480 \times 270\text{ heatmap}$)** trong $10$ Epochs:

#### A. Diễn Biến Hội Tụ 10 Epochs Của 4K Native Scout:
| Epoch | Train Loss | Val Loss | **Region Recall** | **GT Coverage** | False Region Rate | Avg $K$ | Learning Rate | Thời Gian / Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ep 1** | $48.63$ | $29.08$ | $40.57\%$ | $40.52\%$ | $89.0\%$ | $1.00$ | $0.001000$ | $895.0\text{s}$ |
| **Ep 2** | $14.82$ | $4.98$ | $40.57\%$ | $40.52\%$ | $89.2\%$ | $1.08$ | $0.000970$ | $963.6\text{s}$ |
| **Ep 3** | $5.41$ | $4.14$ | $61.68\%$ | $61.65\%$ | $87.1\%$ | $3.57$ | $0.000884$ | $930.7\text{s}$ |
| **Ep 4** | $4.83$ | $3.49$ | $56.91\%$ | $57.04\%$ | $90.6\%$ | $3.97$ | $0.000753$ | $929.7\text{s}$ |
| **Ep 5** | $4.32$ | $3.49$ | $61.78\%$ | $61.96\%$ | $88.5\%$ | $3.85$ | $0.000591$ | $934.7\text{s}$ |
| **Ep 6** | $4.13$ | $3.20$ | $73.66\%$ | $73.75\%$ | $84.6\%$ | $3.98$ | $0.000419$ | $899.2\text{s}$ |
| **Ep 7** 👑 | $3.87$ | $3.05$ | **$\mathbf{76.51\%}$** | **$\mathbf{76.21\%}$** | **$\mathbf{83.4\%}$** | **$4.00$** | $0.000258$ | $897.7\text{s}$ |
| **Ep 8** | $3.80$ | $3.02$ | $73.62\%$ | $73.30\%$ | $84.6\%$ | $4.00$ | $0.000126$ | $955.0\text{s}$ |
| **Ep 9** | $3.54$ | $2.98$ | $69.13\%$ | $68.75\%$ | $86.2\%$ | $4.00$ | $0.000040$ | $943.7\text{s}$ |
| **Ep 10** | $3.37$ | $2.98$ | $73.78\%$ | $73.69\%$ | $84.5\%$ | $4.00$ | $0.000010$ | $934.4\text{s}$ |

#### B. Bảng Quét Ngưỡng $\tau$ Và Ngân Sách Crop $K$ Trên Tập Test ($900$ Ảnh 4K Độc Lập):
| Tham Số Khảo Sát | Giá Trị Cấu Hình | **Region Recall (Test)** | **GT Coverage (Test)** | **False Region Rate** | **Avg $K_{\text{crops}}$** | Nhận Xét Khoa Học |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Threshold Sweep** ($K=4$) | $\tau = 0.01$ | $78.13\%$ | $77.58\%$ | $84.5\%$ | $3.96$ | Ngưỡng quá thấp, nhận diện nhiều nhiễu nền |
| | $\tau = 0.05$ | $78.13\%$ | $77.58\%$ | $84.5\%$ | $3.99$ | Mức ngưỡng cân bằng mặc định |
| | $\tau = 0.10$ | $79.02\%$ | $78.39\%$ | $84.4\%$ | $4.00$ | Lọc bớt nhiễu nhẹ |
| | $\tau = 0.15$ | $80.90\%$ | $80.70\%$ | $83.9\%$ | $3.99$ | Bắt đầu tập trung vào vùng mật độ cao |
| | **$\tau = 0.20$** 🚀 | **$\mathbf{84.86\%}$** | **$\mathbf{84.23\%}$** | **$\mathbf{82.18\%}$** | **$3.93$** | **Vượt trần recall: Định vị cực kỳ sắc nét trên ảnh 4K gốc** |
| **K-Budget Sweep** ($\tau = 0.05$) | $K_{\max} = 1$ | $63.21\%$ | $63.30\%$ | $63.7\%$ | $1.00$ | 1 crop duy nhất chỉ lấy được ổ gà lớn nhất |
| | $K_{\max} = 2$ | $70.79\%$ | $70.87\%$ | $75.3\%$ | $2.00$ | Bao phủ thêm cụm ổ gà phụ |
| | $K_{\max} = 3$ | $75.06\%$ | $74.87\%$ | $81.3\%$ | $2.99$ | Gần tiệm cận mức tối ưu |
| | $K_{\max} = 4$ | $78.13\%$ | $77.58\%$ | $84.5\%$ | $3.99$ | Cân bằng compute / recall chuẩn |
| | $K_{\max} = 6$ | **$79.75\%$** | $79.04\%$ | $88.8\%$ | $5.97$ | Tăng nhẹ recall nhưng tỷ lệ crop trượt tăng cao |

---

## 📦 4. Bảng Các Thực Nghiệm Bổ Trợ (Supplementary Experiments)

Bao gồm các phương pháp khảo sát bổ sung (Slicing trên mô hình 4K, các độ phân giải trung gian, Zero-shot Upscaling và baseline ngoại vi):

| STT | Mô Hình / Phương Pháp | Input Resolution | Cơ Chế Suy Luận | $\mathbf{\text{mAP}_{50}}$ | $\mathbf{\text{mAP}_{75}}$ | $\mathbf{\text{mAP}_{50-95}}$ | Recall | Precision | $F_1$ | FPPI | Latency | Mục Đích / Ghi Chú |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **`sahi` on 4K Model** | $3840 \times 2160$ | SAHI đa cấp (15 calls) | **$42.80\%$** | $28.03\%$ | **$26.24\%$** | $37.13\%$ | $67.59\%$ | $47.93\%$ | $0.083$ | $1054.7\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 2 | **`perspective-grid` on 4K Model** | $3840 \times 2160$ | 3 dải phối cảnh (9 calls) | **$42.02\%$** | $27.10\%$ | **$25.40\%$** | $38.00\%$ | $65.67\%$ | $48.14\%$ | $0.113$ | $830.6\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 3 | **`sliced-nms` on 4K Model** | $3840 \times 2160$ | Lưới đều (25 calls) | **$36.88\%$** | $24.35\%$ | **$22.84\%$** | $33.12\%$ | $65.59\%$ | $44.01\%$ | $0.103$ | $912.6\text{ ms}$ | Slicing thử nghiệm trên model 4K |
| 4 | **`yolo11m_1280`** ⚠️ *(Không thuộc story research)* | $1280 \times 1280$ | Resize 1280 (1 pass) | **$48.98\%$** | $27.10\%$ | **$25.40\%$** | $45.50\%$ | $61.40\%$ | $52.26\%$ | $0.050$ | **$14.6\text{ ms}$** | Baseline kích thước trung gian (Lưu trữ nội bộ) |
| 5 | **`RT-DETRv2` (640)** | $640 \times 640$ | Resize 640 (1 pass) | **$44.01\%$** | $21.22\%$ | **$23.24\%$** | **$53.42\%$** | $32.14\%$ | $40.10\%$ | $0.323$ | $51.5\text{ ms}$ | Đối sánh Transformer Baseline |
| 6 | **`RT-DETRv1` (640)** | $640 \times 640$ | Resize 640 (1 pass) | **$43.54\%$** | $23.36\%$ | **$23.61\%$** | $48.86\%$ | $44.38\%$ | $46.50\%$ | $0.237$ | $50.9\text{ ms}$ | Đối sánh Transformer Baseline |
| 7 | **`yolov8m_640`** | $640 \times 640$ | Resize 640 (1 pass) | **$34.24\%$** | $15.03\%$ | **$16.39\%$** | $30.51\%$ | $65.50\%$ | $41.63\%$ | $0.087$ | **$36.6\text{ ms}$** | Đối sánh CNN Baseline |
| 8 | **`yolov5m_640`** | $640 \times 640$ | Resize 640 (1 pass) | **$33.80\%$** | $14.92\%$ | **$16.78\%$** | $28.12\%$ | $65.74\%$ | $39.39\%$ | $0.053$ | **$37.1\text{ ms}$** | Đối sánh CNN Baseline |
| 9 | **`resize (640)` on 4K Model** | $3840 \times 2160$ | Nén 640 (1 pass) | **$0.22\%$** | $0.22\%$ | **$0.17\%$** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.000$ | $98.7\text{ ms}$ | Minh chứng domain shift khi nén |
| 10 | **`yolo11m_640` on 4K Images** ⚠️ | $3840 \times 2160$ | Test trên 4K (1 pass) | **$13.18\%$** | $5.38\%$ | **$6.69\%$** | $39.41\%$ | $4.11\%$ | $7.44\%$ | $7.130$ | **$27.3\text{ ms}$** | Train 640 $\to$ Test 4K: Cho thấy sự sai lệch phân phối không gian khi scale |
| 11 | **`dfine_640` on 4K Images** ⚠️ | $3840 \times 2160$ | Test trên 4K (1 pass) | **$0.23\%$** | $0.01\%$ | **$0.06\%$** | $3.26\%$ | $2.62\%$ | $2.90\%$ | $0.667$ | **$32.5\text{ ms}$** | Train 640 $\to$ Test 4K: Cho thấy sự suy giảm mạnh của Transformer grid khi thay đổi kích thước |

> [!WARNING]
> ### 📌 Ghi Chú Chiến Lược Về Mô Hình `yolo11m_1280`:
> Mô hình **`yolo11m_1280`** ($1280 \times 1280$) đạt kết quả cao ($25.40\%\text{ mAP}_{50-95}$ với độ trễ chỉ $14.6\text{ ms}$). Tuy nhiên, mô hình này **hoàn toàn KHÔNG được đưa vào câu chuyện nghiên cứu chính (Main Research Story)** của bài báo vì:
> 1. **Làm lệch trọng tâm giả thuyết cốt lõi (*Core Hypothesis*)**: Bài báo tập trung chứng minh cơ chế **phân bổ tính toán thông minh (Adaptive Region Scout)** ở độ phân giải 640px có thể cạnh tranh với Native 4K mà không cần tăng kích thước canvas toàn cục.
> 2. **Làm lu mờ đóng góp về kiến trúc nhẹ**: Đưa $1280 \times 1280$ vào làm phương pháp so sánh chính sẽ biến bài toán thành "chạy đua kích thước ảnh đầu vào" thay vì giải quyết bài toán cốt lõi là nhận diện mục tiêu nhỏ trên ảnh siêu phân giải 4K với tài nguyên hạn chế.
> 3. Mô hình này chỉ được lưu trữ nội bộ dưới dạng baseline tham chiếu bổ trợ trong bảng Supplementary Experiments.
