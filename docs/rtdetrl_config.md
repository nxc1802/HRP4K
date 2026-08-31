### RT-DETR-L — Fixed Training Config (Thực Tế Dự Án)

| Parameter | Fixed Value (Thực Tế) | Ghi Chú & Implementation Chi Tiết |
| :--- | :---: | :--- |
| **Model Architecture** | **RT-DETR-L (Real-Time DEtection TRansformer)** | Backbone HGNetv2 + Multi-scale Deformable Attention Decoder (`rtdetr-l.pt`, 32.8M params) |
| **Pretrained Weights** | **COCO Pretrained** | Nạp weights gốc Transformer từ COCO benchmark |
| **Epochs** | **150** | Huấn luyện đủ 150 epochs |
| **Optimizer** | **AdamW** | Optimizer chuẩn cho Transformer (`weight_decay = 1e-4`) |
| **Initial LR (`lr0`)** | **1e-4 (`0.0001`)** | Base learning rate cho Transformer attention layers |
| **Final LR (`lrf`)** | **1e-6 (`0.000001`)** | `lrf = 0.01` (Decay $0.01 \times \text{lr0}$) |
| **Weight Decay** | **1e-4 (`0.0001`)** | L2 Regularization chống overfitting |
| **Warmup** | **3.0 Epochs** | `warmup_momentum = 0.8`, `warmup_bias_lr = 0.0` |
| **Scheduler** | **Cosine Annealing** | Cosine learning rate decay cho Transformer |
| **Effective Batch Size** | **16** | Cố định cho mọi độ phân giải |
| **Batch Adaptation** | **Gradient Accumulation** | - 640 & 1K: `batch = 16` (1x accum)<br>- 2K: `batch = 4` (4x accum)<br>- 4K: `batch = 2` (8x accum $\to$ Effective Batch = 16) |
| **AMP / Precision** | **FP32 cho 4K / FP16 cho 640** | `amp = False` ở 4K để chống lỗi tràn số NaN trong Deformable Attention 4K; `amp = True` ở 640 |
| **Rectangular (`rect`)** | **True (cho 4K/2K/1K)** | Bảo toàn tỷ lệ khung hình $16:9$, tránh padding đen thừa |
| **Gradient Clipping** | **0.1 (Max Norm)** | Chống bùng nổ gradient trong Attention Decoder |
| **Early Stopping** | **patience = 10** | Dừng sớm nếu không cải thiện sau 10 epoch |
| **Augmentation** | **Chuẩn HRP4K Transformer** | `mosaic = 1.0` (đóng ở 10 epoch cuối), `translate = 0.1`, `scale = 0.5`, `fliplr = 0.5`, `hsv_h = 0.015`, `hsv_s = 0.7`, `hsv_v = 0.4`, `erasing = 0.4` |
| **Seed** | **42** | Deterministic = True |
| **Dataset Split** | **HRP4K Canonical** | Train: 4,202 ảnh \| Valid: 901 ảnh \| Test: 900 ảnh (600 pos + 300 neg) |
| **Checkpoint Selection** | **Best Validation** | Lưu tại `weights/best.pt` dựa trên best mAP50-95 |

---

### Ma Trận Thực Nghiệm Resolution (RT-DETR-L)

| Thí Nghiệm | Input Resolution | Aspect Ratio | Batch (Physical) | Accumulation | Effective Batch | Optimizer | Base LR | AMP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **rtdetr-l-resolution-4k** | **$3840 \times 2160$** | $16:9$ | $2$ | $8\times$ | **$16$** | AdamW | $1\times 10^{-4}$ | False (FP32) |
| **rtdetr-l-resolution-2k** | **$1920 \times 1080$** | $16:9$ | $4$ | $4\times$ | **$16$** | AdamW | $1\times 10^{-4}$ | True (FP16) |
| **rtdetr-l-resolution-1k** | **$960 \times 540$** | $16:9$ | $16$ | $1\times$ | **$16$** | AdamW | $1\times 10^{-4}$ | True (FP16) |
| **rtdetr-l-resolution-640** | **$640 \times 640$ / $640 \times 360$** | $1:1$ / $16:9$ | $16$ | $1\times$ | **$16$** | AdamW | $1\times 10^{-4}$ | True (FP16) |

---

### Spatial Decomposition (Khai Thác Không Gian ở Base 640)

Sau khi huấn luyện mô hình **RT-DETR-L-640**, mô hình này được làm backbone hạt nhân cho các chiến lược phân tách không gian:

```text
RT-DETR-L-640
├── 1. Full Image (Uniform Baseline 640) : Nén đều toàn cảnh về 640×640 (1 pass)
├── 2. Sliced-NMS (25 crops)             : Cắt 25 patches 960p chồng lấn 20%, NMS hợp nhất
├── 3. SAHI (Standard Slicing)          : Cắt slicing theo lưới trượt SAHI
└── 4. Perspective Grid (9 crops)       : Phân bổ 9 patches phi đối xứng theo phối cảnh mặt đường
```
