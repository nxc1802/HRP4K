### YOLO11m — Fixed Training Config (Thực Tế Dự Án)

| Parameter | Fixed Value (Thực Tế) | Ghi Chú & Implementation Chi Tiết |
| :--- | :---: | :--- |
| **Model** | **YOLO11m** | Ultralytics YOLOv11 Medium (`yolo11m.pt`) |
| **Pretrained Weights** | **COCO Pretrained** | Nạp weights gốc từ Ultralytics COCO |
| **Epochs** | **150** | Huấn luyện đủ 150 epochs |
| **Optimizer** | **SGD** | `momentum = 0.937`, `nbs = 64` (mặc định tối ưu của YOLO) |
| **Initial LR (`lr0`)** | **1e-2 (`0.01`)** | Base learning rate cho SGD |
| **Final LR (`lrf`)** | **1e-4 (`0.0001`)** | `lrf = 0.01` (tỷ lệ decay $0.01 \times \text{lr0}$) |
| **Weight Decay** | **5e-4 (`0.0005`)** | L2 Regularization |
| **Warmup** | **3.0 Epochs** | `warmup_momentum = 0.8`, `warmup_bias_lr = 0.1` |
| **Scheduler** | **Linear / Cosine** | Decay theo schedule chuẩn của Ultralytics |
| **Effective Batch Size** | **16** | Cố định cho mọi độ phân giải |
| **Batch Adaptation** | **Gradient Accumulation** | - 640 & 1K: `batch = 16` (1x accum)<br>- 2K: `batch = 4` (4x accum)<br>- 4K: `batch = 2` (8x accum $\to$ Effective Batch = 16) |
| **AMP / Precision** | **FP16 (AMP = True)** | Tự động bật mixed precision tăng tốc và tiết kiệm VRAM |
| **Rectangular (`rect`)** | **True (cho 4K/2K/1K)** | Bảo toàn tỷ lệ khung hình $16:9$, tránh padding đen thừa |
| **Early Stopping** | **patience = 10** | Dừng sớm nếu không cải thiện sau 10 epoch |
| **Augmentation** | **Chuẩn HRP4K** | `mosaic = 1.0` (đóng ở 10 epoch cuối), `translate = 0.1`, `scale = 0.5`, `fliplr = 0.5`, `hsv_h = 0.015`, `hsv_s = 0.7`, `hsv_v = 0.4`, `erasing = 0.4` |
| **Seed** | **42** | Deterministic = True |
| **Dataset Split** | **HRP4K Canonical** | Train: 4,202 ảnh \| Valid: 901 ảnh \| Test: 900 ảnh (600 pos + 300 neg) |
| **Checkpoint Selection** | **Best Validation** | Lưu tại `weights/best.pt` dựa trên best mAP50-95 |

---

### Ma Trận Thực Nghiệm Resolution (YOLO11m)

| Thí Nghiệm | Input Resolution | Aspect Ratio | Batch (Physical) | Accumulation | Effective Batch | Optimizer | Base LR | AMP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Resolution-4K** | **$3840 \times 2160$** | $16:9$ | $2$ | $8\times$ | **$16$** | SGD | $0.01$ | True |
| **Resolution-2K** | **$1920 \times 1080$** | $16:9$ | $4$ | $4\times$ | **$16$** | SGD | $0.01$ | True |
| **Resolution-1K** | **$960 \times 540$** | $16:9$ | $16$ | $1\times$ | **$16$** | SGD | $0.01$ | True |
| **Resolution-640** | **$640 \times 640$ / $640 \times 360$** | $1:1$ / $16:9$ | $16$ | $1\times$ | **$16$** | SGD | $0.01$ | True |

---

### Spatial Decomposition (Khai Thác Không Gian ở Base 640)

Toàn bộ các phương pháp phân tách không gian đều sử dụng **mô hình base 640** làm detector hạt nhân:

```text
YOLO11m-640
├── 1. Full Image (Uniform Baseline 640) : Nén đều toàn cảnh về 640×640 (1 pass)
├── 2. Sliced-NMS (25 crops)             : Cắt 25 patches 960p chồng lấn 20%, NMS hợp nhất
├── 3. SAHI (Standard Slicing)          : Cắt slicing theo lưới trượt SAHI
└── 4. Perspective Grid (9 crops)       : Phân bổ 9 patches phi đối xứng theo phối cảnh mặt đường
```
