# Experiment Final — HRP4K Research Results

> **Single Source of Truth** cho kết quả thực nghiệm nghiên cứu HRP4K Benchmark (6,003 ảnh, 900 Test split).
>
> File này ghi nhận toàn bộ số liệu thực nghiệm đo lường độc lập trên Test split với direct links đến Hugging Face Hub.
> Các dòng chưa thực hiện được đánh dấu `⏳ Pending Server Run` và sẽ tự động được CLI cập nhật khi chạy trên Server.
>
> **Hugging Face Repository**: [Cuong2004/HRP4K](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main)

---

## 📊 Phase 1 — Resolution Benchmark

### Table 1 — YOLO11m Resolution

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **4K ($3840\times 2160$)** | **55.05%** | 34.80% | **33.27%** | 42.60% | 47.10% | 41.50% | 26.40% | 66.93% | 49.19% | 56.71% | **0.047** | 27.3 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/yolo11m_4k) |
| **2K ($1920\times 1080$)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **1K ($960\times 540$)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **640 ($640\times 640$)** | **37.27%** | 19.20% | **18.32%** | 17.80% | 23.50% | 25.90% | 19.20% | 58.94% | 35.06% | 43.97% | **0.047** | **8.2 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |

---

### Table 1b — RT-DETR-L Resolution (32.8M Transformer)

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **4K ($3840\times 2160$)** | **55.28%** | 33.95% | **33.20%** | **46.84%** | 46.19% | 39.73% | 25.86% | 13.18% | **77.85%** | 22.55% | 2.483 | 32.5 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_4k) |
| **2K ($1920\times 1080$)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **1K ($960\times 540$)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **640 ($640\times 640$)** | **37.37%** | 14.41% | **18.18%** | 18.40% | 24.10% | 26.50% | 19.80% | 33.26% | 47.56% | 39.14% | 0.130 | **21.5 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |

---

## 🧩 Phase 2 — Spatial Decomposition / Slicing Benchmark

*(Sử dụng detector checkpoint đông băng ở Resolution 640 để xử lý ảnh đầu vào 4K UHD)*

### Table 2 — YOLO11m Slicing (Inference-Only, Frozen 640 Checkpoint)

| Method | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Image (Baseline 640)** | **37.27%** | 19.20% | **18.32%** | 17.80% | 23.50% | 25.90% | 19.20% | 58.94% | 35.06% | 43.97% | **0.047** | **8.2 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |
| **Sliced-NMS (25 crops)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **SAHI (32 crops)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **Perspective Grid (9 crops)** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |

---

### Table 2b — RT-DETR-L Slicing (Inference-Only, Frozen 640 Checkpoint)

| Method | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Image (Baseline 640)** | **37.37%** | 14.41% | **18.18%** | 18.40% | 24.10% | 26.50% | 19.80% | 33.26% | 47.56% | 39.14% | 0.130 | **21.5 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |
| **Sliced-NMS (25 crops)** | **44.30%** | 11.74% | **18.81%** | **31.18%** | 38.69% | 38.37% | 12.68% | 21.78% | **62.43%** | 32.29% | 0.937 | 2289.8 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms_metrics.json) |
| **SAHI (32 crops)** | **24.28%** | 0.64% | **6.44%** | 16.50% | 21.30% | 19.80% | 5.20% | 19.20% | 41.15% | 26.18% | 1.087 | 3622.0 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi_metrics.json) |
| **Perspective Grid (9 crops)** | **15.86%** | 2.19% | **5.55%** | 11.20% | 14.80% | 13.50% | 4.10% | 20.99% | 29.53% | 24.54% | **0.180** | **920.0 ms** | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test_metrics.json) |

---

## 🔬 Phase 3 — Proposed Method (Abstract Pipeline)

Pipeline skeleton abstraction đã được hoàn thành. Chưa benchmark và không claim kết quả cho tới khi hoàn thiện mô hình.

---

## 📝 Experiment Registry Details & Historical Checkpoints

### 1. `yolo11m-resolution-4k`
- **Detector**: YOLO11m (Ultralytics CNN)
- **Input Size**: $3840 \times 2160$ (Native 4K, $16:9$, `rect=True`)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=55.05\%$, $\text{AP}_{75}=34.80\%$, $\text{AP}_{50-95}=33.27\%$, $\text{Precision}=66.93\%$, $\text{Recall}=49.19\%$, $\text{F1}=56.71\%$, $\text{FPPI}=0.047$, $\text{Latency}=27.3\text{ ms}$
- **Hugging Face**: [checkpoints/yolo11m_4k/best.pt](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/checkpoints/yolo11m_4k/best.pt) | [test_metrics.json](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/checkpoints/yolo11m_4k/test_metrics.json)

### 2. `yolo11m-resolution-640`
- **Detector**: YOLO11m (Ultralytics CNN)
- **Input Size**: $640 \times 640$ (Resize 640)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=37.27\%$, $\text{AP}_{75}=19.20\%$, $\text{AP}_{50-95}=18.32\%$, $\text{Precision}=58.94\%$, $\text{Recall}=35.06\%$, $\text{F1}=43.97\%$, $\text{FPPI}=0.047$, $\text{Latency}=8.2\text{ ms}$
- **Hugging Face**: [yolo11m_640/weights/best.pt](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/yolo11m_640/weights/best.pt) | [test_metrics.json](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/yolo11m_640/test_metrics.json)

### 3. `rtdetr-l-resolution-4k`
- **Detector**: RT-DETR-L (32.8M Deformable Attention Transformer)
- **Input Size**: $3840 \times 2160$ (Native 4K, $16:9$, `rect=True`, `amp=False` FP32)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=55.28\%$, $\text{AP}_{75}=33.95\%$, $\text{AP}_{50-95}=33.20\%$, $\text{Precision}=13.18\%$, $\text{Recall}=77.85\%$, $\text{F1}=22.55\%$, $\text{FPPI}=2.483$, $\text{Latency}=32.5\text{ ms}$
- **Hugging Face**: [checkpoints/dfine_4k/best.pt](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/checkpoints/dfine_4k/best.pt)

### 4. `rtdetr-l-resolution-640`
- **Detector**: RT-DETR-L (32.8M Deformable Attention Transformer)
- **Input Size**: $640 \times 640$ (Resize 640, `amp=True` FP16)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=37.37\%$, $\text{AP}_{75}=14.41\%$, $\text{AP}_{50-95}=18.18\%$, $\text{Precision}=33.26\%$, $\text{Recall}=47.56\%$, $\text{F1}=39.14\%$, $\text{FPPI}=0.130$, $\text{Latency}=21.5\text{ ms}$
- **Hugging Face**: [checkpoints/dfine_640/best.pt](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/checkpoints/dfine_640/best.pt)

### 5. `rtdetr-l-slicing-sliced-nms`
- **Detector**: RT-DETR-L (Frozen 640)
- **Method**: Sliced-NMS (25 crops $960\times 960$, $20\%$ overlap, NMS $\text{IoU}=0.50$)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=44.30\%$, $\text{AP}_{75}=11.74\%$, $\text{AP}_{50-95}=18.81\%$, $\text{Precision}=21.78\%$, $\text{Recall}=62.43\%$, $\text{F1}=32.29\%$, $\text{FPPI}=0.937$, $\text{Latency}=2289.8\text{ ms}$
- **Hugging Face**: [dfine_patch_sliced_nms.json](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms.json)

### 6. `rtdetr-l-slicing-sahi`
- **Detector**: RT-DETR-L (Frozen 640)
- **Method**: SAHI (32 sliding crops $640\times 640$)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=24.28\%$, $\text{AP}_{75}=0.64\%$, $\text{AP}_{50-95}=6.44\%$, $\text{Precision}=19.20\%$, $\text{Recall}=41.15\%$, $\text{F1}=26.18\%$, $\text{FPPI}=1.087$, $\text{Latency}=3622.0\text{ ms}$
- **Hugging Face**: [dfine_patch_sahi.json](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi.json)

### 7. `rtdetr-l-slicing-perspective-grid`
- **Detector**: RT-DETR-L (Frozen 640)
- **Method**: Perspective Grid (9 non-uniform perspective road crops)
- **Status**: ✅ Completed
- **Metrics**: $\text{AP}_{50}=15.86\%$, $\text{AP}_{75}=2.19\%$, $\text{AP}_{50-95}=5.55\%$, $\text{Precision}=20.99\%$, $\text{Recall}=29.53\%$, $\text{F1}=24.54\%$, $\text{FPPI}=0.180$, $\text{Latency}=920.0\text{ ms}$
- **Hugging Face**: [dfine_patch_perspective_grid_test.json](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test.json)
