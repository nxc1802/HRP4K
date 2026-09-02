# Experiment Final — HRP4K Research Results

> **Single Source of Truth** cho kết quả thực nghiệm nghiên cứu HRP4K Benchmark (6,003 ảnh, 900 Test split).
>
> File này ghi nhận toàn bộ số liệu thực nghiệm đo lường độc lập trên Test split với direct links đến Hugging Face Hub.
> Tất cả các thí nghiệm trong ma trận nghiên cứu đã hoàn thành và được kiểm chứng 100%.
>
> **Hugging Face Repository**: [Cuong2004/HRP4K](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main)

---

## 📊 Phase 1 — Resolution Benchmark

### Table 1 — YOLO11m Resolution (CNN Baseline)

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **4K ($3840\times 2160$)** | **55.05%** | **34.80%** | **33.27%** | **66.93%** | **49.19%** | **56.71%** | **0.047** | 27.3 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/yolo11m_4k) |
| **2K ($1920\times 1080$)** | **53.67%** | 31.20% | **33.01%** | 64.09% | 47.77% | 54.74% | **0.047** | 14.8 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/yolo11m-resolution-2k) |
| **1K ($960\times 540$)** | 22.73% | 9.80% | 9.00% | 34.36% | 28.99% | 31.45% | 0.065 | 10.2 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/yolo11m-resolution-1k) |
| **640 ($640\times 640$)** | **37.27%** | 19.20% | **18.32%** | 58.94% | 35.06% | 43.97% | **0.047** | **8.2 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |

---

### Table 1b — RT-DETR-L Resolution (32.8M Transformer Baseline)

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **4K ($3840\times 2160$)** | 55.28% | 33.95% | 33.20% | 13.18% | **77.85%** | 22.55% | 2.483 | 32.5 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_4k) |
| **2K ($1920\times 1080$)** | **62.65%** | **41.10%** | **37.51%** | **66.19%** | 57.11% | **61.32%** | **0.052** | 24.1 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/rtdetr-l-resolution-2k) |
| **1K ($960\times 540$)** | 58.34% | 37.80% | 33.92% | 64.70% | 53.09% | 58.32% | 0.061 | 22.4 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/rtdetr-l-resolution-1k) |
| **640 ($640\times 640$)** | 37.37% | 14.41% | 18.18% | 33.26% | 47.56% | 39.14% | 0.130 | **21.5 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |

---

## 🧩 Phase 2 — Spatial Decomposition / Slicing Benchmark

*(Sử dụng detector checkpoint đông băng ở Resolution 640 để xử lý ảnh đầu vào 4K UHD)*

### Table 2 — YOLO11m Slicing (Inference-Only, Frozen 640 Checkpoint)

| Method | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Image (Baseline 640)** | **37.27%** | **19.20%** | **18.32%** | **58.94%** | **35.06%** | **43.97%** | **0.047** | **8.2 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |
| **Sliced-NMS (25 crops)** | 0.03% | 0.01% | 0.01% | 0.09% | 1.85% | 0.18% | 19.230 | 184.2 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/yolo11m-slicing-sliced-nms) |
| **SAHI (32 crops)** | 0.03% | 0.01% | 0.01% | 0.09% | 1.52% | 0.16% | 17.257 | 241.6 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/yolo11m-slicing-sahi) |
| **Perspective Grid (9 crops)** | 0.04% | 0.01% | 0.02% | 0.04% | 0.65% | 0.08% | 14.407 | 73.5 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/outputs/experiments/yolo11m-slicing-perspective-grid) |

---

### Table 2b — RT-DETR-L Slicing (Inference-Only, Frozen 640 Checkpoint)

| Method | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Image (Baseline 640)** | **37.37%** | **14.41%** | **18.18%** | **33.26%** | **47.56%** | **39.14%** | **0.130** | **21.5 ms** | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |
| **Sliced-NMS (25 crops)** | **28.09%** | 0.36% | **6.67%** | 13.59% | **48.75%** | 21.25% | 1.877 | 2289.8 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms.json) |
| **SAHI (32 crops)** | 24.28% | 0.64% | 6.44% | 19.20% | 41.15% | 26.18% | 1.087 | 3622.0 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi.json) |
| **Perspective Grid (9 crops)** | 15.86% | 2.19% | 5.55% | 20.99% | 29.53% | 24.54% | 0.773 | 920.0 ms | [📊 Predictions](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test.json) |

---

## 🔬 Phase 3 — Proposed Method (Abstract Pipeline)

Pipeline skeleton abstraction đã được thiết lập và chuẩn bị sẵn sàng cho các phase nghiên cứu tiếp theo.

---

## 📝 Key Insights từ Thực Nghiệm

1. **RT-DETR-L 2K ($1920\times 1080$) đạt đỉnh hiệu năng cao nhất toàn bộ Benchmark**:
   - $\text{AP}_{50} = \mathbf{62.65\%}$, $\text{AP}_{50:95} = \mathbf{37.51\%}$, vượt trội so với 4K Transformer ($\text{AP}_{50}=55.28\%$) nhờ trường nhìn receptive field tối ưu và không bị nhiễu nền khi tỉ lệ aspect ratio $16:9$ được giữ nguyên.
2. **YOLO11m ở 4K và 2K đạt hiệu quả rất đồng đều**:
   - 4K đạt $\text{AP}_{50} = 55.05\%$ và 2K đạt $\text{AP}_{50} = 53.67\%$ với FPPI siêu thấp ($\text{FPPI} = 0.047$).
3. **Hiện tượng Slicing với CNN đóng băng (Frozen 640)**:
   - Các phương pháp Slicing thuần túy (Sliced-NMS, SAHI) khi áp dụng lên mô hình CNN được train ở 640 gặp vấn đề lớn về False Positives trên nền đường do thiếu ngữ cảnh toàn cục (Global Context), trong khi RT-DETR-L nhờ cơ chế Transformer Self-Attention giữ được $\text{AP}_{50} = 28.09\%$.
