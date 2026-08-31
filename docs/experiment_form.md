### Table 1 — Resolution Benchmark

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| ---------- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: | -----------------: | --------: | -----: | -: | ---: | ------: | :--- |
| **YOLO11m 4K** | 55.05% | 34.80% | 33.27% | 42.60% | 47.10% | 41.50% | 26.40% | 66.93% | 49.19% | 56.71% | 0.047 | 27.3 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/yolo11m_4k) |
| **YOLO11m 2K** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **YOLO11m 1K** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **YOLO11m 640** | 37.27% | 19.20% | 18.32% | 17.80% | 23.50% | 25.90% | 19.20% | 58.94% | 35.06% | 43.97% | 0.047 | 8.2 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |
| **RT-DETR-L 4K** | 55.28% | 33.95% | 33.20% | 46.84% | 46.19% | 39.73% | 25.86% | 13.18% | 77.85% | 22.55% | 2.483 | 32.5 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_4k) |
| **RT-DETR-L 2K** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **RT-DETR-L 1K** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **RT-DETR-L 640** | 37.37% | 14.41% | 18.18% | 18.40% | 24.10% | 26.50% | 19.80% | 33.26% | 47.56% | 39.14% | 0.130 | 21.5 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |

---

### Table 2 — Low-Resolution Spatial Decomposition: Inference (Frozen 640 Checkpoint)

| Method | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency | Hugging Face Link |
| :--- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: | -----------------: | --------: | -----: | -: | ---: | ------: | :--- |
| **YOLO11m Full Image (640)** | 37.27% | 19.20% | 18.32% | 17.80% | 23.50% | 25.90% | 19.20% | 58.94% | 35.06% | 43.97% | 0.047 | 8.2 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/yolo11m_640) |
| **YOLO11m Sliced-NMS** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **YOLO11m SAHI** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **YOLO11m Perspective Grid** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | `⏳ Pending Server Run` |
| **RT-DETR-L Full Image (640)** | 37.37% | 14.41% | 18.18% | 18.40% | 24.10% | 26.50% | 19.80% | 33.26% | 47.56% | 39.14% | 0.130 | 21.5 ms | [📦 Checkpoint](https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main/checkpoints/dfine_640) |
| **RT-DETR-L Sliced-NMS** | 44.30% | 11.74% | 18.81% | 31.18% | 38.69% | 38.37% | 12.68% | 21.78% | 62.43% | 32.29% | 0.937 | 2289.8 ms | [📊 Preds](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sliced_nms_metrics.json) |
| **RT-DETR-L SAHI** | 24.28% | 0.64% | 6.44% | 16.50% | 21.30% | 19.80% | 5.20% | 19.20% | 41.15% | 26.18% | 1.087 | 3622.0 ms | [📊 Preds](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/predictions/dfine_patch_sahi_metrics.json) |
| **RT-DETR-L Perspective Grid** | 15.86% | 2.19% | 5.55% | 11.20% | 14.80% | 13.50% | 4.10% | 20.99% | 29.53% | 24.54% | 0.180 | 920.0 ms | [📊 Preds](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test.json) \| [📈 Metrics](https://huggingface.co/datasets/Cuong2004/HRP4K/blob/main/outputs/phase2_benchmark/dfine_patch_perspective_grid_test_metrics.json) |
