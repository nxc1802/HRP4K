### Table 1 — Resolution

| Resolution | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency |
| ---------- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: | -----------------: | --------: | -----: | -: | ---: | ------: |
| 4K         |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| 2K         |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| 1K         |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| 640        |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |

### Table 2 — Low-Resolution Spatial Decomposition: Inference

| Method                | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency |
| --------------------- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: | -----------------: | --------: | -----: | -: | ---: | ------: |
| Full Image (Baseline) |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| Sliced-NMS            |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| SAHI                  |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |
| Perspective Grid      |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |

### Table 3 — Low-Resolution Spatial Decomposition: Training + Inference

| Method                | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>50:95</sub> | AP<sub>Ultra-Fine</sub> | AP<sub>Small</sub> | AP<sub>Medium</sub> | AP<sub>Large</sub> | Precision | Recall | F1 | FPPI | Latency |   |
| --------------------- | --------------: | --------------: | -----------------: | ----------------------: | -----------------: | ------------------: | -----------------: | --------: | -----: | -: | ---: | ------: | - |
| Full Image (Baseline) |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |   |
| Sliced-NMS            |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |   |
| SAHI                  |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |   |
| Perspective Grid      |                 |                 |                    |                         |                    |                     |                    |           |        |    |      |         |   |
