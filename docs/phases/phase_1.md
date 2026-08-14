# Final plan — **Baseline & Benchmark cho HRP4K**

Mục tiêu của project này nên được giữ rất rõ: **xây một benchmark suite tái lập và phân tích toàn diện cho HRP4K**. Benchmark này sẽ trở thành nền tảng chuẩn hóa để đánh giá công bằng các mô hình phát hiện đối tượng và chiến lược xử lý hình ảnh.

Paper HRP4K công bố 6.003 ảnh 4K 3840×2160, gồm 4.003 ảnh positive, 2.000 negative và 7.217 pothole instances; dữ liệu có cả YOLO và COCO annotation.  Dataset đã có split ở **video-level** nhằm tránh leakage, với **4.203 train / 900 validation / 900 test**. 

---

## 1. Phạm vi project

Project cuối cùng nên có 4 module chính:

```text
HRP4K-Benchmark
│
├── 1. Dataset & Dataset Description
│
├── 2. Baseline Models / Related Work
│
├── 3. Unified Training + Evaluation Source Code
│
└── 4. Benchmark Results + Analysis
```

Output cuối cùng không chỉ là một notebook chạy model, mà là một **reproducible benchmark repository**:

```text
Dataset
   ↓
Dataset validation
   ↓
6 official baselines
   ↓
Unified predictions
   ↓
Unified evaluator
   ↓
Overall / scale / material / negative-set analysis
   ↓
Benchmark tables + plots + checkpoints
```

---

# 2. Dataset Description

Đây nên là phần đầu tiên được hoàn thiện vì toàn bộ benchmark phụ thuộc vào việc dataset được hiểu và xử lý thống nhất.

## 2.1. Basic dataset description

Document và source phải tự động xác nhận:

| Thuộc tính        |        HRP4K |
| ----------------- | -----------: |
| Resolution        |  3840 × 2160 |
| Total images      |        6.003 |
| Positive images   |        4.003 |
| Negative images   |        2.000 |
| Pothole instances |        7.217 |
| Classes           |  1 — pothole |
| Annotation        | Bounding box |
| Formats           |  YOLO + COCO |
| Train             |        4.203 |
| Validation        |          900 |
| Test              |          900 |

Dataset được thu bằng camera Sony Alpha 7 IV và Alpha 9 III từ phương tiện, trên khoảng 1.100 km đường tại Hangzhou, Huzhou và Jiaxing, chủ yếu trong điều kiện daylight/dry. 

### Geographic distribution

* Hangzhou: **2.343 — 39.0%**
* Huzhou: **3.041 — 50.7%**
* Jiaxing: **619 — 10.3%**

### Pavement

* Asphalt: **5.269 — 87.8%**
* Concrete: **734 — 12.2%**

Sự mất cân bằng asphalt/concrete này đặc biệt quan trọng vì paper cho thấy detector suy giảm mạnh trên concrete. 

---

## 2.2. Object-scale statistics

Đây là đặc điểm quan trọng nhất của HRP4K và benchmark phải giữ lại.

Paper định nghĩa scale theo:

[
r=\frac{A_{bbox}}{A_{image}}\times100%
]

và chia thành:

| Scale      | Relative area | Instances |
| ---------- | ------------: | --------: |
| Ultra-fine |       < 0.05% |     3.833 |
| Fine       |     0.05–0.1% |     1.078 |
| Medium     |     0.1–0.25% |     1.099 |
| Large      |       ≥ 0.25% |     1.207 |

Bounding box trung vị khoảng **100 × 35 px**, trong khi những object nhỏ nhất chỉ khoảng **10 × 4 px** trên ảnh 3840×2160.

Đây chính là lý do **mAP@0.5:0.95 phải được xem là metric rất quan trọng**, vì chỉ một sai số nhỏ vài pixel cũng làm IoU giảm mạnh.

### Source cần có

```text
tools/dataset/
├── verify_dataset.py
├── validate_annotations.py
├── dataset_statistics.py
├── scale_statistics.py
├── pavement_statistics.py
├── split_statistics.py
└── visualize_samples.py
```

Khi chạy:

```bash
python tools/dataset/dataset_statistics.py
```

phải sinh:

```text
outputs/dataset/
├── dataset_summary.json
├── instances_per_image.png
├── bbox_area_distribution.png
├── bbox_width_height.png
├── pavement_distribution.png
├── geographic_distribution.png
└── dataset_report.md
```

---

# 3. Dataset split — tuyệt đối không split lại ngẫu nhiên

Đây là một điểm rất quan trọng.

Paper sử dụng **video-level split**, tức mọi frame đến từ cùng một video phải nằm cùng một subset để tránh information leakage. Positive data đến từ 170 video clips; negative samples cũng giữ video-level grouping. 

Do đó benchmark chính thức phải sử dụng:

```text
train = official HRP4K train
valid = official HRP4K valid
test  = official HRP4K test
```

Không được làm:

```python
random_split(dataset, [0.7, 0.15, 0.15])
```

Nếu làm vậy, kết quả không còn trực tiếp so sánh được với paper.

---

# 4. Related Work → Baseline Methods

Đây là phần **Related Work thực sự được đưa vào benchmark**, thay vì chỉ viết survey.

Paper benchmark 6 detector. 

## Nhóm 1 — YOLO family

### YOLOv5

Baseline CNN truyền thống, đại diện cho thế hệ YOLO đã được sử dụng rất rộng rãi.

Paper:

* Params: 35.7M
* Precision: 0.651
* Recall: 0.430
* F1: 0.518
* mAP50: 0.555
* mAP50-95: 0.343
* FPPI: 0.050

### YOLOv8

Baseline YOLO hiện đại hơn:

* 25.9M
* Precision: 0.648
* Recall: 0.466
* F1: 0.542
* mAP50: 0.580
* mAP50-95: 0.390
* FPPI: 0.113

### YOLOv11

Baseline YOLO mạnh nhất trong paper về localization/false positive:

* 20.1M
* **Precision: 0.742**
* Recall: 0.422
* F1: 0.538
* mAP50: 0.596
* **mAP50-95: 0.407**
* **FPPI: 0.030**

Paper mô tả nhóm YOLO là dense one-stage detector, dựa trên CNN + feature pyramid/PAN-style neck và NMS. 

---

# 5. Transformer-based baselines

## RT-DETRv1

Đại diện cho end-to-end set-prediction detector:

* Params: 36.4M
* Precision: 0.641
* Recall: 0.560
* **F1: 0.598**
* mAP50: 0.606
* mAP50-95: 0.370
* FPPI: 0.140

## RT-DETRv2

* Params: 36.0M
* Precision: 0.621
* **Recall: 0.570**
* F1: 0.594
* mAP50: 0.605
* mAP50-95: 0.373
* FPPI: 0.113

Khác YOLO, RT-DETR sử dụng sparse queries và one-to-one matching thay vì dense prediction + NMS. 

---

# 6. D-FINE

Đây nên được xem là **accuracy baseline chính** của project.

Paper báo cáo:

* Params: **19.2M**
* Precision: 0.694
* Recall: 0.520
* F1: 0.595
* **mAP50: 0.611 — cao nhất**
* mAP50-95: 0.383
* FPPI: 0.130

D-FINE sử dụng **Fine-grained Distribution Refinement — FDR**, biểu diễn/refine các cạnh bounding box dưới dạng distribution để cải thiện localization cho object có hình dạng không đều như pothole. 

---

# 7. Baseline matrix cuối cùng

Project không cần đưa hàng chục detector vào ngay.

**Official Benchmark V1 nên cố định đúng 6 model:**

| Model     | Family      | Vai trò                           |
| --------- | ----------- | --------------------------------- |
| YOLOv5    | CNN / dense | legacy baseline                   |
| YOLOv8    | CNN / dense | modern YOLO baseline              |
| YOLOv11   | CNN / dense | localization / precision baseline |
| RT-DETRv1 | Transformer | DETR baseline                     |
| RT-DETRv2 | Transformer | improved DETR baseline            |
| D-FINE    | Transformer | best mAP50 baseline               |

Về sau có thể có:

```text
benchmark/
├── official/
└── extended/
```

`official/` luôn giữ nguyên 6 model của paper.

`extended/` chứa các cấu hình thử nghiệm mở rộng (YOLO11n, YOLO11s, SAHI, tiling, v.v.).

Như vậy benchmark suite chính thức luôn giữ được tính nhất quán và độc lập.

---

# 8. Training protocol & Optimization

Paper sử dụng:

* medium-sized variants;
* COCO-pretrained weights;
* **150 epochs**;
* official default hyperparameters;
* cùng protocol để đảm bảo fair comparison.

### 8.1. Tối ưu tốc độ Huấn luyện bằng Mixed Precision (FP16 / BF16)

Toàn bộ các mô hình (YOLOv5, YOLOv8, YOLOv11, RT-DETRv1, RT-DETRv2, D-FINE) khi huấn luyện **bắt buộc phải bật Automatic Mixed Precision (AMP - FP16 / BF16)**:

* **Tối ưu VRAM GPU**: Giảm 40–50% dung lượng VRAM tiêu thụ, cho phép tăng batch size phù hợp.
* **Tăng tốc độ tính toán**: Tận dụng triệt để Tensor Cores trên GPU (NVIDIA L40S, A100, RTX 4090, v.v.), nâng cao throughput (FPS) khi huấn luyện.
* **Đảm bảo độ chính xác**: Gradient scaling tự động giúp bảo toàn độ chính xác của các trọng số mà không làm suy giảm mAP.

### 8.2. Chế độ Chạy nhanh Kiểm thử (Smoke Mode for Fast Debugging)

Tất cả các module huấn luyện và đánh giá phải hỗ trợ cờ lệnh `--smoke` (`smoke_mode: true`) để phục vụ việc kiểm tra và sửa lỗi nhanh (dry-run pipeline verification):

* **Số lượng ảnh tối thiểu**: Chỉ load khoảng 50 – 100 ảnh ngẫu nhiên từ dataset (hoặc ~1% dữ liệu).
* **Số epoch tối thiểu**: Chạy từ **1 đến 2 epochs**.
* **Mục tiêu**: Đảm bảo pipeline code, data loader, loss calculation, evaluation metric và GPU allocation hoạt động trơn tru 100% trước khi kích hoạt đợt huấn luyện chính thức 150-epoch.

Do đó config chuẩn cho huấn luyện có dạng:

```yaml
experiment:
  dataset: hrp4k
  epochs: 150
  pretrained: coco
  seed: 42
  smoke_mode: false # Đặt true để chạy thử nghiệm nhanh (1-2 epochs, ~50 samples)

optimization:
  amp: true        # Automatic Mixed Precision
  precision: fp16  # fp16 hoặc bf16 tùy cứng GPU

dataset:
  train: data/HRP4K/train
  val: data/HRP4K/valid
  test: data/HRP4K/test

model:
  family: yolo11
  size: medium

evaluation:
  official_metrics: true
```

### 8.3. Một vấn đề reproducibility cần ghi rõ

Paper **không công bố đầy đủ mọi training parameter cần thiết để bit-for-bit reproduction**, chẳng hạn exact software commit/version và một số preprocessing/training details.

Vì vậy mục tiêu thực tế nên là:

> **protocol reproduction**, không tuyên bố exact numerical reproduction.

Repository phải pin:

```text
Python version
PyTorch version
CUDA version
Ultralytics version
RT-DETR commit
D-FINE commit
pycocotools version
AMP / Precision mode (FP16/BF16)
```

và lưu toàn bộ resolved config sau mỗi run.

---

# 9. Source-code architecture

Tôi đề xuất cấu trúc cuối:

```text
HRP4K-Benchmark/
│
├── README.md
├── requirements/
│   ├── base.txt
│   ├── yolo.txt
│   ├── rtdetr.txt
│   └── dfine.txt
│
├── configs/
│   ├── dataset/
│   │   └── hrp4k.yaml
│   │
│   ├── official/
│   │   ├── yolov5m.yaml
│   │   ├── yolov8m.yaml
│   │   ├── yolov11m.yaml
│   │   ├── rtdetr_v1.yaml
│   │   ├── rtdetr_v2.yaml
│   │   └── dfine.yaml
│   │
│   └── evaluation/
│       ├── official.yaml
│       ├── scale.yaml
│       └── pavement.yaml
│
├── hrp4k/
│   ├── data/
│   │   ├── loader.py
│   │   ├── coco.py
│   │   ├── yolo.py
│   │   └── metadata.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── ultralytics_adapter.py
│   │   ├── rtdetr_adapter.py
│   │   └── dfine_adapter.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── runner.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   └── export_predictions.py
│   │
│   └── evaluation/
│       ├── coco_eval.py
│       ├── detection_metrics.py
│       ├── fppi.py
│       ├── scale_eval.py
│       └── pavement_eval.py
│
├── tools/
│   ├── dataset/
│   ├── benchmark/
│   └── visualization/
│
├── scripts/
│   ├── train_all.sh
│   ├── evaluate_all.sh
│   └── benchmark_all.sh
│
└── outputs/
```

Điểm quan trọng nhất của architecture này là:

> **training framework có thể khác nhau nhưng evaluation phải đi qua một evaluator thống nhất.**

Không lấy trực tiếp con số `mAP` mà từng framework in ra rồi ghép vào bảng.

---

# 10. Unified prediction format

Sau inference, tất cả model phải export về cùng format, tốt nhất là COCO detection JSON:

```json
{
  "image_id": 123,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "score": 0.873
}
```

Pipeline:

```text
YOLO ────────┐
             │
RT-DETR ─────┼──→ COCO prediction JSON
             │
D-FINE ──────┘
                       ↓
                Unified evaluator
```

Đây là điều giúp benchmark thực sự công bằng.

---

# 11. Metric Evaluation — Official metrics

Benchmark bắt buộc tái tạo chính xác các metric paper sử dụng:

### Detection

* Precision
* Recall
* F1-score
* mAP@0.5
* mAP@0.5:0.95

Paper sử dụng mAP@0.5:0.95 trung bình từ IoU **0.50 → 0.95 với step 0.05**. 

### Negative-set false alarm

* **FPPI — False Positives Per Image**

Paper tính FPPI riêng trên **300 negative images của test set**. 

Về bản chất:

[
FPPI=\frac{N_{false\ positives}}{N_{negative\ images}}
]

Metric này rất đáng giữ vì với pothole detection ngoài thực tế, một detector có mAP cao nhưng liên tục báo pothole trên crack, bóng, tar repair hay concrete joint vẫn không tốt.

---

# 12. Metric Evaluation — Extended benchmark

Đây là phần tôi khuyên **thêm vào benchmark của chúng ta**, nhưng phải ghi rõ là extension chứ không phải bảng official của paper.

## Scale-aware AP

Vì HRP4K đã định nghĩa 4 nhóm scale:

```text
AP_ultra-fine
AP_fine
AP_medium
AP_large
```

Đây sẽ là metric cực kỳ giá trị cho việc đánh giá chi tiết theo từng quy mô đối tượng.

Ví dụ:

```text
Model      AP-UF   AP-F   AP-M   AP-L
YOLO11m
D-FINE
SAHI
```

Một phương pháp xử lý ảnh 4K chỉ có ý nghĩa rõ ràng nếu nó thực sự cải thiện **ultra-fine/fine targets**, không chỉ tăng aggregate mAP.

---

# 13. Pavement robustness benchmark

Paper đã chỉ ra concrete khó hơn asphalt rõ rệt. 

Ví dụ YOLOv11:

```text
             Asphalt    Concrete
F1           0.593      0.284
mAP50        0.640      0.385
mAP50-95     0.436      0.280
```

D-FINE:

```text
             Asphalt    Concrete
F1           0.636      0.436
mAP50        0.655      0.425
mAP50-95     0.413      0.260
```

Do đó evaluator nên xuất:

```text
overall
asphalt
concrete
```

và thêm:

[
\Delta AP_{material}
====================

AP_{asphalt}-AP_{concrete}
]

Metric gap này đặc biệt hữu ích khi đánh giá generalization.

---

# 14. Efficiency metrics

Paper chính chủ yếu benchmark detection accuracy + parameter count.

Benchmark mở rộng của project nên thêm:

```text
Parameters
GFLOPs
Latency / image
FPS
Peak VRAM
Model checkpoint size
```

Nhưng phải **tách khỏi official accuracy reproduction**:

```text
Accuracy Benchmark
Efficiency Benchmark
```

vì latency phụ thuộc phần cứng.

Protocol latency phải cố định:

```text
GPU
CUDA
PyTorch
batch size = 1
warmup iterations
measured iterations
precision FP32 / FP16
input resolution
preprocessing included/excluded
```

---

# 15. Final benchmark table

Project cuối cùng nên tự sinh bảng dạng:

| Model     |    Params | Precision |   Recall |       F1 |   FPPI ↓ |    mAP50 | mAP50-95 |
| --------- | --------: | --------: | -------: | -------: | -------: | -------: | -------: |
| YOLOv5    |     35.7M |      .651 |     .430 |     .518 |     .050 |     .555 |     .343 |
| YOLOv8    |     25.9M |      .648 |     .466 |     .542 |     .113 |     .580 |     .390 |
| YOLOv11   |     20.1M |  **.742** |     .422 |     .538 | **.030** |     .596 | **.407** |
| RT-DETRv1 |     36.4M |      .641 |     .560 | **.598** |     .140 |     .606 |     .370 |
| RT-DETRv2 |     36.0M |      .621 | **.570** |     .594 |     .113 |     .605 |     .373 |
| D-FINE    | **19.2M** |      .694 |     .520 |     .595 |     .130 | **.611** |     .383 |

Đây là các giá trị reference từ paper để kiểm tra reproduction. 

Repository nên tự thêm:

```text
Paper value
Our reproduced value
Absolute difference
```

Ví dụ:

| Model  | Paper mAP50 | Reproduced |   Δ |
| ------ | ----------: | ---------: | --: |
| YOLO11 |        .596 |       .xxx | xxx |
| D-FINE |        .611 |       .xxx | xxx |

Cách này hữu ích hơn việc chỉ ghi một bảng kết quả mới.

---

# 16. Các biểu đồ benchmark

Tự động generate tối thiểu:

```text
01_map50_comparison.png
02_map5095_comparison.png
03_precision_recall.png
04_fppi_comparison.png

05_scale_ap_comparison.png
06_asphalt_concrete_gap.png

07_params_vs_map.png
08_flops_vs_map.png
09_latency_vs_map.png

10_failure_examples.png
11_false_positive_examples.png
12_small_object_failures.png
```

Đặc biệt nên có:

### Accuracy–parameter Pareto

```text
mAP50-95
   ↑
   |          YOLO11m
   | D-FINE
   |
   | ...
   +----------------→ Params
```

Sau này các phương pháp xử lý khác chỉ cần được thêm vào cùng plot.

---

# 17. Reproducibility

Mỗi experiment nên tạo:

```text
runs/
└── yolov11m_seed42/
    ├── config.yaml
    ├── environment.json
    ├── train_log.csv
    ├── best.pt
    ├── predictions.json
    ├── metrics.json
    ├── scale_metrics.json
    ├── pavement_metrics.json
    └── efficiency.json
```

`environment.json` ghi:

```json
{
  "python": "...",
  "torch": "...",
  "cuda": "...",
  "gpu": "...",
  "framework": "...",
  "git_commit": "..."
}
```

---

# 18. Các phase phát triển

## Phase 1 — Dataset foundation

Hoàn thiện:

* download/setup HRP4K;
* validate file structure;
* YOLO ↔ COCO consistency check;
* check image/label pairs;
* xác nhận official split;
* dataset statistics;
* object-scale bins;
* pavement metadata.

**Definition of Done:** mọi statistic chính trong paper được source tái tạo.

---

## Phase 2 — Unified evaluator

Implement trước khi train models:

```text
Precision
Recall
F1
mAP50
mAP50-95
FPPI
```

Test evaluator bằng prediction giả và bằng checkpoint có sẵn.

**DoD:** cùng prediction → mọi framework trả cùng metric.

Đây nên là phần được ưu tiên nhất của repository.

---

## Phase 3 — YOLO baselines

Implement:

```text
YOLOv5
YOLOv8
YOLOv11
```

Do cả ba có hệ sinh thái tương đối gần nhau nên hoàn thiện nhóm này trước.

Output:

```text
3 checkpoints
3 prediction JSON
3 complete benchmark reports
```

---

## Phase 4 — Transformer baselines

Tiếp tục:

```text
RT-DETRv1
RT-DETRv2
D-FINE
```

Dùng official implementation của từng method, đúng như paper mô tả. 

Mỗi framework chỉ chịu trách nhiệm:

```text
train
load checkpoint
predict
```

Sau đó predictions phải đi về evaluator chung.

---

## Phase 5 — Official HRP4K benchmark reproduction

Chạy toàn bộ:

```bash
python benchmark.py --suite official
```

Sinh:

```text
benchmark_official.csv
benchmark_official.json
paper_comparison.csv
```

Kết quả không cần trùng từng phần nghìn với paper, nhưng phải phân tích rõ mọi deviation.

---

## Phase 6 — Extended HRP4K benchmark

Sau official reproduction mới thêm:

* scale AP;
* asphalt/concrete;
* computational efficiency;
* negative failure analysis;
* qualitative visualization.

Kết quả:

```text
benchmark_extended.csv
```

---

# 19. Phạm vi tập trung của Project

Dự án tập trung 100% vào việc **xây dựng Benchmark Suite chuẩn hóa, đánh giá toàn diện và phân tích chuyên sâu** về bộ dữ liệu HRP4K cũng như các mô hình và chiến lược xử lý hiện có.

Cấu trúc định hướng của dự án:

```text
HRP4K Benchmark & Analysis Suite
    ├── Dataset Analysis & Integrity (Phase 0)
    ├── Official Six Detector Baselines (Phase 1)
    ├── Resolution Allocation Benchmark (Phase 2)
    └── Deep Diagnostic Analysis & Explanation (Phase 3)
```

---

# 20. Final deliverables

Khi project hoàn thiện, cần có đúng **5 deliverable**:

1. **`DATASET.md`** — mô tả HRP4K, split, scale, material, annotation và thống kê.
2. **`RELATED_WORK.md`** — YOLOv5/v8/v11, RT-DETRv1/v2, D-FINE; lý do chúng trở thành baseline.
3. **Source code** — train/inference adapters + unified evaluator + dataset tools.
4. **`BENCHMARK.md`** — official reproduction + extended benchmark + bảng/biểu đồ/failure analysis.
5. **Checkpoints + prediction files + configs** — đủ để người khác kiểm tra lại mà không cần train lại từ đầu.

### Luồng phát triển cuối cùng tôi đề xuất

```text
Dataset Validation
        ↓
Dataset Description + Statistics
        ↓
Unified COCO Evaluator
        ↓
YOLOv5 / YOLOv8 / YOLOv11
        ↓
RT-DETRv1 / RT-DETRv2 / D-FINE
        ↓
Official Benchmark Reproduction
        ↓
Scale + Pavement + FPPI Analysis
        ↓
Efficiency Benchmark
        ↓
Final HRP4K Benchmark & Diagnostic Suite
```

Nếu cần ưu tiên công việc, thứ tự nên là **Dataset → Evaluator → 6 baseline → reproduction table → extended metrics → deep diagnosis**. Không nên bắt đầu bằng train cả 6 model trước khi evaluator và dataset validation đã ổn định.
