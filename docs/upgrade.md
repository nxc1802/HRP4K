Tôi đã đọc toàn bộ `main` hiện tại của `nxc1802/HRP4K`: toàn bộ package `hrp4k_suite`, `pyproject.toml`, README, Phase 0/1/2/2.1/3, script full pipeline và tài liệu paper đi kèm. Repo hiện khá nhỏ về code nhưng rất lớn về **research specification**: chỉ có 6 module Python lõi, trong khi phần roadmap khoa học nằm chủ yếu trong `docs/`.

## 1. Kết luận tổng quan

**HRP4K hiện là một research/benchmark prototype tốt, nhưng chưa phải publication-grade benchmark suite.**

Tôi đánh giá trạng thái hiện tại như sau:

| Khía cạnh                  |   Đánh giá | Nhận xét                                         |
| -------------------------- | ---------: | ------------------------------------------------ |
| Research framing           |   **9/10** | Câu hỏi nghiên cứu, protocol, limitation rất rõ  |
| Dataset methodology        |   **8/10** | Giữ official split, xử lý missing image hợp lý   |
| Code architecture hiện tại |   **7/10** | Nhỏ, dễ hiểu, interface thống nhất               |
| Evaluation foundation      | **7.5/10** | Đã có pycocotools + scale metrics                |
| Reproducibility            | **5.5/10** | Chưa pin dependency/commit/framework đầy đủ      |
| Phase 1 completion         |   **3/10** | 1/6 detector thực chất được support              |
| Phase 2 completion         |   **4/10** | Chỉ classical baselines, learned methods chưa có |
| Phase 3 completion         |   **3/10** | Diagnostic hiện đơn giản hơn spec rất nhiều      |
| Publication readiness      |   **4/10** | Còn một số vấn đề metric/protocol quan trọng     |

Điểm mạnh nhất là repo **không giả vờ đã reproduce các learned method**. README ghi rõ AutoFocus, AdaZoom, FOVEA, learned TPP và ZoomDet chưa được reproduce và `perspective-bands` không được gọi nhầm là TPP. Đây là cách làm khoa học đúng.

---

# 2. Dự án thực sự đang làm gì?

Research question lớn của project đã được định hình khá tốt:

> Với ảnh road 4K chứa rất nhiều pothole cực nhỏ, hiệu năng detector bị giới hạn bởi **architecture detector** đến đâu và bởi **cách phân bổ độ phân giải đầu vào** đến đâu?

Toàn bộ project được chia thành:

```text
HRP4K Dataset
      │
      ▼
Phase 0
Dataset integrity + dataset-conditioned analysis
      │
      ▼
Phase 1
Detector Benchmark
YOLOv5 / YOLOv8 / YOLOv11
RT-DETRv1 / RT-DETRv2 / D-FINE
      │
      ▼
Phase 2
Resolution Allocation Benchmark
Resize / Tiling / SAHI
AutoFocus / AdaZoom / FOVEA / TPP / ZoomDet
      │
      ▼
Phase 3
Deep Diagnostics
scale / resolution / localization / spatial
material / negatives / compute / failure analysis
```

Phase 3 cuối cùng muốn chuyển câu hỏi từ **“model nào mAP cao nhất?”** thành **“method nào tốt cho loại pothole nào, tại đâu trong ảnh, vì sao, và với giá compute bao nhiêu?”**. Đây là framing có giá trị nghiên cứu cao hơn leaderboard thông thường.

---

# 3. Kiến trúc code hiện tại

Code thực tế đơn giản hơn rất nhiều so với architecture cuối trong tài liệu:

```text
hrp4k_suite/
├── cli.py
├── dataset.py
├── training.py
├── processing.py
├── evaluation.py
└── diagnostics.py
```

Luồng hiện tại:

```text
                         ┌──────────────┐
                         │ HRP4K COCO   │
                         └──────┬───────┘
                                │
                         dataset.py
                  integrity / statistics
                                │
                                ▼
                      smoke/full subset
                                │
                                ▼
                         training.py
                           YOLO only
                                │
                                ▼
                         processing.py
        ┌────────┬────────┬───────┬─────────────┐
        ▼        ▼        ▼       ▼             ▼
     resize   uniform2 uniform3  sliced     perspective
                                inference      bands
        └────────┴────────┴───────┴─────────────┘
                                │
                           COCO JSON
                                │
                                ▼
                         evaluation.py
                  pycocotools + custom metrics
                                │
                                ▼
                        diagnostics.py
                                │
                                ▼
                         Phase 3 report
```

Điểm kiến trúc tốt nhất là **mọi processing method đều quay về prediction format chung rồi đi qua evaluator chung**. Đây chính xác là nguyên tắc Phase 1 đặt ra: framework training có thể khác nhau, nhưng evaluator không được khác nhau.

---

# 4. Phase 0 — Dataset Analysis

Đây là phần code hoàn chỉnh nhất.

`dataset.py` thực hiện được khá nhiều thứ: đọc official `train/valid/test`, kiểm tra image thật tồn tại, invalid bbox, positive/negative image, scale bins, bbox statistics, spatial distribution, object density, image-quality sample và xây `difficulty_index.csv`. Missing image được bỏ qua thay vì resplit dataset.

Cơ chế này phù hợp với vấn đề local dataset hiện tại: tài liệu nói local có đủ 900 valid + 900 test nhưng chỉ có 2.286/4.203 train image. Phase 0 chủ ý bảo toàn official video-level split và không tạo một random split mới.

Một thiết kế rất tốt khác là smoke subset được lấy deterministic với seed và cố giữ khoảng 2/3 positive, 1/3 negative. Dataset smoke dùng **symlink**, tránh duplicate ảnh 4K.

Tuy nhiên, tài liệu Phase 0 tham vọng hơn code khá nhiều. Spec yêu cầu Spearman correlation, conditional variance, split-distribution distance như KS/Jensen–Shannon, texture analysis, richer positive-vs-negative analysis và domain subgroup.  Code hiện chủ yếu có Pearson correlation, basic distribution statistics và brightness/contrast/Laplacian sharpness trên một số sample.

Vì vậy tôi sẽ gọi Phase 0 hiện tại là:

**“Engineering Phase 0 complete, research Phase 0 khoảng 65–75%.”**

---

# 5. Phase 1 — Detector benchmark

Research spec Phase 1 rất tốt.

Official matrix cố định 6 model:

| Model     | Role                        |
| --------- | --------------------------- |
| YOLOv5m   | legacy CNN                  |
| YOLOv8m   | modern YOLO                 |
| YOLOv11m  | precision/localization      |
| RT-DETRv1 | Transformer                 |
| RT-DETRv2 | improved DETR               |
| D-FINE    | distributional localization |

Spec còn yêu cầu mọi framework chỉ chịu trách nhiệm `train → predict`, sau đó đều export COCO JSON để evaluator chung xử lý.

**Nhưng code hiện chỉ implement Ultralytics YOLO.**

`training.py` đơn giản chỉ:

```python
from ultralytics import YOLO
model = YOLO(weights)
model.train(...)
```

Không có adapter YOLOv5 riêng, RT-DETRv1/v2 hay D-FINE.

Do đó Phase 1 hiện tại mới có **infrastructure skeleton**, chưa có “Official Six Detector Baseline”.

---

# 6. Có một lỗi protocol quan trọng trong training

Tài liệu full pipeline tuyên bố dùng:

```text
Optimizer = SGD
lr0 = 0.01
momentum = 0.937
weight_decay = 0.0005
```

và thậm chí gọi đây là hyperparameter chuẩn theo paper.

Nhưng `training.py` **không truyền**:

```python
optimizer="SGD"
```

vào `model.train()`.

Với Ultralytics hiện đại, `optimizer` có thể mặc định ở chế độ `auto`. Khi đó framework có quyền chọn optimizer khác và thậm chí override một số hyperparameter.

Đây là điểm nên sửa **trước khi chạy benchmark 150 epochs**:

```python
optimizer="SGD"
```

Nếu mục tiêu là protocol reproduction.

Ngoài ra, docs lại tự thừa nhận paper không pin đủ software version và configuration để exact numerical reproduction.  Vì vậy câu trong `RUN_FULL_PIPELINE.md` kiểu **“khớp 100% Hyperparameters Paper”** nên bỏ. Chính xác hơn nên gọi:

> Paper-aligned reproduction protocol.

---

# 7. Một contradiction khá rõ: full training guard

README hiện nói:

> “Full local training bị chặn có chủ đích. `train` bắt buộc có `--smoke`.”

Nhưng CLI hoàn toàn cho phép:

```bash
hrp4k train ...   # không --smoke
```

và `training.py` khi `smoke=False` sẽ chạy toàn bộ `epochs`.

Thậm chí chính `run_full_pipeline.sh` gọi 150 epoch **không có `--smoke`**.

Vậy hiện tại:

```text
README
    ↓
full training forbidden

CODE + FULL PIPELINE
    ↓
full training allowed
```

Phải chọn một trong hai và đồng bộ lại.

Theo roadmap dự án, tôi nghĩ **code hiện tại hợp lý hơn README**: smoke để debug, full mode cho benchmark thật.

---

# 8. Một vấn đề khoa học lớn hơn: “Full Official Benchmark” chưa thực sự official

Full pipeline chuẩn bị:

```text
Train = 2286
Valid = 900
Test  = 900
```

do local thiếu 1.917 train images.

Trong khi official train phải là:

```text
4203 images
```

Phase 1 cũng nhấn mạnh không được thay đổi official split.

Giữ đúng validation/test là rất tốt, nhưng train chỉ còn:

[
2286/4203 \approx 54.4%
]

Tức thiếu khoảng **45.6% training images**.

Do đó kết quả model được train theo `run_full_pipeline.sh` **không nên được gọi là official HRP4K reproduction** và không nên lấy deviation với paper để đánh giá khả năng reproduce model.

Tên đúng hơn:

> **Full Local-Available Benchmark**

Còn publication run chỉ nên mở khi đủ 4.203 training images.

---

# 9. Phase 2 — Resolution Allocation

Đây là phần có research value cao nhất của project.

Research spec phân biệt rất đúng:

```text
Global
Resize

Exhaustive
Uniform tiling

Slicing
SAHI

Adaptive crops
AutoFocus / AdaZoom

Adaptive resampling
FOVEA / TPP / ZoomDet
```

và yêu cầu compare accuracy–compute Pareto chứ không chỉ mAP.

Current code support:

```text
resize                 ✅
uniform-2              ✅
uniform-3              ✅
sahi-like slicing      ✅
perspective-bands      ✅

AutoFocus              ❌
AdaZoom                ❌
FOVEA                   ❌
Learned TPP             ❌
ZoomDet                 ❌
```

Status matrix trong code nói chính xác điều này.

Phase 2.1 cũng có quyết định kỹ thuật rất hợp lý: **không cố port tất cả ancient research code vào một environment PyTorch mới ngay lập tức**. AutoFocus là MXNet/custom stack, FOVEA/TPP dùng MMDetection cũ, ZoomDet có implementation riêng; container isolation là hướng tốt hơn.

---

# 10. “SAHI” hiện tại thực ra không phải SAHI package

Implementation hiện tự viết slicing:

```python
if method == "sahi":
    ...
    create overlapping views
    detector(view)
    ...
    global NMS
```

README đã cẩn thận ghi đây là:

> “framework-independent sliced inference”, không phải SAHI sliced fine-tuning.

Tôi khuyên rename method code-level thành:

```text
sliced
```

hoặc:

```text
sliced-nms
```

và khi thật sự tích hợp `obss/sahi` thì dùng:

```text
sahi-i
sahi-sf
```

Như vậy benchmark sẽ không bị reviewer bắt lỗi naming.

---

# 11. `perspective-bands` hiện chưa thực sự phân bổ thêm resolution

Đây là một vấn đề kỹ thuật khá quan trọng.

Method hiện chia ảnh thành ba horizontal bands:

```python
[0, 0.45H, 0.72H, H]
```

nhưng mỗi band vẫn có **full width W**.

Ví dụ ảnh:

```text
3840 × 2160
```

resize baseline vào YOLO `imgsz=640` sẽ scale width:

```text
3840 → 640
```

Một horizontal band vẫn rộng:

```text
3840
```

nên cũng scale:

```text
3840 → 640
```

Kết quả: object không thực sự được phóng to theo chiều ngang. Nó chủ yếu loại bớt vertical context.

Do đó `perspective-bands` hiện **không phải một strong geometry resolution-allocation baseline**.

Nếu muốn geometry baseline thực sự có ý nghĩa, nên crop cả `x` và `y`, hoặc warp/resample với magnification factor thay đổi theo `y`.

---

# 12. Compute accounting hiện chưa đủ

Code đo:

```text
detector_calls
processed_source_pixels
processed_area_ratio
latency
fusion suppression
```

Nhưng với uniform tiling 2×2:

```text
4 crops
```

tổng source area vẫn đúng bằng diện tích ảnh gốc, nên:

```text
processed_area_ratio = 1.0
```

Trong khi detector thực tế chạy **4 lần**.

3×3 cũng:

```text
processed_area_ratio = 1.0
detector_calls = 9
```

Vì thế processed source area không thể đại diện cho compute.

Phase 2 spec thực ra đã yêu cầu thêm:

```text
detector-input pixels
GFLOPs
P95 latency
Peak VRAM
```

Tôi sẽ bổ sung một metric rất đơn giản:

[
CAF_{input}
===========

\frac{\sum_i W_i^{detector}H_i^{detector}}
{W_{baseline}^{detector}H_{baseline}^{detector}}
]

để biết mỗi method khuếch đại detector compute khoảng bao nhiêu lần.

---

# 13. Evaluator: foundation tốt nhưng có một vấn đề FPPI

`evaluation.py` hiện đã tốt hơn nhiều so với tài liệu Phase 2 cũ: nó dùng `pycocotools.COCOeval` cho AP50/AP75/AP50:95 và có NumPy fallback.

Nhưng code hiện xuất hai metric:

```python
"FPPI": fp / len(all_images)

"negative_FPPI":
    predictions_on_negative_images / len(negative_images)
```

Theo protocol mà chính Phase 1 mô tả, HRP4K FPPI được tính trên **300 negative test images**.

Vì vậy metric thực sự tương ứng với paper hiện là:

```text
negative_FPPI
```

chứ không phải field:

```text
FPPI
```

Tôi khuyên đổi thành:

```json
{
  "FPPI_official": ...,
  "FPPI_all_images": ...
}
```

Nếu không, rất dễ lấy nhầm field khi sinh benchmark table.

---

# 14. Cần kiểm tra lại `category_id`

Prediction exporter hiện hard-code:

```python
"category_id": 0
```

Nhưng documentation về unified COCO format lại đưa ví dụ:

```json
"category_id": 1
```

Tôi chưa thấy official dataset JSON được commit trong repo nên không thể kết luận ID thật là `0` hay `1`.

Nhưng đây là **schema invariant bắt buộc phải validate**, bởi COCOeval yêu cầu prediction category ID khớp với GT category ID.

Không nên hard-code.

Nên đọc từ:

```python
gt["categories"][0]["id"]
```

hoặc xây mapping explicit.

---

# 15. Có một bug cụ thể trong full pipeline report

`run_full_pipeline.sh` inference ba method:

```text
resize
sahi
perspective-bands
```

nhưng chỉ evaluate:

```text
resize
sahi
```

Không evaluate `perspective-bands`.

Sau đó:

```bash
diagnose --predictions outputs/predictions/*.json
```

sẽ đọc cả `perspective_bands.json`.

Trong `diagnostics.py`, nếu corresponding:

```text
perspective_bands_metrics.json
```

không tồn tại thì:

```python
metrics = {}
```

và report render:

```python
metric.get("AP50", 0)
metric.get("AP50_95", 0)
```

Tức report có thể hiển thị:

```text
perspective-bands AP50 = 0
AP50:95 = 0
```

**dù method chưa được evaluate**, không phải score thật bằng zero.

Đây là bug cần sửa sớm.

Missing metric phải hiện:

```text
N/A / NOT_EVALUATED
```

và full script nên evaluate mọi prediction trước `diagnose`.

---

# 16. Phase 3 hiện cách rất xa specification

Tài liệu Phase 3 rất mạnh. Nó yêu cầu:

```text
resolution sensitivity
effective object size
scale-conditioned ranking
gain decomposition
IoU-threshold decay
localization taxonomy
spatial/perspective heatmaps
material robustness
negative FP taxonomy
density
compute Pareto
architecture × processing
bootstrap CI
paired-image analysis
difficulty modeling
gap profiles
failure gallery
method suitability map
```

Current `diagnostics.py` chỉ chủ yếu có:

```text
effective bbox size
method metrics table
number predictions
mean calls
mean latency
status matrix
```

`evaluation.py` có tạo per-image TP/FP/FN/localization sidecar, đây là nền tảng tốt, nhưng `diagnostics.py` chưa thực sự tận dụng nó.

Do đó Phase 3 theo code chỉ khoảng **20–30% specification**.

---

# 17. Reproducibility hiện còn yếu

Project muốn làm benchmark reproducible, nhưng `pyproject.toml` hiện dùng dependency dạng:

```text
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
ultralytics>=8.3
pycocotools>=2.0.7
PyYAML>=6
```

Không có:

```text
lock file
exact torch version
exact ultralytics version
CUDA matrix
Docker image
GitHub Actions
tests
```

và repo hiện cũng chưa có `tests/` hay `.github/workflows/`.

Điều này chưa tương thích với claim benchmark reproducibility.

Tối thiểu mỗi official experiment nên pin:

```text
Python
PyTorch
CUDA
cuDNN
framework version/commit
model implementation commit
dataset manifest/hash
seed
resolved config
GPU name
```

`training.environment_snapshot()` hiện đã có Python, platform, Torch, CUDA, Ultralytics và git commit, tức foundation có sẵn.

---

# 18. Điểm tôi đánh giá rất cao trong research design

Điểm quan trọng nhất là project đã từ bỏ hướng ban đầu kiểu:

```text
“I have a new AdaPoth method,
now prove that it works.”
```

và chuyển thành:

```text
Dataset facts
    ↓
controlled baselines
    ↓
resolution allocation benchmark
    ↓
deep diagnostics
    ↓
data-driven conclusion
```

Đây là hướng mạnh hơn rất nhiều về research.

Phase 0 thậm chí ghi rõ không được dùng dataset analysis để “chứng minh trước” một method cụ thể.

Phase 3 cũng yêu cầu:

```text
Dataset fact
→ Observation
→ Hypothesis
→ Controlled evidence
→ Explanation
```

thay vì suy diễn mechanism chỉ từ một con số mAP.

Đây là phần tôi nghĩ nên **giữ nguyên tuyệt đối**.

---

# 19. Thứ tự tôi khuyên nâng cấp

Nếu mục tiêu là biến repo này thành benchmark thực sự mạnh, tôi sẽ ưu tiên đúng thứ tự sau:

1. **Correctness gate:** sửa FPPI naming, category mapping, missing-metric=0 bug, explicit SGD, README/full-mode contradiction.
2. **Reproducibility gate:** lock dependency + Docker + dataset manifest + config snapshot + unit tests + CI.
3. **Phase 0 completion:** split-shift metrics, richer difficulty index và visualization/raw tables.
4. **Phase 1:** xây `DetectorAdapter` rồi hoàn thành đúng 6 official models.
5. **Phase 1 reproduction:** đủ 4.203 train images trước khi so với paper.
6. **Phase 2 classical:** resize resolution sweep + uniform + official SAHI trước.
7. **Compute accounting:** detector pixels, FLOPs, VRAM, P50/P95 latency.
8. **Phase 2 learned:** ZoomDet → TPP → FOVEA → AutoFocus → AdaZoom như Phase 2.1 đã đề xuất.
9. **Phase 3:** xây analysis database từ GT + saved predictions, sau đó mới làm bootstrap, paired analysis, taxonomy và Pareto.
10. **Publication layer:** tự generate benchmark tables, paper-vs-reproduced delta, figures và final report.

---

# 20. Architecture tôi nghĩ repo nên tiến tới

Không cần refactor thành một framework khổng lồ. Với quy mô này, architecture phù hợp sẽ là:

```text
hrp4k/
├── data/
│   ├── dataset.py
│   ├── manifest.py
│   └── analysis.py
│
├── detectors/
│   ├── base.py
│   ├── ultralytics.py
│   ├── rtdetr.py
│   └── dfine.py
│
├── processors/
│   ├── base.py
│   ├── resize.py
│   ├── uniform.py
│   ├── sahi.py
│   └── external/
│       ├── zoomdet/
│       ├── tpp/
│       ├── fovea/
│       ├── autofocus/
│       └── adazoom/
│
├── prediction/
│   ├── schema.py
│   ├── transforms.py
│   └── fusion.py
│
├── evaluation/
│   ├── coco.py
│   ├── scale.py
│   ├── negative.py
│   └── efficiency.py
│
├── analysis/
│   ├── localization.py
│   ├── spatial.py
│   ├── paired.py
│   ├── statistics.py
│   └── pareto.py
│
└── cli.py
```

Điểm quyết định là có hai abstraction độc lập:

```text
DetectorAdapter
Processor
```

Sau đó pipeline chung:

```text
Image
  ↓
Processor
  ↓
ProcessedView[]
  ↓
DetectorAdapter
  ↓
inverse transform
  ↓
fusion
  ↓
canonical COCO predictions
  ↓
Evaluator
```

Như vậy sau này thay YOLO → D-FINE hoặc Resize → ZoomDet không phá evaluator.

---

## Kết luận

**Hướng nghiên cứu của HRP4K hiện mạnh hơn mức độ hoàn thiện của source code.** Đây không phải vấn đề xấu: research specification đã khá chín, trong khi implementation mới ở `v0.1.0` và hiện chủ yếu là một **end-to-end engineering smoke harness**.

Nếu sửa các vấn đề correctness tôi nêu ở trên, sau đó hoàn thiện 6 detector trước khi lao vào learned Phase 2, thì project có thể trở thành một benchmark có giá trị thực sự. Ngược lại, nếu chạy ngay full 150 epochs và bắt đầu tạo bảng “paper reproduction” từ 2.286 train images hiện tại, kết quả rất dễ trông hoàn chỉnh nhưng **không đủ mạnh về mặt khoa học**.

Nếu xét riêng code hiện tại, **ưu tiên số 1 tôi sẽ làm tiếp là refactor `src/hrp4k_suite` thành kiến trúc production/research-grade và sửa các lỗi evaluator/pipeline trên trước khi thêm model mới**.
