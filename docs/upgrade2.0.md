Tôi đã đọc lại `main` mới nhất và so trực tiếp với snapshot trước `7cf14bd`. Bản nâng cấp hiện nằm ở commit `b8c268c...` — **“Clarify local-available benchmark pipeline and safeguards”**.

Kết luận ngắn: **bản nâng cấp tốt hơn đáng kể và đã sửa gần như toàn bộ lỗi correctness quan trọng tôi chỉ ra trước đó.** Repo giờ đã từ “research prototype có evaluator tương đối tốt” thành một **benchmark engineering framework khá chắc**, nhưng vẫn chưa phải publication-grade hoàn chỉnh vì Phase 1/2 còn thiếu phần lớn model/method và tôi vẫn thấy khoảng 3–4 vấn đề kỹ thuật đáng sửa trước khi chạy các experiment lớn.

## 1. Đánh giá mới

| Thành phần             |  Trước |      Hiện tại | Nhận xét                                |
| ---------------------- | -----: | ------------: | --------------------------------------- |
| Research framing       |   9/10 |      **9/10** | Vốn đã mạnh                             |
| Dataset methodology    |   8/10 |    **8.7/10** | Có thêm split-shift, Spearman, manifest |
| Architecture           |   7/10 |      **8/10** | Đã có `DetectorAdapter`                 |
| Evaluation correctness | 7.5/10 |      **9/10** | Các lỗi quan trọng gần như đã sửa       |
| Reproducibility        | 5.5/10 |    **7.5/10** | Lock + manifest + pip freeze + CI       |
| Phase 0 implementation |   ~70% |   **~80–85%** | Khá hoàn chỉnh                          |
| Phase 1 implementation |   ~30% |      **~35%** | Infrastructure tốt hơn, model vẫn thiếu |
| Phase 2 implementation |   ~40% |   **~45–50%** | Classical benchmark tốt hơn             |
| Phase 3 implementation |   ~25% |      **~40%** | Có Pareto + paired analysis             |
| Publication readiness  |  ~4/10 | **~5.5–6/10** | Correctness cao hơn, scope vẫn thiếu    |

Điểm khác biệt quan trọng là: **publication readiness chưa tăng quá nhiều vì chưa có full research results, nhưng engineering reliability tăng rất mạnh.**

---

# 2. Các lỗi tôi chỉ ra trước đây đã được sửa thế nào?

### `category_id` — đã sửa đúng

Trước đây prediction hard-code:

```python
category_id = 0
```

Bây giờ code đọc trực tiếp category từ COCO:

```python
category_ids = [
    int(category["id"])
    for category in coco.get("categories", [])
]

if len(category_ids) != 1:
    raise ValueError(...)

category_id = category_ids[0]
```

và `UltralyticsAdapter` nhận ID này.

Evaluator cũng reject prediction dùng category không tồn tại trong GT.

Test riêng đã được thêm:

```python
with self.assertRaises(ValueError):
    evaluate(... category_id=0 ...)
```

trong khi GT dùng `category_id=7`.

**=> Sửa hoàn chỉnh.**

---

# 3. FPPI — đã sửa rất tốt

Trước đây field `FPPI` thực ra tính trên toàn test set, trong khi HRP4K official FPPI cần negative images.

Bây giờ evaluator tách rõ:

```text
FPPI_official
FPPI_all_images
negative_image_false_alarm_rate
```

Trong đó:

[
FPPI_{official}
===============

\frac{\text{FP on negative images}}
{\text{number of negative images}}
]

Test cũng xác nhận:

```text
1 negative image
1 false positive

FPPI_official = 1.0
FPPI_all_images = 0.5
```

Đây là thay đổi rất quan trọng vì nó loại được một nguồn sai số benchmark lớn.

**=> Sửa hoàn chỉnh.**

---

# 4. Optimizer SGD — đã sửa

Trước:

```python
model.train(
    lr0=0.01,
    momentum=0.937,
    ...
)
```

nhưng không chỉ định `optimizer`, nên Ultralytics có thể dùng `auto`.

Bây giờ có cả config:

```python
"optimizer": "SGD"
```

và actual call:

```python
model.train(
    optimizer="SGD",
    lr0=0.01,
    ...
)
```

Docs cũng đổi cách diễn đạt từ “100% hyperparameters paper” sang **project-resolved paper-aligned protocol**, chính xác hơn về mặt khoa học.

**=> Sửa rất đúng.**

---

# 5. Full training guard — đã sửa đúng bản chất

Đây là upgrade tôi đánh giá cao.

Bây giờ muốn chạy non-smoke phải explicit:

```bash
--allow-full
```

Nếu dataset thiếu official 4.203 train images còn cần:

```bash
--allow-incomplete-train
```

Nếu không:

```python
raise ValueError(...)
```

Full local script giờ gọi rõ:

```text
LOCAL-AVAILABLE TRAINING
(NOT OFFICIAL REPRODUCTION)
```

và sử dụng cả hai flag.

Đây chính là distinction cần có:

```text
smoke
≠
local-available full training
≠
official reproduction
```

Trước đây ba trạng thái này hơi lẫn nhau. Bây giờ rõ ràng hơn rất nhiều.

---

# 6. Reproducibility đã được nâng đáng kể

`prepare_dataset` giờ tạo manifest chứa:

```text
source
seed
official_reference
annotation_sha256
declared_images
available_images
selected_images
official_training_complete
benchmark_label
```

Training lưu:

```text
resolved_config.json
environment.json
```

và `environment.json` bây giờ còn chứa toàn bộ:

```text
pip freeze --all
```

cùng Python, Torch, CUDA, Ultralytics và git commit.

Ngoài ra có environment lock cho smoke đã verify:

```text
numpy==2.5.1
opencv-python==5.0.0.93
ultralytics==8.4.108
pycocotools==2.0.11
torch==2.13.0
...
```

Đây là bước tiến lớn.

---

# 7. CI thực sự đã chạy thành công

Repo bây giờ có GitHub Actions:

```yaml
python-version:
  - "3.11"
  - "3.12"

python -m pip install -e .
python -m unittest discover -v
```

Tôi kiểm tra workflow run của chính commit mới: CI đã **completed / success**.

Hai job Python 3.11 và 3.12 cũng đều pass.

Điều này tốt hơn rất nhiều so với bản trước vốn không có automated regression protection.

---

# 8. Phase 0 đã tiến gần specification hơn

Lần trước tôi nói Phase 0 thiếu:

```text
Spearman
conditional variance
KS
Jensen–Shannon
positive-vs-negative quality
```

Bây giờ gần như tất cả đã xuất hiện.

Có:

```python
spearman_y_bottom_log_area
spearman_y_center_log_area
```

và y-bands có:

```python
conditional_variance
```

Split shift giờ có:

```text
scale_js_divergence
area_ratio_ks_distance
y_center_ks_distance
aspect_ratio_ks_distance
```

Image quality cũng đã tách:

```text
positive
vs
negative
```

với brightness, contrast và sharpness.

Vì vậy tôi nâng Phase 0 từ khoảng **65–75% lên 80–85% specification**.

---

# 9. Architecture đã bắt đầu đúng hướng

Một module mới rất đáng giá:

```text
hrp4k_suite/detectors.py
```

với interface:

```python
class DetectorAdapter(Protocol):
    def predict(
        self,
        image,
        image_size,
        confidence
    ) -> list[dict]:
        ...
```

và implementation đầu tiên:

```text
UltralyticsAdapter
```

Đây đúng với architecture tôi đề xuất trước:

```text
Processor
     │
     ▼
ProcessedView
     │
     ▼
DetectorAdapter
     │
     ▼
canonical prediction
     │
     ▼
Unified evaluator
```

Đây là foundation rất tốt để thêm RT-DETR/D-FINE sau này mà không phá processing/evaluation.

Tuy nhiên **training side chưa có abstraction tương ứng**. `training.py` vẫn gọi Ultralytics trực tiếp.

Do đó Phase 1 vẫn chưa được xem là multi-framework framework hoàn chỉnh.

---

# 10. Naming SAHI đã được sửa đúng

Trước đây:

```text
method = sahi
```

nhưng algorithm thực tế chỉ là self-implemented sliced inference.

Bây giờ primary name là:

```text
sliced-nms
```

còn:

```text
sahi
```

chỉ là deprecated compatibility alias.

`METHOD_STATUS` cũng viết rõ:

```text
sliced-nms:
implemented in-house sliced inference;
not official SAHI
```

Đây là cách naming khoa học chính xác hơn.

---

# 11. `perspective-bands` đã được thay bằng baseline tốt hơn

Đây cũng trực tiếp giải quyết critique trước.

Old:

```text
full image width
+
horizontal bands
```

nên object không thực sự được magnify theo chiều ngang.

New:

```text
perspective-grid
```

chia thành ba band:

```text
far    → 4 horizontal crops
mid    → 3 crops
near   → 2 crops
```

Tức far road thực sự nhận nhiều detector passes hơn:

```text
           road image
              │
     ┌────────┴────────┐
     │  far: 4 crops  │
     ├───────────────┤
     │  mid: 3 crops  │
     ├───────────────┤
     │ near: 2 crops  │
     └───────────────┘
```

Test còn assert:

```python
len(perspective) == 9
source_width < 3840
```

Đây mới thực sự là một **hand-designed resolution-allocation baseline**.

Và code vẫn cẩn thận không gọi nó là TPP.

---

# 12. Compute benchmark tốt hơn đáng kể

Trước đây chỉ có:

```text
mean latency
calls
source pixels
```

Bây giờ thêm:

```text
detector_input_pixels
compute_amplification_input
P50 latency
P95 latency
P95 detector calls
Peak VRAM
warmup
```

Ví dụ nominal:

```text
Resize
calls = 1
CAF = 1×

Uniform 2×2
calls = 4
CAF ≈ 4×

Uniform 3×3
calls = 9
CAF ≈ 9×
```

Đây phù hợp hơn nhiều với Phase 2 research question:

> accuracy tăng bao nhiêu cho mỗi unit compute tăng thêm?

---

# 13. Phase 3 không còn chỉ là report table

`diagnostics.py` hiện tính:

```text
effective object size
scale sensitivity
AP50–AP75 localization gap
paired per-image method comparison
accuracy–compute Pareto frontier
```

Đặc biệt Pareto selection:

```python
AP50_95 ↑
compute_amplification_input ↓
```

và loại các method bị dominated.

Per-image comparison cũng bắt đầu trả lời:

```text
Resize wins
Sliced wins
ties
```

thay vì chỉ nhìn aggregate mAP.

Phase 3 vì vậy đã tiến từ khoảng 25% lên khoảng **40% research spec**.

---

# 14. Bug “missing metric = 0” đã được sửa

Trước đây method chưa evaluate có thể xuất:

```text
AP50 = 0
```

làm người đọc tưởng model thực sự đạt zero.

Bây giờ:

```python
evaluation_status:
    evaluated
    not_evaluated
```

và `_format_metric(None)` trả:

```text
N/A
```

Có regression test riêng cho vấn đề này.

**=> Sửa hoàn chỉnh.**

---

# 15. Full pipeline cũng đã evaluate đủ method

Trước đây `perspective-bands` được predict nhưng không được evaluate.

Bây giờ script evaluate cả:

```text
resize
sliced-nms
perspective-grid
```

trước Phase 3.

Và `run-smoke` trong CLI cũng loop qua cả ba method rồi evaluate từng prediction.

Đây là một regression fix tốt.

---

# 16. Nhưng tôi vừa tìm thấy một bug mới/bug còn sót khá quan trọng

Trong full script:

```bash
python -m hrp4k_suite diagnose \
  --predictions outputs/predictions/*.json
```

Nhưng cùng thư mục đó chứa:

```text
resize_640.json
resize_640_metrics.json
resize_640_metrics_per_image.json

sliced_nms_960.json
sliced_nms_960_metrics.json
sliced_nms_960_metrics_per_image.json

...
```

`diagnose()` **không filter schema hoặc filename**.

Nó đọc bất kỳ JSON nào truyền vào:

```python
predictions, payload = _load_predictions(path)
method = payload.get("method") or path.stem
```

Do đó wildcard có khả năng tạo các “method” giả như:

```text
resize_640_metrics
resize_640_metrics_per_image
sliced_nms_960_metrics
...
```

trong Phase 3 report.

Đây là lỗi cần sửa.

Tốt nhất chuyển cấu trúc thành:

```text
outputs/
├── predictions/
│   ├── resize.json
│   ├── sliced.json
│   └── perspective.json
│
└── metrics/
    ├── resize.json
    ├── sliced.json
    └── perspective.json
```

hoặc đơn giản script truyền explicit:

```bash
--predictions \
  outputs/predictions/resize_640.json \
  outputs/predictions/sliced_nms_960.json \
  outputs/predictions/perspective_grid.json
```

**Mức ưu tiên: cao.**

---

# 17. Official completeness guard vẫn còn một loophole

Manifest hiện quyết định:

```python
official_training_complete =
    selected_train_images == 4203
```

Nhưng official training protocol còn cần validation split đầy đủ.

Ví dụ sau này khi có full dataset, người dùng chạy:

```bash
prepare-dataset \
  --train-limit 4203 \
  --valid-limit 12 \
  --test-limit 12
```

thì:

```text
official_training_complete = True
```

dù validation chỉ có 12/900 ảnh.

Điều này có thể làm model selection/validation protocol sai nghiêm trọng.

Nên đổi thành ít nhất:

```python
official_training_complete = (
    train.selected_images == 4203
    and valid.selected_images == 900
)
```

và có thêm:

```python
official_benchmark_complete = (
    train == 4203
    and valid == 900
    and test == 900
)
```

Sau đó training guard dùng cái đầu, benchmark/report guard dùng cái sau.

---

# 18. `prepare-dataset` có default hơi nguy hiểm

Hiện `prepare-smoke` và `prepare-dataset` dùng cùng default:

```text
train = 24
valid = 12
test = 12
```

Tức:

```bash
hrp4k prepare-dataset
```

nghe như chuẩn bị dataset thật nhưng thực tế lại tạo dataset gần giống smoke.

Tôi sẽ để:

```text
prepare-smoke
24 / 12 / 12

prepare-dataset
all available / all available / all available
```

hoặc `limit=None`.

Đồng thời rename internal function:

```python
prepare_smoke_dataset()
```

thành:

```python
prepare_dataset_view()
```

vì hiện function này được dùng cho cả smoke và full/local dataset.

Đây chủ yếu là API cleanliness, nhưng giúp tránh experiment configuration mistake.

---

# 19. Compute metric vẫn chưa hoàn toàn chính xác

Code hiện ghi:

```python
detector_input_pixels =
    len(views) * image_size * image_size
```

Điều này thực chất gần với:

> nominal detector canvas budget

chứ chưa chắc là số pixel tensor detector thực sự xử lý, vì Ultralytics có letterbox/aspect-ratio/stride behavior.

Do đó tên hiện tại:

```text
detector_input_pixels
```

hơi mạnh.

Tôi sẽ gọi:

```text
nominal_detector_canvas_pixels
```

hoặc đo tensor shape sau preprocessing.

Tương tự:

```python
started = time.perf_counter()
```

được đặt **sau khi**:

```python
views = make_views(...)
```

nhưng metadata lại ghi:

```text
latency_includes_preprocessing_and_fusion = True
```

Trong thực tế timing không bao gồm toàn bộ `make_views()` và cũng không bao gồm image decoding.

Với Resize/tiling thì sai khác có thể nhỏ. Nhưng sau này với:

```text
FOVEA
TPP
ZoomDet
```

preprocessing/warp chính là một phần cost rất quan trọng.

Tôi khuyên định nghĩa hai metric:

```text
processor_latency_ms
detector_latency_ms
end_to_end_latency_ms
```

và benchmark headline dùng end-to-end.

---

# 20. CI hiện tốt, nhưng chưa test “vision stack”

Đây là limitation lớn nhất của CI hiện tại.

Workflow cài:

```bash
pip install -e .
```

chứ không phải:

```bash
pip install -e '.[vision]'
```

Mà base dependency trong `pyproject.toml` chỉ có:

```text
numpy
```

Do đó CI hiện tại **không test**:

```text
pycocotools backend
OpenCV
Ultralytics
training
actual inference
GPU/CPU vision integration
```

Evaluator tests trên CI nhiều khả năng sử dụng NumPy fallback chứ không phải official COCO backend.

CI này rất hữu ích cho regression logic, nhưng chưa phải integration CI.

---

# 21. Phase 1 vẫn là bottleneck lớn nhất

Dù có `DetectorAdapter`, status hiện vẫn là:

```text
YOLOv11         implemented
YOLOv8          adapter-compatible
YOLOv5          adapter-compatible

RT-DETRv1       external required
RT-DETRv2       external required
D-FINE          external required
```

Tức contribution quan trọng:

```text
Official Six Detector Benchmark
```

vẫn chưa tồn tại ở mức experiment.

Architecture đã chuẩn bị tốt hơn, nhưng **research completion của Phase 1 gần như chưa thay đổi nhiều**.

---

# 22. Phase 2 cũng tương tự

Hiện đã có classical/in-house:

| Method           | Status |
| ---------------- | ------ |
| Resize           | ✅      |
| Uniform 2×2      | ✅      |
| Uniform 3×3      | ✅      |
| Sliced-NMS       | ✅      |
| Perspective-grid | ✅      |
| Official SAHI    | ❌      |
| ZoomDet          | ❌      |
| TPP              | ❌      |
| FOVEA            | ❌      |
| AutoFocus        | ❌      |
| AdaZoom          | ❌      |

Status code vẫn nói rõ các learned method cần external reproduction.

Vì vậy **không nên tiếp tục dành nhiều thời gian polish classical infrastructure nữa** trước khi bắt đầu Phase 1 official models hoặc Phase 2 official methods.

Foundation hiện đã đủ tốt.

---

# 23. Một technical debt nhỏ: `docs/upgrade.md`

File mới `docs/upgrade.md` dài hơn 800 dòng và thực chất chứa nguyên bản review trước của tôi.

Tôi không khuyên giữ file này trong final repository.

Nó chứa các assessment đã stale như:

```text
Reproducibility = 5.5/10
CI chưa có
SGD chưa explicit
...
```

trong khi code mới đã sửa các vấn đề đó.

Nên chuyển thành:

```text
CHANGELOG.md
```

với nội dung ngắn:

```text
v0.2.0
- add training safeguards
- fix official FPPI
- infer COCO category ID
- add detector adapter
- add perspective-grid
- add compute metrics
- add Phase 0 statistics
- add tests/CI
```

Còn review cũ nên xóa.

---

# 24. Kiến trúc hiện tại tôi đánh giá thế nào?

Bây giờ có thể mô tả repo như sau:

```text
                    HRP4K
                      │
             ┌────────▼────────┐
             │ Dataset Layer   │
             │ manifest/hash   │
             │ integrity/stats │
             └────────┬────────┘
                      │
                dataset view
                      │
             ┌────────▼────────┐
             │ Training       │
             │ currently YOLO │
             └────────┬────────┘
                      │
                 checkpoint
                      │
              ┌───────▼────────┐
              │ Processor      │
              │ Resize         │
              │ Uniform        │
              │ Sliced-NMS     │
              │ Perspective    │
              └───────┬────────┘
                      │
               ProcessedViews
                      │
              ┌───────▼────────┐
              │DetectorAdapter │
              │ Ultralytics    │
              └───────┬────────┘
                      │
                 inverse map
                      │
                  global NMS
                      │
              canonical COCO JSON
                      │
         ┌────────────▼────────────┐
         │ Unified Evaluator       │
         │ COCO / FPPI / Scale     │
         │ IoU curves / errors     │
         └────────────┬────────────┘
                      │
              ┌───────▼────────┐
              │ Diagnostics    │
              │ effective size │
              │ paired analysis│
              │ Pareto         │
              └────────────────┘
```

Đây là architecture **rất hợp lý cho scope của project**.

Tôi không nghĩ cần refactor lớn thêm lúc này.

---

# 25. Việc nên làm tiếp theo

Tôi sẽ ưu tiên theo thứ tự:

1. **Sửa wildcard Phase 3** vì đây là bug có thể làm report sai ngay.
2. **Sửa official dataset completeness** thành train + valid + test đúng protocol.
3. **Sửa `prepare-dataset` default** để thật sự lấy all available.
4. **Sửa latency accounting** thành processor/detector/end-to-end riêng.
5. **Thêm integration CI nhỏ với `pycocotools`**, không nhất thiết kéo toàn Ultralytics stack vào mọi CI.
6. **Dừng polish infrastructure và bắt đầu Phase 1 models**: YOLOv5m → YOLOv8m → YOLOv11m → RT-DETR → D-FINE.
7. Sau khi unified 6-detector benchmark ổn, làm **official SAHI → ZoomDet → TPP** trước các method khó hơn.

---

## Kết luận mới

Bản `v0.2.0` này là **một upgrade thực chất**, không chỉ thay documentation. Các điểm correctness quan trọng của evaluator/training/pipeline đã được sửa, code đã có abstraction tốt hơn, Phase 0 sâu hơn, compute tracking tốt hơn và đã có automated tests + CI.

Ở phiên bản trước, tôi còn khá e ngại việc chạy experiment dài vì một số lỗi protocol có thể khiến kết quả “trông hợp lệ nhưng thực ra sai”. **Ở phiên bản hiện tại, tôi đã khá tin tưởng foundation để bắt đầu chạy các baseline thật**, sau khi sửa 2 lỗi đáng chú ý nhất là `diagnose *.json` và completeness guard.

Quan trọng nhất bây giờ là **không tiếp tục over-engineer framework**. Bottleneck của HRP4K đã chuyển từ “code benchmark chưa đủ chắc” sang **“chưa có các experiment/model reproduction thật”**. Đây là thời điểm hợp lý để chuyển sức sang Phase 1 training và thu prediction chuẩn.
