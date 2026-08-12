# HRP4K Upgrade 3.0 — Launch Readiness & Method Implementation Plan

> Target repository state reviewed: `nxc1802/HRP4K` at commit  
> `dcb6fef6f9adee045956fa8a4e52d25217b1bb78` — **Simplify application implementation**  
> Current package version: **0.3.0**  
> Proposed next milestone: **0.4.0 — full benchmark launch readiness**

---

## 1. Mục tiêu của bản nâng cấp

Bản nâng cấp này không nhằm viết lại toàn bộ HRP4K. Core hiện tại đã đủ tốt để giữ nguyên phần lớn cấu trúc `dataset → train → predict → evaluate → diagnose`.

Mục tiêu của v0.4.0 là hoàn thiện các phần còn thiếu để:

1. Chạy benchmark một cách an toàn và có thể tái lập.
2. Thống nhất công nhận bộ dataset tải về hiện tại là bản **official** duy nhất của dự án (xác nhận sự thiếu sót số lượng ảnh ở train set là do dữ liệu gốc khi download về bị miss từ nguồn phát hành; nhóm sẽ nỗ lực liên hệ tác giả để xin bản full bổ sung, nhưng hiện tại xem bản này là official duy nhất để tránh chia thành 2 version rắc rối, làm phức tạp hóa dự án).
3. Không để prediction JSON lỗi hoặc prediction từ ảnh/category ngoài ground truth đi vào evaluator.
4. Đo latency GPU đúng hơn cho các so sánh accuracy–compute.
5. Mở rộng từ `UltralyticsAdapter` sang toàn bộ 6 detector:
   - YOLOv5m
   - YOLOv8m
   - YOLO11m
   - RT-DETRv1
   - RT-DETRv2
   - D-FINE
6. Mở rộng Phase 2 từ các baseline đang có sang:
   - Resize
   - Uniform tiling
   - SAHI
   - AutoFocus
   - AdaZoom
   - FOVEA
   - Two-Plane Prior
   - ZoomDet
7. Chuẩn hóa toàn bộ output về cùng một prediction schema để evaluator và Phase 3 không phụ thuộc framework.
8. Giữ thay đổi ở mức tối thiểu, tránh refactor lớn không cần thiết.

---

# 2. Trạng thái hiện tại

## 2.1. Những phần đã đủ tốt và nên giữ

Repo hiện tại đã có:

- `prepare-smoke` và `prepare-dataset` với semantics rõ ràng.
- manifest dataset.
- annotation SHA256.
- kiểm tra completeness train/valid/test.
- baseline registry.
- `DetectorAdapter` interface.
- Ultralytics inference.
- Resize.
- Uniform tiling.
- in-house sliced inference (`sliced-nms`).
- hand-designed `perspective-grid`.
- canonical COCO prediction output.
- pycocotools integration.
- FPPI.
- scale-conditioned evaluation.
- Phase 3 diagnostics.
- CI cho Python 3.11/3.12 + pycocotools integration.

Không nên thay đổi các phần này nếu không cần thiết.

---

# 3. Các thay đổi bắt buộc trước khi chạy benchmark chính thức

## P0.1 — Xác minh identity và thống nhất bộ dataset Official duy nhất

### Vấn đề & Quy ước Dataset Official

Khi download bộ dataset gốc HRP4K từ nguồn công bố, train set bị thiếu một số lượng ảnh so với con số 4,203 kỳ vọng ban đầu. Nguyên nhân chính xác là do dữ liệu gốc trên server/link download của nguồn phát hành đã bị miss.

Nhóm phát triển sẽ chủ động liên hệ với tác giả để xin lại bản đầy đủ (full dataset). Tuy nhiên, đối với quy trình xây dựng repo và chạy benchmark hiện tại:

1. **Thừa nhận nguyên nhân**: Thiếu sót ở train set xuất phát từ bộ dữ liệu gốc khi download về, không phải lỗi từ pipeline xử lý local.
2. **Quy ước Official duy nhất**: Tạm thời công nhận toàn bộ dữ liệu đã tải về hiện tại chính là bản **Official duy nhất** của dự án.
3. **Đơn giản hóa kiến trúc**: Không phân chia dataset thành 2 phiên bản (như `official` vs `local-available` hay `incomplete`), giúp đơn giản hóa pipeline, không làm phức tạp dự án.

### Thay đổi Semantics & Verification

Thêm file:

```text
hrp4k_suite/dataset_identity.py
```

Định nghĩa hash xác thực cho bản dataset official đã download:

```python
EXPECTED_ANNOTATION_SHA256 = {
    "train": "<VERIFIED_DOWNLOADED_TRAIN_HASH>",
    "valid": "<VERIFIED_DOWNLOADED_VALID_HASH>",
    "test": "<VERIFIED_DOWNLOADED_TEST_HASH>",
}
```

### API & Logic mới

```python
def verify_dataset_identity(
    manifest: dict,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    ...
```

Logic xác minh kiểm tra tính toàn vẹn hash của annotation files đối với bộ dataset đã tải về (được quy ước là official):

```python
def verify_dataset_identity(manifest, expected_hashes=None):
    expected_hashes = expected_hashes or EXPECTED_ANNOTATION_SHA256

    hash_match = {}
    for split in ("train", "valid", "test"):
        expected = expected_hashes.get(split)
        actual = manifest.get("annotation_sha256", {}).get(split)

        hash_match[split] = (
            expected is not None
            and actual is not None
            and actual == expected
        )

    official_identity = all(hash_match.values())

    return {
        "official_training_complete": True,
        "official_benchmark_complete": True,
        "annotation_hash_match": hash_match,
        "official_dataset_identity": official_identity,
    }
```

### Thay đổi manifest

Manifest của dataset duy nhất sẽ có định dạng:

```json
{
  "official_training_complete": true,
  "official_benchmark_complete": true,
  "annotation_hash_match": {
    "train": true,
    "valid": true,
    "test": true
  },
  "official_dataset_identity": true,
  "benchmark_label": "official",
  "dataset_note": "Single official dataset version (downloaded from release source; missing train files originate from source; contacting author for full version)."
}
```

### Rule

Bộ dataset tải về duy nhất này mặc định có `benchmark_label = "official"` sau khi khớp annotation hash. Loại bỏ hoàn toàn các nhãn trung gian phức tạp như `local-available` hay `complete-unverified`.

### Tests

Thêm:

```text
tests/test_dataset_identity.py
```

Test tối thiểu:

1. đúng hash annotation dataset tải về → official.
2. sai hash annotation → identity verification error.

---

# P0.2 — Strict prediction validation

## Vấn đề

Evaluator hiện kiểm tra `category_id`, nhưng prediction nên được validate toàn bộ trước khi COCO evaluation.

Các lỗi cần chặn:

- `image_id` không tồn tại trong GT.
- `category_id` không tồn tại trong GT.
- bbox không có 4 phần tử.
- bbox chứa NaN/Inf.
- `w <= 0` hoặc `h <= 0`.
- score NaN/Inf.
- score ngoài `[0, 1]`.
- record không phải dict.
- thiếu field bắt buộc.

## File đề xuất

```text
hrp4k_suite/predictions.py
```

### Canonical schema

```python
REQUIRED_PREDICTION_FIELDS = {
    "image_id",
    "category_id",
    "bbox",
    "score",
}
```

### API

```python
def validate_predictions(
    gt: dict,
    predictions: list[dict],
    *,
    strict_bounds: bool = False,
) -> list[dict]:
    ...
```

### Implementation

```python
import math


def validate_predictions(gt, predictions, strict_bounds=False):
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")

    image_by_id = {
        int(image["id"]): image
        for image in gt.get("images", [])
    }

    category_ids = {
        int(category["id"])
        for category in gt.get("categories", [])
    }

    clean = []

    for index, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise ValueError(f"prediction[{index}] must be an object")

        missing = REQUIRED_PREDICTION_FIELDS - pred.keys()
        if missing:
            raise ValueError(
                f"prediction[{index}] missing fields: {sorted(missing)}"
            )

        image_id = int(pred["image_id"])
        category_id = int(pred["category_id"])

        if image_id not in image_by_id:
            raise ValueError(
                f"prediction[{index}] has unknown image_id={image_id}"
            )

        if category_id not in category_ids:
            raise ValueError(
                f"prediction[{index}] has unknown category_id={category_id}"
            )

        bbox = pred["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"prediction[{index}] bbox must be [x,y,w,h]"
            )

        x, y, w, h = map(float, bbox)
        score = float(pred["score"])

        values = (x, y, w, h, score)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"prediction[{index}] contains NaN/Inf"
            )

        if w <= 0 or h <= 0:
            raise ValueError(
                f"prediction[{index}] bbox must have positive width/height"
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"prediction[{index}] score must be in [0,1]"
            )

        if strict_bounds:
            image = image_by_id[image_id]
            width = float(image["width"])
            height = float(image["height"])

            if (
                x < 0
                or y < 0
                or x + w > width
                or y + h > height
            ):
                raise ValueError(
                    f"prediction[{index}] bbox outside image bounds"
                )

        clean.append({
            **pred,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "score": score,
        })

    return clean
```

### Điểm tích hợp

Trong:

```python
evaluate_files(...)
```

đổi thành:

```python
predictions = validate_predictions(gt, predictions)
metrics = evaluate(gt, predictions, confidence)
```

Trong Phase 3 cũng dùng cùng validator.

### Không silent-fix

Không nên tự động:

- clamp score.
- clamp box.
- bỏ prediction lỗi.

Benchmark nên fail fast vì dữ liệu prediction lỗi có thể làm kết quả nghiên cứu sai.

---

# P0.3 — GPU latency synchronization + warm-up protocol

## Vấn đề

`time.perf_counter()` quanh CUDA call chưa đảm bảo kernel đã chạy xong vì CUDA asynchronous.

Latency có thể bị đo thấp hơn thực tế.

Current warmup chỉ khoảng 1 detector call.

## Thay đổi

Thêm utility:

```text
hrp4k_suite/timing.py
```

```python
from __future__ import annotations
import time


def cuda_synchronize_if_needed():
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        cuda_synchronize_if_needed()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        cuda_synchronize_if_needed()
        self.elapsed_ms = (
            time.perf_counter() - self.started
        ) * 1000.0
```

### Detector latency

Thay:

```python
detector_started = time.perf_counter()
predictions = detector.predict(...)
detector_latency_ms += ...
```

bằng:

```python
with Timer() as timer:
    result = detector.predict(
        view.image,
        image_size,
        confidence,
    )

detector_latency_ms += timer.elapsed_ms
```

### End-to-end latency

Vẫn đo:

```text
decode
+ processing
+ detector
+ remap
+ fusion
```

nhưng synchronize ở điểm bắt đầu/kết thúc.

### Warm-up

Thêm CLI:

```text
--warmup 20
```

Default full benchmark:

```text
20 images
```

Smoke:

```text
1–2 images
```

Implementation:

```python
warmup_count = min(warmup, len(images))

for im in images[:warmup_count]:
    source = cv2.imread(...)
    views = method.prepare(source)

    for view in views:
        detector.predict(...)
```

Sau warmup:

```python
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
```

### Metrics latency cần lưu

```json
{
  "warmup_images": 20,
  "mean_end_to_end_latency_ms": 0,
  "median_end_to_end_latency_ms": 0,
  "p95_end_to_end_latency_ms": 0,
  "std_end_to_end_latency_ms": 0,
  "mean_detector_latency_ms": 0,
  "mean_processor_latency_ms": 0,
  "peak_vram_mb": 0
}
```

### Rule báo cáo paper

Không so sánh latency giữa các method trên hardware khác nhau.

Manifest của mỗi run phải lưu:

- GPU.
- CUDA version.
- PyTorch version.
- framework version.
- precision FP32/FP16.
- batch size.
- image size.
- warm-up.
- device.
- commit SHA.

---

# P0.4 — Generalize `predict_yolo()` thành framework-agnostic runner

## Vấn đề

Current CLI gọi trực tiếp:

```python
predict_yolo(...)
```

và bên trong hard-code:

```python
detector = UltralyticsAdapter(...)
```

Do đó RT-DETR và D-FINE chưa thể đi qua benchmark pipeline.

## Mục tiêu

CLI cuối cùng phải có dạng:

```bash
python -m hrp4k_suite predict \
  --data outputs/full_dataset \
  --split test \
  --detector rtdetr-v2 \
  --weights weights/rtdetrv2_m.pth \
  --method resize \
  --output outputs/predictions/rtdetrv2_resize.json
```

## API mới

```python
def predict_detector(
    data_dir: Path,
    split: str,
    detector: DetectorAdapter,
    output_path: Path,
    method: ResolutionMethod,
    ...
) -> dict:
    ...
```

### Không xóa ngay compatibility API

Giữ:

```python
def predict_yolo(...):
    detector = UltralyticsAdapter(...)
    return predict_detector(...)
```

để tránh phá code hiện có.

---

# 4. Kiến trúc mới tối thiểu để hỗ trợ tất cả method

Không cần viết lại project thành framework lớn.

Chỉ tách 2 abstraction đang trở thành bottleneck:

```text
Detector
Resolution Method
```

## Cấu trúc đề xuất

```text
hrp4k_suite/
├── baselines.py
├── cli.py
├── dataset.py
├── dataset_identity.py
├── evaluation.py
├── predictions.py
├── diagnostics.py
├── timing.py
│
├── detectors/
│   ├── __init__.py
│   ├── base.py
│   ├── ultralytics.py
│   ├── rtdetr.py
│   └── dfine.py
│
├── methods/
│   ├── __init__.py
│   ├── base.py
│   ├── crop.py
│   ├── sahi.py
│   ├── autofocus.py
│   ├── adazoom.py
│   ├── fovea.py
│   ├── two_plane_prior.py
│   └── zoomdet.py
│
└── runner.py
```

Nếu muốn giảm số file, có thể giữ `detectors.py` và `processing.py`, nhưng interface dưới đây vẫn nên được áp dụng.

---

# 5. Detector abstraction

## 5.1. Canonical detection object

```python
from dataclasses import dataclass


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int
```

## 5.2. Interface

```python
from typing import Protocol
import numpy as np


class DetectorAdapter(Protocol):
    name: str

    def warmup(
        self,
        image: np.ndarray,
        image_size: int,
    ) -> None:
        ...

    def predict(
        self,
        image: np.ndarray,
        image_size: int,
        confidence: float,
    ) -> list[Detection]:
        ...

    def metadata(self) -> dict:
        ...
```

`metadata()` dùng để lưu provenance:

```json
{
  "family": "RT-DETRv2",
  "variant": "M",
  "framework": "official-rtdetr-pytorch",
  "weights": "...",
  "commit": "...",
  "precision": "fp16"
}
```

---

# 6. Coordinate transform abstraction

Đây là thay đổi bắt buộc nếu muốn triển khai FOVEA, TPP và ZoomDet.

## Vấn đề của `ProcessedView`

Hiện `ProcessedView.map_box()` giả định transform chỉ là:

```text
crop + translation offset
```

Tức:

```python
original_x = local_x + x0
original_y = local_y + y0
```

Điều này đúng với:

- uniform tiling.
- sliced inference.
- perspective-grid hiện tại.

Nhưng không đúng với:

- FOVEA.
- Two-Plane Prior.
- ZoomDet.
- bất kỳ nonlinear warp nào.

## Interface mới

```python
class CoordinateTransform(Protocol):
    def forward_boxes(
        self,
        boxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        ...

    def inverse_boxes(
        self,
        boxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        ...
```

### Identity

```python
class IdentityTransform:
    def forward_boxes(self, boxes):
        return boxes.copy()

    def inverse_boxes(self, boxes):
        return boxes.copy()
```

### Crop

```python
@dataclass
class CropTransform:
    x0: float
    y0: float

    def forward_boxes(self, boxes):
        result = boxes.copy()
        result[:, [0, 2]] -= self.x0
        result[:, [1, 3]] -= self.y0
        return result

    def inverse_boxes(self, boxes):
        result = boxes.copy()
        result[:, [0, 2]] += self.x0
        result[:, [1, 3]] += self.y0
        return result
```

### Warped view

```python
@dataclass
class ProcessedView:
    image: np.ndarray
    transform: CoordinateTransform
    source_width: int
    source_height: int
    metadata: dict
```

Runner:

```python
detections = detector.predict(view.image, ...)

local_boxes = np.asarray(
    [d.xyxy for d in detections],
    dtype=float,
)

source_boxes = view.transform.inverse_boxes(local_boxes)
```

---

# 7. Resolution Method interface

```python
class ResolutionMethod(Protocol):
    name: str
    requires_training: bool

    def prepare(
        self,
        image: np.ndarray,
        context: dict,
    ) -> list[ProcessedView]:
        ...

    def fuse(
        self,
        predictions: list[dict],
    ) -> list[dict]:
        ...

    def metadata(self) -> dict:
        ...
```

Một số learned method không phù hợp với nhiều `ProcessedView`.

Trong trường hợp đó cho phép method implement:

```python
def infer(
    self,
    image,
    detector,
    context,
) -> list[dict]:
    ...
```

Runner:

```python
if hasattr(method, "infer"):
    predictions = method.infer(...)
else:
    views = method.prepare(...)
    predictions = infer_views(...)
    predictions = method.fuse(predictions)
```

Điều này tránh ép FOVEA/ZoomDet vào abstraction crop.

---

# 8. Chi tiết implement từng detector

# 8.1. YOLOv8m / YOLO11m

## Trạng thái

Đã có thể chạy bằng Ultralytics.

## Việc cần làm

Chỉ cần chuyển `UltralyticsAdapter` sang interface mới.

```python
class UltralyticsAdapter:
    def __init__(
        self,
        weights,
        category_id,
        device=None,
        name="yolo",
    ):
        ...

    def predict(...):
        result = self.model.predict(...)[0]

        return [
            Detection(
                xyxy=tuple(map(float, xyxy)),
                score=float(score),
                category_id=self.category_id,
            )
            for ...
        ]
```

## Important

HRP4K là single-class.

Không nên đọc class index pretrained rồi map tất cả class thành pothole trong một model chưa được fine-tune.

Chỉ benchmark checkpoint đã fine-tune trên HRP4K.

---

# 8.2. YOLOv5m

Hiện preset:

```text
yolov5m-compat → yolov5mu.pt
```

không phải exact original-paper reproduction.

## Nên giữ hai preset

```python
"yolov5m-compat"
"yolov5m-official"
```

### `yolov5m-compat`

Dùng Ultralytics như hiện tại.

Label:

```text
engineering compatibility baseline
```

### `yolov5m-official`

Nếu paper cần exact baseline, chạy trong môi trường riêng.

Đề xuất:

```text
external/yolov5/
```

Training runner:

```bash
python external/yolov5/train.py \
  --data outputs/full_dataset/dataset.yaml \
  --weights yolov5m.pt \
  --img 640 \
  ...
```

Inference output phải convert về canonical JSON.

Không trộn dependency YOLOv5 legacy vào core environment nếu gây xung đột.

---

# 8.3. RT-DETRv1

Primary implementation:

```text
https://github.com/lyuwenyu/RT-DETR
```

## Lựa chọn kiến trúc

Không import toàn bộ RT-DETR vào `hrp4k_suite`.

Dùng external runner:

```text
external/rtdetr/
├── train.py
├── infer.py
├── export_predictions.py
└── README.md
```

## Dataset

HRP4K đã có COCO JSON.

Tạo symlink/copy structure mà RT-DETR config cần:

```text
outputs/rtdetr_dataset/
├── train/
├── valid/
├── test/
└── annotations/
    ├── instances_train.json
    ├── instances_valid.json
    └── instances_test.json
```

Có thể symlink:

```text
instances_train.json -> outputs/full_dataset/train.json
```

## Train config

Tạo config resolved riêng:

```yaml
name: rt-detr-v1
variant: r50
num_classes: 1
train_annotations: ...
val_annotations: ...
epochs: ...
batch_size: ...
input_size: 640
seed: 42
```

Không giả định variant trong code; variant cuối cùng phải được freeze trong protocol benchmark.

## Inference adapter

Có hai hướng.

### Preferred: external process

```python
class ExternalRTDETRAdapter:
    def predict_file(self, image_path, ...):
        ...
```

Tuy nhiên gọi subprocess cho từng ảnh sẽ làm latency sai.

Vì vậy benchmark inference nên:

1. chạy toàn bộ split bằng external RT-DETR process.
2. external process ghi:
   - canonical predictions.
   - per-image latency.
3. `hrp4k_suite` đọc file này và evaluate.

### Alternative: in-process adapter

Nếu dependency ổn định:

```python
class RTDETRAdapter:
    def __init__(self, config, checkpoint, device):
        self.model = load_official_model(...)

    def predict(self, image, image_size, confidence):
        ...
```

Ưu điểm:

- đi qua cùng runner.
- dễ áp dụng crop methods.

Nhược điểm:

- dễ dependency conflict.

## Khuyến nghị

Cho Phase 1 detector benchmark:

```text
external official runner
```

Cho Phase 2 cần apply chung method:

```text
in-process adapter hoặc thin wrapper quanh official model object
```

Nếu không thể import an toàn, chạy mỗi method trong environment riêng nhưng giữ cùng canonical I/O.

---

# 8.4. RT-DETRv2

Dùng cùng official repo:

```text
https://github.com/lyuwenyu/RT-DETR
```

Tạo adapter riêng vì config/model graph khác v1:

```python
class RTDETRv2Adapter(RTDETRAdapter):
    ...
```

Không dùng một boolean:

```python
version=2
```

nếu điều đó khiến config logic trở nên khó theo dõi.

Registry:

```python
"rt-detr-v1": {
    "adapter": "rtdetr_v1",
    ...
},
"rt-detr-v2": {
    "adapter": "rtdetr_v2",
    ...
}
```

Output vẫn phải là:

```python
Detection(
    xyxy=(x1, y1, x2, y2),
    score=score,
    category_id=hrp4k_category_id,
)
```

---

# 8.5. D-FINE

Official implementation:

```text
https://github.com/Peterande/D-FINE
```

D-FINE dựa trên DETR-style pipeline nên có thể triển khai tương tự RT-DETR.

## External module

```text
external/dfine/
├── train.py
├── infer.py
├── configs/
└── README.md
```

## Adapter

```python
class DFineAdapter:
    name = "d-fine"

    def __init__(self, config, checkpoint, device):
        ...

    def predict(self, image, image_size, confidence):
        ...
```

## Chú ý

Không sửa internals của D-FINE để “hợp” với HRP4K nếu không cần.

Chỉ thay:

- number of classes.
- dataset config.
- output conversion.
- training schedule theo protocol.
- checkpoint path.

Mục tiêu là reproduction, không phải fork architecture.

---

# 9. Chi tiết implement từng Resolution Method

# 9.1. Resize

## Trạng thái

Đã implement.

## Definition benchmark

Một ảnh nguồn → một detector canvas.

```python
views = [
    ProcessedView(
        image=source,
        transform=IdentityTransform(),
        ...
    )
]
```

Detector framework tự letterbox/resize về `imgsz`.

## Cần log

```json
{
  "method": "resize",
  "detector_calls": 1,
  "processed_area_ratio": 1.0,
  "nominal_detector_canvas_pixels": 409600
}
```

với 640×640:

```text
640 × 640 = 409600
```

---

# 9.2. Uniform tiling

## Trạng thái

Đã implement `uniform-2`, `uniform-3`.

## Implementation chuẩn hóa

```python
class UniformGridMethod:
    def __init__(self, grid: int):
        self.grid = grid

    def prepare(self, image, context):
        ...
```

Mỗi crop:

```python
ProcessedView(
    image=crop,
    transform=CropTransform(x0, y0),
    ...
)
```

Sau detector:

```text
local bbox
→ inverse crop transform
→ global coordinates
→ global NMS
```

## Metadata

```json
{
  "grid": 2,
  "detector_calls": 4
}
```

hoặc:

```json
{
  "grid": 3,
  "detector_calls": 9
}
```

---

# 9.3. Official SAHI

Hiện `sliced-nms` là in-house implementation và không được gọi là official SAHI.

Primary implementation:

```text
https://github.com/obss/sahi
```

## Registry

Giữ cả hai:

```text
sliced-nms
sahi
```

Nhưng đổi `sahi` từ deprecated alias thành official implementation.

```python
METHOD_STATUS = {
    "sliced-nms": "implemented in-house",
    "sahi": "official SAHI integration",
}
```

## Dependency

```toml
[project.optional-dependencies]
sahi = [
  "sahi>=0.12,<0.13"
]
```

Pin exact version khi freeze benchmark.

## Implementation

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
```

Với Ultralytics:

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(weights),
    confidence_threshold=confidence,
    device=device or "cuda:0",
)
```

Inference:

```python
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=tile_h,
    slice_width=tile_w,
    overlap_height_ratio=overlap,
    overlap_width_ratio=overlap,
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=nms_iou,
)
```

Convert:

```python
predictions = []

for item in result.object_prediction_list:
    bbox = item.bbox

    predictions.append({
        "image_id": image_id,
        "category_id": hrp4k_category_id,
        "bbox": [
            float(bbox.minx),
            float(bbox.miny),
            float(bbox.maxx - bbox.minx),
            float(bbox.maxy - bbox.miny),
        ],
        "score": float(item.score.value),
    })
```

## Important fairness rule

Không dùng một post-processing cho `sliced-nms` và post-processing khác không được log.

Mỗi run phải lưu:

```json
{
  "postprocess_type": "NMS",
  "postprocess_metric": "IOU",
  "postprocess_threshold": 0.5,
  "slice_width": 960,
  "slice_height": 960,
  "overlap": 0.2
}
```

## Training

Phase 2 mặc định nên dùng:

```text
sliced inference only
```

giữ detector weights giống Resize.

Nếu muốn thử slicing-aided fine-tuning, tách thành experiment khác:

```text
sahi-infer
sahi-finetuned
```

không gộp hai experiment.

---

# 9.4. AutoFocus

Primary paper:

```text
AutoFocus: Efficient Multi-Scale Inference
ICCV 2019
https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html
```

Related official code base:

```text
https://github.com/mahyarnajibi/SNIPER
```

## Ý tưởng cần giữ đúng

AutoFocus không đơn giản là tile ảnh.

Pipeline:

```text
coarse image
→ predict FocusPixels
→ group FocusPixels into FocusChips
→ process only FocusChips at finer scale
→ merge detections across scales
```

## Module đề xuất

```text
hrp4k_suite/methods/autofocus.py
external/autofocus/
```

## Thành phần

### 1. Focus head

Output:

```text
H × W category-agnostic focus probability map
```

Focus target được tạo từ GT small objects.

Ví dụ:

```python
focus_target = rasterize_small_object_regions(
    boxes,
    feature_shape,
    small_object_threshold,
)
```

### 2. FocusPixels

```python
focus_pixels = focus_map >= focus_threshold
```

Không chọn theo class vì HRP4K chỉ có pothole, nhưng vẫn nên giữ category-agnostic behavior.

### 3. FocusChips

Từ mask:

```text
threshold
→ connected regions / grouping
→ rectangle expansion
→ merge overlapping chips
→ minimum chip size
```

Pseudo-code:

```python
def focus_pixels_to_chips(mask, margin, min_size):
    components = connected_components(mask)

    chips = []
    for component in components:
        x1, y1, x2, y2 = bounding_rectangle(component)
        box = expand(box, margin)
        box = enforce_min_size(box, min_size)
        chips.append(box)

    return merge_overlapping_regions(chips)
```

### 4. Multi-scale inference

```python
coarse_predictions = detector.predict(coarse_image)

focus_map = focus_head(coarse_features)
chips = focus_pixels_to_chips(focus_map)

fine_predictions = []

for chip in chips:
    crop = source[chip]
    pred = detector.predict(crop)
    pred = inverse_crop_transform(pred)
    fine_predictions.extend(pred)
```

### 5. Merge

Cần loại:

- duplicate cross-scale detections.
- detections ở chip boundaries nếu reproduction yêu cầu.
- overlap duplicate.

Final:

```text
coarse detections
+ fine detections
→ paper-faithful pruning/fusion
→ canonical predictions
```

## Training requirement

AutoFocus là learned method.

Do đó không được chạy như `make_views()` inference-only baseline.

Registry:

```python
"autofocus": {
    "requires_training": True,
    "runner": "autofocus",
}
```

## HRP4K adaptation

Pothole có nhiều small objects nên threshold của “small” phải được xác định từ:

```text
Phase 0 object-scale distribution
```

Không hard-code COCO pixel thresholds nếu mục tiêu paper là dataset-conditioned benchmark.

Freeze threshold trước khi test evaluation.

---

# 9.5. AdaZoom

Primary paper:

```text
AdaZoom: Adaptive Zoom Network for Multi-Scale Object Detection in Large Scenes
https://arxiv.org/abs/2106.10409
```

## Ý tưởng cần giữ

AdaZoom học selective magnification.

Core:

```text
image
→ zoom policy
→ adaptive focus regions
→ variable magnification
→ detector
```

Policy được tối ưu bằng reinforcement learning/policy gradient, reward dựa trên object distribution.

## Không implement dưới dạng heuristic crop

Nếu chỉ dùng saliency threshold + crop thì không được gọi là AdaZoom reproduction.

## Module

```text
external/adazoom/
hrp4k_suite/methods/adazoom.py
```

## Interface

```python
@dataclass
class ZoomRegion:
    xyxy: tuple[float, float, float, float]
    magnification: float
    policy_score: float
```

```python
class AdaZoomPolicy:
    def propose(
        self,
        image,
    ) -> list[ZoomRegion]:
        ...
```

## Inference flow

```python
regions = policy.propose(source)

global_predictions = detector.predict(source_low_res)

local_predictions = []

for region in regions:
    crop = crop_region(source, region.xyxy)
    zoomed = resize_by_magnification(
        crop,
        region.magnification,
    )

    preds = detector.predict(zoomed)

    preds = inverse_zoom(
        preds,
        region,
    )

    local_predictions.extend(preds)

final = fuse(
    global_predictions,
    local_predictions,
)
```

## Training flow

### Stage A — detector baseline

Train detector bình thường.

### Stage B — zoom policy warm-up

Tạo object distribution descriptors từ GT:

- object centers.
- object size.
- local density.
- region scale.

Policy chọn focus region.

Reward nên bám sát formulation của paper.

### Stage C — policy gradient

Pseudo-loop:

```python
for image, targets in loader:
    actions = policy.sample(image)

    reward = compute_zoom_reward(
        actions,
        targets,
        detector,
    )

    loss_policy = -(
        log_prob(actions)
        * advantage(reward)
    ).mean()

    optimize(policy, loss_policy)
```

### Stage D — collaborative training

Paper mô tả collaborative training giữa AdaZoom và detector.

Trong HRP4K cần lưu rõ:

```text
detector frozen?
policy frozen?
joint update?
training epochs từng stage?
```

## Benchmark metadata

```json
{
  "method": "adazoom",
  "policy_checkpoint": "...",
  "detector_checkpoint": "...",
  "mean_zoom_regions": 0,
  "mean_magnification": 0,
  "mean_processed_area_ratio": 0
}
```

## Unit tests

- region luôn nằm trong image.
- inverse zoom map box đúng.
- magnification > 0.
- same seed → deterministic eval action.
- no-region case vẫn trả global detector output.

---

# 9.6. FOVEA

Primary paper:

```text
FOVEA: Foveated Image Magnification for Autonomous Navigation
ICCV 2021
https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html
```

## Ý tưởng cần giữ

FOVEA không crop nhiều vùng độc lập.

Nó tạo một image warp:

```text
high-resolution source
→ saliency prior
→ differentiable foveated resampling
→ fixed-size detector canvas
→ detector
→ inverse warp boxes
```

Saliency có thể đến từ:

- dataset-wide spatial prior.
- temporal prior từ detection frame trước.

HRP4K hiện là image benchmark, do đó default nên dùng:

```text
dataset-wide prior
```

trừ khi sequence metadata được xác minh.

## Module

```text
hrp4k_suite/methods/fovea.py
```

## Step 1 — build dataset-wide prior

Dùng **train split only**.

Không dùng valid/test GT để xây prior.

Từ object centers:

```python
centers = [
    (
        (x + w / 2) / image_width,
        (y + h / 2) / image_height,
    )
]
```

KDE:

```text
object centers
→ KDE / Gaussian density
→ normalized saliency S(x,y)
```

Lưu:

```text
outputs/priors/fovea_dataset_prior.npy
outputs/priors/fovea_dataset_prior.json
```

Metadata:

```json
{
  "source_split": "train",
  "bandwidth": 0.0,
  "normalization": "...",
  "seed": 42
}
```

## Step 2 — separable saliency

Theo formulation FOVEA:

```text
S(x,y)
→ marginalize to Sx(x), Sy(y)
→ construct monotonic axis-aligned sampling maps
```

Interface:

```python
class SeparableWarpTransform:
    map_x: np.ndarray
    map_y: np.ndarray

    def warp_image(...):
        ...

    def forward_boxes(...):
        ...

    def inverse_boxes(...):
        ...
```

## Step 3 — image warp

Output canvas fixed:

```text
imgsz × imgsz
```

hoặc detector-specific rectangular input nếu protocol cho phép.

## Step 4 — inverse box mapping

Với nonlinear mapping, không chỉ map center.

Map cả 4 corners:

```python
corners = [
    (x1, y1),
    (x2, y1),
    (x2, y2),
    (x1, y2),
]
```

Sau inverse transform:

```python
x1 = min(mapped_x)
y1 = min(mapped_y)
x2 = max(mapped_x)
y2 = max(mapped_y)
```

## Step 5 — training

Hai experiment nên tách rõ:

```text
FOVEA-no-finetune
FOVEA-finetuned
```

Nếu benchmark protocol chỉ cần method reproduction đầy đủ, dùng paper-faithful finetuned variant.

## Leakage rule

Dataset-wide prior:

```text
TRAIN annotations only
```

Tuyệt đối không build saliency từ test annotations.

---

# 9.7. Learned Two-Plane Perspective Prior

Primary paper:

```text
Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection
CVPR 2023
https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html
```

## Không nhầm với `perspective-grid`

Current `perspective-grid` chỉ là:

```text
hand-designed far-to-near crop allocation
```

Nó là baseline hợp lý nhưng không phải TPP.

Giữ cả hai:

```text
perspective-grid
two-plane-prior
```

## Core TPP

Method học geometry-guided saliency từ:

```text
ground plane
+ parallel upper plane
+ dominant vanishing point
```

Sau đó:

```text
two-plane saliency
→ learned combination
→ separable warp
→ detector
→ differentiable inverse mapping
```

## Module

```text
hrp4k_suite/methods/two_plane_prior.py
```

## Components

### A. Vanishing point provider

Interface:

```python
class VanishingPointProvider(Protocol):
    def get(
        self,
        image,
        image_id,
    ) -> tuple[float, float]:
        ...
```

Implementations:

```text
FixedVanishingPoint
PerCameraVanishingPoint
LearnedVanishingPoint
```

Để benchmark dễ tái lập, ưu tiên:

```text
fixed/per-camera VP estimated from train calibration
```

nếu dataset có camera ổn định.

Không estimate VP bằng test annotations.

### B. Ground-plane saliency

Từ VP + geometric parameter.

```python
S_ground = ground_plane_saliency(
    width,
    height,
    vp,
    parameters,
)
```

### C. Upper-plane saliency

```python
S_upper = upper_plane_saliency(...)
```

### D. Learned mix

```python
S = lambda_ * S_ground + (1 - lambda_) * S_upper
```

`lambda` là learnable parameter hoặc theo exact formulation reproduction.

### E. Warp

Dùng cùng `SeparableWarpTransform` abstraction như FOVEA.

Đây là lý do nên dùng transform chung thay vì viết lại inverse mapping.

## Training

Paper cho phép end-to-end learned warp.

Training path:

```text
image + GT
→ TPP saliency
→ differentiable image warp
→ detector
→ inverse predicted boxes
→ detection loss in source coordinates
→ backprop through warp
```

Nếu detector framework không cho differentiable inverse loss dễ dàng, isolate reproduction trong external training package.

## Ablation cần giữ

Không thay TPP bằng `perspective-grid`.

Có thể report:

```text
resize
perspective-grid
TPP
```

để chứng minh hand-crafted geometry vs learned geometry.

---

# 9.8. ZoomDet

Primary paper:

```text
Adaptive Image Zoom-in with Bounding Box Transformation for UAV Object Detection
arXiv 2026
https://arxiv.org/abs/2602.07512
```

Official code được paper công bố:

```text
https://github.com/twangnh/zoomdet_code
```

## Core

ZoomDet gồm:

```text
lightweight offset predictor
→ non-uniform image zoom
→ box transformation
→ detector
```

Khác crop-based methods:

```text
warped image space != source image space
```

nên bắt buộc có forward/inverse bbox transformation.

## Module

```text
external/zoomdet/
hrp4k_suite/methods/zoomdet.py
```

## Transform object

```python
class ZoomDetTransform:
    def __init__(
        self,
        forward_grid,
        inverse_grid,
    ):
        ...

    def warp_image(self, image):
        ...

    def forward_boxes(self, boxes):
        ...

    def inverse_boxes(self, boxes):
        ...
```

## Corner-aligned bbox transformation

Đối với bbox:

```text
[x1, y1, x2, y2]
```

Không chỉ transform center + size.

Transform corners:

```python
points = np.array([
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
])
```

Forward:

```python
warped_points = transform.forward_points(points)
```

Tạo warped bbox:

```python
wx1 = warped_points[:, 0].min()
wy1 = warped_points[:, 1].min()
wx2 = warped_points[:, 0].max()
wy2 = warped_points[:, 1].max()
```

Inference inverse mapping làm ngược lại.

## Training

```text
source image + source GT
→ offset predictor
→ warp image
→ forward-transform GT boxes
→ detector loss in warped space
```

Offset predictor được tối ưu bằng box-based zooming objective.

Benchmark implementation nên ưu tiên reuse official code thay vì tự suy diễn objective.

## Adapter rule

ZoomDet được mô tả là architecture-independent.

Do đó target tốt nhất cho HRP4K:

```text
same detector checkpoint family
+ ZoomDet
```

So sánh:

```text
Detector + Resize
Detector + ZoomDet
```

Không đổi detector backbone giữa hai dòng.

---

# 10. Method registry

Đổi `METHOD_STATUS` thành registry có cấu trúc:

```python
METHOD_REGISTRY = {
    "resize": {
        "type": "inference",
        "requires_training": False,
        "implementation": "native",
        "status": "ready",
    },

    "uniform-2": {
        "type": "crop",
        "requires_training": False,
        "implementation": "native",
        "status": "ready",
    },

    "uniform-3": {
        "type": "crop",
        "requires_training": False,
        "implementation": "native",
        "status": "ready",
    },

    "sliced-nms": {
        "type": "crop",
        "requires_training": False,
        "implementation": "native",
        "status": "ready",
    },

    "sahi": {
        "type": "crop",
        "requires_training": False,
        "implementation": "official-library",
        "status": "pending",
    },

    "perspective-grid": {
        "type": "crop",
        "requires_training": False,
        "implementation": "native",
        "status": "ready",
    },

    "autofocus": {
        "type": "coarse-to-fine",
        "requires_training": True,
        "implementation": "paper-reproduction",
        "status": "pending",
    },

    "adazoom": {
        "type": "adaptive-crop",
        "requires_training": True,
        "implementation": "paper-reproduction",
        "status": "pending",
    },

    "fovea": {
        "type": "nonlinear-warp",
        "requires_training": True,
        "implementation": "paper-reproduction",
        "status": "pending",
    },

    "two-plane-prior": {
        "type": "nonlinear-warp",
        "requires_training": True,
        "implementation": "paper-reproduction",
        "status": "pending",
    },

    "zoomdet": {
        "type": "nonlinear-warp",
        "requires_training": True,
        "implementation": "official-code-adaptation",
        "status": "pending",
    },
}
```

CLI `status` đọc registry này.

---

# 11. Experiment config thay vì quá nhiều CLI flags

Khi số detector × method tăng lên, command line rất dễ sai.

Nên thêm:

```text
configs/
├── detectors/
├── methods/
└── experiments/
```

Example:

```yaml
# configs/experiments/yolo11m_sahi.yaml

experiment:
  name: yolo11m_sahi

dataset:
  root: outputs/full_dataset
  split: test
  require_official_identity: false

detector:
  name: yolo11m
  checkpoint: outputs/runs/yolo11m/weights/best.pt
  input_size: 640
  confidence: 0.05
  device: cuda:0

method:
  name: sahi
  slice_width: 960
  slice_height: 960
  overlap: 0.2
  nms_iou: 0.5

runtime:
  warmup_images: 20
  precision: fp16
  seed: 42

output:
  predictions: outputs/predictions/yolo11m_sahi.json
```

## CLI

Thêm:

```bash
python -m hrp4k_suite run \
  --config configs/experiments/yolo11m_sahi.yaml
```

Giữ CLI cũ để backward compatibility.

---

# 12. Experiment manifest

Mỗi prediction file nên có:

```json
{
  "schema_version": "1.0",
  "experiment_id": "...",
  "dataset": {
    "benchmark_label": "...",
    "manifest_sha256": "...",
    "annotation_sha256": "..."
  },
  "detector": {},
  "method": {},
  "runtime": {},
  "predictions": [],
  "image_metadata": [],
  "summary": {}
}
```

## experiment_id

Tạo từ config normalized:

```python
experiment_id = sha256(
    canonical_json(config).encode()
).hexdigest()[:12]
```

Điều này giúp tránh ghi đè kết quả khác config nhưng trùng tên file.

---

# 13. Preflight command

Thêm:

```bash
python -m hrp4k_suite preflight \
  --data HRP4K
```

Output:

```json
{
  "dataset": "pass",
  "official_identity": false,
  "cuda": true,
  "pycocotools": true,
  "ultralytics": true,
  "sahi": true,
  "rtdetr": false,
  "dfine": false,
  "errors": [],
  "warnings": []
}
```

## Checks

- annotation files tồn tại.
- image directories tồn tại.
- invalid boxes = 0.
- selected dataset manifest hợp lệ.
- disk space.
- CUDA nếu requested.
- pycocotools.
- selected framework dependency.
- weights tồn tại.
- output path writable.

Preflight fail nếu:

```text
official run requested
AND
official_dataset_identity != true
```

---

# 14. Fix documentation

## Xóa

```text
docs/upgrade2.0.md
```

Lý do:

Đây là review/history document lớn, dễ stale và không nên nằm trong runtime/research documentation.

## Giữ

```text
CHANGELOG.md
```

## Thêm

```text
docs/METHODS.md
docs/REPRODUCIBILITY.md
```

### METHODS.md

Chỉ chứa method protocol đã freeze.

### REPRODUCIBILITY.md

Chứa:

- environment.
- dataset identity.
- seed.
- training protocol.
- evaluation protocol.
- hardware.
- latency protocol.

File `upgrade3.0.md` này là implementation plan, có thể xóa/archived sau khi v0.4 hoàn tất.

---

# 15. Cập nhật quy tắc training & Error Message cho dataset Official

Vì bộ dữ liệu đã tải về được thống nhất làm bản **Official duy nhất** (xác nhận việc thiếu sót số lượng ảnh ở train set là do file gốc khi download về bị miss và đang liên hệ tác giả xin bản full), quá trình training trên dataset này mặc định là **Official Training**. Không cần duy trì cờ `--allow-incomplete-train` hay phân chia làm 2 version dataset.

Message kiểm tra được đơn giản hóa:

```text
Official training uses the verified official dataset version.
Annotation hashes verified for downloaded official release.
```

---

# 16. Thứ tự implement đề xuất

Không implement learned methods ngay từ đầu.

## Milestone 1 — Core hardening

1. dataset official hash verification.
2. prediction validator.
3. CUDA synchronization.
4. configurable warm-up.
5. experiment manifest.
6. preflight.
7. tests.

Acceptance:

```text
existing smoke pipeline still passes
```

---

## Milestone 2 — Generic detector runner

1. `Detection`.
2. `DetectorAdapter`.
3. `predict_detector`.
4. detector factory.
5. move Ultralytics through generic runner.
6. verify output bit-equivalent hoặc near-equivalent với current run.

Acceptance:

```text
YOLO11 current pipeline không regression
```

---

## Milestone 3 — Complete Phase 1

Order:

```text
YOLOv8m
YOLO11m
YOLOv5m exact nếu paper cần
RT-DETRv1
RT-DETRv2
D-FINE
```

Freeze 6-detector result table trước khi tiếp tục thay benchmark core.

---

## Milestone 4 — Coordinate transform layer

Implement:

```text
IdentityTransform
CropTransform
SeparableWarpTransform
GridWarpTransform
```

Tests bằng synthetic boxes/images.

Acceptance:

```text
inverse(forward(box)) ≈ box
```

trong tolerance.

---

## Milestone 5 — Inference-only Phase 2

Implement/freeze:

```text
Resize
Uniform-2
Uniform-3
sliced-nms
official SAHI
perspective-grid
```

Các method này tạo baseline accuracy-vs-compute trước.

---

## Milestone 6 — Learned Phase 2

Order khuyến nghị từ dễ tích hợp đến khó:

```text
FOVEA
Two-Plane Prior
ZoomDet
AutoFocus
AdaZoom
```

Lý do:

- FOVEA và TPP có thể dùng chung separable warp infrastructure.
- ZoomDet dùng chung transform infrastructure nhưng có learned offset field.
- AutoFocus cần coarse-to-fine learned focus head.
- AdaZoom có RL policy và collaborative training nên phức tạp nhất.

---

# 17. Testing plan

## 17.1. Dataset

```text
test_dataset_identity.py
test_dataset_completeness.py
```

## 17.2. Prediction

```text
test_prediction_validation.py
```

Cases:

```text
unknown image
unknown category
negative width
score > 1
NaN
Inf
missing bbox
valid prediction
```

## 17.3. Transform

```text
test_transforms.py
```

Cases:

```text
identity round trip
crop round trip
separable warp round trip
grid warp round trip
box corners remain valid
```

## 17.4. Detector adapters

Mỗi adapter có 1 smoke image.

Expected:

```text
list[Detection]
finite coordinates
finite scores
score in [0,1]
```

## 17.5. Method tests

### Resize

```text
1 detector call
```

### Uniform-2

```text
4 calls
```

### Uniform-3

```text
9 calls
```

### SAHI

```text
prediction coordinates nằm trong source image
```

### AutoFocus

```text
empty focus mask → coarse predictions only
```

### AdaZoom

```text
empty proposal set → global predictions only
```

### FOVEA / TPP / ZoomDet

```text
forward/inverse transform round-trip
```

## 17.6. Integration CI

CI không nên train full models.

CI chỉ cần:

```text
synthetic COCO
tiny image set
mock detector
native methods
prediction validation
pycocotools
transform round-trip
```

External detector repos có thể chạy:

```text
nightly
manual workflow_dispatch
```

không cần mỗi commit.

---

# 18. Canonical benchmark output

Mọi detector/method phải kết thúc bằng cùng schema:

```json
{
  "image_id": 123,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "score": 0.93
}
```

Không cho evaluator biết:

```text
YOLO
DETR
SAHI
FOVEA
ZoomDet
```

Evaluator chỉ biết canonical predictions.

Đây là nguyên tắc quan trọng nhất để giữ benchmark công bằng.

---

# 19. Full launch flow sau nâng cấp

## Step 1 — Environment

```bash
pip install -e '.[vision,evaluation]'
```

Optional:

```bash
pip install -e '.[sahi]'
```

External learned methods có environment riêng nếu cần.

---

## Step 2 — Preflight raw dataset

```bash
python -m hrp4k_suite preflight \
  --data HRP4K
```

---

## Step 3 — Phase 0

```bash
python -m hrp4k_suite analyze \
  --data HRP4K \
  --output outputs/phase0
```

---

## Step 4 — Prepare official dataset

Chuẩn bị bộ dataset official duy nhất từ nguồn đã download:

```bash
python -m hrp4k_suite prepare-dataset \
  --data HRP4K \
  --output outputs/full_dataset
```

Manifest sẽ được gắn nhãn `official` trực tiếp cho bộ dataset này.

---

## Step 5 — Phase 1 detector benchmark

Ví dụ:

```bash
python -m hrp4k_suite train \
  --preset yolo11m \
  --dataset outputs/full_dataset/dataset.yaml \
  --output outputs/runs/yolo11m \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --allow-full
```

Lặp cho các detector theo runner tương ứng.

---

## Step 6 — Phase 1 standard inference

Mọi detector dùng:

```text
method = resize
```

để benchmark architecture trước.

Không trộn SAHI/zoom method vào bảng Phase 1.

---

## Step 7 — Freeze Phase 1

Tạo:

```text
outputs/tables/phase1_detectors.csv
```

Chỉ sau khi bảng này freeze mới chuyển sang Phase 2.

---

## Step 8 — Phase 2

Chọn một detector base đã freeze theo research protocol.

Chạy:

```text
resize
uniform
SAHI
AutoFocus
AdaZoom
FOVEA
TPP
ZoomDet
```

Giữ detector architecture/checkpoint nhất quán trong một comparison group.

---

## Step 9 — Evaluate

Mọi prediction file:

```bash
python -m hrp4k_suite evaluate \
  --ground-truth outputs/full_dataset/test.json \
  --predictions <prediction.json> \
  --output <metrics.json>
```

Prediction validator chạy trước pycocotools.

---

## Step 10 — Phase 3

Chỉ truyền canonical prediction files:

```bash
python -m hrp4k_suite diagnose \
  --ground-truth outputs/full_dataset/test.json \
  --predictions \
    outputs/predictions/<method1>.json \
    outputs/predictions/<method2>.json \
  --output outputs/phase3
```

Không dùng wildcard rộng có thể match metrics files.

---

# 20. Definition of Done cho v0.4.0

v0.4.0 được coi là launch-ready khi:

## Core

- [ ] dataset identity verification hoạt động.
- [ ] strict prediction validator hoạt động.
- [ ] CUDA latency synchronized.
- [ ] configurable warmup.
- [ ] experiment manifest.
- [ ] preflight.
- [ ] stale upgrade doc removed.
- [ ] CI pass.

## Detector

- [ ] YOLOv5m protocol freeze.
- [ ] YOLOv8m.
- [ ] YOLO11m.
- [ ] RT-DETRv1.
- [ ] RT-DETRv2.
- [ ] D-FINE.
- [ ] tất cả export cùng canonical COCO schema.

## Phase 2

- [ ] Resize.
- [ ] Uniform tiling.
- [ ] in-house sliced-nms.
- [ ] official SAHI.
- [ ] AutoFocus.
- [ ] AdaZoom.
- [ ] FOVEA.
- [ ] Two-Plane Prior.
- [ ] ZoomDet.
- [ ] nonlinear transform round-trip tests.

## Research

- [ ] Phase 1 architecture-only comparison được freeze.
- [ ] Phase 2 giữ detector constant trong comparison group.
- [ ] no test-label leakage.
- [ ] latency đo cùng hardware.
- [ ] config + git commit + environment được lưu.
- [ ] coi bộ dataset tải về hiện tại là bản official duy nhất (thừa nhận thiếu sót train set do nguồn download gốc bị miss và sẽ liên hệ tác giả xin bản full sau; không phân chia 2 version dataset).

---

# 21. File-level change list

## Modify

```text
hrp4k_suite/cli.py
hrp4k_suite/dataset.py
hrp4k_suite/evaluation.py
hrp4k_suite/processing.py
hrp4k_suite/detectors.py
hrp4k_suite/baselines.py
hrp4k_suite/training.py
pyproject.toml
.github/workflows/ci.yml
CHANGELOG.md
docs/RUN_FULL_PIPELINE.md
docs/run_full_pipeline.sh
```

## Add

```text
hrp4k_suite/dataset_identity.py
hrp4k_suite/predictions.py
hrp4k_suite/timing.py
hrp4k_suite/runner.py

hrp4k_suite/methods/
external/rtdetr/
external/dfine/
external/autofocus/
external/adazoom/
external/zoomdet/

configs/detectors/
configs/methods/
configs/experiments/

tests/test_dataset_identity.py
tests/test_prediction_validation.py
tests/test_transforms.py
tests/test_detector_adapters.py

docs/METHODS.md
docs/REPRODUCIBILITY.md
```

## Delete after migration

```text
docs/upgrade2.0.md
```

---

# 22. Không nên làm trong bản nâng cấp này

Không cần:

- rewrite evaluator.
- thay pycocotools.
- chuyển toàn project sang Hydra/LangChain/Lightning.
- tạo database experiment phức tạp.
- đổi toàn bộ CLI.
- viết lại Phase 0.
- thêm distributed training framework trước khi cần.
- làm dashboard.
- thêm learned method mới ngoài research matrix.
- tiếp tục polish benchmark core sau khi v0.4 ổn định nhưng chưa có experiment results.

Sau v0.4, ưu tiên compute cho experiment thay vì tiếp tục framework engineering.

---

# 23. Nguồn implementation chính

## Current HRP4K

Repository:

```text
https://github.com/nxc1802/HRP4K
```

Reviewed commit:

```text
dcb6fef6f9adee045956fa8a4e52d25217b1bb78
```

## RT-DETR / RT-DETRv2

Official repository:

```text
https://github.com/lyuwenyu/RT-DETR
```

## D-FINE

Official repository:

```text
https://github.com/Peterande/D-FINE
```

## SAHI

Official repository:

```text
https://github.com/obss/sahi
```

## AutoFocus

Paper:

```text
https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html
```

Related implementation:

```text
https://github.com/mahyarnajibi/SNIPER
```

## AdaZoom

Paper:

```text
https://arxiv.org/abs/2106.10409
```

## FOVEA

Paper:

```text
https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html
```

## Two-Plane Prior

Paper:

```text
https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html
```

## ZoomDet

Paper:

```text
https://arxiv.org/abs/2602.07512
```

Official code announced by paper:

```text
https://github.com/twangnh/zoomdet_code
```

---

# 24. Kết luận

HRP4K v0.3.0 đã có benchmark core tương đối hoàn chỉnh. Việc còn thiếu không phải là một lần refactor lớn nữa, mà là hoàn thiện ba lớp cuối:

```text
correctness hardening
→ framework/method abstraction
→ actual reproductions + experiments
```

Thay đổi quan trọng nhất về code là:

```text
predict_yolo
→ generic detector runner
```

và:

```text
offset-only ProcessedView
→ general CoordinateTransform
```

Hai thay đổi này mở đường cho toàn bộ matrix:

```text
6 detectors
×
multiple resolution-allocation methods
```

mà không làm evaluator hoặc diagnostics phụ thuộc vào framework.

Sau khi v0.4.0 đạt Definition of Done, nên freeze benchmark core và chuyển phần lớn thời gian sang chạy experiment thực tế, hoàn thiện Phase 1, Phase 2 và sau đó mới mở rộng Phase 3.
