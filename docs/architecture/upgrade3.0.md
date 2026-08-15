# Những lỗi cần sửa trong HRP4K

## 1. P0 — CI của `main` đang fail sau Upgrade 3.0

Đây là lỗi rõ ràng nhất hiện tại.

Workflow mới nhất chạy 4 job nhưng `unit (3.11)`, `evaluation-integration`, `vision-runner-integration` đều fail. Nguyên nhân không nằm ở model mà ở migration cấu trúc project.

### 1.1. `PyYAML` đang bị đặt sai dependency

Config system hiện sử dụng YAML cho:

```text
configs/base.yaml
configs/detectors/*
configs/methods/*
configs/profiles/*
configs/experiments/*
```

`resolve()` gọi `load_yaml()`, nhưng `PyYAML` chỉ nằm trong extra `vision`, trong khi CI core chỉ cài:

```bash
pip install -e .
```

`pyproject.toml` hiện có core dependency chỉ là `numpy`; `PyYAML` nằm trong `vision`.

Kết quả là toàn bộ config contract chết ở:

```text
ModuleNotFoundError: No module named 'yaml'
```

### Cách sửa

`PyYAML` nên là core dependency vì config system không phải vision functionality.

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "PyYAML>=6",
]

[project.optional-dependencies]
evaluation = ["pycocotools>=2.0.7"]

vision = [
    "opencv-python>=4.8",
    "matplotlib>=3.7",
    "ultralytics>=8.3",
    "pycocotools>=2.0.7",
]
```

Sau đó:

```bash
pip install -e .
python -m unittest discover -v
```

phải chạy được config tests mà không cần install cả Ultralytics/OpenCV.

---

## 2. P0 — CI vẫn gọi đường dẫn test cũ

Upgrade 3.0 chuyển test từ:

```text
tests/test_*.py
```

sang:

```text
tests/
├── contracts/
├── integration/
├── scientific/
└── unit/
```

Nhưng `.github/workflows/ci.yml` vẫn gọi:

```bash
python -m unittest tests.test_coco_integration -v
python -m unittest tests.test_runner tests.test_transforms -v
```

Trong khi file thật hiện nằm ở:

```text
tests/integration/test_coco_integration.py
tests/contracts/test_runner.py
tests/unit/test_transforms.py
```

### Sửa CI thành

```yaml
evaluation-integration:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - run: python -m pip install --upgrade pip
    - run: python -m pip install -e '.[evaluation]'

    - run: >
        python -m unittest
        tests.integration.test_coco_integration
        -v
      env:
        HRP4K_REQUIRE_PYCOCOTOOLS: "1"

vision-runner-integration:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - run: python -m pip install --upgrade pip
    - run: python -m pip install -e . opencv-python-headless

    - run: >
        python -m unittest
        tests.contracts.test_runner
        tests.unit.test_transforms
        -v
```

Hai lỗi CI này nên sửa đầu tiên vì chúng làm `main` hiện không phản ánh đúng trạng thái code.

---

# 3. P0 — Config system đang có hai source of truth

Đây là lỗi kiến trúc nghiêm trọng hơn CI.

Upgrade 3.0 đã xây:

```text
YAML
 ↓
resolve()
 ↓
HRP4KConfig
 ↓
validate()
```

Resolver hỗ trợ merge:

```text
base
→ detector
→ method
→ profile
→ experiment
→ override
```

Nhưng:

```bash
hrp4k run --config ...
```

**không dùng hệ thống này**.

Nó lại làm:

```python
config = yaml.safe_load(...)
detector = config["detector"]
method = config["method"]
runtime = config.get("runtime", {})
```

Tức:

```text
config show / validate
```

và:

```text
run
```

đang có hai parser khác nhau.

---

## 3.1. Có lỗi thực tế với `method.parameters`

Typed schema định nghĩa:

```python
class MethodConfig:
    name: str = "resize"
    parameters: dict = ...
```

Ví dụ sliced NMS:

```yaml
method:
  name: sliced-nms
  parameters:
    tile_size: 960
    overlap: 0.2
```

Nhưng raw `run` lại đọc:

```python
method.get("tile_size", 960)
method.get("overlap", 0.2)
```

chứ không đọc:

```python
method["parameters"]["tile_size"]
```

Vì vậy config:

```yaml
parameters:
  tile_size: 1280
```

có thể validate đúng, nhưng lúc `run` lại silently chạy `960`.

Đây là kiểu bug rất nguy hiểm trong experiment framework: **config file ghi một thứ nhưng experiment chạy một thứ khác**.

---

## Cách sửa

`run` chỉ được làm:

```python
resolved = resolve(config_path=args.config)

errors = validate(resolved)
if errors:
    raise ValueError("\n".join(errors))

run_resolved_experiment(resolved)
```

Ví dụ:

```python
def run_resolved_experiment(config: HRP4KConfig):
    method = config.method
    detector = config.detector
    runtime = config.runtime

    return run_phase_2(
        data_dir=Path(config.dataset.root),
        split=config.dataset.split,
        weights=Path(detector.checkpoint),
        output_path=Path(config.output.predictions),
        method=method.name,
        limit=runtime.limit,
        image_size=detector.input_size,
        confidence=detector.confidence,
        tile_size=method.tile_size,
        overlap=method.overlap,
        device=runtime.device,
        warmup=runtime.warmup,
        detector_name=detector.name,
        precision=runtime.precision,
    )
```

Sau đó **xóa toàn bộ `yaml.safe_load()` riêng trong command `run`**.

---

# 4. P0 — Schema hiện silently bỏ field sai

Resolver hiện làm:

```python
valid = {f.name for f in fields(cls)}

return cls(**{
    k: v
    for k, v in raw.items()
    if k in valid
})
```

Tức field không tồn tại bị **bỏ qua**, không báo lỗi.

Điều này không phù hợp với benchmark.

Ví dụ typo:

```yaml
runtime:
  precison: fp16
```

sẽ không fail.

Nó chỉ biến mất và:

```text
precision = fp32
```

được dùng.

### Nên fail-fast

```python
def _build_section(cls, raw):
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid

    if unknown:
        raise ValueError(
            f"Unknown fields in {cls.__name__}: "
            f"{sorted(unknown)}"
        )

    return cls(**raw)
```

Research config nên ưu tiên:

> sai config → fail

thay vì:

> sai config → dùng default.

---

# 5. P0 — Experiment YAML hiện không khớp schema

Ví dụ experiment hiện có:

```yaml
experiment:
  name: yolo11m_resize_smoke

detector:
  device: cpu

runtime:
  warmup_images: 1
```

Trong schema lại là:

```python
ExperimentConfig:
    experiment_name

RuntimeConfig:
    device
    warmup
```

Tức đang có ba mismatch:

```text
experiment.name
        ↓
experiment.experiment_name

detector.device
        ↓
runtime.device

runtime.warmup_images
        ↓
runtime.warmup
```

Do resolver silently ignore unknown field nên bug này hiện bị che mất.

### Config chuẩn nên là

```yaml
experiment:
  experiment_name: yolo11m_resize_smoke
  profile: smoke

dataset:
  root: outputs/full_dataset
  split: test

detector:
  name: yolo11m
  checkpoint: outputs/runs/yolo11m_full/weights/best.pt
  input_size: 320
  confidence: 0.01

method:
  name: resize
  parameters: {}

runtime:
  device: cpu
  warmup: 1
  precision: fp32
  limit: 2

output:
  predictions: outputs/config_smoke/yolo11m_resize.json
```

---

# 6. P0 — Detector ID không đồng nhất

Config hiện có:

```yaml
# configs/detectors/rtdetr.yaml
detector:
  name: rtdetr
```

Nhưng core nhận:

```text
rt-detr-v1
rt-detr-v2
```

Tương tự:

```yaml
# dfine.yaml
name: dfine
```

nhưng core dùng:

```text
d-fine
```

Điều này có thể khiến config-driven runner đi từ:

```text
rtdetr
```

tới:

```python
create_detector("rtdetr")
```

rồi gặp:

```text
Unknown detector: rtdetr
```

thay vì route sang external runtime.

---

## Nên chọn một canonical naming

Mình đề xuất:

```text
yolov5m-official
yolov8m
yolo11m

rt-detr-v1
rt-detr-v2

d-fine
```

Và config files:

```text
configs/detectors/
├── yolov5m_official.yaml
├── yolov8m.yaml
├── yolo11m.yaml
├── rt_detr_v1.yaml
├── rt_detr_v2.yaml
└── d_fine.yaml
```

Nội dung phải dùng canonical ID.

Không nên giữ một `rtdetr.yaml` chung vì v1 và v2 là hai experiment khác nhau.

---

# 7. P1 — `seed` có trong config nhưng training không dùng nó

Bạn chưa cần multi-seed, điều đó hoàn toàn khác với việc **seed phải được plumbing đúng**.

Config có:

```python
TrainingConfig:
    seed: int = 42
```

Nhưng `train_yolo()` hard-code:

```python
"seed": 42
```

và:

```python
model.train(
    ...
    seed=42,
)
```

Nghĩa là nếu sau này config ghi:

```yaml
training:
  seed: 123
```

thì nó **không có tác dụng**.

### Sửa

```python
def train_yolo(
    ...,
    seed: int = 42,
):
    config = {
        ...
        "seed": seed,
    }

    model.train(
        ...
        seed=seed,
        deterministic=True,
    )
```

và truyền từ:

```text
TrainingConfig.seed
↓
run_phase_1
↓
train_yolo
```

Hiện vẫn dùng một seed duy nhất; chỉ là seed đó trở thành reproducible configuration thật sự.

---

# 8. P1 — `preflight` external detector đang detect dependency không đáng tin

Hiện có:

```python
"rtdetr": _available("src") or _available("rtdetr")
```

`src` quá generic.

Một environment có package/module tên `src` không có nghĩa là official RT-DETR runtime đã sẵn sàng.

Tương tự việc kiểm tra:

```python
_available("dfine")
```

không đủ để xác nhận:

* upstream commit;
* checkpoint;
* config;
* Python env;
* dataset mapping;
* export adapter.

### External readiness nên manifest-driven

Ví dụ:

```json
{
  "name": "rt-detr-v2",
  "status": "ready",
  "upstream_repo": "lyuwenyu/RT-DETR",
  "upstream_commit": "...",
  "python": ".../envs/rtdetr/bin/python",
  "config": "...",
  "checkpoint": "...",
  "adapter": "external/rtdetr/run.py"
}
```

Preflight:

```python
def check_external_runtime(manifest):
    errors = []

    for key in ("python", "config", "checkpoint", "adapter"):
        path = Path(manifest[key])
        if not path.exists():
            errors.append(f"missing {key}: {path}")

    return not errors, errors
```

Như vậy:

```text
"external_available": true
```

mới có ý nghĩa thực tế.

---

# 9. Dataset: giữ official, nhưng nên sửa semantics metadata

Theo quyết định của bạn, release hiện tại là:

> **official available HRP4K release**, dù upstream không publish đủ train files.

Vậy mình không khuyến nghị block full training nữa.

Nhưng hiện naming hơi gây hiểu nhầm vì `verify_dataset_identity()` trả:

```python
official_training_complete = identity
official_benchmark_complete = identity
```

chỉ dựa trên annotation hashes.

Nên đổi semantics cho rõ:

```json
{
  "official_release_identity": true,
  "official_release_scope": "available-upstream-release",

  "release_file_completeness": "partial",
  "known_missing_train_images": 1917,

  "benchmark_label": "official",
  "upstream_limitation": true
}
```

Điểm quan trọng là:

**partial ≠ unofficial**.

Nhưng artifact/paper phải có đủ provenance để reviewer biết benchmark dùng chính xác release nào.

---

# 10. P1 — External system hiện mới là “contract”, chưa thực sự là adapter framework

Core có interface rất đẹp:

```python
class DetectorAdapter(Protocol):
    def warmup(...)
    def predict(...)
    def metadata(...)
```

Nhưng external detectors lại chỉ:

```python
raise RuntimeError(
    "... requires its official isolated runtime ..."
)
```

Đây là hợp lý ở giai đoạn đầu, nhưng chưa đủ cho benchmark automation.

## Tuy nhiên: không nên import external detector vào core

Ví dụ TPP official code dùng Python 3.8.5, PyTorch 1.6.0, mmdetection 2.20.0 và mmcv 1.3.17. ([GitHub][1])

FOVEA official code cũng dùng Python 3.8.5/PyTorch 1.6 nhưng mmdetection 2.7.0. ([GitHub][2])

Core HRP4K yêu cầu Python ≥3.10.

Vì vậy giải pháp đúng là:

```text
HRP4K core
    │
    │ experiment contract
    ▼
External isolated environment
    │
    │ run whole split
    ▼
canonical prediction artifact
    │
    ▼
HRP4K evaluate / diagnose
```

**Không spawn subprocess cho từng ảnh.**

Project RT-DETR README hiện cũng đã ghi rõ yêu cầu này.

---

# 11. External adapter framework mình đề xuất

Tạo:

```text
external/
├── common/
│   ├── canonical.py
│   └── runtime.py
│
├── yolov5/
│   ├── manifest.yaml
│   ├── run.py
│   └── README.md
│
├── rtdetr/
│   ├── manifest_v1.yaml
│   ├── manifest_v2.yaml
│   ├── run.py
│   └── README.md
│
├── dfine/
│   ├── manifest.yaml
│   └── run.py
│
├── fovea/
├── two_plane_prior/
├── zoomdet/
├── autofocus/
└── adazoom/
```

## Common canonical writer

Code này có thể chạy cả trên Python 3.8 external env:

```python
# external/common/canonical.py

import json
import time


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    ]


def make_detection(
    image_id,
    category_id,
    xyxy,
    score,
):
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": xyxy_to_xywh(xyxy),
        "score": float(score),
    }


def save_prediction_document(
    output,
    method,
    detector,
    predictions,
    image_metadata,
    provenance,
):
    payload = {
        "schema_version": "1.0",
        "method": method,
        "detector": detector,
        "predictions": predictions,
        "image_metadata": image_metadata,
        "external_provenance": provenance,
    }

    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
```

Core HRP4K sẽ validate phần `predictions` bằng validator hiện tại:

```text
image_id
category_id
bbox = [x,y,w,h]
score
```

và reject NaN, unknown image/category, width/height <= 0...

---

# 12. YOLOv5 official adapter

Đây nên là external detector đầu tiên hoàn thiện vì dễ nhất.

Current project đã xác định rất đúng:

```text
yolov5m-compat
≠
yolov5m-official
```

Official YOLOv5 repo hỗ trợ dataset YAML và lệnh training dạng:

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 150 \
  --data HRP4K/dataset.yaml \
  --weights yolov5m.pt
```

([GitHub][3])

### Nên freeze upstream

```yaml
name: yolov5m-official

upstream:
  repo: ultralytics/yolov5
  commit: <FROZEN_COMMIT>

runtime:
  python: /envs/yolov5/bin/python

model:
  weights: yolov5m.pt
  image_size: 640

dataset:
  yaml: outputs/official_dataset/dataset.yaml
```

### Adapter nhiệm vụ duy nhất

```text
official YOLOv5 prediction
            ↓
original-image XYXY
            ↓
canonical XYWH
            ↓
HRP4K JSON
```

Không cần sửa architecture YOLOv5.

---

# 13. RT-DETR v1/v2 adapter

Đây là external detector thứ hai mình sẽ làm.

Official `lyuwenyu/RT-DETR` chứa cả RT-DETR và RT-DETRv2; upstream cũng hướng dẫn custom data bằng cách tắt `remap_mscoco_category` và chỉnh `img_folder`/`ann_file`. ([GitHub][4])

HRP4K đã là COCO, nên integration khá thuận lợi.

### Tạo hai config

```text
external/rtdetr/configs/
├── hrp4k_v1.yml
└── hrp4k_v2.yml
```

Ví dụ concept:

```yaml
num_classes: 1
remap_mscoco_category: False

train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: /data/hrp4k/train/images
    ann_file: /data/hrp4k/train.json

val_dataloader:
  dataset:
    type: CocoDetection
    img_folder: /data/hrp4k/valid/images
    ann_file: /data/hrp4k/valid.json
```

RT-DETR official PyTorch runner dùng:

```bash
python tools/train.py \
  -c configs/rtdetr/...yml
```

và evaluation với:

```bash
python tools/train.py \
  -c configs/rtdetr/...yml \
  -r checkpoint.pth \
  --test-only
```

([GitHub][5])

### HRP4K adapter

Không nên subprocess per image.

Một `run.py` load model một lần:

```python
model = load_rtdetr(config, checkpoint)

for image in dataset:
    start = synchronize_and_time()

    outputs = model(image)

    elapsed = synchronize_and_stop(start)

    for box, score in decode(outputs):
        predictions.append(
            make_detection(
                image_id=image_id,
                category_id=category_id,
                xyxy=box,
                score=score,
            )
        )
```

Sau đó ghi:

```json
{
  "method": "resize",
  "detector": {
    "name": "rt-detr-v2"
  },
  "predictions": [],
  "image_metadata": []
}
```

---

# 14. D-FINE adapter

D-FINE official repo đã hỗ trợ **custom COCO dataset** trực tiếp và yêu cầu:

```yaml
remap_mscoco_category: False
```

cùng custom dataloader/config. ([GitHub][6])

Project HRP4K README hiện cũng đã xác định:

> giữ architecture internals, chỉ adapt dataset paths, schedule, checkpoint và canonical output.

Đây là đúng.

### HRP4K config

```yaml
task: detection

num_classes: 1
remap_mscoco_category: False

train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: /data/hrp4k/train/images
    ann_file: /data/hrp4k/train.json

val_dataloader:
  dataset:
    type: CocoDetection
    img_folder: /data/hrp4k/valid/images
    ann_file: /data/hrp4k/valid.json
```

Official repo cung cấp pattern:

```bash
torchrun \
  --nproc_per_node=4 \
  train.py \
  -c configs/dfine/custom/...yaml \
  --use-amp \
  --seed=42
```

và test:

```bash
torchrun \
  --nproc_per_node=4 \
  train.py \
  -c ... \
  --test-only \
  -r model.pth
```

([GitHub][6])

Adapter export giống RT-DETR.

---

# 15. FOVEA — hiện đang thiếu `external/fovea/`

Core đã support:

```text
SeparableWarpTransform
```

và registry đã liệt kê `fovea`.

Nhưng repo HRP4K hiện không có:

```text
external/fovea/
```

Đây là thiếu sót thực tế.

Official FOVEA repo có code đầy đủ, dùng Python 3.8.5, PyTorch 1.6.0 và mmdetection 2.7.0. ([GitHub][2])

### Không nên port FOVEA vào core

Tạo:

```text
external/fovea/
├── README.md
├── manifest.yaml
├── hrp4k_config.py
└── run.py
```

FOVEA bản chất:

```text
saliency prior
    ↓
separable warp
    ↓
small fixed canvas
    ↓
detector
    ↓
inverse warp
    ↓
original coordinates
```

Paper sử dụng spatial/temporal priors và differentiable backward mapping. ([CVPR][7])

HRP4K integration nên ưu tiên **dataset-wide spatial prior trước**, vì dữ liệu hiện là ảnh độc lập trong benchmark framework và Phase 0 đã có spatial distribution.

Nhưng prior phải học từ **train**, không lấy test annotation.

---

# 16. Two-Plane Prior — nên ưu tiên cao nhất trong learned methods

Project đã có:

```yaml
method:
  name: two-plane-prior
```

nhưng chưa có:

```text
external/two_plane_prior/
```

Trong khi code official thực sự tồn tại ở `geometriczoom/two-plane-prior`. Nó dùng learnable geometry-guided warp với ground plane + second plane, và official code pin Python 3.8.5, PyTorch 1.6, mmdetection 2.20, mmcv 1.3.17, kornia 0.5.11. ([Geometric Zoom][8])

Đây rất phù hợp với hypothesis của HRP4K:

```text
far road region
      ↓
small potholes
      ↓
more sampling density
```

### Nhưng có một adaptation issue quan trọng

Official TPP repo chứa:

```text
data/vps/
```

và phương pháp phụ thuộc vào vanishing-point/scene geometry. ([GitHub][1])

HRP4K hiện chưa có vanishing-point metadata.

Vì vậy adapter không chỉ là “đổi dataset path”.

Cần giải quyết:

```text
HRP4K image/video
      ↓
vanishing point estimation
      ↓
TPP geometry
      ↓
learnable warp
```

Không được fit vanishing point bằng test ground-truth annotations vì vi phạm leakage boundary.

Có thể dùng:

* geometric image features;
* lane/road perspective;
* train-only calibration;
* video-level/global prior.

Đây mới là phần nghiên cứu thật sự khi port TPP sang HRP4K.

---

# 17. ZoomDet — nên thêm external adapter đầy đủ

Current HRP4K đã có `external/zoomdet/README`, nhưng mới chỉ là contract.

Official ZoomDet code hiện có và dựa trên mmdetection; repository cung cấp Faster R-CNN implementation và chỉ sang một YOLO variant riêng. ([GitHub][9])

ZoomDet học:

```text
offset predictor
       ↓
non-uniform warp
       ↓
detector
       ↓
corner-aligned inverse bbox transform
```

Paper/code nhấn mạnh việc warp cả GT sang zoomed space khi training rồi map prediction ngược về original space. ([arXiv][10])

Core HRP4K đã chuẩn bị đúng abstraction cho loại này bằng:

```python
GridWarpTransform
```

Nhưng khi reproduction chính thức, nên **dùng transform của ZoomDet upstream**, không viết lại GridWarp rồi tự gọi đó là ZoomDet.

`GridWarpTransform` trong core nên dùng cho:

* validation;
* test transform roundtrip;
* normalization/handoff;
* future native implementation.

---

# 18. AutoFocus

Current contract ghi chính xác:

> phải có category-agnostic focus head và FocusPixel → FocusChip grouping; heuristic crop không được gọi là AutoFocus.

Paper AutoFocus sử dụng coarse-to-fine inference:

```text
coarse image
   ↓
FocusPixel prediction
   ↓
FocusChip grouping
   ↓
higher-resolution detector
   ↓
cross-scale merge
```

([arXiv][11])

Đây không phải adapter đơn giản.

Cần ít nhất:

```text
Focus head training
Chip generator
Multi-scale inference
Duplicate suppression
Canonical export
```

Do đó mình sẽ xếp AutoFocus **sau TPP/FOVEA/ZoomDet**, vì cost reproduction lớn hơn.

---

# 19. AdaZoom

AdaZoom còn phức tạp hơn.

Project contract đã ghi cần:

* learned zoom policy;
* policy-gradient stages;
* detector collaboration.

Paper đúng là sử dụng policy-gradient để tạo adaptive focus regions và collaborative training với detector. ([arXiv][12])

Nên pipeline của nó là:

```text
image
 ↓
policy network
 ↓
zoom action
 ↓
variable magnification
 ↓
detector
 ↓
reward
 ↓
policy gradient
```

Không thể thay bằng:

```python
crop = most_salient_region(image)
```

rồi gọi đó là AdaZoom.

Đây nên là một trong những reproduction cuối cùng.

---

# 20. Thứ tự external adapters mình đề xuất

Với mục tiêu **tiết kiệm compute trong giai đoạn nghiên cứu**, mình sẽ không triển khai tất cả cùng lúc.

| Thứ tự | Adapter         | Lý do                                   |
| ------ | --------------- | --------------------------------------- |
| **1**  | YOLOv5 official | dễ nhất, hoàn thiện detector matrix     |
| **2**  | RT-DETRv2       | official support tốt, COCO-native       |
| **3**  | D-FINE          | custom COCO support rõ                  |
| **4**  | Two-Plane Prior | sát research hypothesis HRP4K nhất      |
| **5**  | ZoomDet         | adaptive warp + official code có sẵn    |
| **6**  | FOVEA           | foundation rất liên quan nhưng stack cũ |
| **7**  | AutoFocus       | cần focus-head reproduction             |
| **8**  | AdaZoom         | RL/collaborative training phức tạp nhất |

RT-DETR official hiện cũng khuyến nghị hướng v2 cho người bắt đầu với RT-DETR, nên nếu compute hạn chế thì không cần ưu tiên v1 trước v2. ([GitHub][13])

---

# Danh sách sửa cuối cùng

Nếu cô đặc thành task list thực tế, mình sẽ dùng:

```text
P0
[ ] Fix PyYAML dependency
[ ] Fix CI test module paths
[ ] Remove raw yaml.safe_load() from `hrp4k run`
[ ] Make run use resolve() + validate()
[ ] Fix nested method.parameters handling
[ ] Reject unknown config fields
[ ] Fix experiment.name / warmup_images / detector.device schema mismatch
[ ] Normalize detector IDs: rt-detr-v1/v2, d-fine

P1
[ ] Propagate training.seed instead of hard-coded 42
[ ] Replace importlib-based external preflight with manifest-based checks
[ ] Clarify "official partial upstream release" metadata semantics
[ ] Add common external canonical-export framework
[ ] Add external/fovea/
[ ] Add external/two_plane_prior/

External reproduction
[ ] YOLOv5 official runner
[ ] RT-DETRv2 runner
[ ] D-FINE runner
[ ] TPP HRP4K adaptation
[ ] ZoomDet HRP4K adaptation
[ ] FOVEA HRP4K adaptation
[ ] AutoFocus reproduction
[ ] AdaZoom reproduction
```

**Ba việc đáng làm ngay nhất vẫn là `CI → unified config → external runner contract`.** Sau đó YOLOv5/RT-DETR/D-FINE chỉ còn là những implementation cụ thể bám cùng một contract; còn TPP/ZoomDet/FOVEA mới là phần nên dành thời gian nghiên cứu sâu, đặc biệt TPP vì nó khớp trực tiếp với giả thuyết perspective-aware resolution allocation của HRP4K.

[1]: https://github.com/geometriczoom/two-plane-prior "GitHub - geometriczoom/two-plane-prior: Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection · GitHub"
[2]: https://github.com/tchittesh/fovea "GitHub - tchittesh/fovea: Code for FOVEA: Foveated Image Magnification for Autonomous Navigation (ICCV 2021) · GitHub"
[3]: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data?utm_source=chatgpt.com "Train Custom Data · ultralytics/yolov5 Wiki · GitHub"
[4]: https://github.com/lyuwenyu/RT-DETR?utm_source=chatgpt.com "GitHub - lyuwenyu/RT-DETR: [CVPR 2024] Official RT-DETR (RTDETR paddle pytorch), Real-Time DEtection TRansformer, DETRs Beat YOLOs on Real-time Object Detection. 🔥 🔥 🔥 · GitHub"
[5]: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch "RT-DETR/rtdetr_pytorch at main · lyuwenyu/RT-DETR · GitHub"
[6]: https://github.com/Peterande/D-FINE "GitHub - Peterande/D-FINE: D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement [ICLR 2025 Spotlight] · GitHub"
[7]: https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html?utm_source=chatgpt.com "ICCV 2021 Open Access Repository"
[8]: https://geometriczoom.github.io/ "Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection"
[9]: https://github.com/twangnh/zoomdet_code "GitHub - twangnh/zoomdet_code: implementation of paper submitted to ISPRS Journal of Photogrammetry and Remote Sensing · GitHub"
[10]: https://arxiv.org/abs/2602.07512?utm_source=chatgpt.com "Adaptive Image Zoom-in with Bounding Box Transformation for UAV Object Detection"
[11]: https://arxiv.org/abs/1812.01600?utm_source=chatgpt.com "AutoFocus: Efficient Multi-Scale Inference"
[12]: https://arxiv.org/abs/2106.10409?utm_source=chatgpt.com "AdaZoom: Adaptive Zoom Network for Multi-Scale Object Detection in Large Scenes"
[13]: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/configs/rtdetr/readme.md?utm_source=chatgpt.com "RT-DETR/rtdetrv2_pytorch/configs/rtdetr/readme.md at main · lyuwenyu/RT-DETR · GitHub"
