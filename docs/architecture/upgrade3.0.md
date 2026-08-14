
# 15. Kiến trúc HRP4K tôi đề xuất

Tôi sẽ chuyển thành:

```text
HRP4K/
│
├── pyproject.toml
├── README.md
├── CHANGELOG.md
│
├── configs/
│   ├── base.yaml
│   │
│   ├── profiles/
│   │   ├── smoke.yaml
│   │   ├── research.yaml
│   │   └── benchmark.yaml
│   │
│   ├── detectors/
│   │   ├── yolov5m.yaml
│   │   ├── yolov8m.yaml
│   │   ├── yolo11m.yaml
│   │   ├── rtdetr.yaml
│   │   └── dfine.yaml
│   │
│   ├── methods/
│   │   ├── resize.yaml
│   │   ├── sliced_nms.yaml
│   │   ├── sahi.yaml
│   │   ├── perspective_grid.yaml
│   │   └── learned_tpp.yaml
│   │
│   └── experiments/
│       ├── baselines/
│       ├── resolution/
│       ├── ablations/
│       └── final/
│
├── src/
│   └── hrp4k/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       │
│       ├── config/
│       │   ├── schema.py
│       │   ├── loader.py
│       │   ├── resolver.py
│       │   └── validation.py
│       │
│       ├── data/
│       │   ├── identity.py
│       │   ├── io.py
│       │   ├── audit.py
│       │   ├── manifest.py
│       │   └── views.py
│       │
│       ├── detectors/
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── ultralytics.py
│       │
│       ├── methods/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── resize.py
│       │   ├── sliced.py
│       │   ├── sahi.py
│       │   └── perspective.py
│       │
│       ├── training/
│       │   ├── runner.py
│       │   ├── presets.py
│       │   └── provenance.py
│       │
│       ├── inference/
│       │   ├── runner.py
│       │   ├── fusion.py
│       │   └── schema.py
│       │
│       ├── evaluation/
│       │   ├── coco.py
│       │   ├── validation.py
│       │   ├── scale.py
│       │   └── fppi.py
│       │
│       ├── experiments/
│       │   ├── runner.py
│       │   ├── artifacts.py
│       │   ├── manifest.py
│       │   └── lifecycle.py
│       │
│       ├── phases/
│       │   ├── phase_0.py
│       │   ├── phase_1.py
│       │   ├── phase_2.py
│       │   └── phase_3.py
│       │
│       ├── diagnostics/
│       │   ├── errors.py
│       │   ├── spatial.py
│       │   └── report.py
│       │
│       ├── reports/
│       │   ├── tables.py
│       │   ├── plots.py
│       │   └── summary.py
│       │
│       └── infra/
│           ├── environment.py
│           ├── serialization.py
│           ├── hashing.py
│           └── timing.py
│
├── external/
│
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── integration/
│   └── scientific/
│
└── docs/
```

Đây là kiến trúc tôi nghĩ phù hợp nhất với HRP4K.

---

# 16. Không nên dùng một `config.yaml` khổng lồ như CystoDS

Đây là một chỗ tôi sẽ **không copy CystoDS**.

CystoDS có một model + nhiều ablation nên central config khá hợp lý.

HRP4K có dimensions:

```text
detector
×
checkpoint
×
resolution strategy
×
inference method
×
runtime
×
dataset
×
evaluation protocol
```

Vì vậy modular composition hợp lý hơn:

```text
base
+
detector
+
method
+
profile
+
experiment
+
environment
+
CLI override
```

Ví dụ:

```text
configs/base.yaml
        +
configs/detectors/yolo11m.yaml
        +
configs/methods/sliced_nms.yaml
        +
configs/profiles/research.yaml
        +
experiment override
```

Sau đó resolver sinh:

```text
resolved_config.yaml
```

Đó mới là config thực sự của experiment.

---

# 17. Config schema nên trở thành first-class contract

Tôi đề xuất schema canonical:

```text
schema_version

experiment
    study_id
    experiment_name
    seed
    profile

dataset
    root
    release_id
    split
    expected_hash

detector
    name
    framework
    checkpoint
    checkpoint_hash
    input_size
    confidence

method
    name
    parameters

runtime
    device
    precision
    warmup
    batch_size
    deterministic

evaluation
    coco
    fppi
    scale_metrics
    thresholds

output
    root
```

Và validation phải chạy **trước khi GPU được initialize**.

CLI mới nên có:

```bash
hrp4k config show ...
hrp4k config validate ...
hrp4k experiment id ...
```

giống lợi ích mà CystoDS đang có với `config show` và `validate`.

---

# 18. Experiment ID phải thực sự immutable

HRP4K hiện đã hash config để tạo `experiment_id`. Đây là nền móng rất tốt.

Nhưng nên nâng lên:

```text
experiment_id =
hash(
    resolved config
    +
    dataset identity
    +
    checkpoint identity
    +
    code version
)
```

Vì:

```text
same YAML
+
different checkpoint
```

không phải cùng experiment.

Tương tự:

```text
same config
+
different dataset release
```

cũng không phải cùng experiment.

---

# 19. Chuẩn hóa output layout

Hiện output path khá command-specific.

Tôi đề xuất mỗi experiment luôn có:

```text
outputs/
└── <study_id>/
    └── <experiment_id>/
        ├── status.json
        ├── resolved_config.yaml
        │
        ├── provenance/
        │   ├── dataset.json
        │   ├── environment.json
        │   ├── git.json
        │   ├── packages.txt
        │   └── checkpoint.json
        │
        ├── checkpoints/
        │
        ├── predictions/
        │   └── coco.json
        │
        ├── metrics/
        │   ├── coco.json
        │   ├── scale.json
        │   └── fppi.json
        │
        ├── diagnostics/
        │
        ├── figures/
        │
        └── logs/
```

Sau đó mọi command đều chỉ đọc/write contract này.

---

# 20. Tách prediction schema khỏi runner

Hiện `runner.py` vừa:

* load dataset;
* warmup;
* inference;
* view generation;
* NMS;
* timing;
* metadata;
* experiment hashing;
* serialization.

Nó nên trở thành:

```text
ExperimentRunner
     │
     ├── Method
     ├── Detector
     ├── Timer
     ├── PredictionWriter
     └── ArtifactStore
```

và canonical prediction:

```text
inference/schema.py
```

Như vậy external code chỉ cần implement:

```text
DetectorAdapter
```

hoặc export canonical COCO predictions.

---

# 21. Registry cần được thống nhất

Hiện HRP4K có:

```text
BASELINE_PRESETS
DETECTOR_STATUS
METHOD_REGISTRY
METHOD_STATUS
```

rải ở nhiều module. Điều này sớm muộn sẽ bị drift. `cli.py` hiện cũng phải import nhiều registry/status khác nhau.

Nên có:

```python
DetectorSpec
MethodSpec
```

với metadata thống nhất:

```text
id
display_name
implementation
status
framework
checkpoint
paper
official_repository
config_schema
supports_training
supports_inference
reproduction_level
```

`reproduction_level` cực kỳ có giá trị:

```text
official
faithful
compatibility
heuristic
external_required
unimplemented
```

---

# 22. Phase orchestration nên học trực tiếp CystoDS

HRP4K đang có semantic Phase 0–3 rất rõ.

Vậy hãy giữ nó:

```text
phase_0
Dataset audit + identity

phase_1
Detector benchmark/training

phase_2
Resolution-allocation benchmark

phase_3
Diagnostics + scientific comparison
```

Mỗi file chỉ nên khoảng:

```text
50–150 LOC
```

và gọi các subsystem.

Ví dụ:

```text
phase_2.py

resolve experiment matrix
        ↓
verify Phase 1 checkpoint receipts
        ↓
run methods
        ↓
evaluate
        ↓
save phase artifact
```

Không implement NMS hay detector trong file phase.

---

# 23. Cross-phase artifacts là nâng cấp quan trọng nhất sau config

Có thể định nghĩa:

```text
Phase 0
dataset_receipt.json

Phase 1
detector_receipt.json

Phase 2
prediction_receipt.json

Phase 3
benchmark_receipt.json
```

Ví dụ:

```json
{
  "schema_version": "hrp4k.detector_receipt.v1",
  "experiment_id": "...",
  "dataset_sha256": "...",
  "detector": "yolo11m",
  "checkpoint_sha256": "...",
  "training_config_sha256": "...",
  "git_sha": "..."
}
```

Phase 2 phải check:

```text
dataset_sha == current dataset
checkpoint_sha == selected checkpoint
```

Nếu không:

```text
FAIL FAST
```

Đây chính là pattern rất mạnh của CystoDS.

Nhưng tôi sẽ **không tìm artifact mới nhất theo mtime**.

Dùng:

```text
upstream_experiment_id
```

rõ ràng trong config.

---

# 24. Scientific gates riêng cho HRP4K

Nên tạo:

```text
scientific/
```

hoặc:

```text
protocol/
```

chứa các invariant sau:

```text
official dataset required for benchmark
smoke != scientific result
no automatic re-split
test annotations evaluation-only
canonical prediction required
unknown image/category forbidden
invalid bbox forbidden
checkpoint provenance required
latency requires warmup
latency comparison requires same hardware/protocol
heuristic != paper reproduction
external-required cannot silently fallback
```

Thực ra các nguyên tắc này HRP4K đã viết rất tốt trong `REPRODUCIBILITY.md`.

Bước tiếp theo chỉ là:

> **biến documentation contract thành executable contract.**

---

# 25. Refactor `dataset.py`

`dataset.py` hiện khoảng 22 KB và đang ôm quá nhiều responsibility.

Tôi sẽ chia:

```text
dataset.py
    ↓

data/
├── identity.py
├── coco.py
├── audit.py
├── manifest.py
├── paths.py
└── views.py
```

Mapping:

| Current                 | New                |
| ----------------------- | ------------------ |
| `dataset_identity.py`   | `data/identity.py` |
| COCO loading            | `data/coco.py`     |
| integrity/stats         | `data/audit.py`    |
| dataset manifest        | `data/manifest.py` |
| smoke/full symlink view | `data/views.py`    |
| image resolution/path   | `data/paths.py`    |

`official split` vẫn giữ nguyên.

---

# 26. Refactor inference methods

Hiện `processing.py` xử lý các transforms/methods, và `runner.py` thậm chí import private `_starts` từ `processing`.

Đó là dấu hiệu abstraction boundary chưa sạch.

Nên chuyển:

```text
methods/
├── base.py
├── resize.py
├── slicing.py
├── perspective.py
└── sahi.py
```

Contract:

```python
class ResolutionMethod:
    def prepare_views(image) -> list[View]:
        ...

    def fuse(predictions) -> list[Detection]:
        ...
```

Runner không cần biết:

```text
tile_size
perspective grid
SAHI implementation
```

Runner chỉ biết:

```text
Method → Views → Detector → Fuse
```

---

# 27. Tách detector framework

Tương tự:

```text
detectors/
├── base.py
├── registry.py
├── ultralytics.py
└── external.py
```

Contract:

```text
load()
warmup()
predict()
metadata()
```

Sau này:

```text
Ultralytics
RT-DETR
D-FINE
YOLOv5 official
```

không làm `runner.py` lớn thêm.

---

# 28. Testing architecture nên nâng thành 4 tầng

HRP4K hiện đã có test dataset, evaluation, prediction validation, runner, transforms... khá tốt.

Sau refactor:

```text
tests/
├── unit/
│
├── contracts/
│   ├── test_config_contract.py
│   ├── test_prediction_schema.py
│   ├── test_detector_contract.py
│   └── test_method_contract.py
│
├── scientific/
│   ├── test_official_split.py
│   ├── test_no_test_leakage.py
│   ├── test_smoke_not_benchmark.py
│   └── test_reproduction_labels.py
│
└── integration/
    ├── test_phase0.py
    ├── test_phase1_smoke.py
    ├── test_phase2_smoke.py
    └── test_pipeline_smoke.py
```

Đây sẽ gần với triết lý CystoDS nhưng phù hợp HRP4K hơn.

---

# 29. `pyproject.toml` cũng nên nâng cấp

CystoDS đã khai báo:

* pytest;
* ruff;
* src layout;
* pytest path;
* ruff target;
* CLI entrypoint.

HRP4K `pyproject.toml` hiện còn khá tối giản.

Tôi sẽ thêm:

```text
dev:
pytest
pytest-cov
ruff
mypy

tool.pytest
tool.ruff
tool.coverage
```

và chuyển package sang:

```text
src/hrp4k/
```

`src layout` giúp test tránh tình trạng vô tình import source directory thay vì installed package.

---

# 30. CI mục tiêu

Giữ CI hiện tại nhưng nâng thành:

```text
lint
    ↓
unit
    ↓
contract
    ↓
evaluation integration
    ↓
vision integration
    ↓
smoke pipeline
```

Matrix:

```text
Python 3.10
Python 3.11
Python 3.12
```

Không chạy GPU benchmark trên GitHub Actions.

CI chỉ cần:

```text
tiny deterministic fixture
```

cho inference.

---

# 31. Kế hoạch migration cụ thể

Tôi khuyên triển khai theo **8 PR**, tuyệt đối không big-bang rewrite.

### PR1 — Freeze current behavior

Không refactor logic.

Thêm:

* CLI snapshot tests;
* current prediction schema fixtures;
* current experiment output fixture;
* dataset manifest fixtures;
* smoke pipeline regression test.

Mục tiêu:

```text
old implementation
==
refactored implementation
```

---

### PR2 — Config subsystem

Tạo:

```text
src/hrp4k/config/
```

với:

```text
schema
loader
resolver
validation
```

Thêm:

```bash
hrp4k config show
hrp4k config validate
```

Chuyển `run --config` sang resolved config.

Đây là PR quan trọng nhất.

---

### PR3 — Data + infra decomposition

Move:

```text
dataset_identity.py
dataset.py
timing.py
```

thành:

```text
data/
infra/
```

Không thay đổi behavior.

Compatibility imports có thể giữ tạm:

```python
# old dataset.py
from hrp4k.data import *
```

---

### PR4 — Detector + Method abstractions

Tạo:

```text
DetectorAdapter
ResolutionMethod
```

Sau đó migrate:

```text
Ultralytics
Resize
SlicedNMS
PerspectiveGrid
SAHI
```

Một method một module.

---

### PR5 — Experiment lifecycle

Tạo:

```text
experiments/
├── runner.py
├── artifacts.py
├── manifest.py
└── lifecycle.py
```

Chuẩn hóa run directory.

Thêm:

```text
pending
running
completed
failed
```

vào `status.json`.

Mọi run sinh:

```text
resolved_config
environment
dataset receipt
git receipt
checkpoint receipt
```

---

### PR6 — Phase orchestration

Di chuyển logic khỏi `cli.py`.

Tạo:

```text
phases/
├── phase_0.py
├── phase_1.py
├── phase_2.py
└── phase_3.py
```

`run-smoke` trở thành:

```python
run_pipeline(profile="smoke")
```

thay vì hard-code từng call.

---

### PR7 — Evaluation + diagnostics decomposition

Chuyển:

```text
evaluation.py
diagnostics.py
predictions.py
```

thành packages.

Đặc biệt tách:

```text
prediction validation
COCO metrics
scale metrics
FPPI
spatial diagnostics
report generation
```

---

### PR8 — Scientific gates + CI + cleanup

Thêm scientific contract tests.

Xóa compatibility modules.

Bật:

```text
ruff check
pytest
coverage
```

trên CI.

Freeze:

```text
hrp4k.core.v1
hrp4k.prediction.v1
hrp4k.artifact.v1
```

---

# 32. Thứ tự priority nếu muốn refactor ngay

Tôi sẽ ưu tiên theo thứ tự:

```text
1. Config architecture
        ↓
2. Experiment/artifact contract
        ↓
3. Package/domain decomposition
        ↓
4. Method/Detector abstraction
        ↓
5. Phase orchestration
        ↓
6. Scientific gates
        ↓
7. Reports
        ↓
8. CI polish
```

Đừng bắt đầu bằng việc chỉ:

```text
move file → folder
```

Nếu làm vậy thì HRP4K chỉ đẹp directory hơn, architecture thực tế không đổi.

---

# 33. Những thứ tuyệt đối không nên làm

Có 6 điều tôi đặc biệt tránh:

1. **Không thay đổi scientific logic đồng thời với refactor.** Move architecture trước, thuật toán sau.

2. **Không copy split subsystem từ CystoDS.** HRP4K có official split và phải giữ nguyên contract đó.

3. **Không đưa external implementations trực tiếp vào core package.** Giữ canonical contract.

4. **Không tạo một `config.yaml` 1.000 dòng.** HRP4K hợp với compositional config hơn.

5. **Không thêm cùng lúc Hydra + MLflow + DVC + Lightning.** Sẽ biến refactor architecture thành migration framework.

6. **Không dùng “latest result” làm dependency mặc định.** Mọi upstream artifact nên được tham chiếu bằng `experiment_id`.

---

# 34. Target flow cuối cùng

Sau refactor, một experiment HRP4K lý tưởng sẽ là:

```text
                    configs
                       │
                       ▼
               Config Resolver
                       │
                       ▼
              Resolved Config
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Dataset Identity                Preflight
        │                             │
        └──────────────┬──────────────┘
                       ▼
               Experiment Runner
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Method              Detector
             │                   │
             └─────────┬─────────┘
                       ▼
                 Predictions
                       │
                       ▼
              Canonical Validator
                       │
                       ▼
                  Evaluation
                       │
                       ▼
                 Diagnostics
                       │
                       ▼
              Scientific Gates
                       │
                       ▼
                Artifact Store
                       │
                       ▼
                    Report
```

Và bên trên toàn bộ hệ thống:

```text
Phase 0
   ↓
Phase 1
   ↓
Phase 2
   ↓
Phase 3
```

mỗi phase chỉ là thin orchestrator.

---

# 35. Cấu trúc nào của hai repo nên thắng?

Tóm gọn:

| Thành phần                     | Nên lấy từ                     |
| ------------------------------ | ------------------------------ |
| Package decomposition          | **CystoDS**                    |
| Thin stage orchestration       | **CystoDS**                    |
| Config resolver                | **CystoDS**, nhưng modular hơn |
| Artifact dependency            | **CystoDS**, nhưng explicit ID |
| Scientific tests               | **CystoDS**                    |
| Official dataset identity      | **HRP4K**                      |
| Official split policy          | **HRP4K**                      |
| Canonical prediction           | **HRP4K**                      |
| Latency protocol               | **HRP4K**                      |
| External reproduction contract | **HRP4K**                      |
| Reproduction status matrix     | **HRP4K**                      |
| GitHub CI                      | **HRP4K**                      |
| Final architecture             | **Hybrid**                     |

### Mục tiêu không phải:

```text
HRP4K → CystoDS architecture
```

mà là:

```text
CystoDS software architecture
            +
HRP4K benchmark rigor
            +
explicit experiment provenance
            +
modular configs
            =
HRP4K v1 research framework
```

Theo tôi, **refactor quan trọng nhất không phải chia `hrp4k_suite` thành nhiều folder**. Đó chỉ là phần nhìn thấy bên ngoài. Refactor thật sự là đưa **Config → Experiment → Artifact → Phase → Scientific Contract** thành 5 abstraction chính của repo. Khi 5 abstraction này ổn định, bạn có thể thêm YOLO, RT-DETR, D-FINE, AutoFocus, AdaZoom, ZoomDet hay learned TPP mà core architecture gần như không phải thay đổi. Đây chính là mức mà HRP4K nên hướng đến nếu mục tiêu là một repo nghiên cứu có thể publish, reproduce và tiếp tục phát triển lâu dài.
