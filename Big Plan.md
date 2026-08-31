# HRP4K — Final Refactor, Experiment Pipeline & Server-Ready Implementation Plan

## 1. Mục tiêu

Refactor toàn bộ repository `nxc1802/HRP4K` thành một research codebase **tối giản, dễ maintain, dễ theo dõi và ready-to-train trên GPU Server**.

Không tiếp tục mở rộng benchmark framework cũ.

Mục tiêu cuối cùng:

```text
Clean Repository
        ↓
One-command Setup
        ↓
One-command Experiment
        ↓
Train → Val → Test
        ↓
Per-epoch HF checkpoint + metrics sync
        ↓
Resume from HF without data loss
        ↓
Automatic final experiment report
        ↓
Experiment Final
```

Repository sau refactor chỉ chứa những thành phần cần thiết cho research hiện tại.

---

# 2. Research Scope — BẮT BUỘC

Chỉ nghiên cứu hai detector:

```text
YOLO11m
D-FINE-M
```

Không implement hoặc giữ lại experiment/runtime cho:

```text
YOLOv5m
YOLOv8m
RT-DETRv1
RT-DETRv2
AutoFocus
AdaZoom
FOVEA
Two-Plane Prior
TPP
ZoomDet implementation cũ
```

Các thành phần trên nếu không còn dependency với core pipeline phải được **xóa hoàn toàn**, không chỉ đánh dấu deprecated.

Không giữ code/docs/config chỉ vì chúng từng được sử dụng trong experiment cũ.

---

# 3. Experiment Scope

Chỉ giữ ba phase nghiên cứu:

## Phase 1 — Resolution

Implement đầy đủ cho:

```text
YOLO11m
D-FINE-M
```

Resolution matrix:

```text
4K
2K
1K
640
```

Mỗi resolution là một experiment độc lập.

Mục tiêu là hoàn thành:

### Table 1 — Resolution

Theo `docs/experiment_form.md`.

Các metric cần có:

```text
AP50
AP75
AP50:95
AP Ultra-Fine
AP Small
AP Medium
AP Large
Precision
Recall
F1
FPPI
Latency
```

---

# 4. Phase 2 — Spatial Decomposition / Slicing

Phase 2 sử dụng detector checkpoint đã train ở Resolution 640.

Implement đầy đủ cho:

```text
Full Image / Resize
Sliced-NMS
SAHI
Perspective Grid
```

Không implement thêm learned high-resolution methods.

## Table 2 — Inference

Hoàn thành:

```text
Full Image (Baseline)
Sliced-NMS
SAHI
Perspective Grid
```

## Table 3 — Training + Inference

Chỉ thực hiện nếu experiment protocol yêu cầu training riêng cho spatial decomposition.

Không tự động mở rộng thêm các method ngoài experiment form.

Hai detector:

```text
YOLO11m
D-FINE-M
```

phải có experiment matrix tương ứng.

Detector checkpoint phải được giữ nhất quán trong cùng một comparison group.

---

# 5. Phase 3 — Proposed Method

Chỉ tạo **pipeline abstraction**, KHÔNG triển khai một proposed method cụ thể.

Mục tiêu:

```text
Scout
  ↓
Region Selection
  ↓
High-resolution processing
  ↓
Global / Local detector
  ↓
Coordinate mapping
  ↓
Fusion
```

Pipeline phải có interface rõ ràng để sau này có thể thay:

```text
Scout implementation
Crop policy
High-resolution allocation strategy
Detector
Fusion strategy
```

Nhưng hiện tại:

```text
NO training
NO benchmark
NO claimed result
NO ablation
NO fake implementation
```

Không đưa proposed method vào các bảng kết quả hiện tại.

---

# 6. Experiment Form là nguồn sự thật

`docs/experiment_form.md` là source of truth cho cấu trúc bảng experiment.

Không tự ý tạo thêm bảng research mới.

Nếu code hiện tại có experiment không tương ứng với:

```text
Phase Resolution
Phase Slicing
Phase Proposed Method
```

thì phải xem xét xóa.

---

# 7. Experiment Final

Tạo một file duy nhất:

```text
docs/Experiment_Final.md
```

Đây là **single source of truth cho kết quả cuối cùng của research experiment**.

Không tiếp tục sử dụng:

```text
experiments_master.md
```

nếu nội dung hiện tại chứa kết quả/protocol cũ hoặc out-of-scope experiment.

`Experiment_Final.md` chỉ chứa:

1. Experiment ID
2. Detector
3. Phase
4. Experiment configuration
5. Dataset identity
6. Train information
7. Validation metrics
8. Test metrics
9. Scale-wise metrics
10. Material-wise metrics nếu metadata chính thức tồn tại
11. Latency
12. Checkpoint HF link
13. Prediction/metric HF links
14. Status

---

# 8. Experiment Registry

Mỗi experiment phải có deterministic ID.

ID được tạo từ normalized experiment configuration:

```text
SHA256(canonical_config)
```

Ví dụ:

```text
yolo11m-resolution-4k-<experiment_id>
dfine-m-resolution-4k-<experiment_id>
```

Experiment ID phải thay đổi nếu bất kỳ thông số research-relevant nào thay đổi.

Không dùng filename làm identity.

---

# 9. Không được ghi đè Experiment cũ

Đây là yêu cầu bắt buộc.

Mỗi experiment có một immutable identity.

Nếu experiment đang tồn tại trên Hugging Face:

```text
DO NOT restart from scratch
DO NOT overwrite historical training information
DO NOT replace previous experiment metadata
```

Thay vào đó:

```text
read remote experiment state
        ↓
find latest valid checkpoint
        ↓
download last checkpoint
        ↓
restore training state
        ↓
continue training
        ↓
append new information
```

---

# 10. Hugging Face Storage Model

Hugging Face là **persistent experiment storage**.

Mỗi experiment có thư mục riêng:

```text
experiments/
└── <experiment_id>/
    ├── manifest.json
    ├── config.json
    ├── environment.json
    ├── training/
    │   ├── history.jsonl
    │   ├── results.csv
    │   └── epochs/
    │       ├── epoch-001/
    │       │   ├── last.pt
    │       │   ├── best.pt
    │       │   └── metrics.json
    │       ├── epoch-002/
    │       └── ...
    ├── checkpoints/
    │   ├── last.pt
    │   └── best.pt
    ├── validation/
    │   └── metrics.json
    ├── test/
    │   ├── predictions.json
    │   └── metrics.json
    └── Experiment_Final.md
```

Không upload toàn bộ dataset lên experiment folder.

---

# 11. Checkpoint policy

Sau **mỗi epoch**:

```text
last.pt
best.pt
training metrics
validation metrics
training history
```

phải được đồng bộ lên Hugging Face.

Checkpoint historical theo epoch không được ghi đè:

```text
epoch-001
epoch-002
epoch-003
...
```

`checkpoints/last.pt` và `checkpoints/best.pt` chỉ là convenience pointers/current copies.

Historical checkpoints phải được giữ riêng.

---

# 12. Resume Policy

Khi chạy:

```bash
hrp4k experiment ...
```

pipeline phải tự kiểm tra HF.

Nếu experiment chưa tồn tại:

```text
START FROM SCRATCH
```

Nếu experiment tồn tại:

```text
RESUME
```

từ checkpoint hợp lệ mới nhất.

Không yêu cầu user phải tự chỉ định checkpoint nếu HF đã có experiment state.

Ví dụ:

```bash
hrp4k experiment yolo11m-resolution-4k
```

có thể tiếp tục experiment đã bị server shutdown.

---

# 13. Resume phải restore đầy đủ

Resume phải khôi phục:

```text
model weights
optimizer
scheduler
epoch
best metric
training history
experiment configuration
random seed/state nếu framework hỗ trợ
```

Không được chỉ load weights rồi bắt đầu lại optimizer/scheduler.

---

# 14. Không mất dữ liệu khi upload thất bại

HF upload failure:

```text
MUST NOT crash training
```

nhưng phải ghi:

```text
sync status
failed files
retry status
timestamp
```

Training chỉ được đánh dấu:

```text
completed
```

sau khi local artifacts cần thiết đã tồn tại.

Final experiment report chỉ được đánh dấu:

```text
finalized
```

sau khi train + validation + test + required uploads hoàn tất.

---

# 15. Training Metrics

Mỗi epoch phải lưu đầy đủ thông tin framework cung cấp.

Tối thiểu:

```text
epoch
learning rate
training loss
validation loss nếu framework cung cấp
precision
recall
mAP50
mAP50:95
```

Không hard-code một tập metric nếu detector framework có metric tương đương khác.

Raw framework metrics phải được giữ.

---

# 16. Validation Metrics

Sau training:

```text
best checkpoint
last checkpoint
```

phải được validation.

Lưu:

```text
overall metrics
scale-wise metrics
material-wise metrics nếu có metadata hợp lệ
operating-point metrics
FPPI
```

---

# 17. Test Metrics

Sau khi training + validation hoàn tất:

```text
BEST checkpoint → TEST
```

Test phải chạy trên canonical test split.

Không dùng test result để:

```text
select checkpoint
tune hyperparameter
change model configuration
```

Test result chỉ được tạo sau khi configuration đã freeze.

---

# 18. Scale-wise Metrics

Metric phải được xuất riêng cho:

```text
Ultra-Fine
Fine / Small
Medium
Large
```

Tên scale phải thống nhất với research protocol trong dataset/evaluation implementation.

Không hard-code scale definition ở nhiều file.

Chỉ có một source of truth.

---

# 19. Road-surface Metrics

Cần hỗ trợ:

```text
Asphalt
Concrete
```

NHƯNG:

**Không được tự suy đoán material từ ảnh hoặc filename.**

Trước khi implement phải kiểm tra dataset/release có metadata chính thức tương ứng hay không.

Nếu metadata tồn tại:

```text
material_id → image_id
```

và evaluator dùng mapping đó.

Nếu metadata không tồn tại trong canonical release:

```text
DO NOT fabricate material labels
```

Thay vào đó:

```text
material metrics = unavailable
```

và ghi rõ lý do trong Experiment Final.

---

# 20. CLI Design

Toàn bộ experiment phải chạy bằng CLI.

Không yêu cầu user mở notebook.

Không yêu cầu user chạy nhiều shell command cho một experiment.

---

# 21. Setup CLI

Chỉ cần một lệnh:

```bash
hrp4k setup
```

Setup phải tự động:

1. Clone/update repository nếu cần.
2. Install dependencies.
3. Configure environment.
4. Validate Python/runtime.
5. Validate CUDA nếu có.
6. Configure Hugging Face credentials.
7. Download/prepare HRP4K dataset.
8. Verify dataset identity.
9. Prepare required directories.
10. Run basic environment check.

Không yêu cầu user phải nhớ:

```text
git clone
pip install
export HF_TOKEN
export HF_REPO
download dataset
```

---

# 22. Experiment CLI

Mỗi experiment chỉ cần **một dòng CLI**.

Ví dụ:

```bash
hrp4k experiment yolo11m-resolution-4k
```

```bash
hrp4k experiment yolo11m-resolution-2k
```

```bash
hrp4k experiment yolo11m-resolution-1k
```

```bash
hrp4k experiment yolo11m-resolution-640
```

```bash
hrp4k experiment dfine-m-resolution-4k
```

...

Slicing:

```bash
hrp4k experiment yolo11m-slicing-sliced-nms
```

```bash
hrp4k experiment yolo11m-slicing-sahi
```

```bash
hrp4k experiment yolo11m-slicing-perspective-grid
```

v.v.

CLI không được yêu cầu user truyền hàng chục flag nếu experiment protocol đã được freeze.

---

# 23. Config vẫn tồn tại, nhưng user không cần quản lý nó

Internal config có thể nằm trong:

```text
configs/
```

Nhưng:

```text
CLI → Experiment ID → resolve config
```

User không cần tự viết YAML cho các experiment chính thức.

Config phải chứa toàn bộ resolved parameters để reproducibility.

---

# 24. Default Configuration

Default phải lấy từ:

```text
docs/yolo11m_config.md
docs/dfinem_config.md
```

Không tự ý thay đổi default trong quá trình refactor.

Các giá trị như:

```text
epochs
optimizer
lr0
lrf
weight_decay
warmup
batch
effective batch
AMP
rect
seed
augmentation
confidence
```

phải được resolve từ protocol tương ứng của detector.

Nếu config docs và code hiện tại mâu thuẫn:

```text
research protocol/documentation
→ verify
→ resolve explicitly
→ update one source of truth
```

Không âm thầm chọn một giá trị.

---

# 25. Detector Abstraction

Chỉ cần:

```text
Detector
├── YOLO11m
└── D-FINE-M
```

Interface tối thiểu:

```text
train()
validate()
predict()
export_predictions()
load_checkpoint()
save_checkpoint()
```

YOLO11m sử dụng official Ultralytics implementation.

D-FINE-M phải sử dụng implementation/runtime phù hợp với **official D-FINE-M**, không được giả lập D-FINE-M bằng RT-DETR hoặc alias framework khác.

---

# 26. Evaluation Abstraction

Evaluator phải độc lập với detector.

Input:

```text
ground truth
+
canonical predictions
+
experiment metadata
```

Output:

```text
overall metrics
scale metrics
material metrics
FPPI
latency
```

Canonical prediction schema phải thống nhất:

```json
{
  "image_id": 123,
  "category_id": 1,
  "bbox": [x, y, width, height],
  "score": 0.93
}
```

---

# 27. Resolution Pipeline

Resolution experiment phải thực hiện:

```text
Dataset
 ↓
Train detector
 ↓
Save checkpoint every epoch
 ↓
Validation
 ↓
Best checkpoint
 ↓
Test
 ↓
Scale/material evaluation
 ↓
Latency
 ↓
HF sync
 ↓
Experiment Final update
```

Các resolution:

3840 × 2160
1920 × 1080
960 × 540
640 × 360

Phải tuân thủ detector-specific training protocol.

---

# 28. Slicing Pipeline

Slicing experiment:

```text
Frozen 640 detector checkpoint
        ↓
4K test image
        ↓
Spatial method
        ↓
Canonical predictions
        ↓
Coordinate remapping
        ↓
NMS / fusion
        ↓
Evaluation
```

Không train lại detector nếu method là inference-only.

---

# 29. Proposed Method Skeleton

Tạo abstraction:

```text
ProposedMethod
├── Scout
├── RegionSelector
├── HighResolutionProcessor
├── CoordinateTransform
├── Detector
└── Fusion
```

Nhưng implementation hiện tại chỉ cần:

```text
interfaces
configuration schema
pipeline orchestration
placeholder implementation
tests
```

Không tạo result.

Không tạo fake metric.

Không claim proposed method hoạt động.

---

# 30. Repository Cleanup

Audit toàn bộ repository.

Xóa:

```text
outdated code
unused code
dead code
obsolete experiment code
obsolete CLI commands
obsolete configs
obsolete external runners
obsolete scripts
obsolete experiment results
obsolete reports
obsolete Markdown documentation
obsolete upgrade plans
```

Không giữ một file chỉ vì:

```text
"có thể dùng sau này"
```

Nếu chưa nằm trong scope research hiện tại thì xóa.

---

# 31. Documentation Cleanup

Documentation tối thiểu nên còn:

```text
README.md
commands.md
docs/
├── experiment_form.md
├── yolo11m_config.md
├── dfinem_config.md
└── Experiment_Final.md
```

Có thể giữ paper gốc nếu cần làm scientific reference:

```text
docs/paper/
```

Nhưng không giữ các analysis/report cũ nếu chúng chứa protocol hoặc kết quả đã obsolete.

---

# 32. Không giữ Experiment Master cũ

Nếu:

```text
experiments_master.md
```

chứa:

* experiment ngoài scope
* result cũ
* claimed proposed method
* detector ngoài YOLO11m/D-FINE-M
* protocol không còn dùng

thì phải xóa và thay bằng:

```text
docs/Experiment_Final.md
```

Không migrate các kết quả out-of-scope vào file mới.

Chỉ giữ:

```text
experiment
result
HF link
```

của experiment nằm trong final research matrix.

---

# 33. Checkpoint folder

Không commit checkpoint vào Git repository.

Nếu repository/local workspace đang có:

```text
checkpoints/
```

và các checkpoint đã được đồng bộ đầy đủ lên Hugging Face:

```text
remove checkpoints/
```

Code không được phụ thuộc vào việc repository có checkpoint folder.

Resume phải lấy checkpoint từ:

```text
local run state
OR
Hugging Face experiment storage
```

---

# 34. Outputs

Không commit training outputs vào Git.

Local structure:

```text
outputs/
└── experiments/
    └── <experiment_id>/
```

`outputs/` phải nằm trong `.gitignore`.

Hugging Face là persistent storage cho experiment artifacts.

---

# 35. Không tạo framework phức tạp

Không sử dụng:

```text
Hydra
MLflow
Weights & Biases
database
dashboard
distributed experiment manager
```

trừ khi repository hiện tại thực sự yêu cầu.

Mục tiêu là:

```text
Python + YAML + CLI + Hugging Face
```

và càng ít abstraction càng tốt.

---

# 36. Smoke Test

Sau khi refactor hoàn tất, bắt buộc chạy local smoke test.

Smoke test phải sử dụng:

```text
minimal dataset
minimal epochs
minimal image size
```

nhưng phải chạy **tất cả experiment types được định nghĩa trong final matrix**.

Tối thiểu:

```text
YOLO11m Resolution
    4K
    2K
    1K
    640

D-FINE-M Resolution
    4K
    2K
    1K
    640

YOLO11m Slicing
    Full Image
    Sliced-NMS
    SAHI
    Perspective Grid

D-FINE-M Slicing
    Full Image
    Sliced-NMS
    SAHI
    Perspective Grid

Proposed Method
    pipeline-only smoke
```

Smoke test không cần tạo research result.

Mục tiêu là kiểm tra:

```text
CLI
→ config
→ dataset
→ model
→ checkpoint
→ validation
→ test
→ metrics
→ HF sync mock/local
→ report generation
→ resume
```

---

# 37. Resume Smoke Test

Không chỉ test training từ đầu.

Phải có test:

```text
start experiment
↓
stop after epoch N
↓
simulate clean restart
↓
load experiment state
↓
resume from epoch N
↓
continue to N+1
```

Kiểm tra:

```text
no overwrite
no duplicate epoch
history preserved
checkpoint restored
experiment ID unchanged
```

---

# 38. Hugging Face Resume Smoke Test

Phải mô phỏng:

```text
Local run
↓
upload epoch 1
↓
delete local checkpoint
↓
restart
↓
download checkpoint from HF
↓
resume epoch 2
```

Nếu không thể test live HF trong CI:

```text
mock HF storage
```

nhưng phải có ít nhất một manual integration test với HF thật trước khi server deployment.

---

# 39. Final Experiment Generation

Sau khi:

```text
train complete
validation complete
test complete
```

pipeline tự động cập nhật:

```text
docs/Experiment_Final.md
```

Không yêu cầu user copy/paste metric thủ công.

Experiment Final phải chứa link đến:

```text
HF experiment folder
checkpoint best
checkpoint last
metrics
predictions
config
```

---

# 40. Experiment Final không được ghi đè lịch sử

Nếu experiment được resume:

```text
Experiment_Final.md
```

phải update experiment hiện tại bằng thông tin mới.

Không xóa:

```text
previous training history
previous checkpoint references
previous sync status
```

Nếu cần cập nhật metric:

```text
preserve provenance
```

và ghi:

```text
updated_at
resumed_from_epoch
final_epoch
```

---

# 41. Acceptance Criteria

Refactor chỉ được coi là hoàn thành khi:

## Repository

* [ ] Không còn detector ngoài YOLO11m/D-FINE-M.
* [ ] Không còn learned method ngoài proposed-method skeleton.
* [ ] Không còn obsolete experiment code.
* [ ] Không còn obsolete CLI commands.
* [ ] Không còn obsolete configs.
* [ ] Không còn obsolete experiment reports.
* [ ] Không còn obsolete upgrade documents.
* [ ] Checkpoint folder đã bị xóa khỏi repository.
* [ ] Outputs không được commit.

## CLI

* [ ] `hrp4k setup` hoạt động.
* [ ] Mỗi official experiment chạy bằng một CLI command.
* [ ] Không cần notebook.
* [ ] Không cần manually copy checkpoint.
* [ ] Resume tự động.

## Experiment

* [ ] Resolution 4K/2K/1K/640 cho YOLO11m.
* [ ] Resolution 4K/2K/1K/640 cho D-FINE-M.
* [ ] Slicing experiments cho cả hai detector.
* [ ] Proposed method pipeline skeleton tồn tại nhưng chưa benchmark.

## Metrics

* [ ] Train metrics được lưu.
* [ ] Validation metrics được lưu.
* [ ] Test metrics được lưu.
* [ ] Scale-wise metrics được lưu.
* [ ] Material-wise metrics chỉ được lưu khi có authoritative metadata.
* [ ] FPPI được lưu.
* [ ] Latency được lưu.
* [ ] Config/provenance được lưu.

## Hugging Face

* [ ] Checkpoint được sync sau mỗi epoch.
* [ ] Historical epoch checkpoints không bị overwrite.
* [ ] Best/last checkpoint được lưu.
* [ ] Metrics được sync.
* [ ] Experiment metadata được sync.
* [ ] Resume từ HF hoạt động.
* [ ] Resume không mất history.

## Experiment Final

* [ ] Tự động cập nhật sau train/val/test.
* [ ] Chỉ chứa in-scope experiments.
* [ ] Có HF links (https://huggingface.co/datasets/Cuong2004/HRP4K/tree/main, using HF_TOKEN).
* [ ] Không có fake/placeholder research results.

## Smoke Test

* [ ] Tất cả official experiment commands chạy thành công.
* [ ] Proposed pipeline smoke chạy thành công.
* [ ] Resume smoke chạy thành công.
* [ ] HF resume integration test chạy thành công.
* [ ] Repository ở trạng thái ready-to-train trên Server.

---

# 42. Final Principle

Sau refactor, repository phải được hiểu đơn giản như sau:

```text
HRP4K
│
├── setup
│
├── experiment
│   ├── YOLO11m
│   └── D-FINE-M
│
├── resolution
│   ├── 4K
│   ├── 2K
│   ├── 1K
│   └── 640
│
├── slicing
│   ├── Full Image
│   ├── Sliced-NMS
│   ├── SAHI
│   └── Perspective Grid
│
└── proposed-method
    └── pipeline skeleton only
```

Mọi thứ khác chỉ được tồn tại nếu cần thiết để thực hiện các mục trên.

**Không optimize framework.**
**Không mở rộng research scope.**
**Không giữ legacy code.**
**Không giữ kết quả ngoài scope.**
**Không tạo fake result.**
**Không mất experiment history.**

Mục tiêu là một repository nhỏ, deterministic, reproducible và có thể chạy toàn bộ research experiment bằng CLI.
