Có, nhưng mức độ “có code” rất khác nhau. **Không cần reimplement toàn bộ các SOTA method từ paper**, nhưng cũng không thể clone tất cả rồi thay dataset HRP4K là chạy được. Vấn đề lớn nhất là nhiều repo dùng framework rất cũ hoặc gắn chặt với detector gốc.

| Method              | Code hiện có                               | Với HRP4K nên làm gì?          | Mức công sức      |
| ------------------- | ------------------------------------------ | ------------------------------ | ----------------- |
| Resize-only         | Không cần repo                             | Tự implement                   | 🟢 Rất thấp       |
| Uniform tiling      | Không cần repo                             | Tự implement crop/remap/NMS    | 🟢 Thấp           |
| **SAHI**            | ✅ Official, maintained                     | Dùng trực tiếp                 | 🟢 Thấp           |
| **AutoFocus**       | ✅ Official                                 | Port/adapt khá nhiều           | 🔴 Cao            |
| **AdaZoom**         | ⚠️ Chưa xác minh được official public code | Khả năng phải reimplement      | 🔴 Rất cao        |
| **FOVEA**           | ✅ Official                                 | Dùng code làm reference + port | 🟠 Trung bình-cao |
| **Two-Plane Prior** | ✅ Official minimal implementation          | Dùng code + adapt HRP4K        | 🟠 Trung bình     |
| **ZoomDet**         | ✅ Official, cả Faster R-CNN và YOLO        | Adapt từ code chính thức       | 🟡 Trung bình     |

### SAHI — gần như không phải reimplement

SAHI là trường hợp dễ nhất. Repo `obss/sahi` vẫn được maintain và framework hỗ trợ sliced inference với Ultralytics cùng nhiều detector khác. Có thể cài package rồi đưa checkpoint HRP4K vào. ([GitHub][1])

Với project của chúng ta, phần cần tự viết chủ yếu là wrapper:

```text
HRP4K image
 → SAHI
 → YOLO11 checkpoint
 → prediction
 → COCO JSON
 → unified evaluator
```

Tức **giữ nguyên SAHI**, chỉ làm integration.

---

### AutoFocus — có official code, nhưng thực tế gần như phải port

AutoFocus có code chính thức trong repo **SNIPER/AutoFocus** của tác giả. Repo thậm chí có config train FocusPixel branch, checkpoint AutoFocus và pipeline evaluation. ([GitHub][2])

Vấn đề là implementation dựa trên stack cũ:

```text
MXNet
custom MXNet fork
CUDA/C++ code
SNIPER
ResNet-101 style detector
```

và repo yêu cầu compile MXNet fork cùng các extension C++. ([GitHub][2])

Do đó tôi **không khuyên kéo nguyên AutoFocus stack vào HRP4K benchmark hiện đại**.

Nên lấy algorithm/reference code chính thức rồi port ba thành phần:

```text
FocusPixel prediction
       ↓
FocusChip generation
       ↓
coarse → fine inference + fusion
```

sang PyTorch.

Tức đây là **port/reimplementation có đối chiếu official code**, chứ không phải viết lại mù từ paper.

---

### FOVEA — có official code khá đầy đủ

FOVEA có official repository của tác giả, bao gồm source, configs, scripts và cả pretrained checkpoints. ([GitHub][3])

Nhưng environment gốc là:

```text
Python 3.8.5
PyTorch 1.6.0
torchvision 0.7
MMDetection 2.7
MMCV 1.1.5
CUDA 10.2
```

([GitHub][3])

Vì vậy có hai lựa chọn.

**Reproduction track:** chạy repo gốc trong một Docker riêng, đổi dataset sang HRP4K.

**Controlled benchmark track:** lấy phần quan trọng:

```text
saliency generator
KDE mapping
foveated resampler
inverse bbox mapping
```

rồi port sang PyTorch hiện đại để dùng cùng detector của benchmark.

Tôi thiên về cách thứ hai cho final benchmark, nhưng **code chính thức là reference**, nên rủi ro thấp hơn nhiều so với reimplement từ paper.

---

### Two-Plane Perspective Prior — có official minimal implementation

Đây cũng là tin tốt. Tác giả release repo `geometriczoom/two-plane-prior`, tự mô tả là **minimal implementation** của paper CVPR 2023, có source `tpp`, config, experiment script và checkpoint. ([GitHub][4])

Nó được xây một phần dựa trên FOVEA và dùng:

```text
Python 3.8.5
PyTorch 1.6
MMDetection 2.20
MMCV 1.3.17
Kornia 0.5.11
```

([GitHub][4])

Do đó:

> **Không reimplement TPP từ đầu.**

Clone official code → reproduce → port resampling/prior module sang benchmark stack.

Phần riêng cần làm cho HRP4K chủ yếu là **perspective/vanishing-point adaptation**, vì repo có sẵn dữ liệu vanishing point cho setting gốc nhưng HRP4K không đi kèm annotation tương đương. ([GitHub][4])

---

### ZoomDet — tình hình tốt nhất trong nhóm adaptive learned methods

ZoomDet 2026 đã release official code. Paper chính thức cũng chỉ rõ repository. ([ScienceDirect][5])

Đặc biệt có **hai implementation**:

```text
zoomdet_code
→ Faster R-CNN / MMDetection

zoomdet_yolo
→ YOLOv8 / MMYOLO
```

Repo chính thức xác nhận bản Faster R-CNN và liên kết trực tiếp sang bản YOLO. ([GitHub][6]) Bản YOLO sử dụng YOLOv8 làm base architecture và có sẵn config/train/test cho VisDrone, UAVDT và SeaDroneSee. ([GitHub][7])

Đây là lợi thế lớn cho HRP4K.

Ta có thể:

```text
ZoomDet-YOLOv8 official
        ↓
verify reproduction
        ↓
HRP4K
```

trước.

Nếu sau đó muốn controlled experiment:

```text
Resize + YOLO11n
SAHI + YOLO11n
ZoomDet + YOLO11n
```

thì mới port ZoomDet module từ YOLOv8/MMYOLO sang YOLO11.

**Không cần reverse-engineer method từ paper.**

---

### AdaZoom — đây mới là vấn đề lớn

AdaZoom paper mô tả method khá đầy đủ: PolicyNet ba nhánh cho fixation/scale/aspect ratio, policy-gradient RL, history map, object-scale reward và collaborative training với detector. ([arXiv][8])

Tuy nhiên, sau khi kiểm tra arXiv, trang publication của nhóm tác giả và tìm repository, tôi **chưa xác minh được một official author-released public implementation**. ArXiv paper cũng không dẫn tới code repository. ([arXiv][8])

Vì vậy hiện tại nên coi:

> **AdaZoom = reimplementation required**, trừ khi liên hệ tác giả hoặc tìm được code release khác có thể xác minh.

Và đây là method tốn công nhất vì không chỉ có một module zoom:

```text
backbone
 ↓
PolicyNet
 ├ fixation
 ├ scale
 └ aspect ratio
 ↓
sequential state/history
 ↓
policy-gradient reward
 ↓
adaptive crops
 ↓
detector
 ↓
collaborative retraining
```

Paper còn thay đổi reward trong collaborative training dựa trên confidence/error của detector. ([arXiv][8])

Tôi sẽ **không đưa AdaZoom lên đầu danh sách implementation**.

---

## Vậy thực tế phải viết bao nhiêu?

Không nên nghĩ:

```text
8 methods
→ implement 8 algorithms
```

Thực tế gần hơn với:

```text
Resize            → tự viết ~100%
Uniform           → tự viết ~100%
SAHI              → wrapper ~90%, algorithm 0%

AutoFocus         → port core algorithm
FOVEA             → port official implementation
Two-Plane Prior   → port official implementation
ZoomDet           → adapt official implementation

AdaZoom           → reimplement gần như toàn bộ
```

Trong đó Resize/Uniform rất đơn giản.

---

## Tôi sẽ thay đổi kế hoạch Phase 2 một chút

Thay vì ép toàn bộ method ngay lập tức vào **một codebase YOLO11**, nên làm hai stage.

**Stage A — Reproduction/native implementation**

```text
SAHI           → official SAHI
AutoFocus      → official SNIPER/AutoFocus
FOVEA          → official FOVEA
TPP            → official TPP
ZoomDet        → official ZoomDet-YOLO
AdaZoom        → reimplementation
```

Mục tiêu là xác nhận:

> implementation hoạt động đúng trước khi sửa architecture.

Sau đó **Stage B — HRP4K standardized benchmark**:

```text
                 Common HRP4K data
                        ↓
              Method-specific processor
                        ↓
                 common prediction
                        ↓
                    COCO JSON
                        ↓
                unified evaluator
```

Không nhất thiết ép source code nội bộ giống nhau.

---

## Thậm chí nên dùng container riêng

Vì dependency khác nhau rất mạnh:

```text
containers/
├── sahi/
├── autofocus/
├── fovea/
├── tpp/
├── zoomdet/
└── adazoom/
```

Ví dụ FOVEA cần PyTorch 1.6/MMDet 2.7, còn TPP dùng MMDet 2.20. ([GitHub][3]) AutoFocus lại là MXNet custom fork. ([GitHub][2])

Không nên cố:

> “upgrade tất cả lên PyTorch 2.x trước rồi mới benchmark.”

Việc đó vừa tốn thời gian vừa dễ vô tình thay đổi behavior của method.

---

## Thứ tự implementation sau khi biết tình trạng code

Tôi sẽ sửa roadmap thành:

```text
1. Resize-only
2. Uniform tiling
3. SAHI
      ↓
   easy baselines complete

4. ZoomDet
      ↓
   modern official adaptive method

5. Two-Plane Prior
6. FOVEA
      ↓
   official adaptive-resampling methods

7. AutoFocus
      ↓
   legacy port

8. AdaZoom
      ↓
   full reproduction/reimplementation
```

Điểm quan trọng là **5/6 method không-trivial trong danh sách đã có official code hoặc framework có thể dùng làm ground truth implementation**: SAHI, AutoFocus, FOVEA, Two-Plane Prior và ZoomDet. AdaZoom là trường hợp tôi sẽ dự trù phải reimplement. Vì vậy scope Phase 2 vẫn hoàn toàn khả thi; phần nặng nhất không phải “viết lại các SOTA paper”, mà là **chuẩn hóa dataset interface, dependency/container, detector adaptation và unified evaluation**.

[1]: https://github.com/obss/SAHI?utm_source=chatgpt.com "GitHub - obss/sahi: Framework agnostic sliced/tiled inference + interactive ui + error analysis plots · GitHub"
[2]: https://github.com/MahyarNajibi/SNIPER "GitHub - mahyarnajibi/SNIPER: SNIPER / AutoFocus is an efficient multi-scale object detection training / inference algorithm · GitHub"
[3]: https://github.com/tchittesh/fovea "GitHub - tchittesh/fovea: Code for FOVEA: Foveated Image Magnification for Autonomous Navigation (ICCV 2021) · GitHub"
[4]: https://github.com/geometriczoom/two-plane-prior "GitHub - geometriczoom/two-plane-prior: Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection · GitHub"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0924271626000444?utm_source=chatgpt.com "Adaptive image zoom-in with bounding box transformation for UAV object detection - ScienceDirect"
[6]: https://github.com/twangnh/zoomdet_code "GitHub - twangnh/zoomdet_code: implementation of paper submitted to ISPRS Journal of Photogrammetry and Remote Sensing · GitHub"
[7]: https://github.com/twangnh/zoomdet_yolo "GitHub - twangnh/zoomdet_yolo · GitHub"
[8]: https://arxiv.org/abs/2106.10409 "AdaZoom: Adaptive Zoom Network for Multi-Scale Object Detection in Large Scenes"
