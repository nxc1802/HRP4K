Tôi đã kiểm tra **commit mới nhất thực tế trên GitHub** của `nxc1802/HRP4K`.

Commit mới nhất là:

**`54b2ca3a925d5e7438dc5c5882581796034e455b` — `update`**, tạo lúc 05:02 UTC ngày 03/09/2026.

Đáng chú ý: commit `54b2ca3` hiện đang chứa thay đổi tiếp theo sau `263ec153`, nên đánh giá trước đó cần cập nhật.

## Kết luận nhanh

Kiến trúc đã đi **đúng hướng**:

> **Frozen RT-DETR + Lightweight Dense P2 Head**

và các lỗi nghiêm trọng nhất của kiến trúc cũ đã được xử lý:

* Không còn P2 Transformer → vấn đề attention memory lớn đã được loại bỏ.
* RT-DETR đã frozen.
* Optimizer chỉ nhận P2 branch/head.
* Checkpoint đã lưu P2 state + optimizer + metadata.
* Fusion vẫn là `native + P2 → concatenate → class-aware NMS`.

**Nhưng chưa nên chạy feasibility dài ngay. Có 4 vấn đề tôi đánh giá là cần sửa/verify trước.**

---

# P0 — 1. `extract_c2_backbone()` đang sai semantics của Ultralytics graph

Đây là **bug nghiêm trọng nhất hiện tại**.

`find_c2_backbone_stage()` đã hiểu rằng Ultralytics layer có `m.f`:

```text
f = -1
f = previous layer
f = [layer_a, layer_b]
```

nhưng `extract_c2_backbone()` lại chạy kiểu sequential:

```python
curr = x

for i in range(c2_layer_idx + 1):
    m = sub_modules[i]
    curr = m(curr)
```

Nghĩa là nó **bỏ qua connection graph `f`**.

### Vì sao nguy hiểm?

Nếu C2 nằm sau layer có skip/concat:

```text
layer 0 ──┐
          ├── layer 3
layer 2 ──┘
```

thì code hiện tại truyền:

```text
output layer 2
      ↓
layer 3
```

thay vì:

```text
output layer 0 + output layer 2
             ↓
          layer 3
```

→ C2 feature có thể **hoàn toàn sai**.

### Fix

`extract_c2_backbone()` phải reproduce semantics của `f`:

```python
y = []

for i, m in enumerate(sub_modules[:c2_layer_idx + 1]):
    f = getattr(m, "f", -1)

    if f != -1:
        if isinstance(f, int):
            curr = y[f]
        else:
            curr = [
                curr if j == -1 else y[j]
                for j in f
            ]

    curr = m(curr)
    y.append(curr if i in save_indices else None)

return curr
```

Đây là **P0 — phải sửa trước khi train**.

---

# P0 — 2. Logic tìm C2 chưa deterministic

Hiện tại:

```python
if stride_h == 4 and stride_w == 4 and i < 6:
    c2_idx = i
```

nhưng không `break`.

Nếu có nhiều feature map stride 4, code có thể lấy **layer stride-4 cuối cùng** thỏa điều kiện thay vì feature đầu tiên.

### Fix tối thiểu

```python
if stride_h == 4 and stride_w == 4:
    c2_idx = i
    c2_channels = x.shape[1]
    break
```

Tốt hơn nữa là xác định:

> **first semantic feature map at stride 4**

thay vì hard-code `i < 6`.

### Tôi đánh giá

Đây là P0 vì nếu chọn nhầm feature thì toàn bộ Proposed experiment mất ý nghĩa.

---

# P0/P1 — 3. Dense loss đang assign quá nhiều positive locations

Đây là vấn đề mới đáng chú ý.

Hiện tại `DenseP2Loss` đang coi **toàn bộ grid nằm bên trong GT box** là positive.

Ví dụ:

```text
GT pothole
┌─────────────────────┐
│ + + + + + + + + +   │
│ + + + + + + + + +   │
│ + + + + + + + + +   │
└─────────────────────┘
```

Một pothole lớn có thể tạo ra hàng chục/hàng trăm positive locations, trong khi pothole tiny chỉ có khoảng một cell.

Nghiêm trọng hơn, nếu hai GT overlap, target bbox của GT sau có thể overwrite target của GT trước.

### Cách tôi khuyên sửa

**Không cần phát minh loss mới.**

Với feasibility, dùng:

```text
GT center
    ↓
nearest P2 grid cell
    ↓
1 positive / object
```

Sau đó:

```text
L_P2 =
    L_cls
  + λ1 L_L1
  + λ2 L_GIoU
```

Điều này có hai lợi ích:

1. Một object → một positive assignment.
2. P2 branch thực sự tập trung vào localization tiny object.

Đặc biệt phù hợp với hypothesis của bạn.

---

# P1 — 4. P2 đã bỏ Transformer, nhưng vẫn hơi nặng

Hiện tại P2 path đại khái:

```text
C2
 ↓
1×1 Conv → 256
 ↓
3×3 Conv → 256
 ↓
2 × 3×3 Conv classification
+
2 × 3×3 Conv bbox
```

Ở 2K:

```text
~480 × 272
≈ 130K spatial locations
```

Nên nó không còn là **memory explosion** như Transformer nữa, nhưng vẫn tương đối nhiều computation ở stride 4.

### Tuy nhiên: chưa sửa

Tôi **không khuyên giảm 256 → 128 ngay**.

Giữ:

```text
256 channels
```

cho feasibility đầu tiên.

Nếu chạy được thì tốt.

Nếu profiling cho thấy bottleneck mới thử:

```text
256 → 128
```

Đây là **optimization**, không phải correctness bug.

---

# P1 — 5. Decode bbox cần kiểm tra numerical stability

Hiện tại bbox được sinh từ offsets.

Cần đảm bảo:

```text
x1 < x2
y1 < y2
```

và không xuất hiện bbox âm/vô nghĩa.

Commit hiện tại đã có logic offset tương đối hợp lý, nhưng vẫn nên test các trường hợp:

```text
zero feature
large activation
very small object
object touching image boundary
```

Nếu cần có thể enforce:

```python
x1 = min(...)
x2 = max(...)
```

hoặc clamp width/height. Đây chưa phải blocker lớn.

---

# P1 — 6. `resume=True` chưa hoàn chỉnh

Checkpoint hiện tại đã tốt hơn trước:

```text
p2_state_dict
optimizer_state_dict
epoch
mean_p2_loss
base_checkpoint
image_size
architecture
```

Đây là đúng hướng.

Nhưng `resume` chưa thực sự restore đầy đủ:

```text
checkpoint
 ↓
P2 weights
 ↓
optimizer
 ↓
epoch
 ↓
continue training
```

### Không ảnh hưởng feasibility lần đầu

Nếu chạy fresh training:

```text
resume=False
```

thì có thể bỏ qua.

Sửa sau.

---

# P1 — 7. Proposed inference đang square-resize

Đây là vấn đề **fairness** khá quan trọng.

Code hiện tại có logic:

```python
cv2.resize(image, (image_size, image_size))
```

Ví dụ:

```text
3840 × 2160
      ↓
2048 × 2048
```

→ distortion hình học và thay đổi scale distribution.

Trong khi baseline RT-DETR dùng preprocessing của Ultralytics.

### Cần sửa

P2 phải nhận **chính tensor/preprocessing mà native RT-DETR nhận**.

Không được để:

```text
Native RT-DETR:
Ultralytics preprocessing

P2:
custom square resize
```

Mà phải:

```text
                    Input
                      │
                SAME preprocessing
                      │
               ┌──────┴──────┐
               ↓             ↓
          RT-DETR         C2 extraction
               │             │
               ↓             ↓
          Native det      P2 Head
```

Nếu không, comparison sẽ không còn hoàn toàn fair.

---

# P1 — 8. Training DataLoader chưa chắc reproduce baseline protocol

Commit mới đang tự build:

```text
build_yolo_dataset()
build_dataloader()
```

thay vì trực tiếp reuse pipeline của baseline runner.

Có nguy cơ khác:

* augmentation
* normalization
* rect
* mosaic
* scale
* flip
* collate
* padding

### Nhưng nuance

Vì RT-DETR **frozen**, chúng ta không cần augmentation cho native detector.

Tuy nhiên P2 vẫn phải train trên distribution hợp lý.

### Tôi khuyên

Reuse càng nhiều càng tốt:

```text
dataset
split
image preprocessing
normalization
augmentation config
```

của baseline.

---

# Những phần hiện tại tôi đánh giá ĐÃ ĐÚNG

### Frozen RT-DETR

Đúng:

```text
RT-DETR parameters
       ↓
requires_grad=False

P2 Adapter + P2 Head
       ↓
optimizer
```

Đúng với thiết kế đã thống nhất.

### P2 không dùng P3

**Giữ nguyên.**

```text
C2 → P2
```

Không cần:

```text
C2 + Upsample(P3)
```

### Không dùng Transformer

**Giữ nguyên.**

Đây chính là cách giải quyết memory mà ta đã thống nhất.

### Fusion

Đúng:

```text
Native detections
       +
P2 detections
       ↓
concatenate
       ↓
class-aware NMS
```

Không có score boosting, WBF, learned fusion hay scale gating.

---

# Verdict của tôi với commit mới nhất

| Thành phần                       | Trạng thái  |
| -------------------------------- | ----------- |
| Frozen RT-DETR                   | ✅           |
| Lightweight Dense P2             | ✅           |
| Bỏ Transformer memory bottleneck | ✅           |
| P2 optimizer riêng               | ✅           |
| Checkpoint cơ bản                | ✅           |
| Native + P2 + NMS                | ✅           |
| **C2 extraction semantics**      | ❌ **P0**    |
| **C2 discovery**                 | ❌ **P0**    |
| **Dense positive assignment**    | ❌ **P0/P1** |
| Bbox numerical stability         | ⚠️          |
| Resume                           | ⚠️          |
| Preprocessing fairness           | ⚠️ **P1**   |
| DataLoader parity                | ⚠️ **P1**   |

## Tôi sẽ **chưa chạy training 2K dài**.

Chỉ cần sửa/verify theo thứ tự:

```text
1. Fix C2 graph extraction
          ↓
2. Fix deterministic C2 selection
          ↓
3. Fix 1-GT → 1-center-cell assignment
          ↓
4. Verify same preprocessing
          ↓
5. 1-batch / 1-image overfit
          ↓
6. Verify P2 loss ↓
7. Verify P2 grad != None
8. Verify RT-DETR grad == None
          ↓
9. Short 2K feasibility
```

**Quan trọng:** commit `54b2ca3` là commit mới nhất, nhưng commit message chỉ là `update`; phần diff cho thấy nó đang là bước tiếp theo trên nền `263ec153`, và các vấn đề P0 ở trên vẫn còn.

Nếu mục tiêu hiện tại là **“sửa xong để có thể chạy feasibility”**, tôi sẽ chỉ sửa **3 P0 đầu tiên + preprocessing**, chưa động vào architecture 256-channel hay các optimization khác.
