Đúng. Nếu chốt hướng **Frozen RT-DETR + lightweight P2 head**, thì cách giải quyết sẽ đơn giản hơn đáng kể và loại bỏ luôn phần custom joint-training phức tạp.

Dưới đây là toàn bộ các lỗi hiện tại và cách sửa theo đúng hướng này.

---

# 1. P2 branch hiện tại **không thực sự được train**

### Bug

Trong `proposed.py` hiện tại về logic là:

```text
p2_model = RTDETRP2Model(...)
       ↓
rtdetr.train(...)
```

Nhưng `rtdetr.train()` chỉ train object RT-DETR gốc.

`p2_model` không nằm trong training graph của Ultralytics trainer.

Do đó:

```text
P2 Adapter
P2 Head
    ↓
không được optimizer update
```

Thậm chí sau training, code lại save:

```python
p2_model.state_dict()
```

nhưng P2 weights gần như vẫn là initialization ban đầu.

### Cách sửa

Với hướng mới:

> **Frozen RT-DETR + lightweight P2 head**

không cần custom joint trainer nữa.

Pipeline:

```text
Baseline RT-DETR-L checkpoint
          │
          │ freeze toàn bộ
          ▼
    RT-DETR-L backbone
          │
         C2
          │
          ▼
   Lightweight P2 Head
          │
          ▼
     P2 predictions
```

Chỉ optimize:

```text
P2 Adapter
+
P2 Head
```

Không optimize:

```text
RT-DETR backbone
RT-DETR encoder
RT-DETR decoder
RT-DETR native heads
```

Như vậy có thể dùng một training loop riêng cực kỳ đơn giản cho P2 head, không phải can thiệp vào native RT-DETR loss.

---

# 2. Không nên dùng `L_native + λL_P2` nữa

### Bug / thiết kế cũ

Code hiện tại có:

```text
L_total = L_native + λ L_P2
```

Nhưng implementation chưa thực sự kết nối được hai loss này.

Quan trọng hơn, với thiết kế Frozen RT-DETR thì **không cần native loss**.

### Cách sửa

Chỉ train:

```text
L = L_P2
```

RT-DETR được coi như một feature extractor + native detector frozen.

Điều này làm experiment rất sạch:

```text
RT-DETR-L baseline
       │
       ├── native prediction
       │
       └── C2 → P2 head → P2 prediction
                         │
                         ▼
                  concatenate + NMS
```

P2 head học cách detect tiny objects từ C2.

---

# 3. Memory của P2 Transformer hiện tại quá lớn

Đây là vấn đề bạn vừa chốt lại cách giải quyết.

### Bug

`P2QueryHead` hiện tại flatten toàn bộ P2 feature map:

```python
feat_proj = feat.flatten(2).permute(0, 2, 1)
```

Sau đó đưa toàn bộ spatial tokens vào Transformer:

```text
P2
↓
H/4 × W/4
↓
flatten
↓
~130K tokens @ 2K
~500K tokens @ 4K
↓
Transformer cross-attention
```

Đây là thiết kế quá nặng.

---

# 4. Cách sửa memory: **Frozen RT-DETR + Lightweight P2 Head**

Đây là thay đổi quan trọng nhất.

Không cần P2 Transformer 300 queries kiểu DETR nữa.

Thay bằng một **lightweight dense detection head**.

Ví dụ:

```text
C2
 │
 ▼
1×1 Conv
 │
 ▼
Lightweight Conv Block
 │
 ├───────────────┐
 ▼               ▼
Cls Head       Box Head
 │               │
 ▼               ▼
class logits    bbox
```

Có thể dùng kiến trúc rất đơn giản:

```text
C2
 ↓
Conv 1×1
 ↓
Conv 3×3
 ↓
Conv 3×3
 ↓
 ┌──────────────┐
 │              │
 ▼              ▼
Cls             Box
```

Không có:

* Transformer decoder
* 300 object queries
* Hungarian matching
* global attention
* full-resolution attention matrix

### Memory lúc này

P2 chỉ đi qua convolution:

```text
[B, C, H/4, W/4]
```

CNN complexity gần tuyến tính theo số pixel.

Ví dụ 2K:

```text
~480 × 272
≈ 130K spatial locations
```

130K locations với Conv hoàn toàn khả thi hơn rất nhiều so với đưa 130K tokens vào Transformer attention.

4K cũng tương tự.

---

# 5. Không cần P2 Head phải là DETR-style Query Head

Đây là một điểm nên sửa luôn.

Tên hiện tại:

```text
P2QueryHead
```

không còn phù hợp.

Nên chuyển thành:

```text
P2DenseHead
```

hoặc:

```text
LightweightP2Head
```

Mục tiêu của head:

```text
C2 → dense objectness/classification + bbox
```

Sau đó convert thành detection list:

```text
[x1, y1, x2, y2, score, class]
```

để đưa vào fusion.

---

# 6. Loss của P2 cũng phải đổi

Nếu bỏ Query Transformer + Hungarian matching thì:

```text
P2HeadLoss
```

hiện tại cũng không còn phù hợp.

Không cần:

```text
Hungarian matching
```

Thay bằng dense loss.

Có thể dùng:

```text
L_P2 =
    L_cls
  + λ_box L_box
  + λ_iou L_GIoU
```

Trong feasibility version, không cần phát minh loss mới.

Tốt nhất là sử dụng một loss đơn giản, ổn định và dễ giải thích.

Ví dụ:

```text
L_cls = BCE/Focal
L_box = L1
L_iou = GIoU
```

Điểm quan trọng của paper không nằm ở một loss mới.

Nó nằm ở:

> **bổ sung high-resolution P2 representation cho tiny pothole detection trong khi giữ nguyên RT-DETR detector và chỉ thêm một lightweight auxiliary branch.**

---

# 7. P2 không cần P3 → P2

Thiết kế hiện tại:

```text
C2 → P2
```

là đúng với hypothesis của experiment.

Không cần:

```text
C2 + Upsample(P3)
```

Bởi vì mục tiêu là kiểm tra:

> Liệu raw high-resolution C2 feature có chứa thông tin hữu ích cho ultra-fine potholes mà P3/P4/P5 đã mất không?

Do đó:

```text
C2
 ↓
P2
 ↓
Lightweight P2 Head
```

là thiết kế clean nhất.

---

# 8. RT-DETR phải được freeze hoàn toàn

Hiện tại code cần đảm bảo:

```python
for p in rtdetr.parameters():
    p.requires_grad = False
```

và:

```python
rtdetr.eval()
```

Trong training P2:

```text
RT-DETR
    ↓
no_grad()
    ↓
C2
```

sau đó:

```text
C2
 ↓
P2 head
 ↓
loss
 ↓
backward
```

Chỉ P2 parameters nhận gradient.

Có thể verify bằng:

```python
assert all(
    p.grad is None
    for p in rtdetr.parameters()
)
```

và:

```python
assert any(
    p.grad is not None
    for p in p2_head.parameters()
)
```

---

# 9. Proposed không nên khởi tạo từ `rtdetr-l.pt`

### Bug hiện tại

Registry đang kiểu:

```text
weights = "rtdetr-l.pt"
```

Đây là pretrained COCO checkpoint.

Nhưng experiment của bạn cần:

```text
HRP4K-trained RT-DETR-L
```

### Cách sửa

Pipeline phải là:

```text
HRP4K RT-DETR-L 2K baseline checkpoint
                │
                ▼
        Frozen RT-DETR-L
                │
                ▼
           C2 feature
                │
                ▼
       train Lightweight P2
```

Không train từ COCO nếu mục tiêu là đánh giá incremental improvement trên baseline HRP4K.

---

# 10. Dùng baseline 2K checkpoint

Đây là checkpoint hợp lý nhất cho feasibility.

Baseline hiện tại:

| Model     | Resolution |      AP50 |   AP50:95 |
| --------- | ---------: | --------: | --------: |
| RT-DETR-L |         2K | **62.65** | **37.51** |

Do đó:

```text
Baseline:
RT-DETR-L @ 2K
```

→ freeze

→ train P2 head

→ inference:

```text
Native RT-DETR predictions
+
P2 predictions
↓
NMS
```

Nếu Proposed vượt baseline thì mới mở rộng sang 4K.

---

# 11. Checkpoint của Proposed hiện tại cần sửa

Không nên chỉ save:

```python
p2_model.state_dict()
```

Nên save tối thiểu:

```python
{
    "p2_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "best_metric": ...,
    "base_checkpoint": ...,
    "image_size": 2048,
    "architecture": "frozen_rtdetr_l_p2",
}
```

Trong đó:

```text
base_checkpoint
```

phải xác định chính xác RT-DETR baseline nào được dùng.

---

# 12. Inference cũng cần sửa tương ứng

Pipeline inference nên là:

```text
Input image
      │
      ▼
Frozen RT-DETR-L
      │
      ├───────────────┐
      │               │
      ▼               ▼
Native P3/P4/P5     C2
      │               │
      ▼               ▼
Native decoder      P2 Head
      │               │
      ▼               ▼
 Native dets        P2 dets
      │               │
      └───────┬───────┘
              ▼
        Concatenate
              ▼
        Class-aware NMS
              ▼
         Final output
```

Đây vẫn giữ nguyên quyết định trước đó:

> **Fusion Version 0 = concatenate + NMS**

Không thêm:

* score boosting
* WBF
* learned fusion
* scale-aware weighting
* confidence calibration

ở feasibility experiment.

---

# 13. `np` đang thiếu import

Trong `proposed.py` đang dùng:

```python
np.ndarray
```

và các thao tác numpy nhưng thiếu:

```python
import numpy as np
```

### Fix

Thêm ngay đầu file:

```python
import numpy as np
```

Đây là bug runtime đơn giản nhưng chắc chắn phải sửa.

---

# 14. C2 hook hiện tại có thể giữ cho feasibility

Code hiện tại dùng forward hook để lấy C2.

Nó chưa phải kiến trúc đẹp nhất, nhưng:

> **không cần refactor ngay.**

Với feasibility:

```text
RT-DETR
 ↓
hook C2
 ↓
P2 Head
```

là đủ.

Sau khi chứng minh Proposed hoạt động, mới refactor thành native forward wrapper nếu cần.

---

# 15. Không cần tạo thêm metrics/protocol/threshold system

Điểm này **giữ nguyên như cũ**.

Không tạo:

```text
metrics.py
threshold.py
protocol.py
```

riêng cho Proposed.

Proposed phải dùng:

```text
same dataset
same test split
same image size
same confidence protocol
same COCO evaluator
same scale definitions
same reporting
```

để comparison:

```text
RT-DETR-L
vs
RT-DETR-L + P2
```

thực sự fair.

---

# 16. Không cần Stage 1 → Stage 2 nữa

Với thiết kế Frozen RT-DETR:

### Không cần:

```text
Stage 1:
freeze backbone / train detector

Stage 2:
unfreeze / joint training
```

Mà chỉ:

```text
Step 1
Load fully-trained HRP4K RT-DETR-L

Step 2
Freeze RT-DETR

Step 3
Attach lightweight P2 head

Step 4
Train P2 head

Step 5
Fuse predictions

Step 6
Evaluate
```

Đây là feasibility experiment rất sạch.

---

# Kiến trúc cuối cùng tôi khuyến nghị

```text
                 HRP4K RT-DETR-L
                  pretrained
                      │
                 ┌────┴────┐
                 │ FROZEN  │
                 └────┬────┘
                      │
          ┌───────────┴───────────┐
          │                       │
      P3/P4/P5                   C2
          │                       │
          ▼                       ▼
   Native RT-DETR          Lightweight P2
      Decoder                  Adapter
          │                       │
          ▼                       ▼
 Native Predictions          P2 Dense Head
          │                       │
          │                       ▼
          │                  P2 Predictions
          │                       │
          └───────────┬───────────┘
                      ▼
                Concatenate
                      ▼
                Class-aware NMS
                      ▼
                 Final Detections
```

Và training:

```text
                 Frozen RT-DETR
                      │
                     C2
                      │
                      ▼
              Lightweight P2 Head
                      │
                      ▼
                    L_P2
                      │
                      ▼
                   Backward
                      │
                      ▼
             ONLY P2 parameters
```

---

## Thứ tự sửa code

Tôi sẽ ưu tiên chính xác theo thứ tự này:

| Priority | Vấn đề                     | Cách sửa                                       |
| -------- | -------------------------- | ---------------------------------------------- |
| **P0**   | P2 không được train        | Tách training loop cho P2                      |
| **P0**   | Transformer P2 quá nặng    | Bỏ Query Transformer → Lightweight Dense P2    |
| **P0**   | Generic `rtdetr-l.pt`      | Load **HRP4K 2K baseline checkpoint**          |
| **P0**   | Joint loss không hoạt động | Bỏ `L_native`, chỉ train `L_P2`                |
| **P1**   | Checkpoint semantics       | Save P2 + optimizer + base checkpoint metadata |
| **P1**   | Inference                  | Native + P2 → concat → NMS                     |
| **P1**   | `np` missing               | `import numpy as np`                           |
| **P2**   | C2 hook                    | Giữ nguyên cho feasibility                     |
| **P2**   | Metrics/config/protocol    | **Không sửa**, reuse baseline                  |

### Và quan trọng nhất

Experiment đầu tiên **không cần 4K**.

Chạy:

```text
RT-DETR-L HRP4K @ 2K
       ↓
     frozen
       ↓
Lightweight P2 Head
       ↓
short training
       ↓
compare AP50 / AP50:95
       ↓
Ultra-Fine / Small / Medium / Large
```

Nếu kết quả cho thấy:

```text
Overall AP ↑
Ultra-Fine AP ↑
Small AP ↑
```

mà không tạo FP quá lớn, lúc đó mới đáng đầu tư tiếp vào **P2 architecture / ablation / 4K**.

Đây cũng là phiên bản có câu chuyện research sạch hơn rất nhiều: **không cố thay thế RT-DETR, mà kiểm tra liệu một high-resolution auxiliary detection path từ C2 có thể bổ sung thông tin scale cực nhỏ bị mất trong native P3/P4/P5 hay không.**
