Tôi đã kiểm tra **commit mới nhất** của `nxc1802/HRP4K`.

Commit mới nhất hiện tại là:

**`263ec153e52fc132ded4eb47656539935f11964f`**
`feat(proposed): implement Frozen RT-DETR + Lightweight Dense P2 Head feasibility architecture`

So với commit trước `4e88b4c`, hướng kiến trúc đã được đổi đúng sang **Frozen RT-DETR + Lightweight Dense P2**. Tuy nhiên, tôi vẫn thấy **một số vấn đề cần sửa trước khi chạy feasibility**.

---

# 1. P0 — `extract_c2_backbone()` đang bỏ qua graph connection `f`

Đây là lỗi quan trọng nhất tôi phát hiện trong commit mới.

Trong `find_c2_backbone_stage()`, code đã xử lý đúng Ultralytics graph:

```python
f = getattr(m, "f", -1)

if f != -1:
    ...
```

Nhưng `extract_c2_backbone()` lại làm:

```python
curr = x
for i in range(c2_layer_idx + 1):
    m = sub_modules[i]
    curr = m(curr)
```

Tức là nó **giả định toàn bộ backbone là sequential thuần túy**.

Trong Ultralytics, các layer có thể có:

```text
f = -1
f = previous_layer
f = [layer_a, layer_b]
```

Do đó discovery và extraction hiện tại **không sử dụng cùng một execution semantics**.

### Sửa

`extract_c2_backbone()` phải mirror logic của `find_c2_backbone_stage()`:

```python
def extract_c2_backbone(
    model: nn.Module,
    x: torch.Tensor,
    c2_layer_idx: int,
) -> torch.Tensor:
    _, sub_modules, save_indices = _unwrap_sequential(model)

    curr = x
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

**Đây là P0.**

---

# 2. P0 — `find_c2_backbone_stage()` có nguy cơ chọn sai C2

Code hiện tại:

```python
if stride_h == 4 and stride_w == 4 and i < 6:
    c2_idx = i
    c2_channels = x.shape[1]
```

Nó **không break khi tìm thấy C2**.

Nếu có nhiều layer stride-4 trước layer 6 thì layer cuối cùng thỏa điều kiện sẽ được chọn.

Với một backbone cụ thể có thể vẫn đúng, nhưng discovery logic nên deterministic hơn:

```python
if stride_h == 4 and stride_w == 4:
    c2_idx = i
    c2_channels = x.shape[1]
    break
```

Hoặc tốt hơn nữa: xác định **first semantic feature stage at stride 4**, thay vì chỉ dựa vào `i < 6`.

### Tôi khuyến nghị

Trong feasibility:

```text
first feature map with stride=4
```

là đủ.

---

# 3. P0 — `P2Adapter` thực tế vẫn không phải "lightweight" nếu giữ 256 channels

Commit mới đã loại bỏ Transformer nên memory đã được giải quyết về mặt **complexity**.

`P2Adapter`:

```text
C2
 ↓
1×1 Conv
 ↓
256 channels
 ↓
3×3 Conv
 ↓
256 channels
```

và `LightweightP2Head` lại có:

```text
2 × 3×3 Conv
```

cho classification

và thêm:

```text
2 × 3×3 Conv
```

cho bbox.

Tức là tổng cộng P2 path đang có khá nhiều convolution ở **stride 4 resolution**.

Ở 2K:

```text
~480 × 272 ≈ 130K locations
```

Không còn nguy hiểm như Transformer, nhưng vẫn khá nặng.

### Không phải bug

Đây là **optimization concern**, chưa cần sửa ngay.

Tôi sẽ giữ 256 trước để feasibility có đủ capacity.

Nếu latency/memory quá lớn mới thử:

```text
256 → 128
```

---

# 4. P0 — Loss hiện tại có vấn đề về duplicate positive assignment

`DenseP2Loss` đang assign **toàn bộ grid points nằm trong bounding box** là positive:

```text
GT box
┌─────────────────────┐
│ + + + + + + + + +   │
│ + + + + + + + + +   │
│ + + + + + + + + +   │
└─────────────────────┘
```

Điều này có thể tạo ra **rất nhiều positive locations cho một pothole**.

Đặc biệt với pothole lớn:

```text
one GT → dozens/hundreds of positive cells
```

Trong khi tiny pothole:

```text
one GT → 1 cell
```

Do đó training distribution giữa object size bị lệch mạnh.

### Đáng chú ý hơn

Nếu hai GT boxes overlap, code:

```python
target_cls[b, c_idx, inside] = 1.0
target_boxes_ltrb[b, :, inside] = ...
```

GT sau có thể **ghi đè target box của GT trước**.

Đây là vấn đề thực sự.

### Cách sửa cho feasibility

Không assign toàn bộ box.

Chỉ assign:

> **center region / center point của mỗi GT**

Ví dụ đơn giản nhất:

```text
GT center
   ↓
nearest P2 grid cell
   ↓
1 positive location / object
```

Điều này cũng rất phù hợp với mục tiêu tiny pothole.

Tôi đánh giá đây là **P0/P1**, nên sửa trước training dài.

---

# 5. P0 — `box_offsets` chưa được bảo vệ đủ về numerical stability

Hiện tại:

```python
box_offsets = self.box_conv(p2_feat) * self.stride
```

với:

```python
Conv → ReLU
```

nên offsets ≥ 0.

Điều này hợp lý.

Nhưng prediction có thể tạo:

```text
x1 > x2
```

hoặc:

```text
y1 > y2
```

nếu offset nhỏ/không hợp lệ.

Nên decode có thể enforce:

```python
x1 = min(x1, x2)
x2 = max(x1, x2)
```

hoặc clamp width/height.

Đây không phải blocker lớn vì training GIoU có thể giúp, nhưng nên kiểm tra.

---

# 6. P1 — Frozen RT-DETR đã được thực hiện đúng về mặt optimizer

Đây là phần **đã sửa tốt**.

Commit mới:

```python
p2_params = (
    list(p2_model.p2_branch.parameters())
    + list(p2_model.p2_head.parameters())
)

optimizer = torch.optim.AdamW(p2_params, ...)
```

và native model:

```python
for p in self.native_model.parameters():
    p.requires_grad = False
```

Do đó:

```text
RT-DETR parameters
        ↓
frozen

P2 adapter
P2 head
        ↓
trainable
```

Đúng với thiết kế mới.

---

# 7. P1 — Nhưng `RTDETRP2Model` có một điểm cần kiểm tra

Trong training:

```python
with torch.no_grad():
    c2_feat = extract_c2_backbone(
        p2_model.native_model,
        img,
        c2_layer_idx=p2_model.c2_layer_idx
    )
```

Điều này đúng.

Nhưng trong `RTDETRP2Model.__init__`:

```python
self.native_model.eval()
```

Sau đó training loop:

```python
p2_model.p2_branch.train()
p2_model.p2_head.train()
```

Không gọi:

```python
p2_model.train()
```

Đây thực ra lại khá tốt vì tránh việc `.train()` lan xuống native model.

Tuy nhiên cần đặc biệt đảm bảo **P2 BatchNorm được train**.

Hiện tại:

```text
P2Adapter
 ├── BN
 └── BN

P2Head
 ├── BN
 ├── BN
 ├── BN
 └── BN
```

đang được `.train()` nên ổn.

---

# 8. P1 — Checkpoint hiện tại đã tốt hơn nhiều

Commit mới save:

```text
p2_state_dict
p2_adapter_state_dict
optimizer_state_dict
epoch
mean_p2_loss
base_checkpoint
image_size
architecture
```

Đây là đúng hướng.

Đặc biệt:

```text
base_checkpoint
```

giúp biết P2 được train trên baseline nào.

---

# 9. P1 — Nhưng `resume=True` hiện chưa thực sự được implement đầy đủ

Hàm có:

```python
resume: bool = False
```

nhưng logic training tôi thấy trong commit mới vẫn chủ yếu:

```text
load baseline
initialize P2
create optimizer
train
```

Chưa thấy flow đầy đủ:

```text
load best_p2.pt
        ↓
restore P2
        ↓
restore optimizer
        ↓
restore epoch
        ↓
continue
```

Vì vậy:

> `resume` hiện tại chưa nên được coi là production-ready.

Không ảnh hưởng feasibility lần đầu, nên **P1**.

---

# 10. P1 — Native prediction và P2 prediction đang được resize về square

Trong adapter:

```python
resized = cv2.resize(
    image,
    (image_size, image_size)
)
```

Đây là điểm tôi **không muốn giữ** nếu experiment chính dùng pipeline baseline với aspect ratio / rect preprocessing.

Ví dụ ảnh HRP4K:

```text
3840 × 2160
```

bị biến thành:

```text
2048 × 2048
```

thay vì letterbox/Ultralytics preprocessing.

Điều này có thể làm:

* distortion
* thay đổi object geometry
* scale distribution khác baseline

và khiến comparison không hoàn toàn fair.

### Cần sửa

Proposed inference phải sử dụng **chính preprocessing của baseline RT-DETR**.

Đây là một điểm khá quan trọng cho paper.

---

# 11. P1 — Training DataLoader đang không chắc đã reproduce baseline augmentation

Commit mới tự tạo:

```python
build_yolo_dataset(...)
build_dataloader(...)
```

thay vì reuse toàn bộ training pipeline của `runner.py`.

Điều này có nghĩa:

```text
Baseline:
Ultralytics RT-DETR training protocol

Proposed:
custom P2 dataloader
```

có thể khác:

* augmentation
* preprocessing
* normalization
* rect behavior
* mosaic
* scale
* hsv
* flip
* collate

Đây là vấn đề **fairness**.

### Nhưng có một nuance quan trọng

Vì RT-DETR frozen nên bạn **không cần replicate augmentation 100%** để native model học.

Nhưng P2 phải học trên distribution tương đương baseline.

Do đó tốt nhất là reuse dataset configuration/protocol của baseline càng nhiều càng tốt.

---

# 12. P1 — `batch_data["img"]` có thể không cùng preprocessing semantics với RT-DETR inference

Training:

```python
img = batch_data["img"].float() / 255.0
```

Nếu Ultralytics dataloader đã trả image tensor đúng `[0,255]`, điều này ổn.

Nhưng cần verify:

```text
dtype
range
H/W
rect
padding
```

trên server bằng một smoke run.

Không nên assume.

---

# 13. P2 — Fusion implementation ổn

`p2_fusion.py` hiện thực đúng:

```text
Native detections
       +
P2 detections
       ↓
concatenate
       ↓
class-aware NMS
```

Không có:

* WBF
* score boost
* learned fusion
* scale gating

Đúng với Version 0 đã thống nhất.

---

# 14. P2 — `decode_dense_p2_predictions()` có một vấn đề nhỏ

Nó lấy:

```python
b_scores, b_classes = scores[b].max(dim=0)
```

tức là:

```text
mỗi spatial location
→ chỉ giữ class có score cao nhất
```

Với HRP4K:

```text
nc = 1
```

nên **không có vấn đề**.

Có thể giữ nguyên.

---

# 15. Tổng kết trạng thái commit mới

Tôi đánh giá commit mới **đã giải quyết được 2 blocker lớn nhất của commit cũ**:

### Đã fix

```text
❌ P2 không được train
        ↓
✅ Dedicated P2 training loop
```

và:

```text
❌ Transformer với 130K–500K tokens
        ↓
✅ Lightweight dense CNN head
```

Đồng thời:

```text
❌ Joint native + P2 training
        ↓
✅ Frozen native RT-DETR + L_P2
```

Đây là một thay đổi **đúng hướng**.

---

## Nhưng trước khi chạy server, tôi sẽ sửa 3 thứ này trước

### **P0-1 — Fix C2 extraction**

Đồng bộ `extract_c2_backbone()` với Ultralytics `f` graph.

### **P0-2 — Fix positive assignment**

Không dùng toàn bộ bbox làm positive.

Dùng **center/nearest-grid assignment** để mỗi GT có positive anchor/location rõ ràng.

### **P0-3 — Verify preprocessing**

Đảm bảo:

```text
training P2
       ↕
baseline RT-DETR inference
```

dùng cùng geometry/preprocessing semantics.

---

### Sau đó mới chạy smoke test

Không train 150 epochs ngay.

Test:

```text
1 batch
↓
C2 shape
↓
P2 shape
↓
cls shape
↓
box shape
↓
loss finite
↓
backward
↓
P2 grad != None
↓
RT-DETR grad == None
↓
optimizer.step()
```

Nếu pass toàn bộ thì mới chạy **1 epoch/short feasibility @ 2K**.

Commit hiện tại đã ở mức **gần có thể chạy feasibility**, nhưng tôi **chưa khuyến nghị server train dài ngay** vì `extract_c2_backbone()` và target assignment vẫn có thể làm kết quả sai hoặc training behavior không đúng.
