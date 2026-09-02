# Plan Proposed Method — Feasibility Test

## 1. Architecture

Giữ **native RT-DETR-L nguyên bản**:

```text
Backbone
├── C2 ──► P2 Adapter ──► P2 Auxiliary Detector Head
│
├── P3 ──► Native RT-DETR
├── P4 ──► Native RT-DETR
└── P5 ──► Native RT-DETR
```

### P2

```text
C2 → 1×1 Conv → 3×3 Conv → P2
```

* Không đưa P3 vào P2.
* Không hard-code channel/shape.
* Channel và spatial shape phải lấy từ **model runtime**.

### P2 Head

Dùng **P2 auxiliary detector**.

* Query-based detection head.
* Có classification + bounding box prediction.
* P2 branch có loss riêng để train.
* Native RT-DETR decoder/loss giữ nguyên.

---

## 2. Prediction Fusion

```text
Native RT-DETR predictions
             +
P2 predictions
             ↓
       concatenate
             ↓
           NMS
             ↓
        final output
```

Không:

* score boosting
* WBF
* scale-aware weighting
* learned fusion
* hard routing theo object size

Mục tiêu là kiểm tra **P2 itself** trước.

---

## 3. Training

Dùng:

```text
Baseline RT-DETR-L checkpoint
          ↓
Load pretrained weights
          ↓
Attach randomly initialized P2 branch
          ↓
Fine-tune entire model
          ↓
Native RT-DETR loss + P2 auxiliary loss
```

Tức là:

> **Baseline HF checkpoint là initialization cho toàn bộ native detector; chỉ P2 branch bắt đầu từ random initialization.**

Không freeze native RT-DETR trong feasibility test.

### Loss

```text
L_total = L_native_RTDETR + λ L_P2 (với λ = 0.25)
```

Các training settings khác:

> **copy đúng baseline hiện tại.**

Không tạo một protocol mới.

---

# 4. Exact Runtime Shape Check

Thực hiện **local trước khi train server**.

Mục tiêu:

```text
model runtime
    ↓
identify C2
    ↓
verify C2 shape
    ↓
verify native P3/P4/P5
    ↓
construct P2 adapter
    ↓
forward test
```

Không hard-code kiểu:

```python
C2 = 128
P3 = 256
```

hay phụ thuộc cứng vào layer index nếu runtime inspection có thể xác định được.

Sau khi local forward pass + shape check + gradient check OK → commit → server CLI training.

---

# 5. Server Training

CLI chỉ cần chạy experiment mới dựa trên baseline training pipeline hiện có.

Không xây:

* threshold framework mới
* metrics framework mới
* config framework mới
* dataset framework mới
* training runner mới nếu runner hiện tại có thể reuse
* evaluation framework mới

**Reuse toàn bộ baseline infrastructure.**

---

# 6. Evaluation

So sánh trực tiếp:

### E0 — Baseline

```text
RT-DETR-L pretrained checkpoint
P3/P4/P5
```

### E1 — Proposed

```text
RT-DETR-L baseline checkpoint
+
C2 → P2 auxiliary detector
+
Native predictions + P2 predictions → NMS
```

Giữ **toàn bộ evaluation protocol giống baseline**.

Bao gồm metrics hiện tại:

* AP50
* AP75
* AP50:95
* Precision
* Recall
* F1
* FPPI
* scale-specific AP
* latency nếu baseline đang đo

Đặc biệt kiểm tra:

```text
Ultra-Fine
Fine
Medium
Large
```

Không cần xây metric mới.

---

# 7. Feasibility GO / NO-GO

### GO

Nếu P2:

* cải thiện rõ **Ultra-Fine**
* overall AP không giảm đáng kể / tăng
* không tạo explosion về FPPI
* Medium/Large không bị phá đáng kể

→ tiếp tục nghiên cứu Proposed Method.

### NO-GO

Nếu:

* P2 không cải thiện Ultra-Fine
* AP giảm đáng kể
* FPPI tăng mạnh
* P2 chủ yếu tạo false positives

→ dừng hướng này.

---

# Final implementation scope

```text
src/hrp4k/
├── models/
│   ├── p2_branch.py
│   └── p2_head.py
│
├── inference/
│   └── p2_fusion.py
│
└── experiments/
    └── proposed.py
```

**Chỉ cần thêm tối thiểu các thành phần này**, đồng thời tận dụng native:

```text
RT-DETR backbone
RT-DETR Hybrid Encoder
RT-DETR Decoder
RT-DETR loss
training runner
dataset
evaluation
logging
checkpointing
CLI
```

### Pipeline cuối cùng

```text
                  ┌──────────────── P3 ─┐
                  ├──────────────── P4 ─┤
Input → Backbone ┤                    RT-DETR native
                  └──────────────── P5 ─┘
                   
                  C2
                   │
                   ▼
              P2 Adapter
                   │
                   ▼
               P2 Head
                   │
                   ▼
              P2 predictions
                   │
                   ├──────────────┐
                   │              │
Native RT-DETR predictions ───────┤
                                  ▼
                                 NMS
                                  │
                                  ▼
                           Final predictions
```

**Đây là phiên bản tôi khuyên implement ngay.** Nó tối giản, tận dụng tối đa baseline đã train, và quan trọng nhất là experiment chỉ thêm **một biến chính: P2 auxiliary detector**.
