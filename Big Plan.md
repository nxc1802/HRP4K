Đồng ý. Với mục tiêu hiện tại, tôi sẽ **đóng băng architecture P2 hiện tại** và xây plan theo đúng nguyên tắc:

> **Không training → training với compute ~1× → chỉ giữ thay đổi nếu có evidence → nếu toàn bộ không cải thiện đáng kể thì kết luận P2 đã gần đạt ceiling và chuyển sang public paper.**

## Proposed Method Optimization Plan

### Phase 0 — Baseline cố định

Giữ nguyên baseline hiện tại:

* Frozen RT-DETR-L
* P2 Adapter + Lightweight P2 Head
* 1920×1920 canonical evaluation
* Seed = 42
* Dataset/split cố định
* Training config hiện tại
* Fusion = concat + class-aware NMS

Baseline cần ghi nhận:

* Overall AP50 / AP75 / AP50:95
* Precision / Recall / F1 / FPPI
* **Ultra-fine Recall / AP50**
* Fine / Medium / Large metrics

Đặc biệt lấy **Native + P2-only + Fused** làm baseline.

---

# Phase 1 — Optimization **không cần training**

Mục tiêu: tìm improvement "free" trước khi tiêu thêm GPU.

### 1.1 Top-K sweep

Hiện tại:

```text
P2 predictions
      ↓
Top-K = 300
      ↓
NMS
```

Test:

```text
K = 300
K = 500
K = 1000
K = 2000
```

### 1.2 P2 confidence threshold sweep

Test:

```text
0.001
0.003
0.005
0.01
0.02
```

Có thể chạy grid:

```text
Top-K × P2 threshold
```

vì hai yếu tố này tương tác với nhau.

### 1.3 NMS IoU sweep

Nếu compute inference cho phép, test thêm:

```text
IoU = 0.4
0.5   ← current
0.6
0.7
```

### Output Phase 1

Chọn:

> **Best inference configuration**

theo mục tiêu:

1. Ultra-fine Recall ↑
2. Ultra-fine AP50 ↑
3. F1 ↑
4. FPPI ↓
5. Overall AP không giảm đáng kể

**Không retrain.**

Nếu Phase 1 đã tạo improvement đáng kể → giữ configuration này làm baseline cho Phase 2.

---

# Phase 2 — Multi-positive Target Assignment

Đây là **training experiment đầu tiên**.

Current:

```text
GT
 ↓
nearest center cell
 ↓
1 positive
```

Upgrade:

```text
GT
 ↓
center region
 ↓
multiple positive cells
```

Bắt đầu đơn giản nhất:

### Variant A

```text
1 × 1
```

baseline.

### Variant B

```text
3 × 3 center region
```

Không cần thử quá nhiều biến thể ngay.

Nếu 3×3 tốt rõ ràng mới cân nhắc 5×5.

### Giữ nguyên

* model
* optimizer
* LR
* batch
* epochs
* seed
* resolution
* loss

Chỉ thay **target assignment**.

→ Compute khoảng **1× baseline**.

### Điều cần quan sát

Không chỉ nhìn overall AP.

Đặc biệt:

```text
Ultra-fine Recall
Ultra-fine AP50
P2-only AP50
P2-only Recall
Fused AP50
Fused F1
```

Nếu multi-positive làm P2 mạnh lên nhưng fused không tăng → vấn đề có thể nằm ở fusion/calibration, không phải P2 capacity.

---

# Phase 3 — Focal Loss / Quality Focal Loss

Sau khi xác định target assignment tốt nhất.

### 3.1 Focal Loss

Baseline:

```text
BCE
```

thử:

```text
Focal Loss
γ = 2
```

Giữ target assignment tốt nhất từ Phase 2.

→ Compute ≈ **1×**.

### 3.2 Quality Focal Loss

Nếu Focal Loss có improvement:

```text
Focal
   ↓
Quality Focal Loss
```

QFL đặc biệt đáng thử vì bài toán của bạn không chỉ cần:

> object / background

mà còn cần score phản ánh:

> localization quality.

Điều này có thể giúp:

* AP50
* AP75
* AP50:95
* score calibration
* fusion với Native

### Decision

Nếu:

```text
BCE → Focal
```

không cải thiện → **không cần QFL**.

Nếu Focal có gain → mới chạy QFL.

Như vậy không lãng phí GPU.

---

# Phase 4 — Scale-aware Loss

Đây là optimization cuối cùng trong nhóm **rẻ nhưng rất đúng research hypothesis**.

Current dataset cho thấy P2 thực sự hữu ích nhất ở:

> **Ultra-fine**

Vì vậy loss nên ưu tiên object nhỏ.

Ví dụ:

```text
Ultra-fine → weight 3.0
Fine       → weight 2.0
Medium     → weight 1.0
Large      → weight 0.5
```

Nhưng tôi sẽ không hard-code các con số này ngay.

Chạy một sweep nhỏ:

```text
Baseline:
1 / 1 / 1 / 1

Scale-aware A:
2 / 1.5 / 1 / 0.5

Scale-aware B:
3 / 2 / 1 / 0.5
```

Trong đó thứ tự là:

```text
Ultra-fine / Fine / Medium / Large
```

Không cần thay architecture.

→ Compute khoảng **1× cho mỗi candidate**.

---

# Phase 5 — Build Best P2

Sau 4 phase trên, lấy **winner của từng component**.

Ví dụ nếu kết quả là:

```text
Target assignment → 3×3
Loss               → QFL
Scale-aware        → B
Inference          → TopK=1000, threshold=0.005
```

thì tạo:

```text
Best P2
```

và train/evaluate **một final run**.

Quan trọng:

> Không tiếp tục stack thêm 10 ý tưởng.

Đây là lúc xác định **ceiling thực tế của architecture hiện tại**.

---

# Phase 6 — Final Evaluation

So sánh chính thức:

```text
Native RT-DETR-L
        vs
Original P2
        vs
Optimized P2
        vs
Fused Proposed
```

Các metric bắt buộc:

### Overall

* AP50
* AP75
* AP50:95
* Precision
* Recall
* F1
* FPPI

### Scale-wise

* Ultra-fine Recall
* Ultra-fine AP50
* Fine
* Medium
* Large

### Quan trọng nhất

Tính:

```text
Δ Native → Fused
Δ Original P2 → Optimized P2
```

---

# Decision Gate — điểm kết thúc nghiên cứu

Đây là phần tôi nghĩ nên **đặt thành nguyên tắc cứng**.

## Case A — Có improvement rõ

Ví dụ:

```text
Ultra-fine Recall
91.10 → 94–96%

Ultra-fine AP50
50.36 → 52–55%

Overall F1
48.33 → 50%+

FPPI
↓
```

→ **Tiếp tục research.**

Lúc này mới đáng đầu tư:

* Native-failure-aware training
* C2+C3 fusion
* hard-negative mining
* multi-scale/crop training

---

## Case B — P2 tốt hơn nhưng Fused gần như không đổi

Ví dụ:

```text
P2 AP50
4.9 → 8.0%

nhưng

Fused AP50
62.51 → 62.6%
```

→ Đây là dấu hiệu **P2 đang bị giới hạn bởi fusion/complementarity**, chứ chưa chắc architecture ceiling.

Khi đó chỉ cần thử **một** hướng:

> Native-aware/failure-aware training.

Nếu vẫn không cải thiện → stop.

---

## Case C — Tất cả optimization đều không cải thiện

Ví dụ:

```text
Original:
Ultra-fine Recall = 92.80%

Optimized:
92.5–93.0%

Overall AP:
~62.5%

F1:
~48%
```

và các variant:

* Multi-positive ❌
* Focal/QFL ❌
* Scale-aware ❌
* Top-K/threshold ❌

đều không tạo gain đáng kể.

### → Kết luận:

> **P2 auxiliary branch đã đạt practical ceiling trong architecture/training budget hiện tại.**

**Không tiếp tục thêm module.**

Không:

* Transformer attention
* BiFPN
* Deformable Conv
* Stride-2
* DFL
* learned fusion
* WBF
* dynamic head
* thêm 5 loại augmentation

Chỉ để cố lấy thêm 0.x%.

Khi đó:

> **Chấp nhận kết quả và public paper ở mức thấp hơn.**

---

# Compute Budget tổng thể

Tôi sẽ thiết kế budget như sau:

```text
                    CURRENT P2
                       │
                       ▼
              ┌────────────────┐
              │ Phase 1         │
              │ Top-K/Threshold │
              │ NO TRAINING     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Phase 2         │
              │ Multi-positive  │
              │ ~1×             │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Phase 3         │
              │ Focal / QFL     │
              │ ~1×             │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Phase 4         │
              │ Scale-aware     │
              │ ~1×             │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ Best combination│
              │ ~1×             │
              └───────┬────────┘
                      │
                      ▼
                 FINAL RESULT
```

**GPU budget thực tế có thể giữ quanh 4× baseline** bằng cách không chạy toàn bộ grid một cách mù quáng:

* Phase 1: free
* Phase 2: 1 run
* Phase 3: 1 run
* Phase 4: 1 run
* Final: 1 run

Nếu cần tiết kiệm hơn nữa, **chỉ chạy Final khi intermediate experiments thực sự có gain**.

---

## Research strategy tôi khuyên chốt

Từ giờ proposed method không nên phát triển theo kiểu:

> "Có idea nào hay thì thêm vào."

Mà theo kiểu:

> **POC → cheap optimization → controlled training ablation → best configuration → final evaluation → stop/go decision.**

Và **Phase 1–4 chính là toàn bộ “low-compute optimization budget”**.

Nếu sau chúng mà kết quả vẫn chỉ quanh:

> **62.5% AP50 overall + ~93% ultra-fine recall**

thì tôi hoàn toàn đồng ý với bạn: **đừng đốt thêm GPU để cứu architecture**. Khi đó hãy đóng contribution ở mức *lightweight auxiliary high-resolution P2 branch for ultra-fine pothole detection*, làm experimental analysis thật sạch và public một paper tầm thấp.
