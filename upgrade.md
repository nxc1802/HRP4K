
---

# Proposal: AdaPoth-Lite

## 1. Research Question

Benchmark hiện tại đã cho thấy ba vấn đề:

### ① Low-resolution làm mất thông tin

YOLO11m:

* 640: **18.32% mAP50-95**
* 1280: **25.40%**
* 4K: **33.27%**

Trong khi latency lần lượt là 8.2, 14.6 và 27.3 ms.

### ② Train 640 → test trực tiếp 4K không giải quyết được vấn đề

YOLO11m 640:

> 18.32% → **6.69% mAP50-95**

D-FINE 640:

> 18.18% → **0.06%**



### ③ Exhaustive slicing recover được information nhưng quá đắt

YOLO11m patch + sliced-NMS:

> **18.81% mAP50-95**, Recall **62.43%**, nhưng **3.62 s/image**, 25 calls/image.



Do đó câu hỏi nghiên cứu của AdaPoth-Lite là:

> **Can adaptive region selection recover the small-object benefit of high-resolution processing while processing only a small fraction of the 4K image?**

---

# 2. Core hypothesis

Giả thuyết:

> Không cần chạy detector trên toàn bộ ảnh 4K. Một model rất nhẹ có thể xác định phần lớn vùng có khả năng chứa pothole; detector chỉ cần xử lý các vùng đó ở high resolution.

Do đó:

$$
\boxed{
\text{Low-cost Global Scout}
\rightarrow
\text{Sparse HR Regions}
\rightarrow
\text{Lightweight Detector}
}
$$

Khác với slicing thông thường:

```text
Exhaustive slicing

4K
 ↓
25 crops
 ↓
25 detector calls
```

AdaPoth:

```text
4K
 ↓
Scout
 ↓
1–4 candidate regions
 ↓
1 global + 1–4 local calls
```

Đây chính là **adaptive compute allocation**, là contribution trung tâm của proposal. 

---

# 3. Kiến trúc chốt

Tôi đề xuất **không mở rộng kiến trúc thêm nữa ở MVP**.

```text
                     HRP4K 3840×2160
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Resize 960×540             Original 4K
              │                         │
              ▼                         │
    MobileNetV3-Small                 │
         Scout                         │
              │                       │
              ▼                       │
       Heatmap regions                │
              │                       │
              ▼                       │
    Connected Components             │
       + Region NMS                  │
              │                       │
              ▼                       │
       Dynamic Top-K                 │
         K ≤ 4                       │
              │                       │
       ┌──────┴──────┐               │
       ▼             ▼               ▼
   Local crop 1  Local crop 2 ... Global image
       │             │               │
       └──────┬──────┘               │
              ▼                      ▼
        YOLO11n-P2 ← shared weights
              │
              ▼
       Coordinate mapping
              │
              ▼
       Score calibration
              │
              ▼
            Fusion
              │
              ▼
       Final detections
```

Proposal gốc đã xác định chính xác cấu hình này: MobileNetV3-Small Scout ở 960×540, Dynamic Top-K `K≤4`, và YOLO11n-P2 shared detector.

---

# 4. Module A — Region Scout

## 4.1 Model

Dùng:

**MobileNetV3-Small pretrained**

Input:

$$
960\times540
$$

Output:

$$
60\times34
$$

tức output stride 16.

Output chỉ có **1 channel heatmap**.

Head:

```text
Depthwise Conv 3×3
       ↓
Pointwise Conv 1×1
       ↓
Sigmoid
       ↓
Heatmap
```

Target:

> **<1.5M parameters**

theo proposal. 

---

# 5. Scout không phải detector

Đây là điểm rất quan trọng.

Scout **không cần predict bounding box chính xác**.

Nó chỉ cần:

> "Ở khu vực này có khả năng tồn tại pothole."

Ví dụ:

```text
Heatmap

0.01 0.02 0.03 0.01
0.02 0.84 0.92 0.04
0.01 0.76 0.81 0.02
0.01 0.02 0.01 0.00
```

Sau đó chuyển heatmap thành candidate regions.

Điều này giúp bài toán Scout đơn giản hơn detection rất nhiều.

---

# 6. Scout Ground Truth

Từ mỗi GT pothole:

### Bước 1

Scale bounding box từ:

$$
3840\times2160
$$

sang:

$$
960\times540
$$

### Bước 2

Expand box **25%**.

### Bước 3

Tạo elliptical Gaussian:

$$
H(x,y)=
\exp\left(
-\frac{(x-x_0)^2}{2\sigma_x^2}
-\frac{(y-y_0)^2}{2\sigma_y^2}
\right)
$$

với:

$$
\sigma_x=0.35w
$$

$$
\sigma_y=0.50h
$$

Các giá trị này là cấu hình khởi đầu trong proposal, đặc biệt phù hợp với pothole có hình dạng kéo dài ngang. 

---

# 7. Scout Loss

Dùng:

$$
L_{scout}
=
L_{focal}
+
\lambda_{cov}L_{coverage}
$$

với:

$$
\lambda_{cov}=2.0
$$

Mục tiêu **không phải tối đa hóa heatmap precision**.

Mục tiêu số 1 là:

$$
\boxed{Region\ Recall \geq 97\%}
$$

Tức nếu có 100 potholes thì candidate regions phải cover được ít nhất khoảng 97 potholes.

Proposal cũng xác định rõ checkpoint Scout phải được chọn theo **region recall trước**, sau đó mới tối ưu false-region rate. 

---

# 8. Module B — Candidate Generation

Sau Scout:

### Step 1 — Threshold

$$
H>\tau
$$

### Step 2 — Connected components

Các pixel gần nhau được gom thành region.

### Step 3 — Region score

Có thể bắt đầu bằng:

$$
S_r=\max(H_r)
$$

và ablation:

$$
S_r=\alpha\max(H_r)+(1-\alpha)\operatorname{mean}(H_r)
$$

### Step 4 — Map về 4K

Chuyển region từ 960×540 về tọa độ ảnh gốc.

### Step 5 — Expand context

Context margin:

$$
20\%
$$

### Step 6 — Region NMS

IoU:

$$
0.35
$$

### Step 7 — Dynamic Top-K

$$
K\le4
$$

Cấu hình MVP:

```text
0 components
    → safety crop nếu global confidence thấp

1 component
    → K = 1

2 components
    → K = 2

3 components
    → K = 3

>3 components
    → K = 4
```

Đây là **dynamic K**, không phải fixed 4 crops. 

---

# 9. Safety mechanism

Đây là thành phần tôi cho rằng **không nên bỏ**.

Nếu Scout miss toàn bộ:

```text
Scout
 ↓
0 candidate
```

pipeline vẫn phải hoạt động.

Do đó:

> Global detector luôn chạy.

Và nếu Scout không tìm thấy region nhưng global detector có confidence thấp/không chắc chắn, lấy **top-1 safety crop**.

Như vậy AdaPoth không bị biến thành:

> Scout miss → entire image miss.

Proposal cũng đã đề xuất global branch + safety top-1 để xử lý Scout miss. 

---

# 10. Module C — Lightweight Detector

Đây là phần cần train thêm ngoài Scout.

Không dùng YOLO11m.

Dùng:

$$
\boxed{\text{YOLO11n-P2-lite}}
$$

## Cấu hình

Feature levels:

* P2 — stride 4
* P3 — stride 8
* P4 — stride 16

**Bỏ P5** trong cấu hình chính.

Channels:

```text
P2 = 48
P3 = 96
P4 = 192
```

Neck dùng depthwise separable convolution.

Proposal đặt mục tiêu detector khoảng **3–5M parameters**. 

---

# 11. Tại sao phải train detector mới?

Đây là chỗ cần sửa cách hiểu từ câu hỏi trước.

Không phải:

> "Tôi đã có YOLO11m 4K rồi, chỉ cần thêm Scout."

**Nếu mục tiêu là paper AdaPoth-Lite thì không nên làm vậy.**

Vì khi đó:

```text
Scout rất nhẹ
+
YOLO11m
```

vẫn là một medium detector.

Contribution efficiency sẽ yếu.

AdaPoth-Lite cần chứng minh:

> **Một detector nhỏ + intelligent region allocation có thể cạnh tranh với detector medium/full-resolution.**

Do đó phải train **YOLO11n-P2-lite**.

---

# 12. Shared Global–Local Detector

Đây là một điểm novelty quan trọng.

Không train:

```text
Global detector
+
Local detector
```

riêng biệt.

Chỉ có:

$$
\boxed{\text{ONE detector}}
$$

dùng chung weights.

Nó nhận:

### Global

```text
3840×2160
 ↓
960×544
 ↓
YOLO11n-P2
```

### Local

```text
crop từ ảnh 4K
 ↓
768×512
 ↓
YOLO11n-P2
```

Như vậy local branch nhìn thấy pothole với scale lớn hơn, trong khi global branch giữ contextual information.

Proposal gọi đây là **shared global-local detector**.

---

# 13. Training detector

Đây là phần tôi sẽ triển khai theo 2 bước.

## Stage 1 — Full-image baseline

Train YOLO11n-P2-lite:

```text
Input = 960×544
Pretrained = COCO
Epoch = 150–200
Seeds = 3
```

Đây là baseline bắt buộc.

---

# 14. Stage 2 — Local crop training

Tạo training crops.

Phân phối:

```text
50% positive local crops
25% hard negatives
25% full-image samples
```

Positive crops:

* center jitter
* scale jitter
* random context
* random offset
* horizontal flip
* brightness/contrast
* nhẹ perspective augmentation

Hard negatives:

* crack
* tar repair
* shadow
* water
* concrete joints
* rough pavement texture

Proposal xác định đúng các nhóm hard negative này vì chúng là nguồn false positive quan trọng. 

---

# 15. Stage 3 — Scout-generated crop fine-tuning

Đây mới là bước khiến detector thích nghi với inference thực tế.

Sau khi Scout đã train:

```text
Training image
      ↓
Scout
      ↓
candidate regions
      ↓
crop
      ↓
YOLO11n-P2
```

Sau đó fine-tune detector bằng:

```text
60% Scout-generated crops
40% GT/full-image samples
```

Mục đích:

> train distribution ≈ inference distribution.

Proposal đã quy định chính xác tỷ lệ này. 

---

# 16. Module D — Global + Local inference

Với mỗi ảnh:

### Global

Chạy:

$$
D_g=Detector(I_{960})
$$

### Local

Với từng candidate:

$$
D_i=Detector(C_i)
$$

Sau đó inverse mapping:

$$
B_i^{4K}=T_i^{-1}(B_i^{crop})
$$

Tất cả bounding box cuối cùng được đưa về hệ tọa độ:

$$
3840\times2160
$$

---

# 17. Fusion

Có hai phương án:

### Baseline

Class-agnostic NMS:

$$
IoU=0.5
$$

### Ablation

Weighted Box Fusion.

Cấu hình chính nên bắt đầu bằng NMS để pipeline đơn giản.

Ngoài ra:

### Crop-boundary penalty

Nếu box nằm sát crop boundary:

$$
s'=s\cdot p_{boundary}
$$

để giảm false detection do object bị cắt.

Proposal cũng đề xuất ưu tiên local score đối với small boxes và dùng boundary penalty. 

---

# 18. Calibration

Global và local prediction không nhất thiết có cùng score distribution.

Do đó:

$$
s_g'=T_g(s_g)
$$

$$
s_l'=T_l(s_l)
$$

Temperature scaling trên validation set.

Sau đó mới fusion.

Đây là một phần nhỏ nhưng nên giữ vì nó giúp pipeline có cơ sở phương pháp luận tốt hơn. 

---

# 19. Experimental Plan

Đây là phần quan trọng nhất của paper.

## Experiment 1 — Lightweight detector baseline

So sánh:

```text
YOLO11n
YOLO11n-P2
YOLO11n-P2-lite
```

Mục tiêu:

> chứng minh P2-lite thực sự có lợi cho ultra-fine objects.

---

## Experiment 2 — Oracle crop upper bound

Dùng GT để tạo crop:

```text
GT crop
 ↓
YOLO11n-P2-lite
```

Đây là **oracle experiment**.

Nó trả lời:

> Nếu chúng ta biết chính xác pothole nằm ở đâu, local high-resolution detector có đủ khả năng recover information không?

Nếu oracle crop không tốt → vấn đề nằm ở detector.

Nếu oracle crop rất tốt → Scout trở thành bottleneck.

Đây là experiment cực kỳ quan trọng.

---

# 20. Experiment 3 — Fixed vs adaptive

So sánh:

| Method            |   K |
| ----------------- | --: |
| Global only       |   0 |
| Random crop       |   2 |
| Fixed crop        |   4 |
| Scout + fixed K   |   4 |
| Scout + Dynamic K | 1–4 |

Mục tiêu chứng minh:

> **adaptive allocation tốt hơn việc cứ cố định số crop.**

---

# 21. Experiment 4 — Scout quality

Không chỉ báo accuracy của detector.

Phải báo:

| Metric            | Ý nghĩa                            |
| ----------------- | ---------------------------------- |
| Region Recall     | GT potholes được cover             |
| GT Coverage       | diện tích/box được candidate cover |
| False Region Rate | candidate vô ích                   |
| Average K         | số crop trung bình                 |
| Max K             | worst case                         |

Metric quan trọng nhất:

$$
\boxed{Region\ Recall}
$$

Target:

$$
97\%-99\%
$$

theo proposal. 

---

# 22. Experiment 5 — K ablation

Chạy:

```text
Kmax = 2
Kmax = 4
Kmax = 6
```

Proposal hiện tại đã đề xuất đúng ba mức này. 

Tôi dự kiến **Kmax=4 là main configuration**.

Lý do:

* đủ candidate coverage;
* giới hạn worst-case compute;
* dễ chứng minh efficiency.

---

# 23. Experiment 6 — Context margin

```text
10%
20%
30%
```

Main:

$$
20\%
$$

Cần đo:

* mAP
* region recall
* crop area
* latency.

Nếu 30% làm accuracy tăng nhưng compute tăng mạnh thì không nhất thiết chọn 30%.

---

# 24. Experiment 7 — Scout resolution

```text
640×360
960×540
1280×720
```

Proposal đã xác định đây là ablation chính. 

Main:

$$
960\times540
$$

vì đây là điểm cân bằng giữa scout quality và overhead.

---

# 25. Experiment 8 — Shared detector ablation

Đây là experiment để chứng minh contribution:

```text
Global only
Global + Local separate detector
Global + Local shared detector
```

Nếu shared detector đạt gần separate detector nhưng parameter thấp hơn đáng kể → rất có giá trị.

---

# 26. Experiment 9 — So sánh với benchmark hiện tại

Đây sẽ là bảng **main result**.

| Method             |  mAP50-95 |    Recall |   Params |       Latency |
| ------------------ | --------: | --------: | -------: | ------------: |
| YOLO11m 640        |     18.32 |     35.06 |   medium |        8.2 ms |
| YOLO11m 1280       |     25.40 |     45.50 |   medium |       14.6 ms |
| YOLO11m 4K         | **33.27** |     49.19 |   medium |       27.3 ms |
| D-FINE 4K          | **33.20** | **77.85** |   medium |       32.5 ms |
| Patch + sliced-NMS |     18.81 | **62.43** |   medium | **2.3–3.6 s** |
| **AdaPoth-Lite**   |     **?** |     **?** | **4–7M** |         **?** |

Các baseline hiện tại đã được đánh giá trên cùng test framework 900 ảnh, nên đây sẽ là comparison rất sạch.

---

# 27. Metrics bắt buộc

## Accuracy

* mAP50
* mAP75
* mAP50-95
* Recall
* Precision
* F1
* FPPI

## Small-object

* AP ultra-fine
* AP fine
* AP medium
* AP large

## Efficiency

* Parameters
* GFLOPs
* latency
* P95 latency
* FPS
* peak VRAM
* average K
* processed-area ratio

Proposal cũng yêu cầu báo cáo riêng scale, material và compute. 

---

# 28. Metric quan trọng nhất của AdaPoth

Không nên chỉ nói:

> "AdaPoth đạt X mAP."

Phải có:

$$
\boxed{
Accuracy
\quad\leftrightarrow\quad
Compute
}
$$

Ví dụ:

```text
Method                 mAP50-95      Latency

YOLO11m 640              18.32         8 ms
YOLO11m 1280             25.40        15 ms
YOLO11m 4K               33.27        27 ms
Sliced-NMS               18.81      2300 ms
AdaPoth                   XX           XX
```

Sau đó vẽ:

**Accuracy–Latency Pareto frontier**

và:

**Accuracy–Processed Area Pareto frontier.**

Đây mới là figure có khả năng trở thành **main figure của paper**.

---

# 29. Mục tiêu thực tế

Proposal cũ đặt mục tiêu khá tham vọng:

* mAP50: **0.62–0.64**
* mAP50-95: **0.42–0.44**
* total parameters: **4–6.5M**
* average K ≤ 3
* region recall ≥97%
* giảm ≥50% compute so với baseline medium/exhaustive processing. 

Tôi khuyên **không coi 0.62/0.42 là điều kiện bắt buộc**.

Hãy coi chúng là **stretch goal**.

Điều kiện thành công khoa học nên là:

### Minimum success

$$
RegionRecall\ge97\%
$$

và AdaPoth phải tạo được **Pareto improvement** so với các baseline.

Ví dụ:

> 28–30 mAP50-95 với latency thấp và chỉ 1–3 crop/image

có thể đáng giá hơn:

> 33 mAP nhưng chạy 200 ms.

---

# 30. Failure analysis

Bắt buộc phân tích 4 loại:

### A. Scout miss

```text
GT pothole
↓
Scout không tạo candidate
```

### B. Crop truncation

```text
Pothole nằm sát boundary
↓
crop cắt object
```

### C. False positive

Đặc biệt:

* concrete joint
* cracks
* tar
* shadows
* rough texture.

### D. Duplicate

```text
Global prediction
+
Local prediction
↓
duplicate boxes
```

Proposal cũng xác định bốn nhóm failure/risk này. 

---

# 31. Training schedule cuối cùng

Tôi sẽ chốt thành:

```text
                    TRAINING
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
   YOLO11n-P2-lite              MobileNetV3 Scout
        │                             │
        │                         Heatmap training
        │                             │
        ▼                             ▼
 Full-image baseline             Region Recall
        │                             │
        ▼                             ▼
 Local crop pretraining          Dynamic K
        │
        ▼
 Scout-generated crop
 fine-tuning
        │
        └──────────────┬──────────────┘
                       ▼
                   Calibration
                       │
                       ▼
                  Final pipeline
```

**Không joint end-to-end ở phiên bản chính.**

Điều này đã được proposal xác định để hệ thống dễ tái tạo và dễ phân tích contribution của từng module. 

---

# 32. Scope thực tế: bạn cần train những gì?

Đây là phần trả lời trực tiếp câu hỏi trước của bạn.

### Đã có

```text
YOLO11m 4K
YOLO11m 640
D-FINE 4K
D-FINE 640
...
```

→ dùng làm **external baselines**.

### Cần train mới

**Model 1 — MobileNetV3-Small Scout**

```text
~1–1.5M params
```

**Model 2 — YOLO11n-P2-lite**

```text
~3–5M params
```

Nhưng **Model 2 không phải train nhiều detector khác nhau**. Một checkpoint chính là đủ; các variant chủ yếu phục vụ ablation.

---

# 33. Thứ tự triển khai tôi khuyên dùng

Đừng code toàn bộ pipeline ngay.

### Phase 1 — Detector

```text
YOLO11n
   ↓
YOLO11n-P2
   ↓
YOLO11n-P2-lite
```

**Nếu P2-lite không đánh bại baseline nhỏ → dừng và sửa trước.**

---

### Phase 2 — Oracle

```text
GT regions
   ↓
crop
   ↓
YOLO11n-P2-lite
```

Nếu oracle local processing tốt:

→ hypothesis đúng.

---

### Phase 3 — Scout

```text
MobileNetV3
 ↓
heatmap
 ↓
region recall
```

Chưa cần detector.

Chỉ tối ưu:

$$
Recall \ge97\%
$$

---

### Phase 4 — Adaptive pipeline

```text
Scout
 ↓
Dynamic K
 ↓
YOLO11n-P2-lite
 ↓
fusion
```

---

### Phase 5 — Paper experiments

```text
Ablation
+
3 seeds
+
Pareto
+
Scale
+
Material
+
Failure analysis
```

---

# 34. Contribution cuối cùng của paper

Nếu kết quả đạt mục tiêu, tôi sẽ viết contribution thành **3 contribution chính**, không tham quá nhiều:

### C1 — Adaptive high-resolution processing

> A learned region allocation mechanism that selectively routes high-resolution computation to informative regions instead of exhaustively processing the 4K image.

### C2 — Shared global-local lightweight detector

> A lightweight P2-enhanced detector shared across global and local views, enabling high-resolution local refinement without maintaining separate detectors.

### C3 — Accuracy–efficiency benchmark

> A systematic evaluation showing the trade-off among native 4K inference, resizing, exhaustive slicing, and adaptive high-resolution processing on HRP4K.

Các ý này phù hợp với contribution dự kiến trong proposal hiện tại. 

---

## Chốt lại

**AdaPoth-Lite không phải là “train thêm một model Scout rồi ghép với YOLO11m”.**

Phiên bản paper-grade nên là:

$$
\boxed{
\underbrace{\text{MobileNetV3-Small}}_{\text{Scout}}
+
\underbrace{\text{Dynamic Top-K}}_{\text{Adaptive allocation}}
+
\underbrace{\text{YOLO11n-P2-lite}}_{\text{Shared detector}}
+
\underbrace{\text{Global–Local Fusion}}_{\text{Final prediction}}
}
$$

Trong đó:

* **Scout** = model mới ~1–1.5M;
* **Detector** = model mới ~3–5M;
* **Dynamic Top-K** = algorithm, không phải model;
* **Global/Local sharing** = cùng một detector;
* **Oracle crop** = experiment để chứng minh local HR processing có tiềm năng;
* **YOLO11m/D-FINE 4K** = giữ nguyên làm benchmark/upper-bound;
* **900 test images** = chỉ dùng cuối cùng để report final result.

Và quan trọng nhất: **benchmark hiện tại của bạn đã làm xong phần “tại sao cần AdaPoth”. Việc còn lại là chứng minh “AdaPoth giải quyết được điều đó với ít compute hơn”.** 
