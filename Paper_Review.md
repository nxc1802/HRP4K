Mình đã đọc toàn bộ paper 19 trang. Nhìn tổng thể, đây là **một paper có cấu trúc khá tốt và ý tưởng tương đối rõ**, nhưng hiện tại vẫn có một số vấn đề quan trọng về **novelty, experimental rigor và tính nhất quán số liệu** cần xử lý nếu mục tiêu là conference/journal có chất lượng.

### Đánh giá nhanh

| Thành phần               | Đánh giá hiện tại |
| ------------------------ | ----------------- |
| Problem formulation      | ⭐⭐⭐⭐⭐             |
| Motivation               | ⭐⭐⭐⭐⭐             |
| Architecture             | ⭐⭐⭐⭐              |
| Novelty                  | ⭐⭐⭐½              |
| Experimental design      | ⭐⭐⭐⭐              |
| Ablation                 | ⭐⭐⭐⭐              |
| Practical relevance      | ⭐⭐⭐⭐⭐             |
| Mathematical formulation | ⭐⭐⭐½              |
| Writing/presentation     | ⭐⭐⭐⭐              |
| Reproducibility          | ⭐⭐⭐⭐              |
| **Tổng thể**             | **~7.5–8/10**     |

**Điểm mạnh nhất:** paper không đơn thuần nói "thêm P2 vào detector", mà xây dựng được một câu chuyện khá hoàn chỉnh:

> **4K imagery → ultra-fine object → 2K là sweet spot → 4K direct inference không hiệu quả → slicing phá perspective → cần high-resolution representation nhưng không muốn xử lý toàn bộ 4K → frozen 2K macro detector + lightweight P2 branch → scale-aware fusion.**

Đây là narrative tốt.

---

# 1. Paper đang thực sự đóng góp gì?

Core contribution của paper là **P2-DETR**:

* RT-DETR-L chạy ở **2K**, frozen.
* Lấy feature **C2** ở độ phân giải cao.
* Tạo thêm **P2 dense detection head, stride 4**.
* P2 chỉ chuyên xử lý ultra-fine objects.
* Dùng **Sigmoid Focal Loss** vì imbalance cực lớn.
* Cuối cùng dùng **Scale-Partitioned Prior NMS** để hợp nhất hai nhánh.

Architecture này được mô tả khá rõ trong paper. P2 head chỉ thêm **2.98M parameters** vào backbone 32.8M. 

Đặc biệt, paper không cố claim rằng P2 head thay thế RT-DETR. Nó đóng vai trò **specialist branch** trong khi RT-DETR giữ macro-level context.

Đây là framing đúng và thuyết phục.

---

# 2. Điểm mạnh lớn nhất: Problem rất rõ

Phần Introduction là một trong những phần tốt nhất.

Paper không bắt đầu bằng:

> "Existing methods are bad, therefore we propose P2-DETR."

Mà bắt đầu từ đặc tính vật lý của bài toán.

Bạn đưa ra:

$$
S \propto \frac{1}{z^2}
$$

tức là apparent area của pothole giảm theo bình phương khoảng cách. 

Sau đó nối nó với:

* 4K imagery
* distant potholes
* ultra-fine targets
* downsampling
* FPN stride
* Transformer computational cost
* real-time requirement.

Đây là **problem-driven research**, tốt hơn nhiều so với kiểu "architecture-driven".

---

# 3. Benchmark 4K → 2K → 1K → 640 rất có giá trị

Table 1 là một experiment quan trọng.

RT-DETR-L:

| Resolution |      AP50 |     Latency |
| ---------- | --------: | ----------: |
| 640        |     33.49 |     19.5 ms |
| 1K         |     58.14 |     22.1 ms |
| **2K**     | **62.48** | **43.2 ms** |
| 4K         |     58.07 |    220.4 ms |



Đây tạo ra một observation rất tốt:

> **Simply increasing resolution does not monotonically improve detection.**

4K thậm chí **kém 2K về AP50**, trong khi latency tăng hơn 5×. Paper sử dụng observation này để justify frozen 2K macro branch. 

Đây là một trong những experimental findings có khả năng trở thành selling point của paper.

---

# 4. So sánh SAHI/tiling cũng khá tốt

Table 2 là phần mình đánh giá cao.

Bạn không chỉ benchmark proposed method với baseline, mà kiểm tra các **alternative solutions**:

* Sliced-NMS
* SAHI
* Perspective Grid
* ZoomDet

Kết quả rất mạnh về mặt narrative.

Ví dụ RT-DETR-L:

* Full image: 37.37 AP50
* SAHI: 24.28 AP50
* SAHI latency: **3622 ms**
* Sliced-NMS: **2289.8 ms**



Điều này giúp paper trả lời một câu hỏi reviewer rất hay hỏi:

> "Why don't you simply use tiling?"

Paper đã có empirical answer.

---

# 5. P2 branch có một property rất đẹp

Table 3 có một observation rất đáng chú ý:

| Scale      | P2-only Recall |
| ---------- | -------------: |
| Ultra-Fine |     **70.34%** |
| Fine       |         27.22% |
| Medium     |          4.08% |
| Large      |         **0%** |



Điều này chứng minh P2 branch thực sự học được **scale specialization**.

Đây là điểm quan trọng.

Nếu chỉ nói:

> "P2 improves recall"

thì novelty khá yếu.

Nhưng:

> **P2 branch almost exclusively detects ultra-fine objects while native RT-DETR preserves macro objects**

thì câu chuyện **Scale-Specialized Feature Gating** trở nên hợp lý hơn.

---

# 6. Scale-Partitioned Prior NMS là contribution thứ hai khá ổn

Table 4 cho thấy:

| Method        |      AP50 | UF Recall |        F1 |       FPPI |  Latency |
| ------------- | --------: | --------: | --------: | ---------: | -------: |
| Native        |     62.49 |     91.10 |     47.18 |     1.5033 |    43.20 |
| Greedy NMS    |     62.13 |     93.22 |     46.78 |     1.5278 |     9.59 |
| **ScalePart** | **62.32** |     92.16 | **47.22** | **1.4967** | **4.24** |
| WBF           |      3.81 |     37.08 |     11.94 |     0.4289 |     5.11 |
| MLP           |     35.18 |     93.43 |      1.13 |     163.89 |     7.07 |



Đây là một experiment khá hay vì nó cho thấy:

**Recall cao ≠ good detector.**

MLP có UF Recall 93.43%, nhưng FPPI lên **163.89**.

ScalePart đạt balance tốt hơn.

Về research storytelling, phần này tốt.

---

# 7. Ablation cũng tương đối đầy đủ

Table 5 kiểm tra:

* BCE
* QFL
* Focal Loss
* 1×1 assignment
* 3×3 assignment.

Kết quả:

**Pure Focal + 1×1**

cho:

* AP50 = 62.13%
* AP75 = 38.98%
* AP50:95 = 37.23%
* UF Recall = 93.22%
* F1 = 46.78%



Điều này giúp chứng minh design choice không hoàn toàn arbitrary.

---

# 8. Nhưng vấn đề lớn nhất của paper hiện tại: Novelty

Đây là vấn đề mình sẽ quan tâm nhất nếu đóng vai reviewer.

Reviewer hoàn toàn có thể nói:

> "This is essentially RT-DETR + P2 head + focal loss + scale-aware NMS."

Và câu hỏi tiếp theo:

> **What is fundamentally new?**

Bởi vì:

* P2 head → đã có từ lâu.
* Focal Loss → đã rất established.
* anchor-free dense detection → FCOS.
* NMS → established.
* scale-aware filtering → conceptually không quá mới.

Paper hiện tại đang **ghép các thành phần existing techniques rất hợp lý**, nhưng chưa hoàn toàn biến chúng thành một **novel methodological principle**.

Đây là điểm cần nâng cấp mạnh nhất.

---

# 9. Tên "Scale-Specialized Feature Gating" hơi overclaim

Title hiện tại:

> **Scale-Specialized Feature Gating for Real-Time Ultra-Fine Pothole Detection...**

Nhưng architecture hiện tại thực chất không có một **learned feature gating mechanism**.

P2 branch được tạo ra từ C2 và sau đó candidate được filter bởi area:

$$
Area(d)<1024
$$

Đây là **scale-partitioned proposal gating**, chứ chưa thật sự là feature gating.

Thậm chí conclusion cũng nói:

> future work will explore learned cross-attention feature gating



Điều này vô tình làm reviewer hỏi:

**"Nếu learned feature gating là future work, tại sao title gọi current method là Feature Gating?"**

Mình sẽ cân nhắc đổi terminology thành:

> **Scale-Specialized Dual-Branch Detection**

hoặc:

> **Scale-Partitioned Dual-Branch Detection**

sẽ chính xác hơn.

---

# 10. Có một vấn đề rất đáng chú ý về số liệu

Abstract nói:

> **93.22%–94.07% Ultra-Fine Recall**

Nhưng Table 3 nói:

* P2-DETR = 93.22%
* sweep optimized = 94.07%

Trong khi Table 4:

* ScalePart = 92.16%
* Greedy NMS = 93.22%.

 

Tức là reader có thể rất dễ bị confused:

**93.22 là baseline fusion nào?**

**94.07 là configuration nào?**

**92.16 mới là final ScalePart configuration?**

Trong Fig.2 lại ghi:

> UF-Rec = 92.16%



Như vậy paper đang có ít nhất **3 con số UF Recall nổi bật**:

> 92.16 / 93.22 / 94.07

Đây là vấn đề presentation rất lớn.

Bạn cần định nghĩa rõ:

* **Final model**
* **Best validation-tuned model**
* **Greedy NMS**
* **ScalePart**
* **Threshold sweep result**

Nếu không reviewer sẽ nghi ngờ cherry-picking.

---

# 11. Một vấn đề còn lớn hơn: test-set tuning

Paper tuyên bố:

> 900 test images hoàn toàn untouched.



Đây là claim rất tốt.

Nhưng Table 4 và một số phần discussion phải cực kỳ cẩn thận về cách hyperparameter được chọn.

Ví dụ:

* area threshold 1024
* 2304
* 4096
* confidence
* IoU = 0.60
* Top-K
* threshold sweep.

Paper nói các hyperparameter được tuning trên validation. 

Nếu thực tế code có bất kỳ việc nào chọn configuration dựa trên 900 test images, thì **toàn bộ "independent test" claim sẽ bị ảnh hưởng**.

Đây là thứ mình sẽ audit rất kỹ trước submission.

---

# 12. 20 FPS claim cần cẩn thận hơn

Paper nói:

> total pipeline latency = 47.4 ms → 21.1 FPS.



Điều này hợp lý về mặt arithmetic:

$$
1/0.0474 \approx 21.1 FPS
$$

Nhưng cần phân biệt:

### Detection inference FPS

và

### End-to-end vehicular system FPS.

Nếu 47.4 ms chưa bao gồm:

* image capture
* image transfer
* preprocessing
* resize
* postprocessing ngoài ScalePart
* visualization
* data writing
* communication

thì không nên gọi đơn giản là:

> "vehicular throughput"

Nên gọi:

> **detector inference throughput**

hoặc ghi rõ measurement boundary.

Reviewer computer vision khá hay bắt lỗi điểm này.

---

# 13. "Real-time" cũng cần định nghĩa rõ

20 FPS là một reasonable operational requirement, nhưng paper hiện đang hơi giống như **20 FPS là universal threshold**.

Bạn nên viết rõ:

> We define real-time operation as ≥20 FPS for this experimental deployment setting.

Thay vì implication rằng:

> <20 FPS = inherently non-real-time.

---

# 14. Một điểm methodology cần kiểm tra: C2

Paper viết:

> RT-DETR-L backbone → C3, C4, C5

nhưng sau đó:

> directly tapping early backbone stage **C2**



Điều này hoàn toàn có thể implement được, nhưng paper cần làm rõ architecture của RT-DETR-L:

```text
Backbone
 ├── C2 → P2 auxiliary head
 ├── C3 ─┐
 ├── C4 ─┼→ Hybrid Encoder → Transformer
 └── C5 ─┘
```

Hiện figure có thể khiến reader nghĩ C2 không tồn tại trong official RT-DETR feature pipeline.

Nên giải thích rõ:

> C2 is an intermediate backbone feature tapped before the native RT-DETR multi-scale encoder.

---

# 15. Equation 1–2 tốt về motivation nhưng chưa thật sự cần nhiều như vậy

Optical derivation:

$$
x=fX/z
$$

$$
S\propto1/z^2
$$

là hợp lý. 

Nhưng phần:

> "0.3m at 25m occupies <24×24 pixels"

nên có derivation cụ thể hoặc camera parameter assumption.

Nếu không reviewer có thể hỏi:

> Where does 24×24 come from?

Nên thêm:

* focal length
* sensor/image geometry
* camera calibration assumption.

Hoặc đơn giản bỏ numerical example và giữ scaling law.

---

# 16. Related Work hiện tại chưa đủ mạnh để bảo vệ novelty

Related Work đang khá "textbook":

* road damage
* tiny object
* RT-DETR
* NMS.



Nhưng để submit tốt hơn, phần này nên chuyển từ:

> "A, B, C exist"

sang:

> **"Existing approaches fall into three design families, and each fails under our specific constraint."**

Ví dụ:

### Family 1 — Resolution scaling

4K direct inference
→ high compute + attention dilution.

### Family 2 — Spatial decomposition

SAHI / slicing
→ destroys global perspective + latency.

### Family 3 — High-resolution feature architectures

HRNet / P2 / FPN modifications
→ preserve spatial detail but often increase memory/compute or modify backbone.

Sau đó:

> **Our method occupies a fourth design point: retain global 2K context while selectively introducing high-resolution representation only for ultra-fine objects.**

Cách framing này mạnh hơn nhiều.

---

# 17. Paper hiện tại thiên về "engineering paper" hơn "methodology paper"

Đây là đánh giá quan trọng.

Hiện tại paper có:

> benchmark → identify bottleneck → engineer solution → ablation → deployment.

Điều này **không xấu**.

Thậm chí rất phù hợp với applied CV / intelligent transportation.

Nhưng nếu target:

### Workshop / applied conference

→ **paper hiện tại khá ổn.**

### Mid-tier conference

→ **có tiềm năng tốt**, sau khi polish.

### Strong CV conference

→ novelty hiện tại **chưa đủ chắc**.

### Q1 journal

→ có thể được, nhưng cần experimental depth + stronger methodological justification.

---

# 18. Nếu muốn nâng paper lên một level, mình sẽ không thêm 10 module

Đây là điểm mình đặc biệt khuyên.

**Không nên** tiếp tục:

> P2 + attention + deformable conv + dynamic NMS + transformer + ensemble + ...

Paper sẽ mất focus.

Thay vào đó, chỉ cần biến central idea thành một **principle**:

> **Scale-specialized asymmetric representation is more efficient than uniform high-resolution processing for perspective ultra-fine detection.**

Sau đó chứng minh principle này bằng:

1. Resolution scaling.
2. Tiling.
3. Warping.
4. P2-only.
5. Native-only.
6. Dual branch.
7. Alternative fusion.
8. Scale thresholds.

Thực ra paper hiện tại đã có gần đủ ingredients để làm việc đó.

---

# 19. Experiment mình muốn bổ sung nhất

Nếu chỉ được thêm **3 experiments**, mình sẽ chọn:

### Experiment A — Backbone generality

Không chỉ RT-DETR-L.

Ví dụ:

```text
YOLO + P2
RT-DETR + P2
RT-DETR + proposed
```

Mục đích:

> P2-DETR là contribution cho RT-DETR hay là general principle?

Nếu chỉ RT-DETR thì claim nên giới hạn.

---

### Experiment B — Parameter/compute efficiency

So sánh:

```text
RT-DETR 4K
RT-DETR 2K
RT-DETR + P2
HRNet-based
RT-DETR + higher-resolution backbone
```

với:

* Params
* FLOPs
* VRAM
* latency
* AP50
* UF Recall.

Khi đó paper sẽ có một very strong:

> **accuracy–compute Pareto frontier**

---

### Experiment C — Cross-dataset/generalization

Đây là thứ có thể nâng paper rất mạnh.

Hiện tại toàn bộ story đều nằm trên:

> HRP4K.



Nếu HRP4K là dataset mới và paper chỉ test trên nó, reviewer có thể nói:

> "The method may simply exploit HRP4K's annotation distribution."

Một external dataset—even smaller—sẽ rất valuable.

---

# 20. Một vấn đề mình sẽ kiểm tra kỹ: HRP4K và references

Paper gọi:

> "canonical HRP4K benchmark"

và reference [6] là:

> Chen et al., *A high-resolution perspective-view road image dataset for pothole detection*, Scientific Data 13(1), 961 (2026).



Nếu đây là dataset mới xuất bản 2026 thì việc gọi nó là **canonical benchmark** hơi mạnh.

Nên dùng:

> **HRP4K benchmark**

hoặc:

> **the HRP4K dataset**

trừ khi paper gốc thực sự đã established "canonical benchmark".

---

# 21. Điểm mình thích nhất về paper

Không phải P2.

Mà là **experimental question decomposition**.

Paper thực tế trả lời lần lượt:

### Q1

Có nên đưa 4K trực tiếp vào detector?

→ Không.

### Q2

Có nên slicing?

→ Không.

### Q3

Có nên warping?

→ Không.

### Q4

Vậy làm thế nào giữ high-resolution information?

→ Early P2 feature.

### Q5

P2 có thực sự specialized không?

→ P2-only scale-wise analysis.

### Q6

Làm sao merge hai branches?

→ ScalePart.

### Q7

Tại sao không WBF/MLP?

→ Table 4.

### Q8

Tại sao Focal/1×1?

→ Table 5.

Đây là một **rất tốt research structure**.

---

# 22. Đánh giá cuối cùng của mình

Nếu paper này là **version hiện tại**, mình sẽ đánh giá:

> **Good applied computer vision paper, with a clear problem and coherent experimental story, but novelty is currently moderate rather than strong.**

Điểm số cá nhân:

**Technical quality:** 8/10
**Experimental quality:** 8/10
**Problem significance:** 8.5/10
**Novelty:** 6.5–7/10
**Writing:** 8/10
**Reproducibility:** 7.5/10
**Overall:** **~7.8/10**

### Quan trọng nhất:

**Paper không cần thêm thật nhiều technology.**

Nó cần:

1. **Làm rõ novelty của "asymmetric scale-specialized representation".**
2. **Sửa toàn bộ inconsistency 92.16 / 93.22 / 94.07.**
3. **Audit tuyệt đối test-set leakage.**
4. **Làm rõ measurement của 21.1 FPS.**
5. **Củng cố Related Work để bảo vệ novelty.**
6. Nếu có thể, thêm **1 external dataset hoặc 1 backbone-generalization experiment**.

Nếu xử lý được 6 điểm này, paper sẽ chuyển từ cảm giác **"một engineering solution khá tốt"** sang **"một empirical/methodological contribution có luận điểm rõ ràng"**.

Và theo mình, **điểm mạnh thực sự của paper nằm ở benchmark + analysis về resolution/slicing + asymmetric design**, chứ không nên cố biến P2 hay Focal Loss thành novelty riêng lẻ.
