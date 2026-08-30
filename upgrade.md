# 2. Vấn đề lớn nhất hiện tại không hẳn nằm ở backbone

Code hiện tại có một số điểm khiến tôi nghĩ **80% chưa phản ánh giới hạn thật của Scout**.

### A. GT đang là Gaussian, không phải expanded region

`generate_raw4k_scout_gt()` tạo Gaussian quanh center của bbox; `expand_ratio=0.20` chỉ làm Gaussian rộng hơn.

Trong khi mục tiêu thật của Scout là:

> **ROI phải cover object.**

Đây là hai objective khác nhau.

Model có thể học rất tốt:

```text
"center của pothole nằm ở đây"
```

nhưng vẫn tạo ROI không đủ rộng để cover 75% object.

**Tôi sẽ sửa objective theo coverage trước**, thay vì cố làm heatmap giống CenterNet.

---

### B. `coverage_loss` hiện tại chưa thực sự là coverage loss

Code hiện tại chỉ lấy prediction tại **center pixel** rồi phạt nếu:

$$
p_{center}<0.90
$$

Điều này thực chất là:

> **Center confidence loss**

chứ chưa phải:

> **ROI coverage loss**.

Đây là một điểm tôi đánh giá **rất đáng sửa**.

Scout cần học:

$$
\boxed{\text{“bao phủ object”}}
$$

không chỉ:

$$
\text{“đánh peak tại center”}.
$$

---

# 3. Tôi sẽ nâng cấp Global theo thứ tự này

## Upgrade 1 — Đổi target thành "coverage-aware target"

Thay vì Gaussian thuần túy:

```text
          Gaussian
             ↓
       center peak
```

dùng:

```text
GT bbox
   ↓
expand 20–30%
   ↓
soft region target
```

Tức là toàn bộ vùng mà ROI cần cover đều nhận supervision.

Mục tiêu:

$$
B_{GT}\subseteq R_{Scout}
$$

thay vì chỉ:

$$
center(B_{GT})\rightarrow high\ score.
$$

---

## Upgrade 2 — Coverage loss thật sự

Thay vì:

```text
prediction tại center
```

tính trực tiếp mức activation trên GT region.

Ví dụ:

$$
L_{cov}
=
\max(0,\tau-\operatorname{mean}(H_{GT}))
$$

hoặc mạnh hơn:

$$
L_{cov}
=
1-\frac{|GT\cap R|}{|GT|}
$$

sau bước differentiable region selection.

**Đây là upgrade tôi ưu tiên số 1.**

---

# 4. Upgrade 3 — Đừng để threshold `0.05` quyết định quá nhiều

Current candidate generator:

```text
threshold = 0.05
effective threshold = max(0.05, 0.15 × max_heatmap)
```

Điều này khá heuristic.

Đặc biệt khi heatmap có:

```text
max = 0.20
```

thì threshold chỉ:

$$
0.05
$$

hoặc:

$$
0.15\times0.20=0.03
$$

→ vẫn dùng 0.05.

Trong khi image khác:

```text
max = 0.95
```

→ threshold:

$$
0.1425.
$$

Như vậy **cùng một Scout nhưng effective threshold thay đổi khá mạnh theo confidence calibration của từng ảnh**.

Tôi sẽ chuyển sang:

> **Top-K peak / percentile-based candidate extraction**

hoặc calibration threshold trên validation set.

---

# 5. Upgrade 4 — Crop hiện tại cần được đánh giá bằng GT coverage, không phải candidate overlap

Code hiện tại định nghĩa:

$$
Coverage=
\frac{|GT\cap ROI|}{|GT|}
$$

cái này đúng.

Nhưng **False Region Rate** lại chỉ kiểm tra:

> ROI có giao với GT hay không.

Chỉ cần overlap 1 pixel cũng được coi là có GT.

Điều này quá dễ.

Nên sửa thành:

$$
IoU(R,GT)>\tau
$$

hoặc tốt hơn với Scout:

$$
\frac{|R\cap GT|}{|GT|}>\tau.
$$

Tức là false ROI chỉ được coi là useful nếu **thực sự cover đủ object**.

---

# 6. Upgrade 5 — Phân tích riêng Ultra-fine

Đây là thứ tôi rất muốn làm với kết quả 80%.

Code đã có `scale_bin_recalls`.

Nhưng cần xem:

```text
Ultra-fine     ?
Fine           ?
Medium         ?
Large          ?
```

Nếu kết quả là:

```text
Large      97%
Medium     94%
Fine       87%
UltraFine  65%
```

thì **80% overall không phải do Scout architecture yếu**.

Nó cho thấy:

> Raw 4K → Stem+S1/S2 vẫn chưa giữ đủ information cho ultra-fine objects.
