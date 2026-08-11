## Phase 0 — Phân tích Dataset HRP4K

Phase 0 chỉ có một mục tiêu: **hiểu dataset một cách đầy đủ, trung lập và có hệ thống trước khi đưa ra bất kỳ giả thuyết hay phương pháp nào**. Không tối ưu analysis để chứng minh một method cụ thể.

Kết quả của phase này phải trở thành **nguồn dữ liệu nền** cho các bước sau: chọn hướng nghiên cứu, giải thích kết quả model, phân tích failure case, thiết kế ablation, viết Dataset Analysis/Discussion và giải thích vì sao một phương pháp hoạt động hoặc thất bại.

### 0.1. Dataset integrity & structure

Trước hết xác nhận dataset thực tế khớp với mô tả chính thức.

Cần kiểm tra số ảnh theo split, positive/negative images, số annotation, kích thước ảnh, category, bbox hợp lệ, duplicate annotation, bbox ngoài ảnh, bbox cực nhỏ/lớn bất thường, ảnh thiếu/hỏng và phân bố số object trên mỗi ảnh.

Đầu ra nên có một `dataset_integrity_report` với các sanity checks và các anomaly được phát hiện.

---

### 0.2. Global dataset description

Mô tả dataset ở mức tổng thể mà không gắn với model.

Các thống kê chính gồm số ảnh, số instance, positive/negative ratio, objects/image, ảnh có nhiều pothole, image resolution và distribution bbox.

Đối với bbox cần mô tả riêng:

[
w,\quad h,\quad area,\quad \frac{area}{WH},\quad \frac{w}{h}
]

với mean, median, std, P5/P10/P25/P50/P75/P90/P95 và extreme values.

Đầu ra phải đủ để trả lời những câu như:

> Một pothole điển hình trong HRP4K lớn bao nhiêu?

> Dataset có thực sự dominated by small objects không?

> Distribution có long-tail không?

---

### 0.3. Object scale analysis

Phân tích đầy đủ các nhóm:

[
\text{Ultra-fine / Fine / Medium / Large}
]

theo định nghĩa của HRP4K.

Ngoài tỷ lệ từng class, cần phân tích width, height, aspect ratio, objects/image và số ảnh chứa từng scale.

Đặc biệt nên phân biệt:

[
\text{instance distribution}
]

và:

[
\text{image distribution}
]

vì 3.000 ultra-small objects nằm trong 500 ảnh khác hoàn toàn với việc chúng phân bố trên 3.000 ảnh.

---

### 0.4. Spatial distribution

Đây là phân tích **object nằm ở đâu trong ảnh**, chưa xét tới phương pháp xử lý.

Dùng normalized coordinates:

[
x_c,\quad y_c,\quad y_{bottom}
]

để xây:

* object density map;
* horizontal/vertical marginal distributions;
* 2D grid occupancy;
* center-region vs edge-region distribution;
* vùng ảnh gần như không xuất hiện pothole.

Nên xuất raw grid dưới dạng CSV/JSON bên cạnh visualization để các analysis sau có thể sử dụng lại.

---

### 0.5. Joint Position × Scale analysis

Phần đã làm hiện tại nên được giữ nhưng coi là **một dataset property**, không phải justification cho method nào.

Phân tích:

[
P(scale\mid x,y)
]

và các quan hệ:

[
y_c \leftrightarrow area
]

[
y_c \leftrightarrow w,h
]

[
y_{bottom}\leftrightarrow area
]

bao gồm Pearson, Spearman, quantiles và conditional variance.

`y_center` và `y_bottom` phải được báo cáo riêng vì:

[
y_{bottom}=y_c+\frac{h}{2H}
]

nên `y_bottom` có coupling với bbox height.

Không nên chỉ báo median; mỗi spatial band nên có:

[
P10,P25,P50,P75,P90
]

để nhìn được overlap và uncertainty.

---

### 0.6. Shape analysis

Pothole không chỉ khác nhau về area mà còn khác về hình dạng.

Phân tích:

[
aspect=\frac{w}{h}
]

cùng width-height joint distribution.

Cần xem aspect ratio thay đổi như thế nào theo:

[
scale,\quad x,\quad y
]

và xác định các extreme elongated boxes.

Điều này về sau hữu ích khi giải thích tại sao detector miss những pothole rất thấp nhưng rộng.

---

### 0.7. Object density & co-occurrence

Phân tích cấu trúc ở mức image.

Cần biết distribution:

[
N_{objects/image}
]

và đối với ảnh nhiều pothole:

* khoảng cách giữa objects;
* mức clustering;
* số object nằm gần nhau;
* scale của các object trong cùng ảnh;
* một ảnh có đồng thời tiny + large objects hay không.

Đây là thông tin rất quan trọng cho bất kỳ analysis high-resolution nào sau này.

---

### 0.8. Positive vs Negative image analysis

HRP4K có lượng negative images đáng kể, vì vậy không nên bỏ qua.

Cần so sánh positive/negative image về:

* brightness;
* contrast;
* road-region statistics;
* texture;
* blur;
* potentially confusing structures.

Nếu chỉ dùng annotation mà chưa muốn xử lý ảnh sâu ở Phase 0 thì ít nhất phải phân tích số lượng và phân bố split của negative images.

Sau này nhóm này sẽ rất hữu ích cho false-positive analysis.

---

### 0.9. Split analysis

Train / validation / test phải được phân tích độc lập.

Không chỉ report số lượng mà phải so sánh:

[
P(scale),\quad P(position),\quad P(objects/image),\quad P(aspect)
]

và các joint distributions quan trọng.

Có thể sử dụng KS-test, Jensen–Shannon divergence hoặc các distribution distance phù hợp để phát hiện split shift.

Một kết quả như:

[
\rho_{train}=0.26,\qquad
\rho_{val}=0.52
]

không nên chỉ ghi là “generalizes”; nó phải được xem như một tín hiệu cần giải thích về distribution difference giữa các split.

---

### 0.10. Domain/subgroup analysis

Nếu metadata hỗ trợ, phân tích riêng theo:

[
city,\quad road\ material,\quad camera/route,\quad capture\ condition
]

Ví dụ HRP4K có asphalt/concrete và nhiều khu vực thu thập khác nhau, nên cần xem:

[
P(scale|\text{city})
]

[
P(position|\text{city})
]

[
P(scale|\text{road material})
]

Nếu một property chỉ xuất hiện ở một city thì đó là domain characteristic, không nên mô tả như một property chung của toàn HRP4K.

---

### 0.11. Image-quality analysis

Đây là phần nên bổ sung vì annotation geometry không mô tả hết độ khó.

Với ảnh hoặc local object region có thể tính các proxy như:

[
brightness,\quad contrast,\quad sharpness/blur
]

và nếu cần:

* over/under-exposure;
* shadow level;
* texture complexity;
* color statistics.

Sau đó phân tích chúng theo scale và position.

Điều này sau này cực kỳ hữu ích để giải thích:

> Hai pothole cùng 30×10 px nhưng tại sao một cái detect được còn một cái không?

---

### 0.12. Difficulty taxonomy

Cuối Phase 0 nên xây một taxonomy **dựa trên dataset**, chưa dựa trên bất kỳ model nào.

Ví dụ một instance có thể được mô tả bằng:

[
D_i=
(scale,\ position,\ aspect,\ image\ quality,\ local\ density,\ domain)
]

Từ đó hình thành các nhóm như:

```text
tiny + distant
tiny + near
extreme-horizontal
dense-object region
isolated object
low-contrast
blurred
asphalt
concrete
```

Đây sẽ là bộ dimension chuẩn để dùng cho tất cả experiment sau.

---

## Output structure đề xuất

Phase 0 nên tạo ra một bộ output ổn định:

| Output                         | Mục đích                         |
| ------------------------------ | -------------------------------- |
| `dataset_integrity.json`       | Sanity check                     |
| `dataset_summary.json`         | Global statistics                |
| `scale_analysis.json`          | Scale distributions              |
| `spatial_analysis.json`        | Position distributions           |
| `position_scale_analysis.json` | Joint position × scale           |
| `shape_analysis.json`          | Width/height/aspect              |
| `image_object_density.json`    | Objects/image + clustering       |
| `split_analysis.json`          | Train/val/test shift             |
| `domain_analysis.json`         | City/material/domain             |
| `image_quality_analysis.json`  | Visual difficulty proxies        |
| `difficulty_index.csv`         | Per-instance analysis table      |
| `figures/`                     | Publication-ready visualizations |
| `dataset_analysis_report.md`   | Human-readable interpretation    |

Quan trọng nhất là **không chỉ lưu hình**. Các figure phải có raw CSV/JSON tương ứng để sau này có thể query, aggregate và liên kết với prediction của model.

---

# Dataset Analysis Master Table

Tôi đặc biệt khuyến nghị tạo một record cho mỗi pothole:

```text
image_id
split
city
road_material

x_center
y_center
y_bottom

width_px
height_px
width_rel
height_rel
area_ratio
log_area
aspect_ratio
scale_class

objects_in_image
nearest_object_distance
local_object_density

brightness
contrast
sharpness
```

Sau này khi có prediction, chỉ cần nối thêm:

```text
detected
confidence
IoU
TP / FP / FN
model_name
```

thì có thể hỏi trực tiếp:

[
Recall(scale)
]

[
Recall(position)
]

[
Recall(scale,position)
]

[
Recall(blur,scale)
]

[
Error(city,material)
]

mà **không cần làm lại Dataset Analysis từ đầu**.

---

## Boundary của Phase 0

Phase 0 **không làm**:

```text
Không thiết kế architecture
Không chọn AutoFocus
Không chọn Adaptive Zoom
Không quyết định crop size
Không chứng minh một hypothesis cụ thể
Không tối ưu threshold theo model
Không dùng test data để học prior cho model
```

Phase 0 chỉ trả lời:

> **HRP4K thực sự là một dataset như thế nào, có những distribution/property/bias/difficulty nào, và chúng ta có thể dùng những dimensions nào để giải thích các experiment sau này?**

### Definition of Done

Phase 0 có thể coi là hoàn thành khi ta có thể chọn **bất kỳ một image hoặc pothole instance nào** và mô tả nó theo các dimension chuẩn; đồng thời có thể phân tích toàn dataset, từng split và từng subgroup bằng cùng một hệ thống thống kê.

Khi đó Phase 1 mới có thể bắt đầu từ dữ liệu, thay vì bắt đầu từ một method đã định trước.
