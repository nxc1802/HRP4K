**Không phải thêm “distance module” vào AutoFocus**, mà là thêm một **perspective/scale-aware module** để AutoFocus không chỉ trả lời:

> “Vùng nào có khả năng chứa object?”

mà còn trả lời:

> “Object ở vùng đó có khả năng nhỏ đến mức nào trên ảnh, và cần zoom bao nhiêu?”

Với HRP4K, cách này hợp lý hơn việc gọi trực tiếp là distance-aware, vì dataset **không có depth hay khoảng cách vật lý**. Paper cũng nói rõ kích thước pothole chỉ là *visual scale trong image plane do perspective*, không tương ứng trực tiếp với kích thước vật lý khi không có camera calibration/depth. 

## Thứ cần làm ngay bây giờ: Data Description về `position × size`

May mắn là **chưa cần annotation mới**. HRP4K đã có 7.217 bounding boxes ở YOLO/COCO format.  Từ mỗi bbox, bạn có thể derive toàn bộ dữ liệu cần thiết.

Với mỗi object, tôi sẽ tạo record như sau:

| Field          | Ý nghĩa                            |
| -------------- | ---------------------------------- |
| `image_id`     | ảnh chứa object                    |
| `x_center`     | vị trí ngang chuẩn hóa 0–1         |
| `y_center`     | vị trí dọc chuẩn hóa 0–1           |
| `y_bottom`     | đáy bbox, rất quan trọng           |
| `width`        | width / image width                |
| `height`       | height / image height              |
| `area_ratio`   | bbox area / image area             |
| `log_area`     | log(area_ratio)                    |
| `aspect_ratio` | width / height                     |
| `scale_class`  | ultra-fine / fine / medium / large |

Ví dụ:

[
x_c=\frac{x_1+x_2}{2W}
]

[
y_c=\frac{y_1+y_2}{2H}
]

[
y_b=\frac{y_2}{H}
]

[
w=\frac{x_2-x_1}{W},\qquad
h=\frac{y_2-y_1}{H}
]

và:

[
A=\frac{(x_2-x_1)(y_2-y_1)}{WH}
]

Trong bài toán của bạn, **`y_bottom` có thể còn quan trọng hơn `y_center`**.

Lý do là pothole nằm trên mặt đường. Trong camera perspective cố định tương đối:

```text
Horizon
────────────────────────
       • tiny
          • small
             • medium

                  █ large

──────────────────────── camera
y = 1
```

Object càng gần camera thường càng nằm thấp trong ảnh và có projected size lớn hơn.

Do đó bạn đang muốn học gần như:

[
P(\text{object scale}\mid x,y)
]

hoặc đơn giản hơn:

[
\hat s=f(x_c,y_b)
]

---

# Data Description quan trọng nhất nên có 4 phân tích

### 1. `Y-position vs Object Size`

Đây là figure quan trọng nhất.

Trục X:

[
y_{bottom}
]

Trục Y:

[
\log(A)
]

Mỗi điểm = một pothole.

Ví dụ kỳ vọng:

```text
log(area)
   ↑
   |                         •
   |                    • •••
   |              • ••••••
   |         • •••••
   |    •••••
   |••••
   +────────────────────────────→ y_bottom
    far                         near
```

Nếu xuất hiện correlation rõ, bạn đã có bằng chứng cho premise của method:

> **Perspective position carries information about expected object scale.**

Đây chính là nền tảng để justify module mới.

---

### 2. Chia ảnh thành các horizontal bands

Ví dụ chia:

[
y\in
[0,0.1),
[0.1,0.2),
...
[0.9,1.0]
]

Sau đó với mỗi band, report:

| y range | N objects | median W | median H | median area | ultra-fine % |
| ------- | --------: | -------: | -------: | ----------: | -----------: |
| 0.2–0.3 |         … |        … |        … |           … |            … |
| 0.3–0.4 |         … |        … |        … |           … |            … |
| 0.4–0.5 |         … |        … |        … |           … |            … |
| …       |         … |        … |        … |           … |            … |

Nếu bạn thấy chẳng hạn:

```text
Far region:
median ≈ 30 × 10

Middle:
median ≈ 90 × 30

Near:
median ≈ 250 × 100
```

thì đây là bằng chứng cực kỳ đẹp.

Paper gốc hiện mới mô tả **distribution kích thước tổng thể**: 3.833 instances ultra-fine, median bbox khoảng `100×35 px`, extreme case khoảng `10×4 px`. 

Điều paper chưa cung cấp trong phần chúng ta đang quan tâm là **joint distribution giữa spatial position và object scale**.

Đó chính là thứ bạn nên bổ sung.

---

### 3. 2D Spatial Scale Map

Cái này còn hay hơn.

Chia ảnh thành grid, chẳng hạn:

[
12\times8
]

Mỗi cell tính:

[
S_{ij}=
\operatorname{median}
\left(
A_k
\right)
]

của các pothole center rơi vào cell đó.

Bạn sẽ có:

```text
          Image

 ┌───────────────────────────┐
 │        tiny tiny          │
 │      tiny tiny tiny       │
 │    small small small      │
 │  medium medium medium     │
 │ large   large   large     │
 └───────────────────────────┘
```

Tức là tạo ra một:

[
\boxed{\text{Perspective Scale Prior Map}}
]

Đây thậm chí có thể trở thành input/prior trực tiếp cho model.

---

### 4. Position × Scale-class distribution

Dataset đã có natural scale categories dựa trên area ratio; paper báo 3.833 ultra-fine `<0.05%`, 1.078 fine `0.05–0.1%`, 1.099 medium `0.1–0.25%`, còn lại lớn hơn `0.25%`. 

Ta có thể hỏi:

[
P(
ScaleClass
\mid
y
)
]

Ví dụ:

```text
                 Ultra     Fine     Medium    Large

Far road          80%       15%       4%       1%
Mid road          45%       25%      20%      10%
Near road         10%       15%      30%      45%
```

Nếu HRP4K thực sự cho pattern như vậy, nó là justification rất mạnh cho **position-aware adaptive zoom**.

---

# Sau khi có Data Description này, module mới rất tự nhiên

AutoFocus cơ bản có:

[
Feature
\rightarrow
Objectness/FocusMap
]

Bạn nâng thành:

[
Feature
\rightarrow
\begin{cases}
H(x,y) &: \text{objectness}\
S(x,y) &: \text{expected object scale}
\end{cases}
]

Sau đó crop policy sử dụng cả hai:

[
CropPolicy(H,S)
]

Ví dụ:

```text
Scout
  │
  ├── Focus Head ─────→ Có object ở đâu?
  │
  └── Scale Head ─────→ Object có thể nhỏ bao nhiêu?
                             │
                             ↓
                       Zoom Controller
```

Policy có thể rất trực quan:

[
z_i=f(\hat s_i)
]

Object dự đoán rất nhỏ:

[
z=4\times
]

Object nhỏ:

[
z=2\times
]

Object đủ lớn:

[
z=1\times
]

hoặc thậm chí **không chạy local branch**.

Đây chính là điểm làm nó khác Dynamic Top-K hiện tại của AdaPoth-Lite, vốn đang dùng crop cố định `768×512`; proposal hiện tại mới dùng Scout → components → region NMS → Dynamic K, chứ chưa học scale-conditioned zoom. 

---

## Tôi sẽ chưa làm model ngay

Bước tiếp theo hợp lý nhất là làm một **dataset diagnostic notebook** duy nhất, đọc `train.json + valid.json + test.json` và sinh khoảng 6 outputs:

1. `y_bottom vs log(area)` scatter;
2. `y_center vs width/height`;
3. vertical-band statistics;
4. 2D object-density heatmap;
5. 2D median-scale heatmap;
6. `P(scale class | y-band)`.

Kèm các correlation:

[
\rho(y_{bottom}, \log A)
]

[
\rho(y_{bottom}, w)
]

[
\rho(y_{bottom}, h)
]

Nếu các correlation và conditional distributions đủ mạnh, **lúc đó mới có căn cứ khoa học để thiết kế Perspective-Aware AutoFocus**.

Một điểm rất quan trọng: camera của HRP4K dùng optics/capture setup tương đối chuẩn hóa, nên giả thuyết position→visual scale có cơ sở để thử.  Nhưng vì ảnh được thu từ nhiều loại xe, nếu metadata cho phép, tôi cũng sẽ kiểm tra relation này **theo camera/vehicle subgroup**, để tránh một prior học tốt trên toàn tập nhưng bị lệch bởi khác biệt camera mounting.

**Tóm lại: đúng, thứ cần nhất bây giờ chính là joint data description `vị trí × kích thước`. Và bbox annotation hiện có đã đủ để làm bước này, chưa cần depth annotation.** Nếu kết quả cho thấy một quy luật perspective rõ, đó sẽ là bằng chứng đầu tiên cho việc biến AdaPoth-Lite từ adaptive cropping đơn thuần thành **Perspective/Scale-Aware Adaptive Zoom**.
