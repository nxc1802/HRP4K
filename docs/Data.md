Hiện tại **bbox + position × size đã đủ để chứng minh premise ban đầu**. Thứ tôi cần tiếp theo không phải thêm annotation thủ công, mà là các thống kê giúp trả lời câu hỏi quan trọng hơn:

> **Position prior có đủ ổn định và đủ predictive để đưa vào Adaptive Zoom hay không?**

Tôi ưu tiên theo thứ tự này:

1. **Phân tích riêng theo train / valid / test**
   Hiện bạn đang gộp cả 3 split. Tôi cần mỗi split có:

* (\rho(y_{center}, \log area))
* (\rho(y_{bottom}, \log area))
* distribution `ScaleClass | y-band`

Mục tiêu là xem relation có generalize hay chỉ là artefact của toàn dataset.

2. **`y_center` vs size, không chỉ `y_bottom`**
   Đây là quan trọng nhất ngay lúc này, vì:

[
y_{bottom}=y_{center}+\frac{h}{2H}
]

nên correlation giữa `y_bottom` và height bị coupling một phần.

Cần report:

[
corr(y_{center}, area)
]

[
corr(y_{center}, width)
]

[
corr(y_{center}, height)
]

Nếu trend vẫn rõ thì justification perspective rất chắc.

3. **Conditional variance — không chỉ median**
   Hiện ta biết median tăng theo vị trí, nhưng Adaptive Zoom cần biết độ bất định.

Mỗi y-band nên có thêm:

* P10 / P25 / median / P75 / P90 của bbox area;
* width/height quantiles;
* standard deviation hoặc IQR.

Ví dụ nếu tại `y=0.4–0.5`:

[
P_{10}=0.005%,\quad
P_{90}=0.20%
]

thì scale overlap rất lớn → position chỉ nên là prior.

Ngược lại nếu interval khá hẹp thì có thể dùng prior mạnh hơn.

4. **Predictability experiment**

Đây mới là dữ liệu quyết định architecture.

Thử predict:

[
\log(area)
]

bằng:

[
[x,y]
]

với các model cực nhẹ:

* linear regression;
* polynomial regression;
* Random Forest / tiny MLP.

Report trên test:

* (R^2);
* MAE;
* RMSE.

Và thử:

[
[x,y]\rightarrow
{\text{ultra,fine,medium,large}}
]

report:

* accuracy;
* macro-F1;
* confusion matrix.

Nếu position-only predict được khá tốt thì Perspective Prior có giá trị thực sự.

---

5. **2D position prior: `x × y × scale`**

Hiện y là chính, nhưng road perspective không nhất thiết chỉ phụ thuộc vertical position.

Tôi muốn biết:

[
P(scale|x,y)
]

Ví dụ:

```text
left lane     center lane      right lane
 small           tiny             small
 medium         small            medium
```

Nếu center/vanishing-point region khác hai bên rõ rệt thì dùng:

[
S_{prior}(x,y)
]

sẽ tốt hơn:

[
S_{prior}(y)
]

2D heatmap bạn đã tạo là đúng hướng; cần export raw grid values và số sample/cell để biết cell nào đáng tin.

---

6. **Object density × scale**

Adaptive cropping không chỉ cần biết object nhỏ hay lớn mà còn:

> một crop có khả năng chứa bao nhiêu pothole?

Cần thống kê:

[
P(N_{objects}|region)
]

và khoảng cách giữa các bbox.

Ví dụ 3 ultra-small potholes gần nhau thì:

```text
1 crop lớn
```

có thể tốt hơn:

```text
3 crop nhỏ
```

Đây sẽ ảnh hưởng trực tiếp tới Dynamic K.

---

### Sau đó có 3 nhóm “nice to have”

Nếu dataset có thể suy ra được, tôi cũng muốn:

* **aspect ratio vs position**: pothole xa có bị “flatten” mạnh hơn không;
* **image-level distribution**: số pothole/image, đặc biệt positive vs negative;
* **camera/domain consistency**: nếu có vehicle/camera/route metadata, kiểm tra position–scale relation theo từng subgroup.

Cái cuối rất giá trị. Nếu:

[
P(scale|x,y,\text{camera A})
\neq
P(scale|x,y,\text{camera B})
]

thì một fixed Perspective Prior toàn dataset có thể generalize kém.

---

## Nếu chỉ làm thêm một vòng analysis

Tôi sẽ không làm quá nhiều. Chỉ cần bổ sung **5 outputs**:

1. `y_center vs log(area)` + correlation.
2. Correlation riêng `train / val / test`.
3. Quantile table P10/P25/P50/P75/P90 theo y-band.
4. Position-only regression/classification benchmark.
5. Raw `12×8` grid cho `count + median scale + IQR`.

Sau 5 phân tích này, chúng ta có được kết luận thực nghiệm rõ ràng về cách vị trí không gian điều kiện hóa kích thước thị giác ($P(\text{scale} \mid x, y)$):

* Vị trí không gian ($y_{center}, y_{bottom}$) tạo nên một **Prior định hướng mạnh** ($\text{Far} \to \text{Ultra-fine}$, $\text{Near} \to \text{Large}$).
* Tuy nhiên, biến thiên kích thước cục bộ vẫn đáng kể (Position-only $R^2 \approx 0.125$), chứng minh rằng **vị trí không gian hoạt động như một Prior định hướng (Position Prior), cần kết hợp với đặc trưng thị giác cục bộ (Visual Feature)** để giải thích chính xác kích thước đối tượng.

