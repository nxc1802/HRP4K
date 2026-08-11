# Phase 3 — **Deep Analysis & Explanation of HRP4K Small-Object Benchmark**

Phase 3 tập trung 100% vào phân tích chuyên sâu, so sánh và diễn giải kết quả. Mục tiêu chuyển từ:

> **“Method nào đạt mAP cao nhất?”**

sang:

> **“Tại sao từng method thắng/thua trên HRP4K, trong điều kiện dataset nào, đối với loại pothole nào, và với cái giá compute bao nhiêu?”**

Đây nên là phase biến kết quả của **Phase 1 — detector benchmark** và **Phase 2 — small-object processing benchmark** thành một nghiên cứu có giá trị giải thích.

HRP4K đặc biệt phù hợp cho kiểu phân tích này vì dataset có 7.217 potholes với phân bố scale rất lệch: 3.833 ultra-fine, 1.078 fine, 1.099 medium và 1.207 large; median box chỉ khoảng `100×35 px`, còn những instance cực nhỏ xuống tới khoảng `10×4 px`.  Đồng thời dataset có perspective cố định tương đối rõ, pavement imbalance lớn và một tập negative riêng đủ để nghiên cứu false alarm.  

---

# 1. Vai trò của Phase 3 trong toàn project

Pipeline tổng thể nên trở thành:

```text
Phase 1
Detector Benchmark
YOLO / RT-DETR / D-FINE
        ↓
Phase 2
Small-object Processing Benchmark
Resize / Tiling / SAHI / AutoFocus
AdaZoom / FOVEA / TPP / ZoomDet
        ↓
Phase 3
Dataset-conditioned Analysis
        ↓
Why?
When?
Where?
For which object?
At what computational cost?
        ↓
Final conclusions about
4K perspective-view small-object detection
```

Tức **Phase 3 không cố tạo SOTA mới**.

Contribution của Phase 3 nằm ở:

> **benchmark interpretation + systematic diagnosis + dataset-conditioned explanation.**

---

# 2. Nguyên tắc quan trọng: Explanation không được chỉ là suy đoán

Mỗi conclusion nên có 4 tầng bằng chứng:

```text
Dataset fact
    ↓
Observed benchmark pattern
    ↓
Hypothesis
    ↓
Controlled analysis / statistical evidence
    ↓
Supported explanation
```

Ví dụ không nên chỉ viết:

> “SAHI tốt hơn Resize vì pothole nhỏ.”

Nên chứng minh thành:

```text
FACT:
53.1% potholes là ultra-fine.

OBSERVATION:
SAHI tăng +X AP_UF
nhưng chỉ +Y AP_L.

MECHANISM:
Slicing làm tăng effective pixel size của target.

EVIDENCE:
Gain có tương quan nghịch với original object area.

CONCLUSION:
Phần lớn gain của SAHI trên HRP4K đến từ
resolution recovery cho ultra-small targets,
không phải improvement đồng đều trên mọi scale.
```

Các số liệu `X`, `Y` chỉ được điền sau Phase 2.

---

# 3. Xây một **Dataset → Challenge → Method** framework

Đây nên là khung trung tâm của Phase 3.

| Đặc điểm HRP4K                 | Challenge                    | Phân tích cần làm            |
| ------------------------------ | ---------------------------- | ---------------------------- |
| 3840×2160                      | Downsampling mất detail      | Resolution sensitivity       |
| 53% ultra-fine                 | Small-object recall          | AP/Recall by scale           |
| box median 100×35              | Localization khó             | AP50 vs AP75/AP90            |
| perspective-view               | scale thay đổi theo vị trí   | Spatial/perspective analysis |
| 87.8% asphalt / 12.2% concrete | material bias                | Material robustness          |
| concrete joints/textures       | false positive               | FP taxonomy                  |
| 2.000 negatives                | deployment false alarms      | FPPI analysis                |
| đa số ảnh chỉ 1 pothole        | sparse scenes                | object-density analysis      |
| background clutter             | context confusion            | failure analysis             |
| daylight, mostly dry           | limited environmental domain | generalization limitation    |

Các fact về scale, pavement distribution, acquisition và negative samples đều trực tiếp được dataset paper công bố.   

---

# 4. Analysis A — **Resolution Sensitivity**

Đây nên là analysis quan trọng nhất.

Từ Resize-only sweep:

```text
640
960
1280
1920
```

vẽ:

```text
Input Resolution
      ↓
Overall AP
AP_UF
AP_F
AP_M
AP_L
Recall_UF
```

Điều cần tìm không phải chỉ là resolution nào tốt nhất.

Ta cần trả lời:

### RQ1

**Gain từ việc tăng resolution tập trung vào scale nào?**

Giả thuyết:

```text
Ultra-fine : gain rất lớn
Fine       : gain lớn
Medium     : gain vừa
Large      : gain nhỏ
```

Nếu thực nghiệm đúng pattern này thì nó trực tiếp hỗ trợ lập luận rằng **HRP4K là resolution-limited benchmark**.

Điều này đặc biệt có cơ sở vì paper cho thấy hơn một nửa object thuộc ultra-fine và median box chỉ `100×35 px`. 

---

# 5. Quantify **effective object size after resize**

Đây là một analysis rất đáng thêm.

Ví dụ ảnh:

```text
3840×2160
        ↓
960×540
```

là downsample khoảng 4× mỗi chiều.

Một pothole:

```text
100×35 px
```

sẽ còn xấp xỉ:

```text
25×9 px
```

trong input detector.

Một pothole cực nhỏ:

```text
10×4 px
```

thậm chí xuống chỉ khoảng:

```text
2.5×1 px
```

Đây là phép suy ra trực tiếp từ kích thước ảnh/object mà paper báo cáo, không phải thống kê mới của paper. 

Ta nên tính cho **từng GT instance**:

```text
original_width
original_height

effective_width_at_640
effective_height_at_640

effective_width_at_960
effective_height_at_960
...
```

Sau đó plot:

```text
Recall
  ↑
  │
  │
  │
  └──────────────────→ effective bbox height
```

Nếu recall tăng nhanh sau một threshold pixel nhất định, ta có thể định lượng:

> detector bắt đầu thất bại mạnh khi pothole chỉ còn khoảng N pixels chiều cao.

Đây có thể là một trong những insight mạnh nhất toàn benchmark.

---

# 6. Analysis B — **Scale-conditioned Method Ranking**

Thay vì một bảng:

```text
Method → mAP
```

hãy có:

```text
Method × Scale
```

Ví dụ:

| Method    | AP_UF | AP_F | AP_M | AP_L | Overall |
| --------- | ----: | ---: | ---: | ---: | ------: |
| Resize    |       |      |      |      |         |
| Uniform   |       |      |      |      |         |
| SAHI      |       |      |      |      |         |
| AutoFocus |       |      |      |      |         |
| AdaZoom   |       |      |      |      |         |
| FOVEA     |       |      |      |      |         |
| TPP       |       |      |      |      |         |
| ZoomDet   |       |      |      |      |         |

Sau đó phân tích **ranking inversion**.

Ví dụ có thể xảy ra:

```text
Overall:
ZoomDet > SAHI > Resize

Ultra-fine:
SAHI > ZoomDet >> Resize

Large:
Resize ≈ ZoomDet > SAHI
```

Nếu vậy, conclusion không phải:

> “Method A tốt nhất.”

mà là:

> “Method A tốt nhất cho aggregate benchmark, trong khi Method B vẫn là lựa chọn tốt nhất cho ultra-fine objects.”

Đây là phân tích sâu hơn rất nhiều so với standard leaderboard.

---

# 7. Analysis C — **Where does improvement come from?**

Cho mỗi method (M), tính:

[
\Delta AP_s =
AP_s(M)-AP_s(\text{Resize})
]

với:

```text
s ∈ {UF, F, M, L}
```

Rồi normalize contribution:

```text
Ultra-fine contribution
Fine contribution
Medium contribution
Large contribution
```

Mục tiêu là xác định:

> tổng improvement của method được tạo ra ở đâu.

Ví dụ:

```text
Total gain +4.2 AP

Ultra-fine      +8.4
Fine            +4.1
Medium          +1.3
Large           -0.2
```

Conclusion mạnh hơn rất nhiều:

> Method không “cải thiện detection nói chung”; nó chủ yếu giải quyết đúng bottleneck ultra-small của HRP4K.

---

# 8. Analysis D — **Localization difficulty**

Paper đã quan sát một gap đáng kể giữa `mAP@0.5` và `mAP@0.5:0.95`; ví dụ D-FINE đạt `0.611` tại AP50 nhưng chỉ `0.383` tại AP50:95. Authors giải thích rằng với ultra-small objects, sai lệch vài pixel đã làm IoU giảm mạnh. 

Phase 3 nên mở rộng analysis này.

Thay vì chỉ:

```text
AP50
AP50:95
```

thêm:

```text
AP50
AP60
AP70
AP75
AP80
AP90
```

và tách theo scale.

Có thể xuất:

```text
IoU threshold
    ↑

AP_UF ─────────\
                \
AP_F  ──────────\
                  \
AP_L  ───────────────
```

Nếu ultra-fine AP sụt rất nhanh khi threshold tăng:

> bottleneck không chỉ là **finding the pothole**, mà còn là **precise localization**.

---

# 9. Tách **Detection failure** và **Localization failure**

Mỗi GT có thể được phân loại:

```text
Correct
    IoU ≥ 0.5

Localization error
    predicted near GT
    nhưng IoU < threshold

Miss
    không có prediction tương ứng

Duplicate
    nhiều prediction cho cùng GT
```

Từ đó biết method nào:

```text
finds more objects
```

và method nào:

```text
boxes them more accurately
```

Điều này rất quan trọng khi so:

```text
SAHI
vs
Zoom/warp methods
vs
D-FINE
```

vì cùng tăng AP nhưng mechanism có thể khác hoàn toàn.

---

# 10. Analysis E — **Perspective & Spatial Analysis**

HRP4K được chụp bằng forward-facing vehicle cameras ở độ cao khoảng `1.55–1.70 m`; vùng pavement trung tâm thường nằm cách camera khoảng `3.0–4.5 m`, với downward pitch ước tính khoảng `19–30°`. 

Do đó không gian ảnh không đồng nhất.

Chia image thành các horizontal bands:

```text
Top
│ Far road
│
├──────── band 1
│
├──────── band 2
│
├──────── band 3
│
├──────── band 4
│
│ Near road
Bottom
```

Sau đó tính:

```text
GT density
median bbox area
median bbox width
median bbox height
Recall
AP
```

cho từng band.

---

# 11. Spatial heatmap

Generate hai heatmap từ toàn test set:

### GT center heatmap

```text
Where are potholes located?
```

### Error heatmap

```text
Where are potholes missed?
```

Sau đó cho từng method:

```text
Recall(x,y)
```

Ta có thể phát hiện pattern như:

```text
far-field / upper pavement
       → nhỏ hơn
       → resize failure

near-field
       → larger
       → resize already sufficient
```

Nếu Two-Plane Perspective Prior hoặc một perspective-aware method cải thiện chủ yếu ở vùng xa thì đây là evidence rất trực tiếp cho mechanism của method.

---

# 12. Scale × Perspective interaction

Không dừng ở:

```text
AP_UF
```

mà phân chia tiếp:

```text
Ultra-fine / Far
Ultra-fine / Mid
Ultra-fine / Near
```

Bởi hai pothole cùng thuộc `<0.05%` chưa chắc có cùng difficulty.

Ta có thể tạo:

```text
             Far    Mid    Near
Ultra-fine
Fine
Medium
Large
```

để xem:

> difficulty đến từ **scale**, **perspective position**, hay cả hai.

---

# 13. Analysis F — **Pavement Material**

HRP4K rất mất cân bằng:

```text
Asphalt   5269 = 87.8%
Concrete   734 = 12.2%
```



Paper đã quan sát suy giảm lớn trên concrete cho cả YOLOv11 và D-FINE. Authors đưa ra hai explanation: imbalance khoảng 7× và texture/linear joints của concrete có thể giống boundary của pothole. 

Phase 3 nên mở rộng cho **tất cả methods**.

Tính:

```text
AP_asphalt
AP_concrete

Recall_asphalt
Recall_concrete

FPPI_asphalt
FPPI_concrete
```

và:

[
Gap_{material}
==============

AP_{asphalt}-AP_{concrete}
]

---

# 14. Scale × Material

Điểm này còn thú vị hơn.

Ví dụ:

```text
                 Asphalt    Concrete
Ultra-fine
Fine
Medium
Large
```

Từ đó biết:

> concrete khó vì texture nói chung, hay concrete đặc biệt gây khó cho **small potholes**?

Nếu:

```text
AP_UF concrete << AP_UF asphalt
```

nhưng:

```text
AP_L concrete ≈ AP_L asphalt
```

thì interaction giữa **texture ambiguity + low resolution** có thể là vấn đề trung tâm.

---

# 15. Analysis G — **Negative Set & False Positive Explanation**

HRP4K có 2.000 negative images và test subset giữ riêng 300 negative images để đánh giá FPPI. 

Đây là tài nguyên cực kỳ có giá trị cho slicing/zoom benchmark.

Với từng method:

```text
FPPI
FP/image distribution
% images with ≥1 FP
% images with ≥2 FP
max FP/image
```

Đặc biệt cần xem:

```text
Resize
vs
Tiling
vs
SAHI
```

vì exhaustive processing quan sát nhiều high-resolution texture hơn và có nhiều detector calls hơn, nên **có khả năng** làm tăng cơ hội sinh false positive. Đây là hypothesis cần kiểm chứng bằng benchmark, không nên giả định trước.

---

# 16. False-positive taxonomy

Manually review hoặc semi-automatically cluster FP thành:

```text
crack
concrete joint
tar repair
shadow
water / dark patch
rough texture
road marking
object/background clutter
unknown
```

Paper chỉ annotation class pothole; các loại distress khác như cracks/rutting không được annotate. 

Sau đó tạo:

```text
               Resize SAHI Tiling Zoom...
Crack
Joint
Shadow
Patch
Texture
...
```

Đây sẽ trả lời:

> method nào đổi **false-positive profile**, không chỉ FPPI tổng.

---

# 17. Analysis H — **Object Density**

HRP4K không phải dense detection dataset.

Trong 4.003 positive images:

* 2.594 ảnh, tức 64.8%, có một pothole;
* 1.200 ảnh có 2–4;
* 209 ảnh có ≥5;
* chỉ một ảnh đạt maximum 28 potholes. 

Do đó chia:

```text
Sparse:
1 object

Moderate:
2–4

Dense:
≥5
```

Rồi đo:

```text
Recall
AP
latency
number of crops
duplicates
```

Đặc biệt quan trọng cho adaptive methods:

```text
AutoFocus
AdaZoom
```

vì số regions/crops có thể phụ thuộc số objects.

---

# 18. Accuracy–compute **conditional analysis**

Không chỉ plot:

```text
AP vs latency
```

cho toàn dataset.

Cần plot riêng:

```text
AP_UF vs latency
AP_F vs latency
AP_concrete vs latency
AP_far-field vs latency
```

Điều này có thể dẫn tới conclusion kiểu:

> SAHI cho accuracy tốt nhất đối với ultra-fine targets nhưng compute tăng đáng kể.

Trong khi:

> adaptive warp đạt ít AP_UF hơn một chút nhưng nằm trên Pareto frontier về latency.

Hoặc kết quả có thể ngược lại — Phase 3 phải để data quyết định.

---

# 19. **Compute amplification factor**

Định nghĩa baseline Resize cost:

[
C_R=1
]

và:

[
CAF_M =
\frac{C_M}{C_R}
]

Ví dụ:

```text
Resize       1.0×
2×2 tile     ~4×
3×3 tile     ~9×
SAHI         N×
...
```

Nhưng nên tính từ **measured FLOPs / detector calls / latency**, không chỉ suy ra lý thuyết.

Sau đó:

[
EfficiencyGain =
\frac{\Delta AP_{UF}}{CAF-1}
]

không nhất thiết dùng đây làm headline metric, nhưng rất hữu ích cho analysis.

---

# 20. Accuracy gain per additional pixel processed

Vì bản chất Phase 2 là resolution allocation, metric rất hợp lý là:

[
G_{pixel}
=========

\frac{\Delta AP_{UF}}
{\text{additional processed pixels}}
]

Từ đó:

```text
Uniform tiling
```

có thể đạt AP cao nhưng pixel efficiency thấp;

trong khi:

```text
adaptive zoom
```

có thể đạt AP thấp hơn chút nhưng phân bổ pixel hiệu quả hơn.

Đây chính là cách giải thích **tại sao các phương pháp adaptive processing đạt được sự cân bằng hiệu năng**.

---

# 21. Analysis I — **Architecture × Processing interaction**

Phase 1 đã có:

```text
YOLO
RT-DETR
D-FINE
```

Phase 2 có:

```text
Resize
Tiling
SAHI
...
```

Không cần chạy Cartesian product toàn bộ vì quá đắt.

Nhưng nên chọn 2–3 representative detectors:

```text
YOLO11
D-FINE

optional:
lightweight YOLO
```

rồi chạy:

```text
Detector × Processing Method
```

Mục tiêu:

> small-object processing có phải architecture-agnostic hay chỉ đặc biệt hiệu quả với một detector?

Ví dụ:

```text
SAHI gain:
YOLO       +X
D-FINE     +Y
```

Nếu cả hai đều tăng mạnh:

> bottleneck nhiều khả năng nằm ở input resolution.

Nếu chỉ YOLO tăng:

> interaction với detector architecture đáng kể.

---

# 22. Analysis J — **Precision–Recall behavior**

Official benchmark đã cho thấy architecture trade-off khá rõ:

* YOLOv11: Precision `0.742`, FPPI `0.030`, mAP50:95 `0.407`;
* RT-DETRv1: F1 `0.598`;
* RT-DETRv2: Recall `0.570`;
* D-FINE: mAP50 `0.611`. 

Phase 3 nên xem Phase 2 methods **dịch chuyển operating point** như thế nào.

Không chỉ báo cáo:

```text
Precision
Recall
```

mà plot full:

```text
Precision–Recall curve
```

Ví dụ tiling có thể:

```text
↑ recall
↓ precision
```

trong khi adaptive warp có thể:

```text
↑ recall
precision gần giữ nguyên
```

Đây sẽ cho explanation tốt hơn một con số mAP.

---

# 23. Statistical significance

Nếu Phase 2 chạy 3 seeds:

```text
mean ± std
```

chưa đủ.

Cho các comparison quan trọng nên thêm:

```text
bootstrap 95% CI
```

trên test images.

Ví dụ:

[
\Delta AP_{UF}
==============

## AP_{UF}^{SAHI}

AP_{UF}^{Resize}
]

báo cáo:

```text
ΔAP_UF = ...
95% CI [...]
```

Mục tiêu là phân biệt:

```text
+0.3 AP
```

do noise hay improvement đáng tin.

Dataset paper cũng sử dụng bootstrap trong technical validation của annotation consistency; ba annotator trên 200 ảnh đạt mean IoU 0.81. 

---

# 24. Per-image paired analysis

Một kỹ thuật rất mạnh là không chỉ so aggregate AP.

Với cùng một image:

```text
Resize prediction
vs
SAHI prediction
vs
ZoomDet prediction
...
```

Tạo các nhóm:

```text
Resize fail → SAHI success

Resize fail → Zoom success

SAHI fail → Resize success

All fail

All success
```

Sau đó xem dataset characteristics của từng nhóm:

```text
bbox size
vertical position
material
object count
```

Đây là cách trực tiếp nhất để tìm:

> **“Method B giải quyết chính xác những sample nào mà Method A không giải quyết được?”**

---

# 25. Difficulty modeling

Sau khi có đủ results, có thể xây một **analysis model**, không phải detection model.

Ví dụ logistic regression:

[
P(\text{detected})
==================

f(
bbox\ area,
bbox\ height,
y\ position,
material,
object\ count
)
]

cho từng method.

Không dùng nó để dự đoán pothole.

Dùng nó để giải thích:

```text
Which dataset factors drive detection failure?
```

Ta có thể so coefficient/effect size giữa methods.

Ví dụ:

```text
Resize:
strong negative effect from tiny bbox

SAHI:
much weaker effect from tiny bbox
```

Đây là evidence rất mạnh rằng SAHI thực sự làm giảm **scale sensitivity**.

---

# 26. Một metric mới rất hữu ích: **Scale Sensitivity**

Không cần tuyên bố là metric chuẩn mới, chỉ dùng như analysis statistic.

Ví dụ:

[
SS =
AP_L-AP_{UF}
]

Nếu:

```text
Resize:
SS = 40 AP

SAHI:
SS = 18 AP
```

thì SAHI không chỉ tăng overall mAP mà còn:

> giảm performance disparity giữa large và ultra-fine objects.

Tương tự:

[
MaterialGap
===========

AP_{asphalt}-AP_{concrete}
]

và:

[
PerspectiveGap
==============

AP_{near}-AP_{far}
]

Khi đó mỗi method có một **robustness profile**.

---

# 27. Radar không nên dùng — dùng **Gap Profile**

Thay vì radar chart khó đọc:

```text
Method
│
├── Scale gap
├── Material gap
├── Perspective gap
├── FPPI
├── AP
└── Compute
```

biểu diễn bằng normalized bar/heatmap.

Ví dụ:

```text
                ScaleGap MaterialGap PerspectiveGap FPPI
Resize             ████      ██          ████       █
SAHI               ██        ██          ██         ███
TPP                 ██        ██          █          ██
...
```

Nó cho thấy **strength/weakness** từng strategy.

---

# 28. Qualitative analysis phải được chọn có hệ thống

Không cherry-pick 4 ảnh đẹp.

Tự động chọn:

```text
Best gains
Worst regressions
Ultra-fine successes
Localization failures
Concrete failures
Far-field failures
False positives
Crop boundary errors
Duplicate errors
```

Mỗi figure dùng cùng một image và đặt prediction của methods cạnh nhau:

```text
GT

Resize
Uniform
SAHI
AutoFocus
Adaptive Zoom
```

Nhìn vào một ảnh duy nhất có thể thấy mechanism khác nhau rõ hơn.

---

# 29. Xây **Failure Taxonomy**

Sau Phase 3, toàn bộ error nên được đưa về taxonomy:

```text
Failure
│
├── Resolution
│   ├─ object vanished after resize
│   └─ insufficient boundary detail
│
├── Localization
│   └─ low IoU on thin/small pothole
│
├── Perspective
│   └─ far-field target
│
├── Appearance
│   ├─ concrete texture
│   ├─ shadow
│   ├─ water
│   └─ crack / repair
│
├── Processing
│   ├─ crop truncation
│   ├─ slice boundary
│   └─ warp distortion
│
└── Fusion
    ├─ duplicates
    └─ suppression error
```

Sau đó tính tỷ lệ từng loại nếu có thể.

---

# 30. Explanation riêng cho từng family

Phase 3 cuối cùng nên tạo explanation ở **family level**, không chỉ từng model.

### Resize-only

Phải trả lời:

> Resolution loss nghiêm trọng tới mức nào, và scale nào chịu ảnh hưởng mạnh nhất?

### Uniform tiling

> Accuracy ceiling của brute-force high-resolution processing là bao nhiêu, và compute phải trả bao nhiêu?

### SAHI

> Overlap/slicing cải thiện ultra-fine recall bao nhiêu và có gây FP/duplicate overhead không?

### AutoFocus / AdaZoom

> Region selection có giữ được phần lớn benefit của exhaustive high-resolution processing với ít regions hơn không?

### FOVEA / TPP / ZoomDet

> Non-uniform resolution allocation có đạt được trade-off tốt hơn multi-pass inference không, và improvement có tập trung ở far/small targets đúng như cơ chế kỳ vọng không?

Không cần method nào thắng tuyệt đối.

---

# 31. Quan trọng nhất: **Method suitability map**

Final analysis nên kết thúc bằng dạng:

```text
User priority
        ↓

Maximum AP_UF
        → Method A

Low latency
        → Method B

Low FPPI
        → Method C

Concrete robustness
        → Method D

Far-field detection
        → Method E

Simple implementation
        → Resize / SAHI

Accuracy regardless of compute
        → exhaustive tiling
```

Tên method thực tế chỉ điền sau khi benchmark hoàn tất.

Đây hữu ích hơn câu:

> “X là SOTA.”

---

# 32. Dataset limitations phải xuất hiện trong explanation

Không được biến kết quả HRP4K thành universal conclusion.

Paper nêu rõ HRP4K:

* chỉ được thu tại ba thành phố thuộc Zhejiang;
* capture vào daylight và predominantly dry conditions;
* sử dụng forward-facing oblique perspective;
* không có aerial/nadir view;
* scale là projected visual area chứ không phải kích thước pothole vật lý. 

Do đó wording cuối cùng nên là:

> “On HRP4K and similar forward-facing high-resolution road imagery...”

thay vì:

> “For small-object detection in general...”

Đây là giới hạn khoa học rất quan trọng.

---

# 33. Source-code cho Phase 3

Tôi đề xuất thêm riêng:

```text
hrp4k/
└── analysis/
    ├── aggregate_results.py
    │
    ├── scale_analysis.py
    ├── resolution_analysis.py
    ├── localization_analysis.py
    ├── spatial_analysis.py
    ├── perspective_analysis.py
    ├── material_analysis.py
    ├── density_analysis.py
    ├── negative_analysis.py
    │
    ├── efficiency_analysis.py
    ├── pareto_analysis.py
    │
    ├── paired_analysis.py
    ├── statistical_tests.py
    │
    ├── error_taxonomy.py
    └── qualitative_gallery.py
```

Quan trọng là Phase 3 phải đọc **prediction JSON đã lưu từ Phase 1/2**, không inference lại model trừ khi thiếu data.

---

# 34. Mỗi prediction cần lưu metadata

Prediction result nên đi cùng:

```json
{
  "image_id": 123,
  "method": "SAHI",
  "latency_ms": 82.4,
  "detector_calls": 6,
  "processed_pixels": 5529600,
  "predictions": []
}
```

Dataset analysis database bổ sung:

```json
{
  "image_id": 123,
  "city": "...",
  "material": "concrete",
  "object_count": 2,
  "objects": [
    {
      "scale": "ultra-fine",
      "area_ratio": 0.00023,
      "width": 52,
      "height": 17,
      "cx_norm": 0.48,
      "cy_norm": 0.62
    }
  ]
}
```

Như vậy analysis có thể thực hiện mà không phụ thuộc framework detector.

---

# 35. Các figure bắt buộc của Phase 3

Output cuối cùng nên có khoảng **12–15 figure thật sự hữu ích**, thay vì hàng chục chart nhỏ:

```text
01 resolution_vs_scale_ap
02 effective_object_size_vs_recall

03 method_by_scale_heatmap
04 ap_gain_decomposition

05 iou_threshold_decay
06 localization_error_by_scale

07 spatial_gt_heatmap
08 spatial_recall_heatmap

09 material_gap
10 negative_fp_taxonomy

11 accuracy_compute_pareto
12 apuf_compute_pareto

13 paired_success_failure_examples
14 failure_taxonomy
15 method_suitability_summary
```

---

# 36. Final Phase 3 report structure

Paper/report cuối có thể được tổ chức:

```text
1. HRP4K Dataset Characteristics

2. Benchmark Setup
   ├─ detector baselines
   └─ small-object processing methods

3. Overall Benchmark

4. Scale-aware Analysis

5. Resolution Sensitivity

6. Localization Analysis

7. Perspective-aware Analysis

8. Pavement-material Robustness

9. Negative-set / False-positive Analysis

10. Accuracy–Efficiency Analysis

11. Failure Analysis

12. Discussion
    ├─ Why methods behave differently
    ├─ When each strategy is useful
    └─ Dataset limitations

13. Practical Recommendations

14. Conclusion
```

---

# 37. Các research questions chính của Phase 3

Tôi sẽ chốt Phase 3 quanh **6 câu hỏi**:

1. **Resolution:** Hiệu năng HRP4K bị giới hạn bởi downsampling tới mức nào?
2. **Scale:** Method nào thực sự cải thiện ultra-fine potholes thay vì chỉ aggregate AP?
3. **Localization:** Small potholes thất bại chủ yếu do miss detection hay inaccurate localization?
4. **Perspective/material:** Vị trí trong ảnh và pavement type ảnh hưởng đến các method như thế nào?
5. **Compute:** Accuracy tăng thêm có tương xứng với số pixel/FLOPs/latency tăng thêm không?
6. **Method selection:** Không có một method universally best thì strategy nào phù hợp với từng deployment requirement?

---

# 38. Đóng góp khoa học của Dự án

Dự án này là:

> **Một systematic benchmark & diagnostic study hoàn chỉnh về small-object detection trên ảnh road 4K, phân tích đồng thời detector architecture, resolution allocation, visual scale, perspective, pavement material, localization quality, false alarms và computational efficiency.**

Đóng góp của dự án được định hình qua ba trụ cột chính:

```text
1. Comprehensive Benchmark Suite
   Đánh giá công bằng các mô hình phát hiện và các kỹ thuật phân bổ độ phân giải.

2. Dataset-Conditioned Diagnosis
   Chẩn đoán chuyên sâu theo đặc tính dữ liệu (scale, perspective, material, negatives).

3. Mechanistic Explanation & Decision Matrix
   Giải thích cơ chế tại sao / khi nào / ở đâu mỗi chiến lược thành công hoặc thất bại, cùng chi phí compute tương ứng.
```

---

## Final Roadmap Tổng thể

```text
PHASE 0: Dataset Analysis & Integrity
(Mô tả đặc tính hình học, phân bố scale, vị trí không gian, chất liệu mặt đường)
          ↓
PHASE 1: HRP4K Detector Baseline
(Tái lập và đánh giá chuẩn hóa YOLOv5, YOLOv8, YOLOv11, RT-DETRv1, RT-DETRv2, D-FINE)
          ↓
PHASE 2: HRP4K Small-Object Resolution Allocation Benchmark
(Thử nghiệm Resize, Uniform Tiling, SAHI, AutoFocus, AdaZoom, FOVEA, Two-Plane Prior, ZoomDet)
          ↓
PHASE 3: Deep Dataset-Conditioned Analysis & Explanation
(Phân tích độ nhạy độ phân giải, scale, localization, perspective, material, false alarms, efficiency, failure taxonomy, và lập bảng tư vấn ứng dụng)
          ↓
LỘ TRÌNH ĐÁNH GIÁ VÀ GIẢI THÍCH TOÀN DIỆN HRP4K
```

Lộ trình **Phase 0 → Phase 1 → Phase 2 → Phase 3** xây dựng một **bộ benchmark + chẩn đoán khoa học hoàn chỉnh về high-resolution small-object detection**, trong đó chính các đặc tính rất riêng của HRP4K — ultra-fine dominance, độ phân giải 4K, oblique perspective, material imbalance và negative set — được khai thác triệt để để giải thích kết quả và định hướng ứng dụng thực tế.
