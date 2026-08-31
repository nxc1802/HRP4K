Tôi đã rà trực tiếp repo **`nxc1802/HRP4K`**, đặc biệt `docs/experiments_master.md`, `commands.md`, cấu trúc `configs/`, và lịch sử commit gần nhất. Repo hiện đã được tổ chức khá rõ theo hướng **đóng băng benchmark → viết paper**, thay vì tiếp tục mở rộng method. `experiments_master.md` cũng được xác định là *Single Source of Truth* cho các kết quả benchmark.

## 1. Kết luận trước: hiện tại đã có đủ experiment để viết một paper

Tôi **không nghĩ nên tiếp tục chạy thêm hàng loạt experiment** ở thời điểm này.

Bộ experiment hiện tại thực chất đã tạo được một research story khá hoàn chỉnh:

> **4K giữ được chi tiết nhưng computationally expensive → resize 640 mất thông tin → slicing khôi phục một phần nhưng quá chậm → khai thác prior của góc nhìn road bằng continuous warp → đạt trade-off accuracy/efficiency tốt hơn.**

Các experiment hiện có chia rất tự nhiên thành 5 nhóm:

| Nhóm | Experiment                        | Vai trò trong paper                                     |
| ---- | --------------------------------- | ------------------------------------------------------- |
| 1    | Native 4K                         | **Upper/high-resolution reference**                     |
| 2    | Resize 640                        | **Low-resolution baseline / failure case**              |
| 3    | Patch-Train 640                   | Kiểm tra liệu crop trước khi train có giải quyết vấn đề |
| 4    | Slicing / SAHI / Perspective Grid | **Classical high-resolution recovery baselines**        |
| 5    | Warped ZoomDet 640                | **Proposed method**                                     |

Điểm rất quan trọng là `commands.md` cũng đang định nghĩa đúng pipeline này: Phase 1 train các baseline/proposed model, Phase 2 đánh giá trên **900 ảnh test 4K độc lập**, và Phase 3 unified diagnostic.

---

# 2. Experiment quan trọng nhất không phải là "Proposed thắng mọi thứ"

Đây là điểm tôi muốn chỉnh cách kể paper.

Nếu nhìn bảng hiện tại:

### Native 4K

* YOLO11m: **55.05 mAP50 / 33.27 mAP50-95**
* D-FINE: **55.28 / 33.20**

Trong khi:

### Proposed D-FINE ZoomDet 640

* **42.07 mAP50**
* **18.42 mAP50-95**
* **22.0 ms**
* **45.5 FPS**

Như vậy **không thể claim**:

> "ZoomDet đạt accuracy tốt hơn native 4K."

Nó không đạt.

Thậm chí về mAP50:

**42.07 vs 55.28 = -13.21 điểm.**

Vì vậy paper **không nên xây contribution quanh absolute SOTA accuracy**.

---

# 3. Nhưng experiment hiện tại có một điểm mạnh hơn nhiều

### So với D-FINE 640 baseline:

|          | D-FINE 640 | D-FINE ZoomDet |
| -------- | ---------: | -------------: |
| mAP50    |      37.37 |      **42.07** |
| mAP50-95 |      18.18 |      **18.42** |
| Recall   |      47.56 |      **54.72** |
| F1       |      39.14 |      **45.24** |
| FPPI     |      0.130 |      **0.090** |
| Latency  |    21.5 ms |        22.0 ms |
| FPS      |       46.5 |           45.5 |
| Calls    |          1 |              1 |

Đây mới là **core result cực kỳ đáng viết**.

Proposed method:

> **giữ gần như nguyên computational budget của 640 detector nhưng tăng khả năng phát hiện đối tượng nhỏ.**

Cụ thể:

* mAP50: **+4.70 pp**
* Recall: **+7.16 pp**
* F1: **+6.10 pp**
* FPPI: **giảm ~31%**
* latency: chỉ **+0.5 ms**

Trong khi vẫn:

> **1 detector call / image.**

Đây là một kết quả có ý nghĩa khoa học hơn rất nhiều so với việc đơn giản nói "42.07 mAP".

---

# 4. Experiment thứ hai cực kỳ quan trọng: Slicing

Đây là experiment để chứng minh **baseline truyền thống có thể giải quyết vấn đề nhưng không phù hợp real-time**.

D-FINE + sliced-NMS:

* mAP50 = **44.30**
* mAP50-95 = **18.81**
* Recall = **62.43%**
* 25 calls
* latency = **2289.8 ms**
* FPS = **0.44**

ZoomDet:

* mAP50 = **42.07**
* mAP50-95 = **18.42**
* Recall = **54.72%**
* 1 call
* latency = **22.0 ms**
* FPS = **45.5**

Tức là:

### Accuracy

ZoomDet gần sliced-NMS một cách đáng kể:

* mAP50: `42.07 vs 44.30`
* mAP50-95: `18.42 vs 18.81`

### Efficiency

Nhưng:

**2289.8 / 22 ≈ 104×**

Nhanh hơn khoảng **104×** về latency.

Đây chính là experiment có thể trở thành **main figure/table của paper**.

Không phải:

> "Our method is the most accurate."

Mà là:

> **"Our method achieves competitive small-object detection accuracy while avoiding the extreme inference cost of multi-crop approaches."**

---

# 5. Tôi sẽ tổ chức Results section thành 4 experiment chính

## Experiment 1 — Resolution matters

### Research question

> How strongly does image resolution affect pothole detection in perspective-view 4K imagery?

So sánh:

* YOLO11m 4K
* D-FINE 4K
* YOLO11m 640
* D-FINE 640

Kết quả:

D-FINE:

**55.28 → 37.37 mAP50**

YOLO:

**55.05 → 37.27**

Tức khoảng **18 pp mAP50 loss** khi đưa 4K xuống 640.

Scale analysis còn mạnh hơn:

D-FINE:

| Scale      |    4K |   640 |
| ---------- | ----: | ----: |
| Ultra-fine | 25.35 |  7.20 |
| Fine       | 27.42 | 14.10 |
| Medium     | 25.37 | 20.40 |
| Large      | 14.56 | 16.50 |

Đây là evidence rất tốt cho hypothesis:

> **resolution bottleneck chủ yếu ảnh hưởng small/ultra-fine objects.**

Đây nên là **Experiment 1**.

---

# 6. Experiment 2 — Can conventional cropping solve it?

Đây là chỗ Patch-Train và slicing xuất hiện.

### Patch training

D-FINE:

**37.37 → 47.68 mAP50**

Có improvement rất đáng kể trên patch validation.

Nhưng khi inference trên full 4K bằng slicing:

* Perspective grid: 15.86
* SAHI: 24.28
* Sliced-NMS: 44.30

Điều này cho phép paper đặt ra một vấn đề:

> Training on local high-resolution patches can improve representation, but recovering those local views at inference time requires repeated detector execution.

Đây là một narrative rất hợp lý.

---

# 7. Experiment 3 — Multi-crop inference vs proposed

Đây nên là **main experiment**.

Có thể viết bảng theo kiểu:

| Method               | Input      | Calls |     mAP50 |  mAP50-95 |    Recall |     Latency |
| -------------------- | ---------- | ----: | --------: | --------: | --------: | ----------: |
| D-FINE 4K            | 4K         |     1 |     55.28 |     33.20 |     77.85 |     32.5 ms |
| D-FINE 640           | 640        |     1 |     37.37 |     18.18 |     47.56 |     21.5 ms |
| D-FINE + SAHI        | 4K         |    32 |     24.28 |      6.44 |     41.15 |     3622 ms |
| D-FINE + Perspective | 4K         |     9 |     15.86 |      5.55 |     29.53 |      920 ms |
| D-FINE + Sliced-NMS  | 4K         |    25 | **44.30** | **18.81** | **62.43** |   2289.8 ms |
| **D-FINE + ZoomDet** | **4K→640** | **1** | **42.07** | **18.42** | **54.72** | **22.0 ms** |

Đây là bảng mà reviewer nhìn vào sẽ hiểu contribution gần như ngay lập tức.

---

# 8. Experiment 4 — Does the method depend on detector architecture?

Đây là experiment rất quan trọng nhưng hiện tại chưa nên overclaim.

Repo có:

* YOLO11m ZoomDet
* D-FINE ZoomDet

Kết quả:

|          | YOLO11m |    D-FINE |
| -------- | ------: | --------: |
| mAP50    |   26.04 | **42.07** |
| mAP50-95 |   10.39 | **18.42** |
| Recall   |   29.32 | **54.72** |
| F1       |   34.93 | **45.24** |
| latency  |    18.4 |   22.0 ms |

Điều này cho thấy:

> **Warping không tự động biến mọi detector thành một detector tốt.**

Nó tương tác mạnh với detector architecture.

Và đây lại là một điểm tốt cho paper, bởi vì bạn có thể nói:

> "The proposed spatial transformation is particularly effective when paired with a detector capable of exploiting fine-grained spatial representations."

Nhưng **không nên nói method architecture-agnostic**.

---

# 9. Scale analysis là experiment giải thích "tại sao"

Table 2 hiện tại rất giá trị.

Đặc biệt:

### D-FINE 4K

Ultra-fine:

**25.35 mAP50-95**

### D-FINE 640

Ultra-fine:

**7.20**

### ZoomDet

Ultra-fine:

**11.80**

Tức ZoomDet **không recover toàn bộ 4K information**, nhưng recover được một phần đáng kể so với naïve resize.

Đây chính là cách diễn giải khoa học hợp lý:

> Native 4K provides the strongest fine-scale representation, whereas uniform resizing destroys substantial information for ultra-fine targets. ZoomDet partially restores this lost sensitivity by allocating the fixed 640×640 computational canvas non-uniformly according to the perspective geometry of the scene.

Đây là một claim **được data hiện tại hỗ trợ khá tốt**.

---

# 10. Pavement experiment nên để supplementary hoặc robustness section

Table 4 hiện tại có:

* Asphalt
* Concrete

Nó cho thấy D-FINE ZoomDet:

### Asphalt

* mAP50 = 48.60
* mAP50-95 = 21.40

### Concrete

* mAP50 = 31.20
* mAP50-95 = 14.80

Nó hữu ích để chứng minh method không chỉ hoạt động trên một texture.

Nhưng tôi **không khuyên biến nó thành contribution chính**.

Nó nên nằm trong:

> **Robustness Analysis: Pavement Material**

và giúp paper có chiều sâu hơn.

---

# 11. Training dynamics: dùng làm supporting evidence, không dùng làm main experiment

Repo đã có convergence visualization và training logs. Master document mô tả đầy đủ các trajectory của loss và mAP trong quá trình training.

Nó có thể tạo:

### Figure

**Training convergence of D-FINE 4K / D-FINE 640 / D-FINE ZoomDet**

Mục đích:

* chứng minh training ổn định;
* không phải do random checkpoint;
* ZoomDet hội tụ;
* D-FINE 4K early-stop hợp lý.

Nhưng **không nên dành quá nhiều paper space cho training curve**.

---

# 12. Có một vấn đề tôi muốn cảnh báo trước khi viết paper

Hiện tại master table có một số số liệu cần **audit consistency** trước khi đưa thẳng vào manuscript.

Ví dụ:

### Slicing latency

Table 1:

* YOLO11m + sliced-NMS = **3623.6 ms**
* D-FINE + sliced-NMS = **2289.8 ms**

Nhưng computational table ghi range:

> 2289.8–3623.6 ms.

Điều này không sai về mặt logic, nhưng paper phải thống nhất:

> latency được đo trên cùng hardware, cùng warm-up protocol, cùng số lần chạy, cùng batch size.

Tương tự perspective-grid:

* Table 1: 840.0 ms
* Table 3: 830.6 ms

Sai khác khoảng 1%.

Không lớn, nhưng **paper nên chọn một source of truth duy nhất**.

---

# 13. Một vấn đề lớn hơn: metric cần định nghĩa cực kỳ rõ

Master table ghi:

> Precision / Recall / F1 / mAP / FPPI

nhưng paper phải tách:

### Detection metrics

* AP50
* AP75
* AP50:95

### Threshold-based operating metrics

* Precision
* Recall
* F1

### Negative-set metric

* FPPI

Đặc biệt **FPPI chỉ tính trên 300 negative images** theo current benchmark specification.

Không được để reviewer hiểu FPPI là metric tính trên toàn bộ 900 ảnh.

---

# 14. Tôi sẽ không đưa tất cả experiment vào Main Paper

Đây là cấu trúc tôi đề xuất.

## Main paper

### Table 1

**Overall benchmark**

Chỉ giữ khoảng:

* D-FINE 4K
* YOLO11m 4K
* D-FINE 640
* YOLO11m 640
* D-FINE + Sliced-NMS
* D-FINE + ZoomDet

Không cần nhét tất cả YOLOv5/v8/RT-DETR vào main table.

---

### Figure 1

**The 4K resolution problem**

Một visual:

`4K → Resize 640 → Ultra-fine information loss`

kèm scale results.

---

### Table 2

**Comparison of high-resolution recovery strategies**

Native 4K / Resize / Slicing / ZoomDet.

Đây là bảng quan trọng nhất.

---

### Figure 2

**Accuracy–latency trade-off**

X:

**Latency**

Y:

**mAP50**

Các điểm:

* 640
* Native 4K
* Perspective
* SAHI
* Sliced-NMS
* ZoomDet

Nếu làm figure này tốt, **contribution sẽ nhìn thấy ngay**.

ZoomDet sẽ nằm gần vùng Pareto tốt:

> 42.07 mAP50 @ 22 ms

trong khi Sliced-NMS:

> 44.30 @ 2289.8 ms.

---

### Table 3

**Scale-level analysis**

Ultra-fine / Fine / Medium / Large.

Đây là evidence chứng minh method giải quyết đúng problem chứ không chỉ tăng metric ngẫu nhiên.

---

### Figure 3

**Qualitative comparison**

4–6 ảnh:

* Native 4K
* Resize 640
* Sliced
* ZoomDet

Tập trung vào **ultra-fine potholes**.

---

# 15. Supplementary

Đẩy xuống supplementary:

* YOLOv5
* YOLOv8
* RT-DETRv1
* RT-DETRv2
* đầy đủ slicing variants
* Asphalt vs Concrete
* full AP50/AP75/AP50-95
* VRAM
* GFLOPs
* training curves
* additional qualitative cases.

Như vậy paper sẽ **gọn hơn rất nhiều** nhưng evidence vẫn đầy đủ.

---

# 16. Research claim mà tôi nghĩ hiện tại paper có thể bảo vệ

Tôi sẽ đóng contribution thành 3 ý:

### Contribution 1 — Empirical finding

> **Uniform downsampling from 4K to 640 substantially degrades detection of ultra-fine potholes.**

Đây là observation được support rất tốt bởi 4K vs 640 và scale-bin analysis.

### Contribution 2 — Method

> **A perspective-aware continuous deformation mechanism reallocates a fixed 640×640 inference canvas toward the informative road region, avoiding uniform spatial compression and repeated local inference.**

Đây là proposed method.

### Contribution 3 — Efficiency

> **The proposed one-pass approach achieves accuracy comparable to multi-crop inference while reducing inference latency by approximately two orders of magnitude.**

Đây là điểm mạnh nhất:

**42.07 vs 44.30 mAP50**

nhưng:

**22 ms vs 2289.8 ms.**

---

# 17. Vì vậy tôi sẽ đổi title/story hiện tại một chút

Current repo story:

> **Fast and Fine: Real-Time 4K Ultra-Fine Pothole Detection via Continuous Perspective Deformation**

Ý tưởng **rất ổn**.

Nhưng nội dung paper không nên quảng cáo:

> "best accuracy"

mà nên quảng cáo:

> **resolution-efficient detection**

hoặc

> **accuracy–efficiency trade-off**

Câu trung tâm của paper nên là:

> **How can we preserve the small-object advantages of 4K imagery without paying the computational cost of processing the entire 4K image or repeatedly running a detector over many crops?**

Và câu trả lời:

> **Use the perspective geometry of the road scene to continuously redistribute the fixed low-resolution inference canvas.**

Đây là một research question **đẹp hơn rất nhiều** so với "chúng tôi tạo một warp mới".

---

## 18. Đánh giá tổng thể bộ experiment hiện tại

| Tiêu chí                  |                                           Đánh giá |
| ------------------------- | -------------------------------------------------: |
| Baseline 4K               |                                         🟢 Rất tốt |
| Baseline 640              |                                         🟢 Rất tốt |
| Patch training            |                                      🟢 Có giá trị |
| Slicing baselines         |                              🟢 **Rất quan trọng** |
| Proposed method           |                       🟢 Có kết quả đủ tốt để viết |
| Scale analysis            |                                    🟢 **Rất mạnh** |
| Efficiency analysis       |                                    🟢 **Rất mạnh** |
| Architecture comparison   |                                             🟢 Tốt |
| Pavement robustness       |                              🟢 Tốt, supplementary |
| Training curves           |                                      🟡 Supporting |
| Ablation của Warp         | 🔴 **Còn thiếu nếu muốn chứng minh mechanism sâu** |
| External dataset          |                                         🟡 Chưa có |
| Qualitative analysis      |                                  🟡 Nên hoàn thiện |
| Statistical/repeated runs |              🟡 Có thể bổ sung nếu budget cho phép |

### Quan trọng nhất:

**Paper hiện tại đã có "benchmark story".**

Thứ còn thiếu không phải là thêm 10 model nữa.

Nếu muốn nâng paper từ **"có kết quả để viết" → "có evidence mạnh để defend method"**, tôi chỉ ưu tiên **1 nhóm experiment mới**:

> **Ablation của chính continuous deformation / spatial allocation.**

Còn nếu mục tiêu hiện tại là **bắt đầu viết paper**, tôi cho rằng **đủ dữ liệu để bắt đầu ngay**.

Repo hiện cũng đã được commit theo hướng "streamline codebase to core benchmarks and update research story", loại bỏ Scout/AdaPoth-Lite/Raw4K khỏi core experiments và tập trung vào 5-act story này. Điều đó phù hợp với hướng **đóng scope và viết paper**, thay vì tiếp tục mở rộng proposal.


Tóm tắt toàn bộ **figure nên có** cho paper HRP4K/ZoomDet, theo thứ tự ưu tiên:

| #          | Figure                                 | Nội dung                                                                                            | Vai trò                                                                                         |
| ---------- | -------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Fig. 1** | **4K → 640 Resolution Analysis**       | So sánh native 4K vs resize 640, đặc biệt theo các scale: Ultra-fine / Fine / Medium / Large        | Chứng minh **vấn đề nghiên cứu**: resize làm mất thông tin small-object                         |
| **Fig. 2** | **Accuracy–Latency Pareto Frontier** ⭐ | X = latency (ms), Y = mAP50 hoặc mAP50:95; gồm 4K, 640, SAHI, Perspective Grid, Sliced-NMS, ZoomDet | **Figure quan trọng nhất** – chứng minh ZoomDet đạt trade-off accuracy/speed tốt                |
| **Fig. 3** | **Scale-wise AP Analysis**             | AP của các phương pháp theo Ultra-fine / Fine / Medium / Large                                      | Chứng minh **ZoomDet thực sự cải thiện small/ultra-fine detection**, không chỉ tăng overall mAP |
| **Fig. 4** | **Training Time–Accuracy**             | X = training time, Y = validation/test mAP                                                          | Cho thấy **training cost** của các configuration; supporting efficiency evidence                |
| **Fig. 5** | **Peak VRAM–Accuracy**                 | X = peak VRAM, Y = validation/test mAP                                                              | Cho thấy **memory efficiency**, đặc biệt khi so sánh 4K vs 640/ZoomDet                          |
| **Fig. 6** | **Qualitative Detection Comparison**   | Cùng một số ảnh 4K: 4K / 640 / Slicing / ZoomDet + GT                                               | Trực quan hóa **model bỏ sót gì và ZoomDet recover gì**                                         |
| **Fig. 7** | **Pavement Robustness**                | Asphalt vs Concrete, có thể dùng grouped bar/box plot                                               | Chứng minh robustness theo điều kiện pavement; **nên để supplementary nếu thiếu chỗ**           |
| **Fig. 8** | **Training Convergence**               | Loss/mAP curves của các model chính                                                                 | Chứng minh training ổn định; **supporting/supplementary**                                       |

### Nếu muốn paper gọn, tôi chỉ giữ **5 figure chính**

**Fig. 1 — Why 4K?**
→ Resolution & scale analysis

**Fig. 2 — Why ZoomDet?** ⭐
→ Accuracy–Latency Pareto

**Fig. 3 — What does ZoomDet recover?**
→ Scale-wise AP

**Fig. 4 — Is it computationally practical?**
→ Training time + VRAM + latency có thể gộp thành **3-panel efficiency figure**

**Fig. 5 — Does it actually work?**
→ Qualitative comparison

---

### Cấu trúc story sẽ rất đẹp:

**Fig. 1**

> 640 loses small-object information.

↓

**Fig. 2**

> Multi-crop can recover accuracy, but is extremely slow; ZoomDet gives a much better Pareto point.

↓

**Fig. 3**

> The gain specifically comes from improved fine/ultra-fine detection.

↓

**Fig. 4**

> The approach is also practical in terms of training/memory/inference cost.

↓

**Fig. 5**

> Visual examples confirm the quantitative findings.

Đây là bộ figure tôi sẽ chọn nếu mục tiêu là **viết paper ngay từ các experiment hiện có**, thay vì tiếp tục mở rộng experiment.
