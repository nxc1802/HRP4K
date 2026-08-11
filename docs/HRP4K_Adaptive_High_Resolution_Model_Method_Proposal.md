# PROPOSAL NGHIÊN CỨU CỤ THỂ

## AdaPoth-Lite: Adaptive High-Resolution Processing với MobileNetV3-Small Scout và YOLO11n-P2 cho HRP4K

## 1. Tóm tắt đề xuất

Nghiên cứu đề xuất **AdaPoth-Lite**, một hệ thống phát hiện ổ gà nhẹ cho ảnh đường bộ 4K. Hệ thống không xử lý đồng đều toàn bộ ảnh ở độ phân giải cao. Thay vào đó:

1. **MobileNetV3-Small Scout** quan sát ảnh 960×540 và sinh heatmap vùng nghi ngờ.
2. Một policy **Dynamic Top-K** chọn tối đa bốn crop từ ảnh 3840×2160 gốc.
3. **YOLO11n-P2** dùng chung trọng số xử lý:
   - ảnh global ở 960×544;
   - các crop local ở 768×512.
4. Prediction global và local được chuyển tọa độ, hiệu chỉnh score và hợp nhất bằng class-agnostic NMS hoặc Weighted Box Fusion.

Mục tiêu là vượt đường biên Pareto của các baseline medium trên HRP4K bằng một hệ thống khoảng 4–7 triệu tham số, tập trung vào ultra-fine potholes và chi phí suy luận trung bình thấp.

---

## 2. Cơ sở lựa chọn

HRP4K gồm:

- 6.003 ảnh 4K;
- 4.003 ảnh dương tính;
- 2.000 ảnh âm tính;
- 7.217 pothole instances;
- độ phân giải cố định 3840×2160.

Phần lớn đối tượng rất nhỏ:

- 3.833 ultra-fine instances dưới 0,05% diện tích ảnh;
- bounding box trung vị khoảng 100×35 pixel;
- instance nhỏ nhất xấp xỉ 10×4 pixel;
- nhiều ổ gà có hình dạng kéo dài theo chiều ngang.

Paper benchmark các biến thể medium. D-FINE medium đạt mAP@0.5 cao nhất 0,611 với 19,2M tham số; YOLOv11 medium đạt mAP@0.5:0.95 tốt nhất trong nhóm baseline được báo cáo trước đó. Điều này mở ra khoảng trống cho một hệ thống nhỏ hơn nhưng phân bổ độ phân giải tốt hơn.

---

## 3. Mục tiêu nghiên cứu

### Mục tiêu chính

Thiết kế một pipeline dưới khoảng 7M tham số có khả năng:

- vượt mAP@0.5:0.95 của YOLOv11m;
- tiệm cận hoặc vượt mAP@0.5 của D-FINE medium;
- cải thiện đáng kể AP trên nhóm ultra-fine;
- giảm average compute so với exhaustive tiling;
- tạo đường biên Pareto accuracy–efficiency tốt hơn baseline medium.

### Mục tiêu kỹ thuật

- Scout region recall từ 97% đến 99%.
- Average crop count từ 1 đến 3 crop/ảnh.
- Kmax không quá 4.
- Processed-area ratio thấp hơn đáng kể so với chia toàn ảnh.
- Detector tổng khoảng 3–5M tham số.
- Scout khoảng 1–1,5M tham số.
- Tổng hệ thống khoảng 4–6,5M tham số.

---

## 4. Kiến trúc AdaPoth-Lite

```text
Ảnh HRP4K 3840×2160
       │
       ├── Resize 960×540
       │        │
       │        ▼
       │ MobileNetV3-Small Scout
       │        │
       │        ▼
       │ Heatmap stride 16
       │        │
       │        ▼
       │ Connected components
       │ + region NMS
       │ + Dynamic Top-K, K≤4
       │
       ├── Global path
       │   resize 960×544
       │        │
       │        ▼
       │     YOLO11n-P2
       │
       └── Local path
           crop 768×512 từ ảnh gốc
                    │
                    ▼
                YOLO11n-P2
            dùng chung trọng số
                    │
                    ▼
          Coordinate remapping
                    │
                    ▼
       Score calibration + fusion
                    │
                    ▼
              Final detections
```

---

## 5. MobileNetV3-Small Region Scout

### 5.1 Input và output

- Input: 960×540.
- Backbone: MobileNetV3-Small.
- Output stride: 16.
- Heatmap output: xấp xỉ 60×34.
- Output channels: 1.
- Head: depthwise 3×3 + pointwise 1×1 + sigmoid.
- Target parameters: dưới 1,5M sau khi loại classification head.

### 5.2 Heatmap target

Mỗi ground-truth box được:

1. scale về hệ tọa độ scout;
2. mở rộng 25%;
3. chuyển thành Gaussian elliptical heatmap.

Do pothole thường kéo dài ngang, độ lệch chuẩn được tính riêng:

\[
\sigma_x=\alpha w,\qquad \sigma_y=\beta h
\]

Thiết lập khởi đầu:

- \(\alpha=0.35\);
- \(\beta=0.50\).

Việc tăng \(\beta\) giúp bù sai số theo trục dọc khi box rất thấp.

### 5.3 Scout loss

\[
L_{scout}=L_{focal}+\lambda_{cov}L_{coverage}
\]

Thiết lập ban đầu:

- focal heatmap loss;
- \(\lambda_{cov}=2.0\);
- positive pixel weighting cao;
- threshold sweep trên validation set.

Coverage loss được tính theo mức heatmap mass nằm trong vùng ground-truth mở rộng.

### 5.4 Region selection

Quy trình:

1. threshold heatmap tại \(\tau\);
2. connected-component grouping;
3. tính region score bằng max score và mean score;
4. map về ảnh 4K;
5. tạo crop 768×512;
6. thêm context margin 20%;
7. region NMS với IoU 0,35;
8. chọn tối đa K=4.

Dynamic K:

```text
Không có component vượt ngưỡng:
    lấy top-1 safety crop nếu global confidence thấp

1 component:
    K = 1

2–3 component:
    K = số component

Trên 3 component:
    K = min(4, số component)
```

Một biến thể nghiên cứu sẽ dùng learned crop-count head, nhưng không nằm trong MVP.

---

## 6. YOLO11n-P2 Shared Detector

### 6.1 Backbone

Khởi đầu từ YOLO11n pretrained trên COCO.

Giữ backbone chính nhưng điều chỉnh neck và head để ưu tiên small objects.

### 6.2 Feature levels

Sử dụng:

- P2, stride 4;
- P3, stride 8;
- P4, stride 16.

P5 được loại bỏ hoặc giữ ở phiên bản giảm channel trong ablation. Cấu hình chính đề xuất loại P5 để giảm FLOPs vì local crop chủ yếu chứa đối tượng nhỏ đến trung bình.

### 6.3 Lightweight P2 neck

Channel mục tiêu:

- P2: 48;
- P3: 96;
- P4: 192.

Module:

- depthwise separable 3×3;
- pointwise 1×1;
- một top-down fusion;
- một bottom-up fusion;
- không dùng BiFPN lặp nhiều vòng.

Pseudo-structure:

```text
C4 → lateral → P4
P4 upsample + C3 → DWConv → P3
P3 upsample + C2 → DWConv → P2
P2 downsample + P3 → DWConv
P3 downsample + P4 → DWConv
```

### 6.4 Detection head

Dùng decoupled head:

- classification branch nhỏ;
- regression branch lớn hơn.

Phân bổ khởi đầu:

- classification: 32 channels;
- regression: 64 channels;
- single-class output;
- Distribution Focal Loss cho box regression;
- IoU-based box loss.

Loss:

\[
L_{det}=L_{cls}+\lambda_{box}L_{IoU}
+\lambda_{dfl}L_{DFL}
\]

Thiết lập khởi đầu:

- \(\lambda_{box}\) tăng nhẹ so với mặc định;
- giữ classification weight thấp hơn do chỉ có một class;
- giới hạn small-box weighting để tránh instability.

### 6.5 Shared weights

Global và local path sử dụng đúng một bộ trọng số YOLO11n-P2.

Điều này giúp:

- không nhân đôi model size;
- detector học được cả context và detail;
- triển khai đơn giản;
- batch global/local crops trong cùng pipeline.

---

## 7. Input configuration

### Global branch

- Input: 960×544.
- Letterbox giữ aspect ratio.
- Mục tiêu: context, large objects và safety recall.

### Local branch

- Crop gốc: 768×512.
- Aspect ratio 3:2, phù hợp hơn với pothole kéo dài ngang.
- Nếu crop vượt biên ảnh: padding phản chiếu hoặc constant padding.
- Mỗi crop được resize nhẹ hoặc giữ nguyên 768×512.

### Biến thể ablation

- 640×640;
- 768×512;
- 960×640;
- 960×960.

Cấu hình chính ban đầu: 768×512 vì cân bằng giữa detail và compute.

---

## 8. Prediction fusion

### 8.1 Coordinate remapping

Prediction local được chuyển về ảnh 4K:

\[
x_g=x_l+x_{crop},\qquad y_g=y_l+y_{crop}
\]

Có xử lý scale nếu local input được resize.

### 8.2 Score calibration

Do cùng detector được chạy ở hai miền input khác nhau, score có thể lệch. Sử dụng:

\[
s'=T_g(s)
\]

cho global và:

\[
s'=T_l(s)
\]

cho local, trong đó \(T_g,T_l\) được hiệu chỉnh bằng temperature scaling trên validation set.

### 8.3 Fusion

Hai lựa chọn được so sánh:

- class-agnostic NMS, IoU 0,5;
- Weighted Box Fusion.

Cấu hình chính khởi đầu:

- NMS IoU 0,5;
- ưu tiên score local khi box nhỏ;
- crop-boundary penalty nếu box chạm sát mép crop.

---

## 9. Quy trình huấn luyện

### Stage A — YOLO11n-P2 baseline

Huấn luyện full-image trước:

- pretrained COCO;
- 150–200 epoch;
- input 960×544;
- cùng split với paper;
- lưu baseline theo ba seed.

### Stage B — Local crop pretraining

Sampling:

- 50% positive local crops;
- 25% hard-negative crops;
- 25% full-image samples.

Positive crop:

- center jitter;
- scale jitter;
- random context;
- cho phép box nằm lệch tâm;
- horizontal flip;
- brightness/contrast;
- perspective nhẹ.

Hard negatives:

- crack patterns;
- tar repairs;
- shadows;
- water patches;
- concrete joints;
- rough textures.

### Stage C — Scout training

- MobileNetV3-Small pretrained;
- heatmap supervision;
- focal + coverage loss;
- chọn checkpoint theo region recall trước, sau đó theo false-region rate.

### Stage D — Scout-generated crop fine-tuning

- chạy scout trên training set;
- tạo crop thực tế;
- thêm crop miss-near-boundary;
- fine-tune YOLO11n-P2;
- trộn 60% scout crops và 40% ground-truth/full-image samples.

### Stage E — Calibration

- tối ưu threshold scout;
- Kmax;
- detector confidence;
- global/local temperature;
- fusion IoU;
- crop-boundary penalty.

Không joint train end-to-end trong phiên bản chính để giữ hệ thống dễ tái tạo.

---

## 10. Thiết lập baseline

### Model baselines

- YOLO11n full-image;
- YOLO11s full-image;
- YOLO11n-P2 full-image;
- YOLO11n-P2 exhaustive tiling;
- YOLO11n-P2 fixed four crops;
- YOLO11n-P2 oracle ground-truth crops;
- YOLO11n-P2 + MobileNetV3 scout;
- các medium models trong paper HRP4K.

### Processing baselines

- resize-only;
- uniform 2×2 tiling;
- overlapping exhaustive tiling;
- random K crops;
- saliency heuristic crops;
- learned adaptive crops.

---

## 11. Ablation study

### 11.1 Kiến trúc detector

| Cấu hình | P2 | P5 | DW neck | Shared global/local |
|---|---:|---:|---:|---:|
| YOLO11n | – | ✓ | mặc định | – |
| YOLO11n-P2 | ✓ | ✓ | ✓ | – |
| YOLO11n-P2-lite | ✓ | – | ✓ | – |
| Full detector | ✓ | – | ✓ | ✓ |

### 11.2 Adaptive processing

| Cấu hình | Scout | Dynamic K | Global | Local |
|---|---:|---:|---:|---:|
| Full-image | – | – | ✓ | – |
| Exhaustive tiles | – | – | ✓ | ✓ |
| Fixed K | ✓ | – | ✓ | ✓ |
| Local only | ✓ | ✓ | – | ✓ |
| Full AdaPoth-Lite | ✓ | ✓ | ✓ | ✓ |

### 11.3 Hyperparameter ablation

- scout input: 640×360, 960×540, 1280×720;
- output stride: 8 so với 16;
- crop size;
- Kmax: 2, 4, 6;
- context margin: 10%, 20%, 30%;
- threshold;
- NMS so với WBF;
- có/không hard-negative mining;
- có/không scout-generated crop fine-tuning.

---

## 12. Metrics

### Accuracy

- Precision;
- Recall;
- F1;
- mAP@0.5;
- mAP@0.5:0.95;
- FPPI.

### Theo kích thước

- AP ultra-fine;
- AP fine;
- AP medium;
- AP large.

### Theo vật liệu

- asphalt mAP;
- concrete mAP;
- asphalt/concrete generalization gap.

### Scout

- region recall;
- GT coverage;
- false-region rate;
- average K;
- processed-area ratio.

### Efficiency

- Scout parameters;
- Detector parameters;
- Total parameters;
- average GFLOPs/image;
- worst-case GFLOPs/image;
- average latency;
- P95 latency;
- FPS;
- peak VRAM.

Chi phí trung bình:

\[
C_{avg}=C_{MobileNetV3}
+C_{global}
+\mathbb{E}[K]C_{local}
\]

---

## 13. Mục tiêu kết quả

### Mục tiêu an toàn

- total parameters dưới 7M;
- region recall ít nhất 97%;
- mAP@0.5:0.95 ít nhất 0,407;
- mAP@0.5 từ 0,60 trở lên;
- average K không quá 3;
- compute thấp hơn exhaustive tiling ít nhất 40%.

### Mục tiêu conference tốt

- mAP@0.5 từ 0,62 đến 0,64;
- mAP@0.5:0.95 từ 0,42 đến 0,44;
- ultra-fine AP tăng rõ so với YOLO11n-P2 full-image;
- tổng tham số khoảng 4–6,5M;
- giảm 50% hoặc hơn compute so với medium model hoặc exhaustive tiling tại accuracy tương đương;
- kết quả trung bình ± độ lệch chuẩn trên ba seed.

---

## 14. Đóng góp dự kiến

1. **MobileNetV3 heatmap scout nhẹ** được tối ưu cho region recall thay vì detection hoàn chỉnh.
2. **Dynamic Top-K high-resolution crop allocation** cho ảnh 4K perspective-view.
3. **YOLO11n-P2 shared global-local detector** với neck stride-4 nhẹ và bỏ P5.
4. **Global–local calibration và crop-boundary-aware fusion**.
5. **Đánh giá Pareto đầy đủ** trên HRP4K theo accuracy, scale, material và compute.

---

## 15. Rủi ro kỹ thuật

### MobileNetV3 Scout miss ultra-fine targets

Biện pháp:

- heatmap target mở rộng;
- safety top-1 crop;
- global detector;
- threshold thấp;
- stride-8 scout ablation.

### YOLO11n-P2 tăng FLOPs quá mức

Biện pháp:

- P2 chỉ 48 channels;
- depthwise separable convolution;
- bỏ P5;
- giảm số block neck;
- profile từng module.

### Nhiều crop làm latency dao động

Biện pháp:

- Kmax=4;
- báo cáo P95 latency;
- batch local crops;
- dynamic early exit;
- tối ưu average K.

### Concrete joints gây false positive

Biện pháp:

- hard-negative mining;
- oversample concrete;
- score calibration;
- báo cáo riêng concrete FPPI.

### Crop cắt đôi ổ gà

Biện pháp:

- context margin 20%;
- crop-boundary penalty;
- global prediction;
- merge các region gần nhau trước khi crop.

---

## 16. Kế hoạch triển khai

### Milestone 1 — Baseline

- chuẩn hóa dataset;
- tái tạo YOLO11n;
- thêm P2-lite;
- profile params/FLOPs/latency.

### Milestone 2 — Crop upper bound

- ground-truth crop inference;
- xác định crop size tốt nhất;
- đo tiềm năng tối đa của local processing.

### Milestone 3 — Scout

- MobileNetV3-Small heatmap model;
- dynamic top-K;
- region recall analysis.

### Milestone 4 — Full pipeline

- shared global/local inference;
- coordinate remapping;
- score calibration;
- fusion.

### Milestone 5 — Paper experiments

- three-seed evaluation;
- full ablation;
- scale/material metrics;
- Pareto plots;
- qualitative examples;
- failure analysis.

---

## 17. Tên paper dự kiến

**AdaPoth-Lite: Adaptive High-Resolution Processing with a Shared Lightweight Global–Local Detector for 4K Pothole Detection**

Tên ngắn của phương pháp:

**AdaPoth-Lite**

---

## 18. Kết luận

AdaPoth-Lite chốt một kiến trúc đủ cụ thể để triển khai ngay:

\[
\boxed{
\text{MobileNetV3-Small Scout at 960×540}
+
\text{Dynamic Top-K, } K\le4
+
\text{Shared YOLO11n-P2}
}
\]

Trong đó YOLO11n-P2 xử lý ảnh global 960×544 và crop local 768×512, sử dụng neck nhẹ P2–P4, bỏ P5 và hợp nhất prediction trong hệ tọa độ ảnh 4K.

Cấu hình này giữ contribution tập trung vào adaptive high-resolution processing, đồng thời đủ nhỏ, rõ ràng và có thể thực hiện ablation đầy đủ cho một bài conference.

---

## Nguồn nền tảng

- Paper HRP4K: *A high-resolution perspective-view road image dataset for pothole detection*, Scientific Data, 2026.
- Các số liệu dataset, phân bố kích thước và baseline trong proposal được lấy từ paper HRP4K.
