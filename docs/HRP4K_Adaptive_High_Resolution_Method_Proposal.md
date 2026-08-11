# PROPOSAL NGHIÊN CỨU

## Adaptive High-Resolution Processing cho phát hiện ổ gà bằng mô hình nhẹ trên HRP4K

## 1. Tóm tắt đề xuất

Nghiên cứu đề xuất một phương pháp **Adaptive High-Resolution Processing** dành cho bài toán phát hiện ổ gà trên ảnh đường bộ 4K. Thay vì resize toàn bộ ảnh độ phân giải cao xuống một kích thước cố định hoặc chia toàn ảnh thành các tile đồng đều, phương pháp sử dụng một mạng nhẹ để quan sát toàn cảnh ở độ phân giải thấp, xác định các vùng có khả năng chứa ổ gà, rồi chỉ xử lý chi tiết một số vùng được chọn trực tiếp từ ảnh 4K gốc.

Mục tiêu là đạt được sự cân bằng tốt hơn giữa:

- độ chính xác phát hiện;
- độ chính xác định vị;
- khả năng phát hiện vật thể cực nhỏ;
- số tham số;
- FLOPs trung bình trên mỗi ảnh;
- độ trễ suy luận;
- bộ nhớ sử dụng.

Đề xuất không chốt trước backbone hoặc detector cụ thể. Phần đóng góp chính nằm ở cơ chế **phân bổ độ phân giải và tài nguyên tính toán thích ứng**, kết hợp một detector nhẹ dùng chung cho quan sát toàn ảnh và các crop cục bộ.

---

## 2. Bối cảnh và động lực

HRP4K gồm 6.003 ảnh độ phân giải 3840×2160, trong đó có 4.003 ảnh chứa ổ gà, 2.000 ảnh âm tính và tổng cộng 7.217 bounding box. Dataset được thu từ góc nhìn camera gắn trên phương tiện, vì vậy ổ gà thường xuất hiện nhỏ, xa, bị biến dạng phối cảnh và nằm trong nền đường có texture phức tạp.

Phân bố kích thước đối tượng cho thấy:

- 3.833 instance thuộc nhóm ultra-fine, nhỏ hơn 0,05% diện tích ảnh;
- 1.078 instance thuộc nhóm fine, từ 0,05% đến 0,1%;
- 1.099 instance thuộc nhóm medium, từ 0,1% đến 0,25%;
- 1.207 instance lớn hơn hoặc bằng 0,25%;
- bounding box trung vị khoảng 100×35 pixel;
- instance nhỏ nhất xấp xỉ 10×4 pixel.

Các detector baseline trong paper được huấn luyện bằng biến thể medium, pretrained trên COCO, trong 150 epoch với hyperparameter mặc định. D-FINE medium đạt mAP@0.5 cao nhất là 0,611 với 19,2 triệu tham số. Tuy nhiên, benchmark chưa tập trung vào mô hình nano/tiny, xử lý ảnh 4K thích ứng hoặc phân tích đường biên Pareto giữa accuracy và computation.

Vấn đề cốt lõi là: nếu resize toàn ảnh 4K xuống một kích thước nhỏ, nhiều ổ gà sẽ mất gần hết chi tiết; nếu xử lý toàn bộ ảnh ở độ phân giải cao hoặc chia exhaustive tiles, chi phí tính toán tăng mạnh. Do đó cần một cơ chế chỉ đưa tài nguyên độ phân giải cao đến những vùng có giá trị.

---

## 3. Câu hỏi nghiên cứu

Nghiên cứu tập trung trả lời ba câu hỏi:

1. Có thể dùng một hệ thống nhẹ để chọn thích ứng các vùng độ phân giải cao mà vẫn giữ region recall gần như tuyệt đối hay không?
2. Một detector nhỏ dùng chung cho ảnh toàn cảnh và crop cục bộ có thể vượt hoặc tiệm cận các baseline medium trên HRP4K hay không?
3. Phương pháp có tạo ra đường biên Pareto tốt hơn giữa mAP, FLOPs, latency và số tham số so với resize toàn ảnh, exhaustive tiling và detector medium hay không?

---

## 4. Giả thuyết nghiên cứu

Giả thuyết chính:

> Hiệu năng trên HRP4K bị giới hạn đáng kể bởi cách phân bổ độ phân giải đầu vào, không chỉ bởi kích thước detector. Một mô hình nhỏ có thể đạt kết quả tốt hơn nếu chỉ xử lý chi tiết các vùng giàu thông tin từ ảnh 4K.

Các giả thuyết phụ:

- Một region scout chỉ cần tối ưu recall, không cần thực hiện detection hoàn chỉnh.
- Dynamic top-K crop hiệu quả hơn fixed-K hoặc exhaustive tiling.
- Detector dùng chung trọng số cho global image và local crops giúp giảm số tham số và ổn định huấn luyện.
- Nhánh global giúp giảm rủi ro scout bỏ sót và hỗ trợ phát hiện đối tượng lớn.
- Feature map độ phân giải cao, đặc biệt mức stride nhỏ, có vai trò quan trọng đối với nhóm ultra-fine.

---

## 5. Phương pháp đề xuất

### 5.1 Kiến trúc tổng quát

```text
Ảnh 4K gốc
   │
   ├── Resize xuống độ phân giải thấp
   │          │
   │          ▼
   │   Lightweight Region Scout
   │          │
   │          ▼
   │   Heatmap / region scores
   │          │
   │          ▼
   │   Dynamic region selection
   │
   ├── Global branch:
   │   ảnh resize → lightweight detector
   │
   └── Local branch:
       crop K vùng từ ảnh 4K gốc
                 │
                 ▼
       cùng lightweight detector
                 │
                 ▼
     Coordinate remapping + fusion + NMS
                 │
                 ▼
            Detection cuối
```

Hệ thống gồm hai mạng:

1. **Region Scout:** mạng rất nhỏ, chỉ xác định vùng đáng chú ý.
2. **Shared Detector:** detector nhẹ dùng chung cho global image và local crops.

Scout và detector có thể được lựa chọn từ nhiều backbone khác nhau; proposal này chỉ chốt chức năng và giao diện giữa các thành phần.

---

### 5.2 Region Scout

Scout nhận ảnh đã resize và sinh heatmap:

\[
H \in [0,1]^{h\times w}
\]

Mỗi điểm hoặc ô trên heatmap biểu diễn xác suất vùng tương ứng chứa ổ gà.

Scout không cần dự đoán bounding box chính xác. Mục tiêu là:

- region recall cao;
- chi phí rất thấp;
- heatmap ổn định trước texture nền;
- hỗ trợ chọn số crop thích ứng.

Nhãn scout được sinh từ bounding box bằng một trong các cách:

- binary expanded-box mask;
- Gaussian heatmap quanh tâm box;
- soft rectangular mask có trọng số giảm dần về biên.

Loss tổng quát:

\[
L_{scout}=L_{heatmap}+\lambda_{cov}L_{coverage}
\]

Trong đó `coverage loss` phạt mạnh trường hợp các region được chọn không bao phủ ground-truth.

Metric riêng cho scout:

- region recall;
- mean GT coverage;
- average selected regions per image;
- processed-area ratio;
- false-region rate.

Mục tiêu thực nghiệm ban đầu là region recall từ 97% đến 99%.

---

### 5.3 Adaptive Region Selection

Từ heatmap, hệ thống sinh candidate windows, loại các vùng trùng lặp và chọn một tập region giới hạn.

Quy trình:

1. threshold hoặc tìm local maxima;
2. gom connected components;
3. sinh candidate windows;
4. mở rộng context margin;
5. suppression giữa các window;
6. chọn top-K với giới hạn ngân sách.

Số crop không cố định mà phụ thuộc độ khó ảnh:

\[
K=f(H,\tau,K_{max})
\]

Một policy đơn giản:

- không có score vượt ngưỡng: lấy top-1 làm safety crop hoặc bỏ local branch nếu global confidence đủ cao;
- ít vùng đáng nghi: K nhỏ;
- nhiều vùng đáng nghi: tăng K đến Kmax;
- tránh nhiều crop hội tụ vào cùng một vị trí bằng region NMS.

Cơ chế này khác exhaustive tiling ở chỗ chỉ xử lý một phần nhỏ diện tích ảnh.

---

### 5.4 Shared Global–Local Detector

Cùng một detector được sử dụng cho:

- ảnh toàn cảnh đã resize;
- các crop độ phân giải cao lấy từ ảnh gốc.

Global branch đảm nhiệm:

- ngữ cảnh toàn cảnh;
- đối tượng trung bình hoặc lớn;
- bảo vệ trước scout miss;
- giảm false positive cục bộ.

Local branch đảm nhiệm:

- ultra-fine và fine objects;
- biên và texture chi tiết;
- định vị chính xác hơn.

Prediction local được chuyển về hệ tọa độ ảnh gốc:

\[
B^{global}=T(B^{local},R)
\]

Sau đó hợp nhất với prediction global:

\[
D_{final}=Fuse(D_{global},T(D_{local}))
\]

Có thể sử dụng:

- class-agnostic NMS;
- weighted box fusion;
- score calibration giữa global và local;
- crop-boundary suppression để giảm box bị cắt.

---

### 5.5 Detector nhẹ được tinh chỉnh theo small-object

Proposal không chốt detector cụ thể nhưng yêu cầu kiến trúc đáp ứng:

- tổng tham số thấp;
- neck nhẹ;
- có feature map stride nhỏ hoặc nhánh bảo toàn chi tiết;
- regression branch đủ năng lực;
- classification branch nhỏ do HRP4K chỉ có một class;
- hỗ trợ cùng trọng số cho global và local input;
- có thể loại hoặc thu nhỏ nhánh dành cho đối tượng rất lớn nếu không cần thiết.

Các lựa chọn có thể thử:

- nano/tiny one-stage CNN detector;
- lightweight anchor-free detector;
- compact transformer detector;
- custom detector dựa trên depthwise/partial convolution.

---

## 6. Quy trình huấn luyện

### Giai đoạn 1 — Huấn luyện detector nhẹ

Detector được huấn luyện bằng hỗn hợp:

- full-image resized samples;
- ground-truth-centered positive crops;
- random-offset positive crops;
- hard-negative road crops;
- ảnh âm tính hoàn toàn.

Mục tiêu là tránh detector chỉ hoạt động trên crop hoàn hảo.

### Giai đoạn 2 — Huấn luyện scout

Scout được huấn luyện từ heatmap sinh bởi annotation gốc. Ưu tiên recall hơn precision.

### Giai đoạn 3 — Huấn luyện với scout-generated crops

Chạy scout trên training set, sinh crop thực tế rồi fine-tune detector. Bước này giảm domain gap giữa crop từ ground truth và crop tại inference.

### Giai đoạn 4 — Tinh chỉnh hệ thống

Có thể tinh chỉnh:

- threshold;
- Kmax;
- crop size;
- context margin;
- fusion rule;
- loss weights;
- score calibration.

Joint end-to-end training là tùy chọn mở rộng, không bắt buộc trong phiên bản đầu.

---

## 7. Baseline thực nghiệm

Các baseline tối thiểu:

1. lightweight detector với full-image resize;
2. lightweight detector với fixed tiling;
3. lightweight detector với exhaustive overlapping tiling;
4. lightweight detector với random hoặc uniform top-K crops;
5. lightweight detector với oracle ground-truth crops;
6. proposed adaptive crop pipeline;
7. các medium baseline được paper HRP4K báo cáo.

Oracle crop đóng vai trò upper bound để xác định tiềm năng tối đa của high-resolution local processing.

---

## 8. Ablation study

| Thí nghiệm | Scout | Dynamic K | Global branch | Local branch | Shared weights |
|---|---:|---:|---:|---:|---:|
| Full-image baseline | – | – | ✓ | – | – |
| Fixed tiling | – | – | ✓ | ✓ | ✓ |
| Fixed top-K | ✓ | – | ✓ | ✓ | ✓ |
| Adaptive local only | ✓ | ✓ | – | ✓ | – |
| Adaptive global-local | ✓ | ✓ | ✓ | ✓ | ✓ |
| Full method | ✓ | ✓ | ✓ | ✓ | ✓ |

Ablation bổ sung:

- heatmap type;
- crop size và aspect ratio;
- Kmax;
- region threshold;
- context margin;
- stride của feature map nhỏ nhất;
- NMS so với weighted box fusion;
- có/không crop-boundary suppression;
- có/không fine-tune trên scout-generated crops.

---

## 9. Bộ metric đánh giá

### Accuracy

- Precision;
- Recall;
- F1;
- mAP@0.5;
- mAP@0.5:0.95;
- FPPI trên negative images.

### Scale-aware metrics

Báo cáo riêng cho:

- ultra-fine;
- fine;
- medium;
- large.

### Material-aware metrics

Báo cáo riêng trên:

- asphalt;
- concrete.

### Efficiency

- số tham số;
- model size;
- average GFLOPs/image;
- worst-case GFLOPs/image;
- average selected crops/image;
- processed-area ratio;
- average latency;
- P95 latency;
- throughput;
- peak VRAM;
- năng lượng nếu có thiết bị đo.

Chi phí adaptive phải được tính theo:

\[
C_{avg}=C_{scout}+C_{global}+\mathbb{E}[K]C_{crop}
\]

Không được chỉ báo cáo FLOPs của detector đơn lẻ.

---

## 10. Tiêu chí thành công

### Mức tối thiểu

- tổng tham số thấp hơn rõ rệt so với baseline medium;
- mAP@0.5:0.95 bằng hoặc vượt 0,407;
- mAP@0.5 tiệm cận hoặc vượt 0,611;
- average compute thấp hơn exhaustive tiling;
- region recall scout tối thiểu 97%.

### Mức paper conference tốt

- tổng tham số dưới khoảng 10M;
- mAP@0.5:0.95 tăng ít nhất 1,5–2 điểm so với baseline tốt nhất;
- cải thiện rõ trên ultra-fine AP;
- giảm ít nhất 40–50% FLOPs hoặc latency so với baseline medium tương đương;
- tạo Pareto frontier tốt nhất trong accuracy–efficiency plot;
- kết quả ổn định trên ít nhất ba random seeds.

---

## 11. Đóng góp dự kiến

1. **Adaptive sparse high-resolution processing:** chỉ xử lý độ phân giải cao tại vùng có giá trị.
2. **Dynamic compute allocation:** số region phụ thuộc độ khó của từng ảnh.
3. **Shared global–local lightweight detection:** một detector nhỏ dùng chung cho toàn ảnh và crop.
4. **HRP4K scale-aware efficiency benchmark:** đánh giá đồng thời accuracy, scale, pavement material và chi phí thực tế.
5. **Phân tích failure mode:** scout miss, crop truncation, duplicate detections và texture-induced false positives.

---

## 12. Rủi ro và phương án xử lý

### Scout bỏ sót

Giải pháp:

- ưu tiên recall;
- mở rộng heatmap target;
- giữ global branch;
- safety top-1 region;
- giảm threshold trong inference.

### Crop làm mất ngữ cảnh

Giải pháp:

- context margin;
- nhiều aspect ratio;
- phối hợp global prediction;
- huấn luyện với random-offset crops.

### Chi phí tăng do nhiều crop

Giải pháp:

- dynamic K;
- Kmax cứng;
- early exit;
- batching crops;
- tối ưu scout rất nhẹ.

### False positive từ texture concrete

Giải pháp:

- hard-negative mining;
- oversample concrete;
- global-local score calibration;
- đánh giá riêng theo pavement material.

### Novelty bị xem là gần SAHI

Giải pháp:

- nhấn mạnh learned region allocation;
- dynamic compute budget;
- shared global-local detector;
- so sánh trực tiếp với exhaustive slicing;
- báo cáo processed-area ratio và average compute.

---

## 13. Kế hoạch triển khai

### Phase 1 — Reproduce và lightweight baselines

- tái tạo split và metric HRP4K;
- huấn luyện nhiều detector nano/tiny;
- xác định bottleneck theo scale;
- chạy full-image và exhaustive tiling baselines.

### Phase 2 — Region Scout

- xây heatmap labels;
- thử nhiều loss;
- tối ưu region recall;
- chọn policy crop đơn giản.

### Phase 3 — Global–Local Pipeline

- map tọa độ;
- merge predictions;
- fine-tune trên scout-generated crops;
- xây efficiency profiler.

### Phase 4 — Tối ưu và ablation

- dynamic K;
- crop geometry;
- detector small-object refinement;
- score fusion;
- hard-negative mining.

### Phase 5 — Paper-grade evaluation

- ba random seeds;
- scale/material analysis;
- latency và VRAM;
- failure cases;
- accuracy–efficiency Pareto plots;
- cross-dataset validation nếu tài nguyên cho phép.

---

## 14. Kết luận

Đề xuất tập trung vào một luận điểm rõ ràng: đối với HRP4K, tăng kích thước mô hình không phải cách duy nhất để tăng accuracy. Một hệ thống nhẹ có thể khai thác tốt hơn thông tin 4K bằng cách quan sát toàn cảnh ở chi phí thấp, chọn thích ứng vùng cần phóng đại và sử dụng chung một detector nhỏ cho global–local inference.

Phương pháp cốt lõi được chốt là:

\[
\boxed{
\text{Lightweight Region Scout}
+
\text{Dynamic Sparse High-Resolution Crops}
+
\text{Shared Global–Local Lightweight Detector}
}
\]

Backbone, detector, crop size và các hyperparameter cụ thể được giữ mở để lựa chọn bằng thực nghiệm.

---

## Nguồn nền tảng

- Paper HRP4K: *A high-resolution perspective-view road image dataset for pothole detection*, Scientific Data, 2026.
- Các số liệu dataset, phân bố kích thước và baseline trong proposal được lấy từ paper HRP4K.
