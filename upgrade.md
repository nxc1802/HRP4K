
## Proposal — Raw-4K Shallow Scout (MobileNetV3-Small Stem + Stage 1)

### 1. Objective

Đánh giá khả năng của một Scout cực nhẹ trong việc **trực tiếp định vị các vùng đáng ngờ trên ảnh raw 4K**, chỉ bằng phần đầu của MobileNetV3-Small.

$$
\boxed{4K\ Image \rightarrow Stem+Stage1 \rightarrow Region\ Heatmap}
$$

### 2. Architecture

* **Backbone:** MobileNetV3-Small pretrained.
* **Input:** ảnh **raw 4K (3840×2160)**, không resize trước backbone.
* **Backbone depth:** chỉ **Stem + Stage 1**.
* **Các stage còn lại:** loại khỏi forward path.
* **Head:** lightweight convolutional heatmap head.
* Output: dense **region/objectness heatmap**.

```text
Raw 4K
  ↓
MobileNetV3-Small Stem
  ↓
Stage 1
  ↓
Lightweight Scout Head
  ↓
Region Heatmap
  ↓
Candidate ROI
```

### 3. Ground Truth

Từ GT bounding boxes tạo **expanded binary region target**, mở rộng khoảng **20%** quanh object để ưu tiên coverage thay vì localization chính xác.

### 4. Training

* Khởi tạo Stem + Stage 1 từ pretrained weights.
* Scout head khởi tạo ngẫu nhiên.
* **Full fine-tune toàn bộ Stem + Stage 1 + Scout Head.**
* Objective chính: tối đa hóa khả năng **cover GT object**.

Loss:

$$
L=L_{\text{heatmap}}+\lambda L_{\text{coverage}}
$$

### 5. Region Generation

Heatmap:

$$
H\rightarrow Threshold
\rightarrow Connected\ Components
\rightarrow Context\ Expansion
\rightarrow NMS
\rightarrow Top-K
$$

Các ROI cuối cùng sẽ được dùng về sau làm input cho Local Detector.

### 6. Evaluation

Metric chính:

$$
\boxed{\text{Region Recall @ 0.75}}
$$

Một GT được coi là covered nếu:

$$
\frac{|GT\cap ROI|}{|GT|}\ge0.75
$$

Báo cáo thêm:

* Region Recall @ 0.50 / 0.90
* Mean GT Coverage
* Average K
* Processed Area Ratio
* False Region Rate
* GFLOPs
* Peak VRAM
* Latency

### 7. Thành công được định nghĩa

Scout được coi là đạt yêu cầu nếu **Region Recall rất cao trong khi computation vẫn đủ thấp để justify việc xử lý raw 4K**.

Kết quả cuối cùng cần trả lời một câu hỏi:

> **Stem + Stage 1 của MobileNetV3-Small có đủ khả năng nhìn trực tiếp ảnh 4K để làm Region Scout hay không?**

Nếu **có**, đây sẽ là Global encoder được mang sang bước phát triển **Local Detector** tiếp theo.
