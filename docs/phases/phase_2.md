# Next phase — **HRP4K Small-Object Baseline / Benchmark / SOTA**

Ở phase tiếp theo, mục tiêu nên chuyển từ “benchmark detector nào tốt” sang câu hỏi khó hơn:

> **Với cùng một detector và cùng ảnh HRP4K 4K, chiến lược phân bổ độ phân giải nào giúp phát hiện ultra-small pothole tốt nhất trên mỗi đơn vị compute?**

Điều này rất phù hợp với HRP4K vì 3.833/7.217 instances thuộc nhóm ultra-fine `<0.05%`, median box chỉ khoảng `100×35 px`, và các instance cực nhỏ xuống tới khoảng `10×4 px`.  HRP4K còn có forward-facing oblique perspective tương đối ổn định: camera cao khoảng 1,55–1,70 m và pitch ước tính 19–30°, khiến các phương pháp khai thác perspective như Two-Plane Prior đặc biệt đáng thử. 

---

## 0. Gate bắt buộc trước khi làm phase mới

Một vấn đề cần sửa ngay: evaluator hiện tại **không thể dùng cho benchmark publication-grade**.

Trong code hiện tại:

```python
'mAP@0.5': precision * recall
```

đây không phải cách tính Average Precision. 

Phase mới chỉ nên bắt đầu sau khi toàn bộ prediction được đưa về COCO JSON và đánh giá bằng một unified COCO evaluator:

```text
all methods
    ↓
COCO prediction JSON
    ↓
pycocotools / unified evaluator
    ↓
AP50
AP50:95
AP_UF / AP_F / AP_M / AP_L
FPPI
```

Đây là **Milestone 0**.

---

# 1. Taxonomy benchmark mới

Tôi đề xuất chia toàn bộ method thành 4 tầng:

| Tier | Nhóm                | Methods                         | Câu hỏi cần trả lời                                                    |
| ---- | ------------------- | ------------------------------- | ---------------------------------------------------------------------- |
| B0   | Global processing   | Resize-only                     | Resize làm mất bao nhiêu small-object information?                     |
| B1   | Exhaustive local    | Uniform tiling                  | Nếu compute không giới hạn, crop đều cải thiện tới đâu?                |
| B2   | Slicing             | SAHI                            | Overlap + merging tốt hơn tiling đơn giản bao nhiêu?                   |
| B3   | Learned adaptive    | AutoFocus, AdaZoom              | Có thể chỉ chọn vùng đáng zoom không?                                  |
| B4   | Adaptive resampling | FOVEA, Two-Plane Prior, ZoomDet | Có thể “phân phối lại pixel” mà không cần nhiều detector passes không? |

Quan trọng là **không gộp tất cả thành “tiling methods”**. Chúng giải quyết small object theo ba triết lý rất khác nhau:

```text
                HIGH-RES 4K IMAGE
                       │
       ┌───────────────┼─────────────────┐
       │               │                 │
       ▼               ▼                 ▼
   Exhaustive       Sparse crops       Image warp
   processing       / regions          / resampling
       │               │                 │
 Uniform/SAHI   AutoFocus/AdaZoom   FOVEA/TPP/ZoomDet
```

---

# 2. Detector phải được cố định trước

Không nên để:

```text
Resize → YOLO11n
SAHI → YOLO11s
ZoomDet → Faster-RCNN
AdaZoom → custom detector
```

rồi so mAP trực tiếp.

Ta cần một **Controlled Small-Object Benchmark**.

### Main detector

Tôi đề xuất:

```text
YOLO11n
input canvas ≈ 960×544
COCO pretrained
same HRP4K split
same optimizer/training budget
3 seeds
```

Lý do dùng vanilla `YOLO11n`, chưa dùng `YOLO11n-P2`, là để **không trộn improvement từ detector architecture với improvement từ resolution allocation**.

Sau khi benchmark processing methods xong mới chạy:

```text
Top methods
   ×
YOLO11n-P2
```

để xem method có cộng hưởng với small-object detector hay không.

### Secondary reference

Giữ:

```text
YOLO11m
```

để bridge với official HRP4K benchmark, nơi YOLOv11 medium đạt mAP@0.5:0.95 cao nhất trong sáu detector được paper thử. 

---

# 3. Baseline 1 — Resize-only

Đây là **lower bound quan trọng nhất**.

Không chỉ chạy một resolution.

Tôi đề xuất resolution sweep:

```text
R640   → ~640×384
R960   → ~960×544   ← main setting
R1280  → ~1280×736
R1920  → ~1920×1088
```

Giữ aspect ratio gần 16:9 và padding khi cần.

Mục tiêu không đơn giản là tìm resolution tốt nhất mà dựng được curve:

```text
input pixels
    ↓
AP_UF
AP_F
mAP
latency
VRAM
```

Từ đây chúng ta biết:

> **Small-object detection trên HRP4K bị giới hạn bởi model hay bởi input resolution?**

Nếu tăng 640 → 960 → 1280 làm AP_UF tăng mạnh thì lập luận cho high-resolution processing trở nên rất mạnh.

---

# 4. Baseline 2 — Uniform tiling

Đây là **brute-force upper baseline**.

Các setting chính:

```text
2×2 tiles
3×3 tiles

overlap:
0%
10%
20%
30%
```

Mỗi tile được resize về cùng detector canvas.

Ví dụ:

```text
3840×2160
      ↓ 2×2
4 × 1920×1080 source tiles
      ↓
4 × YOLO11n inference
      ↓
coordinate remapping
      ↓
NMS
```

AdaZoom cũng sử dụng uniform partition như một baseline và cho thấy việc chia đều có thể cải thiện detection, nhưng càng chia nhiều thì crop truncation, repeated/incomplete detections và compute càng tăng. ([arXiv][1])

Cần có hai version:

```text
Uniform-local-only
Uniform-global+local
```

Global+local đặc biệt quan trọng vì local crops dễ mất context.

---

# 5. Baseline 3 — SAHI

Đây nên là **small-object baseline bắt buộc**.

SAHI chia ảnh thành overlapping slices, inference trên từng slice rồi merge prediction; framework hiện hỗ trợ các detector như Ultralytics, RT-DETR, MMDetection, TorchVision... ([GitHub][2])

Paper SAHI báo cáo improvement đáng kể trên VisDrone và xView; slicing-aided fine-tuning còn cải thiện thêm so với chỉ sliced inference. ([arXiv][3])

Ta cần chạy cả:

```text
SAHI-I
= detector full-image trained
+ sliced inference

SAHI-SF
= sliced fine-tuning
+ sliced inference
```

### Grid search

```text
slice size:
640
768
960
1280

overlap:
0.10
0.20
0.30
```

Ngoài ra phải tách:

```text
SAHI sliced only
SAHI global + sliced
```

vì SAHI hiện hỗ trợ cả standard + sliced multi-stage inference. ([GitHub][4])

---

# 6. AutoFocus

AutoFocus là **ancestor rất quan trọng của nhóm learned region selection**.

ICCV 2019 AutoFocus không chạy toàn bộ image pyramid. Nó chạy coarse image trước, dự đoán category-agnostic **FocusPixels**, gom chúng thành **FocusChips**, rồi chỉ chạy finer-scale detector trong các chip này. Paper báo cáo có thể giảm lượng pixel xử lý trong image pyramid khoảng 5× với giảm khoảng 1 AP trong setting của họ. ([CVF Open Access][5])

HRP4K adaptation:

```text
4K
 │
 ▼
coarse detector
 │
 ├── normal detection
 │
 └── FocusPixel head
          │
          ▼
     FocusChips
          │
          ▼
  high-res inference
          │
          ▼
       fusion
```

### HRP4K FocusPixel target

Không dùng COCO small/medium/large.

Dùng chính scale bins HRP4K:

```text
Ultra-fine → strongest focus supervision
Fine       → focus supervision
Medium     → optional
Large      → global path
```

Đây sẽ là baseline learned-region-selection rất tốt để so trực tiếp với các phương pháp adaptive sau này.

---

# 7. AdaZoom

AdaZoom đi xa hơn AutoFocus.

Nó dùng **reinforcement learning / policy gradient** để quyết định:

```text
fixation
+
region scale
+
aspect ratio
```

sau đó magnify region với mức zoom phụ thuộc scale. Reward ưu tiên object nhỏ và được cập nhật dựa trên độ khó của detector; paper còn collaborative-train detector và zoom policy. ([arXiv][1])

Pipeline:

```text
image
  ↓
PolicyNet
  ↓
center + scale + aspect ratio
  ↓
adaptive crop
  ↓
magnification
  ↓
detector
  ↓
repeat T steps
  ↓
merge
```

Đây là method rất gần research question của chúng ta:

> **Learned region allocation có tốt hơn exhaustive slicing không?**

Nhưng đây cũng là method khó reproduce nhất trong nhóm crop-based vì có:

```text
RL policy
sequential state
history map
reward design
collaborative training
detector-policy interaction
```

Do đó tôi không khuyên implement AdaZoom trước ZoomDet/TPP.

---

# 8. FOVEA

FOVEA giải quyết bài toán theo hướng khác hoàn toàn.

Thay vì cắt nhiều crop, nó tạo **một ảnh output có kích thước cố định**, nhưng non-uniformly resample ảnh gốc sao cho vùng quan trọng nhận nhiều pixel hơn và background nhận ít pixel hơn. Paper dùng saliency cues như dataset-specific spatial priors hoặc temporal priors; trên autonomous-driving benchmark, authors báo cáo cải thiện mạnh small-object accuracy mà gần như không tăng compute detector. ([arXiv][6])

Official implementation được public với MMDetection. ([GitHub][7])

Pipeline:

```text
4K image
   │
   ▼
saliency distribution
   │
   ▼
differentiable resampler
   │
   ▼
960×544 warped image
   │
   ▼
one detector pass
   │
   ▼
inverse coordinate mapping
```

Đối với HRP4K **image benchmark**, không dùng temporal prior.

Ta nên chạy:

```text
FOVEA-Spatial
```

trong đó spatial prior chỉ được tính từ **training annotations**.

Điều này đặc biệt thú vị vì pothole distribution trong ảnh forward-facing không đồng đều theo không gian.

---

# 9. Learned Two-Plane Perspective Prior

Đây là method tôi đánh giá **rất đáng ưu tiên cho HRP4K**.

CVPR 2023 Two-Plane Perspective Prior sử dụng rough scene geometry gồm ground plane và một plane phía trên để xây differentiable saliency warp. Vùng xa theo perspective được sampling dày hơn để cải thiện small/far-object detection. ([CVF Open Access][8])

Official implementation cũng đã public. ([GitHub][9])

HRP4K gần như là setting tự nhiên cho ý tưởng này vì camera forward-facing nhìn xiên xuống mặt đường. 

### Nhưng có một vấn đề

Original method phụ thuộc vào **vanishing point**.

HRP4K không cung cấp vanishing-point annotation trong dataset description. Do đó cần 3 experiment:

```text
TPP-Fixed
global/fixed VP prior

TPP-AutoVP
vanishing point estimated automatically

TPP-Learned
learned perspective parameters
```

Và thêm một ablation rất đáng giá:

```text
Two-Plane
vs
Ground-Plane-Only
```

Vì pothole nằm gần như hoàn toàn trên mặt đường, plane thứ hai có thể không cần thiết.

Đây có thể trở thành một insight riêng của HRP4K benchmark.

---

# 10. ZoomDet — ưu tiên SOTA candidate cao nhất

Trong các method baseline được so sánh, **ZoomDet là một phương pháp learned-zoom đáng chú ý để làm reference**.

Bài *Adaptive Image Zoom-in with Bounding Box Transformation for UAV Object Detection* xuất bản tại ISPRS Journal of Photogrammetry and Remote Sensing năm 2026. ZoomDet học **non-uniform zoom transformation** bằng lightweight offset prediction + box-based zoom objective, sau đó transform ground-truth box sang zoomed space khi train và map prediction ngược về original space khi inference. Authors thử với cả Faster R-CNN và YOLOv8 và mô tả method là architecture-independent. ([ScienceDirect][10])

Pipeline:

```text
4K image
    │
    ▼
lightweight zoom predictor
    │
    ▼
adaptive non-uniform warp
    │
    ▼
fixed-size zoomed image
    │
    ▼
YOLO11n
    │
    ▼
box inverse transform
```

Điểm khác quan trọng:

```text
SAHI          → many detector passes
AutoFocus     → selected detector passes
AdaZoom       → selected crop passes
ZoomDet       → usually one adaptively warped canvas
```

Do đó nó có khả năng nằm ở một vùng Pareto rất khác.

---

# 11. SOTA không nên được định nghĩa bằng một con số mAP duy nhất

Ở thời điểm hiện tại, tôi sẽ **không gọi bất kỳ method nào là universal SOTA small-object detection**. SOTA phụ thuộc dataset, detector và compute budget.

Đối với benchmark HRP4K, ta nên định nghĩa 4 “SOTA”:

```text
SOTA-Accuracy
highest mAP50:95

SOTA-UltraFine
highest AP_UF

SOTA-Efficiency
highest AP under latency/GFLOPs budget

SOTA-Pareto
non-dominated accuracy–compute methods
```

Ví dụ một method đạt:

```text
AP_UF = 38
latency = 500 ms
```

không nhất thiết tốt hơn:

```text
AP_UF = 36
latency = 60 ms
```

---

# 12. Ba benchmark protocol bắt buộc

Để tránh comparison không công bằng, tôi đề xuất **ba bảng kết quả độc lập**.

### Protocol A — Same detector / inference processing

Cùng một checkpoint YOLO11n:

```text
Resize
Uniform tiling
SAHI-I
```

Không retrain detector.

Mục tiêu: đo riêng tác động của inference processing.

### Protocol B — Method-native training & Mixed Precision (AMP FP16 / BF16)

Cho phép fine-tuning hoặc retraining các mô hình xử lý độ phân giải:

```text
SAHI sliced fine-tuning
AutoFocus FocusPixel training
AdaZoom RL collaborative training
FOVEA training
TPP training
ZoomDet training
```

**Yêu cầu bắt buộc về Tối ưu hóa & Debugging:**
1. **Mixed Precision (AMP FP16 / BF16)**: Tất cả các quá trình training/fine-tuning bắt buộc phải bật Automatic Mixed Precision (AMP) để tối đa tốc độ tính toán trên Tensor Cores và tiết kiệm VRAM.
2. **Smoke Mode (`--smoke`)**: Mọi pipeline training & inference của Phase 2 bắt buộc hỗ trợ cờ `--smoke` (chạy 1–2 epochs, ~50 samples) để debug và kiểm tra tính hợp lệ của code trước khi chạy đợt huấn luyện chính thức.

Mục tiêu:

> mỗi method đạt performance tốt nhất theo đúng philosophy của nó trên cùng điều kiện tối ưu phần cứng.

### Protocol C — Compute-matched

Ví dụ budgets:

```text
1× baseline compute
2×
4×
8×
```

Mỗi method được tune để không vượt budget.

Đây mới là comparison trực tiếp quan trọng nhất khi so sánh các phương pháp xử lý.

---

# 13. Metric suite mới cho small-object benchmark

Giữ official HRP4K:

```text
Precision
Recall
F1
mAP50
mAP50:95
FPPI
```

Paper dùng FPPI riêng trên 300 negative test images, rất hữu ích vì high-resolution crops dễ làm detector phát sinh thêm false positives. 

Nhưng phase này phải thêm:

### Scale accuracy

```text
AP_UF     ultra-fine
AP_F      fine
AP_M      medium
AP_L      large

Recall_UF
Recall_F
```

HRP4K đã định nghĩa chính xác bốn scale bin này. 

### Processing metrics

```text
Average detector calls / image
P95 detector calls

High-resolution processed-area ratio

Average source pixels processed
Average detector-input pixels

Average GFLOPs / image
P95 GFLOPs

Mean latency
P95 latency

Peak VRAM
```

### Adaptive-specific metrics

Cần thêm:

```text
Region Coverage Recall
Ultra-Fine Region Recall

Average selected crops
Crop truncation rate

Duplicate predictions / GT
Fusion suppression count
```

Đối với FOVEA/TPP/ZoomDet:

```text
average magnification at GT
magnification_UF
magnification_F
```

Như vậy chúng ta biết method thắng **vì thực sự đưa thêm resolution vào pothole nhỏ**, chứ không chỉ tình cờ do detector.

---

# 14. Một architecture source code chung

Thay vì copy code từng paper thành project riêng:

```text
hrp4k/
└── processing/
    ├── base.py
    │
    ├── resize.py
    ├── uniform_tiling.py
    ├── sahi.py
    │
    ├── autofocus/
    ├── adazoom/
    │
    ├── fovea/
    ├── two_plane_prior/
    └── zoomdet/
```

Interface nên là:

```python
processor.process(image)
    -> ProcessedViews
```

Một `ProcessedView` chứa:

```text
image
source_region
forward_transform
inverse_transform
cost_metadata
```

Sau đó:

```text
Processor
   ↓
Views
   ↓
Shared Detector
   ↓
Prediction
   ↓
Inverse Mapping
   ↓
Unified Fusion
   ↓
COCO JSON
```

Điều này cực kỳ quan trọng vì **crop-based và warp-based methods vẫn có thể dùng chung detector/evaluator**.

---

# 15. Thứ tự phát triển tôi khuyên dùng

Không nên triển khai theo tuổi paper.

Thứ tự tối ưu về research value là:

```text
M0  Correct COCO evaluator
 │
 ▼
M1  Resize-only
 │
 ▼
M2  Uniform tiling
 │
 ▼
M3  SAHI
 │
 ├─────────────── Classical baseline complete
 │
 ▼
M4  ZoomDet
 │
 ▼
M5  Two-Plane Perspective Prior
 │
 ▼
M6  AutoFocus
 │
 ▼
M7  FOVEA
 │
 ▼
M8  AdaZoom
```

**ZoomDet nên được làm sớm** vì đây là modern adaptive zoom method 2026 và paper có architecture-independent framing. ([arXiv][11])

**Two-Plane nên đứng ngay sau** vì geometry của HRP4K đặc biệt phù hợp với ground-plane perspective. ([CVF Open Access][8])

**AdaZoom làm sau cùng** vì RL + collaborative training làm chi phí reproduction cao hơn rõ rệt. ([arXiv][1])

---

# 16. Benchmark table cuối cùng

Bảng quan trọng nhất của phase này nên có dạng:

| Method      | Type          | AP50:95 ↑ | AP_UF ↑ | AP_F ↑ | Recall_UF ↑ | FPPI ↓ | Avg calls ↓ | Area ratio ↓ | Latency ↓ |
| ----------- | ------------- | --------: | ------: | -----: | ----------: | -----: | ----------: | -----------: | --------: |
| Resize      | Global        |           |         |        |             |        |           1 |         1.00 |           |
| Uniform 2×2 | Exhaustive    |           |         |        |             |        |           4 |         1.00 |           |
| Uniform 3×3 | Exhaustive    |           |         |        |             |        |           9 |         1.00 |           |
| SAHI        | Slicing       |           |         |        |             |        |             |              |           |
| AutoFocus   | Adaptive crop |           |         |        |             |        |             |              |           |
| AdaZoom     | Adaptive crop |           |         |        |             |        |             |              |           |
| FOVEA       | Warp          |           |         |        |             |        |          ~1 |              |           |
| Two-Plane   | Geometry warp |           |         |        |             |        |          ~1 |              |           |
| ZoomDet     | Learned warp  |           |         |        |             |        |          ~1 |              |           |

Sau đó tạo 3 Pareto plots:

```text
AP_UF       vs latency
AP_UF       vs GFLOPs
mAP50:95    vs processed pixels
```

Đây sẽ là **bộ khung Benchmark chuẩn hóa để đánh giá toàn diện các chiến lược xử lý độ phân giải khác nhau**.

---

# 17. Ý nghĩa của Phase 2 đối với Benchmark Suite

Phase 2 thiết lập sự so sánh trực tiếp và khách quan giữa 8 chiến lược xử lý độ phân giải:

```text
Resize-only
      ↓
Uniform Tiling
      ↓
SAHI
      ↓
AutoFocus
      ↓
AdaZoom
      ↓
FOVEA
      ↓
Two-Plane Perspective Prior
      ↓
ZoomDet
```

Mục tiêu chính là xây dựng một **Pareto Frontier chuẩn hóa giữa Accuracy (AP / AP_UF) và Compute Efficiency (FLOPs / Latency / Pixel Count)**.

---

## Scope chốt của Dự án

Dự án chốt phạm vi thực hiện theo 3 giai đoạn đồng bộ:

```text
Phase 1: HRP4K Detector Benchmark
(YOLOv5, YOLOv8, YOLOv11, RT-DETRv1, RT-DETRv2, D-FINE)
          │
          ▼
Phase 2: HRP4K Resolution Allocation Benchmark
(Resize, Uniform Tiling, SAHI, AutoFocus, AdaZoom, FOVEA, Two-Plane Prior, ZoomDet)
          │
          ▼
Phase 3: Deep Dataset-Conditioned Analysis & Explanation
(Resolution sensitivity, Scale analysis, Localization, Perspective/spatial, Material robustness, Negative/FPPI, Compute efficiency, Failure taxonomy, Synthesis & Decision Matrix)
```

Đây là lộ trình nghiên cứu hệ thống: **Phase 1 xác lập baseline mô hình, Phase 2 xác định giới hạn thực nghiệm (SOTA / Pareto Frontier) của các kỹ thuật phân bổ độ phân giải, và Phase 3 đi sâu phân tích, so sánh và diễn giải nguyên lý đằng sau các kết quả.**

[1]: https://arxiv.org/abs/2106.10409 "AdaZoom: Adaptive Zoom Network for Multi-Scale Object Detection in Large Scenes"
[2]: https://github.com/obss/sahi?utm_source=chatgpt.com "GitHub - obss/sahi: Framework agnostic sliced/tiled inference + interactive ui + error analysis plots · GitHub"
[3]: https://arxiv.org/abs/2202.06934?utm_source=chatgpt.com "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"
[4]: https://github.com/obss/sahi/blob/main/docs/cli.md?utm_source=chatgpt.com "sahi/docs/cli.md at main · obss/sahi · GitHub"
[5]: https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html?utm_source=chatgpt.com "ICCV 2019 Open Access Repository"
[6]: https://arxiv.org/abs/2108.12102?utm_source=chatgpt.com "FOVEA: Foveated Image Magnification for Autonomous Navigation"
[7]: https://github.com/tchittesh/fovea "GitHub - tchittesh/fovea: Code for FOVEA: Foveated Image Magnification for Autonomous Navigation (ICCV 2021) · GitHub"
[8]: https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html?utm_source=chatgpt.com "CVPR 2023 Open Access Repository"
[9]: https://github.com/geometriczoom/two-plane-prior "GitHub - geometriczoom/two-plane-prior: Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection · GitHub"
[10]: https://www.sciencedirect.com/science/article/abs/pii/S0924271626000444?utm_source=chatgpt.com "Adaptive image zoom-in with bounding box transformation for UAV object detection - ScienceDirect"
[11]: https://arxiv.org/abs/2602.07512?utm_source=chatgpt.com "Adaptive Image Zoom-in with Bounding Box Transformation for UAV Object Detection"
