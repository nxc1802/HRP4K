Script có cấu trúc khá rõ và có thể dùng làm prototype kiểm tra luồng, nhưng hiện chưa đủ độ đúng và độ chặt để dùng làm baseline nghiên cứu. Các điểm cần cải thiện quan trọng nhất:

1. Lỗi nghiêm trọng: kích thước heatmap không nhất quán

Dataset tạo heatmap bằng:

grid_h = 540 // 16 = 33
grid_w = 960 // 16 = 60

Scout cũng mặc định trả về 33 × 60, nên phần này khớp về code. Tuy nhiên mô tả trong nhiều chỗ lại ghi 34 × 60 hoặc “stride 16”, trong khi 540 / 16 = 33.75.

Điều này cho thấy output hiện không thực sự là feature map stride 16 chuẩn mà là feature map được nội suy về kích thước tùy chọn. Nên:

* Chọn input chia hết cho stride, ví dụ 960 × 544, cho output 60 × 34.
* Hoặc giữ 960 × 540, nhưng gọi đây là heatmap kích thước 33 × 60, không khẳng định stride chính xác là 16.
* Tốt hơn là lấy kích thước output trực tiếp từ backbone, thay vì hard-code rồi interpolate.

2. MobileNetV3 Scout không thực sự lấy feature stride 16

Code ghi:

self.features = backbone.features

nhưng đây là toàn bộ MobileNetV3-Small backbone, thường tạo feature cuối có stride lớn hơn 16. Sau đó script resize feature bằng interpolation về 33 × 60.

Như vậy Scout không còn là một head stride-16 đúng nghĩa. Đặc biệt, feature cuối đã mất nhiều thông tin không gian của pothole nhỏ.

Nên cải thiện:

* Cắt backbone tại một stage thực sự có stride 16.
* Dùng multi-scale feature, chẳng hạn kết hợp stride 8 và stride 16.
* Thêm FPN-lite hoặc decoder upsampling.
* Kiểm tra số channel bằng forward thử, thay vì hard-code 576.

Đây là điểm rất quan trọng vì pothole nhỏ có thể biến mất trước khi đến feature cuối.

3. Preprocessing không đúng với pretrained weights

Ảnh chỉ được chuẩn hóa:

tensor.float() / 255.0

Trong khi MobileNetV3 pretrained trên ImageNet thường yêu cầu normalization theo mean/std tương ứng. Detector của Torchvision có transform nội bộ, nhưng Scout thì không.

Do đó backbone pretrained của Scout đang nhận phân phối đầu vào khác với lúc pretrain.

Nên dùng:

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

hoặc trực tiếp dùng transform từ MobileNet_V3_Small_Weights.DEFAULT.transforms().

4. Full 4K image được stack theo batch, rất tốn RAM

Dataset trả về toàn bộ tensor ảnh 4K:

image_4k: [3, 2160, 3840]

Sau đó collate_fn dùng:

torch.stack(...)

Với batch_size=2, riêng tensor ảnh float32 đã khoảng:

2 \times 3 \times 2160 \times 3840 \times 4
\approx 199\text{ MB}

Chưa tính Scout image, target, crops và model. Ngoài ra DataLoader phải copy lượng dữ liệu lớn liên tục.

Nên:

* Giữ ảnh dưới dạng uint8 hoặc NumPy cho đến khi cần.
* Không stack ảnh 4K; trả về list.
* Chỉ decode và crop vùng cần thiết.
* Với detector global, resize trực tiếp từ ảnh CPU.
* Với local branch, crop trước rồi mới chuyển sang float tensor.
* Dùng pin_memory=True khi chạy CUDA.

Đây có thể là nút thắt lớn nhất về tốc độ và bộ nhớ.

5. Script đang giả định mọi ảnh đều đúng 3840 × 2160

Dataset có đọc:

orig_h, orig_w = image_cv.shape[:2]

nhưng nhiều phần sau lại dùng:

config.ORIGINAL_WIDTH
config.ORIGINAL_HEIGHT

Ví dụ crop extraction, remapping global box và detector training đều dùng kích thước hard-code.

Nếu dataset có ảnh bị resize, rotated, hoặc có kích thước khác, toàn bộ tọa độ sẽ sai.

Nên truyền orig_w, orig_h thực tế vào:

* extract_crops
* global remapping
* crop clipping
* target scaling
* evaluation

Không nên dùng kích thước ảnh cấu hình làm nguồn chân lý.

6. Fallback ảnh đen che giấu lỗi dữ liệu

Khi cv2.imread thất bại, script tạo một ảnh đen:

image_cv = np.zeros(...)

Điều này rất nguy hiểm trong training vì script vẫn tiếp tục và gán annotation thật lên ảnh giả. Model sẽ học từ cặp ảnh–nhãn sai.

Nên:

* Raise exception rõ ràng.
* Hoặc log và bỏ sample.
* Kiểm tra toàn bộ file ảnh trước khi training.
* Báo danh sách file bị thiếu/hỏng.

Không nên âm thầm thay bằng ảnh synthetic.

7. Focal loss gần như chỉ coi một pixel là positive

Code dùng:

pos_mask = gt_heatmap.eq(1.0)

Gaussian chỉ đạt đúng 1.0 tại tâm nếu tâm rơi chính xác vào tọa độ grid integer. Với tâm bbox thường là số thực, có thể không có pixel nào bằng đúng 1.

Khi đó:

num_pos == 0

và model chỉ bị tối ưu như toàn bộ heatmap là negative.

Đây là một lỗi thuật toán quan trọng.

Nên dùng một trong các cách:

pos_mask = gt_heatmap >= 0.99

hoặc focal loss kiểu CenterNet, trong đó vị trí center được ép chính xác bằng 1:

heatmap[int(cy), int(cx)] = 1.0

Tốt hơn nữa là dùng implementation focal heatmap chuẩn của CenterNet/CornerNet.

8. Coverage loss chưa thực sự đo coverage

Coverage loss hiện chỉ tính tại pos_mask, vốn gần như là pixel tâm:

pos_coverage = (pos_mask * (1 - pred)).pow(2).sum()

Nó không đo xem vùng crop có bao phủ bbox hay không.

Một coverage objective hợp lý hơn có thể là:

* Soft recall trên toàn Gaussian region.
* Max-pooling heatmap trong mỗi GT box.
* Penalize khi tổng activation trong GT region thấp.
* Trực tiếp tối ưu xác suất GT box nằm trong ít nhất một crop.
* Dùng differentiable top-k hoặc surrogate crop coverage loss.

Hiện tên “Coverage Loss” mạnh hơn chức năng thực tế của nó.

9. Detector được train bằng GT heatmap nhưng inference dùng predicted heatmap

Trong training local detector:

crops = crop_extractor.extract_crops(gt_heatmaps[i], img_4k)

Trong inference:

crops = crop_extractor.extract_crops(pred_heatmap, image_4k)

Đây là train–test mismatch. Detector chỉ học trên crop gần như lý tưởng từ ground truth, nhưng lúc test phải xử lý crop lệch, crop thiếu context hoặc crop false positive từ Scout.

Nên thực hiện curriculum:

1. Giai đoạn đầu dùng GT crops.
2. Sau khi Scout đủ tốt, dùng predicted crops.
3. Trộn GT crops và predicted crops.
4. Thêm jitter vị trí, scale và context.
5. Thêm hard-negative crops từ Scout.

Đây là cải thiện quan trọng nhất để pipeline thực tế ổn định.

10. Crop mở rộng nhưng cuối cùng lại bị ép về kích thước cố định

Code tính:

w_crop = max(768, component_width * 1.2)
h_crop = max(512, component_height * 1.2)

nhưng lúc extract lại luôn ép:

x2 = x1 + 768
y2 = y1 + 512

Như vậy nếu connected component lớn hơn crop chuẩn, phần mở rộng đã tính bị mất. Context margin cũng không được bảo toàn đúng.

Cần chọn rõ một thiết kế:

* Fixed-size crops: luôn 768 × 512, component chỉ quyết định tâm.
* Variable-size regions: giữ box mở rộng và resize toàn vùng về detector input.
* Tiled coverage: nếu component quá lớn, chia thành nhiều crop fixed-size.

Hiện code pha trộn cả hai nhưng thực tế chỉ chạy fixed-size.

11. Crop tại biên ảnh có thể không giữ đúng kích thước

Fallback crop dùng:

c_xmin = max(0, ...)
c_xmax = min(width, c_xmin + crop_w)

Nếu tâm gần mép phải, c_xmax bị cắt nhưng c_xmin không dịch ngược lại, khiến crop nhỏ hơn kích thước mong muốn. Sau đó tensor được resize, nhưng tọa độ remap dùng box nhỏ thật, gây biến đổi scale khác nhau.

Nên dùng hàm crop box chuẩn:

x1 = round(cx - crop_w / 2)
x1 = min(max(x1, 0), image_w - crop_w)
x2 = x1 + crop_w

Tương tự cho trục y.

12. Top-K hiện không thực sự “dynamic”

Pipeline luôn chọn tối đa K component sau threshold. Nếu không có component, nó vẫn bắt buộc tạo một crop.

Như vậy mỗi ảnh luôn có ít nhất một local crop, kể cả khi ảnh không có pothole hoặc Scout rất thiếu tự tin.

Một cơ chế dynamic hợp lý hơn:

* K = 0 nếu max score thấp.
* K phụ thuộc số lượng vùng và uncertainty.
* Có tổng compute budget.
* Có minimum distance giữa các crop.
* Có thể chọn thêm crop khi vùng phân tán rộng.
* Dừng khi cumulative heatmap mass đạt ngưỡng.

Nếu HRP4K có ảnh âm tính, fallback Top-1 sẽ làm tăng false positive và chi phí.

13. Score weighting có thể tạo score lớn hơn 1

Code dùng:

local_score * 1.1

Score có thể vượt quá 1.0. Ngoài ra nhân score thủ công không phải calibration đáng tin cậy.

Nên:

* Clamp về [0, 1], ít nhất về mặt kỹ thuật.
* Tốt hơn dùng temperature scaling trên validation.
* Học fusion model từ branch type, score, box size, crop confidence.
* Dùng Weighted Boxes Fusion hoặc Soft-NMS.
* Không nên mặc định local luôn đáng tin hơn global.

14. Fusion là class-agnostic và mất label

Hiện pipeline chỉ hợp nhất boxes và scores, không giữ labels.

Với một class thì chưa gây lỗi ngay, nhưng làm pipeline khó mở rộng và không tương thích đầy đủ với COCO evaluation.

Nên trả về:

{
    "boxes": ...,
    "scores": ...,
    "labels": ...
}

và dùng batched_nms theo class.

15. map_50 hiện là F1, không phải mAP

Đây là lỗi đánh giá nghiêm trọng:

"map_50": round(f1_score, 2)

F1 tại một confidence threshold không phải Average Precision. Không nên gọi đây là mAP@0.5 hoặc dùng trong paper.

Script đã import COCOeval nhưng không sử dụng.

Nên tạo prediction COCO format:

{
  "image_id": ...,
  "category_id": ...,
  "bbox": [x, y, w, h],
  "score": ...
}

sau đó chạy:

COCOeval(coco_gt, coco_dt, "bbox")

Các metric nên có:

* AP@[0.50:0.95]
* AP50
* AP75
* AP small/medium/large
* AR@1, AR@10, AR@100

Đối với bài toán high-resolution, nên thêm AP theo kích thước pothole riêng.

16. Matching prediction–GT chưa chuẩn AP

Evaluation dùng greedy matching theo thứ tự output hiện tại. Dù Faster R-CNN thường trả score giảm dần, code vẫn nên sort rõ theo confidence.

Ngoài ra chỉ tính một operating point tại CONF_THRESH=0.25, không sweep threshold, nên không thể tính precision–recall curve và AP.

Nên:

* Sort prediction theo score.
* Tính matching cho toàn bộ confidence levels.
* Dùng COCOeval hoặc torchmetrics.
* Báo confidence threshold tối ưu trên validation, không chọn tùy ý.

17. Scout region recall đang quá khắt khe và thiếu metric bổ sung

Một GT chỉ được coi là covered khi toàn bộ bbox nằm hoàn toàn trong crop:

gx1 >= cx1 and ...

Metric này hữu ích nhưng chưa đủ.

Nên báo đồng thời:

* Full-box coverage recall.
* Center coverage recall.
* Recall với intersection-over-GT ≥ 0.5, 0.75, 0.9.
* Recall theo kích thước pothole.
* Recall@K.
* Crop area ratio.
* FLOPs hoặc pixel processing ratio.

Điều này giúp đánh giá đúng trade-off accuracy–compute của AdaPoth-Lite.

18. “Best checkpoint” thực tế chỉ là checkpoint cuối

File được đặt tên:

scout_best.pt
detector_best.pt

nhưng không có validation từng epoch và không có cơ chế chọn best.

Nên:

* Evaluate mỗi epoch.
* Lưu last.pt.
* Lưu best_scout_recall.pt.
* Lưu best_map.pt.
* Có early stopping.
* Lưu optimizer, scheduler, epoch, config và random state.

19. Chỉ train 2 epoch trên 20 sample nhưng report ghi “success”

Chế độ debug hữu ích để kiểm tra code, nhưng không thể dùng để kết luận model hoạt động tốt. Báo cáo hiện ghi:

Successfully Executed Local Test

điều này chấp nhận được về execution, nhưng cần tránh diễn giải như validation khoa học.

Nên tách rõ:

* smoke_test
* debug_overfit
* full_training
* benchmark

Một smoke test tốt nên kiểm tra model có overfit được 2–5 ảnh hay không.

20. Không có augmentation

Hiện ảnh chỉ resize. Với pothole detection, nên có:

* Horizontal flip.
* Color jitter.
* Brightness/contrast.
* Blur và noise.
* Shadow/sun glare.
* Weather augmentation.
* Perspective nhẹ.
* Random crop/scale.
* Mosaic hoặc Copy-Paste có kiểm soát.

Tuy nhiên cần bảo toàn đặc trưng bề mặt và không dùng augmentation quá mạnh làm pothole biến dạng phi thực tế.

21. Không xử lý annotation iscrowd, category mapping và bbox clipping

Script gán mọi annotation:

labels.append(1)

Điều này chỉ đúng nếu JSON chỉ chứa pothole và không có category khác.

Nên:

* Kiểm tra category_id.
* Mapping COCO category ID sang contiguous label.
* Bỏ hoặc xử lý iscrowd.
* Clip bbox vào biên ảnh.
* Loại bbox quá nhỏ hoặc bất hợp lệ.
* Báo thống kê số annotation bị loại.

22. Shared detector chưa thật sự tối ưu cho hai domain

Global input và local crop có phân phối rất khác nhau:

* Global: vật thể rất nhỏ.
* Local: vật thể lớn hơn và chi tiết hơn.

Dùng shared weights là hợp lý để tiết kiệm tham số, nhưng BatchNorm và feature distribution có thể xung đột.

Có thể thử:

* Shared backbone, branch-specific normalization.
* Shared backbone nhưng separate detection heads.
* Branch embedding hoặc branch token.
* Global/local feature alignment loss.
* Freeze BatchNorm hoặc dùng GroupNorm.
* Ablation giữa fully shared và partially shared.

23. Detector batch có số ảnh biến thiên mạnh

Mỗi ảnh đầu vào có thể sinh:

* 1 global image.
* Từ 1 đến 4 local crops.

Với batch size 2, detector có thể nhận tới 10 ảnh cùng lúc. Faster R-CNN trên các tensor lớn có thể OOM.

Nên:

* Micro-batch local crops.
* Accumulate gradient.
* Giới hạn tổng pixels mỗi step.
* Group crops theo size.
* AMP mixed precision.
* Gradient clipping.
* Log peak VRAM.

24. Không có AMP, scheduler và gradient stabilization

Nên thêm:

torch.autocast(...)
GradScaler
clip_grad_norm_
CosineAnnealingLR hoặc OneCycleLR
warmup

Đặc biệt Faster R-CNN dễ tốn VRAM; AMP sẽ mang lại lợi ích rõ rệt.

25. Chưa đảm bảo reproducibility đầy đủ

Script mới seed Python, NumPy và Torch. Nên bổ sung:

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

và seed cho DataLoader worker.

Cần lưu:

* Phiên bản Python.
* CUDA/cuDNN.
* Torch/Torchvision.
* Git commit.
* Config đầy đủ.
* Split IDs.
* Dataset checksum hoặc version.

26. Cài package tự động trong script không phù hợp cho research run

pip install ngay khi import có thể:

* Làm thay đổi môi trường bất ngờ.
* Gây version mismatch Torch/Torchvision.
* Không tái lập.
* Không phù hợp máy offline hoặc cluster.

Nên dùng:

* requirements.txt
* pyproject.toml
* Dockerfile
* hoặc một cell cài đặt riêng trong notebook.

Đặc biệt phải pin Torch và Torchvision tương thích.

27. Đường dẫn đang hard-code cho máy cá nhân

Các đường dẫn /Volumes/WorkSpace/... làm script không chạy trực tiếp trên Kaggle, Colab hoặc server.

Nên dùng:

* argparse
* biến môi trường
* YAML config
* đường dẫn tương đối
* auto-detect Kaggle input

Ví dụ:

python train.py \
  --data-dir /kaggle/input/hrp4k \
  --results-dir /kaggle/working/results

28. Chưa có test cho các phép biến đổi tọa độ

Pipeline phụ thuộc mạnh vào mapping:

* Original → Scout grid.
* Original → Global input.
* Original → Local crop.
* Local prediction → Original.
* Crop clipping tại biên.

Nên có unit test round-trip:

GT 4K box
→ local coordinates
→ map back 4K
≈ original box

và test crop tại bốn góc ảnh.

Sai lệch vài pixel có thể ảnh hưởng rõ rệt đến AP75.

29. Visualization thiếu confidence và branch source

Hình hiện chỉ vẽ box đỏ, không có:

* Confidence.
* Global/local origin.
* Crop score.
* Ground-truth match.
* False positive/false negative.

Nên dùng màu riêng cho:

* Global prediction.
* Local prediction.
* Fused prediction.
* Matched TP.
* FP.
* FN.

Điều này rất hữu ích khi phân tích failure mode.

30. Thiếu baseline và ablation

Để script có giá trị nghiên cứu, cần so sánh ít nhất:

* Global-only detector.
* Uniform tiling.
* Random crops.
* GT oracle crops.
* Scout top-1.
* Scout top-2.
* Scout top-4.
* Global + local không fusion calibration.
* Shared detector và separate detector.
* Fixed-K và dynamic-K.

Các metric phải kèm:

* AP.
* Recall.
* Latency.
* Peak VRAM.
* Số megapixel xử lý.
* FLOPs.
* Số crop trung bình.

Thứ tự ưu tiên sửa

Mức P0 — phải sửa trước khi tin kết quả

1. Thay map_50 = F1 bằng COCOeval thật.
2. Sửa Focal Loss vì gt_heatmap.eq(1.0) có thể không có positive.
3. Loại train–test mismatch giữa GT crop và predicted crop.
4. Không hard-code kích thước 3840 × 2160 trong coordinate mapping.
5. Sửa crop logic để context margin không bị mất.
6. Dùng preprocessing đúng cho MobileNet pretrained.
7. Không fallback ảnh đen khi file lỗi.

Mức P1 — cần sửa để train thực tế

1. Không stack toàn bộ ảnh 4K float32.
2. Thêm AMP và micro-batching.
3. Thêm augmentation.
4. Validation mỗi epoch và best checkpoint thật.
5. Dynamic K có thể bằng 0.
6. Calibration và fusion đúng hơn.
7. Lưu labels và dùng COCO prediction format.

Mức P2 — để nâng lên baseline paper

1. Feature stride-8/16 hoặc FPN-lite cho Scout.
2. Predicted-crop curriculum và hard-negative crops.
3. Ablation đầy đủ.
4. Evaluation theo kích thước pothole.
5. Latency/FLOPs/VRAM benchmark.
6. Unit test coordinate mapping.
7. Reproducibility package và config chuẩn.

Kết luận: script hiện là một end-to-end proof of concept khá hoàn chỉnh về mặt bố cục, nhưng metric và một số chi tiết training khiến kết quả thu được chưa đáng tin cậy về mặt nghiên cứu. Hai lỗi nguy hiểm nhất là mAP giả bằng F1 và Focal Loss có khả năng không nhận diện được positive pixel. Sau khi sửa các điểm P0, script mới phù hợp để chạy baseline nghiêm túc trên HRP4K.