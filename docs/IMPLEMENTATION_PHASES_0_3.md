# Phân tích paper và kế hoạch triển khai Phase 0–3

## 1. Kết luận từ paper gốc

HRP4K là benchmark phát hiện ổ gà có ba đặc tính chi phối thiết kế dự án:

1. Ảnh gốc đồng nhất 3840×2160 nhưng object rất nhỏ: 3.833/7.217 instance là ultra-fine (`area/image < 0,05%`), bbox trung vị khoảng 100×35 px. Downsample về canvas detector làm mất mạnh thông tin theo cả hai chiều.
2. Official split là video-level 4.203/900/900, vì vậy không được random split. Local hiện thiếu 1.917 ảnh train nhưng valid/test đầy đủ; mọi loader phải lọc theo file ảnh thật.
3. Dataset có 2.000 negative image và chênh lệch asphalt/concrete lớn. Vì vậy AP tổng là chưa đủ; FPPI, scale-aware recall/AP và material gap mới phản ánh khả năng triển khai. Tuy nhiên COCO JSON local không chứa city/material per image, nên material/city analysis cần metadata bổ sung từ nguồn chính thức.

Paper benchmark sáu detector medium trong 150 epoch. Vì paper không pin đủ version/hyperparameter, mục tiêu đúng là protocol reproduction, không phải bit-for-bit reproduction. Với ràng buộc local smoke, kết quả sinh ra trong repository chỉ chứng minh pipeline hoạt động.

## 2. Phase 0 — Dataset Analysis & Integrity

Đã triển khai:

- xác nhận declared/available image, orphan/missing image, invalid bbox và official split;
- global/scale/shape/spatial/density/split/quality statistics;
- raw `difficulty_index.csv`, spatial grid CSV/JSON và difficulty tags;
- bỏ qua ảnh local bị thiếu mà không thay đổi annotation/split chính thức;
- report machine-readable và human-readable.

Không thể tái tạo trung thực city/material subgroup do metadata vắng trong ba COCO JSON.

## 3. Phase 1 — Detector Baseline

Đã triển khai nền tảng chung:

- subset smoke deterministic, giữ nguyên official split, dùng symlink để không nhân bản ảnh 4K;
- YOLO training adapter với AMP, environment/config snapshot và full-training guard;
- unified COCO prediction format;
- evaluator pycocotools cho AP50/AP75/AP50:95, operating-point precision/recall/F1/FPPI và HRP4K scale bins;
- per-image error records phục vụ Phase 3.

Local smoke chỉ chạy YOLO11n. Matrix YOLOv5/v8/v11, RT-DETRv1/v2 và D-FINE cần môi trường/checkpoint chính chủ riêng khi chạy benchmark thật; không thể kết luận reproduction từ một epoch smoke.

## 4. Phase 2 — Resolution Allocation

Đã triển khai qua một interface chung:

- resize-only;
- uniform 2×2/3×3;
- sliced inference có overlap và global NMS (`sahi`);
- perspective-band baseline;
- đo detector calls, source pixels, processed-area ratio, latency và fusion suppression.

Các tên AutoFocus, AdaZoom, FOVEA, learned Two-Plane Prior và ZoomDet được ghi trong reproduction status nhưng cố ý không map sang heuristic. Chúng cần loss/training/coordinate transform của paper gốc; gắn tên learned method vào một crop rule đơn giản sẽ làm benchmark sai.

## 5. Phase 3 — Deep Diagnostics

Đã triển khai luồng đọc prediction đã lưu, không inference lại:

- effective bbox size tại canvas 640/960/1280/1920;
- per-image TP/FP/FN/localization errors;
- scale metrics, FPPI, detector-call/latency summary;
- report có interpretation boundary và reproduction-status matrix.

Các phân tích material, bootstrap nhiều seed, failure taxonomy thủ công và method suitability map chỉ có ý nghĩa sau khi có prediction benchmark thật và metadata tương ứng. Smoke report không được dùng để xếp hạng method.

## 6. Tiêu chí hoàn thành thực tế

Phase 0–3 ở mức engineering smoke hoàn thành khi một lệnh CLI có thể đi từ dữ liệu local → kiểm tra → smoke train → prediction COCO → unified metrics → diagnostic report. Publication-grade completion vẫn cần full official training trên compute phù hợp, official external implementations, ít nhất ba seed cho comparison chính và metadata material/city.
