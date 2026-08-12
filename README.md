# HRP4K Benchmark & Analysis Suite

CLI tái lập pipeline Phase 0–3 cho HRP4K. Repository ưu tiên dữ liệu thật, official split và unified COCO prediction; không tự động split lại hay coi smoke result là benchmark khoa học.

## Cài đặt

```bash
python -m venv .venv
.venv/bin/pip install -e '.[vision]'
```

Môi trường smoke đã xác minh được pin tại `requirements/benchmark-lock.txt`. Mỗi run còn lưu toàn bộ `pip freeze`, phiên bản Torch/CUDA/Ultralytics, Git commit và dataset manifest/hash trong thư mục run.

Trong workspace hiện tại có thể dùng trực tiếp `venv/bin/python -m hrp4k_suite`.

## Chạy nhanh toàn pipeline

```bash
venv/bin/python -m hrp4k_suite run-smoke \
  --data HRP4K \
  --weights yolo11n.pt \
  --train-limit 24 \
  --eval-limit 8 \
  --imgsz 320
```

Lệnh này chạy tuần tự:

1. Phase 0: integrity + thống kê annotation + quality sample.
2. Phase 1: tạo dataset nhỏ bằng symlink, train YOLO11n đúng 1 epoch với AMP.
3. Phase 2: smoke inference `resize`, `sliced-nms`, `perspective-grid` và export unified COCO JSON.
4. Phase 3: evaluate + sinh diagnostic report từ prediction đã lưu.

Full training cần cờ xác nhận `--allow-full` và manifest của single official downloaded release đã khớp ba annotation hash. Bản release hiện thiếu một số file train từ nguồn tải; dự án coi đây là official version duy nhất trong khi liên hệ tác giả để xin archive đầy đủ. Smoke subset vẫn luôn mang nhãn `smoke`.

## CLI theo phase

```bash
# Phase 0
venv/bin/python -m hrp4k_suite analyze --data HRP4K --output outputs/phase0

# Chuẩn bị subset deterministic, giữ official split
venv/bin/python -m hrp4k_suite prepare-smoke --data HRP4K --output outputs/smoke/dataset

# Phase 1 smoke training
venv/bin/python -m hrp4k_suite train --smoke \
  --dataset outputs/smoke/dataset/dataset.yaml \
  --weights yolo11n.pt --output outputs/smoke/runs/yolo11n

# Preflight identity/dependencies
venv/bin/python -m hrp4k_suite preflight --data HRP4K --require-official

# Phase 2 inference: resize | uniform-2 | uniform-3 | sliced-nms | sahi | perspective-grid
venv/bin/python -m hrp4k_suite predict \
  --data outputs/smoke/dataset --split test \
  --weights outputs/smoke/runs/yolo11n/weights/best.pt \
  --method sliced-nms --output outputs/smoke/predictions/sliced_nms.json

# Config-driven equivalent
venv/bin/python -m hrp4k_suite run --config configs/experiments/yolo11m_resize_smoke.yaml

# Unified evaluator
venv/bin/python -m hrp4k_suite evaluate \
  --ground-truth outputs/smoke/dataset/test.json \
  --predictions outputs/smoke/predictions/sliced_nms.json \
  --output outputs/smoke/predictions/sliced_nms_metrics.json

# Phase 3: không inference lại
venv/bin/python -m hrp4k_suite diagnose \
  --ground-truth outputs/smoke/dataset/test.json \
  --predictions \
    outputs/smoke/predictions/resize.json \
    outputs/smoke/predictions/sliced-nms.json \
    outputs/smoke/predictions/perspective-grid.json \
  --output outputs/smoke/phase3
```

Ba preset medium cho Phase 1 có thể được chọn bằng `--preset yolov5m-compat|yolov8m|yolo11m`. Preset YOLOv5 dùng compatibility checkpoint `yolov5mu.pt` và không được coi là exact reproduction của original YOLOv5 repository. `hrp4k status` hiển thị provenance và trạng thái của toàn bộ sáu baseline.

`prepare-dataset` mặc định chọn toàn bộ ảnh official release khả dụng; `prepare-smoke` mới dùng giới hạn 24/12/12. Evaluator fail-fast với prediction lỗi; diagnostics cũng dùng cùng validator và bỏ qua metrics/per-image JSON nếu người dùng vô tình truyền wildcard rộng.

`hrp4k status` cho biết baseline nào đã được triển khai và learned method nào còn cần reproduction từ official repository.

## Ranh giới kết quả

- Các file COCO công khai không có metadata city/material trên từng ảnh, nên suite không suy đoán hai trường này.
- `sliced-nms` là sliced inference tự triển khai với overlap + global NMS. `sahi` là integration riêng qua optional official library (`pip install -e '.[sahi]'`).
- `perspective-grid` là geometry baseline thủ công phân bổ nhiều detector crop hơn cho far field, không phải reproduction của learned Two-Plane Prior.
- AutoFocus, AdaZoom, FOVEA, learned TPP và ZoomDet được giữ trong status matrix nhưng chưa tạo số liệu giả.

Protocol đã freeze nằm tại [METHODS](docs/METHODS.md) và [REPRODUCIBILITY](docs/REPRODUCIBILITY.md). Generic runner lưu schema version, experiment ID, detector/method/runtime provenance, CUDA-synchronized latency và canonical predictions trong mỗi output.
