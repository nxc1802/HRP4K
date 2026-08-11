# HRP4K Benchmark & Analysis Suite

CLI tái lập pipeline Phase 0–3 cho HRP4K. Repository ưu tiên dữ liệu thật, official split và unified COCO prediction; không tự động split lại hay coi smoke result là benchmark khoa học.

## Cài đặt

```bash
python -m venv .venv
.venv/bin/pip install -e '.[vision]'
```

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
3. Phase 2: inference resize baseline và export unified COCO JSON.
4. Phase 3: evaluate + sinh diagnostic report từ prediction đã lưu.

Full local training bị chặn có chủ đích. `train` bắt buộc có `--smoke`.

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

# Phase 2 inference: resize | uniform-2 | uniform-3 | sahi | perspective-bands
venv/bin/python -m hrp4k_suite predict \
  --data outputs/smoke/dataset --split test \
  --weights outputs/smoke/runs/yolo11n/weights/best.pt \
  --method sahi --output outputs/smoke/predictions/sahi.json

# Unified evaluator
venv/bin/python -m hrp4k_suite evaluate \
  --ground-truth outputs/smoke/dataset/test.json \
  --predictions outputs/smoke/predictions/sahi.json \
  --output outputs/smoke/predictions/sahi_metrics.json

# Phase 3: không inference lại
venv/bin/python -m hrp4k_suite diagnose \
  --ground-truth outputs/smoke/dataset/test.json \
  --predictions outputs/smoke/predictions/*.json \
  --output outputs/smoke/phase3
```

`hrp4k status` cho biết baseline nào đã được triển khai và learned method nào còn cần reproduction từ official repository.

## Ranh giới kết quả

- Các file COCO công khai không có metadata city/material trên từng ảnh, nên suite không suy đoán hai trường này.
- `sahi` ở đây là sliced-inference tương thích framework với overlap + global NMS, không phải SAHI sliced fine-tuning.
- `perspective-bands` là geometry baseline minh bạch, không được gọi là reproduction của learned Two-Plane Prior.
- AutoFocus, AdaZoom, FOVEA, learned TPP và ZoomDet được giữ trong status matrix nhưng chưa tạo số liệu giả.
