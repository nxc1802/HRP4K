# HRP4K Documentation Hub

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật và nghiên cứu của dự án **HRP4K Benchmark & Analysis Suite**.

Tài liệu được phân loại theo 5 chuyên mục chính:

```
docs/
├── README.md                      # Trang mục lục & tổng quan hệ thống tài liệu
│
├── paper/                         # Bối cảnh khoa học & bài báo gốc
│   ├── s41597-026-07317-w.pdf    # Bài báo gốc công bố trên Nature Scientific Data (2026)
│   └── s41597-026-07317-w.md     # Bản phân tích chi tiết toàn bộ nội dung bài báo
│
├── architecture/                  # Kiến trúc phần mềm & nguyên tắc tái lập
│   ├── upgrade3.0.md             # Đặc tả thiết kế kiến trúc Upgrade 3.0 (v0.5.0)
│   └── reproducibility.md        # Giao thức tái lập & các chốt chặn khoa học (Invariants)
│
├── phases/                        # Đặc tả chi tiết từng Phase trong Pipeline
│   ├── overview.md               # Tổng quan lộ trình Phase 0–3
│   ├── phase_0.md                # Phase 0: Phân tích & Kiểm tra toàn vẹn Dataset
│   ├── phase_1.md                # Phase 1: Huấn luyện Baseline & Detector Adapters
│   ├── phase_2.md                # Phase 2: Phân bổ Độ phân giải (Resolution Allocation)
│   ├── phase_2_1.md              # Phase 2.1: Tích hợp thư viện SAHI & Sliced Inference
│   └── phase_3.md                # Phase 3: Chẩn đoán sâu, Phân tích lỗi & Biên Pareto
│
├── methodology/                   # Phương pháp luận & ma trận thuật toán
│   └── methods.md                # Danh mục các phương pháp, phép biến đổi hình học & ranh giới heuristic
│
└── guides/                        # Sổ tay hướng dẫn vận hành & thực thi
    ├── run_full_pipeline.md      # Hướng dẫn chạy toàn bộ pipeline (Local & Server)
    └── run_full_pipeline.sh      # Script bash tự động hóa thực thi pipeline
```

---

## 1. Bối cảnh Khoa học (`paper/`)

- [**Bài báo Nature Scientific Data (PDF)**](paper/s41597-026-07317-w.pdf): Bản in chính thức của bài báo *"A high-resolution perspective-view road image dataset for pothole detection"* (Nature Portfolio, 2026, 13:961).
- [**Phân tích & Tóm lược Bài báo (Markdown)**](paper/s41597-026-07317-w.md): Tổng quan toàn diện về động lực nghiên cứu, quy trình HITL (Human-in-the-loop), đặc thù đối tượng ultra-fine (<0.05% diện tích) và kết quả benchmark baseline.

---

## 2. Kiến trúc Hệ thống & Giao thức Tái lập (`architecture/`)

- [**Đề án Kiến trúc Upgrade 3.0**](architecture/upgrade3.0.md): Tài liệu thiết kế chi tiết quá trình chuyển đổi sang package layout `src/hrp4k/`, modular YAML configuration, và 5 abstraction cốt lõi (*Config, Experiment, Artifact, Phase, Scientific Contract*).
- [**Giao thức Tái lập (Reproducibility Protocol)**](architecture/reproducibility.md): Các quy chuẩn bắt buộc về khóa băm SHA-256 dữ liệu, split cấp video, đo độ trễ đồng bộ CUDA với warm-up và ranh giới không rò rỉ dữ liệu (leakage boundaries).

---

## 3. Quy trình Triển khai theo Phase (`phases/`)

- [**Tổng quan Kế hoạch Phase 0–3**](phases/overview.md): Tóm lược mục tiêu, đầu vào và đầu ra của 4 Phase trong pipeline.
- [**Phase 0 — Dataset Audit & Integrity**](phases/phase_0.md): Quy chuẩn kiểm tra bbox, tính tương quan Spearman không gian $y_{\text{bottom}} \leftrightarrow \log(\text{Area})$, đo distribution shift (JS/KS) và tạo symlink view.
- [**Phase 1 — Detector Baseline**](phases/phase_1.md): Thiết lập Ultralytics adapter, cấu hình SGD 150 epoch với AMP và cơ chế chặn full training nếu thiếu dữ liệu chính thức.
- [**Phase 2 — Resolution Allocation**](phases/phase_2.md): Cơ chế phân bổ độ phân giải (Resize, Uniform Crop, Sliced-NMS, Perspective-Grid) và ánh xạ ngược tọa độ thuận/nghịch.
- [**Phase 2.1 — SAHI Integration**](phases/phase_2_1.md): Tích hợp thư viện SAHI chính thức với bộ metrics tương thích chuẩn COCO.
- [**Phase 3 — Deep Diagnostics**](phases/phase_3.md): Đánh giá COCO AP/AR, phân tích kích thước bbox hiệu dụng ở các canvas, ma trận chuyển dịch cứu vật thể (object rescue transition) và tìm biên Pareto (Accuracy vs. CAF).

---

## 4. Phương pháp luận (`methodology/`)

- [**Methods Registry & Transforms**](methodology/methods.md): Bảng phân loại chi tiết trạng thái của từng phương pháp (`ready`, `optional-ready`, `external-required`), nguyên tắc không gán heuristic đơn giản cho phương pháp learned (như TPP, FOVEA, ZoomDet).

---

## 5. Hướng dẫn Vận hành (`guides/`)

- [**Hướng dẫn Chạy Toàn bộ Pipeline**](guides/run_full_pipeline.md): Hướng dẫn chi tiết từng bước từ cài đặt môi trường, phân tích dữ liệu, smoke run đến chạy benchmark chính thức trên GPU.
- [**Script Tự động hóa Pipeline (`run_full_pipeline.sh`)**](guides/run_full_pipeline.sh): Script shell hỗ trợ chạy nhanh toàn bộ quy trình với các tham số cấu hình linh hoạt.
