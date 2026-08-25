# HRP4K Benchmark Suite — Documentation Hub

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật và nghiên cứu của dự án **HRP4K Benchmark & Analysis Suite**.

Tài liệu được tinh gọn tập trung thành các danh mục sau:

```
docs/
├── README.md                 # Trang mục lục & tổng quan hệ thống tài liệu
├── experiments_master.md     # 📊 BẢNG TỔNG HỢP MASTER KẾT QUẢ THỰC NGHIỆM DUY NHẤT
│
├── paper/                    # Bối cảnh khoa học & bài báo gốc
│   ├── s41597-026-07317-w.pdf# Bài báo gốc công bố trên Nature Scientific Data (2026)
│   └── s41597-026-07317-w.md # Bản phân tích chi tiết toàn bộ nội dung bài báo
│
└── guides/                   # Sổ tay hướng dẫn vận hành & thực thi
    ├── run_full_pipeline.md  # Hướng dẫn chi tiết chạy toàn bộ pipeline
    ├── run_full_pipeline.sh  # Script bash tự động hóa thực thi
    └── run_missing_slicing.sh# Script chạy toàn bộ benchmark Slicing còn thiếu
```

---

## 📊 Bảng Kết Quả Thực Nghiệm Chính Thức

Toàn bộ kết quả thực nghiệm chuẩn hóa trên tập **$900$ ảnh Test split** được lưu trữ duy nhất tại:
👉 [**`docs/experiments_master.md`**](experiments_master.md)

Tài liệu bao gồm:
1. **Bảng Kết Quả Cốt Lõi (`YOLO11m` & `D-FINE`)**:
   - **Native 4K UHD**: `yolo11m_4k` ($\mathbf{55.05\%}\text{ mAP}_{50}$), `dfine_4k` *(đang huấn luyện)*.
   - **Resize 640x640**: `yolo11m_640` ($37.27\%$), `dfine_640` ($37.37\%$).
   - **Patch 640x640**: `yolo11m_patch640`, `dfine_patch640`.
   - **Slicing Methods (`perspective-grid`, `sahi`, `sliced-nms`)**: Áp dụng trên Resize/4K Model và Patch Model.
2. **Bảng Thực Nghiệm Bổ Trợ**: `yolo11m_1280` ($48.98\%$), `dfine_zoomdet640` ($42.07\%$), `RT-DETRv2/v1`, `YOLOv8m`, `YOLOv5m`.
3. **Bóc tách theo 4 Dải Kích Thước (Scale Bins)**: Ultra-fine ($<16\text{ px}$), Fine, Medium, Large.
4. **Phân tích Hiệu Năng & Độ Trễ (Efficiency & Latency)**: Đo thời gian suy luận (ms), số detector calls, peak VRAM và throughput FPS.

---

## 📖 Bối Cảnh Khoa Học (`paper/`)

- [**Bài báo Nature Scientific Data (PDF)**](paper/s41597-026-07317-w.pdf): Bản in chính thức của bài báo *"A high-resolution perspective-view road image dataset for pothole detection"* (Nature Portfolio, 2026, 13:961).
- [**Phân tích & Tóm lược Bài báo (Markdown)**](paper/s41597-026-07317-w.md): Tổng quan toàn diện về bộ dữ liệu $6.003$ ảnh 4K UHD, đối tượng siêu nhỏ ultra-fine và video-level split.

---

## 🚀 Hướng Dẫn Vận Hành (`guides/`)

- [**Hướng dẫn Chạy Toàn bộ Pipeline (`guides/run_full_pipeline.md`)**](guides/run_full_pipeline.md): Hướng dẫn chi tiết từng bước từ tải dữ liệu Hugging Face, kiểm tra toàn vẹn, huấn luyện baseline đến chạy benchmark slicing.
- [**Script Tự Động Hóa (`guides/run_full_pipeline.sh`)**](guides/run_full_pipeline.sh): Script bash một lệnh thực thi toàn bộ pipeline.
- [**Sổ Tay Lệnh Toàn Năng (`commands.md`)**](file:///Volumes/WorkSpace/Project/HRP4K/commands.md): Tra cứu nhanh mọi câu lệnh CLI `hrp4k`, Marimo Lab, GPU Server và Kaggle.
