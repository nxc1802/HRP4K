Tôi đề xuất bộ **experiment tables** như sau:

### Table 1 — Main benchmark

Đây là bảng chính trong paper, dùng để so sánh các configuration/method hiện có.

**Metric nên có:**

* Precision
* Recall
* F1
* mAP@0.5
* mAP@0.5:0.95
* FPPI
* Latency (ms/image)
* FPS

Không cần đưa quá nhiều thông tin model vào bảng này nếu đã có ở Experimental Setup.

---

### Table 2 — Scale-level performance

**Nên có riêng một table**, vì HRP4K có phân bố cực mạnh về small/ultra-fine objects: 3,833 ultra-fine, 1,078 fine, 1,099 medium và 1,207 large. 

Cấu trúc nên là:

| Method | Overall | Ultra-fine | Fine | Medium | Large |
| ------ | ------: | ---------: | ---: | -----: | ----: |
| ...    |     ... |        ... |  ... |    ... |   ... |

Metric chính:

**mAP@0.5:0.95**

Có thể thêm **mAP@0.5** nếu bảng không quá rộng.

Tôi **không khuyến nghị** chia Precision/Recall/F1 thành từng scale. Như vậy bảng sẽ quá lớn mà giá trị giải thích không tăng tương ứng.

---

### Table 3 — Efficiency

Tách khỏi accuracy:

| Method | Params | GFLOPs | Detector Calls | Latency | FPS | Peak VRAM |
| ------ | -----: | -----: | -------------: | ------: | --: | --------: |

Trong đó đặc biệt giữ:

* Params
* GFLOPs
* Detector calls
* Latency
* FPS
* Peak VRAM

Các metric này phù hợp với hướng benchmark hiện tại của repo; tài liệu benchmark cũng đã định hướng đánh giá đồng thời accuracy, scale và computational cost. 

---

### Table 4 — Material-level performance

Nếu experiment Asphalt/Concrete hiện đã có đầy đủ cho các method cần so sánh:

| Method | Asphalt mAP50 | Asphalt mAP50:95 | Concrete mAP50 | Concrete mAP50:95 |
| ------ | ------------: | ---------------: | -------------: | ----------------: |

**Không nhất thiết phải là main table.** Có thể để supplementary nếu paper đã có quá nhiều bảng.

Lý do giữ experiment này là paper gốc đã xác định rõ concrete khó hơn asphalt và đã benchmark theo pavement type. 

---

## Full metrics nên xử lý thế nào?

Nếu experiment hiện tại có **nhiều metric hơn** nữa, không cần bỏ dữ liệu.

Tạo một **Supplementary Table S1 — Full Benchmark Results**:

| Method |  P |  R | F1 | AP50 | AP75 | AP50:95 | FPPI | Latency | FPS | VRAM | Calls |
| ------ | -: | -: | -: | ---: | ---: | ------: | ---: | ------: | --: | ---: | ----: |

Như vậy:

* **Main paper:** reviewer nhìn nhanh được kết quả quan trọng.
* **Supplementary:** toàn bộ kết quả vẫn được công khai, reproducible.
* Không phải hy sinh các metric đã chạy.

Paper gốc HRP4K cũng sử dụng Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 và FPPI làm bộ metric benchmark chính. 

---

# Requirement chốt cho experiment tables

**Main paper:**

1. **Overall benchmark table**

   * P / R / F1 / AP50 / AP50:95 / FPPI / Latency / FPS

2. **Scale-level table**

   * Overall / Ultra-fine / Fine / Medium / Large
   * ưu tiên AP50:95

3. **Efficiency table**

   * Params / GFLOPs / Calls / Latency / FPS / VRAM

4. **Material-level table** *(nếu cần giữ trong main)*

   * Asphalt vs Concrete
   * AP50 và AP50:95

**Supplementary:**

5. **Full metrics table**

   * toàn bộ metric đã experiment
   * không bỏ dữ liệu chỉ vì main table cần gọn.

### Một nguyên tắc quan trọng

**Không tạo table chỉ vì “đã có metric”.** Mỗi table phải trả lời một câu hỏi:

* **Table 1:** Model/method nào tốt hơn overall?
* **Table 2:** Performance thay đổi thế nào theo object scale?
* **Table 3:** Accuracy đó phải trả giá bao nhiêu về computation?
* **Table 4:** Performance có phụ thuộc pavement material không?

Đó là cấu trúc tôi cho là hợp lý nhất để **viết paper từ experiment hiện có**, thay vì biến paper thành một catalogue toàn bộ số liệu.
