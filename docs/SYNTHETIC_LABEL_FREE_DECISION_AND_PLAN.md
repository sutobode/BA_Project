# Báo cáo Quyết định & Kế hoạch: Chuyển Synthetic Data sang Label-Free Generation

**Ngày:** 2026-07-23
**Phạm vi:** Module 1 — Synthetic Data Generation (và ảnh hưởng lan sang Module 3 Cleaning + phần report)
**Trạng thái:** Đề xuất — cần nhóm/giảng viên duyệt hướng trước khi sửa code

---

## 1. Bối cảnh & xung đột cần giải quyết

Pipeline sinh synthetic data hiện tại sinh 5 field **có điều kiện theo nhãn `isFraud`** (label-conditional), ví dụ:

```python
p_new_device = 0.12 if is_fraud else 0.04
```

Một thành viên (Data Engineer) phản biện rằng cách này sai và yêu cầu chuyển sang **label-free generation** (hàm sinh không đọc `isFraud`). Tài liệu này phân tích trung lập cả hai hướng, chỉ ra một điểm kỹ thuật **cả hai phía đều bỏ sót**, và chốt giải pháp cuối cùng kèm kế hoạch thực hiện.

---

## 2. Phân tích kỹ thuật

### 2.1. Sự thật nền: cả hai cách đều tạo dữ liệu giả

Các field bối cảnh (`account_age`, `new_device_flag`, `ip_country`, `shipping_billing_mismatch`, `failed_payment_attempts_24h`...) **không tồn tại trong PaySim**. Do đó câu hỏi không phải "cách nào thật hơn" (không cách nào thật), mà là "cách dựng dữ liệu giả nào **bảo vệ được** theo mục tiêu dự án". Ràng buộc quyết định: **đề bài bắt buộc deploy API chấm điểm giao dịch mới theo thời gian thực (Module 6).**

### 2.2. Hai tiêu chí quan trọng nhất — và chúng xung đột nhau

| Tiêu chí | Ý nghĩa |
|---|---|
| **(1) Serve-time consistency** | Feature phải tính lại được cho giao dịch mới lúc score (khi chưa biết `isFraud`) |
| **(2) Không thành proxy của nhãn** | Feature không được tương quan gần-hoàn-hảo với `isFraud`, nếu không bài toán bị tầm thường hóa và model chỉ học lại giả định đã tiêm |

### 2.3. Đánh giá hai giải pháp

| Tiêu chí | Giải pháp A (label-conditional, hiện tại) | Giải pháp B (label-free, đề xuất) |
|---|---|---|
| (1) Tính lại được lúc score | ❌ Không — feature chỉ định nghĩa qua nhãn, giao dịch mới không có nhãn để tính | ✅ Có — feature là hàm của observable, giao dịch mới có sẵn observable |
| (2) Không thành proxy | ✅ Có — đã ép AUC<0.75 / Cramér's V<0.5 | ⚠️ **Không đảm bảo** — xem 2.4 |
| Qua kiểm tra vệ sinh ("không có target trong input") | ❌ Không | ✅ Có |

### 2.4. Điểm mấu chốt cả hai phía bỏ sót

> **Trên PaySim, chính các cột observable gốc (`type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`) đã gần như xác định `isFraud`.**

Fraud trong PaySim được định nghĩa bằng cơ chế rút cạn tài khoản: fraud ⟺ `TRANSFER`/`CASH_OUT` + `amount = oldbalanceOrg` + `newbalanceOrig = 0`. Một luật đơn giản trên các cột số dư này bắt fraud gần như hoàn hảo (lý do bản pipeline cũ từng đạt AUC-PR ~0.9988).

**Hệ quả:** nếu sinh feature label-free nhưng lấy từ đúng các cột số dư đó, feature sẽ **thừa hưởng tính gần-xác-định** và có thể **leak mạnh hơn cả giải pháp A** (vốn đã bị ép giới hạn). Nói cách khác:

> **"Label-free" chỉ nghĩa là không đọc cột nhãn — KHÔNG đảm bảo không leak.**

Đây là điểm teammate hiểu chưa đủ (tưởng label-free là tự động an toàn), và cũng là điểm giải pháp A hiện tại không mắc phải vì có bước check leakage.

### 2.5. Làm rõ thuật ngữ (tránh sửa sai hướng)

Cụm "label-independent" mà teammate dùng **dễ gây hiểu nhầm**. Phải chốt rõ:

- ✅ **Đúng:** "label-free generation" = hàm sinh **không đọc `isFraud`**. Feature **vẫn được phép tương quan** với fraud (một cách yếu, gián tiếp qua observable).
- ❌ **Sai:** "feature độc lập thống kê với `isFraud`" (tương quan = 0). Nếu hiểu thế này, 12 field thành nhiễu vô nghĩa, model không học được gì.

---

## 3. Giải pháp cuối cùng (chốt)

**Không phải A thuần, cũng không phải B thuần.** Lấy **kiến trúc của B + kỷ luật của A + một ràng buộc mới**:

1. **Sinh label-free** — hàm sinh không nhận `isFraud` làm input. (theo hướng B → deploy được, qua vệ sinh)
2. **Giữ nguyên bước check leakage** (AUC<0.75 / Cramér's V<0.5) và giới hạn cường độ. (theo kỷ luật A → vì label-free KHÔNG tự động hết leak)
3. **Ràng buộc mới bắt buộc:** khi xây risk-proxy từ observable, **TRÁNH các cột định nghĩa fraud** (`oldbalanceOrg`, `newbalanceOrig`, tỷ lệ rút cạn số dư). Chỉ dùng observable **liên hệ lỏng** với fraud (giờ giao dịch, loại giao dịch, độ lớn `amount` tương đối) + nhiễu có kiểm soát → feature mang tín hiệu **yếu, thực tế**, không phải proxy.

### 3.1. `isFraud` được dùng khi nào?

- **KHÔNG** dùng lúc sinh feature.
- **CHỈ** dùng **sau khi sinh xong**, để: (a) đo association / leakage check, (b) train model, (c) evaluate.

### 3.2. Thừa nhận trung thực (bắt buộc ghi trong report)

Dù làm cách nào, feature vẫn là **giả và có tính vòng lặp** — tương quan với fraud là do thiết kế, không phải bằng chứng về sức mạnh dự đoán trên fraud thật. Cách duy nhất thoát vòng lặp hoàn toàn là model thẳng trên cột thật của PaySim, nhưng đề bài bắt buộc sinh contextual data, nên hybrid label-free ở trên là phương án đúng nhất trong khuôn khổ đề bài.

---

## 4. Phạm vi thay đổi theo từng field

| # | Field | Hiện tại | Cần sửa? | Ghi chú |
|---|---|---|---|---|
| 1 | `hour_of_day` | Derived từ `step` | ✅ Giữ nguyên | Đã label-free |
| 2 | `is_night_transaction` | Derived từ `hour_of_day` | ✅ Giữ nguyên | Đã label-free. **Chốt định nghĩa:** `between(0,5)` hay `between(2,5)` — nhóm chọn 1 |
| 3 | `customer_account_age_days` | Conditional theo `isFraud` | ❌ **Sửa** | Sinh từ observable an toàn + nhiễu |
| 4 | `device_id` | Random độc lập | ✅ Giữ nguyên | Đã label-free |
| 5 | `browser` | Random độc lập | ✅ Giữ nguyên | Đã label-free |
| 6 | `device_type` | Random độc lập | ✅ Giữ nguyên | Đã label-free |
| 7 | `new_device_flag` | Conditional theo `isFraud` | ❌ **Sửa** | Sinh từ observable an toàn + nhiễu |
| 8 | `billing_country` | Random độc lập | ✅ Giữ nguyên | Đã label-free |
| 9 | `ip_country` | Conditional theo `isFraud` | ❌ **Sửa** | Sinh từ observable an toàn + nhiễu |
| 10 | `ip_billing_distance_km` | Derived từ `ip_country` | ❌ Tự sửa khi #9 sửa | |
| 11 | `shipping_billing_mismatch` | Conditional theo `isFraud` | ❌ **Sửa** | Sinh từ observable an toàn + nhiễu |
| 12 | `failed_payment_attempts_24h` | Conditional theo `isFraud` | ❌ **Sửa** | Sinh từ observable an toàn + nhiễu |

**Tổng: 5 field đọc nhãn cần thiết kế lại (+1 field suy theo). 6 field còn lại đã label-free sẵn.**

---

## 5. Kế hoạch thực hiện

### Nguyên tắc thứ tự (bắt buộc, không làm ngược)

```
1. Nhóm duyệt hướng (tài liệu này) + chốt 2 quyết định chung (mục 5.1)
2. Sửa code sinh dữ liệu (label-free) + thêm/sửa test
3. Chạy lại → regenerate artifacts → chạy lại leakage check (SỐ MỚI)
4. RỒI mới viết lại report theo số mới
```

> ⚠️ **Tuyệt đối không viết report "label-free" khi code còn đọc nhãn** — sẽ thành mô tả sai sự thật.

### 5.1. Bước 0 — Hai quyết định chung cần chốt trước

- [ ] **Chốt thuật ngữ:** thống nhất "label-free generation" (hàm không đọc nhãn), KHÔNG phải "feature không tương quan với nhãn". (mục 2.5)
- [ ] **Chốt định nghĩa time feature:** `is_night_transaction = hour_of_day.between(0,5)` hay `between(2,5)`? Chọn 1 con số, dùng chung cho Member 4 & 6.

### 5.2. Bước 1 — Brainstorm + Spec thiết kế risk-proxy label-free

- [ ] Với mỗi field trong 5 field cần sửa, xác định: **sinh từ observable nào** (loại trừ cột số dư định nghĩa fraud), phân phối gì, tham số bao nhiêu, nhiễu bao nhiêu.
- [ ] Viết spec: `docs/superpowers/specs/YYYY-MM-DD-label-free-synthetic-design.md`
- [ ] Ghi rõ business assumption + valid range cho từng field.

### 5.3. Bước 2 — Sửa code (TDD)

- [ ] Sửa `src/data_generation/generate_synthetic_fields.py`: 5 hàm sinh field bỏ tham số `is_fraud`, thay bằng input observable.
- [ ] Tách **shared time-feature utility** để Member 4 & 6 tái sử dụng, không viết lại nhiều nơi.
- [ ] Thêm test bắt buộc:
  - [ ] `assert "isFraud" not in <generation inputs>` — nhãn không được truyền vào hàm sinh
  - [ ] không field nào tạo trực tiếp từ target
  - [ ] reproducible với seed cố định (đã có)
  - [ ] schema / range / dtype đúng (đã có)
  - [ ] class distribution không đổi (đã có — `test_generate_all_synthetic_fields_does_not_alter_class_distribution`)
  - [ ] không cột trùng tên (đã có — `test_generate_all_synthetic_fields_produces_no_duplicate_columns`)
- [ ] **Giữ nguyên** `check_leakage.py` — vẫn chạy AUC/Cramér's V (quan trọng hơn bao giờ hết vì label-free không tự động hết leak).

### 5.4. Bước 3 — Regenerate toàn bộ artifacts + gắn version

- [ ] `transactions_synthetic.parquet` / `.csv` (enriched dataset)
- [ ] `transactions_cleaned.parquet` / `.csv` (cleaned dataset)
- [ ] `docs/DATA_DICTIONARY.md` (tự sinh)
- [ ] `docs/CLEANING_REPORT.md` (tự sinh)
- [ ] Gắn nhãn version, ví dụ `dataset_version: v2_label_free`
- [ ] Chạy lại leakage check trên dữ liệu mới → **số liệu sẽ khác**, xác nhận 12/12 vẫn PASS (và không có field nào rơi về ~0 hoàn toàn → mất tín hiệu).

### 5.5. Bước 4 — Sửa report

- [ ] Viết lại `docs/report/data_sources_and_generation.tex` Section Synthetic:
  - [ ] Mô tả pipeline label-free: input observable của từng field, cơ chế, business assumption, seed, valid range, leakage check, giới hạn.
  - [ ] **Xóa** mọi đoạn khẳng định `isFraud` được dùng để điều kiện phân phối.
  - [ ] Cập nhật bảng: synthetic fields, leakage-validation results (số mới), data dictionary excerpt, số cột, field definitions, time-window definition.
- [ ] Rà soát `docs/report/data_cleaning_and_preprocessing.tex`: đảm bảo dùng đúng dataset version cuối (row count, column count, duplicate count, missing count, before/after).
- [ ] Đồng bộ `README.md`, `docs/PROJECT_SUMMARY.md`, slide.

### 5.6. Bước 5 — Kiểm tra tổng thể

- [ ] Chạy full test suite — toàn bộ PASS.
- [ ] Whole-branch review độc lập.
- [ ] Xác nhận artifact khớp code (chạy lại và so byte-for-byte).

---

## 6. Việc cần code ngoài repo này

Ba dấu hiệu sau cho thấy có code của thành viên khác chưa gộp vào repo này — cần lấy về để xử lý tận gốc:

1. **"Hai mô tả pipeline khác nhau"** (report Section 3 vs Section 6) — repo này chỉ có 1 pipeline.
2. **"Cột `customer_account_age_days` trùng sau merge"** — repo này không có cột trùng (đã test).
3. **"Time feature `between(2,5)`"** — khác bản hiện tại `between(0,5)` → có ít nhất 1 bản implementation khác.

→ Cần code/branch của Member 4 & 6 để thống nhất shared utility và loại nguyên nhân trùng cột.

---

## 7. Đầu ra cuối cùng (Definition of Done)

- [ ] `generate_synthetic_fields.py` — không đọc `isFraud` trong bất kỳ hàm sinh nào
- [ ] Shared time-feature utility, tái sử dụng bởi Member 4 & 6
- [ ] Leakage check giữ nguyên, 12/12 PASS trên dữ liệu mới
- [ ] Cleaned final dataset (version mới)
- [ ] Data dictionary mới (tự sinh)
- [ ] Report Section Synthetic + Cleaning đã cập nhật theo số mới
- [ ] Full test suite PASS
- [ ] Limitations ghi rõ: dữ liệu vẫn là synthetic có tính vòng lặp, không phải bằng chứng dự đoán trên fraud thật
