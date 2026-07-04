# Synthetic Contextual Data — Chiến lược & Cách chạy

Module này sinh thêm các trường bối cảnh e-commerce/hành vi (device, IP, thời gian, tài khoản, thanh toán thất bại...) cho dataset PaySim gốc, phục vụ Module 1 (Business Understanding & Data Generation) của đề bài Real-Time Payment Fraud Detection — phần việc của **Người 2 (Data Engineer)**.

Tài liệu đầy đủ (lập luận chi tiết, đối chiếu yêu cầu đề bài, kế hoạch triển khai từng bước):
- Spec (thiết kế + lý do): [`docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md`](docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md)
- Plan (12 task TDD): [`docs/superpowers/plans/2026-07-03-synthetic-data-nguoi2-plan.md`](docs/superpowers/plans/2026-07-03-synthetic-data-nguoi2-plan.md)
- Data dictionary (sinh tự động từ code, có số liệu đo thực tế): [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)

## 1. Chiến lược tóm tắt

**Nguồn dữ liệu:** `Data/PS_20174392719_1491204439457_log.csv` (PaySim/Online Payments Fraud Dataset, 6.362.620 dòng, dùng toàn bộ — không lấy mẫu).

**3 sự thật dữ liệu quyết định thiết kế** (đo trực tiếp trên file gốc trước khi code):
1. 99,85% giá trị `nameOrig` chỉ xuất hiện đúng 1 lần → **không có lịch sử khách hàng lặp lại** đáng kể, nên mọi field được sinh **theo từng dòng giao dịch (row-level)**, không xây "customer profile" bền vững qua nhiều giao dịch.
2. Fraud trong PaySim chỉ tồn tại ở `TRANSFER`/`CASH_OUT` (rút/chuyển tiền) → bản chất là **account-takeover** (chiếm quyền tài khoản), không phải gian lận thẻ tại checkout — các field như device lạ, IP lệch quốc gia, giờ bất thường khớp tự nhiên với narrative này.
3. `step` = số giờ kể từ lúc mô phỏng bắt đầu (1–743) → `hour_of_day`/`is_night_transaction` **suy trực tiếp bằng công thức**, không cần sinh ngẫu nhiên.

**Nguyên tắc tiêm tín hiệu fraud:**
- Chỉ field có lý do hành vi thực sự (new device, IP lệch, giờ đêm, tài khoản mới, mismatch địa chỉ, số lần thất bại) mới điều kiện theo `isFraud`. Field không có cơ sở hành vi (`browser`, `device_type`, `device_id`) sinh **độc lập với fraud** — tránh "mọi field đều tương quan" (dấu hiệu leakage giả tạo).
- Mọi hệ số (odds-ratio, Poisson λ, median gap) **giới hạn 2–4 lần baseline**, ghi rõ là giả định nghiệp vụ tự đặt (không bịa số liệu thực tế).
- Sau khi sinh, **đo leakage khách quan**: AUC đơn biến (numeric/boolean) hoặc Cramér's V (categorical) so với `isFraud`. Ngưỡng FAIL: AUC ≥ 0.75 hoặc Cramér's V ≥ 0.5. Nếu FAIL → giảm hệ số, sinh lại — không đổi ngưỡng để "cho qua".

## 2. 12 field synthetic

| Field | Loại | Base → Fraud |
|---|---|---|
| `hour_of_day` | derived | `(step - 1) % 24` |
| `is_night_transaction` | derived | `hour_of_day ∈ [0,5]` |
| `customer_account_age_days` | conditional (lognormal) | median 400 → 275 ngày¹ |
| `device_id` | độc lập | pool cố định 50.000 UUID (Faker) |
| `browser` | độc lập | Chrome 55% / Safari 20% / Edge 12% / Firefox 8% / Other 5% |
| `device_type` | độc lập | mobile 65% / desktop 30% / tablet 5% |
| `new_device_flag` | conditional (Bernoulli) | p = 0.04 → 0.12 |
| `billing_country` | độc lập | categorical, 20 quốc gia |
| `ip_country` | conditional | match rate 0.93 → 0.80 |
| `ip_billing_distance_km` | derived | haversine(centroid[ip_country], centroid[billing_country]) |
| `shipping_billing_mismatch` | conditional (Bernoulli) | p = 0.05 → 0.15 |
| `failed_payment_attempts_24h` | conditional (Poisson) | λ = 0.15 → 0.6 |

¹ Giá trị 275 (thay vì 150 như thiết kế ban đầu) đã được **tinh chỉnh sau khi kiểm tra leakage trên dữ liệu thật** — xem mục 4.

Lập luận chi tiết cho từng con số: xem spec mục 4, hoặc cột `business_assumption` trong `docs/DATA_DICTIONARY.md`.

## 3. Cấu trúc code

```
src/data_generation/
  country_centroids.py       # Bảng tọa độ 20 quốc gia + haversine distance
  generate_synthetic_fields.py  # 12 hàm sinh field + orchestrator + CLI (CSV -> Parquet)
  check_leakage.py           # Đo AUC/Cramér's V + sinh docs/DATA_DICTIONARY.md
tests/data_generation/       # 49 unit test (pytest)
```

Mọi hàm sinh dữ liệu đều **vectorized** (numpy/pandas), dùng chung **một** `numpy.random.Generator` (`seed=42`) cho cả lượt chạy — không loop qua từng dòng, không nhiều nguồn random rời rạc.

## 4. Kết quả chạy trên dữ liệu thật

Chạy trên toàn bộ 6.362.620 dòng: row count và tỷ lệ fraud giữ nguyên (0,1291%) so với file gốc — bước sinh dữ liệu **không làm thay đổi class imbalance**, đúng phạm vi Module 1 (xử lý imbalance kỹ thuật là việc của Module 4).

Lượt kiểm tra leakage đầu tiên phát hiện `customer_account_age_days` **vượt ngưỡng thật** (AUC 0,8753). Đã giảm hệ số fraud (median 150 → 275 ngày) theo đúng quy trình ở mục "cơ chế chống leakage", sinh lại và đạt AUC 0,6689. Sau điều chỉnh, **12/12 field PASS**:

| Field | Metric | Giá trị |
|---|---|---|
| `hour_of_day` | AUC | 0.6336 |
| `is_night_transaction` | AUC | 0.6217 |
| `customer_account_age_days` | AUC | 0.6689 |
| `device_id` | Cramér's V | 0.0879 |
| `browser` | Cramér's V | 0.0007 |
| `device_type` | Cramér's V | 0.0007 |
| `new_device_flag` | AUC | 0.5419 |
| `billing_country` | Cramér's V | 0.0017 |
| `ip_country` | Cramér's V | 0.0052 |
| `ip_billing_distance_km` | AUC | 0.5651 |
| `shipping_billing_mismatch` | AUC | 0.5495 |
| `failed_payment_attempts_24h` | AUC | 0.6589 |

Số liệu đầy đủ (kèm data type, unit, formula, business assumption): [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## 5. Cách chạy

Yêu cầu: Python 3.13 (ví dụ `C:\ProgramData\miniconda3\python.exe`), chạy trong git-bash/MSYS.

```bash
# 1. Tạo venv và cài dependency (chỉ cần 1 lần)
"/c/ProgramData/miniconda3/python.exe" -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. Chạy test
.venv/Scripts/python.exe -m pytest tests/ -v

# 3. Sinh synthetic data từ dataset gốc (Data/PS_20174392719_1491204439457_log.csv)
PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.generate_synthetic_fields
# -> data/processed/transactions_synthetic.parquet
# -> data/processed/transactions_synthetic_sample.csv (mẫu ~5.000 dòng, stratified theo isFraud)

# 4. Kiểm tra leakage + sinh data dictionary
PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.check_leakage
# -> docs/DATA_DICTIONARY.md
```

Nếu bước 4 báo FAIL cho field nào: mở `generate_synthetic_fields.py`, giảm hằng số fraud của field đó (theo hướng về gần baseline hơn), chạy lại bước 3 rồi bước 4 đến khi tất cả PASS.

## 6. Giới hạn đã biết

- Các hệ số odds-ratio/λ là giả định nghiệp vụ tự đặt, không suy từ số liệu fraud thực tế công khai nào (PaySim không đi kèm dữ liệu loại này).
- `shipping_billing_mismatch` được diễn giải lại thành "địa chỉ giao dịch khác địa chỉ đăng ký" do fraud trong PaySim là account-takeover, không phải gian lận thẻ tại checkout.
- 9.313 `nameOrig` có lặp lại (0,15%) được xử lý như dòng độc lập, không có logic đặc biệt.

## 7. Bàn giao cho các vai trò khác

- **Feature Engineer:** các field string (`device_id`, `billing_country`, `ip_country`, `browser`, `device_type`) cần encode; `ip_billing_distance_km`, `failed_payment_attempts_24h` đã là numeric, dùng trực tiếp được.
- **ML Engineer:** các feature balance hiện có (`sender_balance_delta`...) cho AUC-PR rất cao (~0.9988) — khả năng leak sẵn có trong PaySim (fraud thường rút sạch số dư). Nên train có/không các feature đó để so sánh, tránh đánh giá sai giá trị của field synthetic mới.
