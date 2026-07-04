# Thiết kế: Data Cleaning cho transactions_synthetic

**Ngày:** 2026-07-03
**Input:** `data/processed/transactions_synthetic.parquet` (6.362.620 dòng, 23 cột — 11 cột PaySim gốc + 12 field synthetic)
**Output:** `data/processed/transactions_cleaned.parquet` + `docs/CLEANING_REPORT.md`

## 1. Mục tiêu

Kiểm tra và xử lý 4 nhóm vấn đề chất lượng dữ liệu bắt buộc theo checklist (missing values, duplicates, invalid categories, outliers), có báo cáo before/after, mà **không làm mất tín hiệu fraud thật** — nguyên tắc xuyên suốt của module này.

## 2. Sự thật dữ liệu đã khảo sát (trước khi thiết kế, đo trực tiếp trên `transactions_synthetic.parquet`)

| Khảo sát | Kết quả | Kết luận |
|---|---|---|
| Missing values (toàn bộ 23 cột) | 0 | Không cần xử lý, nhưng vẫn cần check code (defensive) |
| Duplicate toàn dòng | 0 | Không cần xử lý |
| Duplicate theo key giao dịch (`step,type,amount,nameOrig,nameDest`) | 0 | Không cần xử lý |
| `type` ngoài 5 giá trị PaySim hợp lệ | 0 | Không cần xử lý |
| `isFraud`/`isFlaggedFraud` ngoài {0,1} | 0 | Không cần xử lý |
| `nameOrig`/`nameDest` sai format (prefix C/M + số) | 0 | Không cần xử lý |
| `step` có khoảng trống bất thường trong [1,743] | 0 gap | Không cần xử lý |
| Cột boolean synthetic (`is_night_transaction`, `new_device_flag`, `shipping_billing_mismatch`) ngoài {True,False} | 0 | Không cần xử lý |
| Bất biến `is_night_transaction` ↔ `hour_of_day` | 0 sai lệch | Không cần xử lý |
| Bất biến `ip_billing_distance_km` ↔ `ip_country`/`billing_country` | 0 sai lệch | Không cần xử lý |
| Số dư âm (4 cột balance) | 0 | Không cần xử lý |
| `amount = 0` | **16 dòng** | Toàn bộ 16 dòng đều `isFraud=1`, `type=CASH_OUT` — **giữ lại, chỉ flag** |
| `amount` outlier (Tukey IQR: Q1−1.5×IQR, Q3+1.5×IQR = (−279.608, 501.719)) | **338.078 dòng (5,31%)** | Giá trị lớn có thể là tín hiệu fraud — **giữ lại, chỉ flag** |
| `oldbalanceOrg − amount ≠ newbalanceOrig` (sai lệch > 0.01) | **5.118.892 dòng (80,45%)** | Đặc điểm đã biết của PaySim (nhiều giao dịch đến merchant không track số dư đích), không phải lỗi nhập liệu — **giữ lại, chỉ flag + ghi rõ trong report là giới hạn nguồn dữ liệu** |

**Kết luận khảo sát:** dataset đầu vào đã rất sạch về cấu trúc (0 missing/duplicate/invalid category/format lỗi). Công việc cleaning thực chất ở đây là: (a) xây dựng đầy đủ các bước kiểm tra/dedupe mang tính **defensive** (không có tác dụng trên lần chạy này vì input hiện tại sạch, nhưng sẽ hoạt động đúng nếu chạy trên dữ liệu khác có vấn đề), và (b) **flag 3 loại bất thường thật đã tìm thấy** mà không xoá dòng nào, để tránh làm mất mẫu fraud hiếm.

## 3. Nguyên tắc thiết kế

1. **Flag, không xoá, với mọi bất thường có thể liên quan đến fraud.** Lý do: 16 dòng `amount=0` đều là fraud thật; xoá outlier `amount` lớn có thể xoá đúng tín hiệu fraud giá trị cao; 80% "balance inconsistent" là đặc điểm nguồn dữ liệu, không phải lỗi.
2. **Chỉ xoá dòng khi đó là lỗi cấu trúc thật** (duplicate toàn dòng, category không nằm trong danh sách hợp lệ, thiếu giá trị ở cột trọng yếu) — những trường hợp này không mang thông tin fraud, chỉ là nhiễu/lỗi.
3. **Mọi check đều vectorized**, dùng chung kiến trúc với `src/data_generation/` (không loop qua 6,3 triệu dòng).
4. **Report before/after là bằng chứng khách quan**, tự sinh từ code (giống `DATA_DICTIONARY.md`), không điền tay.
5. **Không tái tạo lại các check đã có ở module leakage** (`check_leakage.py` đã validate phân phối/tương quan của 12 field synthetic) — module này chỉ tập trung vào tính hợp lệ *cấu trúc* (structural validity), không lặp lại việc đo AUC/Cramér's V.

## 4. Kiến trúc code

```
src/data_cleaning/
  __init__.py
  clean_transactions.py   # Các hàm check/flag/dedupe (vectorized) + orchestrator clean_dataset() + CLI
  cleaning_report.py      # build_cleaning_report_markdown() + CLI ghi docs/CLEANING_REPORT.md
tests/data_cleaning/
  test_clean_transactions.py
  test_cleaning_report.py
```

Kiến trúc song song với `src/data_generation/` (generate + check/report tách riêng), giữ nguyên quy ước: 1 module = 1 trách nhiệm, có test riêng.

## 5. Chi tiết các check

| # | Check | Hàm | Hành động | Cột/kết quả |
|---|---|---|---|---|
| 1 | Missing values | `check_missing_critical(df)` | Đếm NaN trên **toàn bộ 23 cột** (dùng cho report); chỉ **xoá dòng** nếu NaN rơi vào 1 trong 4 cột trọng yếu (`step`,`type`,`amount`,`isFraud` — thiếu 1 trong 4 cột này thì dòng không dùng được cho phân tích fraud); NaN ở cột khác chỉ ghi nhận trong report, không xoá | Trả về `(df_sạch, per_column_na_counts, n_removed)` |
| 2 | Duplicate toàn dòng | `dedupe_exact(df)` | `df.drop_duplicates()`; log số dòng xoá | Trả về `(df_sạch, n_removed)` |
| 3 | Invalid categories | `check_invalid_categories(df)` | Validate `type` ∈ 5 giá trị PaySim; `browser`/`device_type`/`billing_country`/`ip_country` ∈ tập giá trị hợp lệ đã dùng khi sinh; 3 cột boolean ∈ {True,False}. Xoá dòng nào sai bất kỳ cột nào trong nhóm này; log số dòng xoá theo từng cột | Trả về `(df_sạch, {cột: n_removed})` |
| 4 | Outlier `amount` | `flag_amount_outliers(df)` | Tukey IQR fence trên `amount`; **không xoá** | Thêm cột `is_amount_outlier: bool` |
| 5 | Zero-amount | `flag_zero_amount(df)` | `amount == 0`; **không xoá** | Thêm cột `is_zero_amount: bool` |
| 6 | Balance inconsistency | `flag_balance_inconsistency(df)` | `abs(oldbalanceOrg - amount - newbalanceOrig) > 0.01`; **không xoá** | Thêm cột `is_balance_inconsistent: bool` |

**Thứ tự chạy trong `clean_dataset(df)`:** check 1 → 2 → 3 (loại bỏ lỗi cấu trúc trước) → 4 → 5 → 6 (thêm flag trên dữ liệu đã loại lỗi cấu trúc). Mỗi bước ghi lại số liệu before/after vào 1 dict kết quả, dùng cho report ở mục 6.

## 6. Cleaning Report (`docs/CLEANING_REPORT.md`)

Cấu trúc bảng before/after cho mỗi check:

`check_name | rows_before | rows_flagged_or_removed | action (removed/flagged) | rows_after | note`

Phần đầu report ghi rõ: **80,45% `is_balance_inconsistent` là đặc điểm đã biết của PaySim** (không phải lỗi cần sửa) — để người đọc report không hiểu nhầm là lỗi nghiêm trọng.

## 7. Output

- `data/processed/transactions_cleaned.parquet` — cùng 23 cột input + 3 cột flag mới (`is_amount_outlier`, `is_zero_amount`, `is_balance_inconsistent`) = 26 cột. Row count = 6.362.620 trừ đi số dòng bị xoá thật ở check 1-3 (dự kiến 0 dựa trên khảo sát hiện tại, nhưng code phải xử lý đúng nếu có).
- `data/processed/transactions_cleaned_sample.csv` — mẫu ~5.000 dòng stratified theo `isFraud` (giữ quy ước như module synthetic data).
- `docs/CLEANING_REPORT.md` — sinh tự động từ `cleaning_report.py`.

## 8. Giới hạn đã biết

- Ngưỡng outlier IQR (1.5×IQR, chuẩn Tukey) là lựa chọn thống kê phổ biến, không phải ngưỡng "đúng duy nhất" — ghi rõ trong report để người đọc biết đây là 1 lựa chọn phương pháp, có thể điều chỉnh nếu bước modeling sau cần ngưỡng khác.
- Outlier/inconsistency chỉ được **flag**, không loại khỏi dataset — bước feature engineering/modeling phía sau cần tự quyết định có dùng các cột flag này làm feature hay không.
- Check "invalid categories" cho cột synthetic categorical dựa trên chính danh sách giá trị đã dùng để sinh (`BROWSER_WEIGHTS.keys()`, `DEVICE_TYPE_WEIGHTS.keys()`, `COUNTRY_WEIGHTS.keys()` từ `src/data_generation/`) — nếu ai đó sinh lại dữ liệu với danh sách category khác, cần đồng bộ lại.

## 9. Compliance Checklist

| Yêu cầu | Đáp ứng | Trạng thái |
|---|---|---|
| Missing values | `check_missing_critical` | ✅ |
| Duplicates | `dedupe_exact` | ✅ |
| Invalid categories | `check_invalid_categories` | ✅ |
| Outliers | `flag_amount_outliers` (+ 2 flag đặc thù fraud bổ sung) | ✅ |
| Before/after comparison | `docs/CLEANING_REPORT.md` | ✅ |
| Deliverable: Cleaned dataset | `data/processed/transactions_cleaned.parquet` | ✅ |
| Deliverable: Cleaning summary/report | `docs/CLEANING_REPORT.md` | ✅ |

## 10. Deliverables cụ thể (đường dẫn file)

- `src/data_cleaning/clean_transactions.py`
- `src/data_cleaning/cleaning_report.py`
- `data/processed/transactions_cleaned.parquet` + `data/processed/transactions_cleaned_sample.csv`
- `docs/CLEANING_REPORT.md`
