# Tài liệu tổng kết công việc — Data Pipeline (Synthetic Generation & Cleaning)

Tài liệu này ghi lại **đầy đủ, chi tiết** những gì đã thực hiện trong dự án: mục tiêu, quy trình làm việc, quyết định thiết kế, các vấn đề phát hiện và cách xử lý, kết quả kiểm thử, và toàn bộ deliverable. Dùng để làm căn cứ viết báo cáo hoặc bàn giao.

Khác với [`README.md`](../README.md) (giải thích *chiến lược và cách chạy*), tài liệu này ghi lại **quá trình đã làm** — cái gì đã xong, xong như thế nào, phát hiện gì trong lúc làm.

---

## 1. Mục tiêu & phạm vi

Xây dựng pipeline dữ liệu cho bài toán phát hiện gian lận thanh toán (fraud detection), gồm 2 giai đoạn:

1. **Synthetic Data Generation** — sinh thêm 13 trường bối cảnh e-commerce/hành vi (device, IP, thời gian, tài khoản, thanh toán thất bại) cho dataset PaySim gốc, vì dataset gốc chỉ có dữ liệu giao dịch tài chính thô.
2. **Data Cleaning** — kiểm tra và xử lý missing values, duplicates, invalid categories, outliers trên dataset đã sinh, có báo cáo before/after.

**Nguồn dữ liệu:** `Data/PS_20174392719_1491204439457_log.csv` — PaySim/Online Payments Fraud Dataset, 6.362.620 dòng, dùng toàn bộ (không lấy mẫu) xuyên suốt cả 2 giai đoạn.

## 2. Quy trình làm việc

Cả 2 giai đoạn đều đi qua đầy đủ quy trình sau (không code trực tiếp mà không có thiết kế trước):

```mermaid
flowchart LR
    A["Khảo sát dữ liệu thật<br/>(không giả định)"] --> B["Brainstorm<br/>+ đặt câu hỏi làm rõ"]
    B --> C["Viết Design Spec<br/>(docs/superpowers/specs/)"]
    C --> D["Viết Implementation Plan<br/>(docs/superpowers/plans/, TDD)"]
    D --> E["Code từng task theo TDD<br/>(RED -> GREEN -> commit)"]
    E --> F["Review từng task<br/>(spec compliance + code quality)"]
    F --> G["Chạy thật trên<br/>toàn bộ 6.362.620 dòng"]
    G --> H["Review tổng thể toàn nhánh<br/>(whole-branch review)"]
```

**Đặc điểm quan trọng của quy trình:**
- Mọi quyết định thiết kế đều dựa trên **số liệu đo thật** trên dataset, không suy đoán (ví dụ: kiểm tra `nameOrig` có lặp lại không trước khi quyết định sinh dữ liệu theo customer profile hay row-level).
- Mỗi task code đều theo TDD: viết test trước (RED), implement, xác nhận pass (GREEN), rồi mới commit.
- Mỗi task đều được **review độc lập** (không phải người viết code tự chấm), kiểm tra cả spec compliance lẫn code quality.
- Có **review tổng thể toàn bộ nhánh** ở cuối mỗi giai đoạn, dùng model mạnh nhất, tự tính toán lại độc lập (không tin báo cáo cũ) để xác nhận đúng đắn.

---

## 3. GIAI ĐOẠN 1: Synthetic Data Generation

### 3.1. Khảo sát dữ liệu trước khi thiết kế

| Khảo sát | Kết quả | Ảnh hưởng thiết kế |
|---|---|---|
| `nameOrig` duy nhất | 6.353.307 / 6.362.620 (99,85%) | Không có lịch sử khách hàng lặp lại → sinh dữ liệu **row-level**, không xây customer profile |
| Fraud theo loại giao dịch | Chỉ ở `TRANSFER` (4.097) và `CASH_OUT` (4.116); `PAYMENT`/`CASH_IN`/`DEBIT` = 0 | Bản chất fraud là **account-takeover**, không phải gian lận thẻ tại checkout |
| `step` | 1–743, không có khoảng trống | `hour_of_day` suy trực tiếp bằng công thức, không cần random |
| Tỷ lệ fraud tổng | 8.213/6.362.620 = 0,1291% | Khớp với audit đã có trước đó — xác nhận đúng file dữ liệu |

### 3.2. 13 field đã sinh

`hour_of_day`, `is_night_transaction`, `customer_account_age_days`, `device_id`, `browser`, `device_type`, `new_device_flag`, `billing_country`, `ip_country`, `ip_billing_distance_km`, `ip_billing_country_mismatch`, `shipping_billing_mismatch`, `failed_payment_attempts_24h`.

Nguyên tắc: chỉ field có lý do hành vi thật mới điều kiện theo **risk proxy label-free** (`compute_risk_proxy`, tính từ `type`/`amount`/`hour_of_day` — không đọc `isFraud`), giới hạn hệ số 2–4 lần baseline; field không có cơ sở hành vi sinh độc lập. Chi tiết công thức từng field: xem README mục 7 hoặc [`docs/DATA_DICTIONARY.md`](DATA_DICTIONARY.md). **Lịch sử thiết kế:** thiết kế ban đầu điều kiện trực tiếp theo `isFraud` (`if is_fraud: p=0.12 else 0.04`); được viết lại sang label-free sau khi review phát hiện đây là dấu hiệu leakage kinh điển và làm feature không tái tạo được cho giao dịch mới tại thời điểm scoring — xem mục 5, vấn đề #5.

### 3.3. Code

- `src/data_generation/country_centroids.py` — bảng toạ độ 20 quốc gia + haversine distance
- `src/data_generation/split_manifest.py` — split manifest chung 60/20/20 (mục 5, vấn đề #7)
- `src/data_generation/generate_synthetic_fields.py` — 13 hàm sinh field + orchestrator + CLI
- `src/data_generation/check_leakage.py` — đo leakage (AUC/Cramér's V) + sinh data dictionary
- **90 unit test** (bao gồm 11 test của `split_manifest.py`), 100% vectorized (trừ ~18.611 dòng account lặp lại xử lý theo lịch sử, mục 5 vấn đề #6), dùng 1 `numpy.random.Generator(seed=42)` duy nhất cho cả lượt chạy

### 3.4. Kết quả trên dữ liệu thật (6.362.620 dòng)

Row count và tỷ lệ fraud giữ nguyên (0,1291%). Leakage check: **13/13 field PASS** (ngưỡng AUC < 0.75, Cramér's V < 0.5). Chi tiết số đo: xem README mục 9.

---

## 4. GIAI ĐOẠN 2: Data Cleaning

### 4.1. Khảo sát dữ liệu trước khi thiết kế

Chạy trên `transactions_synthetic.parquet` (output Giai đoạn 1):

| Khảo sát | Kết quả |
|---|---|
| Missing values, duplicate toàn dòng, invalid category, số dư âm, format `nameOrig`/`nameDest`, khoảng trống `step`, bất biến chéo giữa các field | **0 tất cả** — dataset sạch về cấu trúc |
| `amount = 0` | 16 dòng — **toàn bộ đều là fraud thật** |
| `amount` outlier (Tukey IQR) | 338.078 dòng (5,31%) |
| `oldbalanceOrg − amount ≠ newbalanceOrig` | 5.118.892 dòng (80,45%) — đặc điểm đã biết của PaySim, không phải lỗi |

### 4.2. Nguyên tắc thiết kế: Flag, không xoá

Vì 16 dòng `amount=0` đều là fraud thật và 80% "balance inconsistent" là đặc điểm nguồn dữ liệu, quyết định: **chỉ xoá dòng khi là lỗi cấu trúc thật** (missing ở cột trọng yếu, duplicate toàn dòng, category không hợp lệ — cả 3 loại này đều = 0 dòng trên dữ liệu thật); mọi bất thường có thể liên quan fraud thì **flag bằng cột boolean, giữ nguyên dòng**.

### 4.3. Code

- `src/data_cleaning/clean_transactions.py` — 6 hàm check/flag (`check_missing_critical`, `dedupe_exact`, `check_invalid_categories`, `flag_amount_outliers`, `flag_zero_amount`, `flag_balance_inconsistency`) + `fit_tukey_fences`/`apply_tukey_fences` (fit chỉ trên train split, mục 5 vấn đề #7) + orchestrator `clean_dataset()` + CLI
- `src/data_cleaning/cleaning_report.py` — sinh `docs/CLEANING_REPORT.md` tự động
- **26 unit test**

### 4.4. Kết quả trên dữ liệu thật

| Check | Hành động | Kết quả |
|---|---|---|
| Missing values | Xoá nếu có | 0 dòng xoá |
| Duplicates | Xoá nếu có | 0 dòng xoá |
| Invalid categories | Xoá nếu có | 0 dòng xoá |
| `is_amount_outlier` | Flag | 338.078 dòng (5,31%) |
| `is_zero_amount` | Flag | 16 dòng |
| `is_balance_inconsistent` | Flag | 5.118.892 dòng (80,45%) |

Row count không đổi: 6.362.620 dòng, 27 cột (24 cột từ Giai đoạn 1 + 3 cột flag mới). Ý nghĩa chi tiết từng cột flag: xem README mục 14 và mục 16 (full field reference).

---

## 5. Các vấn đề đã phát hiện và xử lý trong quá trình làm

Đây là bằng chứng cụ thể cho thấy quy trình kiểm tra hoạt động thật, không chỉ là thủ tục hình thức:

| # | Vấn đề phát hiện | Cách phát hiện | Cách xử lý | Kết quả sau xử lý |
|---|---|---|---|---|
| 1 | `customer_account_age_days` vượt ngưỡng leakage thật (AUC 0,8753 > 0,75) khi chạy trên 6,36M dòng thật | Chạy leakage check lần đầu trên dữ liệu thật | Giảm hệ số fraud (median 150 → 275 ngày), sinh lại | AUC 0,6689 — PASS |
| 2 | Công thức Cramér's V gốc bị lệch dương (bias) với field cardinality lớn — `device_id` có thể báo FAIL giả (~0,48) nếu chạy trên mẫu nhỏ hơn, dù không có tương quan thật | Review độc lập, tự đạo hàm lại toán thay vì tin kết quả cũ | Thay bằng Cramér's V hiệu chỉnh bias (Bergsma, 2013) | `device_id`: 0,0879 → 0,0; vẫn 12/12 PASS, ổn định hơn theo cỡ mẫu |
| 3 | Data dictionary không phân biệt rõ giá trị nào là AUC, giá trị nào là Cramér's V | Review tổng thể toàn nhánh | Thêm nhãn `(AUC)` / `(Cramér's V)` vào từng giá trị | Đọc rõ ràng, không nhầm thang đo |
| 4 | Subagent xác minh dữ liệu thật (Task 8, giai đoạn Cleaning) bị ngắt giữa chừng do hết giới hạn phiên làm việc | Theo dõi trạng thái subagent | Tự kiểm tra lại độc lập từ file parquet thật (không tin báo cáo dở dang), hoàn tất commit còn thiếu | Số liệu khớp chính xác 100% với khảo sát ban đầu |
| 5 | Code sinh 5 field (`customer_account_age_days`, `new_device_flag`, `ip_country`, `shipping_billing_mismatch`, `failed_payment_attempts_24h`) đọc trực tiếp `isFraud` để chọn tham số (`if is_fraud: p=0.12 else 0.04`) — là dấu hiệu leakage kinh điển với người review fraud detection, và feature không tái tạo được cho giao dịch mới tại thời điểm scoring (chưa biết `isFraud`) | Review kỹ thuật độc lập, đối chiếu với yêu cầu Module 6 (API phải score giao dịch mới real-time) | Viết `compute_risk_proxy()` label-free (chỉ dùng `type`/`amount`/`hour_of_day`, cố ý tránh `oldbalanceOrg`/`newbalanceOrig` vì 2 cột này gần-xác-định `isFraud` trong PaySim); 5 hàm sinh field đổi sang nhận `risk_score` thay `is_fraud`; thêm test static + hành vi xác nhận không đọc nhãn; regenerate toàn bộ artifact từ file CSV gốc | 13/13 field vẫn PASS leakage check; AUC 5 field đổi giảm từ 0,55–0,67 xuống 0,51–0,55 (đúng kỳ vọng — tín hiệu yếu hơn vì không còn đọc nhãn trực tiếp); 82/82 test pass |
| 6 | Review kỹ thuật thứ 2 (sau khi đã label-free) phát hiện 4 vấn đề còn sót: (a) `amount_percentile` fit ngay trên batch đang xử lý, leak train/test nếu batch bị chia sau đó; (b) quyết định giảm/tăng hệ số fraud dựa trên AUC đo trên **toàn bộ** dataset — cũng là 1 dạng leak ở tầng quy trình (nhìn nhãn của dòng sẽ-thành-test để chọn tham số); (c) module chưa có ràng buộc kỹ thuật rõ ràng cho việc "offline-only"; (d) `device_id` và `new_device_flag` sinh hoàn toàn độc lập — với ~9.298 account lặp lại (18.611 dòng), `new_device_flag=False` (nghĩa là "thiết bị đã biết") nhưng `device_id` gần như luôn khác thiết bị trước đó của account (~1/50.000 trùng), mâu thuẫn logic | Review kỹ thuật độc lập lần 2, dựa trên checklist 4 điểm cụ thể được yêu cầu rà soát | (a)+(b): tách `fit_amount_percentile_reference()` (chỉ train split) / `apply_amount_percentile()` (mọi dòng); `check_leakage.py` giờ chỉ đo trên train split. (c): thêm module-level docstring OFFLINE-ONLY + docstring chi tiết từng hàm. (d): viết `generate_device_id_and_new_device_flag()` sinh đồng thời, có lịch sử theo account (theo thứ tự `step`), có pool-exhaustion guard | 100/100 test pass (18 test mới); regenerate từ CSV gốc: 13/13 leakage PASS trên train split (70%, tự tách riêng bằng `assign_train_test_split`); verify thực nghiệm trên data thật: 0 vi phạm consistency trên toàn bộ 18.611 dòng thuộc 9.298 account lặp lại |
| 7 | Review từ đồng team (Người 5 — Model Development) chỉ ra 2 vấn đề còn lại: (a) split 70/30 ở vấn đề #6 là utility **riêng của Giai đoạn 1**, không đảm bảo Giai đoạn 2 (`clean_transactions.py`) dùng cùng split khi fit Tukey IQR fences — nếu `is_amount_outlier` dùng làm model feature, fences fit lẫn cả dòng sẽ-thành-test là leak; (b) không có 1 split dùng chung, thống nhất cho toàn bộ pipeline lẫn Model Development, mỗi module tự vẽ split riêng (`seed=123` ở Giai đoạn 1, `seed=456` ở Giai đoạn 2) | Review chéo giữa các thành viên, đối chiếu comment cụ thể trên PR | Cả nhóm chốt quyết định: bỏ 2 split độc lập (70/30 mỗi module), thay bằng **1 split manifest chung 60/20/20** (`src/data_generation/split_manifest.py`, seed=2024), lưu file riêng `data/processed/split_manifest.parquet` (không thêm cột vào `transactions_cleaned.parquet`). `generate_synthetic_fields.py` dùng train split của manifest để fit `amount_percentile_reference`; `clean_transactions.py` dùng đúng cùng manifest để fit Tukey fences (`fit_tukey_fences`); `check_leakage.py` đo trên đúng train split đó. Thêm 60/20/20 thay 70/30 vì Model Development cần thêm tập validation | 116/116 test pass (11 test mới cho `split_manifest.py`); regenerate toàn bộ từ CSV gốc: split manifest đúng tỷ lệ 3.817.572/1.272.524/1.272.524 (60/20/20); 13/13 leakage PASS trên train split mới; 0 vi phạm device consistency (như vấn đề #6); `transactions_cleaned.parquet` giữ nguyên 27 cột, 6.362.620 dòng |

## 6. Kết quả kiểm thử tổng hợp

- **116/116 unit test pass** (90 cho Synthetic Generation + 26 cho Data Cleaning)
- 2 lượt **review tổng thể toàn nhánh** (1 cho mỗi giai đoạn), verdict cả 2 lần: **"Ready to merge: Yes"**, 0 lỗi Critical/Important
- Đã chạy thật và verify độc lập nhiều lần trên toàn bộ 6.362.620 dòng cho cả 2 giai đoạn

## 7. Danh sách đầy đủ deliverable

**Code:**
```
src/data_generation/country_centroids.py
src/data_generation/split_manifest.py
src/data_generation/generate_synthetic_fields.py
src/data_generation/check_leakage.py
src/data_cleaning/clean_transactions.py
src/data_cleaning/cleaning_report.py
tests/data_generation/ (90 test, bao gồm 11 test split_manifest.py)
tests/data_cleaning/ (26 test)
```

**Dữ liệu output** (`data/processed/`, không commit git do dung lượng):
```
split_manifest.parquet                    — split manifest chung 60/20/20 (row_index, split), dùng cho fit reference/fences + Model Development
transactions_synthetic.parquet / .csv     — sau Giai đoạn 1 (24 cột)
transactions_synthetic_sample.csv         — mẫu ~5.000 dòng
transactions_cleaned.parquet / .csv       — sau Giai đoạn 2, BẢN CUỐI CÙNG (27 cột)
transactions_cleaned_sample.csv           — mẫu ~5.000 dòng
```

**Tài liệu:**
```
README.md                                          — chiến lược, logic, cách chạy, full field reference
docs/DATA_DICTIONARY.md                            — data dictionary 13 field synthetic (tự sinh)
docs/CLEANING_REPORT.md                            — báo cáo before/after cleaning (tự sinh)
docs/PROJECT_SUMMARY.md                             — tài liệu này
docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md  — spec thiết kế Giai đoạn 1
docs/superpowers/specs/2026-07-03-data-cleaning-design.md          — spec thiết kế Giai đoạn 2
docs/superpowers/plans/2026-07-03-synthetic-data-nguoi2-plan.md    — kế hoạch triển khai Giai đoạn 1
docs/superpowers/plans/2026-07-03-data-cleaning-plan.md            — kế hoạch triển khai Giai đoạn 2
```

## 8. Giới hạn đã biết (không chặn, đã ghi nhận)

- **Chuyển sang label-free giải quyết leakage-smell và vấn đề tái tạo lúc scoring, nhưng KHÔNG làm dữ liệu thực tế hơn** — 5 field conditional vẫn được tiêm tương quan theo thiết kế (qua risk proxy tự chọn), và về thống kê gần như redundant với `type`/`amount`/`hour_of_day` đã có sẵn. Chi tiết: README mục 19.
- PaySim gốc có leakage sẵn có qua `oldbalanceOrg`/`newbalanceOrig` (không liên quan phần synthetic/cleaning này) — cần Module 5 xử lý riêng, xem README mục 19.
- `split_manifest.py` (60/20/20, seed=2024) chọn dòng bằng permutation ngẫu nhiên, không stratify theo `isFraud`, không time-based theo `step`. Đây **là** split chính thức chung cho cả việc fit reference/fences (Người 2) và Model Development (Người 5) theo quyết định của nhóm (mục 5, vấn đề #7) — không phải utility nội bộ riêng như bản 70/30 trước. Nếu Người 5 cần chiến lược split khác cho mục đích riêng, cần thống nhất lại với cả nhóm trước khi đổi.
- Các hệ số odds-ratio/λ trong sinh dữ liệu synthetic là giả định nghiệp vụ tự đặt, không suy từ số liệu fraud thực tế công khai.
- Ngưỡng outlier IQR (1.5×IQR) là 1 lựa chọn thống kê phổ biến, không phải "đúng duy nhất".
- Cột flag cleaning chỉ đánh dấu, không loại khỏi dataset — quyết định dùng làm feature hay không thuộc về bước sau.
- Một vài ghi chú Minor từ review (vị trí import, 2 CLI tính `clean_dataset` 2 lần, thiếu cột `rows_after` so với spec gốc...) — reviewer xác nhận không cần sửa trước khi dùng.

## 9. Việc chưa làm / ngoài phạm vi

- **EDA, feature engineering, huấn luyện model, deploy, monitoring** — các giai đoạn tiếp theo của pipeline fraud detection, chưa thực hiện trong phạm vi 2 giai đoạn này.
- **File rác trong git chưa xử lý:** `Data.txt` (đã xoá nhưng chưa commit), `phancong.txt`, `yeucau.txt` (chưa track) — cần quyết định commit/gitignore/giữ nguyên.
- **Lịch sử git cũ:** một số commit trước đó có gắn trailer đồng tác giả AI, đã tạm dừng xử lý theo yêu cầu — vẫn còn treo nếu cần dọn dẹp sau này.
