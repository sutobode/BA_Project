# Thiết kế: Sinh Synthetic Contextual Data cho Fraud Detection (Người 2 — Data Engineer)

**Ngày:** 2026-07-03 (cập nhật 2026-07-23 — chuyển sang label-free generation, xem mục 0)
**Phạm vi:** Module 1 (Business Understanding & Data Generation) theo `yeucau.txt`
**Người thực hiện:** Người 2 (Data Engineer), hỗ trợ bởi Người 3 khi cần cho EDA

## 0. Cập nhật 2026-07-23 — Chuyển sang Label-Free Generation

Thiết kế ban đầu (mục 2-9 dưới đây) điều kiện 5 field trực tiếp theo `isFraud` (`if is_fraud: p=0.12 else 0.04`). Sau review, phát hiện 2 vấn đề: (a) field không tái tạo được cho giao dịch mới lúc scoring vì `isFraud` chính là thứ cần dự đoán, chưa biết tại thời điểm đó; (b) đọc trực tiếp nhãn để sinh feature là dấu hiệu leakage kinh điển với người review fraud detection.

**Đã sửa:** thay `isFraud` bằng `risk_score = compute_risk_proxy(type, amount, hour_of_day)` — một risk proxy **label-free** (không đọc `isFraud`, cố ý không dùng `oldbalanceOrg`/`newbalanceOrig` vì 2 cột số dư này gần-xác-định `isFraud` trong PaySim). 5 field vẫn nội suy tuyến tính giữa base và high-risk với đúng biên độ 2-4x như thiết kế gốc — chỉ đổi biến điều kiện. Thêm field mới `ip_billing_country_mismatch` (boolean, derived từ `ip_country != billing_country`).

**Kết quả đo lại trên 6.362.620 dòng thật:** 13/13 field PASS leakage check (ngưỡng không đổi). AUC của 5 field đổi cơ chế giảm xuống 0.51-0.55 (từ 0.55-0.67 cũ) — đúng kỳ vọng, vì risk proxy label-free có tương quan yếu hơn với nhãn thật so với đọc trực tiếp nhãn.

**Đọc mục 2-9 dưới đây với lưu ý:** mọi chỗ nói "điều kiện theo `isFraud`" / "conditional-on-fraud" / số AUC cũ nay đã lỗi thời về mặt cơ chế cụ thể — **nguyên tắc thiết kế** (row-level, giới hạn hệ số 2-4x, derived khi có thể, đo leakage khách quan) vẫn đúng và không đổi, chỉ đổi *biến điều kiện* từ nhãn sang risk proxy quan sát được. Số liệu chính xác nhất: `docs/DATA_DICTIONARY.md` (tự sinh từ code, luôn khớp code hiện tại) và README mục 7-9.

## 1. Mục tiêu

Sinh thêm các trường bối cảnh e-commerce/hành vi (device, IP, thời gian, tài khoản, thanh toán thất bại...) gắn vào dataset PaySim gốc, sao cho:
- Đáp ứng đúng danh sách field mà đề bài liệt kê (account age, device/browser fingerprint, shipping/billing mismatch, failed payment attempts, IP-to-billing-country distance, time-of-day pattern).
- Dữ liệu **thực tế, có căn cứ hành vi**, không phải random vô nghĩa.
- **Không leak nhãn** — field không được tương quan với `isFraud` mạnh đến mức làm bài toán mất ý nghĩa.
- Có **data dictionary đầy đủ** với logic sinh và giả định nghiệp vụ, đúng yêu cầu bắt buộc của đề.

## 2. Sự thật dữ liệu đã kiểm chứng (căn cứ ra quyết định thiết kế)

Trước khi thiết kế, đã chạy khảo sát trực tiếp trên file gốc `Data/PS_20174392719_1491204439457_log.csv` (6.362.620 dòng, không lấy mẫu — dùng toàn bộ theo quyết định đã chốt):

| Sự thật đo được | Giá trị | Hệ quả thiết kế |
|---|---|---|
| Tổng số dòng | 6.362.620 | Bắt buộc xử lý vectorized, không loop per-row |
| `nameOrig` duy nhất | 6.353.307 (99,85%) | **Không có lịch sử khách hàng lặp lại đáng kể** → không thể/không nên xây "customer profile" bền vững qua nhiều giao dịch. Mỗi giao dịch coi như 1 thực thể độc lập. |
| Fraud theo loại giao dịch | `TRANSFER`: 4.097/532.909; `CASH_OUT`: 4.116/2.237.500; `PAYMENT`, `DEBIT`, `CASH_IN`: 0 fraud | Fraud trong PaySim **chỉ tồn tại ở giao dịch chuyển/rút tiền** — đúng bản chất account-takeover (chiếm quyền tài khoản rồi rút tiền), KHÔNG phải gian lận thẻ tại checkout. Đây là câu chuyện nghiệp vụ đúng cần dùng, không gượng ép narrative "e-commerce checkout fraud". |
| Tỷ lệ fraud tổng | 8.213/6.362.620 = 0,1291% | Khớp đúng con số đã có trong `model/eda_summary.json` (theo phancong.txt) → xác nhận đang dùng đúng file dữ liệu đã audit trước đó. |
| `step` | 1–743 | 1 step ≈ 1 giờ, tổng ≈ 31 ngày mô phỏng → dùng trực tiếp để suy ra giờ trong ngày, không cần sinh ngẫu nhiên. |

**Vì sao điều này quan trọng:** nếu bỏ qua bước kiểm chứng này và thiết kế theo trực giác thông thường (xây customer profile bền vững theo `nameOrig`, giả định e-commerce checkout fraud), thiết kế sẽ **sai lệch với chính dữ liệu đang dùng** — 99,85% effort xây profile sẽ vô nghĩa vì không có giao dịch thứ 2 để dùng lại profile đó.

## 3. Nguyên tắc thiết kế

1. **Sinh theo từng dòng giao dịch (row-level), không theo hồ sơ khách hàng.** Lý do: mục 2.
2. **Chỉ tiêm tín hiệu fraud vào field có lý do hành vi thực sự** (new device, IP lệch quốc gia, giờ đêm, tài khoản mới, mismatch địa chỉ, số lần thất bại). Các field không có cơ sở hành vi rõ ràng (`browser`, `device_type`) được sinh **độc lập với `isFraud`**.
   - *Lập luận:* dữ liệu gian lận thật không bao giờ có toàn bộ field tương quan với nhãn — nếu ép mọi field đều "báo hiệu" fraud, đó là dấu hiệu của một dataset giả tạo/leak, không phải một dataset thực tế. Một senior fraud analyst sẽ nghi ngờ ngay một dataset mà feature nào cũng phân tách hoàn hảo 2 lớp.
3. **Mọi hệ số tương quan (odds-ratio, Poisson λ multiplier) giới hạn ở mức 2–4 lần baseline, không vượt quá.**
   - *Lập luận:* Đề bài yêu cầu "justify the realism" — nghĩa là con số phải giải trình được, không phải chọn để đạt AUC cao. Trong thực tế, fraud giỏi thường **giả mạo được hành vi bình thường** (dùng device/IP quen thuộc), nên tín hiệu không bao giờ tuyệt đối. Giới hạn hệ số ở mức vừa phải phản ánh đúng điều này và tránh lặp lại lỗi leakage đã thấy ở các feature balance hiện có (AUC-PR 0.9988 — quá hoàn hảo, đáng ngờ).
4. **Field nào tính được trực tiếp từ dữ liệu gốc (`step`) thì suy ra bằng công thức, không sinh ngẫu nhiên.**
   - *Lập luận:* Không có lý do gì để "đoán" giờ giao dịch khi dữ liệu gốc đã cho biết chính xác qua `step`. Sinh ngẫu nhiên ở đây sẽ làm giảm độ chính xác một cách không cần thiết.
5. **Có bước đo lường và giới hạn leakage khách quan sau khi sinh (không chỉ "cảm thấy hợp lý").**
   - *Lập luận:* "Hợp lý" là chủ quan; đo AUC đơn biến là khách quan, lặp lại được, và là bằng chứng cho phần "rigor" khi chấm điểm.

## 4. Danh sách field & công thức sinh

Quy ước: `base` = phân phối/xác suất khi `isFraud=0`; `fraud` = khi `isFraud=1`. Toàn bộ sinh bằng 1 `numpy.random.default_rng(seed=42)` duy nhất, vectorized trên toàn bộ 6.362.620 dòng (không loop).

| # | Field | Kiểu sinh | Base | Fraud (hệ số) | Lập luận chọn số |
|---|---|---|---|---|---|
| 1 | `hour_of_day` | Derived | `(step - 1) % 24` | — | Suy trực tiếp từ `step`, không cần giả định. |
| 2 | `is_night_transaction` | Derived | `hour_of_day ∈ [0,5]` | — | Định nghĩa "đêm" = 0h–6h, quy ước phổ biến trong các nghiên cứu fraud theo giờ. |
| 3 | `customer_account_age_days` | Conditional (lognormal) | median ≈ 400 ngày | median ≈ 150 ngày (~0.4x)¹ | Tài khoản bị chiếm đoạt/tài khoản mule thường được tạo gần đây hơn tài khoản lâu năm bình thường — hệ số 0.4x là giả định thận trọng, không suy từ số liệu thực (đề bài yêu cầu ghi rõ đây là business assumption). |
| 4 | `device_id` | Độc lập | pool cố định 50.000 UUID (Faker) | giống base | Không tự thân là tín hiệu fraud; tín hiệu nằm ở field `new_device_flag` (#7), tránh trùng lặp thông tin. |
| 5 | `browser` | Độc lập | Chrome 55% / Safari 20% / Edge 12% / Firefox 8% / Khác 5% | giống base | Không có cơ sở hành vi để gắn với fraud — cố ý để trung lập nhằm tránh over-signal giả tạo. |
| 6 | `device_type` | Độc lập | mobile 65% / desktop 30% / tablet 5% | giống base | Tương tự #5. |
| 7 | `new_device_flag` | Conditional (Bernoulli) | p = 0.04 | p = 0.12 (3x) | ~4% giao dịch hợp pháp đến từ thiết bị mới (đổi điện thoại...) là hợp lý; fraud tăng gấp 3 vì account-takeover thường từ thiết bị lạ, nhưng không tuyệt đối (fraud tinh vi có thể giả mạo device quen). |
| 8 | `billing_country` | Độc lập | categorical, ~20 quốc gia, trọng số giả lập theo thị trường | giống base | Mô phỏng cơ cấu khách hàng nền tảng, không có lý do gắn với fraud tại bước này (tín hiệu nằm ở mismatch, #9). |
| 9 | `ip_country` | Conditional | P(=billing_country) = 0.93 | P(=billing_country) = 0.80 (odds mismatch ~2.6x) | Giao dịch hợp pháp đa số dùng IP đúng quốc gia cư trú; fraud có xác suất lệch cao hơn nhưng vẫn phần lớn trùng (VPN/proxy giúp fraud giả mạo IP). |
| 10 | `ip_billing_distance_km` | **Derived** từ #8, #9 | haversine(centroid[ip_country], centroid[billing_country]) | — | Tính trực tiếp từ 2 field trên bằng bảng tọa độ trung tâm quốc gia cố định — đảm bảo **nhất quán nội tại** (không random riêng distance rồi mâu thuẫn với mismatch flag). |
| 11 | `shipping_billing_mismatch` | Conditional (Bernoulli) | p = 0.05 | p = 0.15 (3x) | Một số khách hàng hợp pháp có địa chỉ giao khác địa chỉ đăng ký (quà tặng, công ty); fraud tăng vì kẻ gian có thể đổi hướng nhận tiền/hàng. Ghi rõ trong dictionary: khái niệm được diễn giải lại thành "địa chỉ giao dịch khác địa chỉ đăng ký" do bản chất fraud của PaySim là account-takeover, không phải checkout thẻ. |
| 12 | `failed_payment_attempts_24h` | Conditional (Poisson) | λ = 0.15 | λ = 0.6 (4x) | Đa số giao dịch hợp pháp không có lần thất bại trước đó; kẻ gian thường thử nhiều lần (dò mật khẩu/thẻ) trước khi thành công. |

¹ Khi chạy leakage check (mục 5) trên dữ liệu thật, `customer_account_age_days` ban đầu vượt ngưỡng (AUC 0.8753). Theo đúng quy trình mục 5, hệ số fraud được giảm bớt (median 150 → 275 ngày, tức 0.4x → 0.6875x) và sinh lại, đạt AUC 0.6689 (PASS). Xem commit `ebfe5cc` và `docs/DATA_DICTIONARY.md` để biết giá trị cuối cùng đã dùng.

## 5. Cơ chế chống leakage (bắt buộc)

Sau khi sinh xong toàn bộ field:
1. Với mỗi field mới, tính **AUC đơn biến** (dùng chính field đó làm score dự đoán `isFraud`) hoặc correlation (Pearson/point-biserial cho numeric, Cramér's V cho categorical).
2. Ngưỡng chấp nhận: AUC đơn biến < 0.75 (tương đương "tín hiệu có nhưng không quyết định"). Nếu vượt ngưỡng → giảm hệ số ở mục 4 và sinh lại.
3. Ghi **giá trị đo được thực tế** vào data dictionary (không chỉ ghi hệ số dự định) — đây vừa là kiểm soát chất lượng vừa là bằng chứng "rigor" khi chấm điểm.

## 6. Lưu trữ & hiệu năng

- Đọc file gốc 1 lần bằng pandas với dtype tối ưu (`category` cho `type`, `float32` cho amount/balance) — 6,36M dòng x 11 cột nằm gọn trong RAM thông thường (không cần chunking ở quy mô này).
- Toàn bộ field mới sinh bằng numpy vectorized (`rng.binomial`, `rng.poisson`, `np.where` theo `isFraud`) — không gọi Faker theo từng dòng (chỉ dùng Faker một lần để tạo pool 50.000 device ID).
- Output chính: **Parquet** (`transactions_synthetic.parquet`) — nhỏ và đọc nhanh hơn CSV 5–10 lần, thuận tiện cho Module 2–7 dùng lại nhiều lần.
- Xuất kèm 1 file CSV mẫu nhỏ (stratified ~5.000 dòng theo `isFraud`) để minh họa trong report/data dictionary, tránh phải mở file Parquet lớn khi chỉ cần xem ví dụ.

## 7. Cấu trúc Data Dictionary (bắt buộc theo đề bài)

Mỗi field ghi đủ các cột sau (Markdown hoặc Excel):

`field_name | data_type | unit | valid_range | generation_type (derived / independent-random / conditional-on-risk-proxy) | logic_or_formula | business_assumption | measured_univariate_AUC_vs_isFraud`

*(Giá trị enum `conditional-on-risk-proxy` cập nhật theo mục 0 — bản gốc ghi `conditional-on-fraud`, đã lỗi thời.)*

## 8. Giới hạn & rủi ro đã biết (ghi minh bạch, không che giấu)

- Các hệ số odds-ratio/λ là **giả định tự đặt**, không suy từ số liệu fraud thực tế công khai nào (vì không có nguồn dữ liệu như vậy đi kèm PaySim) — cần nêu rõ trong report là giả định nghiệp vụ (business assumption), đúng như đề bài yêu cầu, không trình bày như sự thật đã kiểm chứng.
- Do PaySim chỉ gắn fraud cho `TRANSFER`/`CASH_OUT`, các field như `shipping_billing_mismatch` mang tính diễn giải lại (account-takeover thay vì checkout fraud thật) — cần nêu rõ giới hạn này trong phần "limitations" của report cuối (Module 8).
- 9.313 `nameOrig` có lặp lại (0,15%) bị xử lý như các dòng độc lập (không có xử lý đặc biệt) — chấp nhận được vì tỷ lệ quá nhỏ để ảnh hưởng kết quả.

## 9. Đối chiếu với yêu cầu đề bài (Compliance Checklist)

Đối chiếu trực tiếp từng câu chữ liên quan trong `yeucau.txt` để đảm bảo thiết kế đáp ứng **tối thiểu** yêu cầu của giảng viên trước khi cho phép code:

| Yêu cầu trong `yeucau.txt` | Đáp ứng trong thiết kế này | Trạng thái |
|---|---|---|
| "generate additional contextual data using Python (e.g., the Faker library plus custom business logic)" (dòng 36-37) | Faker dùng để tạo pool 50.000 device ID; các field còn lại dùng numpy + business logic tự viết (mục 4, 6) | ✅ |
| "customer account age" (dòng 37) | `customer_account_age_days` (mục 4, #3) | ✅ |
| "device/browser fingerprint" (dòng 38) | `device_id`, `browser`, `device_type`, `new_device_flag` (mục 4, #4-7) | ✅ |
| "shipping vs. billing address mismatch" (dòng 41) | `shipping_billing_mismatch` (mục 4, #11) — có ghi chú diễn giải lại do đặc thù PaySim (mục 8) | ✅ |
| "number of failed payment attempts" (dòng 42) | `failed_payment_attempts_24h` (mục 4, #12) | ✅ |
| "IP-to-billing-country distance" (dòng 42) | `ip_country`, `billing_country`, `ip_billing_distance_km` — khoảng cách số (km), không phải flag nhị phân (mục 4, #8-10) | ✅ |
| "time-of-day pattern" (dòng 42) | `hour_of_day`, `is_night_transaction` (mục 4, #1-2) | ✅ |
| "teams must justify the realism of the data they generate and document it rigorously" (dòng 44) | Cột "Lập luận chọn số" trong bảng field (mục 4) + mục 8 nêu rõ giới hạn/giả định, không che giấu | ✅ |
| "must explicitly address the class-imbalance challenge inherent to fraud data" (dòng 45) | Giữ nguyên tỷ lệ fraud gốc 0,1291% khi sinh field (không oversample/undersample ở bước này) — kỹ thuật xử lý imbalance (SMOTE/class weight) thuộc phạm vi Module 4, nêu rõ để không nhầm phạm vi trách nhiệm | ✅ (trong phạm vi Module 1) |
| "Data Dictionary Requirement: ... column name, data type, unit, valid range, and the generation logic or business assumption used" (dòng 46-48) | Mục 7 có đủ 5 cột tối thiểu + thêm cột đo AUC (vượt tối thiểu) | ✅ |
| Module 1 — "Produce a complete data dictionary" | Mục 7 + deliverable cụ thể ở mục 10 | ✅ |
| Deliverable: "Data dictionary — Excel or Markdown file" (dòng 116) | Chốt cụ thể: `docs/DATA_DICTIONARY.md` (mục 10) | ✅ |
| "Originality of code — all code must be authored by team members" (dòng 155) | Toàn bộ logic tự viết (numpy/pandas/Faker), không dùng AutoML/no-code | ✅ |

**Kết luận:** thiết kế hiện tại đáp ứng đủ tất cả các điểm bắt buộc liên quan đến synthetic data ở Module 1. Không có điểm nào bị thiếu ở mức tối thiểu; phần "etc." (mở rộng thêm field ngoài 5 field nêu tên) là tùy chọn, không bắt buộc để đạt điểm tối thiểu.

## 10. Deliverables cụ thể (đường dẫn file)

- Code sinh dữ liệu: `src/data_generation/generate_synthetic_fields.py`
- Bảng tra tọa độ quốc gia (dùng cho `ip_billing_distance_km`): `src/data_generation/country_centroids.py`
- Script đo leakage (mục 5): `src/data_generation/check_leakage.py`
- Output dữ liệu: `data/processed/transactions_synthetic.parquet` + `data/processed/transactions_synthetic_sample.csv` (~5.000 dòng stratified)
- Data dictionary: `docs/DATA_DICTIONARY.md` (theo cấu trúc mục 7, sinh tự động từ script hoặc điền tay sau khi chạy `check_leakage.py`)

## 11. Bàn giao cho vai trò khác

- **Người 4 (Feature Engineer):** các field string (`device_id`, `billing_country`, `ip_country`, `browser`, `device_type`) cần encode; `ip_billing_distance_km` và `failed_payment_attempts_24h` đã là numeric, dùng trực tiếp được.
- **Người 5 (ML Engineer):** cảnh báo các feature balance hiện có (`sender_balance_delta`...) đang cho AUC-PR 0.9988 — nghi ngờ leakage sẵn có trong PaySim (fraud thường rút sạch số dư). Nên train model có/không các feature đó để so sánh, tránh đánh giá sai giá trị của field synthetic mới.
