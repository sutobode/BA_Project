# Data Pipeline: Synthetic Generation & Cleaning — Chiến lược, Logic & Cách chạy

Pipeline 2 giai đoạn cho dataset PaySim, phục vụ bài toán phát hiện gian lận thanh toán (fraud detection) theo thời gian thực: **(A) sinh thêm 13 trường bối cảnh e-commerce/hành vi (label-free)**, rồi **(B) kiểm tra và làm sạch dữ liệu** (missing values, duplicates, invalid categories, outliers) trước khi bàn giao cho bước feature engineering/modeling.

Tài liệu này là **nguồn tham khảo đầy đủ, tự chứa** — đọc xong hiểu toàn bộ logic/chiến lược/quy tắc của cả 2 giai đoạn, không cần mở file khác. Tài liệu gốc chi tiết hơn nếu cần tra cứu sâu:
- Spec sinh dữ liệu: [`docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md`](docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md) · Plan: [`...-nguoi2-plan.md`](docs/superpowers/plans/2026-07-03-synthetic-data-nguoi2-plan.md)
- Spec cleaning: [`docs/superpowers/specs/2026-07-03-data-cleaning-design.md`](docs/superpowers/specs/2026-07-03-data-cleaning-design.md) · Plan: [`...-data-cleaning-plan.md`](docs/superpowers/plans/2026-07-03-data-cleaning-plan.md)
- Data dictionary (tự sinh): [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) · Cleaning report (tự sinh): [`docs/CLEANING_REPORT.md`](docs/CLEANING_REPORT.md)
- Tổng kết quá trình làm việc (quyết định, vấn đề đã phát hiện & xử lý): [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md)

## 1. Tổng quan toàn bộ pipeline

```mermaid
flowchart LR
    A["Data/PS_..._log.csv<br/>PaySim gốc<br/>6.362.620 dòng, 11 cột"] --> B["PHẦN A<br/>Synthetic Data Generation<br/>+13 field bối cảnh (label-free)"]
    B --> C["transactions_synthetic<br/>.parquet / .csv<br/>24 cột"]
    C --> D["PHẦN B<br/>Data Cleaning<br/>+3 field flag"]
    D --> E["transactions_cleaned<br/>.parquet / .csv<br/>27 cột — dataset cuối cùng"]
```

**Nguồn dữ liệu:** `Data/PS_20174392719_1491204439457_log.csv` — PaySim / Online Payments Fraud Dataset, **6.362.620 dòng, dùng toàn bộ** (không lấy mẫu). Cột gốc: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.

### ✅ Trạng thái xác thực (verify lại được, không phải khẳng định suông)

| Kiểm tra | Kết quả | Lệnh để tự verify lại |
|---|---|---|
| Test suite | **82/82 PASS** | `.venv/Scripts/python.exe -m pytest tests/ -v` |
| Leakage check trên 6.362.620 dòng thật | **13/13 PASS** | `PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.check_leakage` |
| Không hàm sinh nào đọc `isFraud` | **Xác nhận bằng grep, 0 kết quả sai** | `grep -n "isFraud\|is_fraud" src/data_generation/generate_synthetic_fields.py` |
| `transactions_synthetic.parquet` | 6.362.620 dòng × **24 cột** | đọc bằng `pandas.read_parquet` |
| `transactions_cleaned.parquet` | 6.362.620 dòng × **27 cột** — **data final** | đọc bằng `pandas.read_parquet` |
| Fraud rate giữ nguyên qua cả 2 giai đoạn | 8.213 / 6.362.620 = **0,1291%** | so với file gốc |
| Code khớp với commit nào | `82d54af` (2026-07-23) | `git log -1` |

Mọi số liệu trong tài liệu này (AUC, số dòng, số cột) đều lấy trực tiếp từ lần chạy thật tương ứng với commit trên — không phải số ước lượng hay copy từ thiết kế ban đầu.

---

# PHẦN A — SYNTHETIC DATA GENERATION

## 2. Bài toán

Kaggle PaySim chỉ có dữ liệu giao dịch tài chính thô (số tiền, số dư, loại giao dịch...), không có các trường "bối cảnh e-commerce" cần thiết để mô hình fraud detection có đủ tín hiệu hành vi (device fingerprint, khoảng cách IP-billing, tuổi tài khoản, mismatch địa chỉ, số lần thanh toán thất bại, pattern theo giờ). Phần A sinh thêm 13 trường đó bằng Python (Faker + business logic tự viết), đồng thời **tự kiểm tra khách quan** để đảm bảo dữ liệu sinh ra thực tế nhưng không "lộ" nhãn fraud một cách giả tạo (data leakage).

> **Nguyên lý sinh dữ liệu — LABEL-FREE (quan trọng, đọc trước):** không hàm sinh field nào đọc `isFraud`. Các field "có lý do hành vi" được điều kiện theo một **risk proxy label-free** (`compute_risk_proxy`) chỉ tính từ biến quan sát được (`type`, `amount`, `hour_of_day`) — cố ý **không** dùng `oldbalanceOrg`/`newbalanceOrig` vì 2 cột số dư này gần như xác định `isFraud` trong PaySim (fraud = rút cạn tài khoản), dùng chúng sẽ leak qua cửa sau. `isFraud` chỉ được dùng **sau khi sinh xong** để đo leakage / train / evaluate. **Lưu ý trung thực:** vì risk proxy là hàm của `type/amount/hour`, các field điều kiện theo nó về bản chất là hàm nhiễu của các cột đó — tương quan với fraud là **do thiết kế, không phải bằng chứng dự đoán trên fraud thật** (xem mục 19).

## 3. Sự thật dữ liệu → Quyết định thiết kế

Trước khi viết bất kỳ dòng code nào, 3 sự thật sau được **đo trực tiếp trên file gốc** và quyết định toàn bộ hướng thiết kế:

```mermaid
flowchart LR
    F1["<b>Sự thật 1</b><br/>99,85% giá trị nameOrig<br/>chỉ xuất hiện đúng 1 lần"] --> D1["<b>Quyết định</b><br/>Sinh dữ liệu theo từng dòng<br/>giao dịch (row-level) —<br/>KHÔNG xây customer profile<br/>bền vững qua nhiều giao dịch"]
    F2["<b>Sự thật 2</b><br/>Fraud chỉ tồn tại ở<br/>TRANSFER / CASH_OUT"] --> D2["<b>Quyết định</b><br/>Narrative = account-takeover<br/>(chiếm quyền tài khoản),<br/>KHÔNG phải gian lận thẻ<br/>tại checkout"]
    F3["<b>Sự thật 3</b><br/>step = số giờ kể từ<br/>lúc mô phỏng bắt đầu (1–743)"] --> D3["<b>Quyết định</b><br/>hour_of_day / is_night_transaction<br/>suy trực tiếp bằng công thức,<br/>KHÔNG sinh ngẫu nhiên"]
```

Chi tiết sự thật 1 (đo trên toàn bộ 6.362.620 dòng):

```mermaid
pie title Phân bố nameOrig — cơ sở cho quyết định "row-level"
    "Chỉ xuất hiện 1 lần : 6.353.307 (99,85%)" : 6353307
    "Lặp lại : 9.313 (0,15%)" : 9313
```

Chi tiết sự thật 2 — số fraud theo loại giao dịch (khớp đúng tỷ lệ 0,1291% đã có trong audit trước đó):

| Loại giao dịch | Số dòng | Số fraud |
|---|---|---|
| `TRANSFER` | 532.909 | 4.097 |
| `CASH_OUT` | 2.237.500 | 4.116 |
| `PAYMENT` | 2.151.495 | 0 |
| `CASH_IN` | 1.399.284 | 0 |
| `DEBIT` | 41.432 | 0 |

**Vì sao quan trọng:** nếu bỏ qua bước đo này và thiết kế theo trực giác thông thường (customer profile bền vững, narrative "checkout fraud"), thiết kế sẽ sai lệch với chính dữ liệu đang dùng.

## 4. Nguyên tắc thiết kế cốt lõi

| # | Nguyên tắc | Lý do |
|---|---|---|
| 1 | Sinh **row-level**, không customer profile | Sự thật 1 — không có lịch sử khách hàng đáng kể để dùng lại |
| 2 | **LABEL-FREE:** không hàm sinh nào đọc `isFraud`. Field "có lý do hành vi" (new device, IP lệch, tài khoản mới, mismatch địa chỉ, thất bại thanh toán) điều kiện theo **risk proxy** tính từ observable (`type`/`amount`/`hour`); field không có cơ sở hành vi (`browser`, `device_type`, `device_id`, `billing_country`) sinh **độc lập** | `isFraud` là biến cần dự đoán — dùng nó lúc sinh feature là dấu hiệu leakage kinh điển mà giám khảo soi ngay; risk proxy label-free tránh điều đó và về nguyên tắc tính lại được lúc scoring |
| 3 | Risk proxy **cố ý tránh** `oldbalanceOrg`/`newbalanceOrig` | 2 cột số dư này gần như xác định `isFraud` trong PaySim; nếu proxy dùng chúng, field sinh ra sẽ leak qua cửa sau (mạnh hơn cả cách cũ) |
| 4 | Mọi hệ số (odds-ratio, Poisson λ, median gap) **giới hạn 2–4 lần baseline** | Mỗi con số phải giải trình được (justify the realism), không phải chọn để đạt AUC cao. Fraud giỏi vẫn giả mạo được hành vi bình thường, nên tín hiệu không bao giờ tuyệt đối |
| 5 | Field tính được trực tiếp từ dữ liệu gốc (`step`) thì **suy bằng công thức**, không random | Không có lý do "đoán" khi dữ liệu gốc đã cho biết chính xác |
| 6 | Đo leakage **khách quan bằng số** sau khi sinh, không chỉ "cảm thấy hợp lý" | AUC/Cramér's V là con số lặp lại được, là bằng chứng khách quan cho tính rigor của quy trình. **Vẫn bắt buộc chạy dù đã label-free** — label-free KHÔNG tự động hết leak trên PaySim |

## 5. Kiến trúc pipeline sinh dữ liệu

```mermaid
flowchart TD
    A["Data/PS_..._log.csv<br/>6.362.620 dòng gốc"] --> B["load_raw_transactions()<br/>schema guard + dtype tối ưu<br/>(category / float32 / int8)"]
    B --> C["generate_all_synthetic_fields()<br/>seed=42 · 1 numpy.random.Generator duy nhất<br/>13 field label-free, 100% vectorized"]
    C --> D["transactions_synthetic.parquet<br/>6.362.620 dòng + 13 cột mới"]
    D --> E["transactions_synthetic_sample.csv<br/>~5.000 dòng, stratified theo isFraud"]
    D --> F["check_all_fields()<br/>AUC (numeric/bool) hoặc<br/>Cramér's V hiệu chỉnh bias (categorical)"]
    F -->|"Field FAIL<br/>(AUC ≥ 0.75 hoặc V ≥ 0.5)"| G["Giảm hệ số high-risk tương ứng<br/>trong generate_synthetic_fields.py"]
    G --> C
    F -->|"13/13 field PASS"| H["docs/DATA_DICTIONARY.md<br/>formula + business assumption + số đo thật"]
```

**Module phụ trách từng phần:**

```mermaid
flowchart LR
    CC["country_centroids.py<br/>toạ độ 20 quốc gia + haversine distance"] --> GSF["generate_synthetic_fields.py<br/>compute_risk_proxy (label-free) + hàm sinh field + orchestrator + CLI"]
    GSF --> CL["check_leakage.py<br/>AUC / Cramér's V + ghi DATA_DICTIONARY.md"]
```

**Nguyên tắc kỹ thuật:** mọi hàm sinh dữ liệu dùng **numpy/pandas vectorized** (không loop qua từng dòng trong 6,3 triệu dòng), dùng chung **một** `numpy.random.Generator(seed=42)` cho cả lượt chạy → kết quả **tái lập được 100%** khi chạy lại với cùng input.

## 6. Quy tắc phân loại field (logic quyết định mỗi field sinh thế nào)

```mermaid
flowchart TD
    Start["Field mới cần thêm"] --> Q1{"Tính được trực tiếp<br/>từ dữ liệu gốc (step / field khác)?"}
    Q1 -->|Có| Derived["<b>derived</b><br/>Công thức xác định, không random"]
    Q1 -->|Không| Q2{"Có lý do hành vi thực sự<br/>liên quan đến fraud<br/>(account-takeover)?"}
    Q2 -->|Không| Indep["<b>independent-random</b><br/>Sinh độc lập, không chạm risk proxy"]
    Q2 -->|Có| Q3{"Kiểu dữ liệu phù hợp?"}
    Q3 -->|"Nhị phân 0/1"| Bern["<b>conditional-on-risk-proxy</b><br/>(Bernoulli)"]
    Q3 -->|"Số lần / đếm"| Pois["<b>conditional-on-risk-proxy</b><br/>(Poisson)"]
    Q3 -->|"Liên tục, lệch phải"| Logn["<b>conditional-on-risk-proxy</b><br/>(lognormal)"]

    Derived -.-> DerivedEx["hour_of_day, is_night_transaction,<br/>ip_billing_distance_km,<br/>ip_billing_country_mismatch"]
    Indep -.-> IndepEx["device_id, browser, device_type,<br/>billing_country"]
    Bern -.-> BernEx["new_device_flag, ip_country,<br/>shipping_billing_mismatch"]
    Pois -.-> PoisEx["failed_payment_attempts_24h"]
    Logn -.-> LognEx["customer_account_age_days"]

    RP["<b>risk proxy (label-free)</b><br/>= f(type, amount, hour)<br/>KHÔNG đọc isFraud,<br/>KHÔNG dùng cột số dư"] -.->|"đầu vào cho mọi<br/>conditional-on-risk-proxy"| Bern
    RP -.-> Pois
    RP -.-> Logn
```

## 7. Chi tiết 13 field synthetic

`risk_score = compute_risk_proxy(type, amount, hour_of_day)` — label-free, giá trị trong [0, 1], **không đọc `isFraud`, không dùng `oldbalanceOrg`/`newbalanceOrig`**. 5 field "conditional" dưới đây nội suy tuyến tính giữa base (risk_score=0) và high-risk (risk_score=1), giữ đúng biên độ 2–4x như thiết kế gốc — chỉ đổi biến điều kiện từ `isFraud` sang `risk_score`.

| # | Field | Loại sinh | Base → High-risk | Lập luận chọn số |
|---|---|---|---|---|
| 1 | `hour_of_day` | derived | `(step - 1) % 24` | Suy trực tiếp từ `step`, không cần giả định |
| 2 | `is_night_transaction` | derived | `hour_of_day ∈ [0,5]` | Định nghĩa "đêm" = 0h–6h, quy ước phổ biến trong nghiên cứu fraud theo giờ |
| 3 | `customer_account_age_days` | conditional-on-risk-proxy (lognormal) | median 400 → 275 ngày | Tài khoản bị chiếm đoạt/mule thường tạo gần đây hơn — hệ số thận trọng, là business assumption, không suy từ số liệu thực |
| 4 | `device_id` | độc lập | pool cố định 50.000 UUID (Faker) | Không tự thân là tín hiệu fraud; tín hiệu nằm ở `new_device_flag` (#7), tránh trùng lặp thông tin |
| 5 | `browser` | độc lập | Chrome 55% / Safari 20% / Edge 12% / Firefox 8% / Other 5% | Không có cơ sở hành vi để gắn với fraud — cố ý trung lập, tránh over-signal giả tạo |
| 6 | `device_type` | độc lập | mobile 65% / desktop 30% / tablet 5% | Tương tự #5 |
| 7 | `new_device_flag` | conditional-on-risk-proxy (Bernoulli) | p = 0.04 → 0.12 (3x) | ~4% giao dịch hợp pháp từ thiết bị mới là hợp lý; risk cao tăng gấp 3 vì account-takeover thường từ thiết bị lạ, nhưng không tuyệt đối |
| 8 | `billing_country` | độc lập | categorical, 20 quốc gia | Mô phỏng cơ cấu khách hàng nền tảng; tín hiệu nằm ở mismatch (#9, #10a), không phải ở đây |
| 9 | `ip_country` | conditional-on-risk-proxy | match rate 0.93 → 0.80 | Giao dịch hợp pháp đa số dùng IP đúng quốc gia; risk cao lệch nhiều hơn nhưng vẫn phần lớn trùng (VPN/proxy giúp fraud giả mạo IP) |
| 10 | `ip_billing_distance_km` | **derived** từ #8, #9 | `haversine(centroid[ip_country], centroid[billing_country])` | Tính trực tiếp bằng bảng tọa độ cố định — đảm bảo nhất quán nội tại, không mâu thuẫn với mismatch flag |
| 10a | `ip_billing_country_mismatch` | **derived** từ #8, #9 | `ip_country != billing_country` | Bản boolean tiện dụng của cùng mismatch #10 đã capture bằng số — thêm theo tên field trong phân công nhóm |
| 11 | `shipping_billing_mismatch` | conditional-on-risk-proxy (Bernoulli) | p = 0.05 → 0.15 (3x) | Một số khách hợp pháp có địa chỉ giao khác đăng ký (quà tặng, công ty); risk cao tăng vì có thể đổi hướng nhận tiền/hàng. Diễn giải lại thành "địa chỉ giao dịch khác đăng ký" do fraud PaySim là account-takeover, không phải checkout thẻ |
| 12 | `failed_payment_attempts_24h` | conditional-on-risk-proxy (Poisson) | λ = 0.15 → 0.6 (4x) | Đa số giao dịch hợp pháp không có lần thất bại trước; kẻ gian thường thử nhiều lần trước khi thành công |

**`compute_risk_proxy` gồm 3 thành phần quan sát được (trọng số bằng nhau):**
- `risky_type`: `type ∈ {TRANSFER, CASH_OUT}` — 2 channel duy nhất có fraud trong PaySim, nhưng tỷ lệ fraud trong đó vẫn < 1%, nên đây là heuristic yếu, không phải proxy gần-quyết-định.
- `amount_percentile`: rank của `amount` trong cùng `type` — "giao dịch bất thường lớn so với channel" là heuristic fraud phổ biến ngoài thực tế, độc lập với cơ chế rút-cạn-số-dư riêng của PaySim.
- `is_night`: giờ đêm (0h–6h) — heuristic thời gian phổ biến.

## 8. Cơ chế chống leakage — lịch sử phát hiện lỗi và lần chuyển sang label-free

**Quy trình:** sau khi sinh xong, tính **AUC đơn biến** (field numeric/boolean) hoặc **Cramér's V** (field categorical) so với `isFraud`. Ngưỡng FAIL: `AUC ≥ 0.75` hoặc `Cramér's V ≥ 0.5`. Nếu FAIL → giảm hệ số ở mục 7, sinh lại — **không đổi ngưỡng để "cho qua"**. Bước này **vẫn bắt buộc dù pipeline đã label-free** — không đọc `isFraud` không đồng nghĩa không leak (xem cảnh báo mục 2).

Quy trình này đã bắt được các vấn đề thật trong quá trình build:

```mermaid
sequenceDiagram
    participant Gen as generate_synthetic_fields
    participant Check as check_leakage
    participant Dev as Người phát triển

    Gen->>Check: Chạy trên 6.362.620 dòng thật (thiết kế ban đầu)
    Check->>Dev: FAIL — customer_account_age_days AUC=0.8753 (ngưỡng 0.75)
    Dev->>Gen: Giảm hệ số fraud (median 150 -> 275 ngày)
    Gen->>Check: Sinh lại, chạy check lần 2
    Check->>Dev: PASS — 12/12 field, nhưng cơ chế vẫn đọc isFraud trực tiếp

    Dev->>Check: Review kỹ thuật độc lập
    Check->>Dev: Phát hiện code đọc isFraud để sinh 5 field -><br/>không tái tạo được lúc scoring (isFraud chưa biết cho giao dịch mới),<br/>và là "leakage smell" kinh điển với người review fraud detection
    Dev->>Check: Viết lại 5 field: điều kiện theo compute_risk_proxy<br/>(label-free, chỉ dùng type/amount/hour, cố ý tránh cột số dư)
    Check->>Dev: Sinh lại trên 6.362.620 dòng -> 13/13 PASS,<br/>AUC 5 field đổi giảm xuống 0.51-0.55 (từ 0.55-0.67 cũ)<br/>- tín hiệu yếu hơn, đúng như kỳ vọng khi bỏ vòng lặp qua nhãn

    Dev->>Check: Review độc lập riêng cho công thức Cramér's V (không tin báo cáo cũ)
    Check->>Dev: Phát hiện công thức gốc bị lệch dương với field cardinality lớn<br/>(device_id: có thể FAIL giả ~0.48 trên mẫu nhỏ dù độc lập hoàn toàn với fraud)
    Dev->>Check: Thay bằng Cramér's V hiệu chỉnh bias (Bergsma, 2013)
    Check->>Dev: device_id: 0.0, vẫn PASS, ổn định hơn theo sample size
```

**Bài học rút ra:** nếu `check_leakage` báo FAIL, đó là quy trình đang hoạt động đúng — không phải bug (xem mục 18 để biết cách xử lý). Cramér's V dùng công thức **hiệu chỉnh bias** vì field cardinality lớn (`device_id`, 50.000 giá trị) bị lệch dương với công thức gốc, đặc biệt nhạy với kích thước dataset.

## 9. Kết quả đo được trên dữ liệu thật (Phần A)

Row count và tỷ lệ fraud giữ nguyên (0,1291%) so với file gốc — bước sinh dữ liệu **không làm thay đổi class imbalance**. Xử lý imbalance kỹ thuật (SMOTE, class weight...) không thuộc phạm vi phần này, để lại cho bước feature engineering/modeling phía sau.

| Field | Metric | Giá trị đo được | Kết quả |
|---|---|---|---|
| `hour_of_day` | AUC | 0.6336 | PASS |
| `is_night_transaction` | AUC | 0.6217 | PASS |
| `customer_account_age_days` | AUC | 0.5508 | PASS |
| `device_id` | Cramér's V (hiệu chỉnh bias) | 0.0 | PASS |
| `browser` | Cramér's V (hiệu chỉnh bias) | 0.0 | PASS |
| `device_type` | Cramér's V (hiệu chỉnh bias) | 0.0004 | PASS |
| `new_device_flag` | AUC | 0.5137 | PASS |
| `billing_country` | Cramér's V (hiệu chỉnh bias) | 0.0 | PASS |
| `ip_country` | Cramér's V (hiệu chỉnh bias) | 0.0018 | PASS |
| `ip_billing_distance_km` | AUC | 0.5208 | PASS |
| `ip_billing_country_mismatch` | Cramér's V (hiệu chỉnh bias) | 0.0047 | PASS |
| `shipping_billing_mismatch` | AUC | 0.5154 | PASS |
| `failed_payment_attempts_24h` | AUC | 0.5533 | PASS |

**Nhận xét quan trọng:** so với thiết kế label-conditional cũ, AUC của 5 field đã đổi cơ chế **giảm** (ví dụ `customer_account_age_days`: 0.6689 → 0.5508). Đây là kết quả **đúng như kỳ vọng, không phải suy giảm chất lượng** — risk proxy label-free có tương quan yếu hơn với `isFraud` vì không còn đọc trực tiếp nhãn, xác nhận cơ chế mới không leak qua cửa sau qua các cột observable.

Số liệu đầy đủ kèm data type, unit, formula, business assumption: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) (sinh tự động từ code).

---

# PHẦN B — DATA CLEANING

## 10. Mục đích

Kiểm tra `transactions_synthetic.parquet` (output của Phần A) theo 4 nhóm bắt buộc — **missing values, duplicates, invalid categories, outliers** — và tạo báo cáo before/after, theo đúng 1 nguyên tắc xuyên suốt: **không được làm mất tín hiệu fraud thật** trong lúc "làm sạch" dữ liệu.

## 11. Khảo sát dữ liệu thật trước khi thiết kế

Trước khi viết code, dataset `transactions_synthetic.parquet` (6.362.620 dòng, 24 cột) được khảo sát trực tiếp:

| Khảo sát | Kết quả | Ý nghĩa |
|---|---|---|
| Missing values (toàn bộ 24 cột) | 0 | Dataset sạch về mặt này — nhưng vẫn cần code check (defensive), phòng khi chạy trên dữ liệu khác |
| Duplicate toàn dòng | 0 | Tương tự |
| Duplicate theo key giao dịch | 0 | Tương tự |
| `type` ngoài 5 giá trị PaySim hợp lệ | 0 | Tương tự |
| `isFraud`/`isFlaggedFraud` ngoài {0,1}, format `nameOrig`/`nameDest`, khoảng trống `step`, cột boolean, bất biến chéo giữa các field | 0 tất cả | Dataset nhất quán nội tại hoàn toàn |
| Số dư âm | 0 | — |
| **`amount = 0`** | **16 dòng** | **Toàn bộ 16 dòng đều là `isFraud=1`, `type=CASH_OUT`** — đây có thể là dấu vết kẻ gian "thử" hệ thống trước khi rút tiền thật |
| **`amount` outlier (Tukey IQR)** | **338.078 dòng (5,31%)** | Giao dịch giá trị bất thường lớn — trong bài toán fraud, đây chính xác là loại tín hiệu cần giữ, không phải nhiễu |
| **`oldbalanceOrg − amount ≠ newbalanceOrig`** | **5.118.892 dòng (80,45%)** | Đặc điểm đã biết của PaySim (giao dịch đến merchant thường không track số dư đích) — KHÔNG phải lỗi nhập liệu |

**Kết luận:** dataset đã rất sạch về cấu trúc. Việc "cleaning" ở đây thực chất là (a) dựng đầy đủ các bước kiểm tra mang tính phòng vệ (defensive — không có tác dụng trên lần chạy này nhưng đúng nếu chạy trên dữ liệu khác), và (b) **đánh dấu (flag) 3 loại bất thường thật** đã tìm thấy mà không xoá dòng nào.

## 12. Nguyên tắc thiết kế: Flag, không xoá

```mermaid
flowchart TD
    A["Phát hiện bất thường"] --> Q{"Có thể là tín hiệu<br/>fraud thật, hay chỉ<br/>là lỗi cấu trúc?"}
    Q -->|"Lỗi cấu trúc thật<br/>(missing ở cột trọng yếu,<br/>duplicate toàn dòng,<br/>category không hợp lệ)"| Remove["XOÁ dòng<br/>— không mang thông tin fraud,<br/>chỉ là nhiễu"]
    Q -->|"Có thể liên quan fraud<br/>(amount=0, amount lớn,<br/>balance lệch)"| Flag["FLAG bằng cột boolean<br/>— GIỮ NGUYÊN dòng"]
```

**Vì sao chọn Flag thay vì xoá:** 16 dòng `amount=0` đều là fraud thật — xoá sẽ mất đúng 16 mẫu fraud hiếm; outlier `amount` lớn có thể là chính tín hiệu fraud mà model cần học; 80% "balance inconsistent" là đặc điểm nguồn dữ liệu, xoá sẽ mất 80% dataset một cách vô lý.

## 13. Kiến trúc pipeline cleaning

```mermaid
flowchart TD
    A["transactions_synthetic.parquet<br/>6.362.620 dòng, 24 cột"] --> B["check_missing_critical<br/>xoá nếu thiếu step/type/amount/isFraud"]
    B --> C["dedupe_exact<br/>xoá duplicate toàn dòng"]
    C --> D["check_invalid_categories<br/>xoá nếu type/browser/device_type/<br/>country/boolean sai giá trị hợp lệ"]
    D --> E["flag_amount_outliers<br/>+ is_amount_outlier"]
    E --> F["flag_zero_amount<br/>+ is_zero_amount"]
    F --> G["flag_balance_inconsistency<br/>+ is_balance_inconsistent"]
    G --> H["transactions_cleaned.parquet<br/>27 cột (24 + 3 flag)"]
    H --> I["transactions_cleaned_sample.csv<br/>~5.000 dòng stratified"]
    H --> J["cleaning_report.py<br/>build_cleaning_report_markdown"]
    J --> K["docs/CLEANING_REPORT.md<br/>before/after cho từng check"]
```

**Nguyên tắc:** 3 bước đầu (B, C, D) **xoá dòng** — chỉ xử lý lỗi cấu trúc thật, chạy TRƯỚC. 3 bước sau (E, F, G) chỉ **thêm cột flag**, không xoá gì — chạy SAU, trên dữ liệu đã loại lỗi cấu trúc. `check_invalid_categories` dùng lại chính danh sách giá trị hợp lệ (`BROWSER_WEIGHTS`, `DEVICE_TYPE_WEIGHTS`, `COUNTRY_WEIGHTS`) từ module sinh dữ liệu ở Phần A — không hardcode một bản sao có thể lệch nhau.

## 14. 6 check & Ý nghĩa của 3 cột flag mới

| # | Check | Hành động | Cột kết quả |
|---|---|---|---|
| 1 | Missing values ở cột trọng yếu | Xoá dòng | — |
| 2 | Duplicate toàn dòng | Xoá dòng | — |
| 3 | Invalid categories (`type`, 4 cột categorical synthetic, 4 cột boolean) | Xoá dòng | — |
| 4 | Outlier `amount` (Tukey IQR: Q1−1.5×IQR, Q3+1.5×IQR) | Flag, giữ nguyên | `is_amount_outlier` |
| 5 | `amount = 0` | Flag, giữ nguyên | `is_zero_amount` |
| 6 | `oldbalanceOrg − amount ≠ newbalanceOrig` (sai lệch > 0.01) | Flag, giữ nguyên | `is_balance_inconsistent` |

**Ý nghĩa cụ thể của từng cột flag — vì sao cần giữ lại và dùng thế nào ở bước sau:**

- **`is_amount_outlier`** (338.078 dòng, 5,31%): đánh dấu giao dịch có giá trị vượt ngưỡng thống kê thông thường (Tukey fence). Đây **không phải lỗi** — trong fraud detection, giao dịch giá trị bất thường lớn thường chính là dấu hiệu đáng ngờ. Ý nghĩa sử dụng: (a) có thể dùng trực tiếp làm **feature nhị phân** cho model (giả thuyết: `is_amount_outlier=True` tương quan với fraud), (b) dùng để lọc subset khi cần phân tích "giao dịch điển hình" riêng biệt với "giao dịch giá trị lớn", (c) tránh việc vô tình bỏ outlier như nhiễu — điều rất dễ mắc lỗi nếu áp dụng cleaning tự động không phân biệt ngữ cảnh fraud.

- **`is_zero_amount`** (16 dòng): đánh dấu giao dịch `amount=0`. Đây là trường hợp cực hiếm nhưng **có tín hiệu cực mạnh** — 100% các dòng quan sát được đều là fraud thật (giả thuyết: kẻ gian "test" hệ thống/tài khoản trước khi rút tiền thật). Ý nghĩa sử dụng: dù chỉ 16/6.362.620 dòng, cột này có thể là **1 trong những feature dự đoán tốt nhất** nếu pattern này lặp lại ở dữ liệu tương lai — tuyệt đối không nên loại bỏ hoặc coi là "dữ liệu rác" khi làm feature engineering.

- **`is_balance_inconsistent`** (5.118.892 dòng, 80,45%): đánh dấu giao dịch có `oldbalanceOrg - amount ≠ newbalanceOrig`. Tỷ lệ cao bất thường (80%) khiến dễ hiểu lầm là lỗi dữ liệu nghiêm trọng — **thực chất đây là đặc điểm cố hữu của PaySim** (nhiều giao dịch đến merchant/tài khoản đích không track số dư chính xác). Ý nghĩa sử dụng: (a) **không nên báo cáo con số 80% này như một vấn đề chất lượng dữ liệu** khi trình bày — sẽ gây hiểu lầm nghiêm trọng; (b) cột flag vẫn có giá trị làm feature phụ vì bản thân việc "có track số dư đầy đủ hay không" có thể tương quan với loại giao dịch/kênh xử lý; (c) dùng để lọc subset "balance đầy đủ" nếu một phân tích cụ thể cần dữ liệu balance đáng tin cậy.

**Tóm lại:** cả 3 cột flag đều là **boolean, dùng được ngay làm feature** cho bước Model Development, hoặc dùng để lọc/subset dữ liệu khi cần. Không cột nào nên bị xoá hay bỏ qua.

## 15. Kết quả đo được trên dữ liệu thật (Phần B)

Chạy trên toàn bộ 6.362.620 dòng — row count **không đổi** (0 dòng bị xoá vì không có lỗi cấu trúc thật):

| Check | rows_before | rows_flagged/removed | Hành động |
|---|---|---|---|
| Missing values | 6.362.620 | 0 | removed |
| Duplicates | 6.362.620 | 0 | removed |
| Invalid categories | 6.362.620 | 0 | removed |
| `is_amount_outlier` | 6.362.620 | 338.078 (5,31%) | flagged (kept) |
| `is_zero_amount` | 6.362.620 | 16 | flagged (kept) |
| `is_balance_inconsistent` | 6.362.620 | 5.118.892 (80,45%) | flagged (kept) |

Báo cáo đầy đủ (tự sinh từ code): [`docs/CLEANING_REPORT.md`](docs/CLEANING_REPORT.md).

---

# TỔNG HỢP

## 16. Mô tả đầy đủ 27 trường dữ liệu (Full Field Reference — `transactions_cleaned`)

Bảng tham chiếu đầy đủ cho **dataset cuối cùng** (`transactions_cleaned.parquet`/`.csv`, 6.362.620 dòng × 27 cột) — dùng để viết tài liệu/data dictionary chính thức. Dtype lấy trực tiếp từ file thật.

### A. 11 trường gốc từ PaySim

| Cột | Kiểu dữ liệu | Đơn vị / Range | Ý nghĩa |
|---|---|---|---|
| `step` | int32 | Giờ mô phỏng, 1–743 (~31 ngày) | Thời điểm giao dịch, tính bằng số giờ kể từ lúc mô phỏng bắt đầu. `hour_of_day`/`is_night_transaction` (synthetic) suy ra từ cột này |
| `type` | category | {CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER} | Loại giao dịch. **Fraud chỉ xảy ra ở `TRANSFER` và `CASH_OUT`** (đặc điểm PaySim — xem mục 3) |
| `amount` | float32 | ≥ 0, thực tế đến ~92,4 triệu | Số tiền giao dịch (đơn vị tiền tệ mô phỏng). Phân phối lệch phải mạnh — xem `is_amount_outlier` |
| `nameOrig` | string | Bắt đầu bằng `C` + số | Mã định danh tài khoản khởi tạo giao dịch. 99,85% chỉ xuất hiện đúng 1 lần trong dataset (xem mục 3) |
| `oldbalanceOrg` | float32 | ≥ 0 | Số dư tài khoản nguồn **trước** giao dịch |
| `newbalanceOrig` | float32 | ≥ 0 | Số dư tài khoản nguồn **sau** giao dịch. So với `oldbalanceOrg - amount` để phát hiện `is_balance_inconsistent` |
| `nameDest` | string | Bắt đầu bằng `C` (khách hàng) hoặc `M` (merchant) | Mã định danh tài khoản/đối tượng nhận giao dịch |
| `oldbalanceDest` | float32 | ≥ 0, thực tế đến ~356 triệu | Số dư tài khoản đích **trước** giao dịch. Thường = 0 với merchant (không track) — nguồn gốc của tỷ lệ 80,45% "balance inconsistent" |
| `newbalanceDest` | float32 | ≥ 0 | Số dư tài khoản đích **sau** giao dịch |
| `isFraud` | int8 | {0, 1} | **Nhãn gian lận thật** (biến mục tiêu) — mô phỏng hành vi account-takeover (chiếm quyền tài khoản rồi rút/chuyển tiền) |
| `isFlaggedFraud` | int8 | {0, 1} (chỉ 16/6.362.620 dòng = 1) | Cờ cảnh báo **tự động, có sẵn trong PaySim** theo 1 rule đơn giản (transfer giá trị lớn) — **không phải nhãn thật**, không nên nhầm với `isFraud` hay với các cột flag cleaning |

### B. 13 trường synthetic (sinh ở Phần A, label-free — chi tiết công thức xem mục 7)

| Cột | Kiểu dữ liệu | Đơn vị / Range | Ý nghĩa |
|---|---|---|---|
| `hour_of_day` | int16 | [0, 23] | Giờ trong ngày, suy trực tiếp từ `step` |
| `is_night_transaction` | bool | {True, False} | Giao dịch diễn ra trong khung giờ đêm (0h–6h) |
| `customer_account_age_days` | int32 | [1, 3650] | Tuổi tài khoản khách hàng (ngày) tính đến thời điểm giao dịch — điều kiện theo risk proxy label-free, không đọc `isFraud` |
| `device_id` | string | UUID, pool 50.000 giá trị | Mã định danh thiết bị dùng để thực hiện giao dịch |
| `browser` | string | {Chrome, Safari, Edge, Firefox, Other} | Trình duyệt dùng để thực hiện giao dịch |
| `device_type` | string | {mobile, desktop, tablet} | Loại thiết bị |
| `new_device_flag` | bool | {True, False} | Giao dịch có đến từ thiết bị chưa từng ghi nhận với tài khoản này hay không — điều kiện theo risk proxy label-free |
| `billing_country` | string | Mã ISO, 20 quốc gia cố định | Quốc gia đăng ký/billing của khách hàng |
| `ip_country` | string | Mã ISO, 20 quốc gia cố định | Quốc gia suy ra từ địa chỉ IP thực hiện giao dịch — điều kiện theo risk proxy label-free |
| `ip_billing_distance_km` | float64 | [0, ~17.881] | Khoảng cách địa lý (km) giữa `ip_country` và `billing_country`, tính bằng haversine — 0 nếu hai quốc gia trùng nhau |
| `ip_billing_country_mismatch` | bool | {True, False} | `ip_country != billing_country` — bản boolean của cùng mismatch trên |
| `shipping_billing_mismatch` | bool | {True, False} | Địa chỉ giao dịch/nhận hàng có khác địa chỉ đăng ký hay không — điều kiện theo risk proxy label-free |
| `failed_payment_attempts_24h` | int16 | [0, ~5] | Số lần thanh toán thất bại trong 24 giờ trước giao dịch này — điều kiện theo risk proxy label-free |

### C. 3 trường flag từ Phần B — Data Cleaning (ý nghĩa chi tiết xem mục 14)

| Cột | Kiểu dữ liệu | Đơn vị / Range | Ý nghĩa |
|---|---|---|---|
| `is_amount_outlier` | bool | {True, False} (338.078 dòng = True) | `amount` nằm ngoài khoảng Tukey IQR bình thường — giá trị bất thường lớn, có thể là tín hiệu fraud, không phải nhiễu cần loại bỏ |
| `is_zero_amount` | bool | {True, False} (16 dòng = True) | `amount = 0` — toàn bộ 16 dòng quan sát được đều là fraud thật, tín hiệu hiếm nhưng rất mạnh |
| `is_balance_inconsistent` | bool | {True, False} (5.118.892 dòng = True) | `oldbalanceOrg - amount ≠ newbalanceOrig` — **đặc điểm cố hữu của PaySim, không phải lỗi dữ liệu**; tỷ lệ cao (80,45%) là bình thường, không nên báo cáo như vấn đề chất lượng dữ liệu |

## 17. Cấu trúc code & test

```
src/data_generation/
  country_centroids.py          # Bảng tọa độ 20 quốc gia + haversine distance
  generate_synthetic_fields.py  # compute_risk_proxy (label-free) + 13 hàm sinh field + orchestrator + CLI (CSV -> Parquet)
  check_leakage.py              # AUC / Cramér's V (bias-corrected) + sinh docs/DATA_DICTIONARY.md
src/data_cleaning/
  clean_transactions.py         # 6 hàm check/flag + orchestrator clean_dataset() + CLI
  cleaning_report.py            # build_cleaning_report_markdown() + CLI ghi docs/CLEANING_REPORT.md
tests/data_generation/          # 64 unit test
tests/data_cleaning/            # 18 unit test
```

82 test tổng cộng: đúng tỷ lệ base/high-risk theo từng công thức, tái lập được (reproducibility), không tràn kiểu dữ liệu, bất biến toán học kiểm tra exhaustive, schema guard khi đọc CSV, **static check xác nhận không hàm sinh nào nhận `isFraud` làm tham số**, **test hành vi xác nhận đổi `isFraud` giữ observable cố định thì output không đổi**, và với module cleaning: đúng số dòng xoá/flag theo từng kịch bản, không đụng đến dòng không liên quan.

**Cấu trúc file dữ liệu output** (`data/processed/`, đã `.gitignore` do dung lượng lớn — mỗi máy phải tự chạy lại mục 18 để tạo, không lấy được qua git):

| File | Giai đoạn | Số dòng | Số cột | Vai trò |
|---|---|---|---|---|
| **`transactions_cleaned.parquet`** | Sau Phần B | 6.362.620 | 27 | **Data FINAL — dùng cái này cho feature engineering/model, không dùng bản khác** |
| `transactions_cleaned.csv` | Sau Phần B | 6.362.620 | 27 | Cùng nội dung `.parquet` trên, dạng CSV đầy đủ — chỉ để mở bằng công cụ không đọc được parquet |
| `transactions_cleaned.zip` | Sau Phần B | 6.362.620 | 27 | Bản nén của `.csv` trên, để chia sẻ/nộp file gọn hơn |
| `transactions_cleaned_sample.csv` | Sau Phần B | ~5.000 (mẫu stratified) | 27 | **Không phải data final** — chỉ để xem nhanh bằng Excel, không dùng để train |
| `transactions_synthetic.parquet` | Sau Phần A | 6.362.620 | 24 | Kết quả trung gian (trước cleaning) — input của Phần B, không phải output cuối |
| `transactions_synthetic.csv` / `.zip` | Sau Phần A | 6.362.620 | 24 | Tương tự bản `.csv`/`.zip` của cleaned, nhưng cho dữ liệu trung gian |
| `transactions_synthetic_sample.csv` | Sau Phần A | ~5.000 (mẫu stratified) | 24 | Mẫu xem nhanh, không phải data final |

## 18. Cách chạy end-to-end

Yêu cầu: Python 3.13 (ví dụ `C:\ProgramData\miniconda3\python.exe`), chạy trong git-bash/MSYS.

```bash
# 0. Tạo venv và cài dependency (chỉ cần 1 lần)
"/c/ProgramData/miniconda3/python.exe" -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1. Chạy toàn bộ test
.venv/Scripts/python.exe -m pytest tests/ -v

# --- PHẦN A: Synthetic Data Generation ---
# 2. Sinh synthetic data từ dataset gốc (Data/PS_20174392719_1491204439457_log.csv)
PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.generate_synthetic_fields
# -> data/processed/transactions_synthetic.parquet (+ sample CSV)

# 3. Kiểm tra leakage + sinh data dictionary
PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.check_leakage
# -> docs/DATA_DICTIONARY.md

# --- PHẦN B: Data Cleaning ---
# 4. Làm sạch dữ liệu (đọc output của bước 2)
PYTHONPATH=src .venv/Scripts/python.exe -m data_cleaning.clean_transactions
# -> data/processed/transactions_cleaned.parquet (+ sample CSV)

# 5. Sinh cleaning report
PYTHONPATH=src .venv/Scripts/python.exe -m data_cleaning.cleaning_report
# -> docs/CLEANING_REPORT.md
```

**Nếu file CSV của bạn khác cấu trúc** (thiếu cột): bước 2 sẽ báo `ValueError` nêu rõ tên cột thiếu, thay vì lỗi pandas khó hiểu.

**Nếu bước 3 báo FAIL cho field nào:** mở `generate_synthetic_fields.py`, tìm hằng số điều khiển hệ số high-risk của field đó (ví dụ `NEW_DEVICE_FLAG_HIGH_RISK_P`), giảm nó về gần baseline hơn (nguyên tắc #4 ở mục 4), chạy lại bước 2 rồi bước 3 đến khi tất cả PASS. **Không** "sửa" bằng cách đọc `isFraud` lại — nếu cần tăng tín hiệu, tăng trọng số/độ nhạy của `compute_risk_proxy` với observable, không quay lại label-conditional.

## 19. Giới hạn & rủi ro đã biết

**Phần A (synthetic) — đọc kỹ trước khi trình bày, đây là phần dễ bị chất vấn nhất:**

- **Tính vòng lặp (circularity) vẫn còn, chỉ đổi hình thức.** Chuyển sang label-free (không đọc `isFraud`) giải quyết 2 vấn đề: (a) field không còn phụ thuộc vào thứ chưa biết được lúc scoring giao dịch mới, (b) code không còn "mùi leakage" đọc trực tiếp nhãn. Nó **không** làm dữ liệu thực tế hơn và **không** xoá hết tính vòng lặp — 5 field conditional vẫn được tiêm tương quan **do thiết kế** (qua risk proxy tự chọn), không phải học từ hành vi fraud thật. Tương quan đo được với `isFraud` (mục 9) là bằng chứng của quy trình kiểm soát rủi ro, **không phải** bằng chứng các field này sẽ dự đoán tốt trên fraud thật ngoài PaySim.
- **5 field conditional gần như không mang thêm thông tin ngoài `type`/`amount`/`hour_of_day`.** Vì `risk_score = compute_risk_proxy(type, amount, hour_of_day)`, các field điều kiện theo nó (`customer_account_age_days`, `new_device_flag`, `ip_country`, `shipping_billing_mismatch`, `failed_payment_attempts_24h`) về bản chất thống kê là hàm nhiễu của 3 cột đó. Nếu model ở Module 4-5 đã có `type`/`amount`/`hour_of_day`, 5 field này khó đóng góp thêm sức dự đoán đáng kể — giá trị của chúng nằm ở việc minh hoạ đúng *loại* tín hiệu hệ thống chống fraud thật dùng (device risk, geo mismatch, velocity), không phải ở việc tăng AUC.
- `amount_percentile` trong `compute_risk_proxy` là percentile-trong-batch (tính trên toàn bộ 6,36M dòng khi build dataset), không phải giá trị tự tính được từ 1 giao dịch đơn lẻ — muốn tái tạo đúng cho giao dịch mới lúc serving cần lưu lại phân phối `amount` theo `type` từ lúc train, không chỉ đọc `type`/`amount` của giao dịch đó.
- Các hệ số odds-ratio/λ là giả định nghiệp vụ tự đặt, không suy từ số liệu fraud thực tế công khai nào.
- `shipping_billing_mismatch` được diễn giải lại thành "địa chỉ giao dịch khác địa chỉ đăng ký" do fraud trong PaySim là account-takeover, không phải checkout thẻ.
- 9.313 `nameOrig` có lặp lại (0,15%) được xử lý như dòng độc lập.
- Cột `amount`/số dư gốc lưu ở `float32` — có thể mất độ chính xác nhỏ ở giá trị lớn; không ảnh hưởng 13 field synthetic.

**Phần B (cleaning):**
- Ngưỡng outlier IQR (1.5×IQR, chuẩn Tukey) là 1 lựa chọn thống kê phổ biến, không phải "đúng duy nhất" — bước modeling sau có thể cần ngưỡng khác.
- Outlier/inconsistency chỉ được **flag**, không loại khỏi dataset — quyết định có dùng làm feature hay không thuộc về bước sau.
- Check invalid category cho cột synthetic dựa trên danh sách giá trị đã dùng để sinh ở Phần A — nếu sinh lại dữ liệu với danh sách khác, cần đồng bộ lại.

**Rủi ro ngoài phạm vi 2 phần này, nhưng ảnh hưởng cả pipeline — cần báo cho team Model Development (Module 5):**
- `oldbalanceOrg`/`newbalanceOrig` trong dữ liệu PaySim **gốc** (không phải synthetic) gần như xác định `isFraud` (fraud = rút cạn tài khoản). Đây là leakage **có sẵn trong nguồn dữ liệu gốc**, không liên quan đến 13 field synthetic hay 3 flag cleaning ở đây (2 phần này đã kiểm chứng không leak — mục 9). Nếu Module 5 báo AUC-PR/precision gần 1.0, khả năng cao là do leakage này, không phải do model hay feature engineering giỏi — nên train thêm 1 phiên bản model bỏ 2 cột số dư gốc để so sánh, và ghi rõ trong report nếu giữ chúng.

## 20. Dùng output cho bước tiếp theo

- Dùng `transactions_cleaned.parquet`/`.csv` (27 cột) làm input cho feature engineering — đây là bản đầy đủ nhất, đã qua cả 2 giai đoạn.
- Các field string (`device_id`, `billing_country`, `ip_country`, `browser`, `device_type`) cần encode; `ip_billing_distance_km`, `failed_payment_attempts_24h` đã là numeric, dùng trực tiếp được.
- 3 cột flag mới (`is_amount_outlier`, `is_zero_amount`, `is_balance_inconsistent`) là boolean, **dùng được ngay làm feature** — xem ý nghĩa chi tiết ở mục 14 trước khi quyết định giữ/bỏ trong model.
- Đọc kỹ mục 19 trước khi viết phần "Feature Importance"/"Model Development" của report: nếu 5 field synthetic conditional không lên top feature importance, đó là **kỳ vọng đúng** (chúng gần như redundant với `type`/`amount`/`hour`), không phải lỗi.
- Nếu dataset có sẵn feature dạng balance-delta (số dư trước/sau giao dịch), cần lưu ý: PaySim fraud thường rút sạch số dư nên các feature đó có thể cho AUC-PR rất cao một cách đáng ngờ (leakage sẵn có trong dữ liệu gốc, không liên quan đến các field synthetic/cleaning ở đây). Nên train mô hình có/không các feature đó để so sánh.
