# Data Dictionary — Synthetic Contextual Fields

| field_name | data_type | unit | valid_range | generation_type | logic_or_formula | business_assumption | measured_univariate_association_with_isFraud |
|---|---|---|---|---|---|---|---|
| `hour_of_day` | int16 | hour (0-23) | [0, 23] | derived | (step - 1) % 24 | step in PaySim represents elapsed hours since simulation start. | 0.6336 (AUC) |
| `is_night_transaction` | bool | boolean | {True, False} | derived | hour_of_day in [0, 5] | Night defined as 00:00-05:59. | 0.6217 (AUC) |
| `customer_account_age_days` | int32 | days | [1, 3650] | conditional-on-fraud | lognormal(median=400 base / 275 fraud, sigma=0.6), clipped | Compromised/mule accounts assumed more often recently created (business assumption, not empirically derived). | 0.6689 (AUC) |
| `device_id` | string | UUID | 50,000-value pool | independent-random | uniform sample from a fixed 50,000-UUID Faker-generated pool | Device identity alone carries no fraud signal; signal lives in new_device_flag. | 0.0879 (Cramér's V) |
| `browser` | category | categorical | {Chrome, Safari, Edge, Firefox, Other} | independent-random | weighted categorical: Chrome 55%, Safari 20%, Edge 12%, Firefox 8%, Other 5% | No behavioral link to fraud assumed; kept neutral to avoid manufactured signal. | 0.0007 (Cramér's V) |
| `device_type` | category | categorical | {mobile, desktop, tablet} | independent-random | weighted categorical: mobile 65%, desktop 30%, tablet 5% | No behavioral link to fraud assumed. | 0.0007 (Cramér's V) |
| `new_device_flag` | bool | boolean | {True, False} | conditional-on-fraud | Bernoulli(p=0.04 base / 0.12 fraud) | Account-takeover fraud plausibly more often from an unrecognized device; odds-ratio capped at 3x (business assumption). | 0.5419 (AUC) |
| `billing_country` | category | ISO country code | 20 fixed countries | independent-random | weighted categorical over 20 countries (simulated customer base geography) | No direct fraud link; signal lives in ip_country mismatch. | 0.0017 (Cramér's V) |
| `ip_country` | category | ISO country code | 20 fixed countries | conditional-on-fraud | = billing_country with p=0.93 base / 0.80 fraud, else a different country | Legit traffic mostly matches home country; fraud has higher (but not certain) mismatch odds since VPNs let fraud spoof location too. | 0.0052 (Cramér's V) |
| `ip_billing_distance_km` | float64 | km | [0, ~20000] | derived | haversine(centroid[ip_country], centroid[billing_country]) | Derived directly from ip_country/billing_country for internal consistency. | 0.5651 (AUC) |
| `shipping_billing_mismatch` | bool | boolean | {True, False} | conditional-on-fraud | Bernoulli(p=0.05 base / 0.15 fraud) | Reinterpreted as 'transaction address differs from registered address' given PaySim's account-takeover fraud pattern (not card-present checkout). | 0.5495 (AUC) |
| `failed_payment_attempts_24h` | int16 | count | [0, ~10] | conditional-on-fraud | Poisson(lambda=0.15 base / 0.6 fraud) | Attackers often attempt multiple times before succeeding; odds-ratio capped at 4x (business assumption). | 0.6589 (AUC) |
