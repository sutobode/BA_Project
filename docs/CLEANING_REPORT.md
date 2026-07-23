# Cleaning Report — transactions_synthetic

Input rows: 6362620 | Output rows: 6362620

| check | rows_before | rows_flagged_or_removed | action | note |
|---|---|---|---|---|
| missing_values (critical columns) | 6362620 | 0 | removed | NaN counts per column: {'step': 0, 'type': 0, 'amount': 0, 'nameOrig': 0, 'oldbalanceOrg': 0, 'newbalanceOrig': 0, 'nameDest': 0, 'oldbalanceDest': 0, 'newbalanceDest': 0, 'isFraud': 0, 'isFlaggedFraud': 0, 'hour_of_day': 0, 'is_night_transaction': 0, 'customer_account_age_days': 0, 'device_id': 0, 'new_device_flag': 0, 'browser': 0, 'device_type': 0, 'billing_country': 0, 'ip_country': 0, 'ip_billing_distance_km': 0, 'ip_billing_country_mismatch': 0, 'shipping_billing_mismatch': 0, 'failed_payment_attempts_24h': 0} |
| duplicates (full-row) | 6362620 | 0 | removed | Exact duplicate rows only |
| invalid_categories | 6362620 | 0 | removed | Per-column counts: {'type': 0, 'browser': 0, 'device_type': 0, 'billing_country': 0, 'ip_country': 0, 'is_night_transaction': 0, 'new_device_flag': 0, 'shipping_billing_mismatch': 0, 'ip_billing_country_mismatch': 0} |
| amount_outliers (Tukey IQR) | 6362620 | 338078 | flagged (kept) | Column: is_amount_outlier. Large amounts may be genuine fraud signal, not removed. |
| zero_amount | 6362620 | 16 | flagged (kept) | Column: is_zero_amount. Observed zero-amount rows are confirmed fraud, not removed. |
| balance_inconsistent | 6362620 | 5118892 | flagged (kept) | Column: is_balance_inconsistent. **This is a known PaySim data characteristic (destination/merchant balances often untracked), not a data-entry error** - do not interpret a high count here as a data quality problem. |
