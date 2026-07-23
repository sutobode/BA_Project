import pandas as pd
import pytest

from data_cleaning import clean_transactions as ct


def test_check_missing_critical_removes_rows_with_na_in_critical_columns():
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "PAYMENT", "PAYMENT"],
        "amount": [10.0, None, 30.0],
        "isFraud": [0, 0, 1],
        "nameOrig": ["C1", "C2", "C3"],
    })
    cleaned, na_counts, n_removed = ct.check_missing_critical(df)
    assert n_removed == 1
    assert len(cleaned) == 2
    assert na_counts["amount"] == 1
    assert list(cleaned["step"]) == [1, 3]


def test_check_missing_critical_keeps_rows_with_na_only_in_noncritical_columns():
    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "PAYMENT"],
        "amount": [10.0, 20.0],
        "isFraud": [0, 0],
        "nameOrig": ["C1", None],
    })
    cleaned, na_counts, n_removed = ct.check_missing_critical(df)
    assert n_removed == 0
    assert len(cleaned) == 2
    assert na_counts["nameOrig"] == 1


def test_check_missing_critical_reports_zero_when_no_missing_values():
    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [10.0, 20.0],
        "isFraud": [0, 1],
    })
    cleaned, na_counts, n_removed = ct.check_missing_critical(df)
    assert n_removed == 0
    assert all(v == 0 for v in na_counts.values())
    assert len(cleaned) == 2


def test_dedupe_exact_removes_full_row_duplicates():
    df = pd.DataFrame({
        "step": [1, 1, 2],
        "type": ["PAYMENT", "PAYMENT", "TRANSFER"],
        "amount": [10.0, 10.0, 20.0],
    })
    deduped, n_removed = ct.dedupe_exact(df)
    assert n_removed == 1
    assert len(deduped) == 2


def test_dedupe_exact_keeps_rows_that_differ_in_any_column():
    df = pd.DataFrame({
        "step": [1, 1],
        "type": ["PAYMENT", "PAYMENT"],
        "amount": [10.0, 10.1],
    })
    deduped, n_removed = ct.dedupe_exact(df)
    assert n_removed == 0
    assert len(deduped) == 2


def test_dedupe_exact_reports_zero_when_no_duplicates_exist():
    df = pd.DataFrame({"step": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    deduped, n_removed = ct.dedupe_exact(df)
    assert n_removed == 0
    assert len(deduped) == 3


def _valid_category_row(**overrides):
    row = {
        "type": "PAYMENT",
        "browser": "Chrome",
        "device_type": "mobile",
        "billing_country": "US",
        "ip_country": "US",
        "is_night_transaction": True,
        "new_device_flag": False,
        "shipping_billing_mismatch": False,
        "ip_billing_country_mismatch": False,
    }
    row.update(overrides)
    return row


def test_check_invalid_categories_removes_unknown_transaction_type():
    df = pd.DataFrame([
        _valid_category_row(type="PAYMENT"),
        _valid_category_row(type="BOGUS_TYPE"),
        _valid_category_row(type="TRANSFER"),
    ])
    cleaned, counts, n_removed = ct.check_invalid_categories(df)
    assert n_removed == 1
    assert counts["type"] == 1
    assert len(cleaned) == 2


def test_check_invalid_categories_removes_unknown_country():
    df = pd.DataFrame([
        _valid_category_row(billing_country="US"),
        _valid_category_row(billing_country="ZZ"),
    ])
    cleaned, counts, n_removed = ct.check_invalid_categories(df)
    assert n_removed == 1
    assert counts["billing_country"] == 1


def test_check_invalid_categories_reports_zero_when_all_valid():
    df = pd.DataFrame([
        _valid_category_row(type="PAYMENT", browser="Chrome", device_type="mobile", billing_country="US", ip_country="US"),
        _valid_category_row(type="TRANSFER", browser="Safari", device_type="desktop", billing_country="VN", ip_country="VN"),
        _valid_category_row(type="CASH_OUT", browser="Edge", device_type="tablet", billing_country="GB", ip_country="GB"),
        _valid_category_row(type="CASH_IN", browser="Firefox", device_type="mobile", billing_country="US", ip_country="US"),
        _valid_category_row(type="DEBIT", browser="Other", device_type="desktop", billing_country="VN", ip_country="VN"),
    ])
    cleaned, counts, n_removed = ct.check_invalid_categories(df)
    assert n_removed == 0
    assert len(cleaned) == 5
    assert all(v == 0 for v in counts.values())


def test_flag_amount_outliers_flags_values_beyond_tukey_fence():
    amounts = pd.Series([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 5000.0])
    result = ct.flag_amount_outliers(amounts)
    assert result.iloc[-1] == True
    assert not result.iloc[:-1].any()


def test_flag_amount_outliers_flags_nothing_for_tight_distribution():
    amounts = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8])
    result = ct.flag_amount_outliers(amounts)
    assert not result.any()


def test_flag_zero_amount_flags_only_zero_values():
    amounts = pd.Series([0.0, 10.0, 0.0, 5.0])
    result = ct.flag_zero_amount(amounts)
    assert list(result) == [True, False, True, False]


def test_flag_balance_inconsistency_flags_mismatched_rows():
    old = pd.Series([100.0, 100.0, 50.0])
    amount = pd.Series([30.0, 30.0, 20.0])
    new = pd.Series([70.0, 60.0, 30.0])
    result = ct.flag_balance_inconsistency(old, amount, new)
    assert list(result) == [False, True, False]


def test_flag_balance_inconsistency_respects_tolerance():
    old = pd.Series([100.0])
    amount = pd.Series([30.0])
    new = pd.Series([70.005])
    result = ct.flag_balance_inconsistency(old, amount, new)
    assert list(result) == [False]


def _full_schema_row(**overrides):
    row = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 10.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 90.0,
        "isFraud": 0,
        "browser": "Chrome",
        "device_type": "mobile",
        "billing_country": "US",
        "ip_country": "US",
        "is_night_transaction": True,
        "new_device_flag": False,
        "shipping_billing_mismatch": False,
        "ip_billing_country_mismatch": False,
    }
    row.update(overrides)
    return row


def test_clean_dataset_adds_three_flag_columns_and_preserves_row_count_when_no_issues():
    df = pd.DataFrame([
        _full_schema_row(step=1, amount=10.0, oldbalanceOrg=100.0, newbalanceOrig=90.0, isFraud=0),
        _full_schema_row(step=2, amount=20.0, oldbalanceOrg=200.0, newbalanceOrig=180.0, isFraud=0),
        _full_schema_row(step=3, amount=30.0, oldbalanceOrg=300.0, newbalanceOrig=270.0, isFraud=1, type="CASH_OUT"),
    ])
    cleaned, report_data = ct.clean_dataset(df)
    assert len(cleaned) == 3
    assert report_data["input_rows"] == 3
    assert report_data["output_rows"] == 3
    assert {"is_amount_outlier", "is_zero_amount", "is_balance_inconsistent"}.issubset(cleaned.columns)


def test_clean_dataset_removes_rows_with_missing_critical_values():
    df = pd.DataFrame([
        _full_schema_row(step=1, type="PAYMENT"),
        _full_schema_row(step=2, type=None),
    ])
    cleaned, report_data = ct.clean_dataset(df)
    assert len(cleaned) == 1
    assert report_data["missing_values"]["rows_removed"] == 1
    assert report_data["output_rows"] == 1


def test_clean_dataset_flags_zero_amount_without_removing_it():
    df = pd.DataFrame([
        _full_schema_row(step=1, amount=0.0, oldbalanceOrg=0.0, newbalanceOrig=0.0, isFraud=1, type="CASH_OUT"),
        _full_schema_row(step=2, amount=10.0, oldbalanceOrg=100.0, newbalanceOrig=90.0, isFraud=0),
    ])
    cleaned, report_data = ct.clean_dataset(df)
    assert len(cleaned) == 2
    assert report_data["zero_amount"]["rows_flagged"] == 1
    assert cleaned.loc[cleaned["step"] == 1, "is_zero_amount"].iloc[0] == True
