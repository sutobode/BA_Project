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
