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
