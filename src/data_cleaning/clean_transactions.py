import pandas as pd

CRITICAL_COLUMNS = ["step", "type", "amount", "isFraud"]


def check_missing_critical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_column_na_counts = df.isna().sum().to_dict()
    critical_na_mask = df[CRITICAL_COLUMNS].isna().any(axis=1)
    n_removed = int(critical_na_mask.sum())
    cleaned = df[~critical_na_mask].reset_index(drop=True)
    return cleaned, per_column_na_counts, n_removed


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(deduped)
    return deduped, n_removed


from data_generation.generate_synthetic_fields import BROWSER_WEIGHTS, DEVICE_TYPE_WEIGHTS
from data_generation.country_centroids import COUNTRY_WEIGHTS

VALID_TRANSACTION_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}
VALID_BROWSERS = set(BROWSER_WEIGHTS.keys())
VALID_DEVICE_TYPES = set(DEVICE_TYPE_WEIGHTS.keys())
VALID_COUNTRIES = set(COUNTRY_WEIGHTS.keys())
BOOLEAN_COLUMNS = ["is_night_transaction", "new_device_flag", "shipping_billing_mismatch"]


def check_invalid_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_check_invalid_counts = {}
    invalid_mask = pd.Series(False, index=df.index)

    category_checks = [
        ("type", VALID_TRANSACTION_TYPES),
        ("browser", VALID_BROWSERS),
        ("device_type", VALID_DEVICE_TYPES),
        ("billing_country", VALID_COUNTRIES),
        ("ip_country", VALID_COUNTRIES),
    ]
    for column, valid_values in category_checks:
        column_invalid = ~df[column].isin(valid_values)
        per_check_invalid_counts[column] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    for column in BOOLEAN_COLUMNS:
        column_invalid = ~df[column].isin([True, False])
        per_check_invalid_counts[column] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    n_removed = int(invalid_mask.sum())
    cleaned = df[~invalid_mask].reset_index(drop=True)
    return cleaned, per_check_invalid_counts, n_removed
