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
BOOLEAN_COLUMNS = [
    "is_night_transaction", "new_device_flag", "shipping_billing_mismatch",
    "ip_billing_country_mismatch",
]


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


def flag_amount_outliers(amount: pd.Series) -> pd.Series:
    """Tukey IQR fence: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
    Flagged, not removed - a large transaction amount can itself be a fraud signal.
    """
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (amount < lower) | (amount > upper)


def flag_zero_amount(amount: pd.Series) -> pd.Series:
    """amount == 0 transactions are flagged, not removed - observed real-data
    zero-amount CASH_OUT rows are all confirmed fraud (isFraud=1)."""
    return amount == 0


def flag_balance_inconsistency(
    old_balance_org: pd.Series, amount: pd.Series, new_balance_orig: pd.Series, tolerance: float = 0.01
) -> pd.Series:
    """Flags rows where oldbalanceOrg - amount != newbalanceOrig beyond tolerance.
    This is a known PaySim data characteristic (destination/merchant balances
    often untracked), not a data-entry error - flagged, not removed."""
    return (old_balance_org - amount - new_balance_orig).abs() > tolerance


from pathlib import Path
from data_generation.generate_synthetic_fields import build_stratified_sample

INPUT_PARQUET_PATH = "data/processed/transactions_synthetic.parquet"
OUTPUT_PARQUET_PATH = "data/processed/transactions_cleaned.parquet"
OUTPUT_SAMPLE_CSV_PATH = "data/processed/transactions_cleaned_sample.csv"
SAMPLE_SIZE = 5000


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report_data = {"input_rows": len(df)}

    df, na_counts, n_removed_missing = check_missing_critical(df)
    report_data["missing_values"] = {"per_column_na_counts": na_counts, "rows_removed": n_removed_missing}

    df, n_removed_dupes = dedupe_exact(df)
    report_data["duplicates"] = {"rows_removed": n_removed_dupes}

    df, category_counts, n_removed_categories = check_invalid_categories(df)
    report_data["invalid_categories"] = {"per_check_invalid_counts": category_counts, "rows_removed": n_removed_categories}

    out = df.copy()
    out["is_amount_outlier"] = flag_amount_outliers(out["amount"])
    out["is_zero_amount"] = flag_zero_amount(out["amount"])
    out["is_balance_inconsistent"] = flag_balance_inconsistency(
        out["oldbalanceOrg"], out["amount"], out["newbalanceOrig"]
    )

    report_data["amount_outliers"] = {"rows_flagged": int(out["is_amount_outlier"].sum())}
    report_data["zero_amount"] = {"rows_flagged": int(out["is_zero_amount"].sum())}
    report_data["balance_inconsistent"] = {"rows_flagged": int(out["is_balance_inconsistent"].sum())}
    report_data["output_rows"] = len(out)

    return out, report_data


def main():
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    cleaned, report_data = clean_dataset(df)
    Path(OUTPUT_PARQUET_PATH).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    sample = build_stratified_sample(cleaned, sample_size=SAMPLE_SIZE, seed=42)
    sample.to_csv(OUTPUT_SAMPLE_CSV_PATH, index=False)
    print(f"Input rows: {report_data['input_rows']}")
    print(f"Output rows: {report_data['output_rows']}")
    print(f"Wrote {len(cleaned)} rows to {OUTPUT_PARQUET_PATH}")
    print(f"Wrote {len(sample)} sample rows to {OUTPUT_SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()
