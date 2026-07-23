import pandas as pd

CRITICAL_COLUMNS = ["step", "type", "amount", "isFraud"]


def check_missing_critical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_column_na_counts = df.isna().sum().to_dict()
    critical_na_mask = df[CRITICAL_COLUMNS].isna().any(axis=1)
    n_removed = int(critical_na_mask.sum())
    cleaned = df[~critical_na_mask]
    return cleaned, per_column_na_counts, n_removed


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    deduped = df.drop_duplicates()
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
    cleaned = df[~invalid_mask]
    return cleaned, per_check_invalid_counts, n_removed


def fit_tukey_fences(amount: pd.Series) -> tuple[float, float]:
    """FIT step (train-only if is_amount_outlier will be used as a model
    feature): Tukey IQR fence bounds computed from amount.

    Call this on the TRAIN split only when is_amount_outlier is going to be
    consumed downstream as a model feature - fitting Q1/Q3 on the full
    dataset (including rows that will end up in a held-out validation/test
    split) lets those rows' amounts influence the fence used to flag every
    row, the same category of train/test leakage that
    fit_amount_percentile_reference() in generate_synthetic_fields.py exists
    to prevent. If is_amount_outlier is only used for reporting/EDA (not fed
    into a model), fitting on the full dataset is fine - the leakage concern
    only applies when the flag becomes a feature a model is trained and
    evaluated on.
    """
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(lower), float(upper)


def apply_tukey_fences(amount: pd.Series, fences: tuple[float, float]) -> pd.Series:
    """TRANSFORM step: flag values outside the given (lower, upper) fence,
    fitted elsewhere via fit_tukey_fences(). Needs no other rows - reproducible
    for a single new transaction at scoring time given only the persisted
    fence bounds."""
    lower, upper = fences
    return (amount < lower) | (amount > upper)


def flag_amount_outliers(amount: pd.Series, fences: tuple[float, float] | None = None) -> pd.Series:
    """Tukey IQR fence: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
    Flagged, not removed - a large transaction amount can itself be a fraud signal.

    fences: optional (lower, upper) bounds, typically from fit_tukey_fences()
    on a train split (see that function's docstring for when this matters).
    When omitted, fences are fit on-the-fly from this call's own `amount` -
    correct only when this call's input has no further train/test split
    downstream, or when is_amount_outlier will not be used as a model
    feature. clean_dataset() accepts a train_mask for exactly this reason.
    """
    if fences is None:
        fences = fit_tukey_fences(amount)
    return apply_tukey_fences(amount, fences)


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


def clean_dataset(df: pd.DataFrame, split_manifest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """split_manifest: optional dataframe from split_manifest.py (columns
    row_index, split), aligned to df's ORIGINAL row order/index before any
    rows are removed below. When provided, Tukey fences for is_amount_outlier
    are fit on the train-labeled rows ONLY (looked up by original row_index,
    which correctly survives even if the three row-removal checks below
    actually remove rows - see split_manifest.train_mask_for_row_indices)
    and then applied to every surviving row. This matters only if
    is_amount_outlier will be consumed downstream as a model feature (see
    fit_tukey_fences() docstring); if it's only used for reporting/EDA, omit
    split_manifest and fences are fit on the full cleaned dataset (the
    previous, simpler behavior).

    Team decision: split_manifest is the SAME shared 60/20/20 manifest
    (split_manifest.py) that generate_synthetic_fields.py uses for
    amount_percentile_reference and that Model Development (Module 5) is
    expected to reuse - not a locally-drawn split. Passing a manifest built
    from a different row count than df's input will raise (see
    train_mask_for_row_indices), rather than silently misaligning.

    df's index is used as the original row_index for this alignment - do
    not pass a df whose index was already reset/shuffled relative to the
    row order it was loaded in (the CLI in main() reads it fresh from
    parquet, which preserves this by construction).
    """
    report_data = {"input_rows": len(df)}

    df, na_counts, n_removed_missing = check_missing_critical(df)
    report_data["missing_values"] = {"per_column_na_counts": na_counts, "rows_removed": n_removed_missing}

    df, n_removed_dupes = dedupe_exact(df)
    report_data["duplicates"] = {"rows_removed": n_removed_dupes}

    df, category_counts, n_removed_categories = check_invalid_categories(df)
    report_data["invalid_categories"] = {"per_check_invalid_counts": category_counts, "rows_removed": n_removed_categories}

    out = df.copy()
    if split_manifest is not None:
        from data_generation.split_manifest import train_mask_for_row_indices
        train_mask = train_mask_for_row_indices(split_manifest, out.index)
        fences = fit_tukey_fences(out.loc[train_mask, "amount"])
    else:
        fences = None
    out["is_amount_outlier"] = flag_amount_outliers(out["amount"], fences=fences)
    out["is_zero_amount"] = flag_zero_amount(out["amount"])
    out["is_balance_inconsistent"] = flag_balance_inconsistency(
        out["oldbalanceOrg"], out["amount"], out["newbalanceOrig"]
    )

    report_data["amount_outliers"] = {"rows_flagged": int(out["is_amount_outlier"].sum())}
    report_data["zero_amount"] = {"rows_flagged": int(out["is_zero_amount"].sum())}
    report_data["balance_inconsistent"] = {"rows_flagged": int(out["is_balance_inconsistent"].sum())}
    report_data["output_rows"] = len(out)

    out = out.reset_index(drop=True)
    return out, report_data


def main():
    from data_generation.split_manifest import get_or_create_split_manifest

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    # Fit Tukey fences on the SHARED train split (same manifest used by
    # generate_synthetic_fields.py for amount_percentile_reference), since
    # is_amount_outlier is documented (README) as usable directly as a model
    # feature - see clean_dataset()/fit_tukey_fences() docstrings.
    manifest = get_or_create_split_manifest(len(df))
    cleaned, report_data = clean_dataset(df, split_manifest=manifest)
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
