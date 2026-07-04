import pandas as pd

CRITICAL_COLUMNS = ["step", "type", "amount", "isFraud"]


def check_missing_critical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_column_na_counts = df.isna().sum().to_dict()
    critical_na_mask = df[CRITICAL_COLUMNS].isna().any(axis=1)
    n_removed = int(critical_na_mask.sum())
    cleaned = df[~critical_na_mask].reset_index(drop=True)
    return cleaned, per_column_na_counts, n_removed
