from pathlib import Path
import pandas as pd

from data_cleaning.clean_transactions import clean_dataset, INPUT_PARQUET_PATH


def build_cleaning_report_markdown(report_data: dict) -> str:
    lines = [
        "# Cleaning Report — transactions_synthetic\n\n",
        f"Input rows: {report_data['input_rows']} | Output rows: {report_data['output_rows']}\n\n",
        "| check | rows_before | rows_flagged_or_removed | action | note |\n",
        "|---|---|---|---|---|\n",
    ]

    lines.append(
        f"| missing_values (critical columns) | {report_data['input_rows']} | "
        f"{report_data['missing_values']['rows_removed']} | removed | "
        f"NaN counts per column: {report_data['missing_values']['per_column_na_counts']} |\n"
    )
    lines.append(
        f"| duplicates (full-row) | {report_data['input_rows']} | "
        f"{report_data['duplicates']['rows_removed']} | removed | Exact duplicate rows only |\n"
    )
    lines.append(
        f"| invalid_categories | {report_data['input_rows']} | "
        f"{report_data['invalid_categories']['rows_removed']} | removed | "
        f"Per-column counts: {report_data['invalid_categories']['per_check_invalid_counts']} |\n"
    )
    lines.append(
        f"| amount_outliers (Tukey IQR) | {report_data['output_rows']} | "
        f"{report_data['amount_outliers']['rows_flagged']} | flagged (kept) | "
        f"Column: is_amount_outlier. Large amounts may be genuine fraud signal, not removed. |\n"
    )
    lines.append(
        f"| zero_amount | {report_data['output_rows']} | "
        f"{report_data['zero_amount']['rows_flagged']} | flagged (kept) | "
        f"Column: is_zero_amount. Observed zero-amount rows are confirmed fraud, not removed. |\n"
    )
    lines.append(
        f"| balance_inconsistent | {report_data['output_rows']} | "
        f"{report_data['balance_inconsistent']['rows_flagged']} | flagged (kept) | "
        f"Column: is_balance_inconsistent. Direction-aware check on the ORIGIN account only "
        f"(oldbalanceOrg/newbalanceOrig vs amount and type - CASH_IN deposits, other types "
        f"withdraw); does not read destination/merchant balances at all. **Most flags on the "
        f"real dataset come from PaySim recording amount > oldbalanceOrg (an over-draft/"
        f"insufficient-funds transaction) with newbalanceOrig floored at 0 rather than rejected "
        f"or negative - a known PaySim data characteristic, not a data-entry error** - do not "
        f"interpret a high count here as a data quality problem. |\n"
    )
    return "".join(lines)


def main():
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    _, report_data = clean_dataset(df)
    markdown = build_cleaning_report_markdown(report_data)
    Path("docs/CLEANING_REPORT.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Wrote docs/CLEANING_REPORT.md")


if __name__ == "__main__":
    main()
