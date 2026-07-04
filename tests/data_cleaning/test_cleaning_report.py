from data_cleaning import cleaning_report as cr


def test_build_cleaning_report_markdown_includes_all_checks_and_counts():
    report_data = {
        "input_rows": 100,
        "output_rows": 95,
        "missing_values": {"per_column_na_counts": {"amount": 0}, "rows_removed": 2},
        "duplicates": {"rows_removed": 1},
        "invalid_categories": {"per_check_invalid_counts": {"type": 2}, "rows_removed": 2},
        "amount_outliers": {"rows_flagged": 5},
        "zero_amount": {"rows_flagged": 1},
        "balance_inconsistent": {"rows_flagged": 80},
    }
    markdown = cr.build_cleaning_report_markdown(report_data)
    assert "100" in markdown
    assert "95" in markdown
    assert "is_amount_outlier" in markdown
    assert "is_zero_amount" in markdown
    assert "is_balance_inconsistent" in markdown
    assert "PaySim data characteristic" in markdown
    assert markdown.startswith("# Cleaning Report")
