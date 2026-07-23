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


# --- fit/transform split for Tukey fences (train-only fitting) ---


def test_fit_tukey_fences_returns_lower_and_upper_bounds():
    amounts = pd.Series([10.0, 12.0, 11.0, 13.0, 10.0, 12.0])
    lower, upper = ct.fit_tukey_fences(amounts)
    assert lower < amounts.min()
    assert upper > amounts.max()


def test_apply_tukey_fences_matches_flag_amount_outliers_with_same_fences():
    amounts = pd.Series([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 5000.0])
    fences = ct.fit_tukey_fences(amounts)
    applied = ct.apply_tukey_fences(amounts, fences)
    direct = ct.flag_amount_outliers(amounts)  # fits its own fences from the same data
    pd.testing.assert_series_equal(applied, direct)


def test_apply_tukey_fences_reproducible_for_a_single_new_row_using_persisted_fences():
    # Core claim: given ONLY a persisted (lower, upper) tuple - no access to
    # any other rows - a single new transaction's outlier flag is computable
    # deterministically. This is what makes is_amount_outlier reproducible at
    # scoring time instead of depending on whatever batch happens to be
    # passed in.
    train_amounts = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8])
    fences = ct.fit_tukey_fences(train_amounts)

    new_row_amount = pd.Series([5000.0])
    result_1 = ct.apply_tukey_fences(new_row_amount, fences)
    result_2 = ct.apply_tukey_fences(new_row_amount, fences)
    assert result_1.iloc[0] == result_2.iloc[0] == True


def test_flag_amount_outliers_with_explicit_fences_does_not_refit_from_input():
    # If fences are supplied, flag_amount_outliers must use them as-is rather
    # than silently refitting from `amount` - otherwise passing fences would
    # be a no-op and train/test leakage would persist.
    train_amounts = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8])
    fences = ct.fit_tukey_fences(train_amounts)

    # A test-only amount far outside the training range - if
    # flag_amount_outliers refit from this call's own input (batch of size
    # 1), it would trivially flag nothing (a single value can't be an
    # outlier against itself). Using the persisted train fences correctly,
    # it must be flagged.
    test_amount = pd.Series([5000.0])
    result = ct.flag_amount_outliers(test_amount, fences=fences)
    assert result.iloc[0] == True


def test_flag_zero_amount_flags_only_zero_values():
    amounts = pd.Series([0.0, 10.0, 0.0, 5.0])
    result = ct.flag_zero_amount(amounts)
    assert list(result) == [True, False, True, False]


def test_flag_balance_inconsistency_flags_mismatched_withdrawal_rows():
    type_ = pd.Series(["TRANSFER", "TRANSFER", "CASH_OUT"])
    old = pd.Series([100.0, 100.0, 50.0])
    amount = pd.Series([30.0, 30.0, 20.0])
    new = pd.Series([70.0, 60.0, 30.0])
    result = ct.flag_balance_inconsistency(type_, old, amount, new)
    assert list(result) == [False, True, False]


def test_flag_balance_inconsistency_respects_tolerance():
    type_ = pd.Series(["TRANSFER"])
    old = pd.Series([100.0])
    amount = pd.Series([30.0])
    new = pd.Series([70.005])
    result = ct.flag_balance_inconsistency(type_, old, amount, new)
    assert list(result) == [False]


def test_flag_balance_inconsistency_uses_deposit_formula_for_cash_in():
    # CASH_IN adds money TO the origin account: oldbalanceOrg + amount ==
    # newbalanceOrig is the correct identity, not oldbalanceOrg - amount.
    type_ = pd.Series(["CASH_IN", "CASH_IN"])
    old = pd.Series([100.0, 100.0])
    amount = pd.Series([30.0, 30.0])
    new = pd.Series([130.0, 70.0])  # row 0: correctly recorded deposit; row 1: inconsistent
    result = ct.flag_balance_inconsistency(type_, old, amount, new)
    assert list(result) == [False, True]


def test_flag_balance_inconsistency_does_not_flag_correctly_recorded_cash_in_that_would_fail_withdrawal_formula():
    # Regression guard for the fixed bug: a correctly-recorded CASH_IN
    # deposit fails the old (withdrawal-only) formula but must NOT be
    # flagged under the corrected, direction-aware formula.
    type_ = pd.Series(["CASH_IN"])
    old = pd.Series([1000.0])
    amount = pd.Series([500.0])
    new = pd.Series([1500.0])
    result = ct.flag_balance_inconsistency(type_, old, amount, new)
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


def test_clean_dataset_with_split_manifest_still_produces_all_required_columns():
    # Regression guard: passing a split manifest must change which rows the
    # Tukey fences are fit on, without crashing or silently ignoring the
    # argument, and must not change the contract (columns, row count).
    from data_generation.split_manifest import build_split_manifest

    n = 500
    df = pd.DataFrame([
        _full_schema_row(step=i + 1, amount=float(10 + i % 5), oldbalanceOrg=100.0, newbalanceOrig=90.0 - (i % 5), isFraud=0)
        for i in range(n)
    ])
    manifest = build_split_manifest(n, seed=1)
    with_manifest, report_with = ct.clean_dataset(df, split_manifest=manifest)
    without_manifest, report_without = ct.clean_dataset(df, split_manifest=None)
    assert {"is_amount_outlier", "is_zero_amount", "is_balance_inconsistent"}.issubset(with_manifest.columns)
    assert len(with_manifest) == len(df)
    assert len(without_manifest) == len(df)
    assert report_with["output_rows"] == report_without["output_rows"] == n


def test_clean_dataset_with_split_manifest_is_reproducible():
    from data_generation.split_manifest import build_split_manifest

    n = 300
    df = pd.DataFrame([
        _full_schema_row(step=i + 1, amount=float(10 + i % 7), oldbalanceOrg=100.0, newbalanceOrig=90.0)
        for i in range(n)
    ])
    manifest = build_split_manifest(n, seed=456)
    result_1, _ = ct.clean_dataset(df, split_manifest=manifest)
    result_2, _ = ct.clean_dataset(df, split_manifest=manifest)
    pd.testing.assert_series_equal(result_1["is_amount_outlier"], result_2["is_amount_outlier"])


def test_clean_dataset_split_manifest_fences_fit_only_on_train_rows():
    # Direct verification that the manifest-driven path actually restricts
    # fitting to train rows, not the full dataset: construct a case where the
    # train subset and full dataset would produce different Tukey fences,
    # then confirm the flag matches the train-only fence.
    from data_generation.split_manifest import build_split_manifest, train_mask_for_row_indices

    n = 200
    df = pd.DataFrame([
        _full_schema_row(step=i + 1, amount=float(10 + (i % 5)), oldbalanceOrg=100.0, newbalanceOrig=90.0)
        for i in range(n)
    ])
    # Inject one extreme value into what will be a non-train row, so it
    # would widen the fence if (incorrectly) included in fitting.
    manifest = build_split_manifest(n, seed=9)
    train_mask = train_mask_for_row_indices(manifest, pd.RangeIndex(n))
    non_train_positions = [i for i in range(n) if not train_mask[i]]
    df.loc[non_train_positions[0], "amount"] = 999_999.0

    cleaned, _ = ct.clean_dataset(df, split_manifest=manifest)
    expected_fences = ct.fit_tukey_fences(df.loc[train_mask, "amount"])
    expected_flags = ct.apply_tukey_fences(cleaned["amount"], expected_fences)
    pd.testing.assert_series_equal(cleaned["is_amount_outlier"], expected_flags, check_names=False)


def test_clean_dataset_raises_when_row_indices_are_not_covered_by_manifest():
    # A manifest built for a smaller/different dataset that doesn't cover
    # this df's row_index range must raise, not silently misalign.
    from data_generation.split_manifest import build_split_manifest

    df = pd.DataFrame([_full_schema_row(step=i + 1) for i in range(50)])
    manifest = build_split_manifest(10)  # only covers row_index 0..9
    with pytest.raises(ValueError):
        ct.clean_dataset(df, split_manifest=manifest)
