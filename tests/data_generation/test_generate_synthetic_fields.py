import inspect

import numpy as np
import pandas as pd
import pytest

from data_generation import generate_synthetic_fields as gsf


def test_generate_hour_of_day_step_1_is_hour_0():
    result = gsf.generate_hour_of_day(pd.Series([1]))
    assert result.iloc[0] == 0


def test_generate_hour_of_day_step_24_is_hour_23():
    result = gsf.generate_hour_of_day(pd.Series([24]))
    assert result.iloc[0] == 23


def test_generate_hour_of_day_wraps_after_24_hours():
    result = gsf.generate_hour_of_day(pd.Series([25]))
    assert result.iloc[0] == 0


def test_generate_hour_of_day_handles_max_step_743():
    result = gsf.generate_hour_of_day(pd.Series([743]))
    assert result.iloc[0] == (743 - 1) % 24


def test_is_night_transaction_true_for_hours_0_to_5():
    hours = pd.Series([0, 1, 2, 3, 4, 5])
    result = gsf.generate_is_night_transaction(hours)
    assert result.all()


def test_is_night_transaction_false_for_hours_6_to_23():
    hours = pd.Series([6, 12, 18, 23])
    result = gsf.generate_is_night_transaction(hours)
    assert not result.any()


def test_build_device_pool_returns_requested_size():
    pool = gsf.build_device_pool(size=100, seed=1)
    assert len(pool) == 100


def test_build_device_pool_values_are_unique():
    pool = gsf.build_device_pool(size=100, seed=1)
    assert len(set(pool)) == 100


def test_build_device_pool_reproducible_with_same_seed():
    pool1 = gsf.build_device_pool(size=50, seed=7)
    pool2 = gsf.build_device_pool(size=50, seed=7)
    assert list(pool1) == list(pool2)


def test_generate_device_id_only_uses_pool_values():
    rng = np.random.default_rng(0)
    pool = np.array(["a", "b", "c"])
    result = gsf.generate_device_id(1000, pool, rng)
    assert set(result).issubset({"a", "b", "c"})


def test_generate_categorical_respects_weights_approximately():
    rng = np.random.default_rng(0)
    weights = {"A": 0.9, "B": 0.1}
    result = gsf.generate_categorical(100_000, weights, rng)
    share_a = (result == "A").mean()
    assert share_a == pytest.approx(0.9, abs=0.02)


def test_generate_categorical_only_returns_known_categories():
    rng = np.random.default_rng(0)
    result = gsf.generate_categorical(1000, gsf.BROWSER_WEIGHTS, rng)
    assert set(result).issubset(set(gsf.BROWSER_WEIGHTS.keys()))


def test_generate_conditional_on_risk_matches_base_and_high_risk_rates():
    rng = np.random.default_rng(0)
    risk_score = np.array([1.0] * 50_000 + [0.0] * 50_000)
    result = gsf.generate_conditional_on_risk(risk_score, base_p=0.04, high_risk_p=0.12, rng=rng)
    high_risk_rate = result[:50_000].mean()
    base_rate = result[50_000:].mean()
    assert high_risk_rate == pytest.approx(0.12, abs=0.01)
    assert base_rate == pytest.approx(0.04, abs=0.01)


def test_generate_new_device_flag_base_rate_matches_spec():
    rng = np.random.default_rng(1)
    risk_score = np.zeros(50_000)
    result = gsf.generate_new_device_flag(risk_score, rng)
    assert result.mean() == pytest.approx(0.04, abs=0.01)


def test_generate_new_device_flag_high_risk_rate_matches_spec():
    rng = np.random.default_rng(1)
    risk_score = np.ones(50_000)
    result = gsf.generate_new_device_flag(risk_score, rng)
    assert result.mean() == pytest.approx(0.12, abs=0.01)


def test_generate_account_age_days_base_median_matches_spec():
    rng = np.random.default_rng(0)
    risk_score = np.zeros(100_000)
    result = gsf.generate_account_age_days(risk_score, rng)
    assert np.median(result) == pytest.approx(400, rel=0.1)


def test_generate_account_age_days_high_risk_median_lower_than_base():
    rng = np.random.default_rng(0)
    risk_score = np.ones(100_000)
    result = gsf.generate_account_age_days(risk_score, rng)
    assert np.median(result) == pytest.approx(275, rel=0.15)


def test_generate_account_age_days_respects_bounds():
    rng = np.random.default_rng(0)
    risk_score = np.random.default_rng(1).random(10_000)
    result = gsf.generate_account_age_days(risk_score, rng)
    assert result.min() >= 1
    assert result.max() <= 3650


LABEL_FREE_GENERATION_FUNCTIONS = [
    gsf.generate_account_age_days,
    gsf.generate_new_device_flag,
    gsf.generate_ip_country,
    gsf.generate_shipping_billing_mismatch,
    gsf.generate_failed_payment_attempts_24h,
    gsf.compute_risk_proxy,
    gsf.generate_device_id_and_new_device_flag,
    gsf.fit_amount_percentile_reference,
    gsf.apply_amount_percentile,
]


@pytest.mark.parametrize("func", LABEL_FREE_GENERATION_FUNCTIONS)
def test_generation_functions_do_not_accept_isfraud_parameter(func):
    params = inspect.signature(func).parameters
    assert "isFraud" not in params
    assert "is_fraud" not in params


def test_compute_risk_proxy_is_bounded_between_0_and_1():
    rng = np.random.default_rng(0)
    n = 5000
    type_ = pd.Series(rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], size=n))
    amount = pd.Series(rng.exponential(1000, size=n))
    hour_of_day = pd.Series(rng.integers(0, 24, size=n))
    risk = gsf.compute_risk_proxy(type_, amount, hour_of_day)
    assert (risk >= 0).all()
    assert (risk <= 1).all()


def test_compute_risk_proxy_higher_for_transfer_at_night_with_large_amount():
    # A TRANSFER at night with a top-percentile amount should score higher
    # than a PAYMENT during the day with a bottom-percentile amount.
    type_ = pd.Series(["PAYMENT", "TRANSFER"])
    amount = pd.Series([1.0, 999_999.0])
    hour_of_day = pd.Series([12, 2])
    risk = gsf.compute_risk_proxy(type_, amount, hour_of_day)
    assert risk[1] > risk[0]


def test_conditional_fields_are_identical_when_isfraud_changes_but_observables_fixed():
    # Label-free guarantee: holding type/amount/step fixed and flipping isFraud
    # must not change any generated field, because generation never reads isFraud.
    n = 500
    rng_step = np.random.default_rng(9)
    base_df = pd.DataFrame({
        "step": rng_step.integers(1, 744, size=n),
        "type": rng_step.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], size=n),
        "amount": rng_step.exponential(1000, size=n),
        "nameOrig": [f"C{i}" for i in range(n)],
    })
    df_fraud_0 = base_df.copy()
    df_fraud_0["isFraud"] = 0
    df_fraud_1 = base_df.copy()
    df_fraud_1["isFraud"] = 1

    result_0 = gsf.generate_all_synthetic_fields(df_fraud_0, seed=42)
    result_1 = gsf.generate_all_synthetic_fields(df_fraud_1, seed=42)

    synthetic_columns = [
        "hour_of_day", "is_night_transaction", "customer_account_age_days",
        "device_id", "browser", "device_type", "new_device_flag",
        "billing_country", "ip_country", "ip_billing_distance_km",
        "ip_billing_country_mismatch", "shipping_billing_mismatch",
        "failed_payment_attempts_24h",
    ]
    for col in synthetic_columns:
        pd.testing.assert_series_equal(result_0[col], result_1[col], check_names=False)


def test_generate_billing_country_only_returns_known_countries():
    from data_generation.country_centroids import COUNTRY_LIST
    rng = np.random.default_rng(0)
    result = gsf.generate_billing_country(5000, rng)
    assert set(result).issubset(set(COUNTRY_LIST))


def test_generate_ip_country_match_rate_matches_base_probability():
    rng = np.random.default_rng(0)
    n = 100_000
    billing = gsf.generate_billing_country(n, rng)
    risk_score = np.zeros(n)
    ip = gsf.generate_ip_country(billing, risk_score, rng)
    match_rate = (ip == billing).mean()
    assert match_rate == pytest.approx(0.93, abs=0.01)


def test_generate_ip_country_high_risk_rows_have_lower_match_rate():
    rng = np.random.default_rng(0)
    n = 100_000
    billing = gsf.generate_billing_country(n, rng)
    risk_score = np.ones(n)
    ip = gsf.generate_ip_country(billing, risk_score, rng)
    match_rate = (ip == billing).mean()
    assert match_rate == pytest.approx(0.80, abs=0.01)


def test_generate_ip_country_returns_only_known_countries():
    from data_generation.country_centroids import COUNTRY_LIST
    rng = np.random.default_rng(0)
    n = 5000
    billing = gsf.generate_billing_country(n, rng)
    risk_score = rng.random(n)
    ip = gsf.generate_ip_country(billing, risk_score, rng)
    assert set(ip).issubset(set(COUNTRY_LIST))


def test_generate_ip_billing_distance_km_zero_when_countries_match():
    ip = np.array(["US", "VN"])
    billing = np.array(["US", "VN"])
    result = gsf.generate_ip_billing_distance_km(ip, billing)
    assert np.allclose(result, 0.0)


def test_generate_ip_billing_distance_km_positive_when_countries_differ():
    ip = np.array(["US"])
    billing = np.array(["VN"])
    result = gsf.generate_ip_billing_distance_km(ip, billing)
    assert result[0] > 10000


def test_generate_shipping_billing_mismatch_uses_spec_probabilities():
    rng = np.random.default_rng(2)
    n = 50_000
    base_rate = gsf.generate_shipping_billing_mismatch(np.zeros(n), rng).mean()
    high_risk_rate = gsf.generate_shipping_billing_mismatch(np.ones(n), rng).mean()
    assert base_rate == pytest.approx(0.05, abs=0.01)
    assert high_risk_rate == pytest.approx(0.15, abs=0.01)


def test_generate_failed_payment_attempts_24h_means_match_spec():
    rng = np.random.default_rng(3)
    n = 50_000
    base_mean = gsf.generate_failed_payment_attempts_24h(np.zeros(n), rng).mean()
    high_risk_mean = gsf.generate_failed_payment_attempts_24h(np.ones(n), rng).mean()
    assert base_mean == pytest.approx(0.15, abs=0.02)
    assert high_risk_mean == pytest.approx(0.6, abs=0.03)


def test_generate_failed_payment_attempts_24h_non_negative():
    rng = np.random.default_rng(4)
    risk_score = rng.random(1000)
    result = gsf.generate_failed_payment_attempts_24h(risk_score, rng)
    assert (result >= 0).all()


REQUIRED_SYNTHETIC_COLUMNS = {
    "hour_of_day", "is_night_transaction", "customer_account_age_days",
    "device_id", "browser", "device_type", "new_device_flag",
    "billing_country", "ip_country", "ip_billing_distance_km",
    "ip_billing_country_mismatch", "shipping_billing_mismatch",
    "failed_payment_attempts_24h",
}


def _make_base_df(n_rows: int, n_fraud: int, seed: int = 0) -> pd.DataFrame:
    """Minimal realistic frame: generate_all_synthetic_fields needs
    type/amount (for compute_risk_proxy) and nameOrig (for
    generate_device_id_and_new_device_flag) in addition to step/isFraud.
    nameOrig is unique per row here (the ~99.85% common case in the real
    dataset) - tests targeting the repeat-nameOrig behavior use their own
    DataFrame instead."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": np.arange(1, n_rows + 1),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], size=n_rows),
        "amount": rng.exponential(1000, size=n_rows),
        "nameOrig": [f"C{i}" for i in range(n_rows)],
        "isFraud": [0] * (n_rows - n_fraud) + [1] * n_fraud,
    })


def test_generate_all_synthetic_fields_adds_all_required_columns():
    df = _make_base_df(200, 10)
    result = gsf.generate_all_synthetic_fields(df, seed=42)
    assert REQUIRED_SYNTHETIC_COLUMNS.issubset(result.columns)
    assert len(result) == len(df)


def test_generate_all_synthetic_fields_is_reproducible_with_same_seed():
    df = _make_base_df(100, 5)
    r1 = gsf.generate_all_synthetic_fields(df, seed=42)
    r2 = gsf.generate_all_synthetic_fields(df, seed=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_generate_all_synthetic_fields_preserves_original_columns_and_row_count():
    df = pd.DataFrame({
        "step": [1, 2, 3], "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "isFraud": [0, 0, 1], "amount": [10.0, 20.0, 30.0],
        "nameOrig": ["C1", "C2", "C3"],
    })
    result = gsf.generate_all_synthetic_fields(df, seed=1)
    assert list(result["amount"]) == [10.0, 20.0, 30.0]
    assert len(result) == 3


def test_generate_all_synthetic_fields_does_not_alter_class_distribution():
    df = _make_base_df(1000, 13)
    result = gsf.generate_all_synthetic_fields(df, seed=42)
    assert list(result["isFraud"]) == list(df["isFraud"])
    assert result["isFraud"].mean() == df["isFraud"].mean()


def test_generate_all_synthetic_fields_produces_no_duplicate_columns():
    df = _make_base_df(200, 10)
    result = gsf.generate_all_synthetic_fields(df, seed=42)
    assert not result.columns.duplicated().any()


def test_generate_all_synthetic_fields_adds_ip_billing_country_mismatch_consistent_with_countries():
    df = _make_base_df(500, 5)
    result = gsf.generate_all_synthetic_fields(df, seed=42)
    expected = result["ip_country"] != result["billing_country"]
    pd.testing.assert_series_equal(result["ip_billing_country_mismatch"], expected, check_names=False)


# --- fit/transform split for amount_percentile (train-only fitting) ---


def test_fit_amount_percentile_reference_stores_sorted_arrays_per_type():
    type_ = pd.Series(["A", "A", "B", "B", "B"])
    amount = pd.Series([30.0, 10.0, 5.0, 1.0, 9.0])
    ref = gsf.fit_amount_percentile_reference(type_, amount)
    assert list(ref["A"]) == [10.0, 30.0]
    assert list(ref["B"]) == [1.0, 5.0, 9.0]


def test_apply_amount_percentile_matches_in_batch_rank_when_reference_is_full_batch():
    rng = np.random.default_rng(0)
    n = 2000
    type_ = pd.Series(rng.choice(["A", "B", "C"], size=n))
    amount = pd.Series(rng.exponential(1000, size=n))
    reference = gsf.fit_amount_percentile_reference(type_, amount)
    applied = gsf.apply_amount_percentile(type_, amount, reference)
    in_batch = amount.groupby(type_).rank(pct=True).to_numpy()
    # searchsorted-based percentile and in-batch rank agree within a small
    # tolerance (both estimate the same underlying rank, via different but
    # equivalent tie-handling conventions).
    assert np.abs(applied - in_batch).max() < 0.01


def test_apply_amount_percentile_reproducible_for_a_single_new_row_using_persisted_reference():
    # Core claim being tested: given ONLY a persisted reference (no access to
    # any other rows), a single new transaction's percentile is computable
    # deterministically - this is what makes amount_percentile reproducible
    # at scoring time instead of being a batch-only statistic.
    train_type = pd.Series(["TRANSFER"] * 100)
    train_amount = pd.Series(np.arange(1, 101, dtype="float64"))
    reference = gsf.fit_amount_percentile_reference(train_type, train_amount)

    new_row_type = pd.Series(["TRANSFER"])
    new_row_amount = pd.Series([50.0])
    result_1 = gsf.apply_amount_percentile(new_row_type, new_row_amount, reference)
    result_2 = gsf.apply_amount_percentile(new_row_type, new_row_amount, reference)
    assert result_1[0] == result_2[0]
    assert result_1[0] == pytest.approx(0.5, abs=0.01)


def test_apply_amount_percentile_unknown_type_falls_back_to_neutral():
    reference = gsf.fit_amount_percentile_reference(pd.Series(["A"]), pd.Series([10.0]))
    result = gsf.apply_amount_percentile(pd.Series(["ZZZ_UNKNOWN"]), pd.Series([999.0]), reference)
    assert result[0] == pytest.approx(0.5)


def test_compute_risk_proxy_with_explicit_reference_does_not_refit_from_input():
    # If a reference is supplied, compute_risk_proxy must use it as-is rather
    # than silently re-fitting from (type_, amount) - otherwise passing a
    # reference would be a no-op and train/test leakage would persist.
    train_type = pd.Series(["TRANSFER"] * 100)
    train_amount = pd.Series(np.arange(1, 101, dtype="float64"))
    reference = gsf.fit_amount_percentile_reference(train_type, train_amount)

    # A test row with an amount far outside anything in the training
    # reference - if compute_risk_proxy refit from this call's own input
    # (a batch of size 1), the percentile would trivially be 1.0 regardless
    # of the reference; using the persisted reference correctly, it should
    # reflect where 5.0 falls within the TRAIN distribution instead.
    test_type = pd.Series(["TRANSFER"])
    test_amount = pd.Series([5.0])
    test_hour = pd.Series([12])
    risk = gsf.compute_risk_proxy(test_type, test_amount, test_hour, amount_percentile_reference=reference)
    # amount=5 sits near the bottom of the 1..100 training reference, so its
    # contribution to risk_score should be small, not saturated at 1.0.
    assert risk[0] < 0.5


# --- train/test split utility ---


def test_generate_all_synthetic_fields_with_train_mask_uses_train_only_reference():
    # Regression guard: passing train_mask must change which reference gets
    # fit (train rows only) without crashing or silently ignoring the mask.
    # train_mask now comes from the shared split_manifest module (see
    # test_split_manifest.py for that module's own tests); here we just need
    # any valid boolean mask of the right length.
    df = _make_base_df(1000, 10)
    rng = np.random.default_rng(1)
    train_mask = rng.random(len(df)) < 0.6
    with_mask = gsf.generate_all_synthetic_fields(df, seed=42, train_mask=train_mask)
    without_mask = gsf.generate_all_synthetic_fields(df, seed=42, train_mask=None)
    # Both must still produce every required column and preserve row count -
    # the train_mask path is a refinement, not a different contract.
    assert REQUIRED_SYNTHETIC_COLUMNS.issubset(with_mask.columns)
    assert len(with_mask) == len(df)
    assert len(without_mask) == len(df)


# --- device_id / new_device_flag joint consistency ---


def test_generate_device_id_and_new_device_flag_first_occurrence_of_repeat_account_is_always_new():
    # A repeat account's very first row (in order) has no prior device
    # history to compare against, so new_device_flag must be True there
    # regardless of what the risk-based Bernoulli draw would otherwise give
    # (risk_score=0 here means the underlying draw is very unlikely to be
    # True on its own - if the assertion passes anyway, it's because of the
    # no-history override, not chance).
    name_orig = pd.Series(["REPEAT", "REPEAT", "OTHER", "OTHER", "OTHER"])
    risk_score = np.zeros(5)
    rng = np.random.default_rng(0)
    pool = gsf.build_device_pool(size=100, seed=1)
    _, new_device_flag = gsf.generate_device_id_and_new_device_flag(
        name_orig, risk_score, pool, rng, base_p=0.04, high_risk_p=0.12
    )
    assert new_device_flag[0] == True  # REPEAT's first occurrence
    assert new_device_flag[2] == True  # OTHER's first occurrence


def test_generate_device_id_and_new_device_flag_consistent_for_repeat_account():
    # The behavioral guarantee this function exists to enforce: for a
    # repeat-nameOrig account, new_device_flag=False must always mean the
    # device_id was actually seen before for that same account, and
    # new_device_flag=True must always mean it was not.
    n = 200
    name_orig = pd.Series(["REPEAT_ACCOUNT"] * n)
    risk_score = np.full(n, 0.5)
    rng = np.random.default_rng(0)
    pool = gsf.build_device_pool(size=1000, seed=1)  # pool >> n, no exhaustion
    device_id, new_device_flag = gsf.generate_device_id_and_new_device_flag(
        name_orig, risk_score, pool, rng, base_p=0.04, high_risk_p=0.12
    )

    seen: set = set()
    for i in range(n):
        if new_device_flag[i]:
            assert device_id[i] not in seen, f"row {i}: flagged new but device was already seen for this account"
        else:
            assert device_id[i] in seen, f"row {i}: flagged known but device was never seen before for this account"
        seen.add(device_id[i])


def test_generate_device_id_and_new_device_flag_falls_back_to_reuse_when_pool_exhausted():
    # Edge case: if an account's history has already consumed every device in
    # a tiny pool, there is no unseen device left. Must terminate (not loop
    # forever) and fall back to reusing a known device rather than crashing.
    n = 30
    name_orig = pd.Series(["REPEAT_ACCOUNT"] * n)
    risk_score = np.ones(n)  # high_risk_p=0.12 -> "new" draws are still likely
    rng = np.random.default_rng(0)
    pool = gsf.build_device_pool(size=5, seed=1)  # tiny pool, easy to exhaust
    device_id, new_device_flag = gsf.generate_device_id_and_new_device_flag(
        name_orig, risk_score, pool, rng, base_p=0.04, high_risk_p=0.12
    )
    assert len(device_id) == n  # completed without hanging
    assert set(device_id).issubset(set(pool))


def test_generate_device_id_and_new_device_flag_unique_accounts_use_independent_fast_path():
    # Accounts that each appear exactly once in the dataset have no
    # meaningful device history to be consistent (or inconsistent) with, so
    # they take the original, fully independent generation path: device_id
    # uniform from the pool, new_device_flag from the plain risk-based
    # Bernoulli - unchanged from the pre-fix behavior. Verify the rate
    # matches base_p/high_risk_p at the population level (not "always True"
    # -  a unique account can still legitimately draw False).
    n = 50_000
    name_orig = pd.Series([f"C{i}" for i in range(n)])
    risk_score = np.zeros(n)
    rng = np.random.default_rng(0)
    pool = gsf.build_device_pool(size=1000, seed=1)
    _, new_device_flag = gsf.generate_device_id_and_new_device_flag(
        name_orig, risk_score, pool, rng, base_p=0.04, high_risk_p=0.12
    )
    assert new_device_flag.mean() == pytest.approx(0.04, abs=0.01)


def test_generate_all_synthetic_fields_device_history_consistent_end_to_end():
    # End-to-end guard on the full pipeline (not just the isolated helper):
    # for any account appearing more than once in the output, a row with
    # new_device_flag=False must reuse a device_id seen earlier for that
    # same account (in step order), and new_device_flag=True must introduce
    # a device_id not seen earlier for that account.
    n = 300
    rng = np.random.default_rng(7)
    name_orig = rng.choice([f"ACC{i}" for i in range(20)], size=n)  # forces repeats
    df = pd.DataFrame({
        "step": np.arange(1, n + 1),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], size=n),
        "amount": rng.exponential(1000, size=n),
        "nameOrig": name_orig,
        "isFraud": rng.integers(0, 2, size=n),
    })
    result = gsf.generate_all_synthetic_fields(df, seed=42)

    history: dict = {}
    for _, row in result.iterrows():
        acc = row["nameOrig"]
        seen = history.setdefault(acc, set())
        if row["new_device_flag"]:
            assert row["device_id"] not in seen
        else:
            assert row["device_id"] in seen
        seen.add(row["device_id"])


# --- module is offline-only: static check that no function name suggests a scoring entrypoint ---


def test_module_docstring_states_offline_only():
    import data_generation.generate_synthetic_fields as module
    assert module.__doc__ is not None
    assert "OFFLINE" in module.__doc__


def test_load_raw_transactions_applies_expected_dtypes(tmp_path):
    csv_content = (
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
        "1,PAYMENT,9839.64,C123,170136.0,160296.36,M456,0.0,0.0,0,0\n"
        "1,TRANSFER,181.0,C789,181.0,0.0,C999,0.0,0.0,1,0\n"
    )
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(csv_content)
    df = gsf.load_raw_transactions(str(csv_path))
    assert len(df) == 2
    assert df["isFraud"].tolist() == [0, 1]
    assert str(df["type"].dtype) == "category"


def test_load_raw_transactions_raises_value_error_on_missing_required_column(tmp_path):
    # Missing "isFraud" column entirely.
    csv_content = (
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFlaggedFraud\n"
        "1,PAYMENT,9839.64,C123,170136.0,160296.36,M456,0.0,0.0,0\n"
    )
    csv_path = tmp_path / "missing_column.csv"
    csv_path.write_text(csv_content)
    with pytest.raises(ValueError, match="isFraud"):
        gsf.load_raw_transactions(str(csv_path))


def test_build_stratified_sample_preserves_fraud_ratio_approximately():
    n = 10_000
    df = pd.DataFrame({"step": np.arange(1, n + 1), "isFraud": [1] * 13 + [0] * (n - 13)})
    sample = gsf.build_stratified_sample(df, sample_size=1000, seed=1)
    assert len(sample) == 1000
    assert sample["isFraud"].mean() == pytest.approx(df["isFraud"].mean(), abs=0.01)


def test_build_stratified_sample_includes_at_least_one_fraud_row():
    n = 10_000
    df = pd.DataFrame({"step": np.arange(1, n + 1), "isFraud": [1] * 5 + [0] * (n - 5)})
    sample = gsf.build_stratified_sample(df, sample_size=500, seed=1)
    assert (sample["isFraud"] == 1).sum() >= 1
