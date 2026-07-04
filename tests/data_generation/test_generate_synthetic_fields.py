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


def test_generate_conditional_bernoulli_matches_base_and_fraud_rates():
    rng = np.random.default_rng(0)
    is_fraud = np.array([1] * 50_000 + [0] * 50_000)
    result = gsf.generate_conditional_bernoulli(is_fraud, base_p=0.04, fraud_p=0.12, rng=rng)
    fraud_rate = result[:50_000].mean()
    base_rate = result[50_000:].mean()
    assert fraud_rate == pytest.approx(0.12, abs=0.01)
    assert base_rate == pytest.approx(0.04, abs=0.01)


def test_generate_new_device_flag_base_rate_matches_spec():
    rng = np.random.default_rng(1)
    is_fraud = np.zeros(50_000, dtype=int)
    result = gsf.generate_new_device_flag(is_fraud, rng)
    assert result.mean() == pytest.approx(0.04, abs=0.01)


def test_generate_new_device_flag_fraud_rate_matches_spec():
    rng = np.random.default_rng(1)
    is_fraud = np.ones(50_000, dtype=int)
    result = gsf.generate_new_device_flag(is_fraud, rng)
    assert result.mean() == pytest.approx(0.12, abs=0.01)


def test_generate_account_age_days_base_median_matches_spec():
    rng = np.random.default_rng(0)
    is_fraud = np.zeros(100_000, dtype=int)
    result = gsf.generate_account_age_days(is_fraud, rng)
    assert np.median(result) == pytest.approx(400, rel=0.1)


def test_generate_account_age_days_fraud_median_lower_than_base():
    rng = np.random.default_rng(0)
    is_fraud = np.ones(100_000, dtype=int)
    result = gsf.generate_account_age_days(is_fraud, rng)
    assert np.median(result) == pytest.approx(150, rel=0.15)


def test_generate_account_age_days_respects_bounds():
    rng = np.random.default_rng(0)
    is_fraud = np.random.default_rng(1).integers(0, 2, size=10_000)
    result = gsf.generate_account_age_days(is_fraud, rng)
    assert result.min() >= 1
    assert result.max() <= 3650


def test_generate_billing_country_only_returns_known_countries():
    from data_generation.country_centroids import COUNTRY_LIST
    rng = np.random.default_rng(0)
    result = gsf.generate_billing_country(5000, rng)
    assert set(result).issubset(set(COUNTRY_LIST))


def test_generate_ip_country_match_rate_matches_base_probability():
    rng = np.random.default_rng(0)
    n = 100_000
    billing = gsf.generate_billing_country(n, rng)
    is_fraud = np.zeros(n, dtype=int)
    ip = gsf.generate_ip_country(billing, is_fraud, rng)
    match_rate = (ip == billing).mean()
    assert match_rate == pytest.approx(0.93, abs=0.01)


def test_generate_ip_country_fraud_rows_have_lower_match_rate():
    rng = np.random.default_rng(0)
    n = 100_000
    billing = gsf.generate_billing_country(n, rng)
    is_fraud = np.ones(n, dtype=int)
    ip = gsf.generate_ip_country(billing, is_fraud, rng)
    match_rate = (ip == billing).mean()
    assert match_rate == pytest.approx(0.80, abs=0.01)


def test_generate_ip_country_returns_only_known_countries():
    from data_generation.country_centroids import COUNTRY_LIST
    rng = np.random.default_rng(0)
    n = 5000
    billing = gsf.generate_billing_country(n, rng)
    is_fraud = rng.integers(0, 2, size=n)
    ip = gsf.generate_ip_country(billing, is_fraud, rng)
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
    base_rate = gsf.generate_shipping_billing_mismatch(np.zeros(n, dtype=int), rng).mean()
    fraud_rate = gsf.generate_shipping_billing_mismatch(np.ones(n, dtype=int), rng).mean()
    assert base_rate == pytest.approx(0.05, abs=0.01)
    assert fraud_rate == pytest.approx(0.15, abs=0.01)


def test_generate_failed_payment_attempts_24h_means_match_spec():
    rng = np.random.default_rng(3)
    n = 50_000
    base_mean = gsf.generate_failed_payment_attempts_24h(np.zeros(n, dtype=int), rng).mean()
    fraud_mean = gsf.generate_failed_payment_attempts_24h(np.ones(n, dtype=int), rng).mean()
    assert base_mean == pytest.approx(0.15, abs=0.02)
    assert fraud_mean == pytest.approx(0.6, abs=0.03)


def test_generate_failed_payment_attempts_24h_non_negative():
    rng = np.random.default_rng(4)
    is_fraud = rng.integers(0, 2, size=1000)
    result = gsf.generate_failed_payment_attempts_24h(is_fraud, rng)
    assert (result >= 0).all()


REQUIRED_SYNTHETIC_COLUMNS = {
    "hour_of_day", "is_night_transaction", "customer_account_age_days",
    "device_id", "browser", "device_type", "new_device_flag",
    "billing_country", "ip_country", "ip_billing_distance_km",
    "shipping_billing_mismatch", "failed_payment_attempts_24h",
}


def test_generate_all_synthetic_fields_adds_all_required_columns():
    df = pd.DataFrame({"step": np.arange(1, 201), "isFraud": [0] * 190 + [1] * 10})
    result = gsf.generate_all_synthetic_fields(df, seed=42)
    assert REQUIRED_SYNTHETIC_COLUMNS.issubset(result.columns)
    assert len(result) == len(df)


def test_generate_all_synthetic_fields_is_reproducible_with_same_seed():
    df = pd.DataFrame({"step": np.arange(1, 101), "isFraud": [0] * 95 + [1] * 5})
    r1 = gsf.generate_all_synthetic_fields(df, seed=42)
    r2 = gsf.generate_all_synthetic_fields(df, seed=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_generate_all_synthetic_fields_preserves_original_columns_and_row_count():
    df = pd.DataFrame({"step": [1, 2, 3], "isFraud": [0, 0, 1], "amount": [10.0, 20.0, 30.0]})
    result = gsf.generate_all_synthetic_fields(df, seed=1)
    assert list(result["amount"]) == [10.0, 20.0, 30.0]
    assert len(result) == 3


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
