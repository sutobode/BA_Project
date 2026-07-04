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
