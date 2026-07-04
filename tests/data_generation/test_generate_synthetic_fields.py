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
