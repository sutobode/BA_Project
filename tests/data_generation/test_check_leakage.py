import numpy as np
import pandas as pd
import pytest

from data_generation import check_leakage as cl


def test_univariate_auc_perfect_separation_is_one():
    feature = pd.Series([0.0] * 50 + [1.0] * 50)
    label = pd.Series([0] * 50 + [1] * 50)
    assert cl.univariate_auc(feature, label) == pytest.approx(1.0)


def test_univariate_auc_inverse_separation_is_still_flagged_as_high():
    feature = pd.Series([1.0] * 50 + [0.0] * 50)
    label = pd.Series([0] * 50 + [1] * 50)
    assert cl.univariate_auc(feature, label) == pytest.approx(1.0)


def test_univariate_auc_no_separation_is_near_half():
    rng = np.random.default_rng(0)
    feature = pd.Series(rng.random(10_000))
    label = pd.Series(rng.integers(0, 2, size=10_000))
    assert cl.univariate_auc(feature, label) == pytest.approx(0.5, abs=0.02)


def test_cramers_v_perfect_association_is_one():
    feature = pd.Series(["A"] * 50 + ["B"] * 50)
    label = pd.Series([0] * 50 + [1] * 50)
    assert cl.cramers_v(feature, label) == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_independent_association_is_near_zero():
    rng = np.random.default_rng(0)
    feature = pd.Series(rng.choice(["A", "B", "C"], size=10_000))
    label = pd.Series(rng.integers(0, 2, size=10_000))
    assert cl.cramers_v(feature, label) < 0.05
