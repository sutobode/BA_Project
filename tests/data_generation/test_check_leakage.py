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


def test_check_all_fields_flags_perfectly_separating_numeric_fields():
    n = 1000
    label = pd.Series([0] * 500 + [1] * 500)
    df = pd.DataFrame({"isFraud": label})
    for meta in cl.FIELD_METADATA:
        if meta["metric_type"] == "auc":
            df[meta["field_name"]] = [0] * 500 + [1] * 500
        else:
            df[meta["field_name"]] = ["X"] * 1000
    report = cl.check_all_fields(df)
    auc_rows = report[report["metric_type"] == "auc"]
    assert (auc_rows["status"] == "FAIL").all()


def test_check_all_fields_passes_fields_with_no_association():
    n = 1000
    rng = np.random.default_rng(0)
    label = pd.Series(rng.integers(0, 2, size=n))
    df = pd.DataFrame({"isFraud": label})
    for meta in cl.FIELD_METADATA:
        if meta["metric_type"] == "auc":
            df[meta["field_name"]] = rng.random(n)
        else:
            df[meta["field_name"]] = rng.choice(["A", "B", "C"], size=n)
    report = cl.check_all_fields(df)
    assert (report["status"] == "PASS").all()


def test_build_data_dictionary_markdown_contains_all_fields():
    report = pd.DataFrame([
        {"field_name": m["field_name"], "metric_type": m["metric_type"], "metric_value": 0.1, "status": "PASS"}
        for m in cl.FIELD_METADATA
    ])
    markdown = cl.build_data_dictionary_markdown(report)
    for meta in cl.FIELD_METADATA:
        assert f"`{meta['field_name']}`" in markdown
    assert markdown.startswith("# Data Dictionary")


def test_build_data_dictionary_markdown_labels_metric_with_type():
    report = pd.DataFrame([
        {"field_name": m["field_name"], "metric_type": m["metric_type"], "metric_value": 0.1, "status": "PASS"}
        for m in cl.FIELD_METADATA
    ])
    markdown = cl.build_data_dictionary_markdown(report)
    # Check that AUC-type fields have (AUC) label
    auc_fields = [m["field_name"] for m in cl.FIELD_METADATA if m["metric_type"] == "auc"]
    for field_name in auc_fields:
        assert f"`{field_name}`" in markdown
        # Find the row for this field and verify it contains (AUC)
        lines = markdown.split("\n")
        field_line = [l for l in lines if f"`{field_name}`" in l][0]
        assert "0.1 (AUC)" in field_line, f"Field {field_name} should have (AUC) label"

    # Check that Cramér's V-type fields have (Cramér's V) label
    cramers_fields = [m["field_name"] for m in cl.FIELD_METADATA if m["metric_type"] == "cramers_v"]
    for field_name in cramers_fields:
        assert f"`{field_name}`" in markdown
        # Find the row for this field and verify it contains (Cramér's V)
        lines = markdown.split("\n")
        field_line = [l for l in lines if f"`{field_name}`" in l][0]
        assert "0.1 (Cramér's V)" in field_line, f"Field {field_name} should have (Cramér's V) label"
