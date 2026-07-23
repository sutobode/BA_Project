import numpy as np
import pandas as pd
import pytest

from data_generation import split_manifest as sm


def test_build_split_manifest_covers_every_row_exactly_once():
    manifest = sm.build_split_manifest(1000)
    assert len(manifest) == 1000
    assert sorted(manifest["row_index"]) == list(range(1000))
    assert manifest["split"].isin(sm.SPLIT_VALUES).all()


def test_build_split_manifest_approximately_matches_60_20_20():
    manifest = sm.build_split_manifest(100_000)
    counts = manifest["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.60, abs=0.01)
    assert counts["val"] == pytest.approx(0.20, abs=0.01)
    assert counts["test"] == pytest.approx(0.20, abs=0.01)


def test_build_split_manifest_reproducible_with_same_seed():
    m1 = sm.build_split_manifest(5000, seed=7)
    m2 = sm.build_split_manifest(5000, seed=7)
    pd.testing.assert_frame_equal(
        m1.sort_values("row_index").reset_index(drop=True),
        m2.sort_values("row_index").reset_index(drop=True),
    )


def test_build_split_manifest_rejects_fractions_summing_to_one_or_more():
    with pytest.raises(ValueError):
        sm.build_split_manifest(100, train_fraction=0.7, val_fraction=0.3)


def test_build_split_manifest_custom_fractions():
    manifest = sm.build_split_manifest(100_000, train_fraction=0.7, val_fraction=0.15)
    counts = manifest["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.70, abs=0.01)
    assert counts["val"] == pytest.approx(0.15, abs=0.01)
    assert counts["test"] == pytest.approx(0.15, abs=0.01)


def test_save_and_load_split_manifest_roundtrip(tmp_path):
    path = str(tmp_path / "manifest.parquet")
    manifest = sm.build_split_manifest(500, seed=1)
    sm.save_split_manifest(manifest, path)
    loaded = sm.load_split_manifest(path)
    pd.testing.assert_frame_equal(
        manifest.reset_index(drop=True), loaded.reset_index(drop=True), check_dtype=False
    )


def test_get_or_create_split_manifest_creates_when_missing(tmp_path):
    path = str(tmp_path / "manifest.parquet")
    assert not (tmp_path / "manifest.parquet").exists()
    manifest = sm.get_or_create_split_manifest(300, path)
    assert len(manifest) == 300
    assert (tmp_path / "manifest.parquet").exists()


def test_get_or_create_split_manifest_reuses_existing_file(tmp_path):
    path = str(tmp_path / "manifest.parquet")
    first = sm.get_or_create_split_manifest(300, path)
    second = sm.get_or_create_split_manifest(300, path)
    # Same file reused, not regenerated - identical row-to-split assignment.
    pd.testing.assert_frame_equal(
        first.sort_values("row_index").reset_index(drop=True),
        second.sort_values("row_index").reset_index(drop=True),
    )


def test_get_or_create_split_manifest_raises_on_row_count_mismatch(tmp_path):
    path = str(tmp_path / "manifest.parquet")
    sm.get_or_create_split_manifest(300, path)
    with pytest.raises(ValueError):
        sm.get_or_create_split_manifest(301, path)


def test_train_row_mask_matches_manifest_train_rows():
    manifest = sm.build_split_manifest(1000, seed=3)
    mask = sm.train_row_mask(manifest, 1000)
    assert mask.dtype == bool
    assert mask.sum() == (manifest["split"] == "train").sum()
    # Spot-check alignment: every row_index marked True in the mask must be
    # labeled "train" in the manifest, and vice versa.
    ordered = manifest.sort_values("row_index").reset_index(drop=True)
    assert list(mask) == list(ordered["split"] == "train")


def test_train_row_mask_raises_on_row_count_mismatch():
    manifest = sm.build_split_manifest(100)
    with pytest.raises(ValueError):
        sm.train_row_mask(manifest, 50)
