"""
Shared train/validation/test split manifest.

Team decision: a SINGLE 60/20/20 split, computed once from the raw PaySim
row count, saved to disk, and reused by every stage that needs a split -
Member 2's amount_percentile_reference fit (generate_synthetic_fields.py),
Member 2's Tukey fence fit (clean_transactions.py), and Member 5's model
training/evaluation. This replaces the earlier design where
generate_synthetic_fields.py and clean_transactions.py each drew their own
independent 70/30 split with no guarantee of alignment between them or with
whatever Model Development would later use.

The manifest is a row-index -> split-label mapping, not a column merged
into transactions_cleaned.parquet: the final dataset keeps every row and no
split column, exactly as it did before. Any stage that needs the split reads
this file separately and aligns it by row position with whichever dataframe
it currently holds (the raw CSV, the synthetic parquet, or the cleaned
parquet), all of which preserve the raw file's row order and row count
throughout this pipeline (0 rows removed by cleaning on the real dataset -
see README section 15).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_MANIFEST_PATH = "data/processed/split_manifest.parquet"

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20
SPLIT_SEED = 2024

SPLIT_VALUES = ("train", "val", "test")


def build_split_manifest(
    n: int,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Builds a fresh 60/20/20 (by default) train/val/test assignment for n
    rows, indexed 0..n-1 in the same row order as the raw PaySim CSV.

    A single numpy.random.Generator(seed) draws one shuffled permutation of
    row indices and slices it into three contiguous blocks - this guarantees
    the three fractions partition all n rows exactly once each (no overlap,
    no gaps), which a per-row independent random draw (e.g. three separate
    Bernoulli calls) would not guarantee at the boundaries.

    test_fraction is implied as 1 - train_fraction - val_fraction, so the
    three fractions always sum to exactly 1 by construction.
    """
    if not (0 < train_fraction < 1) or not (0 < val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must each be in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1 (test gets the remainder)")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)

    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))

    split = np.empty(n, dtype=object)
    split[shuffled[:n_train]] = "train"
    split[shuffled[n_train:n_train + n_val]] = "val"
    split[shuffled[n_train + n_val:]] = "test"

    return pd.DataFrame({"row_index": np.arange(n), "split": pd.Categorical(split, categories=SPLIT_VALUES)})


def save_split_manifest(manifest: pd.DataFrame, path: str = SPLIT_MANIFEST_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)


def load_split_manifest(path: str = SPLIT_MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_or_create_split_manifest(n: int, path: str = SPLIT_MANIFEST_PATH) -> pd.DataFrame:
    """The canonical entry point every stage should call: reuse the manifest
    already on disk if one exists and matches n rows, otherwise build and
    save a fresh one. This is what makes the split genuinely SHARED across
    separate script runs (generate_synthetic_fields.py, clean_transactions.py,
    and eventually Model Development) rather than each independently
    recomputing its own - once created, the manifest file is the single
    source of truth.

    Raises if an existing manifest's row count doesn't match n, rather than
    silently rebuilding it - a mismatch means the underlying dataset changed
    (e.g. rows were added/removed upstream) and the manifest must be
    regenerated deliberately, not overwritten unnoticed.
    """
    manifest_path = Path(path)
    if manifest_path.exists():
        existing = load_split_manifest(path)
        if len(existing) != n:
            raise ValueError(
                f"Existing split manifest at {path} has {len(existing)} rows, "
                f"but the current dataset has {n} rows. Delete or rebuild the "
                f"manifest deliberately (do not silently overwrite it) before proceeding."
            )
        return existing
    manifest = build_split_manifest(n)
    save_split_manifest(manifest, path)
    return manifest


def train_row_mask(manifest: pd.DataFrame, n: int) -> np.ndarray:
    """Boolean array of length n, True for rows assigned "train" in the given
    manifest, aligned by row_index. Raises if the manifest doesn't cover
    exactly rows 0..n-1, so a caller never silently gets a mask that's
    misaligned with the dataframe it's about to index."""
    if len(manifest) != n or set(manifest["row_index"]) != set(range(n)):
        raise ValueError(f"Split manifest does not cover exactly rows 0..{n - 1}; cannot build an aligned mask.")
    ordered = manifest.sort_values("row_index")
    return (ordered["split"] == "train").to_numpy()


def train_mask_for_row_indices(manifest: pd.DataFrame, row_indices) -> np.ndarray:
    """Boolean array aligned with the given sequence of original row_index
    values (not necessarily 0..n-1, and not necessarily contiguous), True
    where the manifest labels that row_index "train".

    This is what a stage that has REMOVED some rows (e.g.
    clean_transactions.py's three structural checks) should use: after
    removal, the surviving rows' original row_index values are no longer a
    contiguous 0..n-1 range, so train_row_mask()'s strict full-coverage
    requirement would reject them. This function instead looks up each
    given row_index individually, so it works correctly whether zero rows
    were removed (the real dataset's current state) or some were (a
    different input where the structural checks actually trigger).

    Raises if any row_index in the input is not present in the manifest -
    that would mean the manifest is stale relative to the dataset it's
    being applied to (e.g. built for a different row count or a dataset
    that has since changed), and silently treating an unknown row_index as
    non-train would be a bug, not a safe default.
    """
    manifest_by_index = manifest.set_index("row_index")["split"]
    row_indices = pd.Index(row_indices)
    missing = row_indices.difference(manifest_by_index.index)
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} row_index value(s) not found in the split manifest "
            f"(e.g. {list(missing[:5])}) - the manifest may be stale relative to this dataset."
        )
    return (manifest_by_index.loc[row_indices] == "train").to_numpy()
