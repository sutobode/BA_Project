# Data Cleaning Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, test, and validate a data-cleaning module that checks `transactions_synthetic.parquet` for missing values, duplicates, invalid categories, and outliers, producing `transactions_cleaned.parquet` and `docs/CLEANING_REPORT.md`.

**Architecture:** Two small modules mirroring `src/data_generation/`'s split: `clean_transactions.py` (pure, vectorized check/flag/dedupe functions + orchestrator + CLI) and `cleaning_report.py` (markdown report builder + CLI). Every check function takes and returns plain pandas objects so it is unit-testable with small hand-built DataFrames — the real 6.36M-row parquet is only touched in the final integration task.

**Tech Stack:** Python 3.13, pandas, numpy (already installed in `.venv/`). No new dependencies.

## Global Constraints

- **Flag, don't remove**, any anomaly that could be fraud-relevant: zero-amount transactions, amount outliers, balance inconsistency. Verified reason (design spec section 2): all 16 real zero-amount rows are `isFraud=1`; large amounts can themselves be fraud signal; the 80.45% balance-inconsistency rate is a known PaySim data characteristic, not an error.
- **Only remove rows** for genuine structural corruption: missing values in critical columns (`step`, `type`, `amount`, `isFraud`), full-row duplicates, and category values outside the known valid sets.
- Every function must be vectorized (pandas/numpy) — no per-row Python loops over the 6.36M-row dataset.
- Invalid-category checks for synthetic categorical fields must validate against the exact value sets already used to generate them (`BROWSER_WEIGHTS`, `DEVICE_TYPE_WEIGHTS` from `src/data_generation/generate_synthetic_fields.py`; `COUNTRY_WEIGHTS` from `src/data_generation/country_centroids.py`) — do not hardcode a second, possibly-drifting copy of these lists.
- Outlier threshold for `amount`: Tukey IQR fence, `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` — this exact method, not a different one, per the approved design spec.
- Balance-inconsistency tolerance: `abs(oldbalanceOrg - amount - newbalanceOrig) > 0.01`.
- Do NOT create `tests/data_cleaning/__init__.py` — a prior module (`tests/data_generation/`) hit a pytest package-shadowing bug from exactly this file existing alongside a `pythonpath = src`-resolved package of the same name (`data_cleaning` here). If you see `ImportError: cannot import name 'X' from 'data_cleaning'` during any task, check for and remove a stray `tests/data_cleaning/__init__.py` before doing anything else.
- Invoke the venv's Python directly by path in every command — `.venv/Scripts/python.exe` — do not rely on shell activation.
- Deliverable paths: `src/data_cleaning/clean_transactions.py`, `src/data_cleaning/cleaning_report.py`, `data/processed/transactions_cleaned.parquet`, `data/processed/transactions_cleaned_sample.csv`, `docs/CLEANING_REPORT.md`.

---

## File Structure

- `src/data_cleaning/__init__.py` — empty package marker
- `src/data_cleaning/clean_transactions.py` — `check_missing_critical`, `dedupe_exact`, `check_invalid_categories`, `flag_amount_outliers`, `flag_zero_amount`, `flag_balance_inconsistency`, orchestrator `clean_dataset`, I/O constants, `main()`
- `src/data_cleaning/cleaning_report.py` — `build_cleaning_report_markdown`, `main()`
- `tests/data_cleaning/test_clean_transactions.py`
- `tests/data_cleaning/test_cleaning_report.py`

---

### Task 1: `check_missing_critical`

**Files:**
- Create: `src/data_cleaning/__init__.py`
- Create: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Produces: `CRITICAL_COLUMNS: list[str]`, `check_missing_critical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], int]` — returns `(cleaned_df, per_column_na_counts, rows_removed)`.

- [ ] **Step 1: Create the package marker and test scaffolding**

`src/data_cleaning/__init__.py`:
```python
```

- [ ] **Step 2: Write the failing tests**

`tests/data_cleaning/test_clean_transactions.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_cleaning.clean_transactions'`

- [ ] **Step 4: Implement**

`src/data_cleaning/clean_transactions.py`:
```python
import pandas as pd

CRITICAL_COLUMNS = ["step", "type", "amount", "isFraud"]


def check_missing_critical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_column_na_counts = df.isna().sum().to_dict()
    critical_na_mask = df[CRITICAL_COLUMNS].isna().any(axis=1)
    n_removed = int(critical_na_mask.sum())
    cleaned = df[~critical_na_mask].reset_index(drop=True)
    return cleaned, per_column_na_counts, n_removed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/data_cleaning/__init__.py src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add check_missing_critical for data cleaning module"
```

---

### Task 2: `dedupe_exact`

**Files:**
- Modify: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Produces: `dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]` — returns `(deduped_df, rows_removed)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_cleaning/test_clean_transactions.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `AttributeError: module 'data_cleaning.clean_transactions' has no attribute 'dedupe_exact'`

- [ ] **Step 3: Implement**

Append to `src/data_cleaning/clean_transactions.py`:
```python
def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(deduped)
    return deduped, n_removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add dedupe_exact for data cleaning module"
```

---

### Task 3: `check_invalid_categories`

**Files:**
- Modify: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Consumes: `BROWSER_WEIGHTS`, `DEVICE_TYPE_WEIGHTS` from `data_generation.generate_synthetic_fields`; `COUNTRY_WEIGHTS` from `data_generation.country_centroids` (both already exist in the codebase, unchanged by this task).
- Produces: `VALID_TRANSACTION_TYPES: set[str]`, `VALID_BROWSERS: set[str]`, `VALID_DEVICE_TYPES: set[str]`, `VALID_COUNTRIES: set[str]`, `BOOLEAN_COLUMNS: list[str]`, `check_invalid_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], int]` — returns `(cleaned_df, per_check_invalid_counts, rows_removed)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_cleaning/test_clean_transactions.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `AttributeError: module 'data_cleaning.clean_transactions' has no attribute 'check_invalid_categories'`

- [ ] **Step 3: Implement**

Append to `src/data_cleaning/clean_transactions.py`:
```python
from data_generation.generate_synthetic_fields import BROWSER_WEIGHTS, DEVICE_TYPE_WEIGHTS
from data_generation.country_centroids import COUNTRY_WEIGHTS

VALID_TRANSACTION_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}
VALID_BROWSERS = set(BROWSER_WEIGHTS.keys())
VALID_DEVICE_TYPES = set(DEVICE_TYPE_WEIGHTS.keys())
VALID_COUNTRIES = set(COUNTRY_WEIGHTS.keys())
BOOLEAN_COLUMNS = ["is_night_transaction", "new_device_flag", "shipping_billing_mismatch"]


def check_invalid_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    per_check_invalid_counts = {}
    invalid_mask = pd.Series(False, index=df.index)

    category_checks = [
        ("type", VALID_TRANSACTION_TYPES),
        ("browser", VALID_BROWSERS),
        ("device_type", VALID_DEVICE_TYPES),
        ("billing_country", VALID_COUNTRIES),
        ("ip_country", VALID_COUNTRIES),
    ]
    for column, valid_values in category_checks:
        column_invalid = ~df[column].isin(valid_values)
        per_check_invalid_counts[column] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    for column in BOOLEAN_COLUMNS:
        column_invalid = ~df[column].isin([True, False])
        per_check_invalid_counts[column] = int(column_invalid.sum())
        invalid_mask = invalid_mask | column_invalid

    n_removed = int(invalid_mask.sum())
    cleaned = df[~invalid_mask].reset_index(drop=True)
    return cleaned, per_check_invalid_counts, n_removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add check_invalid_categories for data cleaning module"
```

---

### Task 4: `flag_amount_outliers`

**Files:**
- Modify: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Produces: `flag_amount_outliers(amount: pd.Series) -> pd.Series` (bool Series).

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_cleaning/test_clean_transactions.py`:
```python
def test_flag_amount_outliers_flags_values_beyond_tukey_fence():
    amounts = pd.Series([10.0, 12.0, 11.0, 13.0, 10.0, 12.0, 5000.0])
    result = ct.flag_amount_outliers(amounts)
    assert result.iloc[-1] == True
    assert not result.iloc[:-1].any()


def test_flag_amount_outliers_flags_nothing_for_tight_distribution():
    amounts = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8])
    result = ct.flag_amount_outliers(amounts)
    assert not result.any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `AttributeError: module 'data_cleaning.clean_transactions' has no attribute 'flag_amount_outliers'`

- [ ] **Step 3: Implement**

Append to `src/data_cleaning/clean_transactions.py`:
```python
def flag_amount_outliers(amount: pd.Series) -> pd.Series:
    """Tukey IQR fence: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
    Flagged, not removed - a large transaction amount can itself be a fraud signal.
    """
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (amount < lower) | (amount > upper)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add flag_amount_outliers (Tukey IQR) for data cleaning module"
```

---

### Task 5: `flag_zero_amount` and `flag_balance_inconsistency`

**Files:**
- Modify: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Produces: `flag_zero_amount(amount: pd.Series) -> pd.Series`, `flag_balance_inconsistency(old_balance_org: pd.Series, amount: pd.Series, new_balance_orig: pd.Series, tolerance: float = 0.01) -> pd.Series` (both bool Series).

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_cleaning/test_clean_transactions.py`:
```python
def test_flag_zero_amount_flags_only_zero_values():
    amounts = pd.Series([0.0, 10.0, 0.0, 5.0])
    result = ct.flag_zero_amount(amounts)
    assert list(result) == [True, False, True, False]


def test_flag_balance_inconsistency_flags_mismatched_rows():
    old = pd.Series([100.0, 100.0, 50.0])
    amount = pd.Series([30.0, 30.0, 20.0])
    new = pd.Series([70.0, 60.0, 30.0])
    result = ct.flag_balance_inconsistency(old, amount, new)
    assert list(result) == [False, True, False]


def test_flag_balance_inconsistency_respects_tolerance():
    old = pd.Series([100.0])
    amount = pd.Series([30.0])
    new = pd.Series([70.005])
    result = ct.flag_balance_inconsistency(old, amount, new)
    assert list(result) == [False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `AttributeError: module 'data_cleaning.clean_transactions' has no attribute 'flag_zero_amount'`

- [ ] **Step 3: Implement**

Append to `src/data_cleaning/clean_transactions.py`:
```python
def flag_zero_amount(amount: pd.Series) -> pd.Series:
    """amount == 0 transactions are flagged, not removed - observed real-data
    zero-amount CASH_OUT rows are all confirmed fraud (isFraud=1)."""
    return amount == 0


def flag_balance_inconsistency(
    old_balance_org: pd.Series, amount: pd.Series, new_balance_orig: pd.Series, tolerance: float = 0.01
) -> pd.Series:
    """Flags rows where oldbalanceOrg - amount != newbalanceOrig beyond tolerance.
    This is a known PaySim data characteristic (destination/merchant balances
    often untracked), not a data-entry error - flagged, not removed."""
    return (old_balance_org - amount - new_balance_orig).abs() > tolerance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add flag_zero_amount and flag_balance_inconsistency for data cleaning module"
```

---

### Task 6: Orchestrator `clean_dataset` and CSV/Parquet CLI

**Files:**
- Modify: `src/data_cleaning/clean_transactions.py`
- Test: `tests/data_cleaning/test_clean_transactions.py`

**Interfaces:**
- Consumes: every function from Tasks 1-5, and `build_stratified_sample` from `data_generation.generate_synthetic_fields` (existing function, signature `build_stratified_sample(df: pd.DataFrame, sample_size: int = 5000, seed: int = 42) -> pd.DataFrame`).
- Produces: `clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`, `INPUT_PARQUET_PATH`, `OUTPUT_PARQUET_PATH`, `OUTPUT_SAMPLE_CSV_PATH`, `SAMPLE_SIZE`, `main()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_cleaning/test_clean_transactions.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: FAIL with `AttributeError: module 'data_cleaning.clean_transactions' has no attribute 'clean_dataset'`

- [ ] **Step 3: Implement**

Append to `src/data_cleaning/clean_transactions.py`:
```python
from pathlib import Path
from data_generation.generate_synthetic_fields import build_stratified_sample

INPUT_PARQUET_PATH = "data/processed/transactions_synthetic.parquet"
OUTPUT_PARQUET_PATH = "data/processed/transactions_cleaned.parquet"
OUTPUT_SAMPLE_CSV_PATH = "data/processed/transactions_cleaned_sample.csv"
SAMPLE_SIZE = 5000


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report_data = {"input_rows": len(df)}

    df, na_counts, n_removed_missing = check_missing_critical(df)
    report_data["missing_values"] = {"per_column_na_counts": na_counts, "rows_removed": n_removed_missing}

    df, n_removed_dupes = dedupe_exact(df)
    report_data["duplicates"] = {"rows_removed": n_removed_dupes}

    df, category_counts, n_removed_categories = check_invalid_categories(df)
    report_data["invalid_categories"] = {"per_check_invalid_counts": category_counts, "rows_removed": n_removed_categories}

    out = df.copy()
    out["is_amount_outlier"] = flag_amount_outliers(out["amount"])
    out["is_zero_amount"] = flag_zero_amount(out["amount"])
    out["is_balance_inconsistent"] = flag_balance_inconsistency(
        out["oldbalanceOrg"], out["amount"], out["newbalanceOrig"]
    )

    report_data["amount_outliers"] = {"rows_flagged": int(out["is_amount_outlier"].sum())}
    report_data["zero_amount"] = {"rows_flagged": int(out["is_zero_amount"].sum())}
    report_data["balance_inconsistent"] = {"rows_flagged": int(out["is_balance_inconsistent"].sum())}
    report_data["output_rows"] = len(out)

    return out, report_data


def main():
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    cleaned, report_data = clean_dataset(df)
    Path(OUTPUT_PARQUET_PATH).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    sample = build_stratified_sample(cleaned, sample_size=SAMPLE_SIZE, seed=42)
    sample.to_csv(OUTPUT_SAMPLE_CSV_PATH, index=False)
    print(f"Input rows: {report_data['input_rows']}")
    print(f"Output rows: {report_data['output_rows']}")
    print(f"Wrote {len(cleaned)} rows to {OUTPUT_PARQUET_PATH}")
    print(f"Wrote {len(sample)} sample rows to {OUTPUT_SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_clean_transactions.py -v`
Expected: 17 passed

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests across every module pass (52 prior + 17 new = 69)

- [ ] **Step 6: Commit**

```bash
git add src/data_cleaning/clean_transactions.py tests/data_cleaning/test_clean_transactions.py
git commit -m "feat: add clean_dataset orchestrator and CLI for data cleaning module"
```

---

### Task 7: `cleaning_report.py`

**Files:**
- Create: `src/data_cleaning/cleaning_report.py`
- Test: `tests/data_cleaning/test_cleaning_report.py`

**Interfaces:**
- Consumes: `clean_dataset`, `INPUT_PARQUET_PATH` from `data_cleaning.clean_transactions` (Task 6).
- Produces: `build_cleaning_report_markdown(report_data: dict) -> str`, `main()`.

- [ ] **Step 1: Write the failing tests**

`tests/data_cleaning/test_cleaning_report.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_cleaning_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_cleaning.cleaning_report'`

- [ ] **Step 3: Implement**

`src/data_cleaning/cleaning_report.py`:
```python
from pathlib import Path
import pandas as pd

from data_cleaning.clean_transactions import clean_dataset, INPUT_PARQUET_PATH


def build_cleaning_report_markdown(report_data: dict) -> str:
    lines = [
        "# Cleaning Report — transactions_synthetic\n\n",
        f"Input rows: {report_data['input_rows']} | Output rows: {report_data['output_rows']}\n\n",
        "| check | rows_before | rows_flagged_or_removed | action | note |\n",
        "|---|---|---|---|---|\n",
    ]

    lines.append(
        f"| missing_values (critical columns) | {report_data['input_rows']} | "
        f"{report_data['missing_values']['rows_removed']} | removed | "
        f"NaN counts per column: {report_data['missing_values']['per_column_na_counts']} |\n"
    )
    lines.append(
        f"| duplicates (full-row) | {report_data['input_rows']} | "
        f"{report_data['duplicates']['rows_removed']} | removed | Exact duplicate rows only |\n"
    )
    lines.append(
        f"| invalid_categories | {report_data['input_rows']} | "
        f"{report_data['invalid_categories']['rows_removed']} | removed | "
        f"Per-column counts: {report_data['invalid_categories']['per_check_invalid_counts']} |\n"
    )
    lines.append(
        f"| amount_outliers (Tukey IQR) | {report_data['output_rows']} | "
        f"{report_data['amount_outliers']['rows_flagged']} | flagged (kept) | "
        f"Column: is_amount_outlier. Large amounts may be genuine fraud signal, not removed. |\n"
    )
    lines.append(
        f"| zero_amount | {report_data['output_rows']} | "
        f"{report_data['zero_amount']['rows_flagged']} | flagged (kept) | "
        f"Column: is_zero_amount. Observed zero-amount rows are confirmed fraud, not removed. |\n"
    )
    lines.append(
        f"| balance_inconsistent | {report_data['output_rows']} | "
        f"{report_data['balance_inconsistent']['rows_flagged']} | flagged (kept) | "
        f"Column: is_balance_inconsistent. **This is a known PaySim data characteristic "
        f"(destination/merchant balances often untracked), not a data-entry error** - do not "
        f"interpret a high count here as a data quality problem. |\n"
    )
    return "".join(lines)


def main():
    df = pd.read_parquet(INPUT_PARQUET_PATH)
    _, report_data = clean_dataset(df)
    markdown = build_cleaning_report_markdown(report_data)
    Path("docs/CLEANING_REPORT.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Wrote docs/CLEANING_REPORT.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_cleaning/test_cleaning_report.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass (69 prior + 1 new = 70)

- [ ] **Step 6: Commit**

```bash
git add src/data_cleaning/cleaning_report.py tests/data_cleaning/test_cleaning_report.py
git commit -m "feat: add cleaning_report markdown builder and CLI"
```

---

### Task 8: Full pipeline run and verification on the real dataset (manual verification)

**Files:** none created — this task exercises Tasks 1-7 against the real `data/processed/transactions_synthetic.parquet` (6,362,620 rows).

**Interfaces:** consumes `clean_transactions.main()` and `cleaning_report.main()`.

- [ ] **Step 1: Run the cleaner against the real synthetic parquet**

Run:
```bash
PYTHONPATH=src .venv/Scripts/python.exe -m data_cleaning.clean_transactions
```
Expected output: `Input rows: 6362620`, `Output rows: 6362620` (per the design spec's survey, 0 rows should be removed by any of the 3 removal checks), `Wrote 6362620 rows to data/processed/transactions_cleaned.parquet`, `Wrote 5000 sample rows to data/processed/transactions_cleaned_sample.csv`.

- [ ] **Step 2: Verify flag counts match the design spec's real-data survey exactly**

Run:
```bash
.venv/Scripts/python.exe -c "import pandas as pd; df = pd.read_parquet('data/processed/transactions_cleaned.parquet'); print('rows:', len(df)); print('amount_outliers:', df['is_amount_outlier'].sum()); print('zero_amount:', df['is_zero_amount'].sum()); print('balance_inconsistent:', df['is_balance_inconsistent'].sum())"
```
Expected (exact match to design spec section 2 — these are deterministic checks with no randomness, so an exact match, not an approximation, is expected):
`rows: 6362620`, `amount_outliers: 338078`, `zero_amount: 16`, `balance_inconsistent: 5118892`.

If any number differs from these, STOP and investigate before proceeding — either the design spec's survey was run against different code, or a bug was introduced.

- [ ] **Step 3: Run the report generator and inspect it**

Run:
```bash
PYTHONPATH=src .venv/Scripts/python.exe -m data_cleaning.cleaning_report
```
Expected: prints the markdown, ends with `Wrote docs/CLEANING_REPORT.md`. Open `docs/CLEANING_REPORT.md` and confirm all 6 check rows are present with the same numbers as Step 2, and the balance-inconsistency row's note about it being a known PaySim characteristic is present.

- [ ] **Step 4: Commit the cleaning report (not the generated data files - already gitignored)**

```bash
git add docs/CLEANING_REPORT.md
git commit -m "docs: generate cleaning report from real transactions_synthetic run"
```

---

## Self-Review Notes

- **Spec coverage:** all 4 required checks (missing values, duplicates, invalid categories, outliers) map to Tasks 1-5; the flag-vs-remove policy from spec section 3 (only remove structural corruption) is encoded directly in `clean_dataset`'s call order (removal checks run first, flag checks run after and never filter rows); the before/after report requirement maps to Task 7; both required deliverables (`Cleaned dataset`, `Cleaning summary/report`) map to Task 8's real-data outputs.
- **Type/interface consistency:** `check_missing_critical` and `check_invalid_categories` both return the `(df, dict, int)` shape; `dedupe_exact` returns `(df, int)` (no per-column detail needed since it's a single full-row check) — this asymmetry is intentional, not an oversight, and `clean_dataset` unpacks each according to its own shape.
- **No placeholder values:** every numeric expectation in Task 8 (338078, 16, 5118892) is copied verbatim from the design spec's real-data survey (section 2), not invented.
