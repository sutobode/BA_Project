# Synthetic Contextual Data Generation (Người 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, test, and validate the synthetic contextual-data generator for the PaySim fraud dataset described in `docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md`, producing `data/processed/transactions_synthetic.parquet` and `docs/DATA_DICTIONARY.md`.

**Architecture:** Three small, independently testable modules under `src/data_generation/`: `country_centroids.py` (pure geo math), `generate_synthetic_fields.py` (vectorized field generation + I/O), `check_leakage.py` (leakage metrics + data dictionary writer). Every generation function is a pure, vectorized function operating on `numpy`/`pandas` arrays so it can be unit-tested with small hand-built DataFrames — the real 6.36M-row CSV is only touched in the final integration task.

**Tech Stack:** Python 3.13 (`C:\ProgramData\miniconda3\python.exe`), pandas, numpy, Faker, scikit-learn (`roc_auc_score`), pyarrow (Parquet I/O), pytest.

## Global Constraints

- All values (probabilities, medians, λ) MUST match `docs/superpowers/specs/2026-07-03-synthetic-data-nguoi2-design.md` section 4 exactly — no silent rounding/changes.
- Every random draw MUST go through a single `numpy.random.Generator` seeded with `seed=42` (reproducibility requirement from the spec).
- No per-row Python loops over the 6.36M-row dataset — all generation must be vectorized (spec section 6).
- Leakage gate: any field with univariate AUC ≥ 0.75 or Cramér's V ≥ 0.5 against `isFraud` fails the check (spec section 5; the 0.5 Cramér's V cutoff is fixed here as this plan's concrete interpretation of the spec's "objective threshold" requirement).
- Invoke the venv's Python directly by path in every command — `.venv/Scripts/python.exe` (Windows venv layout) — do not rely on shell activation.
- Deliverable paths are fixed by the spec (section 10): `src/data_generation/generate_synthetic_fields.py`, `src/data_generation/country_centroids.py`, `src/data_generation/check_leakage.py`, `data/processed/transactions_synthetic.parquet`, `data/processed/transactions_synthetic_sample.csv`, `docs/DATA_DICTIONARY.md`.

---

## File Structure

- `requirements.txt` — pinned dependency floors
- `pytest.ini` — makes `src/` importable as `data_generation.*` in tests
- `.gitignore` — excludes `.venv/`, `__pycache__/`, `Data/`, `data/processed/`
- `src/data_generation/__init__.py` — empty package marker
- `src/data_generation/country_centroids.py` — country centroid table + haversine distance
- `src/data_generation/generate_synthetic_fields.py` — all 12 field generators, orchestrator, CSV→Parquet CLI
- `src/data_generation/check_leakage.py` — leakage metrics, field metadata, data-dictionary writer, CLI
- `tests/data_generation/test_country_centroids.py`
- `tests/data_generation/test_generate_synthetic_fields.py`
- `tests/data_generation/test_check_leakage.py`

---

### Task 0: Project environment setup

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `src/data_generation/__init__.py`
- Create: `tests/data_generation/__init__.py`
- Create: `tests/data_generation/test_setup_sanity.py`

**Interfaces:**
- Produces: a working venv at `.venv/`, `src` on the pytest import path (`data_generation` importable), for every later task to build on.

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=2.2
numpy>=1.26
faker>=20.0
scikit-learn>=1.4
pyarrow>=15.0
pytest>=8.0
```

- [ ] **Step 2: Create the virtual environment and install dependencies**

Run:
```bash
"/c/ProgramData/miniconda3/python.exe" -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Expected: install completes with no errors; `.venv/Scripts/python.exe -m pip list` shows pandas, numpy, faker, scikit-learn, pyarrow, pytest.

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
pythonpath = src
testpaths = tests
```

- [ ] **Step 4: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
Data/
data/processed/
```

- [ ] **Step 5: Write a sanity test proving the import path works**

`tests/data_generation/test_setup_sanity.py`:
```python
def test_data_generation_package_is_importable():
    import data_generation
    assert data_generation is not None
```

- [ ] **Step 6: Run it to verify it fails (package does not exist yet)**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_setup_sanity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_generation'`

- [ ] **Step 7: Create package/test package markers**

`src/data_generation/__init__.py`:
```python
```

`tests/data_generation/__init__.py`:
```python
```

- [ ] **Step 8: Run the test again to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_setup_sanity.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add requirements.txt pytest.ini .gitignore src/data_generation/__init__.py tests/data_generation/__init__.py tests/data_generation/test_setup_sanity.py
git commit -m "chore: set up Python environment and test scaffolding for synthetic data generation"
```

---

### Task 1: Country centroid table and haversine distance

**Files:**
- Create: `src/data_generation/country_centroids.py`
- Test: `tests/data_generation/test_country_centroids.py`

**Interfaces:**
- Produces: `COUNTRY_CENTROIDS: dict[str, tuple[float, float]]`, `COUNTRY_WEIGHTS: dict[str, float]` (sums to 1.0), `COUNTRY_LIST: list[str]`, `COUNTRY_INDEX: dict[str, int]`, `N_COUNTRIES: int`, `haversine_distance_km(lat1, lon1, lat2, lon2) -> float | np.ndarray`, `distance_between_countries(country_a: np.ndarray, country_b: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/data_generation/test_country_centroids.py`:
```python
import numpy as np
import pandas as pd
import pytest

from data_generation import country_centroids


def test_country_weights_sum_to_one():
    assert sum(country_centroids.COUNTRY_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_country_weights_and_centroids_have_same_countries():
    assert set(country_centroids.COUNTRY_WEIGHTS.keys()) == set(country_centroids.COUNTRY_CENTROIDS.keys())


def test_haversine_same_point_is_zero():
    d = country_centroids.haversine_distance_km(10.0, 20.0, 10.0, 20.0)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_london_paris():
    # London ~51.5074,-0.1278 ; Paris ~48.8566,2.3522 -> real-world distance ~344 km
    d = country_centroids.haversine_distance_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert d == pytest.approx(344, rel=0.05)


def test_distance_between_countries_same_country_is_zero():
    a = np.array(["US", "US"])
    b = np.array(["US", "US"])
    d = country_centroids.distance_between_countries(a, b)
    assert np.allclose(d, 0.0)


def test_distance_between_countries_far_apart_is_large():
    a = np.array(["US"])
    b = np.array(["VN"])
    d = country_centroids.distance_between_countries(a, b)
    assert d[0] > 10000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_country_centroids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_generation.country_centroids'`

- [ ] **Step 3: Implement `src/data_generation/country_centroids.py`**

```python
import numpy as np
import pandas as pd

COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (39.8283, -98.5795),
    "GB": (55.3781, -3.4360),
    "DE": (51.1657, 10.4515),
    "FR": (46.2276, 2.2137),
    "VN": (14.0583, 108.2772),
    "SG": (1.3521, 103.8198),
    "JP": (36.2048, 138.2529),
    "AU": (-25.2744, 133.7751),
    "CA": (56.1304, -106.3468),
    "BR": (-14.2350, -51.9253),
    "IN": (20.5937, 78.9629),
    "CN": (35.8617, 104.1954),
    "ZA": (-30.5595, 22.9375),
    "MX": (23.6345, -102.5528),
    "IT": (41.8719, 12.5674),
    "ES": (40.4637, -3.7492),
    "NL": (52.1326, 5.2913),
    "KR": (35.9078, 127.7669),
    "RU": (61.5240, 105.3188),
    "AE": (23.4241, 53.8478),
}

COUNTRY_WEIGHTS: dict[str, float] = {
    "US": 0.21, "GB": 0.10, "DE": 0.08, "FR": 0.07, "VN": 0.10,
    "SG": 0.05, "JP": 0.05, "AU": 0.04, "CA": 0.05, "BR": 0.04,
    "IN": 0.05, "CN": 0.03, "ZA": 0.02, "MX": 0.02, "IT": 0.02,
    "ES": 0.02, "NL": 0.02, "KR": 0.01, "RU": 0.01, "AE": 0.01,
}

COUNTRY_LIST: list[str] = list(COUNTRY_WEIGHTS.keys())
COUNTRY_INDEX: dict[str, int] = {country: i for i, country in enumerate(COUNTRY_LIST)}
N_COUNTRIES: int = len(COUNTRY_LIST)

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1r, lon1r, lat2r, lon2r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def distance_between_countries(country_a: np.ndarray, country_b: np.ndarray) -> np.ndarray:
    """country_a/country_b: arrays of ISO codes present in COUNTRY_CENTROIDS. Returns km distances."""
    lat_map = {code: coords[0] for code, coords in COUNTRY_CENTROIDS.items()}
    lon_map = {code: coords[1] for code, coords in COUNTRY_CENTROIDS.items()}
    lat1 = pd.Series(country_a).map(lat_map).to_numpy(dtype="float64")
    lon1 = pd.Series(country_a).map(lon_map).to_numpy(dtype="float64")
    lat2 = pd.Series(country_b).map(lat_map).to_numpy(dtype="float64")
    lon2 = pd.Series(country_b).map(lon_map).to_numpy(dtype="float64")
    return haversine_distance_km(lat1, lon1, lat2, lon2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_country_centroids.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/country_centroids.py tests/data_generation/test_country_centroids.py
git commit -m "feat: add country centroid table and haversine distance for IP-billing distance"
```

---

### Task 2: Derived time fields (`hour_of_day`, `is_night_transaction`)

**Files:**
- Create: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Produces: `generate_hour_of_day(step: pd.Series) -> pd.Series`, `generate_is_night_transaction(hour_of_day: pd.Series) -> pd.Series`.

- [ ] **Step 1: Write the failing tests**

`tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_generation.generate_synthetic_fields'`

- [ ] **Step 3: Implement the derived time fields**

`src/data_generation/generate_synthetic_fields.py`:
```python
import numpy as np
import pandas as pd


def generate_hour_of_day(step: pd.Series) -> pd.Series:
    """step in PaySim = elapsed hours since simulation start (1-indexed)."""
    return ((step - 1) % 24).astype("int16")


def generate_is_night_transaction(hour_of_day: pd.Series) -> pd.Series:
    """Night defined as 00:00-05:59 (business assumption)."""
    return hour_of_day.between(0, 5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add derived time-of-day fields (hour_of_day, is_night_transaction)"
```

---

### Task 3: Device pool and independent categorical fields (`device_id`, `browser`, `device_type`)

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: none new.
- Produces: `DEVICE_POOL_SIZE: int`, `BROWSER_WEIGHTS: dict[str, float]`, `DEVICE_TYPE_WEIGHTS: dict[str, float]`, `build_device_pool(size: int, seed: int) -> np.ndarray`, `generate_categorical(n: int, weights: dict, rng: np.random.Generator) -> np.ndarray`, `generate_device_id(n: int, device_pool: np.ndarray, rng: np.random.Generator) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: module 'data_generation.generate_synthetic_fields' has no attribute 'build_device_pool'`

- [ ] **Step 3: Implement device pool and categorical generators**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
from faker import Faker

DEVICE_POOL_SIZE = 50_000
BROWSER_WEIGHTS = {"Chrome": 0.55, "Safari": 0.20, "Edge": 0.12, "Firefox": 0.08, "Other": 0.05}
DEVICE_TYPE_WEIGHTS = {"mobile": 0.65, "desktop": 0.30, "tablet": 0.05}


def build_device_pool(size: int = DEVICE_POOL_SIZE, seed: int = 42) -> np.ndarray:
    faker = Faker()
    Faker.seed(seed)
    return np.array([faker.uuid4() for _ in range(size)])


def generate_categorical(n: int, weights: dict, rng: np.random.Generator) -> np.ndarray:
    categories = np.array(list(weights.keys()))
    probs = np.array(list(weights.values()), dtype="float64")
    probs = probs / probs.sum()
    return rng.choice(categories, size=n, p=probs)


def generate_device_id(n: int, device_pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(device_pool, size=n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add device pool and independent categorical fields (device_id, browser, device_type)"
```

---

### Task 4: Conditional Bernoulli helper and `new_device_flag`

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: none new.
- Produces: `generate_conditional_bernoulli(is_fraud: np.ndarray, base_p: float, fraud_p: float, rng: np.random.Generator) -> np.ndarray`, `NEW_DEVICE_FLAG_BASE_P = 0.04`, `NEW_DEVICE_FLAG_FRAUD_P = 0.12`, `generate_new_device_flag(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_conditional_bernoulli'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
def generate_conditional_bernoulli(
    is_fraud: np.ndarray, base_p: float, fraud_p: float, rng: np.random.Generator
) -> np.ndarray:
    p = np.where(is_fraud == 1, fraud_p, base_p)
    return rng.binomial(1, p).astype(bool)


NEW_DEVICE_FLAG_BASE_P = 0.04
NEW_DEVICE_FLAG_FRAUD_P = 0.12


def generate_new_device_flag(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_bernoulli(is_fraud, NEW_DEVICE_FLAG_BASE_P, NEW_DEVICE_FLAG_FRAUD_P, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add conditional Bernoulli generator and new_device_flag"
```

---

### Task 5: `customer_account_age_days`

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: none new.
- Produces: `ACCOUNT_AGE_BASE_MEDIAN_DAYS = 400`, `ACCOUNT_AGE_FRAUD_MEDIAN_DAYS = 150`, `ACCOUNT_AGE_SIGMA = 0.6`, `ACCOUNT_AGE_MIN_DAYS = 1`, `ACCOUNT_AGE_MAX_DAYS = 3650`, `generate_account_age_days(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_account_age_days'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
ACCOUNT_AGE_BASE_MEDIAN_DAYS = 400
ACCOUNT_AGE_FRAUD_MEDIAN_DAYS = 150
ACCOUNT_AGE_SIGMA = 0.6
ACCOUNT_AGE_MIN_DAYS = 1
ACCOUNT_AGE_MAX_DAYS = 3650


def generate_account_age_days(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    median = np.where(is_fraud == 1, ACCOUNT_AGE_FRAUD_MEDIAN_DAYS, ACCOUNT_AGE_BASE_MEDIAN_DAYS)
    mu = np.log(median.astype("float64"))
    raw = rng.lognormal(mean=mu, sigma=ACCOUNT_AGE_SIGMA)
    clipped = np.clip(raw, ACCOUNT_AGE_MIN_DAYS, ACCOUNT_AGE_MAX_DAYS)
    return clipped.round().astype("int32")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add customer_account_age_days conditional lognormal generator"
```

---

### Task 6: `billing_country`, `ip_country`, `ip_billing_distance_km`

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: `COUNTRY_WEIGHTS`, `COUNTRY_LIST`, `COUNTRY_INDEX`, `N_COUNTRIES`, `distance_between_countries` from `country_centroids` (Task 1).
- Produces: `IP_COUNTRY_MATCH_BASE_P = 0.93`, `IP_COUNTRY_MATCH_FRAUD_P = 0.80`, `generate_billing_country(n: int, rng: np.random.Generator) -> np.ndarray`, `generate_ip_country(billing_country: np.ndarray, is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray`, `generate_ip_billing_distance_km(ip_country: np.ndarray, billing_country: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_billing_country'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
from data_generation.country_centroids import (
    COUNTRY_WEIGHTS,
    COUNTRY_LIST,
    COUNTRY_INDEX,
    N_COUNTRIES,
    distance_between_countries,
)

IP_COUNTRY_MATCH_BASE_P = 0.93
IP_COUNTRY_MATCH_FRAUD_P = 0.80


def generate_billing_country(n: int, rng: np.random.Generator) -> np.ndarray:
    return generate_categorical(n, COUNTRY_WEIGHTS, rng)


def generate_ip_country(billing_country: np.ndarray, is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(billing_country)
    match_p = np.where(is_fraud == 1, IP_COUNTRY_MATCH_FRAUD_P, IP_COUNTRY_MATCH_BASE_P)
    matches = rng.binomial(1, match_p).astype(bool)
    billing_idx = pd.Series(billing_country).map(COUNTRY_INDEX).to_numpy()
    # Offset by 1..N-1 guarantees a genuinely different country when not matching -
    # no accidental same-country coincidence, keeping the mismatch rate exact.
    offset = rng.integers(1, N_COUNTRIES, size=n)
    mismatch_idx = (billing_idx + offset) % N_COUNTRIES
    mismatch_country = np.array(COUNTRY_LIST)[mismatch_idx]
    return np.where(matches, billing_country, mismatch_country)


def generate_ip_billing_distance_km(ip_country: np.ndarray, billing_country: np.ndarray) -> np.ndarray:
    return distance_between_countries(ip_country, billing_country)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add billing_country, ip_country and ip_billing_distance_km generators"
```

---

### Task 7: `shipping_billing_mismatch` and `failed_payment_attempts_24h`

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: `generate_conditional_bernoulli` (Task 4).
- Produces: `SHIPPING_MISMATCH_BASE_P = 0.05`, `SHIPPING_MISMATCH_FRAUD_P = 0.15`, `generate_shipping_billing_mismatch(is_fraud, rng) -> np.ndarray`, `FAILED_ATTEMPTS_BASE_LAMBDA = 0.15`, `FAILED_ATTEMPTS_FRAUD_LAMBDA = 0.6`, `generate_failed_payment_attempts_24h(is_fraud, rng) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_shipping_billing_mismatch'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
SHIPPING_MISMATCH_BASE_P = 0.05
SHIPPING_MISMATCH_FRAUD_P = 0.15


def generate_shipping_billing_mismatch(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_bernoulli(is_fraud, SHIPPING_MISMATCH_BASE_P, SHIPPING_MISMATCH_FRAUD_P, rng)


FAILED_ATTEMPTS_BASE_LAMBDA = 0.15
FAILED_ATTEMPTS_FRAUD_LAMBDA = 0.6


def generate_failed_payment_attempts_24h(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lam = np.where(is_fraud == 1, FAILED_ATTEMPTS_FRAUD_LAMBDA, FAILED_ATTEMPTS_BASE_LAMBDA)
    return rng.poisson(lam).astype("int16")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 27 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add shipping_billing_mismatch and failed_payment_attempts_24h generators"
```

---

### Task 8: Orchestrator (`generate_all_synthetic_fields`) and CSV→Parquet CLI

**Files:**
- Modify: `src/data_generation/generate_synthetic_fields.py`
- Test: `tests/data_generation/test_generate_synthetic_fields.py`

**Interfaces:**
- Consumes: every `generate_*` function from Tasks 2-7 and `build_device_pool`, `DEVICE_POOL_SIZE` from Task 3.
- Produces: `generate_all_synthetic_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame`, `load_raw_transactions(csv_path: str) -> pd.DataFrame`, `build_stratified_sample(df: pd.DataFrame, sample_size: int = 5000, seed: int = 42) -> pd.DataFrame`, `main()`, constants `INPUT_CSV_PATH`, `OUTPUT_PARQUET_PATH`, `OUTPUT_SAMPLE_CSV_PATH`, `SAMPLE_SIZE`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_generate_synthetic_fields.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_all_synthetic_fields'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/generate_synthetic_fields.py`:
```python
from pathlib import Path

INPUT_CSV_PATH = "Data/PS_20174392719_1491204439457_log.csv"
OUTPUT_PARQUET_PATH = "data/processed/transactions_synthetic.parquet"
OUTPUT_SAMPLE_CSV_PATH = "data/processed/transactions_synthetic_sample.csv"
SAMPLE_SIZE = 5000


def generate_all_synthetic_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    is_fraud = df["isFraud"].to_numpy()
    out = df.copy()

    out["hour_of_day"] = generate_hour_of_day(df["step"])
    out["is_night_transaction"] = generate_is_night_transaction(out["hour_of_day"])
    out["customer_account_age_days"] = generate_account_age_days(is_fraud, rng)

    device_pool = build_device_pool(size=DEVICE_POOL_SIZE, seed=seed)
    out["device_id"] = generate_device_id(n, device_pool, rng)
    out["browser"] = generate_categorical(n, BROWSER_WEIGHTS, rng)
    out["device_type"] = generate_categorical(n, DEVICE_TYPE_WEIGHTS, rng)
    out["new_device_flag"] = generate_new_device_flag(is_fraud, rng)

    out["billing_country"] = generate_billing_country(n, rng)
    out["ip_country"] = generate_ip_country(out["billing_country"].to_numpy(), is_fraud, rng)
    out["ip_billing_distance_km"] = generate_ip_billing_distance_km(
        out["ip_country"].to_numpy(), out["billing_country"].to_numpy()
    )

    out["shipping_billing_mismatch"] = generate_shipping_billing_mismatch(is_fraud, rng)
    out["failed_payment_attempts_24h"] = generate_failed_payment_attempts_24h(is_fraud, rng)

    return out


def load_raw_transactions(csv_path: str = INPUT_CSV_PATH) -> pd.DataFrame:
    dtype = {
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "nameOrig": "string",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "nameDest": "string",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    return pd.read_csv(csv_path, dtype=dtype)


def build_stratified_sample(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE, seed: int = 42) -> pd.DataFrame:
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0]
    fraud_frac = len(fraud) / len(df)
    n_fraud_sample = min(len(fraud), max(1, round(sample_size * fraud_frac)))
    n_non_fraud_sample = min(len(non_fraud), sample_size - n_fraud_sample)
    sampled = pd.concat([
        fraud.sample(n=n_fraud_sample, random_state=seed),
        non_fraud.sample(n=n_non_fraud_sample, random_state=seed),
    ])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    df = load_raw_transactions()
    result = generate_all_synthetic_fields(df, seed=42)
    Path(OUTPUT_PARQUET_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    sample = build_stratified_sample(result, sample_size=SAMPLE_SIZE, seed=42)
    sample.to_csv(OUTPUT_SAMPLE_CSV_PATH, index=False)
    print(f"Wrote {len(result)} rows to {OUTPUT_PARQUET_PATH}")
    print(f"Wrote {len(sample)} sample rows to {OUTPUT_SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_generate_synthetic_fields.py -v`
Expected: 33 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/generate_synthetic_fields.py tests/data_generation/test_generate_synthetic_fields.py
git commit -m "feat: add generate_all_synthetic_fields orchestrator and CSV-to-Parquet CLI"
```

---

### Task 9: Leakage metrics (`univariate_auc`, `cramers_v`)

**Files:**
- Create: `src/data_generation/check_leakage.py`
- Test: `tests/data_generation/test_check_leakage.py`

**Interfaces:**
- Produces: `AUC_FAIL_THRESHOLD = 0.75`, `CRAMERS_V_FAIL_THRESHOLD = 0.5`, `univariate_auc(feature: pd.Series, label: pd.Series) -> float`, `cramers_v(feature: pd.Series, label: pd.Series) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests/data_generation/test_check_leakage.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_check_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_generation.check_leakage'`

- [ ] **Step 3: Implement**

`src/data_generation/check_leakage.py`:
```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AUC_FAIL_THRESHOLD = 0.75
CRAMERS_V_FAIL_THRESHOLD = 0.5


def univariate_auc(feature: pd.Series, label: pd.Series) -> float:
    score = feature.astype("float64")
    auc = roc_auc_score(label, score)
    # Direction-agnostic: a field that predicts "not fraud" strongly is just as leaky
    # as one that predicts "fraud" strongly.
    return max(auc, 1 - auc)


def _chi2_statistic(contingency: pd.DataFrame) -> float:
    observed = contingency.to_numpy(dtype="float64")
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    return float(((observed - expected) ** 2 / expected).sum())


def cramers_v(feature: pd.Series, label: pd.Series) -> float:
    contingency = pd.crosstab(feature, label)
    chi2 = _chi2_statistic(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    denom = min(r - 1, k - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_check_leakage.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/data_generation/check_leakage.py tests/data_generation/test_check_leakage.py
git commit -m "feat: add univariate AUC and Cramer's V leakage metrics"
```

---

### Task 10: Field metadata, `check_all_fields`, and data-dictionary writer

**Files:**
- Modify: `src/data_generation/check_leakage.py`
- Test: `tests/data_generation/test_check_leakage.py`

**Interfaces:**
- Consumes: `univariate_auc`, `cramers_v`, `AUC_FAIL_THRESHOLD`, `CRAMERS_V_FAIL_THRESHOLD` from Task 9. Field names must exactly match those produced by `generate_all_synthetic_fields` (Task 8).
- Produces: `FIELD_METADATA: list[dict]`, `check_all_fields(df: pd.DataFrame, label_col: str = "isFraud") -> pd.DataFrame`, `build_data_dictionary_markdown(leakage_report: pd.DataFrame) -> str`, `main()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data_generation/test_check_leakage.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_check_leakage.py -v`
Expected: FAIL with `AttributeError: module 'data_generation.check_leakage' has no attribute 'FIELD_METADATA'`

- [ ] **Step 3: Implement**

Append to `src/data_generation/check_leakage.py`:
```python
from pathlib import Path

FIELD_METADATA: list[dict] = [
    {
        "field_name": "hour_of_day", "data_type": "int16", "unit": "hour (0-23)",
        "valid_range": "[0, 23]", "generation_type": "derived",
        "logic_or_formula": "(step - 1) % 24",
        "business_assumption": "step in PaySim represents elapsed hours since simulation start.",
        "metric_type": "auc",
    },
    {
        "field_name": "is_night_transaction", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "derived",
        "logic_or_formula": "hour_of_day in [0, 5]",
        "business_assumption": "Night defined as 00:00-05:59.",
        "metric_type": "auc",
    },
    {
        "field_name": "customer_account_age_days", "data_type": "int32", "unit": "days",
        "valid_range": "[1, 3650]", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "lognormal(median=400 base / 150 fraud, sigma=0.6), clipped",
        "business_assumption": "Compromised/mule accounts assumed more often recently created (business assumption, not empirically derived).",
        "metric_type": "auc",
    },
    {
        "field_name": "device_id", "data_type": "string", "unit": "UUID",
        "valid_range": "50,000-value pool", "generation_type": "independent-random",
        "logic_or_formula": "uniform sample from a fixed 50,000-UUID Faker-generated pool",
        "business_assumption": "Device identity alone carries no fraud signal; signal lives in new_device_flag.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "browser", "data_type": "category", "unit": "categorical",
        "valid_range": "{Chrome, Safari, Edge, Firefox, Other}", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical: Chrome 55%, Safari 20%, Edge 12%, Firefox 8%, Other 5%",
        "business_assumption": "No behavioral link to fraud assumed; kept neutral to avoid manufactured signal.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "device_type", "data_type": "category", "unit": "categorical",
        "valid_range": "{mobile, desktop, tablet}", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical: mobile 65%, desktop 30%, tablet 5%",
        "business_assumption": "No behavioral link to fraud assumed.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "new_device_flag", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Bernoulli(p=0.04 base / 0.12 fraud)",
        "business_assumption": "Account-takeover fraud plausibly more often from an unrecognized device; odds-ratio capped at 3x (business assumption).",
        "metric_type": "auc",
    },
    {
        "field_name": "billing_country", "data_type": "category", "unit": "ISO country code",
        "valid_range": "20 fixed countries", "generation_type": "independent-random",
        "logic_or_formula": "weighted categorical over 20 countries (simulated customer base geography)",
        "business_assumption": "No direct fraud link; signal lives in ip_country mismatch.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "ip_country", "data_type": "category", "unit": "ISO country code",
        "valid_range": "20 fixed countries", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "= billing_country with p=0.93 base / 0.80 fraud, else a different country",
        "business_assumption": "Legit traffic mostly matches home country; fraud has higher (but not certain) mismatch odds since VPNs let fraud spoof location too.",
        "metric_type": "cramers_v",
    },
    {
        "field_name": "ip_billing_distance_km", "data_type": "float64", "unit": "km",
        "valid_range": "[0, ~20000]", "generation_type": "derived",
        "logic_or_formula": "haversine(centroid[ip_country], centroid[billing_country])",
        "business_assumption": "Derived directly from ip_country/billing_country for internal consistency.",
        "metric_type": "auc",
    },
    {
        "field_name": "shipping_billing_mismatch", "data_type": "bool", "unit": "boolean",
        "valid_range": "{True, False}", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Bernoulli(p=0.05 base / 0.15 fraud)",
        "business_assumption": "Reinterpreted as 'transaction address differs from registered address' given PaySim's account-takeover fraud pattern (not card-present checkout).",
        "metric_type": "auc",
    },
    {
        "field_name": "failed_payment_attempts_24h", "data_type": "int16", "unit": "count",
        "valid_range": "[0, ~10]", "generation_type": "conditional-on-fraud",
        "logic_or_formula": "Poisson(lambda=0.15 base / 0.6 fraud)",
        "business_assumption": "Attackers often attempt multiple times before succeeding; odds-ratio capped at 4x (business assumption).",
        "metric_type": "auc",
    },
]


def check_all_fields(df: pd.DataFrame, label_col: str = "isFraud") -> pd.DataFrame:
    label = df[label_col]
    rows = []
    for meta in FIELD_METADATA:
        name = meta["field_name"]
        if meta["metric_type"] == "auc":
            value = univariate_auc(df[name], label)
            status = "FAIL" if value >= AUC_FAIL_THRESHOLD else "PASS"
        else:
            value = cramers_v(df[name], label)
            status = "FAIL" if value >= CRAMERS_V_FAIL_THRESHOLD else "PASS"
        rows.append({
            "field_name": name,
            "metric_type": meta["metric_type"],
            "metric_value": round(value, 4),
            "status": status,
        })
    return pd.DataFrame(rows)


def build_data_dictionary_markdown(leakage_report: pd.DataFrame) -> str:
    metric_by_field = leakage_report.set_index("field_name")["metric_value"].to_dict()
    lines = [
        "# Data Dictionary — Synthetic Contextual Fields\n\n",
        "| field_name | data_type | unit | valid_range | generation_type | logic_or_formula | "
        "business_assumption | measured_univariate_association_with_isFraud |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for meta in FIELD_METADATA:
        metric = metric_by_field.get(meta["field_name"], "n/a")
        lines.append(
            f"| `{meta['field_name']}` | {meta['data_type']} | {meta['unit']} | {meta['valid_range']} | "
            f"{meta['generation_type']} | {meta['logic_or_formula']} | {meta['business_assumption']} | {metric} |\n"
        )
    return "".join(lines)


def main():
    df = pd.read_parquet("data/processed/transactions_synthetic.parquet")
    report = check_all_fields(df)
    print(report.to_string(index=False))
    failures = report[report["status"] == "FAIL"]
    if len(failures) > 0:
        raise SystemExit(f"Leakage check FAILED for fields: {list(failures['field_name'])}")
    markdown = build_data_dictionary_markdown(report)
    Path("docs/DATA_DICTIONARY.md").write_text(markdown, encoding="utf-8")
    print("Wrote docs/DATA_DICTIONARY.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/data_generation/test_check_leakage.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests across all three modules pass (41 total)

- [ ] **Step 6: Commit**

```bash
git add src/data_generation/check_leakage.py tests/data_generation/test_check_leakage.py
git commit -m "feat: add field metadata, check_all_fields aggregator and data dictionary writer"
```

---

### Task 11: Full pipeline run and verification on the real dataset (manual verification)

**Files:** none created — this task exercises Tasks 1-10 against the real 6,362,620-row file.

**Interfaces:** consumes `generate_synthetic_fields.main()` and `check_leakage.main()`.

- [ ] **Step 1: Run the generator against the real PaySim CSV**

Run:
```bash
.venv/Scripts/python.exe -m data_generation.generate_synthetic_fields
```
(Run from repo root with `src` on `PYTHONPATH`, e.g. `PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.generate_synthetic_fields` in git-bash.)

Expected output: `Wrote 6362620 rows to data/processed/transactions_synthetic.parquet` and `Wrote 5000 sample rows to data/processed/transactions_synthetic_sample.csv`.

- [ ] **Step 2: Verify row count and fraud rate are unchanged**

Run:
```bash
.venv/Scripts/python.exe -c "import pandas as pd; df = pd.read_parquet('data/processed/transactions_synthetic.parquet'); print(len(df)); print(df['isFraud'].mean())"
```
Expected: `6362620` and a fraud rate ≈ `0.001291` (matches the 0.1291% verified against the raw file in the spec — confirms the generator did not alter row count or label distribution).

- [ ] **Step 3: Run the leakage check and generate the data dictionary**

Run:
```bash
PYTHONPATH=src .venv/Scripts/python.exe -m data_generation.check_leakage
```
Expected: a printed table of 12 fields all with `status=PASS`, and the message `Wrote docs/DATA_DICTIONARY.md`. If any field shows `FAIL`, stop and reduce the corresponding odds-ratio/λ constant in `generate_synthetic_fields.py` (per spec section 5), regenerate (Step 1), and re-check.

- [ ] **Step 4: Confirm `docs/DATA_DICTIONARY.md` has all 12 fields**

Run: open `docs/DATA_DICTIONARY.md` and confirm it has one table row per field in `FIELD_METADATA` with a real (non-"n/a") `measured_univariate_association_with_isFraud` value for each.

- [ ] **Step 5: Commit the data dictionary (not the generated data files — already gitignored)**

```bash
git add docs/DATA_DICTIONARY.md
git commit -m "docs: generate data dictionary from real PaySim dataset run"
```

---

## Self-Review Notes

- **Spec coverage:** all 12 fields from spec section 4, the leakage gate from section 5, storage choices from section 6, and the data dictionary structure from section 7 each map to a task above (Tasks 2-10). Section 8 limitations are captured as `business_assumption` text in `FIELD_METADATA` (Task 10) so they ship with the deliverable, not just the spec doc.
- **Country weight arithmetic:** the spec did not pin exact per-country weights; this plan fixes a concrete 20-country list summing to exactly 1.0 (`US` adjusted to 0.21, not 0.22, to correct a rounding overshoot) so `rng.choice(..., p=...)` never raises at runtime.
- **Cramér's V threshold:** the spec left this unset for categorical fields (only AUC < 0.75 was pinned); this plan fixes 0.5 as the concrete cutoff, consistent with the spec's intent of an objective, moderate-strength ceiling.
