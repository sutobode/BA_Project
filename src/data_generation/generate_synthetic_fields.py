"""
OFFLINE-ONLY MODULE - DO NOT CALL FROM SERVING/SCORING CODE.

Every function in this module is a batch dataset-construction helper, run
once (or occasionally re-run) to build/enrich the training dataset from the
raw PaySim CSV. None of it is safe to call from a real-time scoring API:
- The Bernoulli/Poisson draws below are for SIMULATING plausible historical
  values so the training set has contextual columns PaySim lacks. At serving
  time, a real system observes these values directly (real device
  fingerprint, real IP, real payment-retry count) - it does not re-roll them.
- amount_percentile (inside compute_risk_proxy) is a statistic fitted against
  a reference distribution; scoring a single live transaction correctly
  requires the SAME persisted reference used at training time (see
  fit_amount_percentile_reference / apply_amount_percentile below), not a
  fresh in-batch computation.
"""

import numpy as np
import pandas as pd
from faker import Faker


def generate_hour_of_day(step: pd.Series) -> pd.Series:
    """step in PaySim = elapsed hours since simulation start (1-indexed)."""
    return ((step - 1) % 24).astype("int16")


def generate_is_night_transaction(hour_of_day: pd.Series) -> pd.Series:
    """Night defined as 00:00-05:59 (business assumption)."""
    return hour_of_day.between(0, 5)


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


def generate_device_id_and_new_device_flag(
    name_orig: pd.Series,
    risk_score: np.ndarray,
    device_pool: np.ndarray,
    rng: np.random.Generator,
    base_p: float,
    high_risk_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint generation so device_id and new_device_flag stay semantically
    consistent for accounts (nameOrig) that appear more than once in the
    dataset (~9.3k rows out of 6.36M - the 0.15% of nameOrig values that
    repeat, per README section 3). Previously the two fields were generated
    completely independently: device_id was resampled uniformly from the
    50,000-device pool on EVERY row regardless of nameOrig, so even when
    new_device_flag happened to come out False for a repeat account's second
    transaction (claiming "this is a device already associated with this
    account"), the independently-resampled device_id almost always differed
    from the device seen on that account's earlier transaction (~99.998% of
    the time, since the pool has 50,000 values) - directly contradicting the
    flag's own meaning.

    Semantics enforced here (repeat accounts only - see below):
    - new_device_flag is still drawn from the same risk-based
      Bernoulli(base_p, high_risk_p) as before - the injected fraud signal
      and its overall population rate are unchanged.
    - device_id is then made to AGREE with that draw using this account's
      history so far: True -> sample a device not already in the account's
      history (and add it); False -> reuse a device already in the
      account's history.
    - Edge case: if new_device_flag draws False on the very first
      transaction seen for an account (no device history exists yet to
      reuse), it is overridden to True, since "known device" is impossible
      to satisfy with zero history. This only affects a first-occurrence row
      that happened to sample False, a rare, narrow correction - not a
      population-wide behavior change.

    For the ~99.85% of rows whose nameOrig is unique in the dataset, there is
    no history to be consistent (or inconsistent) with, so both fields keep
    the original, fully independent, fully vectorized generation: device_id
    uniform from the pool, new_device_flag from the usual risk-based
    Bernoulli - unchanged from before this fix.

    Performance: the history-consistency bookkeeping only runs over rows
    belonging to a repeat-nameOrig account (~9.3k of 6.36M rows) via a
    Python-level loop - negligible cost relative to the fully vectorized
    fast path used for the other 99.85% of rows.
    """
    n = len(name_orig)
    risk_score = np.asarray(risk_score, dtype="float64")
    p_new = base_p + risk_score * (high_risk_p - base_p)

    name_orig_values = name_orig.to_numpy()
    counts = name_orig.value_counts()
    repeat_accounts = set(counts[counts > 1].index)
    is_repeat_row = name_orig.isin(repeat_accounts).to_numpy()

    # Fast path (99.85% of rows): fully independent, exactly as before.
    device_id = rng.choice(device_pool, size=n)
    new_device_flag = rng.binomial(1, p_new).astype(bool)

    # Slow-but-tiny path: overwrite only the repeat-account rows with
    # history-consistent draws, processed in original row order (which
    # follows step order in the raw PaySim file) so history builds up
    # causally.
    # Per-account history: a list (for random.choice-style reuse) plus a set
    # (for fast "already seen" membership checks) kept in sync together.
    history: dict = {}
    for pos in np.where(is_repeat_row)[0]:
        acc = name_orig_values[pos]
        entry = history.setdefault(acc, {"devices": [], "seen": set()})
        known, seen = entry["devices"], entry["seen"]
        # Pool exhaustion guard: if this account has already accumulated
        # every device in the pool (only possible with a tiny pool and a very
        # long history, e.g. a stress test - the real 50,000-device pool
        # against ~9.3k repeat-account rows never comes close), there is no
        # genuinely unseen device left to assign. Fall back to reuse rather
        # than loop forever searching for a nonexistent "new" device.
        can_be_new = len(seen) < len(device_pool)
        is_new = can_be_new and (bool(rng.binomial(1, p_new[pos])) or len(known) == 0)
        if is_new:
            device = rng.choice(device_pool)
            while device in seen:
                device = rng.choice(device_pool)
            device_id[pos] = device
            new_device_flag[pos] = True
            known.append(device)
            seen.add(device)
        else:
            device_id[pos] = known[int(rng.integers(0, len(known)))]
            new_device_flag[pos] = False

    return device_id, new_device_flag


RISK_PROXY_TYPES = {"TRANSFER", "CASH_OUT"}


def fit_amount_percentile_reference(type_: pd.Series, amount: pd.Series) -> dict[str, np.ndarray]:
    """FIT step (train-only): for each transaction type, store the sorted
    array of training-set amounts. This is the reference distribution that
    apply_amount_percentile() looks values up against.

    Call this ONCE on the training split only. Never re-fit on data that
    includes validation/test/live rows - doing so lets information about
    those rows' amounts leak into every training row's percentile (the
    train/test leakage this pair of functions exists to prevent).
    """
    reference: dict[str, np.ndarray] = {}
    for type_value, group in amount.groupby(type_):
        reference[str(type_value)] = np.sort(group.to_numpy(dtype="float64"))
    return reference


def apply_amount_percentile(type_: pd.Series, amount: pd.Series, reference: dict[str, np.ndarray]) -> np.ndarray:
    """TRANSFORM step: percentile rank of each amount within its type,
    looked up against a reference distribution fitted elsewhere (train-only)
    via fit_amount_percentile_reference(). This is what makes the statistic
    reproducible for a single new transaction at scoring time - it only
    needs the persisted reference array for that type plus its own
    (type, amount), never the rest of the current batch.

    Unknown types (not present in reference) fall back to percentile 0.5
    (neutral) rather than raising, since real-time traffic can include a
    type combination that never appeared in the training split.
    """
    out = np.empty(len(amount), dtype="float64")
    amount_values = amount.to_numpy(dtype="float64")
    type_values = type_.astype(str).to_numpy()
    for i in range(len(amount_values)):
        ref = reference.get(type_values[i])
        if ref is None or len(ref) == 0:
            out[i] = 0.5
        else:
            # searchsorted gives the count of reference values <= this amount;
            # dividing by len(ref) gives a percentile rank in [0, 1] without
            # needing this row's own value to be part of the reference array.
            out[i] = np.searchsorted(ref, amount_values[i], side="right") / len(ref)
    return np.clip(out, 0.0, 1.0)


def compute_risk_proxy(
    type_: pd.Series,
    amount: pd.Series,
    hour_of_day: pd.Series,
    amount_percentile_reference: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Label-free composite risk proxy in [0, 1]. Never reads isFraud, and
    deliberately never reads oldbalanceOrg/newbalanceOrig - those balance
    columns near-perfectly determine isFraud in PaySim (fraud = drain the
    account), so using them would leak the target through the back door.

    This is an OFFLINE dataset-construction helper, not an inference-time
    function - see the module docstring.

    Components (equal weight, business assumption - not fit to the label):
    - risky_type: TRANSFER/CASH_OUT are the only channels PaySim fraud uses,
      but the vast majority of transactions on these channels are legitimate
      (fraud rate within them is under 1%), so this is a weak channel-risk
      heuristic, not a near-deterministic proxy. Computable per-transaction.
    - amount_percentile: rank of amount WITHIN its own transaction type,
      against amount_percentile_reference (fit via
      fit_amount_percentile_reference() on the TRAIN split only - see that
      function's docstring). If amount_percentile_reference is None, it is
      fit on-the-fly from this call's own (type_, amount) - this in-batch
      fallback is only correct when this call's input IS the full training
      set with no held-out split; generate_all_synthetic_fields() takes a
      reference for exactly this reason, so callers who do split their data
      are not silently exposed to leakage.
    - is_night: transaction occurs in the 00:00-05:59 window - standard
      time-of-day risk heuristic. Computable per-transaction.

    IMPORTANT (do not over-claim): because risk_score is a function of
    (type, amount, hour), the conditional fields derived from it carry no
    predictive information beyond what type/amount/hour already provide. Their
    fraud association is injected by design, not learned from real behavioral
    data, and is NOT evidence of real-world predictive power. See the
    Limitations section of the report/README.
    """
    risky_type = type_.isin(RISK_PROXY_TYPES).astype("float64").to_numpy()
    if amount_percentile_reference is None:
        amount_percentile_reference = fit_amount_percentile_reference(type_, amount)
    amount_percentile = apply_amount_percentile(type_, amount, amount_percentile_reference)
    is_night = hour_of_day.between(0, 5).astype("float64").to_numpy()
    return (risky_type + amount_percentile + is_night) / 3.0


def generate_conditional_on_risk(
    risk_score: np.ndarray, base_p: float, high_risk_p: float, rng: np.random.Generator
) -> np.ndarray:
    """Label-free: risk_score in [0, 1] comes from compute_risk_proxy(), never
    from isFraud. Linear interpolation between base_p (risk_score=0) and
    high_risk_p (risk_score=1) preserves the same bounded odds-ratio as the
    previous label-conditional design, just driven by an observable proxy."""
    p = base_p + np.asarray(risk_score) * (high_risk_p - base_p)
    return rng.binomial(1, p).astype(bool)


NEW_DEVICE_FLAG_BASE_P = 0.04
NEW_DEVICE_FLAG_HIGH_RISK_P = 0.12


def generate_new_device_flag(risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_on_risk(risk_score, NEW_DEVICE_FLAG_BASE_P, NEW_DEVICE_FLAG_HIGH_RISK_P, rng)


ACCOUNT_AGE_BASE_MEDIAN_DAYS = 400
ACCOUNT_AGE_HIGH_RISK_MEDIAN_DAYS = 275
ACCOUNT_AGE_SIGMA = 0.6
ACCOUNT_AGE_MIN_DAYS = 1
ACCOUNT_AGE_MAX_DAYS = 3650


def generate_account_age_days(risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    risk_score = np.asarray(risk_score, dtype="float64")
    median = ACCOUNT_AGE_BASE_MEDIAN_DAYS + risk_score * (
        ACCOUNT_AGE_HIGH_RISK_MEDIAN_DAYS - ACCOUNT_AGE_BASE_MEDIAN_DAYS
    )
    mu = np.log(median)
    raw = rng.lognormal(mean=mu, sigma=ACCOUNT_AGE_SIGMA)
    clipped = np.clip(raw, ACCOUNT_AGE_MIN_DAYS, ACCOUNT_AGE_MAX_DAYS)
    return clipped.round().astype("int32")


from data_generation.country_centroids import (
    COUNTRY_WEIGHTS,
    COUNTRY_LIST,
    COUNTRY_INDEX,
    N_COUNTRIES,
    distance_between_countries,
)

IP_COUNTRY_MATCH_BASE_P = 0.93
IP_COUNTRY_MATCH_HIGH_RISK_P = 0.80


def generate_billing_country(n: int, rng: np.random.Generator) -> np.ndarray:
    return generate_categorical(n, COUNTRY_WEIGHTS, rng)


def generate_ip_country(billing_country: np.ndarray, risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(billing_country)
    risk_score = np.asarray(risk_score, dtype="float64")
    match_p = IP_COUNTRY_MATCH_BASE_P + risk_score * (IP_COUNTRY_MATCH_HIGH_RISK_P - IP_COUNTRY_MATCH_BASE_P)
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


SHIPPING_MISMATCH_BASE_P = 0.05
SHIPPING_MISMATCH_HIGH_RISK_P = 0.15


def generate_shipping_billing_mismatch(risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_on_risk(risk_score, SHIPPING_MISMATCH_BASE_P, SHIPPING_MISMATCH_HIGH_RISK_P, rng)


FAILED_ATTEMPTS_BASE_LAMBDA = 0.15
FAILED_ATTEMPTS_HIGH_RISK_LAMBDA = 0.6


def generate_failed_payment_attempts_24h(risk_score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    risk_score = np.asarray(risk_score, dtype="float64")
    lam = FAILED_ATTEMPTS_BASE_LAMBDA + risk_score * (FAILED_ATTEMPTS_HIGH_RISK_LAMBDA - FAILED_ATTEMPTS_BASE_LAMBDA)
    return rng.poisson(lam).astype("int16")


from pathlib import Path

INPUT_CSV_PATH = "Data/PS_20174392719_1491204439457_log.csv"
OUTPUT_PARQUET_PATH = "data/processed/transactions_synthetic.parquet"
OUTPUT_SAMPLE_CSV_PATH = "data/processed/transactions_synthetic_sample.csv"
SAMPLE_SIZE = 5000

def generate_all_synthetic_fields(
    df: pd.DataFrame,
    seed: int = 42,
    train_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Label-free: isFraud is never read by any generation function below.
    Every conditional field is driven by compute_risk_proxy(), a composite
    score built only from observable transaction attributes (type, amount,
    hour_of_day) that are available for a brand-new transaction at real-time
    scoring time - see compute_risk_proxy() docstring for why balance columns
    are deliberately excluded from the proxy.

    train_mask: optional boolean array (same length as df), True for rows
    that belong to the training split. When provided, the amount_percentile
    component of compute_risk_proxy is FIT on train_mask rows only (via
    fit_amount_percentile_reference) and then APPLIED to every row -
    preventing the train/test leakage that fitting on the full dataset would
    cause. When omitted (None), the reference is fit on the full input,
    which is only appropriate when this call's input has no further
    train/test split downstream.

    Team decision: the train_mask passed in by main() comes from the SHARED
    60/20/20 split manifest (see split_manifest.py), not a locally-drawn
    split - the same manifest clean_transactions.py uses for Tukey fences
    and that Model Development (Module 5) is expected to reuse, so "train"
    means the same set of rows everywhere in the pipeline.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    out = df.copy()

    out["hour_of_day"] = generate_hour_of_day(df["step"])
    out["is_night_transaction"] = generate_is_night_transaction(out["hour_of_day"])

    if train_mask is not None:
        amount_percentile_reference = fit_amount_percentile_reference(
            df.loc[train_mask, "type"], df.loc[train_mask, "amount"]
        )
    else:
        amount_percentile_reference = None
    risk_score = compute_risk_proxy(
        df["type"], df["amount"], out["hour_of_day"], amount_percentile_reference=amount_percentile_reference
    )

    out["customer_account_age_days"] = generate_account_age_days(risk_score, rng)

    device_pool = build_device_pool(size=DEVICE_POOL_SIZE, seed=seed)
    out["device_id"], out["new_device_flag"] = generate_device_id_and_new_device_flag(
        df["nameOrig"], risk_score, device_pool, rng, NEW_DEVICE_FLAG_BASE_P, NEW_DEVICE_FLAG_HIGH_RISK_P
    )
    out["browser"] = generate_categorical(n, BROWSER_WEIGHTS, rng)
    out["device_type"] = generate_categorical(n, DEVICE_TYPE_WEIGHTS, rng)

    out["billing_country"] = generate_billing_country(n, rng)
    out["ip_country"] = generate_ip_country(out["billing_country"].to_numpy(), risk_score, rng)
    out["ip_billing_distance_km"] = generate_ip_billing_distance_km(
        out["ip_country"].to_numpy(), out["billing_country"].to_numpy()
    )
    out["ip_billing_country_mismatch"] = out["ip_country"] != out["billing_country"]

    out["shipping_billing_mismatch"] = generate_shipping_billing_mismatch(risk_score, rng)
    out["failed_payment_attempts_24h"] = generate_failed_payment_attempts_24h(risk_score, rng)

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
    df = pd.read_csv(csv_path, dtype=dtype)
    required_columns = set(dtype.keys())
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return df


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
    from data_generation.split_manifest import get_or_create_split_manifest, train_row_mask

    df = load_raw_transactions()
    # Fit amount_percentile on the shared train split, not the full dataset,
    # so this enrichment stage itself doesn't leak test-row amounts into the
    # reference distribution used to score every row - see
    # fit_amount_percentile_reference() and split_manifest.py. This is the
    # SAME manifest clean_transactions.py and Model Development use, created
    # once and reused (not redrawn) on every subsequent run.
    manifest = get_or_create_split_manifest(len(df))
    train_mask = train_row_mask(manifest, len(df))
    result = generate_all_synthetic_fields(df, seed=42, train_mask=train_mask)
    Path(OUTPUT_PARQUET_PATH).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    sample = build_stratified_sample(result, sample_size=SAMPLE_SIZE, seed=42)
    sample.to_csv(OUTPUT_SAMPLE_CSV_PATH, index=False)
    print(f"Wrote {len(result)} rows to {OUTPUT_PARQUET_PATH}")
    print(f"Wrote {len(sample)} sample rows to {OUTPUT_SAMPLE_CSV_PATH}")


if __name__ == "__main__":
    main()
