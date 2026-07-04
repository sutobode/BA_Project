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


def generate_conditional_bernoulli(
    is_fraud: np.ndarray, base_p: float, fraud_p: float, rng: np.random.Generator
) -> np.ndarray:
    p = np.where(is_fraud == 1, fraud_p, base_p)
    return rng.binomial(1, p).astype(bool)


NEW_DEVICE_FLAG_BASE_P = 0.04
NEW_DEVICE_FLAG_FRAUD_P = 0.12


def generate_new_device_flag(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_bernoulli(is_fraud, NEW_DEVICE_FLAG_BASE_P, NEW_DEVICE_FLAG_FRAUD_P, rng)


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


SHIPPING_MISMATCH_BASE_P = 0.05
SHIPPING_MISMATCH_FRAUD_P = 0.15


def generate_shipping_billing_mismatch(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return generate_conditional_bernoulli(is_fraud, SHIPPING_MISMATCH_BASE_P, SHIPPING_MISMATCH_FRAUD_P, rng)


FAILED_ATTEMPTS_BASE_LAMBDA = 0.15
FAILED_ATTEMPTS_FRAUD_LAMBDA = 0.6


def generate_failed_payment_attempts_24h(is_fraud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lam = np.where(is_fraud == 1, FAILED_ATTEMPTS_FRAUD_LAMBDA, FAILED_ATTEMPTS_BASE_LAMBDA)
    return rng.poisson(lam).astype("int16")


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
