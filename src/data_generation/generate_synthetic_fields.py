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
