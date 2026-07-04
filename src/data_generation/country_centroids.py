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
