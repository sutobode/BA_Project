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
