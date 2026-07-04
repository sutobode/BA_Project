import numpy as np
import pandas as pd


def generate_hour_of_day(step: pd.Series) -> pd.Series:
    """step in PaySim = elapsed hours since simulation start (1-indexed)."""
    return ((step - 1) % 24).astype("int16")


def generate_is_night_transaction(hour_of_day: pd.Series) -> pd.Series:
    """Night defined as 00:00-05:59 (business assumption)."""
    return hour_of_day.between(0, 5)
