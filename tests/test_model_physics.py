"""
Physics-sanity tests for the trained model.

These test the *model*, not just the plumbing: a correctly-fit surrogate must
reproduce the monotonic relationships baked into the generator. Each test holds
a batch of random operating points fixed, sweeps one feature low -> high, and
asserts the mean predicted temperature moves in the physically-correct
direction by a clear margin (in degC, well above forest noise).

A small, fast forest is trained once for the module -- this is independent of
the committed 42 MB model (which is gitignored and absent in CI).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import generate_data as gd

FEATURES = [
    "ambient_temp_c",
    "discharge_current_a",
    "state_of_charge_pct",
    "internal_resistance_mohm",
    "airflow_speed_mps",
    "cell_position",
    "elapsed_load_time_min",
    "initial_temp_c",
]
TARGET = "cell_temperature_c"
MARGIN_C = 1.0  # required temperature shift, in degC


@pytest.fixture(scope="module")
def model():
    df = gd.generate_data(n_samples=6000, seed=0)
    m = RandomForestRegressor(
        n_estimators=80, min_samples_leaf=5, random_state=0, n_jobs=1
    )
    m.fit(df[FEATURES], df[TARGET])
    return m


def _baselines(n=200, seed=123):
    """A batch of representative operating points across the sampled domain."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ambient_temp_c": rng.uniform(*gd.AMBIENT_RANGE, n),
        "discharge_current_a": rng.uniform(*gd.CURRENT_RANGE, n),
        "state_of_charge_pct": rng.uniform(*gd.SOC_RANGE, n),
        "internal_resistance_mohm": rng.uniform(*gd.RESISTANCE_RANGE, n),
        "airflow_speed_mps": rng.uniform(*gd.AIRFLOW_RANGE, n),
        "cell_position": rng.integers(0, 10, n),
        "elapsed_load_time_min": rng.uniform(*gd.LOAD_TIME_RANGE, n),
        "initial_temp_c": rng.uniform(*gd.INITIAL_TEMP_RANGE, n),
    })
    return df[FEATURES]


def _mean_pred_with(model, base, **overrides):
    """Mean prediction over the batch with some feature(s) pinned to a value."""
    X = base.copy()
    for key, value in overrides.items():
        X[key] = value
    return float(model.predict(X).mean())


def test_higher_discharge_current_raises_temp(model):
    base = _baselines()
    low = _mean_pred_with(model, base, discharge_current_a=8.0)
    high = _mean_pred_with(model, base, discharge_current_a=36.0)
    assert high > low + MARGIN_C


def test_more_airflow_lowers_temp(model):
    base = _baselines()
    weak = _mean_pred_with(model, base, airflow_speed_mps=1.0)
    strong = _mean_pred_with(model, base, airflow_speed_mps=7.5)
    assert strong < weak - MARGIN_C


def test_center_cell_hotter_than_edge(model):
    base = _baselines()
    edge = _mean_pred_with(model, base, cell_position=0)
    center = _mean_pred_with(model, base, cell_position=9)
    assert center > edge + MARGIN_C


def test_higher_ambient_raises_temp(model):
    base = _baselines()
    cool = _mean_pred_with(model, base, ambient_temp_c=18.0)
    hot = _mean_pred_with(model, base, ambient_temp_c=43.0)
    assert hot > cool + MARGIN_C
