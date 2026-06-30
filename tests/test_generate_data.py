"""
Plumbing tests for the synthetic data generator.

These check the *code* contract -- shape, column names, value ranges, and
determinism -- independent of any trained model.
"""
import pandas as pd

import generate_data as gd

EXPECTED_COLUMNS = [
    "ambient_temp_c",
    "discharge_current_a",
    "state_of_charge_pct",
    "internal_resistance_mohm",
    "airflow_speed_mps",
    "cell_position",
    "elapsed_load_time_min",
    "initial_temp_c",
    "cell_temperature_c",
]


def test_shape_and_columns():
    df = gd.generate_data(n_samples=500, seed=0)
    assert len(df) == 500
    assert list(df.columns) == EXPECTED_COLUMNS


def test_features_within_sampled_ranges():
    df = gd.generate_data(n_samples=5000, seed=1)
    assert df["ambient_temp_c"].between(*gd.AMBIENT_RANGE).all()
    assert df["discharge_current_a"].between(*gd.CURRENT_RANGE).all()
    assert df["state_of_charge_pct"].between(*gd.SOC_RANGE).all()
    assert df["internal_resistance_mohm"].between(*gd.RESISTANCE_RANGE).all()
    assert df["airflow_speed_mps"].between(*gd.AIRFLOW_RANGE).all()
    assert df["elapsed_load_time_min"].between(*gd.LOAD_TIME_RANGE).all()
    assert df["initial_temp_c"].between(*gd.INITIAL_TEMP_RANGE).all()
    # cell_position is an integer index 0..9
    assert df["cell_position"].between(0, 9).all()
    assert df["cell_position"].dtype.kind in "iu"


def test_target_in_plausible_li_ion_band():
    df = gd.generate_data(n_samples=5000, seed=2)
    t = df["cell_temperature_c"]
    # Above freezing, below thermal-runaway territory (generator guards at 105).
    assert t.min() > 5.0
    assert t.max() < 105.0


def test_generation_is_deterministic_per_seed():
    a = gd.generate_data(n_samples=1000, seed=42)
    b = gd.generate_data(n_samples=1000, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = gd.generate_data(n_samples=1000, seed=42)
    c = gd.generate_data(n_samples=1000, seed=7)
    assert not a.equals(c)
