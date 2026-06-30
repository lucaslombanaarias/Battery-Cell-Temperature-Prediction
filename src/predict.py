"""
predict.py -- query the trained surrogate the way the design team would.

The whole point of the model is to replace a minutes-to-hours CFD solve with a
millisecond lookup. This script makes that concrete:

  1. Predicts cell temperature for a few labeled operating points -- a tiny
     "design screen" spanning a cool, well-cooled edge cell to a hot,
     poorly-ventilated center cell -- so you can see the model spread
     temperatures sensibly.
  2. Times the queries (single-query latency and a 1,000-point bulk screen) so
     the "milliseconds, not minutes" claim is measured, not asserted.

Loads the model saved by train.py; run `python src/train.py` first if it is
missing. Every number printed is from the synthetic-data model -- see README.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

import generate_data as gd  # reuse the same feature domain for the bulk screen

MODEL_PATH = "models/rf_model.joblib"

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

# Labeled operating points -- same columns/order the model was trained on. All
# values sit inside the training ranges (the forest does not extrapolate).
SCENARIOS = [
    ("Edge cell, light load, strong airflow",
     dict(ambient_temp_c=22, discharge_current_a=8,  state_of_charge_pct=85,
          internal_resistance_mohm=14, airflow_speed_mps=7.0, cell_position=0,
          elapsed_load_time_min=30, initial_temp_c=22)),
    ("Mid-pack cell, nominal load",
     dict(ambient_temp_c=30, discharge_current_a=20, state_of_charge_pct=60,
          internal_resistance_mohm=25, airflow_speed_mps=3.0, cell_position=5,
          elapsed_load_time_min=45, initial_temp_c=28)),
    ("Center cell, heavy load, weak airflow",
     dict(ambient_temp_c=35, discharge_current_a=36, state_of_charge_pct=20,
          internal_resistance_mohm=42, airflow_speed_mps=1.0, cell_position=9,
          elapsed_load_time_min=60, initial_temp_c=34)),
    ("Hot ambient, sustained heavy load",
     dict(ambient_temp_c=44, discharge_current_a=30, state_of_charge_pct=40,
          internal_resistance_mohm=33, airflow_speed_mps=2.0, cell_position=7,
          elapsed_load_time_min=55, initial_temp_c=38)),
]


def _load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)
    # Score on a single thread: for these batch sizes the worker-spawn overhead
    # dwarfs any parallel speedup, so this is both faster and the honest latency
    # an online single-query path would see.
    model.n_jobs = 1
    return model


def _single_query_ms(model, X, repeats=200):
    """Mean wall-clock latency of predicting one operating point at a time."""
    row = X.iloc[[0]]
    model.predict(row)  # warm up (first call pays one-time overhead)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(row)
    return (time.perf_counter() - t0) / repeats * 1000.0


def _bulk_screen_ms(model, n=1000, seed=0):
    """Wall-clock time to score n random candidate operating points at once."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ambient_temp_c": rng.uniform(*gd.AMBIENT_RANGE, n),
        "discharge_current_a": rng.uniform(*gd.CURRENT_RANGE, n),
        "state_of_charge_pct": rng.uniform(*gd.SOC_RANGE, n),
        "internal_resistance_mohm": rng.uniform(*gd.RESISTANCE_RANGE, n),
        "airflow_speed_mps": rng.uniform(*gd.AIRFLOW_RANGE, n),
        "cell_position": rng.integers(0, 10, n),
        "elapsed_load_time_min": rng.uniform(*gd.LOAD_TIME_RANGE, n),
        "initial_temp_c": rng.uniform(*gd.INITIAL_TEMP_RANGE, n),
    })[FEATURES]
    model.predict(X)  # warm up
    t0 = time.perf_counter()
    model.predict(X)
    return (time.perf_counter() - t0) * 1000.0


def main():
    model = _load_model()
    X = pd.DataFrame([d for _, d in SCENARIOS])[FEATURES]
    preds = model.predict(X)

    print("=" * 60)
    print("BATTERY CELL TEMPERATURE -- SURROGATE INFERENCE DEMO")
    print("(synthetic-data model; see README)")
    print("=" * 60)
    print(f"Loaded {MODEL_PATH}  ({model.n_estimators} trees)\n")

    print(f"{'Operating point':44s}{'Pred. cell temp':>16s}")
    print("-" * 60)
    for (label, _), t in zip(SCENARIOS, preds):
        print(f"{label:44s}{t:11.1f} degC")

    n_bulk = 1000
    single = _single_query_ms(model, X)
    bulk = _bulk_screen_ms(model, n=n_bulk)
    print("\nLatency")
    print("-" * 60)
    print(f"{'Single query (mean of 200)':44s}{single:9.2f} ms")
    print(f"{'Bulk screen of 1,000 points':44s}{bulk:9.2f} ms")
    print(f"{'  -> per point':44s}{bulk / n_bulk * 1000.0:9.2f} us")
    print("\nReplacing a minutes-to-hours CFD solve with a query like this is")
    print("the entire point of the surrogate.")


if __name__ == "__main__":
    main()
