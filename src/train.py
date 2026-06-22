"""
Train a RandomForestRegressor to predict battery cell temperature.

Design choices (interview-defensible):

  * 80/20 train/test split. With 12,000 rows this leaves ~2,400 held-out
    samples -- enough for stable error estimates without starving training.
    The split is seeded so the held-out set is identical on every run.

  * Random forest. The target is a nonlinear function of interacting inputs
    (Joule heating I^2*R, divided by an airflow/position-dependent
    conductance, approached transiently over time). Trees capture those
    interactions with no feature engineering, need no feature scaling
    (they are invariant to monotonic transforms of individual inputs), and
    expose interpretable feature importances.

  * Hyperparameters -- kept simple and explainable, not grid-searched:
      n_estimators=300   more trees -> lower-variance averaging; returns
                         flatten well before 300, so it is a safe default.
      min_samples_leaf=5 each leaf averages >=5 cells, which smooths the
                         +/-1.2 degC sensor noise instead of memorizing it.
                         This is the single regularization knob; max_depth is
                         left unbounded and controlled through the leaf size.
      oob_score=True     free out-of-bag generalization estimate (each tree
                         scores the ~1/3 of samples it did not see).

  * The train and test splits are written to data/ so evaluate.py scores the
    exact same held-out rows and runs cross-validation on training data only.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

SEED = 42
TEST_SIZE = 0.20

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

DATA_PATH = "data/battery_thermal_data.csv"
MODEL_PATH = "models/rf_model.joblib"


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Persist the exact splits so evaluation is fully reproducible.
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    train_df.to_csv("data/train_set.csv", index=False)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv("data/test_set.csv", index=False)

    print(f"Trained on {len(X_train)} samples; held-out test set: {len(X_test)}")
    print(f"Out-of-bag R^2 (training-time generalization estimate): {model.oob_score_:.4f}")
    print(f"Model saved -> {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
