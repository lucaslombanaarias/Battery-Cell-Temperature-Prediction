"""
Evaluate the trained model.

Produces, all from the same seeded split:
  * Held-out test metrics: RMSE, MAE, R^2.
  * A trivial baseline (predict the training mean) for comparison, so the
    model's value is quantified rather than assumed.
  * 5-fold cross-validation on the TRAINING set only (RMSE and R^2,
    mean +/- std) -- the test set stays untouched until the final score.
  * Two feature-importance views:
       - Impurity-based (MDI): fast, but biased toward high-cardinality
         continuous features.
       - Permutation importance on the held-out test set: model-agnostic and
         unbiased; the more defensible number. Reported with std over repeats.
  * Predicted-vs-actual, residual, and feature-importance plots.
  * A machine-readable reports/metrics.json.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
import joblib

SEED = 42
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
REPORTS_DIR = "reports"


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate():
    train_df = pd.read_csv("data/train_set.csv")
    test_df = pd.read_csv("data/test_set.csv")
    model = joblib.load("models/rf_model.joblib")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    y_pred = model.predict(X_test)

    # --- Held-out test metrics ---
    rmse = _rmse(y_test, y_pred)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("=" * 56)
    print("HELD-OUT TEST METRICS")
    print("=" * 56)
    print(f"  RMSE : {rmse:.3f} degC")
    print(f"  MAE  : {mae:.3f} degC")
    print(f"  R^2  : {r2:.4f}")
    if getattr(model, "oob_score_", None) is not None:
        print(f"  (training-time out-of-bag R^2: {model.oob_score_:.4f})")

    # --- Trivial baseline: predict the training mean ---
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    base_rmse = _rmse(y_test, y_base)
    base_mae = float(mean_absolute_error(y_test, y_base))
    base_r2 = float(r2_score(y_test, y_base))
    rmse_gain = (1 - rmse / base_rmse) * 100

    print("\nBASELINE (predict training mean):")
    print(f"  RMSE : {base_rmse:.3f} degC   MAE : {base_mae:.3f} degC   R^2 : {base_r2:.4f}")
    print(f"  -> Model cuts RMSE by {rmse_gain:.1f}% vs. this baseline.")

    # --- 5-fold cross-validation on TRAINING data only ---
    cv_rmse = -cross_val_score(
        model, X_train, y_train, cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    print("\n5-FOLD CROSS-VALIDATION (training set):")
    print(f"  RMSE : {cv_rmse.mean():.3f} +/- {cv_rmse.std():.3f} degC")
    print(f"  R^2  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    # --- Feature importances: MDI and permutation (test set) ---
    mdi = model.feature_importances_
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1
    )
    order = np.argsort(perm.importances_mean)[::-1]

    print("\nFEATURE IMPORTANCE (permutation on test set | MDI):")
    for i in order:
        print(f"  {FEATURES[i]:26s} {perm.importances_mean[i]:.4f} +/- "
              f"{perm.importances_std[i]:.4f}   | MDI {mdi[i]:.4f}")

    # --- Plots ---
    os.makedirs(REPORTS_DIR, exist_ok=True)
    _plot_pred_vs_actual(y_test, y_pred, rmse, mae, r2)
    _plot_residuals(y_pred, y_test - y_pred)
    _plot_importances(mdi, perm, order)
    print(f"\nPlots saved -> {REPORTS_DIR}/")

    # --- Machine-readable metrics ---
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test": {"rmse": rmse, "mae": mae, "r2": r2},
        "baseline_mean": {"rmse": base_rmse, "mae": base_mae, "r2": base_r2},
        "rmse_improvement_pct_vs_baseline": float(rmse_gain),
        "oob_r2": float(model.oob_score_) if getattr(model, "oob_score_", None) is not None else None,
        "cv5_train": {
            "rmse_mean": float(cv_rmse.mean()), "rmse_std": float(cv_rmse.std()),
            "r2_mean": float(cv_r2.mean()), "r2_std": float(cv_r2.std()),
        },
        "feature_importance_mdi": {FEATURES[i]: float(mdi[i]) for i in order},
        "feature_importance_permutation": {
            FEATURES[i]: {"mean": float(perm.importances_mean[i]),
                          "std": float(perm.importances_std[i])} for i in order
        },
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {REPORTS_DIR}/metrics.json")


def _plot_pred_vs_actual(y_test, y_pred, rmse, mae, r2):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.25, s=10, color="steelblue", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set(xlabel="Actual cell temperature (degC)",
           ylabel="Predicted cell temperature (degC)",
           title="Predicted vs. Actual (held-out test set)",
           xlim=lims, ylim=lims)
    ax.text(0.05, 0.95, f"R^2 = {r2:.3f}\nRMSE = {rmse:.2f} degC\nMAE = {mae:.2f} degC",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "predicted_vs_actual.png"), dpi=150)
    plt.close(fig)


def _plot_residuals(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.25, s=10, color="steelblue", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set(xlabel="Predicted cell temperature (degC)",
           ylabel="Residual = actual - predicted (degC)",
           title=f"Residuals (held-out test set)  |  std = {residuals.std():.2f} degC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "residuals.png"), dpi=150)
    plt.close(fig)


def _plot_importances(mdi, perm, order):
    labels = [FEATURES[i] for i in order][::-1]  # most important at top
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax1.barh(labels, [perm.importances_mean[i] for i in order][::-1],
             xerr=[perm.importances_std[i] for i in order][::-1],
             color="steelblue")
    ax1.set(xlabel="Mean R^2 drop when shuffled", title="Permutation importance (test set)")
    ax2.barh(labels, [mdi[i] for i in order][::-1], color="darkorange")
    ax2.set(xlabel="Mean decrease in impurity", title="MDI importance (training)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "feature_importances.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
