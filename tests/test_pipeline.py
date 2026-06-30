"""
End-to-end integration test: generate -> train -> evaluate.

Runs the real scripts on a small dataset inside a temp directory and asserts
the pipeline produces its artifacts -- including reports/metrics.json -- and
that the model beats the trivial baseline. Marked `slow` because it trains a
model; CI runs it.
"""
import json
import os

import pytest

import generate_data as gd
import train
import evaluate


@pytest.mark.slow
def test_end_to_end_writes_metrics(tmp_path, monkeypatch):
    # The scripts use paths relative to the working directory.
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    # Small dataset so the full pipeline (incl. CV + permutation) stays quick.
    gd.generate_data(n_samples=1500, seed=0).to_csv(
        "data/battery_thermal_data.csv", index=False
    )

    train.train()
    assert os.path.exists("models/rf_model.joblib")
    assert os.path.exists("data/train_set.csv")
    assert os.path.exists("data/test_set.csv")

    evaluate.evaluate()

    metrics_path = os.path.join("reports", "metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        metrics = json.load(f)

    for key in ("test", "baseline_mean", "cv5_train",
                "feature_importance_permutation"):
        assert key in metrics

    assert metrics["n_train"] == 1200
    assert metrics["n_test"] == 300
    assert 0.0 < metrics["test"]["r2"] <= 1.0
    # The whole point: the model must beat predicting the mean.
    assert metrics["test"]["rmse"] < metrics["baseline_mean"]["rmse"]

    for png in ("predicted_vs_actual.png", "residuals.png",
                "feature_importances.png"):
        assert os.path.exists(os.path.join("reports", png))
