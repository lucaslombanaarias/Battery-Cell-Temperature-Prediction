"""
hotspot_map.py -- turn the model into the design artifact it was built for.

The surrogate's whole purpose is to flag where cells run hot so cooling and
layout can be targeted. This script sweeps the two design-controllable levers --
airflow and cell position -- across a grid while holding a fixed heavy load
constant, queries the model at every point, and renders the predicted-temperature
field as a heat map. The result is a literal hot-spot map: it shows that interior
cells under weak airflow are the danger zone, which is the quantitative basis for
"the model guided heat-sink placement and cell-layout decisions."

Loads the model saved by train.py; run `python src/train.py` first if missing.
Every number is from the synthetic-data model -- see README.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

MODEL_PATH = "models/rf_model.joblib"
OUT_PATH = "reports/hotspot_map.png"

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

# Fixed heavy-but-in-range operating point. The two swept features (airflow,
# cell_position) are overwritten per grid cell; the rest stay at these values.
FIXED = {
    "ambient_temp_c": 35.0,
    "discharge_current_a": 32.0,     # heavy, sustained load
    "state_of_charge_pct": 30.0,     # low SOC -> higher effective resistance
    "internal_resistance_mohm": 35.0,
    "elapsed_load_time_min": 60.0,   # fully warmed up
    "initial_temp_c": 35.0,
}

POSITIONS = np.arange(0, 10)              # 0 = edge ... 9 = center
AIRFLOWS = np.linspace(0.5, 8.0, 40)      # m/s, low (hot) -> high (cool)


def _load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    model = joblib.load(MODEL_PATH)
    model.n_jobs = 1
    return model


def _predict_grid(model):
    """Predicted cell temperature on the (airflow x position) grid."""
    aa, pp = np.meshgrid(AIRFLOWS, POSITIONS, indexing="ij")
    grid = pd.DataFrame({k: np.full(aa.size, v) for k, v in FIXED.items()})
    grid["airflow_speed_mps"] = aa.ravel()
    grid["cell_position"] = pp.ravel()
    preds = model.predict(grid[FEATURES])
    return preds.reshape(aa.shape)  # rows = airflow, cols = position


def main():
    model = _load_model()
    temps = _predict_grid(model)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        temps, aspect="auto", origin="lower", cmap="inferno",
        extent=[POSITIONS.min() - 0.5, POSITIONS.max() + 0.5,
                AIRFLOWS.min(), AIRFLOWS.max()],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted cell temperature (degC)")

    # Mark the predicted hottest cell on the grid.
    r, c = np.unravel_index(np.argmax(temps), temps.shape)
    ax.scatter(POSITIONS[c], AIRFLOWS[r], marker="x", s=90, color="cyan",
               linewidths=2, label=f"hottest: {temps[r, c]:.1f} degC")

    ax.set(
        xlabel="Cell position (0 = edge / well-cooled, 9 = center / interior)",
        ylabel="Airflow speed (m/s)",
        title="Predicted hot-spot map at a fixed heavy load\n"
              f"(I={FIXED['discharge_current_a']:.0f} A, ambient={FIXED['ambient_temp_c']:.0f} degC, "
              f"SOC={FIXED['state_of_charge_pct']:.0f}%, t={FIXED['elapsed_load_time_min']:.0f} min)",
    )
    ax.set_xticks(POSITIONS)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Hot-spot map saved -> {OUT_PATH}")
    print(f"Predicted range over the grid: {temps.min():.1f} - {temps.max():.1f} degC")
    print(f"Hottest: position {POSITIONS[c]} (interior) at {AIRFLOWS[r]:.1f} m/s airflow "
          f"-> {temps[r, c]:.1f} degC")


if __name__ == "__main__":
    main()
