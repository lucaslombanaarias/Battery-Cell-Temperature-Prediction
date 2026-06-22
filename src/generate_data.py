"""
Synthetic battery-cell thermal data generator.

The generative model is a first-order lumped-capacitance thermal model -- the
standard textbook description of a body heating under a constant load while
losing heat to its surroundings by convection. Every term is physically
motivated so the downstream ML story is defensible.

----------------------------------------------------------------------------
PHYSICS
----------------------------------------------------------------------------
1. Heat generation (Joule heating):
       Q_gen = I^2 * R_eff                                   [W]
   Internal resistance rises as the cell depletes, so the *effective*
   resistance that produces heat is modeled as:
       R_eff = R_nominal * (1 + soc_gain * (1 - SOC/100))
   i.e. a cell at low state-of-charge dissipates more heat for the same
   current (a real, if secondary, effect).

2. Heat removal (Newton's law of cooling):
       Q_out = hA * (T_cell - T_ambient)                     [W]
   The conductance to ambient (hA, in W/degC) grows with forced airflow and
   shrinks for poorly-exposed interior cells:
       hA = base_hA * position_factor * (1 + k_air * airflow)
       position_factor = 1 - pos_penalty * (cell_position / 9)
   position 0 = edge cell (best exposed), position 9 = center cell (worst).

3. Steady-state temperature (set Q_gen = Q_out):
       T_steady = T_ambient + Q_gen / hA

4. Transient approach to steady state (first-order lumped capacitance):
       approach = 1 - exp(-t / tau)
       T(t)     = T_initial + (T_steady - T_initial) * approach
   tau is the thermal time constant. At t=0 the cell sits at T_initial; as
   t -> infinity it approaches T_steady. This makes both elapsed load time
   and initial temperature physically meaningful.

5. Sensor noise:
   Add Gaussian measurement noise to mimic a real thermistor/thermocouple.

----------------------------------------------------------------------------
CALIBRATION NOTE (stated honestly):
   The absolute electrical magnitudes (per-cell watts) are synthetic lumped
   values; the constants below were chosen so cell temperatures land in a
   physically plausible Li-ion operating band (roughly ambient up to ~90 degC,
   below thermal-runaway territory). The *functional relationships* are the
   physics; absolute calibration would come from real bench data.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 12_000

# --- Generative-model constants (lumped/synthetic; see CALIBRATION NOTE) ---
SOC_RESISTANCE_GAIN = 0.30   # low-SOC resistance bump (dimensionless)
BASE_HA = 1.6                # base cell-to-ambient conductance [W/degC]
K_AIR = 0.25                 # forced-convection gain per m/s of airflow
POSITION_PENALTY = 0.50      # center cell loses this fraction of conductance
TAU_MIN = 25.0               # thermal time constant [min]
NOISE_STD_C = 1.2            # sensor noise standard deviation [degC]

# --- Feature sampling ranges ---
AMBIENT_RANGE = (15.0, 45.0)        # degC
CURRENT_RANGE = (5.0, 38.0)         # A (per-cell / lumped)
SOC_RANGE = (10.0, 100.0)           # %
RESISTANCE_RANGE = (12.0, 45.0)     # mOhm
AIRFLOW_RANGE = (0.5, 8.0)          # m/s
POSITION_RANGE = (0, 9)             # index: 0=edge ... 9=center
LOAD_TIME_RANGE = (10.0, 60.0)      # min (bench tests run long enough to warm up)
INITIAL_TEMP_RANGE = (18.0, 40.0)   # degC


def generate_data(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ambient_temp_c = rng.uniform(*AMBIENT_RANGE, n_samples)
    discharge_current_a = rng.uniform(*CURRENT_RANGE, n_samples)
    state_of_charge_pct = rng.uniform(*SOC_RANGE, n_samples)
    internal_resistance_mohm = rng.uniform(*RESISTANCE_RANGE, n_samples)
    airflow_speed_mps = rng.uniform(*AIRFLOW_RANGE, n_samples)
    cell_position = rng.integers(POSITION_RANGE[0], POSITION_RANGE[1] + 1, n_samples)
    elapsed_load_time_min = rng.uniform(*LOAD_TIME_RANGE, n_samples)
    initial_temp_c = rng.uniform(*INITIAL_TEMP_RANGE, n_samples)

    # 1. Heat generation: Q = I^2 * R_eff  (R in ohms; low SOC raises R_eff)
    r_eff_ohm = (internal_resistance_mohm / 1000.0) * (
        1.0 + SOC_RESISTANCE_GAIN * (1.0 - state_of_charge_pct / 100.0)
    )
    q_gen_w = discharge_current_a ** 2 * r_eff_ohm

    # 2. Conductance to ambient: better with airflow, worse for center cells
    position_factor = 1.0 - POSITION_PENALTY * (cell_position / 9.0)
    hA = BASE_HA * position_factor * (1.0 + K_AIR * airflow_speed_mps)

    # 3. Steady-state temperature (energy balance Q_gen = Q_out)
    t_steady_c = ambient_temp_c + q_gen_w / hA

    # 4. First-order transient approach from the initial temperature.
    #    approach = fraction of the way from T_initial to T_steady reached by
    #    time t. At t=0 -> T_initial; as t -> inf -> T_steady.
    approach = 1.0 - np.exp(-elapsed_load_time_min / TAU_MIN)
    cell_temperature_c = initial_temp_c + (t_steady_c - initial_temp_c) * approach

    # 5. Sensor measurement noise
    cell_temperature_c = cell_temperature_c + rng.normal(0, NOISE_STD_C, n_samples)

    df = pd.DataFrame({
        "ambient_temp_c": np.round(ambient_temp_c, 2),
        "discharge_current_a": np.round(discharge_current_a, 2),
        "state_of_charge_pct": np.round(state_of_charge_pct, 2),
        "internal_resistance_mohm": np.round(internal_resistance_mohm, 2),
        "airflow_speed_mps": np.round(airflow_speed_mps, 2),
        "cell_position": cell_position,
        "elapsed_load_time_min": np.round(elapsed_load_time_min, 2),
        "initial_temp_c": np.round(initial_temp_c, 2),
        "cell_temperature_c": np.round(cell_temperature_c, 2),
    })
    return df


def _summarize(df: pd.DataFrame) -> None:
    t = df["cell_temperature_c"]
    pct = t.quantile([0.50, 0.95, 0.99, 0.999]).round(1)
    print(f"Generated {len(df)} samples -> data/battery_thermal_data.csv")
    print(
        f"Cell temp (degC): min={t.min():.1f}  median={pct[0.50]}  "
        f"p95={pct[0.95]}  p99={pct[0.99]}  p99.9={pct[0.999]}  max={t.max():.1f}"
    )
    # Physical-plausibility guard: warn if temps stray into runaway territory.
    if t.max() > 105:
        print(f"  WARNING: max temp {t.max():.1f} degC exceeds plausible Li-ion band.")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv("data/battery_thermal_data.csv", index=False)
    _summarize(df)
