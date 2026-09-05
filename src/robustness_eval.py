"""Robustness of the expected-output model under degraded sensor input.

Real weather sensors drift, drop out, or get miscalibrated. A model that
only performs well on clean data is a weak basis for an operational
underperformance-detection system, since noisy input could itself produce
a false "underperforming inverter" alarm. We measure test RMSE as
Gaussian noise is added to the weather sensor readings, and separately as
an increasing fraction of readings are replaced with a mean-imputed
missing value (simulating sensor dropout).
"""

import json
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import chronological_split, daylight_mask, get_feature_matrix, load_plant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
MISSING_FRACTIONS = [0.0, 0.05, 0.10, 0.20]
SENSOR_COLS_IDX = [0, 1, 2]  # AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    rng = np.random.default_rng(42)
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(train_day)
    X_test, y_test = get_feature_matrix(test_day)
    feature_std = X_train[:, SENSOR_COLS_IDX].std(axis=0)
    feature_mean = X_train[:, SENSOR_COLS_IDX].mean(axis=0)

    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    noise_results = {}
    for level in NOISE_LEVELS:
        X_noisy = X_test.copy()
        noise = rng.normal(0, level * feature_std, size=(len(X_test), len(SENSOR_COLS_IDX)))
        X_noisy[:, SENSOR_COLS_IDX] = X_noisy[:, SENSOR_COLS_IDX] + noise
        pred = model.predict(X_noisy)
        noise_results[f"noise_std_frac_{level}"] = _metrics(y_test, pred)

    missing_results = {}
    for frac in MISSING_FRACTIONS:
        X_missing = X_test.copy()
        mask = rng.random(size=(len(X_test), len(SENSOR_COLS_IDX))) < frac
        for j, col_idx in enumerate(SENSOR_COLS_IDX):
            X_missing[mask[:, j], col_idx] = feature_mean[j]
        pred = model.predict(X_missing)
        missing_results[f"missing_frac_{frac}"] = _metrics(y_test, pred)

    out_path = RESULTS_DIR / "metrics" / "robustness.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "random_forest",
                "gaussian_noise_on_sensor_features": noise_results,
                "mean_imputed_missing_sensor_features": missing_results,
            },
            f,
            indent=2,
        )

    print("noise robustness:")
    for k, m in noise_results.items():
        print(f"  {k:25s} RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}")
    print("missing-data robustness:")
    for k, m in missing_results.items():
        print(f"  {k:25s} RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
