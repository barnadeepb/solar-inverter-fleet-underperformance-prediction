"""Cross-plant generalization check.

Every other benchmark trains and tests within Plant 1. Here we train on all
of Plant 1 and test, zero-shot, on Plant 2 -- a genuine domain-shift
evaluation (different inverter hardware, different site/microclimate)
rather than a same-plant chronological split. A model that only memorized
Plant 1's specific weather-to-power mapping should degrade sharply here;
one that has learned the underlying physical relationship should not.
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import add_time_features, daylight_mask, get_feature_matrix, load_plant
from models_classical import get_classical_models

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    plant1 = add_time_features(load_plant(1))
    plant2 = add_time_features(load_plant(2))

    plant1_day = plant1[daylight_mask(plant1)].reset_index(drop=True)
    plant2_day = plant2[daylight_mask(plant2)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(plant1_day)
    X_test, y_test = get_feature_matrix(plant2_day)

    results = {}
    for name, model in get_classical_models(random_state=42).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = _metrics(y_test, pred)

    out_path = RESULTS_DIR / "metrics" / "cross_plant_generalization.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "train": "Plant 1 (all 34 days)",
                "test": "Plant 2 (all days, zero-shot)",
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "results": results,
            },
            f,
            indent=2,
        )

    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{name:20s} RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
