"""Quantify how much a random train/test split overstates accuracy on this
dataset relative to the chronological split used everywhere else in this
project. Referenced in the paper's discussion of evaluation methodology;
committed here (rather than left as a remembered number) so the claim is
reproducible from this repository.
"""

import json
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split

from data_prep import add_time_features, daylight_mask, get_feature_matrix, load_plant
from models_classical import get_classical_models

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    df = add_time_features(load_plant(1))
    day = df[daylight_mask(df)].reset_index(drop=True)
    X, y = get_feature_matrix(day)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in get_classical_models(random_state=42).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = rmse(y_test, pred)

    with open(RESULTS_DIR / "metrics" / "regression_benchmark.json") as f:
        chrono = json.load(f)["results"]

    out = {"split": "random 80/20, row-wise shuffle", "random_split_rmse": results, "chronological_split_rmse": {
        name: chrono[name]["rmse"]["mean"] for name in results
    }}
    out_path = RESULTS_DIR / "metrics" / "split_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    for name in results:
        c = out["chronological_split_rmse"][name]
        r = results[name]
        pct = (c - r) / c * 100
        print(f"{name:20s} random={r:.2f}  chronological={c:.2f}  chronological is {pct:.1f}% higher")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
