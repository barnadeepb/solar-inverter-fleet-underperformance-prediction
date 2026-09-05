"""Main regression benchmark: predict AC_POWER from weather for Plant 1.

Evaluation design:
  - Primary split is chronological (first 24 days train / last ~10 days
    test), not a random shuffle, so the reported numbers reflect genuine
    forecasting into an unseen future period rather than interpolation
    between neighboring timestamps of the same day.
  - Each stochastic model (random forest, xgboost, lightgbm, mlp, cnn) is
    refit under 3 random seeds; we report mean +/- std of test metrics.
    Deterministic models (naive mean, linear regression, ridge) have zero
    variance across seeds by construction.
  - Night-time rows (IRRADIATION == 0) are excluded: a model that always
    predicts ~0 at night would otherwise dominate the error metrics with a
    trivially "solved" portion of the data.
"""

import json
import pathlib
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import (
    build_sequences,
    chronological_split,
    daylight_mask,
    get_feature_matrix,
    load_plant,
)
from models_classical import get_classical_models
from models_neural import get_neural_models

SEEDS = [0, 1, 2]
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _aggregate(runs: list) -> dict:
    keys = runs[0].keys()
    return {
        k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
        for k in keys
    }


def run_tabular_models(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    for seed in SEEDS:
        tabular_models = get_classical_models(seed)
        tabular_models["mlp"] = get_neural_models(seed)["mlp"]
        for name, model in tabular_models.items():
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = model.predict(X_test)
            predict_seconds = time.perf_counter() - t0

            run = _metrics(y_test, pred)
            run["fit_seconds"] = fit_seconds
            run["predict_seconds_per_sample"] = predict_seconds / len(X_test)
            results.setdefault(name, []).append(run)
    return {name: _aggregate(runs) for name, runs in results.items()}


def run_cnn(train_day, test_day, window: int = 4) -> dict:
    from models_neural import CNN1DRegressor

    Xtr, ytr = build_sequences(train_day, window=window)
    Xte, yte = build_sequences(test_day, window=window)

    runs = []
    for seed in SEEDS:
        model = CNN1DRegressor(epochs=40, random_state=seed)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model.predict(Xte)
        predict_seconds = time.perf_counter() - t0

        run = _metrics(yte, pred)
        run["fit_seconds"] = fit_seconds
        run["predict_seconds_per_sample"] = predict_seconds / len(Xte)
        runs.append(run)
    return {"cnn_1d_sequence": _aggregate(runs)}


def main():
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)

    print(f"train daylight rows: {len(train_day)}, test daylight rows: {len(test_day)}")

    X_train, y_train = get_feature_matrix(train_day)
    X_test, y_test = get_feature_matrix(test_day)

    results = run_tabular_models(X_train, y_train, X_test, y_test)
    results.update(run_cnn(train_day, test_day))

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics" / "regression_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "Plant 1",
                "split": "chronological (first 24 days train, remaining days test)",
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "seeds": SEEDS,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")

    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse"]["mean"]):
        print(f"{name:20s} RMSE={m['rmse']['mean']:.2f}+-{m['rmse']['std']:.2f}  R2={m['r2']['mean']:.4f}")


if __name__ == "__main__":
    main()
