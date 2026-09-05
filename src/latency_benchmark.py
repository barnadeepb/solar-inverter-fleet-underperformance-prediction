"""Single-sample inference latency as a deployability proxy.

train.py already reports batched predict_seconds_per_sample, which favors
models that vectorize well and understates latency for a real-time,
one-reading-at-a-time deployment (e.g. an edge gateway scoring each new
inverter reading as it arrives). This measures wall-clock time for
one-row-at-a-time prediction, repeated many times per model, and reports
median and p95 latency plus a serialized-model-size proxy for memory
footprint.
"""

import json
import pathlib
import pickle
import time

import numpy as np

from data_prep import chronological_split, daylight_mask, get_feature_matrix, load_plant
from models_classical import get_classical_models
from models_neural import get_neural_models

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
N_REPEATS = 500


def _single_row_latencies(model, X: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), size=N_REPEATS)
    latencies = np.empty(N_REPEATS)
    for i, row_idx in enumerate(idx):
        row = X[row_idx : row_idx + 1]
        t0 = time.perf_counter()
        model.predict(row)
        latencies[i] = time.perf_counter() - t0
    return latencies


def main():
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(train_day)
    X_test, _ = get_feature_matrix(test_day)

    models = get_classical_models(random_state=42)
    models["mlp"] = get_neural_models(random_state=42)["mlp"]

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        latencies_ms = _single_row_latencies(model, X_test) * 1000
        try:
            model_size_kb = len(pickle.dumps(model)) / 1024
        except Exception:
            model_size_kb = None
        results[name] = {
            "median_latency_ms": float(np.median(latencies_ms)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "model_size_kb": model_size_kb,
        }

    out_path = RESULTS_DIR / "metrics" / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({"n_repeats": N_REPEATS, "results": results}, f, indent=2)

    for name, m in sorted(results.items(), key=lambda kv: kv[1]["median_latency_ms"]):
        size = f"{m['model_size_kb']:.0f} KB" if m["model_size_kb"] else "n/a"
        print(f"{name:20s} median={m['median_latency_ms']:.3f}ms  p95={m['p95_latency_ms']:.3f}ms  size={size}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
