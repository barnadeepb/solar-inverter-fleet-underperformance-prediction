"""Generate the two headline figures for the paper from the results
already written to results/metrics/ -- kept as a separate script (rather
than inline in train.py/anomaly_detection.py) so figures can be
regenerated any time without rerunning the underlying experiments.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def plot_model_comparison():
    with open(RESULTS_DIR / "metrics" / "regression_benchmark.json") as f:
        bench = json.load(f)
    results = bench["results"]

    names = sorted(results.keys(), key=lambda n: results[n]["rmse"]["mean"])
    rmse_means = [results[n]["rmse"]["mean"] for n in names]
    rmse_stds = [results[n]["rmse"]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if n == "naive_mean" else "#2b6cb0" for n in names]
    ax.barh(names, rmse_means, xerr=rmse_stds, color=colors)
    ax.set_xlabel("Test RMSE (kW), chronological split")
    ax.set_title("Plant 1 AC power prediction: model comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_inverter_ranking():
    df = pd.read_csv(RESULTS_DIR / "metrics" / "inverter_performance_ratios.csv")
    df = df.sort_values("mean_ratio")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if flagged else "#2b6cb0" for flagged in df["flagged_underperforming"]]
    ax.barh(df["SOURCE_KEY"], df["mean_ratio"] * 100, color=colors)
    ax.axvline(100, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean performance ratio (actual / weather-expected AC power, %)")
    ax.set_title("Plant 1 inverter fleet: performance ratio (34 days)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "inverter_performance_ranking.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_model_comparison()
    plot_inverter_ranking()
    print(f"wrote figures to {RESULTS_DIR / 'figures'}")
