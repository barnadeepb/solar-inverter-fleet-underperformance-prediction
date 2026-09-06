"""Generate the five figures used in the full paper, each as a matching
SVG (vector, used in the paper and README) and PNG (raster fallback used
inside the Word document, since not every viewer renders embedded SVG).

Kept separate from make_figures.py, which produces the two working
figures used during analysis; this script produces the polished,
paper-ready versions with consistent styling.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    }
)

BLUE = "#2b6cb0"
RED = "#c0392b"
GRAY = "#7f8c8d"


def save_both(fig, name):
    fig.savefig(FIG_DIR / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison():
    with open(RESULTS_DIR / "metrics" / "regression_benchmark.json") as f:
        bench = json.load(f)
    results = bench["results"]
    names_map = {
        "naive_mean": "Naive mean",
        "linear_regression": "Linear regression",
        "ridge": "Ridge",
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "cnn_1d_sequence": "1D-CNN",
        "vertex_automl_tables": "Vertex AI AutoML",
    }
    order = sorted(results.keys(), key=lambda n: results[n]["rmse"]["mean"])
    labels = [names_map.get(n, n) for n in order]
    means = [results[n]["rmse"]["mean"] for n in order]
    stds = [results[n]["rmse"]["std"] for n in order]
    colors = [RED if n == "naive_mean" else BLUE for n in order]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.barh(labels, means, xerr=stds, color=colors, height=0.6)
    ax.set_xlabel("Test RMSE (kW), chronological split")
    ax.set_title("AC power prediction: model comparison (Plant 1)")
    ax.invert_yaxis()
    fig.tight_layout()
    save_both(fig, "fig1_model_comparison")


def fig_inverter_ranking():
    df = pd.read_csv(RESULTS_DIR / "metrics" / "inverter_performance_ratios.csv")
    df = df.sort_values("mean_ratio", ascending=False)
    colors = [RED if flagged else BLUE for flagged in df["flagged_underperforming"]]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh(range(len(df)), df["mean_ratio"] * 100, color=colors, height=0.65)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["SOURCE_KEY"], fontsize=7)
    ax.axvline(100, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean performance ratio, actual / weather-expected AC power (%)")
    ax.set_title("Inverter fleet performance ratio, 34 days (Plant 1)")
    fig.tight_layout()
    save_both(fig, "fig2_inverter_ranking")


def fig_interpretability():
    with open(RESULTS_DIR / "metrics" / "interpretability.json") as f:
        interp = json.load(f)
    shap_pct = interp["mean_abs_shap_value_pct"]
    order = sorted(shap_pct.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ").title() for k, _ in order]
    values = [v for _, v in order]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.barh(labels, values, color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (% of total)")
    ax.set_title("Feature importance for predicted AC power (Plant 1)")
    fig.tight_layout()
    save_both(fig, "fig3_interpretability")


def fig_robustness():
    with open(RESULTS_DIR / "metrics" / "robustness.json") as f:
        rob = json.load(f)

    noise = rob["gaussian_noise_on_sensor_features"]
    missing = rob["mean_imputed_missing_sensor_features"]

    noise_x = [float(k.split("_")[-1]) * 100 for k in noise.keys()]
    noise_y = [v["rmse"] for v in noise.values()]
    missing_x = [float(k.split("_")[-1]) * 100 for k in missing.keys()]
    missing_y = [v["rmse"] for v in missing.values()]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(noise_x, noise_y, marker="o", color=BLUE, label="Gaussian sensor noise")
    ax.plot(missing_x, missing_y, marker="s", color=RED, label="Mean-imputed missing readings")
    ax.set_xlabel("Severity (%)")
    ax.set_ylabel("Test RMSE (kW)")
    ax.set_title("Robustness of the expected-output model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_both(fig, "fig4_robustness")


def fig_cross_plant():
    with open(RESULTS_DIR / "metrics" / "cross_plant_generalization.json") as f:
        cross = json.load(f)
    with open(RESULTS_DIR / "metrics" / "regression_benchmark.json") as f:
        bench = json.load(f)

    names_map = {
        "linear_regression": "Linear regression",
        "ridge": "Ridge",
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    common = [n for n in names_map if n in cross["results"] and n in bench["results"]]
    in_plant = [bench["results"][n]["r2"]["mean"] for n in common]
    cross_plant = [cross["results"][n]["r2"] for n in common]
    labels = [names_map[n] for n in common]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar([i - width / 2 for i in x], in_plant, width, label="In-plant (Plant 1)", color=BLUE)
    ax.bar([i + width / 2 for i in x], cross_plant, width, label="Cross-plant (train Plant 1, test Plant 2)", color=RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R-squared")
    ax.set_title("In-plant versus cross-plant generalization")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_both(fig, "fig5_cross_plant")


if __name__ == "__main__":
    fig_model_comparison()
    fig_inverter_ranking()
    fig_interpretability()
    fig_robustness()
    fig_cross_plant()
    print(f"wrote 5 SVG+PNG figure pairs to {FIG_DIR}")
