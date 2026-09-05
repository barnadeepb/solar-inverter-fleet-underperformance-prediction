"""Interpretability check: does the model's learned feature importance
match known solar-generation physics? Solar output should be overwhelmingly
governed by irradiation, with ambient/module temperature playing a much
smaller secondary role (temperature affects panel efficiency, not the
primary energy input). A model that instead weighted, say, hour-of-day
heavily would suggest it has memorized a daily pattern rather than the
actual physical driver, which would undermine trust in it as an
"expected output" reference for anomaly detection.

Uses SHAP TreeExplainer on the random forest model (the same model used as
the "expected output" reference in anomaly_detection.py) and cross-checks
against the model's built-in impurity-based feature_importances_.
"""

import json
import pathlib

import matplotlib
import numpy as np
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from data_prep import chronological_split, daylight_mask, get_feature_matrix, load_plant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
FEATURE_NAMES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "HOUR_SIN", "HOUR_COS"]


def main():
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(train_day)
    X_test, _ = get_feature_matrix(test_day)

    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    sample = X_test[np.random.default_rng(42).choice(len(X_test), size=min(2000, len(X_test)), replace=False)]
    shap_values = explainer.shap_values(sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = {name: float(v) for name, v in zip(FEATURE_NAMES, mean_abs_shap)}
    shap_importance_pct = {k: v / sum(mean_abs_shap) * 100 for k, v in shap_importance.items()}

    builtin_importance = {name: float(v) for name, v in zip(FEATURE_NAMES, model.feature_importances_)}

    result_dir = RESULTS_DIR / "metrics"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "interpretability.json", "w") as f:
        json.dump(
            {
                "model": "random_forest",
                "mean_abs_shap_value": shap_importance,
                "mean_abs_shap_value_pct": shap_importance_pct,
                "builtin_feature_importance": builtin_importance,
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    order = sorted(shap_importance_pct.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in order], [v for _, v in order], color="#2b6cb0")
    ax.set_xlabel("mean |SHAP value| (% of total)")
    ax.set_title("Feature importance for predicted AC power (Plant 1)")
    fig.tight_layout()
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "shap_feature_importance.png", dpi=150)

    print("SHAP importance (%):", shap_importance_pct)
    print("built-in importance:", builtin_importance)


if __name__ == "__main__":
    main()
