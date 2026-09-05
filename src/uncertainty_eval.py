"""Uncertainty quantification for the expected-output model.

A point estimate alone is a weak basis for flagging underperformance: is a
inverter at 90% of predicted output a real fault, or within the model's
own noise band? We build a 90% prediction interval from the spread of
individual tree predictions in the random forest (each tree is itself a
plausible estimate of the conditional mean; their spread approximates
predictive uncertainty), then check calibration -- on well-calibrated
intervals, close to 90% of held-out true values should fall inside the
predicted interval.
"""

import json
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data_prep import chronological_split, daylight_mask, get_feature_matrix, load_plant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
LOWER_Q, UPPER_Q = 5, 95


def tree_quantile_interval(model: RandomForestRegressor, X: np.ndarray):
    tree_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    lower = np.percentile(tree_preds, LOWER_Q, axis=0)
    upper = np.percentile(tree_preds, UPPER_Q, axis=0)
    return lower, upper


def main():
    df = load_plant(1)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)
    test_day = test_df[daylight_mask(test_df)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(train_day)
    X_test, y_test = get_feature_matrix(test_day)

    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    lower, upper = tree_quantile_interval(model, X_test)
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = float(np.mean(covered))
    mean_width = float(np.mean(upper - lower))
    median_width = float(np.median(upper - lower))

    nominal_coverage = (UPPER_Q - LOWER_Q) / 100.0
    result = {
        "model": "random_forest",
        "interval": f"[{LOWER_Q}th, {UPPER_Q}th] percentile of per-tree predictions",
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": coverage,
        "mean_interval_width_kw": mean_width,
        "median_interval_width_kw": median_width,
        "n_test": int(len(X_test)),
    }

    out_path = RESULTS_DIR / "metrics" / "uncertainty_calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"nominal coverage: {nominal_coverage:.2f}, empirical coverage: {coverage:.3f}")
    print(f"mean interval width: {mean_width:.1f} kW (median {median_width:.1f} kW)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
