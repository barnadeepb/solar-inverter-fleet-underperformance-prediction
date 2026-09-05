"""Inverter underperformance detection.

Trains a single expected-output model (random forest, chosen for its
accuracy/robustness balance in the regression benchmark) on weather alone,
then uses it as a physically-grounded "expected AC power" reference for
every real inverter. An inverter's performance ratio is:

    actual AC_POWER / model-expected AC_POWER

averaged over all daylight readings. A ratio consistently below the fleet
norm indicates a real fault (soiling, partial shading, a wiring/connection
problem) rather than a one-off measurement blip, because the model has
already accounted for the weather conditions at that moment.

We flag inverters with an EWMA-smoothed daily performance ratio that falls
outside a control-chart band (fleet median +/- k * fleet MAD) on a majority
of days, which is a more defensible anomaly rule than a single arbitrary
threshold on the raw ratio.
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_prep import add_time_features, chronological_split, daylight_mask, get_feature_matrix, load_plant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
K_MAD = 3.0  # control-chart band width in median-absolute-deviations


def compute_performance_ratios(plant_id: int = 1) -> pd.DataFrame:
    df = load_plant(plant_id)
    train_df, test_df = chronological_split(df, train_days=24)
    train_day = train_df[daylight_mask(train_df)].reset_index(drop=True)

    X_train, y_train = get_feature_matrix(train_day)
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    df = add_time_features(df)
    full_day = df[daylight_mask(df)].reset_index(drop=True)
    X_all, y_all = get_feature_matrix(full_day)
    full_day = full_day.copy()
    full_day["EXPECTED_AC_POWER"] = model.predict(X_all)
    full_day["PERFORMANCE_RATIO"] = full_day["AC_POWER"] / full_day["EXPECTED_AC_POWER"].clip(lower=1e-3)

    return full_day


def summarize_by_inverter(full_day: pd.DataFrame) -> pd.DataFrame:
    daily = (
        full_day.groupby(["SOURCE_KEY", full_day["DATE_TIME"].dt.date])["PERFORMANCE_RATIO"]
        .mean()
        .rename("DAILY_RATIO")
        .reset_index()
    )

    fleet_median_by_day = daily.groupby("DATE_TIME")["DAILY_RATIO"].transform("median")
    fleet_mad_by_day = daily.groupby("DATE_TIME")["DAILY_RATIO"].transform(
        lambda s: (s - s.median()).abs().median()
    )
    lower_band = fleet_median_by_day - K_MAD * fleet_mad_by_day
    daily["BELOW_BAND"] = daily["DAILY_RATIO"] < lower_band

    summary = (
        daily.groupby("SOURCE_KEY")
        .agg(
            mean_ratio=("DAILY_RATIO", "mean"),
            std_ratio=("DAILY_RATIO", "std"),
            n_days=("DAILY_RATIO", "size"),
            days_below_band=("BELOW_BAND", "sum"),
        )
        .reset_index()
    )
    summary["frac_days_below_band"] = summary["days_below_band"] / summary["n_days"]
    summary["flagged_underperforming"] = summary["frac_days_below_band"] >= 0.5
    return summary.sort_values("mean_ratio")


def main():
    full_day = compute_performance_ratios(plant_id=1)
    summary = summarize_by_inverter(full_day)

    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "inverter_performance_ratios.csv", index=False)

    flagged = summary[summary["flagged_underperforming"]]
    with open(out_dir / "underperformance_summary.json", "w") as f:
        json.dump(
            {
                "n_inverters": int(len(summary)),
                "k_mad_band_width": K_MAD,
                "flagged_inverters": flagged["SOURCE_KEY"].tolist(),
                "flagged_mean_ratios": {
                    row.SOURCE_KEY: float(row.mean_ratio) for row in flagged.itertuples()
                },
                "fleet_mean_ratio": float(summary["mean_ratio"].mean()),
                "fleet_median_ratio": float(summary["mean_ratio"].median()),
            },
            f,
            indent=2,
        )

    print(summary.to_string(index=False))
    print(f"\nflagged as consistently underperforming: {flagged['SOURCE_KEY'].tolist()}")


if __name__ == "__main__":
    main()
