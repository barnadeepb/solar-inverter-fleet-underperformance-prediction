"""Load, merge and feature-engineer the solar plant generation/weather data.

Known dataset quirks handled here:
  - Generation timestamps are DD-MM-YYYY HH:MM; weather timestamps are
    YYYY-MM-DD HH:MM:SS. Both are parsed to the same dtype before merging.
  - Plant 1's DC_POWER is reported on a ~10x different scale than AC_POWER
    (a documented artifact of this public dataset). We model AC_POWER, the
    physically meaningful delivered power, and never mix DC_POWER across
    plants.
  - Weather is recorded once per plant (one sensor), not per inverter, so it
    is joined on (DATE_TIME, PLANT_ID) rather than per-inverter SOURCE_KEY.
"""

import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

WEATHER_FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]


def _load_generation(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True)
    return df


def _load_weather(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df.drop(columns=["SOURCE_KEY"])


def load_plant(plant_id: int) -> pd.DataFrame:
    """Return merged generation+weather rows for plant 1 or 2."""
    if plant_id not in (1, 2):
        raise ValueError("plant_id must be 1 or 2")

    gen = _load_generation(DATA_DIR / f"Plant_{plant_id}_Generation_Data.csv")
    weather = _load_weather(DATA_DIR / f"Plant_{plant_id}_Weather_Sensor_Data.csv")

    merged = gen.merge(weather, on=["DATE_TIME", "PLANT_ID"], how="inner", suffixes=("", "_WEATHER"))
    merged = merged.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HOUR"] = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24.0)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24.0)
    df["DAY_OF_STUDY"] = (df["DATE_TIME"].dt.normalize() - df["DATE_TIME"].dt.normalize().min()).dt.days
    return df


def chronological_split(df: pd.DataFrame, train_days: int = 24):
    """Split by calendar day (not randomly) so the test set is a genuinely
    future period relative to training -- a stronger generalization test
    than a random row shuffle for time-series data."""
    df = add_time_features(df)
    cutoff = df["DAY_OF_STUDY"].min() + train_days
    train = df[df["DAY_OF_STUDY"] < cutoff].reset_index(drop=True)
    test = df[df["DAY_OF_STUDY"] >= cutoff].reset_index(drop=True)
    return train, test


def daylight_mask(df: pd.DataFrame) -> pd.Series:
    """Restrict to readings with nonzero irradiation, i.e. actual daylight
    operating conditions -- night-time zeros would otherwise dominate error
    metrics with a trivial "predict zero" result."""
    return df["IRRADIATION"] > 0


def get_feature_matrix(df: pd.DataFrame):
    feature_cols = WEATHER_FEATURES + ["HOUR_SIN", "HOUR_COS"]
    return df[feature_cols].to_numpy(dtype=np.float32), df["AC_POWER"].to_numpy(dtype=np.float32)


def build_sequences(df: pd.DataFrame, window: int = 4):
    """Build per-inverter lag windows of weather features for sequence
    models (CNN/LSTM). Sequences never cross inverter or day boundaries."""
    feature_cols = WEATHER_FEATURES + ["HOUR_SIN", "HOUR_COS"]
    xs, ys = [], []
    for _, group in df.groupby(["SOURCE_KEY", "DAY_OF_STUDY"]):
        group = group.sort_values("DATE_TIME")
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        target = group["AC_POWER"].to_numpy(dtype=np.float32)
        for i in range(window - 1, len(group)):
            xs.append(feats[i - window + 1 : i + 1])
            ys.append(target[i])
    return np.stack(xs), np.array(ys, dtype=np.float32)
