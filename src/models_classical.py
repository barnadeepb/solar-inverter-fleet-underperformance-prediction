"""Baseline, linear and tree-ensemble regressors for AC power prediction.

Every model is wrapped behind a fit/predict interface identical to
scikit-learn's so the training harness in train.py can iterate over them
uniformly.
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


class NaiveMeanBaseline:
    """Predicts the training-set mean for every input -- the floor any
    real model must clear."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=np.float32)


def get_classical_models(random_state: int = 42) -> dict:
    return {
        "naive_mean": NaiveMeanBaseline(),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "xgboost": XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=random_state
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.1, random_state=random_state, verbosity=-1
        ),
    }
