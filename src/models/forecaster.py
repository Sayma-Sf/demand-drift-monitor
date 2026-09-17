"""Train, save and score the XGBoost demand forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config


def train(frame: pd.DataFrame, features: list[str], n_jobs: int = -1) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**config.XGB_PARAMS, random_state=config.SEED, n_jobs=n_jobs)
    model.fit(frame[features], frame["sales"])
    return model


def predict(model: xgb.XGBRegressor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(model.predict(frame[features]), 0, None)


def save(model: xgb.XGBRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def forecast_errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """SMAPE (the competition metric), MAE, and bias: total forecast vs total actual sales.

    Bias is the number a planner feels: -8% means ordering 8% too little stock.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(2 * np.abs(predicted - actual), denominator,
                      out=np.zeros_like(denominator), where=denominator > 0)
    return {
        "smape": float(100 * ratio.mean()),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "bias_pct": float(100 * (predicted.sum() / actual.sum() - 1)),
    }
