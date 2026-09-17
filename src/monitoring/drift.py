"""Monthly drift checks with Evidently.

Data drift means the data a model sees in production no longer looks like the data it
learned from. A model left running unattended keeps producing confident numbers either
way, so drift has to be measured rather than noticed.

Two choices matter here:

- **What to compare against.** Sales here are strongly seasonal: July runs 28% above the
  yearly average and January 32% below. Comparing July with the whole training window
  flags "drift" every summer even though the model already knows about summer. The
  monitor instead compares a month with the *same calendar month* in the model's most
  recent training year. A naive whole-window comparison is kept alongside to show the
  difference.
- **Which columns.** Both the actual sales (the target, known once the month is over)
  and the model's predictions. When sales drift and predictions don't follow, the
  model has stopped tracking reality. That's the staleness signature.
"""

from __future__ import annotations

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from src import config

MONITORED = ["sales", "prediction"]


def drift_scores(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str] = MONITORED,
    method: str = config.DRIFT_METHOD,
    threshold: float = config.DRIFT_THRESHOLD,
) -> dict[str, dict]:
    """Evidently ValueDrift for each column: score, threshold, and whether it drifted."""
    definition = DataDefinition(numerical_columns=columns)
    report = Report([ValueDrift(column=c, method=method, threshold=threshold) for c in columns])
    snapshot = report.run(
        Dataset.from_pandas(current[columns].reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference[columns].reset_index(drop=True), data_definition=definition),
    )
    scores = {}
    for metric in snapshot.dict()["metrics"]:
        cfg = metric["config"]
        score = float(metric["value"])
        scores[cfg["column"]] = {
            "score": score,
            "threshold": float(cfg["threshold"]),
            "method": cfg["method"],
            # Distance-based tests: a larger distance than the threshold means drift.
            "drifted": bool(score >= float(cfg["threshold"])),
        }
    return scores


def seasonal_reference(training: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    """The same calendar month in the most recent training year that contains it."""
    same_month = training[training["date"].dt.month == month_start.month]
    if same_month.empty:
        raise ValueError(f"Training data has no {month_start:%B} to compare with")
    latest_year = same_month["date"].dt.year.max()
    return same_month[same_month["date"].dt.year == latest_year]


def naive_reference(training: pd.DataFrame, rows: int = config.NAIVE_REFERENCE_ROWS,
                    seed: int = config.SEED) -> pd.DataFrame:
    """A random sample of the whole training window, seasons mixed together."""
    return training.sample(min(rows, len(training)), random_state=seed)
