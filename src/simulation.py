"""Replay 2016-2017 as if the forecast were running in production, one month at a time.

For every month:
1. The model version in service forecasts every day for all 500 store-item series.
   Each forecast row is logged with the version that made it.
2. When the month ends, actual sales arrive. The monitor scores the month: forecast
   errors, and Evidently drift of sales and predictions against the model's training data.
3. Depending on the strategy, the model is retrained on everything up to that month.
   The new version gets a manifest and takes over from the next month.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.features.calendar import FEATURES
from src.ingest.load import month_slice, months
from src.lineage.manifest import build_manifest, close_serving, write_manifest
from src.models.forecaster import forecast_errors, predict, save, train
from src.monitoring.drift import drift_scores, naive_reference, seasonal_reference


def should_retrain(strategy: str, seasonal: dict) -> bool:
    if strategy == "monthly":
        return True
    if strategy == "drift_triggered":
        return seasonal["sales"]["drifted"]
    if strategy == "static":
        return False
    raise ValueError(f"Unknown strategy: {strategy}")


class Deployment:
    """The model currently serving one strategy, plus the data it was trained on."""

    def __init__(self, strategy: str, sales: pd.DataFrame, artifacts_dir: Path,
                 source_path: Path, source_sha256: str, code_hash: str, n_jobs: int,
                 features: list[str] = FEATURES):
        self.strategy, self.sales, self.features = strategy, sales, features
        self.artifacts_dir, self.n_jobs = artifacts_dir, n_jobs
        self.source_path, self.source_sha256, self.code_hash = source_path, source_sha256, code_hash
        self.number = 0
        self.model = None
        self.training = None

    @property
    def version(self) -> str:
        return f"v{self.number:03d}"

    @property
    def lineage_dir(self) -> Path:
        return self.artifacts_dir / "lineage"

    def train(self, training_end: pd.Timestamp, trigger: dict) -> None:
        if self.number:
            close_serving(self.lineage_dir, self.strategy, self.version, training_end)
        self.number += 1
        self.training = self.sales[(self.sales["date"] >= config.TRAIN_START)
                                   & (self.sales["date"] <= training_end)]
        self.model = train(self.training, self.features, n_jobs=self.n_jobs)
        model_path = self.artifacts_dir / "models" / self.strategy / f"{self.version}.ubj"
        save(self.model, model_path)
        write_manifest(build_manifest(
            model_version=self.version, strategy=self.strategy, training=self.training,
            features=self.features, trigger=trigger, model_path=model_path,
            source_path=self.source_path, source_sha256=self.source_sha256,
            code_hash=self.code_hash, served_from=training_end + pd.Timedelta(days=1),
        ), self.lineage_dir)

    def with_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(prediction=predict(self.model, frame, self.features))


def run_strategy(
    strategy: str,
    sales: pd.DataFrame,
    artifacts_dir: Path,
    source_path: Path,
    source_sha256: str,
    code_hash: str,
    production_start: str = config.PRODUCTION_START,
    production_end: str = config.PRODUCTION_END,
    n_jobs: int = -1,
) -> dict:
    deployment = Deployment(strategy, sales, artifacts_dir, source_path, source_sha256,
                            code_hash, n_jobs)
    first = pd.Timestamp(production_start)
    deployment.train(first - pd.Timedelta(days=1), {"type": "initial"})

    records, logs = [], []
    production = months(production_start, production_end)
    for i, month in enumerate(production):
        current = deployment.with_predictions(month_slice(sales, month))
        logs.append(current[["date", "store", "item", "prediction"]]
                    .assign(model_version=deployment.version))

        # The month is over: actuals are in, so score it against the serving model's data.
        seasonal_ref = deployment.with_predictions(seasonal_reference(deployment.training, month))
        naive_ref = deployment.with_predictions(naive_reference(deployment.training))
        seasonal = drift_scores(current, seasonal_ref)
        naive = drift_scores(current, naive_ref)
        record = {
            "strategy": strategy,
            "month": month.strftime("%Y-%m"),
            "model_version": deployment.version,
            "reference_month": seasonal_ref["date"].min().strftime("%Y-%m"),
            **forecast_errors(current["sales"], current["prediction"]),
            "actual_total": int(current["sales"].sum()),
            "forecast_total": float(current["prediction"].sum()),
        }
        for label, scores in (("seasonal", seasonal), ("naive", naive)):
            for column, result in scores.items():
                record[f"{label}_{column}_drift"] = result["score"]
                record[f"{label}_{column}_drifted"] = result["drifted"]

        is_last = i == len(production) - 1
        record["retrained"] = bool(should_retrain(strategy, seasonal) and not is_last)
        if record["retrained"]:
            month_end = month + pd.offsets.MonthEnd(0)
            trigger = ({"type": "schedule", "month": record["month"]} if strategy == "monthly"
                       else {"type": "drift", "month": record["month"], "column": "sales",
                             "method": seasonal["sales"]["method"],
                             "score": round(seasonal["sales"]["score"], 4),
                             "threshold": seasonal["sales"]["threshold"],
                             "reference_month": record["reference_month"]})
            deployment.train(month_end, trigger)
        records.append(record)

    close_serving(deployment.lineage_dir, strategy, deployment.version,
                  pd.Timestamp(production_end))
    predictions = pd.concat(logs, ignore_index=True)
    return {"records": records, "predictions": predictions, "versions": deployment.number}
