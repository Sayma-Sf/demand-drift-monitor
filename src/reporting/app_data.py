"""Data helpers for the Streamlit app: results files and lineage manifests only.

No competition data, no model files: the deployed demo installs a handful of packages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.lineage.manifest import load_manifests

STRATEGY_LABELS = {
    "static": "Never retrain",
    "drift_triggered": "Retrain when drift is flagged",
    "monthly": "Retrain every month",
    "static_with_recent_sales": "Comparison: model with recent-sales features (never retrained)",
}
REFERENCE_LABELS = {
    "seasonal": "Same calendar month in the training data",
    "naive": "Whole training window (naive)",
}


def load_results(results_dir: Path = config.RESULTS_DIR) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(results_dir / "monthly.csv")
    monthly["when"] = pd.to_datetime(monthly["month"])
    return summary, monthly


def load_lineage(lineage_dir: Path = config.ARTIFACTS_DIR / "lineage") -> dict[str, list[dict]]:
    return {s: load_manifests(lineage_dir, s) for s in config.STRATEGIES
            if (lineage_dir / s).exists()}


def drift_frame(monthly: pd.DataFrame, strategy: str, reference: str) -> pd.DataFrame:
    """Long format: one row per month and monitored column, ready for a line chart."""
    part = monthly[monthly["strategy"] == strategy]
    rows = []
    for column, label in (("sales", "Actual sales"), ("prediction", "Forecasts")):
        score, flag = f"{reference}_{column}_drift", f"{reference}_{column}_drifted"
        if score not in part:
            continue
        rows.append(pd.DataFrame({
            "when": pd.to_datetime(part["month"]), "month": part["month"], "signal": label,
            "score": part[score],
            "drifted": part[flag] if flag in part else part[score] >= config.DRIFT_THRESHOLD,
        }))
    return pd.concat(rows, ignore_index=True)


def versions_table(manifests: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Version": m["model_version"],
        "Why": ("initial model" if m["trigger"]["type"] == "initial"
                else f"{m['trigger']['type']} after {m['trigger']['month']}"
                + (f" (score {m['trigger']['score']:.3f})" if "score" in m["trigger"] else "")),
        "Trained on": f"{m['training_data']['date_from']} to {m['training_data']['date_to']}",
        "Rows": m["training_data"]["rows"],
        "Served": f"{m['served']['from']} to {m['served']['to']}",
        "Snapshot SHA-256": m["training_data"]["snapshot_sha256"][:16] + "...",
    } for m in manifests])
