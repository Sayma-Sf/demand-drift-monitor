"""Replay 2016-2017 for every retraining strategy and write results, lineage and figures.

    python -m src.pipeline           # full replay, ~25 min on a laptop (24 monthly retrains)
    python -m src.pipeline --quick   # 4 months with small models, into results_quick/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version

import pandas as pd
from joblib import Parallel, delayed

from src import config
from src.features.calendar import HISTORY_FEATURES, add_calendar, add_history
from src.ingest.load import load_sales, month_slice, months
from src.lineage.manifest import code_sha256, file_sha256, load_manifests, serving_version, verify
from src.models.forecaster import forecast_errors, predict, train
from src.monitoring.drift import drift_scores, seasonal_reference
from src.simulation import run_strategy

TRACE_EXAMPLE = {"strategy": "drift_triggered", "date": "2017-03-14", "store": 3, "item": 17}


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def comparison_model(history: pd.DataFrame, production: list[pd.Timestamp]) -> list[dict]:
    """A static model that also sees recent sales: its inputs drift, but does it go stale?"""
    first = production[0]
    training = history[(history["date"] >= config.TRAIN_START) & (history["date"] < first)]
    model = train(training, HISTORY_FEATURES)
    records = []
    for month in production:
        current = month_slice(history, month)
        current = current.assign(prediction=predict(model, current, HISTORY_FEATURES))
        reference = seasonal_reference(training, month)
        reference = reference.assign(prediction=predict(model, reference, HISTORY_FEATURES))
        drift = drift_scores(current, reference)
        records.append({
            "strategy": "static_with_recent_sales",
            "month": month.strftime("%Y-%m"),
            **forecast_errors(current["sales"], current["prediction"]),
            "seasonal_sales_drift": drift["sales"]["score"],
            "seasonal_prediction_drift": drift["prediction"]["score"],
        })
    return records


def summarise(monthly: pd.DataFrame) -> dict:
    out = {}
    for strategy, frame in monthly.groupby("strategy", sort=False):
        years = frame["month"].str[:4]
        entry = {
            "months": int(len(frame)),
            "smape_mean": round(float(frame["smape"].mean()), 3),
            "bias_pct_mean": round(float(frame["bias_pct"].mean()), 3),
            "abs_bias_pct_mean": round(float(frame["bias_pct"].abs().mean()), 3),
            "by_year": {
                year: {"smape_mean": round(float(part["smape"].mean()), 3),
                       "bias_pct_mean": round(float(part["bias_pct"].mean()), 3)}
                for year, part in frame.groupby(years)
            },
        }
        # Flags are recomputed from scores so the comparison model (scores only) counts too.
        for column in ("sales", "prediction"):
            scores = frame[f"seasonal_{column}_drift"]
            entry[f"months_seasonal_{column}_drift"] = int((scores >= config.DRIFT_THRESHOLD).sum())
        if frame["retrained"].notna().any():
            entry["model_versions"] = int(frame["retrained"].astype(bool).sum()) + 1
            entry["months_naive_sales_drift"] = int(
                (frame["naive_sales_drift"] >= config.DRIFT_THRESHOLD).sum()
            )
        out[strategy] = entry
    return out


def trace_example(sales: pd.DataFrame, artifacts_dir, example: dict) -> dict:
    """Follow one logged forecast to its model version and verify the training snapshot."""
    predictions = pd.read_parquet(artifacts_dir / "predictions" / f"{example['strategy']}.parquet")
    row = predictions[(predictions["date"] == example["date"])
                      & (predictions["store"] == example["store"])
                      & (predictions["item"] == example["item"])].iloc[0]
    manifest = serving_version(load_manifests(artifacts_dir / "lineage", example["strategy"]),
                               example["date"])
    check = verify(manifest, sales)
    return {
        **example,
        "prediction": round(float(row["prediction"]), 2),
        "logged_model_version": row["model_version"],
        "manifest_model_version": manifest["model_version"],
        "training_window": [manifest["training_data"]["date_from"],
                            manifest["training_data"]["date_to"]],
        "trigger": manifest["trigger"],
        **check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    results_dir = config.ROOT / "results_quick" if args.quick else config.RESULTS_DIR
    artifacts_dir = config.ROOT / "artifacts_quick" if args.quick else config.ARTIFACTS_DIR
    figures_dir = results_dir / "figures" if args.quick else config.FIGURES_DIR
    production_end = "2016-04-30" if args.quick else config.PRODUCTION_END
    if args.quick:
        config.XGB_PARAMS = {**config.XGB_PARAMS, "n_estimators": 60}
    for sub in ("lineage", "models", "predictions"):  # every run rebuilds its own lineage
        shutil.rmtree(artifacts_dir / sub, ignore_errors=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_sales(config.DATA_PATH)
    sales = add_calendar(raw)
    source_hash, code_hash = file_sha256(config.DATA_PATH), code_sha256()
    production = months(config.PRODUCTION_START, production_end)
    log(f"{len(sales):,} rows; replaying {len(production)} months for {config.STRATEGIES}")

    # Quick mode stays in one process so the smaller model settings apply.
    outputs = Parallel(n_jobs=1 if args.quick else len(config.STRATEGIES))(
        delayed(run_strategy)(s, sales, artifacts_dir, config.DATA_PATH, source_hash, code_hash,
                              production_end=production_end, n_jobs=1 if args.quick else 3)
        for s in config.STRATEGIES
    )
    (artifacts_dir / "predictions").mkdir(parents=True, exist_ok=True)
    for strategy, output in zip(config.STRATEGIES, outputs, strict=True):
        output["predictions"].to_parquet(artifacts_dir / "predictions" / f"{strategy}.parquet",
                                         index=False)
        log(f"    {strategy}: {output['versions']} model version(s)")

    log("Comparison: a static model that also sees recent sales")
    comparison = comparison_model(add_history(raw), production)
    monthly = pd.DataFrame([r for o in outputs for r in o["records"]] + comparison)
    monthly.round(5).to_csv(results_dir / "monthly.csv", index=False)

    summary = summarise(monthly)
    for strategy, entry in summary.items():
        versions = f", {entry['model_versions']} versions" if "model_versions" in entry else ""
        log(f"    {strategy:>26}: SMAPE {entry['smape_mean']:.2f}, "
            f"bias {entry['bias_pct_mean']:+.2f}%{versions}")

    example = ({**TRACE_EXAMPLE, "date": production[-1].strftime("%Y-%m-14")} if args.quick
               else TRACE_EXAMPLE)
    trace = trace_example(raw, artifacts_dir, example)
    log(f"Trace {trace['date']} store {trace['store']} item {trace['item']}: "
        f"{trace['manifest_model_version']}, snapshot verified={trace['verified']}")

    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "quick": args.quick, "train_start": config.TRAIN_START,
            "production": [config.PRODUCTION_START, production_end],
            "strategies": config.STRATEGIES, "drift_method": config.DRIFT_METHOD,
            "drift_threshold": config.DRIFT_THRESHOLD, "xgb_params": config.XGB_PARAMS,
        },
        "data": {"rows": int(len(raw)), "stores": int(raw["store"].nunique()),
                 "items": int(raw["item"].nunique()),
                 "yearly_mean_sales": {str(k): round(float(v), 3) for k, v in
                                       raw.groupby(raw["date"].dt.year)["sales"].mean().items()},
                 "source_sha256": source_hash},
        "summary": summary,
        "trace_example": trace,
        "versions": {lib: version(lib) for lib in ["xgboost", "evidently", "pandas", "streamlit"]},
        "runtime_seconds": round(time.time() - started, 1),
    }
    (results_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    from src.reporting.figures import render_all

    written = render_all(results, monthly, artifacts_dir / "lineage", figures_dir)
    log(f"Wrote {results_dir}, {artifacts_dir / 'lineage'} and {len(written)} figures "
        f"in {results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
