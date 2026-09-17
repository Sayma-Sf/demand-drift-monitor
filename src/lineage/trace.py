"""Trace a forecast back to the data its model was trained on.

    python -m src.lineage.trace --date 2017-03-14 --store 3 --item 17
    python -m src.lineage.trace --date 2016-07-01 --strategy monthly

Prints the model version that made the forecast, why that version exists, its training
window, and whether the training-snapshot fingerprint still matches data/train.csv.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config
from src.ingest.load import load_sales
from src.lineage.manifest import load_manifests, serving_version, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--store", type=int)
    parser.add_argument("--item", type=int)
    parser.add_argument("--strategy", default="drift_triggered", choices=config.STRATEGIES)
    args = parser.parse_args()

    manifest = serving_version(load_manifests(config.ARTIFACTS_DIR / "lineage", args.strategy),
                               args.date)
    data = manifest["training_data"]
    print(f"Forecast for {args.date} ({args.strategy}) was made by {manifest['model_version']}")
    print(f"  why it exists:   {json.dumps(manifest['trigger'])}")
    print(f"  trained on:      {data['date_from']} to {data['date_to']}, {data['rows']:,} rows")
    print(f"  snapshot sha256: {data['snapshot_sha256']}")
    model = manifest["model"]
    print(f"  model file:      {model['file']} ({model['file_sha256'][:16]}...)")

    log_path = config.ARTIFACTS_DIR / "predictions" / f"{args.strategy}.parquet"
    if args.store and args.item and log_path.exists():
        log = pd.read_parquet(log_path)
        row = log[(log["date"] == args.date) & (log["store"] == args.store)
                  & (log["item"] == args.item)]
        if not row.empty:
            logged = row.iloc[0]
            print(f"  logged forecast: {logged['prediction']:.1f} units "
                  f"by {logged['model_version']}")

    if config.DATA_PATH.exists():
        check = verify(manifest, load_sales(config.DATA_PATH))
        status = "VERIFIED" if check["verified"] else "MISMATCH"
        print(f"  {status}: recomputed snapshot {check['snapshot_found'][:16]}... "
              f"from {check['rows_found']:,} rows")
    else:
        print("  data/train.csv not found, so the snapshot can't be re-verified here")


if __name__ == "__main__":
    main()
