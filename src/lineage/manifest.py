"""Lineage manifests: every model version records exactly what it learned from.

A manifest is a small JSON file written the moment a model version is trained:

- the training window (first and last date) and row count,
- a SHA-256 fingerprint of those exact training rows, and of the source file,
- the features, hyperparameters, library versions and a fingerprint of the code,
- why the version exists (initial model, drift alarm, or schedule) and the evidence,
- the months it served in production and the fingerprint of the saved model file.

Every logged prediction carries its model version, so any forecast can be followed to
a manifest and from there to the rows it was trained on. ``verify`` recomputes the
fingerprint from the data file, which turns "trained on 2013-2016 data" from a claim
into something anyone with the data can check.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

LIBRARIES = ["xgboost", "evidently", "pandas", "numpy"]


def relative(path: Path) -> str:
    """Repo-relative path with forward slashes, so manifests never contain a local folder."""
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return Path(path).name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sha256(frame: pd.DataFrame) -> str:
    """Fingerprint of training rows that doesn't depend on file format or row order.

    Canonical form: rows sorted by date, store, item; then a 64-bit integer matrix of
    (days since 1970-01-01, store, item, sales), hashed as raw little-endian bytes.
    """
    rows = frame.sort_values(["date", "store", "item"])
    days = (rows["date"].to_numpy("datetime64[D]").astype("int64"))
    matrix = np.column_stack([
        days,
        rows["store"].to_numpy("int64"),
        rows["item"].to_numpy("int64"),
        rows["sales"].to_numpy("int64"),
    ]).astype("<i8")
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


# The code that decides what a model version is: data loading, features, training,
# monitoring and retraining. Reporting and the dashboard are left out on purpose.
MODEL_CODE = ["config.py", "ingest", "features", "models", "monitoring", "simulation.py",
              "lineage/manifest.py"]


def code_sha256(src_dir: Path = config.ROOT / "src") -> str:
    """Fingerprint of the model-producing source files (paths and contents, sorted)."""
    paths = set()
    for entry in MODEL_CODE:
        target = src_dir / entry
        paths.update(target.rglob("*.py") if target.is_dir() else [target])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(src_dir).as_posix().encode())
        # Normalise line endings so a Windows checkout and CI produce the same fingerprint.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build_manifest(
    *,
    model_version: str,
    strategy: str,
    training: pd.DataFrame,
    features: list[str],
    trigger: dict,
    model_path: Path,
    source_path: Path,
    source_sha256: str,
    code_hash: str,
    served_from: pd.Timestamp,
) -> dict:
    return {
        "schema_version": 1,
        "model_version": model_version,
        "strategy": strategy,
        "trigger": trigger,
        "training_data": {
            "source_file": relative(source_path),
            "source_sha256": source_sha256,
            "date_from": training["date"].min().strftime("%Y-%m-%d"),
            "date_to": training["date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(training)),
            "series": int(training.groupby(["store", "item"]).ngroups),
            "snapshot_sha256": snapshot_sha256(training),
            "snapshot_definition": "sha256 of int64 [days since epoch, store, item, sales] "
                                   "rows sorted by date, store, item",
        },
        "model": {
            "type": "xgboost.XGBRegressor",
            "features": features,
            "hyperparameters": {**config.XGB_PARAMS, "random_state": config.SEED},
            "file": relative(model_path),
            "file_sha256": file_sha256(model_path),
        },
        "code_sha256": code_hash,
        "library_versions": {lib: version(lib) for lib in LIBRARIES},
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "served": {"from": served_from.strftime("%Y-%m-%d"), "to": None},
    }


def manifest_path(lineage_dir: Path, strategy: str, model_version: str) -> Path:
    return lineage_dir / strategy / f"{model_version}.json"


def write_manifest(manifest: dict, lineage_dir: Path) -> Path:
    path = manifest_path(lineage_dir, manifest["strategy"], manifest["model_version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def close_serving(lineage_dir: Path, strategy: str, model_version: str, last_day: pd.Timestamp):
    path = manifest_path(lineage_dir, strategy, model_version)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["served"]["to"] = last_day.strftime("%Y-%m-%d")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifests(lineage_dir: Path, strategy: str) -> list[dict]:
    paths = sorted((lineage_dir / strategy).glob("v*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def serving_version(manifests: list[dict], date: str | pd.Timestamp) -> dict:
    """The manifest of the model version that produced forecasts for ``date``."""
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    for manifest in manifests:
        served = manifest["served"]
        if served["from"] <= day and (served["to"] is None or day <= served["to"]):
            return manifest
    raise LookupError(f"No model version served {day}")


def verify(manifest: dict, sales: pd.DataFrame) -> dict:
    """Recompute the training-snapshot fingerprint from the data and compare."""
    data = manifest["training_data"]
    window = sales[(sales["date"] >= data["date_from"]) & (sales["date"] <= data["date_to"])]
    actual = snapshot_sha256(window)
    return {
        "rows_expected": data["rows"],
        "rows_found": int(len(window)),
        "snapshot_expected": data["snapshot_sha256"],
        "snapshot_found": actual,
        "verified": actual == data["snapshot_sha256"] and len(window) == data["rows"],
    }
