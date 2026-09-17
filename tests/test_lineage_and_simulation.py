import json

import pandas as pd
import pytest

from src import config
from src.features.calendar import add_calendar
from src.lineage.manifest import (
    code_sha256,
    file_sha256,
    load_manifests,
    serving_version,
    snapshot_sha256,
    verify,
)
from src.simulation import run_strategy, should_retrain


def test_snapshot_hash_ignores_row_order_and_date_format(sales):
    reordered = sales.sample(frac=1, random_state=3).assign(date=lambda d: d["date"].astype(str))
    reordered["date"] = pd.to_datetime(reordered["date"])
    assert snapshot_sha256(sales) == snapshot_sha256(reordered)


def test_snapshot_hash_changes_when_one_value_changes(sales):
    tampered = sales.copy()
    tampered.loc[100, "sales"] += 1
    assert snapshot_sha256(sales) != snapshot_sha256(tampered)


def test_should_retrain_rules():
    drifted = {"sales": {"drifted": True}}
    calm = {"sales": {"drifted": False}}
    assert should_retrain("monthly", calm)
    assert should_retrain("drift_triggered", drifted)
    assert not should_retrain("drift_triggered", calm)
    assert not should_retrain("static", drifted)
    with pytest.raises(ValueError):
        should_retrain("weekly", calm)


@pytest.fixture
def replay(sales, small_models, tmp_path):
    source = tmp_path / "train.csv"
    sales.to_csv(source, index=False)
    frame = add_calendar(sales)
    kwargs = dict(sales=frame, artifacts_dir=tmp_path, source_path=source,
                  source_sha256=file_sha256(source), code_hash=code_sha256(),
                  production_start="2016-01-01", production_end="2016-02-29", n_jobs=1)
    return {s: run_strategy(s, **kwargs) for s in config.STRATEGIES}, tmp_path


def test_strategies_create_the_expected_versions(replay):
    outputs, _ = replay
    assert outputs["static"]["versions"] == 1
    assert outputs["monthly"]["versions"] == 2  # retrained after January, not after the last month
    records = outputs["drift_triggered"]["records"]
    assert outputs["drift_triggered"]["versions"] == 1 + records[0]["retrained"]
    assert {"seasonal_sales_drift", "naive_sales_drift", "bias_pct"} <= set(records[0])


def test_every_logged_forecast_traces_to_a_verified_manifest(replay, sales):
    outputs, root = replay
    manifests = load_manifests(root / "lineage", "monthly")
    served = [(m["served"]["from"], m["served"]["to"]) for m in manifests]
    assert served == [("2016-01-01", "2016-01-31"), ("2016-02-01", "2016-02-29")]

    log = outputs["monthly"]["predictions"]
    for date in ("2016-01-15", "2016-02-15"):
        manifest = serving_version(manifests, date)
        assert set(log[log["date"] == date]["model_version"]) == {manifest["model_version"]}
        assert verify(manifest, sales)["verified"]

    second = manifests[1]
    assert second["trigger"] == {"type": "schedule", "month": "2016-01"}
    assert second["training_data"]["date_to"] == "2016-01-31"
    assert not any(":" in json.dumps(m["model"]["file"]) for m in manifests)  # repo-relative


def test_verify_catches_changed_data(replay, sales):
    _, root = replay
    manifest = load_manifests(root / "lineage", "static")[0]
    tampered = sales.copy()
    tampered.loc[tampered["date"] == "2014-06-01", "sales"] += 5
    assert not verify(manifest, tampered)["verified"]
