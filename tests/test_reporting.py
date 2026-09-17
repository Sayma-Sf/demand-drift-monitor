import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.reporting.app_data import drift_frame, versions_table
from src.reporting.figures import render_all


def fake_monthly() -> pd.DataFrame:
    months = pd.date_range("2016-01-01", periods=6, freq="MS").strftime("%Y-%m")
    rows = []
    for strategy in (*config.STRATEGIES, "static_with_recent_sales"):
        for i, month in enumerate(months):
            row = {"strategy": strategy, "month": month, "smape": 12 + i * 0.1,
                   "bias_pct": -2.0 * i if strategy == "static" else -0.5,
                   "seasonal_sales_drift": 0.15, "seasonal_prediction_drift": 0.02}
            if strategy != "static_with_recent_sales":
                row.update({"model_version": "v001", "seasonal_sales_drifted": True,
                            "seasonal_prediction_drifted": False, "naive_sales_drift": 0.5,
                            "naive_sales_drifted": True, "naive_prediction_drift": 0.4,
                            "naive_prediction_drifted": True,
                            "retrained": strategy != "static" and i < 5})
            rows.append(row)
    return pd.DataFrame(rows)


def write_manifests(lineage_dir):
    folder = lineage_dir / "drift_triggered"
    folder.mkdir(parents=True)
    windows = [("2015-12-31", "2016-01-01", "2016-01-31"),
               ("2016-01-31", "2016-02-01", "2016-06-30")]
    for n, (train_to, serve_from, serve_to) in enumerate(windows, start=1):
        trigger = ({"type": "initial"} if n == 1 else
                   {"type": "drift", "month": "2016-01", "score": 0.18, "threshold": 0.1,
                    "reference_month": "2015-01"})
        manifest = {"model_version": f"v{n:03d}", "trigger": trigger,
                    "training_data": {"date_from": "2013-01-01", "date_to": train_to,
                                      "rows": 1000 * n, "snapshot_sha256": "ab" * 32},
                    "served": {"from": serve_from, "to": serve_to}}
        (folder / f"v{n:03d}.json").write_text(json.dumps(manifest))


def test_drift_frame_is_long_format():
    frame = drift_frame(fake_monthly(), "static", "seasonal")
    assert set(frame["signal"]) == {"Actual sales", "Forecasts"}
    assert len(frame) == 12


def test_versions_table_explains_each_version(tmp_path):
    write_manifests(tmp_path)
    from src.lineage.manifest import load_manifests

    table = versions_table(load_manifests(tmp_path, "drift_triggered"))
    assert table["Why"].tolist() == ["initial model", "drift after 2016-01 (score 0.180)"]


def test_render_all_writes_light_and_dark_figures(tmp_path):
    write_manifests(tmp_path / "lineage")
    written = render_all({"config": {"drift_threshold": 0.1}}, fake_monthly(),
                         tmp_path / "lineage", tmp_path / "figures")
    assert len(written) == 6
    assert all(p.stat().st_size > 10_000 for p in written)
    assert np.all([p.suffix == ".png" for p in written])


@pytest.mark.skipif(not (config.RESULTS_DIR / "summary.json").exists(),
                    reason="no committed results")
def test_app_runs_on_committed_results():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not app.exception
    app.radio[0].set_value("naive").run()
    app.selectbox[0].set_value("monthly").run()
    assert not app.exception
