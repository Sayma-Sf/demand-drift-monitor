import numpy as np
import pandas as pd
import pytest

from src.features.calendar import add_calendar, add_history
from src.ingest.load import load_sales, month_slice, months, validate


def test_validate_sorts_and_parses_dates(sales):
    shuffled = sales.sample(frac=1, random_state=0).assign(date=lambda d: d["date"].astype(str))
    df = validate(shuffled)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing


def test_validate_rejects_a_missing_day(sales):
    with pytest.raises(ValueError, match="one row per day"):
        validate(sales.drop(index=5))


def test_validate_rejects_negative_sales(sales):
    with pytest.raises(ValueError, match="non-negative"):
        validate(sales.assign(sales=sales["sales"] - 1000))


def test_load_sales_points_to_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/download.py"):
        load_sales(tmp_path / "train.csv")


def test_months_and_month_slice(sales):
    starts = months("2016-01-01", "2016-02-29")
    assert [m.strftime("%Y-%m-%d") for m in starts] == ["2016-01-01", "2016-02-01"]
    february = month_slice(sales, starts[1])
    assert february["date"].min() == pd.Timestamp("2016-02-01")
    assert february["date"].max() == pd.Timestamp("2016-02-29")
    assert len(february) == 29 * 4


def test_calendar_features(sales):
    row = add_calendar(sales[sales["date"] == "2016-02-29"]).iloc[0]
    assert (row["dayofweek"], row["month"], row["dayofyear"], row["year"]) == (0, 2, 60, 2016)


def test_history_features_only_use_earlier_months(sales):
    history = add_history(sales)
    one = history[(history["store"] == 1) & (history["item"] == 1)]
    january = sales[(sales["store"] == 1) & (sales["item"] == 1)
                    & (sales["date"].dt.strftime("%Y-%m") == "2016-01")]
    feb_row = one[one["date"] == "2016-02-10"].iloc[0]
    assert feb_row["prev_month_mean"] == pytest.approx(january["sales"].mean())
    lagged = one[one["date"] == pd.Timestamp("2016-02-10") - pd.Timedelta(days=364)]["sales"]
    assert feb_row["lag_364"] == lagged.iloc[0]
    assert np.isnan(one.iloc[0]["prev_month_mean"])  # nothing before the first month
