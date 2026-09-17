import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """Stand-in for the competition data: 2 stores x 2 items, daily, 2013 to Feb 2016,
    with weekly and yearly seasonality and 10% growth a year. No Kaggle data in CI."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2013-01-01", "2016-02-29", freq="D")
    rows = []
    for store in (1, 2):
        for item in (1, 2):
            base = 20 * store + 10 * item
            season = 1 + 0.3 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
            weekly = 1 + 0.15 * (dates.dayofweek >= 5)
            growth = 1.10 ** ((dates - dates[0]).days / 365)
            sales = rng.poisson(base * season * weekly * growth)
            rows.append(pd.DataFrame({"date": dates, "store": store, "item": item, "sales": sales}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def small_models(monkeypatch):
    monkeypatch.setattr(config, "XGB_PARAMS", {**config.XGB_PARAMS, "n_estimators": 30,
                                               "max_depth": 4})
