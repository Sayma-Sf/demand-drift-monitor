"""Features for the planning forecast: store, item and calendar only.

A model built on these can forecast any future date, which is why planning teams
like it. It is also why it goes stale: nothing in its inputs tells it that sales
have grown since it was trained. ``HISTORY_FEATURES`` adds recent sales, for the
comparison model that adapts.
"""

from __future__ import annotations

import pandas as pd

FEATURES = ["store", "item", "dayofweek", "month", "dayofyear", "year"]
HISTORY_FEATURES = ["store", "item", "dayofweek", "month", "dayofyear", "prev_month_mean",
                    "prev_3m_mean", "same_month_last_year", "prev_12m_mean", "lag_364"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["year"] = out["date"].dt.year
    return out


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recent-sales features, each using only months before the month being forecast."""
    out = add_calendar(df)
    out["month_start"] = out["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (out.groupby(["store", "item", "month_start"])["sales"].mean()
               .rename("m").reset_index().sort_values(["store", "item", "month_start"]))
    g = monthly.groupby(["store", "item"])["m"]
    monthly["prev_month_mean"] = g.shift(1)
    monthly["prev_3m_mean"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    monthly["same_month_last_year"] = g.shift(12)
    monthly["prev_12m_mean"] = g.transform(lambda s: s.shift(1).rolling(12).mean())
    out = out.merge(monthly.drop(columns="m"), on=["store", "item", "month_start"], how="left")
    out = out.sort_values(["store", "item", "date"])
    out["lag_364"] = out.groupby(["store", "item"])["sales"].shift(364)
    return out.drop(columns="month_start").sort_values(["date", "store", "item"]).reset_index(
        drop=True)
