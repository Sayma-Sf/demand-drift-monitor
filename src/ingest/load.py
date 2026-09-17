"""Load the daily store-item sales and check the file is what the rest of the code assumes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["date", "store", "item", "sales"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    if df[["store", "item", "sales"]].isna().any().any():
        raise ValueError("Sales data contains missing values")
    if (df["sales"] < 0).any():
        raise ValueError("Sales must be non-negative")
    days = df["date"].nunique()
    per_series = df.groupby(["store", "item"]).size()
    if not (per_series == days).all():
        raise ValueError("Every store-item series must have one row per day")
    return df.sort_values(["date", "store", "item"]).reset_index(drop=True)


def load_sales(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python data/download.py` first.")
    return validate(pd.read_csv(path))


def months(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month from ``start`` to ``end`` inclusive."""
    return list(pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                              pd.Timestamp(end), freq="MS"))


def month_slice(df: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    end = month_start + pd.offsets.MonthEnd(0)
    return df[(df["date"] >= month_start) & (df["date"] <= end)]
