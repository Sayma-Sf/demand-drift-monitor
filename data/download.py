"""Download the Store Item Demand Forecasting data to ``data/train.csv``.

Source: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
It is competition data, so the file is never committed, and Kaggle only serves it to
accounts that have accepted the competition rules. Once per machine:

    python -m kaggle auth login     # browser sign-in, no token to copy
    # then open the competition page, Rules tab, and accept the rules
    python data/download.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "train.csv"
COMPETITION = "demand-forecasting-kernels-only"
EXPECTED_ROWS = 913_000  # 10 stores x 50 items x 1,826 days


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION,
         "-p", str(DATA_DIR)],
        check=False,
    )
    if result.returncode != 0:
        sys.exit("Download failed. Log in with `python -m kaggle auth login` and accept the "
                 f"rules at https://www.kaggle.com/competitions/{COMPETITION}/rules")

    with zipfile.ZipFile(DATA_DIR / f"{COMPETITION}.zip") as archive:
        archive.extract("train.csv", DATA_DIR)

    df = pd.read_csv(TARGET)
    if list(df.columns) != ["date", "store", "item", "sales"] or len(df) != EXPECTED_ROWS:
        sys.exit(f"Unexpected file: {len(df):,} rows, columns {list(df.columns)}")
    print(f"Wrote {len(df):,} rows to {TARGET}")


if __name__ == "__main__":
    main()
