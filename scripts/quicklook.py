"""One-screen sanity check of a raw parquet file.

    python scripts/quicklook.py data/raw/training_2025-01-01_2025-02-01.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026 import config, data  # noqa: E402


def main(path: Path) -> None:
    df = data.normalise(pd.read_parquet(path))
    pd.set_option("display.width", 140)

    print(f"{path.name}: {len(df):,} rows x {df.shape[1]} columns")
    print(f"period: {df[config.MVT_TIME].min()} .. {df[config.MVT_TIME].max()}")
    print("\nphase counts\n", df[config.PHASE].value_counts())
    print("\nmovements per airport\n", df["airport"].value_counts())

    dep = data.departures(df, labelled=True)
    if config.TARGET in dep.columns and dep[config.TARGET].notna().any():
        taxi = dep[config.TARGET] / 60
        print(f"\ntaxi-out (min) over {len(dep):,} labelled departures")
        print(taxi.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2))
        print("\nmedian taxi-out (min) per airport\n",
              taxi.groupby(dep["airport"].to_numpy()).median().round(2).sort_values())
    else:
        print("\nno taxi-out labels in this file (ranking set?)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
