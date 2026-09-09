"""Loading and normalising the movement/flight parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _files(pattern: str) -> list[Path]:
    return sorted(config.RAW_DIR.glob(pattern))


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce time columns to UTC datetimes and add the reporting airport."""
    for col in config.TIME_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        elif col in df.columns and df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")

    if config.PHASE in df.columns:
        is_dep = df[config.PHASE].eq("DEP")
        df["airport"] = df[config.AIRPORT_ARR].where(~is_dep, df[config.AIRPORT_DEP])
        # the "other end" of the movement, i.e. where the flight goes to / comes from
        df["other_airport"] = df[config.AIRPORT_DEP].where(~is_dep, df[config.AIRPORT_ARR])

    if config.TARGET in df.columns and {config.MVT_TIME, config.BLOCK_TIME} <= set(df.columns):
        derived = (df[config.MVT_TIME] - df[config.BLOCK_TIME]).dt.total_seconds()
        df[config.TARGET] = df[config.TARGET].fillna(derived)
    return df


def load_training(months: list[int] | None = None, files: list[Path] | None = None) -> pd.DataFrame:
    """Load the monthly training files (all phases, arrivals included)."""
    paths = files or _files(config.TRAINING_GLOB)
    if not paths:
        raise FileNotFoundError(
            f"No {config.TRAINING_GLOB} under {config.RAW_DIR}. Run `prc2026 download` first."
        )
    if months:
        wanted = {f"{m:02d}" for m in months}
        paths = [p for p in paths if p.stem.split("_")[1].split("-")[1] in wanted]
    frames = [normalise(pd.read_parquet(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def load_ranking() -> pd.DataFrame:
    path = config.RAW_DIR / config.RANKING_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `prc2026 download` first.")
    return normalise(pd.read_parquet(path))


def load_submitting() -> pd.DataFrame:
    path = config.RAW_DIR / config.SUBMITTING_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `prc2026 download` first.")
    return pd.read_parquet(path)


def departures(df: pd.DataFrame, labelled: bool = False) -> pd.DataFrame:
    out = df[df[config.PHASE].eq("DEP")]
    if labelled:
        y = out[config.TARGET]
        out = out[y.between(config.TAXI_MIN_SEC, config.TAXI_MAX_SEC)]
    return out


def audit(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-column non-null share, split by phase. Use it to spot what is blanked."""
    rows = []
    for phase, part in df.groupby(config.PHASE, dropna=False):
        for col in df.columns:
            rows.append(
                {
                    "dataset": name,
                    "phase": phase,
                    "column": col,
                    "n": len(part),
                    "non_null_pct": round(100 * part[col].notna().mean(), 2),
                    "n_unique": part[col].nunique(dropna=True) if part[col].dtype == "O" else None,
                }
            )
    return pd.DataFrame(rows)
