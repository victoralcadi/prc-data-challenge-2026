"""Building and validating a submission file.

The ranking script rejects a submission outright if an `MVT_ID_mvt` does not match,
if rows are missing, or if extra rows are present, so the template's rows and their
order are preserved exactly and checked before anything is uploaded. With a cap of
5 submissions per day, a rejected upload is expensive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _cast_like_template(values: pd.Series, template: pd.DataFrame) -> pd.Series:
    """Match the template's target dtype where it is numeric, else use float64."""
    if config.TARGET not in template.columns:
        return values.astype("float64")
    dtype = template[config.TARGET].dtype
    if pd.api.types.is_integer_dtype(dtype):
        return values.round().astype(dtype)
    if pd.api.types.is_float_dtype(dtype):
        return values.astype(dtype)
    return values.astype("float64")


def fill(template: pd.DataFrame, values: pd.DataFrame | float) -> pd.DataFrame:
    """Fill the template's target column, keeping its rows and order untouched.

    `values` is either a constant or a frame with `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.
    """
    out = template[[config.ID]].copy()
    if isinstance(values, (int, float, np.floating, np.integer)):
        filled = pd.Series(float(values), index=out.index)
    else:
        merged = out.merge(values[[config.ID, config.TARGET]], on=config.ID, how="left")
        filled = pd.Series(merged[config.TARGET].to_numpy(), index=out.index)
    out[config.TARGET] = _cast_like_template(filled, template)
    return out


def validate(frame: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    """Return a list of reasons the ranking script would reject `frame`. Empty is good."""
    problems: list[str] = []

    missing_cols = {config.ID, config.TARGET} - set(frame.columns)
    if missing_cols:
        problems.append(f"missing column(s): {', '.join(sorted(missing_cols))}")
        return problems

    if len(frame) != len(template):
        problems.append(f"row count {len(frame):,} != template {len(template):,}")

    ids, template_ids = frame[config.ID], template[config.ID]
    extra = set(ids) - set(template_ids)
    absent = set(template_ids) - set(ids)
    if extra:
        problems.append(f"{len(extra):,} MVT_ID_mvt not in the template")
    if absent:
        problems.append(f"{len(absent):,} template MVT_ID_mvt missing")
    if ids.duplicated().any():
        problems.append(f"{int(ids.duplicated().sum()):,} duplicated MVT_ID_mvt")
    if not extra and not absent and len(frame) == len(template):
        if not ids.reset_index(drop=True).equals(template_ids.reset_index(drop=True)):
            problems.append("row order differs from the template")

    target = frame[config.TARGET]
    if target.isna().any():
        problems.append(f"{int(target.isna().sum()):,} null predictions")
    if not pd.api.types.is_numeric_dtype(target):
        problems.append(f"{config.TARGET} is {target.dtype}, expected numeric")
    else:
        finite = target.dropna()
        if len(finite) and (finite <= 0).any():
            problems.append(f"{int((finite <= 0).sum()):,} non-positive taxi times")
    return problems


def write(frame: pd.DataFrame, version: int, name: str | None = None) -> Path:
    """Write `<team-name>_v<version>.parquet` into the submissions directory."""
    team = config.TEAM_NAME or "team"
    path = config.SUBMISSION_DIR / (name or f"{team}_v{version}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def summarise(frame: pd.DataFrame) -> str:
    target = frame[config.TARGET]
    return (
        f"{len(frame):,} rows | taxi-out mean {target.mean():.0f}s "
        f"median {target.median():.0f}s min {target.min():.0f}s max {target.max():.0f}s"
    )
