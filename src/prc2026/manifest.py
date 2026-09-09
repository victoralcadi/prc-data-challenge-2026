"""What the competition bucket should contain, and what is actually on disk.

Expected layout of the 2026 bucket (14 objects, all parquet):

    competition-data/
    |-- training_2025-01-01_2025-02-01.parquet
    |-- training_2025-02-01_2025-03-01.parquet
    |-- ...
    |-- training_2025-12-01_2026-01-01.parquet
    |-- ranking.parquet        # Jan + Jul 2026, departure block/taxi times blanked
    `-- submitting.parquet     # template: MVT_ID_mvt + empty TAXITIME_SEC_mvt

Note this differs from the 2024 challenge, whose bucket held daily trajectory
files (`2022-01-01.parquet`, ...) plus `challenge_set.csv`, `submission_set.csv`
and `final_submission_set.csv`.
"""

from __future__ import annotations

import pandas as pd

from . import config

SIZE_TOLERANCE = 0.4  # accept +/-40% against the published sizes


def local_status() -> pd.DataFrame:
    """One row per expected object, with what is present in `data/raw`."""
    rows = []
    for name, expected_mb in config.expected_objects().items():
        path = config.RAW_DIR / name
        exists = path.exists()
        actual_mb = path.stat().st_size / 1e6 if exists else 0.0
        if not exists:
            state = "missing"
        elif abs(actual_mb - expected_mb) > SIZE_TOLERANCE * expected_mb:
            state = "size mismatch"
        else:
            state = "ok"
        rows.append(
            {
                "object": name,
                "state": state,
                "local_mb": round(actual_mb, 1),
                "expected_mb": expected_mb,
            }
        )
    return pd.DataFrame(rows)


def remote_status(bucket: str | None = None, prefix: str = "") -> pd.DataFrame:
    """Compare the bucket listing against the expected layout."""
    from . import io_s3

    listing = {name: size / 1e6 for name, size in io_s3.list_objects(bucket, prefix)}
    expected = config.expected_objects()
    rows = [
        {
            "object": name,
            "in_bucket": name in listing,
            "bucket_mb": round(listing.get(name, 0.0), 1),
            "expected_mb": mb,
        }
        for name, mb in expected.items()
    ]
    rows += [
        {"object": name, "in_bucket": True, "bucket_mb": round(mb, 1), "expected_mb": None}
        for name, mb in listing.items()
        if name not in expected
    ]
    return pd.DataFrame(rows)


def row_counts(phases: bool = True) -> pd.DataFrame:
    """Row counts per downloaded file, read from parquet metadata (no full load)."""
    import pyarrow.parquet as pq

    rows = []
    for name in config.expected_objects():
        path = config.RAW_DIR / name
        if not path.exists():
            continue
        entry = {"object": name, "rows": pq.ParquetFile(path).metadata.num_rows}
        if phases and name != config.SUBMITTING_FILE:
            phase = pq.read_table(path, columns=[config.PHASE])[config.PHASE].to_pylist()
            entry["dep"] = sum(p == "DEP" for p in phase)
            entry["arr"] = sum(p == "ARR" for p in phase)
        rows.append(entry)
    return pd.DataFrame(rows)


def check_movement_total(counts: pd.DataFrame) -> str:
    """Compare the training row total against the published 4,167,797."""
    if counts.empty:
        return "no files downloaded yet, nothing to count"
    training = counts[counts["object"].str.startswith("training_")]
    if len(training) < len(config.TRAINING_SIZES_MB):
        return (
            f"only {len(training)}/12 training files present "
            f"({training['rows'].sum():,} movements so far)"
        )
    total = int(training["rows"].sum())
    delta = total - config.EXPECTED_MOVEMENTS
    verdict = "matches" if delta == 0 else f"differs by {delta:+,} from"
    return (
        f"{total:,} movements across 12 files, {verdict} "
        f"the published {config.EXPECTED_MOVEMENTS:,}"
    )


def describe(status: pd.DataFrame) -> str:
    if "state" in status.columns:
        ok = int(status["state"].eq("ok").sum())
    else:
        ok = int(status["in_bucket"].sum())
    return f"{ok}/{len(config.expected_objects())} expected objects present"
