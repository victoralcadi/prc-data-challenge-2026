"""Paths, column names and env-driven settings."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:  # optional, keeps the package importable without extras
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("PRC_DATA_DIR") or PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

S3_ENDPOINT = os.getenv("PRC_S3_ENDPOINT", "s3.opensky-network.org")
ACCESS_KEY = os.getenv("PRC_ACCESS_KEY", "")
SECRET_KEY = os.getenv("PRC_SECRET_KEY", "")
DATA_BUCKET = os.getenv("PRC_DATA_BUCKET", "competition-data")
SUBMISSION_BUCKET = os.getenv("PRC_SUBMISSION_BUCKET", "")
TEAM_NAME = os.getenv("PRC_TEAM_NAME", "")
TEAM_ID = os.getenv("PRC_TEAM_ID", "")

RANKING_FILE = "ranking.parquet"
SUBMITTING_FILE = "submitting.parquet"
TRAINING_GLOB = "training_*.parquet"

# The bucket holds one movements file per month of 2025, plus the ranking and
# submission-template files. Sizes are the published ones, used only as a rough
# integrity hint (see manifest.py).
TRAINING_SIZES_MB = {
    1: 21, 2: 19, 3: 22, 4: 23, 5: 25, 6: 24,
    7: 25, 8: 25, 9: 24, 10: 25, 11: 22, 12: 22,
}
RANKING_SIZE_MB = 27
SUBMITTING_SIZE_MB = 1.1
TRAINING_YEAR = 2025

# Published total for the 2025 training set, arrivals + departures, across the 10
# reporting airports. `prc2026 status --count` checks the parquet row counts
# against it.
EXPECTED_MOVEMENTS = 4_167_797


def training_file(month: int, year: int = TRAINING_YEAR) -> str:
    """`training_2025-01-01_2025-02-01.parquet` style name for a given month."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"


def expected_objects() -> dict[str, float]:
    """Every object the competition bucket should contain, mapped to its size in MB."""
    objects = {training_file(m): float(mb) for m, mb in TRAINING_SIZES_MB.items()}
    objects[RANKING_FILE] = float(RANKING_SIZE_MB)
    objects[SUBMITTING_FILE] = float(SUBMITTING_SIZE_MB)
    return objects

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PHASE = "PHASE_mvt"
AIRPORT_DEP = "ADEP_mvt"
AIRPORT_ARR = "ADES_mvt"
MVT_TIME = "MVT_TIME_UTC_mvt"
BLOCK_TIME = "BLOCK_TIME_UTC_mvt"
SCHED_TIME = "SCHED_TIME_UTC_mvt"

TIME_COLUMNS = [
    MVT_TIME,
    BLOCK_TIME,
    SCHED_TIME,
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
]

# columns that leak the blanked off-block time; excluded from the model by default
LEAKY_COLUMNS = [BLOCK_TIME, "AOBT_3_flt", TARGET]

# the 10 reporting airports, with the tz used for local time-of-day
AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}
# UTC offset of local *standard* time, used as a cheap tz proxy (hours)
AIRPORT_UTC_OFFSET = {
    "EDDF": 1, "EDDM": 1, "EGLL": 0, "EHAM": 1, "LEBL": 1,
    "LEMD": 1, "LFPG": 1, "LIRF": 1, "LTFM": 3, "LSZH": 1,
}

# plausibility bounds for the target, in seconds (used to trim training rows)
TAXI_MIN_SEC = 60
TAXI_MAX_SEC = 3 * 3600


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, SUBMISSION_DIR):
        path.mkdir(parents=True, exist_ok=True)
