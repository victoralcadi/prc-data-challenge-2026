"""Checks the vectorised window/gap helpers against brute force, plus a pipeline smoke test.

Runnable with pytest or directly: `python tests/test_features.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026 import config, features  # noqa: E402

RNG = np.random.default_rng(7)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(s: pd.Series) -> np.ndarray:
    """Unit-agnostic seconds since epoch, independent of the pandas datetime resolution."""
    return (s - EPOCH).dt.total_seconds().to_numpy().astype("int64")


def synthetic(n: int = 4000) -> pd.DataFrame:
    airports = list(config.AIRPORT_TZ)[:4]
    start = pd.Timestamp("2025-03-01", tz="UTC")
    takeoff = start + pd.to_timedelta(RNG.integers(0, 28 * 86400, n), unit="s")
    phase = RNG.choice(["DEP", "ARR"], n, p=[0.55, 0.45])
    airport = RNG.choice(airports, n)
    taxi = RNG.integers(300, 2400, n)

    df = pd.DataFrame(
        {
            config.ID: np.arange(n),
            config.PHASE: phase,
            config.AIRPORT_DEP: np.where(phase == "DEP", airport, "LFPO"),
            config.AIRPORT_ARR: np.where(phase == "DEP", "LFPO", airport),
            config.MVT_TIME: takeoff,
            config.BLOCK_TIME: takeoff - pd.to_timedelta(taxi, unit="s"),
            config.SCHED_TIME: takeoff - pd.to_timedelta(RNG.integers(-600, 1800, n), unit="s"),
            config.TARGET: taxi.astype("float64"),
            "RUNWAY_mvt": RNG.choice(["25R", "25L", "07C"], n),
            "STAND_mvt": RNG.choice(["A12", "B03", "C21", "V7"], n),
            "AIRCRAFT_TYPE_mvt": RNG.choice(["A20N", "B738", "A359"], n),
            "FLIGHT_RULE_mvt": "I",
            "WK_TBL_CAT_flt": RNG.choice(["M", "H"], n),
            "AIRCRAFT_OPERATOR_flt": RNG.choice(["AAA", "BBB", "CCC"], n),
            "MARKET_SEGMENT_flt": RNG.choice(["Mainline", "Low-Cost"], n),
            "FLIGHT_TYPE_flt": "S",
            "EOBT_1_flt": takeoff - pd.to_timedelta(RNG.integers(600, 2400, n), unit="s"),
            "ARVT_1_flt": takeoff + pd.to_timedelta(RNG.integers(1800, 9000, n), unit="s"),
            "IOBT_flt": takeoff - pd.to_timedelta(RNG.integers(600, 2400, n), unit="s"),
        }
    )
    # a few missing values, because the real data is messy
    df.loc[df.index[:20], "RUNWAY_mvt"] = None
    df.loc[df.index[20:40], config.TARGET] = np.nan
    return df


def test_window_count_matches_brute_force() -> None:
    df = synthetic(600)
    ref, q = df.iloc[:400], df.iloc[400:]
    lo, hi = -1800, 0
    got = features.window_count(
        ref["airport"] if "airport" in ref else ref[config.AIRPORT_DEP],
        ref[config.MVT_TIME],
        q[config.AIRPORT_DEP],
        q[config.MVT_TIME],
        lo,
        hi,
    )
    ref_key = ref[config.AIRPORT_DEP].to_numpy()
    ref_t = epoch_seconds(ref[config.MVT_TIME])
    q_key = q[config.AIRPORT_DEP].to_numpy()
    q_t = epoch_seconds(q[config.MVT_TIME])
    expected = np.array(
        [
            np.sum((ref_key == k) & (ref_t > t + lo) & (ref_t <= t + hi))
            for k, t in zip(q_key, q_t, strict=True)
        ]
    )
    assert np.array_equal(got.astype("int64"), expected)


def test_previous_gap_matches_brute_force() -> None:
    df = synthetic(400)
    key, time = df[config.AIRPORT_DEP], df[config.MVT_TIME]
    got = features.previous_gap(key, time, key, time, back=1)
    secs = epoch_seconds(time)
    keys = key.to_numpy()
    for i in RNG.choice(len(df), 40, replace=False):
        earlier = secs[(keys == keys[i]) & (secs < secs[i])]
        # the event itself is in the reference set, so a same-second twin gives gap 0
        same_second = np.sum((keys == keys[i]) & (secs == secs[i])) > 1
        if same_second:
            continue
        expected = secs[i] - earlier.max() if len(earlier) else np.nan
        assert (np.isnan(got[i]) and np.isnan(expected)) or got[i] == expected


def test_feature_builder_roundtrip() -> None:
    df = features.pd.concat([synthetic(3000)], ignore_index=True)
    from prc2026 import data

    df = data.normalise(df)
    builder = features.FeatureBuilder()
    X, y = builder.fit_transform(df)

    assert len(X) == len(y) and len(X) > 0
    assert y.between(config.TAXI_MIN_SEC, config.TAXI_MAX_SEC).all()
    assert X["dep_takeoffs_prev_60m"].notna().all()
    for suffix in ("mean", "median", "std", "count"):
        assert f"g_airport_runway_{suffix}" in X.columns

    # the ranking case: departures lose their block time and target, arrivals keep theirs
    ranking = df.copy()
    is_dep = ranking[config.PHASE].eq("DEP")
    ranking.loc[is_dep, config.BLOCK_TIME] = pd.NaT
    ranking.loc[is_dep, config.TARGET] = np.nan
    X2, ids = builder.transform(ranking)
    assert list(X2.columns) == list(X.columns)
    assert len(X2) == int(is_dep.sum())
    assert ids.is_unique


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
