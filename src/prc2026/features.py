"""Feature engineering for taxi-out prediction.

Everything here is computable for the ranking set, i.e. **without** the departure
off-block time. Congestion is therefore anchored on the takeoff time
(`MVT_TIME_UTC_mvt`) and on scheduled times, never on push-back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SPAN = 10**10  # seconds of head-room between group codes when packing key+time
PRIOR_WEIGHT = 20.0  # smoothing strength for group statistics

WINDOWS = {"15m": 900, "30m": 1800, "60m": 3600}

CATEGORICALS = [
    "airport",
    "other_airport",
    "RUNWAY_mvt",
    "STAND_mvt",
    "stand_prefix",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "FLIGHT_RULE_mvt",
]

GROUP_KEYS = [
    ("airport", "RUNWAY_mvt"),
    ("airport", "STAND_mvt"),
    ("airport", "STAND_mvt", "RUNWAY_mvt"),
    ("airport", "stand_prefix", "RUNWAY_mvt"),
    ("airport", "AIRCRAFT_OPERATOR_flt"),
    ("airport", "AIRCRAFT_TYPE_mvt"),
    ("airport", "hour_local_int"),
    ("airport", "RUNWAY_mvt", "hour_local_int"),
]


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #
def _epoch_seconds(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (seconds since epoch as int64, mask of missing values)."""
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    arr = s.to_numpy("datetime64[s]")
    return arr.astype("int64"), np.isnat(arr)


def _codes(ref_key: pd.Series, q_key: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Factorise reference and query keys into a shared integer coding."""
    both = pd.concat([ref_key, q_key], ignore_index=True).astype("string").fillna("<NA>")
    codes, _ = pd.factorize(both)
    return codes[: len(ref_key)], codes[len(ref_key) :]


def _packed(codes: np.ndarray, secs: np.ndarray) -> np.ndarray:
    return codes.astype("int64") * SPAN + secs


def window_count(
    ref_key: pd.Series,
    ref_time: pd.Series,
    q_key: pd.Series,
    q_time: pd.Series,
    lo: int,
    hi: int,
) -> np.ndarray:
    """Count reference events with `q_time + lo < ref_time <= q_time + hi`, per key."""
    rc, qc = _codes(ref_key, q_key)
    rsec, rbad = _epoch_seconds(ref_time)
    qsec, qbad = _epoch_seconds(q_time)

    ref = np.sort(_packed(rc[~rbad], rsec[~rbad]))
    q = _packed(qc, qsec)
    counts = np.searchsorted(ref, q + hi, "right") - np.searchsorted(ref, q + lo, "right")
    return np.where(qbad, np.nan, counts).astype("float32")


def previous_gap(
    ref_key: pd.Series,
    ref_time: pd.Series,
    q_key: pd.Series,
    q_time: pd.Series,
    back: int = 1,
) -> np.ndarray:
    """Seconds between `q_time` and the `back`-th preceding reference event on the same key."""
    rc, qc = _codes(ref_key, q_key)
    rsec, rbad = _epoch_seconds(ref_time)
    qsec, qbad = _epoch_seconds(q_time)

    ref = np.sort(_packed(rc[~rbad], rsec[~rbad]))
    q = _packed(qc, qsec)
    idx = np.searchsorted(ref, q, "left") - back
    prev = ref[np.clip(idx, 0, None)]
    # the neighbour only counts if it belongs to the same key
    ok = (~qbad) & (idx >= 0) & (prev // SPAN == qc)
    gap = np.full(len(q), np.nan, dtype="float64")
    gap[ok] = (q[ok] - prev[ok]) / back
    return gap.astype("float32")


def _local_hour(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Local (airport) time of day in fractional hours, DST aware."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for airport, tz in config.AIRPORT_TZ.items():
        mask = df["airport"].eq(airport)
        if not mask.any():
            continue
        local = df.loc[mask, time_col].dt.tz_convert(tz)
        out.loc[mask] = local.dt.hour + local.dt.minute / 60.0
    missing = out.isna() & df[time_col].notna()
    if missing.any():  # airports outside the known list: fall back to UTC
        utc = df.loc[missing, time_col]
        out.loc[missing] = utc.dt.hour + utc.dt.minute / 60.0
    return out


# --------------------------------------------------------------------------- #
# feature blocks
# --------------------------------------------------------------------------- #
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    t = df[config.MVT_TIME]
    f = pd.DataFrame(index=df.index)
    f["hour_local"] = _local_hour(df, config.MVT_TIME)
    f["hour_local_sin"] = np.sin(2 * np.pi * f["hour_local"] / 24)
    f["hour_local_cos"] = np.cos(2 * np.pi * f["hour_local"] / 24)
    f["dow"] = t.dt.dayofweek.astype("float32")
    f["is_weekend"] = t.dt.dayofweek.ge(5).astype("float32")
    f["month"] = t.dt.month.astype("float32")
    f["day_of_year"] = t.dt.dayofyear.astype("float32")

    sched = df[config.SCHED_TIME]
    f["takeoff_vs_sched_sec"] = (t - sched).dt.total_seconds().astype("float32")
    if "EOBT_1_flt" in df.columns:
        f["eobt_vs_sched_sec"] = (df["EOBT_1_flt"] - sched).dt.total_seconds().astype("float32")
        f["takeoff_vs_eobt_sec"] = (t - df["EOBT_1_flt"]).dt.total_seconds().astype("float32")
    if {"EOBT_1_flt", "ARVT_1_flt"} <= set(df.columns):
        f["planned_flight_sec"] = (
            (df["ARVT_1_flt"] - df["EOBT_1_flt"]).dt.total_seconds().astype("float32")
        )
    if {"IOBT_flt", "EOBT_1_flt"} <= set(df.columns):
        f["eobt_vs_iobt_sec"] = (
            (df["EOBT_1_flt"] - df["IOBT_flt"]).dt.total_seconds().astype("float32")
        )
    if {"ADES_flt", "ADES_FILED_flt"} <= set(df.columns):
        f["is_diverted"] = df["ADES_flt"].ne(df["ADES_FILED_flt"]).astype("float32")
    return f


def _congestion(dep: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Traffic pressure around each departure's takeoff time.

    `ref` must hold *all* movements (arrivals included) for the same period, so that
    the ranking frame is self-sufficient.
    """
    f = pd.DataFrame(index=dep.index)
    ref_dep = ref[ref[config.PHASE].eq("DEP")]
    ref_arr = ref[ref[config.PHASE].eq("ARR")]

    ap_q, t_q = dep["airport"], dep[config.MVT_TIME]
    rwy_q = dep["airport"].astype("string") + "|" + dep["RUNWAY_mvt"].astype("string")
    rwy_r = (
        ref_dep["airport"].astype("string") + "|" + ref_dep["RUNWAY_mvt"].astype("string")
    )

    for label, w in WINDOWS.items():
        f[f"dep_takeoffs_prev_{label}"] = window_count(
            ref_dep["airport"], ref_dep[config.MVT_TIME], ap_q, t_q, -w, 0
        )
        f[f"dep_takeoffs_next_{label}"] = window_count(
            ref_dep["airport"], ref_dep[config.MVT_TIME], ap_q, t_q, 0, w
        )
        f[f"arr_landings_prev_{label}"] = window_count(
            ref_arr["airport"], ref_arr[config.MVT_TIME], ap_q, t_q, -w, 0
        )
        f[f"arr_inblocks_prev_{label}"] = window_count(
            ref_arr["airport"], ref_arr[config.BLOCK_TIME], ap_q, t_q, -w, 0
        )
        f[f"sched_deps_prev_{label}"] = window_count(
            ref_dep["airport"], ref_dep[config.SCHED_TIME], ap_q, t_q, -w, 0
        )
        f[f"sched_deps_next_{label}"] = window_count(
            ref_dep["airport"], ref_dep[config.SCHED_TIME], ap_q, t_q, 0, w
        )
        f[f"rwy_takeoffs_prev_{label}"] = window_count(
            rwy_r, ref_dep[config.MVT_TIME], rwy_q, t_q, -w, 0
        )

    # net departure queue built up over the last 3h: scheduled minus already airborne
    three_h = 3 * 3600
    sched_3h = window_count(
        ref_dep["airport"], ref_dep[config.SCHED_TIME], ap_q, t_q, -three_h, 0
    )
    airborne_3h = window_count(
        ref_dep["airport"], ref_dep[config.MVT_TIME], ap_q, t_q, -three_h, 0
    )
    f["dep_queue_3h"] = sched_3h - airborne_3h

    # runway pacing
    f["rwy_gap_prev_1"] = previous_gap(rwy_r, ref_dep[config.MVT_TIME], rwy_q, t_q, 1)
    f["rwy_gap_mean_5"] = previous_gap(rwy_r, ref_dep[config.MVT_TIME], rwy_q, t_q, 5)
    f["arr_dep_ratio_60m"] = f["arr_landings_prev_60m"] / (f["dep_takeoffs_prev_60m"] + 1)
    return f


class GroupStats:
    """Smoothed per-group target statistics, fitted on labelled training rows only."""

    def __init__(self, keys: list[tuple[str, ...]] = GROUP_KEYS, prior: float = PRIOR_WEIGHT):
        self.keys = keys
        self.prior = prior
        self.global_mean_: float = 0.0
        self.tables_: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _name(key: tuple[str, ...]) -> str:
        return "g_" + "_".join(k.replace("_mvt", "").replace("_flt", "").lower() for k in key)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> GroupStats:
        self.global_mean_ = float(y.mean())
        work = df.copy()
        work["_y"] = y.to_numpy()
        for key in self.keys:
            grouped = work.groupby(list(key), dropna=False)["_y"]
            agg = grouped.agg(["mean", "median", "std", "count"])
            agg["smoothed"] = (
                agg["mean"] * agg["count"] + self.global_mean_ * self.prior
            ) / (agg["count"] + self.prior)
            self.tables_[self._name(key)] = agg
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for key in self.keys:
            name = self._name(key)
            agg = self.tables_[name]
            joined = df[list(key)].merge(
                agg, how="left", left_on=list(key), right_index=True, sort=False
            )
            out[f"{name}_mean"] = joined["smoothed"].to_numpy(dtype="float32")
            out[f"{name}_median"] = joined["median"].to_numpy(dtype="float32")
            out[f"{name}_std"] = joined["std"].to_numpy(dtype="float32")
            out[f"{name}_count"] = joined["count"].to_numpy(dtype="float32")
        return out


class FeatureBuilder:
    """Turns raw movement frames into the model matrix."""

    def __init__(self):
        self.group_stats_ = GroupStats()
        self.categories_: dict[str, pd.Index] = {}
        self.feature_names_: list[str] = []

    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        stand = out["STAND_mvt"].astype("string")
        out["stand_prefix"] = stand.str.extract(r"^([A-Za-z]*\d?)", expand=False)
        out["hour_local_int"] = _local_hour(out, config.MVT_TIME).fillna(-1).astype("int16")
        return out

    def _assemble(self, dep: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
        X = pd.concat([_calendar(dep), _congestion(dep, ref)], axis=1)
        X = pd.concat([X, self.group_stats_.transform(dep)], axis=1)
        for col in CATEGORICALS:
            values = dep[col].astype("string")
            cats = self.categories_.get(col)
            X[col] = pd.Categorical(values, categories=cats)
        return X

    def fit_transform(self, all_movements: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        ref = self._base(all_movements)
        dep = ref[ref[config.PHASE].eq("DEP")]
        y_all = dep[config.TARGET]
        keep = y_all.between(config.TAXI_MIN_SEC, config.TAXI_MAX_SEC)
        dep, y = dep[keep], y_all[keep].astype("float32")

        for col in CATEGORICALS:
            self.categories_[col] = pd.Index(dep[col].astype("string").dropna().unique())
        self.group_stats_.fit(dep, y)

        X = self._assemble(dep, ref)
        self.feature_names_ = list(X.columns)
        return X, y

    def transform(self, all_movements: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
        ref = self._base(all_movements)
        dep = ref[ref[config.PHASE].eq("DEP")]
        X = self._assemble(dep, ref)
        X = X.reindex(columns=self.feature_names_)
        return X, dep[config.ID]
