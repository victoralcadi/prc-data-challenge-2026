"""Training, validation and prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import FeatureBuilder

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
}
NUM_ROUNDS = 3000
EARLY_STOPPING = 100


@dataclass
class Artefact:
    """Everything needed to score the ranking set."""

    model: object
    builder: FeatureBuilder
    best_iteration: int
    metrics: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        path = path or config.MODEL_DIR / "taxiout_lgbm.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path | None = None) -> Artefact:
        path = path or config.MODEL_DIR / "taxiout_lgbm.pkl"
        with path.open("rb") as fh:
            return pickle.load(fh)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def split_by_month(
    df: pd.DataFrame, val_months: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month = df[config.MVT_TIME].dt.month
    is_val = month.isin(val_months)
    return df[~is_val], df[is_val]


def _baseline(train_dep: pd.DataFrame, y_train: pd.Series, val_dep: pd.DataFrame) -> np.ndarray:
    """Per-airport median taxi-out: the number any model has to beat."""
    med = y_train.groupby(train_dep["airport"].to_numpy()).median()
    return val_dep["airport"].map(med).fillna(y_train.median()).to_numpy()


def train_validate(
    all_movements: pd.DataFrame,
    val_months: tuple[int, ...] = (1, 7),
    params: dict | None = None,
) -> Artefact:
    """Fit on all months except `val_months`, score on those months.

    The validation frame is featurised on its own, exactly like `ranking.parquet`
    (which contains Jan + Jul 2026 in full), so congestion features never reach
    across the split.
    """
    import lightgbm as lgb

    train_raw, val_raw = split_by_month(all_movements, val_months)
    if val_raw.empty:
        raise ValueError(f"No movements in validation months {val_months}.")

    builder = FeatureBuilder()
    X_train, y_train = builder.fit_transform(train_raw)

    val_builder_view = builder  # reuse fitted encoders / group stats
    X_val, val_ids = val_builder_view.transform(val_raw)
    val_dep = val_raw[val_raw[config.PHASE].eq("DEP")]
    y_val = val_dep[config.TARGET]
    labelled = y_val.between(config.TAXI_MIN_SEC, config.TAXI_MAX_SEC).to_numpy()
    X_val, y_val, val_dep = X_val[labelled], y_val[labelled], val_dep[labelled]

    print(f"train rows {len(X_train):,}  val rows {len(X_val):,}  features {X_train.shape[1]}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        {**PARAMS, **(params or {})},
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dvalid],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    train_dep = train_raw[train_raw[config.PHASE].eq("DEP")]
    train_lab = train_dep[config.TARGET].between(config.TAXI_MIN_SEC, config.TAXI_MAX_SEC)
    metrics = {
        "val_rmse": rmse(y_val, pred),
        "baseline_rmse": rmse(y_val, _baseline(train_dep[train_lab],
                                               train_dep.loc[train_lab, config.TARGET],
                                               val_dep)),
        "val_mae": float(np.mean(np.abs(y_val.to_numpy() - pred))),
        "best_iteration": booster.best_iteration,
    }
    return Artefact(booster, builder, booster.best_iteration, metrics)


def fit_full(all_movements: pd.DataFrame, num_rounds: int, params: dict | None = None) -> Artefact:
    """Refit on every training month for the given number of rounds."""
    import lightgbm as lgb

    builder = FeatureBuilder()
    X, y = builder.fit_transform(all_movements)
    print(f"refit rows {len(X):,}  features {X.shape[1]}  rounds {num_rounds}")
    booster = lgb.train({**PARAMS, **(params or {})}, lgb.Dataset(X, label=y),
                        num_boost_round=num_rounds)
    return Artefact(booster, builder, num_rounds)


def predict(artefact: Artefact, ranking: pd.DataFrame) -> pd.DataFrame:
    X, ids = artefact.builder.transform(ranking)
    pred = artefact.model.predict(X, num_iteration=artefact.best_iteration or None)
    pred = np.clip(pred, config.TAXI_MIN_SEC, config.TAXI_MAX_SEC)
    return pd.DataFrame({config.ID: ids.to_numpy(), config.TARGET: pred})


def importances(artefact: Artefact, top: int = 30) -> pd.DataFrame:
    gain = artefact.model.feature_importance("gain")
    names = artefact.model.feature_name()
    return (
        pd.DataFrame({"feature": names, "gain": gain})
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
