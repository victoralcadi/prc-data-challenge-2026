"""A submission must match the template row for row, or the ranking script rejects it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026 import config, submission  # noqa: E402


def template(n: int = 50) -> pd.DataFrame:
    """Like submitting.parquet: the ids to predict, with an empty target column."""
    return pd.DataFrame(
        {
            config.ID: np.arange(1000, 1000 + n),
            config.TARGET: pd.Series([np.nan] * n, dtype="float64"),
        }
    )


def test_constant_fill_is_accepted() -> None:
    tmpl = template()
    frame = submission.fill(tmpl, 900.0)
    assert submission.validate(frame, tmpl) == []
    assert frame[config.TARGET].eq(900.0).all()
    assert frame[config.ID].tolist() == tmpl[config.ID].tolist()
    assert list(frame.columns) == [config.ID, config.TARGET]


def test_fill_from_predictions_preserves_template_order() -> None:
    tmpl = template()
    shuffled = tmpl[[config.ID]].sample(frac=1, random_state=3).reset_index(drop=True)
    shuffled[config.TARGET] = np.linspace(300, 2000, len(shuffled))

    frame = submission.fill(tmpl, shuffled)
    assert submission.validate(frame, tmpl) == []
    assert frame[config.ID].tolist() == tmpl[config.ID].tolist()
    # values must follow their id, not the row position
    expected = shuffled.set_index(config.ID)[config.TARGET]
    assert np.allclose(frame[config.TARGET], frame[config.ID].map(expected))


def test_validate_catches_every_rejection_reason() -> None:
    tmpl = template()
    good = submission.fill(tmpl, 900.0)

    missing_row = good.iloc[:-1]
    assert any("row count" in p for p in submission.validate(missing_row, tmpl))
    assert any("missing" in p for p in submission.validate(missing_row, tmpl))

    extra = pd.concat([good, good.iloc[[0]].assign(**{config.ID: 999_999})], ignore_index=True)
    assert any("not in the template" in p for p in submission.validate(extra, tmpl))

    duplicated = pd.concat([good.iloc[:-1], good.iloc[[0]]], ignore_index=True)
    assert any("duplicated" in p for p in submission.validate(duplicated, tmpl))

    reordered = good.iloc[::-1].reset_index(drop=True)
    assert any("row order" in p for p in submission.validate(reordered, tmpl))

    nulls = good.copy()
    nulls.loc[0, config.TARGET] = np.nan
    assert any("null predictions" in p for p in submission.validate(nulls, tmpl))

    negative = good.copy()
    negative.loc[0, config.TARGET] = -5.0
    assert any("non-positive" in p for p in submission.validate(negative, tmpl))

    assert any("missing column" in p for p in submission.validate(good[[config.ID]], tmpl))


def test_integer_template_dtype_is_preserved() -> None:
    tmpl = template()
    tmpl[config.TARGET] = pd.Series([0] * len(tmpl), dtype="int64")
    frame = submission.fill(tmpl, 899.6)
    assert frame[config.TARGET].dtype == np.dtype("int64")
    assert frame[config.TARGET].eq(900).all()
    assert submission.validate(frame, tmpl) == []


def test_write_uses_the_team_naming_convention() -> None:
    import tempfile

    tmpl = template()
    frame = submission.fill(tmpl, 900.0)
    original_dir, original_team = config.SUBMISSION_DIR, config.TEAM_NAME
    with tempfile.TemporaryDirectory() as tmp:
        config.SUBMISSION_DIR = Path(tmp)
        config.TEAM_NAME = "team_warm_donkey"
        try:
            path = submission.write(frame, 3)
            assert path.name == "team_warm_donkey_v3.parquet"
            assert pd.read_parquet(path)[config.TARGET].eq(900.0).all()
        finally:
            config.SUBMISSION_DIR, config.TEAM_NAME = original_dir, original_team


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
