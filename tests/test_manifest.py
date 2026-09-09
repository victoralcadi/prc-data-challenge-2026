"""The expected bucket layout: 12 monthly training files, ranking, submitting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026 import config, manifest  # noqa: E402


def test_training_file_names() -> None:
    assert config.training_file(1) == "training_2025-01-01_2025-02-01.parquet"
    assert config.training_file(7) == "training_2025-07-01_2025-08-01.parquet"
    # December rolls the year over
    assert config.training_file(12) == "training_2025-12-01_2026-01-01.parquet"


def test_expected_objects() -> None:
    objects = config.expected_objects()
    assert len(objects) == 14
    assert sum(name.startswith("training_") for name in objects) == 12
    assert config.RANKING_FILE in objects and config.SUBMITTING_FILE in objects
    # every training file must be loadable by the month filter in data.load_training
    for name in objects:
        if name.startswith("training_"):
            assert name.split("_")[1].split("-")[1].isdigit()


def test_local_status_shape() -> None:
    status = manifest.local_status()
    assert list(status.columns) == ["object", "state", "local_mb", "expected_mb"]
    assert len(status) == 14
    assert status["state"].isin({"ok", "missing", "size mismatch"}).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
