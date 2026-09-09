"""Diagnostics must never echo a credential in full."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026 import config  # noqa: E402


def test_mask_secret_hides_the_value() -> None:
    secret = "AKIAVERYSECRETVALUE123"
    masked = config.mask_secret(secret)
    assert secret not in masked
    assert masked.startswith("AKI")
    assert "22 chars" in masked
    # short values reveal nothing at all
    assert config.mask_secret("abcd") == "(set, 4 chars)"
    assert config.mask_secret("") == "(not set)"


def test_settings_masks_keys_and_flags_requirements() -> None:
    original = config.ACCESS_KEY, config.SECRET_KEY
    config.ACCESS_KEY = "ABCDEFGHIJKLMNOP"
    config.SECRET_KEY = "0123456789abcdef"
    try:
        rows = config.settings()
        names = [name for name, _, _ in rows]
        assert "PRC_ACCESS_KEY" in names and "PRC_SUBMISSION_BUCKET" in names
        shown = " ".join(value for _, value, _ in rows)
        assert config.ACCESS_KEY not in shown
        assert config.SECRET_KEY not in shown
        required = {name for name, _, req in rows if req}
        assert {"PRC_ACCESS_KEY", "PRC_SECRET_KEY", "PRC_TEAM_NAME"} <= required
    finally:
        config.ACCESS_KEY, config.SECRET_KEY = original


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
