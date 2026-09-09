"""Read the public leaderboard. No credentials needed.

Useful right after an upload: the ranking job scores submissions periodically, so a
file appearing here with a score is proof the submission was accepted.
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

from . import config

API = "https://datacomp.opensky-network.org/api/competitions/{cid}/leaderboard"


def fetch(pages: int = 1) -> list[dict]:
    """Fetch leaderboard entries, following `nextCursor` for up to `pages` pages."""
    url = API.format(cid=config.COMPETITION_ID)
    items: list[dict] = []
    cursor: str | None = None
    for _ in range(max(1, pages)):
        target = f"{url}?cursor={cursor}" if cursor else url
        with urllib.request.urlopen(target, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
        items.extend(payload.get("items", []))
        cursor = payload.get("nextCursor")
        if not cursor:
            break
    return items


def table(items: list[dict]) -> pd.DataFrame:
    """Submissions as scored, best first."""
    if not items:
        return pd.DataFrame(columns=["team", "file", "rmse_sec", "rows_scored", "processed"])
    frame = pd.DataFrame(items)
    frame = frame.rename(
        columns={
            "teamName": "team",
            "filename": "file",
            "score": "rmse_sec",
            "usedPairs": "rows_scored",
            "processedAt": "processed",
        }
    )
    keep = [c for c in ["team", "file", "rmse_sec", "rows_scored", "processed"] if c in frame]
    return frame[keep].sort_values("rmse_sec").reset_index(drop=True)


def best_per_team(items: list[dict]) -> pd.DataFrame:
    """One row per team, their best submission, which is how teams are ranked."""
    scored = table(items)
    if scored.empty:
        return scored
    best = scored.loc[scored.groupby("team")["rmse_sec"].idxmin()]
    best = best.sort_values("rmse_sec").reset_index(drop=True)
    best.index += 1
    return best
