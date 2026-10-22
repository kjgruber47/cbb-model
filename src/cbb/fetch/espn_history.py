"""Download historical ESPN schedules and team box scores.

Source: the sportsdataverse project's public data releases on GitHub (the same
files the hoopR R package loads). One CSV per season for each table.

Usage:
    python -m cbb.fetch.espn_history                # all seasons 2006..current
    python -m cbb.fetch.espn_history 2024 2025      # specific seasons
    python -m cbb.fetch.espn_history --refresh 2026 # re-download even if present
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
import requests

from cbb.paths import FIRST_SEASON, RAW_ESPN, ensure

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
TABLES = {
    "schedule": ("espn_mens_college_basketball_schedules", "mbb_schedule_{season}.csv"),
    "team_box": ("espn_mens_college_basketball_team_boxscores", "team_box_{season}.csv"),
}


def current_season(today: dt.date | None = None) -> int:
    """Season label = the calendar year the season ends in (2026-27 -> 2027)."""
    today = today or dt.date.today()
    return today.year + 1 if today.month >= 8 else today.year


def is_complete_csv(path) -> bool:
    """A truncated download usually fails to parse or ends mid-row."""
    try:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                return False
        pd.read_csv(path, nrows=5)
        return True
    except Exception:
        return False


def fetch(season: int, table: str, refresh: bool = False) -> str:
    tag, pattern = TABLES[table]
    name = pattern.format(season=season)
    ensure(RAW_ESPN)  # fresh checkouts (e.g. GitHub runners) have no data/raw
    dest = RAW_ESPN / name
    if dest.exists() and not refresh and is_complete_csv(dest):
        return f"have   {name}"
    url = f"{BASE}/{tag}/{name}"
    r = requests.get(url, timeout=180)
    if r.status_code == 404:
        return f"none   {name} (not published yet)"
    r.raise_for_status()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return f"got    {name} ({len(r.content) / 1e6:.1f} MB)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("seasons", nargs="*", type=int)
    p.add_argument("--refresh", action="store_true", help="re-download files already present")
    args = p.parse_args(argv)

    ensure(RAW_ESPN)
    seasons = args.seasons or list(range(FIRST_SEASON, current_season() + 1))
    for season in seasons:
        for table in TABLES:
            print(fetch(season, table, refresh=args.refresh), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
