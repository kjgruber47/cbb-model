"""Download Bart Torvik's season files (T-Rank) for benchmarking.

  {year}_team_results.csv  final (or current) team ratings
  {year}_super_sked.csv    every game, including T-Rank's predicted line

Torvik publishes these for bulk use and asks heavy users to contact him, so
this fetches each file once and only refreshes the current season on request.
Saved gzipped to data/external/torvik/ (committed).

Usage:
    python -m cbb.fetch.torvik                 # 2008 .. current season
    python -m cbb.fetch.torvik --refresh 2027  # refresh one season
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time

import requests

from cbb.fetch.espn_history import current_season
from cbb.paths import TORVIK, ensure

BASE = "https://barttorvik.com"
FILES = ["{year}_team_results.csv", "{year}_super_sked.csv"]
HEADERS = {"User-Agent": "cbb-model research (github.com/kjgruber47/cbb-model)"}


def fetch(year: int, pattern: str, refresh: bool = False) -> str:
    name = pattern.format(year=year)
    dest = TORVIK / f"{name}.gz"
    if dest.exists() and not refresh:
        return f"have {name}"
    r = requests.get(f"{BASE}/{name}", headers=HEADERS, timeout=120)
    if r.status_code == 404 or b"<html" in r.content[:500].lower():
        return f"none {name} (status {r.status_code}, html={b'<html' in r.content[:500].lower()})"
    r.raise_for_status()
    with gzip.open(dest, "wb") as f:
        f.write(r.content)
    time.sleep(2)  # one file at a time, gently
    return f"got  {name} ({len(r.content) / 1e6:.1f} MB)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("years", nargs="*", type=int)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args(argv)
    ensure(TORVIK)
    for year in args.years or range(2008, current_season() + 1):
        for pattern in FILES:
            print(fetch(year, pattern, args.refresh), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
