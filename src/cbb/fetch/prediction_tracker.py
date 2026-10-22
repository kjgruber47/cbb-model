"""Download ThePredictionTracker's college basketball season files.

Each file lists every D-I game with the final score, the market line, the
opening line and the predicted margin from each system the site tracks. Lines
start in 2005-06. File ncaabbYY.csv is the season that starts in 20YY.

Saved gzipped to data/external/prediction_tracker/ (committed; small).

Usage:
    python -m cbb.fetch.prediction_tracker          # 2003-04 .. 2025-26
    python -m cbb.fetch.prediction_tracker 25       # one season by start year
"""
from __future__ import annotations

import argparse
import gzip
import sys

import requests

from cbb.paths import TRACKER, ensure

URL = "https://www.thepredictiontracker.com/ncaabb{yy:02d}.csv"
HEADERS = {"User-Agent": "cbb-model research (github.com/kjgruber47/cbb-model)"}


def fetch(yy: int, refresh: bool = False) -> str:
    dest = TRACKER / f"ncaabb{yy:02d}.csv.gz"
    if dest.exists() and not refresh:
        return f"have {dest.name}"
    r = requests.get(URL.format(yy=yy), headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return f"none ncaabb{yy:02d}.csv"
    r.raise_for_status()
    with gzip.open(dest, "wb") as f:
        f.write(r.content)
    lines = r.content.count(b"\n")
    return f"got  {dest.name} ({lines} lines)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("years", nargs="*", type=int, help="two-digit season start years")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args(argv)
    ensure(TRACKER)
    for yy in args.years or range(3, 26):
        print(fetch(yy, args.refresh), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
