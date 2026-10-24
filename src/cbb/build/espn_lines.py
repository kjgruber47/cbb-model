"""Per-game market numbers from ESPN's multi-book odds history.

ESPN lists several kinds of "providers" per game:
  - sportsbooks (DraftKings, Caesars, Westgate, 5Dimes, Unibet, consensus, ...),
  - an "Opening" entry in older seasons (the opening line),
  - projection sites that are NOT markets (numberfire, TeamRankings, AccuScore),
  - live in-game odds.
Only sportsbooks count here. For each game the market number is the median across
books. Closing numbers use each book's 'close' phase when present, else its last
'current' number (for completed games that is the final pre-game number).
Opening numbers use 'open' phases (2023-24 on) or the 'Opening' entry (older).

Margins follow the repo convention: expected HOME margin (= -home spread).

Output: data/processed/espn_lines.csv.gz
  game_id, season, close_margin, close_total, open_margin, open_total,
  home_ml, away_ml, n_books_close, n_books_open

Usage:
    python -m cbb.build.espn_lines
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cbb.paths import EXTERNAL, PROCESSED, ensure

NOT_MARKETS = ("numberfire", "teamrankings", "accuscore", "unknown")


def is_book(provider: str) -> bool:
    p = str(provider).lower()
    return not any(k in p for k in NOT_MARKETS) and "live" not in p and p != "nan"


def load(season: int) -> pd.DataFrame:
    d = pd.read_csv(EXTERNAL / "espn_odds" / f"odds_{season}.csv.gz", dtype={"game_id": str},
                    low_memory=False)
    d = d[d.game_id.notna()].copy()
    d["season"] = season
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.provider.map(is_book)].copy()
    d["margin"] = -pd.to_numeric(d.home_spread, errors="coerce")
    d["total"] = pd.to_numeric(d.total, errors="coerce")
    for c in ("home_ml", "away_ml"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Implausible values are blanks or typos.
    d.loc[d.margin.abs() > 60, "margin"] = np.nan
    d.loc[(d.total < 90) | (d.total > 220), "total"] = np.nan

    opening_entry = d.provider.str.lower().eq("opening")
    books = d[~opening_entry]

    # Close: each book's 'close' if it has one, else its 'current'.
    has_close = set(zip(books[books.phase == "close"].game_id, books[books.phase == "close"].provider))
    close = books[(books.phase == "close") |
                  ((books.phase == "current") &
                   ~pd.Series(list(zip(books.game_id, books.provider)), index=books.index).isin(has_close))]
    opn = pd.concat([books[books.phase == "open"], d[opening_entry]])

    def med(x, prefix):
        g = x.groupby("game_id")
        out = pd.DataFrame({
            f"{prefix}_margin": g.margin.median(),
            f"{prefix}_total": g.total.median(),
            f"n_books_{prefix}": g.provider.nunique(),
        })
        if prefix == "close":
            out["home_ml"] = g.home_ml.median()
            out["away_ml"] = g.away_ml.median()
        return out

    out = med(close, "close").join(med(opn, "open"), how="outer")
    out["season"] = d.groupby("game_id").season.first()
    return out.reset_index()


def main() -> int:
    seasons = sorted(int(p.stem.split("_")[1].split(".")[0])
                     for p in (EXTERNAL / "espn_odds").glob("odds_*.csv.gz"))
    frames = []
    for s in seasons:
        x = summarize(load(s))
        frames.append(x)
        print(f"{s}: {len(x)} games | close total {x.close_total.notna().sum()} "
              f"open total {x.open_total.notna().sum()} | close spread {x.close_margin.notna().sum()} "
              f"open spread {x.open_margin.notna().sum()} | median books {x.n_books_close.median():.0f}",
              flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["game_id"] = out.game_id.astype(int)
    ensure(PROCESSED)
    out.to_csv(PROCESSED / "espn_lines.csv.gz", index=False)
    print(f"wrote {PROCESSED / 'espn_lines.csv.gz'} ({len(out)} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
