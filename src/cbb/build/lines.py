"""Attach historical market lines to ESPN games.

ThePredictionTracker files name teams their own way, so games are matched on
date and final score instead of names:
  1. exact date + the same two scores (either home/away orientation),
  2. then +/- 1 day for the leftovers (late tips logged on the next date).
Where the tracker's home team is ESPN's away team (common at neutral sites),
every line is flipped so all margins are from ESPN's home team's view.

Output: data/processed/lines.csv.gz, one row per matched game:
  game_id, line (market, home margin), line_open, and the tracked systems'
  predicted home margins (prefixed sys_).

Usage:
    python -m cbb.build.lines
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cbb.paths import PROCESSED, TRACKER

NON_SYSTEM = {"date", "home", "hscore", "road", "rscore", "neutral", "std", "line", "lineopen", "lineavg"}


def load_tracker() -> pd.DataFrame:
    frames = []
    for f in sorted(TRACKER.glob("ncaabb*.csv.gz")):
        yy = int(f.name[6:8])
        d = pd.read_csv(f, na_values=["."])
        d["season"] = 2000 + yy + 1
        d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
        sys_cols = [c for c in d.columns if c.startswith("line") and c not in NON_SYSTEM]
        d = d.rename(columns={c: "sys_" + c[4:] for c in sys_cols})
        d = d.rename(columns={"lineavg": "sys_avg", "lineopen": "line_open"})
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    t = t.dropna(subset=["hscore", "rscore"])
    t["hscore"] = t.hscore.astype(int)
    t["rscore"] = t.rscore.astype(int)
    # An opening line of exactly 0 far from the closing number is a missing value.
    bad_open = (t.line_open == 0) & ((t.line - t.line_open).abs() > 6)
    t.loc[bad_open, "line_open"] = np.nan
    t["tid"] = np.arange(len(t))
    return t


def _score_pass(t, g, used_t, used_g, keys_t=("hscore", "rscore"), extra=None):
    """Match on date (+/- 1 day) and scores, optionally also on team ids."""
    found = []
    for shift in (0, 1, -1):
        tt = t[~t.tid.isin(used_t)].copy()
        tt["date"] = tt["date"] + pd.Timedelta(days=shift)
        gg = g[~g.game_id.isin(used_g)]
        lk_same = ["date", *keys_t] + ([extra[0], extra[1]] if extra else [])
        rk_same = ["date", "home_pts", "away_pts"] + (["home_id", "away_id"] if extra else [])
        rk_swap = ["date", "away_pts", "home_pts"] + (["away_id", "home_id"] if extra else [])
        same = tt.merge(gg, left_on=lk_same, right_on=rk_same)
        same["flip"] = False
        swap = tt.merge(gg, left_on=lk_same, right_on=rk_swap)
        swap["flip"] = True
        m = pd.concat([same, swap], ignore_index=True)
        # Keep only unambiguous pairs (same score twice on one day is common).
        m = m[~m.tid.duplicated(keep=False) & ~m.game_id.duplicated(keep=False)]
        used_t |= set(m.tid)
        used_g |= set(m.game_id)
        found.append(m)
    return pd.concat(found, ignore_index=True)


def crosswalk(m: pd.DataFrame) -> dict:
    """Tracker team name -> ESPN team id, by majority vote over clean matches."""
    a = pd.DataFrame({
        "name": np.concatenate([m.home.to_numpy(), m.road.to_numpy()]),
        "espn": np.concatenate([np.where(m.flip, m.away_id, m.home_id),
                                np.where(m.flip, m.home_id, m.away_id)]),
    })
    votes = a.groupby(["name", "espn"]).size().rename("n").reset_index()
    votes["share"] = votes.n / votes.groupby("name").n.transform("sum")
    best = votes.sort_values("n").drop_duplicates("name", keep="last")
    best = best[(best.share >= 0.8) & (best.n >= 3)]
    return dict(zip(best.name, best.espn))


def match(games: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "season", "game_date", "home_pts", "away_pts", "home_id", "away_id"]].copy()
    g["date"] = pd.to_datetime(g.game_date)

    used_t, used_g = set(), set()
    first = _score_pass(t, g, used_t, used_g)
    xw = crosswalk(first)
    t = t.assign(home_espn=t.home.map(xw), road_espn=t.road.map(xw))
    second = _score_pass(
        t.dropna(subset=["home_espn", "road_espn"]).astype({"home_espn": int, "road_espn": int}),
        g, used_t, used_g, extra=("home_espn", "road_espn"),
    )
    m = pd.concat([first, second], ignore_index=True)

    line_cols = ["line", "line_open"] + [c for c in t.columns if c.startswith("sys_")]
    sign = np.where(m.flip, -1.0, 1.0)
    for c in line_cols:
        m[c] = m[c] * sign
    out = m[["game_id", "tid", "flip", "home", "road", *line_cols]].copy()
    return out


def main() -> int:
    games = pd.read_csv(PROCESSED / "games.csv.gz", low_memory=False)
    t = load_tracker()
    m = match(games, t)
    rate = len(m) / len(t)
    print(f"tracker games {len(t)}, matched {len(m)} ({rate:.1%})")
    by = t.assign(matched=t.tid.isin(m.tid)).groupby("season").matched.mean()
    print("match rate by season:", " ".join(f"{s}:{r:.0%}" for s, r in by.items()))
    m.drop(columns=["tid"]).to_csv(PROCESSED / "lines.csv.gz", index=False)
    print(f"wrote {PROCESSED / 'lines.csv.gz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
