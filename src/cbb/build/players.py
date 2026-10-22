"""Player-season table from ESPN player box scores.

Downloads one large CSV per season from the sportsdataverse releases, keeps only
what the roster model needs, and aggregates to one row per player per team per
season (a mid-season transfer gets a row for each team).

Output: data/processed/player_seasons.csv.gz

Usage:
    python -m cbb.build.players            # 2010..last completed season
    python -m cbb.build.players 2025 2026
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from cbb.paths import PROCESSED, RAW, ensure

BASE = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
        "espn_mens_college_basketball_player_boxscores/player_box_{season}.csv")
RAW_PLAYERS = RAW / "espn_players"
STATS = {
    "minutes": "min", "points": "pts",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf",
}


def download(season: int) -> None:
    ensure(RAW_PLAYERS)
    dest = RAW_PLAYERS / f"player_box_{season}.csv"
    if dest.exists() and dest.stat().st_size > 1e6:
        return
    with requests.get(BASE.format(season=season), stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)


def aggregate(season: int) -> pd.DataFrame:
    cols = ["season", "game_id", "athlete_id", "athlete_display_name", "team_id", "team_location",
            "did_not_play", "starter", *STATS]
    p = pd.read_csv(RAW_PLAYERS / f"player_box_{season}.csv", usecols=cols, low_memory=False)
    p = p.rename(columns=STATS)
    p = p[~p.did_not_play.fillna(False).astype(bool)]
    p = p[p["min"].fillna(0) > 0]
    agg = p.groupby(["season", "team_id", "athlete_id"]).agg(
        name=("athlete_display_name", "last"), team=("team_location", "last"),
        games=("game_id", "nunique"), starts=("starter", "sum"),
        **{c: (c, "sum") for c in STATS.values()},
    ).reset_index()
    team_min = agg.groupby("team_id")["min"].transform("sum")
    agg["min_share"] = agg["min"] / team_min  # share of the team's player-minutes (5 on court)
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seasons", nargs="*", type=int)
    args = ap.parse_args(argv)
    seasons = args.seasons or list(range(2010, 2027))
    frames = []
    out = PROCESSED / "player_seasons.csv.gz"
    if args.seasons and out.exists():
        old = pd.read_csv(out)
        frames.append(old[~old.season.isin(seasons)])
    for s in seasons:
        download(s)
        a = aggregate(s)
        print(f"{s}: {len(a)} player-team rows, {a.athlete_id.nunique()} players", flush=True)
        frames.append(a)
    ensure(PROCESSED)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
