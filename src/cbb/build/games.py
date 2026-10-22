"""Build one row per game from ESPN schedules and team box scores.

Output: data/processed/games.csv.gz with, for each completed game:
  ids, season, date, season type, neutral flag, overtime periods,
  both teams' points and box-score counts, estimated possessions,
  and whether each team is Division I that season.

Division I is inferred: a team counts as D-I in a season if it has at least
10 box-scored games in ESPN's data (non-D-I opponents appear only a handful
of times).

Usage:
    python -m cbb.build.games              # all seasons with raw files
    python -m cbb.build.games 2025 2026
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from cbb.paths import PROCESSED, RAW_ESPN, ensure

BOX_COLS = {
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m",
    "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb",
    "total_turnovers": "tov",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "fouls": "pf",
}
FT_POSS_COEF = 0.475  # KenPom's play-by-play-derived coefficient
MIN_D1_GAMES = 10


def possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + FT_POSS_COEF * fta


def build_season(season: int) -> pd.DataFrame:
    box = pd.read_csv(RAW_ESPN / f"team_box_{season}.csv", low_memory=False)
    sched = pd.read_csv(
        RAW_ESPN / f"mbb_schedule_{season}.csv",
        usecols=["game_id", "neutral_site", "conference_competition", "status_period",
                 "status_type_completed", "venue_id", "notes_headline"],
        low_memory=False,
    ).drop_duplicates("game_id")

    box = box.rename(columns=BOX_COLS)
    keep = ["game_id", "season", "season_type", "game_date", "game_date_time", "team_id",
            "team_location", "team_home_away", "team_score", *BOX_COLS.values()]
    box = box[keep]

    home = box[box.team_home_away == "home"].drop(columns="team_home_away")
    away = box[box.team_home_away == "away"].drop(columns="team_home_away")
    shared = ["game_id", "season", "season_type", "game_date", "game_date_time"]
    home = home.rename(columns={c: f"home_{c}" for c in home.columns if c not in shared})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns if c not in shared})
    g = home.merge(away.drop(columns=shared[1:]), on="game_id", how="inner")
    g = g.rename(columns={
        "home_team_id": "home_id", "away_team_id": "away_id",
        "home_team_location": "home_team", "away_team_location": "away_team",
        "home_team_score": "home_pts", "away_team_score": "away_pts",
    })

    g = g.merge(sched, on="game_id", how="left")
    g = g[g.status_type_completed.fillna(True).astype(bool)]
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    g["conference_game"] = g.conference_competition.fillna(False).astype(bool)
    periods = g.status_period.fillna(2).clip(lower=2)
    g["ot_periods"] = (periods - 2).astype(int)
    g["minutes"] = 40 + 5 * g.ot_periods

    g["home_poss"] = possessions(g.home_fga, g.home_oreb, g.home_tov, g.home_fta)
    g["away_poss"] = possessions(g.away_fga, g.away_oreb, g.away_tov, g.away_fta)
    g["poss"] = (g.home_poss + g.away_poss) / 2
    g["tempo40"] = g.poss * 40 / g.minutes

    counts = pd.concat([g.home_id, g.away_id]).value_counts()
    d1 = set(counts[counts >= MIN_D1_GAMES].index)
    g["home_d1"] = g.home_id.isin(d1)
    g["away_d1"] = g.away_id.isin(d1)

    g["margin"] = g.home_pts - g.away_pts
    g["total"] = g.home_pts + g.away_pts
    g = g.drop(columns=["neutral_site", "conference_competition", "status_period", "status_type_completed"])
    return g.sort_values(["game_date_time", "game_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("seasons", nargs="*", type=int)
    args = p.parse_args(argv)

    seasons = args.seasons or sorted(
        int(f.stem.split("_")[-1]) for f in RAW_ESPN.glob("team_box_*.csv")
    )
    frames = []
    for s in seasons:
        df = build_season(s)
        both = (df.home_d1 & df.away_d1).sum()
        print(f"{s}: {len(df):5d} games, {both:5d} D-I vs D-I, "
              f"{df.loc[df.home_d1 & df.away_d1, 'tempo40'].mean():.1f} poss/40", flush=True)
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)

    bad = games[(games.poss < 40) | (games.poss > 110)]
    if len(bad):
        print(f"warning: {len(bad)} games with implausible possessions (kept, flagged)")
    games["poss_ok"] = games.poss.between(40, 110)

    ensure(PROCESSED)
    out = PROCESSED / "games.csv.gz"
    games.to_csv(out, index=False)
    print(f"wrote {out} ({len(games)} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
