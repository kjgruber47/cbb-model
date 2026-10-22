"""Roster-based preseason ratings.

Predicts each team's season-long offense and defense rating before any games,
from last season's team rating and who is on the roster:
  - prev_off, prev_def          last season's season-end ratings (0 if new to D-I)
  - ret_min                     share of last season's minutes that return
  - ret_val                     production of returning players
  - lost_val                    production of departed players
  - tin_min, tin_val, tin_level incoming D-I transfers: their old role, production,
                                and the strength of the team they came from
  - new_d1                      team was not D-I last season
Player production is a box-score composite (Hollinger's game score per 40
minutes, relative to the league average), weighted by the player's share of
his team's minutes. Weights are fit by least squares on training seasons.

Caveat: "on the roster" means the player appears in that season's box scores.
Before a real season we would use ESPN's preseason rosters instead; the backtest
version therefore knows slightly more (it excludes players who never play).

Usage:
    python -m cbb.model.preseason --train 2011-2019
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from cbb.model.ratings import Params, Prior, fit
from cbb.paths import PROCESSED, ROOT, ensure

FEATURES = ["prev_off", "prev_def", "ret_min", "ret_val", "lost_val",
            "tin_min", "tin_val", "tin_level", "new_d1", "prev_net_x_ret"]
COEF_FILE = ROOT / "src" / "cbb" / "model" / "preseason_coefs.json"


def game_score_40(p: pd.DataFrame) -> pd.Series:
    gs = (p.pts + 0.4 * p.fgm - 0.7 * p.fga - 0.4 * (p.fta - p.ftm) + 0.7 * p.oreb
          + 0.3 * p.dreb + p.stl + 0.7 * p.ast + 0.7 * p.blk - 0.4 * p.pf - p.tov)
    # Shrink low-minute players toward the average with 100 minutes of average play.
    return 40 * gs / (p["min"] + 100)


def team_seasons(games: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Season-end ratings for every D-I team, each season seeded by the last."""
    from cbb.evaluate.backtest import d1_games
    g = d1_games(games)
    p_final = replace(p, lam_eff=p.lam_final, lam_tempo=p.lam_final, recency_floor=1.0)
    prior, rows = Prior.flat(hca=p.hca_prior), []
    for season in sorted(g.season.unique()):
        sg = g[g.season == season]
        teams = np.array(sorted(set(sg.home_id) | set(sg.away_id)))
        r = fit(sg, teams, prior, p_final, as_of=sg.date.max() + pd.Timedelta(days=1))
        prior = Prior.from_ratings(r, p)
        rows.append(pd.DataFrame({"season": season, "team_id": teams, "off": r.O, "def": r.D,
                                  "tempo": r.T, "net": r.O + r.D, "games": r.games}))
    return pd.concat(rows, ignore_index=True)


def features(players: pd.DataFrame, teams: pd.DataFrame,
             live_rosters: pd.DataFrame | None = None, live_teams=None) -> pd.DataFrame:
    """One row per (season, team) for every D-I team-season with a prior season.

    Historical seasons take roster membership from that season's box scores.
    For a season not yet played, pass `live_rosters` (season, team_id, athlete_id)
    and `live_teams` (the team ids to rate).
    """
    pl = players.copy()
    league = pl.groupby("season").apply(
        lambda x: np.average(game_score_40(x), weights=x["min"]), include_groups=False)
    pl["val"] = game_score_40(pl) - pl.season.map(league)
    pl["contrib"] = pl.min_share * pl.val

    t_net = teams.set_index(["season", "team_id"]).net
    seasons = sorted(teams.season.unique())
    if live_rosters is not None:
        seasons = sorted(set(live_rosters.season))
    out = []
    for season in seasons:
        if live_rosters is not None:
            roster = live_rosters[live_rosters.season == season].groupby("team_id").athlete_id.apply(set)
            team_ids = list(live_teams)
        else:
            if season - 1 not in set(pl.season) or season not in set(pl.season):
                continue
            roster = pl[pl.season == season].groupby("team_id").athlete_id.apply(set)
            team_ids = teams[teams.season == season].team_id
        prev = pl[pl.season == season - 1]
        prev_by_team = prev.groupby("team_id")
        # Where each returning/arriving player was last season (largest-minutes team).
        prev_main = prev.sort_values("min").drop_duplicates("athlete_id", keep="last").set_index("athlete_id")
        prev_t = teams[teams.season == season - 1].set_index("team_id")
        for t in team_ids:
            ath = roster.get(t, set())
            row = {"season": season, "team_id": t}
            if t in prev_t.index:
                row.update(prev_off=prev_t.at[t, "off"], prev_def=prev_t.at[t, "def"], new_d1=0)
            else:
                row.update(prev_off=0.0, prev_def=0.0, new_d1=1)
            if t in prev_by_team.groups:
                last = prev_by_team.get_group(t)
                back = last.athlete_id.isin(ath)
                row["ret_min"] = last.loc[back, "min_share"].sum()
                row["ret_val"] = last.loc[back, "contrib"].sum()
                row["lost_val"] = last.loc[~back, "contrib"].sum()
                came_from_here = set(last.athlete_id)
            else:
                row.update(ret_min=0.0, ret_val=0.0, lost_val=0.0)
                came_from_here = set()
            arrivals = [a for a in ath if a not in came_from_here and a in prev_main.index]
            arr = prev_main.loc[arrivals] if arrivals else prev_main.iloc[0:0]
            old_net = np.array([t_net.get((season - 1, ot), np.nan) for ot in arr.team_id]) if len(arr) else np.array([])
            row["tin_min"] = arr.min_share.sum()
            row["tin_val"] = arr.contrib.sum()
            row["tin_level"] = float(np.nansum(arr.min_share.to_numpy() * np.nan_to_num(old_net, nan=-10.0)))
            out.append(row)
    f = pd.DataFrame(out)
    f["prev_net_x_ret"] = (f.prev_off + f.prev_def) * f.ret_min
    return f


def fit_coefs(data: pd.DataFrame) -> dict:
    """Least squares for off and def targets; returns coefficient dicts."""
    X = np.column_stack([np.ones(len(data)), data[FEATURES].to_numpy()])
    coefs = {}
    for target in ("off", "def"):
        beta, *_ = np.linalg.lstsq(X, data[target].to_numpy(), rcond=None)
        coefs[target] = dict(zip(["const", *FEATURES], beta.round(5).tolist()))
    return coefs


def predict_prior(f: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    out = f[["season", "team_id"]].copy()
    for target in ("off", "def"):
        c = coefs[target]
        out[target] = c["const"] + sum(c[k] * f[k] for k in FEATURES)
    return out


def build_inputs(p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    from cbb.evaluate.backtest import load_games
    teams = team_seasons(load_games(), p)
    players = pd.read_csv(PROCESSED / "player_seasons.csv.gz")
    f = features(players, teams)
    data = f.merge(teams[["season", "team_id", "off", "def", "net", "tempo"]], on=["season", "team_id"])
    return data, teams


def build_live_prior(season: int, p: Params) -> pd.DataFrame:
    """Preseason off/def for a season not yet played, from fetched ESPN rosters.

    Coefficients are fit on every completed season before `season`. Writes
    data/external/preseason/prior_{season}.csv (committed; the daily job reads it).
    """
    from cbb.paths import EXTERNAL
    teams_path = PROCESSED / "team_seasons.csv.gz"
    feats_path = PROCESSED / "preseason_features.csv.gz"
    if not (teams_path.exists() and feats_path.exists()):
        data, teams = build_inputs(p)
        data.to_csv(feats_path, index=False)
        teams.to_csv(teams_path, index=False)
    teams = pd.read_csv(teams_path)
    data = pd.read_csv(feats_path)
    players = pd.read_csv(PROCESSED / "player_seasons.csv.gz")
    rosters = pd.read_csv(EXTERNAL / "rosters" / f"rosters_{season}.csv")
    rosters = rosters.dropna(subset=["athlete_id"]).astype({"athlete_id": int})
    d1 = pd.read_csv(EXTERNAL / "d1_teams.csv")

    coefs = fit_coefs(data[data.season < season])
    f = features(players, teams, live_rosters=rosters, live_teams=d1.team_id)
    pred = predict_prior(f, coefs).merge(d1, on="team_id", how="left")
    pred["net"] = pred["off"] + pred["def"]
    pred["net_vs_avg"] = pred["net"] - pred["net"].mean()  # points per 100 poss vs average D-I team
    pred = pred.merge(f[["team_id", "ret_min", "tin_min", "new_d1"]], on="team_id")
    pred["roster_size"] = pred.team_id.map(rosters.groupby("team_id").size()).fillna(0).astype(int)
    out_dir = EXTERNAL / "preseason"
    ensure(out_dir)
    pred = pred.sort_values("net", ascending=False)
    pred.to_csv(out_dir / f"prior_{season}.csv", index=False, float_format="%.3f")
    return pred


def main(argv: list[str] | None = None) -> int:
    from cbb.evaluate.backtest import parse_seasons
    from cbb.predict.daily import load_params
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="2011-2019")
    ap.add_argument("--live", type=int, help="build the preseason prior for this upcoming season")
    args = ap.parse_args(argv)
    p = load_params()
    if args.live:
        pred = build_live_prior(args.live, p)
        print(pred.head(25)[["team", "net_vs_avg", "ret_min", "tin_min", "roster_size"]].round(2).to_string())
        return 0
    data, teams = build_inputs(p)
    ensure(PROCESSED)
    data.to_csv(PROCESSED / "preseason_features.csv.gz", index=False)
    teams.to_csv(PROCESSED / "team_seasons.csv.gz", index=False)

    train = data[data.season.isin(parse_seasons(args.train))]
    test = data[~data.season.isin(parse_seasons(args.train))]
    coefs = fit_coefs(train)
    COEF_FILE.write_text(json.dumps({"train": args.train, **coefs}, indent=2))

    for name, d in (("train", train), ("test", test)):
        pred = predict_prior(d, coefs)
        net_hat = pred["off"] + pred["def"]
        naive = p.regress * (d.prev_off + d.prev_def)
        rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
        print(f"{name}: {len(d)} team-seasons | net RMSE roster model {rmse(d.net - net_hat):.2f} "
              f"vs last-season x {p.regress} {rmse(d.net - naive):.2f}")
    for target in ("off", "def"):
        print(target, {k: round(v, 3) for k, v in coefs[target].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
