"""Daily predictions: refit ratings and price upcoming games against the market.

Steps:
  1. refresh ESPN schedule and box scores for this season (and last season),
  2. seed ratings from last season's season-end fit, regressed toward average,
  3. fit this season's D-I games played before today,
  4. predict today's and the next two days' D-I games,
  5. join the latest DraftKings numbers from data/live/odds/ (current and open),
  6. write data/live/predictions/YYYY-MM-DD.csv, latest.md, and today's ratings.

Margins are always from the HOME team's view (positive = home wins by that many).
A sportsbook spread of -4.5 on the home team is a market home margin of +4.5.

Usage:
    python -m cbb.predict.daily
    python -m cbb.predict.daily --date 2026-11-03
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from cbb.build.games import MIN_D1_GAMES, build_season
from cbb.fetch.espn_history import current_season, fetch as fetch_espn
from cbb.fetch.espn_live import game_meta
from cbb.model.ratings import Params, Prior, fit, predict
from cbb.paths import LIVE, RAW_ESPN, ROOT, ensure

EASTERN = ZoneInfo("America/New_York")
PARAMS_FILE = ROOT / "src" / "cbb" / "model" / "params.json"
MARGIN_SD = 11.0      # spread of actual margins around a good prediction, points
EARLY_GAMES = 8       # below this many games for either team, flag as early-season


def load_params() -> Params:
    if PARAMS_FILE.exists():
        return Params(**json.loads(PARAMS_FILE.read_text()))
    return Params()


def with_roster_prior(base: Prior, season: int) -> Prior:
    """Use the roster-based preseason ratings if they were built for this season."""
    from cbb.paths import EXTERNAL
    path = EXTERNAL / "preseason" / f"prior_{season}.csv"
    if not path.exists():
        print(f"no roster prior at {path}; using last season's ratings")
        return base
    r = pd.read_csv(path)
    r = r[r.roster_size >= 5]  # skip teams whose roster did not come through
    # Keep the base prior's overall offense/defense levels (they set scoring, not margins).
    o_shift = np.mean([base.O.get(t, 0.0) for t in r.team_id]) - r["off"].mean()
    d_shift = np.mean([base.D.get(t, 0.0) for t in r.team_id]) - r["def"].mean()
    O, D = dict(base.O), dict(base.D)
    O.update(zip(r.team_id, r["off"] + o_shift))
    D.update(zip(r.team_id, r["def"] + d_shift))
    print(f"roster prior: {len(r)} teams from {path.name}")
    return Prior(O=O, D=D, T=base.T, mu=base.mu, tmu=base.tmu, hca=base.hca)


def ensure_espn(season: int) -> None:
    for s, refresh in ((season - 1, False), (season, True)):
        for table in ("schedule", "team_box"):
            print(fetch_espn(s, table, refresh=refresh), flush=True)


def completed_games(season: int) -> pd.DataFrame:
    if not (RAW_ESPN / f"team_box_{season}.csv").exists():
        return pd.DataFrame()
    g = build_season(season)
    g["date"] = pd.to_datetime(g.game_date)
    return g[g.home_d1 & g.away_d1 & g.poss_ok]


def d1_teams(season: int, last_season_d1: set) -> set:
    """Last season's D-I teams plus any team with 10+ games on this season's schedule.

    Early in the season ESPN's schedule file is incomplete, so last season's
    membership carries most of the weight until the schedule fills in.
    """
    s = pd.read_csv(RAW_ESPN / f"mbb_schedule_{season}.csv", usecols=["home_id", "away_id"])
    counts = pd.concat([s.home_id, s.away_id]).value_counts()
    return set(last_season_d1) | set(counts[counts >= MIN_D1_GAMES].index)


def games_from_snapshots(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Every game on ESPN's scoreboard: the latest slate file, plus recent raw snapshots."""
    rows = []
    slate = LIVE / "slate_latest.csv"
    try:
        sl = pd.read_csv(slate) if slate.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:  # older runs wrote a bare file on empty days
        sl = pd.DataFrame()
    if len(sl):
        for r in sl.itertuples():
            rows.append({
                "game_id": int(r.game_id), "date": pd.Timestamp(r.game_date), "start_utc": r.start_utc,
                "home_id": int(r.home_id), "away_id": int(r.away_id),
                "home": r.home_location if isinstance(r.home_location, str) else r.home,
                "away": r.away_location if isinstance(r.away_location, str) else r.away,
                "neutral": bool(r.neutral_site),
            })
    for f in sorted((LIVE / "raw").glob("*/espn_*.json.gz"))[-12:]:
        with gzip.open(f, "rt") as fh:
            snap = json.load(fh)
        for board in (snap.get("scoreboards") or {}).values():
            for ev in (board or {}).get("events", []):
                m = game_meta(ev)
                comp = (ev.get("competitions") or [{}])[0]
                teams = {c.get("homeAway"): c.get("team", {}) for c in comp.get("competitors", [])}
                rows.append({
                    "game_id": int(m["game_id"]), "date": pd.Timestamp(m["game_date"]),
                    "start_utc": m["start_utc"],
                    "home_id": int(m["home_id"]), "away_id": int(m["away_id"]),
                    "home": teams.get("home", {}).get("location") or m["home"],
                    "away": teams.get("away", {}).get("location") or m["away"],
                    "neutral": bool(m["neutral_site"]),
                })
    g = pd.DataFrame(rows, columns=["game_id", "date", "start_utc", "home_id", "away_id",
                                    "home", "away", "neutral"])
    if g.empty:
        return g
    g = g.drop_duplicates("game_id", keep="last")
    return g[(g.date >= pd.Timestamp(start)) & (g.date <= pd.Timestamp(end))]


def upcoming(season: int, start: dt.date, days: int, d1: set) -> pd.DataFrame:
    """Games to price: ESPN schedule file plus the fuller slate from live snapshots."""
    s = pd.read_csv(RAW_ESPN / f"mbb_schedule_{season}.csv", low_memory=False).drop_duplicates("game_id")
    s["date"] = pd.to_datetime(s.game_date)
    end = start + dt.timedelta(days=days - 1)
    s = s[(s.date >= pd.Timestamp(start)) & (s.date <= pd.Timestamp(end))]
    sched = pd.DataFrame({
        "game_id": s.game_id.astype(int), "date": s.date, "start_utc": s.start_date,
        "home_id": s.home_id, "away_id": s.away_id,
        "home": s.home_location, "away": s.away_location,
        "neutral": s.neutral_site.fillna(False).astype(bool),
    })
    g = pd.concat([sched, games_from_snapshots(start, end)], ignore_index=True)
    g = g.drop_duplicates("game_id", keep="last")
    return g[g.home_id.isin(d1) & g.away_id.isin(d1)].reset_index(drop=True)


def latest_odds() -> pd.DataFrame:
    files = sorted((LIVE / "odds").glob("*.csv"))
    if not files:
        return pd.DataFrame()
    o = pd.concat([pd.read_csv(f) for f in files[-2:]], ignore_index=True)
    o = o[o.source == "core"].sort_values("snapshot_utc")
    o = o.drop_duplicates(["game_id", "provider", "phase"], keep="last")
    # One provider per game: the one with the most coverage overall.
    top = o.provider.value_counts().index
    o["rank"] = o.provider.map({p: i for i, p in enumerate(top)})
    o = o.sort_values("rank").drop_duplicates(["game_id", "phase"])
    cur = o[o.phase == "current"].set_index("game_id")
    opn = o[o.phase == "open"].set_index("game_id")
    return pd.DataFrame({
        "provider": cur.provider,
        "market_margin": -cur.home_spread,
        "open_margin": -opn.home_spread.reindex(cur.index),
        "market_total": cur.total,
        "open_total": opn.total.reindex(cur.index),
        "home_ml": cur.home_ml, "away_ml": cur.away_ml,
        "odds_as_of": cur.snapshot_utc,
    }).reset_index()


def implied_home_prob(home_ml, away_ml):
    def raw(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    h, a = raw(home_ml), raw(away_ml)
    return h / (h + a)


def run(today: dt.date, days: int = 3) -> pd.DataFrame:
    season = current_season(today)
    p = load_params()
    ensure_espn(season)

    prev = build_season(season - 1)
    prev["date"] = pd.to_datetime(prev.game_date)
    prev = prev[prev.home_d1 & prev.away_d1 & prev.poss_ok]
    prev_teams = np.array(sorted(set(prev.home_id) | set(prev.away_id)))
    p_final = replace(p, lam_eff=p.lam_final, lam_tempo=p.lam_final, recency_floor=1.0)
    prior = Prior.from_ratings(fit(prev, prev_teams, Prior.flat(hca=p.hca_prior), p_final), p)
    prior = with_roster_prior(prior, season)

    d1 = d1_teams(season, set(prev_teams))
    teams = np.array(sorted(d1))
    played = completed_games(season)
    asof = pd.Timestamp(today)
    train = played[played.date < asof] if len(played) else played
    ratings = fit(train, teams, prior, p, as_of=asof)

    games = upcoming(season, today, days, d1)
    pred = predict(ratings, games).drop(columns=["pred_home_pts", "pred_away_pts"])
    out = games.merge(pred, on="game_id")
    odds = latest_odds()
    if len(odds):
        odds["game_id"] = odds.game_id.astype(int)
        out = out.merge(odds, on="game_id", how="left")
    else:
        for c in ["provider", "market_margin", "open_margin", "market_total", "open_total",
                  "home_ml", "away_ml", "odds_as_of"]:
            out[c] = np.nan

    out["edge"] = out.pred_margin - out.market_margin
    out["edge_vs_open"] = out.pred_margin - out.open_margin
    out["total_edge"] = out.pred_total - out.market_total
    out["model_home_prob"] = norm.cdf(out.pred_margin / MARGIN_SD)
    out["market_home_prob"] = implied_home_prob(out.home_ml, out.away_ml)
    out["early_season"] = np.minimum(out.home_games, out.away_games) < EARLY_GAMES
    out["start_et"] = pd.to_datetime(out.start_utc, utc=True).dt.tz_convert(EASTERN).dt.strftime("%a %-I:%M %p")

    cols = ["date", "start_et", "away", "home", "neutral", "pred_margin", "market_margin", "open_margin",
            "edge", "edge_vs_open", "pred_total", "market_total", "open_total", "total_edge", "pred_poss",
            "model_home_prob", "market_home_prob", "home_games", "away_games", "early_season",
            "provider", "odds_as_of", "game_id", "home_id", "away_id"]
    out = out[cols].sort_values(["date", "start_et"])

    ensure(LIVE / "predictions", LIVE / "ratings")
    out.to_csv(LIVE / "predictions" / f"{today.isoformat()}.csv", index=False, float_format="%.2f")
    log_predictions(out)
    names = pd.concat([
        games[["home_id", "home"]].set_axis(["team_id", "team"], axis=1),
        games[["away_id", "away"]].set_axis(["team_id", "team"], axis=1),
        prev[["home_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
    ]).drop_duplicates("team_id")
    table = ratings.table().merge(names, on="team_id", how="left").sort_values("net", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(LIVE / "ratings" / f"{today.isoformat()}.csv", index=False, float_format="%.2f")
    write_markdown(out, table, today, len(train))
    return out


LOG_COLS = ["run_utc", "game_id", "date", "away", "home", "neutral", "pred_margin", "pred_total",
            "market_margin", "market_total", "open_margin", "open_total", "provider", "odds_as_of"]
LOG_CHANGE = ["pred_margin", "pred_total", "market_margin", "market_total"]


def log_predictions(out: pd.DataFrame) -> None:
    """Append-only paper-trading record: a row whenever the model's or the market's number moves."""
    run_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = out.assign(run_utc=run_utc)[LOG_COLS].copy()
    new[LOG_CHANGE] = new[LOG_CHANGE].round(2)
    path = LIVE / "predictions" / f"log_{run_utc[:7]}.csv"
    if path.exists():
        old = pd.read_csv(path)
        last = old.sort_values("run_utc").drop_duplicates("game_id", keep="last").set_index("game_id")[LOG_CHANGE]
        cur = new.set_index("game_id")[LOG_CHANGE]
        prev = last.reindex(cur.index)
        diff = ~((cur.round(2) == prev.round(2)) | (cur.isna() & prev.isna())).all(axis=1)
        new = new[new.game_id.isin(diff[diff].index)]
    if len(new):
        new.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.2f")


def write_markdown(out: pd.DataFrame, table: pd.DataFrame, today: dt.date, n_train: int) -> None:
    def fmt(x, signed=True):
        return "" if pd.isna(x) else (f"{x:+.1f}" if signed else f"{x:.1f}")

    lines = [
        f"# Predictions for {today:%A, %B %-d, %Y}",
        "",
        f"Ratings fit on {n_train} D-I games played this season before today. "
        "Margins are from the home team's view. Edge = model minus market.",
        "",
        "Early-season numbers (either team under 8 games) lean on last season's ratings "
        "and are not yet adjusted for roster changes. Treat them with caution.",
        "",
        "## Largest spread disagreements",
        "",
        "| Date | Time (ET) | Away | Home | Model | Market | Open | Edge | Early |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    priced = out.dropna(subset=["edge"])
    top = priced.loc[priced.edge.abs().sort_values(ascending=False).index].head(15)
    if top.empty:
        lines.append("| | | | No market lines posted yet for these games. | | | | | |")
    for r in top.itertuples():
        lines.append(f"| {r.date:%m-%d} | {r.start_et} | {r.away} | {r.home}{' (N)' if r.neutral else ''} | "
                     f"{fmt(r.pred_margin)} | {fmt(r.market_margin)} | {fmt(r.open_margin)} | "
                     f"{fmt(r.edge)} | {'yes' if r.early_season else ''} |")
    lines += ["", "## Top 25 by rating", "", "| Rank | Team | Net | Off | Def | Tempo | Games |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in table.rename(columns={"def": "defense"}).head(25).itertuples():
        lines.append(f"| {r.rank} | {r.team} | {r.net:+.1f} | {r.off:+.1f} | {r.defense:+.1f} | "
                     f"{r.tempo:.1f} | {r.games} |")
    (LIVE / "predictions" / "latest.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args(argv)
    today = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(EASTERN).date()
    out = run(today, args.days)
    priced = out.market_margin.notna().sum()
    print(f"{today}: {len(out)} games predicted, {priced} with a market line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
