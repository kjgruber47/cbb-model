"""Walk-forward backtest: refit ratings each game day using only earlier games.

For every season, ratings start from last season's (regressed) values. On each
date, the model is fit on that season's D-I vs D-I games played before that
date and predicts the day's games. Nothing from the prediction date or later
is used. The season's final fit becomes next season's prior.

Usage:
    python -m cbb.evaluate.backtest                       # 2008..2026, default params
    python -m cbb.evaluate.backtest --seasons 2017-2026 --recency-floor 0.6 --lam-eff 3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from cbb.model.ratings import Params, Prior, fit, predict
from cbb.paths import BACKTESTS, PROCESSED, ensure


def load_games() -> pd.DataFrame:
    g = pd.read_csv(PROCESSED / "games.csv.gz", low_memory=False)
    g["date"] = pd.to_datetime(g.game_date)
    return g


def d1_games(g: pd.DataFrame) -> pd.DataFrame:
    return g[g.home_d1 & g.away_d1 & g.poss_ok].copy()


def roster_prior(base: Prior, data: pd.DataFrame, season: int, min_train: int = 3) -> Prior:
    """Swap last season's O/D for the roster model's, fit only on earlier seasons."""
    from cbb.model.preseason import fit_coefs, predict_prior
    hist = data[data.season < season]
    cur = data[data.season == season]
    if hist.season.nunique() < min_train or cur.empty:
        return base
    pred = predict_prior(cur, fit_coefs(hist))
    # Keep the base prior's overall offense/defense levels (they set the scoring level, not margins).
    o_shift = np.mean([base.O.get(t, 0.0) for t in pred.team_id]) - pred["off"].mean()
    d_shift = np.mean([base.D.get(t, 0.0) for t in pred.team_id]) - pred["def"].mean()
    return Prior(O=dict(zip(pred.team_id, pred["off"] + o_shift)),
                 D=dict(zip(pred.team_id, pred["def"] + d_shift)),
                 T=base.T, mu=base.mu, tmu=base.tmu, hca=base.hca)


def run(games: pd.DataFrame, seasons: list[int], p: Params, verbose: bool = True,
        roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """Walk-forward predictions. Pass `roster` (preseason features with targets) to use
    the roster-based preseason prior instead of last season's ratings."""
    games = d1_games(games)
    first = min(seasons)
    prior = Prior.flat(hca=p.hca_prior)
    preds = []
    # Warm up from two seasons earlier so the first scored season has a real prior.
    for season in range(max(first - 2, games.season.min()), max(seasons) + 1):
        sg = games[games.season == season].sort_values("date")
        teams = np.array(sorted(set(sg.home_id) | set(sg.away_id)))
        t0 = time.time()
        season_prior = roster_prior(prior, roster, season) if roster is not None else prior
        if season in seasons:
            for d in sg.date.unique():
                train = sg[sg.date < d]
                today = sg[sg.date == d]
                r = fit(train, teams, season_prior, p, as_of=d)
                out = predict(r, today, p)
                out["season"] = season
                out["date"] = d
                preds.append(out)
        # Season-end fit for next year's prior: light shrinkage, no recency decay,
        # so shrinkage does not compound from one season to the next.
        p_final = replace(p, lam_eff=p.lam_final, lam_tempo=p.lam_final, recency_floor=1.0)
        final = fit(sg, teams, season_prior, p_final, as_of=sg.date.max() + pd.Timedelta(days=1))
        prior = Prior.from_ratings(final, p)
        if verbose:
            print(f"  {season}: {'scored' if season in seasons else 'warm-up'} "
                  f"({len(sg)} games, {time.time() - t0:.1f}s)", flush=True)
    return pd.concat(preds, ignore_index=True)


def attach(preds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    lines = pd.read_csv(PROCESSED / "lines.csv.gz")
    cols = ["game_id", "home_team", "away_team", "neutral", "home_pts", "away_pts", "margin", "total",
            "season_type", "conference_game"]
    d = preds.merge(games[cols], on="game_id", how="left")
    return d.merge(lines[["game_id", "line", "line_open", "sys_dok", "sys_avg"]], on="game_id", how="left")


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """Model vs market on the same games (those with a line)."""
    x = d[d.line.notna() & d.pred_margin.notna()].copy()
    x["err_model"] = (x.margin - x.pred_margin).abs()
    x["err_line"] = (x.margin - x.line).abs()
    x["month"] = x.date.dt.month.map({11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Mar"})

    def agg(g):
        return pd.Series({
            "games": len(g),
            "mae_model": g.err_model.mean(),
            "mae_line": g.err_line.mean(),
            "gap": g.err_model.mean() - g.err_line.mean(),
            "bias": (g.pred_margin - g.margin).mean(),
        })

    by_season = x.groupby("season").apply(agg, include_groups=False)
    by_month = x.groupby("month").apply(agg, include_groups=False).reindex(["Nov", "Dec", "Jan", "Feb", "Mar"])
    total = agg(x).to_frame("all").T
    return pd.concat({"season": by_season, "month": by_month, "total": total})


def incremental_value(d: pd.DataFrame) -> dict:
    """Does the model add information the line lacks? Regress margin on both."""
    x = d[d.line.notna() & d.pred_margin.notna()]
    X = np.column_stack([np.ones(len(x)), x.line, x.pred_margin - x.line])
    beta, *_ = np.linalg.lstsq(X, x.margin.to_numpy(), rcond=None)
    resid = x.margin.to_numpy() - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var())
    return {"coef_line": beta[1], "coef_model_minus_line": beta[2], "t_stat": beta[2] / se[2], "n": len(x)}


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default="2010-2026")
    ap.add_argument("--recency-full", type=float)
    ap.add_argument("--recency-floor", type=float)
    ap.add_argument("--lam-eff", type=float)
    ap.add_argument("--lam-final", type=float)
    ap.add_argument("--lam-tempo", type=float)
    ap.add_argument("--regress", type=float)
    ap.add_argument("--blowout-cap", type=float)
    ap.add_argument("--prior", choices=["last", "roster"], default="last",
                    help="preseason prior: last season's ratings, or the roster model")
    ap.add_argument("--params", help="JSON file of settings (default: model/params.json if present)")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args(argv)

    import json
    from cbb.predict.daily import PARAMS_FILE
    pfile = args.params or (PARAMS_FILE if PARAMS_FILE.exists() else None)
    p = Params(**json.loads(open(pfile).read())) if pfile else Params()
    overrides = {k: v for k, v in {
        "recency_full": args.recency_full, "recency_floor": args.recency_floor,
        "lam_eff": args.lam_eff, "lam_final": args.lam_final, "lam_tempo": args.lam_tempo,
        "regress": args.regress, "blowout_cap": args.blowout_cap}.items() if v is not None}
    p = replace(p, **overrides)
    print("params:", asdict(p))

    games = load_games()
    roster = pd.read_csv(PROCESSED / "preseason_features.csv.gz") if args.prior == "roster" else None
    preds = run(games, parse_seasons(args.seasons), p, roster=roster)
    d = attach(preds, games)
    ensure(BACKTESTS)
    d.to_csv(BACKTESTS / f"{args.tag}.csv.gz", index=False)

    pd.set_option("display.width", 140)
    print(summarize(d).round(2).to_string())
    print("incremental value:", {k: round(v, 3) for k, v in incremental_value(d).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
