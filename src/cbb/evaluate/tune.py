"""Tune model settings on training seasons only, by coordinate search.

Objective: mean absolute error of the predicted margin on games with a market
line, over the training seasons. Test seasons are never touched here.

Usage:
    python -m cbb.evaluate.tune --train 2012-2019
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from cbb.evaluate.backtest import attach, load_games, parse_seasons, run
from cbb.model.ratings import Params
from cbb.paths import BACKTESTS, ensure

GRID = {
    "lam_eff": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    "regress": [0.55, 0.65, 0.75, 0.85],
    "recency_floor": [1.0, 0.8, 0.6, 0.4],
    "lam_final": [0.25, 0.5, 1.0, 2.0],
    "hca_prior": [1.8, 2.0, 2.2, 2.4, 2.6],
    "lam_tempo": [1.0, 3.0, 6.0],
    "blowout_cap": [None, 50.0, 40.0, 30.0],
    "new_team_net": [-8.0, -14.0, -20.0],
}

_GAMES = None


def _init():
    global _GAMES
    _GAMES = load_games()


def score(args) -> tuple[dict, float, float]:
    params, seasons = args
    p = Params(**params)
    d = attach(run(_GAMES, seasons, p, verbose=False), _GAMES)
    x = d[d.line.notna() & d.pred_margin.notna()]
    mae = (x.margin - x.pred_margin).abs().mean()
    line = (x.margin - x.line).abs().mean()
    return params, mae, line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="2012-2019")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    seasons = parse_seasons(args.train)

    best = asdict(Params())
    log = []
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        _, best_mae, line_mae = score_once(ex, best, seasons)
        print(f"start: model {best_mae:.3f}  line {line_mae:.3f}", flush=True)
        for rnd in range(args.rounds):
            for name, values in GRID.items():
                cands = [{**best, name: v} for v in values if v != best[name]]
                for params, mae, _ in ex.map(score, [(c, seasons) for c in cands]):
                    log.append({**params, "mae": mae})
                    if mae < best_mae - 1e-4:
                        best, best_mae = params, mae
                print(f"round {rnd + 1} {name:14s} -> {best[name]!s:6s} model {best_mae:.3f} "
                      f"(gap to line {best_mae - line_mae:+.3f})", flush=True)

    ensure(BACKTESTS)
    pd.DataFrame(log).to_csv(BACKTESTS / "tuning_log.csv", index=False)
    (BACKTESTS / "best_params.json").write_text(json.dumps(best, indent=2))
    print("best:", json.dumps(best))
    return 0


def score_once(ex, params, seasons):
    return next(iter(ex.map(score, [(params, seasons)])))


if __name__ == "__main__":
    sys.exit(main())
