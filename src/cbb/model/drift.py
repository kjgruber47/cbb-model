"""Day-of-season drift in pace and scoring rate.

Early-season D-I games are faster and less efficient than the same teams play
over the full season, and the gap fades as the season goes on. Left alone, the
model starts each season at last season's full-year pace and efficiency, which
underestimates opening-week pace, overestimates opening-week efficiency, and
understates blowout margins (margins scale with possessions).

The ratings fit removes the drift from each game before fitting and adds it back
when predicting (Params.season_drift), so team ratings and the league averages
(mu, tmu) describe full-season levels.

The curves are piecewise linear in days since the season opener, measured as
team-adjusted residuals of full-season fits on TRAINING seasons only, and saved in
drift.json next to this file.

Usage:
    python -m cbb.model.drift --fit 2012-2019
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd

from cbb.paths import ROOT

DRIFT_FILE = ROOT / "src" / "cbb" / "model" / "drift.json"
KNOTS = [0, 3, 7, 14, 21, 28, 42, 56, 70, 84, 98, 112, 126, 150]
OPENER_MIN_GAMES = 20


def season_start(dates) -> pd.Timestamp:
    """The opener: first date with OPENER_MIN_GAMES+ games (skips stray early games)."""
    c = pd.Series(pd.to_datetime(pd.Series(dates)).dt.normalize()).value_counts().sort_index()
    big = c[c >= OPENER_MIN_GAMES]
    return big.index[0] if len(big) else c.index[0]


def days_in(dates, start: pd.Timestamp) -> np.ndarray:
    return np.maximum((pd.to_datetime(pd.Series(dates)).dt.normalize() - start).dt.days.to_numpy(), 0)


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DRIFT_FILE.read_text())


def terms(dos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tempo adjustment in poss/40, efficiency adjustment in pts/100) for days into season."""
    c = load()
    dos = np.asarray(dos, dtype=float)
    return np.interp(dos, c["knots"], c["tempo"]), np.interp(dos, c["knots"], c["eff"])


def hat_basis(x: np.ndarray, knots: list[int]) -> np.ndarray:
    """Linear-interpolation weights: row i holds the weights of x[i] on each knot."""
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(x, knots, eye[k]) for k in range(len(knots))])


def estimate(games: pd.DataFrame, seasons: list[int], smooth: float = 50.0) -> dict:
    """Fit the curves from team-adjusted residuals of full-season fits."""
    from cbb.evaluate.backtest import d1_games
    from cbb.model.ratings import Params, Prior, fit

    p = Params()
    p_final = replace(p, lam_eff=p.lam_final, lam_tempo=p.lam_final, recency_floor=1.0)
    rows = []
    for s in seasons:
        sg = d1_games(games[games.season == s])
        teams = np.array(sorted(set(sg.home_id) | set(sg.away_id)))
        r = fit(sg, teams, Prior.flat(hca=p.hca_prior), p_final)
        idx = r.lookup()
        hi, ai = sg.home_id.map(idx).to_numpy(), sg.away_id.map(idx).to_numpy()
        tempo = 40 * sg.poss.to_numpy() / sg.minutes.to_numpy()
        eff = 100 * (sg.home_pts.to_numpy() + sg.away_pts.to_numpy()) / (2 * sg.poss.to_numpy())
        eff_hat = r.mu + (r.O[hi] - r.D[ai] + r.O[ai] - r.D[hi]) / 2  # home court cancels
        rows.append(pd.DataFrame({
            "season": s, "dos": days_in(sg.date, season_start(sg.date)),
            "r_tempo": tempo - (r.tmu + r.T[hi] + r.T[ai]), "r_eff": eff - eff_hat,
            "w": sg.poss.to_numpy() / 68.0,
        }))
    d = pd.concat(rows, ignore_index=True)
    B = hat_basis(d.dos.to_numpy(), KNOTS)
    W = d.w.to_numpy()
    # Second-difference penalty keeps the curve smooth between sparse knots.
    D2 = np.diff(np.eye(len(KNOTS)), n=2, axis=0)
    A = (B.T * W) @ B + smooth * D2.T @ D2
    out = {"knots": KNOTS, "seasons": f"{min(seasons)}-{max(seasons)}"}
    for col, key in (("r_tempo", "tempo"), ("r_eff", "eff")):
        beta = np.linalg.solve(A, (B.T * W) @ d[col].to_numpy())
        out[key] = [round(float(v), 3) for v in beta]
    # Diagnostics: raw residual means by week, to compare against the curve.
    d["week"] = np.minimum(d.dos // 7, 20)
    out["check_by_week"] = (d.groupby("week")[["r_tempo", "r_eff"]].mean().round(2)
                            .reset_index().to_dict(orient="records"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", default="2012-2019", help="training seasons, e.g. 2012-2019")
    args = ap.parse_args(argv)
    from cbb.evaluate.backtest import load_games, parse_seasons
    out = estimate(load_games(), parse_seasons(args.fit))
    for k, t, e in zip(out["knots"], out["tempo"], out["eff"]):
        print(f"day {k:>3}: tempo {t:+.2f} poss/40, efficiency {e:+.2f} pts/100")
    DRIFT_FILE.write_text(json.dumps({k: v for k, v in out.items() if k != "check_by_week"}, indent=1) + "\n")
    print(f"wrote {DRIFT_FILE}")
    print(pd.DataFrame(out["check_by_week"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
