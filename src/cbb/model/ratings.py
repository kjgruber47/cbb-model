"""Team ratings: offense, defense and tempo, fit by weighted ridge regression.

Efficiency model (points per 100 possessions), two rows per game:
    ppp_home = mu + O[home] - D[away] + hca * at_home
    ppp_away = mu + O[away] - D[home] - hca * at_home
Tempo model (possessions per 40 minutes), one row per game:
    tempo    = tmu + T[home] + T[away]

Each rating is pulled toward a prior (last season's rating, regressed toward
average) with a strength measured in games: lam=8 means the prior counts like
8 games of evidence. Games are weighted by recency and by possessions.

Prediction for a game:
    poss       = tmu + T[home] + T[away]
    home_pts   = poss * (mu + O[home] - D[away] + hca * at_home) / 100
    away_pts   = poss * (mu + O[away] - D[home] - hca * at_home) / 100
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Params:
    lam_eff: float = 3.0          # prior strength, in games, for O and D during the season
    lam_tempo: float = 3.0        # prior strength, in games, for T
    lam_final: float = 1.0        # light shrinkage for the season-end fit that seeds next season
    lam_mu: float = 200.0         # prior strength for the league average
    lam_hca: float = 1e7          # effectively fixed: early-season fits otherwise inflate it
    hca_prior: float = 2.2        # per side, points per 100 poss (about 3 pts total)
    recency_full: float = 40.0    # games this many days old or newer get full weight ...
    recency_floor: float = 1.0    # ... then weight falls linearly to this floor at 2x that age
    regress: float = 0.70         # share of last season's rating carried forward
    new_team_net: float = -14.0   # prior net rating for teams new to D-I
    blowout_cap: float | None = None  # cap |ppp_home - ppp_away| at this value


@dataclass
class Ratings:
    teams: np.ndarray                     # team ids, index order of O, D, T
    O: np.ndarray
    D: np.ndarray
    T: np.ndarray
    mu: float
    hca: float
    tmu: float
    games: np.ndarray = field(default=None)  # games played per team

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team_id": self.teams, "off": self.O, "def": self.D, "net": self.O + self.D,
            "tempo": self.tmu + self.T,  # possessions per 40 vs an average opponent
            "games": self.games.astype(int),
        })

    def lookup(self) -> dict:
        return {t: i for i, t in enumerate(self.teams)}


@dataclass
class Prior:
    """Starting values for a season. Missing teams get new-team defaults."""
    O: dict
    D: dict
    T: dict
    mu: float
    tmu: float
    hca: float

    @classmethod
    def flat(cls, mu: float = 103.0, tmu: float = 68.0, hca: float = 1.6) -> "Prior":
        return cls({}, {}, {}, mu, tmu, hca)

    @classmethod
    def from_ratings(cls, r: Ratings, p: Params) -> "Prior":
        k = p.regress
        return cls(
            O=dict(zip(r.teams, k * r.O)), D=dict(zip(r.teams, k * r.D)), T=dict(zip(r.teams, k * r.T)),
            mu=r.mu, tmu=r.tmu, hca=r.hca,
        )

    def vectors(self, teams: np.ndarray, p: Params):
        half = p.new_team_net / 2
        O = np.array([self.O.get(t, half) for t in teams])
        D = np.array([self.D.get(t, half) for t in teams])
        T = np.array([self.T.get(t, 0.0) for t in teams])
        return O, D, T


def recency_weights(days_ago: np.ndarray, full: float, floor: float) -> np.ndarray:
    """Torvik-style: 1.0 up to `full` days old, then linear down to `floor` at 2*full days."""
    if floor >= 1.0:
        return np.ones_like(days_ago, dtype=float)
    slope = (1.0 - floor) / full
    return np.clip(1.0 - slope * (days_ago - full), floor, 1.0)


def fit(games: pd.DataFrame, teams: np.ndarray, prior: Prior, p: Params,
        as_of: pd.Timestamp | None = None) -> Ratings:
    """Fit ratings from `games` (D-I vs D-I, completed, before the prediction date)."""
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    pO, pD, pT = prior.vectors(teams, p)

    if len(games) == 0:
        return Ratings(teams, pO, pD, pT, prior.mu, prior.hca, prior.tmu, np.zeros(n))

    hi = games.home_id.map(idx).to_numpy()
    ai = games.away_id.map(idx).to_numpy()
    at_home = (~games.neutral.to_numpy()).astype(float)
    poss = games.poss.to_numpy()
    ppp_h = 100 * games.home_pts.to_numpy() / poss
    ppp_a = 100 * games.away_pts.to_numpy() / poss
    if p.blowout_cap:
        mid = (ppp_h + ppp_a) / 2
        gap = np.clip(ppp_h - ppp_a, -p.blowout_cap, p.blowout_cap)
        ppp_h, ppp_a = mid + gap / 2, mid - gap / 2

    as_of = as_of if as_of is not None else games.date.max() + pd.Timedelta(days=1)
    days_ago = (as_of - games.date).dt.days.to_numpy().astype(float)
    w_game = recency_weights(days_ago, p.recency_full, p.recency_floor)
    w_eff = w_game * poss / 68.0

    # ---- efficiency: params = [mu, hca, O_0..O_n-1, D_0..D_n-1]
    m = len(games)
    rows = np.arange(2 * m)
    off_team = np.concatenate([hi, ai])
    def_team = np.concatenate([ai, hi])
    hsign = np.concatenate([at_home, -at_home])
    y = np.concatenate([ppp_h, ppp_a])
    w = np.concatenate([w_eff, w_eff])

    k = 2 + 2 * n
    X = sparse.csr_matrix(
        (np.concatenate([np.ones(2 * m), hsign, np.ones(2 * m), -np.ones(2 * m)]),
         (np.concatenate([rows, rows, rows, rows]),
          np.concatenate([np.zeros(2 * m, int), np.ones(2 * m, int), 2 + off_team, 2 + n + def_team]))),
        shape=(2 * m, k),
    )
    lam = np.concatenate([[p.lam_mu, p.lam_hca], np.full(2 * n, p.lam_eff)])
    prior_vec = np.concatenate([[prior.mu, prior.hca], pO, pD])
    XtW = X.T.multiply(w)
    A = (XtW @ X).toarray() + np.diag(lam)
    b = XtW @ y + lam * prior_vec
    beta = np.linalg.solve(A, b)
    mu, hca, O, D = beta[0], beta[1], beta[2:2 + n], beta[2 + n:]

    # ---- tempo: params = [tmu, T_0..T_n-1]
    tempo = 40 * poss / games.minutes.to_numpy()
    Xt = sparse.csr_matrix(
        (np.concatenate([np.ones(m), np.ones(m), np.ones(m)]),
         (np.concatenate([np.arange(m)] * 3), np.concatenate([np.zeros(m, int), 1 + hi, 1 + ai]))),
        shape=(m, 1 + n),
    )
    lam_t = np.concatenate([[p.lam_mu], np.full(n, p.lam_tempo)])
    prior_t = np.concatenate([[prior.tmu], pT])
    XtWt = Xt.T.multiply(w_game)
    At = (XtWt @ Xt).toarray() + np.diag(lam_t)
    bt = XtWt @ tempo + lam_t * prior_t
    bt_ = np.linalg.solve(At, bt)
    tmu, T = bt_[0], bt_[1:]

    played = np.bincount(np.concatenate([hi, ai]), minlength=n)
    return Ratings(teams, O, D, T, mu, hca, tmu, played)


def predict(r: Ratings, games: pd.DataFrame) -> pd.DataFrame:
    """Expected possessions, points, margin and total for each game."""
    idx = r.lookup()
    hi = games.home_id.map(idx)
    ai = games.away_id.map(idx)
    ok = hi.notna() & ai.notna()
    hi = hi.fillna(0).astype(int).to_numpy()
    ai = ai.fillna(0).astype(int).to_numpy()
    at_home = (~games.neutral.to_numpy()).astype(float)
    poss = r.tmu + r.T[hi] + r.T[ai]
    home_ppp = r.mu + r.O[hi] - r.D[ai] + r.hca * at_home
    away_ppp = r.mu + r.O[ai] - r.D[hi] - r.hca * at_home
    out = pd.DataFrame({
        "game_id": games.game_id.to_numpy(),
        "pred_poss": poss,
        "pred_home_pts": poss * home_ppp / 100,
        "pred_away_pts": poss * away_ppp / 100,
        "home_games": r.games[hi], "away_games": r.games[ai],
    })
    out["pred_margin"] = out.pred_home_pts - out.pred_away_pts
    out["pred_total"] = out.pred_home_pts + out.pred_away_pts
    out.loc[~ok.to_numpy(), ["pred_margin", "pred_total"]] = np.nan
    return out
