# cbb-model: notes for working in this repo

College basketball model for betting markets (spreads, totals, moneylines). Python, `src/` layout.
Run modules with `PYTHONPATH=src python -m cbb.<module>` or after `pip install -e .`.

## Conventions
- **Season label = the year the season ends.** 2026-27 is season 2027.
- **Margins are from the home team's view** everywhere (positive = home wins by that much).
  A sportsbook spread of -4.5 on the home team is a market home margin of +4.5
  (`market_margin = -home_spread`). ThePredictionTracker's `line` is already a home margin.
- At neutral sites ESPN still names a "home" team; `neutral=True` turns off home court.
- Division I is inferred: a team with 10+ box-scored games in a season. Live, that count is
  zero for weeks, so `predict.daily` uses last season's D-I teams plus teams with 10+
  *scheduled* games (until Nov 5, 2026 it used the box-score flag and dropped every game).
- Possessions = FGA - OREB + TOV + 0.475 x FTA, averaged over both teams.

## Data
| Path | Source | In git |
| --- | --- | --- |
| `data/raw/espn/` | sportsdataverse GitHub releases: schedules, team box scores (2006 on) | no |
| `data/processed/games.csv.gz` | `cbb.build.games`: one row per game | no |
| `data/processed/lines.csv.gz` | `cbb.build.lines`: tracker lines matched to ESPN game ids | no |
| `data/processed/espn_lines.csv.gz` | `cbb.build.espn_lines`: per-game median of real sportsbooks (close/open spread, total, moneyline) from the ESPN history; projection sites and live odds excluded | no |
| `data/external/prediction_tracker/` | ThePredictionTracker season files: market line, opening line, other systems | yes |
| `data/external/espn_odds/` | ESPN per-game odds history (2012-13 on, many books); hourly chunked backfill, progress in `status.csv` / `progress.log` | yes |
| `data/live/odds/YYYY-MM.csv` | scheduled job: DraftKings open / current / close for today + 2 days | yes |
| `data/live/raw/` | scheduled job: raw ESPN JSON, gzipped | yes |
| `data/live/predictions/` | scheduled job: model vs market per game, plus `latest.md` | yes |
| `data/live/ratings/` | scheduled job: team ratings as of each day | yes |
| `data/live/grades.md` | scheduled job: paper-trading grades vs DraftKings (`cbb.evaluate.paper`) | yes |

## What runs where
- GitHub Actions has open internet; it fetches ESPN live odds, ThePredictionTracker, and the
  odds backfill. `snapshot-loop.yml` runs `scripts/snapshot_loop.sh`: an odds snapshot plus
  prediction refresh every 30 minutes, ~5.5 h per run, each run starting its successor (GitHub's
  cron dropped most scheduled runs; a 3-hourly cron only restarts a broken chain). It is the only
  writer to `data/live/`. This relies on the repo being public (free Actions minutes); a private
  repo would need an external trigger instead. `daily-snapshot.yml` is manual-only.
- Odds rows are appended only when a number changes; `data/live/predictions/log_YYYY-MM.csv`
  is the append-only paper-trading record (model and market numbers with timestamps).
- Claude's cloud workspace can only reach GitHub (sportsdataverse releases and this repo), so
  analysis and backtests run there on data the Actions jobs commit. Actions logs are not
  readable from the workspace: jobs write summaries into committed files instead.
- Torvik's site blocks automated downloads behind a browser check; do not work around it.

## Model (baseline)
`cbb.model.ratings`: weighted ridge regression for offense, defense (points per 100
possessions) and tempo, pulled toward a prior (last season's season-end ratings, regressed).
Home court is fixed (it inflates early in the season if fitted). Tuned settings live in
`src/cbb/model/params.json`; tune only on training seasons (`cbb.evaluate.tune`), report on
held-out seasons (`cbb.evaluate.backtest`). Judge everything against the market line on the
same games, and by month: November is where the market is weakest and the model too.
- Day-of-season drift (`cbb.model.drift`, `drift.json`, `season_drift`): opening-week games run
  ~3.8 poss/40 faster and ~2.9 pts/100 less efficient than full-season levels (team-adjusted,
  fit on 2012-19), fading by mid-season and reversing slightly in conference play. Fit removes
  it, predict adds it back; games need a `dos` column (days since the opener).
- `total_offset` (0.82): pred_total = regulation + offset, the training-season median gap, so
  totals target the typical game.
- Non-conference home calibration (`nc_home_*`, backtest `--nc-home`): built and tested, OFF.
- ESPN lists DraftKings as both "DraftKings" and "Draft Kings" (alternating by snapshot) and adds
  in-game "... - Live Odds" entries: merge the spellings and drop live rows when reading odds.

## Known results and caveats (updated Nov 5, 2026; held-out seasons 2020-26)
- Roster prior: 0.15 pts/game behind the market line overall, 0.25 in November.
- No edge vs the closing line (ATS ~50-51% at any disagreement size).
- Vs the opening line, after cleaning the tracker's broken openers: ~56% ATS when the model
  differs by 4+ points (n ~1,800), 53.6% at 3+; seasons range 52-61%. Before cleaning it read
  58.5% (bad openers inflated it). The closing line moves toward the model ~2/3 of the time.
- Same test on ESPN's own openers (ESPN BET / DraftKings, 2023-24 to 2025-26), after dropping
  ~135 suspect openers (moves > 6 pts, zero openers): 53.5% at 4+ (n=807), 54.6% at 5+ (n=313);
  by season 55.5 / 54.3 / 51.4%; November 49.9%. Thin: barely above break-even (52.4%).
- Totals (Oct 24): no edge. Vs close ~49% at any edge size; vs open ~51% at 4+ (n~2,200).
  Model trails closing totals by ~0.25 pts/game (0.45 in Nov). Totals lines price the typical
  (regulation) game; closing totals run ~0.8 below average actual totals because overtime
  (5.8% of games, ~19 extra pts) skews the mean. `pred_total` is the regulation number;
  `pred_total_with_ot` is the mean, for reference only.
- Suspect openers recur in every source; always filter them before testing against openers.
- Nov 5, 2026 (held-out 2020-26, roster prior): drift on vs off, spread MAE gap to the line
  0.153 -> 0.147 (Nov 0.246 -> 0.243); vs ESPN openers 4+ 53.3% -> 54.0% (n=787), tracker
  openers 4+ 56.4% -> 56.8%; totals gap to close 0.249 -> 0.242, bias vs close +0.57 -> -0.06.
- With last season's ratings as the prior (no roster model), November non-conference home
  games run ~1.10x the model's margin (buy-game favorites win bigger; 20+ pt lines: model
  5.4 pts less extreme than the line, line unbiased). The roster prior removes the slope
  (~0.99); a leftover ~+0.5 home intercept did not help held-out MAE.
- Live 2026-27, Nov 1-4 (150 games, preseason ratings only because of the D-I bug): spread
  MAE 10.69 vs DraftKings close 9.67; 4+ disagreement 22-29 vs close, 19-22 vs open; totals
  ran ~5 pts under the market before the drift fix (~+0.3 after).
- Caveats: backtest roster membership comes from that season's box scores (worth <= ~1 pt);
  model-refit timing vs when openers post (<= ~1 pt); the roster design and the 4-point
  threshold were chosen with test seasons visible. The tracker's opening-line book and time
  are unknown. Validate on our own timestamped DraftKings snapshots before staking money.
