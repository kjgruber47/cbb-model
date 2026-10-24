# cbb-model: notes for working in this repo

College basketball model for betting markets (spreads, totals, moneylines). Python, `src/` layout.
Run modules with `PYTHONPATH=src python -m cbb.<module>` or after `pip install -e .`.

## Conventions
- **Season label = the year the season ends.** 2026-27 is season 2027.
- **Margins are from the home team's view** everywhere (positive = home wins by that much).
  A sportsbook spread of -4.5 on the home team is a market home margin of +4.5
  (`market_margin = -home_spread`). ThePredictionTracker's `line` is already a home margin.
- At neutral sites ESPN still names a "home" team; `neutral=True` turns off home court.
- Division I is inferred: a team with 10+ box-scored games in a season.
- Possessions = FGA - OREB + TOV + 0.475 x FTA, averaged over both teams.

## Data
| Path | Source | In git |
| --- | --- | --- |
| `data/raw/espn/` | sportsdataverse GitHub releases: schedules, team box scores (2006 on) | no |
| `data/processed/games.csv.gz` | `cbb.build.games`: one row per game | no |
| `data/processed/lines.csv.gz` | `cbb.build.lines`: tracker lines matched to ESPN game ids | no |
| `data/external/prediction_tracker/` | ThePredictionTracker season files: market line, opening line, other systems | yes |
| `data/external/espn_odds/` | ESPN per-game odds history (2012-13 on, many books); hourly chunked backfill, progress in `status.csv` / `progress.log` | yes |
| `data/live/odds/YYYY-MM.csv` | scheduled job: DraftKings open / current / close for today + 2 days | yes |
| `data/live/raw/` | scheduled job: raw ESPN JSON, gzipped | yes |
| `data/live/predictions/` | scheduled job: model vs market per game, plus `latest.md` | yes |
| `data/live/ratings/` | scheduled job: team ratings as of each day | yes |

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

## Known results and caveats (Oct 22, 2026; held-out seasons 2020-26)
- Roster prior: 0.15 pts/game behind the market line overall, 0.25 in November.
- No edge vs the closing line (ATS ~50-51% at any disagreement size).
- Vs the opening line, after cleaning the tracker's broken openers: ~56% ATS when the model
  differs by 4+ points (n ~1,800), 53.6% at 3+; seasons range 52-61%. Before cleaning it read
  58.5% (bad openers inflated it). The closing line moves toward the model ~2/3 of the time.
- Caveats: backtest roster membership comes from that season's box scores (worth <= ~1 pt);
  model-refit timing vs when openers post (<= ~1 pt); the roster design and the 4-point
  threshold were chosen with test seasons visible. The tracker's opening-line book and time
  are unknown. Validate on our own timestamped DraftKings snapshots before staking money.
