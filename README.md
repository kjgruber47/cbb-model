# cbb-model

A college basketball predictive model built for betting markets: spreads, totals, moneylines and the other markets books offer across the full Division I slate.

## Layout

| Path | What it holds |
| --- | --- |
| `src/cbb/fetch/` | Downloaders: ESPN history, live ESPN schedule and odds, ThePredictionTracker lines |
| `src/cbb/build/` | Builds the clean one-row-per-game table |
| `src/cbb/model/` | Team ratings and game predictions |
| `src/cbb/evaluate/` | Walk-forward backtests against the market line |
| `data/live/` | Daily snapshots written by the scheduled GitHub Actions job (committed) |
| `data/external/` | Small reference files such as historical lines (committed) |
| `data/raw/`, `data/processed/` | Large or rebuildable data (not committed) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m cbb.fetch.espn_history      # ESPN schedules and team box scores, 2005-06 on
```
