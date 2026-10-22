"""Filesystem layout. Everything is relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

RAW = DATA / "raw"                  # downloaded, rebuildable, not in git
RAW_ESPN = RAW / "espn"
PROCESSED = DATA / "processed"      # built tables, not in git
BACKTESTS = DATA / "backtests"      # backtest outputs, not in git

EXTERNAL = DATA / "external"        # small reference files, committed
TRACKER = EXTERNAL / "prediction_tracker"
TORVIK = EXTERNAL / "torvik"

LIVE = DATA / "live"                # daily snapshots from the scheduled job, committed

# ESPN data starts in 2006; the 2006 "season" is 2005-06.
FIRST_SEASON = 2006


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
