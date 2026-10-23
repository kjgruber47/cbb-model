"""Backfill ESPN's per-game odds history (open / close spread, total, moneyline).

For each completed game in a season's ESPN schedule, request ESPN's odds record
and flatten it with the same parser the live logger uses. ESPN's history starts
in 2012-13 (season 2013).

Built to run in short, resumable chunks (the scheduled GitHub job runs one chunk
an hour until every season is done):
  - games already requested are skipped, so a chunk picks up where the last stopped;
  - `--max-minutes` stops cleanly before the time budget runs out;
  - progress is saved every 250 games and at the end of every chunk;
  - data/external/espn_odds/status.csv records each season's progress, and the
    line ALL_DONE is printed once every season is complete.

Output: data/external/espn_odds/odds_{season}.csv.gz

Usage:
    python -m cbb.fetch.espn_odds_history --next --max-minutes 40   # one chunk
    python -m cbb.fetch.espn_odds_history 2025                      # one season, no time limit
    python -m cbb.fetch.espn_odds_history 2013-2026 --sample 25     # probe coverage
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from cbb.fetch.espn_history import current_season, fetch as fetch_espn_file
from cbb.fetch.espn_live import CORE_ODDS, ODDS_FIELDS, get_json, rows_from_odds_item
from cbb.paths import EXTERNAL, RAW_ESPN, ensure

OUT = EXTERNAL / "espn_odds"
STATUS = OUT / "status.csv"
LOG = OUT / "progress.log"
FIRST_SEASON = 2013  # ESPN odds history begins in 2012-13


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def schedule(season: int) -> pd.DataFrame:
    fetch_espn_file(season, "schedule")
    s = pd.read_csv(RAW_ESPN / f"mbb_schedule_{season}.csv", low_memory=False)
    s = s[s.status_type_completed.fillna(False).astype(bool)].drop_duplicates("game_id")
    return s


def meta(row) -> dict:
    return {
        "game_id": row.game_id, "game_date": row.game_date, "start_utc": row.start_date,
        "status": row.status_type_name, "neutral_site": row.neutral_site,
        "home_id": row.home_id, "home": row.home_display_name,
        "away_id": row.away_id, "away": row.away_display_name,
        "home_score": row.home_score, "away_score": row.away_score,
    }


def load_done(path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str, "fetched_game_id": str}, low_memory=False)
    return pd.DataFrame(columns=[*ODDS_FIELDS, "fetched_game_id"])


def requested_ids(done: pd.DataFrame) -> set:
    ids = set(done.game_id.dropna().astype(str))
    if "fetched_game_id" in done:
        ids |= set(done.fetched_game_id.dropna().astype(str))
    return ids


def read_status() -> pd.DataFrame:
    if STATUS.exists():
        return pd.read_csv(STATUS)
    return pd.DataFrame(columns=["season", "scheduled", "requested", "with_odds", "complete", "updated_utc"])


def write_status(season: int, scheduled: int, done: pd.DataFrame) -> bool:
    req = len(requested_ids(done))
    with_odds = done.game_id.dropna().nunique()
    complete = req >= scheduled
    st = read_status()
    st = st[st.season != season]
    row = pd.DataFrame([{"season": season, "scheduled": scheduled, "requested": req,
                         "with_odds": with_odds, "complete": complete,
                         "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}])
    pd.concat([st, row], ignore_index=True).sort_values("season").to_csv(STATUS, index=False)
    return complete


def backfill(season: int, sample: int, pause: float, deadline: float | None) -> tuple[str, bool]:
    """Work through one season until done or out of time. Returns (message, complete)."""
    ensure(OUT)
    path = OUT / f"odds_{season}.csv.gz"
    done = load_done(path)
    seen = requested_ids(done)

    s = schedule(season)
    if sample:
        s = s.sample(min(sample, len(s)), random_state=season)
    todo = [r for r in s.itertuples() if str(r.game_id) not in seen]

    rows, empty, errors = [], [], []
    stopped_early = False
    for i, r in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            stopped_early = True
            break
        try:
            core = get_json(CORE_ODDS.format(eid=r.game_id))
        except Exception as e:  # skip; a later chunk retries it
            errors.append(f"{r.game_id}: request {type(e).__name__}: {e}"[:300])
            continue
        got = []
        for item in (core or {}).get("items", []):
            try:
                got += rows_from_odds_item(meta(r), item, "core", "history")
            except Exception as e:
                errors.append(f"{r.game_id}: {type(e).__name__}: {e}"[:300])
        if got:
            rows += got
        else:
            empty.append(r.game_id)
        if i % 250 == 0:
            done = _save(path, done, rows, empty)
            rows, empty = [], []
            print(f"  {season}: {i}/{len(todo)} this chunk", flush=True)
        time.sleep(pause)
    done = _save(path, done, rows, empty)

    complete = (not sample) and write_status(season, len(s), done)
    st = read_status().set_index("season")
    msg = (f"{season}: {st.at[season, 'requested'] if season in st.index else '?'} of {len(s)} games requested, "
           f"{done.game_id.dropna().nunique()} with odds"
           f"{' (paused: time budget)' if stopped_early else ''}{' COMPLETE' if complete else ''}")
    if errors:
        msg += f" | {len(errors)} request errors (retried next chunk), first: {errors[0]}"
    return msg, complete


def _save(path, done, rows, empty) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=ODDS_FIELDS)
    misses = pd.DataFrame({"fetched_game_id": [str(g) for g in empty]})
    out = pd.concat([done, new, misses], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def log(lines: list[str]) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    old = LOG.read_text().splitlines() if LOG.exists() else []
    LOG.write_text("\n".join((old + [f"{stamp} {line}" for line in lines])[-300:]) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seasons", nargs="?", help="e.g. 2025 or 2013-2026 (omit with --next)")
    ap.add_argument("--next", action="store_true", help="work on incomplete seasons, oldest first")
    ap.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    ap.add_argument("--sample", type=int, default=0, help="games per season (0 = all)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args(argv)

    ensure(OUT)
    start = time.time()
    before = read_status()
    req_before = int(before.requested.sum()) if len(before) else 0
    deadline = start + 60 * args.max_minutes if args.max_minutes else None
    all_seasons = list(range(FIRST_SEASON, current_season()))  # completed seasons only
    if args.next:
        st = read_status()
        finished = set(st.loc[st.complete.astype(str) == "True", "season"].astype(int))
        seasons = [s for s in all_seasons if s not in finished]
    else:
        seasons = parse_seasons(args.seasons)

    lines = []
    for season in seasons:
        if deadline and time.time() > deadline - 60:
            break
        try:
            msg, _ = backfill(season, args.sample, args.pause, deadline)
        except Exception as e:
            msg = f"{season}: FAILED {type(e).__name__}: {e}"[:400]
        print(msg, flush=True)
        lines.append(msg)
    log(lines or ["nothing to do"])

    st = read_status()
    print(f"PROGRESS {int(st.requested.sum()) - req_before if len(st) else 0}")
    finished = set(st.loc[st.complete.astype(str) == "True", "season"].astype(int))
    remaining = [s for s in all_seasons if s not in finished]
    if args.next and not remaining:
        print("ALL_DONE")
    else:
        print(f"remaining seasons: {remaining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
