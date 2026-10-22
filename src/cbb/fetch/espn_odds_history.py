"""Backfill ESPN's per-game odds history (open / close spread, total, moneyline).

For each completed game in a season's ESPN schedule, request ESPN's odds record
and flatten it with the same parser the live logger uses. Resumable: games
already in the output file are skipped. Meant to run in GitHub Actions (the
odds host is not reachable from every environment).

Output: data/external/espn_odds/odds_{season}.csv.gz

Usage:
    python -m cbb.fetch.espn_odds_history 2025                # whole season
    python -m cbb.fetch.espn_odds_history 2010-2026 --sample 25  # probe coverage
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from cbb.fetch.espn_history import fetch as fetch_espn_file
from cbb.fetch.espn_live import CORE_ODDS, ODDS_FIELDS, get_json, rows_from_odds_item
from cbb.paths import EXTERNAL, RAW_ESPN, ensure

OUT = EXTERNAL / "espn_odds"


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


def backfill(season: int, sample: int, pause: float) -> str:
    ensure(OUT)
    path = OUT / f"odds_{season}.csv.gz"
    done = (pd.read_csv(path, dtype={"game_id": str, "fetched_game_id": str}) if path.exists()
            else pd.DataFrame(columns=ODDS_FIELDS))
    seen = set(done.game_id.astype(str)) | set(done.get("fetched_game_id", pd.Series(dtype=str)).astype(str))

    s = schedule(season)
    if sample:
        s = s.sample(min(sample, len(s)), random_state=season)
    todo = [r for r in s.itertuples() if str(r.game_id) not in seen]

    rows, empty, errors = [], [], []
    for i, r in enumerate(todo, 1):
        try:
            core = get_json(CORE_ODDS.format(eid=r.game_id))
        except Exception as e:  # keep going; a later run retries
            errors.append(f"{r.game_id}: request {type(e).__name__}: {e}"[:300])
            continue
        items = (core or {}).get("items", [])
        got = []
        for item in items:
            try:
                got += rows_from_odds_item(meta(r), item, "core", "history")
            except Exception as e:
                errors.append(f"{r.game_id}: {type(e).__name__}: {e} | item keys: {sorted(item)[:15]}")
        if got:
            rows += got
        else:
            empty.append(r.game_id)
        if i % 250 == 0:
            print(f"  {season}: {i}/{len(todo)}", flush=True)
            _save(path, done, rows, empty)
        time.sleep(pause)
    _save(path, done, rows, empty)

    new = pd.DataFrame(rows)
    with_odds = new.game_id.nunique() if len(new) else 0
    providers = ", ".join(f"{k} {v}" for k, v in new.provider.value_counts().items()) if len(new) else "-"
    phases = ", ".join(f"{k} {v}" for k, v in new.phase.value_counts().items()) if len(new) else "-"
    msg = (f"{season}: {len(todo)} games requested, {with_odds} with odds, {len(empty)} without | "
           f"providers: {providers} | phases: {phases}")
    if errors:
        msg += f"\n  {len(errors)} errors, first: " + "\n  ".join(errors[:3])
    return msg


def _save(path, done, rows, empty):
    new = pd.DataFrame(rows, columns=ODDS_FIELDS)
    misses = pd.DataFrame({"fetched_game_id": [str(g) for g in empty]})
    out = pd.concat([done, new, misses], ignore_index=True)
    out.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seasons")
    ap.add_argument("--sample", type=int, default=0, help="games per season (0 = all)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args(argv)
    summary = []
    ensure(OUT)
    for season in parse_seasons(args.seasons):
        try:
            line = backfill(season, args.sample, args.pause)
        except Exception:
            import traceback
            line = f"{season}: FAILED\n" + traceback.format_exc()
        print(line, flush=True)
        summary.append(line)
        (OUT / "last_run_summary.txt").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
