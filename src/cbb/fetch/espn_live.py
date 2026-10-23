"""Snapshot ESPN's schedule and betting odds for upcoming games.

Runs several times a day from the scheduled GitHub Actions job. Each run:
  1. pulls the D-I scoreboard for today and the next two days,
  2. pulls ESPN's detailed odds for every game on it (open / current / close
     where ESPN provides them),
  3. appends flat rows to data/live/odds/YYYY-MM.csv, and
  4. saves the raw JSON (gzipped) under data/live/raw/ so nothing is lost if the
     flattening below misses a field.

Usage:
    python -m cbb.fetch.espn_live                 # today + 2 days
    python -m cbb.fetch.espn_live --days 1        # today only
    python -m cbb.fetch.espn_live --date 20261103 # one specific date
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from cbb.paths import LIVE, ensure

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)
CORE_ODDS = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
    "/events/{eid}/competitions/{eid}/odds"
)
HEADERS = {"User-Agent": "cbb-model research (github.com/kjgruber47/cbb-model)"}
EASTERN = ZoneInfo("America/New_York")

ODDS_FIELDS = [
    "snapshot_utc", "game_id", "game_date", "start_utc", "status", "neutral_site",
    "home_id", "home", "away_id", "away", "home_score", "away_score",
    "source", "provider_id", "provider", "phase",
    "home_spread", "total", "home_ml", "away_ml", "home_spread_price", "away_spread_price",
    "over_price", "under_price", "details",
]


def get_json(url: str, params: dict | None = None, tries: int = 3) -> dict | None:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def num(x):
    """ESPN mixes numbers, numeric strings and strings like '+150' or 'EVEN'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip().replace("+", "")
    if s.upper() in {"EVEN", "EV", "PK", "PICK"}:
        return 100 if s.upper().startswith("EV") else 0
    try:
        return float(s)
    except ValueError:
        return None


def game_meta(event: dict) -> dict:
    comp = (event.get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    home, away = teams.get("home", {}), teams.get("away", {})
    start = event.get("date")
    start_et = None
    if start:
        start_et = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(EASTERN)
    return {
        "game_id": event.get("id"),
        "game_date": start_et.date().isoformat() if start_et else None,
        "start_utc": start,
        "status": (comp.get("status") or event.get("status") or {}).get("type", {}).get("name"),
        "neutral_site": comp.get("neutralSite"),
        "home_id": home.get("team", {}).get("id"),
        "home": home.get("team", {}).get("displayName"),
        "away_id": away.get("team", {}).get("id"),
        "away": away.get("team", {}).get("displayName"),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
    }


def side(block: dict | None, *keys):
    """Walk nested dicts, returning None at the first missing key."""
    cur = block
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def rows_from_odds_item(meta: dict, item: dict, source: str, snap: str) -> list[dict]:
    """One ESPN odds object can carry current numbers plus 'open'/'close' blocks."""
    provider = item.get("provider") or {}
    out = []

    def row(phase: str, spread, total, hml, aml, hsp, asp, op, up):
        out.append({
            **meta, "snapshot_utc": snap, "source": source,
            "provider_id": provider.get("id"), "provider": provider.get("name"), "phase": phase,
            "home_spread": num(spread), "total": num(total),
            "home_ml": num(hml), "away_ml": num(aml),
            "home_spread_price": num(hsp), "away_spread_price": num(asp),
            "over_price": num(op), "under_price": num(up),
            "details": item.get("details"),
        })

    # Current numbers. ESPN's 'spread' is the home spread in recent feeds.
    home_odds, away_odds = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
    row(
        "current", item.get("spread"), item.get("overUnder"),
        home_odds.get("moneyLine"), away_odds.get("moneyLine"),
        home_odds.get("spreadOdds"), away_odds.get("spreadOdds"),
        item.get("overOdds"), item.get("underOdds"),
    )
    # Newer feeds nest open / current / close under each side and under 'pointSpread'/'total'.
    for phase in ("open", "close"):
        hs = side(item, "pointSpread", "home", phase, "line") or side(home_odds, phase, "pointSpread", "american")
        tot = side(item, "total", "over", phase, "line") or side(item, phase, "total", "american")
        hml = side(item, "moneyline", "home", phase, "odds") or side(home_odds, phase, "moneyLine", "american")
        aml = side(item, "moneyline", "away", phase, "odds") or side(away_odds, phase, "moneyLine", "american")
        hsp = side(item, "pointSpread", "home", phase, "odds") or side(home_odds, phase, "spread", "american")
        asp = side(item, "pointSpread", "away", phase, "odds") or side(away_odds, phase, "spread", "american")
        op = side(item, "total", "over", phase, "odds")
        up = side(item, "total", "under", phase, "odds")
        if isinstance(tot, str) and tot[:1] in "ou":  # e.g. 'o145.5'
            tot = tot[1:]
        if any(v is not None for v in (hs, tot, hml, aml)):
            row(phase, hs, tot, hml, aml, hsp, asp, op, up)
    return out


def snapshot(dates: list[dt.date]) -> tuple[list[dict], dict]:
    snap = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []
    raw: dict = {"snapshot_utc": snap, "scoreboards": {}, "core_odds": {}}
    for d in dates:
        ymd = d.strftime("%Y%m%d")
        board = get_json(SCOREBOARD, {"dates": ymd, "groups": 50, "limit": 500})
        raw["scoreboards"][ymd] = board
        for event in (board or {}).get("events", []):
            meta = game_meta(event)
            comp = (event.get("competitions") or [{}])[0]
            for item in comp.get("odds") or []:
                rows += rows_from_odds_item(meta, item, "scoreboard", snap)
            core = get_json(CORE_ODDS.format(eid=event["id"]))
            raw["core_odds"][event["id"]] = core
            for item in (core or {}).get("items", []):
                rows += rows_from_odds_item(meta, item, "core", snap)
            time.sleep(0.25)  # be polite
    return rows, raw


KEY = ["game_id", "source", "provider", "phase"]
PRICES = ["home_spread", "total", "home_ml", "away_ml", "home_spread_price", "away_spread_price",
          "over_price", "under_price"]


def changed_only(rows: list[dict], path: Path) -> list[dict]:
    """Drop rows whose numbers match the last logged row for the same game/book/phase."""
    if not rows or not path.exists():
        return rows
    old = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    last = old.sort_values("snapshot_utc").drop_duplicates(KEY, keep="last").set_index(KEY)[PRICES]
    keep = []
    for r in rows:
        k = (str(r["game_id"]), r["source"], r["provider"], r["phase"])
        if k not in last.index:
            keep.append(r)
            continue
        prev = last.loc[k]
        if isinstance(prev, pd.DataFrame):
            prev = prev.iloc[-1]
        same = all((pd.isna(r[c]) and pd.isna(prev[c])) or
                   (not pd.isna(r[c]) and not pd.isna(prev[c]) and float(r[c]) == float(prev[c]))
                   for c in PRICES)
        if not same:
            keep.append(r)
    return keep


def append_rows(rows: list[dict]) -> tuple[Path | None, int]:
    """Append rows to this month's file, keeping only numbers that changed."""
    if not rows:
        return None, 0
    month = rows[0]["snapshot_utc"][:7]
    path = LIVE / "odds" / f"{month}.csv"
    ensure(path.parent)
    rows = changed_only(rows, path)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ODDS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    return path, len(rows)


def save_slate(raw: dict) -> Path:
    """Every game on ESPN's scoreboard for the dates fetched (overwritten each run)."""
    games = []
    for board in raw["scoreboards"].values():
        for ev in (board or {}).get("events", []):
            m = game_meta(ev)
            comp = (ev.get("competitions") or [{}])[0]
            teams = {c.get("homeAway"): c.get("team", {}) for c in comp.get("competitors", [])}
            games.append({**m, "home_location": teams.get("home", {}).get("location"),
                          "away_location": teams.get("away", {}).get("location"),
                          "snapshot_utc": raw["snapshot_utc"]})
    path = LIVE / "slate_latest.csv"
    ensure(path.parent)
    pd.DataFrame(games).to_csv(path, index=False)
    return path


def save_raw(raw: dict) -> Path | None:
    """Keep one raw snapshot per UTC day (for debugging the parser), not every run."""
    folder = LIVE / "raw" / raw["snapshot_utc"][:10]
    if folder.exists() and any(folder.glob("espn_*.json.gz")):
        return None
    stamp = raw["snapshot_utc"].replace(":", "").replace("-", "")
    path = folder / f"espn_{stamp}.json.gz"
    ensure(folder)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(raw, f, separators=(",", ":"))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=int, default=3, help="today plus this many days minus one")
    p.add_argument("--date", help="a single date, YYYYMMDD")
    args = p.parse_args(argv)

    if args.date:
        dates = [dt.datetime.strptime(args.date, "%Y%m%d").date()]
    else:
        today = dt.datetime.now(EASTERN).date()
        dates = [today + dt.timedelta(days=i) for i in range(args.days)]

    rows, raw = snapshot(dates)
    games = sum(len((b or {}).get("events", [])) for b in raw["scoreboards"].values())
    csv_path, written = append_rows(rows)
    slate_path = save_slate(raw)
    raw_path = save_raw(raw)
    print(f"{raw['snapshot_utc']}: {games} games, {len(rows)} odds rows, {written} changed and written")
    print(f"  rows  -> {csv_path}")
    print(f"  slate -> {slate_path}")
    print(f"  raw   -> {raw_path or 'skipped (already have one today)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
