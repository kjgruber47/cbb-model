"""Fetch current ESPN rosters for every D-I team (for the preseason model).

Team ids come from data/external/d1_teams.csv (last season's D-I teams plus
anyone added by hand). One request per team, gently paced.

Output: data/external/rosters/rosters_{season}.csv
  season, team_id, athlete_id, name, position, experience, fetched_utc

Usage:
    python -m cbb.fetch.rosters            # current season
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from cbb.fetch.espn_history import current_season
from cbb.fetch.espn_live import HEADERS, get_json
from cbb.paths import EXTERNAL, ensure

ROSTER = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{tid}/roster"


def athletes(payload: dict) -> list[dict]:
    """ESPN returns either a flat list or groups of athletes."""
    items = (payload or {}).get("athletes") or []
    flat = []
    for it in items:
        if isinstance(it, dict) and "items" in it:
            flat += it["items"]
        else:
            flat.append(it)
    return flat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=current_season())
    ap.add_argument("--pause", type=float, default=0.4)
    args = ap.parse_args(argv)

    teams = pd.read_csv(EXTERNAL / "d1_teams.csv")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, missing = [], []
    for tid in teams.team_id:
        try:
            data = get_json(ROSTER.format(tid=tid))
        except Exception as e:
            missing.append(f"{tid} {type(e).__name__}")
            continue
        found = athletes(data)
        if not found:
            missing.append(f"{tid} empty")
        for a in found:
            rows.append({
                "season": args.season, "team_id": tid, "athlete_id": a.get("id"),
                "name": a.get("displayName") or a.get("fullName"),
                "position": (a.get("position") or {}).get("abbreviation"),
                "experience": (a.get("experience") or {}).get("displayValue"),
                "fetched_utc": stamp,
            })
        time.sleep(args.pause)
    out_dir = EXTERNAL / "rosters"
    ensure(out_dir)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"rosters_{args.season}.csv", index=False)
    summary = (f"{stamp}: {out.team_id.nunique() if len(out) else 0} of {len(teams)} teams, "
               f"{len(out)} players; missing: {', '.join(missing[:20]) or 'none'}")
    (out_dir / f"summary_{args.season}.txt").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
