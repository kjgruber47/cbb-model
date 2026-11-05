"""Grade the live paper-trading record against DraftKings and final scores.

For every completed D-I game the model priced before tip-off:
  - the model's number: the last logged prediction before the start time
    (and, for the opener test, the first one logged),
  - DraftKings' opening and closing numbers (pregame only; ESPN's in-game
    "Live Odds" entries are dropped; "Draft Kings" and "DraftKings" are the same book),
  - the final score.

Reports, for spreads and totals:
  - mean absolute error of model, opener and closer against results,
  - against-the-spread record vs the close and vs the open, by size of disagreement
    (pushes excluded; break-even at -110 is 52.4%),
  - line movement: how often the line moved toward the model between open and close.

Usage:
    python -m cbb.evaluate.paper                    # everything logged so far
    python -m cbb.evaluate.paper --since 2026-11-03
    python -m cbb.evaluate.paper --write            # also save data/live/grades.md
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from cbb.paths import LIVE, RAW_ESPN

BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 99)]


def dk_lines() -> pd.DataFrame:
    """Pregame DraftKings open and close per game (home margin = -home spread)."""
    o = pd.concat([pd.read_csv(f) for f in sorted((LIVE / "odds").glob("*.csv"))], ignore_index=True)
    prov = o.provider.astype(str).str.lower().str.replace(" ", "")
    o = o[prov.str.startswith("draftkings") & ~prov.str.contains("live")].copy()
    o["snap"] = pd.to_datetime(o.snapshot_utc, utc=True)
    o["start"] = pd.to_datetime(o.start_utc, utc=True)
    o["margin"] = -pd.to_numeric(o.home_spread, errors="coerce")
    o["total"] = pd.to_numeric(o.total, errors="coerce")
    o = o.sort_values("snap")

    # Close: ESPN's 'close' phase; failing that, the last 'current' seen before tip-off.
    close = o[o.phase == "close"].groupby("game_id")[["margin", "total"]].last()
    pre = o[(o.phase == "current") & (o.snap < o.start)].groupby("game_id")[["margin", "total"]].last()
    close = close.combine_first(pre)
    opn = o[o.phase == "open"].groupby("game_id")[["margin", "total"]].first()
    out = close.add_prefix("close_").join(opn.add_prefix("open_"), how="outer")
    out["start"] = o.groupby("game_id").start.first()
    return out.reset_index()


def model_numbers() -> pd.DataFrame:
    """First and last pregame model numbers per game from the append-only log."""
    log = pd.concat([pd.read_csv(f) for f in sorted((LIVE / "predictions").glob("log_*.csv"))],
                    ignore_index=True)
    log["run"] = pd.to_datetime(log.run_utc, utc=True)
    return log.sort_values("run")[["game_id", "run", "date", "away", "home", "neutral",
                                   "pred_margin", "pred_total"]]


def results(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        f = RAW_ESPN / f"mbb_schedule_{s}.csv"
        if f.exists():
            d = pd.read_csv(f, low_memory=False, usecols=["game_id", "status_type_completed",
                                                          "home_score", "away_score"])
            frames.append(d[d.status_type_completed == True])  # noqa: E712
    r = pd.concat(frames).drop_duplicates("game_id")
    r["actual_margin"] = r.home_score - r.away_score
    r["actual_total"] = r.home_score + r.away_score
    return r[["game_id", "actual_margin", "actual_total"]]


def assemble(seasons: list[int], since: str | None = None) -> pd.DataFrame:
    lines = dk_lines()
    log = model_numbers().merge(lines[["game_id", "start"]], on="game_id")
    pre = log[log.run < log.start]
    first = pre.groupby("game_id")[["pred_margin", "pred_total"]].first().add_suffix("_first")
    last = pre.groupby("game_id").last()
    g = last.join(first).reset_index().drop(columns=["start"])
    g = g.merge(lines, on="game_id").merge(results(seasons), on="game_id")
    if since:
        g = g[g.date >= since]
    return g


def ats(pred, line, actual):
    """+1 win, -1 loss, 0 push, NaN when the model agrees with the line or data is missing."""
    side = np.sign(pred - line)
    res = np.sign(actual - line)
    out = side * res
    out[(side == 0) | np.isnan(side) | np.isnan(res)] = np.nan
    return out


def record(w: pd.Series) -> str:
    w = w.dropna()
    wins, losses = (w == 1).sum(), (w == -1).sum()
    n = wins + losses
    pct = f"{100 * wins / n:.1f}%" if n else "-"
    return f"{wins}-{losses} ({pct})"


def report(g: pd.DataFrame) -> str:
    out = [f"Games graded: {len(g)} ({g.date.min()} to {g.date.max()})", ""]
    for kind, pred, real in (("Spreads", "pred_margin", "actual_margin"),
                             ("Totals", "pred_total", "actual_total")):
        key = "margin" if kind == "Spreads" else "total"
        c, o = f"close_{key}", f"open_{key}"
        out += [f"## {kind}", ""]
        has = g.dropna(subset=[pred, c, real])
        out.append(f"Mean absolute error, {len(has)} games: model {(has[pred] - has[real]).abs().mean():.2f}, "
                   f"DraftKings close {(has[c] - has[real]).abs().mean():.2f}"
                   + (f", open {(has[o] - has[real]).abs().mean():.2f}" if has[o].notna().any() else ""))
        out.append(f"Model minus close, average: {(has[pred] - has[c]).mean():+.2f}")
        out += ["", "| Disagreement | vs close | vs open | Line moved toward model |",
                "| --- | --- | --- | --- |"]
        w_close = pd.Series(ats(g[pred].to_numpy(), g[c].to_numpy(), g[real].to_numpy()), index=g.index)
        first = f"{pred}_first"
        w_open = pd.Series(ats(g[first].to_numpy(), g[o].to_numpy(), g[real].to_numpy()), index=g.index)
        d_close = (g[pred] - g[c]).abs()
        d_open = (g[first] - g[o]).abs()
        move = np.sign(g[c] - g[o]) * np.sign(g[first] - g[o])  # +1 toward model, -1 away
        for lo, hi in BUCKETS + [(2, 99), (4, 99)]:
            label = f"{lo}+ pts" if hi == 99 else f"{lo}-{hi} pts"
            mc = (d_close >= lo) & (d_close < hi)
            mo = (d_open >= lo) & (d_open < hi)
            mv = move[mo].dropna()
            toward = f"{(mv == 1).sum()} of {(mv != 0).sum()} that moved" if len(mv) else "-"
            out.append(f"| {label} | {record(w_close[mc])} | {record(w_open[mo])} | {toward} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="only games on or after this date (YYYY-MM-DD)")
    ap.add_argument("--seasons", default="2027", help="comma-separated season labels")
    ap.add_argument("--write", action="store_true", help="save data/live/grades.md")
    args = ap.parse_args(argv)
    g = assemble([int(s) for s in args.seasons.split(",")], args.since)
    text = report(g)
    print(text)
    if args.write:
        (LIVE / "grades.md").write_text(
            "# Paper-trading grades (DraftKings)\n\n"
            "Model number = last logged before tip-off (opener test: first logged). Before Nov 5, "
            "2026 the live ratings ignored this season's games (bug), so Nov 1-5 games were priced "
            "on preseason ratings alone. Break-even at -110 is 52.4%.\n\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
