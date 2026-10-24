#!/usr/bin/env bash
# Take an odds snapshot and refresh predictions every INTERVAL_MIN minutes for
# LOOP_MINUTES, committing each one. Run by .github/workflows/snapshot-loop.yml,
# which starts a fresh copy of this loop when it ends.
#
# Env (all optional):
#   LOOP_MINUTES   how long to keep looping (default 330)
#   INTERVAL_MIN   minutes between snapshots (default 30)
#   SNAP_CMD       snapshot command (default: python -m cbb.fetch.espn_live)
#   PRED_CMD       prediction command (default: python -m cbb.predict.daily)
#   BRANCH         branch to push to (default: main)
#   LOOP_SECONDS / INTERVAL_SEC override the minute settings (used in tests)
set -u

LOOP_MINUTES=${LOOP_MINUTES:-330}
INTERVAL_MIN=${INTERVAL_MIN:-30}
SNAP_CMD=${SNAP_CMD:-python -m cbb.fetch.espn_live}
PRED_CMD=${PRED_CMD:-python -m cbb.predict.daily}
BRANCH=${BRANCH:-main}

LOOP_SECONDS=${LOOP_SECONDS:-$(( LOOP_MINUTES * 60 ))}
INTERVAL_SEC=${INTERVAL_SEC:-$(( INTERVAL_MIN * 60 ))}
end=$(( $(date +%s) + LOOP_SECONDS ))
ok=0
fail=0

sync() {
  # Bring in anyone else's commits; on any conflict, take the remote state.
  if ! git pull -q --rebase origin "$BRANCH"; then
    git rebase --abort 2>/dev/null
    git fetch -q origin "$BRANCH" && git reset -q --hard "origin/$BRANCH"
  fi
}

while [ "$(date +%s)" -lt "$end" ]; do
  started=$(date +%s)
  sync
  if bash -c "$SNAP_CMD"; then
    bash -c "$PRED_CMD" || echo "prediction step failed (snapshot kept)"
    git add data/live
    if ! git diff --cached --quiet; then
      git commit -q -m "Odds snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
      pushed=0
      for i in 1 2 3 4 5; do
        if git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; then
          pushed=1; break
        fi
        git rebase --abort 2>/dev/null
        sleep $(( RANDOM % 15 + 5 ))
      done
      if [ "$pushed" != 1 ]; then
        echo "push failed after retries; dropping this snapshot"
        git fetch -q origin "$BRANCH" && git reset -q --hard "origin/$BRANCH"
      fi
    fi
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "snapshot failed ($fail so far)"
  fi
  echo "$(date -u +%H:%MZ) snapshots ok: $ok, failed: $fail"

  next=$(( started + INTERVAL_SEC ))
  [ "$next" -ge "$end" ] && break
  now=$(date +%s)
  [ "$next" -gt "$now" ] && sleep $(( next - now ))
done

echo "ok=$ok" >> "${GITHUB_OUTPUT:-/dev/null}"
echo "loop finished: $ok snapshots ok, $fail failed"
