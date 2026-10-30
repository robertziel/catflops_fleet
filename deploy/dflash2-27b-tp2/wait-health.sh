#!/usr/bin/env bash
# Wait for rank 0's /health to answer 200, and say how long it took. Exit 0 on 200, 1 on timeout --
# and on timeout print both ranks' last lines, which are the whole post-mortem of a failed start.
#
#   deploy/dflash2-27b-tp2/wait-health.sh [timeout_s]
set -uo pipefail
cd "$(dirname "$0")"
[[ -r ./env.sh ]] || { echo "no env.sh here: copy env.example.sh to env.sh and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source ./env.sh
TIMEOUT=${1:-$HEALTH_TIMEOUT_S}
start=$(date +%s)
code=000
while :; do
  now=$(date +%s); elapsed=$(( now - start ))
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$API_HOST:$PORT/health" || true)
  if [[ $code == 200 ]]; then
    echo "health=200 after ${elapsed}s"
    exit 0
  fi
  # A rank that has exited will never answer: stop waiting out the clock for it.
  for pair in "$RANK0_HOST:$NAME0" "$RANK1_HOST:$NAME1"; do
    box=${pair%%:*}; name=${pair#*:}
    st=$("${SSH[@]}" "$box" "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $name" 2>/dev/null) || st=""
    if [[ $st == exited* ]]; then
      echo "health=$code after ${elapsed}s -- $box/$name has EXITED ($st)" >&2
      "${SSH[@]}" "$box" "docker logs --tail 40 $name" 2>&1 | sed "s/^/[$name] /" >&2
      exit 1
    fi
  done
  if (( elapsed >= TIMEOUT )); then
    echo "health=$code after ${elapsed}s -- TIMEOUT (${TIMEOUT}s)" >&2
    for pair in "$RANK0_HOST:$NAME0" "$RANK1_HOST:$NAME1"; do
      box=${pair%%:*}; name=${pair#*:}
      "${SSH[@]}" "$box" "docker logs --tail 25 $name" 2>&1 | sed "s/^/[$name] /" >&2
    done
    exit 1
  fi
  printf 'health=%s %ss\n' "$code" "$elapsed"
  sleep 10
done
