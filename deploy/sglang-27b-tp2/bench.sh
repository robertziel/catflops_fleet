#!/usr/bin/env bash
# bench.sh — measure the TP=2 SGLang pair, with a thermal guard between every level.
#
#   deploy/sglang-27b-tp2/bench.sh <outdir>
#
# THE THERMAL RULE, and why it is in the harness and not in a human's head: on 2026-10-02 the dense
# 27B at 16 seats uncapped hard-killed box-a in 7 minutes (GPU power +35%, 118 W spikes,
# thermal-slowdown flags). So this script
#   * never runs more than 8 concurrent streams (bench.py refuses above that too), and
#   * reads nvidia-smi -q -d PERFORMANCE,TEMPERATURE on BOTH boxes before and after every level,
#     and ABORTS the whole sweep the moment any slowdown flag reads Active.
# It never applies a clock cap or a power limit: those are the operator's buttons, and a limit this
# script believes is needed is reported, not set.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -r "$HERE/sglang-tp2.env" ]] || { echo "no $HERE/sglang-tp2.env: copy sglang-tp2.example.env to sglang-tp2.env and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source "$HERE/sglang-tp2.env"
OUT=${1:?usage: bench.sh <outdir>}
mkdir -p "$OUT"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
BASE=http://$SGLANG_API_HOST:$SGLANG_PORT

say() { printf '%s  %s\n' "$(date -Is)" "$*" | tee -a "$OUT/bench.log"; }

# Every slowdown/shutdown flag nvidia-smi reports, plus the temperatures, for one box.
thermal() {
  "${SSH[@]}" "$1" 'nvidia-smi -q -d PERFORMANCE,TEMPERATURE 2>/dev/null \
     | grep -E "Slowdown|Shutdown|SW Power Cap|HW (Slowdown|Thermal)|GPU Current Temp|GPU T.Limit|Clocks (Event|Throttle) Reasons" \
     | sed "s/^ *//"' 2>/dev/null || echo "nvidia-smi unavailable"
}

# True when any slowdown flag is Active on either box.
thermal_tripped() {
  local host t
  for host in "$SGLANG_RANK0_HOST" "$SGLANG_RANK1_HOST"; do
    t=$(thermal "$host")
    printf '%s\n' "--- $host ---" "$t" >> "$OUT/thermal.txt"
    if grep -qE "(Slowdown|Shutdown|Thermal).*: *Active" <<<"$t"; then
      say "THERMAL FLAG ACTIVE on $host:"
      grep -E ".*: *Active" <<<"$t" | tee -a "$OUT/bench.log"
      return 0
    fi
  done
  return 1
}

# SGLang's own counters. The acceptance rate is the engine's, never the harness's arithmetic on
# token counts: --speculative-num-draft-tokens 8 means a step proposes 8 and accepts some, and only
# the server knows how many.
metrics() { curl -fsS -m 10 "$BASE/metrics" 2>/dev/null || echo "# metrics unavailable"; }
# Every non-comment metric whose NAME mentions speculation, acceptance or drafting — matched
# case-insensitively and without assuming the `sglang:` prefix, because a counter this sweep exists
# to read is not worth losing to a renamed namespace. The whole /metrics body is saved beside this
# either way, so nothing is lost if the filter still misses one.
spec_lines() { grep -v '^#' | grep -iE '^[A-Za-z_:][A-Za-z0-9_:]*(spec|accept|draft|verif)' | sort || true; }

say "base=$BASE model=$SGLANG_SERVED_NAME seats=$SGLANG_SEATS"
curl -fsS -m 10 "$BASE/get_server_info" > "$OUT/server_info.json" 2>/dev/null \
  && say "wrote server_info.json" || say "note: /get_server_info did not answer"
metrics > "$OUT/metrics-00-start.txt"
say "spec counters at start:"; spec_lines < "$OUT/metrics-00-start.txt" | tee -a "$OUT/bench.log"

: > "$OUT/thermal.txt"
if thermal_tripped; then
  say "ABORT: a thermal flag is already active before any load. Nothing was run."
  exit 1
fi
say "thermal before the sweep: clean on both boxes"

# ── the sweep ────────────────────────────────────────────────────────────────
# Single stream twice (thinking on, which is this model's default, and off), then 2/4/8 concurrent
# with thinking on. 8 is the ceiling; nothing here goes above it.
run_level() {
  local label=$1 tag=$2
  shift 2
  say "level $label"
  metrics > "$OUT/metrics-$tag-before.txt"
  python3 "$HERE/bench.py" --base "$BASE" --model "$SGLANG_SERVED_NAME" \
    --out "$OUT/bench-$tag.json" "$@" 2>&1 | tee -a "$OUT/bench.log"
  metrics > "$OUT/metrics-$tag-after.txt"
  say "spec counters after $label:"; spec_lines < "$OUT/metrics-$tag-after.txt" | tee -a "$OUT/bench.log"
  if thermal_tripped; then
    say "ABORT after $label: a thermal slowdown flag went Active. Stopping the sweep and reporting."
    exit 1
  fi
  say "thermal after $label: clean"
  sleep 20                                   # let both boxes settle before the next level
}

run_level "single stream, thinking ON"  "c1-think"   --concurrency 1 --max-tokens 400
run_level "single stream, thinking OFF" "c1-nothink" --concurrency 1 --max-tokens 400 --no-thinking
run_level "sweep 2/4/8, thinking ON"    "sweep"      --concurrency 2 4 8 --max-tokens 400

metrics > "$OUT/metrics-99-end.txt"
say "spec counters at end:"; spec_lines < "$OUT/metrics-99-end.txt" | tee -a "$OUT/bench.log"
say "done; everything is under $OUT"
