#!/usr/bin/env bash
# teardown.sh — take the two-box TP=2 SGLang cluster down.
#
#   deploy/sglang-27b-tp2/teardown.sh          # stop AND remove both ranks
#   deploy/sglang-27b-tp2/teardown.sh --halt   # stop both ranks, KEEP them for the post-mortem
#
# Idempotent: a rank that is not there is a success. Only ever touches qwen38-27b-sglang-tp2-r0 and
# -r1 — never qwen38-cluster-vllm (vllm-lb's), never the broker's box agent or its workers
# (broker_parrot's).
#
# Rank 0 goes first: it owns the rendezvous, so dropping it tells rank 1's collectives to fail fast
# instead of sitting in an all-reduce with a peer that is already gone.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -r "$HERE/sglang-tp2.env" ]] || { echo "no $HERE/sglang-tp2.env: copy sglang-tp2.example.env to sglang-tp2.env and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source "$HERE/sglang-tp2.env"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
VERB=down
[[ ${1-} == --halt ]] && VERB=halt

for pair in "$SGLANG_RANK0_HOST:$SGLANG_RANK0_IP" "$SGLANG_RANK1_HOST:$SGLANG_RANK1_IP"; do
  host=${pair%%:*} ip=${pair##*:}
  printf '%-7s ' "$host"
  # shellcheck disable=SC2029  # RANK_HOST_IP is this box's own, deliberately expanded here
  "${SSH[@]}" "$host" "env RANK_HOST_IP=$ip bash -s -- $VERB" < "$HERE/sglang-rank" || true
done
