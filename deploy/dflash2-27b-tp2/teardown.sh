#!/usr/bin/env bash
# Take the DFlash2 pair down. `halt` keeps the containers (their exit codes and logs are the
# post-mortem of a failed start); the default removes them and frees the memory.
#
#   deploy/dflash2-27b-tp2/teardown.sh          # remove both ranks
#   deploy/dflash2-27b-tp2/teardown.sh halt     # stop both, keep them for their logs
#
# It only ever names qwen38-dflash2-tp2-r0 / -r1. vllm-lb's qwen38-cluster-vllm is never touched.
set -uo pipefail
cd "$(dirname "$0")"
[[ -r ./env.sh ]] || { echo "no env.sh here: copy env.example.sh to env.sh and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source ./env.sh
verb=${1:-down}
[[ $verb == down || $verb == halt ]] || die "usage: teardown.sh [down|halt]"
rc=0
# Rank 0 first: it is the endpoint, so clients stop being served before the group is broken.
for pair in "$RANK0_HOST:0" "$RANK1_HOST:1"; do
  box=${pair%%:*}; rank=${pair#*:}
  out=$("${SSH[@]}" "$box" env "RANK=$rank" bash -s -- "$verb" < rank.sh 2>&1) || rc=1
  printf '%-7s %s\n' "$box" "$out"
done
for pair in "$RANK0_HOST:$NAME0" "$RANK1_HOST:$NAME1"; do
  box=${pair%%:*}; name=${pair#*:}
  printf '%-7s free: ' "$box"
  "${SSH[@]}" "$box" "free -g | awk '/^Mem:/{printf \"%s GiB available of %s\\n\", \$7, \$2}'" 2>&1
done
exit $rc
