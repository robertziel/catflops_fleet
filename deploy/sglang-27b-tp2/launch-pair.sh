#!/usr/bin/env bash
# launch-pair.sh — bring up the two-box TP=2 SGLang cluster with ONE command, from box-c.
#
#   deploy/sglang-27b-tp2/launch-pair.sh            # preflight both boxes, then rank 1, then rank 0
#   SGLANG_SEATS=4 deploy/sglang-27b-tp2/launch-pair.sh
#   deploy/sglang-27b-tp2/launch-pair.sh --preflight-only
#
# This is the WHEN; deploy/sglang-27b-tp2/sglang-rank is the HOW, and it is sent to each box on
# stdin over ssh so the copy that runs is always the one in this checkout.
#
# WHY RANK 1 FIRST: rank 0 owns the rendezvous socket (--dist-init-addr) and SGLang's rank 0 blocks
# in the store handshake until every other node has checked in. Starting the headless node first
# means rank 0's wait is short; the other order works too but spends the whole handshake timeout
# budget on a node that is still pulling weights off NFS.
#
# MUTUALLY EXCLUSIVE WITH THE LIVE vLLM CLUSTER, by arithmetic, not by policy: each GB10 box has 128 GB
# of unified memory and one 27B TP=2 pair needs ~70 GiB of it per box. Preflight therefore REFUSES
# to launch while qwen38-cluster-vllm (vllm-lb's, on :8100) or any other engine is up on either box.
# It never stops anything to make room — that container is the LB's and not ours to touch.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../models-env.sh
. "$HERE/../models-env.sh"              # MODELS_SEARCH_DIRS, from lb/.env
[[ -r "$HERE/sglang-tp2.env" ]] || { echo "no $HERE/sglang-tp2.env: copy sglang-tp2.example.env to sglang-tp2.env and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source "$HERE/sglang-tp2.env"
RANK_SCRIPT=$HERE/sglang-rank
[[ -r $RANK_SCRIPT ]] || { echo "cannot read $RANK_SCRIPT" >&2; exit 1; }

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
PREFLIGHT_ONLY=false
[[ ${1-} == --preflight-only ]] && PREFLIGHT_ONLY=true

# The env every box gets. One place, so rank 0 and rank 1 cannot be launched with different engine
# settings — a TP group whose members disagree does not fail cleanly, it hangs.
rank_env() {
  # Every value is shell-quoted for the REMOTE shell: SGLANG_EXTRA_ARGS holds whole flags with
  # spaces in them, and an unquoted one would arrive as separate words and be lost.
  local pairs=(
    "RANK_HOST_IP=$1"
    "SGLANG_IMAGE=$SGLANG_IMAGE"
    "SGLANG_MODELS_DIRS=${SGLANG_MODELS_DIRS:-$MODELS_SEARCH_DIRS}"
    "SGLANG_TARGET=$SGLANG_TARGET"
    "SGLANG_DRAFT=$SGLANG_DRAFT"
    "SGLANG_SERVED_NAME=$SGLANG_SERVED_NAME"
    "SGLANG_PORT=$SGLANG_PORT"
    "SGLANG_DIST_PORT=$SGLANG_DIST_PORT"
    "SGLANG_DIST_ADDR=$SGLANG_DIST_ADDR"
    "SGLANG_IFACE=$SGLANG_IFACE"
    "SGLANG_NCCL_IB_HCA=$SGLANG_NCCL_IB_HCA"
    "SGLANG_CPUSET=$SGLANG_CPUSET"
    "SGLANG_MEM_FRACTION=$SGLANG_MEM_FRACTION"
    "SGLANG_DRAFT_TOKENS=$SGLANG_DRAFT_TOKENS"
    "SGLANG_MAMBA_SLOTS_PER_REQ=$SGLANG_MAMBA_SLOTS_PER_REQ"
    "SGLANG_EXTRA_ARGS=$SGLANG_EXTRA_ARGS"
  )
  local out=() q
  for q in "${pairs[@]}"; do out+=("$(printf %q "$q")"); done
  printf '%s' "${out[*]}"
}

# Run a sglang-rank verb on a box. The script arrives on stdin; the verb and its numbers are argv.
on_box() {
  local host=$1 ip=$2
  shift 2
  # shellcheck disable=SC2029  # the remote shell expands $HOME on purpose: only the box knows it
  "${SSH[@]}" "$host" "env $(rank_env "$ip") bash -s -- $*" < "$RANK_SCRIPT"
}

say() { printf '%s  %s\n' "$(date -Is)" "$*"; }

# ── 1. preflight, BOTH boxes, before anything is started anywhere ────────────
say "preflight on $SGLANG_RANK0_HOST and $SGLANG_RANK1_HOST"
ok=true
for pair in "$SGLANG_RANK0_HOST:$SGLANG_RANK0_IP" "$SGLANG_RANK1_HOST:$SGLANG_RANK1_IP"; do
  host=${pair%%:*} ip=${pair##*:}
  out=$(on_box "$host" "$ip" preflight) || ok=false
  printf '  %-7s %s\n' "$host" "$out"
done
if [[ $ok != true ]]; then
  cat >&2 <<'MSG'

REFUSED: a box is not ready. The usual reason is that another engine holds its memory —
qwen38-cluster-vllm is vllm-lb's TP=2 pair (:8010 / :8100) and must be stopped by the LB,
never by hand. Check the LB panel, let it take the pair down, then run this again.
MSG
  exit 1
fi
$PREFLIGHT_ONLY && { say "preflight only: nothing started"; exit 0; }

# ── 2. the idle memory figure, before either rank allocates ──────────────────
say "idle memory, per box"
for host in "$SGLANG_RANK0_HOST" "$SGLANG_RANK1_HOST"; do
  printf '  %-7s %s\n' "$host" "$("${SSH[@]}" "$host" 'free -g | awk "/^Mem:/ {print \"total=\"\$2\" used=\"\$3\" free=\"\$4\" available=\"\$7}"')"
done

# ── 3. rank 1 (headless), then rank 0 (the API) ──────────────────────────────
T0=$(date +%s)
say "starting rank 1 on $SGLANG_RANK1_HOST (headless)"
on_box "$SGLANG_RANK1_HOST" "$SGLANG_RANK1_IP" up 1 "$SGLANG_SEATS" "$SGLANG_CONTEXT"
say "starting rank 0 on $SGLANG_RANK0_HOST (serves :$SGLANG_PORT)"
on_box "$SGLANG_RANK0_HOST" "$SGLANG_RANK0_IP" up 0 "$SGLANG_SEATS" "$SGLANG_CONTEXT"

# ── 4. the health wait ───────────────────────────────────────────────────────
# Ready is /health answering on rank 0. Three ways to stop waiting, so a bad start never hangs
# forever and always leaves the evidence behind:
#   * either rank's container EXITED            -> report its tail, halt the pair, exit 1
#   * rank 0's log has been quiet for QUIET_ABORT_S -> the same (the FlashInfer-autotune class of
#     hang shows exactly this: a live process, a silent log, and a peer that never joins)
#   * the whole READY_TIMEOUT_S budget is spent -> the same
READY=http://$SGLANG_API_HOST:$SGLANG_PORT/health
say "waiting for $READY (budget ${SGLANG_READY_TIMEOUT_S}s, quiet abort ${SGLANG_QUIET_ABORT_S}s)"
last_len=0 last_change=$T0
while :; do
  now=$(date +%s)
  if curl -fsS -m 5 "$READY" >/dev/null 2>&1; then
    say "READY after $((now - T0))s — http://$SGLANG_API_HOST:$SGLANG_PORT/v1 (model $SGLANG_SERVED_NAME)"
    break
  fi
  # A container that is gone is a failure, whatever the clock says.
  for pair in "$SGLANG_RANK0_HOST:0" "$SGLANG_RANK1_HOST:1"; do
    host=${pair%%:*} r=${pair##*:}
    if ! "${SSH[@]}" "$host" "docker ps --format '{{.Names}}' | grep -qx qwen38-27b-sglang-tp2-r$r"; then
      say "FAILED: rank $r on $host is no longer running"
      "${SSH[@]}" "$host" "docker logs --tail 60 qwen38-27b-sglang-tp2-r$r" 2>&1 | tail -60 || true
      say "halting the pair; the containers are KEPT for the post-mortem (teardown.sh removes them)"
      "$HERE/teardown.sh" --halt || true
      exit 1
    fi
  done
  # Progress is a GROWING LOG, not a live process.
  len=$("${SSH[@]}" "$SGLANG_RANK0_HOST" "docker logs qwen38-27b-sglang-tp2-r0 2>&1 | wc -c" || echo "$last_len")
  if [[ $len != "$last_len" ]]; then
    last_len=$len last_change=$now
  elif (( now - last_change >= SGLANG_QUIET_ABORT_S )); then
    say "FAILED: rank 0's log has not grown for $((now - last_change))s ($len bytes) — giving up"
    "${SSH[@]}" "$SGLANG_RANK0_HOST" "docker logs --tail 60 qwen38-27b-sglang-tp2-r0" 2>&1 | tail -60 || true
    "$HERE/teardown.sh" --halt || true
    exit 1
  fi
  if (( now - T0 >= SGLANG_READY_TIMEOUT_S )); then
    say "FAILED: not ready within ${SGLANG_READY_TIMEOUT_S}s"
    "$HERE/teardown.sh" --halt || true
    exit 1
  fi
  sleep 5
done

# ── 5. what it cost, measured ───────────────────────────────────────────────
say "loaded memory, per box"
for host in "$SGLANG_RANK0_HOST" "$SGLANG_RANK1_HOST"; do
  printf '  %-7s %s\n' "$host" "$("${SSH[@]}" "$host" 'free -g | awk "/^Mem:/ {print \"total=\"\$2\" used=\"\$3\" free=\"\$4\" available=\"\$7}"')"
done
say "served model: $(curl -fsS -m 5 "http://$SGLANG_API_HOST:$SGLANG_PORT/v1/models" | head -c 400)"
