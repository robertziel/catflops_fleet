#!/usr/bin/env bash
# Shared settings for the DFlash2 27B TP=2 pair — THE EXAMPLE. Copy it to env.sh (gitignored) next to it and
# fill in the deployment's own boxes: launch-pair.sh, wait-health.sh and teardown.sh source env.sh and refuse to
# run without it. Every engine value below is overridable from the environment; the boxes, their fabric addresses
# and the interface have NO default — a pair launched at a guessed address lands NCCL on the wrong box.
#
# This pair is NOT the fleet's serving path. vllm-lb owns `qwen38-cluster-vllm` on :8010 and decides
# when it runs; these scripts own `qwen38-dflash2-tp2-r0`/`-r1` on a port of their own and nothing
# else. Both pairs want ~70 GiB a box out of 128 GiB, so they are MUTUALLY EXCLUSIVE: launch-pair.sh
# refuses to start while the cluster container exists, and it never stops it.

# The two boxes, as ssh names them (e.g. box-a, box-b), and their addresses on the pair's own fabric link (e.g.
# 192.0.2.201 and 192.0.2.202 on a /30), and that link's interface on each box (e.g. bond0 — the same for both).
: "${RANK0_HOST:?RANK0_HOST is not set — rank 0, the head that serves the API (copy env.example.sh to env.sh)}"
: "${RANK1_HOST:?RANK1_HOST is not set — rank 1, the headless worker}"
: "${RANK0_IP:?RANK0_IP is not set — rank 0 address on the pair fabric link}"
: "${RANK1_IP:?RANK1_IP is not set — rank 1 address on the pair fabric link}"
: "${IFACE:?IFACE is not set — the fabric link interface on each box, e.g. bond0}"
# The address a CLIENT uses, which is NOT RANK0_IP: the fabric link is the pair's own private /30, which the LB's
# box is not on, so a health check or a benchmark aimed at RANK0_IP times out. API_HOST is the head's address on
# the link the LB's box reaches it over (the one vllm-lb dials too).
: "${API_HOST:?API_HOST is not set — the head address as the LB box reaches it}"
: "${MASTER_ADDR:=$RANK0_IP}"
: "${MASTER_PORT:=25010}"               # NOT 25000: that is the live cluster's rendezvous port
: "${PORT:=8021}"                       # 8020-8029 is this task's range; 8000 and 8010 are taken

: "${IMAGE:=myllmbox/qwen38-flash-next-cluster-vllm:v5}"
: "${MODELS_SUBDIR:=dflash2-27b-tp2}"   # a models subdir only this task writes
: "${MODEL_NAME:=Qwen3.8-27B-NVFP4-bf16head}"
: "${DRAFT_NAME:=Qwen3.8-27B-DFlash2}"
: "${SERVED_MODEL_NAME:=Qwen/Qwen3.8-27B-DFlash2}"

# K = num_speculative_tokens. The draft's dflash_config.block_size is 8 and its own model card
# serves K=7 (one block minus the bonus query); upstream's launcher guards K % 8 == 0 and calls
# K=16 (two blocks) the GB10 peak. vLLM 0.29.0 itself imposes no block-size rule on dflash
# (config/speculative.py checks it only for dspark), so both are launchable and both are measured.
# K=0 means NO speculative config at all: the no-spec baseline, and the greedy A/B reference.
: "${K:=16}"

: "${MAX_MODEL_LEN:=262144}"
: "${MAX_NUM_SEQS:=8}"                  # the thermal cap: 16 seats uncapped killed box-a 2026-10-02
: "${MAX_NUM_BATCHED_TOKENS:=8192}"     # the pair's measured best; 16384 measured worse
: "${GPU_MEMORY_UTILIZATION:=0.70}"
: "${KV_CACHE_MEMORY:=28000000000}"
: "${REASONING_PARSER:=qwen3}"
: "${TOOL_CALL_PARSER:=qwen3_xml}"
: "${HEALTH_TIMEOUT_S:=900}"            # upstream quotes 4-5 min to /health; TP=2 cold starts drift

readonly NAME0=qwen38-dflash2-tp2-r0
readonly NAME1=qwen38-dflash2-tp2-r1
readonly LIVE_CLUSTER=qwen38-cluster-vllm   # vllm-lb's. Never run, stop, kill or remove it.
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

die() { printf '%s\n' "$*" >&2; exit 1; }

# Refuse the port range the fleet already uses, and anything outside this task's own.
check_port() {
  [[ $PORT =~ ^[0-9]+$ ]] || die "PORT is not a number: $PORT"
  (( PORT >= 8020 && PORT <= 8029 )) \
    || die "PORT must be 8020-8029 (this task's range), not $PORT -- 8000 and 8010 are the fleet's"
}

# The live TP=2 cluster and this pair cannot both fit in 128 GiB a box. If the cluster is there at
# all -- running, exited, halted -- this is not our window: say so and change nothing.
require_cluster_absent() {
  local box found
  for box in "$RANK0_HOST" "$RANK1_HOST"; do
    found=$("${SSH[@]}" "$box" "docker ps -a --filter name=^/$LIVE_CLUSTER\$ --format '{{.Names}} {{.Status}}'" 2>&1) \
      || die "cannot reach $box: $found"
    if [[ -n $found ]]; then
      die "$box still holds $LIVE_CLUSTER ($found).
This pair needs the same ~70 GiB a box, and $LIVE_CLUSTER belongs to vllm-lb on :8010.
STOPPING. Nothing was started or removed. Free the boxes through the LB panel, not from here."
    fi
  done
}
