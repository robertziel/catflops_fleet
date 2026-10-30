#!/usr/bin/env bash
# Bring up the DFlash2 27B TP=2 pair: rank 1 (headless) first, then rank 0 (the API), then wait for
# /health. Every parameter comes from env.sh and can be overridden:
#
#   deploy/dflash2-27b-tp2/launch-pair.sh                 # K=16, port 8021
#   K=7 deploy/dflash2-27b-tp2/launch-pair.sh             # one draft block
#   K=0 deploy/dflash2-27b-tp2/launch-pair.sh             # no speculation: the baseline
#   PORT=8022 MAX_NUM_SEQS=4 deploy/dflash2-27b-tp2/launch-pair.sh
#
# WHY rank 1 FIRST: rank 0 is the rendezvous master and starts serving once the group forms. Bring
# rank 0 up first and it waits on a worker that is still loading; upstream reports the handshake
# stalls outright. Rank 1 is started, and its container confirmed running, before rank 0 is sent.
#
# It REFUSES to start while vllm-lb's qwen38-cluster-vllm exists on either box: the two pairs want
# the same ~70 GiB a box. It never stops, kills or removes that container.
set -uo pipefail
cd "$(dirname "$0")"
[[ -r ./env.sh ]] || { echo "no env.sh here: copy env.example.sh to env.sh and fill in this deployment's pair" >&2; exit 2; }
# shellcheck source=/dev/null
source ./env.sh
# shellcheck source=../models-env.sh
source ../models-env.sh            # MODELS_SEARCH_DIRS, from lb/.env

check_port
[[ $K =~ ^[0-9]+$ ]] && (( K <= 32 )) || die "K must be 0..32, not '$K'"
[[ $MASTER_PORT =~ ^[0-9]+$ ]] || die "MASTER_PORT is not a number"
(( MASTER_PORT != 25000 )) || die "MASTER_PORT 25000 is the live cluster's rendezvous port"

echo "== guards =="
require_cluster_absent
echo "ok: $LIVE_CLUSTER absent on $RANK0_HOST and $RANK1_HOST"
for pair in "$RANK0_HOST:$RANK0_IP" "$RANK1_HOST:$RANK1_IP"; do
  box=${pair%%:*}
  busy=$("${SSH[@]}" "$box" "ss -ltnH | grep -w ':$PORT' || true" 2>&1)
  [[ -z $busy ]] || die "$box already has a listener on $PORT: $busy"
  free=$("${SSH[@]}" "$box" "free -g | awk '/^Mem:/{print \$7}'" 2>&1)
  echo "ok: $box port $PORT free, ${free} GiB available"
done

env_common=(
  "IMAGE=$IMAGE" "MODELS_SUBDIR=$MODELS_SUBDIR" "MODELS_SEARCH_DIRS=$MODELS_SEARCH_DIRS" "MODEL_NAME=$MODEL_NAME" "DRAFT_NAME=$DRAFT_NAME"
  "SERVED_MODEL_NAME=$SERVED_MODEL_NAME" "IFACE=$IFACE" "PORT=$PORT"
  "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" "K=$K"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" "KV_CACHE_MEMORY=$KV_CACHE_MEMORY"
  "REASONING_PARSER=$REASONING_PARSER" "TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
)

echo "== rank 1 ($RANK1_HOST, headless) =="
"${SSH[@]}" "$RANK1_HOST" env "${env_common[@]}" RANK=1 "HOST_IP=$RANK1_IP" bash -s -- up < rank.sh \
  || die "rank 1 did not start"
for i in $(seq 1 30); do
  st=$("${SSH[@]}" "$RANK1_HOST" "docker inspect -f '{{.State.Status}}' $NAME1" 2>/dev/null) || st=""
  [[ $st == running ]] && break
  [[ $st == exited ]] && { "${SSH[@]}" "$RANK1_HOST" "docker logs --tail 40 $NAME1" >&2; die "rank 1 exited immediately"; }
  sleep 1
done
[[ $st == running ]] || die "rank 1 is '$st', not running, after 30s"
echo "ok: $NAME1 running -- the group has a worker to rendezvous with"

echo "== rank 0 ($RANK0_HOST, serves :$PORT) =="
"${SSH[@]}" "$RANK0_HOST" env "${env_common[@]}" RANK=0 "HOST_IP=$RANK0_IP" bash -s -- up < rank.sh \
  || die "rank 0 did not start"

echo "== waiting for http://$API_HOST:$PORT/health =="
exec ./wait-health.sh
