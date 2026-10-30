#!/usr/bin/env bash
# ONE rank of the DFlash2 27B TP=2 pair, run ON a GB10 box. Sent over ssh stdin, as tp-rank is, so the
# copy that runs is always the caller's and nothing on a box can drift:
#
#   ssh box-b 'env RANK=1 HOST_IP=... bash -s -- up' < rank.sh
#
# Verbs: up | halt | down | status | logs [n]
#
# Everything is defined first and `main "$@"` is the last line: the script arrives over a network,
# and a connection that drops mid-transfer hands bash a prefix, which must do nothing.
set -uo pipefail

RANK=${RANK:?RANK must be 0 or 1}
NAME=qwen38-dflash2-tp2-r$RANK
LIVE_CLUSTER=qwen38-cluster-vllm

die() { printf '%s\n' "$*" >&2; exit 1; }

cmd_status() {
  docker ps -a --filter "name=^/$NAME\$" --format '{{.Names}}|{{.Status}}|{{.Image}}' || true
}
cmd_logs() { docker logs --tail "${1:-80}" "$NAME" 2>&1 || true; }
cmd_halt() { docker stop -t 10 "$NAME" >/dev/null 2>&1; echo "halted $NAME (kept for its exit code and log)"; }
cmd_down() { docker rm -f "$NAME" >/dev/null 2>&1; echo "removed $NAME"; }

cmd_up() {
  : "${HOST_IP:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}" "${PORT:?}" "${IMAGE:?}"
  : "${MODELS_SUBDIR:?}" "${MODEL_NAME:?}" "${DRAFT_NAME:?}" "${SERVED_MODEL_NAME:?}" "${IFACE:?}"
  : "${K:?}" "${MAX_MODEL_LEN:?}" "${MAX_NUM_SEQS:?}" "${MAX_NUM_BATCHED_TOKENS:?}"
  : "${GPU_MEMORY_UTILIZATION:?}" "${KV_CACHE_MEMORY:?}"

  # The fleet's own pair must not be here. A rank of ours beside it would take memory the LB
  # already promised, so refuse rather than squeeze.
  if [[ -n $(docker ps -a --filter "name=^/$LIVE_CLUSTER\$" --format '{{.Names}}') ]]; then
    die "$(hostname): $LIVE_CLUSTER is present; refusing to start $NAME beside it"
  fi

  # Where this box reads the checkpoints: its own NVMe first, then box-a's export over pair-link.
  local root=
  local dir
  local roots tried=()
  IFS=: read -ra roots <<<"${MODELS_SEARCH_DIRS:?MODELS_SEARCH_DIRS is not set — launch-pair.sh sends it from lb/.env}"
  for dir in "${roots[@]}"; do
    dir="${dir/#\~/$HOME}/$MODELS_SUBDIR"; tried+=("$dir")
    if [[ -f $dir/$MODEL_NAME/config.json && -f $dir/$DRAFT_NAME/config.json ]]; then
      root=$dir; break
    fi
  done
  [[ -n $root ]] || die "$(hostname): no root holds both $MODEL_NAME and $DRAFT_NAME (looked in ${tried[*]})"

  # A FlashInfer autotune cache that PERSISTS hangs every TP=2 start after the first: rank 0 hits
  # the cache, rank 1 tunes alone, and the group waits out a 30-minute gloo timeout. It lives on a
  # tmpfs, so each start tunes on both ranks or neither. (The compile cache below is fine to keep.)
  local autotune=/flashinfer-autotune
  local cache=$HOME/.cache/vllm-dflash2-tp2
  mkdir -p "$cache" || die "cannot create $cache"

  docker rm -f "$NAME" >/dev/null 2>&1 || true

  local spec=()
  if (( K > 0 )); then
    spec=(--speculative-config \
      "{\"method\":\"dflash\",\"model\":\"/models/$DRAFT_NAME\",\"num_speculative_tokens\":$K}")
  fi
  local parsers=()
  [[ -n ${REASONING_PARSER:-} ]] && parsers+=(--reasoning-parser "$REASONING_PARSER")
  [[ -n ${TOOL_CALL_PARSER:-} ]] && parsers+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
  local headless=()
  (( RANK == 1 )) && headless=(--headless)   # rank 1 serves no API; rank 0 is the endpoint

  # --restart no: half a TP pair brought back by docker cannot rejoin the group, it only holds the
  # memory. --cap-add SYS_PTRACE: NCCL shares CUDA memory across processes, and this is the only
  # way this image has ever reached healthy on these GB10 boxes.
  # VLLM_USE_V2_MODEL_RUNNER=1 is load-bearing, not tidiness: the DFlash2 candidate selector exists
  # only in the V2 speculator (v1/worker/gpu/spec_decode/dflash2/speculator.py, chosen on the draft's
  # DFlash2DraftModel architecture). On the V1 runner the same checkpoint drafts through
  # DFlashProposer, which never calls the selector, and the draft degrades to DFlash1 SILENTLY.
  # Asking for V2 outright turns that into a startup error instead of a quiet loss of the feature.
  docker run -d --restart no --name "$NAME" \
    --label "dflash2.rank=$RANK" --label "dflash2.k=$K" \
    --gpus all --ipc=host --network host --cap-add SYS_PTRACE --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
    -v "$root:/models:ro" -v "$cache:/cache" \
    --tmpfs "$autotune:rw,nosuid,nodev,noexec,size=16m" \
    -e "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$autotune" \
    -e VLLM_CACHE_ROOT=/cache/vllm -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180 \
    -e "NCCL_SOCKET_IFNAME=$IFACE" -e "GLOO_SOCKET_IFNAME=$IFACE" \
    -e "VLLM_HOST_IP=$HOST_IP" -e NCCL_IB_DISABLE=0 \
    "$IMAGE" \
    --model "/models/$MODEL_NAME" --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --nnodes 2 --node-rank "$RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-cache-memory "$KV_CACHE_MEMORY" \
    --enable-prefix-caching --enable-mfu-metrics \
    "${parsers[@]}" ${spec[@]+"${spec[@]}"} ${headless[@]+"${headless[@]}"} \
    || die "$(hostname): docker run failed for $NAME"
  echo "$(hostname): started $NAME (rank $RANK, K=$K, models root $root)"
}

main() {
  case ${1:-} in
    up) cmd_up ;;
    halt) cmd_halt ;;
    down) cmd_down ;;
    status) cmd_status ;;
    logs) shift; cmd_logs "$@" ;;
    *) die "usage: RANK=<0|1> ... bash -s -- up|halt|down|status|logs [n]" ;;
  esac
}

main "$@"
