#!/usr/bin/env bash
# Seed broker_parrot's vLLM sidecar on a GB10 box — CREATE the container, never start it.
#
# broker_parrot owns vLLM on the GB10 boxes ("fleet decides", 2026-09-11). A GPU claim lane
# `docker start`s the container named `vllm` when enough queued jobs want vLLM (the LB's
# lease is one such job), `docker stop`s it after its idle TTL, and a parked box kills it on
# sight. The engine can start/stop/recreate a container that EXISTS but never creates one —
# this is the seed.
#
# Never start/stop/rm `vllm` by hand. The old vllmstart did (`docker rm vllm`, then a Ray
# container under the same name) and the fleet had no startable vLLM for a week.
#
# ── The spec below was MEASURED on box-b, 2026-09-11 (worklog/vllm-lb-context.md) ──
#   * qwen3.8 NVFP4 (dense 27B) serves on vLLM v0.23.0 / GB10: architecture resolved,
#     weights 18.74 GiB, cold start 466 s (153 s of it reading 18.36 GiB over NFS at 127 MB/s).
#   * --max-model-len 262144 = the model's trained window (max_position_embeddings). The KV
#     pool at 0.75 holds 1,079,598 tokens = 4.12 concurrent FULL-window requests, so memory
#     is not the limit; the model is. A 251,509-token prompt was read in full and the needle
#     in its first line came back (no silent truncation); prefill ran at 851 tok/s.
#   * ONE served name: broker_parrot's residency enforcer counts every id on /v1/models as a
#     resident copy (cap 1) and would `docker stop` a two-name sidecar ~30 s after warm-up.
#   * --tool-call-parser qwen3_coder matches this model's chat template (<function=…>), and
#     --reasoning-parser qwen3 keeps thinking out of `content` — both verified live through
#     /v1/messages (stop_reason tool_use; content '51' with reasoning in its own field).
#   * --max-num-seqs 8 bounds the hybrid model's per-sequence state (48 of 64 layers are
#     linear-attention); the LB serves a handful of clients, not a batch farm.
#
# Usage: seed-broker-sidecar.sh <ssh-host> [--replace]
#   <ssh-host>  the GPU box, as ~/.ssh/config names it (any box of the fleet; no default)
#   --replace   remove an existing, NOT running `vllm` container first
set -euo pipefail

IMAGE=vllm/vllm-openai:v0.23.0          # == :latest on both boxes (003febc956c8)
MODEL=/models/base.qwen3.8-dense-nvfp4-vllm
SERVED_NAME=qwen3.8:latest              # what the fleet dials; the LB rewrites client names
MAX_MODEL_LEN=262144

# Required, no operator defaults. MODEL_DIR is the checkpoint directory mounted read-only at /models
# (an NFS export the GPU boxes can read as root). The box is the first argument, an ssh alias.
ssh_host="${1:-}"
case "$ssh_host" in
    ""|-*) echo "usage: $0 <ssh-host> [--replace]   (the box's ssh alias, as ~/.ssh/config names it; no default)" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the checkpoint directory mounted at /models (read-only)}"
replace=0
[[ "${2:-}" == "--replace" ]] && replace=1

existing=$(ssh -n "$ssh_host" "docker inspect -f '{{.State.Status}} {{.Config.Image}}' vllm 2>/dev/null" || true)
if [[ -n "$existing" ]]; then
    if (( ! replace )); then
        echo "$ssh_host already has a vllm container ($existing) — pass --replace to reseed" >&2
        exit 1
    fi
    if [[ "$existing" == running* ]]; then
        echo "$ssh_host: vllm is RUNNING (the broker is serving from it) — not replacing" >&2
        exit 1
    fi
    ssh -n "$ssh_host" docker rm vllm
fi

ssh -n "$ssh_host" docker create \
    --name vllm \
    --label owner=broker_parrot \
    --restart no \
    --runtime nvidia \
    --ipc host \
    -p 8000:8000 \
    -v "$MODEL_DIR:/models:ro" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HUB_OFFLINE=1 \
    --log-opt max-size=10m --log-opt max-file=3 \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 8192 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

ssh -n "$ssh_host" "docker inspect -f '{{.Name}} {{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}} {{.Config.Image}}' vllm"
