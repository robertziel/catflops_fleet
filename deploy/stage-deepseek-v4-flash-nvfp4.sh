#!/usr/bin/env bash
# nvidia/DeepSeek-V4-Flash-0731-NVFP4 (MIT, 75 files, 175.6 GB): DeepSeek-V4-Flash-0731 (304B MoE, 13B active) with
# ONLY its routed experts re-encoded by Model Optimizer 0.46 — the source's MXFP4 4-bit values kept, the block scale
# rewritten from one per 32 to one per 16, activations calibrated (README). Its DSpark drafter heads ride in the same
# checkpoint unquantized. Pinned f1caa711 (2026-09-02), read 2026-10-23. Operator 2026-10-23 chose this over the
# native 0731 checkpoint the published 2x DGX Spark recipes run.
# fetch | stage | check: deploy/stage-checkpoint.sh.
STAGE_REPO=nvidia/DeepSeek-V4-Flash-0731-NVFP4 STAGE_REV=f1caa71142bd0be02f728c79f75042ac1e461579 \
  STAGE_NAME=deepseek-v4-flash-cluster/DeepSeek-V4-Flash-0731-NVFP4 exec "$(dirname "$0")/stage-checkpoint.sh" "$@"
