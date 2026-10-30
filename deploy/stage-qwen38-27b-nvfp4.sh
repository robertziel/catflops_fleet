#!/usr/bin/env bash
# nvidia/Qwen3.8-27B-NVFP4 (Apache-2.0, not gated, 20.4 GiB in three shards): ModelOpt mixed precision — NVFP4 on the
# MLPs and lm_head, FP8 on the attention projections, the model's own MTP head in bf16 (`mtp.*`, shard 3), so the
# drafter comes WITH the checkpoint. fetch | stage | check: deploy/stage-checkpoint.sh.
STAGE_REPO=nvidia/Qwen3.8-27B-NVFP4 STAGE_REV=482ca0f3832238542f8f5295dde86b5f22711d80 STAGE_NAME=Qwen3.8-27B-NVFP4 \
  exec "$(dirname "$0")/stage-checkpoint.sh" "$@"
