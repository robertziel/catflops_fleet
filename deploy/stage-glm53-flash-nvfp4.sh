#!/usr/bin/env bash
# RedHatAI/GLM-5.3-Flash-NVFP4 (MIT base model, 21 files, 197.9 GB / 184.3 GiB): zai-org's GLM-5.3-Flash (321B MoE,
# 18B active) in compressed-tensors NVFP4 by llm-compressor, its MTP layer (45) in FP8-block — the drafter comes WITH
# the checkpoint. Pinned 18d55bfd (2026-09-18): the same safetensors as 36c184c6, the revision Classmethod measured
# on 2x DGX Spark with no U+FFFD (the LibertAIDAI quant breaks CJK), plus zai-org's chat-template fix (240131d6).
# fetch | stage | check: deploy/stage-checkpoint.sh.
STAGE_REPO=RedHatAI/GLM-5.3-Flash-NVFP4 STAGE_REV=18d55bfd5a2194887738da73753975c9d3842f46 \
  STAGE_NAME=glm53-flash-cluster/GLM-5.3-Flash-NVFP4 exec "$(dirname "$0")/stage-checkpoint.sh" "$@"
