# syntax=docker/dockerfile:1
# The fleet's vLLM images: ONE Dockerfile, one STAGE per build of the image matrix (lb/vllm_lb/library/images/vllm.yml),
# each over the base its accelerator's pin names. What differs per (build × accelerator) — the base, pinned by digest,
# and the CPU platform — is data in <build>.<accelerator>.env beside this file, never a literal here; what a build ADDS
# to its base (an overlay file, its environment, its entrypoint, a patch) is its stage, and its files sit in ./<build>/.
#
#   deploy/images/build.sh vllm <build> <accelerator> build|ship|check|tag|remote …
#
# reads the pin, takes the tag from the image matrix and runs `docker build --platform <PLATFORM> --target <TARGET>
# --build-arg BASE=<BASE> deploy/images/vllm`. BuildKit builds only the stages the target needs, so the pair stages
# (no RUN) still build on an amd64 host for arm64 even though the `priority` stage below RUNs. Each stage is
# instruction for instruction the image its own Dockerfile built before the fold (deploy/tests/test_engine_images.py
# VllmStages holds each one exactly), so a rebuild changes no file, environment or entrypoint.
#
# A cell with no pin file has no known base: the build refuses it naming the file. By hand (any stage):
#   docker build --platform <platform> --target <stage> --build-arg BASE=<repo>@sha256:<digest> -t <tag> deploy/images/vllm
ARG BASE

# ── glm53: the pair's GLM-5.3-Flash image (catalog `glm-5.3-flash-nvfp4`, operator 2026-10-20) ─────────────────────────
# BASE (nvidia-sm121): tonyd2wild's sm121 v11 — vLLM's per-model glm53-flash-arm64-cu130 image (vllm
# 0.1.dev20051+g487ecf187, torch 2.13.0+cu130) plus the GB10 patch chain the stock image needs to boot this model at all
# (NoPE-MLA backend gate to sm_12x, FlashInfer 0.6.18, NCCL 2.30.7, PDL off on sm_12x, indexer hardening, fp8-KV tile
# cap). The stock image asserts `pe_dim must be 64 for fp8_ds_mla` at warmup on sm_121 whatever --kv-cache-dtype says
# (Classmethod 2026-09-02; himorishige/glm53-flash-2x-dgx-spark-recipe cluster.env.example). Pinned by digest: the one
# both published 2x DGX Spark runs of this checkpoint measured on (README of
# tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark, ghcr 2026-08-30).
FROM ${BASE:?pass--build-arg-BASE=the-vLLM-image-pinned-by-digest} AS glm53

# THE SM121 TOP-K FIX, baked in where the recipe bind-mounts it on both ranks (see glm53/NOTICE): `pool_topk` starts at
# -1 instead of torch.empty, and the persistent_topk kernel runs only on a GPU with >= 78 SMs — GB10 has 48 and 99 KB of
# shared memory, and the kernel aborts at long context there.
COPY glm53/sparse_attn_indexer_kpool_sm121.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py

# THE ENVIRONMENT the multimodal 2x DGX Spark recipe serves under (MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark @aed98a1
# files/Dockerfile.mm-ray; operator 2026-10-21). Its seven sm90 kernel steps and its FlashInfer 0.6.18 / NCCL 2.30.7 /
# cutlass-dsl 4.6.2 are already in the base (probed 2026-10-21), so only this is new: JIT for sm_121a alone (the base
# lists 8.0-12.0), the nightly FlashInfer's version check off, expandable segments on the unified pool, and an hour for
# the engine to report ready — this vLLM's own default is 600 s, under the 10-17 min a cold boot of this model takes.
ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

# ── deepseek-v4-b12x: the pair's DeepSeek-V4-Flash image (catalog `deepseek-v4-flash-nvfp4`, operator 2026-10-23) ─────
# BASE (nvidia-sm121): eugr/spark-vllm-b12x nightly-20261015 — the B12X runner eugr/spark-vllm-docker tests on DGX Spark
# before it advances `latest` (README: "pulls the separately tested eugr/spark-vllm-b12x:latest"), the image its
# recipes/deepseek-v4-flash-0731.yaml launches on 2x DGX Spark. vLLM 0.1.dev21830+g8244d1c44, torch 2.13.0+cu130,
# FlashInfer 0.7.1 (sm121a JIT cache), NCCL 2.29.7, b12x. Read 2026-10-23 off its files: DeepseekV4ForCausalLM, the
# deepseek_v4 tokenizer / tool / reasoning parsers, the b12x loader, and nvidia's NVFP4 experts routed to
# ModelOptNvFp4FusedMoE (models/deepseek_v4/quant_config.py:181-190) with a B12X backend (fused_moe/oracle/nvfp4.py:160).
FROM ${BASE} AS deepseek-v4-b12x

# THE RECIPE'S ENVIRONMENT, only the variables this image's code reads (vllm/envs.py, third_party/tml_fa4): the recipe
# also sets VLLM_MEMORY_PROFILE_INCLUDE_ATTN, VLLM_USE_B12X_{WO_PROJECTION,MHC,FP8_GEMM,MOE,SPARSE_INDEXER},
# B12X_MLA_SM120_UNIFIED and B12X_MOE_FORCE_A8, which nothing in it reads. Plus an hour for the engine to report ready:
# this vLLM's default is 600 s, and a cold boot of a 164 GiB checkpoint over NFS takes longer.
ENV CUTE_DSL_ARCH=sm_121a \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    VLLM_USE_MEGA_AOT_ARTIFACT=1 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_MOE_SKIP_PADDING=0 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

# tp-rank passes vLLM's arguments as the container's command, as vllm/vllm-openai's entrypoint takes them; this base
# enters through nvidia_entrypoint.sh with no command of its own.
ENTRYPOINT ["vllm", "serve"]

# THE DRAFTER'S EXPERTS ARE MXFP4 (generation 233: "strided dtype conversion: mtp.0.ffn.experts.0.w1.scale"): nvidia's
# export quantised only the experts its config lists (`quantized_layers`: layers.0-42.ffn.experts) and ignores `mtp.*`,
# but this vLLM gives every RoutedExperts NVFP4. quant_config.py = the base's own file plus `_is_nvfp4_layer`: NVFP4
# only where the checkpoint lists it, DeepSeek's MXFP4 elsewhere. Re-derive it from the base's file on a base bump.
COPY deepseek-v4-b12x/quant_config.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/quant_config.py

# The b12x weight loader reads through io_uring, which Docker >= 25's default seccomp profile refuses (generation 232):
# tp-rank launches an image with this label without that filter.
LABEL lb.io_uring=1

# ── priority: a one-card node's vLLM (EDGE_NODE_ENGINE=vllm, deploy/standalone/node-engine.vllm.yml) ──────────────────
# Upstream vllm-openai pinned by digest (BASE) plus the fleet's patches (./priority/), each gated by its own check.
#   priority.patch  THE CALLER'S PRIORITY (the LB's X-Priority): the Anthropic /v1/messages request carries `priority`
#                   into the chat request it runs (upstream's adapter dropped it), and a priority the scheduler would
#                   ignore (--scheduling-policy fcfs) is refused 400 instead — upstream's own field description promises
#                   that error and no code raised it. Made against vllm-project/vllm 2eb491e (0.31.1rc1.dev273).
# Its image-matrix entry is build `priority` on nvidia-sm120 × linux/amd64 (the RTX 5090 node's production image,
# 2026-10-30), but no pin file yet: the base its one build ran on is recorded only by a truncated digest
# (lb/vllm_lb/kv_cache_dtypes.json), and a digest is never guessed. Build it by hand with --target priority (above) —
# the built image's id (docker image inspect -f '{{.Id}}') is what the fleet's launch block names for engine vllm —
# or add priority.<accelerator>.env with the full digest, and the shared build takes it.
FROM ${BASE} AS priority
COPY priority/ /opt/fleet-patches/
RUN cd "$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
    && patch -p1 --forward --batch < /opt/fleet-patches/priority.patch
RUN cd /tmp && python3 /opt/fleet-patches/check_priority.py
