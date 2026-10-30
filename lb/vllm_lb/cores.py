"""WHAT THE GPU'S CORES ARE DOING, as an estimate from the engine's own step counters.

A GPU has no counter for it (a GB10 none at all): nvidia-smi's utilisation says a kernel was resident, not that it
was multiplying. So the split is derived, per GPU, from what the engine did between two probes:

  multiplying   real FLOP/s over the measured kernel peak. Real FLOPs are the tokens the layers
                PROCESSED × the model's own per-token cost — NOT vLLM's `estimated_flops`, which models
                every layer as dense full-context attention (87 % phantom on Flash-Next). For an SGLang
                generation it is `modeled_flops` over the tile's window: the COMPUTE tile's own figure, each
                prefill chunk priced at the depth it sits inside its prompt and timed off the engine's busy
                clock — so the two tiles are one number.
  memory        bytes of weights the steps had to stream, over the memory bus. A DENSE model reads every
                layer's weights on every step. A MoE step of T tokens touches 1 − (1 − top_k / experts)^T
                of the routed experts, and a step that carries a prefill chunk touches all of them.
                For an SGLang generation it is EVERY byte the LB models a pass moving — weights, the KV of
                every running context, the drafter's KV window, the GDN state — priced by the catalog's
                `StepTraffic`, the one model the MEMORY BANDWIDTH tile draws too (`traffic`). Weights alone
                were 13.0 GB of a ~41 GB decode pass, so the KV and GDN traffic fell into "other"
                (worklog/sglang-bottleneck-2026-10-03.md).
  other         the rest of the wall clock: kernel launches, the cross-box all-reduces, sampling — and
                the time the engine stood idle between requests.

WHAT A STEP HOLDS. `vllm:iteration_tokens_total` counts the tokens a step PRODUCED: the prompt tokens it
computed plus the output tokens it emitted. The engine processed more than it emitted — every running
sequence is verified at 1 + K positions (K = the drafter's depth) while ~3 are accepted — so the decode
side is charged for `running × (1 + K)` tokens, and only what the count holds beyond that is prefill.
With a drafter, each of its K passes re-reads the drafter's own weights and the lm_head.

It is an ESTIMATE and the page says so. A model with no shape recorded here gets no figure.

HOW MANY CORES is the serving card's, never the model's: the caller passes `cuda_cores` from the card the server runs
on (vllm_lb/topology.py `serving_cards`, its hardware type's CUDA cores — the fleet file names the card). A shape is
the CHECKPOINT's arithmetic and carries no hardware: the GB10's 6,144 used to be written into three rows here, so on
any other card the tile would have counted a GB10's cores. Each row also names the PRECISION its GEMMs run in
(`precision_of`): the COMPUTE tile's ceiling is the serving card's dense peak at that precision.
"""
from __future__ import annotations

from typing import NamedTuple

from vllm_lb.rates import FIRST_SAMPLE, NOT_EXPOSED, PASS_IN_FLIGHT

#: A step's time as `a + b × tokens` for Flash-Next, when the recent probes do not vary enough to fit it:
#: the 2026-10-02 fit of the pair (decode step 180 ms at 80 tokens, 0.34 ms per prefill token). It is also
#: what `fit_step_time` RETURNS when it has nothing to fit — `estimate` then takes the model's own prior.
STEP_FIT = (0.153, 0.00034)

#: Per GPU (one rank of TP=2). `fixed_bytes`: weights EVERY step reads. `expert_bytes`: routed expert
#: weights, read in proportion to the experts a step touches (0 for a dense model). `draft_bytes`: what
#: the drafter reads in ONE pass; `draft_flops`: its maths per drafted POSITION per running sequence;
#: `draft_block`: the positions one pass drafts — 1 for a sequential MTP head, so a step runs K passes
#: of it, and its block for a block drafter, which drafts all K in a single pass (DFlash 2: 8).
#: `attn_flops_per_ctx_token`: the attention core, per processed token per token of context, where
#: attention runs over the WHOLE context (0 where an indexer bounds it). `step_fit`: the prior for a
#: step's time, see STEP_FIT — the one field here that is a MEASUREMENT on an engine rather than
#: arithmetic off a checkpoint, so it does not carry over from one engine to another.

#: THE DENSE 27B's ARCHITECTURE — nvidia/Qwen3.8-27B-NVFP4's 64 layers — shared by every row that serves
#: them, because a FLOP does not depend on the dtype a checkpoint stores a weight in. Read off its three
#: shard headers and its config.json (2026-10-02, /srv/models/Qwen3.8-27B-NVFP4): DENSE (no experts),
#: 64 layers, hidden 5,120, FFN 17,408, 24 q / 4 kv heads x 256 = 12 + 2 a rank, vocab 248,320, 48 linear
#: + 16 full attention layers. 24.32e9 matmul elements in the layers -> 2 FLOP each, halved by TP; the
#: lm_head is 5,120 x 248,320 -> 2.543e9 FLOP, 1.271e9 a rank. The 16 full-attention layers read the whole
#: context: QK and AV, 2 FLOP x 12 heads x 256 each = 196,608 a processed token a context token.
_DENSE_27B = {
    "experts": 0, "top_k": 0, "expert_bytes": 0.0,
    "flops_per_token": 24.32e9,
    "flops_per_logit": 1.271e9,
    "attn_flops_per_ctx_token": 196_608.0,
}

SHAPES = {
    # Checkpoint + config.json (memory note vllm-mfu-estimator-overcounts-attention): 48 layers, hidden
    # 2560, 512 experts top-10, expert intermediate 640 (320 a rank), 12 + 1 q/kv heads a rank × 256,
    # vocab 248,320 (half a rank). Sparse attention (2,048 keys) and the drafter are left out: under 10 %.
    "qwen3.8-flash-next-hibrid48": {
        "precision": "nvfp4",               # the routed experts' GEMMs (FLASHINFER_CUTLASS nvfp4) dominate the FLOPs
        "experts": 512, "top_k": 10,
        "expert_bytes": 34.0e9,             # routed expert weights a rank
        "fixed_bytes": 1.1e9,               # attention projections + lm_head a rank, read every step
        "flops_per_token": 3.995e9,         # qkv + out projections + routed experts, 48 layers
        "flops_per_logit": 0.636e9,         # lm_head, per token that needs logits
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        "step_fit": STEP_FIT,
    },
    # THE SAME FLASH-NEXT CHECKPOINT, SERVED BY SGLANG (catalog `qwen3.8-flash-next-sglang`, 2026-10-05).
    # Every FLOP and byte term is the row above's, unchanged and deliberately so: a FLOP does not depend on
    # which server issues it, and these are the SAME 21 shards under the SAME NVFP4 quantisation. What
    # differs between the two profiles is not in this table — the engine, the drafter's ALGORITHM (SGLang
    # reads the `mtp.` tensors itself as NEXTN, where vLLM is handed a `-mtp` view) and the PLE table's
    # PLACEMENT (--ple-offload-embedding puts its 320,001,536 rows in host memory). None of the three moves
    # a per-step weight byte a rank reads: the PLE table is GATHERED BY TOKEN, never streamed, which is why
    # it is absent from `fixed_bytes` in both rows and why offloading it costs throughput nothing
    # measurable upstream (-0.07 % on H200 TP=4, lmsys.org/blog/2026-08-26-qwen-flash-next).
    # THE DRAFTER stays uncharged at 0.0, as in the row above, because it is the same `mtp.` head in the
    # same checkpoint; charging it here and not there would make the two profiles' tiles differ for a
    # reason that is not real.
    "qwen3.8-flash-next-sglang": {
        "precision": "nvfp4",               # the same checkpoint: its experts' GEMMs are nvfp4
        "experts": 512, "top_k": 10,
        "expert_bytes": 34.0e9,             # routed expert weights a rank
        "fixed_bytes": 1.1e9,               # attention projections + lm_head a rank, read every step
        "flops_per_token": 3.995e9,         # qkv + out projections + routed experts, 48 layers
        "flops_per_logit": 0.636e9,         # lm_head, per token that needs logits
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        # NOT MEASURED, for the same reason the dense SGLang row below says so: a step time is a
        # measurement on an ENGINE, and this pair has never run these weights on SGLang. Carries the
        # table's shared prior until the first 15-minute generation's (steps_s, tokens_step) probes can be
        # fitted the way `qwen3.8-27b-nvidia-nvfp4`'s pair was.
        "step_fit": STEP_FIT,
    },
    # THE SAME FLASH-NEXT ARCHITECTURE ON TENSORFOLD (catalog `qwen3.8-flash-next-tensorfold-*`, 2026-10-29). Every
    # FLOP term is the rows' above (48 layers, 512 experts top-10, the same heads and vocab: a FLOP does not depend on
    # the checkpoint's format). COMPUTE is `modeled_flops` over TensorFold's own pass, computed-prompt and busy-time
    # counters — the fleet image's patch 0014 (deploy/images/tensorfold); before it there were none and nothing was drawn.
    # nvidia/Qwen3.8-Flash-Next-NVFP4 (rev fc694b54): NVIDIA's own NVFP4 export of the same layers — its experts are
    # NVFP4 at the hibrid48 rows' width (0.5625 B a weight): 33,973,862,400 B a rank off its own shard headers
    # (2026-10-30). Its non-expert linears are BF16 as stored (TensorFold's loader: "bf16 linears as stored"), so a pass
    # streams 5.10e9 B of them a rank, not hibrid48's 1.1e9 — the catalog row's `traffic` (RoutedStepTraffic) carries
    # both, and with it the tiles price a pass from there; these two serve only a window with no traffic.
    # step_fit: the table's shared prior, never measured on this engine.
    "qwen3.8-flash-next-tensorfold-nvfp4": {
        "precision": "nvfp4",               # NVFP4 x NVFP4 in the experts ("checkpoint" precision, the native default)
        "experts": 512, "top_k": 10,
        "expert_bytes": 34.0e9,             # routed expert weights a rank, at NVFP4's width
        "fixed_bytes": 5.10e9,              # bf16 projections, hyper-connections, router, shared expert, lm_head a rank
        "flops_per_token": 3.995e9,         # qkv + out projections + routed experts, 48 layers
        "flops_per_logit": 0.636e9,         # lm_head, per token that needs logits
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        "step_fit": STEP_FIT,
    },
    # TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP (rev 2b170fa6): MLX affine 4-bit, one fp16 scale and one fp16 bias per
    # 32 weights = 5 bits = 0.625 B a weight against NVFP4's 0.5625, so the expert bytes scale by 10/9 (34.0e9 ->
    # 37.8e9). Its GEMMs dequantise to bf16 ("prompts: bf16 activations", the spike's rank 0), so the ceiling is bf16's.
    "qwen3.8-flash-next-tensorfold-mlx-mtp": {
        "precision": "bf16",
        "experts": 512, "top_k": 10,
        "expert_bytes": 37.8e9,             # 34.0e9 x 0.625 / 0.5625
        "fixed_bytes": 1.1e9,               # NOT re-derived for the MLX layout: the NVFP4 rows' figure
        "flops_per_token": 3.995e9,
        "flops_per_logit": 0.636e9,
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        "step_fit": STEP_FIT,
    },
    # nvidia/Qwen3.8-27B-NVFP4 served with its own in-checkpoint MTP head, on vLLM. Per rank, off the three
    # shard headers (2026-10-02): MLPs 8.965 GiB (NVFP4 weight 7.969 + scales 0.996) + linear attention
    # 5.204 + self attention 1.563 (FP8), halved = 8.45e9 B, + half the NVFP4 lm_head (weight 0.592 +
    # scales 0.074 GiB) 0.36e9. The MTP head is bf16: 0.42e9 B a rank, + the lm_head it shares with the
    # target = 0.782e9 a PASS, and it drafts one token a pass, so a step of K drafts runs K passes.
    "qwen3.8-27b-nvidia-nvfp4": {
        **_DENSE_27B,
        "precision": "nvfp4",               # the MLPs (most of a token's FLOPs) are NVFP4 weights
        "fixed_bytes": 8.80e9,              # every layer's weights + the lm_head, every step
        "draft_bytes": 0.782e9,             # the drafter's layer + the lm_head, per pass
        "draft_flops": 1.696e9,             # …per drafted position per running sequence
        "draft_block": 1,                   # a sequential MTP head: one token a pass
        # First hour of generation 103 (3 running): 93 ms a decode step, 0.18-0.32 ms per prefill token.
        "step_fit": (0.090, 0.00025),
    },
    # The SAME 27B layers under a BF16 lm_head, drafted by its own DFlash2 checkpoint (catalog
    # `qwen3.8-27b-dflash2`: path dflash2-27b-tp2/Qwen3.8-27B-NVFP4-bf16head, method "dflash", 7 drafts).
    # Read on box-c 2026-10-02, box-a being down. THE LAYERS are the row above's byte for byte; only the
    # head's dtype differs — 248,320 x 5,120 BF16 = 2.543e9 B, 1.271e9 a rank, against the NVFP4 head's
    # 0.36e9 — which is the +1.70 GiB the catalog's own `size_gib` records between the two paths (22.1
    # against 20.4), and the reason the head was converted at all: the drafter's selector needs a dense one.
    # THE DRAFTER is not an MTP head. dflash2-27b-tp2/Qwen3.8-27B-DFlash2/model.safetensors holds
    # 3,848,808,960 B of BF16 tensors = 1.924e9 parameters (5 sliding-attention layers of hidden 5,120 and
    # FFN 17,408, an `fc` over 5 target layers' hidden states, two 248,320 x 256 selector codebooks), and
    # ships NO lm_head and no embeddings — "the draft does not ship one … runs on a head taken from the
    # **target**" (worklog/dflash2-27b-tp2.md). So a pass reads the whole drafter, REPLICATED per rank (that
    # file is the 3.58 GiB the catalog measured as this profile's `mtp_gib` on a real start), PLUS the
    # target's BF16 head a rank: 3.85 + 1.27 = 5.12e9 B, and 2 FLOP a parameter plus the head's 1.271e9 for
    # every position it drafts — the same two terms the MTP row above carries, on a bigger drafter.
    # ONE PASS A STEP, not seven: the block is paid in full every step. The session's own counters make it
    # exact — 83,048 draft tokens over 11,864 drafts is 7.00 each, and accepted positions are recorded for
    # 0-6 only (lb/state/benchq/sessions/S-20260101-000002/metrics-after.txt) — and `draft_block` 8 is the
    # drafter's `dflash_config.block_size`, which deploy/dflash2-27b-tp2/env.sh reads the same way: "K=7
    # … one block minus the bonus query", "K=16 (two blocks)". Charged as seven passes, the drafter would
    # have read 35.8 GB a step off a 273 GB/s bus instead of 5.1, and called a quiet pair memory-bound.
    "qwen3.8-27b-dflash2": {
        **_DENSE_27B,
        "precision": "nvfp4",               # the same NVFP4 layers; only the head and drafter are BF16
        "fixed_bytes": 9.72e9,              # 8.45e9 of layers + 1.271e9 of BF16 lm_head, every step
        "draft_bytes": 5.12e9,              # the whole drafter + the target's head, once a pass, a rank
        "draft_flops": 5.12e9,              # 2 FLOP x 1.924e9 parameters + the head, a drafted position
        "draft_block": 8,                   # a block drafter: all 7 drafts in one pass
        # MEASURED on generation 109 (K=7, budget 4,096, lb/state/perf/perf-2026-10-02.jsonl): 106.0 ms a
        # decode step at one seat (9 windows, 5,023 steps) against the MTP profile's 93, and 302.1 ms at
        # eight seats (4 windows, 5,732) — a seat term this table does not carry, so the prior is the
        # single-seat step, as the row above's is. The slope is `fit_step_time`'s own over that
        # generation's ten (steps_s, tokens_step) probes: 0.298 ms a token, beside the MTP row's 0.25.
        "step_fit": (0.106, 0.000298),
    },
    # THE SAME CHECKPOINT AND THE SAME DRAFTER, SERVED BY SGLANG. The architecture is shared, so every
    # FLOP term comes from _DENSE_27B untouched. ITS BYTES ARE NOT READ FROM THIS ROW when proxy.py passes the
    # catalog's `traffic` — which it does for every SGLang generation: a decode pass is then priced whole
    # (`estimate`'s `traffic`), and SGLang SHARDS the drafter (2.25 GiB a rank in its own load log, not the
    # 3.85e9 file below). The two byte figures here are the weights-only fallback. What differs is the
    # bytes — this profile serves
    # nvidia/Qwen3.8-27B-NVFP4 with its lm_head still NVFP4, where the vLLM row above serves the copy
    # whose head was dequantised to BF16 because vLLM's DFlash2 selector refuses a quantised one
    # (worklog/dflash2-27b-tp2.md). The head is 248,320 x 5,120: 1.271e9 B a rank at BF16, 0.358e9 at
    # NVFP4's 0.5625 B a weight. The drafter still borrows the target's head, so both byte figures move
    # by the same 0.913e9.
    "qwen3.8-27b-sglang-dflash2": {
        **_DENSE_27B,
        "precision": "nvfp4",               # the NVFP4 layers and head
        "fixed_bytes": 8.80e9,              # 8.45e9 of layers + 0.358e9 of NVFP4 lm_head, every step
        "draft_bytes": 4.21e9,              # the whole drafter + the target's NVFP4 head, once a pass
        "draft_flops": 5.12e9,              # unchanged: FLOPs do not depend on the dtype stored
        "draft_block": 8,                   # the same DFlash2 drafter, and SGLang reports block_size 8
        # NOT MEASURED, deliberately: a step time is a measurement on an ENGINE, and SGLang's step is not
        # vLLM's for identical weights. The 2026-10-02 run gives decode passes and decode tokens over one
        # 62 s window (1,443 passes, 19,138 tokens) — one equation, which cannot fit two parameters — so
        # this carries the table's shared prior and the tile is an estimate on a prior, as it says it is.
        # Replace it with a pair fitted the way the row above's was once a generation's probes exist.
        # Only a window that ran ONE kind of pass, or too few probes to fit, reads it: `modeled_flops`
        # times the passes off the engine's own busy clock.
        "step_fit": STEP_FIT,
        # THE DRAFTER DURING PREFILL: DFlash2 projects every new prompt token's target hidden states into
        # its own KV (srt/models/dflash.py `project_target_hidden`, `kv_proj_only`) — its REPLICATED `fc`,
        # 5 x 5,120 -> 5,120 (2 x 131.07e6), and its five layers' K and V, 5,120 -> 2 x 8 x 128, sharded
        # (2 x 52.43e6 / 2): 0.3146e9 a prompt token, 1.3 % of the layers' 24.32e9.
        "draft_prefill_flops": 2 * 5 * 5120 * 5120 + 5 * 5120 * 1024 * 2,
        # SGLANG'S OWN FLOP ESTIMATOR'S TWO CONSTANTS for this checkpoint at TP=2, PER GPU — what its
        # `sglang:estimated_flops_per_gpu_total` charges (managers/scheduler_components/metrics_reporter.py
        # :476-620 in the pinned image, `_init_estimated_perf_constants`): a token's linear layers as 64
        # dense GQA layers (`num_attention_layers` = `num_hidden_layers`, configs/model_config.py:1242) with
        # the MLP unsharded, (2·h·d·(q+2kv) + 2·h·d·q + 6·h·I) x 64 at h 5,120, d 256, q 12, kv 2, I 17,408;
        # and a query-key pair as 4·q·d x 64. Against this row's real 24.32e9 and 196,608 that is x1.60 and
        # x4.00. NOT drawn: `modeled_flops` uses them only to read back the causal pairs the counter holds.
        "engine_linear_flops": (2 * 5120 * 256 * (12 + 2 * 2) + 2 * 5120 * 256 * 12 + 6 * 5120 * 17408) * 64,
        "engine_attn_flops": 4 * 12 * 256 * 64,
    },
    # zai-org GLM-5.3-Flash in RedHatAI's compressed-tensors NVFP4 (catalog `glm-5.3-flash-nvfp4`, operator
    # 2026-10-20). Per rank, off the ten shard headers (rev 18d55bfd, read 2026-10-20; NVFP4 packed U8 counted two
    # elements a byte): 45 layers (11 sparse-MLA, 34 KDA linear attention; 3 dense MLPs, 42 MoE), hidden 4,096,
    # 288 routed experts top-8 + 1 shared. Routed experts 159.469 GiB (304.4e9 elements), halved = 85.62e9 B; every
    # other non-embedding matmul (attention, indexer, KDA, shared expert, dense MLPs, lm_head) 15.44 GiB, halved =
    # 8.29e9 B, read every step. A token's FLOPs: 7.610e9 non-expert + 304.4e9 x 8/288 routed = 16.07e9 parameters,
    # 2 FLOPs each, halved = 16.07e9 a rank; the lm_head 4,096 x 154,880 = 0.634e9 a rank per logit. The MTP layer
    # (model_mtp.safetensors, FP8-block, its own 288 experts) and the sparse indexer are left out, as Flash-Next's
    # drafter and sparse attention are above: one pass of 45+1 layers a draft. The vision tower is not loaded
    # (language_model_only). NOT MEASURED on an engine: the shared prior until the first generation's probes fit.
    "glm-5.3-flash-nvfp4": {
        "precision": "nvfp4",               # the routed experts (Marlin NVFP4) dominate the FLOPs
        "experts": 288, "top_k": 8,
        "expert_bytes": 85.62e9,            # routed expert weights a rank
        "fixed_bytes": 8.29e9,              # every non-expert matmul + the lm_head a rank, read every step
        "flops_per_token": 16.07e9,         # non-expert matmuls + 8 routed experts, 45 layers, a rank
        "flops_per_logit": 0.634e9,         # lm_head, per token that needs logits
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        "step_fit": STEP_FIT,
    },
    # DeepSeek-V4-Flash-0731 in nvidia's NVFP4 (catalog `deepseek-v4-flash-nvfp4`, operator 2026-10-23). Per rank, off
    # the 48 shard headers (rev f1caa711, read 2026-10-23 by HTTP range; NVFP4 packed U8 counted two elements a byte):
    # 43 layers, hidden 4,096, 256 routed experts top-6 + 1 shared, expert intermediate 2,048. Routed experts 277.03e9
    # elements, 145.13 GiB, halved = 77.92e9 B; every other non-embedding matmul (attention + compressor, shared
    # expert, router, hyper-connections) and the lm_head 7.23 GiB, halved = 3.88e9 B, read every step. A token's
    # FLOPs: 6.212e9 non-expert + 277.03e9 x 6/256 routed = 12.71e9 parameters (the card's "13B activated"), 2 FLOPs
    # each, halved = 12.71e9 a rank; the lm_head 4,096 x 129,280 = 0.530e9 a rank per logit. The DSpark drafter (mtp.*,
    # its experts MXFP4) and the compressed/sparse attention core are left out, as Flash-Next's are above.
    # NOT MEASURED on an engine: the shared prior until the first generation's probes fit.
    "deepseek-v4-flash-nvfp4": {
        "precision": "nvfp4",               # the routed experts (B12X NVFP4 MoE) dominate the FLOPs
        "experts": 256, "top_k": 6,
        "expert_bytes": 77.92e9,            # routed expert weights a rank
        "fixed_bytes": 3.88e9,              # every non-expert matmul + the lm_head a rank, read every step
        "flops_per_token": 12.71e9,         # non-expert matmuls + 6 routed experts, 43 layers, a rank
        "flops_per_logit": 0.530e9,         # lm_head, per token that needs logits
        "draft_bytes": 0.0, "draft_flops": 0.0, "draft_block": 1, "attn_flops_per_ctx_token": 0.0,
        "step_fit": STEP_FIT,
    },
}


def precision_of(model: str | None) -> str | None:
    """The precision `model`'s GEMMs run in — the COMPUTE tile's peak is the serving card's dense peak at it — or None
    for a model with no shape recorded here (then no compute peak is claimed for it)."""
    shape = SHAPES.get(model or "")
    return shape.get("precision") if shape is not None else None


def mean_context(kv_fraction, *, running, pool_blocks, blocks) -> float | None:
    """The mean tokens of CONTEXT a running request holds: the KV gauge counts the blocks the running
    requests reference, F of which per request are the model's fixed state (`KvBlocks.fixed_blocks`), not
    context — counting them charged 18 × 816 = 14.7k tokens of attention per request on the 27B. None when
    nothing runs or the pool's block count is unknown; never negative."""
    nums = (kv_fraction, running, pool_blocks)
    if blocks is None or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in nums):
        return None
    if running <= 0 or pool_blocks <= 0:
        return None
    held = kv_fraction * pool_blocks / running
    return max(0.0, held - blocks.fixed_blocks) * blocks.block_tokens


def fit_step_time(pairs) -> tuple[float, float]:
    """Seconds-per-step as `a + b × tokens-per-step`, fitted over recent (steps_s, tokens_step) probes —
    or STEP_FIT when there are too few, they do not vary, or the line is not a cost (a or b ≤ 0).

    Theil-Sen (the median slope of every pair of probes), not least squares: a probe interval in which the
    engine stood idle part of the time reports few steps a second, i.e. a "step" seconds long, and one such
    outlier bends a least-squares line (live 2026-10-02: 2,123 ms beside 93 ms steps gave a = 184 ms)."""
    pts = [(float(t), 1.0 / s) for s, t in pairs
           if isinstance(s, (int, float)) and isinstance(t, (int, float)) and s > 0 and t > 0]
    if len(pts) < 8 or max(x for x, _ in pts) - min(x for x, _ in pts) < 25.0:   # nothing to fit a slope to
        return STEP_FIT
    slopes = sorted((y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(pts) for x2, y2 in pts[i + 1:]
                    if abs(x2 - x1) >= 25.0)
    if not slopes:
        return STEP_FIT
    b = slopes[len(slopes) // 2]
    rest = sorted(y - b * x for x, y in pts)
    a = rest[len(rest) // 2]
    return (a, b) if a > 0 and b > 0 else STEP_FIT

# ── THE MODELED FLOPs OF AN SGLANG GENERATION: the one figure COMPUTE and CUDA CORES both draw ─────────
#
# SGLang's own `sglang:estimated_flops_per_gpu_total` is not that figure: it charges every one of the 27B's 64
# layers as dense attention with an unsharded MLP (the row's `engine_*` constants: x1.60 a token's linear
# layers, x4.00 a query-key pair), so it read 236.8 TFLOP/s beside a cores tile at 742 of 6,144 on the same
# prefill-heavy minute (2026-10-03). What it DOES hold exactly is the depth of every prefill chunk inside its
# prompt — `_prefill_attention_pairs` sums c·p + c(c+1)/2 over each batch's extend and prefix lengths — and
# that depth is the one thing no other counter carries. So the pairs are read back OUT of it with its own
# constants, and priced with the real ones:
#
#     P = (ΔE − L_sg x (T_prefill + T_decode) − A_sg x n_decode x Σctx) / A_sg
#
# (its decode term is L_sg per EMITTED token plus A_sg per context token a pass, metrics_reporter.py:597-
# 620). Five consecutive 8,192-token chunks of one cold prompt read back as (k + ½) x 8,192 + ½ for k = 14..18,
# to the token (tests/test_sglang_compute_model.py). Without the counter (`--enable-mfu-metrics` off) the
# depth is the cached prefix a pass + half a pass — which misses how deep a CONTINUATION chunk sits, and
# `depth_from` says which it was.
#
# THE TIME is SGLang's own `scheduler_stage_seconds_total{category="run_batch"}` (`busy_seconds_total`),
# credited when a pass returns, as the pass counter is. It is split between the two kinds of pass by what was
# measured: all of it to the one kind when only one ran; else a least-squares fit over the window's probe
# intervals — busy = t_decode x decode passes + prefill FLOPs / prefill FLOP rate — when their mixes vary;
# else the step-time prior's ratio, scaled to the measured total. The chunk a prefill pass carries is the
# ENGINE's (computed tokens over prefill passes), never the panel's budget: the two have differed.

#: The engine's own counters and gauges `modeled_flops` reads at each probe (probe.py's canonical keys), kept
#: per server by the hub (`Hub.engine_reads`).
READ_COUNTERS = ("steps_total", "decode_steps_total", "prompt_computed_total", "prompt_cached_total",
                 "gen_tokens_total", "est_flops_total", "busy_seconds_total")
READ_GAUGES = ("decode_context_tokens", "running", "spec_verify_width")
READ_KEYS = READ_COUNTERS + READ_GAUGES
#: The counters an interval cannot be priced without.
_REQUIRED = ("steps_total", "decode_steps_total", "prompt_computed_total")
#: No prompt token sits deeper than the model's context window: the read-back pairs' ceiling.
MAX_DEPTH = 262_144

#: How a window's busy time was split between its decode and its prefill passes (`split`).
SPLIT_ONE, SPLIT_FIT, SPLIT_PRIOR, SPLIT_WALL = "one kind", "fit", "prior", "prior, wall clock"
#: Where a prefill pass's attention depth came from (`depth_from`).
DEPTH_ENGINE, DEPTH_CACHED = "engine", "cached prefix"
#: A fit needs intervals whose mixes of decode and prefill VARY: below this share of independent variation
#: (1 − r² of the two regressors) the two times are not separable, and the prior's ratio is used instead.
_FIT_MIN_SPREAD = 0.02

#: The keys of the `compute_model` figure (saturation.py), every one present, null where not known.
FLOP_MODEL_KEYS = ("tflops_per_gpu", "window_s", "why", "engine_tflops", "prefill_passes", "prefill_tokens",
                   "prefill_chunk", "prefill_depth", "depth_from", "prefill_flops", "prefill_s", "prefill_tflops",
                   "decode_passes", "decode_flops", "decode_s", "decode_tflops", "busy_s", "split",
                   "context_tokens", "seqs", "verify_width")


class Interval(NamedTuple):
    """What the engine did between two consecutive reads, by its own counters; the gauges are the two reads'
    mean (SGLang sets `decode_sum_seq_lens` every 40th decode pass)."""
    seconds: float
    decode_passes: float
    prefill_passes: float
    prefill_tokens: float
    cached_tokens: float
    decode_tokens: float | None
    engine_flops: float | None
    busy_s: float | None
    context: float
    seqs: float
    width: float | None


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _gauge(a: dict, b: dict, key: str) -> float | None:
    seen = [v for v in (_num(a.get(key)), _num(b.get(key))) if v is not None]
    return sum(seen) / len(seen) if seen else None


def intervals(reads) -> list[Interval]:
    """Every interval between two consecutive reads that both carry the counters — never across a read that
    failed (None), and never across a restart (a counter that went backwards)."""
    out = []
    for (a_at, a), (b_at, b) in zip(reads, reads[1:]):
        if not a or not b or b_at <= a_at:
            continue
        delta: dict[str, float | None] = {}
        for key in READ_COUNTERS:
            x, y = _num(a.get(key)), _num(b.get(key))
            delta[key] = None if x is None or y is None else y - x
        if any(delta[k] is None for k in _REQUIRED) or any(v is not None and v < 0 for v in delta.values()):
            continue
        decode = delta["decode_steps_total"]
        width = _num(b.get("spec_verify_width"))
        out.append(Interval(seconds=b_at - a_at, decode_passes=decode,
                            prefill_passes=max(0.0, delta["steps_total"] - decode),
                            prefill_tokens=delta["prompt_computed_total"],
                            cached_tokens=delta["prompt_cached_total"] or 0.0,
                            decode_tokens=delta["gen_tokens_total"], engine_flops=delta["est_flops_total"],
                            busy_s=delta["busy_seconds_total"],
                            context=_gauge(a, b, "decode_context_tokens") or 0.0,
                            seqs=_gauge(a, b, "running") or 0.0, width=width))
    return out


def verify_flops(shape: dict, *, context: float, seqs: float, width) -> float:
    """One DECODE pass's FLOPs: seqs x D verified positions through the layers and the lm_head, every one
    attending to its sequence's whole context (D x Σctx query-key pairs), and the drafter's one block of D
    positions a sequence — the CUDA CORES tile's decode pass (`estimate`). No drafter: one position a sequence."""
    drafting = isinstance(width, (int, float)) and width >= 2
    w = float(width) if drafting else 1.0
    return (seqs * w * (shape["flops_per_token"] + shape["flops_per_logit"])
            + w * context * shape["attn_flops_per_ctx_token"]
            + (w * seqs * shape["draft_flops"] if drafting else 0.0))


def _pairs(shape: dict, iv: Interval) -> tuple[float, str | None]:
    """The causal query-key pairs of an interval's computed prompt tokens, and where they came from."""
    if iv.prefill_passes <= 0 or iv.prefill_tokens <= 0:
        return 0.0, None
    chunk = iv.prefill_tokens / iv.prefill_passes
    lin, att = shape.get("engine_linear_flops"), shape.get("engine_attn_flops")
    if iv.engine_flops is not None and iv.decode_tokens is not None and lin and att:
        pairs = (iv.engine_flops - lin * (iv.prefill_tokens + iv.decode_tokens)
                 - att * iv.decode_passes * iv.context) / att
        # Never fewer than every pass cold (Σ c(c+1)/2 >= T(T/n + 1)/2), never deeper than the window.
        return min(max(pairs, iv.prefill_tokens * (chunk + 1) / 2), iv.prefill_tokens * MAX_DEPTH), DEPTH_ENGINE
    return iv.prefill_tokens * (iv.cached_tokens / iv.prefill_passes + chunk / 2), DEPTH_CACHED


def _prefill_flops(shape: dict, tokens: float, pairs: float) -> float:
    return tokens * (shape["flops_per_token"] + shape.get("draft_prefill_flops", 0.0)) \
        + pairs * shape["attn_flops_per_ctx_token"]


def _fit_busy(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """(seconds a decode pass, seconds a prefill FLOP) by least squares over (decode passes, prefill FLOPs,
    busy seconds) per interval — None when the mixes do not vary enough to separate them, or the fit is not
    a cost (either term <= 0)."""
    s11 = sum(d * d for d, _, _ in rows)
    s12 = sum(d * f for d, f, _ in rows)
    s22 = sum(f * f for _, f, _ in rows)
    det = s11 * s22 - s12 * s12
    if s11 <= 0 or s22 <= 0 or det <= _FIT_MIN_SPREAD * s11 * s22:
        return None
    r1 = sum(d * b for d, _, b in rows)
    r2 = sum(f * b for _, f, b in rows)
    x, y = (r1 * s22 - r2 * s12) / det, (s11 * r2 - s12 * r1) / det
    return (x, y) if x > 0 and y > 0 else None


def _priced(shape: dict, ivs: list[Interval]) -> list[tuple[Interval, float, float, str | None, float]]:
    """Each interval with its prefill FLOPs, its prompt tokens' query-key pairs and where they came from, and
    its decode FLOPs."""
    out = []
    for iv in ivs:
        pairs, source = _pairs(shape, iv)
        out.append((iv, _prefill_flops(shape, iv.prefill_tokens, pairs), pairs, source,
                    iv.decode_passes * verify_flops(shape, context=iv.context, seqs=iv.seqs, width=iv.width)))
    return out


def _fit_rows(priced) -> list[tuple[float, float, float]]:
    return [(iv.decode_passes, f_pre, iv.busy_s) for iv, f_pre, _, _, _ in priced if iv.busy_s is not None]


def modeled_flops(model: str | None, reads, *, peak_tflops: float | None = None, peak_gbps: float | None = None,
                  traffic=None, fit=STEP_FIT, history=None) -> dict | None:
    """FLOP/s PER GPU over `reads` — [(monotonic time, the engine's canonical scrape or None)], oldest first —
    priced with this model's own shape, and its parts (`FLOP_MODEL_KEYS`). None for a model with no shape; a figure
    with `why` and a null rate when the reads cannot give one. The shape's SGLang estimator constants
    (`engine_linear_flops`, `engine_attn_flops`) are read only to take a chunk's depth back out of SGLang's own counter
    (`_pairs`); without them — or without that counter — the depth is the cached prefix plus half a pass.

    The rate is the window's FLOPs over max(wall clock, the busy time its passes were credited): a pass is
    counted when it returns, so a window shorter than the passes it counted is widened to their time — one
    8,192-token chunk took 6.5 s inside a 5.0 s pair of reads.

    THE SPLIT of the busy time: all of it to the one kind that ran; else the least-squares fit over the
    window's intervals, else over `history` (the longer run of reads the window ends, when the window's own
    mixes did not vary), else the step-time prior's ratio (`fit`) — scaled to the measured total. Then a kind
    is given at least its ROOFLINE — its FLOPs at `peak_tflops`, and with `traffic` its bytes
    (`StepTraffic`) at `peak_gbps` — taken from the other kind's share: a fit once gave 11 decode passes 0.4 s,
    a third of what their 37 GB each need on a 273 GB/s bus. So no figure here exceeds either peak."""
    shape = SHAPES.get(model or "")
    if shape is None:
        return None
    reads = list(reads or [])
    ivs = intervals(reads)
    last = next((r for _, r in reversed(reads) if r), None)
    out = dict.fromkeys(FLOP_MODEL_KEYS)
    if last is not None:
        out.update(context_tokens=_num(last.get("decode_context_tokens")), seqs=_num(last.get("running")),
                   verify_width=_num(last.get("spec_verify_width")))
        # THE ENGINE'S OWN mean prefill pass since it started, for the prefill case a window without one
        # draws at 0 % of the time (`estimate`): never the panel's budget, which the engine may not be using.
        passes = (_num(last.get("steps_total")) or 0.0) - (_num(last.get("decode_steps_total")) or 0.0)
        computed = _num(last.get("prompt_computed_total"))
        if passes > 0 and computed:
            chunk = computed / passes
            out.update(prefill_chunk=chunk, depth_from=DEPTH_CACHED,
                       prefill_depth=(_num(last.get("prompt_cached_total")) or 0.0) / passes + chunk / 2)
    if not ivs:
        out["why"] = NOT_EXPOSED if last is not None and any(_num(last.get(k)) is None for k in _REQUIRED) \
            else FIRST_SAMPLE
        return out
    wall = sum(iv.seconds for iv in ivs)
    n_d = sum(iv.decode_passes for iv in ivs)
    n_p = sum(iv.prefill_passes for iv in ivs)
    if n_d + n_p <= 0:
        if (out["seqs"] or 0) > 0:
            out["why"] = PASS_IN_FLIGHT          # work running, and no pass came back in the window
        else:
            out.update(tflops_per_gpu=0.0, window_s=wall, prefill_passes=0.0, decode_passes=0.0)
        return out
    priced = _priced(shape, ivs)
    tokens = sum(iv.prefill_tokens for iv in ivs)
    pairs = sum(p for _, _, p, _, _ in priced)
    sources = {src for _, _, _, src, _ in priced if src is not None}
    prefill = sum(f for _, f, _, _, _ in priced)
    decode = sum(f for _, _, _, _, f in priced)
    engine = [iv.engine_flops for iv in ivs]
    busy = [iv.busy_s for iv in ivs]
    busy_s = sum(busy) if all(b is not None for b in busy) else None

    # THE SPLIT of the engine's busy clock between the two kinds of pass.
    a, b = shape["step_fit"] if fit is STEP_FIT else fit
    seqs = max(1.0, sum(iv.decode_passes * iv.seqs for iv in ivs) / n_d) if n_d else 1.0
    widths = [iv.width for iv in ivs if iv.decode_passes and iv.width is not None]
    w = max(1.0, widths[-1]) if widths and widths[-1] >= 2 else 1.0
    prior = (n_d * (a + b * seqs * w), n_p * (a + b * (tokens / n_p if n_p else 0.0)))
    if busy_s is None:
        d_s, p_s = prior
        scale = wall / (d_s + p_s) if d_s + p_s > wall else 1.0
        d_s, p_s, split = d_s * scale, p_s * scale, SPLIT_WALL
    elif not n_p or not n_d:
        (d_s, p_s), split = ((busy_s, 0.0) if not n_p else (0.0, busy_s)), SPLIT_ONE
    else:
        fitted = _fit_busy(_fit_rows(priced))
        if fitted is None and history:
            fitted = _fit_busy(_fit_rows(_priced(shape, intervals(list(history)))))
        (d_s, p_s), split = ((fitted[0] * n_d, fitted[1] * prefill), SPLIT_FIT) if fitted else (prior, SPLIT_PRIOR)
        scale = busy_s / (d_s + p_s) if d_s + p_s > 0 else 0.0
        d_s, p_s = d_s * scale, p_s * scale
    # THE ROOFLINE: a kind that ran took at least its FLOPs at peak and its bytes over the bus — taken from the
    # other kind's share of a measured clock, never added to it, while that one stays above its own.
    peak = peak_tflops * 1e12 if isinstance(peak_tflops, (int, float)) and peak_tflops > 0 else None
    bus = peak_gbps * 1e9 if traffic is not None and isinstance(peak_gbps, (int, float)) and peak_gbps > 0 else None
    d_floor = max(decode / peak if peak else 0.0,
                  sum(iv.decode_passes * sum(traffic.decode_pass_bytes(context=iv.context, seqs=iv.seqs,
                                                                       verify_width=iv.width))
                      for iv in ivs) / bus if bus else 0.0)
    p_floor = max(prefill / peak if peak else 0.0,
                  n_p * traffic.prefill_bytes(tokens / n_p if n_p else None) / bus if bus else 0.0)
    if busy_s is not None and split != SPLIT_WALL:
        if d_s < d_floor:
            d_s, p_s = d_floor, max(p_floor, p_s - (d_floor - d_s))
        if p_s < p_floor:
            p_s, d_s = p_floor, max(d_floor, d_s - (p_floor - p_s))
    d_s, p_s = max(d_s, d_floor), max(p_s, p_floor)
    if n_d and d_s <= 0:
        d_s = prior[0]                       # a kind that ran took SOME time, whatever the clock said
    if n_p and p_s <= 0:
        p_s = prior[1]
    window = max(wall, d_s + p_s)
    rate = (prefill + decode) / window
    out.update(tflops_per_gpu=(min(rate, peak) if peak else rate) / 1e12, window_s=window, why=None,
               engine_tflops=(sum(engine) / window / 1e12 if all(e is not None for e in engine) else None),
               prefill_passes=n_p, prefill_tokens=tokens, prefill_flops=prefill, prefill_s=p_s,
               prefill_tflops=prefill / p_s / 1e12 if p_s > 0 else None,
               decode_passes=n_d, decode_flops=decode, decode_s=d_s,
               decode_tflops=decode / d_s / 1e12 if d_s > 0 else None,
               busy_s=busy_s, split=split)
    if n_p:
        out.update(prefill_chunk=tokens / n_p, prefill_depth=pairs / tokens if tokens else 0.0,
                   depth_from=DEPTH_ENGINE if sources == {DEPTH_ENGINE} else DEPTH_CACHED)
    return out


def estimate(model: str | None, *, steps_s, tokens_step, running, speculative, budget,
             peak_tflops, peak_gbps, cuda_cores: int | None, fit=STEP_FIT, context=None, traffic=None,
             verify_width=None, flops=None) -> dict | None:
    """The split of one GPU's cores, time-averaged over the probe window — or None when the model has
    no recorded shape or the engine completed no step in the window.

    `context`: the mean tokens of context a running request holds, for the models whose attention runs
    over all of it; None leaves that term out (an under-count of the multiplying share for them).

    `traffic`, `verify_width`: an SGLang generation's catalog `StepTraffic` and the tokens its verify
    carries per sequence (`spec_num_draft_tokens`, bonus included). With them a decode pass is priced by
    `StepTraffic.decode_pass_bytes` — the MEMORY BANDWIDTH tile's own model — and a prefill pass by its
    weights, as that tile's floor is; the verify carries `verify_width` tokens a sequence (4 on this pair —
    not 1 + K = 5: SGLang resolves DFlash2's block to the width, 'block_size=4' in its start log) and the
    drafter one block of that many positions.

    `flops`: an SGLang generation's `modeled_flops` over the tile's window. With it and `traffic` the tile is
    drawn from THAT window alone (`_from_window`) — the passes it counted, the chunk the ENGINE ran, each kind
    timed off the engine's busy clock — and its multiplying cores are the COMPUTE tile's FLOP/s; `steps_s`,
    `tokens_step` and `budget` are not read.

    `cuda_cores`: the serving card's CUDA cores (its hardware type). None — no card named, or none known — is no
    tile: a count nobody stated is never stood in for."""
    if not isinstance(cuda_cores, int) or isinstance(cuda_cores, bool) or cuda_cores <= 0:
        return None
    shape = SHAPES.get(model or "")
    if shape is not None and traffic is not None and flops is not None:
        return _from_window(shape, flops, traffic, cuda_cores=cuda_cores, peak_tflops=peak_tflops,
                            peak_gbps=peak_gbps, fit=fit)
    nums = (steps_s, tokens_step, running, peak_tflops, peak_gbps)
    if shape is None or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in nums):
        return None
    if steps_s <= 0 or tokens_step <= 0 or peak_tflops <= 0 or peak_gbps <= 0:
        return None
    drafts = max(0, int(speculative or 0))
    # THE DRAFTER'S PASSES, which is not its tokens: a sequential MTP head runs once per draft token, a
    # block drafter proposes the whole block in ONE pass and re-reads its weights once, not K times
    # (`draft_block`). Its MATHS still scales with the positions it computes — a block at a time.
    block = max(1, int(shape["draft_block"]))
    passes = -(-drafts // block)
    positions = passes * block
    seqs = max(1.0, float(running))
    decode = seqs * (1 + drafts)                     # tokens the target VERIFIES a step, emitted or not
    width = None
    if traffic is not None:
        # The ENGINE's verify width when it said one; the catalog's 1 + K only when it did not.
        width = int(verify_width) if isinstance(verify_width, (int, float)) else (1 + drafts if drafts else 0)
        drafting = width >= 2
        decode = seqs * (width if drafting else 1)
        passes, positions = (1, width) if drafting else (0, 0)
    prefill = max(0.0, tokens_step - min(tokens_step, decode))       # computed prompt tokens a mean step
    mixed = 0.0 if not budget else max(0.0, min(1.0, prefill / budget))
    chunk = prefill / mixed if mixed > 0 else float(budget or 0)     # …and in a step that carries a chunk
    touched = (1.0 - (1.0 - shape["top_k"] / shape["experts"]) ** decode) if shape["experts"] else 0.0
    ctx = float(context) if isinstance(context, (int, float)) and not isinstance(context, bool) and context > 0 else 0.0
    per_token = shape["flops_per_token"] + ctx * shape["attn_flops_per_ctx_token"]

    if traffic is not None:
        decode_bytes = sum(traffic.decode_pass_bytes(context=ctx * seqs, seqs=seqs, verify_width=width))
        mixed_bytes = traffic.prefill_bytes(chunk)
    else:
        decode_bytes = shape["fixed_bytes"] + shape["expert_bytes"] * touched + passes * shape["draft_bytes"]
        mixed_bytes = shape["fixed_bytes"] + shape["expert_bytes"] + passes * shape["draft_bytes"]
    decode_flops = decode * (per_token + shape["flops_per_logit"]) + positions * seqs * shape["draft_flops"]
    step_bytes = (1.0 - mixed) * decode_bytes + mixed * mixed_bytes
    flops_s = steps_s * (decode_flops + prefill * per_token)
    mul = min(1.0, flops_s / (peak_tflops * 1e12))
    mem = min(1.0 - mul, steps_s * step_bytes / (peak_gbps * 1e9))
    n = cuda_cores
    multiplying, memory = round(n * mul), round(n * mem)

    # THE TWO KINDS OF STEP, each on its own: what the cores do DURING a decode step and DURING a step
    # that carries a prefill chunk, and the share of the wall clock each kind takes. Step times are the
    # fit's (the model's own prior when the probes gave none). Steps that would add up to MORE than the
    # wall clock are scaled down to it; steps that add up to less leave the rest IDLE — a quiet engine
    # completes few steps a second, and reading that as long steps would invent work nobody did.
    a, b = shape["step_fit"] if fit is STEP_FIT else fit
    t_dec, t_mix = a + b * decode, a + b * (decode + chunk)
    busy = steps_s * ((1.0 - mixed) * t_dec + mixed * t_mix)
    scale = 1.0 / busy if busy > 1.0 else 1.0

    def case(flops: float, bytes_: float, seconds: float, share: float) -> dict:
        m = min(1.0, flops / seconds / (peak_tflops * 1e12))
        r = min(1.0 - m, bytes_ / seconds / (peak_gbps * 1e9))
        cm, cr = round(n * m), round(n * r)
        return {"multiplying": cm, "memory": cr, "other": n - cm - cr, "step_ms": round(seconds * 1e3),
                "time_share": round(share, 3)}

    # BOTH kinds always: a kind that did not run in the window is what its step WOULD look like, at
    # 0 % of the time — a grid that vanished and came back with the load read as a broken page.
    cases = {"decode": case(decode_flops, decode_bytes, t_dec * scale,
                            (1.0 - mixed) * t_dec * scale * steps_s),
             "prefill": (case(decode_flops + chunk * per_token, mixed_bytes, t_mix * scale,
                              mixed * t_mix * scale * steps_s) if budget else None)}
    return {"cores": n, "cases": cases,
            "multiplying": multiplying, "memory": memory, "other": n - multiplying - memory,
            "tflops": round(flops_s / 1e12, 2), "gbps": round(steps_s * step_bytes / 1e9, 1),
            # WHAT `gbps` and the memory share count: the weights alone, or every byte a pass is modeled
            # to move (`traffic`). The page names which in the tile's hover.
            "bytes": "modeled" if traffic is not None else "weights",
            "steps_s": round(steps_s, 2), "tokens_step": round(tokens_step, 1),
            "mixed_share": round(mixed, 3), "idle_share": round(max(0.0, 1.0 - min(1.0, busy)), 3)}


def _from_window(shape: dict, m: dict, traffic, *, cuda_cores: int, peak_tflops, peak_gbps, fit) -> dict | None:
    """The tile for an SGLang generation, from `modeled_flops`'s window `m`: the same FLOPs, the same passes,
    the same seconds — so its multiplying cores ARE the COMPUTE tile's FLOP/s over the peak. A decode pass is
    priced by `StepTraffic.decode_pass_bytes` (the MEMORY BANDWIDTH tile's model) and `verify_flops`; a
    prefill pass by its weights (that tile's floor) and the window's own prefill FLOPs a pass. A kind that did
    not run is what its pass WOULD look like at 0 % of the time: decode at the window's gauges, prefill at the
    engine's own mean pass since it started, each on the step-time prior. None while `m` has no figure."""
    nums = (m.get("tflops_per_gpu"), m.get("window_s"), peak_tflops, peak_gbps)
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in nums):
        return None
    if m["window_s"] <= 0 or peak_tflops <= 0 or peak_gbps <= 0:
        return None
    n, win = cuda_cores, float(m["window_s"])
    n_d, n_p = float(m.get("decode_passes") or 0.0), float(m.get("prefill_passes") or 0.0)
    seqs, ctx, width = max(1.0, float(m.get("seqs") or 0.0)), float(m.get("context_tokens") or 0.0), m.get("verify_width")
    drafting = isinstance(width, (int, float)) and width >= 2
    w = float(width) if drafting else 1.0
    a, b = shape["step_fit"] if fit is STEP_FIT else fit
    decode_bytes = sum(traffic.decode_pass_bytes(context=ctx, seqs=seqs, verify_width=width))
    prefill_bytes = traffic.prefill_bytes(m.get("prefill_chunk"))

    def case(flops: float, bytes_: float, seconds: float, share: float) -> dict:
        mul = min(1.0, flops / seconds / (peak_tflops * 1e12))
        mem = min(1.0 - mul, bytes_ / seconds / (peak_gbps * 1e9))
        cm, cr = round(n * mul), round(n * mem)
        return {"multiplying": cm, "memory": cr, "other": n - cm - cr, "step_ms": round(seconds * 1e3),
                "time_share": round(min(1.0, share), 3)}

    if n_d:
        decode = case(m["decode_flops"] / n_d, decode_bytes, m["decode_s"] / n_d, m["decode_s"] / win)
    else:
        decode = case(verify_flops(shape, context=ctx, seqs=seqs, width=width), decode_bytes, a + b * seqs * w, 0.0)
    chunk, depth = m.get("prefill_chunk"), m.get("prefill_depth") or 0.0
    if n_p:
        prefill = case(m["prefill_flops"] / n_p, prefill_bytes, m["prefill_s"] / n_p, m["prefill_s"] / win)
    elif chunk:
        prefill = case(_prefill_flops(shape, chunk, chunk * depth), prefill_bytes, a + b * chunk, 0.0)
    else:
        prefill = None
    mul = min(1.0, m["tflops_per_gpu"] / peak_tflops)
    gbps = (n_d * decode_bytes + n_p * prefill_bytes) / win / 1e9
    mem = min(1.0 - mul, gbps / peak_gbps)
    multiplying, memory = round(n * mul), round(n * mem)
    passes = n_d + n_p
    busy = (m.get("decode_s") or 0.0) + (m.get("prefill_s") or 0.0)
    return {"cores": n, "cases": {"decode": decode, "prefill": prefill},
            "multiplying": multiplying, "memory": memory, "other": n - multiplying - memory,
            "tflops": round(m["tflops_per_gpu"], 2), "gbps": round(gbps, 1), "bytes": "modeled",
            # WHAT `tflops` IS: the modeled FLOPs of the window COMPUTE draws (`modeled_flops`), not a model of
            # its own — the page says so in the hover.
            "flops": "modeled", "window_s": round(win, 1),
            "steps_s": round(passes / win, 2),
            "tokens_step": round((n_d * seqs * w + (m.get("prefill_tokens") or 0.0)) / passes, 1) if passes else 0.0,
            "mixed_share": round(n_p / passes, 3) if passes else 0.0,
            "idle_share": round(max(0.0, 1.0 - busy / win), 3)}
