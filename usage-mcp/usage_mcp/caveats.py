"""Numbers on this fleet mean different things depending on what produced them.

Two of those were measured on 2026-10-02/03 and are large enough to invert a conclusion, so they
travel attached to the figures rather than living in a document the caller has not read:

1. **ITL is per TOKEN on SGLang and per speculative STEP on vLLM.** Quoting a vLLM engine's
   `inter_token_latency_seconds` raw overstated perceived latency by 3.2x
   (worklog/sglang-27b-tp2-lb.md). Since 2026-10-03 the LB's scrape states it outright, as
   `itl_basis` with the values "token" and "step" (`vllm_lb/probe.py`), and this module uses the
   same two strings so the vocabularies cannot drift apart. The field is NOT yet persisted into the
   perf windows or published in `/lb/status`, so for recorded data `itl_semantics()` also *measures*
   it from the window: `gen_tokens / itl.n` is 1.00 when each sample is a token and ~3 when each
   sample is a step. Over the real logs that ratio is 1.004 for qwen3.8-27b-sglang-dflash2 and
   2.91-3.36 for the three vLLM model choices. When both are available they are cross-checked and a
   disagreement is reported rather than resolved silently.
2. **One swarm session's throughput is not a configuration's throughput.** Mean concurrency across
   the recorded deep-research sessions spans 2.36 to 11.64 and throughput 47.8 to 130.4 tok/s - a
   2.7x spread, larger than any configuration effect ever measured here (the biggest was +8.4%).
   worklog/dflash2-27b-tp2-tuning.md.
"""
from __future__ import annotations

from typing import Any

#: Below this, each ITL sample is one token; above it, one speculative step carrying several.
PER_TOKEN_MAX_RATIO = 1.05

#: The LB's own vocabulary (`vllm_lb/probe.py`: `ITL_PER_TOKEN, ITL_PER_STEP = "token", "step"`).
ITL_PER_TOKEN = "token"
ITL_PER_STEP = "step"
ITL_UNKNOWN = "unknown"
ITL_BASIS_VALUES = (ITL_PER_TOKEN, ITL_PER_STEP)

#: An impossible prefix-cache share, and why a recorded window can carry one.
PREFIX_RATE_DEFECT = (
    "This selection's prefix_hit_tokens EXCEED its prefix_query_tokens, which cannot be a share, "
    "so no hit rate is reported for it. Cause: until lb 9c32807 (2026-10-03) the probe recorded "
    "SGLang's prefill_effective_tokens_total{mode=\"input\"} as the queried total, and that series "
    "EXCLUDES the cache hits - it equals the COMPUTED tokens. Windows recorded before that fix "
    "therefore have no queried total to divide by. The raw token counts are returned as recorded; "
    "do not derive a share from them without knowing which side of the fix a window is on."
)

SWARM_SPREAD = (
    "A benchmark swarm session's mean concurrency has ranged 2.36-11.64 and its throughput "
    "47.8-130.4 tok/s across the recorded sessions - a 2.7x spread, wider than any configuration "
    "effect measured on this fleet (largest +8.4%). One session's number is NOT its "
    "configuration's number; comparing configurations needs several sessions each "
    "(worklog/dflash2-27b-tp2-tuning.md)."
)

PREFIX_CACHE = (
    "prompt_tokens is what clients sent; prompt_computed_tokens is what the engine actually "
    "prefilled. The gap is the prefix cache, so a high prompt_tok_s with a high prefix_hit_rate is "
    "cache replay, not prefill throughput."
)

MFU_PHANTOM = (
    "tflops_per_gpu / gbps_per_gpu come from the engine's own estimator, which on this fleet counts "
    "48 dense full-context attention layers: ~87% of the FLOPs and ~53% of the bytes are phantom. "
    "Treat them as trend only, never as utilisation."
)

NOT_ANTHROPIC = (
    "These are tokens the LOCAL two-box gateway served. This fleet records no Anthropic-account "
    "usage at all - see the usage_data_inventory tool."
)


def itl_semantics(gen_tokens: float, itl_n: float,
                  basis: str | None = None) -> dict[str, Any]:
    """What one ITL sample counts, and how to convert it to per-token.

    `basis` is the LB's own `itl_basis` ("token" / "step") when a caller has it; it is authoritative
    and is reported as `stated`. The ratio `gen_tokens / itl.n` is computed either way, because the
    field is not in the recorded windows and because it independently corroborates the label - 1.00
    when each sample is a token, ~3 when each sample is a step. If the two disagree, both are
    reported and `disagreement` says so; nothing is quietly picked.
    """
    stated = basis if basis in ITL_BASIS_VALUES else None
    if not itl_n or not gen_tokens or itl_n <= 0 or gen_tokens <= 0:
        return {
            "means": stated or ITL_UNKNOWN,
            "stated_by_the_engine": stated,
            "tokens_per_sample": None,
            "per_token_divisor": 1.0 if stated == ITL_PER_TOKEN else None,
            "disagreement": None,
            "note": "no ITL samples in this selection - the engine publishes the histogram only "
                    "once it has served a token interval"
                    + (f"; the engine states itl_basis={stated!r}" if stated else ""),
        }
    ratio = gen_tokens / itl_n
    measured = ITL_PER_TOKEN if ratio <= PER_TOKEN_MAX_RATIO else ITL_PER_STEP
    means = stated or measured
    per_token = means == ITL_PER_TOKEN
    disagreement = None
    if stated and stated != measured:
        disagreement = (f"the engine states itl_basis={stated!r} but gen_tokens / itl.n = "
                        f"{ratio:.3f} implies {measured!r}; the stated basis is used and this "
                        "mismatch is worth chasing before quoting either")
    return {
        "means": means,
        "stated_by_the_engine": stated,
        "tokens_per_sample": round(ratio, 3),
        "per_token_divisor": 1.0 if per_token else round(ratio, 3),
        "disagreement": disagreement,
        "note": (
            ("stated by the engine as itl_basis='token'; " if stated else "")
            + "measured: gen_tokens / itl.n = %.3f, so each ITL sample is one TOKEN and the raw "
              "figure is already the user-visible interval" % ratio
            if per_token else
            ("stated by the engine as itl_basis='step'; " if stated else "")
            + "measured: gen_tokens / itl.n = %.3f, so each ITL sample is one speculative STEP "
              "carrying that many tokens - divide the raw figure by it before quoting a per-token "
              "latency, and never compare it raw against an engine whose samples are per token"
              % ratio
        ),
    }


def engine_of(model_choice: str | None, catalog_engines: dict[str, str] | None = None) -> str:
    """The engine a model choice runs on: the catalog's own `launch.cluster.engine`, else vllm.

    The LB's models view fills the same default in, so a catalog entry without the key is vLLM.
    """
    if not model_choice:
        return "unknown"
    if catalog_engines and model_choice in catalog_engines:
        return catalog_engines[model_choice]
    return "unknown"
