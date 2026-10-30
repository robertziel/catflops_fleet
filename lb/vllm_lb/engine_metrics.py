"""The engines' metric VOCABULARIES — one definition of what each inference engine's `/metrics` says.

Pure and stdlib-only: the domain of "which engine wrote this exposition, and what it means in the LB's
canonical keys" (`SCRAPE_KEYS`), apart from the transport that fetches it (probe.py) and from every
reader of the result. vLLM, SGLang, TensorFold and the transcription service each speak their own
families; each has ONE parser here, and `parse_metrics` picks it off the text itself (`engine_of`).

Two readers share it: the LB's probe (vllm_lb/probe.py, which re-exports these names) and the swarm's
session (lb/bench_session.py, inside the SEALED runner image, which therefore cannot import config or
state — hence nothing here does). A new engine is a new vocabulary row and parser, never a second list
of prefixes in a caller (the swarm kept its own `("vllm:", "sglang:")` and refused a live TensorFold
pair as "no model is loaded", 2026-10-29).
"""
from __future__ import annotations

import json
import math

_COUNT, _SUM, _GAUGE = "count", "sum", "gauge"

#: Exposed SAMPLE name -> (scrape key, the one label value it must carry, kind). EXACT names: a
#: `startswith` also matched `vllm:num_requests_waiting_by_reason`, which repeats the same figure
#: per reason — the LB reported wait=10 where vLLM said 5 — and the histogram
#: `vllm:iteration_tokens_total` exposes `…_total_sum`, which a prefix match adds to a counter.
#: Family names and meanings: worklog/vllm-gateway-throughput.md Q2.
SCRAPE_FAMILIES = {
    "vllm:num_requests_waiting": ("waiting", None, _COUNT),
    "vllm:num_requests_running": ("running", None, _COUNT),
    "vllm:generation_tokens_total": ("gen_tokens_total", None, _COUNT),
    "vllm:prompt_tokens_total": ("prompt_tokens_total", None, _COUNT),
    # Only locally computed prompt tokens are prefill work; cache hits are counted apart.
    "vllm:prompt_tokens_by_source_total": ("prompt_computed_total", ("source", "local_compute"), _COUNT),
    "vllm:prompt_tokens_cached_total": ("prompt_cached_total", None, _COUNT),
    # Set every engine step, even one with no request output: the one signal that moves mid-prefill.
    "vllm:kv_cache_usage_perc": ("kv_cache_usage", None, _GAUGE),
    "vllm:request_prefill_kv_computed_tokens_sum": ("prefill_computed_tokens_sum", None, _SUM),
    "vllm:request_prefill_time_seconds_sum": ("prefill_time_sum", None, _SUM),
    "vllm:request_prefill_time_seconds_count": ("prefill_requests_total", None, _COUNT),
    # Exposed only where speculative decoding is configured (the TP=2 cluster's MTP).
    "vllm:spec_decode_num_drafts_total": ("spec_drafts_total", None, _COUNT),
    "vllm:spec_decode_num_draft_tokens_total": ("spec_draft_tokens_total", None, _COUNT),
    "vllm:spec_decode_num_accepted_tokens_total": ("spec_accepted_total", None, _COUNT),
    # WHAT THE ENGINE ITSELF ESTIMATES it did, per GPU, every step (vllm/v1/metrics/perf.py): the only
    # honest source of a FLOP/s or a GB/s on this fleet, since nvidia-smi gives neither on a GB10.
    # vLLM REGISTERS these three whatever it was started with and only feeds them under
    # `--enable-mfu-metrics`, so "exposed" does not mean "measured" — throughput.py tells those apart.
    "vllm:estimated_flops_per_gpu_total": ("est_flops_total", None, _COUNT),
    "vllm:estimated_read_bytes_per_gpu_total": ("est_read_bytes_total", None, _COUNT),
    "vllm:estimated_write_bytes_per_gpu_total": ("est_write_bytes_total", None, _COUNT),
    # The process fingerprint: a new value is a restart even when the counters ended higher.
    "process_start_time_seconds": ("process_start_time", None, _GAUGE),
    # ── what the real-traffic performance log reads (perflog.py), off this same scrape ──
    # Every finished request, whatever its `finished_reason` (stop, length, abort, error): summed.
    "vllm:request_success_total": ("requests_finished_total", None, _COUNT),
    "vllm:num_preemptions_total": ("preemptions_total", None, _COUNT),
    # Both in TOKENS: queried at scheduling, and found in the prefix cache.
    "vllm:prefix_cache_queries_total": ("prefix_queries_total", None, _COUNT),
    "vllm:prefix_cache_hits_total": ("prefix_hits_total", None, _COUNT),
    # A histogram's sum and count give its MEAN over any two scrapes; its quantiles need the buckets
    # (HISTOGRAM_BUCKETS below).
    # THE WHOLE REQUEST, arrival to last byte: what a transcription's latency is (`throughput.Throughput.speech`).
    "vllm:e2e_request_latency_seconds_sum": ("request_seconds_sum", None, _SUM),
    "vllm:e2e_request_latency_seconds_count": ("request_seconds_count", None, _COUNT),
    "vllm:time_to_first_token_seconds_sum": ("ttft_sum", None, _SUM),
    "vllm:time_to_first_token_seconds_count": ("ttft_count", None, _COUNT),
    "vllm:inter_token_latency_seconds_sum": ("itl_sum", None, _SUM),
    "vllm:inter_token_latency_seconds_count": ("itl_count", None, _COUNT),
    # ENGINE STEPS and the tokens they carried (a histogram's count and sum): steps a second and
    # tokens a step over any two scrapes — what cores.py splits the GPU's time with.
    "vllm:iteration_tokens_total_count": ("steps_total", None, _COUNT),
    "vllm:iteration_tokens_total_sum": ("step_tokens_sum", None, _SUM),
    # ── A KV TIER IN THE HOST'S RAM (vLLM's OffloadingConnector, `--kv-offloading-backend native`) ──
    # Bytes moved RAM → GPU (a RELOAD of a prefix the GPU pool evicted) and GPU → RAM (a STORE), and the seconds
    # each transfer took by its own CUDA events (v1/kv_offload/cpu/gpu_worker.py:708-712 @2eb491e) — so bytes over
    # seconds is the copy's own speed. The flat names (distributed/kv_transfer/kv_connector/v1/offloading/
    # metrics.py:25-30): their `transfer_type` twins are deprecated (:132-151). A flat counter is CREATED at its
    # first increment (:410-423), so a tier that has moved nothing yet exposes none of the four: not exposed, never 0.
    "vllm:kv_offload_load_bytes_total": ("kv_offload_load_bytes_total", None, _COUNT),
    "vllm:kv_offload_load_time_total": ("kv_offload_load_seconds_total", None, _SUM),
    "vllm:kv_offload_store_bytes_total": ("kv_offload_store_bytes_total", None, _COUNT),
    "vllm:kv_offload_store_time_total": ("kv_offload_store_seconds_total", None, _SUM),
}

#: ONE FAMILY, A KEY PER LABEL VALUE: exposed sample name -> (label, {label value: scrape key}, kind). WHY A REQUEST
#: WAITS, in the engine's own words (vLLM 2eb491e `vllm:num_requests_waiting_by_reason`): `capacity` = waiting for
#: scheduling capacity (a seat, KV blocks, the step's token budget), `deferred` = held by a transient constraint (a
#: KV transfer — the RAM tier's reload — a LoRA budget, a blocked status); the two sum to `num_requests_waiting`. An
#: engine that does not publish the family reads None for both, never 0.
SCRAPE_BY_LABEL = {
    "vllm:num_requests_waiting_by_reason": ("reason", {"capacity": "waiting_capacity",
                                                       "deferred": "waiting_deferred"}, _COUNT),
    # THE RAM TIER'S TRANSFERS, second spelling (vLLM v0.31.1, the 5090 node, 2026-10-26): the flat counters above are
    # created at a tier's FIRST transfer, these `transfer_type` families at engine start — so before a first store only
    # these say the tier is EMPTY (0), where the flat ones say nothing. CPU_to_GPU a reload, GPU_to_CPU a store. Both
    # spell ONE counter (`SECOND_SPELLINGS`): read only where no flat sample filled the key, never summed with it.
    "vllm:kv_offload_total_bytes_total": ("transfer_type", {"CPU_to_GPU": "kv_offload_load_bytes_total",
                                                            "GPU_to_CPU": "kv_offload_store_bytes_total"}, _COUNT),
    "vllm:kv_offload_total_time_total": ("transfer_type", {"CPU_to_GPU": "kv_offload_load_seconds_total",
                                                           "GPU_to_CPU": "kv_offload_store_seconds_total"}, _SUM),
}

#: Families that are a SECOND SPELLING of a flat counter in SCRAPE_FAMILIES (the same count, another vLLM's name): read
#: into their keys only where the flat family wrote none — never added to it.
SECOND_SPELLINGS = frozenset({"vllm:kv_offload_total_bytes_total", "vllm:kv_offload_total_time_total"})

#: INFO METRICS: the figure is in the LABELS and the sample's value is a constant 1.0, so these cannot
#: go in SCRAPE_FAMILIES — there a label is a FILTER on which sample to read, and the value is the
#: reading. Exposed sample name -> {label name: (scrape key, kind)}. Same pass, same `bad` discipline:
#: a label that is not a finite non-negative number costs its key for this scrape, never the scrape.
#:
#: WHY THE KV POOL IS READ RATHER THAN ASSUMED: the pin is an operator setting now
#: (`cluster.KV_CACHE_MEMORY_RANGE`), so a figure derived from a hardcoded 28 GB is wrong the moment
#: anyone changes it — silently, which is the worst way. The engine already publishes what it really
#: allocated; a recorded minute carries it (perflog) and the page reads it from there.
SCRAPE_INFO_LABELS = {
    "vllm:cache_config_info": {
        "kv_cache_size_tokens": ("kv_pool_tokens", _COUNT),
        "kv_cache_memory_bytes": ("kv_pool_bytes", _COUNT),
        "kv_cache_max_concurrency": ("kv_max_concurrency", _GAUGE),
        # The pool in BLOCKS, and the tokens one block holds once the engine resolved it: the seats a pool
        # holds are blocks over blocks-per-request (cluster_defaults.seats_for), the engine's own rule.
        "num_gpu_blocks": ("kv_pool_blocks", _COUNT),
        "block_size": ("kv_block_tokens", _COUNT),
        # THE GPU MEMORY FRACTION it was started with (--gpu-memory-utilization): what sized an unpinned pool, and
        # the one launch setting a node this LB only observes publishes about its memory (topology.py).
        "gpu_memory_utilization": ("gpu_memory_utilization", _GAUGE),
        # THE KV TIER IN THE HOST'S RAM, in GiB (`--kv-offloading-size`: the whole buffer, summed over TP ranks,
        # config/cache.py:266-270 @2eb491e) — "None" where none runs, which costs this key alone. Its backend is the
        # word below: only "native" makes it this engine's own RAM (config/vllm.py:1106-1127).
        "kv_offloading_size": ("kv_offload_ram_gib", _GAUGE),
    },
}

#: Not a figure but the engine's own WORD for how it runs, carried verbatim — never turned into a number or guessed
#: when absent. `_WORD` passes the label through; `_FLAG` reads Python's own True/False (vLLM writes its bools that
#: way) and nothing else.
_WORD, _FLAG = "word", "flag"

#: INFO LABELS THAT ARE WORDS: exposed sample name -> {label: (scrape key, kind)}. A node's KV cache dtype and its
#: prefix caching are settings the panel shows for a node it only observes, and the engine says both here.
SCRAPE_INFO_WORDS = {
    "vllm:cache_config_info": {
        "cache_dtype": ("kv_cache_dtype", _WORD),
        "enable_prefix_caching": ("prefix_caching", _FLAG),
        # Printed whether a tier runs or not ("native" is vLLM's default word): it means something only beside a size.
        "kv_offloading_backend": ("kv_offload_backend", _WORD),
    },
}

#: Histograms whose BUCKETS a quantile needs (the performance log's TTFT and inter-token p90): exposed
#: sample name -> scrape key. Each parses to ((le, cumulative count), ...) ascending, `+Inf` last as
#: `inf`, summed across label sets — or None when the family is not exposed or any of its lines could
#: not be read, exactly as a scalar family. Never serialised as-is: `inf` is not JSON.
HISTOGRAM_BUCKETS = {
    "vllm:time_to_first_token_seconds_bucket": "ttft_buckets",
    "vllm:inter_token_latency_seconds_bucket": "itl_buckets",
}

#: A histogram's parsed buckets, which are never int-converted.
_BUCKETS = "buckets"
#: Not a reading at all but the CONVENTION behind one: a short string, passed through verbatim.
_BASIS = "basis"

#: WHAT ONE INTER-TOKEN-LATENCY SAMPLE COVERS, carried in the scrape beside `itl_sum` / `itl_count`
#: so nothing downstream can pool two engines' gaps as though they were the same quantity. The two
#: vocabularies use the same family NAME for different things, and MEASURED on real captures:
#:
#:   vLLM    one sample per ENGINE STEP. `vllm:inter_token_latency_seconds_count` is EXACTLY
#:           `vllm:spec_decode_num_drafts_total` in both reads of the captured TP=2 head
#:           (fixtures/throughput/vllm-v0.29.0-cluster-head-{1,2}.metrics: 7 667 == 7 667 and
#:           7 864 == 7 864), which is 0.324 of its 23 656 generated tokens — i.e. 1/accept_length,
#:           because a verify step emits `accept_length` tokens behind ONE recorded gap.
#:   SGLang  one sample per EMITTED TOKEN. `sglang:inter_token_latency_seconds_count` is 169 554
#:           against 169 889 decode tokens (0.998) and 56 122 verify steps (3.02x) in
#:           fixtures/sglang-metrics-2026-10-03-cached-b.txt.
#:
#: So a mean ITL (`itl_sum / itl_count`) compared across the two engines is wrong by the accept
#: length — ~3x on this fleet — in opposite directions, and a consumer that pools them silently
#: reports the SGLang pair as three times the faster per gap at the same tok/s. Normalising here
#: would destroy the engine's own histogram BUCKETS, which are the p90's only source and cannot be
#: rescaled after the fact, so the value stays as published and the basis travels with it. The
#: CONSUMERS are where the two are told apart — `perflog.COUNTERS` does not record this key, so a
#: recorded window still carries no basis and /benchmarks still labels both axes the same way:
#: that fix is perflog's and benchmarks.js's, not the probe's.
ITL_PER_TOKEN, ITL_PER_STEP = "token", "step"

#: The ITL figures the basis describes. It is published only where one of them was really read: a
#: basis with no value behind it would claim there is an inter-token latency to describe.
_ITL_KEYS = ("itl_sum", "itl_count", "itl_buckets")

#: THE INPUTS OF THE MODELED MEMORY BANDWIDTH (`throughput.Throughput.bandwidth_model`), which only SGLang
#: feeds — so they are null on every vLLM scrape, and vLLM's figures never read them. Where each is read
#: from, and why that family and not another, is at its mapping in `SGLANG_FAMILIES`.
#:
#:   decode_steps_total     decode FORWARD PASSES (`steps_total` holds them and the prefill ones)
#:   decode_context_tokens  the context tokens summed over the running decode batch — what attention reads
#:   spec_verify_width      the tokens a verify carries per sequence, bonus included; 0 = no drafter
#:   busy_seconds_total     the wall clock the scheduler spent in forward passes — the modeled COMPUTE's
#:                          time (cores.py `modeled_flops`)
MODEL_INPUT_KEYS: tuple[tuple[str, str], ...] = (
    ("decode_steps_total", _COUNT), ("decode_context_tokens", _COUNT), ("spec_verify_width", _GAUGE),
    ("busy_seconds_total", _SUM))

#: THE INPUT OF A POOL AN ENGINE RESERVES PER SEAT (TensorFold): its live streams' fills of their own windows, summed
#: — tokens held in units of one window. Over the seats (`with_health`) it IS the share of the pool in use. Null on
#: every engine that reports its pool's use itself.
POOL_INPUT_KEYS: tuple[tuple[str, str], ...] = (("kv_stream_fill_sum", _GAUGE),)

#: THE SPEECH FIGURES only a transcription service publishes (`TRANSCRIPTION_FAMILIES`): the seats and the queue it
#: admits to, its answers by outcome, the audio it transcribed, and the GPU memory its process holds.
SPEECH_KEYS: tuple[tuple[str, str], ...] = (
    ("seats", _COUNT), ("queue_capacity", _COUNT), ("requests_silence_total", _COUNT),
    ("requests_failed_total", _COUNT), ("requests_rejected_total", _COUNT), ("audio_seconds_total", _SUM),
    ("gpu_memory_bytes", _COUNT))

#: EVERY canonical scrape key, with the kind that decides how it is rendered. The ONE shape a scrape
#: has, whatever engine it was read from: both parsers build their result from this, so no consumer
#: can tell the two apart by the keys it gets, and a key added here reaches both at once.
SCRAPE_KEYS: tuple[tuple[str, str], ...] = (
    *((key, kind) for key, _, kind in SCRAPE_FAMILIES.values()),
    *((key, kind) for _, keys, kind in SCRAPE_BY_LABEL.values() for key in keys.values()),
    *((key, kind) for fields in SCRAPE_INFO_LABELS.values() for key, kind in fields.values()),
    *((key, kind) for fields in SCRAPE_INFO_WORDS.values() for key, kind in fields.values()),
    *((key, _BUCKETS) for key in HISTOGRAM_BUCKETS.values()),
    *MODEL_INPUT_KEYS,
    *POOL_INPUT_KEYS,
    *SPEECH_KEYS,
    # The keys that are not figures: which convention produced `itl_sum` / `itl_count` above, and WHICH ENGINE
    # wrote the exposition — its vocabulary (`vllm` / `sglang`), as the server itself says what it is: a node this
    # LB only observes was launched by nothing that could tell it.
    ("itl_basis", _BASIS),
    ("engine", _BASIS),
)

#: The engine each parser reads, named as `engine` says it — the transcription service's as its node engine is named
#: (fleet.ENGINES).
ENGINE_VLLM, ENGINE_SGLANG, ENGINE_TRANSCRIPTION = "vllm", "sglang", "faster-whisper"
ENGINE_TENSORFOLD = "tensorfold"


def _word(kind: str, raw: str):
    """One info label as the scrape carries it: the word as written (`_WORD`), or True/False (`_FLAG`, Python's own
    spelling and nothing else — a word that is neither is no reading)."""
    if kind == _FLAG:
        return {"True": True, "False": False}.get(raw)
    return raw if raw.strip() else None


def _itl_basis(out: dict, basis: str) -> str | None:
    """`ITL_PER_TOKEN` or `ITL_PER_STEP` for a scrape that carried an inter-token latency, None for
    one that did not — so the basis never outlives the value it describes, and an empty scrape stays
    entirely empty."""
    return basis if any(out.get(key) is not None for key in _ITL_KEYS) else None


# ────────────────────────────────── SGLang ──────────────────────────────────
#
# SGLang serves the same OpenAI HTTP API as vLLM — `/health`, `/v1/models` with `max_model_len`, and
# `/metrics` — but exposes an ENTIRELY DIFFERENT metrics vocabulary. The rest of the LB reads the
# canonical `vllm:*` keys above across throughput.py, saturation.py, state.py, cluster_defaults.py,
# cores.py and perflog.py, so the second vocabulary is normalised HERE, at the boundary, and nothing
# downstream learns a second set of names.
#
# EVERY MAPPING BELOW WAS READ OFF A REAL SCRAPE of the SGLang + DFlash2 TP=2 pair that served
# qwen38-27b-sglang-tp2 on the two GB10 boxes at 16 seats on 2026-10-02 (box-a:8020), kept as
# tests/fixtures/sglang-metrics-2026-10-02-{busy-a,busy-b,idle}.txt. busy-a and busy-b are the same
# engine 62 s apart, so the counters in them have a real delta to reason from. Lines marked
# UNVERIFIED are the ones that capture could not settle, because the figure was zero throughout.
#
# WHERE SGLANG HAS NO EQUIVALENT THIS PUBLISHES NOTHING — never a zero. A fabricated 0.0 is the bug
# class worklog/gateway-figures-audit.md finding 14 calls out: the page cannot tell an invented zero
# from a measured one, so an absent figure must stay null and read as "not exposed".

#: Labels that REPLICATE a figure rather than partition it. SGLang tags its scheduler families with
#: the rank that published them, and on a TP=2 pair every rank reports the SAME scheduler state — so
#: summing across them would report 32 running on a 16-seat engine and double every token counter.
#: The adapter keeps the lowest rank it sees instead (`_replica`). Ordered by significance, not
#: alphabetically, so the selected group is rank 0's and not merely the lowest `pid`.
#:
#: The captured pair's HTTP head published rank 0 only (`tp_rank="0"` on all 201 of its per-rank
#: series, no `tp_rank="1"` anywhere), so the guard is not exercised by the real sample — a test
#: relabels the capture's own rank-0 lines to rank 1 to exercise it.
SGLANG_REPLICA_LABELS: tuple[str, ...] = ("tp_rank", "pp_rank", "moe_ep_rank", "pid")

#: Keys read only to DERIVE a canonical one (`_sglang_derived`); stripped from the result. The verify width
#: is derived from AND published: the modeled bandwidth reads it as it stands (`MODEL_INPUT_KEYS`).
_KV_GIB, _GEN_FINISHED = "~kv_gib", "~gen_finished"
_SPEC_WIDTH = "spec_verify_width"

#: Prefix-cache hits, by where the reused pages were found.
_HIT_MODES = frozenset({"device_hit", "host_hit", "storage_hit"})

#: EVERY mode of `sglang:prefill_effective_tokens_total`, which is the BILLED prompt: what the
#: requests brought, whether a layer computed it or the prefix cache served it. `mode="input"` is
#: only the computed part — see the family's own note below — so the hit modes belong in this sum and
#: not beside it.
#:
#: THE HIT MODES ARE A SUBSET OF THIS SET BY CONSTRUCTION, which is what makes a prefix hit rate
#: outside [0,1] impossible rather than merely unobserved: `prefix_hits_total` reads `_HIT_MODES` and
#: its denominator `prefix_queries_total` reads this one.
_PROMPT_MODES = _HIT_MODES | frozenset({"input"})

#: The modes of `sglang:realtime_tokens_total` whose tokens a forward pass really PROCESSED — the
#: CUDA CORES tile's numerator. `prefill_cache` is the third mode and is reused KV pages: no FLOPs,
#: no weight read, not a token any layer saw.
_STEP_MODES = frozenset({"prefill_compute", "decode"})

#: Exposed SGLang sample name -> the canonical keys it feeds, each ((label, accepted values) | None,
#: kind). One name may feed several keys, and a label filter picks which of its series does.
SGLANG_FAMILIES: dict[str, tuple[tuple[str, tuple[str, frozenset[str]] | None, str], ...]] = {
    # ── 1. the scheduler's depth: drawn on the page, and what the saturation logic reads ──
    # `sglang:num_queue_reqs` is the waiting queue. SGLang also has a grammar queue and four
    # disaggregated-serving queues; all six were 0 through every capture, and whether they overlap
    # num_queue_reqs is UNVERIFIED, so adding them would be a guess. Only the plain queue is read.
    "sglang:num_queue_reqs": (("waiting", None, _COUNT),),
    "sglang:num_running_reqs": (("running", None, _COUNT),),
    # The whole request, arrival to last byte, summed over `is_streaming` (2026-10-02 busy-b: 1 + 24 requests).
    "sglang:e2e_request_latency_seconds_sum": (("request_seconds_sum", None, _SUM),),

    # ── 2. the token counters every throughput figure is taken from ──
    # WHICH TOKEN COUNTER, AND WHY IT MATTERS: SGLang publishes two.
    # `sglang:generation_tokens_total` is incremented when a REQUEST FINISHES — across busy-a and
    # busy-b, 62 s in which this engine emitted 19 138 tokens, it does not move at all, because the
    # 16 in-flight requests had not returned yet. `sglang:realtime_tokens_total` is the scheduler's
    # own per-step counter and moved by exactly those 19 138. A rate must be taken over the one that
    # advances while the work is happening, or every throughput figure reads 0 mid-generation and
    # then spikes when a batch completes.
    "sglang:realtime_tokens_total": (
        ("gen_tokens_total", ("mode", frozenset({"decode"})), _COUNT),
        ("prompt_computed_total", ("mode", frozenset({"prefill_compute"})), _COUNT),
        ("prefill_computed_tokens_sum", ("mode", frozenset({"prefill_compute"})), _SUM),
        # ── 3. the tokens the engine's steps carried: the CUDA CORES tile's numerator (cores.py) ──
        # ONLY WHAT A FORWARD PASS PROCESSED. This filter was `None` — every mode — and the live
        # scrape has a THIRD mode the two captures below never exposed, `prefill_cache`, which is
        # reused KV pages at zero FLOPs. Over the 83 s between the two 2026-10-03 captures it was 86 %
        # of the sum: 239 554 tokens over 731 steps = 327.7 a step, against 19 066 / 731 = 26.1 really
        # processed, a x12.56 numerator for cores.py and so for the TFLOP/s and cores figures it draws.
        # vLLM's `vllm:iteration_tokens_total_sum` carries no cache hits, so this was an engine
        # assumption and not a shared one. The family's HELP line names all three
        # ("mode: prefill_compute, prefill_cache, decode"); only the two here were ever processed.
        ("step_tokens_sum", ("mode", _STEP_MODES), _SUM),
    ),
    # THE BILLED PROMPT AND THE PART OF IT THE CACHE SERVED. `mode="input"` is the COMPUTED prompt,
    # not the billed one — MEASURED now, no longer assumed, on a pair whose prefix cache was really
    # hitting (fixtures/sglang-metrics-2026-10-03-cached-{a,b}.txt, read off the serving head at
    # :8100 while generation 123 ran; the three 2026-10-02 captures could not settle it because every
    # hit mode in them is 0):
    #
    #     prefill_effective{mode="input"}      801 500  ==    realtime{mode="prefill_compute"}
    #     prefill_effective{mode="device_hit"} 7 481 424  ==  realtime{mode="prefill_cache"}
    #
    # exact in both captures — and the family's own HELP line, present in every capture the repo has,
    # states the same rule in words: "Windowed prefix cache hit rate = rate(sum of *_hit) / rate(sum
    # of ALL MODES)". `input` EXCLUDES the hits, so the previous mapping — `prompt_tokens_total`
    # and `prefix_queries_total` both from `input` alone — made the hit rate hits/computed: 9.33
    # cumulative and 16.00 over that window, a "hit rate" of 1 600 %, and left perflog's
    # `computed = billed - hits` at a hard 0 in 31 of 50 recorded windows.
    #
    # So, mode by mode:
    #   input                         -> the COMPUTED prompt   (`prompt_computed_total`, taken off
    #                                    realtime's identical `prefill_compute` above)
    #   device_hit|host_hit|storage_hit -> what the CACHE SERVED (`prompt_cached_total`, `prefix_hits_total`)
    #   all four (`_PROMPT_MODES`)    -> what the REQUESTS BROUGHT, which is the billed prompt and the
    #                                    only honest hit-rate denominator
    # and because `_HIT_MODES` is a subset of `_PROMPT_MODES`, hits/queries cannot leave [0,1].
    "sglang:prefill_effective_tokens_total": (
        ("prompt_tokens_total", ("mode", _PROMPT_MODES), _COUNT),
        ("prefix_queries_total", ("mode", _PROMPT_MODES), _COUNT),
        ("prompt_cached_total", ("mode", _HIT_MODES), _COUNT),
        ("prefix_hits_total", ("mode", _HIT_MODES), _COUNT),
    ),

    # ── 3. ENGINE STEPS: what cores.py splits the GPU's time with ──
    # One forward pass per scheduler iteration, labelled by whether a CUDA graph was used
    # (`decode_cuda_graph`, `prefill_none`), so every mode summed is every step.
    # PROVED A STEP IS AN ITERATION, NOT A FORWARD: busy-a -> busy-b is 477 passes carrying 19 138
    # decode tokens over 16 running sequences = 2.51 tokens per sequence per pass, which is this
    # engine's own `spec_accept_length` (2.56). Were a draft forward counted separately the figure
    # would be 5.0, above the 4 draft tokens a step can even propose.
    "sglang:cuda_graph_passes_total": (
        ("steps_total", None, _COUNT),
        # DECODE PASSES ALONE, the modeled bandwidth's pass rate. +1 per decode batch result, with or without
        # a drafter (batch_result_processor.py:929-932 `increment_decode_cuda_graph_pass`), +1 per prefill
        # batch in the `prefill_*` modes (metrics_reporter.py:726-729) — one forward each, counted when it
        # COMPLETES. `sglang:spec_verify_calls_total` is not a pass rate: it is added to when a REQUEST
        # finishes, by that request's own verify count (metrics_collector.py:1803-1804, the count bumped per
        # verify at batch_result_processor.py:773), and stood at 3,730 through all three 2026-10-02 captures
        # while 477 decode passes ran between busy-a and busy-b.
        ("decode_steps_total", ("mode", frozenset({"decode_cuda_graph", "decode_none"})), _COUNT),
    ),
    # THE CONTEXT ATTENTION READS: `_decode_total_seq_lens(batch)`, the sum of the running decode batch's
    # sequence lengths (metrics_reporter.py:84-88), set at each decode log interval — every 40th decode pass
    # (metrics_reporter.py:1005, `decode_log_interval` 40) — so it lags the batch by at most that many passes.
    # Not the KV pool's occupancy: prefix sharing makes that the DEDUPLICATED context (16 % of the pool at
    # 1.35 M summed tokens, worklog/sglang-bottleneck-2026-10-03.md), while every sequence's attention reads
    # its own whole context.
    "sglang:decode_sum_seq_lens": (("decode_context_tokens", None, _COUNT),),
    # THE TIME THE PASSES TOOK: the scheduler loop's wall clock in `run_batch`, the stage that launches a
    # forward pass and — with speculative decoding's syncs — waits it out (live 2026-10-03: 0.966 s a second
    # under load, every other stage 0.025; worklog/sglang-bottleneck-2026-10-03.md). The stages partition the
    # loop's time exclusively (observability/scheduler_stage_metrics.py `SchedulerStageMetricsRecorder`) and
    # are drained into the counter at a loop iteration at most once a second
    # (metrics_reporter.py:60 `_SCHEDULER_TIME_ACCOUNTING_INTERVAL_NS`, :1263-1296) — so a pass longer than
    # that is credited when it RETURNS, as `cuda_graph_passes_total` counts it: one 8,192-token chunk moved
    # it by 6.507 s and the pass counter by 1 between the two prefill captures, which were 5.005 s apart.
    # The modeled COMPUTE prices its prefill passes with it (cores.py `modeled_flops`).
    "sglang:scheduler_stage_seconds_total": (
        ("busy_seconds_total", ("category", frozenset({"run_batch"})), _SUM),),
    # ── THE COMPUTE AND MEMORY BANDWIDTH TILES ──
    # SGLang publishes the SAME three estimator counters vLLM does, under its own prefix. Read off the
    # PINNED IMAGE's own source (lmsysorg/sglang@sha256:00205b89..., an editable install at
    # /sgl-workspace/sglang/python, git 708f51e44bc6), not from documentation:
    #
    #   srt/observability/metrics_collector.py:947-970  `SchedulerMetricsCollector.__init__` constructs
    #       all three as prometheus_client `Counter`s, the `sglang:` prefix a literal in the name;
    #   srt/observability/metrics_collector.py:1344-1361  `increment_estimated_perf` — each counter
    #       `.inc()`s only when its own value is > 0;
    #   managers/scheduler_components/metrics_reporter.py:735-741 (prefill) and :840-851 (decode) call
    #       it, so BOTH phases feed them, every step.
    #
    # Raw FLOPs and raw BYTES — the engine's own log divides by 1e12 and 1e9 — so they land on the same
    # canonical keys as vLLM's identically-named counters, unconverted, and `throughput.py`'s divisors
    # stay the engine's own arithmetic.
    #
    # PER GPU AS NAMED, NOT AS A MEASUREMENT, and the two tiles are not equally safe. SGLang's estimator
    # (metrics_reporter.py:476-620) is an analytical PER-TOKEN model, biased the way vLLM's is — memory
    # note vllm-mfu-estimator-overcounts-attention — and on the 27B, read against its own constants:
    #
    #   * COMPUTE is in range. `_linear_flops_per_token` 3.892e10 at tp=2; at this pair's measured
    #     278.69 tok/s (see `sglang_cmd`'s note) the linear term is 10.8 TFLOP/s against a MEASURED
    #     nvfp4 peak of 323.2 TFLOP/s. Overcounted (all 64 hidden layers charged as attention though 48
    #     are GDN; the MLP not TP-sharded), but the same class of overcount the vLLM tile already draws.
    #
    #   * MEMORY BANDWIDTH IS NOT. It charges the full per-rank WEIGHT READ for every GENERATED TOKEN
    #     rather than once per step, and at bf16 itemsize whatever the checkpoint is: 3.892e10 B/token
    #     x 278.69 tok/s = 10 847 GB/s, which is 39.7x this pair's 273 GB/s DRAM limit. Its KV term is
    #     131 072 B/token where the real fp8 KV is 16 x 2 x 2 x 256 = 16 384 (8x). So once a pair starts
    #     with the switch on, `mem bw / GPU` will draw about ten thousand GB/s and the saturation verdict
    #     will read BANDWIDTH_BOUND off it — a visibly wrong page, not a cautious one.
    #
    # The mapping stays, because it is what the engine publishes and the keys mean "what the engine
    # ESTIMATES it did". Whether the page should DRAW the bytes side for an SGLang generation is a
    # consumer decision (throughput.py / saturation.py), it is not settled here, and nothing has drawn
    # it yet: no pair has been started with the switch.
    #
    # WHAT IT TAKES TO GET THEM. `--enable-mfu-metrics` (store_true, default False:
    # srt/arg_groups/fields/observability.py:85-88) AND `--enable-metrics`, which gates it
    # (metrics_reporter.py:253-254 sets the flag only inside `if self.enable_metrics:`).
    # `lb/cluster/tp-rank`'s `sglang_cmd` passes both on every SGLang start. Until it passed the first,
    # the two tiles read "not exposed by this server" on every SGLang generation — and the note under
    # SGLANG_UNAVAILABLE claimed SGLang had no counterpart to the MFU estimator at all, which the image
    # refutes. Only the head rank is scrapeable anyway (rank 1 runs no HTTP server), and only the
    # attn_tp_rank 0 scheduler emits without `--enable-metrics-for-all-schedulers`.
    #
    # ABSENT UNTIL THE FIRST POSITIVE INCREMENT, because prometheus_client renders a labelled Counter
    # only once a child exists: with the switch ON but before the first step, these keys are still null
    # and the page still says "not exposed" — transient, and indistinguishable from the switch being
    # off from the scrape alone. That is why `SGLANG_BEHIND_A_FLAG` is about the START, not the scrape.
    #
    # DO NOT give `http_get` an `Accept: application/openmetrics-text` header: under OpenMetrics the
    # SAMPLE names come back with the colon turned into an underscore (`sglang_estimated_flops_...`)
    # while HELP/TYPE keep the colon, and every name in this table would stop matching. The getter
    # sends only `Host`, which is what the captures were taken with.
    #
    # VERIFIED AGAINST A LIVE SCRAPE since 2026-10-03: the serving head's two reads kept as
    # tests/fixtures/sglang-metrics-2026-10-03-mfu-{a,b}.txt carry all three names. The older tests that
    # exercise them still build their own exposition (lb/tests/test_sglang_probe.py `_with_mfu`).
    "sglang:estimated_flops_per_gpu_total": (("est_flops_total", None, _COUNT),),
    # THE TWO BYTE COUNTERS ARE DELIBERATELY NOT MAPPED, though the engine publishes them and the flag
    # is on. At the pair's own measured 278.69 tok/s they work out to 10,847 GB/s a GPU — 39.7x the
    # 273 GB/s this hardware physically has — because SGLang charges the FULL per-rank weight read for
    # every generated token, at bf16 itemsize whatever the checkpoint stores, and its KV term is 8x the
    # real fp8 KV. The saturation verdict would then read BANDWIDTH_BOUND off a figure that cannot be
    # true. An estimator that overcounts within a plausible range is a documented bias and this gateway
    # already draws one (vLLM's, which counts 48 dense full-context attention layers); a number forty
    # times over a hard physical limit is not a bias, and a panel that has spent this long learning to
    # say "not exposed" rather than invent a figure should not start here.
    #
    # MEASURED LIVE, 2026-10-03 (fixtures/sglang-metrics-2026-10-03-mfu-{a,b}.txt): 3.70e14 B in 5.03 s,
    # 73,600 GB/s a GPU. The MEMORY BANDWIDTH tile draws the LB's OWN MODEL instead — each forward pass
    # priced by the catalog's `StepTraffic` at SGLang's pass counts and gauges (`MODEL_INPUT_KEYS`,
    # throughput.py `bandwidth_model`): 99.8 GB/s over those same two reads.
    #
    # To draw it anyway, add the two lines back and move their keys out of SGLANG_BEHIND_A_FLAG — the
    # mapping itself is correct, the arithmetic behind it is what is not.
    # "sglang:estimated_read_bytes_per_gpu_total": (("est_read_bytes_total", None, _COUNT),),
    # "sglang:estimated_write_bytes_per_gpu_total": (("est_write_bytes_total", None, _COUNT),),

    # ── 5a. the KV cache's own occupancy ──
    # `sglang:full_token_usage`, NOT `sglang:token_usage`. `token_usage` is the WORST of the engine's
    # pools, and DFlash2's draft state lives in a tiny mamba pool that read 0.8 while the KV cache was
    # 1.3 % full — reading it would put the KV tile at 80 % on an almost empty cache. In the capture
    # `full_token_usage` equals kv_used_tokens / max_total_num_tokens exactly (45072 / 3432455).
    "sglang:full_token_usage": (("kv_cache_usage", None, _GAUGE),),

    # ── 5b. latency, where SGLang's family names happen to match vLLM's ──
    "sglang:time_to_first_token_seconds_sum": (("ttft_sum", None, _SUM),),
    "sglang:time_to_first_token_seconds_count": (("ttft_count", None, _COUNT),),
    "sglang:inter_token_latency_seconds_sum": (("itl_sum", None, _SUM),),
    "sglang:inter_token_latency_seconds_count": (("itl_count", None, _COUNT),),
    # Every request that reached an end, whatever its reason — the analogue of
    # `vllm:request_success_total` summed over finish reasons. `sglang:num_requests_total` counts
    # ARRIVALS, so it is not this figure.
    "sglang:e2e_request_latency_seconds_count": (("requests_finished_total", None, _COUNT),
                                                 ("request_seconds_count", None, _COUNT)),
    # Prefill speed: the forward stage's own time and the requests through it.
    "sglang:per_stage_req_latency_seconds_sum": (
        ("prefill_time_sum", ("stage", frozenset({"prefill_forward"})), _SUM),),
    "sglang:per_stage_req_latency_seconds_count": (
        ("prefill_requests_total", ("stage", frozenset({"prefill_forward"})), _COUNT),),

    # ── the pool the engine really allocated (vLLM puts these in `vllm:cache_config_info`'s labels) ──
    "sglang:max_total_num_tokens": (("kv_pool_tokens", None, _COUNT),),
    "sglang:num_pages": (("kv_pool_blocks", None, _COUNT),),
    "sglang:page_size": (("kv_block_tokens", None, _COUNT),),
    "sglang:kv_cache_memory_usage_gb": ((_KV_GIB, None, _GAUGE),),

    # ── 4. speculative decoding ──
    # SGLANG'S OWN ARITHMETIC, confirmed to the last digit in all three captures:
    #     spec_accept_rate == (spec_accept_length - 1) / (spec_num_draft_tokens - 1)
    #   idle    (2.470125786163522 - 1) / 3 == 0.490041928721174
    #   busy-a  (2.6171875         - 1) / 3 == 0.5390625
    #   busy-b  (2.55625           - 1) / 3 == 0.51875
    # So a verify step proposes `spec_num_draft_tokens - 1` DRAFT tokens and always emits 1 bonus
    # token of the target's own. `spec_accept_length` and `spec_accept_rate` are instantaneous GAUGES
    # of the latest batch, so neither can become a cumulative counter — the canonical triple is built
    # from the two real counters instead, and both are COMPLETION-SIDE, which keeps them in the same
    # units as each other and as vLLM's.
    "sglang:spec_verify_calls_total": (("spec_drafts_total", None, _COUNT),),
    # THE VERIFY WIDTH, bonus included: `--speculative-num-draft-tokens`, to which SGLang also resolves the
    # DFlash2 drafter's block (dflash_worker_v2.py:340, logged 'block_size=4' over the checkpoint's 8, and
    # 'num_tokens_per_req=4' at the verify graph capture). 0 when no drafter runs (metrics_reporter.py:889).
    "sglang:spec_num_draft_tokens": ((_SPEC_WIDTH, None, _GAUGE),),
    "sglang:generation_tokens_total": ((_GEN_FINISHED, None, _COUNT),),
}

#: Its two latency histograms' buckets, for the performance log's p90s.
SGLANG_HISTOGRAM_BUCKETS = {
    "sglang:time_to_first_token_seconds_bucket": "ttft_buckets",
    "sglang:inter_token_latency_seconds_bucket": "itl_buckets",
}

#: Canonical key -> kind, for the families above.
_SGLANG_KINDS = {key: kind for entries in SGLANG_FAMILIES.values() for key, _, kind in entries}

#: WHAT AN SGLANG GENERATION WILL NOT SHOW, WHATEVER IT WAS STARTED WITH. Every canonical key with no
#: SGLang source at all, and why. These stay null, so the page says "not exposed" instead of drawing an
#: invented zero.
#:
#: A figure SGLang DOES publish but only behind a switch does not belong here, it belongs in
#: `SGLANG_BEHIND_A_FLAG`: this list is a property of the ENGINE and that one of HOW IT WAS STARTED, and
#: a reader asking "can the page ever show this?" gets opposite answers from the two. Keeping the MFU
#: estimators here — which is what this list did until the image was read — made the second question's
#: answer look like the first's.
SGLANG_UNAVAILABLE = {
    # The restart fingerprint. SGLang's metrics carry no epoch gauge — not even the prometheus_client
    # default collectors — only `sglang:startup_time_seconds`, which is a per-phase DURATION. So an
    # SGLang engine restart cannot be detected from its scrape, and a counter that restarts at 0
    # looks like a counter that went backwards.
    "process_start_time": "no epoch gauge; startup_time_seconds is a duration, not a start time",
    # `sglang:num_retracted_reqs` is a GAUGE of what is retracted RIGHT NOW, not a cumulative count,
    # so a delta of it would be meaningless. It was 0 in every capture (UNVERIFIED either way).
    "preemptions_total": "num_retracted_reqs is an instantaneous gauge, not a counter",
    # Only in `/get_server_info` (`max_running_requests`), which is not a metrics family.
    "kv_max_concurrency": "only in /get_server_info, not in /metrics",
    # How it was started — its server args (`mem_fraction_static`, `kv_cache_dtype`, `disable_radix_cache`) — is in
    # `/get_server_info` alone; no family in any capture under tests/fixtures/sglang-metrics-* carries it.
    "gpu_memory_utilization": "only in /get_server_info (mem_fraction_static), not in /metrics",
    "kv_cache_dtype": "only in /get_server_info (kv_cache_dtype), not in /metrics",
    "prefix_caching": "only in /get_server_info (disable_radix_cache), not in /metrics",
    # A KV TIER IN THE HOST'S RAM is vLLM's OffloadingConnector (`--kv-offloading-size`, `vllm:kv_offload_*`); no family
    # in any capture under tests/fixtures/sglang-metrics-* reports a host KV tier, and the pair runs none.
    **dict.fromkeys(("kv_offload_ram_gib", "kv_offload_backend", "kv_offload_load_bytes_total",
                     "kv_offload_load_seconds_total", "kv_offload_store_bytes_total", "kv_offload_store_seconds_total"),
                    "vLLM's OffloadingConnector figures; no SGLang family reports a host KV tier"),
    # WHY A REQUEST WAITS is vLLM's `num_requests_waiting_by_reason`; SGLang splits its waiting by QUEUE (grammar,
    # prefill bootstrap, decode transfer…), not by what it waits for — no family in any capture says capacity/deferred.
    **dict.fromkeys(("waiting_capacity", "waiting_deferred"),
                    "vLLM's num_requests_waiting_by_reason; SGLang publishes its queues, not why they wait"),
    **dict.fromkeys((k for k, _ in SPEECH_KEYS), "a transcription service's figure (transcription_*); no SGLang family"),
    **dict.fromkeys((k for k, _ in POOL_INPUT_KEYS),
                    "a pool reserved one window a seat (TensorFold's); SGLang reports its shared pool's use itself"),
}

#: WHAT SGLANG PUBLISHES ONLY WHEN ASKED. Canonical key -> the launch switch that publishes it.
#:
#: vLLM REGISTERS its MFU counters whatever it was started with and merely leaves them at 0, so on vLLM
#: "exposed" and "measured" are different questions and `throughput.py` tells them apart from the value.
#: SGLang does not create the families at all without the switch, so with it off these keys are ABSENT
#: from the scrape, come back null, and the page reads "not exposed by this server" — never a fabricated
#: 0. `lb/cluster/tp-rank`'s `sglang_cmd` passes the switch on every SGLang start it makes, and
#: `CLUSTER_MFU_METRICS=0` — the same opt-out the vLLM path honours — turns it back off on either engine.
#: The switch also needs `--enable-metrics`, which gates it inside the engine and which that same
#: command already passes; where each of those was read in the pinned image is cited at the mapping.
#:
#: EVERY CAPTURE under `lb/tests/fixtures/` predates the switch — `sglang-lb-start-2026-10-03-rank0.log`
#: carries SGLang's own `server_args` with `'enable_mfu_metrics': False` — which is why these three are
#: still null there while no longer being claimed impossible.
SGLANG_BEHIND_A_FLAG = {
    "est_flops_total": "--enable-mfu-metrics",
    "est_read_bytes_total": "--enable-mfu-metrics",
    "est_write_bytes_total": "--enable-mfu-metrics",
}


def _replica(labels: dict[str, str]) -> tuple:
    """Which REPLICA of a figure this sample is — rank 0's or rank 1's. Sorts numerically, so rank 2
    beats rank 10, and an empty tuple for a family that is published once for the whole engine (so it
    is its own only group and is never dropped)."""
    out = []
    for name in SGLANG_REPLICA_LABELS:
        value = labels.get(name)
        if value is not None:
            out.append((name, int(value) if value.lstrip("-").isdigit() else math.inf, value))
    return tuple(out)


def parse_sglang_metrics(text: str) -> dict:
    """One scrape of SGLang's `/metrics` as the LB's own canonical scrape (`SCRAPE_KEYS`).

    Same discipline as `parse_vllm_metrics`, and the same result shape: a family the text does not
    expose is None — never 0 — and one unreadable line costs its own key for this scrape, never the
    scrape. Counts are summed across the labels that PARTITION the work (`is_streaming`, `mode`,
    `stage`) but only within ONE rank, because the ranks of a TP pair replicate the same figure.
    """
    totals: dict[str, dict[tuple, float]] = {}
    buckets: dict[str, dict[tuple, dict[float, float]]] = {}
    bad: set[str] = set()
    sglang = False
    for line in text.splitlines():
        sample = _sample(line)
        if sample is None:
            continue
        name, labels, raw = sample
        sglang = sglang or name.startswith("sglang:")
        rank = _replica(labels)
        if name in SGLANG_HISTOGRAM_BUCKETS:
            _sglang_bucket(sample, SGLANG_HISTOGRAM_BUCKETS[name], rank, buckets, bad)
            continue
        for key, want, kind in SGLANG_FAMILIES.get(name, ()):
            if want is not None and labels.get(want[0]) not in want[1]:
                continue
            try:
                value = float(raw)
            except ValueError:
                bad.add(key)
                continue
            if not math.isfinite(value) or (kind != _GAUGE and value < 0):
                bad.add(key)
                continue
            group = totals.setdefault(key, {})
            group[rank] = max(group[rank], value) if kind == _GAUGE and rank in group \
                else group.get(rank, 0.0) + value
    # ONE RANK PER KEY, chosen per key because the families do not all carry the same rank labels:
    # the tokenizer-side ones carry none and would otherwise lose to a rank-labelled group.
    read = {key: None if key in bad or key not in totals else totals[key][min(totals[key])]
            for key in _SGLANG_KINDS}
    read.update(_sglang_derived(read))
    out = {key: None for key, _ in SCRAPE_KEYS}
    for key, kind in SCRAPE_KEYS:
        value = read.get(key)
        if value is not None:
            out[key] = int(value) if kind == _COUNT else value
    for key, edges in buckets.items():
        out[key] = None if key in bad else tuple(sorted(edges[min(edges)].items()))
    # SGLang records one gap per EMITTED TOKEN, vLLM one per ENGINE STEP (see `ITL_PER_TOKEN`).
    out["itl_basis"] = _itl_basis(out, ITL_PER_TOKEN)
    out["engine"] = ENGINE_SGLANG if sglang else None
    return out


def _sglang_derived(read: dict[str, float | None]) -> dict[str, float | None]:
    """The canonical figures SGLang states only indirectly. Each is None unless every input it needs
    was really read, so a missing input leaves a null rather than a plausible number."""
    gib, width, finished = read.get(_KV_GIB), read.get(_SPEC_WIDTH), read.get(_GEN_FINISHED)
    drafts = read.get("spec_drafts_total")
    # GiB, not GB: 52.3751220703125 GiB over 3 432 455 tokens is 16 384.00 bytes a token — exactly
    # 2**14, which a 10**9 reading (15 258.79) would not give.
    pool_bytes = None if gib is None else gib * (1024 ** 3)
    # Each verify step proposes `width - 1` draft tokens (the identity above); a width of 1 or less
    # means nothing was drafted, so there is no draft-token figure to publish.
    draft_tokens = None if drafts is None or width is None or width <= 1 else drafts * (width - 1)
    # Accepted DRAFT tokens: the tokens those verify steps produced, less the one bonus token each
    # step always emits. Both counters are completion-side, so they advance together.
    # UNVERIFIED that they advance ATOMICALLY: a scrape landing between the two increments of one
    # request would dip this by that request's tokens and recover on the next scrape, which the
    # consumers' delta logic reads as a counter reset — bounded, and never a fabricated figure.
    accepted = None if finished is None or drafts is None or finished < drafts else finished - drafts
    return {"kv_pool_bytes": pool_bytes, "spec_draft_tokens_total": draft_tokens,
            "spec_accepted_total": accepted}


def _sglang_bucket(sample: tuple[str, dict[str, str], str], key: str, rank: tuple,
                   buckets: dict, bad: set) -> None:
    """One SGLang `_bucket{le="…"}` line into its histogram, per rank, exactly as `_bucket` does
    within one rank: summed across the label sets that partition the work."""
    _, labels, raw = sample
    try:
        le, value = float(labels.get("le", "")), float(raw)
    except ValueError:
        bad.add(key)
        return
    if math.isnan(le) or not math.isfinite(value) or value < 0:
        bad.add(key)
        return
    counts = buckets.setdefault(key, {}).setdefault(rank, {})
    counts[le] = counts.get(le, 0.0) + value


# ──────────────────────────────── transcription ────────────────────────────────
#
# A transcription server (the node engines `faster-whisper` and `nemo`) publishes the `transcription_*` family this
# gateway's transcription-server contract names (docs/design/engines.md; GET /metrics), normalised HERE as SGLang's is.
# Read off tests/fixtures/transcription-metrics-2026-10-23-{busy,idle}.txt.

#: Exposed name -> canonical key. `transcription_gpu_memory_bytes` is summed over its `region` (vram, gtt).
TRANSCRIPTION_FAMILIES = {
    "transcription_requests_running": "running",
    "transcription_requests_waiting": "waiting",
    "transcription_seats": "seats",
    "transcription_queue_capacity": "queue_capacity",
    "transcription_audio_seconds_total": "audio_seconds_total",
    "transcription_processing_seconds_sum": "request_seconds_sum",
    "transcription_processing_seconds_count": "request_seconds_count",
    "transcription_gpu_memory_bytes": "gpu_memory_bytes",
}
#: `transcription_requests_total{outcome}` -> the canonical counters it feeds: an answer (ok, or silence: answered
#: with no speech) is a FINISHED request as vLLM's `request_success_total` is; every rejection reason is one count.
TRANSCRIPTION_OUTCOMES = {
    "ok": ("requests_finished_total",),
    "silence": ("requests_finished_total", "requests_silence_total"),
    "failed": ("requests_failed_total",),
    "rejected_busy": ("requests_rejected_total",),
    "rejected_queue_full": ("requests_rejected_total",),
    "rejected_deferred": ("requests_rejected_total",),
}
_TRANSCRIPTION_KEYS = {*TRANSCRIPTION_FAMILIES.values(), *(k for keys in TRANSCRIPTION_OUTCOMES.values() for k in keys),
                       "engine"}
#: Every other canonical key, null off a transcription scrape — and why.
TRANSCRIPTION_UNAVAILABLE = dict.fromkeys(
    (k for k, _ in SCRAPE_KEYS if k not in _TRANSCRIPTION_KEYS),
    "a CTranslate2 transcription service: no token, KV-cache, prefill, speculative or MFU counters, no process start")


def parse_transcription_metrics(text: str) -> dict:
    """One scrape of the transcription service's `/metrics` as the canonical scrape — same discipline as the others:
    a family it does not expose is None, one unreadable line costs its own key."""
    kinds = dict(SCRAPE_KEYS)
    totals: dict[str, float] = {}
    bad: set[str] = set()
    for line in text.splitlines():
        sample = _sample(line)
        if sample is None:
            continue
        name, labels, raw = sample
        keys = ((TRANSCRIPTION_FAMILIES[name],) if name in TRANSCRIPTION_FAMILIES
                else TRANSCRIPTION_OUTCOMES.get(labels.get("outcome"), ()) if name == "transcription_requests_total"
                else ())
        for key in keys:
            try:
                value = float(raw)
            except ValueError:
                bad.add(key)
                continue
            if not math.isfinite(value) or value < 0:
                bad.add(key)
                continue
            totals[key] = totals.get(key, 0.0) + value
    out = {key: None for key, _ in SCRAPE_KEYS}
    for key, value in totals.items():
        if key not in bad:
            out[key] = int(value) if kinds[key] == _COUNT else value
    out["engine"] = ENGINE_TRANSCRIPTION if totals or bad else None
    return out


# ──────────────────────────────── TensorFold ────────────────────────────────
#
# TensorFold (the pair's third engine, 2026-10-29) writes `tensorfold:*` — read in its own source, not a capture: the
# native server's zig/src/server/metrics.zig `Metrics.render` (TensorFold db281878, the image the spike ran) and 0.6.6's
# src/tensorfold/server/metrics.py `render` (cb2ebf05), which write the same families ("family for family as the Python
# server writes it") except where noted. tests/fixtures/tensorfold-metrics-native-db28187.txt is rendered in that exact
# format. The pair was busy with a measurement when this was written, so NO LIVE SCRAPE backs it yet: the first serving
# generation's /metrics should be kept beside the fixture and checked against it.

#: Exposed name -> (canonical key, kind). Mirrors (`num_requests_running`, the `e2e_`/`_time_` histograms) are the
#: same values under vLLM's names; each figure is read once, from its own family.
TENSORFOLD_FAMILIES = {
    "tensorfold:requests_running": ("running", _COUNT),
    "tensorfold:requests_waiting": ("waiting", _COUNT),
    # FINISHED requests' tokens only (both servers' HELP: "of finished requests") — `gen_tokens_total` adds the live
    # streams' (`_TF_GEN_RUNNING`) so the count moves while the work happens, not when a batch completes.
    "tensorfold:prompt_tokens_total": ("prompt_tokens_total", _COUNT),
    "tensorfold:spec_decode_num_draft_tokens_total": ("spec_draft_tokens_total", _COUNT),
    "tensorfold:spec_decode_num_accepted_tokens_total": ("spec_accepted_total", _COUNT),
    "tensorfold:request_latency_seconds_sum": ("request_seconds_sum", _SUM),
    "tensorfold:request_latency_seconds_count": ("request_seconds_count", _COUNT),
    # One observation per finished request: the count of finished requests, as vLLM's request_success_total is.
    "tensorfold:time_to_first_token_seconds_sum": ("ttft_sum", _SUM),
    "tensorfold:time_to_first_token_seconds_count": ("ttft_count", _COUNT),
    # The native server only (0.6.6 writes no prefill histogram): absent there, so null there.
    "tensorfold:request_prefill_seconds_sum": ("prefill_time_sum", _SUM),
    "tensorfold:request_prefill_seconds_count": ("prefill_requests_total", _COUNT),
    "tensorfold:preemptions_total": ("preemptions_total", _COUNT),
    # THE ENGINE'S OWN WORK — the fleet image's patch 0014 (deploy/images/tensorfold; absent before -fleet.2, so null
    # there). Counted where the lane core hands the backend its calls (core/lanes/counting.zig) and where the Flash-Next
    # engine commits a prompt chunk (cuda_engine `notePromptChunk`): the same quantities SGLang's pass families give.
    "tensorfold:decode_passes_total": ("decode_steps_total", _COUNT),
    "tensorfold:prompt_tokens_computed_total": ("prompt_computed_total", _COUNT),
    "tensorfold:prompt_tokens_cached_total": ("prompt_cached_total", _COUNT),
    # the LAST decode forward's: its streams' context summed, and the rows a stream carried (pending token + drafts)
    "tensorfold:decode_context_tokens": ("decode_context_tokens", _COUNT),
    "tensorfold:verify_width": ("spec_verify_width", _GAUGE),
    # seconds the round loop spent inside the backend's calls — the GPU's work and the host's around it
    "tensorfold:busy_seconds_total": ("busy_seconds_total", _SUM),
    # the computed prompt of exactly the finished requests `request_prefill_seconds` timed: prefill / request's tokens
    "tensorfold:request_prefill_computed_tokens_total": ("prefill_computed_tokens_sum", _COUNT),
}
_TF_GEN_FINISHED, _TF_GEN_RUNNING = "tensorfold:generation_tokens_total", "tensorfold:generation_tokens_running"
#: Prompt chunks (patch 0014): `steps_total` is them PLUS the decode forwards — only with both, as for generated tokens.
_TF_PROMPT_PASSES = "tensorfold:prompt_passes_total"
#: ONE STREAM's tokens over ITS window (native server/metrics.zig `pools`: "Tokens in a stream cache divided by that
#: stream's context window"), a sample per live stream — an idle server prints one at 0. Summed, then divided by the
#: seats in `with_health`; the per-stream `kv_cache_usage_perc` mirror carries the same ratios and is not read twice.
_TF_STREAM_FILL = "tensorfold:kv_cache_usage_ratio"
_TF_FINISHED = "tensorfold:request_latency_seconds_count"
TENSORFOLD_HISTOGRAM_BUCKETS = {"tensorfold:time_to_first_token_seconds_bucket": "ttft_buckets"}
_TF_KEYS = {key for key, _ in TENSORFOLD_FAMILIES.values()} | {"gen_tokens_total", "requests_finished_total",
                                                              "ttft_buckets", "engine", "steps_total"}
#: WHAT A TENSORFOLD GENERATION WILL NOT SHOW — every other canonical key, null, and why.
TENSORFOLD_UNAVAILABLE = {
    # `kv_cache_usage` is not here: /metrics gives each stream's fill of its window, and /health gives the seats those
    # windows are reserved for — together the pool in use (`with_health`), never one of them alone.
    **dict.fromkeys(("kv_pool_tokens", "kv_pool_bytes", "kv_pool_blocks", "kv_block_tokens", "kv_max_concurrency",
                     "gpu_memory_utilization"),
                    "TensorFold budgets its sequence memory itself and publishes no pool figure in /metrics"),
    **dict.fromkeys(("est_flops_total", "est_read_bytes_total", "est_write_bytes_total"),
                    "TensorFold publishes no FLOP or byte estimate"),
    **dict.fromkeys(("prefix_queries_total", "prefix_hits_total"),
                    "its prompt cache's lookups are a log line ('info: prompt cache: …'), not a family; the tokens it "
                    "restored are (patch 0014 `prompt_tokens_cached_total`)"),
    "process_start_time": "no epoch gauge: a restarted engine's counters are read as a reset",
    **dict.fromkeys(("itl_sum", "itl_count", "itl_buckets", "step_tokens_sum", "spec_drafts_total"),
                    "no per-step token or inter-token family (its time per output token is per REQUEST)"),
}


def parse_tensorfold_metrics(text: str) -> dict:
    """One scrape of TensorFold's `/metrics` as the canonical scrape — the same discipline as the others: a family it
    does not expose is None, never 0, and one unreadable line costs its own key. One process writes it (rank 0's HTTP
    server), so nothing is replicated across ranks and nothing is summed but a histogram's own buckets."""
    read: dict[str, float] = {}
    bad: set[str] = set()
    buckets: dict[str, dict[float, float]] = {}
    for line in text.splitlines():
        sample = _sample(line)
        if sample is None:
            continue
        name, labels, raw = sample
        if name in TENSORFOLD_HISTOGRAM_BUCKETS:
            key = TENSORFOLD_HISTOGRAM_BUCKETS[name]
            try:
                le, value = float(labels.get("le", "")), float(raw)
            except ValueError:
                bad.add(key)
                continue
            buckets.setdefault(key, {})[le] = value
            continue
        if name == _TF_STREAM_FILL:
            try:
                fill = float(raw)
            except ValueError:
                bad.add(name)
                continue
            if not math.isfinite(fill) or not 0 <= fill <= 1:
                bad.add(name)
                continue
            read[name] = read.get(name, 0.0) + fill
            continue
        if name not in TENSORFOLD_FAMILIES and name not in (_TF_GEN_FINISHED, _TF_GEN_RUNNING, _TF_PROMPT_PASSES):
            continue
        try:
            value = float(raw)
        except ValueError:
            bad.add(name)
            continue
        if not math.isfinite(value) or value < 0:
            bad.add(name)
            continue
        read[name] = value
    out = {key: None for key, _ in SCRAPE_KEYS}
    for name, (key, kind) in TENSORFOLD_FAMILIES.items():
        if name in read and name not in bad:
            out[key] = int(read[name]) if kind == _COUNT else read[name]
    if _TF_FINISHED in read and _TF_FINISHED not in bad:
        out["requests_finished_total"] = int(read[_TF_FINISHED])
    # GENERATED TOKENS SO FAR = finished requests' + the live streams'. Only with both: one alone is a count that sits
    # still for a whole request (finished) or falls when one ends (running). A stream a client leaves takes its running
    # tokens with it un-finished — a dip the consumers' delta logic reads as a reset, bounded, never invented.
    if all(n in read and n not in bad for n in (_TF_GEN_FINISHED, _TF_GEN_RUNNING)):
        out["gen_tokens_total"] = int(read[_TF_GEN_FINISHED] + read[_TF_GEN_RUNNING])
    # EVERY FORWARD = the prompt chunks + the decode forwards (patch 0014), only with both.
    if out["decode_steps_total"] is not None and _TF_PROMPT_PASSES in read and _TF_PROMPT_PASSES not in bad:
        out["steps_total"] = int(read[_TF_PROMPT_PASSES]) + out["decode_steps_total"]
    for key, edges in buckets.items():
        out[key] = None if key in bad else tuple(sorted(edges.items()))
    if _TF_STREAM_FILL in read and _TF_STREAM_FILL not in bad:
        out["kv_stream_fill_sum"] = read[_TF_STREAM_FILL]
    out["engine"] = ENGINE_TENSORFOLD if read or buckets or bad else None
    return out


def _labels(line: str, i: int) -> tuple[dict[str, str] | None, int]:
    """The label set starting just after `{`, and the index after its `}`. Values are quoted and
    may hold `}`, spaces, `\\"` and `+Inf`; anything malformed is (None, i)."""
    labels: dict[str, str] = {}
    n = len(line)
    while i < n:
        while i < n and line[i] in " ,":
            i += 1
        if i < n and line[i] == "}":
            return labels, i + 1
        eq = line.find("=", i)
        if eq == -1 or eq + 1 >= n or line[eq + 1] != '"':
            return None, i
        name, i, value = line[i:eq].strip(), eq + 2, []
        while i < n and line[i] != '"':
            if line[i] == "\\" and i + 1 < n:
                i += 1
                value.append("\n" if line[i] == "n" else line[i])
            else:
                value.append(line[i])
            i += 1
        if i >= n:
            return None, i
        labels[name] = "".join(value)
        i += 1
    return None, i


def _sample(line: str) -> tuple[str, dict[str, str], str] | None:
    """(name, labels, value text) of one exposition line; None for a comment, a blank or garbage."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    brace, space = line.find("{"), line.find(" ")
    if brace != -1 and (space == -1 or brace < space):
        labels, end = _labels(line, brace + 1)
        name, rest = line[:brace], line[end:]
        if labels is None:
            return None
    else:
        name, _, rest = line.partition(" ")
        labels = {}
    parts = rest.split()             # the value, then an optional timestamp
    return (name, labels, parts[0]) if name and parts else None


def parse_vllm_metrics(text: str) -> dict:
    """What one scrape of vLLM's `/metrics` says, by scrape key (`SCRAPE_FAMILIES`).

    A family the text does not expose is None — never 0: `--disable-log-stats` removes every
    `vllm:` family, and a 0 would read as idle. So is a family any of whose lines could not be read
    (NaN, ±Inf, a negative count, garbage): one bad line costs its family for this scrape, not the
    whole scrape. Counts are summed across label sets (engines, models); a gauge takes the largest.
    """
    totals: dict[str, float] = {}
    bad: set[str] = set()
    buckets: dict[str, dict[float, float]] = {}
    words: dict[str, object] = {}
    spelled: dict[str, float] = {}                # SECOND_SPELLINGS' readings, kept apart until the flat ones are known
    vllm = False
    for line in text.splitlines():
        sample = _sample(line)
        if sample is None:
            continue
        vllm = vllm or sample[0].startswith("vllm:")
        if sample[0] in HISTOGRAM_BUCKETS:
            _bucket(sample, buckets, bad)
            continue
        if sample[0] in SCRAPE_INFO_LABELS:
            _info(sample, totals, bad)
            for label, (key, kind) in SCRAPE_INFO_WORDS.get(sample[0], {}).items():
                if label in sample[1]:
                    words[key] = _word(kind, sample[1][label])
            continue
        name, labels, raw = sample
        if name in SCRAPE_BY_LABEL:
            label_name, keys, kind = SCRAPE_BY_LABEL[name]
            key = keys.get(labels.get(label_name))
            if key is None:
                continue
        elif name in SCRAPE_FAMILIES:
            key, label, kind = SCRAPE_FAMILIES[name]
            if label is not None and labels.get(label[0]) != label[1]:
                continue
        else:
            continue
        try:
            value = float(raw)
        except ValueError:
            bad.add(key)
            continue
        if not math.isfinite(value) or (kind != _GAUGE and value < 0):
            bad.add(key)
            continue
        if name in SECOND_SPELLINGS:
            spelled[key] = spelled.get(key, 0.0) + value
            continue
        totals[key] = max(totals[key], value) if kind == _GAUGE and key in totals \
            else totals.get(key, 0.0) + value
    for key, value in spelled.items():
        totals.setdefault(key, value)
    out: dict = {}
    for key, kind in SCRAPE_KEYS:
        if kind == _BUCKETS:
            out[key] = (None if key in bad or key not in buckets
                        else tuple(sorted(buckets[key].items())))
        elif kind in (_WORD, _FLAG):
            out[key] = words.get(key)
        elif key in bad or key not in totals:
            out[key] = None
        else:
            out[key] = int(totals[key]) if kind == _COUNT else totals[key]
    # One gap per ENGINE STEP here, not per token — the SGLang side of the same key says so too.
    out["itl_basis"] = _itl_basis(out, ITL_PER_STEP)
    # The vocabulary it was written in: a text with no `vllm:` sample named no engine at all.
    out["engine"] = ENGINE_VLLM if vllm else None
    return out


def _info(sample: tuple[str, dict[str, str], str], totals: dict, bad: set) -> None:
    """One info metric's LABELS into the scrape. An info family is published once per engine with the
    same configuration, so the largest wins rather than the sum: two engines do not have twice the pool.
    A label that is missing or not a finite non-negative number costs its own key, never the others."""
    name, labels, _ = sample
    for label, (key, kind) in SCRAPE_INFO_LABELS[name].items():
        raw = labels.get(label)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            bad.add(key)
            continue
        if not math.isfinite(value) or value < 0:
            bad.add(key)
            continue
        totals[key] = max(totals.get(key, 0.0), value)


def _bucket(sample: tuple[str, dict[str, str], str], buckets: dict, bad: set) -> None:
    """One `_bucket{le="…"}` line into its histogram's {le: cumulative count}, summed across label sets.
    A line with no readable `le`, or a count that is not a finite non-negative number, costs its
    histogram for this scrape — never the scrape."""
    name, labels, raw = sample
    key = HISTOGRAM_BUCKETS[name]
    try:
        le, value = float(labels.get("le", "")), float(raw)
    except ValueError:
        bad.add(key)
        return
    if math.isnan(le) or not math.isfinite(value) or value < 0:
        bad.add(key)
        return
    counts = buckets.setdefault(key, {})
    counts[le] = counts.get(le, 0.0) + value


#: THE VOCABULARIES, by the prefix a scrape's samples carry -> (the engine, its parser). The first sample
#: naming one decides.
VOCABULARIES: tuple[tuple[str, str, object], ...] = (
    ("sglang:", ENGINE_SGLANG, parse_sglang_metrics),
    ("transcription_", ENGINE_TRANSCRIPTION, parse_transcription_metrics),
    ("tensorfold:", ENGINE_TENSORFOLD, parse_tensorfold_metrics),
    ("vllm:", ENGINE_VLLM, parse_vllm_metrics),
)
#: The engines that serve chat completions — what a swarm can talk to. The transcription service speaks a
#: vocabulary of its own and is a resident engine, but not a model a deep-research session can use.
CHAT_ENGINES = frozenset({ENGINE_VLLM, ENGINE_SGLANG, ENGINE_TENSORFOLD})


def engine_of(text: str) -> str | None:
    """The engine whose vocabulary this scrape speaks (`ENGINE_*`), or None when it speaks none of them —
    an exposition with no engine family is no engine at all."""
    for line in text.splitlines():
        for prefix, engine, _ in VOCABULARIES:
            if line.startswith(prefix):
                return engine
    return None


def parse_metrics(text: str) -> dict:
    """What one scrape of a backend's `/metrics` says, in the LB's canonical keys, WHICHEVER engine
    wrote it.

    HOW IT DECIDES: the exposition names its own vocabulary (`engine_of`), so nothing has to be threaded
    in from the catalog's `LaunchProfile.engine`. A real SGLang scrape carries `sglang:` families and not
    one `vllm:` sample (the captured 497 series contain no occurrence of the string "vllm" at all), and a
    vLLM scrape the reverse. Reading the TEXT also keeps the probe honest about what it actually
    reached rather than what the profile intended — a pair that came up as the other engine, or a
    stale proxy, is then read as what it is. A text with neither vocabulary is parsed as vLLM, which
    yields the all-null scrape it does today.
    """
    engine = engine_of(text)
    for _, name, parser in VOCABULARIES:
        if name == engine:
            return parser(text)
    return parse_vllm_metrics(text)


def _tensorfold_health(scrape: dict, health: dict) -> dict:
    """TensorFold's /health `max_batch_size` is its lanes — the `--parallel` it serves (native
    server/status_routes.zig `health`, `srv.info.lanes`) — and its sequence memory is reserved one window a
    lane (rank 0: 'sequence memory: 30.76 GiB' = 16 x 131,072 x 15,749 B). So the pool in use is the streams'
    fills over the lanes. A keyed server's health says only "ok": no seat count, and then no pool figure."""
    seats = health.get("max_batch_size")
    if not isinstance(seats, int) or isinstance(seats, bool) or seats <= 0:
        return scrape
    fill = scrape.get("kv_stream_fill_sum")
    usage = None if fill is None else min(1.0, fill / seats)
    return {**scrape, "seats": seats, "kv_cache_usage": usage}


#: WHAT AN ENGINE'S /health ADDS TO ITS SCRAPE, per engine that says something there the scrape needs. vLLM's and
#: SGLang's /health carry no body the figures read; their pools are in /metrics.
HEALTH_READERS = {ENGINE_TENSORFOLD: _tensorfold_health}


def with_health(scrape: dict, body: bytes | None) -> dict:
    """The scrape completed by the engine's own /health body — for an engine whose figures need both
    (`HEALTH_READERS`); every other scrape, or a body that is not a JSON object, comes back as it was."""
    reader = HEALTH_READERS.get(scrape.get("engine"))
    if reader is None or not body:
        return scrape
    try:
        health = json.loads(body)
    except ValueError:
        return scrape
    return reader(scrape, health) if isinstance(health, dict) else scrape
