"""What one server is generating and loading, MEASURED — and, when a figure is not, why.

The hub keeps one `Throughput` per server (per URL) and feeds it every probe result; `/lb/status`
reads `view` (the `throughput` block) and `loaded` (the `loaded` block).
worklog/vllm-gateway-throughput.md, Design 1-4:

  * a classic vLLM or a cluster's head is measured from its own counters (`source: "metrics"`):
    every client's tokens, fleet jobs included, through one `RateMeter` per series. (The ollama fallback, which
    kept none and was measured off the relayed usage, is retired with the fallback tier, 2026-10-17.)

`null` means not measured and comes with its `why`; a number, 0 included, is a measurement.
"""
from __future__ import annotations

from typing import NamedTuple

from vllm_lb.rates import (CREDITED_AT_FIRST_TOKEN, FIRST_SAMPLE, MFU_NOT_KNOWN_YET, MFU_NOT_MEASURED,
                           NO_REQUEST_YET, NOT_EXPOSED, PASS_IN_FLIGHT, SCRAPE_FAILED,
                           SERVER_DOWN, STEP_IN_FLIGHT, TOO_SHORT_TO_TIME, WINDOW_NOT_REACHED, RateMeter, Reading)
from vllm_lb.state import Backend
from vllm_lb.usage_tap import UNCOUNTED, Usage

METRICS = "metrics"

#: PROMPT is credited at each request's first output token, in one lump: only a window is honest.
PROMPT_WINDOW_S = 60.0
#: Prefill speed comes from FINISHED requests' own timings, so its window must be long enough to hold any.
PREFILL_WINDOW_S = 300.0
#: MTP acceptance, over the same minute as PROMPT.
SPEC_DECODE_WINDOW_S = 60.0

#: SPEECH figures (`Throughput.speech`) come from FINISHED requests' own counters — a transcription is seconds of work
#: and arrives in bursts — so their window holds several, as prefill speed's does.
SPEECH_WINDOW_S = PREFILL_WINDOW_S
#: The answers per minute the speech block states, by outcome: series name -> its word.
SPEECH_OUTCOMES = {"finished": "finished", "silence": "silence", "failed": "failed", "rejected": "rejected"}

#: Series -> (scrape key, window). One RateMeter each, all fed the same stamped probe.
SERIES = {
    "generated": ("gen_tokens_total", None),
    "prompt": ("prompt_tokens_total", PROMPT_WINDOW_S),
    "computed": ("prompt_computed_total", PROMPT_WINDOW_S),
    "cached": ("prompt_cached_total", PROMPT_WINDOW_S),
    "prefill_tokens": ("prefill_computed_tokens_sum", PREFILL_WINDOW_S),
    "prefill_seconds": ("prefill_time_sum", PREFILL_WINDOW_S),
    "prefill_requests": ("prefill_requests_total", PREFILL_WINDOW_S),
    "drafts": ("spec_drafts_total", SPEC_DECODE_WINDOW_S),
    "draft_tokens": ("spec_draft_tokens_total", SPEC_DECODE_WINDOW_S),
    "accepted": ("spec_accepted_total", SPEC_DECODE_WINDOW_S),
    # WHAT THE ENGINE ESTIMATES IT DID, per GPU. Window None — the last two probes, as `generated` is
    # — because the question is "is it saturated NOW", and a lifetime average answers a different one.
    "est_flops": ("est_flops_total", None),
    "est_read_bytes": ("est_read_bytes_total", None),
    "est_write_bytes": ("est_write_bytes_total", None),
    # FORWARD PASSES, every kind and decode alone: the modeled bandwidth's pass rates (`bandwidth_model`).
    # Over the last two probes, as `generated` is.
    "steps": ("steps_total", None),
    "decode_steps": ("decode_steps_total", None),
    # …and the prompt rows those passes computed, over the SAME two probes: a routed-expert model's prompt chunk reads
    # the experts its rows touch (catalog `RoutedStepTraffic.prefill_bytes`), so the window's mean chunk prices it.
    "pass_rows": ("prompt_computed_total", None),
    # THE KV TIER IN THE HOST'S RAM (`kv_offload`): bytes reloaded RAM → GPU and stored GPU → RAM, over the last two
    # probes as `generated` is — the KV tile's own cadence.
    "kv_reload": ("kv_offload_load_bytes_total", None),
    "kv_store": ("kv_offload_store_bytes_total", None),
    # SPEECH (`speech`): the audio transcribed, the seconds the engine spent on whole requests and how many, and the
    # answers by outcome — one window, so every ratio is taken over the same span.
    "audio": ("audio_seconds_total", SPEECH_WINDOW_S),
    "request_seconds": ("request_seconds_sum", SPEECH_WINDOW_S),
    "requests": ("request_seconds_count", SPEECH_WINDOW_S),
    "finished": ("requests_finished_total", SPEECH_WINDOW_S),
    "silence": ("requests_silence_total", SPEECH_WINDOW_S),
    "failed": ("requests_failed_total", SPEECH_WINDOW_S),
    "rejected": ("requests_rejected_total", SPEECH_WINDOW_S),
}

#: How much of a run the engine's counters (and the generated tokens beside them) KEEP beyond their
#: two-probe window: enough to widen a window in which no engine step completed back to the last one
#: that did (`Throughput.engine_perf`). A 16384-token prefill step is ~7 s on this pair; a minute holds
#: several of them.
PERF_KEEP_S = 60.0
KEEP_S = {"generated": PERF_KEEP_S, "est_flops": PERF_KEEP_S, "est_read_bytes": PERF_KEEP_S,
          "est_write_bytes": PERF_KEEP_S, "steps": PERF_KEEP_S, "decode_steps": PERF_KEEP_S}

#: What `engine_perf` says beside a rate it took over a WIDER window than the last two probes, because
#: no engine step completed between them. A number with this note is still a measurement: the work the
#: engine counted, over the window `window_s` states.
WIDENED_PREFILL = ("prefill in progress — vLLM counts a step's FLOPs and bytes when the step completes, "
                   "and a prefill step outlasted the probe window, so the window was widened to the "
                   "last completed step")
WIDENED_STEP = ("an engine step outlasted the probe window — vLLM counts a step's FLOPs and bytes when "
                "the step completes, so the window was widened to the last completed step")
#: Why a MODELED bandwidth (`bandwidth_model`) is a lower bound for its window: a prefill pass is charged the
#: target's weights only. The prefix KV its attention reads over a cached context, and the GDN state of the
#: requests it admits, are not in the model — so the figure can prove a ceiling reached, never one far away.
PREFILL_FLOOR = ("the window held prefill passes, charged their weights only — not the prefix KV their "
                 "attention reads — so the figure is a floor")
#: …and when the window had to be widened back past a pass still running, which is not counted yet.
PASS_FLOOR = ("a forward pass is still running and is not counted yet, so the window was widened to the last "
              "completed pass and the figure is a floor")
#: What the counters prove about `--enable-mfu-metrics` in the engine process being read.
MFU_ON, MFU_OFF = "on", "off"

#: FLOP/s and byte/s as the operator reads them. vLLM's own PromQL for these counters is
#: `rate(...)/1e12` and `(rate(read)+rate(write))/1e9` (v1/metrics/perf.py), so the divisors are the
#: engine's, not ours.
TERA, GIGA = 1e12, 1e9

#: A series nothing has been read for yet.
NOT_YET = Reading(None, FIRST_SAMPLE, None)

#: The keys of each figure in the `throughput` block, besides `why`.
GENERATED_KEYS = ("tok_s", "window_s")
PROMPT_KEYS = ("tok_s", "computed_tok_s", "cached_share", "window_s", "credited", "computed_why")
PREFILL_KEYS = ("tok_s", "requests", "window_s")
SPEC_DECODE_KEYS = ("acceptance_rate", "mean_accepted_len", "window_s")
#: The keys of the `engine_perf` figure: the two rates PER GPU, the window they were taken over, why
#: either is not a number, why the window is wider than two probes (`widened`), and what the counters
#: prove about `--enable-mfu-metrics` ("on" / "off" / null = not known yet).
ENGINE_PERF_KEYS = ("tflops_per_gpu", "gbps_per_gpu", "window_s", "flops_why", "bytes_why", "widened",
                    "mfu_metrics")
#: The keys of the `bandwidth_model` figure: the modeled GB/s PER GPU and its three parts (weights, KV, GDN
#: state), the window, why it is not a number, why it is a floor (null = it is the model's figure), the two
#: pass rates, and the three gauges it was priced at — which the CUDA CORES tile prices its decode pass with.
BANDWIDTH_MODEL_KEYS = ("gbps_per_gpu", "weights_gbps", "kv_gbps", "state_gbps", "window_s", "why", "floor",
                        "decode_passes_s", "prefill_passes_s", "context_tokens", "seqs", "verify_width")
#: The keys of the `engine_config` figure: the engine's own words for how it runs (probe.SCRAPE_INFO_WORDS).
ENGINE_CONFIG_KEYS = ("engine", "gpu_memory_utilization", "kv_cache_dtype", "prefix_caching")
#: The only backend whose `kv_offloading_size` is RAM this engine holds (config/vllm.py:1106-1123 @2eb491e).
NATIVE_TIER = "native"
#: `kv_offloading_size` is GiB (config/cache.py:266-270); the store counter is bytes.
GIB = 2 ** 30
#: How the RAM tier's fill is known — the wire's `occupancy_derived`, the page's hover (`Throughput.kv_offload`).
OCCUPANCY_DERIVED = ("derived: min(tier, bytes stored since the engine started) — an LRU tier frees a chunk only to "
                     "make room for another, so it holds what it stored up to its size; an upper bound (a failed "
                     "store or a cache reset frees chunks the counter does not see). vLLM's own usage gauge counts "
                     "only chunks pinned by copies in flight")
#: The keys of every entry in `loaded.models`; what a server does not say is null.
MODEL_KEYS = ("name", "context", "size_bytes", "gpu_bytes", "offloaded", "expires_in_s", "expiry",
              "kv_usage")


class Sample(NamedTuple):
    """What one taken probe adds to the server's history, and whether work moved since the last.
    `computed_tok_s` is the part of `prompt_tok_s` the engine COMPUTED (`prompt.computed_tok_s`): the
    same window, the same crediting at the first token, the prefix cache's hits left out."""
    gen_tok_s: float | None
    prompt_tok_s: float | None
    moved: bool = False
    computed_tok_s: float | None = None
    #: The KV tier's reload (RAM → GPU) and store (GPU → RAM) bytes a second: `kv_offload`'s own rates.
    kv_reload_bytes_s: float | None = None
    kv_store_bytes_s: float | None = None
    #: The tier's fill, percent of its size (`kv_offload.occupancy_pct`): None where there is no tier or no reading.
    kv_ram_pct: float | None = None


def _value(live: bool, scrape: dict | None, key: str) -> tuple[float | None, str | None]:
    """A series' value in this probe, or None and why nothing was read."""
    if not live:
        return None, SERVER_DOWN
    if scrape is None:
        return None, SCRAPE_FAILED
    value = scrape.get(key)
    return value, (NOT_EXPOSED if value is None else None)


def _changed(before: dict, now: dict, key: str, *, forward: bool) -> bool:
    a, b = before.get(key), now.get(key)
    return a is not None and b is not None and (b > a if forward else b != a)


def _moved(before: dict | None, now: dict | None) -> bool:
    """Work MOVED between two probes that both read the scrape: a generated or a credited prompt token
    (counters, so only forward), or KV usage — which the engine sets every step, the one signal that
    moves while a long prompt is read, since vLLM credits prompt tokens at the first output token
    (Q1b). Across a probe that read nothing, nothing is known to have moved."""
    return before is not None and now is not None and (
        _changed(before, now, "gen_tokens_total", forward=True)
        or _changed(before, now, "prompt_tokens_total", forward=True)
        or _changed(before, now, "kv_cache_usage", forward=False))


def _ram_tier(scrape: dict) -> float | None:
    """The KV tier in the host's RAM a scrape states, in GiB — or None: no size (`kv_offloading_size="None"`), or a
    backend other than vLLM's native one, whose storage is another process's (lmcache: config/vllm.py:1124-1127)."""
    gib = scrape.get("kv_offload_ram_gib")
    return float(gib) if gib and scrape.get("kv_offload_backend") == NATIVE_TIER else None


def _unmeasured(why: str | None, keys: tuple[str, ...], **known) -> dict:
    """A figure whose keys are all null unless known, and the reason it is not a number."""
    return {**dict.fromkeys(keys), **known, "why": why}


def _first_missing(*readings: Reading) -> Reading | None:
    return next((r for r in readings if r.tok_s is None), None)


def _perf(reading: Reading, divisor: float) -> tuple[float | None, str | None]:
    """One engine-estimated rate in its display unit, or None and why."""
    if reading.tok_s is None:
        return None, reading.why
    return reading.tok_s / divisor, None


def _model(**known) -> dict:
    return {key: known.get(key) for key in MODEL_KEYS}


class Throughput:
    """One server's measured figures, kept in memory: a restarted LB starts measuring again."""

    def __init__(self, kind: str, interval_s: float) -> None:
        # Every server the LB routes to keeps vLLM's counters (the ollama fallback, which kept none, is retired).
        self.source = METRICS
        self.interval_s = interval_s
        self._meters = {name: RateMeter(interval_s, window, keep_s=KEEP_S.get(name, 0.0))
                        for name, (_, window) in SERIES.items()}
        #: The last Reading of each series.
        self.readings: dict[str, Reading] = {}
        #: The monotonic stamp, and the scrape, of the last probe whose scrape the meters took; the
        #: scrape is None when the last probe read nothing.
        self._last_read_at: float | None = None
        self._last_scrape: dict | None = None
        #: The monotonic stamp of the probe the last GENERATED reading was taken at: its window ENDS there
        #: (`generated_window`, the REQUESTS card's apportioning).
        self._generated_at: float | None = None
        #: The KV tier in the host's RAM, in GiB, as the last scrape that READ said it (`kv_offload`); None where the
        #: engine runs none. A probe that read nothing keeps it: the tier did not go away because /metrics did.
        self._tier_gib: float | None = None
        #: {since, kv_from} while a prompt is being read: work running, nothing generated, KV rising.
        self.prefilling: dict | None = None
        #: Does the engine have requests running at the last read? A flat MFU counter with work in
        #: flight is a step not yet completed (or a switch proven off), never a measured 0.
        self._running_now = False
        #: The stamp of the last read at which NOTHING was running: a window widened to the last
        #: completed step never reaches back past it, so the idle time before the current work began is
        #: never averaged into its rate. None = no idle read seen in the kept history.
        self._idle_floor: float | None = None
        #: Tokens seen between consecutive reads of one process since the LB started.
        self.totals = {"generated": 0, "prompt": 0}
        #: The last time a request of the LB's to this server ENDED.
        self._request_ended_at: float | None = None
        #: The LB's OWN requests to this server, as UsageTap judged them: how many counted, why the
        #: others were not (the closed set), and the last one counted.
        self.counted = 0
        self.uncounted = dict.fromkeys(UNCOUNTED, 0)
        self.last_request: dict | None = None

    # ── feeding ────────────────────────────────────────────────────────────

    def observe(self, *, live: bool, scrape: dict | None, mono_at: float, at: float | None = None,
                inflight: int = 0) -> Sample | None:
        """Feed one probe result (`at` is its LB wall time). None when it came too soon after the last
        one taken (RateMeter rule 4): then nothing is recorded, not even a history sample."""
        read = live and scrape is not None
        if (read and self._last_read_at is not None
                and mono_at - self._last_read_at < 0.5 * self.interval_s):
            return None
        instance = scrape.get("process_start_time") if read else None
        for name, (key, _) in SERIES.items():
            value, why = _value(live, scrape, key)
            reading = self._meters[name].sample(value, mono_at, instance, why_none=why)
            if reading is not None:
                self.readings[name] = reading
                if name == "generated":
                    self._generated_at = mono_at
        before, now = self._last_scrape, (scrape if read else None)
        if now is not None:
            self._tier_gib = _ram_tier(now)
        self._running_now = now is not None and (now.get("running") or 0) > 0
        if now is not None and not self._running_now:
            self._idle_floor = mono_at
        self._last_read_at, self._last_scrape = (mono_at if read else None), now
        self._count_totals(before, now)
        self._track_prefilling(before, now, at)
        prompt = None if self.prefilling else self.readings["prompt"].tok_s
        # THE FIGURE THE CARD HEADS (`_prompt`), so the line under PREFILL is that number's own past.
        return Sample(self.readings["generated"].tok_s, prompt, _moved(before, now),
                      computed_tok_s=self._prompt()["computed_tok_s"],
                      kv_reload_bytes_s=self.readings["kv_reload"].tok_s,
                      kv_store_bytes_s=self.readings["kv_store"].tok_s,
                      kv_ram_pct=self._tier_fill()[1])

    def request_ended(self, at: float) -> None:
        """A request the LB relayed to this server has ended."""
        self._request_ended_at = at

    def note_usage(self, verdict: Usage | str | None, at: float) -> dict | None:
        """One UsageTap verdict, at LB time `at`. Returns the new `last_request` when it counted."""
        if verdict is None:
            return None
        if isinstance(verdict, str):
            if verdict in self.uncounted:
                self.uncounted[verdict] += 1
            return None
        self.counted += 1
        self.last_request = {"at": at, "path": verdict.path, "prompt_tokens": verdict.prompt_tokens,
                             "output_tokens": verdict.output_tokens,
                             "decode_tok_s": verdict.decode_tok_s,
                             "decode_basis": verdict.decode_basis, "why": verdict.decode_why,
                             "prefill_tok_s": verdict.prefill_tok_s}
        return self.last_request

    def _count_totals(self, before: dict | None, now: dict | None) -> None:
        """Only real deltas: between consecutive reads of ONE process, never a counter's absolute
        value, never across a probe that read nothing, never across a restart."""
        if before is None or now is None:
            return
        if _changed(before, now, "process_start_time", forward=False):
            return
        for total, key in (("generated", "gen_tokens_total"), ("prompt", "prompt_tokens_total")):
            a, b = before.get(key), now.get(key)
            if a is not None and b is not None and b >= a:
                self.totals[total] += b - a

    def _track_prefilling(self, before: dict | None, now: dict | None, at: float | None) -> None:
        """A prompt being read: work running, no token generated since the last read, KV rising."""
        if (before is not None and now is not None and (now.get("running") or 0) > 0
                and before.get("gen_tokens_total") is not None
                and now.get("gen_tokens_total") == before.get("gen_tokens_total")
                and _changed(before, now, "kv_cache_usage", forward=True)):
            if self.prefilling is None:
                self.prefilling = {"since": at, "kv_from": before["kv_cache_usage"]}
        else:
            self.prefilling = None

    # ── reading ────────────────────────────────────────────────────────────

    def generated_window(self) -> dict:
        """GENERATED as the card heads it, and the window it was taken over on the relay's monotonic clock:
        `end_at` the probe it was read at, `window_s` back from there — so the REQUESTS card can count each turn's
        content events in exactly that window (request_queue.py `DERIVED_DECODE`). A server with no token counter
        has no window: `why` says so, never a stand-in."""
        g = self.readings.get("generated", NOT_YET)
        return {"tok_s": g.tok_s, "why": g.why, "window_s": g.window_s,
                "end_at": self._generated_at if g.tok_s is not None else None}

    def last_scrape(self) -> dict | None:
        """The last probe's scrape when it READ one, else None (`observe` keeps None for a probe that read nothing)."""
        return self._last_scrape

    def view(self, backend: Backend, now: float, since: float, traffic=None) -> dict:
        """The `throughput` block of /lb/status for this server. `traffic`: the catalog's `StepTraffic` for the
        model it serves when the LB MODELS its bandwidth (an SGLang generation); None for every other."""
        common = {"last_request": dict(self.last_request) if self.last_request else None,
                  "ours": {"counted": self.counted, "uncounted": dict(self.uncounted)},
                  "totals": {**self.totals, "since": since}}
        generated = self.readings.get("generated", NOT_YET)
        exposes_spec = (self._last_scrape or {}).get("spec_drafts_total") is not None
        # THE ENGINE'S OWN COUNTERS, as last read: every client's tokens since this process loaded the
        # model (a restart starts them at 0). null per figure the engine does not expose, all null
        # while /metrics is not answering — never the LB's `totals`, which count since the LB started.
        scrape = self._last_scrape or {}
        engine_totals = {"generated": scrape.get("gen_tokens_total"),
                         "prompt": scrape.get("prompt_tokens_total"),
                         "computed": scrape.get("prompt_computed_total"),
                         # …and SINCE WHEN: the process's own start (`process_start_time_seconds`), the one clock
                         # an engine this LB did not launch can tell its uptime by. null where it is not exposed.
                         "started_at": scrape.get("process_start_time")}
        # THE POOL THE ENGINE REALLY ALLOCATED (`vllm:cache_config_info`, probe.SCRAPE_INFO_LABELS): the KV
        # tile's denominator while it serves — a catalog figure is only the estimate before a start. null
        # when the engine exposes neither figure, or /metrics is not answering.
        pool = {"tokens": scrape.get("kv_pool_tokens"), "bytes": scrape.get("kv_pool_bytes"),
                "blocks": scrape.get("kv_pool_blocks"), "block_tokens": scrape.get("kv_block_tokens")}
        # WHAT THE ENGINE SAYS IT RUNS WITH, in its own words (probe.SCRAPE_INFO_WORDS): which engine wrote the
        # exposition, the GPU memory fraction it was started with, its KV cache dtype and its prefix caching — the
        # settings a node this LB only observes can be shown by. null per word it does not say.
        engine_config = {key: scrape.get(key) for key in ENGINE_CONFIG_KEYS}
        return {"source": METRICS,
                "generated": {"tok_s": generated.tok_s, "window_s": generated.window_s,
                              "why": generated.why},
                "prompt": self._prompt(),
                "prefill_speed": self.prefill_speed(),
                "spec_decode": self._spec_decode() if exposes_spec else None,
                "prefilling": dict(self.prefilling) if self.prefilling else None,
                "engine_perf": self.engine_perf(),
                "bandwidth_model": self.bandwidth_model(traffic) if traffic is not None else None,
                "engine_totals": engine_totals,
                "kv_pool": pool if any(v is not None for v in pool.values()) else None,
                "kv_offload": self.kv_offload(),
                "engine_config": engine_config,
                "speech": self.speech(),
                **common}

    def kv_offload(self) -> dict | None:
        """THE KV TIER IN THE HOST'S RAM — or None where the engine runs none: no block, never zeros.

        Its size is the engine's own (`kv_offloading_size`, what the applied launch's `--kv-offloading-size` became);
        each direction's rate is a counter's delta over the last two probes, null with the vocabulary's reason; and
        each direction's COPY SPEED is every byte it moved over every second its transfers took, since the engine
        started — the host↔GPU copy's own pace, not the load on it.

        OCCUPANCY IS NOT PUBLISHED by vLLM 2eb491e. `vllm:kv_offload_cpu_cache_usage_perc` is (allocated − free −
        evictable) / chunks (v1/kv_offload/cpu/manager.py:452-459): the chunks PINNED by transfers in flight, every
        stored, evictable block left out — it reads 0.0 with the tier full of prefixes. So the fill is DERIVED
        (`OCCUPANCY_DERIVED`, `occupancy_derived` on the wire): the tier is an LRU cache that frees a chunk only to
        make room for another (prepare_store, manager.py:311-334 — the evicted chunk reused at once), on a failed
        store (complete_store :384-388) or a cache reset (:429-441); so it holds min(its size, the bytes stored since
        the engine started). An UPPER bound: the two frees the counter does not see make it hold less. No store
        counter (vLLM creates it at the first store): not known, never 0."""
        if self._tier_gib is None:
            return None
        scrape = self._last_scrape
        used_gib, occupancy_pct, occupancy_why = self._tier_fill()

        def direction(series: str, moved_key: str, took_key: str) -> dict:
            reading = self.readings.get(series, NOT_YET)
            moved, took = (scrape.get(moved_key), scrape.get(took_key)) if scrape is not None else (None, None)
            if scrape is None:
                copy, copy_why = None, reading.why
            elif moved is None or took is None:
                copy, copy_why = None, NOT_EXPOSED
            elif took <= 0:
                copy, copy_why = None, TOO_SHORT_TO_TIME
            else:
                copy, copy_why = moved / took / GIGA, None
            return {"bytes_s": reading.tok_s, "window_s": reading.window_s, "why": reading.why,
                    "copy_gbps": copy, "copy_why": copy_why}

        return {"tier_gib": self._tier_gib,
                "reload": direction("kv_reload", "kv_offload_load_bytes_total", "kv_offload_load_seconds_total"),
                "store": direction("kv_store", "kv_offload_store_bytes_total", "kv_offload_store_seconds_total"),
                "occupancy_pct": occupancy_pct, "used_gib": used_gib, "occupancy_why": occupancy_why,
                "occupancy_derived": OCCUPANCY_DERIVED}

    def _tier_fill(self) -> tuple[float | None, float | None, str | None]:
        """(used GiB, percent, why) of the RAM tier as the last scrape that READ states it (`kv_offload`'s rule):
        min(its size, the bytes it stored); None without a tier, NOT_EXPOSED without the store counter."""
        if self._tier_gib is None:
            return None, None, None
        stored = (self._last_scrape or {}).get("kv_offload_store_bytes_total")
        if self._last_scrape is None:
            return None, None, self.readings.get("kv_store", NOT_YET).why
        if stored is None:
            return None, None, NOT_EXPOSED
        used = min(self._tier_gib, stored / GIB)
        return used, 100.0 * used / self._tier_gib if self._tier_gib > 0 else None, None

    def engine_perf(self) -> dict:
        """TFLOP/s and GB/s PER GPU, as the engine itself estimated them — or null with the reason.

        WHEN THE COUNTERS MOVE (vLLM 0.29, read in the image): a step's FLOPs and bytes are computed from
        its SchedulerOutput — prefill chunks included — and shipped on SchedulerStats for EVERY step,
        output or not (v1/core/sched/scheduler.py `update_from_output`: "We must return the stats even
        if there are no request outputs this step"). But that runs only after the step's forward pass
        returns (v1/engine/core.py `step`), so the counters move in one lump per COMPLETED step — and a
        16384-token prefill step takes ~7 s on this pair, longer than a probe window.

        So a window in which the counter did not move while requests run is, by itself, a step in
        flight. Only evidence about THIS process says more (`_mfu_evidence`):

          * the counter has left 0: the switch is on. The window is widened back to the last completed
            step — never past the last read at which nothing ran — which is a real count over the
            window `window_s` states, and `widened` says why. With no completed step to widen to, the
            rate is null and says a step is in flight;
          * tokens were produced and the counter is still exactly 0: the switch is off. vLLM registers
            these counters whatever `--enable-mfu-metrics` says, and a "0 TFLOP/s" there would be the
            same invention as a "0 GB VRAM" tile;
          * neither: not known yet.

        A flat counter with nothing running IS a measured 0, and stays one. Read and write bytes are
        summed, as vLLM's own PromQL does, and only when BOTH were read over the same span: half a sum
        is not a bandwidth."""
        flops, read_bytes, written = self._read("est_flops", "est_read_bytes", "est_write_bytes")
        evidence = self._mfu_evidence()
        widened = None
        if evidence == MFU_OFF:
            # PROVEN OFF IS OFF AT REST TOO: a counter that was never counting is no measured zero, idle or busy
            # (site-b 2026-10-11: `compute / GPU 0.0 TFLOPS` after 7,895 tokens with every estimator still at 0.0).
            gap = Reading(None, MFU_NOT_MEASURED, None)
            flops = read_bytes = written = gap
        elif self._running_now and flops.tok_s == 0.0:
            flops, read_bytes, written, widened = self._in_flight(evidence)
        tflops, flops_why = _perf(flops, TERA)
        missing = _first_missing(read_bytes, written)
        if missing is not None:
            gbps, bytes_why = None, missing.why
        elif read_bytes.window_s != written.window_s:
            gbps, bytes_why = None, WINDOW_NOT_REACHED   # two spans do not add into one bandwidth
        else:
            gbps, bytes_why = (read_bytes.tok_s + written.tok_s) / GIGA, None
        window = next((r.window_s for r in (flops, read_bytes) if r.window_s is not None), None)
        return {"tflops_per_gpu": tflops, "gbps_per_gpu": gbps, "window_s": window,
                "flops_why": flops_why, "bytes_why": bytes_why, "widened": widened,
                "mfu_metrics": evidence}

    def bandwidth_model(self, traffic) -> dict:
        """GB/s PER GPU as the LB MODELS it from the engine's own pass counters (SGLang's; TensorFold's patch 0014)
        — or null with the reason.

        SGLang's own byte estimate is not drawn (probe.py `SGLANG_FAMILIES`: 73,600 GB/s a GPU live on a 273
        GB/s part). This prices each FORWARD PASS with the catalog's `StepTraffic` instead, at the engine's
        own pass counts and gauges, over the same two-probe window as every rate here:

            decode passes/s  x (weights + context x KV + the drafter's window of KV + seqs x GDN state x (D+3))
          + prefill passes/s x the target's weights

        A window that held a prefill pass is a FLOOR (`PREFILL_FLOOR`): the prefix KV a prefill's attention
        reads is not in the model. Live, one 8,192-token chunk ran 3.4-4.6 s, so a probe window can end with no
        pass completed while requests run: that is a pass in flight, never a measured 0, and the window widens
        back to the last completed pass — never past the last idle read — exactly as `engine_perf` widens,
        and as a floor (`PASS_FLOOR`). With nothing running, a flat counter is a measured 0.

        D is the engine's own `spec_num_draft_tokens`, which SGLang sets at each decode log interval and holds
        at 0 until the first one after a start (metrics_reporter.py:889-1016): the first 40 decode passes of a
        generation are priced without the drafter — an under-count, never an invention."""
        # The gauges travel with every answer, a number or not: the CUDA CORES tile prices its decode pass
        # at them even while the rate itself is not known yet.
        scrape = self._last_scrape or {}
        context, seqs, width = (scrape.get(key) for key in ("decode_context_tokens", "running",
                                                             "spec_verify_width"))
        gauges = {"context_tokens": context, "seqs": seqs, "verify_width": width}
        steps, decode, rows = self._read("steps", "decode_steps", "pass_rows")
        floor = None
        if self._running_now and steps.tok_s == 0.0:
            base = self._meters["steps"].rise_base(self._idle_floor)
            pair = ([self._meters[name].rate_since(base) for name in ("steps", "decode_steps", "pass_rows")]
                    if base is not None else [None])
            if any(reading is None for reading in pair[:2]):
                return _unmeasured(PASS_IN_FLIGHT, BANDWIDTH_MODEL_KEYS, **gauges)
            (steps, decode, rows), floor = pair, PASS_FLOOR
        missing = _first_missing(steps, decode)
        if missing is not None:
            return _unmeasured(missing.why, BANDWIDTH_MODEL_KEYS, **gauges)
        if steps.window_s != decode.window_s:
            return _unmeasured(WINDOW_NOT_REACHED, BANDWIDTH_MODEL_KEYS, **gauges)
        if context is None or seqs is None:
            return _unmeasured(NOT_EXPOSED, BANDWIDTH_MODEL_KEYS, **gauges)
        prefill_s = max(0.0, steps.tok_s - decode.tok_s)
        # THE MEAN PROMPT CHUNK over the same window, where the engine counts its computed rows; else None, which a
        # routed model prices as a full chunk (every expert) and a dense one ignores.
        chunk = (rows.tok_s / prefill_s if prefill_s > 0 and rows is not None and rows.tok_s is not None
                 and rows.window_s == steps.window_s else None)
        weights, kv, state = traffic.decode_pass_bytes(context=context, seqs=seqs, verify_width=width)
        parts = (decode.tok_s * weights + prefill_s * traffic.prefill_bytes(chunk), decode.tok_s * kv,
                 decode.tok_s * state)
        if prefill_s > 0:
            floor = PREFILL_FLOOR
        # WHAT THE MODEL CANNOT COUNT (`omits`: TensorFold's drafter passes) makes every figure it draws a floor too.
        floor = "; ".join(f for f in (floor, traffic.omits) if f) or None
        return {"gbps_per_gpu": sum(parts) / GIGA, "weights_gbps": parts[0] / GIGA, "kv_gbps": parts[1] / GIGA,
                "state_gbps": parts[2] / GIGA, "window_s": steps.window_s, "why": None, "floor": floor,
                "decode_passes_s": decode.tok_s, "prefill_passes_s": prefill_s, **gauges}

    def _mfu_evidence(self) -> str | None:
        """What the counters PROVE about `--enable-mfu-metrics` in the process last read: "on" once any of
        them has left 0 (vLLM adds to them only with the switch on, and a new process starts them at 0);
        "off" once that process has produced tokens — so steps completed — while all three are still
        exactly 0; None while neither is proven."""
        now = self._last_scrape
        if now is None:
            return None
        counters = [now.get(key) for key in ("est_flops_total", "est_read_bytes_total",
                                             "est_write_bytes_total")]
        if any(value is not None and value > 0 for value in counters):
            return MFU_ON
        tokens = [now.get(key) for key in ("gen_tokens_total", "prompt_tokens_total")]
        if all(value == 0 for value in counters) and any(value and value > 0 for value in tokens):
            return MFU_OFF
        return None

    def _in_flight(self, evidence: str | None) -> tuple[Reading, Reading, Reading, str | None]:
        """The three readings, and the widening note, for a window in which the counter did not move
        while requests run."""
        if evidence == MFU_OFF:
            gap = Reading(None, MFU_NOT_MEASURED, None)
            return gap, gap, gap, None
        if evidence != MFU_ON:
            gap = Reading(None, MFU_NOT_KNOWN_YET, None)
            return gap, gap, gap, None
        base = self._meters["est_flops"].rise_base(self._idle_floor)
        readings = ([self._meters[name].rate_since(base)
                     for name in ("est_flops", "est_read_bytes", "est_write_bytes")]
                    if base is not None else [None])
        if any(reading is None for reading in readings):
            gap = Reading(None, STEP_IN_FLIGHT, None)
            return gap, gap, gap, None
        flops, read_bytes, written = readings
        return flops, read_bytes, written, self._widened_because(base)

    def _widened_because(self, base_at: float) -> str:
        """Which sentence explains a widened window. Fewer tokens came out over it than there are
        requests running, although a step completed in it: requests are still being READ — a decoding
        request emits at least one token per step."""
        generated = self._meters["generated"].rate_since(base_at)
        running = (self._last_scrape or {}).get("running") or 0
        if generated is not None and round(generated.tok_s * generated.window_s) < running:
            return WIDENED_PREFILL
        return WIDENED_STEP

    def prefill_speed(self) -> dict:
        """The `prefill_speed` figure: the engine's computed rate over the window. The hub samples THIS into its `prefill_tok_s` ring, so the line
        under PREFILL is the number's own past and cannot drift from it."""
        return self._prefill_speed()

    def _read(self, *names: str) -> list[Reading]:
        return [self.readings.get(name, NOT_YET) for name in names]

    def _prompt(self) -> dict:
        if self.prefilling:
            return _unmeasured(CREDITED_AT_FIRST_TOKEN, PROMPT_KEYS, credited="at_first_token")
        prompt, computed, cached = self._read("prompt", "computed", "cached")

        def over_the_same_span(other: Reading) -> bool:
            return other.tok_s is not None and other.window_s == prompt.window_s

        # WHY PREFILL IS BLANK beside a PROMPT figure: the computed counter's own reason (an engine that counts no
        # computed prompt token says not exposed), or — a number over another span — not reached yet.
        computed_why = (None if over_the_same_span(computed) else computed.why if computed.tok_s is None
                        else WINDOW_NOT_REACHED)
        return {"tok_s": prompt.tok_s,
                "computed_tok_s": computed.tok_s if over_the_same_span(computed) else None,
                "cached_share": (cached.tok_s / prompt.tok_s
                                 if prompt.tok_s and over_the_same_span(cached) else None),
                "window_s": prompt.window_s, "credited": "at_first_token", "why": prompt.why,
                "computed_why": computed_why}

    def _prefill_speed(self) -> dict:
        """Computed prefill tokens over prefill seconds, both from requests that FINISHED in the window."""
        tokens, seconds, finished = readings = self._read("prefill_tokens", "prefill_seconds",
                                                          "prefill_requests")
        missing = _first_missing(*readings)
        if missing is not None:
            return _unmeasured(missing.why, PREFILL_KEYS)
        if not tokens.window_s == seconds.window_s == finished.window_s:
            return _unmeasured(WINDOW_NOT_REACHED, PREFILL_KEYS)
        requests = round(finished.tok_s * finished.window_s)
        if requests == 0 or seconds.tok_s <= 0:
            return _unmeasured(WINDOW_NOT_REACHED, PREFILL_KEYS, requests=requests,
                               window_s=tokens.window_s)
        return {"tok_s": tokens.tok_s / seconds.tok_s, "requests": requests,
                "window_s": tokens.window_s, "why": None}

    def speech(self) -> dict:
        """THE SPEECH FIGURES over `SPEECH_WINDOW_S`: audio seconds transcribed per second, how many times faster than
        real time (audio seconds ÷ the seconds the engine spent on those requests), the latency per request, answers
        per minute by outcome; the engine's gauges (seats, queue capacity, running, waiting, GPU memory) and counters
        as last read. Each null with its why: an engine that does not count audio (vLLM) says not exposed."""
        audio, seconds, requests, finished = self._read("audio", "request_seconds", "requests", "finished")
        scrape = self._last_scrape or {}
        speed = latency = None
        speed_why = _first_missing(seconds, requests)
        speed_why = speed_why.why if speed_why is not None else None
        if speed_why is None:
            if seconds.window_s != requests.window_s or requests.tok_s <= 0 or seconds.tok_s <= 0:
                speed_why = WINDOW_NOT_REACHED
            else:
                latency = seconds.tok_s / requests.tok_s
                if audio.tok_s is None or audio.window_s != seconds.window_s:
                    speed_why = audio.why or WINDOW_NOT_REACHED
                else:
                    speed = audio.tok_s / seconds.tok_s
        return {"audio_s_per_s": audio.tok_s, "audio_why": audio.why,
                "speed_x": speed, "latency_s": latency, "speed_why": speed_why,
                "per_min": {word: (None if (r := self.readings.get(name, NOT_YET)).tok_s is None else r.tok_s * 60)
                            for name, word in SPEECH_OUTCOMES.items()},
                "window_s": finished.window_s, "why": finished.why,
                **{k: scrape.get(k) for k in ("seats", "queue_capacity", "running", "waiting", "gpu_memory_bytes")},
                "totals": {"audio_seconds": scrape.get("audio_seconds_total"),
                           **{word: scrape.get(f"requests_{name}_total") for name, word in SPEECH_OUTCOMES.items()}}}

    def _spec_decode(self) -> dict:
        """MTP: the share of drafted tokens accepted, and the tokens a draft step emits."""
        drafts, draft_tokens, accepted = readings = self._read("drafts", "draft_tokens", "accepted")
        missing = _first_missing(*readings)
        if missing is not None:
            return _unmeasured(missing.why, SPEC_DECODE_KEYS)
        if (drafts.tok_s <= 0 or draft_tokens.tok_s <= 0
                or not drafts.window_s == draft_tokens.window_s == accepted.window_s):
            return _unmeasured(WINDOW_NOT_REACHED, SPEC_DECODE_KEYS, window_s=drafts.window_s)
        return {"acceptance_rate": accepted.tok_s / draft_tokens.tok_s,
                "mean_accepted_len": 1 + accepted.tok_s / drafts.tok_s,
                "window_s": drafts.window_s, "why": None}

    def loaded(self, backend: Backend, now: float) -> dict:
        """The `loaded` block: the model a server holds, in its own words."""
        if not backend.last_probe_at:
            return {"state": "unknown", "models": [], "why": FIRST_SAMPLE, "other_client_at": None}
        if not backend.live:
            return {"state": "unknown", "models": [], "why": SERVER_DOWN, "other_client_at": None}
        kv = (self._last_scrape or {}).get("kv_cache_usage")
        return {"state": "serving" if backend.running or backend.inflight else "resident",
                "models": [_model(name=name, context=backend.max_model_len, kv_usage=kv)
                           for name in backend.served_models],
                "why": None, "other_client_at": None}

