"""The real-traffic performance log: minute by minute, what the cluster's head engine did while it
served REAL requests — recorded inside the LB, off the /metrics scrape the router already takes.

WHY. The operator stopped benchmarking: "clear benchmark data and collect them on the go when real
cluster runs without benchmarks with real requests". A benchmark is a controlled experiment; real
traffic is not — the mix of prompt lengths, output lengths and concurrency moves from minute to
minute — so this log records WHAT happened under WHICH configuration, and the page that reads it
(/benchmarks) says in words that a difference between two configurations may be the traffic.

A WINDOW. The samples are the probe's own (one every `probe_interval_s`, ~5-6 s). They are cut into
windows on the wall-clock minute (`window_s`) AND at every change of the configuration, so no window
ever mixes two. A counter's increment is taken between two consecutive READS, timed by their
monotonic stamps, and each such pair belongs to the window of its LATER read — so consecutive windows
tile the timeline with no gap and no overlap, and `span_s` is the time the window's pairs covered.
A pair is not counted at all when the reads are more than `MAX_PAIR_S` apart (a probe gap), when
either read saw no engine (down, or its /metrics unreadable), or when the configuration changed
between them. A counter that went DOWN, or a new `process_start_time`, is an engine restart: the
increment is then what the counter has counted since the restart.

WHAT IS KEPT. Only a window with real activity: a request running or waiting, or a token generated or
prefilled (`_active`). An idle minute writes nothing. A kept window is one JSON line (`RECORD_VERSION`):
its span and read count; the configuration it ran under and where that came from (`config_source`);
generated tok/s; prompt tok/s BILLED and COMPUTED (billed minus prefix-cache hits); the prefix hit rate;
requests finished; TTFT and inter-token latency — the mean from the histograms' sum and count, the p90
interpolated inside vLLM's buckets, and the bucket counts themselves so a reader can pool windows
exactly; running, waiting and KV usage (mean, max); THE KV POOL IT RAN UNDER, as the engine reports it
(`kv_pool_tokens`, `kv_pool_bytes`, `kv_max_concurrency`) — recorded, not assumed, because the pin is an
operator knob and a minute has to be self-describing or an A/B across a pin change compares nothing;
speculative acceptance; preemptions; the engine's
own TFLOP/s and GB/s per GPU; and each box's SM fill and power as its own nvidia-smi read them. A figure
nobody measured is null, never 0.

PARALLELISM IS KEPT EXACT. Real traffic's concurrency moves from probe to probe, and performance depends on
it more than on anything else (measured: 147 tok/s aggregate at 8 running vs 184 at 16, per-request speed
the other way round), so a minute that mixed 2 and 12 running requests must not be averaged into one
number. Every window therefore also carries, per PARALLEL BUCKET (`BUCKETS`, the engine's running count),
the sums of its PAIRS that ran at that level (`by_running`): pair count, seconds, request-seconds, tokens,
cache hits, TTFT (mean and bucket counts), inter-token latency, drafts, the engine's FLOPs and bytes, KV,
waiting and each box's SM fill and power — plus the running and waiting count of every read. A pair's
level is the mean of the running count at its two reads; a pair that did nothing is in no bucket. The
page pools these per (configuration, bucket) in the browser: nothing is aggregated on a GPU box, no probe
or SSH call is added, and the sums are the same arithmetic the window already does.

WHERE. `Config.perf_dir` (LB_PERF_DIR, /data/perf — on the state volume the compose file already mounts
from lb/state), one append-only file per UTC day, `perf-YYYY-MM-DD.jsonl`. Bounded twice: a day's file
stops growing at `max_file_bytes`, and once the directory passes `max_total_bytes` its oldest days are
deleted. A restart loses at most the minute in progress (`PerfRecorder.close` writes it on a clean stop).
Beside them, `boots-YYYY-MM-DD.jsonl`: one line per cluster START (`BootRecorder`) — the pair's Start, or an up
intent released to a one-host node's host — the configuration it launched, the seconds to each state it passed and
to each loading phase the LB read, and how it ended — so /benchmarks can say what a configuration costs to bring
up, not only what it does once up, and /lb/sessions which start each minute belongs to (`sessions`).

NEVER IN THE WAY. `observe()` runs in the control loop after each probe: arithmetic on numbers the probe
already parsed, and a `put_nowait` onto a bounded queue. One background thread does all the disk I/O. A
full queue drops the window and counts it; a disk error is logged, counted and shown on the page — the
LB never stops serving, or probing, over its own diary.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import pathlib
import queue
import re
import threading
import time
from typing import Any, NamedTuple

log = logging.getLogger(__name__)

#: Bumped when a record's meaning changes; a reader keeps what it understands of any version.
RECORD_VERSION = 1

#: Two reads further apart than this are a probe GAP, not a pair: whatever the engine did between them
#: cannot be placed inside one minute. Five probe intervals — a slow tick or two is still a pair.
MAX_PAIR_S = 30.0

#: The configuration a window is recorded under: the running pair's LAUNCH values.
#: The settings a window is TAGGED with. The KV pin is one of them: a minute recorded under a 28 GB pool
#: and one recorded under 36 GB are not the same configuration, and pooling them would silently average
#: across the very change an A/B is trying to measure. So is the long-prefill threshold: a minute at 2048 and
#: one with it off share every other setting, and only this key tells them apart (absent on a minute recorded
#: before it existed, when it was off). And WHICH CHECKPOINT: a minute on a model's uncensored twin and one on
#: the original are two configurations (absent before 2026-10-07, when every minute was the original's).
CONFIG_KEYS = ("model_choice", "generation", "max_num_seqs", "max_num_batched_tokens",
               "num_speculative_tokens", "ple_replicate", "kv_cache_memory", "kv_offload_gib",
               "max_model_len", "long_prefill_token_threshold", "uncensored", "kv_cache_dtype")
#: …and what only a ONE-HOST node's `effective` states (topology.py `observed_effective`, its applied intent's
#: declaration over it): its GPU memory fraction and its KV tier in the host's RAM — a minute with a 32 GiB tier and one
#: without are two configurations (2026-10-12: site-b's were recorded with neither, its tier as kv_offload_gib=None).
#: Recorded where the effective carries the key, and ABSENT on the pair's minutes: its effective states neither (its
#: fraction is its catalog profile's, and no RAM tier runs on unified memory), and a null there would claim a live
#: value unknown.
NODE_CONFIG_KEYS = ("gpu_memory_utilization", "kv_offload_ram_gib")
#: `cluster.effective` — what the running pair was actually launched with. The contract with the
#: feat/cluster-knobs branch, which publishes it in /lb/status.
EFFECTIVE = "effective"
#: Until `cluster.effective` exists: `cluster.desired` (generation, max_num_seqs, max_model_len,
#: model_choice) completed from `cluster.defaults` (the same contract) — or, before that is published
#: too, from FALLBACK_DEFAULTS, the values every launch has used so far. Marked on every record, so a
#: window stamped this way is never read as a measured one: until `effective` exists the recorder
#: cannot SEE a speculation or prefill-budget change, only assume it.
FALLBACK = "desired+defaults"
FALLBACK_DEFAULTS = {"max_num_seqs": 8, "max_num_batched_tokens": 8192, "num_speculative_tokens": 4,
                     "max_model_len": 262144}

#: The engine's running count, bucketed: (label, lowest, highest). A pair's level is the mean of the
#: running count at its two reads, rounded; at least 1 when it did any work.
BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9-12", 9, 12), ("13-16", 13, 16),
           ("17-24", 17, 24), ("25+", 25, 10 ** 9))

#: Counters taken as increments between reads (probe.SCRAPE_FAMILIES keys).
COUNTERS = ("gen_tokens_total", "prompt_tokens_total", "prefix_hits_total", "prefix_queries_total",
            "requests_finished_total", "preemptions_total", "spec_drafts_total",
            "spec_draft_tokens_total", "spec_accepted_total", "est_flops_total",
            "est_read_bytes_total", "est_write_bytes_total", "ttft_sum", "ttft_count", "itl_sum",
            "itl_count")
#: Histograms taken as bucket increments (probe.HISTOGRAM_BUCKETS keys).
HISTOGRAMS = ("ttft_buckets", "itl_buckets")

FILE_RE = re.compile(r"^perf-(\d{4})-(\d{2})-(\d{2})\.jsonl$")


def file_re(prefix: str) -> re.Pattern:
    """`<prefix>-YYYY-MM-DD.jsonl` — the perf windows are `perf-`, the cluster boots `boots-`, one dir."""
    return re.compile(rf"^{re.escape(prefix)}-(\d{{4}})-(\d{{2}})-(\d{{2}})\.jsonl$")


#: THE CLUSTER BOOT LOG, beside the perf windows in the same directory: one line per cluster START.
BOOT_PREFIX = "boots"
BOOT_RECORD_VERSION = 1
#: A cluster state a start passes on its way up (cluster.plan). "unknown" is a tick that could not observe,
#: not a step of the start; it is skipped.
BOOT_STATES = frozenset(("checking", "blocked", "freeing", "reserving", "waiting", "restarting", "starting"))
#: What ends an open start, and the outcome it is recorded with.
BOOT_ENDS = {"running": "serving", "failed": "failed", "off": "stopped", "unloading": "stopped",
             "releasing": "stopped", "draining": "stopped"}
#: At most this many windows go to the page per request — 14 days of every minute busy.
MAX_WINDOWS = 20160
#: A malformed line is shown with this much of itself.
EXCERPT = 120


def _num(v) -> float | None:
    """A finite number, or None. A bool is not a number here."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _r(v: float | None, places: int = 3) -> float | None:
    return None if v is None else round(v, places)


def engine_start(v) -> float | None:
    """An engine process's own start (`process_start_time_seconds`, epoch s) as a minute records it and as the
    running session is named by it (/lb/sessions `current`) — one rounding, so the two compare equal."""
    return _r(_num(v), 3)


def _plain(v):
    """A config value as it may be written: a string, true/false, a finite number (an integral one as int),
    or None. BOOL IS CHECKED FIRST and on purpose — `isinstance(True, int)` is true in Python, so the
    numeric path would write the PLE placement as 0/1 and a reader would never get `false` back."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip()[:120] or None
    n = _num(v)
    if n is None:
        return None
    return int(n) if n.is_integer() else n


def effective_config(cluster: Any) -> tuple[dict | None, str]:
    """(the configuration the running pair has, where that came from) — or (None, "") when there is
    no cluster to record. `cluster` is the hub's published cluster view (the `cluster` of /lb/status)."""
    if not isinstance(cluster, dict) or not cluster.get("configured"):
        return None, ""
    eff = cluster.get("effective")
    if isinstance(eff, dict) and eff:
        return {**{k: _plain(eff.get(k)) for k in CONFIG_KEYS},
                **{k: _plain(eff[k]) for k in NODE_CONFIG_KEYS if k in eff}}, EFFECTIVE
    desired = cluster.get("desired")
    if not isinstance(desired, dict):
        return None, ""
    published = cluster.get("defaults") if isinstance(cluster.get("defaults"), dict) else {}
    defaults = {k: (_plain(published.get(k)) if _plain(published.get(k)) is not None else v)
                for k, v in FALLBACK_DEFAULTS.items()}
    config = {k: _plain(desired.get(k)) for k in CONFIG_KEYS}
    # desired never carries the prefill budget or the speculation depth: those are the defaults'.
    config["max_num_batched_tokens"] = defaults["max_num_batched_tokens"]
    config["num_speculative_tokens"] = defaults["num_speculative_tokens"]
    for k in ("max_num_seqs", "max_model_len"):
        if config[k] is None:
            config[k] = defaults[k]
    return config, FALLBACK


#: WHICH WEIGHTS a minute ran: the model AND the checkpoint. An uncensored twin (`models[].uncensored`) is other
#: weights — an abliterated checkpoint, whose tokens a drafter accepts at its own rate — so its minutes are not the
#: original's, and the `model` basis below never crosses from one to the other (2026-10-09: the picker's twin row
#: would otherwise print a rate the twin never ran).
CHECKPOINT_KEYS = ("model_choice", "uncensored")
#: What makes two recorded minutes "the same setup" for a throughput target. `generation` is not in it:
#: it counts restarts, and a restart with the same settings is the same setup.
TARGET_KEYS = CHECKPOINT_KEYS + ("max_num_seqs", "max_num_batched_tokens", "num_speculative_tokens",
                                 "max_model_len")


def _target_value(config: dict, key: str):
    """A setup's value for one TARGET_KEYS key, as two minutes are compared on it. `uncensored` absent or None is
    the ORIGINAL checkpoint: a minute recorded before the twins existed (2026-10-07) ran the only one there was,
    and a launch with None runs the original (`cluster_exec.resolved_launch`)."""
    v = _plain(config.get(key))
    return bool(v) if key == "uncensored" else v
#: A window shorter than this is a cut minute (a config change, a restart): its rate is over too few
#: seconds to be a figure anyone should aim for.
TARGET_MIN_SPAN_S = 30.0


#: The target is the floor of the TOP DECILE of the recorded minutes: the rate that one minute in ten
#: reached. One freak minute (a cache-warm burst) would set a bar nothing else reaches; the single best
#: of fewer than ten minutes is what there is.
TARGET_TOP_SHARE = 0.10


def top_decile(values: list[float]) -> float | None:
    """The lowest value among the top TARGET_TOP_SHARE of `values` — the max when there are fewer than
    ten. None when there are none."""
    xs = sorted((v for v in values if v is not None and v == v), reverse=True)
    if not xs:
        return None
    k = max(1, math.ceil(len(xs) * TARGET_TOP_SHARE))           # how many minutes make the top tenth
    return xs[min(k, len(xs)) - 1]


def throughput_target(windows: list[dict], config: dict | None) -> dict | None:
    """THE RATE TO AIM FOR at this setup: the generated and prompt tok/s that the best tenth of the
    recorded minutes reached under the SAME settings (TARGET_KEYS) — benchmark runs and real traffic
    alike, since both are minutes of this log. With no minute at this exact setup, the same for the same
    MODEL AND CHECKPOINT under any setup (CHECKPOINT_KEYS), and `basis` says which. None when there is nothing to judge against: a tile
    then scales against its own recent peak. A target is measured, never a nameplate, and a minute may
    beat it: the figures shown are real tok/s, not a share."""
    if not isinstance(config, dict) or not config.get("model_choice"):
        return None

    def best(match) -> dict | None:
        gens: list[float] = []
        prompts: list[float] = []
        computeds: list[float] = []
        n = 0
        for w in windows:
            c = w.get("config")
            if not isinstance(c, dict) or not match(c) or (_num(w.get("span_s")) or 0.0) < TARGET_MIN_SPAN_S:
                continue
            n += 1
            g, p = _num(w.get("gen_tok_s")), _num(w.get("prompt_tok_s"))
            k = _num(w.get("prompt_computed_tok_s"))
            if g:
                gens.append(g)
            if p:
                prompts.append(p)
            if k:
                computeds.append(k)
        gen, prompt, computed = top_decile(gens), top_decile(prompts), top_decile(computeds)
        if n == 0 or (not gen and not prompt):
            return None
        # `computed_tok_s`: the PREFILL tile's bar — the prompt tokens the engine COMPUTED per second
        # (a minute's `prompt_computed_tok_s`), which the billed `prompt_tok_s` says nothing about.
        return {"gen_tok_s": _r(gen, 1) if gen else None, "prompt_tok_s": _r(prompt, 1) if prompt else None,
                "computed_tok_s": _r(computed, 1) if computed else None, "windows": n}

    def same(keys):
        return lambda c: all(_target_value(c, k) == _target_value(config, k) for k in keys)

    setup = best(same(TARGET_KEYS))
    if setup:
        return {**setup, "basis": "setup", "config": {k: _target_value(config, k) for k in TARGET_KEYS}}
    model = best(same(CHECKPOINT_KEYS))
    if model:
        return {**model, "basis": "model", "config": {k: _target_value(config, k) for k in CHECKPOINT_KEYS}}
    return None


def bucket_of(running: float) -> str:
    """The parallel bucket a (mean) running count falls in; at least "1"."""
    n = max(1, int(round(running)))
    for label, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return label
    return BUCKETS[-1][0]


def quantile(cumulative: list[tuple[float, float]] | tuple, q: float) -> float | None:
    """Prometheus' `histogram_quantile` over (upper bound, cumulative count) pairs, `+Inf` last: linear
    inside the bucket the rank falls in. A rank in the open-ended bucket returns that bucket's lower
    bound — the largest finite one — because nothing says how far past it the tail went."""
    pairs = [(le, c) for le, c in cumulative if c is not None]
    if not pairs or pairs[-1][1] <= 0:
        return None
    rank = q * pairs[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in pairs:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            if c <= prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


class _Read(NamedTuple):
    at: float
    mono: float
    key: tuple
    scrape: dict


class _Window:
    """The window being filled."""

    def __init__(self, config: dict, source: str, key: tuple, minute: int, t_start: float) -> None:
        self.config, self.source, self.key, self.minute = config, source, key, minute
        self.t_start = self.t_end = t_start
        self.span = 0.0
        self.pairs = 0
        self.reads = 0
        self.restarts = 0
        self.inc = dict.fromkeys(COUNTERS, 0.0)
        self.seen: set[str] = set()
        self.hist: dict[str, dict[float, float]] = {h: {} for h in HISTOGRAMS}
        self.running: list[float] = []
        self.waiting: list[float] = []
        self.kv: list[float] = []
        #: THE KV POOL THIS MINUTE RAN UNDER, as the engine itself reports it (probe.SCRAPE_INFO_LABELS):
        #: tokens, bytes, and vLLM's own max concurrency at this context. Last read wins — the pool is a
        #: constant within an engine, and a window that spans a restart already says so (`engine_restarts`).
        #: RECORDED RATHER THAN ASSUMED because the pin is an operator knob now: a reader that divides by a
        #: hardcoded 28 GB silently misreads every minute recorded under any other pin, and the first thing
        #: anyone will do with the knob is an A/B across it.
        self.pool: dict[str, float] = {}
        #: WHICH ENGINE PROCESS served this minute: its own `process_start_time_seconds` (probe key
        #: `process_start_time`), last read wins — a window across a restart is the new process's (`engine_restarts`
        #: says it spanned one). What a session of a node the LB did not start is keyed by (`sessions`).
        self.engine_started: float | None = None
        self.boxes: dict[str, dict[str, list[float]]] = {}
        #: Per parallel bucket, the sums of the pairs that ran at that level (`_Bucket`).
        self.by: dict[str, "_Bucket"] = {}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _stat(xs: list[float], places: int = 3) -> dict:
    return {"mean": _r(_mean(xs), places), "max": _r(max(xs), places) if xs else None}


def _hist_view(inc: dict, prefix: str, cum: dict[float, float]) -> dict | None:
    """One latency histogram's window: n, mean, p90, and the per-bucket counts ([le, n], +Inf as null)
    a reader pools across windows. None when the engine does not expose it; {"n": 0} when it does and
    nothing completed in the window."""
    n, total = inc.get(f"{prefix}_count"), inc.get(f"{prefix}_sum")
    if n is None:
        return None
    out: dict = {"n": int(n)}
    if n <= 0:
        return out
    out["mean"] = _r(total / n, 4) if total is not None else None
    ordered = sorted(cum.items())
    out["p90"] = _r(quantile(ordered, 0.9), 4) if ordered else None
    per, prev = [], 0.0
    for le, c in ordered:
        if c - prev > 0:
            per.append([None if math.isinf(le) else le, int(round(c - prev))])
        prev = max(prev, c)
    out["buckets"] = per
    return out


class _Bucket:
    """One parallel level's share of a window: sums over its pairs, all additive across windows."""

    __slots__ = ("pairs", "span", "req_s", "inc", "seen", "ttft", "kv_s", "kv_t", "waiting_s", "boxes")

    def __init__(self) -> None:
        self.pairs = 0
        self.span = 0.0
        self.req_s = 0.0
        self.inc = dict.fromkeys(COUNTERS, 0.0)
        self.seen: set[str] = set()
        self.ttft: dict[float, float] = {}
        self.kv_s = self.kv_t = self.waiting_s = 0.0
        self.boxes: dict[str, list[float]] = {}           # box -> [sm_s, sm_t, pwr_s, pwr_t]

    def view(self) -> dict:
        inc = {c: (self.inc[c] if c in self.seen else None) for c in COUNTERS}
        per, prev = [], 0.0
        for le, c in sorted(self.ttft.items()):
            if c - prev > 0:
                per.append([None if math.isinf(le) else le, int(round(c - prev))])
            prev = max(prev, c)
        billed, hits = inc["prompt_tokens_total"], inc["prefix_hits_total"]
        moved = (inc["est_read_bytes_total"], inc["est_write_bytes_total"])
        return {
            "pairs": self.pairs, "span_s": round(self.span, 3), "req_s": round(self.req_s, 3),
            "gen_tokens": _r(inc["gen_tokens_total"], 0), "prompt_tokens": _r(billed, 0),
            "prompt_computed_tokens": None if billed is None or hits is None else _r(max(0.0, billed - hits), 0),
            "prefix_hit_tokens": _r(hits, 0), "prefix_query_tokens": _r(inc["prefix_queries_total"], 0),
            "requests_finished": _r(inc["requests_finished_total"], 0),
            "ttft_n": _r(inc["ttft_count"], 0), "ttft_sum": _r(inc["ttft_sum"], 4), "ttft_buckets": per,
            "itl_n": _r(inc["itl_count"], 0), "itl_sum": _r(inc["itl_sum"], 4),
            "draft_tokens": _r(inc["spec_draft_tokens_total"], 0), "accepted": _r(inc["spec_accepted_total"], 0),
            "flops": _r(inc["est_flops_total"], 0),
            "bytes": None if None in moved else _r(moved[0] + moved[1], 0),
            "kv_s": round(self.kv_s, 4), "kv_t": round(self.kv_t, 3), "waiting_s": round(self.waiting_s, 3),
            "boxes": {box: {"sm_s": round(v[0], 2), "sm_t": round(v[1], 3), "pwr_s": round(v[2], 2),
                            "pwr_t": round(v[3], 3)} for box, v in sorted(self.boxes.items())},
        }


class WindowAccumulator:
    """Reads in, closed windows out. Pure: no clock, no disk, no thread — `observe` returns the windows
    it closed, each as the record to persist, or none when the window was idle."""

    def __init__(self, window_s: float = 60.0) -> None:
        self.window_s = max(1.0, float(window_s))
        self._open: _Window | None = None
        self._prev: _Read | None = None
        self.skipped_idle = 0
        self.last_source = ""

    @property
    def open_window(self) -> _Window | None:
        return self._open

    def observe(self, *, at: float, mono: float | None, live: bool, scrape: Any,
                cluster: Any) -> list[dict]:
        closed: list[dict] = []
        config, source = effective_config(cluster)
        self.last_source = source
        if config is None:                         # no cluster to record: close, forget the last read
            closed += self._close()
            self._prev = None
            return closed
        key = tuple(config.get(k) for k in CONFIG_KEYS + NODE_CONFIG_KEYS)
        minute = math.floor(at / self.window_s)
        if self._open is not None and (self._open.key != key or self._open.minute != minute):
            closed += self._close()
        if not live or not isinstance(scrape, dict) or _num(mono) is None:
            self._prev = None                      # a read that saw no engine: nothing pairs across it
            return closed
        prev = self._prev
        pair = (prev is not None and prev.key == key and 0 < mono - prev.mono <= MAX_PAIR_S)
        if self._open is None:
            self._open = _Window(config, source, key, minute, prev.at if pair else at)
        w = self._open
        w.reads += 1
        w.t_end = at
        self._sample(w, scrape, cluster)
        if pair:
            dt = mono - prev.mono
            w.span += dt
            w.pairs += 1
            delta = self._increments(w, prev.scrape, scrape)
            self._by_running(w, dt, prev.scrape, scrape, delta, cluster)
        self._prev = _Read(at, mono, key, scrape)
        return closed

    def close(self) -> list[dict]:
        """The window in progress, if it had any activity — on a clean stop."""
        out = self._close()
        self._prev = None
        return out

    # ── inside a window ────────────────────────────────────────────────────

    @staticmethod
    def _sample(w: _Window, scrape: dict, cluster: Any) -> None:
        for name, series in (("running", w.running), ("waiting", w.waiting), ("kv_cache_usage", w.kv)):
            v = _num(scrape.get(name))
            if v is not None:
                series.append(v)
        for name in ("kv_pool_tokens", "kv_pool_bytes", "kv_max_concurrency"):
            v = _num(scrape.get(name))
            if v is not None:
                w.pool[name] = v
        started = _num(scrape.get("process_start_time"))
        if started is not None:
            w.engine_started = started
        boxes = cluster.get("boxes") if isinstance(cluster, dict) else None
        for box, view in (boxes or {}).items():
            gpu = view.get("gpu") if isinstance(view, dict) else None
            if not isinstance(gpu, dict):
                continue
            seen = w.boxes.setdefault(str(box), {"sm": [], "pwr": []})
            for name, series in (("sm_pct", seen["sm"]), ("pwr_w", seen["pwr"])):
                v = _num(gpu.get(name))
                if v is not None:
                    series.append(v)

    @staticmethod
    def _increments(w: _Window, before: dict, now: dict) -> dict:
        """Adds one pair's increments to the window and returns them: {counter: delta} for the
        counters it could take, plus {histogram: {le: cumulative delta}}."""
        started = (_num(before.get("process_start_time")), _num(now.get("process_start_time")))
        restarted = None not in started and started[0] != started[1]
        if restarted:
            w.restarts += 1
        delta: dict = {}
        for c in COUNTERS:
            a, b = _num(before.get(c)), _num(now.get(c))
            if b is None or (a is None and not restarted):
                continue                          # not exposed, or exposed only from now on
            d = b - a if a is not None else b
            if restarted or d < 0:
                d = b                             # a restart: what it counted since
            w.inc[c] += d
            w.seen.add(c)
            delta[c] = d
        for h in HISTOGRAMS:
            a, b = before.get(h), now.get(h)
            if not b:
                continue
            prior = dict(a) if a else {}
            if not prior and not restarted:
                continue
            reset = restarted or any(c < prior.get(le, 0.0) for le, c in b)
            acc = w.hist[h]
            got = delta.setdefault(h, {})
            for le, c in b:
                d = max(0.0, c if reset else c - prior.get(le, 0.0))
                acc[le] = acc.get(le, 0.0) + d
                got[le] = d
        return delta

    @staticmethod
    def _by_running(w: _Window, dt: float, before: dict, now: dict, delta: dict, cluster: Any) -> None:
        """One pair's work into its parallel bucket. The level is the mean RUNNING count of the two
        reads — running only, never waiting: a pair with nothing running and no token is an engine
        doing nothing, and it is in no bucket however many requests were queued behind it. (It used
        to fall into bucket "1" on its waiting count alone, and each such pair stretched level 1's
        span with seconds no request ran.) The window still keeps the stall in `waiting`."""
        levels = [v for v in (_num(before.get("running")), _num(now.get("running"))) if v is not None]
        running = sum(levels) / len(levels) if levels else 0.0
        worked = (delta.get("gen_tokens_total") or 0) > 0 or (delta.get("prompt_tokens_total") or 0) > 0
        waiting = [v for v in (_num(before.get("waiting")), _num(now.get("waiting"))) if v is not None]
        if running <= 0 and not worked:
            return
        b = w.by.setdefault(bucket_of(running), _Bucket())
        b.pairs += 1
        b.span += dt
        b.req_s += max(running, 1.0 if worked else 0.0) * dt
        for c in COUNTERS:
            if c in delta:
                b.inc[c] += delta[c]
                b.seen.add(c)
        for le, d in (delta.get("ttft_buckets") or {}).items():
            b.ttft[le] = b.ttft.get(le, 0.0) + d
        kv = _num(now.get("kv_cache_usage"))
        if kv is not None:
            b.kv_s += kv * dt
            b.kv_t += dt
        if waiting:
            b.waiting_s += sum(waiting) / len(waiting) * dt
        boxes = cluster.get("boxes") if isinstance(cluster, dict) else None
        for box, view in (boxes or {}).items():
            gpu = view.get("gpu") if isinstance(view, dict) else None
            if not isinstance(gpu, dict):
                continue
            acc = b.boxes.setdefault(str(box), [0.0, 0.0, 0.0, 0.0])
            sm, pwr = _num(gpu.get("sm_pct")), _num(gpu.get("pwr_w"))
            if sm is not None:
                acc[0] += sm * dt
                acc[1] += dt
            if pwr is not None:
                acc[2] += pwr * dt
                acc[3] += dt

    # ── closing ────────────────────────────────────────────────────────────

    def _close(self) -> list[dict]:
        w, self._open = self._open, None
        if w is None or w.pairs == 0:
            return []
        if not _active(w):
            self.skipped_idle += 1
            return []
        return [record(w)]


def _active(w: _Window) -> bool:
    """Real activity: a request running or waiting at any read, or a token generated or prefilled."""
    return (any(v > 0 for v in w.running) or any(v > 0 for v in w.waiting)
            or w.inc["gen_tokens_total"] > 0 or w.inc["prompt_tokens_total"] > 0)


def record(w: _Window) -> dict:
    """A closed window as the line that is persisted. Every rate is over `span_s`."""
    inc = {c: (w.inc[c] if c in w.seen else None) for c in COUNTERS}
    span = w.span

    def rate(c: str, scale: float = 1.0, places: int = 3):
        return None if inc[c] is None or span <= 0 else _r(inc[c] / span / scale, places)

    billed, hits, queries = inc["prompt_tokens_total"], inc["prefix_hits_total"], inc["prefix_queries_total"]
    computed = None if billed is None or hits is None else max(0.0, billed - hits)
    drafts, accepted = inc["spec_draft_tokens_total"], inc["spec_accepted_total"]
    flops = inc["est_flops_total"]
    moved = inc["est_read_bytes_total"], inc["est_write_bytes_total"]
    worked = (inc["gen_tokens_total"] or 0) > 0 or (computed or 0) > 0
    # The engine's FLOP / byte counters exist whether or not it feeds them (--enable-mfu-metrics): a
    # counter that stayed flat while tokens were produced is "not measured", never 0 TFLOP/s.
    if flops is None:
        mfu = "not exposed"
    elif flops > 0:
        mfu = "measured"
    else:
        mfu = "flat while working" if worked else "idle"
    measured = mfu == "measured"
    return {
        "v": RECORD_VERSION,
        "t_start": round(w.t_start, 3), "t_end": round(w.t_end, 3),
        "span_s": round(span, 3), "reads": w.reads, "pairs": w.pairs,
        "config": dict(w.config), "config_source": w.source,
        "engine_restarts": w.restarts,
        # The engine process's own start (epoch s), null where its /metrics states none. Absent on every minute
        # recorded before 2026-10-12: those are the minutes `sessions` says came "before sessions were recorded".
        "engine_started_at": engine_start(w.engine_started),
        "gen_tokens": _r(inc["gen_tokens_total"], 0), "gen_tok_s": rate("gen_tokens_total"),
        "prompt_tokens": _r(billed, 0), "prompt_tok_s": rate("prompt_tokens_total"),
        "prompt_computed_tokens": _r(computed, 0),
        "prompt_computed_tok_s": None if computed is None or span <= 0 else _r(computed / span),
        "prefix_hit_tokens": _r(hits, 0), "prefix_query_tokens": _r(queries, 0),
        "prefix_hit_rate": None if not queries else _r(hits / queries, 4),
        "requests_finished": _r(inc["requests_finished_total"], 0),
        "ttft": _hist_view(inc, "ttft", w.hist["ttft_buckets"]),
        "itl": _hist_view(inc, "itl", w.hist["itl_buckets"]),
        "running": _stat(w.running, 2), "waiting": _stat(w.waiting, 2), "kv": _stat(w.kv, 4),
        # The pool this minute ran under; absent (null) for a minute whose engine exposed no
        # cache_config_info, and for every minute recorded before this field existed.
        "kv_pool_tokens": _r(w.pool.get("kv_pool_tokens"), 0),
        "kv_pool_bytes": _r(w.pool.get("kv_pool_bytes"), 0),
        "kv_max_concurrency": _r(w.pool.get("kv_max_concurrency"), 3),
        # Every read's own parallelism, in read order: the probe-granular record of concurrency.
        "reads_running": [int(v) for v in w.running], "reads_waiting": [int(v) for v in w.waiting],
        "by_running": {label: w.by[label].view() for label, _, _ in BUCKETS if label in w.by},
        "spec": None if drafts is None else {
            "draft_tokens": int(drafts), "accepted": int(accepted or 0),
            "acceptance": _r(accepted / drafts, 4) if drafts and accepted is not None else None},
        "preemptions": _r(inc["preemptions_total"], 0),
        "tflops_per_gpu": rate("est_flops_total", 1e12) if measured else None,
        "gbps_per_gpu": (_r((moved[0] + moved[1]) / span / 1e9) if measured and None not in moved
                         and span > 0 else None),
        "mfu": mfu,
        "boxes": {box: {"sm_mean": _r(_mean(v["sm"]), 1), "sm_max": _r(max(v["sm"]), 1) if v["sm"] else None,
                        "pwr_mean": _r(_mean(v["pwr"]), 1), "pwr_max": _r(max(v["pwr"]), 1) if v["pwr"] else None,
                        "reads": max(len(v["sm"]), len(v["pwr"]))}
                  for box, v in sorted(w.boxes.items())},
    }


# ── the one thread that touches the disk ─────────────────────────────────────

_STOP = object()


def day_of(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d")


class JsonlWriter:
    """Appends records to `perf-<UTC day>.jsonl` from its own thread. `submit` never blocks and never
    raises; every failure is a count and a sentence (`status`)."""

    def __init__(self, directory: str, *, max_total_bytes: int, max_file_bytes: int,
                 queue_size: int = 1024, prefix: str = "perf") -> None:
        self.dir = pathlib.Path(directory)
        self.prefix = prefix
        self._file_re = file_re(prefix)
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self._q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.written = 0
        self.dropped = 0
        self.capped = 0
        self.errors = 0
        self.deleted = 0
        self.last_error = ""
        self.last_error_at: float | None = None
        self.last_write_at: float | None = None
        self._logged_at = 0.0

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="perf-log-writer", daemon=True)
            self._thread.start()

    def submit(self, rec: dict) -> bool:
        try:
            self._q.put_nowait(rec)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            return False

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._thread = None

    def drain(self) -> None:
        """Write whatever is queued, in this thread — for tests and for `stop` without a thread."""
        while True:
            try:
                rec = self._q.get_nowait()
            except queue.Empty:
                return
            if rec is not _STOP:
                self.write(rec)

    def _run(self) -> None:
        while True:
            rec = self._q.get()
            if rec is _STOP:
                return
            self.write(rec)

    def write(self, rec: dict) -> None:
        try:
            line = json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n"
            path = self.dir / f"{self.prefix}-{day_of(rec['t_start'])}.jsonl"
            self.dir.mkdir(parents=True, exist_ok=True)
            size = path.stat().st_size if path.exists() else 0
            if size + len(line) > self.max_file_bytes:
                with self._lock:
                    self.capped += 1
                self._fail(f"{path.name} reached its {self.max_file_bytes} byte cap; this minute was "
                           "not written")
                return
            # A line cut short by a crash would swallow the next one: start on a fresh line.
            lead = "\n" if size and not _ends_with_newline(path) else ""
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(lead + line)
            with self._lock:
                self.written += 1
                self.last_write_at = time.time()
            self._prune(keep=path.name)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _prune(self, keep: str) -> None:
        """Oldest days first, until the directory is inside its budget. Never the day being written."""
        files = sorted(p for p in self.dir.iterdir() if self._file_re.match(p.name))
        sizes = {p: p.stat().st_size for p in files}
        total = sum(sizes.values())
        for p in files:
            if total <= self.max_total_bytes or p.name == keep:
                break
            p.unlink()
            total -= sizes[p]
            with self._lock:
                self.deleted += 1

    def _fail(self, message: str) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = message[:300]
            self.last_error_at = time.time()
        now = time.monotonic()
        if now - self._logged_at >= 300:          # a full disk must not become a log flood
            self._logged_at = now
            log.warning("[%s] %s", self.prefix, message)

    def status(self) -> dict:
        with self._lock:
            return {"written": self.written, "dropped": self.dropped, "capped": self.capped,
                    "errors": self.errors, "deleted_days": self.deleted,
                    "last_error": self.last_error, "last_error_at": self.last_error_at,
                    "last_write_at": self.last_write_at, "queued": self._q.qsize()}


def _ends_with_newline(path: pathlib.Path) -> bool:
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


class PerfRecorder:
    """The accumulator and the writer, behind the one call the control loop makes (`observe`)."""

    def __init__(self, directory: str, *, window_s: float = 60.0, max_total_bytes: int = 256 << 20,
                 max_file_bytes: int | None = None) -> None:
        self.dir = str(directory)
        self.window_s = window_s
        #: ONE ACCUMULATOR PER SERVER (the view's `name`; "" for a view with none): each engine's counters are its own,
        #: so two servers' minutes are never one engine's deltas, and each closed window names its server (`cluster`).
        self._accs: dict[str, WindowAccumulator] = {}
        self._writer = JsonlWriter(directory, max_total_bytes=max_total_bytes,
                                   max_file_bytes=max_file_bytes or max(1 << 20, max_total_bytes // 8))
        self._lock = threading.Lock()
        self.internal_errors = 0
        self.last_observed_at: float | None = None

    @classmethod
    def from_config(cls, cfg) -> "PerfRecorder | None":
        """The process's recorder, started — or None when switched off. Never raises: a recorder that
        cannot start is a line in the log, not a boot that fails."""
        if not getattr(cfg, "perf_enabled", False):
            return None
        try:
            rec = cls(cfg.perf_dir, window_s=cfg.perf_window_s,
                      max_total_bytes=int(cfg.perf_max_mb * (1 << 20)))
            rec.start()
            return rec
        except Exception as exc:                   # noqa: BLE001 — the diary must never cost the LB
            log.warning("[perf] recorder not started: %s: %s", type(exc).__name__, exc)
            return None

    def start(self) -> None:
        self._writer.start()

    def observe(self, *, at: float, mono: float | None, live: bool, scrape: Any, cluster: Any) -> None:
        try:
            name = cluster.get("name") if isinstance(cluster, dict) else None
            name = name if isinstance(name, str) else ""
            with self._lock:
                acc = self._accs.get(name) or self._accs.setdefault(name, WindowAccumulator(self.window_s))
                closed = acc.observe(at=at, mono=mono, live=live, scrape=scrape, cluster=cluster)
                self.last_observed_at = at
            for rec in closed:
                self._writer.submit({**rec, "cluster": name} if name else rec)
        except Exception:                          # noqa: BLE001 — arithmetic on a surprise shape
            with self._lock:
                self.internal_errors += 1
            log.exception("[perf] a probe could not be recorded")

    def close(self, timeout: float = 5.0) -> None:
        """A clean stop: the minute in progress (when it had activity), then the writer drains."""
        try:
            with self._lock:
                closed = [({**rec, "cluster": name} if name else rec)
                          for name, acc in self._accs.items() for rec in acc.close()]
            for rec in closed:
                self._writer.submit(rec)
        finally:
            self._writer.stop(timeout)

    def status(self) -> dict:
        with self._lock:
            # Per server, by name ("" for a view with none): its minute in progress and the source of its config.
            open_view = {name: {"t_start": round(w.t_start, 3), "t_end": round(w.t_end, 3), "reads": w.reads,
                                "config": dict(w.config), "config_source": w.source}
                         for name, acc in self._accs.items() if (w := acc.open_window) is not None}
            out = {"enabled": True, "dir": self.dir, "window_s": self.window_s,
                   "skipped_idle": sum(acc.skipped_idle for acc in self._accs.values()),
                   "internal_errors": self.internal_errors, "last_observed_at": self.last_observed_at,
                   "config_source": {name: acc.last_source for name, acc in self._accs.items()},
                   "open": open_view}
        return {**out, **self._writer.status()}


# ── the cluster boot log: one line per START ─────────────────────────────────

class BootRecorder:
    """Every cluster START as one JSON line — it reached serving, it failed, or it was stopped — with the
    configuration it launched, so the page pools it under the same settings key as the perf windows.

    Fed once per cluster tick with what the loop publishes (`ClusterLoop.tick` → `observe`): the state,
    its detail, the generation, the configuration as the launch resolves it and `cluster.loading`. A start
    opens when a generation is first seen in a BOOT_STATES state; every state it then passes is stamped
    with the seconds since it was asked (first entry), every loading phase the LB read off the head's log
    likewise; it closes on the first BOOT_ENDS state — or when another generation replaces it (`stopped`).
    Written in the calling thread: a few hundred bytes, a few times an hour, and `JsonlWriter.write` never
    raises. An LB restart forgets a start in progress; nothing is written for it.

    A HUNG ATTEMPT IS A LINE OF ITS OWN (cluster.START_RETRIES): the tick that tears a hung start down to relaunch
    it, and the tick that finally fails one, hand over `hang` (`cluster.StartHang.view()` and where its evidence
    went), which closes the attempt as `failed` with it; the relaunch opens the next line under the same
    generation, numbered by `attempt`. So the hang rate is the lines with a `hang` over the lines.

    THE SAME RECORDER FOR A NODE THE LB REQUESTS (control local, `ClusterLoop._tick_local`): its generation is the
    released intent's id, its configuration the released launch in these keys plus the node's own two
    (NODE_CONFIG_KEYS: its card's share and its KV tier in RAM). A line that opens while the caller KNOWS the start
    already served — that LB saw it serve, then a probe went unanswered — is opened with `read_again`, and the line
    says so (`read_again: true`): no start, whatever the log held before it (`read_again` below)."""

    def __init__(self, directory: str, *, cluster: str = "", max_total_bytes: int = 16 << 20) -> None:
        self.dir = str(directory)
        #: The server whose starts these are (its fleet name): every line says it (`cluster`), as every perf window does.
        self.cluster = cluster
        self._writer = JsonlWriter(directory, prefix=BOOT_PREFIX, max_total_bytes=max_total_bytes,
                                   max_file_bytes=max(1 << 20, max_total_bytes // 8))
        self.open: dict | None = None
        self.internal_errors = 0

    @classmethod
    def from_config(cls, cfg) -> "BootRecorder | None":
        """The process's boot recorder — in the perf log's directory, off with it (LB_PERF=0)."""
        if not getattr(cfg, "perf_enabled", False):
            return None
        return cls(cfg.perf_dir, cluster=getattr(cfg, "cluster_name", "") or "")

    def observe(self, *, now: float, state: str, detail: str, generation: int, config: dict | None,
                config_source: str, loading: dict | None, attempt: int = 1,
                hang: dict | None = None, read_again: bool = False) -> dict | None:
        """One tick's view. `attempt` numbers a start that opens now (1 = the operator's own launch); `hang`
        closes the open one as a hung attempt; `read_again` marks a line that opens now as a start the caller saw
        serve already. Returns the record just closed (and written), else None. Never raises."""
        try:
            return self._observe(now, str(state or ""), str(detail or ""), generation, config,
                                 config_source, loading, attempt, hang, read_again is True)
        except Exception:                          # noqa: BLE001 — the diary must never cost the LB
            self.internal_errors += 1
            log.exception("[boots] a cluster tick could not be recorded")
            return None

    def _observe(self, now, state, detail, generation, config, source, loading, attempt=1,
                 hang=None, read_again=False) -> dict | None:
        if state == "unknown":
            return None
        b = self.open
        closed = None
        if b is not None and generation != b["generation"]:
            closed = self._close(b, now, "stopped", f"replaced by a start of generation {generation}")
            b = None
        if b is None:
            if state not in BOOT_STATES or hang is not None:
                return closed
            b = self.open = {"t_start": now, "generation": generation, "states": {}, "phases": [],
                             "config": None, "config_source": "", "last_state": "",
                             "attempt": attempt if isinstance(attempt, int) and attempt > 0 else 1,
                             "read_again": read_again}
        if isinstance(config, dict) and config:
            # The windows' own keys — and a one-host node's two where its launch states them (absent on the pair's).
            b["config"] = {**{k: _plain(config.get(k)) for k in CONFIG_KEYS},
                           **{k: _plain(config[k]) for k in NODE_CONFIG_KEYS if k in config}}
            b["config_source"] = str(source or "")
        at = round(max(0.0, now - b["t_start"]), 1)
        if state not in b["states"]:
            b["states"][state] = at
        phase = (loading or {}).get("phase") if isinstance(loading, dict) else None
        if isinstance(phase, str) and phase and (not b["phases"] or b["phases"][-1]["phase"] != phase):
            b["phases"].append({"phase": phase[:120], "at_s": at})
        if hang is not None:
            return self._close(b, now, "failed", detail, hang=hang)
        if state in BOOT_ENDS:
            return self._close(b, now, BOOT_ENDS[state], detail)
        b["last_state"] = state
        return closed

    def _close(self, b: dict, now: float, outcome: str, detail: str, hang: dict | None = None) -> dict:
        self.open = None
        failed = outcome != "serving"
        rec = {"v": BOOT_RECORD_VERSION, "kind": "boot", "t_start": round(b["t_start"], 3),
               "t_end": round(now, 3), "generation": b["generation"], "attempt": b.get("attempt", 1),
               "config": b["config"], "config_source": b["config_source"], "outcome": outcome,
               "total_s": round(max(0.0, now - b["t_start"]), 1), "states": b["states"],
               "phases": b["phases"],
               "failure": detail[:300] if failed else None,
               "failed_in": (b["last_state"] or None) if failed else None,
               "failed_phase": (b["phases"][-1]["phase"] if failed and b["phases"] else None),
               "hang": {k: _plain(v) for k, v in hang.items()} if isinstance(hang, dict) else None}
        if self.cluster:
            rec["cluster"] = self.cluster
        if b.get("read_again"):
            rec["read_again"] = True                # written only where true: every older line is a start or not by
        self._writer.write(rec)                     # the earlier-line rule alone (`read_again`)
        return rec

    def status(self) -> dict:
        b = self.open
        return {"dir": self.dir, "internal_errors": self.internal_errors,
                "open": None if b is None else {"t_start": round(b["t_start"], 3), "generation": b["generation"],
                                                "states": dict(b["states"]), "config": b["config"]},
                **self._writer.status()}


#: The state a launch passes (cluster.plan, cluster.plan_local): said on the tick that sends the engine up, and while
#: it comes up — never before (every gate lies between).
LAUNCHED = "starting"


def boot_start(line) -> tuple | None:
    """WHICH START a boot line belongs to: (generation, attempt, model id, twin) — or None for a line that names no
    model or no whole generation. `attempt` absent (before 2026-10-07) is the one launch; `uncensored` null (before the
    twins) launched the original."""
    if not isinstance(line, dict):
        return None
    config = line.get("config")
    model = config.get("model_choice") if isinstance(config, dict) else None
    generation, attempt = line.get("generation"), line.get("attempt")
    if not isinstance(model, str) or not model or not isinstance(generation, int) or isinstance(generation, bool):
        return None
    attempt = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else 1
    return generation, attempt, model, config.get("uncensored") is True


def read_again(line: dict, earlier: set) -> bool:
    """A START READ AGAIN, NOT A START — the one rule the start rates (start_rates.py) and the sessions both keep:
      * a line that says so (`read_again`, BootRecorder): the LB that wrote it had seen that start serve;
      * a line that OPENED in `starting` while an earlier line of the same start (`boot_start`) is in `earlier` — the
        LB finding an engine already up: a redeployed LB beside a serving pair (`starting` → `running` 7 s later;
        generation 177, 2026-10-10: fifteen of them), a serving head that missed a /health probe (173).
    Without either it IS the start: an LB restarted in the middle of one forgets its open line, and the line the new
    LB writes is the only record of how it ended (generation 138). `earlier`: the starts of every line before this
    one, as far back as the log keeps — a pair serving for a month is not started today by today's redeploy."""
    if line.get("read_again") is True:
        return True
    states = line.get("states")
    if not isinstance(states, dict) or not states or not all(_num(v) is not None for v in states.values()):
        return False
    return min(states, key=lambda s: states[s]) == LAUNCHED and boot_start(line) in earlier


# ── reading it back, for the page ────────────────────────────────────────────

def _finite_float(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {text!r}")
    return v


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


class PerfLog:
    """The recorded windows, read off disk for /lb/perf. Parsed files are cached on (mtime, size): a
    closed day never changes, and today's file is re-read only when a window was appended.

    A line that is not a window — cut short by a crash, hand-edited, or from a writer this reader does
    not know — is NOT dropped in silence: it is listed with its file, line and the parser's own words,
    exactly as the saved-runs store lists a killed benchmark."""

    def __init__(self, directory: str, prefix: str = "perf") -> None:
        self.dir = pathlib.Path(directory)
        self.prefix = prefix
        self._file_re = file_re(prefix)
        self._cache: dict[str, tuple[float, int, list[dict], list[dict]]] = {}

    def _files(self) -> tuple[list[pathlib.Path], str]:
        try:
            files = sorted(p for p in self.dir.iterdir() if p.is_file() and self._file_re.match(p.name))
        except FileNotFoundError:
            return [], f"no performance log at {self.dir} yet — nothing has been recorded there"
        except OSError as exc:
            return [], f"cannot read {self.dir}: {exc}"
        return files, ""

    def _read(self, path: pathlib.Path) -> tuple[list[dict], list[dict], str]:
        try:
            st = path.stat()
        except OSError as exc:
            return [], [], f"cannot stat: {exc}"
        hit = self._cache.get(path.name)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2], hit[3], ""
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return [], [], f"cannot read: {exc}"
        windows, bad = [], []
        for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line, parse_float=_finite_float, parse_constant=_reject_constant)
            except ValueError as exc:
                bad.append({"line": i, "error": f"not JSON: {exc}"[:200], "excerpt": line[:EXCERPT]})
                continue
            if (not isinstance(rec, dict) or _num(rec.get("t_start")) is None
                    or _num(rec.get("t_end")) is None):
                bad.append({"line": i, "error": "not a window record (no t_start / t_end)",
                            "excerpt": line[:EXCERPT]})
                continue
            windows.append(rec)
        self._cache[path.name] = (st.st_mtime, st.st_size, windows, bad)
        if len(self._cache) > 400:
            self._cache.clear()
        return windows, bad, ""

    def index(self, days: float | None = 7.0, now: float | None = None, keep=None) -> dict:
        """Every recorded window that ENDED inside the last `days` (None = all of them), oldest first,
        at most MAX_WINDOWS of the newest; plus what could not be read, per file and per line. `cutoff`: the
        range's start (None = everything) — the one definition of it a caller reading another log over the
        same range uses (`sessions(…, since=)`). `keep`: a filter over the lines in range (one server's,
        `of_cluster`), applied BEFORE the MAX_WINDOWS cut and the model counts, so they are that server's own."""
        now = time.time() if now is None else now
        cutoff = None if not days else now - float(days) * 86400.0
        first_day = None if cutoff is None else day_of(cutoff)
        files, error = self._files()
        windows: list[dict] = []
        malformed: list[dict] = []
        listed: list[dict] = []
        for path in files:
            if first_day is not None and path.name < f"{self.prefix}-{first_day}.jsonl":
                continue                           # a whole day before the range
            ws, bad, err = self._read(path)
            listed.append({"file": path.name, "windows": len(ws), "malformed": len(bad), "error": err})
            windows.extend(w for w in ws if cutoff is None or w["t_end"] >= cutoff)
            malformed.extend({"file": path.name, **b} for b in bad)
        if keep is not None:
            windows = keep(windows)
        windows.sort(key=lambda w: (w["t_start"], w["t_end"]))
        truncated = max(0, len(windows) - MAX_WINDOWS)
        # WHOSE minutes the cut ones were: /benchmarks shows one model at a time, and "the N oldest not
        # shown" on a model's page must count that model's minutes only ("" = no model recorded).
        cut: dict[str, int] = {}
        for w in windows[:truncated]:
            cut[window_model(w)] = cut.get(window_model(w), 0) + 1
        return {"dir": str(self.dir), "error": error, "now": now, "days": days, "cutoff": cutoff,
                "windows": windows[-MAX_WINDOWS:], "truncated": truncated, "truncated_by_model": cut,
                # Counted BEFORE the cut: a model whose every minute is older than the newest
                # MAX_WINDOWS is still a model this log recorded, with its real count.
                "models": recorded_models(windows),
                "malformed": malformed[:200], "malformed_total": len(malformed), "files": listed}


def of_cluster(lines, name: str, owners: dict) -> list:
    """THE LINES OF SERVER `name` — perf windows or boot lines, in order. A line names its server (`cluster`, written
    since 2026-10-20); a line from before the name is the server's whose models alone include its model (`owners`:
    {server: names it may run}, router.cluster_models) — a model two servers may run names neither, and such a line
    stays in the unscoped log only, never in one server's figures by a guess."""
    out = []
    for w in lines:
        owner = w.get("cluster")
        if owner is None:
            model = window_model(w)
            # A fleet of ONE server: every line it ever wrote is its own.
            mine = list(owners) if len(owners) == 1 else [n for n, models in owners.items() if model in models]
            owner = mine[0] if len(mine) == 1 else None
        if owner == name:
            out.append(w)
    return out


def window_model(w: dict) -> str:
    """The model id a window was recorded under — "" for a window whose config named none."""
    c = w.get("config")
    m = c.get("model_choice") if isinstance(c, dict) else None
    return m if isinstance(m, str) else ""


def recorded_models(windows) -> list[dict]:
    """EVERY MODEL THIS LOG HOLDS A MINUTE OF, in the order its first minute was recorded:
    {id, minutes, first, last} — `first` the t_start of its oldest window, `last` the t_end of its newest.

    This, and never the catalog, is the list /benchmarks offers (operator 2026-10-12, a one-model deployment's
    page showing four catalog models at "0 MIN": "should not be hardcoded, added on the way when new model loaded"). A
    deployment's models are the ones it RECORDED: a model joins the list the first time a window of it is
    written, so no entry has 0 minutes, a model the catalog does not know is listed under its raw id, and an
    empty log is an empty list. A boot record is not a minute: a model that loaded and served nothing is
    not listed. `windows` are index()'s, oldest first."""
    by: dict[str, dict] = {}
    for w in windows:
        mid = window_model(w)
        m = by.get(mid)
        if m is None:
            m = by[mid] = {"id": mid, "minutes": 0, "first": w["t_start"], "last": w["t_end"]}
        m["minutes"] += 1
        m["first"] = min(m["first"], w["t_start"])
        m["last"] = max(m["last"], w["t_end"])
    return sorted(by.values(), key=lambda m: m["first"])


# ── sessions: one loaded model, from its start to its last busy minute ───────────────────────


#: Bumped when a session's meaning changes, like `RECORD_VERSION` for a window.
SESSION_VERSION = 2


def _acc() -> dict:
    """The sums one session accumulates. Every one is a SUM of what the windows measured — never an
    average of averages: tokens over seconds, hits over queries, accepted over drafted."""
    return {"active_s": 0.0, "windows": 0, "reads": 0, "gen_tokens": 0.0, "prompt_tokens": 0.0,
            "prompt_computed_tokens": 0.0, "prefix_hit_tokens": 0.0, "prefix_query_tokens": 0.0,
            "requests": 0.0, "preemptions": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0,
            "ttft_sum": 0.0, "ttft_n": 0.0, "itl_sum": 0.0, "itl_n": 0.0,
            "running_span": 0.0, "running_sum": 0.0, "running_max": 0.0, "waiting_max": 0.0,
            "kv_max": 0.0, "peak_tok_s": 0.0, "restarts": 0,
            #: which summed counters at least ONE window really carried. A sum over windows that never
            #: carried a counter is "not measured", not 0 — the difference is a price: a session whose
            #: windows lack `prompt_computed_tokens` would otherwise read as "every prompt token was a
            #: cache hit" and be priced at the cache rate.
            "seen": set()}


def _add(a: dict, w: dict) -> None:
    """One window into the sums. A figure the window does not carry is NOT counted as zero: it is left
    out of its own sum, so a session's rate is over the minutes that really measured it."""
    span = _num(w.get("span_s")) or 0.0
    a["windows"] += 1
    a["active_s"] += span
    a["reads"] += int(_num(w.get("reads")) or 0)
    a["restarts"] += int(_num(w.get("engine_restarts")) or 0)
    for key, field in (("gen_tokens", "gen_tokens"), ("prompt_tokens", "prompt_tokens"),
                       ("prompt_computed_tokens", "prompt_computed_tokens"),
                       ("prefix_hit_tokens", "prefix_hit_tokens"),
                       ("prefix_query_tokens", "prefix_query_tokens"),
                       ("requests", "requests_finished"), ("preemptions", "preemptions")):
        v = _num(w.get(field))
        if v is not None:
            a[key] += v
            a["seen"].add(key)
    rate = _num(w.get("gen_tok_s"))
    if rate is not None:
        a["peak_tok_s"] = max(a["peak_tok_s"], rate)
    spec = w.get("spec") if isinstance(w.get("spec"), dict) else None
    if spec:
        a["draft_tokens"] += _num(spec.get("draft_tokens")) or 0.0
        a["accepted_tokens"] += _num(spec.get("accepted")) or 0.0
    for name, sum_key, n_key in (("ttft", "ttft_sum", "ttft_n"), ("itl", "itl_sum", "itl_n")):
        h = w.get(name) if isinstance(w.get(name), dict) else None
        mean, n = (_num(h.get("mean")), _num(h.get("n"))) if h else (None, None)
        if mean is not None and n:
            a[sum_key] += mean * n
            a[n_key] += n
    running = w.get("running") if isinstance(w.get("running"), dict) else {}
    mean = _num(running.get("mean"))
    if mean is not None and span > 0:
        a["running_span"] += span
        a["running_sum"] += mean * span
    for key, holder, field in (("running_max", "running", "max"), ("waiting_max", "waiting", "max"),
                               ("kv_max", "kv", "max")):
        h = w.get(holder) if isinstance(w.get(holder), dict) else {}
        v = _num(h.get(field))
        if v is not None:
            a[key] = max(a[key], v)


def _counted(acc: dict, key: str, session) -> float | None:
    """A session's total of one counter. With NO window at all (a generation that loaded and served
    nothing) the honest total is 0. With windows that never carried the counter it is None — the log
    did not measure it, and 0 would be a claim."""
    if session is not None and key not in acc["seen"]:
        return None
    return _r(acc[key], 0)


def _add_boxes(boxes: dict, w: dict) -> None:
    """Each box's own SM fill and power, span-weighted — the window already averaged its reads."""
    span = _num(w.get("span_s")) or 0.0
    seen = w.get("boxes") if isinstance(w.get("boxes"), dict) else {}
    for box, v in seen.items():
        if not isinstance(v, dict):
            continue
        b = boxes.setdefault(box, {"sm_span": 0.0, "sm_sum": 0.0, "pwr_span": 0.0, "pwr_sum": 0.0,
                                   "sm_max": None, "pwr_max": None})
        for field, span_key, sum_key, max_key in (("sm_mean", "sm_span", "sm_sum", None),
                                                  ("pwr_mean", "pwr_span", "pwr_sum", None)):
            x = _num(v.get(field))
            if x is not None and span > 0:
                b[span_key] += span
                b[sum_key] += x * span
        for field, key in (("sm_max", "sm_max"), ("pwr_max", "pwr_max")):
            x = _num(v.get(field))
            if x is not None:
                b[key] = x if b[key] is None else max(b[key], x)


def _box_view(boxes: dict) -> dict:
    out = {}
    for box, b in sorted(boxes.items()):
        out[box] = {"sm_mean": _r(b["sm_sum"] / b["sm_span"], 1) if b["sm_span"] else None,
                    "sm_max": _r(b["sm_max"], 1),
                    "pwr_mean": _r(b["pwr_sum"] / b["pwr_span"], 1) if b["pwr_span"] else None,
                    "pwr_max": _r(b["pwr_max"], 1)}
    return out


def _watts(boxes: dict) -> float | None:
    """What the PAIR drew while it worked: the boxes' mean power added up. None when no box said."""
    means = [b["pwr_sum"] / b["pwr_span"] for b in boxes.values() if b["pwr_span"]]
    return sum(means) if means else None


def _session_config(variants: dict) -> tuple[dict, int]:
    """THE configuration a session ran: the one its busiest minutes ran under, and how many distinct
    ones the log saw for it. A generation is launched ONCE, but `cluster.effective` takes the engine's
    own word for a knob it reports differently (generation 135: 2,048 sent, 8,192 run), so a session can
    honestly hold more than one — and the page must be able to say so instead of picking in silence."""
    best = max(variants.items(), key=lambda kv: kv[1]["span"])
    return best[1]["config"], len(variants)


def _session_key(w: dict) -> tuple | None:
    """WHICH SESSION a minute belongs to: (generation, None, model) where the LB started what served it — the pair's
    Start, or the intent a node's host applied (`generation`, topology `observed_view`) — else (None, the engine
    process's own start, model): an engine the LB did not start, known by its `process_start_time_seconds`
    (`engine_started_at`). None for a minute that names no model, or neither: nothing recorded says whose it was."""
    c = w.get("config") if isinstance(w.get("config"), dict) else None
    model = c.get("model_choice") if c else None
    if not isinstance(model, str) or not model:
        return None
    gen = _num(c.get("generation"))
    if gen is not None:
        return int(gen), None, model
    started = _num(w.get("engine_started_at"))
    return None if started is None else (None, started, model)


def sessions(windows, boots=(), *, describe=None, since: float | None = None) -> dict:
    """EVERY LOADED-MODEL SESSION the log remembers, newest first.

    A session is ONE START of the model: one generation of the pair, or one intent a node's host applied
    (`generation` either way), from its first busy minute to its last — or, for an engine the LB did not start
    (a node it only observes, or one running the fleet's default before any intent), ONE ENGINE PROCESS, known by
    its own start (`engine_started_at`, `generation` null): no start is invented for it. The windows are the minutes
    it served (`record`), the boot records what it cost to bring up (`BootRecorder`), and everything here is a SUM
    over them — tokens, requests, seconds, drafts, hits — so a session's rate is the tokens it generated over the
    seconds it was busy, never a mean of per-minute means.

    `boots`: every boot line in range, oldest first — and, with `since` (the range's start, `PerfLog.index`
    `cutoff`), the lines before it too: those make no session, they only tell a start READ AGAIN in the range from a
    start (`read_again`, the start rates' own rule). A read again is never an attempt nor a boot time.

    `describe(model_choice)` -> dict is the catalog's say about a model (its label, its engine): the log
    stores an id, and an id is not a name. Left out, a session carries the id alone.

    A generation with a boot record but no busy minute is still a session (it loaded and served nothing),
    and one whose minutes were recorded before boots existed is a session without a boot. Both happen. A minute
    that names neither a start nor its engine's is in no session and is COUNTED (`windows_without_a_session`);
    those of them recorded before minutes named their engine's start (no `engine_started_at` field at all — every
    minute before 2026-10-12) are `windows_before_sessions`.
    """
    by_gen: dict[tuple, dict] = {}
    skipped = before = 0
    for w in windows:
        key = _session_key(w)
        if key is None:
            skipped += 1
            before += "engine_started_at" not in w
            continue
        c = w["config"]
        s = by_gen.get(key)
        if s is None:
            s = by_gen[key] = {"acc": _acc(), "boxes": {}, "variants": {}, "pool": None,
                               "first": None, "last": None}
        _add(s["acc"], w)
        _add_boxes(s["boxes"], w)
        stamp = json.dumps(c, sort_keys=True, default=str)
        v = s["variants"].setdefault(stamp, {"config": c, "span": 0.0})
        v["span"] += _num(w.get("span_s")) or 0.0
        pool = _num(w.get("kv_pool_tokens"))
        if pool:
            s["pool"] = pool if s["pool"] is None else max(s["pool"], pool)
        t0, t1 = _num(w.get("t_start")), _num(w.get("t_end"))
        if t0 is not None:
            s["first"] = t0 if s["first"] is None else min(s["first"], t0)
        if t1 is not None:
            s["last"] = t1 if s["last"] is None else max(s["last"], t1)

    # The starts, per generation: how long it took to come up, how it ended, how many attempts it needed.
    starts: dict[tuple, dict] = {}
    earlier: set = set()
    for b in boots or ():
        if not isinstance(b, dict):
            continue
        again = read_again(b, earlier)
        start = boot_start(b)
        if start is not None:
            earlier.add(start)
        if since is not None and (_num(b.get("t_end")) or 0.0) < since:
            continue                                 # before the range: read only to know a start from a re-read
        c = b.get("config") if isinstance(b.get("config"), dict) else {}
        gen, model = _num(b.get("generation")), c.get("model_choice")
        if gen is None or not isinstance(model, str) or not model:
            continue
        key = (int(gen), None, model)
        at = starts.setdefault(key, {"attempts": 0, "first_t": None, "boot_s": None, "outcome": "",
                                     "failure": "", "config": c})
        t0 = _num(b.get("t_start"))
        if t0 is not None:
            at["first_t"] = t0 if at["first_t"] is None else min(at["first_t"], t0)
        # The LAST record of a generation is how it really ended: a failed attempt that was rescued is
        # not the outcome, it is the cost — and a pair killed after it served (173) ended failed.
        at["outcome"] = str(b.get("outcome") or at["outcome"])
        at["failure"] = str(b.get("failure") or "")[:300]
        if again:
            continue                                 # an engine already up, looked at again: no attempt, no boot time
        at["attempts"] += 1
        total = _num(b.get("total_s"))
        if total is not None and str(b.get("outcome")) == "serving":
            at["boot_s"] = total

    out = []
    for key in set(by_gen) | set(starts):
        gen, engine, model = key
        s = by_gen.get(key)
        boot = starts.get(key)
        acc = s["acc"] if s else _acc()
        boxes = s["boxes"] if s else {}
        config, variants = (_session_config(s["variants"]) if s and s["variants"]
                            else ((boot or {}).get("config") or {}, 1 if boot else 0))
        # It stood from its START: the boot record's (when it was asked), else — for an engine the LB did not
        # start — the engine's own process start, else its first busy minute.
        started = min([t for t in ((s or {}).get("first"), (boot or {}).get("first_t"), engine) if t is not None],
                      default=None)
        ended = (s or {}).get("last")
        watts = _watts(boxes)
        active = acc["active_s"]
        # ENERGY, from the unrounded figure: the row's `wh` is rounded to 0.01 for display and a
        # throughput per kWh divided by a rounded number would carry that rounding into a ratio.
        energy_wh = watts * active / 3600.0 if watts is not None and active > 0 else None
        row = {
            "v": SESSION_VERSION, "generation": gen, "model_choice": model,
            # The engine process a session with no generation IS (null on a session the LB started).
            "engine_started_at": _r(engine, 3),
            "started_at": _r(started, 3), "ended_at": _r(ended, 3),
            # TWO CLOCKS, and they mean different things: `wall_s` is how long the session stood,
            # `active_s` the seconds of it that were BUSY — the log records nothing for an idle minute,
            # so a rate over wall time would price the silence in.
            "wall_s": _r(ended - started, 1) if started is not None and ended is not None else None,
            "active_s": _r(active, 1), "windows": acc["windows"], "reads": acc["reads"],
            "config": config, "config_variants": variants,
            "kv_pool_tokens": _r(s["pool"], 0) if s else None,
            "gen_tokens": _counted(acc, "gen_tokens", s),
            "prompt_tokens": _counted(acc, "prompt_tokens", s),
            "prompt_computed_tokens": _counted(acc, "prompt_computed_tokens", s),
            "requests": _counted(acc, "requests", s),
            "gen_tok_s": _r(acc["gen_tokens"] / active, 2) if active > 0 else None,
            "peak_gen_tok_s": _r(acc["peak_tok_s"], 2) or None,
            "prompt_tok_s": _r(acc["prompt_tokens"] / active, 2) if active > 0 else None,
            "prefix_hit_rate": (_r(acc["prefix_hit_tokens"] / acc["prefix_query_tokens"], 4)
                                if acc["prefix_query_tokens"] else None),
            "tokens_per_request": (_r(acc["gen_tokens"] / acc["requests"], 1) if acc["requests"] else None),
            "ttft_mean_s": _r(acc["ttft_sum"] / acc["ttft_n"], 3) if acc["ttft_n"] else None,
            "itl_mean_s": _r(acc["itl_sum"] / acc["itl_n"], 4) if acc["itl_n"] else None,
            "running_mean": (_r(acc["running_sum"] / acc["running_span"], 2)
                             if acc["running_span"] else None),
            "running_max": _r(acc["running_max"], 0), "waiting_max": _r(acc["waiting_max"], 0),
            "kv_max": _r(acc["kv_max"], 4),
            "acceptance": (_r(acc["accepted_tokens"] / acc["draft_tokens"], 4)
                           if acc["draft_tokens"] else None),
            "preemptions": _r(acc["preemptions"], 0), "engine_restarts": acc["restarts"],
            "boxes": _box_view(boxes),
            # The PAIR's draw while it worked, and the energy that is: watts x busy seconds. Null where
            # no box reported power — never 0, which would claim a measurement of nothing.
            "watts_mean": _r(watts, 1),
            "wh": _r(energy_wh, 2),
            #: The same energy UNROUNDED (`wh` is for display, to 0.01): every aggregate that divides by
            #: energy — tokens per kWh, over a session or a whole history — must divide by this, or the
            #: rounding of each row is carried into the ratio and the total disagrees with its own rows.
            "energy_wh": _r(energy_wh, 6),
            # TOKENS GENERATED PER kWh the boxes drew while this session was busy — the efficiency figure.
            # It needs BOTH sides measured: no power reading is "not measured", not infinite efficiency, and
            # a session that generated nothing has no ratio rather than 0 tokens per kWh.
            "tok_per_kwh": _tok_per_kwh(_counted(acc, "gen_tokens", s), energy_wh),
            "boot_s": _r((boot or {}).get("boot_s"), 1),
            "boot_attempts": (boot or {}).get("attempts", 0),
            "outcome": (boot or {}).get("outcome", ""),
            "failure": (boot or {}).get("failure", ""),
        }
        if describe is not None:
            row.update(describe(model) or {})
        out.append(row)
    # NEWEST FIRST, by when each started: a generation and an engine process are on one clock, and the pair's
    # generations are numbered in the order they were started.
    out.sort(key=lambda r: (_num(r["started_at"]) or 0.0, r["generation"] if r["generation"] is not None else -1),
             reverse=True)
    return {"sessions": out, "totals": _totals(out), "by_model": _by_model(out),
            "windows_without_a_session": skipped, "windows_before_sessions": before}


def _tok_per_kwh(tokens, wh) -> float | None:
    """Generated tokens per kilowatt-hour: tokens over (Wh / 1000). None unless both are real and positive."""
    if not (_num(tokens) and _num(wh)) or tokens <= 0 or wh <= 0:
        return None
    return _r(tokens / (wh / 1000.0), 0)


def _efficiency(rows: list[dict]) -> dict:
    """tokens per kWh over a set of sessions — a RATIO OF SUMS, taken only over the sessions where both are
    measured, so a session with tokens but no power reading cannot inflate the numerator of a ratio whose
    denominator never saw it. `of` is the sessions that generated anything: the figure says how many of them
    it could be computed for."""
    both = [r for r in rows if _num(r.get("energy_wh")) and r["energy_wh"] > 0 and _num(r.get("gen_tokens"))
            and r["gen_tokens"] > 0]
    wh = sum(r["energy_wh"] for r in both)
    tokens = sum(r["gen_tokens"] for r in both)
    return {"tok_per_kwh": _tok_per_kwh(tokens, wh) if both else None, "measured": len(both),
            "of": sum(1 for r in rows if (_num(r.get("gen_tokens")) or 0) > 0)}


def _totals(rows: list[dict]) -> dict:
    """The whole history in one line — what the tiles above the table say."""
    def total(field):
        return sum(_num(r.get(field)) or 0.0 for r in rows)

    active = total("active_s")
    wh = [r for r in rows if _num(r.get("wh")) is not None]
    return {"sessions": len(rows), "models": len({r["model_choice"] for r in rows}),
            "gen_tokens": _r(total("gen_tokens"), 0), "prompt_tokens": _r(total("prompt_tokens"), 0),
            "requests": _r(total("requests"), 0), "active_s": _r(active, 1),
            "wall_s": _r(total("wall_s"), 1),
            "gen_tok_s": _r(total("gen_tokens") / active, 2) if active > 0 else None,
            "wh": _r(sum(_num(r["wh"]) for r in wh), 1) if wh else None,
            "efficiency": _efficiency(rows),
            "first_at": min((_num(r.get("started_at")) for r in rows
                             if _num(r.get("started_at")) is not None), default=None),
            "last_at": max((_num(r.get("ended_at")) for r in rows
                            if _num(r.get("ended_at")) is not None), default=None)}


def _by_model(rows: list[dict]) -> list[dict]:
    """Per MODEL, over every session of it: what it served and how fast, most tokens first."""
    acc: dict[str, dict] = {}
    for r in rows:
        m = acc.setdefault(r["model_choice"], {"model_choice": r["model_choice"], "label": r.get("label"),
                                               "engine": r.get("engine"), "sessions": 0,
                                               "gen_tokens": 0.0, "prompt_tokens": 0.0, "requests": 0.0,
                                               "active_s": 0.0, "wh": 0.0, "wh_seen": False})
        m["sessions"] += 1
        for key, field in (("gen_tokens", "gen_tokens"), ("prompt_tokens", "prompt_tokens"),
                           ("requests", "requests"), ("active_s", "active_s")):
            m[key] += _num(r.get(field)) or 0.0
        if _num(r.get("wh")) is not None:
            m["wh"] += _num(r["wh"])
            m["wh_seen"] = True
    out = []
    for m in acc.values():
        out.append({**{k: v for k, v in m.items() if k != "wh_seen"},
                    "efficiency": _efficiency([r for r in rows if r["model_choice"] == m["model_choice"]]),
                    "gen_tokens": _r(m["gen_tokens"], 0), "prompt_tokens": _r(m["prompt_tokens"], 0),
                    "requests": _r(m["requests"], 0), "active_s": _r(m["active_s"], 1),
                    "wh": _r(m["wh"], 1) if m["wh_seen"] else None,
                    "gen_tok_s": _r(m["gen_tokens"] / m["active_s"], 2) if m["active_s"] > 0 else None})
    out.sort(key=lambda m: -(m["gen_tokens"] or 0))
    return out
