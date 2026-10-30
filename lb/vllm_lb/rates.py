"""Rates from counters: count nothing rather than something wrong.

A server reports monotonic counters, never a rate, so a rate is a DELTA between two samples — and
only a delta both samples can vouch for (worklog/vllm-gateway-throughput.md, Design 2).
`RateMeter.sample` applies the rules in order, and every "no rate" carries its reason from `WHY`:
the one vocabulary the hub, the usage tap, `/lb/status` and the page all speak.

Pure on purpose. It imports nothing from the LB and reads no clock: the caller stamps each sample
with `time.monotonic()` taken when the value was READ. It keeps full precision; rounding is for
display only.
"""
from __future__ import annotations

from collections import deque
from itertools import islice
from typing import NamedTuple

# ── why a figure is not a number: the closed vocabulary ────────────────────────
SERVER_DOWN = "server down"
FIRST_SAMPLE = "first sample"
COUNTER_RESET = "counter reset"
PROBE_GAP = "probe gap"
SCRAPE_FAILED = "scrape failed"
NOT_EXPOSED = "not exposed by this server"
NO_METRICS_ON_OLLAMA = "no /metrics on ollama"
NO_REQUEST_YET = "no request through the LB yet"
CREDITED_AT_FIRST_TOKEN = "credited at first token"
TOO_SHORT_TO_TIME = "too short to time"
WINDOW_NOT_REACHED = "window not reached"
#: The engine's FLOP counter is still exactly 0 in a process that has already produced tokens. vLLM
#: registers the MFU counters whatever it was started with and feeds them only under
#: `--enable-mfu-metrics`; with the switch on, every completed step adds to them. So this — and ONLY
#: this — is evidence the switch is off. A counter that is merely flat for one window is not
#: (`STEP_IN_FLIGHT`). `throughput.Throughput.engine_perf` tells the cases apart.
MFU_NOT_MEASURED = "the engine is not measuring it (no --enable-mfu-metrics)"
#: Requests are running, the switch is known to be on (the counter has moved in this process), and no
#: step has completed within the work in progress. vLLM adds a step's FLOPs and bytes when the step
#: COMPLETES (`scheduler.update_from_output` runs after the forward pass returns), and one prefill step
#: of 16384 tokens takes ~7 s on this pair — longer than a probe window.
STEP_IN_FLIGHT = ("an engine step is still running: vLLM counts a step's FLOPs and bytes only when the "
                  "step completes")
#: Requests are running, the counter is 0 and the engine has produced no token yet. A first step still
#: running and a switch that is off look the same until a token is out, so neither is claimed.
MFU_NOT_KNOWN_YET = ("not known yet: the engine has produced no token and its FLOP counter is still 0 — "
                     "a first step still running and a missing --enable-mfu-metrics look alike until "
                     "a token is out")

#: Requests are running and no FORWARD PASS completed since the last probe, with no earlier completed pass
#: in the kept history to widen back to. SGLang adds to `sglang:cuda_graph_passes_total` once a pass's
#: batch result is processed (batch_result_processor.py:929-932, metrics_reporter.py:726-729), and one
#: 8,192-token prefill chunk ran 3.4-4.6 s on the live pair (2026-10-03) — most of a 5 s probe window.
#: Read by the LB's modeled bandwidth (`throughput.Throughput.bandwidth_model`), never by vLLM's figures.
PASS_IN_FLIGHT = "a forward pass is still running: SGLang counts a pass only when it completes"

#: Every `why` anything may say. `null` means not measured; a number, 0 included, means measured.
WHY = frozenset({SERVER_DOWN, FIRST_SAMPLE, COUNTER_RESET, PROBE_GAP, SCRAPE_FAILED, NOT_EXPOSED,
                 NO_METRICS_ON_OLLAMA, NO_REQUEST_YET, CREDITED_AT_FIRST_TOKEN, TOO_SHORT_TO_TIME,
                 WINDOW_NOT_REACHED, MFU_NOT_MEASURED, STEP_IN_FLIGHT, MFU_NOT_KNOWN_YET, PASS_IN_FLIGHT})


class Reading(NamedTuple):
    """One sample's verdict: a rate over `window_s` seconds, or `tok_s=None` and `why`."""
    tok_s: float | None
    why: str | None
    window_s: float | None


class RateMeter:
    """One counter on one server.

    `interval_s` is the probe interval the gap rules are judged against. `window_s` makes it a
    windowed figure (PROMPT over 60 s, prefill speed over 5 min): the rate is taken over the newest
    unbroken run of samples, at most that long, and `Reading.window_s` says how long it really was.
    `keep_s` keeps that much of the run even when the window is shorter, so `rise_base` and
    `rate_since` can look further back than the reading does; the reading itself is unchanged.
    """

    def __init__(self, interval_s: float, window_s: float | None = None, keep_s: float = 0.0) -> None:
        self.interval_s = interval_s
        self.window_s = window_s
        self.keep_s = keep_s
        #: (mono_at, value), oldest first: one unbroken run. Its first entry is the base.
        self._run: deque[tuple[float, float]] = deque()
        #: The run's process fingerprint (`process_start_time_seconds`), when the server exposes it.
        self._instance: object | None = None

    def sample(self, value: float | None, mono_at: float, instance: object | None = None, *,
               why_none: str = SCRAPE_FAILED) -> Reading | None:
        """This sample's Reading, or None when it was too close to the last one to be a sample at
        all (rule 4) — the caller then changes nothing either."""
        # 1. Nothing was read (server down, scrape failed, family absent). No rate, and no base:
        #    the next good sample is a first sample, so a gap is never bridged nor a rate repeated.
        if value is None:
            self._start()
            return Reading(None, why_none, None)
        # 2. No base yet.
        if not self._run:
            self._start(mono_at, value, instance)
            return Reading(None, FIRST_SAMPLE, None)
        last_at, last_value = self._run[-1]
        # 3. Another process, or a counter that went backwards: its counts are not this run's.
        #    A fingerprint that appears or vanishes proves nothing, so only two known ones compare.
        restarted = instance is not None and self._instance is not None and instance != self._instance
        if restarted or value < last_value:
            self._start(mono_at, value, instance)
            return Reading(None, COUNTER_RESET, None)
        dt = mono_at - last_at
        # 4. Too close together: a spike from two near-simultaneous reads is not a measurement.
        if dt < 0.5 * self.interval_s:
            return None
        # 5. The tick stalled: no rate across the hole.
        if dt > 3 * self.interval_s:
            self._start(mono_at, value, instance)
            return Reading(None, PROBE_GAP, None)
        # 6. A rate. A value of 0 is a value.
        if instance is not None:
            self._instance = instance
        self._run.append((mono_at, value))
        horizon = mono_at - (self.window_s or 0.0)
        kept = mono_at - max(self.window_s or 0.0, self.keep_s)
        while len(self._run) > 2 and self._run[1][0] <= kept:
            self._run.popleft()               # keep the newest sample at or before the kept horizon
        base_at, base_value = self._base(horizon)
        span = mono_at - base_at
        return Reading((value - base_value) / span, None, span)

    def _base(self, horizon: float) -> tuple[float, float]:
        """The newest sample at or before `horizon`, the newest one itself excluded — or the run's
        oldest, when the run is younger than the window."""
        for at, value in islice(reversed(self._run), 1, None):
            if at <= horizon:
                return at, value
        return self._run[0]

    def rise_base(self, floor_at: float | None = None) -> float | None:
        """The stamp of the kept sample just before the counter last ROSE: where the shortest span that
        ends at the newest sample and holds at least one increase begins. Samples taken before
        `floor_at` are not looked at. None when there is no rise to find."""
        if len(self._run) < 2:
            return None
        newest = self._run[-1][1]
        for at, value in islice(reversed(self._run), 1, None):
            if floor_at is not None and at < floor_at:
                break
            if value < newest:
                return at
        return None

    def rate_since(self, base_at: float) -> Reading | None:
        """The rate from the kept sample stamped `base_at` to the newest one — None when this run holds
        no sample with that stamp (it broke since, or this series was not read then)."""
        if len(self._run) < 2:
            return None
        last_at, last_value = self._run[-1]
        for at, value in self._run:
            if at == base_at and at < last_at:
                span = last_at - at
                return Reading((last_value - value) / span, None, span)
        return None

    def _start(self, mono_at: float | None = None, value: float | None = None,
               instance: object | None = None) -> None:
        self._run.clear()
        self._instance = instance
        if value is not None:
            self._run.append((mono_at, value))
