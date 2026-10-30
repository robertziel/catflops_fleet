"""The one shared, lock-guarded state between the proxy threads and the control loop.

Everything a decision needs is read through `snapshot()` — a consistent copy — so the
router and scaler stay pure. Mutation goes through the few verbs below; the `Backend`
objects handed out by `backend_for` are the live ones (the prober and the tests set fields
on them), the ones inside a snapshot are copies.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Callable

from vllm_lb import topology
from vllm_lb.config import Config
from vllm_lb.cluster import CLASSIC, CLUSTER_MODE, limits_view
from vllm_lb.cluster_defaults import recommended_for
from vllm_lb.cores import READ_KEYS
from vllm_lb.state import CLUSTER, LEASE_RUNNING, VLLM, Backend, Lease, Snapshot
from vllm_lb.throughput import Throughput

#: A backend's history, one entry per probe the meters took, oldest first. `at` is the LB's clock, so
#: the page can place each sample in a fixed time bucket; `live` says whether a missing rate is a
#: server that was down or one that was not measured; `tok_s` is the old name of `gen_tok_s`, kept
#: until the page has migrated.
HISTORY_RINGS = ("at", "live", "depth", "running", "gen_tok_s", "prompt_tok_s", "tok_s", "kv_pct",
                 "hit_pct", "prefill_tok_s", "computed_tok_s", "steps_s", "tokens_step",
                 "kv_reload_bytes_s", "kv_store_bytes_s", "kv_ram_pct")
#: The engine's own counters, as read, kept per server for the modeled FLOPs (cores.py `modeled_flops`, which
#: proxy.py takes over the CUDA CORES tile's window). Server-side only: the page never receives them.
ENGINE_READS_KEPT = 32


def switch_route(name: str, one: Config) -> str:
    """WHERE a cluster's Start/Stop is POSTed (`clusters[name].switch`): its own route for a cluster this LB drives
    (fleet.CONTROLS `drives`), "" for one it only observes — the panel offers a switch only where this names one."""
    from vllm_lb.fleet import CONTROLS
    return f"/lb/clusters/{name}" if CONTROLS[one.cluster_control].drives else ""


def _desired_mode(view: dict) -> str | None:
    """The operator's desired cluster mode in a published view: `desired.mode`, or None without one."""
    return (view.get("desired") or {}).get("mode")


class Hub:
    def __init__(self, cfg: Config, *, clock: Callable[[], float] = time.time, cluster_spec=None) -> None:
        #: `cluster_spec`: the ModelSpec the cluster switch holds when this process starts — whose pool and
        #: cost per token the pre-tick `recommended` below is worked out for. None: the configured pool.
        self.cfg = cfg
        self.clock = clock
        self._lock = threading.RLock()
        #: WHAT A HELD TURN WAITS ON (proxy `_hold_for_cluster`): bumped and broadcast by every write that can
        #: make a backend routable — a probe result, the cluster loop's view, the leases. A turn holding for
        #: a cluster that is starting sleeps here until the pair is published as held, instead of polling a
        #: snapshot; `_changes` is how a waiter tells a change it already saw from one it has not.
        self._changed = threading.Condition(self._lock)
        self._changes = 0
        #: EVERY cluster's one-cluster Config, by its fleet name (worklog/parallel-clusters.md) — the context each
        #: cluster's loop, view and backend are built from. A cluster the fleet names but does not complete (a pair
        #: of one host) is not configured: no backend, no view, as before.
        self.cluster_cfgs: dict[str, Config] = {c.name: cfg.for_cluster(c.name) for c in cfg.clusters if c.boot}
        self.cluster_cfgs = {n: c for n, c in self.cluster_cfgs.items() if c.cluster_configured}
        configured = (
            [Backend(kind=VLLM, url=b.url, host_label=b.host_label, box=b.box)
             for b in cfg.vllm_boxes]
            + [Backend(kind=CLUSTER, url=c.cluster_url, cluster=name) for name, c in self.cluster_cfgs.items()]
        )
        #: ONE server per URL, as first configured. The same URL listed twice used to make two
        #: backends, one of them never probed: two cards, and a server counted twice.
        self._by_url: dict[str, Backend] = {}
        for configured_backend in configured:
            self._by_url.setdefault(configured_backend.url, configured_backend)
        self._backends: list[Backend] = list(self._by_url.values())
        #: Per-backend rings, oldest first, appended once per PROBE. The page's own sparkline
        #: history died on every reload; this one outlives the tab.
        self._rings: dict[str, dict[str, deque]] = {
            b.url: {name: deque(maxlen=max(1, cfg.stat_history)) for name in HISTORY_RINGS}
            for b in self._backends}
        #: What each server generates and loads, measured (vllm_lb/throughput.py).
        self._throughput = {b.url: Throughput(b.kind, cfg.probe_interval_s) for b in self._backends}
        #: Per backend: (monotonic read time, the engine's counters or None for a probe that read nothing), one
        #: per probe the meters took, oldest first (`engine_reads`).
        self._engine_reads: dict[str, deque] = {b.url: deque(maxlen=ENGINE_READS_KEPT) for b in self._backends}
        self._leases: list[Lease] = []
        #: DEMAND: when a client request last ARRIVED. Only `note_request` writes it.
        self.last_request_at: float | None = None
        #: RETENTION: when this process adopted running leases from the one before it, and which.
        #: Only `note_recovered` writes them; they keep those leases and ask for nothing.
        self.adopted_at: float | None = None
        self.adopted_lease_ids: frozenset[str] = frozenset()
        #: The Stop: when the cluster's desired mode last went `cluster` → `classic` (`set_cluster`).
        self.cluster_stopped_at: float | None = None
        #: Declared clusters NOT in this run whose stored wish wants them up (`set_wanted_off_run`).
        self.wanted_off_run: list[str] = []
        self.last_pressure_at: float | None = None
        #: Timestamps of admissions that could not get an unsaturated leased vLLM while
        #: a capable box was idle. Our own queueing, made visible to the scaler.
        self._starved: list[float] = []
        self.db_ok = True
        self.db_error = ""
        self.requests_total = 0
        self.started_at = clock()
        self.last_tick_at: float | None = None
        self.last_tick_error = ""
        self.context_state = ""
        self.vllm_capacity = None
        self.vllm_capacity_detail = ""
        self.run_next_state = ""
        self.gpu_slots_state = ""
        self.boxes_on = True
        self.boxes_detail = ""
        #: Each cluster as ITS loop last published it, by name. Until its first tick: configured, not observed — and
        #: never held. A node this LB only observes gets the observed view's own shape (vllm_lb/topology.py).
        self.cluster_views: dict[str, dict] = {name: self._initial_view(c, cluster_spec)
                                               for name, c in self.cluster_cfgs.items()}

    @staticmethod
    def _initial_view(cfg: Config, cluster_spec=None) -> dict:
        if cfg.cluster_control in topology.ONE_HOST:
            return topology.observed_view(cfg, state="unknown", detail="not observed yet", held=False,
                                          since=time.time())
        return {"configured": True, "topology": topology.view(cfg),
                "desired": None, "state": "unknown",
                "detail": "not observed yet", "held": False, "loading": {},
                "boxes": {}, "url": cfg.cluster_url,
                "limits": limits_view(),
                # No pair has been SEEN running yet, so none is claimed; and the
                # defaults are resolved by the loop (the box's settings, the catalog).
                "effective": None, "defaults": None,
                "since": time.time(),
                # The stored model's OWN tuned seats, not a hardcoded 16: a page loaded in the
                # ~2 s before the loop's first tick must already be offered — and
                # the router must already admit — the depth that model is offered. No model
                # (the catalog did not load): no seats, so the router admits nothing there.
                "max_depth": (cluster_spec.profile(CLUSTER_MODE).tuned.max_num_seqs
                              if cluster_spec is not None and cluster_spec.profile(CLUSTER_MODE) is not None else None),
                "recommended": recommended_for(cfg, cluster_spec).view() if cluster_spec is not None else None,
                "clamp": "",
                "declared": None, "applier": None}

    @property
    def cluster_view(self) -> dict:
        """THE cluster's view — the one-cluster alias of `cluster_views` (worklog/parallel-clusters.md): the view when
        exactly one cluster is configured, `{"configured": False}` when none, and `{"configured": False, "clusters":
        [names]}` when several — so no reader of the singular can mistake one of several for the whole."""
        if len(self.cluster_views) == 1:
            return next(iter(self.cluster_views.values()))
        if not self.cluster_views:
            return {"configured": False}
        return {"configured": False, "clusters": list(self.cluster_views)}

    # ── reads ──────────────────────────────────────────────────────────────

    def backends(self) -> list[Backend]:
        with self._lock:
            return list(self._backends)

    def cluster_state(self) -> dict:
        """The cluster view as the cluster loop last published it (`set_cluster` replaces it whole, so
        the dict handed out is never mutated under the reader)."""
        with self._lock:
            return self.cluster_view

    def backend_for(self, url: str) -> Backend:
        return self._by_url[url.rstrip("/")]

    def leases(self) -> list[Lease]:
        with self._lock:
            return list(self._leases)

    def snapshot(self, now: float | None = None) -> Snapshot:
        with self._lock:
            return Snapshot(
                now=self.clock() if now is None else now,
                backends=[replace(b) for b in self._backends],
                leases=list(self._leases),
                last_request_at=self.last_request_at,
                last_pressure_at=self.last_pressure_at,
                adopted_at=self.adopted_at,
                adopted_lease_ids=self.adopted_lease_ids,
                cluster_stopped_at=self.cluster_stopped_at,
                db_ok=self.db_ok,
                db_error=self.db_error,
                boxes_on=self.boxes_on,
                vllm_capacity=self.vllm_capacity,
                extra={"requests_total": self.requests_total,
                       "started_at": self.started_at,
                       "last_tick_at": self.last_tick_at,
                       "last_tick_error": self.last_tick_error,
                       "context_state": self.context_state,
                       "vllm_capacity_detail": self.vllm_capacity_detail,
                       "run_next_state": self.run_next_state,
                       "gpu_slots_state": self.gpu_slots_state,
                       "starved_recent": self._starved_count(
                           self.cfg.scale_up_sustain_s, self.clock() if now is None else now),
                       "boxes_detail": self.boxes_detail,
                       "cluster": dict(self.cluster_view),
                       # EVERY active cluster by name, each carrying it and WHERE its switch is posted (its own route
                       # for a cluster this LB drives; "" for one it only observes): what /lb/status.clusters and the
                       # panel's cards read.
                       "clusters": {n: {**v, "name": n,
                                        "switch": switch_route(n, self.cluster_cfgs[n]) if n in self.cluster_cfgs else "",
                                        # THE DEFAULT CLUSTER (fleet `default`): where a request naming no model goes
                                        "default": n == self.cfg.default_cluster}
                                    for n, v in self.cluster_views.items()},
                       # Declared clusters NOT in this run whose stored wish wants them up (cluster_run.publish_wanted):
                       # their engines may run, so "a cluster is wanted" counts them (Snapshot.cluster_wanted).
                       "wanted_off_run": list(self.wanted_off_run)},
            )

    # ── writes ─────────────────────────────────────────────────────────────

    def set_leases(self, leases: list[Lease]) -> None:
        with self._lock:
            self._leases = list(leases)
            self._changed_now()

    # ── change notification (held turns) ───────────────────────────────────

    def _changed_now(self) -> None:
        """Called with the lock held by every write a held turn may be waiting for."""
        self._changes += 1
        self._changed.notify_all()

    def change_version(self) -> int:
        """Read BEFORE the snapshot a waiter decides on, so a change between the two is never slept through."""
        with self._lock:
            return self._changes

    def notify_change(self) -> None:
        """Wake every waiter to look again — the proxy's drain uses it: a turn held for the cluster must
        stop waiting for a pair this process will no longer serve."""
        with self._lock:
            self._changed_now()

    def wait_for_change(self, seen: int, timeout: float) -> int:
        """Sleep until something changed since `seen` (a `change_version`), or `timeout` passed. Returns
        the version now."""
        with self._lock:
            if self._changes == seen and timeout > 0:
                self._changed.wait(timeout)
            return self._changes

    def note_starved(self, at: float | None = None) -> None:
        """One admission of ours settled for worse — or for nothing — while a box that could
        take a vLLM lease sat idle. That is demand for another server, and the only place it is
        visible is here, because the router sends such a turn to the fallback before the vLLM's
        own `waiting` counter ever moves."""
        with self._lock:
            self._starved.append(self.clock() if at is None else at)
            del self._starved[:-200]

    def _starved_count(self, window_s: float, now: float) -> int:
        return sum(1 for t in self._starved if now - t <= window_s)

    def starved_in(self, window_s: float, now: float | None = None) -> int:
        with self._lock:
            return self._starved_count(window_s, self.clock() if now is None else now)

    def note_recovered(self, leases: list) -> None:
        """RETENTION, not demand: the running leases a restart adopts are kept through their idle
        window, counted from now (`scaler.retained_leases`).

        A fresh process has no request on record and nothing in flight — the exact shape of "idle
        past the window" — so its first tick released every lease it had just recovered and
        re-asked two seconds later; under leased-only routing that gap is a 503. The first fix
        stamped `last_request_at` and each adopted box's `last_used_at`, and the scaler read that
        stamp as DEMAND: on 2026-09-15 an adopted fallback lease became a vLLM lease nine minutes
        later, when the GB10 boxes came back, with no client request at all. So adoption records only
        what it is — which leases, and when — and a lease that then goes unused is still released
        once the window has passed.
        """
        with self._lock:
            running = frozenset(l.lease_id for l in leases if l.status == LEASE_RUNNING)
            if running:
                self.adopted_at, self.adopted_lease_ids = self.clock(), running

    def note_request(self) -> None:
        """An inference request ARRIVED (served or not — demand is demand)."""
        with self._lock:
            self.last_request_at = self.clock()
            self.requests_total += 1

    def note_request_start(self, url: str) -> None:
        """Also stamps `last_used_at`: idleness is judged per BOX, not per fleet."""
        with self._lock:
            b = self._by_url[url]
            if b.waiting + max(b.running, b.inflight) == 0:
                b.progress_at = self.clock()        # the stall clock starts when work arrives
            b.inflight += 1
            b.last_used_at = self.clock()

    def note_request_end(self, url: str) -> None:
        with self._lock:
            b = self._by_url[url]
            b.inflight = max(0, b.inflight - 1)
            self._throughput[b.url].request_ended(self.clock())

    def note_usage(self, url: str, verdict) -> None:
        """What UsageTap made of one response the LB relayed to `url` (exactly once per response): a
        Usage is counted, a reason tallied, None — not an accounted shape — changes nothing.

        A server's history stays its own counters' — every client's tokens, never ours laid over them."""
        with self._lock:
            b = self._by_url[url.rstrip("/")]
            now = self.clock()
            throughput = self._throughput[b.url]
            throughput.note_usage(verdict, now)

    def note_response(self, url: str, status: int) -> int:
        """Record a DELIVERED response's status; returns the backend's 5xx streak.

        Only a 5xx counts: a 4xx is the client's fault and says nothing about the backend. Any
        non-5xx clears the streak, so this only ever fires on a backend that is failing
        consistently — not one having a bad minute.
        """
        with self._lock:
            b = self._by_url[url]
            b.failure_streak = b.failure_streak + 1 if status >= 500 else 0
            return b.failure_streak

    def penalise(self, url: str, seconds: float, error: str) -> None:
        with self._lock:
            b = self._by_url[url]
            b.penalty_until = self.clock() + seconds
            b.last_error = error[:200]

    def history(self, url: str) -> dict[str, list]:
        """This backend's recent past, oldest first. Server-side on purpose: the page used to
        accumulate its own sparkline, so a reload erased every stat's history."""
        with self._lock:
            ring = self._rings.get(url.rstrip("/"))
            return ({k: list(v) for k, v in ring.items()} if ring
                    else {name: [] for name in HISTORY_RINGS})

    def engine_reads(self, url: str) -> list[tuple[float, dict | None]]:
        """The engine's own counters at each probe the meters took, oldest first (`ENGINE_READS_KEPT`)."""
        with self._lock:
            return list(self._engine_reads.get(url.rstrip("/"), ()))

    def engine_gauges(self, url: str, keys) -> dict | None:
        """`keys` of the last scrape the meters took for `url`, all from that ONE read — or None when the last probe
        read nothing (a server down, /metrics not answering): never an older read's figures."""
        with self._lock:
            meter = self._throughput.get(url)
            scrape = meter.last_scrape() if meter is not None else None
            return {k: scrape.get(k) for k in keys} if scrape is not None else None

    def generated_window(self, url: str) -> dict | None:
        """`url`'s GENERATED tok/s and the window it ends at (Throughput `generated_window`); None for a server
        this hub does not know."""
        with self._lock:
            meter = self._throughput.get(url)
            return meter.generated_window() if meter is not None else None

    def throughput(self, backend: Backend, now: float, traffic=None) -> dict:
        """The `throughput` block of /lb/status for `backend` (a snapshot's copy of it). `traffic`: the
        catalog's `StepTraffic` when the LB models this server's bandwidth (throughput.py `bandwidth_model`)."""
        with self._lock:
            return self._throughput[backend.url].view(backend, now, self.started_at, traffic=traffic)

    def loaded(self, backend: Backend, now: float) -> dict:
        """The `loaded` block of /lb/status: the model `backend` holds, in its own words."""
        with self._lock:
            return self._throughput[backend.url].loaded(backend, now)

    def update_backend(self, url: str, **fields) -> None:
        """Apply a probe result, and MEASURE throughput across successive probes.

        A server reports monotonic counters, never a rate, so tok/s is a delta — and the hub is the
        only place that sees both samples. Each server's `Throughput` holds one `RateMeter` per
        series and decides what a delta may say: a first sample, a counter reset (a sidecar restarts
        every time a lease is released and re-taken), a probe gap, or a probe that read nothing —
        server down, scrape failed — gives no rate and says why, and 0 is a measurement. A delta's
        time is `mono_at`, when the probe READ the counters; `last_probe_at` is the tick's wall clock.
        """
        scrape = fields.pop("scrape", None)
        mono_at = fields.pop("mono_at", None)
        with self._lock:
            b = self._by_url[url]
            prev_depth = b.waiting + max(b.running, b.inflight)
            new_at = fields.get("last_probe_at")
            for k, v in fields.items():
                setattr(b, k, v)
            if new_at is not None or "live" in fields:      # a probe result: routability may have moved
                self._changed_now()
            # A PROBE result carries `last_probe_at`; the proxy's inflight bookkeeping does not.
            sample = None
            if new_at is not None:
                read_at = time.monotonic() if mono_at is None else mono_at
                sample = self._throughput[b.url].observe(
                    live=b.live, scrape=scrape, at=new_at, inflight=b.inflight, mono_at=read_at)
                if sample is not None:
                    b.tokens_per_s = sample.gen_tok_s
                    # A probe that read nothing is kept as a None, so no interval is taken across it.
                    self._engine_reads[b.url].append(
                        (read_at, {k: scrape.get(k) for k in READ_KEYS} if b.live and scrape else None))
            # PROGRESS, for `stalled_s`: work MOVED (Throughput `moved`) — a generated or credited
            # prompt token, or KV usage changing, which is all a long prefill shows for minutes — and
            # work arriving at an idle backend starts the clock rather than inheriting an old one.
            if b.waiting + max(b.running, b.inflight) == 0:
                b.progress_at = None
            elif ((sample is not None and sample.moved)
                  or prev_depth == 0 or b.progress_at is None):
                b.progress_at = new_at if new_at is not None else self.clock()
            # Sample the ring on PROBE results only — and only on a probe the meters took. The
            # proxy adjusts `inflight` through here around each request; counting those would
            # sample the same instant many times over and draw the shape of our traffic instead
            # of the backend's load.
            if sample is not None:
                # KV OCCUPANCY and the PREFIX HIT RATE, per probe, so the card can draw the pool filling
                # and the cache it evicts on one history. `kv_cache_usage` is the engine's gauge (a
                # fraction); the hit rate is over THIS interval's prompt tokens — cached against computed
                # — never cumulative, so a cache that just started evicting shows it on the next dot.
                kv = scrape.get("kv_cache_usage") if scrape else None
                kv_pct = None if kv is None else round(float(kv) * 100.0, 1)
                cached = scrape.get("prompt_cached_total") if scrape else None
                computed = scrape.get("prompt_computed_total") if scrape else None
                last = getattr(self, "_prompt_last", None)
                if last is None:
                    last = self._prompt_last = {}
                hit_pct = None
                prev = last.get(b.url)
                if prev is not None and cached is not None and computed is not None:
                    dc, dk = cached - prev[0], computed - prev[1]
                    if dc >= 0 and dk >= 0 and dc + dk > 0:
                        hit_pct = round(100.0 * dc / (dc + dk), 1)
                if cached is not None and computed is not None:
                    last[b.url] = (cached, computed)
                # PREFILL SPEED, per probe, so the saturation row's PREFILL has its own past under it:
                # the same `prefill_speed` /lb/status heads it with (Throughput `prefill_speed`) —
                # computed prefill tokens over prefill seconds, from the requests that FINISHED in the
                # last PREFILL_WINDOW_S. It moves only when a request finishes its prefill or leaves the
                # window, so its line is a staircase; an empty window is a gap, never a 0. The billed
                # `prompt_tok_s` is another quantity, with its own ring under PROMPT.
                prefill = self._throughput[b.url].prefill_speed()["tok_s"]
                # ENGINE STEPS a second and tokens a step, over THIS interval (cores.py reads them):
                # a gap when the engine restarted or completed no step between the two probes.
                steps_s = tokens_step = None
                steps = scrape.get("steps_total") if scrape else None
                stoks = scrape.get("step_tokens_sum") if scrape else None
                slast = getattr(self, "_steps_last", None)
                if slast is None:
                    slast = self._steps_last = {}
                sprev = slast.get(b.url)
                if sprev is not None and steps is not None and stoks is not None:
                    dt, dn, dk = new_at - sprev[0], steps - sprev[1], stoks - sprev[2]
                    if dt > 0 and dn > 0 and dk >= 0:
                        steps_s, tokens_step = round(dn / dt, 3), round(dk / dn, 1)
                if steps is not None and stoks is not None:
                    slast[b.url] = (new_at, steps, stoks)
                else:
                    slast.pop(b.url, None)
                ring = self._rings[b.url]
                for name, value in (("at", new_at), ("live", b.live),
                                    # WHAT THE CARD HEADS, so the dots under a number are that
                                    # number's own past: the engine's running + waiting (gateway.js
                                    # `onServer`), and a GAP when /metrics did not answer — the card
                                    # then heads "—", and the zeros probe.py leaves in running and
                                    # waiting are placeholders, not a queue measured empty. Requests
                                    # this LB forwarded that the engine has not admitted are not in
                                    # it: the card stopped heading them (5660b68, "there's no ours
                                    # that is not running nor waiting"). The ROUTER's depth
                                    # (router.py `depth`) stays `waiting + max(running, inflight)` —
                                    # deliberately pessimistic, right for admission control and
                                    # wrong for a drawn history.
                                    ("depth", (b.running + b.waiting) if b.depth_known else None),
                                    # RUNNING ALONE, beside the depth that adds the queue to it: the
                                    # card divides a rate by this to draw what ONE request was getting
                                    # (gateway.js `perRequestOf`). Dividing by the depth would charge a
                                    # queued request for tokens it was not being given.
                                    ("running", b.running if b.depth_known else None),
                                    ("gen_tok_s", sample.gen_tok_s),
                                    ("prompt_tok_s", sample.prompt_tok_s),
                                    ("tok_s", sample.gen_tok_s),
                                    ("kv_pct", kv_pct), ("hit_pct", hit_pct),
                                    ("prefill_tok_s", prefill),
                                    # PREFILL on the card: the prompt tokens the engine COMPUTED per
                                    # second of wall clock — `prompt.computed_tok_s`, PROMPT's own
                                    # window with the cache hits left out. Not `prefill_tok_s`, which
                                    # is one request's pace per second it spent prefilling.
                                    ("computed_tok_s", sample.computed_tok_s),
                                    ("steps_s", steps_s), ("tokens_step", tokens_step),
                                    # THE KV TIER IN THE HOST'S RAM, reloaded and stored, bytes a second: the
                                    # `kv_offload` block's own rates, so the line under it is their past. A
                                    # server without the tier exposes no counter, and its samples are gaps.
                                    ("kv_reload_bytes_s", sample.kv_reload_bytes_s),
                                    ("kv_store_bytes_s", sample.kv_store_bytes_s),
                                    # …and how full it is (`kv_offload.occupancy_pct`, derived), the tile's top graph.
                                    ("kv_ram_pct", sample.kv_ram_pct)):
                    ring[name].append(value)

    def set_db_state(self, ok: bool, error: str = "") -> None:
        with self._lock:
            self.db_ok, self.db_error = ok, error

    def set_pressure(self, at: float | None) -> None:
        with self._lock:
            self.last_pressure_at = at

    def set_live_boxes(self, labels: set[str]) -> None:
        """Which gpu workers are heartbeating, by host_label. Maps onto each backend's box so
        the UI can say "no server here" instead of "down" for a box that is plainly working."""
        boxes = {b.box for b in self.cfg.vllm_boxes if b.host_label in labels}
        with self._lock:
            for b in self._backends:
                b.box_alive = (b.box in boxes) if b.box else None

    def set_parked_boxes(self, labels: set[str]) -> None:
        """Which gpu lanes the operator has PARKED, by host_label.

        Parked is a decision, not a fault: the broker starts nothing on that box, the LB will
        not lease it, and the card must say so in its own words rather than borrowing the red
        one that means "this machine is not answering"."""
        boxes = {b.box for b in self.cfg.vllm_boxes if b.host_label in labels}
        with self._lock:
            for b in self._backends:
                b.parked = (b.box in boxes) if b.box else None

    def set_vllm_capacity(self, count: int | None, detail: str = "") -> None:
        with self._lock:
            self.vllm_capacity, self.vllm_capacity_detail = count, detail

    def set_context_state(self, state: str) -> None:
        with self._lock:
            self.context_state = state

    def set_run_next_state(self, state: str) -> None:
        with self._lock:
            self.run_next_state = state

    def set_gpu_slots_state(self, state: str) -> None:
        with self._lock:
            self.gpu_slots_state = state

    def add_cluster(self, one: Config, cluster_spec=None) -> None:
        """A cluster JOINS the run (cluster_run.ClusterRun.activate): its one-cluster Config, its backend (probed from the
        next tick on, with its own rings and meters), its initial view. ValueError for a name already here or a URL
        another backend has — one server per URL, as at boot."""
        name = one.cluster_name
        with self._lock:
            if name in self.cluster_cfgs:
                raise ValueError(f"cluster {name!r} is already in this run")
            if one.cluster_url in self._by_url:
                raise ValueError(f"cluster {name!r}: {one.cluster_url} is already a backend of this LB")
            b = Backend(kind=CLUSTER, url=one.cluster_url, cluster=name)
            self._by_url[b.url] = b
            self._backends.append(b)
            self._rings[b.url] = {k: deque(maxlen=max(1, self.cfg.stat_history)) for k in HISTORY_RINGS}
            self._throughput[b.url] = Throughput(b.kind, self.cfg.probe_interval_s)
            self._engine_reads[b.url] = deque(maxlen=ENGINE_READS_KEPT)
            self.cluster_cfgs[name] = one
            self.cluster_views[name] = self._initial_view(one, cluster_spec)
            self._changed_now()

    def remove_cluster(self, name: str) -> None:
        """A cluster LEAVES the run (cluster_run.ClusterRun.deactivate, once it is off): its view, config and backend go."""
        with self._lock:
            one = self.cluster_cfgs.pop(name)
            self.cluster_views.pop(name, None)
            b = self._by_url.pop(one.cluster_url, None)
            if b is not None:
                self._backends = [x for x in self._backends if x.url != b.url]
                for table in (self._rings, self._throughput, self._engine_reads):
                    table.pop(b.url, None)
            self._changed_now()

    def set_cluster(self, view: dict, name: str | None = None) -> None:
        """Publish cluster `name`'s loop view (`name` may be left out while ONE cluster is configured). The router
        reads `held`, `desired` and `max_depth`. ValueError for a name among several that is not one of them.

        Also notes the STOP — the desired mode going from `cluster` to `classic` — because classic
        demand counts from there (`scaler.classic_demand_at`)."""
        with self._lock:
            if name is None:
                if len(self.cluster_views) != 1:
                    raise ValueError(f"set_cluster: {len(self.cluster_views)} clusters "
                                     f"({', '.join(self.cluster_views) or 'none configured'}) — name one")
                name = next(iter(self.cluster_views))
            if self.cluster_views and name not in self.cluster_views:
                raise ValueError(f"set_cluster: no cluster {name!r}; the clusters are {', '.join(self.cluster_views)}")
            if _desired_mode(self.cluster_views.get(name) or {}) == CLUSTER_MODE and _desired_mode(view) == CLASSIC:
                self.cluster_stopped_at = self.clock()
            self.cluster_views[name] = dict(view)
            self._changed_now()

    def set_wanted_off_run(self, names: list[str]) -> None:
        """The declared clusters outside this run whose stored wish wants them up (cluster_run.ClusterRun)."""
        with self._lock:
            self.wanted_off_run = list(names)

    def set_boxes_on(self, on: bool, detail: str = "") -> None:
        with self._lock:
            self.boxes_on, self.boxes_detail = on, detail

    def note_tick(self, error: str = "") -> None:
        with self._lock:
            self.last_tick_at = self.clock()
            self.last_tick_error = error
