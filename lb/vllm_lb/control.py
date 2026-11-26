"""The control loop — probe, refresh leases, measure pressure, plan, apply. Never raises.

One tick is four fenced phases; a failure in one is logged and the others still run, and
the proxy thread never waits on any of it. Creating a lease backs off exponentially after a
failure (a broker outage must not become a write storm); releasing retries every tick until
the broker confirms it.

With the broker OFF (`Config.broker_enabled`) a tick is the probe and the cluster's loop only: `leases` is
None, and the engine, the leases, the pressure, the boxes and the plan are never touched.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable

from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases
from vllm_lb.scaler import pressure_sample, Create, PressureTracker, Release, plan_actions, serving_now
from vllm_lb.state import CLUSTER, LEASE_LOST, LEASE_RUNNING, VLLM, Snapshot


def cluster_wanted(snap: Snapshot) -> bool:
    """Is ANY active cluster wanted (`Snapshot.cluster_wanted`, over every view by name)? While one is, the classic
    vLLM planner stands aside — a lease it asks for could land on a box a cluster is about to reserve. Read off the
    views, never the singular alias, which names no cluster once two run (worklog/parallel-clusters.md R7)."""
    return snap.cluster_wanted()

log = logging.getLogger(__name__)

_BACKOFF_BASE_S = 5.0
#: Probes out at once, at most — one per backend is the rule (a backend is never probed twice at once), and a fleet
#: has a handful; this only bounds a fleet file that declares very many.
_PROBE_WORKERS = 32


class Controller:
    def __init__(self, cfg: Config, hub: Hub, leases: BrokerLeases | None, *,
                 prober: Callable, clock: Callable[[], float] = time.time,
                 ensure_engine: Callable[[], str] | None = None, cluster_loop=None,
                 recorder=None) -> None:
        self.cfg = cfg
        self.hub = hub
        self.leases = leases
        self.prober = prober
        self.clock = clock
        self.backoff_until = 0.0
        self._fail_count = 0
        self._pressure = PressureTracker(threshold=cfg.scale_up_waiting,
                                         sustain_s=cfg.scale_up_sustain_s)
        #: `ensure_engine()` returns "" once the broker engine is importable + configured,
        #: else the reason. Until it succeeds the loop probes and proxies but does not talk
        #: to the broker; on the first success it recovers our open leases. A missing
        #: /opt/broker_parrot mount at boot is therefore a status line, not a crash loop.
        self.ensure_engine = ensure_engine
        self.engine_ready = ensure_engine is None
        #: The TP=2 cluster's loop (`cluster_loop.ClusterLoop`), or None when not configured. It
        #: runs after the boxes phase and before the plan (which must see its view).
        self.cluster_loop = cluster_loop
        #: The real-traffic performance log (`perflog.PerfRecorder`): shown the cluster head's probe
        #: result after each probe, outside the hub's lock. None = nothing is recorded.
        self.recorder = recorder
        #: THE PROBES run on their own threads and a tick waits for them at most one tick (`_probe`): the
        #: backends whose probe is still out, the lock their results land under, and the last landing that failed —
        #: raised by the next tick's probe phase, so it is named in `last_tick_error` like any phase's failure.
        self._probes = ThreadPoolExecutor(max_workers=_PROBE_WORKERS, thread_name_prefix="vllm-lb-probe")
        self._probing: set[str] = set()
        self._probing_lock = threading.Lock()
        self._landing = threading.Lock()
        self._landing_error = ""

    # ── lifecycle ──────────────────────────────────────────────────────────

    def recover(self) -> None:
        recovered = self.leases.recover()
        self.hub.set_leases(recovered)
        # A restart is not idleness: what we just adopted was in use a moment ago, and the
        # idle clock must start NOW rather than judge the fleet idle on the first tick.
        self.hub.note_recovered(recovered)
        self.hub.set_db_state(self.leases.db_ok, self.leases.last_error)

    def run_forever(self, stop: threading.Event) -> None:
        if self.cfg.broker_enabled:
            log.info("[control] loop up: tick every %.1fs, release after %.0fs idle, max %d lease(s)",
                     self.cfg.tick_s, self.cfg.idle_release_s, self.cfg.max_leases)
        else:
            log.info("[control] loop up: tick every %.1fs; broker_parrot off (LB_BROKER_ENABLED unset) — "
                     "no leases, every live configured backend is routable", self.cfg.tick_s)
        # THE PERIOD IS `tick_s`: what is LEFT of it after the tick is slept, so a 1 s tick is one event a
        # second on /lb/stream. A tick longer than `tick_s` (its probes wait up to `tick_s`, its other phases add
        # theirs) is followed at once by the next — a period of the tick's own length, never tick + tick_s.
        while not stop.is_set():
            started = time.monotonic()
            self.tick()
            stop.wait(max(0.0, self.cfg.tick_s - (time.monotonic() - started)))

    # ── one tick ───────────────────────────────────────────────────────────

    def tick(self) -> None:
        now = self.clock()
        cluster = [("cluster", self._cluster_phase)] if self.cluster_loop else []
        if not self.cfg.broker_enabled:
            # BROKER OFF (`LB_BROKER_ENABLED` unset): probe, and run the cluster's loop — nothing else. No engine
            # is set up, no lease is read, asked for or released, and no broker state is published: there is
            # none. The router then routes to whatever configured backend is live (router.py).
            self.hub.note_tick("; ".join(self._run([("probe", self._probe), *cluster], now))[:300])
            return
        phases = [("probe", self._probe)]
        engine_error = self._engine_phase()
        if engine_error is None:
            phases += [("leases", self._refresh_leases), ("pressure", self._pressure_phase),
                       ("boxes", self._boxes_phase), *cluster, ("plan", self._plan_and_apply)]
        errors = self._run(phases, now)
        if engine_error is None:
            self.hub.set_db_state(self.leases.db_ok, self.leases.last_error)
            self.hub.set_context_state(self.leases.context_state)
            self.hub.set_run_next_state(self.leases.run_next_state)
            self.hub.set_gpu_slots_state(self.leases.gpu_slots_state)
        else:
            self.hub.set_db_state(False, f"engine unavailable: {engine_error}"[:300])
        self.hub.note_tick("; ".join(errors)[:300])

    @staticmethod
    def _run(phases, now: float) -> list[str]:
        """Each phase in turn, fenced: one that raises is logged and named, and the rest still run."""
        errors = []
        for name, phase in phases:
            try:
                phase(now)
            except Exception as exc:                  # a phase must never take the loop down
                log.exception("[control] %s phase failed", name)
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return errors

    def _engine_phase(self) -> str | None:
        """None when the broker engine is usable this tick, else why not (retried every tick)."""
        if self.engine_ready:
            return None
        try:
            err = self.ensure_engine() if self.ensure_engine else ""
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        if err:
            return err
        self.engine_ready = True
        log.info("[control] broker engine is up — adopting open leases")
        try:
            self.recover()
        except Exception as exc:
            log.exception("[control] recover after engine start failed")
            return f"recover: {type(exc).__name__}: {exc}"
        return None

    def close(self) -> None:
        """Stop taking probes (a probe out ends on its own timeouts). For tests and a clean shutdown."""
        self._probes.shutdown(wait=False, cancel_futures=True)

    def _probe(self, now: float) -> None:
        """Probe every backend that is due and not still being probed, and wait for them AT MOST ONE TICK.

        WHY NOT WAIT FOR ALL. At one probe a second a tick that waited for its slowest probe — a box that
        is off answers nothing until /health's 1.5 s and /v1/models' 3 s have run out — sent every OTHER backend's
        next probe 4-5 s late, past the rate meters' gap rule (three probe intervals, rates.RateMeter): every live
        server's tok/s went blank for as long as one box was off. Now a probe still out after a tick lands when it
        answers, on its own thread, and its backend is skipped until it has — never probed twice at once."""
        if self._landing_error:
            error, self._landing_error = self._landing_error, ""
            raise RuntimeError(f"a probe's result did not land: {error}")
        with self._probing_lock:
            due = [b for b in self.hub.backends()
                   if now - b.last_probe_at >= self.cfg.probe_interval_s and b.url not in self._probing]
            self._probing.update(b.url for b in due)
        if due:
            wait([self._probes.submit(self._probe_one, b, now) for b in due], timeout=self.cfg.tick_s)

    def _probe_one(self, b, now: float) -> None:
        """One backend's probe, on a probe thread: read it, then land the result — one landing at a time, so the
        hub and the performance log see probes in a single order as they did when the tick landed them."""
        try:
            try:
                fields = self.prober(b, self.cfg, now=now)
            except Exception as exc:
                fields = {"live": False, "served_models": (), "waiting": 0, "running": 0, "last_probe_at": now,
                          "last_error": f"{type(exc).__name__}: {exc}"[:200]}
            with self._landing:
                url = b.url
                before = self.hub.backend_for(url).live
                self.hub.update_backend(url, **fields)
                self._record(url, fields, now)
                if fields.get("live") != before:
                    log.info("[control] %s is now %s%s", url, "LIVE" if fields.get("live") else "down",
                             f" ({fields.get('last_error')})" if fields.get("last_error") else "")
        except Exception as exc:                      # a landing must never die unseen on a probe thread
            log.exception("[control] the probe of %s did not land", b.url)
            self._landing_error = f"{b.url}: {type(exc).__name__}: {exc}"[:300]
        finally:
            with self._probing_lock:
                self._probing.discard(b.url)

    def _record(self, url: str, fields: dict, now: float) -> None:
        """A server head's probe, as the performance log's next read, with ITS OWN view: the recorder keeps each
        server's minutes apart by the view's name and every window names its server (perflog.PerfRecorder). Only
        heads: they are the engines whose minutes the log keeps. Whatever the log does, the probe phase goes on."""
        b = self.hub.backend_for(url)
        if self.recorder is None or b.kind != CLUSTER:
            return
        snap = self.hub.snapshot(now)
        try:
            self.recorder.observe(at=fields.get("last_probe_at") or now, mono=fields.get("mono_at"),
                                  live=bool(fields.get("live")), scrape=fields.get("scrape"),
                                  cluster=snap.cluster_view_of(b))
        except Exception:                          # noqa: BLE001 — never the loop's problem
            log.exception("[control] the performance log could not take a probe")

    def _refresh_leases(self, now: float) -> None:
        current = self.hub.leases()
        if not current:
            return
        fresh = self.leases.refresh(current)
        keep = []
        for lease in fresh:
            if lease.status == LEASE_LOST:
                # Close the run so a lost lease can never linger as a phantom demand signal.
                self.leases.release(lease)
                continue
            keep.append(lease)
        self.hub.set_leases(keep)

    def _pressure_phase(self, now: float) -> None:
        snap = self.hub.snapshot(now)
        self.hub.set_pressure(self._pressure.observe(now=now, waiting=pressure_sample(snap, self.cfg)))

    def _boxes_phase(self, now: float) -> None:
        cap, cap_detail = self.leases.vllm_capacity()
        self.hub.set_vllm_capacity(cap, cap_detail)
        self.hub.set_live_boxes(self.leases.live_worker_labels)
        on, detail = self.leases.vllm_boxes_on()
        if self.leases.parked_labels is not None:   # only once the control plane was readable
            self.hub.set_parked_boxes(self.leases.parked_labels)
        before = self.hub.snapshot(now).boxes_on
        self.hub.set_boxes_on(on, detail)
        if on != before:
            log.info("[control] vLLM boxes %s (%s)", "available" if on else "all parked", detail)

    def _cluster_phase(self, now: float) -> None:
        self.cluster_loop.tick(now)

    def _plan_and_apply(self, now: float) -> None:
        snap = self.hub.snapshot(now)
        wanted = cluster_wanted(snap)
        if wanted:
            # START FREES RAM FIRST (worklog/vllm-gateway-program.md L7). The GB10 boxes are being reserved
            # for the pair, so the classic vLLM planner stands aside: no lease is asked for, and every
            # one held goes back the moment the gateway's turns on it end — the router sends it no new one.
            busy = {b.host_label for b in serving_now(snap, VLLM)}
            for lease in snap.active_leases(kind=VLLM):
                if not (lease.status == LEASE_RUNNING and lease.host_label in busy):
                    self._release(Release(lease_id=lease.lease_id))
        # The vLLM leases we ASK the fleet for (no fallback lease: the tier is retired, 2026-10-17).
        for action in ([] if wanted else plan_actions(snap, self.cfg)):
            if isinstance(action, Create):
                if wanted and action.kind == VLLM:
                    continue      # the GB10 boxes are the pair's: a classic lease would only compete for them
                self._create(action, now)
            elif isinstance(action, Release):
                self._release(action)

    def _create(self, action: Create, now: float) -> None:
        if now < self.backoff_until:
            log.debug("[control] lease wanted but backing off for %.0fs", self.backoff_until - now)
            return
        lease = self.leases.create(action.avoid_boxes, kind=action.kind)
        if lease is None:
            self._fail_count += 1
            wait = min(self.cfg.lease_backoff_max_s, _BACKOFF_BASE_S * 2 ** (self._fail_count - 1))
            self.backoff_until = now + wait
            log.warning("[control] lease create failed (%s); next attempt in %.0fs",
                        self.leases.last_error, wait)
            return
        self._fail_count = 0
        self.hub.set_leases(self.hub.leases() + [lease])

    def _release(self, action: Release) -> None:
        for lease in self.hub.leases():
            if lease.lease_id == action.lease_id:
                if self.leases.release(lease):
                    self.hub.set_leases([l for l in self.hub.leases()
                                         if l.lease_id != lease.lease_id])
                    if lease.status == LEASE_RUNNING:
                        log.info("[control] released %s on %s — the engine stops that vLLM "
                                 "after its idle TTL", lease.lease_id[:8], lease.box or "?")
                return
