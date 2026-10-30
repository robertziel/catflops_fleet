"""How many vLLM leases we want, and what to do about the difference — pure.

Two concepts decide the count, and they are kept apart by name (worklog/vllm-lb-stop-demand.md):

* DEMAND — a client request ARRIVED (`classic_demand_at`). Only demand asks the fleet for a server.
  Nothing arriving and nothing in flight for `idle_release_s` ⇒ 0 — a RELEASE; the engine stops the
  server its own TTL later. Sustained queueing pressure on a leased vLLM in the same window
  (`classic_pressure_at`) ⇒ a second box.
* RETENTION — what a restart's ADOPTION grants (`retained_leases`): the leases it took over are not
  released as idle until the window has passed from the adoption. It keeps; it never asks.

Every classic input is counted FROM THE CLUSTER'S LAST STOP (`_since_stop`) and only over the
backends of its own KIND (`serving_now`). Both halves of that sentence were bought on 2026-09-15,
when a Stop with seven turns in flight on the TP=2 head created a classic lease two seconds later
and a second one nine seconds after that.

The count never exceeds the boxes the LB knows about, so a lease always has somewhere to land.

Why release ≥ cold start: a vLLM restart costs 5–15 min on the GB10s (and, with the engine's
fleet-wide demand count, BOTH boxes pay it). Releasing after 60 s of quiet would re-pay that
on every interactive pause — the ski-rental break-even says hold for at least the restart
cost, which is where the 600 s default comes from.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm_lb.config import Config
from vllm_lb.state import LEASE_QUEUED, VLLM, Backend, Lease, Snapshot


@dataclass(frozen=True)
class Create:
    avoid_boxes: tuple[str, ...] = ()
    kind: str = VLLM


@dataclass(frozen=True)
class Release:
    lease_id: str


Action = Create | Release


def _recent(at: float | None, now: float, window_s: float) -> bool:
    """Did `at` happen within the last `window_s`? Never, if it never happened."""
    return at is not None and now - at <= window_s


def _since_stop(at: float | None, snap: Snapshot) -> float | None:
    """`at`, unless it predates the cluster's last Stop — then it is the CLUSTER's, not classic's.

    The one rule, for every input the classic planner has. A turn the TP=2 pair served, and the
    queue those turns built while they overflowed it, are facts about the pair; neither is a reason
    to start a classic vLLM once the GB10 boxes come back. A classic lease makes broker_parrot start
    vLLM on BOTH GB10 boxes — minutes of cold start each, which is what the phantom leases of
    2026-09-15 cost, twice. Anything at or after the Stop counts at once.
    """
    stopped = snap.cluster_stopped_at
    return None if at is None or (stopped is not None and at < stopped) else at


def classic_demand_at(snap: Snapshot) -> float | None:
    """DEMAND for a classic vLLM: when a client request last ARRIVED, counted from the Stop."""
    return _since_stop(snap.last_request_at, snap)


def classic_pressure_at(snap: Snapshot) -> float | None:
    """PRESSURE for a classic vLLM: when one box was last measurably too few, counted from the Stop.

    Pressure is demand's own second signal, so it starts afresh with demand. On 2026-09-15 the queue
    that asked for the SECOND classic box (job row 23:24:03, 11 s after the Stop) had been built by
    turns starving against a TP=2 pair the router judged full — evidence about the cluster, carried
    across the Stop by a clock nothing reset. Real pressure after a Stop re-fires within
    `scale_up_sustain_s`; this only stops the pair's own queue prepaying for it.
    """
    return _since_stop(snap.last_pressure_at, snap)


def serving_now(snap: Snapshot, kind: str) -> list[Backend]:
    """The backends of `kind` carrying one of OUR requests right now.

    Per KIND, because a lease plan decides only about its own servers: the turns on the TP=2 head are the
    cluster's. Until 2026-09-15 this was
    read over EVERY backend: the operator pressed Stop with seven turns in flight on the head, and
    the next tick asked the fleet for a classic vLLM on their account — turns that Stop was itself
    about to kill (they died 2 s later, all with `IncompleteRead(0 bytes read)`). A backend's
    in-flight work is a reason to keep ITS lease, never to ask for another kind of server.
    """
    return [b for b in snap.backends if b.kind == kind and b.inflight > 0]


def retained_leases(snap: Snapshot, kind: str, window_s: float) -> list[Lease]:
    """RETENTION: the held leases of `kind` that this process ADOPTED from the one before it, until
    `window_s` has passed since the adoption.

    A restart is not idleness: a fresh process has no request on record, and its first tick used to
    release everything it had just adopted (worklog/vllm-lb-restart-keeps-leases.md). Nor is it
    demand: retention only keeps what was adopted and is still held. It never asks for a lease, so
    an adopted lease the broker ends is not replaced on its account.
    """
    if not _recent(snap.adopted_at, snap.now, window_s):
        return []
    return [l for l in snap.active_leases(kind=kind) if l.lease_id in snap.adopted_lease_ids]


def desired_leases(snap: Snapshot, cfg: Config) -> int:
    # A lease is a REQUEST FOR A vLLM. With every configured box parked, no box will start
    # one — the row would only sit queued and get flagged `unassignable`.
    if not snap.boxes_on:
        return 0
    # DEMAND: something ARRIVED within the window, or is still in flight ON A vLLM — a long
    # generation (Claude Code streams for minutes) must not lose its lease mid-stream.
    busy = bool(serving_now(snap, VLLM))
    n = 0
    if busy or _recent(classic_demand_at(snap), snap.now, cfg.idle_release_s):
        n = 2 if _recent(classic_pressure_at(snap), snap.now, cfg.idle_release_s) else 1
    # RETENTION keeps every lease a restart adopted through its window, and never asks for more.
    n = max(n, len(retained_leases(snap, VLLM, cfg.idle_release_s)))
    # THE BROKER'S ANSWER IS THE CEILING. One lease holds one BOX, so wanting more leases
    # than there are boxes advertising vllm only queues rows nothing can claim — which is
    # exactly what stranded a run-next lease for 38 min on 2026-09-12. `None` = not asked
    # yet (or the control plane was unreadable), and then the configured count stands.
    capacity = snap.vllm_capacity
    if capacity is None:
        capacity = len(snap.vllm_backends())
    return max(0, min(n, cfg.max_leases, capacity))


def _release_order(snap: Snapshot, active: list[Lease]) -> list[Lease]:
    """Unclaimed first (nothing is serving from them), then the running lease whose backend
    carries the fewest of our requests, newest first among equals."""
    inflight = {b.host_label: b.inflight for b in snap.vllm_backends()}

    def key(l: Lease):
        return (0 if l.status == LEASE_QUEUED else 1,
                inflight.get(l.host_label, 0), -l.created_at)

    return sorted(active, key=key)


def plan_actions(snap: Snapshot, cfg: Config) -> list[Action]:
    active = snap.active_leases(kind=VLLM)
    # ONE vLLM LEASE PER BOX, reconciled rather than steered. A gpu lane is two wide, so two
    # leases on one box take BOTH lanes for up to lease_max_hold_s (6 h) and buy no capacity —
    # that box's single vLLM already serves us. We cannot steer placement (avoid_box is a hard
    # claim filter that strands rows), so the duplicate is let go after it lands, which can
    # never make anything unclaimable. Newest goes first: it has waited least.
    seen: set[str] = set()
    dupes = []
    for l in sorted((l for l in active if l.box), key=lambda l: l.created_at):
        if l.box in seen:
            dupes.append(l)
        seen.add(l.box)
    if dupes:
        return [Release(lease_id=l.lease_id) for l in dupes]
    want = desired_leases(snap, cfg)
    if len(active) < want:
        # ONE create per tick: the next tick sees where it landed (claimed_by).
        #
        # NO avoid_box, EVER. It reads like a spreading hint but broker_parrot applies it as a
        # hard claim FILTER with no fallback, so on a two-box fleet it is one box away from
        # "unclaimable". 2026-09-12: the 2nd lease carried avoid_box=['box-a'] and sat queued
        # 36 min wearing its run-next flag, because the other vLLM box was 6.5 h into a single
        # job (against a 148 min historical max). Two leases on one box cost nothing — that
        # box's vLLM serves both and the surplus goes back on the normal idle path — while an
        # unclaimable row costs a backend the fleet actually has.
        # EXCLUDE A BOX WE ALREADY HOLD — but only while another candidate exists. A gpu lane
        # is two wide, so two leases on one box take both lanes and buy nothing (the operator
        # watched that happen on box-b). `avoid_box` is a hard claim FILTER, so sending it
        # when it would exclude the LAST candidate is what stranded a lease in pass 11; the
        # guard is the capacity figure, which now counts boxes that could TAKE a lease.
        held = tuple(sorted({l.box for l in active if l.box}))
        capacity = snap.vllm_capacity
        if capacity is None:
            capacity = len(snap.vllm_backends())
        return [Create(avoid_boxes=held if len(held) < capacity else ())]
    if len(active) > want:
        surplus = len(active) - want
        return [Release(lease_id=l.lease_id) for l in _release_order(snap, active)[:surplus]]

    # PER-BOX IDLENESS — checked only when the fleet-wide rules are content, because
    # reduction already prefers an UNCLAIMED lease and must keep that order. `last_request_at`
    # is a single clock, so while ANY traffic flows every held lease is kept — including one on
    # a box that has served nothing for hours, whose gpu lane broker_parrot could be using.
    # Releasing is the node's own cooperative finish: the LB never stops a server, and
    # broker_parrot decides what to do with it. If the box is wanted again, the next tick asks.
    used = {b.host_label: b.last_used_at for b in snap.vllm_backends()}
    busy_boxes = {b.host_label for b in serving_now(snap, VLLM)}
    retained = {l.lease_id for l in retained_leases(snap, VLLM, cfg.idle_release_s)}
    stale = []
    for l in active:
        if l.status == LEASE_QUEUED or not l.host_label:
            continue                       # no box yet — the surplus rule owns that case
        if l.host_label in busy_boxes:
            continue                       # serving right now; inflight always protects a lease
        if l.lease_id in retained:
            continue                       # adopted by a restart, inside its window: not disuse
        # A box that has never served is aged from the CLAIM, so a cold start is not disuse.
        since = used.get(l.host_label) or l.claimed_at or l.created_at
        if snap.now - since > cfg.idle_release_s:
            stale.append(l)
    if stale:
        return [Release(lease_id=l.lease_id) for l in stale]

    return []


def pressure_sample(snap: Snapshot, cfg: Config) -> int:
    """What the pressure tracker sees each tick: the deepest `waiting` on a LEASED live vLLM,
    or the number of our own starved admissions in the sustain window — whichever is larger.

    The second source exists because the first cannot fire from our traffic: the router demotes
    a vLLM at `vllm_max_depth`, long before its `waiting` reaches the scale-up threshold, so our
    queueing never showed up as pressure — while a capable box sat with no server and nobody asked for it."""
    leased = snap.leased_labels()
    waiting = max((b.waiting for b in snap.backends
                   if b.kind == VLLM and b.live and b.host_label in leased), default=0)
    return max(waiting, int(snap.extra.get("starved_recent", 0) or 0))


class PressureTracker:
    """Turns a per-tick `waiting` sample into "sustained pressure": the queue on a leased
    vLLM had to stay ≥ `threshold` for `sustain_s` before it counts, and the last time it
    counted is remembered (that is what `Snapshot.last_pressure_at` carries)."""

    def __init__(self, *, threshold: int, sustain_s: float) -> None:
        self._threshold = int(threshold)
        self._sustain_s = float(sustain_s)
        self._since: float | None = None
        self.last_pressure_at: float | None = None

    def observe(self, *, now: float, waiting: int) -> float | None:
        if waiting >= self._threshold:
            if self._since is None:
                self._since = now
            if now - self._since >= self._sustain_s:
                self.last_pressure_at = now
        else:
            self._since = None
        return self.last_pressure_at
