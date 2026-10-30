"""The state snapshot every decision is made over. Plain data, no I/O, no threads."""
from __future__ import annotations

from dataclasses import dataclass, field

from vllm_lb.catalog import CLUSTER_MODE

VLLM = "vllm"
#: The two-box TP=2 vLLM (worklog/vllm-cluster-control-plane.md): the vLLM wire on its own port,
#: HELD by the cluster loop through one broker_parrot reservation per GB10 box, not a scaler lease.
CLUSTER = "cluster"

LEASE_QUEUED = "queued"     # row exists, nobody has claimed it yet
LEASE_RUNNING = "running"   # a box claimed it: host_label/box known, its vLLM is (coming) up
LEASE_LOST = "lost"         # the row ended without us (failed/cancelled/completed) — forget it


@dataclass
class Backend:
    kind: str                       # VLLM | CLUSTER (the OLLAMA fallback kind is retired, 2026-10-17)
    url: str
    host_label: str | None = None   # VLLM only — what a claim stamps into claimed_by
    box: str | None = None          # VLLM only — what avoid_box speaks
    #: CLUSTER only — the fleet NAME of the cluster this is the head of: the key of its view in `extra["clusters"]`
    #: (worklog/parallel-clusters.md). None = a cluster backend built before names existed (tests).
    cluster: str | None = None
    live: bool = False
    served_models: tuple[str, ...] = ()
    #: The checkpoint each served model was loaded from, as /v1/models' `root` names it ("" never kept).
    model_roots: tuple[str, ...] = ()
    max_model_len: int | None = None
    waiting: int = 0                # vllm:num_requests_waiting (queue pressure)
    running: int = 0                # vllm:num_requests_running
    inflight: int = 0               # requests THIS LB is proxying to it right now
    penalty_until: float = 0.0      # skipped until then, after a connect failure
    failure_streak: int = 0         # consecutive 5xx, no success in between
    depth_known: bool = True        # did /metrics answer? unknown depth = busy
    #: Is the BOX's worker heartbeating? A server can be down on a perfectly healthy box —
    #: that is this fleet's resting state, since broker_parrot starts a vLLM only on demand.
    #: None = not asked yet; never claim a box is down before looking.
    box_alive: bool | None = None
    #: Has the operator PARKED this box's gpu lane (`worker_controls.desired_state='off'`)?
    #: A parked box is deliberate and healthy — the broker simply starts nothing there — so it
    #: must never be shown as a fault. None = the control plane has not been read yet; claiming
    #: "not parked" before looking is the same lie `box_alive=None` exists to avoid.
    parked: bool | None = None
    last_used_at: float | None = None   # when WE last sent this backend a request
    gen_tokens_total: int = 0           # vllm:generation_tokens_total, a monotonic counter
    prompt_tokens_total: int = 0
    #: Measured across two probes, NOT reported by the engine. None = not yet known (one
    #: sample only, or the counter went backwards because the sidecar restarted). None is not
    #: zero: zero means measured, and idle.
    tokens_per_s: float | None = None
    #: When work on this backend last MOVED — a generated or a prefilled token, or work arriving
    #: at an idle backend. None while it has no work. `stalled_s` in /lb/status is measured from it.
    progress_at: float | None = None
    last_probe_at: float = 0.0
    #: When `running` / `waiting` were READ off /metrics (wall clock; probe.py). None while no depth was read.
    depth_read_at: float | None = None
    last_error: str = ""


@dataclass
class Lease:
    lease_id: str                   # == the workflow_runs id we own
    job_id: str
    created_at: float
    #: WHICH SERVER this lease holds a box for: a vLLM lease asks the fleet to start one and keeps it warm (a
    #: CLUSTER reservation is the cluster loop's). The fallback's OLLAMA lease is retired (2026-10-17).
    kind: str = VLLM
    status: str = LEASE_QUEUED
    host_label: str | None = None
    box: str | None = None
    claimed_at: float | None = None
    released_at: float | None = None
    detail: str = ""


@dataclass
class Snapshot:
    now: float
    backends: list[Backend]
    leases: list[Lease]
    #: DEMAND: when a client request last ARRIVED (`Hub.note_request`). Nothing else writes it.
    last_request_at: float | None
    last_pressure_at: float | None
    db_ok: bool
    db_error: str = ""
    #: Is at least one configured vLLM box unparked (worker_controls.desired_state='on')?
    #: False ⇒ a lease would be a request nobody can serve. Fail-open default: True.
    boxes_on: bool = True
    #: How many LIVE gpu workers advertise `vllm` (worker_heartbeats.llm_servers_available),
    #: asked of the broker every tick. The ceiling on leases: one lease holds one BOX, so
    #: wanting more than the fleet can serve only queues rows nothing will claim. None ⇒ not
    #: asked yet; the scaler then falls back to the configured box count.
    vllm_capacity: int | None = None
    #: RETENTION: when this process adopted running leases from the one before it, and which
    #: (`Hub.note_recovered`). `scaler.retained_leases` keeps those through their idle window.
    adopted_at: float | None = None
    adopted_lease_ids: frozenset[str] = frozenset()
    #: The Stop: when the cluster's desired mode last went `cluster` → `classic` (`Hub.set_cluster`).
    #: Classic demand counts from here (`scaler.classic_demand_at`).
    cluster_stopped_at: float | None = None
    extra: dict = field(default_factory=dict)

    def active_leases(self, kind: str | None = None) -> list[Lease]:
        return [l for l in self.leases
                if l.status in (LEASE_QUEUED, LEASE_RUNNING)
                and (kind is None or l.kind == kind)]

    def leased_labels(self) -> set[str]:
        """vLLM labels only — this feeds the router's top tier."""
        return {l.host_label for l in self.leases
                if l.status == LEASE_RUNNING and l.host_label and l.kind == VLLM}

    def cluster_views(self) -> dict[str, dict]:
        """EVERY active cluster's view, by its fleet name (`extra["clusters"]`, `Hub.snapshot`) — the one place a
        reader asks for them (worklog/parallel-clusters.md). The singular `extra["cluster"]` is a display alias:
        with two clusters it names neither, so nothing that decides (router, planner, hold) reads it."""
        return self.extra.get("clusters") or {}

    def cluster_view_of(self, b: Backend) -> dict:
        """The view of the cluster `b` is the head of — `{}` for a backend that names none (it then has no loop
        that could call it held, so it is never routed as one)."""
        return self.cluster_views().get(b.cluster) or {}

    def cluster_wanted(self) -> bool:
        """Does the operator want ANY cluster serving — an active one's `desired.mode == cluster`, or a declared one
        outside this run whose stored wish still wants it up (`wanted_off_run`: an LB restart forgot its activation, not
        its engine)? The classic planner (control.py) stands aside while one does."""
        return (any((v.get("desired") or {}).get("mode") == CLUSTER_MODE for v in self.cluster_views().values())
                or bool(self.extra.get("wanted_off_run")))

    def vllm_backends(self) -> list[Backend]:
        return [b for b in self.backends if b.kind == VLLM]
