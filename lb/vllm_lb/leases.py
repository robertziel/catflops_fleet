"""The broker adapter — every word the LB says to broker_parrot, in one place.

A lease is ONE `workflow_runs` row we own plus ONE gpu node-job:

  * run: `workflow_name=<lease_workflow>`, `status='running'`, `mode='node'`, no out_dir —
    'running' (not 'queued') so the orchestrator's expand loop never tries to load a workflow
    by that name; we cancel the run BEFORE the job returns, which makes the orchestrator's
    completion callback a no-op for it.
  * job: `queue='gpu'`, `server_types=['vllm']`, NO required_model (see below), `priority=0`
    (the operator's "highest priority in the broker"), `avoid_box` for the 2nd box, the node
    module by DOTTED name so the GB10 boxes' lanes import it off the NFS-live tree without a
    restart, `gpu_slots='all'` — the row weighs the claiming box's WHOLE budget (migration
    0030), so the fleet cannot co-schedule its own work into the server we hold — and
    `inputs.model` — the model the lease DECLARES it is for.

WHY THE MODEL IS IN `inputs` AND NOT IN `required_model` (worklog/vllm-lb-lease-model.md).
`required_model` is not a label, it is a hard capability FILTER in the engine's
`claim_next_gpu_job` (`node_queue.py:1064-1075`): a row that names a model is claimable only
where the worker advertised that id, so setting it can only ever SHRINK the set of boxes that
can take the lease. The advertised list is narrowed by observation and by live memory
(`claim_worker.py:2233-2259`, `2771-2820`) — measured on this fleet, box-a-gpu holds
`lm-article` and `lm-title` and advertises neither — and `qwen3.8:latest` survives that gate on
a GB10 box only because the vLLM the lease exists to START is already serving it. Naming it would
strand the lease in exactly the state the lease is for. It would also arm the unassignable
sweep (`node_queue.py:3170-3178`) and move the box lease slot off `__llm_server__`
(`claim_worker.py:4580-4582`), and it buys nothing: the pool/inline split is on `server_types`,
warm affinity sorts BELOW the RUN-NEXT flag we already set, and neither `server_switch` nor
`model_residency` reads a job's `required_model` at all. `inputs` is written at insert and read
by no claim predicate, no ORDER BY and no sweep — and `node_executor` binds node arguments from
the `run()` signature, so a key the node does not declare is ignored rather than splatted.

The engine tells us where it landed through `claimed_by` (a host_label) — never a URL — so
the LB maps host_label → box → URL from its own config.

Release is cooperative: cancelling the run is the sanctioned signal the body's
`cancel_event` polls (~5 s); a still-queued job is cancelled outright. There is no per-job
kill in the engine and we do not want one (it would take the whole worker down).

Every method is fenced: a DB failure sets `db_ok=False` / `last_error` and returns the
conservative value. The daemon keeps proxying whatever is live either way.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from vllm_lb.config import Config
from vllm_lb.cluster import Reservation, is_cluster_purpose, tag_for
from vllm_lb.state import LEASE_LOST, LEASE_QUEUED, LEASE_RUNNING, VLLM, Lease

#: The server kind a FALLBACK lease named (`server_types`), before the tier was retired (2026-10-17): what `recover`
#: releases on sight.
RETIRED_FALLBACK_KIND = "ollama"
#: The `inputs.purpose` a KEPT lease carried for the operator's keep-loaded order, before keep loaded was retired
#: (2026-10-17): what `recover` releases on sight.
RETIRED_KEEP_PURPOSE = "vllm-lb/keep"

log = logging.getLogger(__name__)

_TERMINAL = ("failed", "cancelled", "completed", "skipped")

_RESERVATION_NS = uuid.UUID("c1a5e7b0-2f6d-4d8e-9a51-7f3b2c4d9e10")


def cluster_reservation_id(generation: int, start_id: str | None, box: str) -> str:
    """The `workflow_runs` id of one box's reservation for one Start: DERIVED, never random.

    So a second ask — this LB's, or a restarted one's — finds the row instead of inserting another,
    and a reservation that ended (even one whose run someone else closed) is still found, which is what
    keeps an ended reservation a sticky failure instead of a quiet re-ask. The start id keeps a
    generation number that restarted from a lost state file from naming an old one."""
    return str(uuid.uuid5(_RESERVATION_NS, f"{tag_for(generation)}|{start_id or '-'}|{box}"))


def _reservation(run_id: str, job: dict, *, run_open: bool) -> Reservation:
    return Reservation(reservation_id=run_id, job_id=str(job["id"]), status=str(job.get("status") or ""),
                       claimed_by=str(job["claimed_by"]) if job.get("claimed_by") else None, run_open=run_open)


class BrokerLeases:
    def __init__(self, cfg: Config, *, clock: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self.clock = clock
        self.db_ok = True
        self.last_error = ""
        self._box_of = {b.host_label: b.box for b in cfg.vllm_boxes}
        #: Why the next lease is (un)sized — shown in /lb/status. Decided per create().
        self.context_state = "unknown: no lease created yet"
        #: Whether the last lease got the "run next" flag — shown in /lb/status.
        self.run_next_state = "unknown: no lease created yet"
        #: Whether the lease weighs the whole box (0030) — shown in /lb/status. Per create().
        self.gpu_slots_state = "unknown: no lease created yet"
        #: Every gpu worker seen heartbeating on the last capacity read. A box here is ALIVE
        #: even when no server runs on it — this fleet's resting state.
        self.live_worker_labels: set[str] = set()
        #: gpu lanes the operator has parked (`desired_state='off'`), by host_label. None until
        #: the control plane has been read once — "not parked" must not be asserted before then.
        self.parked_labels: set[str] | None = None

    # ── context sizing (migration 0028) ────────────────────────────────────

    def _context_kwargs(self) -> dict:
        """`{'context_length': N}` — the lease model's catalog window (`Config.lease_context_length`) — when the
        engine takes it, else `{}` and says why.

        Rollout-safe by construction: an engine without the parameter, or a database
        without the column, still gets a lease — unsized — instead of no lease at all. A lease model with
        no catalog row is not a rollout state: it raises, naming the model."""
        import inspect

        from queue_workflows import node_queue

        want = self._cfg.lease_context_length          # the lease model's catalog window; raises when it has none
        if "context_length" not in inspect.signature(node_queue.enqueue_node_job).parameters:
            self.context_state = "unsized: engine predates migration 0028"
            return {}
        self.context_state = f"sized: {want}"
        return {"context_length": int(want)}

    # ── the weight of the lease (migration 0030) ───────────────────────────

    def _gpu_slots_kwargs(self) -> dict:
        """`{'gpu_slots': 'all'}` when the engine can weigh a row, else `{}` — and says why.

        A lease weighed 1 leaves PAR-1 slots open on its box, and the fleet co-schedules its
        own `llm_chat` into the very server the lease was taken to hold (measured 2026-09-12:
        `wait=4 run=1` of fleet load on the leased box-b while our turns queued behind it).
        `all` closes the box: the box-slot arbiter weighs the row as the claiming box's whole
        budget, so nothing else is admitted while we hold it — and the lease itself WAITS
        behind running fleet work rather than squeezing in beside it (it is never stranded:
        it claims the moment the box drains).

        The same three rollout states as `_context_kwargs`, and the same rule: an engine
        without the parameter still gets a lease — unweighted — instead of no lease at all."""
        import inspect

        from queue_workflows import node_queue

        if "gpu_slots" not in inspect.signature(node_queue.enqueue_node_job).parameters:
            self.gpu_slots_state = "unweighted: engine predates migration 0030"
            return {}
        self.gpu_slots_state = "all: the lease holds the whole box"
        return {"gpu_slots": "all"}

    def _run_next(self, node_queue, job_id: str) -> None:
        """Flag the freshly-enqueued row "run next" (`is_priority`, migration 0016).

        `priority=0` only wins the BAND, and the band is the FOURTH sort key: the GPU claim
        orders `is_priority DESC, affinity, warm_affinity, priority ASC, host_dir`. A job whose
        model is already loaded on the box "preempts a colder one even from a worse priority
        band" — and on this fleet every GB10 box holds qwen3.8 resident, so every queued qwen3.8
        job outranks an unflagged lease. `is_priority` sorts ahead of all of it.

        NEVER fatal. The lease row already exists; a missing flag costs queue position, and the
        alternative — discarding a good lease because an optimisation failed — costs a backend.
        A `None` return is not a failure either: it means the row was no longer `queued`, which
        at this age almost always means a worker already claimed it.
        """
        flag = getattr(node_queue, "prioritize_node_job", None)
        if flag is None:
            self.run_next_state = "unflagged: engine predates migration 0016"
            return
        try:
            self.run_next_state = "flagged" if flag(job_id) else "already claimed (or gone)"
        except Exception as exc:                       # an optimisation, never a precondition
            self.run_next_state = f"unflagged: {type(exc).__name__}: {exc}"[:120]
            log.warning("[leases] could not flag %s run-next: %s", job_id[:8], exc)

    # ── fencing ────────────────────────────────────────────────────────────

    def _ok(self) -> None:
        self.db_ok = True
        self.last_error = ""

    def _fail(self, what: str, exc: BaseException) -> None:
        self.db_ok = False
        self.last_error = f"{what}: {type(exc).__name__}: {exc}"[:300]
        log.warning("[leases] %s", self.last_error)

    # ── the four verbs ─────────────────────────────────────────────────────

    def vllm_boxes_on(self) -> tuple[bool, str]:
        """Is any configured vLLM box unparked? `(on, detail)` — never raises.

        A box with no `worker_controls` row counts as ON (the engine's own rule: an absent
        row means the worker runs). An unreadable control plane fails OPEN — a DB blip must
        not silently stop leasing; the worst case is one queued row the sweep flags."""
        from queue_workflows import worker_control

        states = []
        try:
            for box in self._cfg.vllm_boxes:
                row = worker_control.get_worker_control(box.host_label, "gpu")
                states.append((box.host_label, (row or {}).get("desired_state") or "no row"))
        except Exception as exc:
            return True, f"unknown ({type(exc).__name__}: {exc})"[:200]
        # Remembered per label so a card can say PARKED instead of borrowing the red that
        # means "not answering". Only set when the control plane was actually readable.
        self.parked_labels = {label for label, state in states if state == "off"}
        if not states:
            return False, "no vLLM boxes configured"
        detail = " ".join(f"{label}={state}" for label, state in states)
        return any(state != "off" for _, state in states), detail

    def vllm_capacity(self) -> tuple[int, str]:
        """How many boxes could TAKE a vLLM lease? `(count, detail)` — never raises.

        NOT "how many serve vllm right now". A box only ADVERTISES vllm once its sidecar is
        running, and the sidecar only starts when a lease is claimed — so capping leases on
        that number deadlocks a fleet whose vLLMs are all stopped, which is exactly when a
        lease is needed. Observed live: both boxes ConnectionRefused, `desired leases: 0`,
        and nothing would ever ask for one.

        What bounds leases is how many boxes could ACCEPT one: a live gpu worker, inside the
        engine's own stale window and project, whose label the LB has a URL for, and which the
        operator has not parked. What each box currently SERVES still decides routing — that is
        `served_models`, and it is untouched.

        Fails OPEN to the configured box count; a confident zero (workers alive, none of them
        ours or all parked) is returned as zero.
        """
        fallback = len(self._cfg.vllm_boxes)
        try:
            from queue_workflows.node_queue import _stale_worker_after_s
            live, capable = self._gpu_heartbeats(int(_stale_worker_after_s()))
            # Every heartbeating gpu worker, whatever it serves. A box here is ALIVE even with
            # no server running on it — this fleet's resting state, and the difference between
            # "no server" and "box down" on the gateway.
            self.live_worker_labels = set(live)
        except Exception as exc:
            return fallback, f"unknown ({type(exc).__name__}: {exc})"[:160]
        self.live_worker_labels = set(live)   # every heartbeating gpu worker, for box liveness
        if not live:
            # No live gpu worker at all is "we cannot tell", not "nothing can serve" — the
            # engine's unassignable sweep draws the same line.
            return fallback, "unknown (no live gpu worker is heartbeating)"
        return len(capable), (", ".join(capable) or "no live gpu worker can take a vLLM lease")

    def _gpu_heartbeats(self, window: int) -> tuple[list[str], list[str]]:
        """`(live gpu workers, those that could take a vLLM lease)` in the stale window.

        Filtered by PROJECT, because `worker_heartbeats` is keyed
        `(host_label, queue, project)` and every engine query filters on it. De-duplicated, and
        intersected with `LB_VLLM_BOXES`: a box the LB has no URL for can hold a lease but can
        never be routed to. A box the operator has PARKED is excluded — the engine will not
        start anything there.
        """
        from queue_workflows.db import connection
        from queue_workflows.node_queue import _project

        routable = {b.host_label for b in self._cfg.vllm_boxes}
        live: set[str] = set()
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT host_label FROM worker_heartbeats "
                "WHERE queue = 'gpu' AND project = %s "
                "  AND last_seen > now() - make_interval(secs => %s)",
                (_project(None), window))
            for row in cur.fetchall():
                live.add(str(row[0] if isinstance(row, tuple) else row["host_label"]))

        capable = sorted(live & routable)
        if capable:
            try:
                from queue_workflows import worker_control
                capable = [c for c in capable
                           if ((worker_control.get_worker_control(c, "gpu") or {})
                               .get("desired_state") or "on") != "off"]
            except Exception:
                pass                       # control plane unreadable: do not shrink capacity
        return sorted(live), capable

    def create(self, avoid_boxes: tuple[str, ...], *, kind: str = VLLM) -> Lease | None:
        """Ask the broker for a box. ``kind`` decides WHICH box and on what terms.

        A **vllm** lease is a request for a vLLM anywhere the fleet can start one: it names
        the kind and steers away from boxes we already hold. (The ollama FALLBACK lease is retired with
        the fallback tier, 2026-10-17, and the kept lease with keep loaded, the same day.)
        """
        from queue_workflows import node_queue, run_store

        run_id = str(uuid.uuid4())
        max_hold = self._cfg.lease_max_hold_s

        def enqueue(ctx: dict) -> str:
            return node_queue.enqueue_node_job(
                run_id=run_id, node_id="lease", node_module=self._cfg.lease_node_module,
                queue="gpu", server_types=[kind], priority=0,
                avoid_box=list(avoid_boxes) or None,
                inputs={"lease_id": run_id, "max_hold_s": max_hold,
                        # DECLARED, never required: see the module docstring.
                        "model": self._cfg.lease_model_id},
                **ctx,
            )

        try:
            run_store.insert_run(run_id=run_id, workflow_name=self._cfg.lease_workflow,
                                 out_dir=None, status="running", mode="node")
            try:
                job_id = enqueue({**self._context_kwargs(), **self._gpu_slots_kwargs()})
            except Exception as exc:
                # Engine ahead of the database. Both columns are in the INSERT list on EVERY
                # enqueue, so until the migration is applied no lease is possible AT ALL —
                # sized or not, weighted or not. Say exactly that instead of failing with a
                # bare SQL error every tick.
                #
                # Nothing is remembered across calls. An earlier version latched a
                # `_db_lacks_*` flag here and never cleared it, so the first lease after the
                # operator applied the migration was created unsized/UNWEIGHTED while the
                # status line still blamed the migration (independent audit, 2026-09-12,
                # finding 1). There is nothing to cache: a rejected enqueue re-states the gap
                # on the spot, and an accepted one is itself the proof that the column exists.
                if "context_length" in str(exc):
                    self.context_state = ("BLOCKED: database lacks migration 0028 — apply it "
                                          "(this engine writes context_length on every enqueue)")
                if "gpu_slots" in str(exc):
                    self.gpu_slots_state = ("BLOCKED: database lacks migration 0030 — apply it "
                                            "(this engine writes gpu_slots on every enqueue)")
                raise
        except Exception as exc:
            self._fail("create", exc)
            return None
        self._run_next(node_queue, job_id)
        self._ok()
        log.info("[leases] created %s %s (job %s) avoid_box=%s run_next=%s weight=%s",
                 kind, run_id[:8], job_id[:8], list(avoid_boxes) or '-',
                 self.run_next_state, self.gpu_slots_state.split(":", 1)[0])
        return Lease(lease_id=run_id, job_id=job_id, created_at=self.clock(), kind=kind, status=LEASE_QUEUED)

    def refresh(self, leases: list[Lease]) -> list[Lease]:
        """Re-read every non-lost lease's job row. Returns NEW Lease objects (the inputs are
        never mutated); the same list back when the DB is unreadable."""
        from queue_workflows import node_queue

        out: list[Lease] = []
        for lease in leases:
            if lease.status == LEASE_LOST:
                out.append(lease)
                continue
            try:
                row = node_queue.get_node_job(lease.job_id)
            except Exception as exc:
                self._fail("refresh", exc)
                return list(leases)
            out.append(self._reconcile(lease, row))
        self._ok()
        return out

    def _reconcile(self, lease: Lease, row: dict | None) -> Lease:
        now = self.clock()
        if row is None:
            return self._lost(lease, "job row vanished", now)
        status = str(row.get("status") or "")
        if status in _TERMINAL:
            return self._lost(lease, f"job ended without us: {status}", now)
        if status == "running" and row.get("claimed_by"):
            label = str(row["claimed_by"])
            # `_box_of` maps only LB_VLLM_BOXES labels, so a FALLBACK lease claimed by, say,
            # `box-c-gpu` resolved to box=None and — since the router gates on
            # `leased_boxes` — its ollama could never be served from at all. The box it was
            # pinned to with force_box is the answer, and the claim cannot change it.
            box = self._box_of.get(label) or lease.box
            detail = "" if box else f"claimed_by {label!r} is not in LB_VLLM_BOXES"
            if lease.status != LEASE_RUNNING:
                log.info("[leases] %s claimed by %s (box %s)", lease.lease_id[:8], label, box)
            return Lease(lease_id=lease.lease_id, job_id=lease.job_id, kind=lease.kind,
                         created_at=lease.created_at, status=LEASE_RUNNING,
                         host_label=label, box=box,
                         claimed_at=lease.claimed_at or now, detail=detail)
        # queued again (a requeue after a spill / a lane restart) or still queued
        # `kind` is carried on EVERY return: dropping it here once turned a recovered fallback
        # lease into a phantom vLLM lease, which the vLLM planner would immediately discard.
        return Lease(lease_id=lease.lease_id, job_id=lease.job_id, kind=lease.kind,
                     box=lease.box, created_at=lease.created_at, status=LEASE_QUEUED)

    @staticmethod
    def _lost(lease: Lease, why: str, now: float) -> Lease:
        log.warning("[leases] %s lost — %s", lease.lease_id[:8], why)
        return Lease(lease_id=lease.lease_id, job_id=lease.job_id, kind=lease.kind,
                     created_at=lease.created_at, status=LEASE_LOST,
                     host_label=lease.host_label, box=lease.box,
                     claimed_at=lease.claimed_at, released_at=now, detail=why)

    def release(self, lease: Lease) -> bool:
        from queue_workflows import node_queue, run_store

        try:
            run_store.update_run(lease.lease_id, status="cancelled",
                                 finished_at=datetime.now(timezone.utc))
            # A queued job would otherwise sit until the run-cancel guard rejects its claim;
            # end it now so it stops counting as vLLM demand this very tick.
            node_queue.cancel_queued_jobs_for_run(lease.lease_id)
        except Exception as exc:
            self._fail("release", exc)
            return False
        self._ok()
        log.info("[leases] released %s (%s on %s)", lease.lease_id[:8], lease.status,
                 lease.host_label or "-")
        return True

    # ── the cluster's reservation: one broker_parrot node_job per GB10 box ──────
    #
    # The LB ASKS and broker_parrot GRANTS. It never parks a lane: a park is the operator's, and it is
    # only ever READ here (`parked_by`), to say why a reservation waits.

    def cluster_support(self) -> str:
        """"" when broker_parrot's engine can claim a cluster reservation, else why it cannot."""
        from queue_workflows import node_queue

        if not hasattr(node_queue, "CLUSTER_SERVER_KIND"):
            return ("broker_parrot's engine has no node_queue.CLUSTER_SERVER_KIND, so nothing could claim a "
                    "reservation; roll broker_parrot out (local b7cbe34 or later)")
        return ""

    def _cluster_reservation_kwargs(self, *, run_id: str, box: str, purpose: str, model: str,
                                    cluster: str | None = None) -> dict:
        """THE CONTRACT with broker_parrot for one GB10 box's reservation — every word of it, here
        (final: broker_parrot local b7cbe34, live 2026-09-15 21:23Z).

        Made the way `create()` makes a lease, and on its terms: the gpu queue, the lease node by dotted
        name, top band and run-next (`_run_next`), the whole box (`gpu_slots='all'`), NO required_model.
        Pinned to its PHYSICAL box with `force_box`, and of the engine's own cluster kind, so the pinned
        lane claims it without starting a server — at once when idle, after a drain when busy. NO
        `context_length`: a sized row is claimable only by a box whose serving window fits, and a box with
        vLLM stopped may state none. `inputs`: `purpose` names the generation, `model` declares what the
        generation's ModelChoice serves (no claim predicate reads it, and no other setting names a model), and
        `max_hold_s` is a real ceiling — the node's own default is 6 h. `cluster`: the fleet NAME of a one-host node
        reserving through the SAME contract (worklog/parallel-clusters.md) — carried in `inputs.cluster`, so that
        node's own reservations are told from any other cluster's on the box; the pair's rows
        carry none, byte for byte as before. broker_parrot claims ONE cluster row per lane at a time (its per-kind cap),
        so nodes sharing a box share one such row: the first reserves it whole, the others ride it, and the last to stop
        returns it (cluster.plan_node_hold)."""
        from queue_workflows import node_queue

        inputs = {"lease_id": run_id, "purpose": purpose, "model": model,
                  "max_hold_s": self._cfg.cluster_reservation_max_hold_s}
        if cluster is not None:
            inputs["cluster"] = cluster
        return {"run_id": run_id, "node_id": f"cluster-{box}", "node_module": self._cfg.lease_node_module,
                "queue": "gpu", "server_types": [node_queue.CLUSTER_SERVER_KIND], "priority": 0,
                "required_model": None, "force_box": [box], "inputs": inputs,
                **self._gpu_slots_kwargs()}

    def reserve_for_cluster(self, box: str, generation: int, start_id: str | None, *, model: str,
                            cluster: str | None = None) -> str | None:
        """Ask broker_parrot for `box`'s reservation of this generation, declaring `model` — what the generation's
        ModelChoice serves. Returns its job id, or None.

        Idempotent by construction: the run id is derived (`cluster_reservation_id`), so asking again —
        after a crash between two lines, or from a restarted LB — finds the row instead of adding one."""
        from queue_workflows import node_queue, run_store

        run_id = cluster_reservation_id(generation, start_id, box)
        refused = self.cluster_support()
        if refused:                          # a row nothing can claim is a Start that silently never starts
            self.last_error = f"reserve: {refused}"[:300]
            log.warning("[leases] not reserving %s: %s", box, refused)
            return None
        try:
            run = run_store.get_run(run_id)
            jobs = node_queue.list_jobs_for_run(run_id) if run is not None else []
            if run is not None and run.get("status") != "running":
                # A CLOSED run is an ENDED reservation — sticky (`cluster_reservation_id`): never handed back as the
                # job asked for, never re-asked under the same id.
                raise RuntimeError(f"reservation run {run_id[:8]} was closed ({run.get('status')})")
            if jobs:
                self._ok()
                return str(jobs[0]["id"])
            if run is None:
                run_store.insert_run(run_id=run_id, workflow_name=self._cfg.lease_workflow, out_dir=None,
                                     status="running", mode="node")
            job_id = node_queue.enqueue_node_job(**self._cluster_reservation_kwargs(
                run_id=run_id, box=box, purpose=tag_for(generation), model=model, cluster=cluster))
        except Exception as exc:
            self._fail("reserve", exc)
            return None
        self._run_next(node_queue, job_id)
        self._ok()
        log.info("[leases] asked broker_parrot to reserve %s for %s (job %s) run_next=%s weight=%s", box,
                 tag_for(generation), job_id[:8], self.run_next_state, self.gpu_slots_state.split(":", 1)[0])
        return job_id

    def cluster_reservations(self, boxes, generation: int, start_id: str | None) -> dict | None:
        """`{box: (this generation's Reservation | None, open reservations of other generations)}`.

        None overall when the rows cannot be read — never "no reservation". Open ones are found by
        their PURPOSE TAG among our open lease runs, so a restarted LB adopts what it asked for and an
        older generation's is found to be returned. This generation's is then looked up by its derived
        id, so one that ended — even one whose run someone closed — still reads as ended."""
        from queue_workflows import db, node_queue, run_store

        boxes = tuple(boxes)
        purpose = tag_for(generation)
        derived = {b: cluster_reservation_id(generation, start_id, b) for b in boxes}
        current: dict[str, Reservation | None] = {b: None for b in boxes}
        stale: dict[str, list[Reservation]] = {b: [] for b in boxes}
        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT id FROM workflow_runs WHERE workflow_name = %(wf)s AND status = 'running' "
                            "ORDER BY created_at", {"wf": self._cfg.lease_workflow})
                run_ids = [str(r["id"]) for r in cur.fetchall()]
            found = []
            for run_id in run_ids:
                for job in node_queue.list_jobs_for_run(run_id):
                    tag = (job.get("inputs") or {}).get("purpose")
                    box = next((str(b) for b in (job.get("force_box") or []) if b), None)
                    if is_cluster_purpose(tag) and box in current:
                        found.append((run_id, job, tag, box))
            for run_id, job, tag, box in sorted(found, key=lambda f: f[0] != derived[f[3]]):
                if tag == purpose and current[box] is None:
                    current[box] = _reservation(run_id, job, run_open=True)
                else:
                    stale[box].append(_reservation(run_id, job, run_open=True))
            for b in boxes:
                if current[b] is not None:
                    continue
                run = run_store.get_run(derived[b])
                if run is None:
                    continue
                open_run = str(run.get("status")) == "running"
                jobs = node_queue.list_jobs_for_run(derived[b])
                if jobs:
                    current[b] = _reservation(derived[b], jobs[0], run_open=open_run)
                elif not open_run:
                    current[b] = Reservation(derived[b], "", str(run.get("status") or "cancelled"), None, False)
        except Exception as exc:
            self._fail("cluster reservations", exc)
            return None
        return {b: (current[b], tuple(stale[b])) for b in boxes}

    def node_reservation(self, run_id: str) -> tuple[bool, Reservation | None]:
        """`(readable, the reservation)` of one derived run (`cluster_reservation_id`), read WHATEVER its run's status:
        a run closed from outside reads `run_open=False` — the sticky failure, never a quiet re-ask — and a run with no
        job yet reads its status with no job id. `(True, None)`: never asked; `(False, None)`: the rows cannot be read."""
        from queue_workflows import node_queue, run_store

        try:
            run = run_store.get_run(run_id)
            if run is None:
                self._ok()
                return True, None
            open_run = str(run.get("status")) == "running"
            jobs = node_queue.list_jobs_for_run(run_id)
        except Exception as exc:
            self._fail("node reservation", exc)
            return False, None
        self._ok()
        if jobs:
            return True, _reservation(run_id, jobs[0], run_open=open_run)
        if open_run:
            # AN OPEN RUN WITH NO JOB YET (the enqueue failed after the run row, or the LB died between the two): not
            # asked yet — the pair's reading (`cluster_reservations`) — and `reserve_for_cluster` enqueues under it.
            return True, None
        return True, Reservation(run_id, "", str(run.get("status") or ""), None, open_run)

    def node_reservations(self, box: str, cluster: str) -> tuple[Reservation, ...] | None:
        """EVERY OPEN reservation one-host node `cluster` holds or asked for on `box` (its `inputs.cluster`, a cluster
        purpose, pinned there), oldest first — the node's STALE scan, as the pair finds its stale ones: one that is not
        the current start's (its start file lost, a derivation from an older LB) is returned once the node is off,
        never left holding the box for `max_hold_s`. None when the rows cannot be read — never "none"."""
        from queue_workflows import db, node_queue

        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT id FROM workflow_runs WHERE workflow_name = %(wf)s AND status = 'running' "
                            "ORDER BY created_at", {"wf": self._cfg.lease_workflow})
                run_ids = [str(r["id"]) for r in cur.fetchall()]
            out = []
            for run_id in run_ids:
                for job in node_queue.list_jobs_for_run(run_id):
                    inputs = job.get("inputs") or {}
                    if (is_cluster_purpose(inputs.get("purpose")) and inputs.get("cluster") == cluster
                            and box in [str(b) for b in (job.get("force_box") or [])]):
                        out.append(_reservation(run_id, job, run_open=True))
        except Exception as exc:
            self._fail("node reservations", exc)
            return None
        self._ok()
        return tuple(out)

    def parked_by(self, labels) -> dict[str, str | None] | None:
        """`{label: who parked it}` — READ ONLY, for the words on the page: a reservation on a parked
        lane waits, and nothing here ever answers a park. None for a lane that is ON; None overall
        when unreadable."""
        from queue_workflows import worker_control

        out: dict[str, str | None] = {}
        try:
            for label in labels:
                row = worker_control.get_worker_control(label, "gpu") or {}
                out[label] = ((row.get("requested_by") or "unknown")
                              if row.get("desired_state") == "off" else None)
        except Exception as exc:
            self._fail("parks", exc)
            return None
        return out

    def live_lanes(self, labels) -> dict[str, bool] | None:
        """`{label: that gpu lane heartbeated inside the engine's stale window}`. None when unreadable, and
        None when NO gpu lane heartbeats at all: an empty table is "we cannot tell", the same line
        `vllm_capacity` draws — never "every lane is dead"."""
        from queue_workflows import db
        from queue_workflows.node_queue import _project, _stale_worker_after_s

        labels = tuple(labels)
        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT host_label FROM worker_heartbeats "
                    "WHERE queue = 'gpu' AND project = %s "
                    "  AND last_seen > now() - make_interval(secs => %s)",
                    (_project(None), int(_stale_worker_after_s())))
                live = {str(r[0] if isinstance(r, tuple) else r["host_label"]) for r in cur.fetchall()}
        except Exception as exc:
            self._fail("live lanes", exc)
            return None
        if not live:
            return None
        return {label: label in live for label in labels}

    def running_gpu_jobs(self, labels, exclude=()) -> dict[str, int] | None:
        """`{label: running gpu node-jobs claimed by that lane}`, leaving out the job ids in `exclude`
        (our own reservations); None when unreadable — never 0."""
        from queue_workflows import db
        from queue_workflows.node_queue import _project

        labels = tuple(labels)
        exclude = tuple(str(j) for j in exclude if j)
        if not labels:
            return {}
        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT claimed_by, COUNT(*) AS n FROM workflow_node_jobs "
                    "WHERE queue = 'gpu' AND status = 'running' AND project = %s "
                    "AND claimed_by IN (" + ", ".join(["%s"] * len(labels)) + ")"
                    + (" AND id NOT IN (" + ", ".join(["%s"] * len(exclude)) + ")" if exclude else "")
                    + " GROUP BY claimed_by",
                    (_project(None), *labels, *exclude))
                rows = cur.fetchall()
        except Exception as exc:
            self._fail("running gpu jobs", exc)
            return None
        out = {label: 0 for label in labels}
        for r in rows:
            label, n = (r[0], r[1]) if isinstance(r, tuple) else (r["claimed_by"], r["n"])
            out[str(label)] = int(n)
        return out

    def recover(self) -> list[Lease]:
        """Adopt our open leases after a restart: every run of ours still 'running' whose
        job is queued/running. A run whose job already ended is closed (cancelled) so it
        cannot linger as a phantom. Other workflows' runs are never touched."""
        from queue_workflows import db, node_queue

        found: list[Lease] = []
        try:
            with db.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM workflow_runs WHERE workflow_name = %(wf)s "
                    "AND status = 'running' ORDER BY created_at",
                    {"wf": self._cfg.lease_workflow},
                )
                run_ids = [str(r["id"]) for r in cur.fetchall()]
            for run_id in run_ids:
                jobs = node_queue.list_jobs_for_run(run_id)
                if any(is_cluster_purpose((j.get("inputs") or {}).get("purpose")) for j in jobs):
                    continue        # a cluster reservation: the cluster loop's to observe, and to end
                live = [j for j in jobs if str(j.get("status")) in ("queued", "running")]
                if not live:
                    self.release(Lease(lease_id=run_id, job_id="", created_at=self.clock()))
                    continue
                job = live[0]
                # WHICH KIND we adopted, read off the row itself. An OLLAMA (fallback) lease an LB from before
                # the fallback's retirement took (2026-10-17) is nobody's now — no plan would ever release it,
                # and it would block that box's LLM lane until its max-hold ran out — so it is RELEASED here.
                kinds = [str(s) for s in (job.get("server_types") or [])]
                if RETIRED_FALLBACK_KIND in kinds:
                    log.warning("[leases] releasing %s: a fallback lease (the fallback tier is retired)", run_id[:8])
                    self.release(Lease(lease_id=run_id, job_id=str(job["id"]), created_at=self.clock()))
                    continue
                # A KEPT lease (the operator's keep-loaded order, retired 2026-10-17) the same: adopted as demand it
                # would pin its box until the idle rule, for an order nobody can see or end now.
                if (job.get("inputs") or {}).get("purpose") == RETIRED_KEEP_PURPOSE:
                    log.warning("[leases] releasing %s: a kept lease (keep loaded is retired)", run_id[:8])
                    self.release(Lease(lease_id=run_id, job_id=str(job["id"]), created_at=self.clock()))
                    continue
                lease = Lease(lease_id=run_id, job_id=str(job["id"]), kind=VLLM, created_at=self.clock())
                found.append(self._reconcile(lease, job))
        except Exception as exc:
            self._fail("recover", exc)
            return []
        self._ok()
        if found:
            log.info("[leases] recovered %d open lease(s)", len(found))
        return found
