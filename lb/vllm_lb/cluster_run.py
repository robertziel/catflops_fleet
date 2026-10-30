"""THE RUN — which of the fleet's declared clusters this LB runs NOW (worklog/parallel-clusters.md; operator, 2026-10-17).

The fleet file is a CATALOG: every cluster this fleet CAN run, alternatives included (two may share a box). The run is
the subset that is ACTIVE — each with its loop, its hub view, its cluster backend and, for a pair, its switch — and it
lives in memory alone: the clusters marked `run_at_boot` are activated when the LB starts (a one-cluster fleet's one,
as always), the panel's "+" tab activates another for THIS run, and an LB restart forgets every activation. Nothing
here writes the fleet file or lb/state/.

A cluster may be activated or started only while every box of its ranks is FREE (`holders`) — or SHARED: the clusters
holding it and this one all state their memory need and the sum fits the box's one pool (`fleet.share_problem`: a
unified-memory box's gpu_ram_max_gib; each need is `fleet.pool_need` at its stored wish, the Start's at what it asks).
Anything else — an unknown need, a discrete box, a sum over the bound — is refused naming the box, its holders and the
GiB. A box is HELD by any DECLARED cluster — in this run or not — that
  * WANTS to be up: its stored wish says so (a pair's switch `mode: cluster`, a node's intent `state: up`). Read off
    disk, so it holds from the moment a Start is written, before any tick observes it (AUDIT-FINAL P1-1: two Starts
    with no tick between were both taken), and after an LB restart forgot the "+" activation that started it — the
    run is memory, the engine is not (P1-2). An unreadable wish holds too: it cannot be proven free;
  * or, while ACTIVE, has a view anything but `off` — `unknown` (not observed yet), starting, finishing, failed: it may
    still have an engine on the box.
An OBSERVED node (control `none`) starts nothing, so activating one needs no free box; and a one-host node (`none` or
`local`) holds no box against a pair (`fleet.measured_beside`): the memory it uses is the box's, which the pair's own
start check measures (tp-rank's gate on MemAvailable) — never a need it would have to state. A node starting beside a
running pair is not measured, so the pair holds the box against it. Beside a node this LB sizes from stated needs it holds
the box as any holder. A second engine on one GPU beyond what its pool holds is the one thing this must never allow. The same answer gates a Start (proxy.py), and a
Start's check and its write are ONE step under `lock` (`start_guard`), so there is one definition of "this box is
taken" and no window between asking it and taking the box. Deactivation removes an `off` cluster from the run; one that
is not off is stopped first, through its own switch — never dropped while its engine runs. A cluster whose launch the
fleet refuses (`unlaunchable`, e.g. a card with no measured start memory) is declared but never activated.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

from vllm_lb import topology
from vllm_lb.catalog import CLUSTER_MODE
from vllm_lb.cluster_defaults import desired_store
from vllm_lb.config import Config
from vllm_lb.fleet import measured_beside
from vllm_lb.hub import Hub
from vllm_lb.perflog import BootRecorder

log = logging.getLogger("vllm_lb.cluster_run")

#: tp-rank launch settings the LB passes through from its own environment, when set. CLUSTER_IB_HCA / CLUSTER_IB_GID_INDEX
#: are the pair's RoCE ports and their RoCE v2 GID index (lb/.env.example), which an engine that opens them itself
#: needs (TensorFold's one-shot all-gather; tp-rank `tensorfold_cmd` refuses a RoCE launch without them).
#: CLUSTER_WORKER_UNIT is the broker worker's boot backstop `heal` restarts on a box — this deployment's unit name,
#: tp-rank defaults none (unset: heal remounts and says no backstop was named).
TP_RANK_SETTINGS = ("CLUSTER_CACHE_DIR", "CLUSTER_GPU_TELEMETRY", "CLUSTER_IB_HCA", "CLUSTER_IB_GID_INDEX",
                    "CLUSTER_WORKER_UNIT")


def tp_rank_env(cfg: Config) -> dict[str, str]:
    """The settings EVERY tp-rank call carries for `cfg`'s one cluster: what only the fleet file holds — the port the
    head is probed on, the head's interconnect, the NIC the ranks bind, the RAM share a start may take and the
    cluster's prefill budget (`max_num_batched_tokens`: vLLM's --max-num-batched-tokens when neither the operator's
    knob nor the model's own says; CLUSTER_MAX_NUM_BATCHED_TOKENS is retired from the LB's own environment and
    tp-rank defaults none) — plus whichever launch settings the operator set (`TP_RANK_SETTINGS`). A cluster that
    states no budget is refused here, naming the key, never launched at a number nobody chose."""
    (cluster,) = cfg.clusters
    if cluster.max_num_batched_tokens is None:
        raise ValueError(f"the fleet file states no prefill budget for cluster {cluster.name!r}: set "
                         f"clusters.{cluster.name}.max_num_batched_tokens (vLLM --max-num-batched-tokens, e.g. 8192)")
    env = {"CLUSTER_PORT": str(urlsplit(cfg.cluster_url).port or 80),
           "CLUSTER_MASTER_ADDR": cfg.cluster_hosts[0].interconnect,
           # The NIC NCCL and Gloo bind on each rank (the fleet's `iface`): the deployment's link, never the script's guess.
           "CLUSTER_IFACE": cluster.iface or "",     # none named: tp-rank refuses the launch, naming it
           # The fleet's share of each box's MemTotal a start may take: tp-rank's memory guard kills under the rest.
           "CLUSTER_RAM_USABLE_FRACTION": str(cluster.ram_usable_fraction),
           "CLUSTER_MAX_NUM_BATCHED_TOKENS": str(cluster.max_num_batched_tokens)}
    env.update(topology_facts(cfg, cluster.name))
    env.update((key, os.environ[key]) for key in TP_RANK_SETTINGS if key in os.environ)
    return env


def topology_facts(cfg: Config, name: str) -> dict[str, str]:
    """THE TOPOLOGY tp-rank is told (vllm_lb/executors.py `Topology.facts`): the TP size, the node count, the GPUs a
    rank sees, the link, the cards' memory family, the box's platform — from the fleet file, never tp-rank's own
    assumption (it has none since 2026-10-30). One env serves every rank (SshExecutor adds only the box's address), so
    the facts must be the same on every rank box; a fleet whose boxes differ is refused naming the first difference
    (docs/design/engines.md, Open items: per-rank facts)."""
    from vllm_lb.executors import topology_of
    if cfg.fleet is None or name not in cfg.fleet.clusters:
        raise ValueError(f"cluster {name!r} has no fleet file entry to read its topology from (LB_FLEET)")
    fc = cfg.fleet.clusters[name]
    topo = topology_of(fc, cfg.fleet.boxes)
    per_box = {b.name: topo.facts(b) for b in topo.boxes}
    first = next(iter(per_box.values()))
    for box, facts in per_box.items():
        differs = {k: (first[k], v) for k, v in facts.items() if first[k] != v}
        if differs:
            raise ValueError(f"cluster {name!r}: box {box!r} differs from the head in {differs}: one tp-rank env serves "
                             f"every rank")
    return first


class RunRefused(ValueError):
    """An activation or deactivation the run refuses, with the HTTP status the API answers it with."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind = status, kind


class ClusterLoops:
    """EVERY active cluster's loop, by its fleet name, ticked in activation order — the one object the Controller's
    cluster phase holds. Each loop was built from its own one-cluster Config (`Config.for_cluster`) and publishes under
    its own name (`Hub.set_cluster`). Ticks copy the dict: an activation from the proxy thread may land mid-tick."""

    def __init__(self, loops: dict | None = None, after_tick=None) -> None:
        self.loops: dict = dict(loops or {})
        #: Called after every tick (`ClusterRun.publish_wanted`): the clusters wanted outside this run, re-read.
        self.after_tick = after_tick

    def tick(self, now: float) -> None:
        for loop in list(self.loops.values()):
            loop.tick(now)
        if self.after_tick is not None:
            self.after_tick()


def build_cluster_loop(cfg: Config, hub: Hub, leases, switch, catalog=None):
    """ONE cluster's loop (`cfg` its one-cluster Config), or None. Built only AFTER the proxy listens: nothing the
    cluster needs (the SSH executor, its key) may cost the gateway its port. `leases` is None with the broker off.

    A ONE-HOST node (vllm_lb/topology.py) gets the same loop with no executor and no pair switch: it reads the node's
    HTTP API through the probe and publishes the cluster view — and, for a node the LB requests (control `local`),
    releases the operator's wish as the intent its host applies (node_control.py): a file, never a channel."""
    from vllm_lb.cluster_loop import ClusterLoop

    if cfg.cluster_configured and cfg.cluster_control in topology.ONE_HOST:
        # A node the LB requests records its starts in the pair's boot log (one line per up intent released); one it
        # only observes started nothing here, so it records none.
        boots = BootRecorder.from_config(cfg) if cfg.cluster_control == topology.LOCAL else None
        return ClusterLoop(cfg, hub, leases, executor=None, store=None, catalog=catalog, boots=boots)
    if switch is None:
        return None
    from vllm_lb.executors import executor_class

    # The executor its control runs it by (executors.py; the ssh pair: tp-rank over SSH). tp-rank ships next to this
    # package and is fed to each box on every call. It is told what only this LB's configuration holds: the port the
    # head is probed on, the head's interconnect, and the TOPOLOGY (TP, nodes, GPUs, link, memory, platform) — plus
    # whichever launch settings the operator set.
    env = tp_rank_env(cfg)
    executor_cls = executor_class(cfg.cluster_control)
    if executor_cls is None:
        raise ValueError(f"cluster {cfg.cluster_name!r}: control {cfg.cluster_control!r} is run by no executor here")
    executor = executor_cls(
        key_path=cfg.cluster_ssh_key, known_hosts=cfg.cluster_known_hosts,
        script_path=str(Path(__file__).resolve().parent.parent / "cluster" / "tp-rank"), env=env)
    # The cluster boot log lives beside the perf windows and is switched off with them (LB_PERF=0).
    return ClusterLoop(cfg, hub, leases, executor=executor, store=switch, boots=BootRecorder.from_config(cfg))


def build_cluster_loops(cfg: Config, hub: Hub, leases, switches: dict, catalog=None) -> ClusterLoops:
    """One loop per cluster the hub holds a view for (`Hub.cluster_cfgs`: the boot clusters, or whatever is active);
    `switches`: each ssh cluster's switch by name — a driven pair without one gets no loop, as before."""
    loops = {}
    for name, one in hub.cluster_cfgs.items():
        loop = build_cluster_loop(one, hub, leases, switches.get(name), catalog=catalog)
        if loop is not None:
            loops[name] = loop
    return ClusterLoops(loops)


class ClusterRun:
    """The active set (see the module). `hub.cluster_cfgs` IS the set — the hub holds each active cluster's config,
    view and backend; this object adds the loops and the pair switches, and the rules for joining and leaving."""

    def __init__(self, cfg: Config, hub: Hub, catalog=None) -> None:
        self.cfg, self.hub, self.catalog = cfg, hub, catalog
        self.leases = None
        self.switches: dict = {}
        #: Stores of the ssh clusters NOT in this run, kept only to read their switch file (`wants_up`).
        self._stores: dict = {}
        self.loops = ClusterLoops(after_tick=self.publish_wanted)
        #: THE RUN'S ONE LOCK: activation, deactivation and a Start's check-and-write (`start_guard`). Re-entrant: a
        #: holder of it asks `blockers`, which reads the switches it guards.
        self._lock = threading.RLock()
        self.lock = self._lock

    # ── boot ────────────────────────────────────────────────────────────────────────────────────

    def boot_switches(self) -> dict:
        """One switch per ACTIVE ssh cluster, by name — built before the hub's first view so the pre-tick
        `recommended` is the switch's model's (`__main__`)."""
        with self._lock:
            for name, one in self.hub.cluster_cfgs.items():
                if one.cluster_control == topology.SSH and name not in self.switches:
                    self.switches[name] = desired_store(one, catalog=self.catalog)
            return dict(self.switches)

    def start(self, leases) -> ClusterLoops:
        """The loops of every cluster the hub holds (the boot set), once the proxy listens and the broker adapter — if
        any — exists. `leases` is kept for the clusters activated later."""
        self.leases = leases
        self.boot_switches()
        with self._lock:
            loops = build_cluster_loops(self.cfg, self.hub, leases, self.switches, catalog=self.catalog).loops
            self.loops.loops.update({n: self._with_peers(n, loop) for n, loop in loops.items()})
        self.publish_wanted()
        return self.loops

    # ── what each declared cluster's stored wish says ───────────────────────────────────────────

    def _one(self, name: str) -> Config | None:
        """Cluster `name`'s one-cluster Config — the run's for an active one, else derived — or None when incomplete."""
        one = self.hub.cluster_cfgs.get(name) or self.cfg.for_cluster(name)
        return one if one.cluster_configured else None

    def wants_up(self, name: str) -> bool:
        """Does cluster `name`'s STORED wish say it is wanted up — in this run or not? A pair's switch file (`mode:
        cluster`, its active switch or a fresh read of the same file), a node's intent (`state: up`). An observed node
        stores none. An unreadable intent is True: a box is never called free on a file nobody could read."""
        one = self._one(name)
        if one is None:
            return False
        if one.cluster_control == topology.SSH:
            switch = self.switches.get(name)
            if switch is None:                 # not in this run: its switch file, read through a store kept per name
                switch = self._stores.get(name)
                if switch is None:
                    switch = self._stores[name] = desired_store(one, catalog=self.catalog)
            # A switch it could not read is reported as its DEFAULT (classic) with `last_error` set — never free on that.
            return switch.load().mode == CLUSTER_MODE or bool(switch.last_error)
        if one.cluster_control == topology.LOCAL:
            from vllm_lb import node_control as nc
            try:
                intent = nc.IntentStore(one.cluster_state_path).read()
            except ValueError:
                return True
            return intent is not None and intent["state"] == nc.UP
        return False

    def wanted_off_run(self) -> list[str]:
        """The declared clusters NOT in this run whose stored wish still wants them up: an engine this run did not
        start and does not drive (an LB restart forgot its activation)."""
        return [c.name for c in self.cfg.clusters if c.name not in self.hub.cluster_cfgs and self.wants_up(c.name)]

    def publish_wanted(self) -> None:
        """Hand the hub the clusters wanted outside this run, so `Snapshot.cluster_wanted` — the classic planner's and
        the router's "a cluster is wanted" — counts their engines too. Every tick, and once at start."""
        self.hub.set_wanted_off_run(self.wanted_off_run())

    # ── the one definition of "this box is taken" ───────────────────────────────────────────────

    def holding(self, name: str) -> bool:
        """Does declared cluster `name` hold its boxes (see the module): its stored wish wants it up, or — active — its
        view is anything but `off`."""
        view = self.hub.cluster_views.get(name)
        return (view is not None and view.get("state") != "off") or self.wants_up(name)

    def holders(self) -> dict[str, str]:
        """box → the DECLARED cluster holding it (`holding`), active in this run or not, in the fleet's order."""
        out: dict[str, str] = {}
        for c in self.cfg.clusters:
            if self.holding(c.name):
                for host in c.hosts:
                    out.setdefault(host.box, c.name)
        return out

    def blockers(self, name: str, launch: dict | None = None) -> list[dict]:
        """`[{box, cluster, reason}]`: which of cluster `name`'s boxes other clusters hold and it may not share (`cluster`:
        the first holder; `reason` names every holder and the GiB) — [] when it may run. `launch`: the launch keys a Start
        asks for (a local node's wish as it would be written); None = its stored wish's."""
        cluster = self._declared(name)
        if cluster.control == topology.NONE:
            return []  # it starts nothing: activating it routes to what its own deployment already runs
        # A node UP already (its view not off) relaunches in the place it holds: a pair beside it started measuring it.
        up = (self.hub.cluster_views.get(name) or {}).get("state") not in (None, "off")
        out = []
        for h in cluster.hosts:
            held = [c.name for c in self.cfg.clusters
                    if c.name != name and any(x.box == h.box for x in c.hosts) and self.holding(c.name)
                    and not measured_beside(cluster.control, c.control)
                    and not (up and measured_beside(c.control, cluster.control))]
            # BESIDE A PAIR (ssh) that measures the box: judged on its rank's MemAvailable, never on stated needs.
            measured = [n for n in held if measured_beside(self._declared(n).control, cluster.control)]
            for n in measured:
                why = self._beside_problem(n, h.box, self._need(name, launch), name)
                if why:
                    # AN ALERT, NOT A BLOCK (operator 2026-10-24: "just estimate ram usage and show alert, but allow
                    # proceeding"): the Start goes through and the panel says the estimate.
                    out.append({"box": h.box, "cluster": n, "reason": f"box {h.box} is held by cluster {n}: {why}",
                                "alert": True})
            held = [n for n in held if n not in measured]
            if not held:
                continue
            why = self._share_problem(h.box, {name: self._need(name, launch), **{n: self._need(n) for n in held}})
            if why:
                out.append({"box": h.box, "cluster": held[0],
                            "reason": f"box {h.box} is held by cluster {', '.join(held)}: {why}"})
        return out

    def _beside_problem(self, holder: str, box: str, need: float | str, name: str) -> str:
        """May node `name` (`need` GiB, or why unknown) start on `box` beside pair `holder` — "" when the box's
        MemAvailable, as the pair's rank measures it, less the need leaves the danger floor (`cluster_floor_gib`)."""
        view = self.hub.cluster_views.get(holder) or {}
        if view.get("state") != "running":
            return f"it is {view.get('state') or 'not observed'}: its use of the box is not settled"
        avail = ((view.get("boxes") or {}).get(box) or {}).get("mem_available_gib")
        if not isinstance(avail, (int, float)):
            return "the box's free memory is not measured"
        if isinstance(need, str):
            return f"{name}'s memory need is unknown ({need})"
        floor = self.cfg.cluster_floor_gib
        if avail - need < floor:
            return (f"{avail:.1f} GiB available, {name} needs {need:.2f} GiB: it would leave {avail - need:.1f} GiB, "
                    f"under the {floor:g} GiB floor")
        return ""

    def _need(self, name: str, launch: dict | None = None) -> float | str:
        """Cluster `name`'s GiB of its box's pool (`fleet.pool_need`) at `launch`, else at its stored wish — or why
        it cannot be said."""
        from vllm_lb.fleet import pool_need
        if self.cfg.fleet is None:
            return "no fleet file states it"
        if launch is None:
            one = self._one(name)
            if one is not None and one.cluster_control == topology.LOCAL:
                from vllm_lb import node_control as nc
                try:
                    wish = nc.IntentStore(one.cluster_state_path).read()
                except ValueError as exc:
                    return str(exc)
                launch = wish["launch"] if wish is not None else {}
        try:
            return pool_need(self.cfg.fleet.clusters[name], launch)
        except ValueError as exc:
            return str(exc)

    def _share_problem(self, box: str, needs: dict) -> str:
        from vllm_lb.fleet import share_problem
        if self.cfg.fleet is None:
            return f"no fleet file states box {box}'s memory: one cluster holds it whole"
        return share_problem(self.cfg.fleet.boxes[box], needs)

    def peers(self, name: str) -> dict[str, bool]:
        """The other declared clusters on cluster `name`'s boxes → whether each holds its box now (`holding`): whose
        broker reservation a node of `name` may share, and for whom it keeps its own (cluster.plan_node_hold)."""
        boxes = {h.box for h in self._declared(name).hosts}
        with self._lock:
            return {c.name: self.holding(c.name) for c in self.cfg.clusters
                    if c.name != name and boxes & {h.box for h in c.hosts}}

    def start_guard(self, name: str, launch: dict | None = None):
        """The lock a Start of `name` (asking for `launch`, `blockers`) holds from its check to its write (proxy.py), and
        its blockers under it: the caller writes the wish only while `blockers` is [] — one step, so two Starts cannot
        both pass the check."""
        return _Guard(self, name, launch)

    def _with_peers(self, name: str, loop):
        """`loop`, told its peers on the box (`peers`) — or None."""
        if loop is not None:
            loop.peers = lambda: self.peers(name)
        return loop

    # ── the panel's words ───────────────────────────────────────────────────────────────────────

    def declared(self) -> list[dict]:
        """`/lb/status.declared`: every cluster the fleet declares (its name, control, boxes, TP, model, boot) — active or not, and for one that is not, whether it
        may be activated now or which box blocks it and who holds that box. Data for the "+" tab, in the fleet's order."""
        out = []
        for c in self.cfg.clusters:
            fleet_cluster = self.cfg.fleet.clusters.get(c.name) if self.cfg.fleet is not None else None
            found = self.blockers(c.name)
            blocked = [b for b in found if not b.get("alert")]
            # `tp`: the fleet's DERIVED tensor-parallel size (its serving cards) — None where no fleet file says it.
            active = c.name in self.hub.cluster_cfgs
            # `holding`: its stored wish wants it up (or, active, it is not off); `unlaunchable`: why it never activates.
            out.append({"name": c.name, "control": c.control, "boxes": [h.box for h in c.hosts],
                        "tp": fleet_cluster.tp if fleet_cluster is not None else None, "boot": c.boot,
                        "default": c.default,
                        "model": fleet_cluster.model.ref if fleet_cluster is not None and fleet_cluster.model else None,
                        "active": active, "blocked_by": blocked, "holding": self.holding(c.name),
                        "alerts": [b["reason"] for b in found if b.get("alert")],
                        "unlaunchable": c.unlaunchable,
                        "available": not active and not blocked and not c.unlaunchable})
        return out

    # ── joining and leaving the run ─────────────────────────────────────────────────────────────

    def _declared(self, name: str):
        cluster = next((c for c in self.cfg.clusters if c.name == name), None)
        if cluster is None:
            raise RunRefused(404, "cluster_unknown",
                             f"no cluster {name!r}: the fleet declares {', '.join(c.name for c in self.cfg.clusters)}")
        return cluster

    def activate(self, name: str) -> dict:
        """Add declared cluster `name` to the run — its view, backend, switch and loop — for this run only. Refused
        (RunRefused: 404 unknown, 409 already active, 409 blocked naming each box and its holder, 409 not configured)."""
        with self._lock:
            declared = self._declared(name)
            if name in self.hub.cluster_cfgs:
                raise RunRefused(409, "cluster_active", f"cluster {name!r} is already in this run")
            if declared.unlaunchable:
                raise RunRefused(409, "cluster_unlaunchable", f"cluster {name!r} cannot launch: {declared.unlaunchable}")
            blocked = [b for b in self.blockers(name) if not b.get("alert")]
            if blocked:
                raise RunRefused(409, "cluster_blocked", f"cluster {name!r} cannot run now: "
                                 + "; ".join(b["reason"] for b in blocked))
            one = self.cfg.for_cluster(name)
            if not one.cluster_configured:
                raise RunRefused(409, "cluster_incomplete", f"cluster {name!r} is not completely configured")
            self.hub.add_cluster(one)
            switch = None
            if one.cluster_control == topology.SSH:
                switch = self.switches[name] = desired_store(one, catalog=self.catalog)
            loop = self._with_peers(name, build_cluster_loop(one, self.hub, self.leases, switch, catalog=self.catalog))
            if loop is not None:
                self.loops.loops[name] = loop
            log.info("[run] cluster %s activated for this run", name)
            self.publish_wanted()
            return {"active": list(self.hub.cluster_cfgs)}

    def deactivate(self, name: str) -> dict:
        """Remove active cluster `name` from the run. Only an `off` cluster leaves (409 otherwise: stop it first, through
        its own switch) — never one whose engine may still run."""
        with self._lock:
            self._declared(name)
            if name not in self.hub.cluster_cfgs:
                raise RunRefused(409, "cluster_inactive", f"cluster {name!r} is not in this run")
            state = self.hub.cluster_views.get(name, {}).get("state")
            if state != "off":
                raise RunRefused(409, "cluster_not_off", f"cluster {name!r} is {state or 'unknown'}: stop it first "
                                 f"(POST /lb/clusters/{name} {{\"enabled\": false}}), then deactivate it")
            self.loops.loops.pop(name, None)
            self.switches.pop(name, None)
            self.hub.remove_cluster(name)
            log.info("[run] cluster %s left this run", name)
            self.publish_wanted()
            return {"active": list(self.hub.cluster_cfgs)}


class _Guard:
    """`ClusterRun.start_guard`: holds the run's lock; `blockers` asked under it."""

    def __init__(self, run: ClusterRun, name: str, launch: dict | None) -> None:
        self.run, self.name, self.launch = run, name, launch

    def __enter__(self) -> list[dict]:
        self.run.lock.acquire()
        try:
            return [b for b in self.run.blockers(self.name, self.launch) if not b.get("alert")]
        except BaseException:
            self.run.lock.release()
            raise

    def __exit__(self, *exc) -> None:
        self.run.lock.release()
