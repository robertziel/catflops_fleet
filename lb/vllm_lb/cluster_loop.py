"""The cluster's control loop: read the switch, observe the fleet, plan, act, publish.

One phase of the Controller's tick (worklog/vllm-cluster-control-plane.md, worklog/vllm-gateway-program.md
pass 2). Every observation is fresh or explicitly absent, and every action is idempotent, so a crash
between any two lines is recovered by the next tick of the next process — the only thing it keeps is how
long a reservation has sat back in the queue, and losing that only restarts the grace.

SSH is PACED. A GB10 box is asked for its status every `cluster_observe_s` while anything is in
motion and every `cluster_idle_observe_s` at rest; actions are applied only on a tick whose
observation is fresh, and the tick after an action looks again. (The previous attempt asked
both GB10 boxes every 2 s forever and ran box-a's sshd into MaxStartups.) broker_parrot's rows are
read every tick — they are cheap, and the reservation is the fence.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from vllm_lb import topology
from vllm_lb.atomic import write_json
from vllm_lb.catalog import RANK_ROLES, ModelCatalog, availability_key
from vllm_lb.cluster import (CLUSTER_MODE, RELEASE, BoxObs, Desired, DesiredStore, Down, Halt, Heal, Observed, Plan,
                             RamHolder, RankObs, ReleaseReservation, Reservation, Reserve, StartHang, Up, at_rest, blockers,
                             NodeDown, limits_view, plan, plan_local, plan_node_hold, plan_observed)
from vllm_lb.topology import observed_view
from vllm_lb.probe import ollama_ps_reply
from vllm_lb.saturation import gpu_view
from vllm_lb.cluster_defaults import engine_pool, kv_choices_run, recommended_for
from vllm_lb.cluster_exec import LOG_LINES_RANGE, launch_problem, resolved_launch
from vllm_lb.memory_estimate import GIB, MemoryEstimate, estimate_memory
from vllm_lb.cluster_startup import StartupPhase, read_startup, stated_prefill_budget
from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases
from vllm_lb.state import CLUSTER, VLLM, Lease

#: How often a dead lane's box is HEALED (tp-rank `heal`): long enough for one heal to play out — the
#: box's nfs-mount-healer ticks every 2 min, the worker's backstop checks its mounts every 60 s, and the
#: worker then boots and heartbeats — and short enough that a boot race costs minutes, not the 1 h 45 min
#: box-b sat worker-less on 2026-09-28. The planner asks for a heal on every tick; this paces it, and a
#: heal that is not due is not an action, so it never cancels the SSH pacing ("acted: look again").
HEAL_EVERY_S = 180.0

log = logging.getLogger(__name__)

#: What the LB last LAUNCHED, beside the switch it launched from (`DesiredStore.path`), so that a restarted LB
#: can still say what the pair that kept running was launched with: `cluster.effective`.
LAUNCH_FILE = "cluster-launch.json"
#: `cluster.effective`, in the order the contract with the page and the performance log states it.
EFFECTIVE_KEYS = ("model_choice", "generation", "max_num_seqs", "max_num_batched_tokens", "num_speculative_tokens",
                  "ple_replicate", "kv_cache_memory", "kv_offload_gib", "max_model_len",
                  "long_prefill_token_threshold", "uncensored", "kv_cache_dtype")

#: The launch record's key for the prefill budget the HEAD SAYS it runs (`cluster_startup.stated_prefill_budget`):
#: `cluster.effective` states it over the one the launch sent, so a launch the engine did not honour shows.
ENGINE_BUDGET_KEY = "engine_max_num_batched_tokens"

#: THE POST-MORTEM OF A HUNG START, beside the switch on the LB's state volume (`/data` = lb/state): one folder per
#: hung attempt, `gen<N>-attempt<k>`, holding each rank's log (`<box>.log`), each rank's stacks as py-spy read them
#: (`<box>-stacks.txt`) and what the LB knew (`hang.json`) — saved BEFORE the start is torn down, because the
#: process py-spy reads dies with it (`_save_hang_evidence`).
HANG_DIR = "start-hangs"
#: How many of those folders are kept, newest first. A folder the LB did not name is never touched: generation
#: 131's own, taken by hand, is `gen131`.
HANG_EVIDENCE_KEEP = 10
_HANG_NAME = re.compile(r"gen\d+-attempt\d+")
#: The states a relaunched start passes on its way back up, in which the page is told which retry it is on.
RELAUNCHING = ("finishing", "restarting", "checking", "reserving", "waiting", "draining", "freeing", "starting")
#: A hung start halted after its relaunches (cluster.plan): the same Halt as a failure's, another word.
STALLED = "stalled"
#: The two states that do not end on their own — and the planner's words for the TICKS AFTER one: a halted rank
#: is an exited rank to it, so a stall reads `failed` from the second tick on unless the cause is kept.
TERMINAL = ("failed", STALLED)
#: The launch record's key for a hang the LB halted: {generation, start_id, detail, at, hang} — read back by a
#: restarted LB so the halted pair is still said to have hung, not to have failed.
HALT_KEY = "halt"

#: How much of the head's log one read asks for: all tp-rank allows. The StartupPhase and every LoadPass's
#: name are read off the WHOLE start — 289 lines (~60 KB) by the head's first /health 200 on 2026-09-15,
#: when the 200-line tail read before had lost the line that begins the load and every line naming the
#: drafter (worklog/vllm-cluster-loading-status.md F5). Still one read per `cluster_logs_every_s`.
START_LOG_LINES = LOG_LINES_RANGE[1]


def _rank(raw) -> RankObs | None:
    r = raw.get("rank") if isinstance(raw, dict) else None
    if not isinstance(r, dict):
        return None
    gen, code, choice = r.get("gen"), r.get("exit_code"), r.get("model_choice")
    reason = r.get("exit_reason")
    present, running = bool(r.get("present")), bool(r.get("running"))
    # LOST, NOT EXITED: the container started before the box's current boot (both on the box's own clock —
    # docker's StartedAt and /proc/stat's btime), and it is not running. Only with both times in hand: a box
    # that cannot say when it booted has not said its rank was lost.
    started, booted = _started_at(raw), _number(raw, "boot_at")
    lost = present and not running and started is not None and booted is not None and started < booted
    return RankObs(present=present, running=running,
                   gen=gen if isinstance(gen, int) and not isinstance(gen, bool) else None,
                   exit_code=code if isinstance(code, int) and not isinstance(code, bool) else None,
                   model_choice=choice if isinstance(choice, str) and choice else None,
                   exit_reason=reason if isinstance(reason, str) else "", lost=lost)


def _number(raw, key: str) -> float | None:
    v = raw.get(key) if isinstance(raw, dict) else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


#: docker's `StartedAt` as tp-rank's status reports it: RFC 3339, UTC, nanoseconds (`2026-10-03T07:25:54.990330365Z`).
_STARTED_AT = re.compile(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?Z")
#: A line stamped up to this long before its container started is still its own: SGLang and vLLM print whole
#: seconds, docker's StartedAt carries nanoseconds.
STAMP_SLACK_S = 2.0


def _started_at(raw) -> float | None:
    """When the box's rank container started, on the BOX's clock (UTC epoch seconds), or None — no rank, no status,
    or a time docker did not write."""
    r = raw.get("rank") if isinstance(raw, dict) else None
    m = _STARTED_AT.fullmatch(r.get("started_at") or "") if isinstance(r, dict) and isinstance(
        r.get("started_at"), str) else None
    if m is None:
        return None
    try:
        whole = datetime(*(int(v) for v in m.groups()[:6]), tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
    return whole + (float(f"0.{m.group(7)}") if m.group(7) else 0.0)


def _said(view: dict) -> dict:
    """What the head's LOG said in a `loading` view — without `taken_gib`, which is the box's own evidence."""
    return {k: v for k, v in view.items() if k != "taken_gib"}


def _flag(raw, key: str) -> bool | None:
    v = raw.get(key) if isinstance(raw, dict) else None
    return v if isinstance(v, bool) else None


def _holders(raw) -> tuple[RamHolder, ...] | None:
    """What this box says its OWN ollama still holds, through the one ollama reader the LB has
    (`probe.ollama_ps_reply`). None = the box could not be asked — never an empty tuple, which would
    read as "nothing is resident" (worklog/vllm-cluster-hold-robustness.md §6)."""
    ps = ollama_ps_reply(raw.get("ollama") if isinstance(raw, dict) else None)
    if ps is None:
        return None
    return tuple(RamHolder(name=str(m["name"]),
                           gib=float(m.get("gpu_bytes") or m.get("size_bytes") or 0) / GIB,
                           expiry=str(m.get("expiry") or ""))
                 for m in ps["models"] if m.get("name"))


def _reservation_view(r: Reservation | None, desired: Desired) -> dict | None:
    """What the page shows of a box's reservation. After a Stop, one that was given back and has ended
    is nothing of ours any more; while the cluster is wanted an ended one stays on screen, because it is
    the reason the cluster failed."""
    if r is None:
        return None
    live_row = r.status in ("queued", "running")
    if desired.mode != CLUSTER_MODE and not r.run_open and not live_row:
        return None
    return {"job_id": r.job_id, "status": r.status if r.run_open or not live_row else "cancelled",
            "claimed_by": r.claimed_by}


def _read_launch(path: str) -> dict | None:
    """The launch record a previous LB left, or None — missing, unreadable, or not one of ours."""
    try:
        with open(path, "rb") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("generation"), int):
        return None
    return raw


class ClusterLoop:
    #: `leases` is the broker adapter, or None with the broker OFF (`Config.broker_enabled`): the loop then never
    #: asks broker_parrot anything, and the plan launches on what the boxes themselves say.
    #:
    #: `executor` None is a node this LB ONLY OBSERVES (control `none`, vllm_lb/topology.py): there is no switch
    #: (`store` None), nothing is planned to be done to it (`cluster.plan_observed`), and the one thing read is the
    #: node's own HTTP API through the probe every backend gets. `catalog` names what it serves.
    def __init__(self, cfg: Config, hub: Hub, leases: BrokerLeases | None, *, executor,
                 store: DesiredStore | None, clock: Callable[[], float] = time.time, boots=None,
                 catalog: ModelCatalog | None = None) -> None:
        self.cfg = cfg
        self.hub = hub
        self.leases = leases
        self.executor = executor
        self.store = store
        self.catalog = catalog if catalog is not None else (store.catalog if store is not None else ModelCatalog.load())
        self.clock = clock
        #: The cluster boot log (`perflog.BootRecorder`): told every tick's state, or None = nothing is recorded.
        self.boots = boots
        self.hosts = cfg.cluster_hosts
        self._status: dict[str, dict | None] = {}
        self._observed_at: float | None = None
        self._state = "unknown"
        self._since = clock()
        self._loading: dict = {}
        self._logs_at: float | None = None
        #: job id → when it was first seen back in the queue under a standing rank of this generation.
        self._requeued_at: dict[str, float] = {}
        #: The other declared clusters on this cluster's boxes → whether each holds its box now (`ClusterRun.peers`, set
        #: by the run) — whose broker reservation a node shares, and for whom it keeps its own.
        self.peers: Callable[[], dict[str, bool]] = dict
        #: box → what it HAS (spec id → Availability), and when every box was last asked. Paced apart
        #: from the status poll: it is a look at several folders, and the picker needs it, not the tick.
        self._availability: dict[str, dict | None] = {}
        self._availability_at: float | None = None
        #: box → its MemAvailable at the moment its rank was launched, so an estimate can be held against
        #: what the box actually gave up (M8). Forgotten with the rank.
        self._mem_at_launch: dict[str, float] = {}
        #: Per box, what it had free before the rank that has since come DOWN launched — `mem_at_launch`
        #: kept past the teardown, so the memory coming back is what says the handover is finished.
        self._mem_release_target: dict[str, float] = {}
        #: box → this tick's MemoryEstimate as the page reads it (`view()`), computed ONCE in `_observed`
        #: so the number the planner gates on and the number the page shows can never differ (L2).
        self._memory: dict[str, dict | None] = {}
        #: box → its last `cluster_mem_samples` MemAvailable observations. The planner is handed their
        #: LOWEST: one lucky sample of a number that swings by the size of the resident ollama model is
        #: not a free box (worklog/vllm-cluster-hold-robustness.md §3, L3).
        self._mem_samples: dict[str, deque] = {}
        #: What the head SERVES, from its own /v1/models — never the ModelChoice we asked it to load.
        self._loaded = None
        #: THE START'S OWN CLOCK: (what `cluster.loading` last said, when it first said it, when it was last
        #: READ saying it). Quiet time is read-to-read, so a log that could not be read adds nothing to it,
        #: and None — never read, or unreadable — is no evidence of a stall. "When it first said it" is the
        #: HEAD'S time where its line states one (`_printed_since`), so a restarted LB states the head's silence
        #: rather than starting it over; else the LB's first read that saw it.
        self._progress: tuple[dict, float, float] | None = None
        #: THE LOAD'S OWN EVIDENCE, off the box: (the head box's MemAvailable when this start's log was first read, the
        #: LOWEST since). `_load_taken` turns it into `loading.taken_gib` while a weight pass is the step — a figure
        #: that moves only when the box has given up another whole GiB, which the clock above counts as progress.
        self._load_mem: tuple[float, float] | None = None
        #: THE LAST LOOK THAT SAW THIS PAIR SERVE (the LB's clock), kept while it is `running` or has dropped back to
        #: `starting`, cleared by anything else (a rescue, a Stop, a new start). A pair that has served and then
        #: stops answering /health is quiet from HERE, never from its finished start's last line — generation 173
        #: (2026-10-09) served twelve minutes, missed one probe, and was called "hung 774 s in “API server up”"
        #: against a 180 s limit, from a startup line printed at 07:17:55.
        self._served_at: float | None = None
        #: (generation, start id, reason) of the last launch the LB refused to build — `Observed.launch_refused`.
        self._refused: tuple[int, str | None, str] | None = None
        #: box → its own clock minus the LB's, from the `date` its last status reply carried (`net.t`): what a
        #: line the box stamped is measured against. Absent: the box sent no clock, and the LB's is taken.
        self._box_offset: dict[str, float] = {}
        #: (generation, the STATE and the FIRST words it ended with) of a pair that is `failed` or `stalled`. The
        #: planner is pure and judges each tick on its own: the tick after a halt it sees two exited ranks (137)
        #: and says `failed: rank on box-a exited`, which is the CONSEQUENCE of the stall, not its cause. While
        #: the planner keeps saying one of those two words for this generation, the page gets the cause — state
        #: and words — as first said (`_pinned`). A hang's halt is also kept in the launch record (`HALT_KEY`),
        #: so a redeployed LB does not call it a failure either.
        self._failure: tuple[int, str, str] | None = None
        #: box → the last heal of its broker worker, as the page shows it: when, whether it worked, and what
        #: it did or why not. Kept after the lane is back — the record of what brought it back.
        self._heals: dict[str, dict] = {}
        #: box → (t, tx_bytes, rx_bytes) of its last RDMA reading, for `_net_rate`.
        self._net_last: dict[str, tuple | None] = {}
        self._net_prev: dict[str, dict | None] = {}
        #: WHAT THE PAIR WAS LAUNCHED WITH: generation, start id and every setting resolved as the launch
        #: resolved it — written when a generation's ranks are sent up, read back by a restarted LB. On disk,
        #: because an LB redeploy leaves a serving pair running, and the defaults it would otherwise re-resolve
        #: (the box's environment, the catalog) may have changed since. An observed node has no switch beside
        #: which to keep one, and was never launched from here.
        self._launch_path = str(Path(store.path).with_name(LAUNCH_FILE)) if store is not None else None
        self._launched: dict | None = _read_launch(self._launch_path) if self._launch_path else None
        #: THE LAST RESCUE of a hung start — which attempt hung, in what step, for how long, the retries allowed and
        #: where its evidence went — with the generation and start id it belongs to. Kept in the launch record too,
        #: so a redeployed LB still knows which retry a start is on and the bound stays a bound.
        rescue = (self._launched or {}).get("rescue")
        self._rescue: dict | None = rescue if isinstance(rescue, dict) else None
        # A HALT THE LAST LB DECIDED: the hung start it stopped stays `stalled` with its reason, not `failed:
        # rank exited (code 137)` — the only thing a restarted LB would otherwise see of it.
        halt = (self._launched or {}).get(HALT_KEY)
        if (isinstance(halt, dict) and isinstance(halt.get("generation"), int)
                and isinstance(halt.get("detail"), str) and halt.get("generation") == self._launched.get("generation")):
            self._failure = (halt["generation"], STALLED, halt["detail"])
        self._hang_root = Path(store.path).with_name(HANG_DIR) if store is not None else None
        #: (generation, start id, attempt) whose evidence was last saved: a teardown retried on the next tick does
        #: not take it twice.
        self._evidence_saved: tuple | None = None
        #: A node the LB requests: the (id, at) of the released intent this LB last SAW SERVE (`_record_local_boot`).
        self._served_intent: tuple | None = None

    # ── one tick ────────────────────────────────────────────────────────────

    def tick(self, now: float) -> None:
        if not self.cfg.cluster_configured:
            self.hub.set_cluster({"configured": False}, self.cfg.cluster_name)
            return
        if self.cfg.cluster_control == topology.LOCAL:
            return self._tick_local(now)
        if self.executor is None:
            return self._tick_observed(now)
        desired, start_id = self.store.current()
        at_rest = desired.mode != CLUSTER_MODE and self._state == "off"
        interval = self.cfg.cluster_idle_observe_s if at_rest else self.cfg.cluster_observe_s
        fresh = self._observed_at is None or now - self._observed_at >= interval
        if fresh:
            self._observe_boxes()
            self._observed_at = now
        # What the boxes HAVE is asked at the picker's idle cadence — except while a start is WAITING on the answer
        # (`checking`: a share not mounted yet, a box that could not look), when it is asked as often as the
        # boxes are observed, so the start goes on the tick the share is there, not up to a minute later.
        every = self.cfg.cluster_observe_s if self._state == "checking" else self.cfg.cluster_idle_observe_s
        if self._availability_at is None or now - self._availability_at >= every:
            self._observe_availability()
            self._availability_at = now
        boxes = tuple(h.box for h in self.hosts)
        labels = tuple(h.label for h in self.hosts)
        if self.cfg.broker_enabled:
            reservations = self.leases.cluster_reservations(boxes, desired.generation, start_id)
            ours = [r.job_id for current, stale in (reservations or {}).values()
                    for r in ((current,) if current is not None else ()) + tuple(stale)]
            observed = replace(self._observed(desired, reservations, self.leases.parked_by(labels),
                                              self.leases.running_gpu_jobs(labels, exclude=ours), now,
                                              lanes=self.leases.live_lanes(labels)),
                               reservations_unsupported=self.leases.cluster_support(),
                               start_rescues=self._rescues(desired, start_id),
                               launch_refused=self._refusal(desired, start_id))
        else:
            # BROKER OFF: nothing is asked of broker_parrot — no reservation exists (an empty set, READ as
            # such, never an unreadable one), and no park, fleet job or lane is there to read.
            observed = replace(self._observed(desired, {}, None, None, now, lanes=None),
                               start_rescues=self._rescues(desired, start_id),
                               launch_refused=self._refusal(desired, start_id))
        p = plan(desired, observed, boxes=boxes, labels=labels,
                 broker=self.cfg.broker_enabled,
                 # How long the RAM wait has been going on, from the state clock the page already shows.
                 freeing_s=(now - self._since) if self._state == "freeing" else 0.0,
                 freeing_deadline_s=self.cfg.cluster_freeing_deadline_s,
                 start_stall_s=self.cfg.cluster_start_stall_s, kill_floor_gib=self.cfg.cluster_kill_floor_gib,
                 start_retries=self.cfg.cluster_start_retries,
                 finishing_s=(now - self._since) if self._state == "finishing" else 0.0,
                 finishing_limit_s=self.cfg.cluster_finishing_limit_s)
        # THE CAUSE OUTRANKS ITS CONSEQUENCES: a pair that failed or stalled is shown as it first ended, for as long
        # as the planner's own word for this generation is one of the two (`_failure`).
        p = self._pinned(p, desired)
        due = [a for a in p.actions if not isinstance(a, Heal) or self._heal_due(a.box, now)]
        # A HUNG ATTEMPT, for the boot log: once its teardown lands (a rescue), or as soon as it is halted.
        hang = None
        if due and fresh:
            if p.hang is not None:
                # EVIDENCE BEFORE TEARDOWN: the Downs (or Halts) below end the processes py-spy reads.
                self._save_hang_evidence(p.hang, desired, start_id, now)
            self._apply(due, start_id, now, desired)
            self._remember_launch_memory(due, observed)
            self._observed_at = None                     # acted: look again next tick
            if p.hang is not None and p.hang.relaunch:
                hang = self._note_rescue(p.hang, desired, start_id, now)
        if p.hang is not None and not p.hang.relaunch:
            hang = {**p.hang.view(), "evidence": self._evidence_name(p.hang, desired)}
            self._note_halt(p, desired, start_id, now)
        self._track(p, now, desired)
        self._note_rest(p, observed, desired)
        self._note_rescued(p, desired, start_id)
        self._record_boot(p, now, desired, start_id, hang)
        self._publish(desired, observed, p, now, start_id)

    def _tick_observed(self, now: float) -> None:
        """One tick for a node this LB ONLY OBSERVES (control `none`): its head is the cluster backend the probe phase
        has just read — whether it answers, what it serves, when its engine process started — and nothing else is
        asked of anything. No switch to read, no executor to call, no boot to record: the LB started nothing. The
        view is the pair's shape (`topology.observed_view`), so the page draws the same card for it."""
        snap = self.hub.snapshot(now)
        head = next((b for b in snap.backends if b.url == self.cfg.cluster_url), None)
        live = bool(head is not None and head.live)
        error = (head.last_error if head is not None else "") or ""
        box = self.hosts[0].box
        p = plan_observed(live=live, error=error, box=box)
        self._note_state(p, now)
        # HOW LONG IT HAS BEEN UP: the engine's own process start, off the /metrics the probe read
        # (`process_start_time_seconds`) — the node's clock is this box's, so no offset is taken.
        measured = self.hub.throughput(head, now) if live else {}
        started = (measured.get("engine_totals") or {}).get("started_at")
        loaded = self.catalog.loaded_model(head.served_models, head.max_model_len) if live else None
        self.hub.set_cluster(name=self.cfg.cluster_name, view=observed_view(
            self.cfg, state=p.state, detail=p.detail, held=p.held, since=self._since, live=live, error=error,
            up_since=started if p.state == "running" else None,
            loaded_model=loaded.view() if loaded is not None else None, catalog_error=self.catalog.error,
            effective=topology.observed_effective(head, measured) if live else None,
            served=head.served_models if live else ()))

    def _tick_local(self, now: float) -> None:
        """One tick for a node the LB REQUESTS and its host APPLIES (control `local`, vllm_lb/node_control.py): read the
        operator's wish (the node's switch), what was released to the host, the host applier's report and the node's
        own HTTP API; plan (`cluster.plan_local`); RELEASE the wish — write the intent the applier applies — when the
        plan says so; publish the observed node's view with what only this topology has (`topology.local_view`).

        What the node is expected to run is the fleet's launch block moved by the last intent its host APPLIED (exit
        0, up) — what the panel cross-checks the node against and the router's seats — never the wish not yet applied."""
        from vllm_lb import node_control as nc
        wish_store = nc.IntentStore(self.cfg.cluster_state_path)
        node_dir = nc.node_dir(self.cfg.cluster_state_path)
        released_store = nc.IntentStore(f"{node_dir}/{nc.INTENT}")
        problems: list[str] = []
        wish = released = None
        try:
            wish = wish_store.read()
        except ValueError as exc:
            problems.append(str(exc))
        try:
            released = released_store.read()
        except ValueError as exc:
            problems.append(str(exc))
        report = nc.read_report(node_dir, now)
        snap = self.hub.snapshot(now)
        head = next((b for b in snap.backends if b.url == self.cfg.cluster_url), None)
        live = bool(head is not None and head.live)
        error = (head.last_error if head is not None else "") or ""
        busy = head.waiting + max(head.running, head.inflight) if live else 0
        # A BOOT NODE NOBODY HAS ASKED ANYTHING OF, AND NOT UP, IS STARTED: the fleet's run_at_boot is the wish —
        # written once, at its boot default, the moment an applier listens to apply it. A node already answering is
        # left as it runs (observed); an unreadable wish is never overwritten.
        if (wish is None and not problems and not live and report["listening"]
                and topology.fleet_cluster(self.cfg).run_at_boot):
            wish = wish_store.write(nc.UP, {}, now=now, force=False)
            log.info("cluster: %s runs at boot: wish #%d up at the fleet's boot default", self.cfg.cluster_name,
                     wish["id"])
        box = self.hosts[0].box
        p = plan_local(desired=wish, intent=released, report=report, live=live, error=error, busy=busy, box=box)
        # THE BOX IS HELD THROUGH broker_parrot FIRST (`Control.reserves`, the broker on): the pair's reservation
        # contract for this node's one box — an `up` reaches the host only once the box's lane holds it
        # (`plan_node_hold`), and the reservation is returned once the node is off.
        hold = self._node_hold(wish, released, live, box, p, now, report)
        if hold is not None:
            p = hold
        if RELEASE in p.actions:
            released = released_store.write(wish["state"], wish["launch"], now=now, force=wish["force"])
            log.info("cluster: released %s intent #%d %s", wish["state"], released["id"], released["launch"])
            p = plan_local(desired=wish, intent=released, report=report, live=live, error=error, busy=busy, box=box)
            hold = self._node_hold(wish, released, live, box, p, now, report)
            if hold is not None:
                p = hold
        self._note_state(p, now)
        applied = report.get("applied") or {}
        launch = source = desired = None
        if applied.get("exit") == 0 and not applied.get("refused") and applied.get("state") == nc.UP:
            launch = topology.applied_launch(topology.fleet_cluster(self.cfg), applied.get("launch") or {})
            if launch is None:
                problems.append(f"the applied intent #{applied.get('id')}: its model is no longer among the fleet's "
                                f"choices")
            else:
                source = f"{self.cfg.fleet.source} + intent #{applied.get('id')}"
        try:
            desired = topology.local_desired(self.cfg, wish)
        except ValueError as exc:
            problems.append(f"the stored wish: {exc}")
        measured = self.hub.throughput(head, now) if live else {}
        started = (measured.get("engine_totals") or {}).get("started_at")
        loaded = self.catalog.loaded_model(head.served_models, head.max_model_len) if live else None
        progress = read_startup(report["log"], now=now) if p.state == "starting" and report.get("log") else None
        loading = progress.view() if progress is not None else {}
        self._record_local_boot(p, now, released, report, loading, serves=launch is not None
                                and nc.same_intent(released, applied) and p.state == "running")
        self.hub.set_cluster(name=self.cfg.cluster_name, view=self._with_reservation(box, topology.local_view(
            self.cfg, desired=desired, applier=topology.applier_view(report, p.state),
            fit=topology.local_fit(self.cfg, wish["launch"] if wish else {}),
            loading=loading, store_error="; ".join(problems),
            state=p.state, detail=p.detail, held=p.held, since=self._since, live=live, error=error,
            up_since=started if p.state == "running" else None,
            loaded_model=loaded.view() if loaded is not None else None, catalog_error=self.catalog.error,
            effective=topology.observed_effective(head, measured) if live else None,
            served=head.served_models if live else (), launch=launch, source=source,
            # WHICH START serves: the applied intent's id — what every minute the node records is stamped with.
            generation=applied.get("id") if launch is not None else None)))

    def _reserves(self) -> bool:
        """Does this node hold its box through broker_parrot? Its cluster's data (`Config.cluster_reserves`: its
        control's `reserves`, or the fleet's `reserve: false`) and the broker on — never a branch on the control's name."""
        return self.cfg.broker_enabled and self.cfg.cluster_reserves

    def _node_hold(self, wish: dict | None, released: dict | None, live: bool, box: str, local: Plan,
                   now: float, report: dict | None = None) -> Plan | None:
        """Read this start's reservation, plan the hold (`cluster.plan_node_hold`), apply its actions; the plan that
        replaces `local`, or None when nothing is reserved (the broker off: as before). Remembers what it read for the
        view (`_with_reservation`).

        THIS START'S IDENTITY (`hold.json` in the node dir, `_start`): minted when an `up` begins from off — never the
        wish's id alone, which restarts at 1 when the wish file is lost and would name an old, closed run — kept through
        relaunches, and dropped once the node is off with nothing held. The run id is
        `cluster_reservation_id(<the start's wish id>, <cluster>:<start>, <box>)`."""
        self._hold_seen = None
        if not self._reserves():
            return None
        from vllm_lb import node_control as nc
        label = self.hosts[0].label
        up = wish is not None and wish["state"] == nc.UP
        node_up = live or (released is not None and released["state"] == nc.UP)
        # OFF: answers nothing and its host applied a down — or was never told anything.
        applied = (report or {}).get("applied")
        node_off = not live and (released is None or (released["state"] == nc.DOWN and nc.same_intent(released, applied)))
        start = self._start()
        readable, current = (True, None) if start is None else self.leases.node_reservation(self._run_id(start, box))
        # THE STALE SCAN (the pair's): this cluster's other open reservations on the box — a lost start file's.
        every = self.leases.node_reservations(box, self.cfg.cluster_name)
        if every is None:
            readable = False
        mine = current.reservation_id if current is not None else None
        strays = tuple(r for r in (every or ()) if r.reservation_id != mine)
        # ONE ROW PER BOX (`plan_node_hold`): with none of its own, the node rides a peer's open reservation on the box.
        peers = self.peers()
        own, ridden = current, None
        if readable and current is None and not strays:
            for peer in peers:
                rows = self.leases.node_reservations(box, peer)
                if rows is None:
                    readable = False
                    break
                ridden = next((r for r in rows if r.status == "running" and r.claimed_by == label), ridden or
                              next(iter(rows), None))
            current = ridden
        # RE-QUEUED UNDER THE RUNNING NODE (a lane restart): since when, for the pair's grace (`REQUEUE_GRACE_S`).
        requeued_s = None
        if current is not None and current.run_open and current.status == "queued" and node_up:
            at = self._requeued_at.setdefault(current.job_id, now)
            requeued_s = now - at
        else:
            self._requeued_at.pop(getattr(current, "job_id", None), None)
        ours = [r.job_id for r in (current, *strays) if r is not None and r.job_id]
        jobs = (self.leases.running_gpu_jobs((label,), exclude=ours) or {}).get(label)
        actions, plan = plan_node_hold(up=up, start=start and start["id"], current=current, readable=readable,
                                       node_up=node_up, node_off=node_off, strays=strays, requeued_s=requeued_s,
                                       support=self.leases.cluster_support(), box=box, label=label, jobs=jobs,
                                       local=local, kept_for=tuple(n for n, holding in peers.items() if holding))
        for action in actions:
            if isinstance(action, Reserve):
                if start is None:
                    start = self._mint_start(wish)
                job_id = self.leases.reserve_for_cluster(box, start["wish"], f"{self.cfg.cluster_name}:{start['id']}",
                                                         model=self._node_served_name(wish),
                                                         cluster=self.cfg.cluster_name)
                log.info("cluster: %s reserve %s: %s", self.cfg.cluster_name, box,
                         "ok" if job_id else self.leases.last_error)
            elif isinstance(action, ReleaseReservation):
                ok = self.leases.release(Lease(lease_id=action.reservation_id, job_id=action.job_id, created_at=now,
                                               kind=CLUSTER, box=box))
                log.info("cluster: %s release %s's reservation %s: %s", self.cfg.cluster_name, box,
                         action.reservation_id[:8], "ok" if ok else self.leases.last_error)
            elif isinstance(action, NodeDown):
                # THE BOX IS NOT HELD FOR IT: the node comes down at once (force) — its card is broker_parrot's again.
                # Written ONCE: a forced down already released stands while its host applies it.
                if released is not None and released["state"] == nc.DOWN and released["force"]:
                    continue
                doc = nc.IntentStore(f"{nc.node_dir(self.cfg.cluster_state_path)}/{nc.INTENT}").write(
                    nc.DOWN, (released or {}).get("launch") or {}, now=now, force=True)
                log.warning("cluster: %s DOWN intent #%d — %s", self.cfg.cluster_name, doc["id"], plan.detail)
        if not up and local.state == "off" and start is not None and not (
                own is not None and own.run_open and own.status in ("queued", "running")):
            self._end_start()                    # off, nothing held: the next `up` is a new start
        if actions and start is not None and ridden is None:
            readable, current = self.leases.node_reservation(self._run_id(start, box))
        self._hold_seen = (readable, current, jobs)
        return plan

    def _hold_path(self) -> str:
        from vllm_lb import node_control as nc
        return f"{nc.node_dir(self.cfg.cluster_state_path)}/{nc.HOLD}"

    def _start(self) -> dict | None:
        """This start's identity (`{id, wish}`), or None — none begun, or the file lost or unreadable (then a node
        still up is taken down by the hold, never re-reserved under a guessed identity)."""
        import json
        try:
            with open(self._hold_path(), encoding="utf-8") as fh:
                doc = json.load(fh)
            return doc if isinstance(doc, dict) and isinstance(doc.get("id"), str) and isinstance(doc.get("wish"), int) \
                else None
        except (OSError, ValueError):
            return None

    def _mint_start(self, wish: dict) -> dict:
        import uuid
        doc = {"id": uuid.uuid4().hex[:16], "wish": int(wish["id"])}
        write_json(self._hold_path(), doc)
        return doc

    def _end_start(self) -> None:
        import os
        try:
            os.remove(self._hold_path())
        except FileNotFoundError:
            pass

    def _run_id(self, start: dict, box: str) -> str:
        from vllm_lb.leases import cluster_reservation_id
        return cluster_reservation_id(start["wish"], f"{self.cfg.cluster_name}:{start['id']}", box)

    def _node_served_name(self, wish: dict | None) -> str:
        """What the wish's model serves (its catalog row's served name) — the reservation declares it, as the pair's
        declares its ModelChoice's."""
        cluster = topology.fleet_cluster(self.cfg)
        ref = ((wish or {}).get("launch") or {}).get("model") or getattr(cluster.launch, "model", None)
        choice = next((m for m in cluster.choices if m.ref == ref), cluster.model)
        return choice.spec.served_name

    def _with_reservation(self, box: str, view: dict) -> dict:
        """The view with what the box's reservation is — `reservation` {state, detail} and the box's `reservation` and
        `running_jobs` the stepper reads, as the pair's boxes carry them. Broker off: `unreserved`, said."""
        if not self._reserves():
            why = ("the broker is off (LB_BROKER_ENABLED): nothing reserves " + box + " in broker_parrot"
                   if not self.cfg.broker_enabled else f"cluster {self.cfg.cluster_name} reserves nothing (its fleet "
                   f"entry: control {self.cfg.cluster_control}, its reserve rule)")
            return {**view, "reservation": {"state": "unreserved", "detail": why}}
        readable, current, jobs = getattr(self, "_hold_seen", None) or (False, None, None)
        alive = current is not None and current.run_open and current.status in ("queued", "running")
        state = ("unknown" if not readable else "none" if current is None else
                 "held" if alive and current.status == "running" else "queued" if alive else "ended")
        boxes = {**view.get("boxes", {})}
        boxes[box] = {**boxes.get(box, {}), "reservation": {"status": current.status if current else None,
                                                             "job_id": current.job_id if current else None},
                      "running_jobs": jobs}
        return {**view, "boxes": boxes, "reservation": {
            "state": state, "detail": {"held": f"broker_parrot holds {box} for this node",
                                       "queued": f"{box}'s reservation is queued: its lane has not claimed it",
                                       "none": f"{box} is not reserved",
                                       "ended": f"{box}'s reservation ended: Stop, then Start",
                                       "unknown": "broker_parrot's reservations cannot be read"}[state]}}
    def _record_local_boot(self, p: Plan, now: float, released: dict | None, report: dict, loading: dict, *,
                           serves: bool) -> None:
        """The boot log's view of a node the LB requests — the pair's own recorder (perflog.BootRecorder), one line per
        UP INTENT RELEASED: its generation the released intent's id, its configuration that intent's launch
        (`topology.released_config`), its steps the plan's for that intent (`topology.boot_state`: `waiting` until the
        host's applier takes it up), its end serving, failed in the plan's words (vLLM's root cause, the applier's
        refusal) or stopped. A DOWN intent starts nothing: it closes the start it stops under that start's generation,
        in the plan's words. `serves`: the node answers under the released intent — kept, so that a line opening on
        that same intent later (a probe it missed) is marked a read again, never a start: this LB saw it serve.

        An LB that boots over a node already serving writes nothing (the probe answers before the first tick): it saw
        no start, and its minutes join the applied intent's session by its id all the same."""
        if self.boots is None or released is None:
            return
        from vllm_lb.node_control import UP, same_intent
        identity = (released["id"], released["at"])
        if serves:
            self._served_intent = identity
        up = released["state"] == UP
        line = self.boots.open
        generation = released["id"] if up else (line or {}).get("generation", released["id"])
        try:
            config = topology.released_config(self.cfg, released) if up else None
        except Exception:                                 # noqa: BLE001 — a diary line, never the tick
            config = None
        state = topology.boot_state(p.state, released, report)
        steps = [state]
        applied = report.get("applied") or {}
        # LAUNCHED BETWEEN TWO LOOKS: the applier ran this intent's launch (applied, not refused) while the line has
        # not yet said `starting` — a fast `up.sh` the LB never saw as `applying`. The start passed that step, so it
        # is a launch that served or failed, never a refusal (start_rates.py).
        if (up and state != "starting" and line is not None and line["generation"] == generation
                and same_intent(released, applied) and not applied.get("refused")):
            steps.insert(0, "starting")
        for step in steps:
            self.boots.observe(now=now, state=step, detail=p.detail, generation=generation, config=config,
                               config_source=f"{self.cfg.fleet.source} + intent #{released['id']}" if up else "",
                               loading=loading, read_again=up and self._served_intent == identity)

    # ── observe ─────────────────────────────────────────────────────────────

    def _observe_boxes(self) -> None:
        with ThreadPoolExecutor(max_workers=max(1, len(self.hosts))) as pool:
            for host, raw in zip(self.hosts, pool.map(self._status_of, self.hosts)):
                self._status[host.box] = raw
                self._note_memory(host.box, _number(raw, "mem_available_gib"))
                self._note_clock(host.box, raw)

    def _note_clock(self, box: str, raw) -> None:
        """The box's own clock against the LB's, from the `date +%s.%N` tp-rank reads beside its RDMA counters
        (`net.t`) — off by at most the reply's way back. A reply without one keeps the last offset: a clock does
        not drift in a tick."""
        net = raw.get("net") if isinstance(raw, dict) else None
        t = net.get("t") if isinstance(net, dict) else None
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            self._box_offset[box] = float(t) - self.clock()

    def _note_memory(self, box: str, gib: float | None) -> None:
        """One FRESH MemAvailable observation (this runs only where a box was actually asked). A box
        that could not be read forgets its samples: a launch must never lean on readings from before a
        gap it cannot see into."""
        ring = self._mem_samples.setdefault(box, deque(maxlen=max(1, self.cfg.cluster_mem_samples)))
        if gib is None:
            ring.clear()
        else:
            ring.append(gib)

    def _stable_mem(self, box: str) -> float | None:
        """What this box can be COUNTED ON to have: the lowest of its last `cluster_mem_samples`
        observations, or None until it has that many (L3)."""
        ring = self._mem_samples.get(box)
        if not ring or len(ring) < ring.maxlen:
            return None
        return min(ring)

    def _peak_mem(self, box: str) -> float | None:
        """The HIGHEST of the same observations: what the kill floor is judged on."""
        ring = self._mem_samples.get(box)
        return max(ring) if ring and len(ring) == ring.maxlen else None

    def _status_of(self, host):
        try:
            return self.executor.status(host)
        except Exception as exc:                          # the executor must never take a tick down
            log.warning("[cluster] status of %s raised %s: %s", host.box, type(exc).__name__, exc)
            return None

    def _observe_availability(self) -> None:
        """Ask every box what it has, for every cluster CHECKPOINT at once — each model and, where it has
        one, its uncensored twin, keyed apart (`ModelCatalog.checkpoints`). A box that cannot be asked
        stays None — unknown, never "the files are gone"."""
        specs = self.store.catalog.checkpoints(CLUSTER_MODE)
        if not specs or getattr(self.executor, "availability", None) is None:
            return
        with ThreadPoolExecutor(max_workers=max(1, len(self.hosts))) as pool:
            seen = pool.map(lambda h: self._availability_of(h, specs), self.hosts)
            for host, found in zip(self.hosts, seen):
                self._availability[host.box] = found

    def _availability_of(self, host, specs):
        try:
            return self.executor.availability(host, specs)
        except Exception as exc:                          # a look at a folder must never take a tick down
            log.warning("[cluster] availability of %s raised %s: %s", host.box, type(exc).__name__, exc)
            return None

    def _net_rate(self, box: str, net) -> dict | None:
        """GB/s on the TP link, per direction, from two of tp-rank's monotonic RDMA byte readings, and the
        PCIe ceiling they run under (GT/s·lanes × 128/130 ÷ 8 — 31.5 GB/s on a GB10 box).

        None on the first reading, after a gap, or when a counter went BACKWARDS (a box or NIC reset). The
        same reading seen twice in one tick (dt = 0) returns the rate already derived from it."""
        ok = (isinstance(net, dict) and all(isinstance(net.get(k), (int, float))
                                             for k in ("tx_bytes", "rx_bytes", "t")))
        if not ok:
            self._net_last[box] = None
            self._net_prev[box] = None
            return None
        last = self._net_last.get(box)
        if last is not None and net["t"] == last[0]:
            return self._net_prev.get(box)
        self._net_last[box] = (net["t"], net["tx_bytes"], net["rx_bytes"])
        rate = None
        if last is not None:
            dt = net["t"] - last[0]
            dtx, drx = net["tx_bytes"] - last[1], net["rx_bytes"] - last[2]
            if dt > 0 and dtx >= 0 and drx >= 0:
                lanes = net.get("pcie_gt_lanes")
                rate = {"tx_gbps": round(dtx / dt / 1e9, 3), "rx_gbps": round(drx / dt / 1e9, 3),
                        "link_gbps": (round(lanes * 128 / 130 / 8, 1)
                                      if isinstance(lanes, (int, float)) and lanes > 0 else None),
                        "devices": net.get("devices"), "window_s": round(dt, 1)}
        self._net_prev[box] = rate
        return rate

    def _observed(self, desired: Desired, reservations: dict | None, parks: dict | None, jobs: dict | None,
                  now: float, lanes: dict | None = None) -> Observed:
        boxes = {}
        requeued: dict[str, float] = {}
        backends = self.hub.backends()
        head = next((b for b in backends if b.url == self.cfg.cluster_url), None)
        live = bool(head is not None and head.live)
        errors = getattr(self.executor, "last_error", {}) or {}
        for i, h in enumerate(self.hosts):
            raw = self._status.get(h.box)
            rank = _rank(raw)
            current, stale = (reservations or {}).get(h.box, (None, ()))
            requeued_s = None
            # Back in the queue under a standing rank of this generation: a lane restart re-queued it.
            if (current is not None and current.run_open and current.status == "queued" and rank is not None
                    and rank.present and rank.gen == desired.generation):
                requeued[current.job_id] = self._requeued_at.get(current.job_id, now)
                requeued_s = now - requeued[current.job_id]
            seen = BoxObs(
                reservation=current, reservation_known=reservations is not None, stale=tuple(stale),
                parked_by=parks.get(h.label) if parks is not None else None,
                running_jobs=jobs.get(h.label) if jobs is not None else None,
                rank=rank, broker_vllm_running=_flag(raw, "broker_vllm_running"),
                mem_available_gib=_number(raw, "mem_available_gib"),
                mem_total_gib=_number(raw, "mem_total_gib"),
                mem_stable_gib=self._stable_mem(h.box), mem_peak_gib=self._peak_mem(h.box),
                requeued_s=requeued_s,
                availability=self._availability.get(h.box), holds=_holders(raw),
                gpu=gpu_view(raw.get("gpu") if isinstance(raw, dict) else None),
                network=self._net_rate(h.box, raw.get("net") if isinstance(raw, dict) else None),
                lane_live=lanes.get(h.label) if lanes is not None else None,
                # WHY it could not be asked, when it could not: the executor's own words for the planner's line.
                error=str(errors.get(h.box) or "") if rank is None else "")
            # THE REQUIREMENT, computed ONCE per box per tick and used twice: the planner gates on it
            # and the page shows it, so "need N" can never be a different number from the one the plan
            # waits for (worklog/vllm-cluster-hold-robustness.md L2). Per ROLE — the head carries the
            # bigger overhead — so the same tick refuses rank 0 and admits rank 1 at the same GiB free.
            role = RANK_ROLES[i] if i < len(RANK_ROLES) else RANK_ROLES[-1]
            estimate, why = self._estimate(h.box, role, desired, seen, serving=live)
            self._memory[h.box] = estimate.view() if estimate is not None else None
            boxes[h.box] = replace(seen, needs_gib=None if estimate is None else estimate.needed_now_gib,
                                   needs_note="" if estimate is None else estimate.needs_note, needs_why=why,
                                   released_from_gib=self._mem_release_target.get(h.box))
        if reservations is not None:                      # an unreadable broker does not restart the grace
            self._requeued_at = requeued
        # OBSERVED, from the head's own /v1/models — the ModelChoice is what we ASKED for, not what runs.
        self._loaded = self.store.catalog.loaded_model(head.served_models, head.max_model_len) if live else None
        spec = self.store.catalog.get(desired.model_choice)
        serves = None if self._loaded is None or spec is None else self._loaded.served_name == spec.served_name
        return Observed(boxes=boxes, head_live=live,
                        classic_inflight=sum(b.inflight for b in backends if b.kind == VLLM),
                        head_inflight=sum(b.inflight for b in backends if b.kind == CLUSTER),
                        head_serves=serves,
                        head_model=self._loaded.served_name if self._loaded is not None else None,
                        start_quiet_s=None if self._progress is None else self._progress[2] - self._progress[1],
                        start_phase=str((self._progress[0] if self._progress else {}).get("phase") or ""))

    # ── act ─────────────────────────────────────────────────────────────────

    def _heal_due(self, box: str, now: float) -> bool:
        last = self._heals.get(box)
        return last is None or now - last["at"] >= HEAL_EVERY_S

    def _refusal(self, desired: Desired, start_id: str | None) -> str:
        """THIS start's refused launch, if any: a new Start (another generation or start id) is a new launch.

        Asked of the launch's OWN BUILD first (`cluster_exec.launch_problem`, the same function `up` sends what it
        builds): a stored selection the launch cannot build — a KV pin past its profile's ceiling, a twin the model does
        not have — is a refusal on the FIRST plan of the generation, before a box is asked or reserved, with no Up ever
        attempted. Generation 187 (2026-10-10) learned it from `up`, which answers `(False, "up: ValueError: …")` and
        never raises for a box, so the refusal below never fired and Up was planned again on every tick. A refusal an
        executor RAISED is still kept (`_refused`), for a build the loop could not foresee."""
        r = self._refused
        if r is not None and r[0] == desired.generation and r[1] == start_id:
            return r[2]
        if desired.mode != CLUSTER_MODE:
            return ""
        return self._build_problem(desired)

    def _build_problem(self, desired: Desired) -> str:
        """Why the launch of `desired` cannot be BUILT, in the launch's words — "" when it builds."""
        spec = self.store.catalog.get(desired.model_choice)
        if desired.model_choice is not None and spec is None:
            return f"the model choice {desired.model_choice!r} is not in the model catalog"
        return launch_problem(spec, rank=0, gen=desired.generation, max_num_seqs=desired.max_num_seqs,
                              max_model_len=desired.max_model_len,
                              max_num_batched_tokens=desired.max_num_batched_tokens,
                              num_speculative_tokens=desired.num_speculative_tokens,
                              ple_replicate=desired.ple_replicate, kv_cache_memory=desired.kv_cache_memory,
                              kv_offload_gib=desired.kv_offload_gib,
                              long_prefill_token_threshold=desired.long_prefill_token_threshold,
                              uncensored=desired.uncensored, kv_cache_dtype=desired.kv_cache_dtype,
                              estimated=desired.estimated)

    def _apply(self, actions: list, start_id: str | None, now: float, desired: Desired) -> None:
        by_box = {h.box: h for h in self.hosts}
        for action in actions:
            host = by_box[action.box]
            try:
                if isinstance(action, Reserve):
                    # The reservation declares what this generation's ModelChoice serves: ONE model setting.
                    spec = self.store.catalog.get(desired.model_choice)
                    if spec is None:
                        ok, detail = False, f"the model choice {desired.model_choice!r} is not in the model catalog"
                    else:
                        job_id = self.leases.reserve_for_cluster(host.box, action.gen, start_id,
                                                                 model=spec.served_name)
                        ok, detail = job_id is not None, self.leases.last_error
                elif isinstance(action, ReleaseReservation):
                    ok = self.leases.release(Lease(lease_id=action.reservation_id, job_id=action.job_id,
                                                   created_at=now, kind=CLUSTER, box=host.box))
                    detail = self.leases.last_error
                elif isinstance(action, Up):
                    # The ModelChoice travels as its ModelSpec: the executor turns the spec's cluster
                    # LaunchProfile into tp-rank's settings, and the rank is labelled with the id. The
                    # operator's engine knobs travel as stored — None, the default, adds nothing.
                    spec = self.store.catalog.get(action.model_choice)
                    # A LAUNCH THAT DOES NOT BUILD IS REFUSED BEFORE THE EXECUTOR IS ASKED — `_refusal` has already
                    # failed the plan for such a selection, so this is the same check kept at the last step: whatever
                    # shape an executor gives its refusal (`(False, "up: …")`, as the real one does), it is recorded as
                    # THIS start's refusal (`_refused`) and never re-planned.
                    problem = (f"the model choice {action.model_choice!r} is not in the model catalog"
                               if action.model_choice is not None and spec is None else
                               launch_problem(spec, rank=action.rank, gen=action.gen, max_num_seqs=action.max_num_seqs,
                                              max_model_len=action.max_model_len,
                                              max_num_batched_tokens=action.max_num_batched_tokens,
                                              num_speculative_tokens=action.num_speculative_tokens,
                                              ple_replicate=action.ple_replicate, kv_cache_memory=action.kv_cache_memory,
                                              kv_offload_gib=action.kv_offload_gib,
                                              long_prefill_token_threshold=action.long_prefill_token_threshold,
                                              uncensored=action.uncensored, kv_cache_dtype=action.kv_cache_dtype,
                                              estimated=action.estimated))
                    if problem:
                        self._refused = (action.gen, start_id, problem)
                        ok, detail = False, problem
                    else:
                        self._record_launch(action, spec, start_id, now)
                        try:
                            ok, detail = self.executor.up(host, rank=action.rank, gen=action.gen,
                                                          max_num_seqs=action.max_num_seqs,
                                                          max_model_len=action.max_model_len, spec=spec,
                                                          max_num_batched_tokens=action.max_num_batched_tokens,
                                                          num_speculative_tokens=action.num_speculative_tokens,
                                                          ple_replicate=action.ple_replicate,
                                                          kv_cache_memory=action.kv_cache_memory,
                                                          kv_offload_gib=action.kv_offload_gib,
                                                          long_prefill_token_threshold=action.long_prefill_token_threshold,
                                                          uncensored=action.uncensored, kv_cache_dtype=action.kv_cache_dtype)
                        except ValueError as exc:
                            # REFUSED BEFORE ANYTHING REACHED THE BOX: the executor never raises for a box or a
                            # link (cluster_exec: "Never raise"), so this is the launch these settings BUILD, or
                            # tp-rank's refusal of it (exit 2, generation 221) — the same on every retry. The plan fails the start with it (`launch_refused`).
                            self._refused = (action.gen, start_id, str(exc))
                            raise
                elif isinstance(action, Halt):
                    ok, detail = self.executor.halt(host)
                elif isinstance(action, Heal):
                    ok, detail = self.executor.heal(host)
                else:
                    ok, detail = self.executor.down(host)
            except Exception as exc:                      # recorded; the next fresh tick re-plans
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            if isinstance(action, Heal):
                self._heals[action.box] = {"at": now, "ok": ok, "detail": str(detail or "")}
            log.log(logging.INFO if ok else logging.WARNING, "[cluster] %s %s: %s%s%s",
                    type(action).__name__.lower(), action.box, "ok" if ok else "FAILED",
                    self._launch_memory(action) if ok else "",
                    f" ({str(detail)[:200]})" if detail and (not ok or isinstance(action, Heal)) else "")

    def _launch_memory(self, action) -> str:
        """" — or, for a launch, what the box HAD and what this rank NEEDS, in the `up … ok` line (L4), and
        the settings it was launched WITH.

        The decisive number used to be captured in memory and never said, so a post-mortem could not tell
        "we think the gate passed" from "the gate passed at 101.3" — which is exactly what could not be
        reconstructed for generation 10 (worklog/vllm-cluster-hold-robustness.md §8)."""
        if not isinstance(action, Up):
            return ""
        est = self._memory.get(action.box) or {}
        # THE GATE'S OWN NEED (`needed_now_gib`: the rank's cost plus the start headroom), never the cost alone — the
        # line read `needs 101.9` while the gate had waited on 109.9, two figures for one requirement (2026-10-10).
        free, need = est.get("available_gib"), est.get("needed_now_gib")
        memory = "" if free is None or need is None else f" ({free:.1f} GiB free, needs {need:.1f})"
        rec = self._launched if (self._launched or {}).get("generation") == action.gen else None
        if rec is None:
            return memory
        threshold = rec.get("long_prefill_token_threshold")
        return (memory + f" — {rec['max_num_seqs']} seqs, {rec['max_model_len']} tokens, prefill budget "
                         f"{rec['max_num_batched_tokens']}, {rec['num_speculative_tokens']} speculative tokens"
                         + (f", long-prefill threshold {threshold}" if threshold else ""))

    # ── what the pair was launched WITH ─────────────────────────────────────

    def _box_env(self) -> dict:
        """The settings every tp-rank call carries — the LB's own environment's share (`TP_RANK_SETTINGS`)."""
        return dict(getattr(self.executor, "env", None) or {})

    def _resolve(self, desired: Desired) -> dict:
        """`desired` as a launch of it resolves — the operator's knobs, else the box's or the catalog's
        defaults, else tp-rank's — keyed as `cluster.effective`."""
        settings = resolved_launch(self.store.catalog.get(desired.model_choice), self._box_env(),
                                   max_num_seqs=desired.max_num_seqs, max_model_len=desired.max_model_len,
                                   max_num_batched_tokens=desired.max_num_batched_tokens,
                                   num_speculative_tokens=desired.num_speculative_tokens,
                                   ple_replicate=desired.ple_replicate,
                                   kv_cache_memory=desired.kv_cache_memory,
                                   kv_offload_gib=desired.kv_offload_gib,
                                   long_prefill_token_threshold=desired.long_prefill_token_threshold,
                                   uncensored=desired.uncensored, kv_cache_dtype=desired.kv_cache_dtype)
        return {"model_choice": desired.model_choice, "generation": desired.generation, **settings}

    def _record_launch(self, action: Up, spec, start_id: str | None, now: float) -> None:
        """Remember — on disk — what THIS generation is launched with, exactly as the executor resolves it.
        Written when it changes: both ranks get the same settings in the same tick. A record that cannot be
        written costs `cluster.effective` its exactness after an LB restart, never the launch."""
        settings = resolved_launch(spec, self._box_env(), max_num_seqs=action.max_num_seqs,
                                   max_model_len=action.max_model_len,
                                   max_num_batched_tokens=action.max_num_batched_tokens,
                                   num_speculative_tokens=action.num_speculative_tokens,
                                   ple_replicate=action.ple_replicate,
                                   kv_cache_memory=action.kv_cache_memory,
                                   kv_offload_gib=action.kv_offload_gib,
                                   long_prefill_token_threshold=action.long_prefill_token_threshold,
                                   uncensored=action.uncensored, kv_cache_dtype=action.kv_cache_dtype)
        record = {"model_choice": action.model_choice, "generation": action.gen, **settings, "start_id": start_id}
        if self._launched is not None and {key: self._launched.get(key) for key in record} == record:
            return
        self._launched = {**record, "launched_at": now}
        try:
            write_json(self._launch_path, self._launched)
        except OSError as exc:
            log.warning("[cluster] the launch record for generation %d was not saved to %s: %s",
                        action.gen, self._launch_path, exc)

    def _head_pool(self) -> dict | None:
        """`throughput.kv_pool` of the cluster head — what the engine's own `vllm:cache_config_info` said
        about the pool it allocated, as the prober last scraped it. None while it is not answering."""
        snap = self.hub.snapshot()
        head = next((b for b in snap.backends if b.url == self.cfg.cluster_url), None)
        if head is None or not head.live:
            return None
        return self.hub.throughput(head, snap.now).get("kv_pool")

    def _effective(self, desired: Desired, observed: Observed) -> dict | None:
        """`cluster.effective`: the RESOLVED settings the running pair was launched with — null unless both
        ranks are observed running, of one generation.

        From the launch record when it is this pair's: its generation, and the model its head rank says it was
        launched for — so an old pair still standing through a restart is described as what it IS. A pair
        launched before the LB kept a record (generation 52 was) is the stored switch it was launched from,
        resolved now: exact unless a default changed since, because its own values are frozen while it runs."""
        ranks = [(observed.boxes.get(h.box) or BoxObs()).rank for h in self.hosts]
        if not ranks or not all(r is not None and r.present and r.running for r in ranks):
            return None
        gens = {r.gen for r in ranks}
        gen = gens.pop() if len(gens) == 1 else None
        if gen is None:
            return None
        label = ranks[0].model_choice               # what the head was launched FOR; None on a rank from before labels
        rec = self._launched
        if rec is not None and rec.get("generation") == gen and label in (None, rec.get("model_choice")):
            # The prefill budget in the ENGINE's words once it has said it (`_note_stated_budget`), else the launch's.
            said = rec.get(ENGINE_BUDGET_KEY)
            out = {key: (said if key == "max_num_batched_tokens" and isinstance(said, int) else rec.get(key))
                   for key in EFFECTIVE_KEYS}
            return self._engine_own(out)
        if desired.mode == CLUSTER_MODE and desired.generation == gen and label in (None, desired.model_choice):
            return self._resolve(desired)
        return None

    def _engine_own(self, eff: dict) -> dict:
        """`eff` with every knob its model's ENGINE does not take stated as the engine runs it (`resolved_launch`) —
        a record an older LB wrote says what it SENT: generation 135's says a 40 GB pin SGLang never read."""
        spec = self.store.catalog.get(eff.get("model_choice"))
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        if profile is None or not profile.refuses:
            return eff
        try:
            own = resolved_launch(spec, self._box_env(), max_num_seqs=eff["max_num_seqs"],
                                  max_model_len=eff["max_model_len"])
        except (KeyError, TypeError, ValueError):
            return eff
        return {**eff, **{knob: own[knob] for knob in profile.refuses if knob in own}}

    def _note_rest(self, p: Plan, observed: Observed, desired: Desired) -> None:
        """THE RESTING FIGURE: the most each box has had free — on its lowest-of-recent reading, so one lucky
        sample does not count — while THIS generation's pair serves. It is what the box rests at before load
        takes its share, and `auto` needs it so that it never charges the load twice: the caching allocator
        keeps what load grew, so the free memory of a pair that has served is already the loaded figure
        (generation 99: 19.4 GiB at rest, 13.5 under 16 sequences — memory_estimate.KvAutoBox).

        Kept WITH THE LAUNCH RECORD, on disk, so an LB redeployed under a serving pair still knows it. An LB
        that only began watching a pair under load starts from that lower figure, which only ever makes an
        auto pool smaller, never larger."""
        rec = self._launched
        if not p.held or rec is None or rec.get("generation") != desired.generation:
            return
        rest = dict(rec.get("mem_rest_gib") or {})
        higher = {box: o.mem_stable_gib for box, o in observed.boxes.items()
                  if o.mem_stable_gib is not None
                  and not (isinstance(rest.get(box), (int, float)) and rest[box] >= o.mem_stable_gib)}
        if not higher:
            return
        self._launched = {**rec, "mem_rest_gib": {**rest, **higher}}
        try:
            write_json(self._launch_path, self._launched)
        except OSError as exc:
            log.warning("[cluster] the resting memory of generation %d was not saved to %s: %s",
                        desired.generation, self._launch_path, exc)

    def _rest_of(self, box: str, p: Plan, desired: Desired) -> float | None:
        """`mem_rest_gib`, published only while THIS generation's pair serves — the one pair it describes."""
        rec = self._launched or {}
        if not p.held or rec.get("generation") != desired.generation:
            return None
        value = (rec.get("mem_rest_gib") or {}).get(box)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def _remember_launch_memory(self, actions: list, observed: Observed) -> None:
        """What a box had free at the moment its rank was launched — the one number that turns an
        estimate into "and this is what it actually cost" (M8). Forgotten with the rank, but not thrown
        away: a rank coming DOWN leaves it behind as the memory this box should get back before the next
        rank launches here. That is the only evidence the LB has that a handover has finished letting go,
        and the handover is where 2026-09-30's fatal NVRM `Out of memory [NV_ERR_NO_MEMORY]` landed."""
        for action in actions:
            if isinstance(action, Up):
                free = observed.boxes[action.box].mem_available_gib
                if free is not None:
                    self._mem_at_launch[action.box] = free
                self._mem_release_target.pop(action.box, None)     # a new rank is here: the old one is history
            elif isinstance(action, (Down, Halt)):
                had = self._mem_at_launch.pop(action.box, None)
                if had is not None:
                    self._mem_release_target[action.box] = had

    def _prefill_budget(self, desired: Desired) -> int | None:
        """The prefill budget a launch of `desired` would RESOLVE to — the operator's knob, else the model's own,
        else the fleet's (`CLUSTER_MAX_NUM_BATCHED_TOKENS` in every call's env). One resolution for the launch and
        the estimate, so the number the box is judged against is the number the engine is given."""
        value = self._resolve(desired).get("max_num_batched_tokens")
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def _estimate(self, box: str, role: str, desired: Desired, o: BoxObs, *,
                  serving: bool = False) -> tuple[MemoryEstimate | None, str]:
        """M8: what this selection — model, concurrency, context — needs on this box, and once its rank
        was launched, what the box actually gave up: `(estimate, "")` — or `(None, why)` when the choice cannot
        be estimated at all, in the estimate's own words, which the planner refuses the Start with
        (`BoxObs.needs_why`): no flat requirement stands in.

        The ESTIMATE itself, not its view: `needed_now_gib` is the planner's gate (L2) and `view()` is
        what the page reads — one computation behind both.

        `serving`: the head answers. Only then is what the box gave up an ACTUAL to hold the estimate
        against: while the weights load there is no KV pool yet, and the page read `actual 19.6 GiB
        (+247 %) — correct the catalog's constants` about a start that was a third of the way through
        (2026-10-02). The launch figure itself is kept from the launch on, so `leaves` is measured from it."""
        launched = self._mem_at_launch.get(box)
        used = (None if not serving or launched is None or o.mem_available_gib is None
                else launched - o.mem_available_gib)
        try:
            return estimate_memory(
                self.store.catalog.get(desired.model_choice), box=box, role=role,
                max_num_seqs=desired.max_num_seqs, max_model_len=desired.max_model_len, tp=len(self.hosts),
                available_gib=o.mem_available_gib,
                **self.cfg.memory_policy(o.mem_total_gib),
                actual_used_gib=used,
                # The PREFILL BUDGET as a launch would RESOLVE it, never as the operator happened to type
                # it: a budget nobody set is still the budget the engine runs, and it is the knob that
                # took box-a to 0.8 GiB free on 2026-09-30 while this estimate could not see it.
                max_num_batched_tokens=self._prefill_budget(desired),
                # The drafter's weights only when it is launched: the operator's 0 loads none.
                speculative=(None if desired.num_speculative_tokens is None
                             else desired.num_speculative_tokens > 0),
                # The PLE placement and the KV pin as the launch would resolve them: both move GiB on
                # every box, and the gate is the only thing standing between a selection and an NVRM
                # out-of-memory on a pair that shares one LPDDR5X pool.
                ple_replicate=desired.ple_replicate, kv_cache_memory=desired.kv_cache_memory,
                launched_gib=launched, kv_cache_dtype=desired.kv_cache_dtype), ""
        except ValueError as exc:                         # classic vLLM's memory is broker_parrot's; MemTotal unread
            return None, str(exc)

    # ── a hung start: its evidence, and its relaunch ────────────────────────

    def _rescues(self, desired: Desired, start_id: str | None) -> int:
        """How many times THIS start — this generation of this Start — has been relaunched after a hang."""
        r = self._rescue_view(desired, start_id)
        n = (r or {}).get("attempt")
        return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0

    def _rescue_view(self, desired: Desired, start_id: str | None) -> dict | None:
        """`cluster.rescue`: the last rescue, when it belongs to the start being shown — never a previous one's."""
        r = self._rescue
        if not isinstance(r, dict) or r.get("generation") != desired.generation or r.get("start_id") != start_id:
            return None
        return dict(r)

    def _evidence_name(self, hang: StartHang, desired: Desired) -> str:
        return f"{HANG_DIR}/gen{desired.generation}-attempt{hang.attempt}"

    def _save_hang_evidence(self, hang: StartHang, desired: Desired, start_id: str | None, now: float) -> str:
        """EVIDENCE BEFORE TEARDOWN. Both ranks' logs and, through tp-rank `stacks`, what every engine process of
        each is doing (Python and native stacks, every thread's state) and what the box saw (its GPU, its memory,
        its kernel log) — asked of both boxes at once, then written under `HANG_DIR` with what the LB knew
        (`hang.json`). The stacks die with the processes; the log goes with a Down. Generation 131's were taken by
        hand while the pair sat deadlocked, and they were the whole diagnosis. Best effort, never in the way: a
        box that cannot be asked leaves its error in the file, and a disk that cannot be written loses the
        evidence, never the teardown. Returns where it went, relative to the state dir."""
        rel = self._evidence_name(hang, desired)
        key = (desired.generation, start_id, hang.attempt)
        if self._evidence_saved == key:
            return rel
        self._evidence_saved = key
        stacks_of = getattr(self.executor, "stacks", None)

        def grab(job) -> str:
            what, host = job
            try:
                if what == "log":
                    return self.executor.logs(host, START_LOG_LINES) or ""
                return (stacks_of(host) or "") if stacks_of is not None else ""
            except Exception as exc:                      # noqa: BLE001 — evidence is a courtesy
                return f"(the {what} could not be read: {type(exc).__name__}: {exc})\n"

        # Every read at once — both boxes' logs and stacks — so the evidence costs the slowest one (tp-rank
        # bounds `stacks` at ~30 s), never their sum.
        jobs = [(what, h) for h in self.hosts for what in ("log", "stacks")]
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
            texts = list(pool.map(grab, jobs))
        got = [(h.box, texts[2 * i], texts[2 * i + 1]) for i, h in enumerate(self.hosts)]
        errors = getattr(self.executor, "last_error", {}) or {}
        rec = self._launched if (self._launched or {}).get("generation") == desired.generation else None
        where = self._hang_root / rel.split("/", 1)[1]
        try:
            where.mkdir(parents=True, exist_ok=True)
            for box, text, stacks in got:
                (where / f"{box}.log").write_text(text, encoding="utf-8")
                (where / f"{box}-stacks.txt").write_text(
                    stacks or f"(no stacks: {errors.get(box) or 'no rank running, or the box said nothing'})\n",
                    encoding="utf-8")
            write_json(str(where / "hang.json"), {
                "generation": desired.generation, "start_id": start_id, **hang.view(), "at": now,
                "config": ({key: rec.get(key) for key in EFFECTIVE_KEYS} if rec is not None
                           else self._resolve(desired)),
                "loading": dict(self._loading),
                "errors": {box: str(errors.get(box) or "") for box, _, _ in got}})
        except Exception as exc:                          # noqa: BLE001 — never the teardown's problem
            log.warning("[cluster] the evidence of generation %d's hang was not saved to %s: %s",
                        desired.generation, where, exc)
            return rel
        self._prune_hangs()
        log.info("[cluster] generation %d hung %.0f s in “%s”: both ranks' logs and stacks saved to %s",
                 desired.generation, hang.quiet_s, hang.phase, where)
        return rel

    def _prune_hangs(self) -> None:
        """Keep the newest HANG_EVIDENCE_KEEP folders the LB named; leave everything else where it is."""
        def age(path: Path) -> tuple:
            try:
                return path.stat().st_mtime, path.name
            except OSError:
                return 0.0, path.name
        try:
            ours = [d for d in self._hang_root.iterdir() if d.is_dir() and _HANG_NAME.fullmatch(d.name)]
        except OSError:
            return
        for old in sorted(ours, key=age)[:-HANG_EVIDENCE_KEEP]:
            shutil.rmtree(old, ignore_errors=True)

    def _note_rescue(self, hang: StartHang, desired: Desired, start_id: str | None, now: float) -> dict:
        """A hung start's teardown has been sent: remember the rescue — on disk, with the launch record, so the
        retries stay bounded across an LB restart — and start the relaunch's own clock. Returns the boot log's
        account of the attempt that hung."""
        account = {**hang.view(), "evidence": self._evidence_name(hang, desired)}
        self._rescue = {**account, "generation": desired.generation, "start_id": start_id, "at": now}
        self._progress, self._load_mem = None, None
        log.warning("[cluster] generation %d hung %.0f s in “%s” (the limit for that step is %.0f s) — torn down "
                    "and relaunched, retry %d/%d; evidence in %s", desired.generation, hang.quiet_s, hang.phase,
                    hang.limit_s, hang.attempt, hang.retries, account["evidence"])
        rec = self._launched
        if rec is not None and rec.get("generation") == desired.generation:
            self._launched = {**rec, "rescue": self._rescue}
            try:
                write_json(self._launch_path, self._launched)
            except OSError as exc:
                log.warning("[cluster] the rescue of generation %d was not saved to %s: %s",
                            desired.generation, self._launch_path, exc)
        return account

    def _note_rescued(self, p: Plan, desired: Desired, start_id: str | None) -> None:
        """The relaunch SERVES: the rescue is history from here on — still published, still counted against the
        retries, but no longer said over a later restart of the same generation."""
        r = self._rescue_view(desired, start_id)
        if not p.held or r is None or r.get("served"):
            return
        self._rescue = {**r, "served": True}
        rec = self._launched
        if rec is not None and rec.get("generation") == desired.generation:
            self._launched = {**rec, "rescue": self._rescue}
            try:
                write_json(self._launch_path, self._launched)
            except OSError as exc:
                log.warning("[cluster] the rescue of generation %d was not saved to %s: %s",
                            desired.generation, self._launch_path, exc)

    def _note_halt(self, p: Plan, desired: Desired, start_id: str | None, now: float) -> None:
        """A hung start's halt has been decided (`p.hang`, no relaunch left): keep it with the launch record, so an
        LB restarted while the halted ranks stand still says `stalled: hung in …`, not `failed: exited (137)`."""
        rec = self._launched
        if rec is None or rec.get("generation") != desired.generation:
            return
        if (rec.get(HALT_KEY) or {}).get("detail") == p.detail:
            return
        self._launched = {**rec, HALT_KEY: {"generation": desired.generation, "start_id": start_id,
                                            "detail": p.detail, "at": now, "hang": p.hang.view()}}
        try:
            write_json(self._launch_path, self._launched)
        except OSError as exc:
            log.warning("[cluster] the halt of generation %d was not saved to %s: %s",
                        desired.generation, self._launch_path, exc)

    def _pinned(self, p: Plan, desired: Desired) -> Plan:
        """The plan as the page shows it: a pair that ended (`failed`, `stalled`) keeps the state and the words it
        FIRST ended with for this generation, while the planner's own word is still one of the two. Its actions
        are its own — a Halt that did not land is still due."""
        f = self._failure
        if f is None or f[0] != desired.generation or p.state not in TERMINAL:
            return p
        return replace(p, state=f[1], detail=f[2])

    # ── remember only what the page shows ───────────────────────────────────

    def _note_state(self, p: Plan, now: float) -> None:
        """The state clock: a change of state is logged and restarts `since`. All a one-host node's tick keeps."""
        if p.state != self._state:
            log.info("[cluster] %s → %s: %s", self._state, p.state, p.detail)
            self._state, self._since = p.state, now

    def _track(self, p: Plan, now: float, desired: Desired) -> None:
        self._note_state(p, now)
        # THE CAUSE, PINNED: a failure, or the tick a hung start is halted (`p.hang` without a relaunch). The ticks
        # after say `failed: exited (137)` of the ranks the LB itself stopped; `_pinned` answers them with this.
        ended = p.state == "failed" or (p.hang is not None and not p.hang.relaunch)
        if ended:
            if self._failure is None or self._failure[0] != desired.generation:
                self._failure = (desired.generation, p.state, p.detail)
        elif p.state not in TERMINAL:
            self._failure = None
        if p.state == "running":
            self._served_at = now
        elif p.state != "starting":
            self._served_at = None
        if p.state not in ("starting", *TERMINAL) and p.hang is None:
            # Frozen while failed or stalled, and while a rescue's teardown has not landed yet: a Halt or Down
            # that did not land is tried again on the next look, at the quiet that called for it.
            self._progress, self._load_mem = None, None
        if p.state != "starting":
            self._loading, self._logs_at = {}, None
            return
        if self._logs_at is not None and now - self._logs_at < self.cfg.cluster_logs_every_s:
            return
        self._logs_at = now
        try:
            text = self.executor.logs(self.hosts[0], START_LOG_LINES)
        except Exception:                                 # progress is a courtesy, never a fault
            text = ""
        self._note_stated_budget(text or "", desired)
        startup = read_startup(text or "", now=now)
        if startup is None:
            self._progress = None             # a start that cannot be read has not been SEEN to stall
            return
        view = startup.view()
        taken = self._load_taken(startup.phase)
        if taken is not None:
            view["taken_gib"] = taken
        self._loading = view
        prev = self._progress
        if prev is not None and prev[0] == view:
            since = prev[1]                               # nothing new: the clock stands
        elif prev is not None and _said(prev[0]) == _said(view):
            since = now                                   # the log said nothing new; the box gave up memory — seen NOW
        else:
            printed = self._printed_since(startup.printed_at, now)
            since = now if printed is None else min(now, printed)
            if prev is not None:
                since = max(since, prev[1])               # never behind evidence already in hand (memory moved later)
        if self._served_at is not None:
            since = max(since, self._served_at)       # it served after that line: no older silence is its
        self._progress = (dict(view), since, now)

    def _load_taken(self, phase: str) -> int | None:
        """`loading.taken_gib`: the whole GiB the HEAD's box has given up since this start's log was first read — its
        MemAvailable then, less the LOWEST since (`_load_mem`) — while `phase` is a LoadPass's own step; None in any
        other step, and when the box's memory is not known. A low-water mark only ever grows, so a wobble (ollama
        coming and going, the kernel's cache) cannot flip it, and a load that has stopped taking memory stops moving it.

        WHY THE BOX'S MEMORY, and not the log or the container: docker's json-file driver logs one entry per `\\n`,
        and SGLang's shard bar is `\\r` frames with none until the pass is done — `docker logs` shows a seven-minute
        load as nothing at all (generation 188, 2026-10-10: "Load weight begin" 21:04:47, the bar's one entry
        21:11:33; the driver keeps a partial line in memory, nothing reads it back). The rank container's own cgroup
        does not see the weights either: on a GB10 the GPU's memory is the box's but is not charged to the container
        (memory.current 14.0 GiB under a pair serving a 63 GB load, box-a 2026-10-10 21:19Z), and its block I/O
        (io.stat) counts only a local-NVMe read (box-a 47 GB; box-b 3.4 GB — its checkpoint is on NFS). What every
        load does, on either box, whatever the storage, is take the box's MemAvailable: 109.63 → 46.64 GB on the head
        over those 415 s (the head's own `avail mem` figures), which tp-rank already reports every status.
        Only under a load's step, so the FlashInfer-autotune and capture hangs (generations 122, 131) are judged by
        the log alone exactly as before."""
        gib = _number(self._status.get(self.hosts[0].box), "mem_available_gib")
        if gib is None:
            return None
        high, low = self._load_mem if self._load_mem is not None else (gib, gib)
        self._load_mem = (high, min(low, gib))
        if not StartupPhase.is_loading(phase):
            return None
        return max(0, int(high - self._load_mem[1]))

    def _note_stated_budget(self, text: str, desired: Desired) -> None:
        """WHAT THE ENGINE TOOK, beside what the LB sent: the prefill budget the starting head states, kept in this
        generation's launch record (`ENGINE_BUDGET_KEY`, on disk, so a redeployed LB still knows it) and stated by
        `cluster.effective`. A head that runs another budget than the launch sent is said in the LB's log — the
        mismatch generation 135 ran with unseen (2048 sent, 8192 run)."""
        said = stated_prefill_budget(text)
        rec = self._launched
        if (said is None or rec is None or rec.get("generation") != desired.generation
                or rec.get(ENGINE_BUDGET_KEY) == said):
            return
        if said != rec.get("max_num_batched_tokens"):
            log.warning("[cluster] generation %d: the head runs a prefill budget of %d, the launch sent %s",
                        desired.generation, said, rec.get("max_num_batched_tokens"))
        self._launched = {**rec, ENGINE_BUDGET_KEY: said}
        try:
            write_json(self._launch_path, self._launched)
        except OSError as exc:
            log.warning("[cluster] the prefill budget generation %d runs was not saved to %s: %s",
                        desired.generation, self._launch_path, exc)

    def _printed_since(self, printed_at: float | None, now: float) -> float | None:
        """When the head printed the line that last moved its start, on the LB's clock — so the silence survives an
        LB restart (generation 134, 2026-10-03: last line 07:21:13, LB redeployed 07:23:14, "no progress for 131 s"
        at 07:25:29, i.e. counted from ~07:23:18). The AGE is measured on the box: its now (the LB's plus the
        offset its last status carried, `_note_clock`; the LB's own when it sent none — box-a and box-b were
        within 0.05 s of box-c on 2026-10-03, all three NTP-synchronised) minus the line's time, never below 0.

        None — the read-to-read clock stands — when the line states no time, or a time before the head's container
        started (docker's StartedAt, the same box's clock): that is not a line this rank printed, or not in UTC."""
        if printed_at is None:
            return None
        head = self.hosts[0].box
        started = _started_at(self._status.get(head))
        if started is None or printed_at < started - STAMP_SLACK_S:
            return None
        age = now + self._box_offset.get(head, 0.0) - printed_at
        return now - max(0.0, age)

    def _up_since(self) -> float | None:
        """When the head's rank container started, moved from the box's clock onto the LB's by the offset its last
        status carried (`_note_clock`, as `_printed_since` reads it), or None when its status names no start."""
        head = self.hosts[0].box
        started = _started_at(self._status.get(head))
        return None if started is None else started - self._box_offset.get(head, 0.0)

    def _record_boot(self, p: Plan, now: float, desired: Desired, start_id: str | None = None,
                     hang: dict | None = None) -> None:
        """The boot log's view of this tick: the state and its first words, under the configuration the launch
        resolved (the launch record once the ranks were sent up, the switch resolved before that) — numbered by
        attempt, and closing a hung one (`hang`) as its own line."""
        if self.boots is None:
            return
        try:
            rec = self._launched if (self._launched or {}).get("generation") == desired.generation else None
            if rec is not None:
                config, source = {key: rec.get(key) for key in EFFECTIVE_KEYS}, "launch"
            else:
                config, source = self._resolve(desired), "resolved"
        except Exception:                                 # noqa: BLE001 — a diary line, never the tick
            config, source = None, ""
        self.boots.observe(now=now, state=p.state, detail=p.detail, generation=desired.generation, config=config,
                           config_source=source, loading=self._loading,
                           attempt=self._rescues(desired, start_id) + 1, hang=hang)

    def _defaults(self, desired: Desired) -> dict:
        """`cluster.defaults`: what an unwritten switch offers FOR THE CHOSEN MODEL — its own measured-best
        concurrency and context (`DesiredStore.defaults_for`, a `catalog.TunedSetup`), else the configured ones —
        and what a knob left at null launches with for it (its tuned prefill budget, else the box's; its
        drafter's depth)."""
        base = self.store.defaults_for(desired.model_choice)
        return resolved_launch(self.store.catalog.get(desired.model_choice), self._box_env(),
                               max_num_seqs=base.max_num_seqs, max_model_len=base.max_model_len,
                               uncensored=base.uncensored)

    def _setups(self) -> dict:
        """`cluster.setups`: THE PICKER ROWS' FACTS — for every model the cluster can launch and every twin, what a
        Start of that row launches when nothing else is chosen, keyed as the rows are (`catalog.availability_key`:
        `<id>`, `<id>@uncensored`). The SAME resolve as `_defaults` (`defaults_for` + `resolved_launch`), so a row
        and the Start it offers cannot say two things, and none of them moves when another model is chosen. Each
        names its model, so the proxy can look its throughput target up from the setup alone."""
        env, out = self._box_env(), {}
        for spec in self.store.catalog.for_mode(CLUSTER_MODE):
            base = self.store.defaults_for(spec.id)
            for uncensored in ((False, True) if spec.uncensored is not None else (False,)):
                out[availability_key(spec.id, uncensored)] = {
                    "model_choice": spec.id,
                    **resolved_launch(spec, env, max_num_seqs=base.max_num_seqs,
                                      max_model_len=base.max_model_len, uncensored=uncensored)}
        return out

    def _publish(self, desired: Desired, observed: Observed, p: Plan, now: float,
                 start_id: str | None = None) -> None:
        errors = getattr(self.executor, "last_error", {}) or {}
        # How long this wait has been going on, and when it runs out: a stuck handover must be visibly
        # BOUNDED on the page rather than one sentence that never changes (L8).
        waiting = round(max(0.0, now - self._since), 1) if p.state in ("freeing", "draining", "finishing") else 0.0
        boxes = {}
        for i, h in enumerate(self.hosts):
            o = observed.boxes[h.box]
            role = RANK_ROLES[i] if i < len(RANK_ROLES) else RANK_ROLES[-1]
            boxes[h.box] = {"label": h.label, "reservation": _reservation_view(o.reservation, desired),
                            "park": o.parked_by, "running_jobs": o.running_jobs,
                            "broker_vllm_running": o.broker_vllm_running,
                            "mem_available_gib": o.mem_available_gib,
                            # What it can be COUNTED ON to have — the number the planner gates on.
                            "mem_stable_gib": o.mem_stable_gib,
                            "mem_total_gib": o.mem_total_gib,
                            # NOTHING the LB can wait for holds its RAM: what it has free is the most it will free,
                            # so a selection that does not fit it is refused at the Start (`memory_admission`).
                            "at_rest": (o.mem_stable_gib is not None and not (o.rank and o.rank.present)
                                        and at_rest(o, o.mem_stable_gib, broker=self.cfg.broker_enabled)),
                            # …and the most it has had while THIS pair serves: what it rests at before load
                            # takes its share, which an `auto` KV pool is sized from (`_note_rest`).
                            "mem_rest_gib": self._rest_of(h.box, p, desired),
                            # And what it HAD when its rank was launched (L4): the number a post-mortem
                            # needs, kept for as long as the rank stands.
                            "mem_at_launch": self._mem_at_launch.get(h.box),
                            # What a box whose rank has come DOWN owes back before another launches here.
                            "mem_release_target": self._mem_release_target.get(h.box),
                            "rank": asdict(o.rank) if o.rank is not None else None,
                            "availability": (None if o.availability is None
                                             else {i: a.view() for i, a in o.availability.items()}),
                            "memory": self._memory.get(h.box),
                            # WHY this box is not free for the pair yet — the operator asked and the
                            # page could not tell them (L8). `holds` is what the box itself said.
                            "holds": None if o.holds is None else [h.view() for h in o.holds],
                            # What the GPU under this rank is doing — the panel's per-box half.
                            "gpu": o.gpu,
                            "network": o.network,
                            "hold": {
                                "reservation_status": o.reservation.status if o.reservation else None,
                                "claimed_by": o.reservation.claimed_by if o.reservation else None,
                                "waiting_s": waiting,
                                "deadline_s": (self.cfg.cluster_finishing_limit_s if p.state == "finishing"
                                               else self.cfg.cluster_freeing_deadline_s),
                                "blockers": blockers(o, o.needs_gib, broker=self.cfg.broker_enabled)},
                            # The last time its broker worker was healed, and what that did.
                            "heal": self._heals.get(h.box),
                            "error": errors.get(h.box, "")}
        # A failure (or a stall) is published in the FIRST words it ended with, for as long as it stays so — the
        # plan arrives here already `_pinned`. A RELAUNCH says which retry it is, and why, all the way back up:
        # the tick that decided it has its own words, the minutes after it would otherwise read like any start.
        detail = p.detail
        rescue = self._rescue_view(desired, start_id)
        if rescue is not None and not rescue.get("served") and p.hang is None and p.state in RELAUNCHING:
            detail += (f" — retry {rescue.get('attempt')}/{rescue.get('retries')} after the start hung "
                       f"{float(rescue.get('quiet_s') or 0):.0f} s in “{rescue.get('phase')}”")
        eff = self._effective(desired, observed)
        # WHAT THE ENGINE SAYS ITS OWN POOL HOLDS, for the ONE sentence that tells the operator how many
        # requests fit: the figure vLLM printed, never arithmetic over the pin, whenever the pair serving
        # is the pair being described. None before a start, and then the arithmetic answers.
        measured = engine_pool({"held": p.held, "effective": eff},
                               self.store.catalog.get(desired.model_choice), self._head_pool(),
                               kv_cache_memory=desired.kv_cache_memory)
        self.hub.set_cluster(name=self.cfg.cluster_name, view={
            # WHAT THE CLUSTER IS — TP, boxes, control channel, the start steps that apply (vllm_lb/topology.py):
            # what the page names it by, instead of a literal that was true of one deployment.
            "configured": True, "topology": topology.view(self.cfg),
            "desired": asdict(desired), "state": p.state, "detail": detail,
            # The head's start in its own words (`StartupProgress.view`) plus, while a weight pass is the step, the
            # whole GiB its box has given up so far (`taken_gib`, `_load_taken`): the page watches both for change.
            "held": p.held, "loading": dict(self._loading), "boxes": boxes,
            # The last rescue of THIS start (a hang torn down and relaunched): which attempt hung, in which step,
            # for how long against that step's limit, the retries allowed, where its evidence is. Null if none.
            "rescue": rescue,
            # The RAM wait, with its own clock: when it started and when it gives up (L3/L8). None
            # unless we are in it, so the page never draws a countdown for a wait nobody is waiting.
            "freeing": ({"since": float(self._since),
                         "deadline": float(self._since) + self.cfg.cluster_freeing_deadline_s}
                        if p.state == "freeing" else None),
            "url": self.cfg.cluster_url,
            "limits": limits_view(),
            # WHAT THE RUNNING PAIR WAS LAUNCHED WITH, every default resolved; null when no pair runs. The
            # performance log tags each sample with it, so it is what ran — never what is wanted now.
            "effective": eff,
            # What an engine knob left at null resolves to for the chosen model, and what an unwritten switch
            # offers: the page shows these where the operator has chosen nothing.
            "defaults": self._defaults(desired),
            # ...and the same for EVERY row of the model picker, so each row can say what it would launch.
            "setups": self._setups(),
            "since": float(self._since), "max_depth": desired.max_num_seqs,
            # HOW LONG THE PAIR HAS BEEN UP: docker's StartedAt of the HEAD's rank, on the LB's clock; null unless it
            # serves. It does not move when the LB restarts, which `since` (this loop's own last change) does.
            "up_since": self._up_since() if p.state == "running" else None,
            # The measured maximum for the STORED model's own pool and per-token cost at ITS context (its row's
            # tuned one) — per model, so it is worked out each time, never once per process; None for a choice the
            # catalog does not know.
            "recommended": ((lambda sp: recommended_for(self.cfg, sp).view() if sp is not None else None)
                            (self.store.catalog.get(desired.model_choice))),
            "clamp": self.store.published_clamp(desired, engine=measured),
            "store_error": self.store.last_error,
            # ONE ModelCatalog behind both pickers: every ModelSpec the cluster can launch, with the
            # limits that spec may be started with, and what the head is actually serving now.
            "models": [s.view() for s in self.store.catalog.for_mode(CLUSTER_MODE)],
            # …of which THIS deployment offers these rows (fleet `choices`; null = every one): the picker draws only them.
            "offers": list(self.store.offers) if self.store.offers is not None else None,
            # WHICH KV DTYPES each of them can run on THIS cluster's cards, and why not (`ModelSpec.kv_cache_dtype_
            # choices`, the image's own words): the page's KV dtype options, disabled with the switch's refusal.
            "kv_cache_dtypes": {s.id: kv_choices_run(s.kv_cache_dtype_choices(self.store.archs), self.store.kv_dtypes)
                                for s in self.store.catalog.for_mode(CLUSTER_MODE)},
            "loaded_model": self._loaded.view() if self._loaded is not None else None,
            "catalog_error": self.store.catalog.error,
            # What a fleet file declares a node this LB only observes runs (topology.declaration): the pair's
            # launch record is its own, so it has none — nor a host applier (control local's, node_control.py).
            "declared": None, "applier": None,
        })
