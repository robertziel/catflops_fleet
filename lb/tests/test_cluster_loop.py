"""The cluster loop end to end, hermetically: the reservations on the sqlite engine, a fake executor for
the GB10 boxes, the real Hub/Controller/router — and routing by what the loop publishes.

Every tick runs under a spy on the engine's own cursor, so "the LB wrote nothing" is a recorded fact, not a
reading of the code (worklog/vllm-gateway-program.md L1, L3).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

import pytest

from tests.fakes import FakeClusterExecutor, FakeProber, claim_gpu
from tests.start_logs import HEAD, RANK1, log_at, stamped
from vllm_lb import router
from vllm_lb.cluster import parse_hosts, start_stall_limit
from vllm_lb.cluster_defaults import desired_store
from vllm_lb.cluster_exec import LOG_LINES_RANGE
from vllm_lb.cluster_loop import HEAL_EVERY_S, ClusterLoop
from vllm_lb.cluster_startup import LoadPass
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases, cluster_reservation_id
from vllm_lb.state import CLUSTER, LEASE_RUNNING, VLLM

S1, S2, TP = "http://10.0.0.1:8000", "http://10.0.0.2:8000", "http://10.0.0.1:8100"
OFF = {"live": False, "last_error": "refused"}
LIVE_V = {"live": True, "served_models": ("qwen3.8:latest",), "max_model_len": 262144, "waiting": 0, "running": 0,
          "last_error": ""}
LIVE_TP = {"live": True, "served_models": ("Qwen/Qwen3.8-Flash-Next",), "max_model_len": 262144,
           "waiting": 0, "running": 0, "last_error": ""}
BOXES = ("box-a", "box-b")
LABEL = {"box-a": "box-a-gpu", "box-b": "box-b-gpu"}
#: The cluster's ModelChoices (the model library, vllm_lb/library).
FLASH, DENSE = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4"
DFLASH = "qwen3.8-27b-dflash2"
SGLANG = "qwen3.8-27b-sglang-dflash2"
#: Flash-Next on SGLang: the same weights as FLASH, the same engine as SGLANG (catalog.json, 2026-10-05).
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
GLM = "glm-5.3-flash-nvfp4"
DSV4 = "deepseek-v4-flash-nvfp4"


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Writes:
    """Every INSERT/UPDATE/DELETE the engine's cursor executed while the LB was ticking."""

    def __init__(self) -> None:
        self.on = False
        self.statements: list[str] = []


@pytest.fixture
def writes(broker_db, monkeypatch) -> Writes:
    from queue_workflows import db
    rec = Writes()
    real = db._SqliteCursor.execute

    def execute(self, sql, params=None):
        if rec.on and re.match(r"\s*(INSERT|UPDATE|DELETE|REPLACE)\b", sql, re.I):
            rec.statements.append(" ".join(sql.split())[:120])
        return real(self, sql, params)

    monkeypatch.setattr(db._SqliteCursor, "execute", execute)
    return rec


class World:
    def __init__(self, tmp_path, writes: Writes) -> None:
        self.writes = writes
        self.cfg = Config(
            broker_enabled=True,
            vllm_boxes=(VllmBox("box-a-gpu", "box-a", S1), VllmBox("box-b-gpu", "box-b", S2)),
            probe_interval_s=0, idle_release_s=600,
            cluster_name="pair", cluster_hosts=parse_hosts("box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,"
                                      "box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"),
            cluster_url=TP, cluster_state_path=str(tmp_path / "cluster.json"), cluster_max_num_batched_tokens=8192,
            cluster_observe_s=0.0, cluster_idle_observe_s=60.0, cluster_logs_every_s=15.0)
        self.clock = Clock()
        self.executor = FakeClusterExecutor()
        self.prober = FakeProber({S1: OFF, S2: OFF, TP: OFF})
        # Built exactly as `__main__` builds it, so the harness sees the same default — and the
        # same clamp — the real LB does.
        self.store = desired_store(self.cfg)
        self.boot()

    def boot(self) -> None:
        """A (re)started LB: fresh hub, fresh broker adapter, fresh loop — same disk, same fleet."""
        self.hub = Hub(self.cfg, clock=self.clock)
        self.leases = BrokerLeases(self.cfg, clock=self.clock)
        self.loop = ClusterLoop(self.cfg, self.hub, self.leases, executor=self.executor,
                                store=desired_store(self.cfg), clock=self.clock)
        self.ctl = Controller(self.cfg, self.hub, self.leases, prober=self.prober, clock=self.clock,
                              cluster_loop=self.loop)

    def tick(self, seconds: float = 2.0) -> dict:
        self.writes.on = True
        try:
            self.ctl.tick()
        finally:
            self.writes.on = False
        self.clock.t += seconds
        return self.view()

    def view(self) -> dict:
        return self.hub.snapshot().extra["cluster"]

    def settle(self, *, until: str, ticks: int = 8) -> dict:
        """Tick until the loop reaches `until`. A launch now needs each box to clear its own need on
        `cluster_mem_samples` CONSECUTIVE observations (L3), so "the fleet is ready" is a state to
        arrive at, not a fixed number of ticks."""
        v = self.view()
        for _ in range(ticks):
            v = self.tick()
            if v["state"] == until:
                return v
        raise AssertionError(f"never reached {until!r} in {ticks} ticks: {v['state']} — {v['detail']}")

    def calls(self, verb: str) -> list:
        return [c for c in self.executor.calls if c[0] == verb]

    def claim(self, *boxes: str) -> None:
        """broker_parrot's pinned lanes take their reservations."""
        for b in boxes or BOXES:
            assert claim_gpu(LABEL[b], box=b, served_types=("cluster",)) is not None, f"{b} claimed nothing"

    def run_id(self, box: str) -> str:
        desired, start_id = self.store.current()
        return cluster_reservation_id(desired.generation, start_id, box)

    def reservation_jobs(self) -> list[dict]:
        from queue_workflows import node_queue
        return [j for b in BOXES for j in node_queue.list_jobs_for_run(self.run_id(b))]

    def bodies_return(self) -> None:
        """A released reservation's node body sees cancel_event and returns."""
        from queue_workflows import node_queue, run_store
        for b in BOXES:
            if (run_store.get_run(self.run_id(b)) or {}).get("status") != "cancelled":
                continue
            for j in node_queue.list_jobs_for_run(self.run_id(b)):
                if j["status"] == "running":
                    node_queue.mark_completed(j["id"], context_delta={}, seconds=1.0)

    def run_to_running(self, *, serves: str = "Qwen/Qwen3.8-Flash-Next", **settings) -> None:
        """`serves` is what the head's own /v1/models reports once it is up — which is what decides
        whether the pair is held, so a run of another ModelChoice must say so."""
        self.store.set(enabled=True, **settings)
        assert self.tick()["state"] == "reserving"
        self.claim()
        self.settle(until="starting")        # fleet idle, broker vLLM down, memory free on K samples
        self.prober.table[TP] = {**LIVE_TP, "served_models": (serves,)}
        assert self.tick()["state"] == "running"


def vllm_leases(hub) -> list:
    return [lease for lease in hub.leases() if lease.kind == VLLM]


def worker_control_writes(writes: Writes) -> list[str]:
    return [s for s in writes.statements if "worker_control" in s.lower()]


def operator(label: str, state: str) -> None:
    """The operator, in broker_parrot's own panel — not the LB."""
    from queue_workflows import worker_control
    worker_control.set_worker_control(label, "gpu", desired_state=state, stop_policy="drain",
                                      requested_by="cluster-panel")


@pytest.fixture
def world(broker_db, writes, tmp_path) -> World:
    return World(tmp_path, writes)


def test_start_reserves_both_boxes_frees_ram_launches_and_routes_only_once_the_head_answers(world):
    world.executor.broker_vllm = {"box-a": True, "box-b": True}
    world.executor.mem = {"box-a": 30.0, "box-b": 30.0}
    # 12 seats at the FULL 262,144 window do not fit the ranks' KV pool — only 6 do — and the store
    # now clamps a launch to what fits (worklog/vllm-cluster-defaults.md C4). Halving the context
    # makes the 12 honest (13 fit at 131,072) and keeps this test about what it is named for: the
    # operator's own numbers travelling through to `up`, unclamped and unlike any default.
    world.store.set(enabled=True, max_num_seqs=12, max_model_len=131072)

    v = world.tick()
    assert v["state"] == "reserving" and "asking broker_parrot to reserve box-a and box-b" in v["detail"]
    assert len(world.reservation_jobs()) == 2 and not world.calls("up")
    v = world.tick()                                     # broker_parrot has not claimed yet
    jobs = {b: v["boxes"][b]["reservation"] for b in BOXES}
    assert {b: r["status"] for b, r in jobs.items()} == {"box-a": "queued", "box-b": "queued"}
    assert f"(node_job {jobs['box-a']['job_id'][:8]}, {jobs['box-b']['job_id'][:8]})" in v["detail"]
    assert len(world.reservation_jobs()) == 2, "waiting is not asking again"

    world.claim()
    v = world.tick()
    assert v["state"] == "freeing" and "freeing RAM: stopping classic vLLM on box-a, box-b" in v["detail"]
    assert not world.calls("up")

    world.executor.broker_vllm = {"box-a": False, "box-b": False}
    world.executor.mem = {"box-a": 115.0, "box-b": 115.0}
    v = world.settle(until="starting")
    assert world.calls("up") == [("up", "box-a", 0, 1, 12, 131072, FLASH),
                                 ("up", "box-b", 1, 1, 12, 131072, FLASH)]
    assert v["state"] == "starting" and v["held"] is False
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is None or chosen.kind != CLUSTER

    world.prober.table[TP] = LIVE_TP
    v = world.tick()
    assert v["state"] == "running" and v["held"] is True
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is not None and (chosen.kind, chosen.url) == (CLUSTER, TP)
    assert len(world.reservation_jobs()) == 2, "asked once per box per generation"


def test_start_frees_ram_first_the_classic_turn_ends_then_its_lease_goes_then_the_boxes_are_reserved(world):
    """L7, in order: no release while our turn is on classic vLLM; then the release; then the claim;
    then broker_parrot's vLLM goes down; then Up."""
    from queue_workflows import node_queue, run_store
    world.hub.note_request()
    world.tick()                                         # the classic lease is asked for …
    assert claim_gpu("box-a-gpu", box="box-a") is not None
    world.prober.table[S1] = LIVE_V
    world.tick()                                         # … and held
    (classic,) = vllm_leases(world.hub)
    assert classic.status == LEASE_RUNNING
    world.executor.broker_vllm = {"box-a": True, "box-b": True}
    world.executor.mem = {"box-a": 30.0, "box-b": 30.0}
    world.hub.note_request_start(S1)                     # a turn in flight on it

    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "freeing" and "gateway turn" in v["detail"]
    assert [l.lease_id for l in vllm_leases(world.hub)] == [classic.lease_id], "no release mid-turn"
    assert run_store.get_run(classic.lease_id)["status"] == "running"
    assert world.reservation_jobs() == []

    world.hub.note_request_end(S1)
    world.tick()
    assert vllm_leases(world.hub) == [] and run_store.get_run(classic.lease_id)["status"] == "cancelled"
    assert len(world.reservation_jobs()) == 2
    node_queue.mark_completed(classic.job_id, context_delta={}, seconds=1.0)

    world.claim()
    v = world.tick()
    assert v["state"] == "freeing" and not world.calls("up")
    world.executor.broker_vllm = {"box-a": False, "box-b": False}
    world.executor.mem = {"box-a": 115.0, "box-b": 115.0}
    assert world.settle(until="starting") and len(world.calls("up")) == 2
    for _ in range(3):
        world.hub.note_request()
        world.tick()
    assert vllm_leases(world.hub) == [], "no classic vLLM lease while the cluster is wanted"


def test_an_operator_park_or_unpark_mid_run_gets_no_lb_write_of_any_kind(world):
    world.run_to_running()
    before = len(world.executor.calls)
    for label, state in (("box-a-gpu", "off"), ("box-b-gpu", "off"), ("box-a-gpu", "on"), ("box-b-gpu", "on")):
        operator(label, state)
        world.writes.statements.clear()
        v = world.tick()
        assert world.writes.statements == [], (label, state, world.writes.statements)
        assert [c for c in world.executor.calls[before:] if c[0] in ("up", "down")] == []
        assert v["state"] == "running" and v["held"] is True
    operator("box-a-gpu", "off")
    v = world.tick()
    assert "box-a parked by cluster-panel — reservation kept" in v["detail"]
    assert v["boxes"]["box-a"]["park"] == "cluster-panel" and v["boxes"]["box-b"]["park"] is None
    assert v["boxes"]["box-a"]["reservation"]["status"] == "running", "a drain park keeps the held job"


def test_no_worker_controls_write_through_start_run_park_unpark_restart_and_stop(world):
    world.run_to_running()
    operator("box-a-gpu", "off")
    world.tick()
    operator("box-a-gpu", "on")
    world.tick()
    world.boot()
    world.tick()
    world.store.set(enabled=False)
    for _ in range(3):
        world.tick()
        world.bodies_return()
    assert world.tick()["state"] == "off"
    assert worker_control_writes(world.writes) == []


def test_a_lane_restart_requeues_the_reservation_and_the_pair_keeps_running_until_it_is_reclaimed(world):
    from queue_workflows import node_queue
    world.run_to_running()
    before = len(world.executor.calls)
    assert node_queue.requeue_running_for_worker("box-a-gpu", "gpu") == 1
    v = world.tick()
    assert v["state"] == "waiting" and v["held"] is False
    assert "broker_parrot re-queued the reservation on box-a" in v["detail"]
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is None or chosen.kind != CLUSTER, "no traffic while broker_parrot does not hold the box"
    world.claim("box-a")
    v = world.tick()
    assert v["state"] == "running" and v["held"] is True
    assert [c for c in world.executor.calls[before:] if c[0] in ("up", "down")] == []


def test_a_requeue_not_reclaimed_within_600_s_takes_the_pair_down(world):
    from queue_workflows import node_queue
    world.run_to_running()
    node_queue.requeue_running_for_worker("box-a-gpu", "gpu")
    world.tick()
    world.clock.t += 601
    v = world.tick()
    assert sorted(c[1] for c in world.calls("down")) == ["box-a", "box-b"]
    assert v["state"] == "restarting" and v["held"] is False


def test_an_ended_reservation_takes_the_ranks_down_and_is_never_asked_for_again_even_after_a_restart(world):
    from queue_workflows import run_store
    world.run_to_running()
    run_store.update_run(world.run_id("box-a"), status="cancelled")     # someone cancels it in the broker
    v = world.tick()
    assert ("down", "box-a") in world.calls("down")
    assert v["state"] == "failed" and "ended the reservation on box-a" in v["detail"] and v["held"] is False
    world.boot()
    for _ in range(3):
        assert world.tick()["state"] == "failed"
    assert len(world.reservation_jobs()) == 2 and len(world.calls("up")) == 2


def test_a_hold_that_reaches_its_ceiling_takes_the_ranks_down_and_is_not_asked_for_again(world):
    """The holding node ends by itself at `max_hold_s`: a completed job under our still-open run."""
    from queue_workflows import node_queue
    world.run_to_running()
    job_id = world.view()["boxes"]["box-a"]["reservation"]["job_id"]
    node_queue.mark_completed(job_id, context_delta={}, seconds=world.cfg.cluster_reservation_max_hold_s)
    v = world.tick()
    assert ("down", "box-a") in world.calls("down")
    assert v["state"] == "failed" and "completed" in v["detail"] and v["held"] is False
    for _ in range(3):
        world.tick()
    assert len(world.reservation_jobs()) == 2 and len(world.calls("up")) == 2


@pytest.mark.parametrize("mode", ["cluster", "classic"])
def test_a_boot_that_finds_ranks_with_no_reservation_takes_them_down_and_never_holds_them(world, tmp_path, mode):
    """The migration: generation 3 ran under the OLD park design — ranks up, no reservation row. This LB
    cannot undo those parks (it writes no worker_controls); what it must never do is adopt the ranks."""
    (tmp_path / "cluster.json").write_text(json.dumps({"mode": mode, "generation": 3, "max_num_seqs": 16,
                                                      "max_model_len": 262144}))
    world.executor.ranks = {b: {"running": True, "gen": 3, "exit_code": None} for b in BOXES}
    world.prober.table[TP] = LIVE_TP
    world.boot()
    v = world.tick()
    assert sorted(c[1] for c in world.calls("down")) == ["box-a", "box-b"]
    assert not world.calls("up") and v["held"] is False
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is None or chosen.kind != CLUSTER


def test_start_on_an_engine_without_cluster_reservations_is_refused_with_the_reason(world, monkeypatch):
    from queue_workflows import node_queue
    monkeypatch.delattr(node_queue, "CLUSTER_SERVER_KIND")
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "failed" and "CLUSTER_SERVER_KIND" in v["detail"]
    assert world.reservation_jobs() == [] and not world.calls("up")


def test_stop_takes_both_ranks_down_before_either_reservation_is_released(world):
    from queue_workflows import run_store
    world.run_to_running()
    world.store.set(enabled=False)

    v = world.tick()
    assert world.calls("down") == [("down", "box-a"), ("down", "box-b")]
    assert {run_store.get_run(world.run_id(b))["status"] for b in BOXES} == {"running"}, "still reserved while ranks may stand"
    assert v["state"] == "unloading" and v["held"] is False

    v = world.tick()
    assert v["state"] == "releasing"
    assert {run_store.get_run(world.run_id(b))["status"] for b in BOXES} == {"cancelled"}
    world.bodies_return()
    v = world.tick()
    assert v["state"] == "off" and {b: v["boxes"][b]["reservation"] for b in BOXES} == {"box-a": None, "box-b": None}


def held_by_us(world) -> dict[str, bool]:
    """Per box: does at least ONE open reservation of ours stand on it right now — this generation's or
    an older one's? This is what "the fleet cannot have the box back" means."""
    desired, start_id = world.store.current()
    seen = world.leases.cluster_reservations(BOXES, desired.generation, start_id)
    assert seen is not None, "the broker's rows could not be read"
    return {b: (current is not None and current.run_open) or any(s.run_open for s in stale)
            for b, (current, stale) in seen.items()}


def test_a_restart_asks_for_the_next_generations_reservations_before_giving_the_old_ones_back(world):
    """H1 (worklog/vllm-cluster-handover.md). Live at 01:01:15Z the old reservations went back 36 s
    before the new ones were asked for, and the lane filled the gap with a pool `llm_chat` job that
    re-loaded ollama on the box generation 11 was about to need.

    The engine cannot hold two reservations for one box at once (node_queue.py:1326-1337, 1442-1447), so
    the successor is QUEUED before the predecessor is released — which is enough: while a run-next
    reservation is queued for a box, that lane may claim nothing narrower (node_queue.py:1363-1376) and
    starts no LLM server of its own (claim_worker.py:4632-4662)."""
    from queue_workflows import run_store

    world.run_to_running(max_num_seqs=2)
    old_runs = {b: world.run_id(b) for b in BOXES}
    old_jobs = {j["id"] for j in world.reservation_jobs()}

    world.store.set(enabled=True, restart=True, max_num_seqs=8)
    for _ in range(6):
        v = world.tick()
        assert all(held_by_us(world).values()), f"a box was handed back at {v['state']}: {held_by_us(world)}"
        if {run_store.get_run(r)["status"] for r in old_runs.values()} == {"cancelled"}:
            break
        world.bodies_return()
    new_jobs = {j["id"] for j in world.reservation_jobs()}
    assert new_jobs and not (new_jobs & old_jobs), "the new generation asked for its own reservations"
    assert {run_store.get_run(r)["status"] for r in old_runs.values()} == {"cancelled"}, \
        "the old reservations were never given back"
    assert world.view()["desired"]["generation"] == 2


def test_a_restarted_lb_adopts_the_reservations_and_the_running_pair_without_touching_either(world):
    world.run_to_running()
    before = len(world.executor.calls)
    world.boot()
    world.writes.statements.clear()
    v = world.tick()
    assert [c for c in world.executor.calls[before:] if c[0] in ("up", "down")] == []
    assert v["state"] == "running" and v["held"] is True
    assert world.writes.statements == [] and len(world.reservation_jobs()) == 2


def test_a_restart_adopts_the_running_pair_at_ITS_concurrency_not_the_recommended_default(world):
    """C3 — the criterion that outranks the others (worklog/vllm-cluster-defaults.md).

    The pair runs at concurrency 2: the operator's generation 5 on 2026-09-15. The LB restarts, and
    a FRESH store would now offer the measured maximum of 6. Adoption must win outright — same
    generation, same concurrency, no Up, no Down, no second reservation, nothing written to the
    broker — and the router's admission depth must follow the pair that is actually running, not
    the number the page would have offered."""
    world.run_to_running(max_num_seqs=2)
    assert world.store.defaults.max_num_seqs == 6, \
        "the offered default must differ from the running pair, or this test proves nothing"
    before = len(world.executor.calls)
    jobs_before = {j["id"] for j in world.reservation_jobs()}

    world.boot()                                          # docker compose up -d vllm-lb
    world.writes.statements.clear()
    v = world.tick()

    assert [c for c in world.executor.calls[before:] if c[0] in ("up", "down")] == [], \
        "the default restarted the operator's running cluster"
    assert v["state"] == "running" and v["held"] is True
    assert v["desired"]["max_num_seqs"] == 2, "the default overrode a running pair's concurrency"
    assert v["desired"]["generation"] == 1, "the default bumped the generation"
    assert v["max_depth"] == 2, "admission depth must follow the ranks that are actually up"
    assert v["clamp"] == "", "a running pair must not be clamped"
    assert world.writes.statements == []
    assert {j["id"] for j in world.reservation_jobs()} == jobs_before, "the default re-reserved"


def test_with_nothing_stored_the_loop_publishes_the_recommendation_and_offers_it(world):
    """C2 — a fresh container with an empty state/ dir. The switch offers the measured maximum, and
    /lb/status carries where the number came from, so the page can explain a value nobody typed."""
    v = world.tick()
    assert v["desired"]["max_num_seqs"] == 6, "nothing stored: the page must be offered what fits"
    assert v["desired"]["max_model_len"] == 262144
    assert v["recommended"]["max_num_seqs"] == 6
    assert v["recommended"]["max_model_len"] == 262144
    assert v["recommended"]["kv_tokens"] == 1_708_150
    assert "vllm-cluster-ram-headroom" in v["recommended"]["source"]
    assert v["clamp"] == ""


def test_the_published_recommendation_follows_the_stored_model(world):
    """Audit #7: `cluster.recommended` was worked out once per process, for Flash-Next's 28 GB at 16,392
    B/token, whatever model was chosen. It is the STORED model's own pool and cost each time it is published."""
    assert world.tick()["recommended"]["kv_tokens"] == 1_708_150
    world.store.set(enabled=False, model_choice=DENSE)
    r = world.tick()["recommended"]
    assert (r["kv_cache_memory"], r["kv_tokens"], r["max_num_seqs"]) == (48_000_000_000, 1_302_224, 4)
    world.store.set(enabled=False, model_choice=FLASH)
    assert world.tick()["recommended"]["kv_tokens"] == 1_708_150


def test_an_unreachable_box_a_freezes_the_release_instead_of_guessing(world):
    from queue_workflows import run_store
    world.run_to_running()
    world.executor.reachable["box-b"] = False
    world.store.set(enabled=False)
    v = world.tick()
    assert ("down", "box-a") in world.calls("down")
    assert {run_store.get_run(world.run_id(b))["status"] for b in BOXES} == {"running"}
    assert v["state"] == "unknown" and "box-b" in v["detail"]
    assert v["boxes"]["box-b"]["rank"] is None and v["boxes"]["box-b"]["error"]


MODEL_DONE = LoadPass(LoadPass.MODEL, 28, 28, 100)


def launched(world) -> None:
    """Start → reserved → claimed by both pinned lanes → both ranks launched; the head not answering yet."""
    world.store.set(enabled=True)
    world.tick()
    world.claim()
    world.settle(until="starting")


def test_loading_progress_comes_from_the_head_rank_while_starting(world):
    """20:04:25 on 2026-09-15: the model's LoadPass is in, the MTP drafter's is 7 shards into the same
    checkpoint. Rank 1 prints no count, so only the head is read."""
    launched(world)
    world.executor.log_text["box-a"] = log_at(HEAD, "20:04:25.3")
    world.executor.log_text["box-b"] = log_at(RANK1, "20:04:25.3")
    world.clock.t += 20
    v = world.tick()
    assert v["state"] == "starting"
    assert {k: o for k, o in v["loading"].items() if k not in ("steps", "taken_gib")} == {
        "phase": "loading MTP draft weights", "completed": 7, "total": 28, "percent": 25,
        "passes": [MODEL_DONE.view(), LoadPass(LoadPass.MTP_DRAFT, 7, 28, 25).view()]}
    # …and, the step being a weight pass, the whole GiB box-a has given up since the start was first read: none
    # here, a fake box's memory standing still (cluster_loop `_load_taken`; tests/test_start_rescue.py has it move).
    assert v["loading"]["taken_gib"] == 0
    # and the TRAIL rides along, so a reader of this tick need not have caught the earlier steps as they
    # passed — one read per 15 s cannot see a step narrower than that (cluster_startup.StartupProgress).
    assert v["loading"]["steps"][:4] == ["starting vLLM", "initializing the engine",
                                         "connecting the ranks (NCCL)", "initializing model"]
    assert v["loading"]["steps"][-1] == v["loading"]["phase"]
    assert {box for _, box, _ in world.calls("logs")} == {"box-a"}


def test_the_loop_reads_the_heads_whole_start_and_still_only_every_15_s(world):
    """At "Application startup complete" the head's start had printed 286 lines, and a 200-line read had
    lost the line that begins the load and every line that names the drafter — the passes' names with
    them. The loop reads what tp-rank allows (at most 2000 lines; ~60 KB here), once per
    `cluster_logs_every_s` like before."""
    launched(world)
    whole = log_at(HEAD, "20:08:40")
    world.executor.log_text["box-a"] = whole
    world.clock.t += 20
    v = world.tick()
    assert v["state"] == "starting"
    assert {k: o for k, o in v["loading"].items() if k != "steps"} == {
        "phase": "API server up",
        "passes": [MODEL_DONE.view(), LoadPass(LoadPass.MTP_DRAFT, 28, 28, 100).view()]}
    # ONE read of the whole start carries every step it passed through — the nine the 15 s poll happened
    # to catch on 2026-10-02 and the ten it did not, the CUDA-graph capture among them.
    steps = v["loading"]["steps"]
    assert steps[-1] == "API server up"
    for step in ("initializing the engine", "allocating the KV cache", "KV cache allocated",
                 "warming up kernels", "capturing CUDA graphs", "CUDA graphs captured"):
        assert step in steps, step
    asked = {n for _, _, n in world.calls("logs")}
    assert asked and all(whole.count("\n") <= n <= LOG_LINES_RANGE[1] for n in asked), asked

    reads = len(world.calls("logs"))
    for _ in range(7):                                   # 14 s of 2 s ticks: not due yet
        world.tick()
    assert len(world.calls("logs")) == reads
    world.tick()                                         # 16 s since the last read
    assert len(world.calls("logs")) == reads + 1


def test_at_rest_the_boxes_are_observed_once_a_minute_not_every_tick(world):
    for _ in range(10):
        world.tick()
    assert len(world.calls("status")) == 2, "one observation per box in 20 s at rest"


def test_a_starts_reservations_declare_what_its_model_choice_serves(world):
    """ONE model setting: the reservations' `inputs.model` is the ModelChoice's served name, never a second knob."""
    world.store.set(enabled=True, model_choice="qwen3.8-27b-nvidia-nvfp4")
    world.tick()
    jobs = world.reservation_jobs()
    assert len(jobs) == 2 and {j["inputs"]["model"] for j in jobs} == {"Qwen/Qwen3.8-27B"}


def test_the_published_view_carries_the_contract_the_page_reads(world):
    world.run_to_running(max_num_seqs=8, max_model_len=131072)
    v = world.view()
    assert set(v) >= {"configured", "desired", "state", "detail", "held", "loading", "boxes", "url", "limits",
                      "since", "max_depth", "store_error", "models", "loaded_model"}
    assert v["configured"] is True
    assert v["desired"] == {"mode": "cluster", "generation": 1, "max_num_seqs": 8, "max_model_len": 131072,
                            "model_choice": FLASH, "max_num_batched_tokens": None,
                            "num_speculative_tokens": None, "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                            "kv_cache_auto": False, "long_prefill_token_threshold": None, "uncensored": True, "kv_cache_dtype": None}
    assert set(v["boxes"]) == {"box-a", "box-b"}
    box_a = v["boxes"]["box-a"]
    job_id = box_a["reservation"]["job_id"]
    assert len(job_id) == 36
    assert {k: val for k, val in box_a.items() if k not in ("availability", "memory")} == {
        "label": "box-a-gpu", "reservation": {"job_id": job_id, "status": "running", "claimed_by": "box-a-gpu"},
        "park": None, "running_jobs": 0, "broker_vllm_running": False, "mem_available_gib": 115.0,
        # What the box can be COUNTED ON to have — the lowest of its last observations, which is the
        # number the planner gates on (L3). Equal to the newest here: this fleet's memory is steady.
        "mem_stable_gib": 115.0,
        # MemTotal (tp-rank's status): what the fleet's ram_usable_fraction is a share of — and, its rank standing, not
        # AT REST: the Start's admission leaves a serving box to the loop.
        "mem_total_gib": 121.7, "at_rest": False,
        # The most it has had while this pair serves — what it rests at before load takes its share, which an
        # `auto` KV pool is sized from.
        "mem_rest_gib": 115.0,
        # And what it had when this rank was launched (L4).
        "mem_at_launch": 115.0,
        # What a box owes back once its rank has come DOWN — null while one stands, because the memory
        # it is holding is not memory anybody is waiting for.
        "mem_release_target": None,
        # WHY it is not free for the pair — nothing, here: a serving box is blocked by nothing, and
        # `holds` is None because this fake box's ollama was never asked (L8).
        "holds": None,
        "hold": {"reservation_status": "running", "claimed_by": "box-a-gpu", "waiting_s": 0.0,
                 "deadline_s": 600.0, "blockers": []},
        # …and WHY a rank that exited did (49d280c): nothing, for one that runs. `lost` is a rank whose box
        # rebooted under it (its container older than the box's boot): not this one.
        "rank": {"present": True, "running": True, "gen": 1, "exit_code": None, "model_choice": FLASH,
                 "exit_reason": "", "lost": False},
        # The last heal of its broker worker: none — its lane never went quiet.
        "heal": None,
        # What the GPU under the rank is doing: null here, the fake GB10 box has no nvidia-smi — and an
        # unobserved GPU is null, never a reassuring 0 %.
        "gpu": None,
        # The TP link's rate: null until two RDMA readings exist — the fake GB10 box reports none.
        "network": None,
        "error": ""}
    assert v["url"] == TP and v["limits"] == {"max_num_seqs": [1, 48], "max_model_len": [1024, 262144],
                                              "max_num_batched_tokens": [2048, 65536],
                                              "num_speculative_tokens": [0, 8],
                                              "kv_cache_memory": [8000000000, 96000000000],
                                              "kv_offload_gib": [0, 512], "ple_replicate": [0, 1],
                                              "long_prefill_token_threshold": [0, 65536],
                                              # the checkpoint knob (eb92362): original 0, twin 1
                                              "uncensored": [0, 1]}
    assert v["max_depth"] == 8 and isinstance(v["since"], float)
    assert json.loads(json.dumps(v)) == v, "the whole cluster block is what the page is sent"


def test_the_view_offers_every_model_the_cluster_can_launch_with_its_own_limits(world):
    """ONE ModelCatalog behind the picker: each ModelSpec with the limits that spec may be started with."""
    v = world.tick()
    by_id = {m["id"]: m for m in v["models"]}
    assert set(by_id) == {FLASH, DENSE, DFLASH, SGLANG, FLASH_SGLANG, GLM, DSV4, "qwen3.8-flash-next-tensorfold-nvfp4",
                          "qwen3.8-flash-next-tensorfold-mlx-mtp"}, "the classic-only spec is not a cluster choice"
    # TensorFold serves 1..16 streams (--parallel), and the picker offers no more.
    assert by_id["qwen3.8-flash-next-tensorfold-nvfp4"]["limits"] == {"max_model_len": 262144, "max_num_seqs": [1, 16]}
    assert by_id[FLASH]["launch"]["cluster"]["default"] is True
    assert by_id[DENSE]["served_name"] == "Qwen/Qwen3.8-27B"
    assert by_id[FLASH]["limits"] == {"max_model_len": 262144, "max_num_seqs": [1, 48]}
    # The drafter that is its own checkpoint is a choice of its own, and since 2026-10-02 it is offered
    # over the gateway's whole range — 1..48, the same as the dense profile. A dflash step does submit
    # 1 + K queries a seat, but 48 x (1 + 7) = 384 of them sit well inside the 8,192-token prefill budget,
    # and the 28 GB pin holds at most 49 requests at any context: 48 is the ceiling the arithmetic already
    # gave, and what the pool really admits at the chosen context is the page's clamp to say, not a
    # second, lower ceiling hidden in the catalog.
    assert by_id[DFLASH]["limits"] == {"max_model_len": 262144, "max_num_seqs": [1, 48]}
    # 4, not 7: measured over five generations on 2026-10-03 (worklog/dflash2-27b-tp2-tuning.md) —
    # 201.5 tok/s at 8 streams against K=7's 187.6, because acceptance is 0.577 against 0.393.
    assert by_id[DFLASH]["launch"]["cluster"]["num_speculative_tokens"] == 4


def test_the_boxes_say_what_they_have_and_the_view_shows_it_per_box(world):
    world.tick()
    for b in BOXES:
        seen = world.view()["boxes"][b]["availability"]
        assert seen[FLASH]["present"] is True and seen[FLASH]["name"] == "Qwen3.8-Flash-Next-hibrid48"
        assert seen[FLASH]["searched"], "the box says where it looked"
    assert [c for c in world.executor.calls if c[0] == "availability"], "the boxes were asked"


def test_a_box_a_without_the_chosen_model_blocks_the_start_and_reserves_nothing(world):
    """The expensive mistake: taking both GB10 boxes from the fleet for a model that is not on them."""
    world.executor.have["box-b"] = {FLASH: False}
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "blocked" and v["held"] is False
    assert "box-b" in v["detail"]
    assert world.reservation_jobs() == [] and not world.calls("up")


def test_a_box_that_cannot_say_what_it_has_is_checked_not_guessed(world):
    world.executor.have["box-a"] = None
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "checking" and world.reservation_jobs() == []


def test_the_launch_carries_the_model_choice_and_the_rank_remembers_which_one(world):
    world.run_to_running(model_choice=DENSE, serves="Qwen/Qwen3.8-27B")
    assert world.calls("up")[0][-1] == DENSE
    assert world.view()["boxes"]["box-a"]["rank"]["model_choice"] == DENSE
    assert world.view()["loaded_model"]["spec"] == DENSE, "and the head is serving it"


def test_the_view_names_the_model_the_head_actually_serves_never_the_one_we_asked_for(world):
    """LoadedModel is OBSERVED, from the head's own /v1/models."""
    world.run_to_running()
    loaded = world.view()["loaded_model"]
    assert loaded == {"served_name": "Qwen/Qwen3.8-Flash-Next", "spec": FLASH,
                      "label": "Qwen3.8 Flash-Next hibrid48 · MoE NVFP4", "max_model_len": 262144,
                      "efforts": None}


def test_a_head_serving_a_model_we_did_not_choose_is_never_held(world):
    world.run_to_running()
    world.prober.table[TP] = {**LIVE_TP, "served_models": ("Qwen/Qwen3.8-27B",)}
    v = world.tick()
    assert v["held"] is False and v["state"] == "failed"
    assert "Qwen/Qwen3.8-27B" in v["detail"]
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is None or chosen.kind != CLUSTER, "no traffic to a pair serving something else"


def test_the_memory_a_start_waits_for_is_the_measured_need_for_this_selection(world):
    """H3 (worklog/vllm-cluster-handover.md). What the page is told the box needs and what the planner
    actually waits for must be ONE number — the estimator's, for this model at this concurrency and
    context — not the configured constant."""
    world.executor.mem = {"box-a": 96.0, "box-b": 96.0}
    # The table replicated, as the 2026-09-15 start that calibrated the need ran it.
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144, ple_replicate=True)
    world.tick()
    world.claim()
    for _ in range(3):                       # until the box has been observed enough times to judge (L3)
        v = world.tick()
    assert v["state"] == "freeing", v
    est = v["boxes"]["box-a"]["memory"]
    need = est["needed_now_gib"]
    assert need == round(est["total_gib"] + world.cfg.cluster_start_headroom_gib, 1), \
        "the requirement is what the selection costs plus the headroom a start must leave"
    assert f"box-a: 96.0 GiB free, need {need:.1f}" in v["detail"], v["detail"]
    assert not world.calls("up")


def test_a_box_is_judged_on_the_lowest_of_its_last_observations_not_on_one_lucky_sample(world):
    """L3 (worklog/vllm-cluster-hold-robustness.md §3): while a hold and a chat client fight over
    ollama, MemAvailable swings by the size of the model (17.6 GiB) on the unload/reload cycle. One
    sample landing high is not the box being free."""
    world.executor.mem = {"box-a": 77.0, "box-b": 115.0}
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    world.claim()
    assert world.tick()["state"] == "freeing"
    for lucky in (115.0, 77.0, 115.0):                   # the swing, sampled tick by tick
        world.executor.mem["box-a"] = lucky
        v = world.tick()
        assert not world.calls("up"), f"launched on a single {lucky} GiB sample: {v['detail']}"
    assert v["boxes"]["box-a"]["mem_stable_gib"] == 77.0, "the page shows what the box can be counted on"

    world.executor.mem["box-a"] = 115.0                  # and once it STAYS free, it launches
    for _ in range(3):
        v = world.tick()
    assert len(world.calls("up")) == 2 and v["state"] == "starting", v["detail"]


def test_a_freeing_wait_past_its_budget_says_so_names_the_blocker_and_still_launches_when_the_ram_is_back(world):
    """L3: `freeing` had no deadline at all — it could wait silently forever. Past its budget it SAYS so, with
    the blocker; it is not failed (a timeout is not an error — operator, 2026-10-10), and nothing is given
    up: the RAM coming back is what the wait was for."""
    world.executor.mem = {"box-a": 77.0, "box-b": 115.0}
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    world.claim()
    assert world.tick()["state"] == "freeing"
    world.clock.t += world.cfg.cluster_freeing_deadline_s
    v = world.tick()
    assert v["state"] == "freeing" and not world.calls("up")
    assert "box-a: 77.0 GiB free, need" in v["detail"] and "past" in v["detail"] and "600 s" in v["detail"], v["detail"]
    assert "Stop, then Start again" not in v["detail"] and "failed" not in v["detail"], v["detail"]
    v = world.tick()
    assert v["state"] == "freeing" and "past" in v["detail"], "and it stays said, not a one-tick blip"
    world.executor.mem = {"box-a": 115.0, "box-b": 115.0}
    v = world.settle(until="starting")
    assert len(world.calls("up")) == 2, v["detail"]


def test_a_box_whose_nfs_share_is_not_mounted_is_waited_for_and_launches_once_it_can_see(world):
    """THE FALSE FLAG OF 2026-10-10 20:15:22Z, end to end: box-b (just rebooted) looks for the checkpoint over
    NFS from box-a before box-a's export is up. The LB does not say `blocked … choose another model`; it says
    what box-b waits for and why, keeps nothing from the fleet, and — the share mounted by the heal two seconds
    later — launches as if nothing had happened."""
    world.executor.have["box-b"] = {FLASH: ("/mnt/box-a-home/models/qwen38-flash-next-cluster is not mounted: "
                                             "NFS /mnt/box-a-home (192.0.2.201:/) is not up on this box yet")}
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "checking" and v["held"] is False, v
    assert "waiting for box-b to see Qwen3.8-Flash-Next-hibrid48" in v["detail"], v["detail"]
    assert "not mounted" in v["detail"] and "/mnt/box-a-home" in v["detail"], v["detail"]
    assert "choose another model" not in v["detail"] and "no Qwen3.8" not in v["detail"], v["detail"]
    assert world.reservation_jobs() == [] and not world.calls("up")
    seen = v["boxes"]["box-b"]["availability"][FLASH]
    assert seen["present"] is None and seen["dir"] is None and "not mounted" in seen["error"]
    # The share comes up (the heal remounted it). A box the start is WAITING ON is asked again at the status
    # cadence, not the picker's idle one: the next tick sees it, and the start goes on.
    world.executor.have["box-b"] = {}
    v = world.tick()
    assert v["state"] == "reserving", f"{v['state']} — {v['detail']}"
    world.claim()
    world.settle(until="starting")
    assert len(world.calls("up")) == 2


def test_a_box_that_cannot_be_observed_is_unknown_and_the_page_says_why(world):
    """`reserving → unknown: cannot observe box-b` (2026-10-10) — the executor knew it was ssh timing out on a
    box mid-reboot, and the page said nothing of it."""
    world.store.set(enabled=True)
    world.tick()
    world.executor.reachable["box-b"] = False
    v = world.tick()
    assert v["state"] == "unknown" and v["held"] is False
    assert "cannot observe box-b" in v["detail"] and "ssh: connect to host timed out" in v["detail"], v["detail"]
    assert v["boxes"]["box-b"]["error"] == "ssh: connect to host timed out"


def test_a_rank_whose_box_rebooted_is_relaunched_not_failed(world):
    """2026-09-19 08:07Z: a power cut took all three boxes; docker on the GB10 boxes came back with both rank
    containers exited (255); the LB said `failed: rank on box-a exited (code 255); Stop, then Start again` and
    kept both GB10 boxes reserved and idle for 3.5 h. A container older than its box's boot did not exit — the box
    rebooted under it: both ranks come down and the same generation launches again, through every gate."""
    launched(world)
    gen = world.view()["desired"]["generation"]
    booted = world.clock.t + 60.0                           # the box boots a minute after the launch…
    world.executor.started_at["box-a"] = "2026-09-15T19:58:01.123456789Z"
    world.executor.boot_at["box-a"] = 1789502281.0 + 60.0  # …and its clock says so, against the container's start
    world.executor.ranks["box-a"] = {**world.executor.ranks["box-a"], "running": False, "exit_code": 255}
    world.clock.t = booted + 120.0
    v = world.tick()
    assert v["state"] == "restarting" and v["held"] is False, f"{v['state']} — {v['detail']}"
    assert "box-a rebooted" in v["detail"] and "relaunching" in v["detail"], v["detail"]
    assert "exited" not in v["detail"] and "Stop, then Start" not in v["detail"], v["detail"]
    assert sorted(c[1] for c in world.calls("down")) == ["box-a", "box-b"] and not world.calls("halt")
    assert v["boxes"]["box-a"]["rank"]["lost"] is True
    v = world.settle(until="starting")
    ups = world.calls("up")
    assert len(ups) == 4 and ups[2:] == ups[:2], "the same generation, the same settings, launched again"
    assert v["desired"]["generation"] == gen
    assert {j["status"] for j in world.reservation_jobs()} == {"running"}, "the reservations were never let go"


def test_a_rank_that_exited_after_its_box_booted_is_a_failure_as_before(world):
    launched(world)
    world.executor.started_at["box-a"] = "2026-09-15T19:58:01.123456789Z"
    world.executor.boot_at["box-a"] = 1789502281.0 - 3600.0  # the box booted an hour BEFORE the container started
    world.executor.ranks["box-a"] = {**world.executor.ranks["box-a"], "running": False, "exit_code": 255}
    v = world.tick()
    assert v["state"] == "failed" and "exited (code 255)" in v["detail"], v["detail"]
    assert v["boxes"]["box-a"]["rank"]["lost"] is False


RESIDENT = {"date": "Wed, 16 Sep 2026 01:02:00 GMT",
            "ps": {"models": [{"name": "qwen3.8:latest", "size": 18_882_431_548,
                               "size_vram": 18_882_431_548, "context_length": 262144,
                               "expires_at": "2026-09-16T01:07:00.0Z"}]}}


def test_a_box_that_is_not_free_says_what_is_holding_it(world):
    """L8: the per-box `hold` object — what the box is waiting on, in the LB's own words, read from the
    box's own `/api/ps` through tp-rank's status (worklog/vllm-cluster-hold-robustness.md §6)."""
    world.executor.mem = {"box-a": 77.0, "box-b": 115.0}
    world.executor.ollama["box-a"] = RESIDENT
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    world.claim()
    for _ in range(3):
        v = world.tick()
    assert v["state"] == "freeing"
    hold = v["boxes"]["box-a"]["hold"]
    assert [b["code"] for b in hold["blockers"]] == ["ollama_resident", "mem_short"]
    assert "ollama holds qwen3.8:latest, 17.6 GiB" in hold["blockers"][0]["message"]
    need = v["boxes"]["box-a"]["memory"]["needed_now_gib"]
    assert "77.0 GiB free" in hold["blockers"][1]["message"]
    assert f"{need:.1f}" in hold["blockers"][1]["message"], hold["blockers"][1]["message"]
    assert hold["reservation_status"] == "running" and hold["claimed_by"] == "box-a-gpu"
    assert hold["waiting_s"] >= 0 and hold["deadline_s"] == world.cfg.cluster_freeing_deadline_s
    assert "ollama holds qwen3.8:latest, 17.6 GiB" in v["detail"], v["detail"]
    # …and the top-level wait is visibly bounded rather than silent.
    assert v["freeing"]["since"] and v["freeing"]["deadline"] > v["freeing"]["since"]

    free = v["boxes"]["box-b"]["hold"]
    assert free["blockers"] == [], "a box that is ready is not 'blocked' by anything"


def _beat(*labels: str) -> None:
    from queue_workflows import node_queue
    for label in labels:
        node_queue.upsert_worker_heartbeat(host_label=label, queue="gpu", concurrency=1,
                                           llm_servers_available=["ollama"])


def test_a_reservation_no_lane_can_claim_names_the_dead_lane(world):
    """2026-09-23, cluster#18: box-b's worker container had failed its boot start, so its reservation sat
    `queued` for 27 minutes while the page said "reserving" with an EMPTY blocker list."""
    _beat("box-a-gpu")                                   # box-a's lane is up; box-b's is not
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    v = world.tick()
    assert v["state"] == "reserving"
    dead = v["boxes"]["box-b"]["hold"]["blockers"]
    assert [b["code"] for b in dead] == ["no_lane"], dead
    assert "is its worker container up?" in dead[0]["message"]
    assert v["boxes"]["box-a"]["hold"]["blockers"] == [], "a queued reservation on a live lane is not blocked"
    assert "box-b-gpu is not heartbeating: no broker_parrot worker is up on box-b" in v["detail"], v["detail"]
    assert "box-a-gpu is not" not in v["detail"]

    _beat("box-b-gpu")                                  # the worker comes back
    v = world.tick()
    assert v["boxes"]["box-b"]["hold"]["blockers"] == [] and "not heartbeating" not in v["detail"]


def test_an_empty_heartbeat_table_is_not_a_dead_lane(world):
    """No gpu lane heartbeats anywhere is "we cannot tell" (the line vllm_capacity draws), never a blocker."""
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    v = world.tick()
    assert v["state"] == "reserving"
    assert all(v["boxes"][b]["hold"]["blockers"] == [] for b in BOXES)
    assert "not heartbeating" not in v["detail"]
    assert world.calls("heal") == [], "a lane we cannot see is not a lane to heal"


def test_a_dead_lane_is_healed_paced_and_never_failed(world, caplog):
    """2026-09-28, generation 29: box-b booted with /srv/hdd latched, so no worker, so no lane, and its
    reservation sat `queued` 1 h 45 min with box-a held idle while the page asked "is its worker container
    up?". The operator: "we want healing service not killing service". The LB brings the worker back —
    one heal per HEAL_EVERY_S, for as long as it takes — and gives nothing up for it."""
    caplog.set_level(logging.INFO, logger="vllm_lb.cluster_loop")
    _beat("box-a-gpu")                                   # box-a's lane is up; box-b's is not
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    v = world.tick()
    assert v["state"] == "reserving"
    assert world.calls("heal") == [("heal", "box-b")], "only the box whose lane is dead"
    assert "box-b-gpu is not heartbeating: no broker_parrot worker is up on box-b to claim it — healing it" \
        in v["detail"], v["detail"]
    assert v["boxes"]["box-b"]["heal"] == {"at": pytest.approx(world.clock.t - 2.0), "ok": True,
                                            "detail": "remounted /srv/hdd; worker backstop restarted"}
    assert v["boxes"]["box-a"]["heal"] is None
    assert "heal box-b: ok (remounted /srv/hdd; worker backstop restarted)" in caplog.text

    for _ in range(int(HEAL_EVERY_S // 2) - 2):         # one heal plays out before the next
        v = world.tick()
    assert len(world.calls("heal")) == 1 and v["state"] == "reserving", "a dead lane is never a failure"
    world.tick()
    world.tick()
    assert world.calls("heal") == [("heal", "box-b")] * 2, "still dead after HEAL_EVERY_S: healed again"

    _beat("box-b-gpu")                                  # the worker is back
    world.clock.t += HEAL_EVERY_S
    v = world.tick()
    assert len(world.calls("heal")) == 2 and "healing" not in v["detail"]
    assert v["boxes"]["box-b"]["heal"]["ok"] is True, "the record of what brought it back stays"
    world.claim()
    world.settle(until="starting")
    assert world.calls("up"), "the pair starts as it always did"


def test_a_heal_that_fails_is_retried_and_its_reason_is_on_the_page(world):
    _beat("box-a-gpu")
    world.executor.heal_reply = (False, "heal: exit 1: still not mounted: /srv/hdd (mount.nfs: "
                                        "Network is unreachable)")
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    world.tick()
    v = world.tick()
    assert v["state"] == "reserving"
    assert v["boxes"]["box-b"]["heal"]["ok"] is False
    assert "Network is unreachable" in v["boxes"]["box-b"]["heal"]["detail"]
    world.clock.t += HEAL_EVERY_S
    world.tick()
    assert len(world.calls("heal")) == 2, "a failed heal is tried again, never given up on"


def test_a_heal_that_is_not_due_does_not_cancel_the_ssh_pacing(world):
    """The loop looks again right after it ACTS. The planner asks for a heal on every tick; a heal that is
    not due must not count as acting, or a dead lane would poll both GB10 boxes on every tick — the pattern
    that once ran box-a's sshd into MaxStartups."""
    world.cfg = replace(world.cfg, cluster_observe_s=10.0)
    world.boot()
    _beat("box-a-gpu")
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144)
    for _ in range(4):                                   # reserve, heal, look again
        world.tick()
    assert len(world.calls("heal")) == 1
    before = len(world.calls("status"))
    for _ in range(10):                                  # 20 s of a dead lane, no heal due
        world.tick()
    polls = len(world.calls("status")) - before
    assert polls <= 6, f"{polls} status calls in 20 s: paced at 10 s would be 4, every tick 20"


def test_a_running_pair_is_never_healed_even_with_a_lane_that_reads_dead(world):
    """Heal answers a reservation NO lane can claim — a queued one. While both reservations run, the claim
    itself is the proof a lane is there, whatever the heartbeat table says; a serving pair (a benchmark
    at max_num_seqs=16, 2026-09-28) must see no SSH action at all."""
    world.run_to_running()
    _beat("box-b-gpu")                                  # box-a's lane reads dead in the table
    before = len(world.executor.calls)
    for _ in range(int(HEAL_EVERY_S)):                   # six minutes of ticks
        v = world.tick()
        assert v["state"] == "running" and v["held"] is True
    acted = [c for c in world.executor.calls[before:] if c[0] not in ("status", "availability", "logs")]
    assert acted == [], acted
    assert all(v["boxes"][b]["heal"] is None for b in BOXES)


def test_a_worker_that_dies_under_a_serving_pair_is_healed_inside_the_grace(world):
    """The lane that holds a RUNNING pair's reservation dies: broker_parrot re-queues it, and after the
    grace the pair would come down. Heal the worker inside the grace instead."""
    from queue_workflows import node_queue
    world.run_to_running()
    _beat("box-b-gpu")                                  # box-a's lane stopped heartbeating
    before = len(world.executor.calls)
    assert node_queue.requeue_running_for_worker("box-a-gpu", "gpu") == 1
    v = world.tick()
    assert v["state"] == "waiting"
    assert "box-a-gpu is not heartbeating: healing it" in v["detail"], v["detail"]
    acted = [c for c in world.executor.calls[before:] if c[0] in ("heal", "up", "down", "halt")]
    assert acted == [("heal", "box-a")], "healed — and the standing pair left alone"
    world.claim("box-a")                                 # the worker is back and takes it again
    assert world.tick()["state"] == "running"


def test_the_memory_a_box_had_at_launch_is_logged_and_published(world, caplog):
    """L4 (worklog/vllm-cluster-hold-robustness.md): `_mem_at_launch` is captured and nothing says it,
    so the one number that would settle a post-mortem — what the box really had when 99 GB of weights
    were sent to it — cannot be reconstructed. "The difference between 'we think the gate passed' and
    'the gate passed at 101.3'."""
    world.executor.mem = {"box-a": 114.8, "box-b": 114.3}
    # The 2026-09-15 start, run 1, as it ran: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144, ple_replicate=True)
    world.tick()
    world.claim()
    with caplog.at_level(logging.INFO, logger="vllm_lb.cluster_loop"):
        world.settle(until="starting")
    v = world.view()
    assert v["boxes"]["box-a"]["mem_at_launch"] == 114.8
    assert v["boxes"]["box-b"]["mem_at_launch"] == 114.3
    launch = [r.getMessage() for r in caplog.records if "up box-a:" in r.getMessage()]
    assert launch and "114.8" in launch[0], launch
    # …and what the GATE waited for — the cost plus the start headroom, the figure the plan compared — not the
    # cost alone (103.2), which the line named until 2026-10-10 while the gate waited on 111.2.
    need = v["boxes"]["box-a"]["memory"]["needed_now_gib"]
    assert need == 111.2 and f"needs {need:.1f}" in launch[0], "and what it needed, so the two can be compared in the log"

    world.store.set(enabled=False)
    for _ in range(4):
        world.tick()
        world.bodies_return()
    assert world.view()["boxes"]["box-a"]["mem_at_launch"] is None, "forgotten with the rank"


def test_each_box_carries_a_memory_estimate_for_the_selection_and_then_the_actual(world):
    """M8: what this model, at this concurrency and context, needs on each box — and, once the pair
    runs, what the box actually gave up, so a wrong constant shows itself."""
    world.executor.mem = {"box-a": 114.8, "box-b": 114.3}
    # The 2026-09-15 start, run 1, as it ran: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144, ple_replicate=True)
    world.tick()
    est = world.view()["boxes"]["box-a"]["memory"]
    assert (est["box"], est["role"], est["model_choice"]) == ("box-a", "head", FLASH)
    assert est["total_gib"] > 90 and est["leaves_gib"] == round(114.8 - est["total_gib"], 1)
    assert est["actual_used_gib"] is None, "nothing has been loaded yet"
    assert world.view()["boxes"]["box-b"]["memory"]["role"] == "worker"

    world.claim()
    world.settle(until="starting")
    world.prober.table[TP] = LIVE_TP
    world.executor.mem = {"box-a": 11.6, "box-b": 16.6}
    est = world.tick()["boxes"]["box-a"]["memory"]
    assert est["actual_used_gib"] == round(114.8 - 11.6, 1), "what the box gave up since the launch"
    assert abs(est["deviation_pct"]) <= 10.0, est


# ── a start that hangs, and a box that runs dry ──────────────────────────────

HUNG_HEAD = "qwen38-start-2026-09-18-hung-rank0.log"


def hung_at(when: str) -> str:
    """The head's log of the 2026-09-18 hang as `docker logs` held it at `when` (UTC HH:MM:SS)."""
    return "".join(f"{body}\n" for stamp, body in stamped(HUNG_HEAD) if stamp <= "2026-09-18T" + when)


def clock_of(when: str) -> int:
    h, m, sec = (int(v) for v in when.split(":"))
    return h * 3600 + m * 60 + sec


def replay(world, log, start: str, end: str, step: int = 15) -> list[dict]:
    """Tick every `step` seconds from `start` to `end`, the head's log growing as it really did."""
    views = []
    for t in range(clock_of(start), clock_of(end) + 1, step):
        world.executor.log_text["box-a"] = log(f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}")
        views.append(world.tick(seconds=step))
    return views


def no_relaunches(world) -> None:
    """An LB told never to relaunch a hung start (LB_CLUSTER_START_RETRIES=0): the rule before rescues."""
    world.cfg = replace(world.cfg, cluster_start_retries=0)
    world.boot()


def test_the_2026_09_18_hang_is_stalled_ten_minutes_in_and_both_ranks_are_halted_with_their_logs(world, caplog):
    """The head printed "Autotuning process starts" at 22:37:25 and nothing that moved its start for the next
    53 minutes. Replayed at the loop's own cadence: `starting`, saying how long it has been quiet, then
    `stalled` once the quiet reaches the deadline — the ranks HALTED (never removed), and the reason still on
    the page after they are seen exited. The deadline is now the step's own (4 x the slowest healthy autotune,
    335 s, not 600), and this LB is told not to relaunch: test_start_rescue.py has the relaunches. The word is
    `stalled`, never `failed`: a clock ran out, nothing errored (operator, 2026-10-10)."""
    no_relaunches(world)
    launched(world)
    caplog.set_level(logging.INFO, logger="vllm_lb.cluster_loop")
    views = replay(world, hung_at, "22:36:00", "22:50:00")
    states = [v["state"] for v in views]
    assert "stalled" in states, f"53 minutes of silence and the loop never gave up: {set(states)}"
    assert "failed" not in states, f"a hang was called failed: {[v['detail'] for v in views if v['state'] == 'failed']}"
    stalled_at = 15 * states.index("stalled") + clock_of("22:36:00")
    limit = start_stall_limit("autotuning FlashInfer kernels")
    assert clock_of("22:37:25") + limit <= stalled_at <= clock_of("22:37:25") + limit + 45, stalled_at
    quiet = [v["detail"] for v in views if v["state"] == "starting" and "no progress" in v["detail"]]
    assert quiet and "autotuning FlashInfer kernels" in quiet[-1]
    assert sorted(box for _, box in world.calls("halt")) == ["box-a", "box-b"]
    assert not world.calls("down"), "a halted rank's log is its post-mortem: it is halted, never removed"
    assert all(world.executor.ranks[b]["exit_code"] == 137 for b in BOXES)
    last = views[-1]
    assert last["state"] == "stalled" and "hung in" in last["detail"] and "halted" in last["detail"] \
        and "autotuning FlashInfer kernels" in last["detail"], \
        f"the reason left the page once the ranks were seen exited: {last['detail']}"
    assert "137" not in last["detail"], "the exit code of a rank the LB itself stopped is a consequence, not the reason"
    assert [r.message for r in caplog.records if "→ stalled" in r.message and "hung in" in r.message]
    assert not [r.message for r in caplog.records if "→ failed" in r.message]
    # …and the reservations are untouched: the halt is sticky, Stop is the operator's.
    assert {j["status"] for j in world.reservation_jobs()} == {"running"}


def test_the_2026_09_15_start_is_never_mistaken_for_a_hang(world):
    """The healthy start, replayed whole at the same cadence: its longest quiet window is 56 s."""
    launched(world)
    views = replay(world, lambda when: log_at(HEAD, when), "19:56:15", "20:08:40")
    assert {v["state"] for v in views} == {"starting"}
    assert not world.calls("halt") and not any("no progress" in v["detail"] for v in views)


def test_a_start_whose_log_cannot_be_read_is_never_called_stalled(world):
    launched(world)
    for _ in range(60):                                    # 15 minutes of an unreadable log
        world.executor.log_text["box-a"] = ""
        v = world.tick(seconds=15)
    assert v["state"] == "starting" and not world.calls("halt")


def test_a_pinned_stall_is_forgotten_by_the_next_start(world):
    no_relaunches(world)
    launched(world)
    replay(world, hung_at, "22:37:30", "22:48:30")
    assert world.view()["state"] == "stalled"
    world.store.set(enabled=True, restart=True)
    world.executor.log_text["box-a"] = ""
    v = world.tick()
    assert v["state"] not in ("failed", "stalled") and "hung in" not in v["detail"], v


def test_a_box_that_runs_dry_under_a_starting_rank_gets_the_pair_halted_before_the_box_dies(world):
    launched(world)
    world.executor.mem["box-b"] = 0.1                     # box-b, 2026-09-18 23:21-23:34Z
    v = world.tick()
    assert v["state"] == "starting", "one sample is not evidence"
    for _ in range(world.cfg.cluster_mem_samples):
        v = world.tick()
    assert v["state"] == "failed" and "box-b" in v["detail"] and "out of memory" in v["detail"]
    assert sorted(box for _, box in world.calls("halt")) == ["box-a", "box-b"]
    chosen = router.choose(world.hub.snapshot(), world.cfg)
    assert chosen is None or chosen.url != TP, "a pair that is being stopped is not routed to"


def test_a_serving_pair_on_a_tight_box_keeps_serving(world):
    world.run_to_running()
    world.executor.mem["box-a"] = 0.3
    for _ in range(world.cfg.cluster_mem_samples + 2):
        v = world.tick()
    assert v["state"] == "running" and v["held"] and not world.calls("halt")


def test_a_serving_pair_shows_no_box_as_blocked_by_the_memory_its_own_rank_holds(world):
    """LIVE, 2026-09-19 00:30Z, generation 13 SERVING: both box rows on the page read "5.4 GiB free, needs
    103.2 GiB: short 97.8 GiB" and "10.9 GiB free, needs 97.7 GiB: short 86.8 GiB". The memory missing is
    the memory our own rank holds; a box whose rank stands is not waiting for anything, and a row that says
    it is short reads as a cluster that is not running while it is loaded."""
    world.run_to_running()
    for b in BOXES:
        world.executor.mem[b] = 5.4
    for _ in range(world.cfg.cluster_mem_samples + 1):
        v = world.tick()
    assert v["state"] == "running"
    for b in BOXES:
        assert v["boxes"][b]["hold"]["blockers"] == [], f"{b}: {v['boxes'][b]['hold']['blockers']}"


# ── the engine knobs, and what the running pair was launched WITH ─────────────

def _effective(world) -> dict | None:
    return world.view()["effective"]


def _tuned(key: str, model: str = FLASH):
    """The MODEL's own measured-best value for `key` (`catalog.TunedSetup`), read from the shipped catalog so
    these tests move with it rather than asserting a stale copy of a number."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    tuned = ModelCatalog.load().get(model).profile(CLUSTER_MODE).tuned
    return tuned.get(key) if tuned is not None else None


def _ple(model: str = FLASH):
    """The MODEL's own PLE placement (`LaunchProfile.ple_replicate`), read from the shipped catalog like `_tuned`:
    split for Flash-Next since 2026-10-11 (tests/test_flash_next_ple_split.py)."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    return ModelCatalog.load().get(model).profile(CLUSTER_MODE).ple_replicate


def test_effective_is_null_until_a_pair_runs_and_then_says_what_it_was_launched_with(world):
    """THE CONTRACT the performance log tags every sample with: `cluster.effective` is the RESOLVED settings
    the running pair was launched with — the knobs nobody set resolved to this MODEL's measured-best budget
    (4,096; the box's 8,192 only for a model with none measured) and the catalog's 4 — and null while no pair
    runs."""
    assert world.tick()["effective"] is None
    world.run_to_running(max_num_seqs=8, max_model_len=131072)
    assert _effective(world) == {"model_choice": FLASH, "generation": 1, "max_num_seqs": 8, "max_model_len": 131072,
                                 "max_num_batched_tokens": _tuned("max_num_batched_tokens"), "num_speculative_tokens": 4, "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                                 "long_prefill_token_threshold": 0, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}
    assert list(_effective(world)) == ["model_choice", "generation", "max_num_seqs", "max_num_batched_tokens",
                                       "num_speculative_tokens", "ple_replicate", "kv_cache_memory",
                                       "kv_offload_gib", "max_model_len",
                                       "long_prefill_token_threshold", "uncensored", "kv_cache_dtype"], "the contract's own order"
    assert list(world.view()["defaults"]) == ["max_num_seqs", "max_num_batched_tokens", "num_speculative_tokens",
                                              "ple_replicate", "kv_cache_memory", "kv_offload_gib",
                                              "max_model_len", "long_prefill_token_threshold", "uncensored", "kv_cache_dtype"]
    assert world.executor.knobs == {b: {"max_num_batched_tokens": None, "num_speculative_tokens": None,
                                        "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                                        "long_prefill_token_threshold": None,
                                        # …except WHICH CHECKPOINT, which the Start resolves and stores as a bool:
                                        # it changes what is launched, and the planner reads the box's answer
                                        # for that very checkpoint (`Desired.uncensored`).
                                        "uncensored": True}
                                    for b in BOXES}, "a default is left to the launch, never sent as a value"
    world.store.set(enabled=False)
    v = world.tick()
    assert v["effective"]["generation"] == 1, "the tick that takes the ranks down still saw them run"
    assert world.tick()["effective"] is None, "no pair runs once its ranks are down"


def test_the_operators_knobs_reach_both_ranks_and_effective(world):
    # A 16,384 prefill budget is 8.8 GiB more than the calibrated one, and a start must clear the
    # headroom on top: these boxes hold it, so the test stays about the KNOBS reaching both ranks.
    world.executor.mem = {b: 125.0 for b in BOXES}
    world.run_to_running(max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=16384,
                         num_speculative_tokens=2, long_prefill_token_threshold=2048)
    assert world.executor.knobs == {b: {"max_num_batched_tokens": 16384, "num_speculative_tokens": 2,
                                        "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                                        "long_prefill_token_threshold": 2048, "uncensored": True}
                                    for b in BOXES}
    assert _effective(world) == {"model_choice": FLASH, "generation": 1, "max_num_seqs": 8, "max_model_len": 262144,
                                 "max_num_batched_tokens": 16384, "num_speculative_tokens": 2,
                                 "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                                 "long_prefill_token_threshold": 2048, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}


def test_the_box_default_is_what_effective_says_when_the_operator_set_nothing(world):
    """The LB's own CLUSTER_MAX_NUM_BATCHED_TOKENS is the box's default: it travels with every call (never
    re-sent by the loop), and effective resolves it the way tp-rank does."""
    world.executor.env = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "4096"}
    world.run_to_running()
    assert _effective(world)["max_num_batched_tokens"] == 4096
    assert world.view()["defaults"]["max_num_batched_tokens"] == 4096


def test_a_restart_waits_for_the_turns_streaming_on_the_old_pair_and_routes_none_there_meanwhile(world):
    """Generation 135 → the next one cut 34 streams in flight; each Claude Code turn was re-sent whole and
    re-prefilled from nothing. Now the old pair stays up, unrouted, until its last turn ends."""
    world.executor.mem = {b: 125.0 for b in BOXES}
    world.run_to_running(max_num_seqs=8, max_model_len=262144)
    world.hub.note_request_start(TP)
    world.hub.note_request_start(TP)
    world.store.set(enabled=True, restart=True, num_speculative_tokens=3)
    v = world.tick(seconds=60)
    assert v["state"] == "finishing" and "2 turn(s)" in v["detail"] and v["held"] is False
    assert not world.calls("down"), "the old pair stays up while its turns stream"
    assert router.choose(world.hub.snapshot(), world.cfg) is None, "a new turn is held, not sent to the old pair"
    world.hub.note_request_end(TP)
    v = world.tick(seconds=60)
    assert v["state"] == "finishing" and "1 turn(s)" in v["detail"]
    hold = v["boxes"]["box-a"]["hold"]
    assert hold["waiting_s"] >= 60 and hold["deadline_s"] == world.cfg.cluster_finishing_limit_s
    world.hub.note_request_end(TP)
    v = world.tick()
    assert v["state"] == "restarting" and world.calls("down")


def test_a_turn_on_the_old_pair_that_never_ends_holds_the_restart_only_up_to_its_limit(world):
    world.executor.mem = {b: 125.0 for b in BOXES}
    world.run_to_running(max_num_seqs=8, max_model_len=262144)
    world.hub.note_request_start(TP)
    world.store.set(enabled=True, restart=True, num_speculative_tokens=3)
    assert world.tick(seconds=world.cfg.cluster_finishing_limit_s)["state"] == "finishing"
    v = world.tick()
    assert v["state"] == "restarting" and world.calls("down")


def test_a_restart_at_new_knobs_moves_effective_only_once_the_new_pair_runs(world):
    """The panel's "apply & restart". While generation 1 is being replaced it is still what runs, so effective
    says generation 1 — never the settings asked for next; then nothing while no pair runs; then generation
    2's, from its own launch."""
    from queue_workflows import node_queue, run_store

    # A 16,384 prefill budget is 8.8 GiB more than the calibrated one, and a start must clear the
    # headroom on top: these boxes hold it, so the test stays about the KNOBS reaching both ranks.
    world.executor.mem = {b: 125.0 for b in BOXES}
    world.run_to_running(max_num_seqs=8, max_model_len=262144)
    old_runs = [world.run_id(b) for b in BOXES]
    world.store.set(enabled=True, restart=True, max_num_batched_tokens=16384, num_speculative_tokens=3)
    v = world.tick()
    assert v["state"] == "restarting" and v["desired"]["generation"] == 2
    assert v["effective"] == {"model_choice": FLASH, "generation": 1, "max_num_seqs": 8, "max_model_len": 262144,
                              "max_num_batched_tokens": _tuned("max_num_batched_tokens"), "num_speculative_tokens": 4, "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                              "long_prefill_token_threshold": 0, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}
    seen = []
    for _ in range(8):                                   # hold the boxes, then let generation 1's go
        v = world.tick()
        seen.append(v["effective"])
        if {run_store.get_run(r)["status"] for r in old_runs} == {"cancelled"}:
            break
    assert seen and all(e is None for e in seen), f"effective while no pair ran: {seen}"
    for run in old_runs:                                 # generation 1's reservation bodies see the cancel …
        for job in node_queue.list_jobs_for_run(run):
            if job["status"] == "running":
                node_queue.mark_completed(job["id"], context_delta={}, seconds=1.0)
    world.claim()                                        # … and the lanes take generation 2's
    world.settle(until="starting")
    v = world.tick()
    assert v["state"] == "running", v["detail"]
    assert world.executor.knobs == {b: {"max_num_batched_tokens": 16384, "num_speculative_tokens": 3,
                                        "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                                        "uncensored": True,
                                        "long_prefill_token_threshold": None}
                                    for b in BOXES}
    assert v["effective"] == {"model_choice": FLASH, "generation": 2, "max_num_seqs": 8, "max_model_len": 262144,
                              "max_num_batched_tokens": 16384, "num_speculative_tokens": 3,
                              "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                              "long_prefill_token_threshold": 0, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}


def test_a_restarted_lb_says_what_the_running_pair_was_launched_with_not_what_it_would_launch_now(world, tmp_path):
    """An LB redeploy leaves a serving pair running. If the box's default changed in that redeploy, the pair
    still runs what it was launched with — so effective comes from the launch record the LB keeps beside the
    switch, not from resolving the defaults again."""
    world.run_to_running(max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=2048)
    record = json.loads((tmp_path / "cluster-launch.json").read_text())
    assert (record["generation"], record["max_num_batched_tokens"], record["num_speculative_tokens"]) == (1, 2048, 4)
    world.executor.env = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "16384"}      # the redeploy changed the default …
    world.boot()                                                          # … and restarted the LB
    v = world.tick()
    assert v["effective"]["max_num_batched_tokens"] == 2048, "effective re-resolved a default instead of the launch"
    # And the NEXT launch takes the MODEL's own measured budget — which the box's new default cannot move,
    # since a number from the environment does not know which model is being launched.
    assert v["defaults"]["max_num_batched_tokens"] == _tuned("max_num_batched_tokens") != 16384


def test_a_pair_launched_before_the_lb_kept_a_record_is_resolved_from_its_own_switch(world, tmp_path):
    """Generation 52 was launched before this record existed: its switch (frozen while it runs) resolved the
    way the launch resolved it is what it runs — the same 8 / 262,144 / its own 4,096 / 4."""
    world.run_to_running(max_num_seqs=8, max_model_len=262144)
    (tmp_path / "cluster-launch.json").unlink()
    world.boot()
    assert world.tick()["effective"] == {"model_choice": FLASH, "generation": 1, "max_num_seqs": 8,
                                         "max_model_len": 262144,
                                         "max_num_batched_tokens": _tuned("max_num_batched_tokens"),
                                         "num_speculative_tokens": 4, "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                                         "long_prefill_token_threshold": 0, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}


def test_the_view_publishes_what_a_start_of_every_picker_row_launches(world):
    """THE PICKER ROW'S FACTS (operator, 2026-10-09: "make select more detailed"). For every model and every
    twin, the launch a Start of that row resolves to when nothing else is chosen — the SAME resolve as
    `cluster.defaults` (`DesiredStore.defaults_for` + `resolved_launch`), so a row and the Start it offers
    can never say two things. Keyed as the picker's rows are: `<id>`, `<id>@uncensored`."""
    v = world.tick()
    twins = {m["id"] for m in v["models"] if m.get("uncensored")}
    assert twins, "the shipped catalog names twins"
    assert set(v["setups"]) == {m["id"] for m in v["models"]} | {f"{t}@uncensored" for t in twins}
    # The stored model's rows: its default checkpoint's row IS cluster.defaults; the other differs only there.
    assert v["defaults"]["uncensored"] is True
    assert v["setups"][f"{FLASH}@uncensored"] == dict(v["defaults"], model_choice=FLASH)
    assert v["setups"][FLASH] == dict(v["defaults"], model_choice=FLASH, uncensored=False)
    # Another model's row is its OWN resolve — what cluster.defaults says once it is the chosen one — and no
    # row moves when the choice does.
    world.store.set(enabled=False, model_choice=DENSE)
    after = world.tick()
    assert after["setups"] == v["setups"]
    assert after["setups"][DENSE] == dict(after["defaults"], model_choice=DENSE, uncensored=False)
    assert after["setups"][DENSE]["kv_cache_memory"] == 48_000_000_000


def test_the_view_publishes_the_defaults_a_knob_left_at_null_resolves_to(world):
    v = world.tick()
    v = world.tick()
    assert v["defaults"] == {"max_num_seqs": world.store.defaults.max_num_seqs, "max_model_len": 262144,
                             "max_num_batched_tokens": _tuned("max_num_batched_tokens"), "num_speculative_tokens": 4, "ple_replicate": _ple(), "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                             "long_prefill_token_threshold": 0, "uncensored": True, "kv_cache_dtype": "fp8_e4m3"}
    assert (v["defaults"]["max_num_seqs"], v["defaults"]["max_num_batched_tokens"]) == (
        _tuned("max_num_seqs"), _tuned("max_num_batched_tokens")), "the chosen model's own measured-best"
    world.store.set(enabled=False, model_choice=DENSE)
    dense = world.tick()["defaults"]
    assert (dense["num_speculative_tokens"], dense["ple_replicate"], dense["kv_cache_memory"]) == (
        4, None, 48_000_000_000), "the dense model's own: its drafter, no PLE table, its own pool"
    # NOTHING BENCHED FOR THE DENSE 27B: its row still states the seats it is offered (6, the figure the gateway's
    # former default gave it, with that source) and no budget, so the box's stands.
    assert _tuned("max_num_batched_tokens", DENSE) is None
    assert (dense["max_num_seqs"], dense["max_num_batched_tokens"]) == (_tuned("max_num_seqs", DENSE), 8192) == (6, 8192)


def test_the_launch_line_says_the_settings_it_launched_with(world, caplog):
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144, num_speculative_tokens=0)
    world.tick()
    world.claim()
    with caplog.at_level(logging.INFO, logger="vllm_lb.cluster_loop"):
        world.settle(until="starting")
    (line,) = [r.getMessage() for r in caplog.records if "up box-a:" in r.getMessage()]
    assert f"8 seqs, 262144 tokens, prefill budget {_tuned('max_num_batched_tokens')}, 0 speculative tokens" \
        in line, line


def test_speculation_off_frees_the_drafters_memory_in_the_published_estimate(world):
    world.store.set(enabled=False, num_speculative_tokens=0)
    off = world.tick()["boxes"]["box-a"]["memory"]
    world.store.set(enabled=False, num_speculative_tokens=None)
    on = world.tick()["boxes"]["box-a"]["memory"]
    assert (off["mtp_gib"], on["mtp_gib"]) == (0.0, 0.7)


def test_every_start_is_one_line_in_the_boot_log_under_the_launch_record(world, tmp_path):
    """The boot log (perflog.BootRecorder) is fed by the loop itself: a start whose rank dies is one `failed`
    record with the planner's own words and the phase the head's log had reached; a start brought to serving
    is one `serving` record — both keyed exactly like the perf windows (`cluster.effective`)."""
    from vllm_lb.perflog import BootRecorder
    world.loop.boots = BootRecorder(str(tmp_path / "perf"))
    lines = lambda: [json.loads(line) for f in sorted((tmp_path / "perf").glob("boots-*.jsonl"))  # noqa: E731
                     for line in f.read_text().splitlines()]

    world.store.set(enabled=True, max_num_seqs=4, max_model_len=131072)
    world.tick()
    world.claim()
    world.executor.log_text["box-a"] = log_at(HEAD, "20:04:25.3")
    world.settle(until="starting")
    assert lines() == [] and world.loop.boots.open["states"]["reserving"] == 0.0
    world.executor.ranks["box-b"] = {**world.executor.ranks["box-b"], "running": False, "exit_code": 1}
    v = world.settle(until="failed", ticks=4)
    (failed,) = lines()
    assert failed["outcome"] == "failed" and failed["generation"] == 1 and failed["config_source"] == "launch"
    assert failed["config"]["max_num_seqs"] == 4 and failed["config"]["max_model_len"] == 131072
    assert failed["failure"] == v["detail"] and "rank on box-b exited (code 1)" in failed["failure"]
    assert failed["failed_in"] == "starting" and failed["failed_phase"] == "loading MTP draft weights"
    assert [p["phase"] for p in failed["phases"]] == ["loading MTP draft weights"]
    assert set(failed["states"]) >= {"reserving", "starting", "failed"}

    world.store.set(enabled=False)                       # Stop, then Start again at other settings
    for _ in range(6):
        if world.tick()["state"] == "off":
            break
        world.bodies_return()
    assert world.view()["state"] == "off" and len(lines()) == 1, "a Stop after a failure is not a start"
    world.executor.ranks = {b: None for b in BOXES}
    # The budget is given explicitly, and it is the one the fake head's log states it runs: `effective` takes
    # the ENGINE's own word for that knob, so a launch at the model's measured 4,096 against a head printing
    # 8,192 would differ from the record on purpose (the LB warns, and that is its own test above).
    world.run_to_running(max_num_seqs=8, max_model_len=131072, max_num_batched_tokens=8192)
    failed, ok = lines()
    assert ok["outcome"] == "serving" and ok["generation"] == 2 and ok["config_source"] == "launch"
    assert ok["config"] == world.view()["effective"], "keyed like the perf windows: the launch's resolved values"
    assert set(ok["states"]) >= {"reserving", "starting", "running"} and ok["states"]["reserving"] == 0.0
    assert ok["total_s"] == ok["states"]["running"] == round(ok["t_end"] - ok["t_start"], 1) > 0
    assert ok["failure"] is None


# ── the prefill budget is part of what a start needs ──────────────────────────
# 2026-09-30: the pair ran at max_num_batched_tokens 16384 down to 0.8 GiB MemAvailable and both GB10
# boxes hung. The kernel log is NVRM `Out of memory [NV_ERR_NO_MEMORY]` — the CUDA allocator on a box
# whose CPU and GPU share one pool — recurring all evening and last at 20:46-20:47Z, the minute
# generation 80 began. The estimate the planner gates on never saw that knob, so `batched 8192` and
# `batched 16384` asked the box for the same 103.2 GiB (lb/tests/fixtures/
# cluster-b16384-2026-09-30-memory.json holds every number).

#: What box-a and box-b really had free when generation 52 launched on them
#: (worklog/fixtures/gateway-knobs/lb-status-live-pre-knobs.json, a real /lb/status capture).
REAL_MEM = {"box-a": 115.7, "box-b": 116.6}


def test_a_prefill_budget_that_will_not_fit_is_refused_before_either_rank_launches(world):
    """THE ASK: the cluster is not loaded when it would exceed the memory the boxes have. On the
    memory they really had, 16,384 must never reach `up`."""
    world.executor.mem = dict(REAL_MEM)
    # As measured on 2026-09-30: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=16384,
                    ple_replicate=True)
    world.tick()
    world.claim()
    for _ in range(4):                       # long enough for L3's consecutive samples to be in
        v = world.tick()
    assert not world.calls("up"), f"launched 16,384 onto {REAL_MEM}: {v['detail']}"
    assert v["state"] == "freeing", v

    est = v["boxes"]["box-a"]["memory"]
    assert est["max_num_batched_tokens"] == 16384
    assert est["activation_gib"] > 0 and est["needed_now_gib"] > REAL_MEM["box-a"]
    # The refusal reaches the PANEL's box rows …
    assert {"no_headroom", "budget"} <= {w["code"] for w in est["warnings"]}, est["warnings"]
    # … and the cluster state line, naming the knob the operator can act on.
    assert "batched 16,384" in v["detail"], v["detail"]
    assert f"{est['needed_now_gib']:.0f}" in v["detail"], v["detail"]


def test_the_very_same_boxes_start_at_the_calibrated_budget(world):
    """The other half of the ask: the configuration that has been serving for weeks — seqs 8, batched
    8192 — still starts on exactly the memory that refuses 16,384. A guard that stopped this one would
    be a worse bug than the one it fixes."""
    world.executor.mem = dict(REAL_MEM)
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=8192)
    world.tick()
    world.claim()
    v = world.settle(until="starting")
    assert len(world.calls("up")) == 2, v["detail"]
    est = v["boxes"]["box-a"]["memory"]
    assert est["activation_gib"] == 0.0, "the calibrated budget costs nothing extra"
    assert est["needed_now_gib"] <= REAL_MEM["box-a"]


def test_a_pair_already_serving_at_16384_is_never_taken_down_by_the_new_gate(world):
    """40b34b2's property, kept: the memory floor guards a START, never a serving pair. Generations 68
    and 79 reached `serving` at 16,384 and ran on down to 0.8 GiB free — and a verdict that took THEM
    down would have been right about the memory and wrong about the pair: they were answering.

    The boxes start with more free than the new gate asks for, because that is the only way a 16,384
    pair gets up at all now — which is the point of the fix, and is why this test is about what happens
    AFTER it is up."""
    world.executor.mem = {"box-a": 125.0, "box-b": 125.0}
    world.run_to_running(max_num_seqs=16, max_num_batched_tokens=16384)
    world.executor.mem = {"box-a": 0.8, "box-b": 7.2}          # what the boxes really read there
    for _ in range(4):
        v = world.tick()
    assert v["state"] == "running" and v["held"] is True, v["detail"]
    assert not world.calls("down") and not world.calls("halt"), "a serving pair was stopped"
    est = v["boxes"]["box-a"]["memory"]
    assert not {"gate", "no_headroom", "budget", "floor", "near_floor"} & {w["code"] for w in est["warnings"]}


# ── the CONCURRENCY is part of what a start needs ──────────────────────────────
# 2026-10-01, measured on the live pair at batched 8192: 16 seats rests the head at 11.3 GiB free, 24
# seats rests it at 1.7 GiB with NO traffic at all — inside the 0.8-4.2 GiB band that hung both GB10 boxes
# the night before, and with nothing left for the generation handover that is the minute they died. The
# KV pool is pinned and allocated up front, so that is the configuration's RESTING cost, not a load
# effect (lb/tests/fixtures/cluster-seqs24-2026-10-01-memory.json).


def test_a_concurrency_that_will_not_fit_is_refused_before_either_rank_launches(world):
    """The same ask as the prefill budget's, for the other knob: the pair is not loaded at a
    concurrency the boxes cannot rest at. On the memory they really have, 24 seats must never reach
    `up`."""
    world.executor.mem = dict(REAL_MEM)
    # As measured on 2026-10-01: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=24, max_model_len=262144, max_num_batched_tokens=8192,
                    ple_replicate=True)
    world.tick()
    world.claim()
    for _ in range(4):                       # long enough for L3's consecutive samples to be in
        v = world.tick()
    assert not world.calls("up"), f"launched 24 seats onto {REAL_MEM}: {v['detail']}"
    assert v["state"] == "freeing", v

    est = v["boxes"]["box-a"]["memory"]
    assert est["max_num_seqs"] == 24 and est["max_num_batched_tokens"] == 8192
    assert est["concurrency_gib"] > 0 and est["needed_now_gib"] > REAL_MEM["box-a"]
    # The refusal reaches the PANEL's box rows …
    assert {"no_headroom", "concurrency"} <= {w["code"] for w in est["warnings"]}, est["warnings"]
    # … and the cluster state line, naming the dropdown the operator can act on.
    assert "seqs 24" in v["detail"], v["detail"]
    assert f"{est['needed_now_gib']:.1f}" in v["detail"], v["detail"]


def test_the_very_same_boxes_start_at_the_concurrency_that_has_banked_runs(world):
    """The other half: 16 seats — 11.3 GiB free measured, generation after generation of it — still
    starts on exactly the memory that refuses 24. A guard that stopped this one would be the worse bug."""
    world.executor.mem = dict(REAL_MEM)
    world.store.set(enabled=True, max_num_seqs=16, max_model_len=262144, max_num_batched_tokens=8192)
    world.tick()
    world.claim()
    v = world.settle(until="starting")
    assert len(world.calls("up")) == 2, v["detail"]
    est = v["boxes"]["box-a"]["memory"]
    assert est["concurrency_gib"] == 0.0, "the anchor concurrency costs nothing extra"
    assert est["needed_now_gib"] <= REAL_MEM["box-a"]


def test_a_pair_already_serving_at_24_sequences_is_never_taken_down_by_the_new_gate(world):
    """40b34b2's property again, for the concurrency term: generation 90 reached `serving` at 24 seats
    and rested at 1.7 GiB free. A verdict that took it down would be right about the memory and wrong
    about the pair — it was answering."""
    world.executor.mem = {"box-a": 125.0, "box-b": 125.0}
    world.run_to_running(max_num_seqs=24, max_num_batched_tokens=8192)
    world.executor.mem = {"box-a": 1.7, "box-b": 7.2}           # what the box really read there
    for _ in range(4):
        v = world.tick()
    assert v["state"] == "running" and v["held"] is True, v["detail"]
    assert not world.calls("down") and not world.calls("halt"), "a serving pair was stopped"
    est = v["boxes"]["box-a"]["memory"]
    codes = {w["code"] for w in est["warnings"]}
    assert not {"gate", "no_headroom", "budget", "concurrency", "floor", "near_floor"} & codes, codes


def test_the_memory_a_rank_took_is_what_the_box_owes_back_before_the_next_one_launches(world):
    """THE HANDOVER, the moment 2026-09-30's fatal NVRM `Out of memory [NV_ERR_NO_MEMORY]` landed in
    (20:46-20:47Z, one 16,384 generation ending as the next began). `mem_at_launch` used to be dropped
    the instant a rank came down, which threw away the only evidence the LB has that the box has
    finished letting go. It is kept as what the box owes back."""
    world.run_to_running(max_num_seqs=8, max_model_len=262144)
    assert world.view()["boxes"]["box-a"]["mem_at_launch"] == 115.0
    assert world.view()["boxes"]["box-a"]["mem_release_target"] is None

    world.store.set(enabled=False)                       # Stop: both ranks come down
    world.executor.mem = {"box-a": 20.0, "box-b": 20.0}  # …and the driver has not given the pool back
    v = world.tick()
    assert world.calls("down"), "the ranks were never taken down"
    v = world.tick()
    box_a = v["boxes"]["box-a"]
    assert box_a["mem_release_target"] == 115.0, box_a
    assert box_a["mem_at_launch"] is None, "the launch figure belongs to a rank that is gone"


# ── the KV pool at `auto` ─────────────────────────────────────────────────────

def _load_fixture() -> dict:
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent / "fixtures" / "cluster-load-2026-10-01-memory.json")
                      .read_text())


def _per_box(figures: dict) -> dict:
    """A fixture's per-box figures without its notes."""
    return {b: figures[b] for b in BOXES}


def test_a_restart_at_auto_launches_the_number_it_resolved_and_effective_and_the_perflog_record_that_number(world):
    """End to end, on generation 99's own figures: the pair rests, load takes its share, a restart at `auto` is
    sized from what the loop published — and the NUMBER, never the word, is what both ranks are sent, what
    `cluster.effective` says and what the performance log keys the new minutes by, so a 35 GB pool's minutes and
    a 36 GB pool's are never one configuration."""
    from queue_workflows import node_queue, run_store
    from vllm_lb.cluster import KV_AUTO
    from vllm_lb.cluster_defaults import kv_auto_resolver
    from vllm_lb.perflog import effective_config

    g = _load_fixture()["generation_99"]
    c = g["config"]
    world.store.set(enabled=True, max_num_seqs=c["max_num_seqs"], max_model_len=c["max_model_len"],
                    max_num_batched_tokens=c["max_num_batched_tokens"], ple_replicate=c["ple_replicate"],
                    kv_cache_memory=c["kv_cache_memory"])
    assert world.tick()["state"] == "reserving"
    world.claim()
    world.settle(until="starting")
    world.executor.mem = _per_box(g["at_rest_gib"])            # the ranks are up: what the boxes rest at …
    for _ in range(3):
        world.tick()
    world.prober.table[TP] = dict(LIVE_TP)
    v = world.tick()
    assert v["state"] == "running" and v["held"] is True
    assert {b: v["boxes"][b]["mem_rest_gib"] for b in BOXES} == _per_box(g["at_rest_gib"])
    world.executor.mem = _per_box(g["loaded_gib"])             # … and then load takes its share
    for _ in range(3):
        v = world.tick()
    assert {b: v["boxes"][b]["mem_stable_gib"] for b in BOXES} == _per_box(g["loaded_gib"])
    assert {b: v["boxes"][b]["mem_rest_gib"] for b in BOXES} == _per_box(g["at_rest_gib"]), "load never lowers the rest"
    before, _ = effective_config(v)
    old_runs = [world.run_id(b) for b in BOXES]

    world.store.set(enabled=True, restart=True, kv_cache_memory=KV_AUTO,
                    resolve_kv_auto=kv_auto_resolver(world.cfg, world.store.catalog, world.view))
    assert world.store.load().kv_cache_memory == 36_000_000_000
    world.executor.mem = {b: 115.0 for b in BOXES}         # the old ranks come down and give it back
    assert world.tick()["state"] == "restarting"
    for _ in range(8):
        world.tick()
        if {run_store.get_run(r)["status"] for r in old_runs} == {"cancelled"}:
            break
    for run in old_runs:
        for job in node_queue.list_jobs_for_run(run):
            if job["status"] == "running":
                node_queue.mark_completed(job["id"], context_delta={}, seconds=1.0)
    world.claim()
    world.settle(until="starting")
    v = world.tick()
    assert v["state"] == "running", v["detail"]
    assert {b: world.executor.knobs[b]["kv_cache_memory"] for b in BOXES} == {b: 36_000_000_000 for b in BOXES}
    assert (v["effective"]["generation"], v["effective"]["kv_cache_memory"]) == (2, 36_000_000_000)
    assert (v["desired"]["kv_cache_memory"], v["desired"]["kv_cache_auto"]) == (36_000_000_000, True)
    after, source = effective_config(v)
    assert (source, after["kv_cache_memory"], before["kv_cache_memory"]) == ("effective", 36_000_000_000,
                                                                            35_000_000_000)


def test_the_resting_figure_survives_an_lb_redeploy_under_a_serving_pair(world, tmp_path):
    """The LB is redeployed under serving pairs several times a day. A resting figure kept only in memory would
    come back as the LOADED one, and auto would charge the load twice — so it is kept with the launch record."""
    g = _load_fixture()["generation_99"]
    world.run_to_running(max_num_seqs=16, max_model_len=262144)
    rest = {b: v["mem_rest_gib"] for b, v in world.view()["boxes"].items()}
    assert rest == {b: 115.0 for b in BOXES}
    world.executor.mem = _per_box(g["loaded_gib"])
    world.boot()                                           # a redeployed LB, the pair still serving
    world.prober.table[TP] = dict(LIVE_TP)
    for _ in range(4):
        v = world.tick()
    assert v["state"] == "running"
    assert {b: v["boxes"][b]["mem_rest_gib"] for b in BOXES} == rest
    assert {b: v["boxes"][b]["mem_stable_gib"] for b in BOXES} == _per_box(g["loaded_gib"])


def test_the_actual_is_published_only_once_the_pair_serves_and_leaves_is_measured_from_the_launch(world):
    """UX audit 2026-10-02: `actual 19.6 GiB (+247 %) — correct the catalog's constants` was published
    while the weights were still loading, before the KV pool existed. The actual, and the deviation, exist
    once the head SERVES; and from then on what the box leaves is (free at launch − total), not (free with
    the pair loaded − total), which read `leaves -22.2 GiB` beside 45.8 GiB free."""
    world.executor.mem = {"box-a": 114.8, "box-b": 114.3}
    # The 2026-09-15 start, run 1, as it ran: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=2, max_model_len=262144, ple_replicate=True)
    world.tick()
    world.claim()
    world.settle(until="starting")
    world.executor.mem = {"box-a": 60.0, "box-b": 70.0}          # weights loading, no pool yet
    est = world.tick()["boxes"]["box-a"]["memory"]
    assert est["actual_used_gib"] is None and est["deviation_pct"] is None, est
    assert not [w for w in est["warnings"] if w["code"] == "deviation"], est["warnings"]
    assert est["launched_gib"] == 114.8
    assert est["leaves_gib"] == round(114.8 - est["total_gib"], 1), "from what it had at launch"
    world.prober.table[TP] = LIVE_TP
    world.executor.mem = {"box-a": 11.6, "box-b": 16.6}
    est = world.tick()["boxes"]["box-a"]["memory"]
    assert est["actual_used_gib"] == round(114.8 - 11.6, 1), "serving: what the box gave up since the launch"
    assert est["leaves_gib"] == round(114.8 - est["total_gib"], 1)
    assert est["leaves_gib"] > 0, "never the pair subtracted twice"


# ── the KV-pool sentence: ONE seat count, and it is the engine's ──────────────────────────────────
#
# generation 109 (2026-10-02, fixtures/qwen38-dflash2-start-2026-10-02-rank0.log), the dense 27B with a
# dflash drafter on its own 28 GB pin, read off box-a before it went down:
#
#   [interface.py:938]    Setting attention block size to 832 tokens to ensure that attention page size
#                         is >= mamba page size.
#   [kv_cache_utils:2032] GPU KV cache size: 629,682 tokens, Maximum concurrency for 262,144 tokens per
#                         request: 2.40x
#
# and the LB recorded it: perf-2026-10-02.jsonl carries "kv_pool_tokens": 629682.0,
# "kv_max_concurrency": 2.402 on all 19 windows of that generation. The page drew "fits 2 @ 262,144"
# from that pool and "KV pool holds 3 x 262,144-token requests" from the catalog's arithmetic over the
# same pin, in one sentence (worklog/gateway-figures-audit.md §2).

#: What the prober scrapes off the head's /metrics on that generation (`vllm:cache_config_info`,
#: probe.SCRAPE_INFO_LABELS → throughput.kv_pool). `blocks` is the engine's num_gpu_blocks as published.
GEN_109_POOL = {"kv_pool_tokens": 629_682, "kv_pool_bytes": 28_000_000_000,
                "kv_pool_blocks": 3286, "kv_block_tokens": 832}
DFLASH_SERVES = "Qwen/Qwen3.8-27B-DFlash2"


def test_the_published_clamp_is_the_seat_count_the_engine_itself_reported(world):
    """ONE pool, ONE seat count. The engine printed 2.40x for 262,144-token requests out of 629,682
    tokens; the page must not be told "holds 3" in the same breath as "fits 2".

    The catalog's arithmetic now ANSWERS 2 as well, because its block figure was corrected to the 832
    this profile's own start log prints (it had carried 864, read off a generation launched with a
    different prefill budget). That agreement is why this test asserts the engine's sentence rather than
    a disagreement: the figure moves with the launch, so the engine is authoritative whether or not the
    anchor happens to match today. `block_bytes` is still the MTP profile's, measured at its 48 GB pin."""
    world.loop.store = world.store          # ONE DesiredStore behind the switch and the loop, as `__main__`
    world.run_to_running(model_choice=DFLASH, serves=DFLASH_SERVES, max_num_seqs=8, max_model_len=262144)
    world.prober.table[TP] = {**LIVE_TP, "served_models": (DFLASH_SERVES,), "scrape": GEN_109_POOL}
    v = world.tick()

    assert v["held"] is True and v["effective"]["model_choice"] == DFLASH
    assert v["effective"]["kv_cache_memory"] == 28_000_000_000, "the dflash profile's own pin"
    assert "holds 2 x" in world.store.last_clamp, \
        "the arithmetic the write worked out — which the corrected block figure makes agree"
    assert v["clamp"] == ("KV pool holds 2 x 262,144-token requests: concurrency 8 runs 2 at a time "
                          "and queues the rest"), v["clamp"]


def test_the_clamp_falls_back_to_the_arithmetic_when_no_engine_has_measured_the_pool(world):
    """Before a start there is nothing to measure, and the arithmetic is all there is — it must still
    warn. The same selection, with the pair not yet up: the catalog's 2, at the 832-token block its own
    start log prints."""
    world.loop.store = world.store
    world.store.set(enabled=True, model_choice=DFLASH, max_num_seqs=8, max_model_len=262144)
    v = world.tick()
    assert v["held"] is False and v["effective"] is None
    assert "holds 2 x 262,144-token requests" in v["clamp"], v["clamp"]


def test_the_published_clamp_survives_an_lb_restart_under_a_serving_pair(world):
    """`last_clamp` was the last WRITE through one store object: an LB redeployed under a serving pair
    published "" and the operator lost the warning. The sentence is worked out from what is STORED."""
    world.run_to_running(model_choice=DFLASH, serves=DFLASH_SERVES, max_num_seqs=8, max_model_len=262144)
    world.prober.table[TP] = {**LIVE_TP, "served_models": (DFLASH_SERVES,), "scrape": GEN_109_POOL}
    world.tick()
    world.boot()                                          # a fresh LB, same disk, the pair still up
    v = world.tick()
    assert "holds 2 x 262,144-token requests" in v["clamp"], v["clamp"]


# ── a launch the LB itself refuses ────────────────────────────────────────────

REFUSAL = ("SGLang KV pin 20,000,000,000 B needs mem_fraction_static 0.851, above this profile's safe 0.85 ceiling "
           "(maximum 19,918,363,626 B)")


def refusing(world):
    """An executor whose launch BUILD refuses these settings — raised before anything reaches a box."""
    real = world.executor.up

    def up(host, **kw):
        world.executor.calls.append(("up", host.box, kw.get("rank"), kw.get("gen")))
        raise ValueError(REFUSAL)

    world.executor.up = up
    return real


def test_a_launch_the_lb_refuses_fails_the_start_with_its_reason_instead_of_retrying(world):
    """GENERATION 175, 2026-10-09: the stored KV pin was 0.4 % over the SGLang profile's safe ceiling, so building the
    launch raised ValueError on every tick — 566 refusals in 21 minutes while the page said "launching both ranks".
    A refusal raised before anything reaches a box is the same on every retry: the start FAILS, saying why, and is
    not launched again until the operator changes something and starts again."""
    refusing(world)
    world.store.set(enabled=True)
    world.tick()
    world.claim()
    views = [world.tick() for _ in range(12)]
    assert views[-1]["state"] == "failed", [(v["state"], v["detail"][:80]) for v in views]
    assert REFUSAL in views[-1]["detail"] and "refused to launch" in views[-1]["detail"], views[-1]["detail"]
    ups = len(world.calls("up"))
    assert 1 <= ups <= 2, f"one attempt per box, not one per tick: {ups}"
    for _ in range(5):
        world.tick()
    assert len(world.calls("up")) == ups, "a refused launch is never retried by itself"


TP_RANK_REFUSAL = ("tp-rank on box-a refused the launch: up: exit 2: kv_cache_dtype nvfp4 with a drafter is refused: "
                   "generations 214, 215 and 217 died on the drafter, and 219 (drafter on its own pool) served wrong "
                   "output — set num_speculative_tokens 0")


def test_tp_ranks_refusal_fails_the_start_with_its_words_instead_of_retrying(world):
    """GENERATION 221, 2026-10-22: tp-rank refused nvfp4 with a drafter on both ranks (exit 1) and the loop RETRIED the
    up every ~2 s while the page said "launching both ranks". The executor now raises tp-rank's refusal of the PLAN
    (`cluster_exec.LaunchRefused`): the start FAILS with tp-rank's own words and is never launched again by itself."""
    from vllm_lb.cluster_exec import LaunchRefused

    def up(host, **kw):
        world.executor.calls.append(("up", host.box, kw.get("rank"), kw.get("gen")))
        raise LaunchRefused(TP_RANK_REFUSAL)

    world.executor.up = up
    world.store.set(enabled=True)
    world.tick()
    world.claim()
    views = [world.tick() for _ in range(12)]
    assert views[-1]["state"] == "failed", [(v["state"], v["detail"][:80]) for v in views]
    assert TP_RANK_REFUSAL in views[-1]["detail"], views[-1]["detail"]
    ups = len(world.calls("up"))
    assert 1 <= ups <= 2, f"one attempt per box, not one per tick: {ups}"
    for _ in range(5):
        world.tick()
    assert len(world.calls("up")) == ups, "a refused launch is never retried by itself"


def test_a_transient_up_failure_is_still_retried(world):
    """A box that could not be reached (ssh exit 255) is not a refusal: the loop keeps launching, as before."""
    def up(host, **kw):
        world.executor.calls.append(("up", host.box, kw.get("rank"), kw.get("gen")))
        return False, "up: exit 255: ssh: connect to host box-a port 22: No route to host"

    world.executor.up = up
    world.store.set(enabled=True)
    world.tick()
    world.claim()
    views = [world.tick() for _ in range(12)]
    assert all(v["state"] != "failed" for v in views), [(v["state"], v["detail"][:80]) for v in views]
    assert len(world.calls("up")) > 2, f"retried on later ticks: {len(world.calls('up'))}"


def test_a_new_start_after_a_refusal_launches_again(world):
    """The refusal belongs to THAT start: the operator changes the setting and starts again, and it launches."""
    from queue_workflows import node_queue, run_store

    real = refusing(world)
    world.store.set(enabled=True)
    world.tick()
    world.claim()
    for _ in range(12):
        v = world.tick()
    assert v["state"] == "failed"
    old_runs = [world.run_id(b) for b in BOXES]
    world.executor.up = real                               # the operator changed the setting
    world.store.set(enabled=True, restart=True)
    v = world.tick()
    assert v["state"] != "failed" and "refused" not in v["detail"], v
    for _ in range(3):                                     # the new reservations asked, the old ones let go
        world.tick()
    for run_id in old_runs:                                # …and their node bodies return
        assert run_store.get_run(run_id)["status"] == "cancelled"
        for job in node_queue.list_jobs_for_run(run_id):
            if job["status"] == "running":
                node_queue.mark_completed(job["id"], context_delta={}, seconds=1.0)
    world.claim()
    world.settle(until="starting")
    assert any(len(c) > 4 for c in world.calls("up")), "the real executor launched the new generation"


# ── GENERATION 187 (2026-10-10 20:55-21:04Z): a start that LOOPED instead of refusing ────────────────────────────────
# The switch accepted a 24 GB pin on the SGLang Flash-Next row. `SshExecutor.up` turned the launch build's ValueError
# into `(False, "up: ValueError: …")` (cluster_exec: "never raise"), so the loop's `except ValueError` never saw it,
# `launch_refused` stayed "", and Up was planned on every fresh tick — 131 refusals in six minutes under `starting`,
# alternating with `freeing` whenever the box's lowest reading dipped 0.1 GiB under the need.

CEILING = ("SGLang KV pin 24,000,000,000 B needs mem_fraction_static 0.885, above this profile's safe 0.85 ceiling "
           "(maximum 19,918,363,626 B)")
AT_REST = {"date": None, "ps": {"models": []}}                  # the box's ollama, asked: nothing resident


def stored_switch(world, **fields) -> None:
    """A switch file as an older LB (or the operator's curl) left it — past the switch's own refusals."""
    path = world.cfg.cluster_state_path
    json.dump({"mode": "cluster", "generation": 1, "max_num_seqs": 16, "max_model_len": 262144, "start_id": "ab",
               **fields}, open(path, "w"))


def test_a_stored_pin_the_launch_cannot_build_fails_the_start_on_the_first_plan_with_no_launch_attempt(world):
    """The launch's own build check runs for the stored selection on every tick, so the FIRST plan of the generation
    fails with the launch's words — before a reservation is asked for, before the RAM gate, with no `up` at all."""
    stored_switch(world, model_choice=FLASH_SGLANG, uncensored=True, kv_cache_memory=24_000_000_000)
    v = world.tick()
    assert v["state"] == "failed", (v["state"], v["detail"])
    assert CEILING in v["detail"] and "refused to launch generation 1" in v["detail"], v["detail"]
    assert not world.calls("up") and world.reservation_jobs() == [], "nothing was launched, nothing reserved"
    states = {world.tick()["state"] for _ in range(12)}
    assert states == {"failed"}, f"a refused launch never re-enters freeing or starting: {states}"
    assert not world.calls("up")
    # The operator starts again at a pin the profile takes (what generation 188 did): a new generation launches.
    world.store.set(enabled=True, restart=True, kv_cache_memory=None)
    assert world.tick()["state"] == "reserving"
    world.claim()
    world.settle(until="starting")
    assert len(world.calls("up")) == 2


def test_a_need_the_box_at_rest_can_never_meet_fails_the_start_instead_of_waiting_on_it(world):
    """The other half of the flap: with nothing on box-a holding RAM and its reading steady, a need above what it has
    is not a wait with a budget — it is refused with both numbers, on the first plan that has its samples."""
    world.executor.mem = dict(REAL_MEM)
    world.executor.ollama = {b: dict(AT_REST) for b in BOXES}
    # As measured on 2026-09-30: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=16384,
                    ple_replicate=True)
    world.tick()
    world.claim()
    v = world.settle(until="failed")
    assert "box-a: 115.7 GiB free is the most it frees at rest" in v["detail"], v["detail"]
    assert "short" in v["detail"] and "batched 16,384" in v["detail"], v["detail"]
    assert not world.calls("up")
    for _ in range(4):
        v = world.tick()
    assert v["state"] == "failed" and not world.calls("up"), v["detail"]


def test_the_same_need_is_a_wait_while_the_box_has_not_said_what_holds_it(world):
    """`holds` None is "could not be asked", never "nothing": the wait stands, as before (and says so past budget)."""
    world.executor.mem = dict(REAL_MEM)
    # As measured on 2026-09-30: the PLE table replicated.
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=16384,
                    ple_replicate=True)
    world.tick()
    world.claim()
    for _ in range(4):
        v = world.tick()
    assert v["state"] == "freeing" and not world.calls("up"), (v["state"], v["detail"])


def test_the_launch_line_states_the_gates_own_need(world, caplog):
    """`up box-a: ok (115.6 GiB free, needs 101.9)` named the rank's cost while the gate had waited on 109.9 — the cost
    plus the start headroom. One need, in the gate and in the line a post-mortem reads."""
    caplog.set_level(logging.INFO, logger="vllm_lb.cluster_loop")
    world.store.set(enabled=True, max_num_seqs=8, max_model_len=262144)
    world.tick()
    world.claim()
    v = world.settle(until="starting")
    need = v["boxes"]["box-a"]["memory"]["needed_now_gib"]
    assert need > v["boxes"]["box-a"]["memory"]["total_gib"], "the headroom is part of the need"
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("[cluster] up box-a: ok"))
    assert f"needs {need:.1f}" in line, line
