"""A HUNG START IS RESCUED: declared hung on its own step's clock, its evidence saved, relaunched — and only once
the retries run out is it failed and halted as before.

THE FAILURE, MEASURED. The SGLang pair (qwen3.8-27b-sglang-dflash2, 32 seats) hung in "capturing CUDA graphs" on
2 of its 7 starts of 2026-10-02/03 — generations 122 and 131 — where a healthy start takes 124-132 s in all
(tests/fixtures/boots-2026-09-30-to-10-03.jsonl, a copy of lb/state/perf/boots-*.jsonl). Generation 131's own
evidence (tests/fixtures/sglang-hung-start-2026-10-03-gen131-*, cut from lb/state/start-hangs/gen131/): rank 0
blocked in a device→host copy inside FlashInfer's `plan` with the GPU idle, rank 1 waiting at a CPU barrier — a
cross-rank deadlock, and an intermittent one: the identical configuration relaunched came up (122 hung, 123 and
125 served). Until now the LB noticed only after 600 s of nothing new, failed the start and HELD both GB10 boxes until
the operator pressed Stop, then Start.

Pinned here:
  * the LIMIT is the step's own: 4 x the slowest healthy observation of that step in the boot log, never under
    400 s (4.1 x the slowest capture any engine's own log has timed: generation 188's 97.75 s; it was 180 s, 4.6 x
    the 39 s known before), never over the deployment's ceiling (600 s) — which still stands wherever there is no
    measured basis;
  * EVIDENCE comes first: both ranks' logs and stacks are saved under the LB's state dir before anything is torn
    down, bounded to the newest ten;
  * a HANG — only a hang — is relaunched, at the same generation and settings, up to `cluster_start_retries`
    times; then the old rule: failed, halted, logs kept. Never after the operator changed the switch;
  * the panel says so, a held turn keeps waiting through it, and the boot log keeps every hung attempt.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

import vllm_lb.cluster as cluster
from tests.test_cluster_hold import cluster as hold_cluster  # noqa: F401 — a fixture
from tests.test_cluster_hold import pair_up, post
from tests.test_cluster_loop import LIVE_TP, OFF, SGLANG, TP, World, clock_of, writes  # noqa: F401 — `writes` is a fixture
from tests.start_logs import log_at
from tests.test_cluster_plan import box, desired, exited, observed, of, run, running
from vllm_lb.cluster import Down, Halt, Up
from vllm_lb.cluster_startup import LoadPass, StartupPhase
from vllm_lb.perflog import BootRecorder
from vllm_lb.proxy import cluster_coming

FIXTURES = Path(__file__).with_name("fixtures")
#: Every cluster start the LB recorded from 2026-09-30 to 2026-10-03 (through generation 132), as it wrote them.
BOOTS = FIXTURES / "boots-2026-09-30-to-10-03.jsonl"
#: Generation 131's head and worker, as `docker logs` held them while the pair sat deadlocked, and what py-spy
#: said each rank's scheduler was doing.
HUNG_HEAD = "sglang-hung-start-2026-10-03-gen131-rank0.log"
HUNG_WORKER = "sglang-hung-start-2026-10-03-gen131-rank1.log"
STACKS = {"box-a": (FIXTURES / "sglang-hung-start-2026-10-03-gen131-rank0-pyspy.txt").read_text(),
          "box-b": (FIXTURES / "sglang-hung-start-2026-10-03-gen131-rank1-pyspy.txt").read_text()}
#: A healthy SGLang start of the same model (generation 121), whole — what a relaunch prints.
HEALTHY_HEAD = "sglang-lb-start-2026-10-03-rank0.log"
SGLANG_SERVES = "Qwen/Qwen3.8-27B-SGLang"
CAPTURING = StartupPhase.CAPTURING_CUDA_GRAPHS


# ── the limits, and the data they come from ───────────────────────────────────

def reads_checkpoint(phase: str) -> bool:
    """A step whose length is the STORAGE's: a LoadPass counting, or the step that opens the load — which on
    SGLang holds the whole read, its shard bar having no line end until the pass is done."""
    return phase.startswith("loading ") or phase == StartupPhase.INITIALIZING_MODEL


def healthy_steps() -> dict[str, list[float]]:
    """Every step of every start that reached serving: from the read that first named it to the read that
    first named the next one, or to serving — the most `start_quiet_s` could have reached in it, plus a read."""
    seen: dict[str, list[float]] = defaultdict(list)
    for line in BOOTS.read_text().splitlines():
        rec = json.loads(line)
        if rec["outcome"] != "serving":
            continue
        steps = rec["phases"]
        for i, step in enumerate(steps):
            end = steps[i + 1]["at_s"] if i + 1 < len(steps) else rec["total_s"]
            seen[step["phase"]].append(round(end - step["at_s"], 1))
    return seen


def test_every_measured_step_is_its_slowest_healthy_start_in_the_boot_log():
    """The table is the data, not a taste: re-derived here from the boot log it was read off, every step seen on
    at least START_STALL_MIN_SEEN healthy starts and not bounded by the storage. (`pytest -s` prints it.)"""
    seen = healthy_steps()
    expected = {phase: (max(v), len(v)) for phase, v in seen.items()
                if len(v) >= cluster.START_STALL_MIN_SEEN and not reads_checkpoint(phase)}
    for phase, (slowest, n) in sorted(expected.items(), key=lambda kv: -kv[1][0]):
        print(f"{phase:45s} {n:3d} healthy starts, slowest {slowest:5.1f} s -> "
              f"hung after {cluster.start_stall_limit(phase):5.1f} s")
    assert cluster.START_STEP_SLOWEST_S == expected


def test_a_steps_limit_is_four_times_its_slowest_healthy_start_never_under_400_s_never_over_600_s():
    assert cluster.start_stall_limit(CAPTURING) == 400.0, "4 x 16.8 s is 67 s: the floor"
    assert cluster.start_stall_limit(StartupPhase.AUTOTUNING) == 400.0, "4 x 83.8 s is 335 s: the floor"
    assert cluster.start_stall_limit(StartupPhase.STARTING_API_SERVER) == 400.0, "4 x 82.8 s is 331 s: the floor"
    assert cluster.start_stall_limit(StartupPhase.COMPILING_MODEL) == 400.0
    assert cluster.start_stall_limit(StartupPhase.AUTOTUNING, 900.0) == 400.0, "the floor stands under a higher ceiling"
    for phase, (slowest, _) in cluster.START_STEP_SLOWEST_S.items():
        assert cluster.start_stall_limit(phase) == min(600.0, max(400.0, 4 * slowest)), phase


@pytest.mark.parametrize("phase", ["loading model weights", "loading draft weights", "initializing model",
                                   "starting SGLang", "warming up", "", "its start"])
def test_a_step_with_no_measured_basis_keeps_the_600_s_rule(phase):
    """The checkpoint's reads (cold off the HDD the boot log never saw), steps seen on fewer than three healthy
    starts, and anything this LB has not named: the rule they had."""
    assert cluster.start_stall_limit(phase) == cluster.START_STALL_S == 600.0


def test_the_deployments_ceiling_caps_every_step_and_stands_where_nothing_was_measured():
    assert cluster.start_stall_limit(CAPTURING, 60.0) == 60.0
    assert cluster.start_stall_limit(StartupPhase.AUTOTUNING, 900.0) == cluster.START_STALL_FLOOR_S
    assert cluster.start_stall_limit("loading model weights", 900.0) == 900.0


def test_the_floor_clears_four_times_the_slowest_capture_any_engines_own_log_has_timed():
    """The boot log samples every 15 s, so it saw capture as at most 16.8 s; the engines time it themselves. The
    slowest any start in fixtures/ printed was vLLM generation 109's 39 s (180 s was 4.6 x that) until generation
    188 (2026-10-10, SGLang Flash-Next at 16 seats) timed its target-verify capture at 97.75 s — quiet 100 s in the
    boot log against that 180 s. The floor follows the data: 400 s is 4.1 x it."""
    timed = []
    for log in FIXTURES.glob("*.log"):
        text = log.read_bytes().decode("utf-8", "replace")
        timed += [float(s) for s in re.findall(r"Graph capturing finished in (\d+) secs", text)]
        timed += [float(s) for s in re.findall(r"CUDA graph end\. elapsed=([\d.]+) s", text)]
    assert max(timed) == 97.75
    assert cluster.START_STALL_FLOOR_S >= 4 * max(timed)


# ── the planner ───────────────────────────────────────────────────────────────

def hung(quiet: float, *, phase: str = CAPTURING, rescues: int = 0, box_a=None, box_b=None):
    a = box_a if box_a is not None else box("box-a", rank=running())
    b = box_b if box_b is not None else box("box-b", rank=running())
    return replace(observed(a, b, quiet=quiet, phase=phase), start_rescues=rescues)


def test_a_start_hung_in_graph_capture_is_relaunched_at_400_s_not_failed_at_600():
    p = run(desired(), hung(402.0))
    assert p.state == "restarting" and not p.held
    assert p.actions == [Down("box-a"), Down("box-b")], "torn down to relaunch — never halted for a retry"
    assert p.hang == cluster.StartHang(phase=CAPTURING, quiet_s=402.0, limit_s=400.0, attempt=1, retries=2,
                                       relaunch=True)
    assert "hung in “capturing CUDA graphs” for 402 s" in p.detail, p.detail
    assert "relaunching (retry 1/2)" in p.detail, p.detail


def test_the_second_hang_is_retry_two_and_the_third_is_halted_as_before_but_stalled_not_failed():
    """The protective behaviour is unchanged — after the retries the ranks are HALTED and keep their logs — but
    the word is: a start that ran out its clock in one step did not error (operator, 2026-10-10: "there must be
    real error, not timeout to show failed"). The line says what hung, for how long, what the LB did."""
    p = run(desired(), hung(410.0, rescues=1))
    assert p.state == "restarting" and "relaunching (retry 2/2)" in p.detail and p.hang.attempt == 2
    p = run(desired(), hung(410.0, rescues=2))
    assert p.state == "stalled" and p.actions == [Halt("box-a"), Halt("box-b")]
    assert p.hang == cluster.StartHang(phase=CAPTURING, quiet_s=410.0, limit_s=400.0, attempt=3, retries=2,
                                       relaunch=False)
    assert "hung in “capturing CUDA graphs” for 410 s" in p.detail and "after 2 relaunches" in p.detail, p.detail
    assert "halted" in p.detail and "logs" in p.detail and "failed" not in p.detail, p.detail
    assert p.detail.endswith("Stop, then Start again")


def test_with_no_retries_configured_a_hang_is_halted_as_before_and_stalled():
    p = run(desired(), hung(402.0), start_retries=0)
    assert p.state == "stalled" and p.actions == [Halt("box-a"), Halt("box-b")] and not p.hang.relaunch


def test_a_quiet_start_says_what_comes_next_as_a_threshold_never_as_a_thing_done():
    """"(relaunched at 600 s, retry 1/2)" was read as a relaunch that had HAPPENED — by the operator and the watchdog
    alike (generation 188, 2026-10-10: nothing had been relaunched). A quiet start states the threshold ahead and the
    retries left; "relaunched" is said only after one, and says how many."""
    p = run(desired(), hung(150.0))
    assert p.state == "starting" and p.actions == [] and p.hang is None
    assert ("no progress for 150 s in “capturing CUDA graphs” (relaunch at 400 s if still quiet · 2 retries left)"
            in p.detail), p.detail
    assert "relaunched" not in p.detail, p.detail
    p = run(desired(), hung(150.0, rescues=1))
    assert "(relaunched once already · relaunch at 400 s if still quiet · 1 retry left)" in p.detail, p.detail
    p = run(desired(), hung(150.0, rescues=2))
    assert "(relaunched twice already · halt at 400 s if still quiet · no retries left)" in p.detail, p.detail
    assert "fails" not in p.detail and "halted" not in p.detail, p.detail


@pytest.mark.parametrize("case", ["config error", "out of memory", "wrong model"])
def test_only_a_hang_is_relaunched(case):
    """A rank that EXITED (SGLang's exit 2 is a bad argument: generation 120), a box running out of memory under
    its rank, and a head serving another model all fail as before, retries or not: a relaunch would only
    repeat them."""
    if case == "config error":
        o = hung(402.0, box_a=box("box-a", rank=exited(code=2)))
    elif case == "out of memory":
        o = hung(402.0, box_b=box("box-b", rank=running(), mem=0.2, peak=0.2))
    else:
        o = replace(hung(402.0), head_live=True, head_serves=False, head_model="Qwen/Qwen3.8-27B")
    p = run(desired(), o)
    assert p.state == "failed" and p.hang is None and not of(p.actions, Down), p


def test_the_relaunch_is_the_same_generation_with_the_same_settings():
    """Torn down, both reservations still running for this generation: the planner's own launch, of what is
    stored — nothing asked of broker_parrot again, nothing re-resolved."""
    d = desired(seqs=32)
    p = run(d, replace(observed(box("box-a", rank=cluster.RankObs(False, False, None, None)),
                                box("box-b", rank=cluster.RankObs(False, False, None, None))), start_rescues=1))
    assert [(a.box, a.gen, a.max_num_seqs) for a in of(p.actions, Up)] == [("box-a", d.generation, 32),
                                                                          ("box-b", d.generation, 32)]


# ── the loop, replaying generation 131 ────────────────────────────────────────

_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2})")


def sglang_stamped(name: str) -> list[tuple[str, str]]:
    """(HH:MM:SS SGLang stamped it, the line) for every line of an SGLang log; a line SGLang does not stamp — a
    tqdm bar, a warning — carries the stamp before it, the first lines the first stamp."""
    lines = [line for line in (FIXTURES / name).read_bytes().decode("utf-8", "replace").split("\n") if line]
    stamp = next(m.group(1) for line in lines if (m := _STAMP.match(line)))
    out = []
    for line in lines:
        if m := _STAMP.match(line):
            stamp = m.group(1)
        out.append((stamp, line))
    return out


def gen131_at(when: str) -> str:
    """The head's log of generation 131 as `docker logs` held it at `when` (UTC HH:MM:SS, 2026-10-03)."""
    return "".join(f"{line}\n" for stamp, line in sglang_stamped(HUNG_HEAD) if stamp <= when)


def whole(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("utf-8", "replace")


@pytest.fixture
def world(broker_db, writes, tmp_path):  # noqa: F811 — `writes` is the fixture imported above
    w = World(tmp_path, writes)
    arm(w)
    return w


def arm(w: World) -> None:
    """What a booted LB has beside the loop: its boot log; and the fleet as generation 131 left it."""
    w.loop.boots = BootRecorder(str(w.cfg.cluster_state_path.rsplit("/", 1)[0] + "/perf"))
    w.executor.stacks_text = dict(STACKS)
    w.executor.log_text["box-b"] = whole(HUNG_WORKER)


def reboot(w: World) -> None:
    """A fresh LB process: same disk, same fleet, nothing kept in memory."""
    w.boot()
    w.loop.boots = BootRecorder(str(w.cfg.cluster_state_path.rsplit("/", 1)[0] + "/perf"))


def sglang_launched(w: World) -> int:
    """The operator's Start of the SGLang pair at 32 seats, through to both ranks launched. Its generation."""
    w.store.set(enabled=True, model_choice=SGLANG, max_num_seqs=32)
    w.tick()
    w.claim()
    w.settle(until="starting")
    return w.view()["desired"]["generation"]


def replay(w: World, start: str, end: str, *, until=None, log=gen131_at, step: int = 15) -> list[tuple[int, dict]]:
    """Tick every `step` s from `start` to `end`, the head's log as generation 131's grew; stop early once
    `until(view)` holds. (time, view) per tick."""
    out = []
    for t in range(clock_of(start), clock_of(end) + 1, step):
        w.executor.log_text["box-a"] = log(f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}")
        v = w.tick(seconds=step)
        out.append((t, v))
        if until is not None and until(v):
            break
    return out


def rescued(v: dict) -> bool:
    return v["state"] == "restarting" and "relaunching" in v["detail"]


def test_generation_131_is_relaunched_at_the_capture_limit_and_its_evidence_is_saved_first(world, tmp_path):
    gen = sglang_launched(world)
    timeline = replay(world, "06:26:15", "06:38:30", until=rescued)
    at, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    # The draft-verify capture began at 06:27:57; the first 15 s read to see it was 06:28:00. The planner judges
    # the quiet the LAST read measured, so the limit is acted on one read after it is reached: +400 to +420.
    assert clock_of("06:28:00") + 400 <= at <= clock_of("06:28:00") + 420, at
    assert "hung in “capturing CUDA graphs” for" in v["detail"] and "relaunching (retry 1/2)" in v["detail"]
    assert all(s["state"] != "failed" for _, s in timeline)

    # EVIDENCE BEFORE TEARDOWN: both ranks' logs and stacks were asked for before the first Down.
    calls = world.executor.calls
    first_down = next(i for i, c in enumerate(calls) if c[0] == "down")
    asked = {c[:2] for c in calls[:first_down] if c[0] in ("logs", "stacks")}
    assert {("logs", "box-a"), ("logs", "box-b"), ("stacks", "box-a"), ("stacks", "box-b")} <= asked
    where = tmp_path / "start-hangs" / f"gen{gen}-attempt1"
    assert (where / "box-a-stacks.txt").read_text() == STACKS["box-a"]
    assert "plan (flashinfer/prefill.py:2340)" in (where / "box-a-stacks.txt").read_text()
    assert "barrier (torch/distributed/distributed_c10d.py" in (where / "box-b-stacks.txt").read_text()
    # (bytes: a text-mode read would turn tqdm's carriage returns into line ends)
    assert (where / "box-a.log").read_bytes().decode() == gen131_at("06:31:30")
    assert (where / "box-b.log").read_bytes().decode() == whole(HUNG_WORKER)
    meta = json.loads((where / "hang.json").read_text())
    assert (meta["generation"], meta["attempt"], meta["retries"], meta["relaunch"]) == (gen, 1, 2, True)
    assert meta["phase"] == CAPTURING and meta["limit_s"] == 400.0 and meta["quiet_s"] >= 400.0
    assert meta["config"]["model_choice"] == SGLANG and meta["config"]["max_num_seqs"] == 32
    assert v["rescue"]["evidence"] == f"start-hangs/gen{gen}-attempt1" and v["rescue"]["attempt"] == 1

    # TORN DOWN, not halted: the evidence is on the LB's disk now.
    assert sorted(c[1] for c in world.calls("down")) == ["box-a", "box-b"] and not world.calls("halt")

    # RELAUNCHED by itself: the same generation, the same settings, every view still "on its way".
    world.executor.log_text["box-a"] = whole(HEALTHY_HEAD)
    views = [world.tick() for _ in range(8)]
    ups = world.calls("up")
    assert len(ups) == 4 and ups[2:] == ups[:2], ups
    assert world.executor.knobs["box-a"]["max_num_batched_tokens"] is None
    assert all(cluster_coming(s) for s in views) and all(s["state"] != "failed" for s in views)
    relaunching = [s["detail"] for s in views if s["state"] == "starting"]
    assert relaunching and all("retry 1/2" in d and CAPTURING in d for d in relaunching), relaunching

    world.prober.table[TP] = {**LIVE_TP, "served_models": (SGLANG_SERVES,)}
    v = world.tick()
    assert v["state"] == "running" and v["held"], v["detail"]
    assert v["rescue"]["attempt"] == 1, "the record of what brought it back stays with the pair"
    assert v["rescue"]["served"] is True, "…as history: a later restart of this generation is not that rescue"


def test_after_two_relaunches_a_third_hang_is_stalled_and_halted_with_its_logs_kept(world, tmp_path):
    """The same deadlock every time: rescued twice, then the halt — both ranks HALTED (never removed), the
    reason on the page, and every attempt in the boot log so the hang rate stays measurable. The word on the
    page is `stalled`, not `failed`: a start that ran out its clock is not an error (operator, 2026-10-10) —
    and it STAYS `stalled` on the ticks after, when the halted ranks are seen exited (137): the cause outranks
    its consequences, as it always did for the words."""
    gen = sglang_launched(world)
    timeline = replay(world, "06:26:15", "06:58:00", until=lambda v: v["state"] == "stalled")
    v = timeline[-1][1]
    assert v["state"] == "stalled", f"{v['state']} — {v['detail']}"
    assert "after 2 relaunches" in v["detail"] and CAPTURING in v["detail"], v["detail"]
    assert "hung in" in v["detail"] and "halted" in v["detail"] and "failed" not in v["detail"], v["detail"]
    assert all(s["state"] != "failed" for _, s in timeline), "a hang is never FAILED on any tick"
    assert len(world.calls("down")) == 4, "two rescues, both ranks each"
    assert sorted(c[1] for c in world.calls("halt")) == ["box-a", "box-b"]
    assert all(world.executor.ranks[b]["exit_code"] == 137 for b in ("box-a", "box-b")), "kept, with their logs"
    assert not cluster_coming(v), "now it needs the operator"
    launches = len(world.calls("up"))
    for _ in range(3):                                   # the halted ranks are seen exited: still the stall
        v = world.tick(seconds=15)
        assert v["state"] == "stalled" and "hung in" in v["detail"], f"{v['state']} — {v['detail']}"
    reboot(world)                                        # …and a redeployed LB still knows it was a hang
    v = world.tick(seconds=15)
    assert v["state"] == "stalled" and "hung in" in v["detail"] and "137" not in v["detail"], f"{v['state']} — {v['detail']}"
    assert len(world.calls("up")) == launches, "nothing relaunched: the halt is the operator's to lift"
    hangs = tmp_path / "start-hangs"
    assert sorted(p.name for p in hangs.iterdir()) == [f"gen{gen}-attempt{k}" for k in (1, 2, 3)]
    assert json.loads((hangs / f"gen{gen}-attempt3" / "hang.json").read_text())["relaunch"] is False

    boots = [json.loads(line) for f in sorted((tmp_path / "perf").glob("boots-*.jsonl"))
             for line in f.read_text().splitlines()]
    assert [(b["generation"], b["attempt"], b["outcome"]) for b in boots] == [(gen, 1, "failed"), (gen, 2, "failed"),
                                                                             (gen, 3, "failed")]
    assert [b["hang"]["relaunch"] for b in boots] == [True, True, False]
    assert all(b["failed_phase"] == CAPTURING and b["hang"]["phase"] == CAPTURING for b in boots)
    assert boots[0]["hang"]["evidence"] == f"start-hangs/gen{gen}-attempt1"


def test_a_stop_pressed_after_a_rescue_is_never_relaunched(world):
    sglang_launched(world)
    assert rescued(replay(world, "06:26:15", "06:38:30", until=rescued)[-1][1]), "no rescue to stop"
    launches = len(world.calls("up"))
    world.store.set(enabled=False)
    for _ in range(4):
        world.tick()
        world.bodies_return()
    assert world.tick()["state"] == "off"
    assert len(world.calls("up")) == launches, "the operator changed the switch: nothing is relaunched"


def test_a_restart_pressed_after_a_rescue_is_a_new_start_with_a_fresh_budget(world):
    from queue_workflows import node_queue, run_store

    gen = sglang_launched(world)
    old_runs = [world.run_id(b) for b in ("box-a", "box-b")]
    replay(world, "06:26:15", "06:38:30", until=rescued)
    world.store.set(enabled=True, restart=True)
    world.executor.log_text["box-a"] = ""
    for _ in range(4):                                     # the new reservations asked, the old ones let go
        world.tick()
    for run_id in old_runs:                                # …and their node bodies return
        assert run_store.get_run(run_id)["status"] == "cancelled"
        for job in node_queue.list_jobs_for_run(run_id):
            if job["status"] == "running":
                node_queue.mark_completed(job["id"], context_delta={}, seconds=1.0)
    world.claim()
    world.settle(until="starting")
    assert {c[3] for c in world.calls("up")} == {gen, gen + 1} and len(world.calls("up")) == 4, world.calls("up")
    assert world.view()["rescue"] is None, "the rescue belonged to the start the operator replaced"


def test_the_relaunch_budget_survives_an_lb_restart(world):
    """Bounded means bounded: a redeployed LB under a rescued start remembers the retry it is on."""
    gen = sglang_launched(world)
    replay(world, "06:26:15", "06:38:30", until=rescued)
    reboot(world)
    world.executor.log_text["box-a"] = gen131_at("06:38:30")
    v = world.tick()
    assert v["rescue"] is not None and v["rescue"]["attempt"] == 1, v.get("rescue")
    timeline = replay(world, "06:40:00", "06:50:00", until=rescued, log=lambda when: gen131_at("06:38:30"))
    assert "relaunching (retry 2/2)" in timeline[-1][1]["detail"], timeline[-1][1]["detail"]
    assert (Path(world.cfg.cluster_state_path).with_name("start-hangs") / f"gen{gen}-attempt2").is_dir()


def test_hang_evidence_keeps_the_newest_ten_and_never_touches_a_folder_it_did_not_write(world, tmp_path):
    root = tmp_path / "start-hangs"
    old = time.time() - 86400
    for g in range(1001, 1013):
        (root / f"gen{g}-attempt1").mkdir(parents=True)
        os.utime(root / f"gen{g}-attempt1", (old + g, old + g))
    (root / "gen131").mkdir()
    (root / "gen131" / "box-a.log").write_text("the operator's own capture\n")
    gen = sglang_launched(world)
    replay(world, "06:26:15", "06:38:30", until=rescued)
    names = sorted(p.name for p in root.iterdir())
    ours = [n for n in names if re.fullmatch(r"gen\d+-attempt\d+", n)]
    assert len(ours) == 10 and f"gen{gen}-attempt1" in ours, names
    assert "gen1001-attempt1" not in ours and "gen1002-attempt1" not in ours and "gen1003-attempt1" not in ours
    assert (root / "gen131" / "box-a.log").read_text() == "the operator's own capture\n"


# ── a turn held for the pair waits through the rescue ─────────────────────────

def test_a_turn_held_for_the_pair_keeps_waiting_through_a_rescue_and_is_served_by_the_relaunch(
        world, hold_cluster):  # noqa: F811 — the fixture imported above
    """`_hold_for_cluster` re-checks `cluster_coming` on every change and lets a turn go 503 the moment the
    cluster is `failed`. Every view the loop publishes from the hang to the relaunched pair serving must
    keep it waiting — then the same views, replayed into a proxy, must end in a 200."""
    sglang_launched(world)
    views = [v for _, v in replay(world, "06:26:15", "06:38:30", until=lambda v: bool(world.calls("down")))]
    world.executor.log_text["box-a"] = whole(HEALTHY_HEAD)
    views += [world.tick() for _ in range(6)]
    world.prober.table[TP] = {**LIVE_TP, "served_models": (SGLANG_SERVES,)}
    views.append(world.tick())
    assert views[-1]["state"] == "running", views[-1]["detail"]
    assert [v["state"] for v in views if not cluster_coming(v)] == [], "a view that releases a held turn"

    proxy, hub, head, _ = hold_cluster("starting")
    hub.set_cluster(views[0])

    def feed():
        for v in views[1:-1]:
            hub.set_cluster(v)
            time.sleep(0.01)
        pair_up(hub, head)
    threading.Timer(0.2, feed).start()
    status, body, _ = post(proxy)
    assert status == 200, body


# ── the boot log keeps every attempt ──────────────────────────────────────────

T0 = 1_791_008_771.0
CONFIG = {"model_choice": SGLANG, "generation": 131, "max_num_seqs": 32, "max_num_batched_tokens": 8192,
          "num_speculative_tokens": 4, "ple_replicate": None, "kv_cache_memory": 73435823273, "kv_offload_gib": 0,
          "max_model_len": 262144, "long_prefill_token_threshold": 0}


def boot_tick(rec, at, state, detail="", *, loading=None, attempt=1, hang=None):
    return rec.observe(now=T0 + at, state=state, detail=detail, generation=131, config=CONFIG,
                       config_source="launch", loading=loading, attempt=attempt, hang=hang)


def test_a_hung_attempt_is_its_own_failed_line_and_the_relaunch_opens_the_next(tmp_path):
    rec = BootRecorder(str(tmp_path))
    boot_tick(rec, 0, "reserving")
    boot_tick(rec, 4, "starting")
    boot_tick(rec, 100, "starting", loading={"phase": CAPTURING})
    hang = {"phase": CAPTURING, "quiet_s": 182.0, "limit_s": 180.0, "attempt": 1, "retries": 2, "relaunch": True,
            "evidence": "start-hangs/gen131-attempt1"}
    why = "hung in “capturing CUDA graphs” for 182 s (the limit for this step is 180 s) — relaunching (retry 1/2)"
    first = boot_tick(rec, 290, "restarting", why, hang=hang, attempt=2)
    assert first is not None and first["outcome"] == "failed" and first["attempt"] == 1 and first["hang"] == hang
    assert first["failure"] == why and first["failed_phase"] == CAPTURING and first["generation"] == 131
    boot_tick(rec, 295, "freeing", attempt=2)
    boot_tick(rec, 300, "starting", attempt=2)
    second = boot_tick(rec, 425, "running", "TP=2 pair up (generation 131)", attempt=2)
    assert second["outcome"] == "serving" and second["attempt"] == 2 and second["hang"] is None
    assert second["total_s"] == 130.0, "the relaunch is timed from the moment it was decided"
    lines = [json.loads(line) for f in tmp_path.glob("boots-*.jsonl") for line in f.read_text().splitlines()]
    assert lines == [first, second]
    assert sum(1 for b in lines if b["hang"]) / len(lines) == 0.5, "the hang rate, straight off the log"


# ── a pair that has SERVED is not a start ─────────────────────────────────────

#: Generation 173's head, 2026-10-09 (cut from lb/state/start-hangs/gen173-attempt1/): a healthy start ("Application
#: startup complete" 07:17:55), then twelve minutes of serving — decode at 57-84 tok/s, /health 200 every 5 s to
#: 07:30:44 — as `docker logs` held it when the LB tore it down.
SERVED_HEAD = "sglang-gen173-2026-10-09-served-then-dark-rank0.log"
GEN173_STARTED = "2026-10-09T07:16:15Z"            # its container's StartedAt: before its first stamped line


def gen173_at(when: str) -> str:
    return "".join(f"{line}\n" for stamp, line in sglang_stamped(SERVED_HEAD) if stamp <= when)


def gen173_epoch(when: str) -> float:
    h, m, sec = (int(v) for v in when.split(":"))
    return float(calendar.timegm((2026, 10, 9, h, m, sec)))


def gen173_serving(w: World) -> None:
    """Generation 173 up and serving at 07:30:44 (the LB's clock in UTC epoch, so the head's stamps date its lines
    exactly as they did live), its head's log as it then stood."""
    w.executor.started_at = {"box-a": GEN173_STARTED, "box-b": GEN173_STARTED}
    w.clock.t = gen173_epoch("07:16:15")
    sglang_launched(w)
    w.executor.log_text["box-a"] = gen173_at("07:30:44")
    w.clock.t = gen173_epoch("07:30:44")
    w.prober.table[TP] = {**LIVE_TP, "served_models": (SGLANG_SERVES,)}
    v = w.tick(seconds=5)
    assert v["state"] == "running", v["detail"]


def test_one_missed_health_probe_on_a_pair_that_served_is_not_a_hang_of_its_finished_start(world):
    """GENERATION 173, 2026-10-09 07:30:49: one /health probe did not come back in time while the engine kept decoding
    and answering /v1/messages. The pair dropped to `starting`, the LB re-read the head's log, found its START's last
    line at 07:17:55 and called it "hung 774 s in “API server up”" (limit 180 s) — tearing down a pair that had served
    for twelve minutes, whose relaunch the retried turns then took down for good (memory guard, `failed`). A pair that
    has served is quiet from the moment it stopped answering, never from when its start finished."""
    gen173_serving(world)
    world.prober.table[TP] = OFF                             # 07:30:49: the probe that did not come back in time
    dark = [world.tick(seconds=9), world.tick(seconds=9)]   # 07:30:49 and 07:30:58: the two looks that tore it down
    assert not world.calls("down") and not world.calls("halt"), [v["detail"] for v in dark]
    assert [v["state"] for v in dark] == ["starting", "starting"], [v["detail"] for v in dark]
    assert not any("hung" in v["detail"] or "774" in v["detail"] or "783" in v["detail"] for v in dark)
    world.prober.table[TP] = {**LIVE_TP, "served_models": (SGLANG_SERVES,)}
    assert world.tick()["state"] == "running" and not world.calls("down"), "it answered again: it serves on"


def test_a_pair_that_served_and_went_dark_for_good_is_relaunched_on_the_steps_clock_from_when_it_went_dark(world):
    """A served pair that really stopped answering is still rescued — at the step's own limit counted from the last
    look that saw it serve, not at once."""
    gen173_serving(world)
    last_served = world.clock.t - 5                          # the tick that saw it serve ran before the clock moved
    world.prober.table[TP] = OFF
    for _ in range(40):
        v = world.tick(seconds=15)
        if rescued(v):
            break
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    at = world.clock.t - 15
    assert last_served + 400 <= at <= last_served + 400 + 30, at - last_served
    assert sorted(c[1] for c in world.calls("down")) == ["box-a", "box-b"]


# ── generation 188 (2026-10-10): a healthy seven-minute weight load the LB read as silence ─────────────

GEN188_HEAD = "sglang-gen188-2026-10-10-flash-next-rank0.log"
GEN188_WORKER = "sglang-gen188-2026-10-10-flash-next-rank1.log"
GEN188_DAY = "2026-10-10T"
#: What box-a had free, in GiB, between the figures the head itself printed: `Load weight begin. avail mem=109.63 GB`
#: (21:04:47), `Load weight end … avail mem=46.64 GB, mem usage=62.99 GB` (21:11:43); the drafter's pass 50.00 → 48.27
#: (21:11:58 → 21:12:33). The first shard took 33 s (the bar's own `1/145 [00:32<…]`), the rest came steadily. On a
#: GB10 the GPU's memory IS the box's (tp-rank, `observe_memory`), so MemAvailable falls by what the weights take.
GEN188_MEM = [("21:04:30", 109.6), ("21:05:20", 109.6), ("21:11:21", 46.6), ("21:11:43", 46.6), ("21:11:58", 50.0),
              ("21:12:33", 48.3), ("21:20:00", 48.3)]
LOADING = StartupPhase.loading(LoadPass.MODEL)


def gen188_at(when: str) -> str:
    """The head's log as `docker logs` held it at `when` (UTC HH:MM:SS): its `\\r` bar is one line, in at 21:11:33."""
    return log_at(GEN188_HEAD, when, day=GEN188_DAY)


def gen188_mem(when: str) -> float:
    """box-a's MemAvailable at `when`, straight between the figures the head printed."""
    t = clock_of(when)
    for (a, ga), (b, gb) in zip(GEN188_MEM, GEN188_MEM[1:]):
        ta, tb = clock_of(a), clock_of(b)
        if ta <= t <= tb:
            return round(ga + (gb - ga) * (t - ta) / (tb - ta), 1)
    return GEN188_MEM[0][1] if t < clock_of(GEN188_MEM[0][0]) else GEN188_MEM[-1][1]


def hhmmss(t: int) -> str:
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def replay_box(w: World, start: str, end: str, *, log, mem, until=None, step: int = 15) -> list[tuple[int, dict]]:
    """Tick every `step` s from `start` to `end`, the head's log AND box-a's memory as they really went; stop once
    `until(view)` holds. (time, view) per tick."""
    out = []
    for t in range(clock_of(start), clock_of(end) + 1, step):
        w.executor.log_text["box-a"] = log(hhmmss(t))
        w.executor.mem["box-a"] = mem(hhmmss(t))
        v = w.tick(seconds=step)
        out.append((t, v))
        if until is not None and until(v):
            break
    return out


def gen188_launched(w: World) -> int:
    w.executor.log_text["box-b"] = whole(GEN188_WORKER)
    return sglang_launched(w)


def test_generation_188s_seven_minute_weight_load_is_progress_not_silence(world):
    """21:04:47 "Load weight begin"; 21:11:33 the bar, whole, in one docker entry; 21:11:43 "Load weight end" — and at
    21:10 the panel said "no progress for 374 s in “initializing model”", the step painted red, a relaunch due at
    600 s (generation 179 had already been relaunched inside a healthy load). The box knew better: box-a gave up 63 GiB
    over those minutes, weight by weight. Memory the head's box gives up while a weight load is OPEN is progress — the
    clock does not run, the step is the load's, and the page is handed the GiB taken so far."""
    gen188_launched(world)
    timeline = replay_box(world, "21:04:30", "21:12:45", log=gen188_at, mem=gen188_mem)
    assert {v["state"] for _, v in timeline} == {"starting"}, \
        {v["detail"] for _, v in timeline if v["state"] != "starting"}
    assert not world.calls("down") and not world.calls("halt")
    assert not any("no progress" in v["detail"] for _, v in timeline), \
        [v["detail"] for _, v in timeline if "no progress" in v["detail"]]
    at_2110 = next(v for t, v in timeline if t == clock_of("21:10:00"))
    assert at_2110["loading"]["phase"] == LOADING and at_2110["loading"]["steps"][-1] == LOADING, at_2110["loading"]
    assert "completed" not in at_2110["loading"], "no bar in sight: the LB does not invent a count"
    assert 40 <= at_2110["loading"]["taken_gib"] <= 55, at_2110["loading"]
    # The figure grows with the load, whole GiB, never backwards — and is on the wire only while a load is open.
    taken = [v["loading"].get("taken_gib") for t, v in timeline
             if clock_of("21:05:30") <= t <= clock_of("21:11:20")]
    assert all(isinstance(g, int) for g in taken) and taken == sorted(taken) and taken[-1] > taken[0] + 30, taken
    after = next(v for t, v in timeline if t == clock_of("21:11:45"))
    assert after["loading"]["phase"] == StartupPhase.loaded(LoadPass.MODEL) and "taken_gib" not in after["loading"]
    world.prober.table[TP] = {**LIVE_TP, "served_models": (SGLANG_SERVES,)}
    assert world.tick()["state"] == "running"


def test_a_load_whose_memory_stops_moving_is_hung_on_the_same_clock_as_before(world):
    """The strict half: the same start, but from 21:06:00 nothing moves — no line (the bar is a `\\r` bar), no GiB
    taken. A shard read that stalls (a hung NFS mount) looks exactly like this, and it is still rescued at the step's
    limit (600 s, the ceiling: a load's length is the storage's), counted from the last movement."""
    gen188_launched(world)
    freeze = "21:06:00"
    timeline = replay_box(world, "21:04:30", "21:20:00", until=rescued,
                          log=lambda when: gen188_at(min(when, freeze)),
                          mem=lambda when: gen188_mem(min(when, freeze)))
    at, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    assert clock_of(freeze) + 600 <= at <= clock_of(freeze) + 600 + 30, at - clock_of(freeze)
    assert f"hung in “{LOADING}” for" in v["detail"] and "relaunching (retry 1/2)" in v["detail"], v["detail"]
    quiet = [v for _, v in timeline if "no progress" in v["detail"]]
    assert quiet and all(LOADING in v["detail"] for v in quiet), [v["detail"] for v in quiet][:2]
    assert "relaunch at 600 s if still quiet · 2 retries left" in quiet[-1]["detail"], quiet[-1]["detail"]


def test_memory_moving_outside_a_load_is_not_progress(world):
    """The guard: generation 131's capture deadlock, with box-a's MemAvailable wobbling 5 GiB tick to tick (ollama coming
    and going, the kernel's cache). Only an OPEN weight load reads memory as progress, so the FlashInfer/capture hangs
    are rescued at the step's limit exactly as before, and no load figure is published outside a load."""
    sglang_launched(world)
    wobble = iter([100.0, 95.0] * 200)
    timeline = replay_box(world, "06:26:15", "06:38:30", until=rescued, log=gen131_at, mem=lambda when: next(wobble))
    at, v = timeline[-1]
    assert rescued(v), f"never rescued: {v['state']} — {v['detail']}"
    assert clock_of("06:28:00") + 400 <= at <= clock_of("06:28:00") + 420, at
    assert CAPTURING in v["detail"]
    assert all("taken_gib" not in s["loading"] for _, s in timeline
               if not StartupPhase.is_loading(s["loading"].get("phase", "")))
    assert any("taken_gib" in s["loading"] for _, s in timeline), "its own weight load (06:26:32-06:27:17) carried one"
