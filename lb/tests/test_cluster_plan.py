"""S1 — the cluster planner is a pure function of (desired, observed), and its safety rules hold.

A GB10 box belongs to the pair only through its RESERVATION: one broker_parrot node_job per box per
generation, whose states are the lease's (queued → running → ended). Never a park (operator,
2026-09-15: "We must not interfere with parked anywhere in vllm service, we just add node_job with
priority and wait for broker_parrot to take it"). The rules (worklog/vllm-gateway-program.md L2, L3, L7):

  * never `Up` unless BOTH boxes' current-generation reservations are RUNNING and claimed by the
    pinned box's own lane, with no other fleet job there, no broker vLLM and enough free memory —
    all OBSERVED, not assumed;
  * never release a reservation while a rank is observed on its box, or while the box is unobservable;
  * a rank comes `Down` only when its reservation ended, was claimed by another box, or sat back in
    the queue past the engine's own lease duration (600 s) — never on a guess, never for a park;
  * an operator's park or unpark changes the words, NEVER the actions;
  * Start frees RAM first, and says so: `freeing` while a broker vLLM is up;
  * once perturbations stop, the plan converges: `running` in cluster mode, `off` in classic.

Nothing here touches SSH, Docker or a broker. A world simulator stands in for the fleet.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from vllm_lb.catalog import Availability
from vllm_lb.cluster import (FINISHING_LIMIT_S, FREEING_DEADLINE_S, KILL_FLOOR_GIB, REQUEUE_GRACE_S, START_RETRIES, START_STALL_S,
                             BoxObs, Desired, Down, Halt, Observed, RamHolder, RankObs, ReleaseReservation,
                             Reservation, Reserve, Up, blockers, is_cluster_purpose, plan, start_stall_limit,
                             tag_for)

BOXES = ("box-a", "box-b")
LABELS = ("box-a-gpu", "box-b-gpu")
LABEL = dict(zip(BOXES, LABELS))
HEX = {"box-a": "aaaaaaaa", "box-b": "bbbbbbbb"}
#: What the test selection COSTS on a box (`BoxObs.needs_gib`, memory_estimate) — every box the helper builds is
#: sized at it. There is no flat requirement to fall back on: a box the LB cannot size is refused, naming why.
NEED = 90.0
GEN = 3
#: The cluster's ModelChoices (the model library, vllm_lb/library), and what the chosen one serves.
FLASH, DENSE = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4"
SERVES = "Qwen/Qwen3.8-Flash-Next"


def desired(mode="cluster", gen=GEN, seqs=16, ctx=262144, choice=FLASH) -> Desired:
    return Desired(mode=mode, generation=gen, max_num_seqs=seqs, max_model_len=ctx, model_choice=choice)


def availability(present: bool | None, choice: str = FLASH) -> dict:
    """What a box answered about ONE ModelChoice: present, certainly missing, or it could not say."""
    return {choice: Availability(name="Qwen3.8-Flash-Next-hibrid48", searched=("~/models", "/srv/models"),
                                 dir="~/models" if present else None,
                                 image="myllmbox/qwen38-flash-next-cluster-vllm:v5", image_present=present,
                                 observed_at=1.0, error="" if present is not None else "box-a could not look")}


NOT_MOUNTED = ("/mnt/box-a-home/models/qwen38-flash-next-cluster is not mounted: NFS /mnt/box-a-home "
               "(192.0.2.201:/) is not up on this box yet")


def unreadable(error: str = NOT_MOUNTED, choice: str = FLASH) -> dict:
    """A box that LOOKED and could not see: a share not mounted yet, a stale handle, a folder that did not
    answer — tp-rank's own words in `error`, and no folder found. Not "missing": it has not said that."""
    return {choice: Availability(name="Qwen3.8-Flash-Next-hibrid48",
                                 searched=("~/models", "/mnt/box-a-home/models", "/srv/models"),
                                 dir=None, image="myllmbox/qwen38-flash-next-cluster-vllm:v5",
                                 image_present=True, observed_at=1.0, error=error)}


def reservation(name: str, status: str = "running", *, claimed_by: str | None = "", run_open: bool = True,
                gen: int = GEN) -> Reservation:
    if claimed_by == "":
        claimed_by = LABEL[name] if status in ("running", "completed") else None
    return Reservation(reservation_id=f"run-{name}-{gen}",
                       job_id=f"{HEX[name]}-{gen:04d}-4000-8000-000000000000",
                       status=status, claimed_by=claimed_by, run_open=run_open)


def box(name: str, *, res="held", known=True, stale=(), parked_by=None, jobs=0, rank=None,
        broker_vllm=False, mem=110.0, requeued_s=None, has=True, needs=NEED, stable=None,
        holds=None, peak=None, released_from=None, note="", error="", why="") -> BoxObs:
    """`has`: True the box holds the ModelChoice, False it certainly does not, None it was never asked —
    or a ready-made answer, for a plan whose ModelChoice is not the default. `needs`: what THIS selection
    costs on this box (memory_estimate), or None when it cannot be estimated. `stable`: the LOWEST of the
    last few observations — what the box can be COUNTED ON to have; defaults to `mem`, i.e. a box whose
    memory is not moving. `error`: why the box could not be asked, in the executor's words."""
    return BoxObs(reservation=reservation(name) if res == "held" else res, reservation_known=known,
                  stale=tuple(stale), parked_by=parked_by, running_jobs=jobs, rank=rank,
                  broker_vllm_running=broker_vllm, mem_available_gib=mem, requeued_s=requeued_s,
                  availability=has if isinstance(has, dict) else (None if has is None else availability(has)),
                  needs_gib=needs, mem_stable_gib=mem if stable is None else stable, holds=holds,
                  mem_peak_gib=mem if peak is None else peak, released_from_gib=released_from,
                  needs_note=note, needs_why=why, error=error)


def absent() -> RankObs:
    return RankObs(present=False, running=False, gen=None, exit_code=None)


def lost(gen=GEN, code=255) -> RankObs:
    """A rank whose BOX REBOOTED under it: the container started before the box's current boot and is not
    running — docker records 255 for one killed by a host power loss. It did not exit; it was lost."""
    return RankObs(present=True, running=False, gen=gen, exit_code=code, model_choice=FLASH, lost=True)


def running(gen=GEN, choice=FLASH) -> RankObs:
    return RankObs(present=True, running=True, gen=gen, exit_code=None, model_choice=choice)


def exited(gen=GEN, code=1) -> RankObs:
    return RankObs(present=True, running=False, gen=gen, exit_code=code, model_choice=FLASH)


def observed(box_a: BoxObs, box_b: BoxObs, *, head_live=False, classic_inflight=0,
             head_serves=None, head_model=None, quiet=None, phase="", head_inflight=0) -> Observed:
    return Observed(boxes={"box-a": box_a, "box-b": box_b}, head_live=head_live,
                    classic_inflight=classic_inflight, head_inflight=head_inflight,
                    head_serves=head_serves, head_model=head_model,
                    start_quiet_s=quiet, start_phase=phase)


# ── adopting what is already standing ────────────────────────────────────────

def test_a_running_rank_that_carries_no_model_label_is_adopted_not_replaced():
    """THE INCIDENT, 2026-09-16 00:30:32Z. A pair had been launched by code that predated the
    ModelChoice label, so both ranks reported `model_choice=None`. The new planner read `None != the choice`,
    called them "a rank of another model", and took a healthy, SERVING pair down — an eight-minute reload
    nobody asked for:

        00:30:37 [cluster] down box-a: ok
        00:30:40 [cluster] unknown → restarting: replacing a rank of another model on box-a and …

    A rank that cannot say which model it runs is not evidence that it runs the wrong one. What settles that
    is the head's own /v1/models (`head_model`) — and here it agrees with the choice."""
    p = run(desired(), observed(box("box-a", rank=running(choice=None)),
                                box("box-b", rank=running(choice=None)),
                                head_live=True, head_model=FLASH))
    assert not [a for a in p.actions if isinstance(a, Down)], f"an adoptable pair was taken DOWN: {p.state}"
    assert p.state != "restarting", f"a serving pair was replaced: {p.state}"


def test_a_running_rank_that_names_a_different_model_is_still_replaced():
    """The counter-case, so the rule above cannot decay into "never replace": a rank that SAYS it runs
    another model is precisely what a generation number alone cannot catch."""
    p = run(desired(), observed(box("box-a", rank=running(choice=DENSE)),
                                box("box-b", rank=running(choice=DENSE))))
    assert [a for a in p.actions if isinstance(a, Down)], "a rank of a DIFFERENT model must be replaced"
    assert p.state == "restarting", f"expected a replacement, got {p.state}"


# ── the ModelChoice: a box is asked what it HAS before anything is reserved ───

def test_a_start_waits_for_each_box_to_say_what_it_has_before_reserving_anything():
    """Availability is OBSERVED. Until every box has answered, the plan reserves nothing: taking both
    GB10 boxes from the fleet for a model that is not on them is the expensive mistake."""
    p = run(desired(), observed(box("box-a", res=None, rank=absent(), has=None),
                                box("box-b", res=None, rank=absent())))
    assert p.actions == [] and p.state == "checking" and p.held is False
    assert "box-a" in p.detail


def test_a_box_that_does_not_have_the_chosen_model_blocks_the_start_and_says_what_is_missing():
    p = run(desired(), observed(box("box-a", res=None, rank=absent()),
                                box("box-b", res=None, rank=absent(), has=False)))
    assert not of(p.actions, Reserve) and not of(p.actions, Up)
    assert p.state == "blocked" and p.held is False
    assert "box-b" in p.detail and "Qwen3.8-Flash-Next-hibrid48/config.json" in p.detail


def test_a_box_that_could_not_look_blocks_too_and_is_never_read_as_missing():
    p = run(desired(), observed(box("box-a", res=None, rank=absent()),
                                box("box-b", res=None, rank=absent(), has=None)))
    assert p.actions == [] and p.state == "checking"


def test_a_box_whose_share_is_not_mounted_yet_is_waited_for_and_never_told_to_choose_another_model():
    """2026-10-10 20:15:22Z: both GB10 boxes rebooted mid-start; box-b reads the checkpoint over NFS from box-a,
    and box-a's export was not up yet. The LB said `blocked: box-b: no …/config.json in …; choose another
    model, or copy it there` about a checkpoint that was exactly where it always is, and the operator saw a
    failure nobody reacted to. A box that LOOKED and could not see has not said "missing": the start waits,
    names the share and why, and says nothing about choosing another model."""
    p = run(desired(), observed(box("box-a", res=None, rank=absent()),
                                box("box-b", res=None, rank=absent(), has=unreadable())))
    assert p.actions == [] and p.state == "checking" and p.held is False
    assert "waiting for box-b to see Qwen3.8-Flash-Next-hibrid48" in p.detail, p.detail
    assert "/mnt/box-a-home" in p.detail and "not mounted" in p.detail, p.detail
    assert "choose another model" not in p.detail and "no Qwen3.8" not in p.detail, p.detail
    assert "asking" not in p.detail, "it has answered; the LB is not still asking"


def test_a_box_whose_docker_could_not_speak_about_the_image_is_waited_for_too():
    """The files are there; docker on the box did not answer about the image (a daemon restarting after the
    reboot). Not "no image": the box could not say."""
    seen = {FLASH: Availability(name="Qwen3.8-Flash-Next-hibrid48", searched=("~/models",), dir="~/models",
                                image="myllmbox/qwen38-flash-next-cluster-vllm:v5", image_present=None,
                                observed_at=1.0, error="")}
    p = run(desired(), observed(box("box-a", res=None, rank=absent()),
                                box("box-b", res=None, rank=absent(), has=seen)))
    assert p.actions == [] and p.state == "checking"
    assert "box-b" in p.detail and "myllmbox/qwen38-flash-next-cluster-vllm:v5" in p.detail, p.detail
    assert "choose another model" not in p.detail, p.detail


def test_only_a_readable_folder_without_the_file_blocks_the_start():
    """The other half of the rule, so it cannot decay into "never block": a box that could read every folder
    and found no config.json HAS said missing — and that is the one case for "choose another model"."""
    p = run(desired(), observed(box("box-a", res=None, rank=absent()),
                                box("box-b", res=None, rank=absent(), has=False)))
    assert p.state == "blocked" and "choose another model" in p.detail


# ── a box that cannot be observed: unknown, and WHY ───────────────────────────

def test_an_unobservable_box_is_unknown_with_the_executors_own_reason_never_failed():
    """`reserving → unknown: cannot observe box-b` (2026-10-10, box-b rebooting) said nothing the operator
    could act on. The executor knows why — ssh timed out, no route, exit 255 — and that goes in the line."""
    why = "status: exit 255: ssh: connect to host 192.0.2.3 port 22: No route to host"
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=None, error=why)))
    assert p.state == "unknown" and not p.held and "cannot observe box-b" in p.detail
    assert why in p.detail, p.detail
    p = run(desired(mode="classic"), observed(box("box-a", rank=running()), box("box-b", rank=None, error=why)))
    assert p.state == "unknown" and why in p.detail, p.detail
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=None)))
    assert p.state == "unknown" and p.detail == "cannot observe box-b", "no reason known: none invented"


# ── a box that REBOOTED under its rank: relaunched, never failed ──────────────

def test_a_rank_lost_to_a_reboot_relaunches_the_pair_instead_of_failing_it():
    """2026-09-19 08:07Z (power loss, all three boxes) and 2026-10-10 20:12Z (the operator rebooted both
    GB10 boxes): the LB read `rank on box-a exited (code 255); Stop, then Start again`, stayed `failed` and kept
    both GB10 boxes reserved and idle for 3.5 h until the operator did the Stop+Start by hand. A container that
    started BEFORE the box's current boot did not exit — the box went away under it. Nothing in its log
    explains a reboot, so there is nothing to keep: both ranks come down and the same generation launches
    again through every gate a start passes (availability, reservations, memory)."""
    p = run(desired(), observed(box("box-a", rank=lost()), box("box-b", rank=lost())))
    assert p.state == "restarting" and not p.held
    assert sorted(p.actions, key=str) == [Down("box-a"), Down("box-b")] and not of(p.actions, Halt)
    assert "box-a and box-b rebooted" in p.detail and "relaunching" in p.detail, p.detail
    assert "Stop, then Start" not in p.detail and "exited" not in p.detail, p.detail


def test_one_box_rebooting_takes_its_partner_down_for_the_relaunch_whatever_that_partner_did():
    """The partner is a consequence: still running (NCCL has not noticed yet), or exited with the error the
    lost peer caused. Neither is the fault, so neither is failed."""
    for partner in (running(), exited(code=1)):
        p = run(desired(), observed(box("box-a", rank=lost()), box("box-b", rank=partner)))
        assert p.state == "restarting" and "box-a rebooted" in p.detail, (partner, p)
        assert sorted(p.actions, key=str) == [Down("box-a"), Down("box-b")] and not of(p.actions, Halt)


def test_a_serving_pair_whose_box_reboots_is_relaunched_too():
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=lost()), head_live=True))
    assert p.state == "restarting" and not p.held and "box-b rebooted" in p.detail


def test_a_lost_rank_in_classic_mode_simply_comes_down():
    p = run(desired(mode="classic"), observed(box("box-a", rank=lost()), box("box-b", rank=absent())))
    assert p.actions == [Down("box-a")] and p.state == "unloading"


def test_a_rank_that_exited_after_its_box_booted_is_still_a_failure():
    """The counter-case: `lost` is the box's boot time against the container's start, never the exit code."""
    p = run(desired(), observed(box("box-a", rank=exited(code=255)), box("box-b", rank=running())))
    assert p.state == "failed" and "exited (code 255)" in p.detail


def test_availability_never_blocks_a_running_pair_or_a_stop():
    """Whether a box could be asked says nothing about a model already loaded on it."""
    up = observed(box("box-a", rank=running(), has=None), box("box-b", rank=running(), has=False),
                  head_live=True)
    p = run(desired(), up)
    assert p.state == "running" and p.held is True
    assert run(desired(mode="classic"), up).state == "unloading"


def test_the_launch_carries_the_model_choice_the_operator_selected():
    p = run(desired(seqs=12, ctx=131072, choice=DENSE),
            observed(box("box-a", rank=absent(), has=availability_of(DENSE)),
                     box("box-b", rank=absent(), has=availability_of(DENSE))))
    assert p.actions == [Up("box-a", rank=0, gen=GEN, max_num_seqs=12, max_model_len=131072, model_choice=DENSE),
                         Up("box-b", rank=1, gen=GEN, max_num_seqs=12, max_model_len=131072, model_choice=DENSE)]


def test_a_rank_of_this_generation_running_another_model_choice_is_replaced():
    """A generation alone cannot tell two models apart; the rank says which one it was launched for."""
    p = run(desired(), observed(box("box-a", rank=running(choice=DENSE)), box("box-b", rank=running()),
                                head_live=True))
    assert Down("box-a") in p.actions and p.held is False
    assert p.state == "restarting" and "box-a" in p.detail


def test_a_head_serving_another_model_is_never_held_and_says_which():
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=running()),
                                head_live=True, head_serves=False, head_model="Qwen/Qwen3.8-27B"))
    assert p.held is False and p.state == "failed"
    assert "Qwen/Qwen3.8-27B" in p.detail


def availability_of(choice: str) -> dict:
    """A box that holds THIS ModelChoice — for a plan whose choice is not the default."""
    return availability(True, choice)


def run(d: Desired, o: Observed, **kw):
    return plan(d, o, boxes=BOXES, labels=LABELS, **kw)


def of(actions, kind) -> list:
    return [a for a in actions if isinstance(a, kind)]


# ── Start, one tick at a time ─────────────────────────────────────────────────

def test_start_asks_broker_parrot_to_reserve_both_boxes_and_does_nothing_else():
    p = run(desired(), observed(box("box-a", res=None, rank=absent()), box("box-b", res=None, rank=absent())))
    assert p.actions == [Reserve("box-a", gen=GEN), Reserve("box-b", gen=GEN)]
    assert p.state == "reserving" and p.held is False
    assert "asking broker_parrot to reserve box-a and box-b" in p.detail


def test_start_frees_ram_first_letting_gateway_turns_on_classic_vllm_end():
    """A claimed reservation makes broker_parrot stop its own vLLM; our turns on it end first (the
    router already sends no NEW turn to classic vLLM once the cluster is wanted)."""
    o = observed(box("box-a", res=None, rank=absent(), broker_vllm=True),
                 box("box-b", res=None, rank=absent(), broker_vllm=True), classic_inflight=2)
    p = run(desired(), o)
    assert p.actions == [] and p.state == "freeing"
    assert "freeing RAM" in p.detail and "2 gateway turn" in p.detail


def test_queued_reservations_are_waited_for_and_never_asked_for_twice():
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "queued"), rank=absent()),
                                box("box-b", res=reservation("box-b", "queued"), rank=absent())))
    assert p.actions == [] and p.state == "reserving"
    assert "asking broker_parrot to reserve box-a and box-b (node_job aaaaaaaa, bbbbbbbb)" in p.detail


def test_while_a_broker_vllm_is_up_the_step_is_named_freeing():
    p = run(desired(), observed(box("box-a", rank=absent(), broker_vllm=True, mem=30),
                                box("box-b", res=reservation("box-b", "queued"), rank=absent(), broker_vllm=True)))
    assert p.actions == [] and p.state == "freeing"
    assert "freeing RAM: stopping classic vLLM on box-a, box-b" in p.detail


def test_reserved_boxes_wait_for_the_fleets_own_jobs_to_drain():
    p = run(desired(), observed(box("box-a", jobs=1, rank=absent()), box("box-b", rank=absent())))
    assert p.actions == [] and p.state == "draining" and "box-a: 1 fleet job" in p.detail


def test_drained_boxes_wait_for_memory():
    p = run(desired(), observed(box("box-a", rank=absent(), mem=40.0), box("box-b", rank=absent())))
    assert p.actions == [] and p.state == "freeing"
    assert "box-a: 40.0 GiB free" in p.detail and f"need {NEED:.1f}" in p.detail


def test_the_memory_a_box_must_have_free_is_what_the_selection_needs_not_a_constant():
    """H3 (worklog/vllm-cluster-handover.md). The gate was one number for every model, context and
    concurrency. It cuts both ways: a selection that costs MORE than the constant was waved through, and
    one that costs LESS was held back by a number nobody measured."""
    tight = run(desired(), observed(box("box-a", rank=absent(), mem=85.0, needs=103.2),
                                    box("box-b", rank=absent(), mem=85.0, needs=80.0)))
    assert not of(tight.actions, Up) and tight.state == "freeing"
    assert "box-a: 85.0 GiB free, need 103.2" in tight.detail, tight.detail
    assert "box-b" not in tight.detail, "box-b already has what its own rank needs"

    roomy = run(desired(), observed(box("box-a", rank=absent(), mem=85.0, needs=80.0),
                                    box("box-b", rank=absent(), mem=85.0, needs=80.0)))
    assert of(roomy.actions, Up), "a selection that FITS was held back by a constant nobody measured"


def test_the_head_and_the_worker_are_gated_on_their_OWN_need():
    """L2 (worklog/vllm-cluster-hold-robustness.md §4): the flat 100 is too low for the head and too
    high for the worker. The calibrated totals for qwen3.8-flash-next at this selection are 103.2 GiB
    (head: overhead 12.66) and 97.7 GiB (worker: 7.16) — so 99 GiB free must refuse rank 0 and admit
    rank 1, which one constant can never do."""
    p = run(desired(), observed(box("box-a", rank=absent(), mem=99.0, needs=103.2),
                                box("box-b", rank=absent(), mem=99.0, needs=97.7)))
    assert not of(p.actions, Up) and p.state == "freeing"
    assert "box-a: 99.0 GiB free, need 103.2" in p.detail and "box-b" not in p.detail, p.detail
    both = run(desired(), observed(box("box-a", rank=absent(), mem=104.0, needs=103.2),
                                   box("box-b", rank=absent(), mem=99.0, needs=97.7)))
    assert of(both.actions, Up), "the worker's own need was met and the head's too"


def test_one_lucky_sample_of_a_swinging_number_never_launches():
    """L3 (worklog/vllm-cluster-hold-robustness.md §3). At 00:59:06 the gate PASSED — on a single 5 s
    sample of a number swinging 17.6 GiB (the size of the ollama model) while a hold and a chat client
    fought over the box. What a box can be counted on to have is the LOWEST of its last few
    observations, not whichever one the tick happened to land on."""
    p = run(desired(), observed(box("box-a", rank=absent(), mem=101.0, stable=77.0, needs=97.7),
                                box("box-b", rank=absent(), mem=101.0, needs=97.7)))
    assert not of(p.actions, Up), "a 99 GiB load launched on one lucky sample"
    assert p.state == "freeing" and "box-a: 77.0 GiB free, need 97.7" in p.detail, p.detail
    steady = run(desired(), observed(box("box-a", rank=absent(), mem=101.0, needs=97.7),
                                     box("box-b", rank=absent(), mem=101.0, needs=97.7)))
    assert of(steady.actions, Up), "a box whose memory is NOT moving is not held back"


def test_the_wait_says_WHAT_is_holding_the_ram():
    """L8 (worklog/vllm-cluster-hold-robustness.md §6): the operator asked "why does box-b wait" and the
    page could not tell them. `freeing RAM; box-a: 77 GiB free, need 100` is not an answer — "ollama still
    holds 17.6 GiB of qwen3.8, because a chat job on that box reloaded it" is."""
    p = run(desired(), observed(box("box-a", rank=absent(), mem=77.0, needs=103.2,
                                    holds=(RamHolder("qwen3.8:latest", 17.6),)),
                                box("box-b", rank=absent())))
    assert p.state == "freeing"
    assert "box-a: 77.0 GiB free, need 103.2: short 26.2 GiB — ollama holds qwen3.8:latest, 17.6 GiB" in p.detail, p.detail


def test_the_freeing_wait_has_a_budget_and_says_when_it_is_past_it_without_calling_that_a_failure():
    """L3, the other half: `freeing` could wait forever with no escalation, unlike `waiting`, which has
    REQUEUE_GRACE_S — and it said the same one sentence throughout. Past the budget the line says so, with
    what still holds the RAM: that is the escalation. It is not a FAILURE: nothing errored, the RAM may yet
    come back (a chat client's ollama model expiring is the usual way), and then the start goes on. Launching
    short is still never done — the cost of being wrong is a ~13-minute cold start and, as generation 10
    showed, a dead rank."""
    assert FREEING_DEADLINE_S == 600.0, "the cold-start cost is the budget"
    o = observed(box("box-a", rank=absent(), mem=77.0, needs=103.2), box("box-b", rank=absent()))
    p = run(desired(), o, freeing_s=FREEING_DEADLINE_S - 1)
    assert p.state == "freeing" and "past" not in p.detail
    p = run(desired(), o, freeing_s=FREEING_DEADLINE_S)
    assert p.state == "freeing" and not of(p.actions, Up), "a timeout is a wait, never a failure"
    assert "box-a: 77.0 GiB free, need 103.2" in p.detail, p.detail
    assert "600 s" in p.detail and "past" in p.detail and "600 s" in p.detail, p.detail
    assert "Stop, then Start again" not in p.detail and "failed" not in p.detail, p.detail
    # …and once the RAM is back, the same plan launches: nothing was given up.
    freed = observed(box("box-a", rank=absent(), mem=115.0, needs=103.2), box("box-b", rank=absent()))
    assert of(run(desired(), freed, freeing_s=2 * FREEING_DEADLINE_S).actions, Up)


def test_a_selection_that_cannot_be_sized_is_refused_naming_what_is_missing():
    """Classic vLLM's memory is broker_parrot's, and a model with no footprint in the catalog cannot be sized. No
    flat requirement stands in (LB_CLUSTER_MIN_MEM_GIB is retired, 2026-10-29: a guessed 100 GiB launched
    generation 10 onto 77 GiB): the Start is refused with the estimate's own reason, on a box whose memory IS
    known — a box that has not reported yet is a wait, as it always was."""
    why = "classic vLLM's memory is broker_parrot's: qwen3.8-27b-nvfp4 has no cluster LaunchProfile to estimate"
    p = run(desired(), observed(box("box-a", rank=absent(), mem=80.0, needs=None, why=why),
                                box("box-b", rank=absent(), mem=80.0, needs=None, why=why)))
    assert not of(p.actions, Up) and p.state == "failed", p
    assert "cannot size" in p.detail and f"box-a: {why}" in p.detail and "need " not in p.detail, p.detail
    p = run(desired(), observed(box("box-a", rank=absent(), mem=None, needs=None, why=why),
                                box("box-b", rank=absent(), mem=None, needs=None, why=why)))
    assert p.state == "freeing" and "memory unknown" in p.detail and "cannot size" not in p.detail, p


@pytest.mark.parametrize("over", [{"jobs": None}, {"broker_vllm": None}, {"mem": None}])
def test_an_unreadable_gate_is_a_closed_gate(over):
    p = run(desired(), observed(box("box-a", rank=absent(), **over), box("box-b", rank=absent())))
    assert not of(p.actions, Up), p


def test_both_reservations_running_and_clear_launches_both_ranks_head_first():
    p = run(desired(seqs=12, ctx=131072), observed(box("box-a", rank=absent()), box("box-b", rank=absent())))
    assert p.actions == [Up("box-a", rank=0, gen=GEN, max_num_seqs=12, max_model_len=131072, model_choice=FLASH),
                         Up("box-b", rank=1, gen=GEN, max_num_seqs=12, max_model_len=131072, model_choice=FLASH)]
    assert p.state == "starting" and p.held is False


@pytest.mark.parametrize("box_b_res", [
    reservation("box-b", "queued"),
    reservation("box-b", "running", claimed_by="box-a-gpu"),
    reservation("box-b", "running", run_open=False),
    reservation("box-b", "running", gen=2),
    None,
], ids=["queued", "claimed-elsewhere", "run-closed", "older-generation", "none"])
def test_up_needs_both_reservations_running_and_claimed_by_the_pinned_box(box_b_res):
    o = observed(box("box-a", rank=absent()), box("box-b", res=None, rank=absent()))
    if box_b_res is not None and box_b_res.reservation_id.endswith("-2"):
        o = replace(o, boxes={**o.boxes, "box-b": replace(o.boxes["box-b"], stale=(box_b_res,))})
    else:
        o = replace(o, boxes={**o.boxes, "box-b": replace(o.boxes["box-b"], reservation=box_b_res)})
    assert not of(run(desired(), o).actions, Up)


def test_ranks_running_but_head_not_answering_is_starting_not_held():
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=running()), head_live=False))
    assert p.actions == [] and p.state == "starting" and p.held is False


def test_ranks_running_and_head_live_is_running_and_held():
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=running()), head_live=True))
    assert p.actions == [] and p.state == "running" and p.held is True


# ── an operator's park is READ, never answered ────────────────────────────────

def test_a_queued_reservation_on_a_parked_box_waits_and_names_who_parked_it():
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "queued"), parked_by="cluster-panel",
                                    rank=absent()), box("box-b", rank=absent())))
    assert p.actions == [] and p.state == "waiting"
    assert "waiting for broker_parrot to claim box-a — parked by cluster-panel" in p.detail


def test_a_drain_park_on_a_reserved_box_keeps_the_reservation_and_the_ranks_and_only_says_so():
    p = run(desired(), observed(box("box-a", rank=running(), parked_by="cluster-panel"),
                                box("box-b", rank=running()), head_live=True))
    assert p.actions == [] and p.state == "running" and p.held is True
    assert "box-a parked by cluster-panel — reservation kept" in p.detail


PARK_TABLE = {
    "start": observed(box("box-a", res=None, rank=absent()), box("box-b", res=None, rank=absent())),
    "queued": observed(box("box-a", res=reservation("box-a", "queued"), rank=absent()), box("box-b", rank=absent())),
    "clear": observed(box("box-a", rank=absent()), box("box-b", rank=absent())),
    "running": observed(box("box-a", rank=running()), box("box-b", rank=running()), head_live=True),
    "requeued": observed(box("box-a", res=reservation("box-a", "queued"), rank=running(), requeued_s=30.0),
                         box("box-b", rank=running()), head_live=True),
    "ended": observed(box("box-a", res=reservation("box-a", "cancelled", run_open=False), rank=running()),
                      box("box-b", rank=running())),
}


def with_parks(o: Observed, who: str | None, boxes) -> Observed:
    return replace(o, boxes={b: replace(ob, parked_by=who if b in boxes else None) for b, ob in o.boxes.items()})


@pytest.mark.parametrize("mode", ["cluster", "classic"])
@pytest.mark.parametrize("name", sorted(PARK_TABLE))
def test_parking_or_unparking_a_box_never_changes_a_single_action(mode, name):
    """Live, 2026-09-15 20:18: the operator turned both lanes back ON while the pair served, and the
    park-based planner answered with Down + Park — twice. A park is the operator's; the plan does
    not move when it does (L3)."""
    o = PARK_TABLE[name]
    base = run(desired(mode=mode), with_parks(o, None, ()))
    for who in ("cluster-panel", "vllm-lb/cluster#2"):
        for boxes in (("box-a",), ("box-b",), BOXES):
            p = run(desired(mode=mode), with_parks(o, who, boxes))
            assert (p.actions, p.held) == (base.actions, base.held), (who, boxes, p, base)


# ── only a reservation that ended takes a rank down ───────────────────────────

@pytest.mark.parametrize("status", ["cancelled", "failed", "completed"])
def test_an_ended_reservation_takes_its_rank_down_and_is_a_sticky_failure(status):
    p = run(desired(), observed(box("box-a", res=reservation("box-a", status), rank=running()),
                                box("box-b", rank=running()), head_live=True))
    assert Down("box-a") in p.actions and Down("box-b") not in p.actions
    assert not of(p.actions, Reserve) and not of(p.actions, Up)
    assert p.state == "failed" and p.held is False
    assert "ended the reservation on box-a" in p.detail and status in p.detail and "Stop, then Start" in p.detail


def test_a_reservation_whose_run_was_closed_has_ended_even_while_its_job_row_still_runs():
    p = run(desired(), observed(box("box-a", res=reservation("box-a", run_open=False), rank=running()),
                                box("box-b", rank=running()), head_live=True))
    assert Down("box-a") in p.actions and p.state == "failed" and p.held is False


def test_a_reservation_claimed_by_another_box_takes_the_rank_down():
    p = run(desired(), observed(box("box-a", res=reservation("box-a", claimed_by="box-b-gpu"), rank=running()),
                                box("box-b", rank=running()), head_live=True))
    assert Down("box-a") in p.actions and p.state == "failed" and p.held is False
    assert "box-b-gpu" in p.detail


def test_an_ended_reservation_before_any_rank_is_sticky_too_and_never_asked_for_again():
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "cancelled", run_open=False), rank=absent()),
                                box("box-b", rank=absent())))
    assert p.actions == [] and p.state == "failed"


def test_a_requeued_reservation_keeps_the_ranks_inside_the_engines_lease_duration():
    """A lane restart re-queues the row and the same box re-claims it: the pair is not reloaded for
    that — but it carries no traffic while broker_parrot does not hold the box for it."""
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "queued"), rank=running(), requeued_s=120.0),
                                box("box-b", rank=running()), head_live=True))
    assert p.actions == [] and p.state == "waiting" and p.held is False
    assert "broker_parrot re-queued the reservation on box-a" in p.detail


def test_a_requeue_not_reclaimed_within_the_grace_takes_the_pair_down():
    assert REQUEUE_GRACE_S == 600.0, "the engine's own DEFAULT_LEASE_S"
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "queued"), rank=running(),
                                    requeued_s=REQUEUE_GRACE_S), box("box-b", rank=running()), head_live=True))
    assert sorted(of(p.actions, Down), key=str) == [Down("box-a"), Down("box-b")]
    assert not of(p.actions, ReleaseReservation) and not of(p.actions, Reserve)
    assert p.state == "restarting" and "600 s" in p.detail


def test_a_rank_without_a_reservation_comes_down_in_every_mode():
    for mode in ("cluster", "classic"):
        p = run(desired(mode=mode), observed(box("box-a", res=None, rank=running()), box("box-b", rank=running())))
        assert Down("box-a") in p.actions, mode
        assert not of(p.actions, Up), mode


# ── what cannot be observed blocks, and never becomes a guess ─────────────────

def test_unreadable_reservations_block_every_action_and_down_nothing_on_a_guess():
    o = observed(box("box-a", known=False, res=None, rank=running()),
                 box("box-b", known=False, res=None, rank=running()), head_live=True)
    for mode in ("cluster", "classic"):
        p = run(desired(mode=mode), o)
        assert not of(p.actions, ReleaseReservation) and not of(p.actions, Reserve) and not of(p.actions, Up)
        assert p.held is False
    p = run(desired(), o)
    assert p.actions == [] and p.state == "unknown" and "reservations" in p.detail


def test_a_box_that_cannot_be_observed_blocks_up_and_release():
    p = run(desired(), observed(box("box-a", rank=None), box("box-b", rank=absent())))
    assert p.actions == [] and p.state == "unknown" and "box-a" in p.detail
    p = run(desired(mode="classic"), observed(box("box-a", rank=None), box("box-b", rank=absent())))
    assert not of(p.actions, ReleaseReservation) and p.state == "unknown"


# ── generations, crashes, and the pair as one unit ────────────────────────────

def test_a_rank_of_an_older_generation_is_replaced():
    p = run(desired(gen=4), observed(box("box-a", res=reservation("box-a", gen=4), rank=running(gen=3)),
                                     box("box-b", res=reservation("box-b", gen=4), rank=absent())))
    assert p.actions == [Down("box-a")] and p.state == "restarting"


def _old_pair_serving(inflight: int) -> Observed:
    """Generation 3's pair still up and answering, generation 4 asked for — a restart."""
    return observed(box("box-a", res=reservation("box-a", gen=4), rank=running(gen=3)),
                    box("box-b", res=reservation("box-b", gen=4), rank=running(gen=3)),
                    head_live=True, head_serves=True, head_inflight=inflight)


def test_a_restart_lets_the_turns_streaming_on_the_old_pair_finish_first():
    p = run(desired(gen=4), _old_pair_serving(3))
    assert p.state == "finishing" and not of(p.actions, Down), "a Down here cut every stream on the pair"
    assert "3 turn(s)" in p.detail and not p.held, "not held: nothing new may be routed to the old pair"


def test_the_restart_goes_ahead_once_the_last_turn_on_the_old_pair_ended():
    p = run(desired(gen=4), _old_pair_serving(0))
    assert p.state == "restarting" and p.actions[:2] == [Down("box-a"), Down("box-b")]


def test_a_turn_that_never_ends_does_not_stop_the_restart_past_its_limit():
    p = plan(desired(gen=4), _old_pair_serving(1), boxes=BOXES, labels=LABELS,
             finishing_s=FINISHING_LIMIT_S, finishing_limit_s=FINISHING_LIMIT_S)
    assert p.state == "restarting" and Down("box-a") in p.actions


def test_half_an_old_pair_or_a_silent_head_is_not_waited_for():
    half = observed(box("box-a", res=reservation("box-a", gen=4), rank=running(gen=3)),
                    box("box-b", res=reservation("box-b", gen=4), rank=absent()),
                    head_live=True, head_inflight=2)
    assert run(desired(gen=4), half).state == "restarting", "half a pair serves nothing: no turn can finish"
    silent = replace(_old_pair_serving(2), head_live=False)
    assert run(desired(gen=4), silent).state == "restarting"


def test_an_older_generations_reservation_goes_back_only_once_no_rank_stands_on_its_box():
    old = reservation("box-a", gen=2)
    kw = dict(res=reservation("box-a", "queued"), stale=(old,))
    p = run(desired(), observed(box("box-a", rank=running(gen=2), **kw), box("box-b", rank=absent())))
    assert not of(p.actions, ReleaseReservation)
    p = run(desired(), observed(box("box-a", rank=absent(), **kw), box("box-b", rank=absent())))
    assert ReleaseReservation("box-a", reservation_id=old.reservation_id, job_id=old.job_id) in p.actions


def test_an_exited_rank_of_the_current_generation_is_a_sticky_failure():
    p = run(desired(), observed(box("box-a", rank=exited(code=137)), box("box-b", rank=running())))
    assert p.state == "failed" and "box-a" in p.detail and "137" in p.detail
    assert not of(p.actions, Up), "a 99 GiB reload is not a retry primitive"
    p = run(desired(mode="classic"), observed(box("box-a", rank=exited(code=137)), box("box-b", rank=running())))
    assert sorted(p.actions, key=str) == [Down("box-a"), Down("box-b")]


# ── a start that fails lets go of the boxes, and says why ────────────────────

AUTOTUNING = "autotuning FlashInfer kernels"


def test_a_rank_that_exited_takes_its_partner_out_of_service_and_both_keep_their_logs():
    """2026-09-18 23:31:19Z: the head exited (code 1) and the worker stayed up, holding 115 GiB of a box
    whose lane was already logging "0.1 GB available — under the 512 MB KILL floor" every second; that
    morning a dead head's partner held box-b for 37 minutes. Half a pair is nothing (the rule already
    applied to a MISSING partner) — but it is HALTED, not removed: on 09-18 the worker's log was the whole
    diagnosis, and `down` would have destroyed it (L5)."""
    p = run(desired(), observed(box("box-a", rank=exited(code=1)), box("box-b", rank=running())))
    assert p.state == "failed" and "box-a exited (code 1)" in p.detail
    assert p.actions == [Halt("box-b")]
    p = run(desired(), observed(box("box-a", rank=exited(code=1)), box("box-b", rank=exited(code=137))))
    assert p.state == "failed" and p.actions == [], "sticky, and nothing left to stop"


def test_a_start_that_stops_making_progress_is_stalled_not_failed_and_both_ranks_are_halted():
    """THE HANG (fixtures/qwen38-start-2026-09-18-hung-rank0.log): 53 minutes in one StartupPhase, the page
    saying `starting` throughout. A healthy start never goes 60 s without its phase or a shard count
    changing (the 2026-09-15 fixture's longest quiet window is 56 s), and every OTHER wait in this planner
    already has a 600 s bound. Once its relaunches are spent — none here; test_start_rescue.py has the
    relaunches and each step's own limit — it is HALTED, with its logs kept, exactly as before. But a start
    that ran out its clock is not a start that errored (operator, 2026-10-10: "there must be real error, not
    timeout to show failed"): the state is `stalled`, and the line says what the LB did and why."""
    both = (box("box-a", rank=running()), box("box-b", rank=running()))
    limit = start_stall_limit(AUTOTUNING)
    p = run(desired(), observed(*both, quiet=limit, phase=AUTOTUNING), start_retries=0)
    assert p.state == "stalled" and not p.held
    assert p.actions == [Halt("box-a"), Halt("box-b")], "the protective halt stands"
    assert p.hang is not None and p.hang.relaunch is False and p.hang.limit_s == limit
    assert f"hung in “{AUTOTUNING}” for {limit:.0f} s" in p.detail and f"limit for this step is {limit:.0f} s" in p.detail, p.detail
    assert "halted" in p.detail and "logs" in p.detail, p.detail
    assert p.detail.endswith("Stop, then Start again")
    assert "failed" not in p.detail and "stalled" not in p.detail, f"the state word is the headline's: {p.detail}"


@pytest.mark.parametrize("quiet", [None, 0.0, 59.0, start_stall_limit(AUTOTUNING) - 1])
def test_a_start_still_moving_or_one_that_cannot_be_read_is_left_alone(quiet):
    """None = the head's log could not be read: a start nobody can see has not been SEEN to stall."""
    both = (box("box-a", rank=running()), box("box-b", rank=running()))
    p = run(desired(), observed(*both, quiet=quiet, phase=AUTOTUNING))
    assert p.state == "starting" and p.actions == []


def test_a_quiet_start_says_so_long_before_it_gives_up():
    both = (box("box-a", rank=running()), box("box-b", rank=running()))
    p = run(desired(), observed(*both, quiet=240.0, phase=AUTOTUNING))
    assert p.state == "starting" and p.actions == []
    assert "no progress for 240 s" in p.detail and AUTOTUNING in p.detail
    assert f"{start_stall_limit(AUTOTUNING):.0f} s" in p.detail
    assert "no progress" not in run(desired(), observed(*both, quiet=30.0, phase=AUTOTUNING)).detail


def test_a_serving_pair_is_never_judged_by_its_startup_log():
    both = (box("box-a", rank=running()), box("box-b", rank=running()))
    p = run(desired(), observed(*both, head_live=True, head_serves=True, quiet=10 * START_STALL_S,
                                phase=AUTOTUNING))
    assert p.state == "running" and p.held and p.actions == []


def test_the_stall_deadline_is_the_deployments_to_set():
    both = (box("box-a", rank=running()), box("box-b", rank=running()))
    p = run(desired(), observed(*both, quiet=90.0, phase=AUTOTUNING), start_stall_s=60.0, start_retries=0)
    assert p.state == "stalled" and len(of(p.actions, Halt)) == 2
    assert run(desired(), observed(*both, quiet=90.0, phase=AUTOTUNING), start_stall_s=60.0).hang.limit_s == 60.0


def test_a_rank_never_takes_its_box_down_with_it():
    """box-b, 2026-09-18 23:21-23:34Z: MemAvailable at 0.1-0.2 GB for thirteen minutes under a runaway
    rank, NVRM out-of-memory in the kernel log, and then the box was gone. broker_parrot's sentinel sheds
    only ITS OWN servers at that floor — the rank is ours, so the floor is ours to keep. Judged on the
    HIGHEST of the last samples: every one of them under the floor, never one dip."""
    low = KILL_FLOOR_GIB / 2
    starved = box("box-b", rank=running(), mem=low, peak=low)
    p = run(desired(), observed(box("box-a", rank=running()), starved))
    assert p.state == "failed" and not p.held
    assert p.actions == [Halt("box-a"), Halt("box-b")]
    assert "box-b" in p.detail and "out of memory" in p.detail
    dipped = box("box-b", rank=running(), mem=low, peak=9.0)          # one low sample among healthy ones
    unknown = replace(starved, mem_peak_gib=None)                       # too few samples to say
    for ok in (dipped, unknown):
        p = run(desired(), observed(box("box-a", rank=running()), ok))
        assert p.state == "starting" and p.actions == []


def test_a_serving_pair_is_never_stopped_by_the_gateway_for_being_tight_on_memory():
    """LIVE, 2026-09-19 00:56Z: generation 13 SERVING at full load (8 running) with box-a at 1.7 GiB
    MemAvailable, stable — 1.2 GiB above the floor. A pair that answers is working: stopping it for a
    number is an outage by our own hand, and if the box really runs dry the rank dies and the half-pair
    rule takes the partner down. The floor guards a START — the 09-18 shape — never a serving pair."""
    starved = box("box-b", rank=running(), mem=KILL_FLOOR_GIB / 2, peak=KILL_FLOOR_GIB / 2)
    p = run(desired(), observed(box("box-a", rank=running()), starved, head_live=True, head_serves=True))
    assert p.state == "running" and p.held and p.actions == []


def test_a_halt_is_never_planned_on_a_box_whose_rank_is_already_coming_down():
    """Safety first still wins: a rank whose reservation ended is REMOVED, and not also halted."""
    gone = reservation("box-b", "completed", run_open=False)
    p = run(desired(), observed(box("box-a", rank=exited(code=1)), box("box-b", res=gone, rank=running())))
    assert Down("box-b") in p.actions and not of(p.actions, Halt)


# ── the handover between generations: ask before you let go ──────────────────

def test_an_older_generations_reservation_is_kept_until_this_generations_has_been_asked_for():
    """D1, live 2026-09-16 01:01Z (worklog/vllm-cluster-handover.md). The old reservations went back
    and only THEN were new ones asked for; in the 36 s gap the lane resumed normal service and claimed
    a pool `llm_chat` job pinned to qwen3.8, re-loading ollama on the box the next generation needed:

        01:01:15 [cluster-hold] box-b-gpu: RELEASED cluster reservation ba9928b4 — normal service resumes
        01:01:15 [claim-worker:gpu:pool] claimed 452d615a (node=llm_chat/llm_chat model=qwen3.8:latest)
        01:01:51 [cluster-hold] box-b-gpu: HOLDING this box for cluster reservation 5d8e41d9 (…#11)

    broker_parrot cannot hold two reservations for one box at once (node_queue.py:1326-1337 weighs
    `gpu_slots='all'` as the box's whole budget, and node_queue.py:1442-1447 caps one running `cluster`
    row per lane), so the successor cannot be CLAIMED first. What it can be is QUEUED first: a queued
    run-next reservation already forbids the lane's own LLM starts and, being run-next-wide, is the only
    row that lane may claim (node_queue.py:1363-1376). So the predecessor is given back only once the
    successor has been asked for."""
    stale_box_a, stale_box_b = reservation("box-a", gen=2), reservation("box-b", gen=2)
    p = run(desired(), observed(box("box-a", res=None, stale=(stale_box_a,), rank=absent()),
                                box("box-b", res=None, stale=(stale_box_b,), rank=absent())))
    assert of(p.actions, Reserve) == [Reserve("box-a", gen=GEN), Reserve("box-b", gen=GEN)]
    assert not of(p.actions, ReleaseReservation), \
        "a box was handed back to the fleet before its successor was even asked for"


def test_and_it_goes_back_as_soon_as_this_generations_reservation_exists():
    """The counter-case, so the rule above cannot decay into "never release": the moment this
    generation's row exists on that box, the older one is returned."""
    old = reservation("box-a", gen=2)
    p = run(desired(), observed(box("box-a", res=reservation("box-a", "queued"), stale=(old,), rank=absent()),
                                box("box-b", rank=absent())))
    assert ReleaseReservation("box-a", old.reservation_id, old.job_id) in p.actions


def test_a_lone_running_rank_is_taken_down_so_the_pair_restarts_together():
    p = run(desired(), observed(box("box-a", rank=running()), box("box-b", rank=absent())))
    assert p.actions == [Down("box-a")] and p.state == "restarting"


# ── Stop ──────────────────────────────────────────────────────────────────────

def test_stop_takes_the_ranks_down_before_any_reservation_goes_back():
    p = run(desired(mode="classic"), observed(box("box-a", rank=running()), box("box-b", rank=running()),
                                              head_live=True))
    assert sorted(p.actions, key=str) == [Down("box-a"), Down("box-b")]
    assert p.state == "unloading" and p.held is False


def test_stop_releases_the_reservations_only_once_no_box_carries_a_rank():
    p = run(desired(mode="classic"), observed(box("box-a", rank=absent()), box("box-b", rank=running())))
    assert p.actions == [Down("box-b")], "box-a stays reserved: its partner still has a rank"
    p = run(desired(mode="classic"), observed(box("box-a", rank=absent()), box("box-b", rank=absent())))
    assert p.actions == [ReleaseReservation("box-a", reservation_id="run-box-a-3", job_id=reservation("box-a").job_id),
                         ReleaseReservation("box-b", reservation_id="run-box-b-3",
                                            job_id=reservation("box-b").job_id)]
    assert p.state == "releasing" and "box-a and box-b" in p.detail


def test_stop_closes_an_ended_reservation_whose_run_is_still_open_and_leaves_a_closed_one_alone():
    p = run(desired(mode="classic"), observed(box("box-a", res=reservation("box-a", "failed"), rank=absent()),
                                              box("box-b", res=reservation("box-b", "completed", run_open=False),
                                                  rank=absent())))
    assert p.actions == [ReleaseReservation("box-a", reservation_id="run-box-a-3",
                                            job_id=reservation("box-a").job_id)]


def test_a_released_reservation_whose_job_is_still_returning_reads_releasing_and_asks_nothing():
    p = run(desired(mode="classic"), observed(box("box-a", res=reservation("box-a", run_open=False), rank=absent()),
                                              box("box-b", res=None, rank=absent())))
    assert p.actions == [] and p.state == "releasing"


def test_classic_with_nothing_of_ours_left_is_off():
    p = run(desired(mode="classic"), observed(box("box-a", res=None, rank=absent()),
                                              box("box-b", res=reservation("box-b", "completed", run_open=False),
                                                  rank=absent())))
    assert p.actions == [] and p.state == "off" and p.held is False


def test_purpose_tags_are_ours_only_with_our_prefix():
    assert tag_for(3) == "vllm-lb/cluster#3"
    assert is_cluster_purpose("vllm-lb/cluster#3") and is_cluster_purpose("vllm-lb/cluster#12")
    assert not is_cluster_purpose("cluster-panel") and not is_cluster_purpose(None)
    assert not is_cluster_purpose("") and not is_cluster_purpose(7)


def test_an_engine_without_cluster_reservations_refuses_start_with_the_reason():
    o = replace(observed(box("box-a", res=None, rank=absent()), box("box-b", res=None, rank=absent())),
                reservations_unsupported="broker_parrot's engine has no node_queue.CLUSTER_SERVER_KIND")
    p = run(desired(), o)
    assert p.actions == [] and p.state == "failed" and "CLUSTER_SERVER_KIND" in p.detail
    assert run(desired(mode="classic"), o).state == "off", "Stop is never refused"


# ── the randomized sequence: the rules hold at every step, and the plan converges ──

DT = 20.0                                   # seconds per step: 30 steps cover the re-queue grace


class World:
    """A fleet that reacts the way broker_parrot (under the reservation contract) and the GB10 boxes do."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.t = 0.0
        self.res = {b: [] for b in BOXES}             # dict(gen, id, job, status, claimed_by, run_open)
        self.parked = {b: None for b in BOXES}
        self.jobs = {b: rng.choice([0, 1, 2]) for b in BOXES}
        self.broker_vllm = {b: True for b in BOXES}
        self.rank = {b: None for b in BOXES}          # None | dict(gen, running, exit_code)
        self.up_fails = {b: False for b in BOXES}
        self.head_live_in, self.head_live = None, False
        self.executor_down = self.db_down = False
        self.inflight = rng.choice([0, 1])
        self.requeued_at: dict[str, float] = {}
        self.hung = False                             # the pair's start never reaches /health
        self.quiet = 0.0                              # how long its log has said nothing new
        self.rescues: dict[int, int] = {}             # generation → relaunches after a hang (the loop's count)

    def current(self, b: str, gen: int):
        mine = [r for r in self.res[b] if r["gen"] == gen]
        return mine[-1] if mine else None

    @staticmethod
    def _obs(r) -> Reservation:
        return Reservation(reservation_id=r["id"], job_id=r["job"], status=r["status"],
                           claimed_by=r["claimed_by"], run_open=r["run_open"])

    def observe(self, gen: int) -> Observed:
        boxes = {}
        for b in BOXES:
            r = self.rank[b]
            rank = None if self.executor_down else (
                absent() if r is None else RankObs(present=True, running=r["running"], gen=r["gen"],
                                                    exit_code=r["exit_code"],
                                                    model_choice=r.get("choice", FLASH)))
            cur = self.current(b, gen)
            requeued_s = None                           # what the loop measures (cluster_loop.py)
            # Both boxes hold the ModelChoice throughout: availability is exercised by its own tests,
            # and a fleet that keeps answering "yes" leaves these rules to be the only thing under test.
            if cur and cur["run_open"] and cur["status"] == "queued" and rank and rank.present and rank.gen == gen:
                requeued_s = self.t - self.requeued_at.setdefault(cur["job"], self.t)
            elif cur:
                self.requeued_at.pop(cur["job"], None)
            mem = 110.0 if not self.broker_vllm[b] and self.jobs[b] == 0 else 30.0
            boxes[b] = BoxObs(
                reservation=None if self.db_down or cur is None else self._obs(cur),
                reservation_known=not self.db_down,
                stale=() if self.db_down else tuple(self._obs(s) for s in self.res[b]
                                                    if s["gen"] != gen and s["run_open"]),
                parked_by=None if self.db_down else self.parked[b],
                running_jobs=None if self.db_down else self.jobs[b], rank=rank,
                broker_vllm_running=None if self.executor_down else self.broker_vllm[b],
                mem_available_gib=None if self.executor_down else mem,
                # This fleet's memory does not swing: what it has now is what it can be counted on to
                # have. The oscillation is exercised by its own test.
                mem_stable_gib=None if self.executor_down else mem, requeued_s=requeued_s,
                availability=None if self.executor_down else availability(True))
        return Observed(boxes=boxes, head_live=self.head_live, classic_inflight=self.inflight,
                        head_serves=True if self.head_live else None,
                        start_quiet_s=None if self.executor_down else self.quiet,
                        start_phase="capturing CUDA graphs", start_rescues=self.rescues.get(gen, 0))

    def apply(self, actions) -> None:
        for a in actions:
            if isinstance(a, Reserve):
                if self.current(a.box, a.gen) is None:            # the derived run id: never a second row
                    self.res[a.box].append(dict(gen=a.gen, id=f"run-{a.box}-{a.gen}",
                                                job=f"{HEX[a.box]}-{a.gen:04d}", status="queued",
                                                claimed_by=None, run_open=True))
            elif isinstance(a, ReleaseReservation):
                for r in self.res[a.box]:
                    if r["id"] == a.reservation_id:
                        r["run_open"] = False
                        if r["status"] == "queued":
                            r["status"] = "cancelled"
            elif isinstance(a, Up):
                if not self.up_fails[a.box]:
                    self.rank[a.box] = {"gen": a.gen, "running": True, "exit_code": None,
                                        "choice": a.model_choice}
            elif isinstance(a, Down):
                self.rank[a.box] = None
            elif isinstance(a, Halt):
                if self.rank[a.box] and self.rank[a.box]["running"]:
                    self.rank[a.box] = {**self.rank[a.box], "running": False, "exit_code": 137}

    def _requeue(self, b: str) -> None:
        for r in self.res[b]:
            if r["status"] == "running" and r["run_open"]:
                r["status"], r["claimed_by"] = "queued", None

    def tick(self, perturb: bool) -> None:
        self.t += DT
        for b in BOXES:
            for r in self.res[b]:                               # a released body returns
                if not r["run_open"] and r["status"] == "running":
                    r["status"] = "completed"
            ours_running = any(r["status"] == "running" for r in self.res[b])
            asked = any(r["run_open"] and r["status"] in ("queued", "running") for r in self.res[b])
            if asked or self.parked[b]:
                self.jobs[b] = max(0, self.jobs[b] - 1)         # a priority row drains the box
                if (ours_running or self.parked[b]) and self.jobs[b] == 0:
                    self.broker_vllm[b] = False                  # the claimant / ParkQuiesce stops it
            elif self.rng.random() < 0.3:
                self.jobs[b] = self.rng.choice([0, 1])           # the fleet keeps working
                self.broker_vllm[b] = self.rng.random() < 0.5
            if not self.parked[b] and not ours_running and self.jobs[b] == 0:
                queued = [r for r in self.res[b] if r["run_open"] and r["status"] == "queued"]
                if queued:                                       # the pinned box claims, oldest first
                    queued[0]["status"], queued[0]["claimed_by"] = "running", LABEL[b]
        both = all(self.rank[b] and self.rank[b]["running"] for b in BOXES)
        if both and self.hung:
            self.head_live, self.quiet = False, self.quiet + DT
        elif both:
            self.head_live_in = 2 if self.head_live_in is None else max(0, self.head_live_in - 1)
            self.head_live, self.quiet = self.head_live_in == 0, 0.0
        else:
            self.head_live_in, self.head_live, self.quiet, self.hung = None, False, 0.0, False
        self.executor_down = self.db_down = False
        self.up_fails = {b: False for b in BOXES}
        self.inflight = self.rng.choice([0, 0, 1]) if perturb else 0
        if not perturb:
            return
        b = self.rng.choice(BOXES)
        what = self.rng.choice(["crash", "reboot", "executor", "park", "hardpark", "unpark", "db", "upfail",
                                "lane_restart", "cancel", "fail", "hang", "none"])
        open_res = [r for r in self.res[b] if r["run_open"] and r["status"] in ("queued", "running")]
        if what == "hang" and both and not self.head_live:
            self.hung = True                                     # 2026-09-18: a start that never ends
        elif what == "crash" and self.rank[b] and self.rank[b]["running"]:
            self.rank[b] = {**self.rank[b], "running": False, "exit_code": 1}
        elif what == "reboot":
            if self.rank[b]:
                self.rank[b] = {**self.rank[b], "running": False, "exit_code": 255}
            self.jobs[b], self.broker_vllm[b] = 0, self.parked[b] is None
            self._requeue(b)                                     # its lane died: the engine re-queues
        elif what == "executor":
            self.executor_down = True
        elif what == "park":
            self.parked[b] = "cluster-panel"                     # drain: the held job keeps running
        elif what == "hardpark":
            self.parked[b] = "cluster-panel"
            self._requeue(b)
        elif what == "unpark":
            self.parked[b] = None
        elif what == "db":
            self.db_down = True
        elif what == "upfail":
            self.up_fails[b] = True
        elif what == "lane_restart":
            self._requeue(b)
        elif what == "cancel" and open_res:
            r = self.rng.choice(open_res)                        # the operator cancels the run
            r["run_open"] = False
            if r["status"] == "queued":
                r["status"] = "cancelled"
        elif what == "fail" and open_res:
            self.rng.choice(open_res)["status"] = "failed"


def grants(ob: BoxObs, b: str) -> bool:
    r = ob.reservation
    return r is not None and r.run_open and (
        (r.status == "running" and r.claimed_by == LABEL[b])
        or (r.status == "queued" and (ob.requeued_s or 0.0) < REQUEUE_GRACE_S))


def standing(ob: BoxObs, b: str, gen: int) -> bool:
    return bool(ob.rank and ob.rank.running and ob.rank.gen == gen and grants(ob, b))


def _check_rules(p, o: Observed, d: Desired) -> None:
    gen = d.generation
    downs = {a.box for a in p.actions if isinstance(a, Down)}
    halts = {a.box for a in p.actions if isinstance(a, Halt)}
    if halts:
        # A Halt is a failure's, or a hung start's that spent its relaunches (`stalled`): never a wait's.
        assert d.mode == "cluster" and p.state in ("failed", "stalled") and not p.held, f"a Halt outside a failure: {p}"
        assert p.state == "failed" or (p.hang is not None and not p.hang.relaunch), f"stalled without a hang: {p}"
        assert not halts & downs, f"halted AND removed: {p}"
        for b in halts:
            rank = o.boxes[b].rank
            assert rank is not None and rank.running and rank.gen == gen, f"a Halt of nothing on {b}: {rank}"
    if d.mode == "cluster" and p.state == "failed" and all(o.boxes[b].rank is not None for b in BOXES):
        left = [b for b in BOXES if o.boxes[b].rank.running and o.boxes[b].rank.gen == gen
                and b not in halts | downs and any(not o.boxes[c].rank.running and o.boxes[c].rank.present
                                                   and o.boxes[c].rank.gen == gen for c in BOXES)]
        assert not left, f"a failed pair still holds {left}: {p}"
    limit = start_stall_limit(o.start_phase)
    if o.start_quiet_s is not None and o.start_quiet_s >= limit and p.state == "starting":
        raise AssertionError(f"a start quiet for {o.start_quiet_s:.0f} s is still `starting`: {p}")
    if p.hang is not None and p.hang.relaunch:
        assert d.mode == "cluster" and p.state == "restarting" and not halts and not p.held, f"a rescue: {p}"
        assert o.start_quiet_s is not None and o.start_quiet_s >= limit, f"rescued before its limit: {p}"
        assert o.start_rescues < START_RETRIES and p.hang.attempt == o.start_rescues + 1, (o.start_rescues, p)
    for a in p.actions:
        if isinstance(a, Up):
            assert d.mode == "cluster"
            for b in BOXES:
                ob, r = o.boxes[b], o.boxes[b].reservation
                assert ob.reservation_known and r is not None and r.run_open, f"Up with {b} unreserved: {ob}"
                assert r.status == "running" and r.claimed_by == LABEL[b], f"Up with {b} not claimed by it: {ob}"
                assert ob.running_jobs == 0 and ob.broker_vllm_running is False, f"Up while {b} busy: {ob}"
                assert ob.mem_available_gib is not None and ob.mem_available_gib >= NEED, f"Up without memory on {b}"
                assert ob.rank is not None, f"Up with {b} unobserved"
        elif isinstance(a, ReleaseReservation):
            rank = o.boxes[a.box].rank
            assert rank is not None and not rank.present, f"release on {a.box} with rank {rank}"
            if d.mode != "cluster":
                assert all(o.boxes[b].rank is not None and not o.boxes[b].rank.present for b in BOXES)
        elif isinstance(a, Reserve):
            ob = o.boxes[a.box]
            assert d.mode == "cluster" and a.gen == gen and ob.reservation_known and ob.reservation is None, (a, ob)
            assert o.classic_inflight == 0, "asked while our turns were still on classic vLLM"
    for b in BOXES:
        ob = o.boxes[b]
        present = ob.rank is not None and ob.rank.present
        if present and ob.reservation_known and not grants(ob, b):
            assert b in downs, f"a rank on {b} stands without its reservation: {ob}"
        if d.mode == "cluster" and b in downs and standing(ob, b, gen) and not (p.hang and p.hang.relaunch):
            others = [c for c in BOXES if c != b and not standing(o.boxes[c], c, gen)]
            assert others, f"{b}'s rank taken down while its reservation holds and its partner stands: {p}"
        if d.mode == "cluster" and present and not ob.reservation_known and ob.rank.gen == gen:
            assert b not in downs, "a Down on a guess"
    for who, boxes in (("cluster-panel", BOXES), (None, ())):
        q = run(d, with_parks(o, who, boxes))
        assert (q.actions, q.held) == (p.actions, p.held), f"a park moved the plan: {q} vs {p}"
    if p.held:
        assert d.mode == "cluster" and o.head_live and not downs
        for b in BOXES:
            assert standing(o.boxes[b], b, gen) and o.boxes[b].reservation.status == "running"


@pytest.mark.parametrize("seed", range(40))
def test_rules_hold_under_random_perturbation_and_the_plan_converges(seed):
    rng = random.Random(seed)
    world = World(rng)
    d = desired(gen=1)
    for step in range(160):
        perturb = step < 80
        if step == 80:
            world.parked = {b: None for b in BOXES}             # the operator is done
        if perturb and rng.random() < 0.08:                      # the operator flips the switch
            d = replace(d, mode="classic") if d.mode == "cluster" else replace(d, mode="cluster",
                                                                              generation=d.generation + 1)
        o = world.observe(d.generation)
        p = run(d, o)
        _check_rules(p, o, d)
        if p.state in ("failed", "stalled"):                     # the way out: Stop, then Start
            d = replace(d, mode="classic")
        if p.hang is not None and p.hang.relaunch:               # the loop counts what it relaunched
            world.rescues[d.generation] = world.rescues.get(d.generation, 0) + 1
        world.apply(p.actions)
        world.tick(perturb)
    o = world.observe(d.generation)
    p = run(d, o)
    if d.mode == "cluster":
        assert p.state == "running" and p.held, (seed, p, o)
        for b in BOXES:
            cur = world.current(b, d.generation)
            assert world.rank[b]["running"] and cur["status"] == "running" and cur["claimed_by"] == LABEL[b]
    else:
        assert p.state == "off" and not p.held, (seed, p, o)
        assert all(world.rank[b] is None for b in BOXES)
        assert not [r for b in BOXES for r in world.res[b] if r["run_open"] or r["status"] in ("queued", "running")]


# ── the handover: the memory of the rank that just came down ──────────────────

def test_a_start_after_a_teardown_says_when_the_previous_rank_still_holds_the_memory():
    """2026-09-30: the fatal NVRM `Out of memory [NV_ERR_NO_MEMORY]` came at 20:46-20:47Z, the minute one
    16,384 generation ended and the next began. A start whose box is short right after a teardown is short
    for a REASON the operator can read — the rank that came down has not given back what it took — and not
    "freeing RAM", which reads like the fleet is in the way."""
    p = run(desired(), observed(box("box-a", rank=absent(), mem=40.0, needs=120.0, released_from=115.7),
                                box("box-b", rank=absent(), mem=120.0, needs=110.0)))
    assert not of(p.actions, Up), "launched onto a box that has not been given its memory back"
    assert p.state == "freeing"
    assert "has not given back" in p.detail, p.detail
    assert "116 GiB" in p.detail and "40.0 GiB free" in p.detail, p.detail


def test_once_the_memory_really_is_back_the_teardown_is_not_mentioned_again():
    """The note is self-limiting: it explains a shortage, it never becomes a permanent label on the box."""
    p = run(desired(), observed(box("box-a", rank=absent(), mem=118.0, needs=110.0, released_from=115.7),
                                box("box-b", rank=absent(), mem=120.0, needs=110.0)))
    assert len(of(p.actions, Up)) == 2, p.detail
    assert "has not given back" not in p.detail, p.detail


def test_the_knob_that_puts_the_requirement_past_the_box_is_named_in_the_state_line():
    """The refusal has to be actionable: "need 120" is not, "8.8 GiB of that is the 16,384-token prefill
    budget" is. The words come from the estimate, so the page and the state line cannot disagree."""
    note = " — 8.8 GiB of that is the 16,384-token prefill budget (max_num_batched_tokens; at 8,192 it needs 103)"
    p = run(desired(), observed(box("box-a", rank=absent(), mem=115.7, needs=120.0, note=note),
                                box("box-b", rank=absent(), mem=116.6, needs=114.0, note=note)))
    assert not of(p.actions, Up)
    assert "max_num_batched_tokens" in p.detail and "16,384" in p.detail, p.detail
    blocked = blockers(observed(box("box-a", rank=absent(), mem=115.7, needs=120.0, note=note),
                               box("box-b", rank=absent())).boxes["box-a"], 120.0)
    assert any("max_num_batched_tokens" in b["message"] for b in blocked), blocked


# ── a launch the LB refuses to build fails the start BEFORE the GB10 boxes are taken and before the RAM gate ─────────────
# GENERATION 187, 2026-10-10 20:58-21:04Z: `launch_refused` was judged AFTER the RAM gate, so a box whose lowest
# recent reading wandered 0.1 GiB either side of the need alternated `freeing` and `starting` every 20-60 s while the
# launch was refused on every fresh tick (131 times). A refusal of the SETTINGS is the same whatever the RAM says.

REFUSAL = ("SGLang KV pin 24,000,000,000 B needs mem_fraction_static 0.885, above this profile's safe 0.85 ceiling "
           "(maximum 19,918,363,626 B)")


def test_a_refused_launch_fails_the_start_before_the_ram_gate_not_after_it():
    short = observed(box("box-a", rank=absent(), mem=115.4, needs=115.5), box("box-b", rank=absent()))
    p = run(desired(), replace(short, launch_refused=REFUSAL))
    assert p.state == "failed", (p.state, p.detail)
    assert REFUSAL in p.detail and "refused to launch generation 3" in p.detail, p.detail
    assert not of(p.actions, Up) and not of(p.actions, Reserve)
    # …and with the RAM there, the same: the refusal does not depend on the gate.
    roomy = observed(box("box-a", rank=absent()), box("box-b", rank=absent()))
    p = run(desired(), replace(roomy, launch_refused=REFUSAL))
    assert p.state == "failed" and not of(p.actions, Up), (p.state, p.detail)


def test_a_refused_launch_never_takes_the_boxes_from_the_fleet():
    """Reserving both GB10 boxes for a launch that cannot be built is the expensive mistake in its purest form."""
    o = observed(box("box-a", rank=absent(), res=None), box("box-b", rank=absent(), res=None))
    p = run(desired(), replace(o, launch_refused=REFUSAL))
    assert p.state == "failed" and not of(p.actions, Reserve), (p.state, p.detail, p.actions)


def test_a_refused_launch_is_about_a_launch_still_ahead_never_a_pair_already_running():
    """A catalog whose ceiling moved under a serving pair must not take it down: what it RUNS is past this check."""
    o = observed(box("box-a", rank=running()), box("box-b", rank=running()), head_live=True, head_serves=True)
    p = run(desired(), replace(o, launch_refused=REFUSAL))
    assert p.state == "running" and p.held is True, (p.state, p.detail)
    assert not of(p.actions, Halt) and not of(p.actions, Down)


# ── ONE RAM gate: the figure the plan waits on is the figure the box row shows, to a tenth, and a need nothing on
# the box can free is refused with the numbers, not waited on ───────────────────────────────────────────────────────

def test_a_need_nothing_on_the_box_can_free_is_refused_at_once_with_the_numbers():
    """box-a at rest: nothing resident, no fleet job, no classic vLLM, its last rank's memory back, three readings
    within a GiB — and still 0.1 GiB short of a 115.5 GiB need. No wait brings that memory: the selection is
    refused in the first plan that can tell, with both numbers (operator, 2026-10-10: six minutes of flapping)."""
    o = observed(box("box-a", rank=absent(), mem=115.5, stable=115.4, peak=115.5, needs=115.5, holds=()),
                 box("box-b", rank=absent(), mem=116.6, needs=110.8, holds=()))
    p = run(desired(), o)
    assert p.state == "failed", (p.state, p.detail)
    assert "box-a: 115.4 GiB free is the most it frees at rest" in p.detail, p.detail
    assert "need 115.5" in p.detail and "short 0.1 GiB" in p.detail, p.detail
    assert "refused to launch generation 3" in p.detail and "Stop, then Start again" in p.detail, p.detail
    assert not of(p.actions, Up)


@pytest.mark.parametrize("still", [
    dict(holds=(RamHolder("qwen3.8:latest", 17.6),)),        # something resident can expire
    dict(holds=None),                                          # the box could not say what holds it
    dict(peak=117.0),                                          # its reading is still moving
    dict(released_from=117.0),                                 # its last rank has not given it all back
    dict(jobs=1),                                              # a fleet job still runs there
    dict(broker_vllm=None),                                    # whether classic vLLM is up is unknown
], ids=lambda d: next(iter(d)))
def test_the_same_shortfall_is_a_wait_while_something_can_still_free_it(still):
    at_rest = dict(mem=115.5, stable=115.4, peak=115.5, needs=115.5, holds=())
    o = observed(box("box-a", rank=absent(), **{**at_rest, **still}), box("box-b", rank=absent(), holds=()))
    p = run(desired(), o)
    assert p.state in ("freeing", "draining"), (p.state, p.detail)
    assert "refused" not in p.detail and not of(p.actions, Up), p.detail


def test_the_gate_and_the_box_row_judge_one_figure_and_say_it_to_the_tenth():
    """The state line said `116 GiB free, need 116` and the box row `115.5 GiB free, needs 115.5 GiB: short 0.0 GiB`
    while the plan waited on a 0.04 GiB shortfall: two roundings of one comparison, neither showing it. Both now
    print the figure the gate compares — and a shortfall under a tenth is written to the hundredth, never as 0.0."""
    o = observed(box("box-a", rank=absent(), mem=115.46, stable=115.46, needs=115.5, holds=None),
                 box("box-b", rank=absent(), holds=None))
    p = run(desired(), o)
    assert p.state == "freeing", (p.state, p.detail)
    assert "box-a: 115.5 GiB free, need 115.5: short 0.04 GiB" in p.detail, p.detail
    row = [b for b in blockers(o.boxes["box-a"], 115.5) if b["code"] == "mem_short"]
    assert row and row[0]["message"].startswith("115.5 GiB free, needs 115.5 GiB: short 0.04 GiB"), row
    # …and a whole shortfall reads as before, to the tenth.
    o = observed(box("box-a", rank=absent(), mem=77.0, needs=103.2, holds=None), box("box-b", rank=absent()))
    p = run(desired(), o)
    assert "box-a: 77.0 GiB free, need 103.2: short 26.2 GiB" in p.detail, p.detail
