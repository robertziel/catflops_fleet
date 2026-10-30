"""Idleness is measured PER BOX.

`last_request_at` is one fleet-wide clock, so while ANY traffic flows every held lease is kept —
including one on a box that has served nothing for hours, whose gpu lane broker_parrot could be
using. The operator: "If no requests is run or queued by one server for over 10 min then
finalise system.vllm_lease assigned to that server… broker_parrot will know what to do."
"""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.scaler import Create, Release, plan_actions
from vllm_lb.state import (LEASE_QUEUED, LEASE_RUNNING, VLLM, Backend, Lease, Snapshot)

NOW = 10_000.0
CFG = Config(idle_release_s=600.0, max_leases=2)


def box(label, *, inflight=0, last_used=None):
    b = Backend(kind=VLLM, url=f"http://{label}:8000", host_label=label, box=label.split("-")[0])
    b.live, b.served_models, b.inflight = True, ("qwen3.8:latest",), inflight
    b.last_used_at = last_used
    return b


def lease_on(label, *, created_at=NOW - 5000, status=LEASE_RUNNING):
    return Lease(lease_id=f"r-{label}", job_id=f"j-{label}", created_at=created_at,
                 status=status, host_label=label, box=label.split("-")[0])


def snap(backends, leases, *, last_request_at=NOW - 5, capacity=2, last_pressure=None):
    return Snapshot(now=NOW, backends=list(backends), leases=list(leases),
                    last_request_at=last_request_at, last_pressure_at=last_pressure, db_ok=True,
                    vllm_capacity=capacity)


def test_an_idle_box_is_released_even_when_there_is_NO_surplus():
    """THE GAP. Today a lease is only given back when `want` drops below the number held —
    a fleet-wide judgement. Under sustained pressure `want` stays at the maximum, so a box
    that has served nothing for twenty minutes keeps its lease (and its gpu lane) forever."""
    s = snap([box("box-a-gpu", inflight=1, last_used=NOW - 5),
              box("box-b-gpu", last_used=NOW - 1200)],          # 20 min unused
             [lease_on("box-a-gpu"), lease_on("box-b-gpu")],
             last_pressure=NOW)                                  # sustained ⇒ want == 2
    rel = [a for a in plan_actions(s, CFG) if isinstance(a, Release)]
    assert [r.lease_id for r in rel] == ["r-box-b-gpu"], \
        f"only the idle box goes back; got {[r.lease_id for r in rel]}"


def test_a_box_serving_right_now_is_never_released():
    """The KILL line: inflight must always protect a lease."""
    s = snap([box("box-a-gpu", inflight=2, last_used=NOW - 9999)],
             [lease_on("box-a-gpu")], last_request_at=NOW - 9999)
    assert [a for a in plan_actions(s, CFG) if isinstance(a, Release)] == []


def test_a_freshly_claimed_lease_survives_its_cold_start():
    """A vLLM still loading has never served; age is measured from the claim, not from a
    last-used stamp that does not exist yet."""
    s = snap([box("box-a-gpu", last_used=None)],
             [lease_on("box-a-gpu", created_at=NOW - 30)])
    assert [a for a in plan_actions(s, CFG) if isinstance(a, Release)] == []


def test_a_lease_that_never_served_IS_released_once_past_the_window():
    s = snap([box("box-a-gpu", last_used=None)],
             [lease_on("box-a-gpu", created_at=NOW - 1200)], last_request_at=NOW - 1200)
    rel = [a for a in plan_actions(s, CFG) if isinstance(a, Release)]
    assert [r.lease_id for r in rel] == ["r-box-a-gpu"]


def test_the_per_box_rule_does_not_target_a_lease_that_has_no_box_yet():
    """A still-queued lease has no backend to be idle. Under sustained pressure (no surplus)
    the per-box rule must leave it alone; the fleet-wide surplus path owns that case."""
    s = snap([box("box-a-gpu", inflight=1, last_used=NOW - 5)],
             [lease_on("box-a-gpu"), Lease(lease_id="r-q", job_id="j-q", created_at=NOW - 1200,
                                           status=LEASE_QUEUED)],
             last_pressure=NOW)
    rel = [a for a in plan_actions(s, CFG) if isinstance(a, Release)]
    assert "r-q" not in [r.lease_id for r in rel], "no box, so box-idleness cannot apply"
