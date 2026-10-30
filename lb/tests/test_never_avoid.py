"""`avoid_box` is a hard claim FILTER in broker_parrot, not a preference — so the LB must
never send one.

2026-09-12, the operator's queue view: a lease flagged RUN-NEXT, "✗ NOT BOX_A", queued 36
minutes. The LB had set avoid_box=['box-a'] because it already held a lease there, meaning the
second one to land on the other box. But the fleet has exactly two vLLM boxes, and the other
one's single GPU lane was 6.5 hours into one job (against a 148-minute historical maximum):

    box-b-gpu  compose_region_report  running  389 min   lease still renewing
    box-b container `vllm`            Exited (0)

So the only box that could have taken the row was the one box we had excluded, and the
optimisation meant to ADD capacity removed it. Clearing avoid_box in the queue got it claimed
by box-a within seconds.

Two leases on one box are not waste: that box's single vLLM serves both, the surplus is
released by the normal idle path, and the scaler re-asks next tick. An unclaimable row is.
"""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.scaler import Create, plan_actions

from tests.test_scaler import CFG, NOW, lease, snap


def _pressed(leases):
    """Sustained pressure ⇒ the scaler wants a second lease."""
    return snap(last_request_at=NOW, last_pressure_at=NOW, leases=leases)


def test_a_second_lease_avoids_the_held_box_ONLY_while_another_candidate_exists():
    """REVERSED 2026-09-12 by pass 16 (worklog/vllm-lb-capacity-deadlock.md). `avoid_box`
    is BACK, but conditionally: it is sent only while a candidate box remains, so it can
    never exclude the last one. Pass 11's lesson is preserved as the guard, not discarded —
    see test_it_sends_NO_avoid_box_when_the_held_box_is_the_only_candidate."""
    two = _pressed([lease(1, box="box-a", host_label="box-a-gpu")])
    two.vllm_capacity = 2
    assert plan_actions(two, CFG) == [Create(avoid_boxes=("box-a",))]

    one = _pressed([lease(1, box="box-a", host_label="box-a-gpu")])
    one.vllm_capacity = 1
    for a in plan_actions(one, CFG):
        if isinstance(a, Create):
            assert a.avoid_boxes == (), "never exclude the last candidate"


def test_a_second_lease_is_still_created():
    """Dropping the hint must not drop the scale-up itself."""
    s = _pressed([lease(1, box="box-a", host_label="box-a-gpu")])
    assert [a for a in plan_actions(s, CFG) if isinstance(a, Create)], "still scales up"


def test_two_leases_on_the_same_box_are_allowed_and_do_not_loop():
    """The KILL condition: with no avoid_box, max_leases is the only thing capping growth."""
    cfg = Config(max_leases=2)
    s = _pressed([lease(1, box="box-a", host_label="box-a-gpu"),
                  lease(2, box="box-a", host_label="box-a-gpu")])
    assert not [a for a in plan_actions(s, cfg) if isinstance(a, Create)]
