"""A restart is not idleness.

The LB's own log after a redeploy, 31 ms apart: `recovered 1 open lease(s)` then
`released 68b9af51 (running on box-b-gpu)`, then `created vllm …` two seconds later. The
sidecar never stopped; the lease did. A fresh process has `last_request_at = None` and nothing
in flight — the exact shape of "idle past the window" — so its first tick released everything
it had just adopted. Under leased-only routing that gap is a 503.
"""
from __future__ import annotations

from tests.test_control import A, B, LIVE_V, vllm_leases, world  # noqa: F401  (fixture)
from vllm_lb.leases import RETIRED_FALLBACK_KIND, RETIRED_KEEP_PURPOSE
from vllm_lb.state import VLLM


def _held_by_a_previous_process(leases, kind=VLLM, host="box-a-gpu"):
    """What a running LB leaves behind: a claimed, running lease row in the broker — a vLLM lease a
    GB10 box claimed, or (`RETIRED_FALLBACK_KIND`) the fallback lease an LB from before its retirement held."""
    from tests.fakes import claim_gpu
    lease = leases.create(avoid_boxes=(), kind=kind)
    if kind == RETIRED_FALLBACK_KIND:
        claim_gpu("box-c-gpu", served_types=("ollama",))
    else:
        claim_gpu(host)
    return lease


def test_a_recovered_running_lease_survives_the_first_tick(world):
    ctl, hub, leases, prober, clock = world
    _held_by_a_previous_process(leases)
    prober.table[A] = LIVE_V                       # its vLLM is up, as it really was

    ctl.recover()                                  # the NEW process adopts it
    assert len(vllm_leases(hub)) == 1
    ctl.tick()                                     # first tick of the new process
    assert len(vllm_leases(hub)) == 1, \
        "a restart is not idleness: the lease it just adopted must not be released"


def test_a_recovered_lease_that_is_then_truly_unused_is_still_released(world):
    """The KILL line: stamping activity on recover must not turn into holding a box forever."""
    ctl, hub, leases, prober, clock = world
    _held_by_a_previous_process(leases)
    prober.table[A] = LIVE_V
    ctl.recover()
    ctl.tick()
    clock.t += ctl.cfg.idle_release_s + 1          # no request the whole window
    ctl.tick()
    assert vllm_leases(hub) == [], "genuinely idle for the full window ⇒ released as always"


def test_every_lease_a_restart_adopts_is_retained_not_just_one(world):
    """RETENTION is per adopted lease, not one request's worth of demand. Stamped as a request, a
    restart that held BOTH GB10 boxes wanted a single lease — pressure memory does not survive a restart —
    and released the other on its very first tick."""
    ctl, hub, leases, prober, clock = world
    _held_by_a_previous_process(leases, host="box-a-gpu")
    _held_by_a_previous_process(leases, host="box-b-gpu")
    prober.table[A] = prober.table[B] = LIVE_V

    ctl.recover()
    ctl.tick()
    assert sorted(l.box for l in vllm_leases(hub)) == ["box-a", "box-b"], \
        "a restarted LB does not release the leases it adopted"
    clock.t += ctl.cfg.idle_release_s + 1          # no request the whole window
    ctl.tick()
    assert vllm_leases(hub) == [], "unused for the full window after the adoption ⇒ both released"


def test_a_kept_lease_an_older_lb_left_is_released_never_adopted(world, broker_db):
    """Keep loaded is retired (2026-10-17): a lease an LB from before took for the operator's keep-loaded order
    (`inputs.purpose == RETIRED_KEEP_PURPOSE`, pinned with force_box) is nobody's now — adopted as demand it would
    pin its box until the idle rule; it is RELEASED on recover."""
    from queue_workflows import node_queue, run_store
    from tests.fakes import claim_gpu
    ctl, hub, leases, prober, clock = world
    run_store.insert_run(run_id="kept", workflow_name="vllm_lb_lease", out_dir=None, status="running", mode="node")
    node_queue.enqueue_node_job(run_id="kept", node_id="lease", node_module="system.vllm_lease", queue="gpu",
                                server_types=["vllm"], force_box=["box-a"],
                                inputs={"lease_id": "kept", "purpose": RETIRED_KEEP_PURPOSE, "max_hold_s": 21600})
    claim_gpu("box-a-gpu", box="box-a")
    assert [l.lease_id for l in leases.recover()] == []
    assert run_store.get_run("kept")["status"] == "cancelled"
