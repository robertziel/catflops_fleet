"""S2 — the scaler is PURE: desired lease count + the actions to get there.

N=0 after idle_release_s without an inference request; else N>=1; N=2 while a leased live
vLLM showed sustained queueing pressure within the last idle_release_s. Extra leases avoid the
boxes already leased; surplus leases are released unclaimed-first.
"""
from vllm_lb.config import Config, VllmBox
from vllm_lb.scaler import Create, PressureTracker, Release, desired_leases, plan_actions
from vllm_lb.state import Backend, Lease, Snapshot

NOW = 5_000.0
CFG = Config(idle_release_s=600, max_leases=2, scale_up_waiting=2, scale_up_sustain_s=30,
             vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://192.0.2.10:8000"),
                         VllmBox("box-b-gpu", "box-b", "http://192.0.2.11:8000")))


def boxes(n=2):
    labels = ["box-a-gpu", "box-b-gpu"][:n]
    return [Backend(kind="vllm", url=f"http://{l}:8000", host_label=l, box=l.removesuffix("-gpu"))
            for l in labels]


def snap(*, last_request_at, last_pressure_at=None, leases=(), backends=None, now=NOW,
         vllm_capacity=None):
    return Snapshot(now=now, backends=backends if backends is not None else boxes(),
                    leases=list(leases), last_request_at=last_request_at,
                    last_pressure_at=last_pressure_at, db_ok=True,
                    vllm_capacity=vllm_capacity)


def lease(i, status="running", box=None, host_label=None):
    return Lease(lease_id=f"r{i}", job_id=f"j{i}", created_at=NOW - 100 * i, status=status,
                 host_label=host_label, box=box)


# ── desired count ────────────────────────────────────────────────────────────

def test_no_request_ever_means_zero():
    assert desired_leases(snap(last_request_at=None), CFG) == 0


def test_a_recent_request_means_one():
    assert desired_leases(snap(last_request_at=NOW - 1), CFG) == 1


def test_idle_past_the_release_window_means_zero():
    assert desired_leases(snap(last_request_at=NOW - 601), CFG) == 0
    assert desired_leases(snap(last_request_at=NOW - 600), CFG) == 1


def test_a_request_still_in_flight_is_activity_even_past_the_window():
    """Found live: the idle clock starts at ARRIVAL, so a generation streaming longer than
    idle_release_s would have its lease released mid-stream. In-flight = active."""
    a, b = boxes()
    a.inflight = 1
    assert desired_leases(snap(last_request_at=NOW - 900, backends=[a, b]), CFG) == 1


def test_recent_pressure_means_two():
    assert desired_leases(snap(last_request_at=NOW - 1, last_pressure_at=NOW - 10), CFG) == 2


def test_stale_pressure_falls_back_to_one():
    assert desired_leases(snap(last_request_at=NOW - 1, last_pressure_at=NOW - 601), CFG) == 1


def test_capped_by_max_leases_and_by_configured_boxes():
    s = snap(last_request_at=NOW - 1, last_pressure_at=NOW - 1)
    assert desired_leases(s, Config(idle_release_s=600, max_leases=1)) == 1
    s1 = snap(last_request_at=NOW - 1, last_pressure_at=NOW - 1, backends=boxes(1))
    assert desired_leases(s1, CFG) == 1


# ── actions ──────────────────────────────────────────────────────────────────

def test_first_request_creates_one_lease_with_nothing_to_avoid():
    assert plan_actions(snap(last_request_at=NOW), CFG) == [Create(avoid_boxes=())]


def test_one_create_per_tick_even_when_two_are_wanted():
    s = snap(last_request_at=NOW, last_pressure_at=NOW)
    assert plan_actions(s, CFG) == [Create(avoid_boxes=())]


def test_second_lease_avoids_the_leased_box_when_another_candidate_exists():
    """Changed 2026-09-12 (worklog/vllm-lb-never-avoid.md): avoid_box is a hard claim FILTER,
    and steering the 2nd lease off the box we hold stranded it for 36 min while the only other
    vLLM box sat 6.5 h into one job. See tests/test_never_avoid.py."""
    s = snap(last_request_at=NOW, last_pressure_at=NOW, vllm_capacity=2,
             leases=[lease(1, box="box-a", host_label="box-a-gpu")])
    assert plan_actions(s, CFG) == [Create(avoid_boxes=("box-a",))]


def test_a_queued_unclaimed_lease_counts_as_active_so_no_duplicate_is_created():
    s = snap(last_request_at=NOW, leases=[lease(1, status="queued")])
    assert plan_actions(s, CFG) == []


def test_idle_releases_every_active_lease():
    s = snap(last_request_at=NOW - 601,
             leases=[lease(1, box="box-a"), lease(2, status="queued")])
    assert plan_actions(s, CFG) == [Release(lease_id="r2"), Release(lease_id="r1")]


def test_surplus_releases_the_unclaimed_lease_first():
    s = snap(last_request_at=NOW, leases=[lease(1, box="box-a"), lease(2, status="queued")])
    assert plan_actions(s, CFG) == [Release(lease_id="r2")]


def test_surplus_among_running_releases_the_least_loaded_backend():
    a, b = boxes()
    a.inflight, b.inflight = 3, 0
    s = snap(last_request_at=NOW, backends=[a, b],
             leases=[lease(1, box="box-a", host_label="box-a-gpu"),
                     lease(2, box="box-b", host_label="box-b-gpu")])
    assert plan_actions(s, CFG) == [Release(lease_id="r2")]


def test_lost_leases_are_ignored_by_the_planner():
    s = snap(last_request_at=NOW, leases=[lease(1, status="lost")])
    assert plan_actions(s, CFG) == [Create(avoid_boxes=())]


# ── pressure tracker (sustain) ───────────────────────────────────────────────

def test_pressure_must_be_sustained_before_it_counts():
    t = PressureTracker(threshold=2, sustain_s=30)
    assert t.observe(now=0, waiting=5) is None
    assert t.observe(now=29, waiting=5) is None
    assert t.observe(now=30, waiting=5) == 30
    assert t.observe(now=31, waiting=5) == 31


def test_a_dip_resets_the_sustain_clock():
    t = PressureTracker(threshold=2, sustain_s=30)
    t.observe(now=0, waiting=5)
    t.observe(now=20, waiting=0)
    assert t.observe(now=40, waiting=5) is None
    assert t.observe(now=70, waiting=5) == 70


def test_last_pressure_is_remembered_after_it_subsides():
    t = PressureTracker(threshold=2, sustain_s=0)
    assert t.observe(now=10, waiting=2) == 10
    assert t.observe(now=50, waiting=0) == 10
