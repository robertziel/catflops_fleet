"""Do not ask broker_parrot for something no box can give.

A lease is a REQUEST FOR A vLLM. When every configured GB10 box is parked
(`worker_controls.desired_state = 'off'`) no box will ever start one, so the row would sit
queued and broker_parrot's stranded sweep would rightly flag it
`unassignable: no live worker serves any of: vllm`. So the right move is to not
enqueue at all, and to say so. When a box is unparked, leasing resumes on the next tick.

(Putting 'ollama' in the lease's own server_types is NOT the fix: an ollama box would claim it
at once, the row would leave the queue, the vLLM demand counter — which counts QUEUED rows —
would drop to zero, and no vLLM would ever start. Pinned by the last test here.)
"""
from __future__ import annotations

import pytest

from tests.fakes import FakeProber
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.leases import BrokerLeases
from vllm_lb.scaler import desired_leases
from vllm_lb.state import Backend, Snapshot

def vllm_leases(hub):
    """vLLM leases only, and these tests are about parking vLLM boxes."""
    from vllm_lb.state import VLLM
    return [l for l in hub.leases() if l.kind == VLLM]


A, B = "http://10.0.0.1:8000", "http://10.0.0.2:8000"
CFG = Config(broker_enabled=True,
             vllm_boxes=(VllmBox("box-a-gpu", "box-a", A), VllmBox("box-b-gpu", "box-b", B)),
             idle_release_s=600, probe_interval_s=0)
NOW = 500.0


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def _snap(*, boxes_on: bool, last_request_at=NOW - 1):
    return Snapshot(now=NOW,
                    backends=[Backend(kind="vllm", url=A, host_label="box-a-gpu", box="box-a"),
                              Backend(kind="vllm", url=B, host_label="box-b-gpu", box="box-b")],
                    leases=[], last_request_at=last_request_at, last_pressure_at=None,
                    db_ok=True, boxes_on=boxes_on)


def _control(host_label: str, state: str):
    from queue_workflows import worker_control
    worker_control.set_worker_control(host_label, "gpu", desired_state=state,
                                      requested_by="test")


@pytest.fixture
def world(broker_db):
    clock = Clock()
    hub = Hub(CFG, clock=clock)
    leases = BrokerLeases(CFG, clock=clock)
    ctl = Controller(CFG, hub, leases, prober=FakeProber({}), clock=clock)
    return ctl, hub, leases, clock


# ── the pure half ────────────────────────────────────────────────────────────

def test_a_parked_fleet_wants_no_lease_however_recent_the_request():
    assert desired_leases(_snap(boxes_on=False), CFG) == 0


def test_an_unparked_fleet_wants_one():
    assert desired_leases(_snap(boxes_on=True), CFG) == 1


# ── reading the control plane ────────────────────────────────────────────────

def test_all_boxes_off_is_reported_with_the_states(world):
    _, _, leases, _ = world
    _control("box-a-gpu", "off")
    _control("box-b-gpu", "off")
    on, detail = leases.vllm_boxes_on()
    assert on is False and "box-a-gpu=off" in detail and "box-b-gpu=off" in detail


def test_one_box_on_is_enough(world):
    _, _, leases, _ = world
    _control("box-a-gpu", "off")
    _control("box-b-gpu", "on")
    on, _ = leases.vllm_boxes_on()
    assert on is True


def test_a_box_with_no_control_row_counts_as_on(world):
    """The engine's own rule: a worker absent from worker_controls is ON."""
    _, _, leases, _ = world
    on, detail = leases.vllm_boxes_on()
    assert on is True and "no row" in detail


def test_an_unreadable_control_plane_fails_open(world, monkeypatch):
    import queue_workflows as qw
    from queue_workflows import db
    _, _, leases, _ = world
    db.close_pool()
    monkeypatch.setenv("LB_TEST_DB_DOWN", "/nonexistent/dir/broker.db")
    qw.configure(db_backend="sqlite", db_url_env="LB_TEST_DB_DOWN")
    on, detail = leases.vllm_boxes_on()
    assert on is True and "unknown" in detail, "a control-plane blip must not stop leasing"


# ── the loop ─────────────────────────────────────────────────────────────────

def test_no_lease_is_enqueued_while_every_box_is_parked(world):
    from queue_workflows import node_queue
    ctl, hub, _, _ = world
    _control("box-a-gpu", "off")
    _control("box-b-gpu", "off")
    hub.note_request()
    ctl.tick()
    assert vllm_leases(hub) == []
    assert node_queue.count_pending_for_server(None, "vllm", any_model=True) == 0, \
        "nothing queued means nothing for the stranded sweep to flag"
    assert "off" in hub.snapshot().extra["boxes_detail"]


def test_unparking_a_box_resumes_leasing_on_the_next_tick(world):
    ctl, hub, _, _ = world
    _control("box-a-gpu", "off")
    _control("box-b-gpu", "off")
    hub.note_request()
    ctl.tick()
    assert vllm_leases(hub) == []
    _control("box-b-gpu", "on")
    ctl.tick()
    assert len(vllm_leases(hub)) == 1


def test_parking_the_fleet_releases_a_lease_we_already_hold(world):
    ctl, hub, leases, _ = world
    hub.note_request()
    ctl.tick()
    (lease,) = vllm_leases(hub)
    _control("box-a-gpu", "off")
    _control("box-b-gpu", "off")
    ctl.tick()
    assert vllm_leases(hub) == []
    from queue_workflows import run_store
    assert run_store.get_run(lease.lease_id)["status"] == "cancelled"


def test_the_lease_never_offers_ollama_as_a_fallback_kind(world):
    """server_types stays ['vllm'] BY DESIGN: an ollama box would claim the row instantly,
    it would leave the queue, and the vLLM demand counter (queued rows only) would read 0."""
    from queue_workflows import node_queue
    from tests.fakes import claim_gpu
    ctl, hub, _, _ = world
    hub.note_request()
    ctl.tick()
    (lease,) = vllm_leases(hub)
    assert node_queue.get_node_job(lease.job_id)["server_types"] == ["vllm"]
    assert claim_gpu("box-c-gpu", served_types=("ollama",)) is None, \
        "an ollama-only box must not be able to take the vLLM request off the queue"
