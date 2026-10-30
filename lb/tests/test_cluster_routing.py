"""S2 — no lease, no traffic, with a cluster in the fleet: it is routed to only while HELD."""
from __future__ import annotations

from dataclasses import replace

from tests.fakes import FakeBackend
from vllm_lb.cluster import parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.models_view import models_payload
from vllm_lb.probe import probe_backend
from vllm_lb.router import choose
from vllm_lb.state import CLUSTER, VLLM, Backend, Lease, Snapshot

#: One cluster, "pair" (the TP head below): the default cluster an unnamed turn goes to.
CFG = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://s1:8000"),),
             cluster_name="pair",
             cluster_hosts=parse_hosts("box-a-gpu=box-a=operator@192.0.2.10=198.51.100.1,"
                                       "box-b-gpu=box-b=operator@192.0.2.11=198.51.100.2"),
             cluster_url="http://s1:8100")
TP = Backend(kind=CLUSTER, url="http://s1:8100", live=True, served_models=("Qwen/Qwen3.8-Flash-Next",),
             max_model_len=262144, cluster="pair")
CLASSIC_VLLM = Backend(kind=VLLM, url="http://s1:8000", host_label="box-a-gpu", box="box-a", live=True,
                       served_models=("qwen3.8:latest",))
HELD = {"configured": True, "held": True, "state": "running", "desired": {"mode": "cluster"}, "max_depth": 16}
STARTING = {"configured": True, "held": False, "state": "starting", "desired": {"mode": "cluster"}, "max_depth": 16}
CLASSIC = {"configured": True, "held": False, "state": "off", "desired": {"mode": "classic"}, "max_depth": 16}


def lease(kind: str, *, label: str | None = None, box: str | None = None) -> Lease:
    return Lease(lease_id=f"run-{kind}", job_id="job", created_at=0.0, kind=kind, status="running",
                 host_label=label, box=box)


def snap(*backends: Backend, leases=(), cluster=CLASSIC) -> Snapshot:
    return Snapshot(now=100.0, backends=list(backends), leases=list(leases), last_request_at=99.0,
                    last_pressure_at=None, db_ok=True,
                    extra={"cluster": cluster, "clusters": {"pair": cluster}})


def test_a_held_cluster_is_chosen():
    assert choose(snap(TP, cluster=HELD), CFG).kind == CLUSTER


def test_a_live_cluster_that_is_not_held_gets_no_traffic_not_even_in_the_scalers_counterfactual():
    s = snap(TP, cluster=STARTING)
    assert choose(s, CFG) is None
    assert choose(s, CFG, require_lease=False) is None


def test_while_the_cluster_is_wanted_no_new_request_goes_to_classic_vllm():
    s = snap(CLASSIC_VLLM, leases=(lease(VLLM, label="box-a-gpu"),), cluster=STARTING)
    assert choose(s, CFG) is None


def test_classic_mode_routes_exactly_as_before():
    s = snap(CLASSIC_VLLM, leases=(lease(VLLM, label="box-a-gpu"),))
    assert choose(s, CFG).kind == VLLM
    assert choose(replace(s, extra={}), CFG).kind == VLLM, "no cluster key at all is classic"


def test_a_held_cluster_deeper_than_its_concurrency_is_demoted_not_banned():
    deep = replace(TP, running=16, waiting=3)
    assert choose(snap(deep, cluster=HELD), CFG).kind == CLUSTER, "nothing with room: the deep one is still chosen"


def test_models_view_reports_the_cluster_model_and_window():
    body = models_payload(snap(TP, cluster=HELD), CFG)
    assert body["data"][0]["id"] == "Qwen/Qwen3.8-Flash-Next"
    assert body["data"][0]["max_model_len"] == 262144


def test_the_cluster_is_probed_over_the_vllm_wire():
    fake = FakeBackend(served=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144)
    try:
        out = probe_backend(Backend(kind=CLUSTER, url=fake.url), CFG, now=7.0)
    finally:
        fake.close()
    assert out["live"] is True and out["served_models"] == ("Qwen/Qwen3.8-Flash-Next",)
