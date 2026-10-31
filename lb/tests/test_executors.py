"""THE TOPOLOGY AXIS (vllm_lb/executors.py, phase 5 of docs/design/engines.md): how a cluster's cards are joined is
one class per kind — one card, tensor parallel within one host, tensor parallel across hosts — and what tp-rank is told
is FACTS from the fleet file, never the DGX Spark pair's constants it was written with.

The pair's own launches are pinned byte for byte by tests/test_pair_launch_golden.py; here: the facts each kind
derives, the fleet's `link`, and tp-rank honouring a fact that is not the pair's (an ethernet link, a one-host TP,
the GPUs a rank sees, a discrete card's memory, the box's platform) — and refusing a fact that is missing.
"""
from __future__ import annotations

import pytest

from tests.fleet_files import PAIR
from tests.test_tp_rank import (GB10_PAIR_FACTS, GUARD_FRACTION, NAME, box_a, guard_polls, guarded,  # noqa: F401
                                reply, wait_until)
from vllm_lb.executors import OneCard, TpAcrossHosts, TpOneHost, executor_class, topology_of
from vllm_lb.fleet import parse_fleet

#: One box, three cards: a 3090 that serves nothing and two 5090s one engine spans.
TWO_5090 = """
panel:
  style: vibe
boxes:
  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    cards:
      - {index: 0, type: rtx3090, serves: false}
      - {index: 1, type: rtx5090, serves: true}
      - {index: 2, type: rtx5090, serves: true}
clusters: {}
backends:
  solo: {url: http://192.0.2.30:8000}
"""


def _cluster(cards, boxes, link=None, ranks=None):
    """A stand-in fleet Cluster: what `topology_of` reads (its ranks, serving cards, link)."""
    from vllm_lb.fleet import Rank
    return type("C", (), {"ranks": tuple(Rank(b, None) for b in (ranks or boxes)), "serving_cards": tuple(cards),
                          "link": link})


# ── the facts each kind derives ─────────────────────────────────────────────────────────────────────

def test_the_reference_pair_is_tp_across_hosts_and_derives_exactly_the_facts_tp_rank_was_built_for():
    fleet = parse_fleet(PAIR, "fleet.yml")
    topo = topology_of(fleet.clusters["pair"], fleet.boxes)
    assert isinstance(topo, TpAcrossHosts)
    for box in topo.boxes:
        assert topo.facts(box) == GB10_PAIR_FACTS


def test_two_cards_of_one_host_are_tp_within_one_host():
    fleet = parse_fleet(TWO_5090, "fleet.yml")
    box = fleet.boxes["solo"]
    topo = topology_of(_cluster(box.serving_cards, ["solo"], link="pcie"), fleet.boxes)
    assert isinstance(topo, TpOneHost)
    assert topo.facts(box) == {"CLUSTER_TP_SIZE": "2", "CLUSTER_NNODES": "1", "CLUSTER_GPUS": "device=1,2",
                               "CLUSTER_LINK": "pcie", "CLUSTER_MEMORY": "discrete", "CLUSTER_PLATFORM": "linux/amd64"}


def test_one_card_joins_nothing_and_names_no_link():
    fleet = parse_fleet(TWO_5090, "fleet.yml")
    box = fleet.boxes["solo"]
    topo = topology_of(_cluster(box.serving_cards[:1], ["solo"]), fleet.boxes)
    assert isinstance(topo, OneCard) and topo.facts(box)["CLUSTER_GPUS"] == "device=1"
    assert "names no link" in OneCard(box.serving_cards[:1], [box], "roce").problem()


@pytest.mark.parametrize("cls, link", [(TpOneHost, "roce"), (TpAcrossHosts, "pcie"), (TpAcrossHosts, None)])
def test_a_link_the_topology_cannot_have_is_refused_naming_the_ones_it_can(cls, link):
    fleet = parse_fleet(TWO_5090, "fleet.yml")
    box = fleet.boxes["solo"]
    why = cls(box.serving_cards, [box], link).problem()
    assert why and all(w in why for w in cls(box.serving_cards, [box], link).links)


def test_only_the_ssh_control_is_run_by_an_executor_here():
    from vllm_lb.cluster_exec import ClusterExecutor, SshExecutor
    assert executor_class("ssh") is SshExecutor and issubclass(SshExecutor, ClusterExecutor)
    assert executor_class("local") is None and executor_class("none") is None


# ── the fleet file states the link ──────────────────────────────────────────────────────────────────

def test_an_ssh_cluster_names_its_link():
    with pytest.raises(ValueError, match="link"):
        parse_fleet(PAIR.replace("    link: roce\n", ""), "fleet.yml")
    with pytest.raises(ValueError, match="roce, ethernet"):
        parse_fleet(PAIR.replace("link: roce", "link: pcie"), "fleet.yml")
    assert parse_fleet(PAIR.replace("link: roce", "link: ethernet"), "fleet.yml").clusters["pair"].link == "ethernet"


# ── tp-rank honours each fact, and refuses a missing one ────────────────────────────────────────────

def _up(box, **facts):
    return box.run("up", "0", "7", "16", "262144", config={"CLUSTER_HOST_IP": "192.0.2.201", **facts})


def _run_argv(box) -> list:
    return [c for c in box.calls() if c[0] == "run"][-1]


@pytest.mark.parametrize("missing", sorted(GB10_PAIR_FACTS))
def test_up_without_a_topology_fact_is_refused_naming_it_before_docker(box_a, missing):  # noqa: F811
    proc = _up(box_a, **{missing: ""})
    assert proc.returncode == 1 and missing in reply(proc)["error"], proc.stdout
    assert not [c for c in box_a.calls() if c[0] == "run"], "docker never ran"


def test_an_ethernet_link_opens_no_rdma_device_and_tells_nccl(box_a):  # noqa: F811
    assert _up(box_a).returncode == 0
    roce = _run_argv(box_a)
    box_a.set(containers={})
    assert _up(box_a, CLUSTER_LINK="ethernet").returncode == 0
    eth = _run_argv(box_a)
    assert "/dev/infiniband" in roce and "NCCL_IB_DISABLE=0" in roce
    assert "/dev/infiniband" not in eth and "IPC_LOCK" not in eth and "memlock=-1:-1" not in eth
    assert "NCCL_IB_DISABLE=1" in eth and "NCCL_IB_DISABLE=0" not in eth
    rdma = ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1"]
    at = roce.index("--device")
    assert roce[at:at + len(rdma)] == rdma
    expected = roce[:at] + roce[at + len(rdma):]
    assert eth == ["NCCL_IB_DISABLE=1" if w == "NCCL_IB_DISABLE=0" else w for w in expected], \
        "everything else byte for byte the roce launch"


def test_tp_within_one_host_launches_one_node_over_its_cards(box_a):  # noqa: F811
    proc = _up(box_a, CLUSTER_NNODES="1", CLUSTER_TP_SIZE="2", CLUSTER_GPUS="device=1,2", CLUSTER_LINK="pcie",
               CLUSTER_MEMORY="discrete", CLUSTER_PLATFORM="linux/amd64", TP_RANK_MACHINE="x86_64")
    assert proc.returncode == 0, proc.stdout
    run = _run_argv(box_a)
    assert run[run.index("--gpus") + 1] == "device=1,2"
    assert run[run.index("--nnodes") + 1] == "1" and run[run.index("--tensor-parallel-size") + 1] == "2"
    assert "/dev/infiniband" not in run


def test_a_rank_past_the_node_count_is_refused(box_a):  # noqa: F811
    proc = box_a.run("up", "1", "7", "16", "262144", config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_NNODES": "1"})
    assert proc.returncode == 2 and "rank must be 0..0" in reply(proc)["error"], proc.stdout


def test_a_box_whose_cpu_is_not_the_fleets_platform_is_refused(box_a):  # noqa: F811
    proc = _up(box_a, TP_RANK_MACHINE="x86_64")
    assert proc.returncode == 1 and "linux/arm64" in reply(proc)["error"] and "x86_64" in reply(proc)["error"]


def test_discrete_cards_arm_no_ram_guard_unified_ones_do(guarded):  # noqa: F811
    from tests.test_tp_rank import guard_running
    proc = _up(guarded, CLUSTER_GUARD="1", CLUSTER_RAM_USABLE_FRACTION=GUARD_FRACTION, CLUSTER_MEMORY="discrete")
    assert proc.returncode == 0 and reply(proc)["action"] == "started", proc.stdout
    assert not guard_running(guarded), "a discrete card's VRAM is not this box's RAM: no RAM guard watches it"
