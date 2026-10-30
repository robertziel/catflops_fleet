"""A CLUSTER IS ROUTED BY ITS OWN SEATS — the pair by the concurrency its Start set (`desired.max_num_seqs`), a node
this LB only observes by the seats its fleet file declares (`launch.max_num_seqs`) — through one code path: the
cluster view's `max_depth`, which router.py `busy` holds a CLUSTER backend's depth against (demoted past it, never
banned). LB_VLLM_MAX_DEPTH (`vllm_max_depth`) is the CLASSIC boxes' limit only: it never caps a cluster, and a node
whose seats nobody declared has no limit to be within, so it counts as busy — as an unknown depth does — never as
`vllm_max_depth`'s two slots (operator, 2026-10-11)."""
from __future__ import annotations

from dataclasses import replace

from tests.test_cluster_declared import declared_fleet
from tests.test_cluster_one_host import LIVE, NODE, Node, node_cfg
from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.router import choose
from vllm_lb.state import CLUSTER, Backend, Snapshot

#: Another held server of the same model, DEEPER (20) but within its own 64 seats: a head within its seats beats it on
#: depth, a demoted one loses to it — what "demoted" is seen against now no fallback tier exists.
ROOMY = {"configured": True, "held": True, "state": "running", "desired": {"mode": "cluster"}, "max_depth": 64}
OTHER = Backend(kind=CLUSTER, url="http://other:8100", live=True, served_models=("m",), max_model_len=131072,
                running=20, depth_known=True, cluster="other")
#: A cluster backend `running` deep, its depth known.
def head(url: str, running: int) -> Backend:
    return Backend(kind=CLUSTER, url=url, live=True, served_models=("m",), max_model_len=131072, running=running,
                   depth_known=True, cluster="node")


def routed(cluster: dict, backend: Backend, cfg: Config) -> str:
    """The cluster a turn naming `m` lands on: `backend`'s ("node") or the deeper, roomy OTHER ("other")."""
    snap = Snapshot(now=100.0, backends=[backend, OTHER], leases=[], last_request_at=99.0, last_pressure_at=None,
                    db_ok=True, extra={"cluster": cluster, "clusters": {"node": cluster, "other": ROOMY}})
    return choose(snap, cfg, model="m").cluster


# ── the pair: unchanged ─────────────────────────────────────────────────────────────────────────


PAIR = {"configured": True, "held": True, "state": "running", "desired": {"mode": "cluster"}, "max_depth": 16}


def test_the_pair_is_held_to_its_own_concurrency_and_never_to_the_classic_limit():
    cfg = node_cfg(broker_enabled=False, vllm_max_depth=2)   # one cluster ("node"): the default an unnamed turn takes
    assert routed(PAIR, head("http://pair:8100", 16), cfg) == "node", "16 of 16 seats: within"
    assert routed(PAIR, head("http://pair:8100", 17), cfg) == "other", "one past its seats: demoted"


# ── the node: its declared seats ────────────────────────────────────────────────────────────────


def test_a_node_this_lb_observes_is_held_to_the_seats_its_fleet_declares():
    node = Node(LIVE, fleet=declared_fleet(), vllm_max_depth=2)
    view = node.tick()
    assert view["max_depth"] == 4, "launch.max_num_seqs"
    cfg = replace(node.cfg)
    assert routed(view, head(NODE, 4), cfg) == "node", "4 of its 4 seats: within — LB_VLLM_MAX_DEPTH=2 plays no part"
    assert routed(view, head(NODE, 5), cfg) == "other", "past its seats: demoted below a server with room"


def test_the_pre_tick_view_already_carries_the_declared_seats():
    assert Hub(node_cfg(fleet=declared_fleet())).cluster_state()["max_depth"] == 4


def test_a_node_whose_seats_nobody_declared_has_no_limit_to_be_within_and_counts_as_busy():
    view = Node(LIVE, fleet=declared_fleet(launch=False), vllm_max_depth=2).tick()
    assert view["max_depth"] is None, "no seats declared: none claimed — never the classic boxes' limit"
    assert routed(view, head(NODE, 0), node_cfg(vllm_max_depth=2)) == "other", "demoted (not banned), as unknown depth"
    alone = Snapshot(now=100.0, backends=[head(NODE, 0)], leases=[], last_request_at=99.0,
                           last_pressure_at=None, db_ok=True, extra={"cluster": view, "clusters": {"node": view}})
    assert choose(alone, node_cfg(vllm_max_depth=2)).kind == CLUSTER, "…and still chosen when it is all there is"
