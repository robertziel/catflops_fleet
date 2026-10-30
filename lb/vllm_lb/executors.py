"""THE TOPOLOGY AXIS — how a cluster's cards are joined, one class per kind, and the facts its executor is told
(docs/design/engines.md, phase 5).

A cluster's TOPOLOGY is what the fleet file says about its cards and hosts — derived, never typed twice:

  OneCard        one card on one host: TP=1, nothing to join
  TpOneHost      several cards of ONE host, one engine across them (TP = the host's serving cards, one node): joined
                 inside the box — `pcie` or `nvlink`
  TpAcrossHosts  one rank on each of several hosts, each over that host's serving cards (TP = hosts × cards a host,
                 NNODES = hosts): joined between boxes — `roce` (NCCL over RDMA) or `ethernet` (NCCL sockets)

`topology_of(cluster)` picks the class from the cluster's ranks and serving cards (data, never a hardware name);
`Topology.facts` are what lb/cluster/tp-rank receives instead of the DGX Spark pair's constants it was written with —
the TP size, the node count, the GPUs a rank sees, the link, the cards' memory family (read through the hardware
family's own name, never branched on here), the box's platform. The GB10 pair's facts launch it byte for byte as before
(tests/test_pair_launch_golden.py).

THE EXECUTOR is what runs a cluster's verbs (status, availability, up, halt, down, logs, heal): `ClusterExecutor`, the
interface (cluster_exec.py); `SshExecutor`, tp-rank over SSH, the one the LB drives today (`EXECUTORS`). A one-host
node the LB REQUESTS is run by its host's applier (node_control.py, deploy/standalone), not by an executor here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: The links each kind of topology may name (the fleet file's `clusters.<name>.link`).
LINKS_ACROSS_HOSTS = ("roce", "ethernet")
LINKS_ONE_HOST = ("pcie", "nvlink")


@dataclass(frozen=True)
class Topology:
    """How one cluster's serving cards are joined. `cards`: fleet Cards, head first; `boxes`: its rank boxes, head
    first; `link`: the fleet's word for the join (None where nothing is joined)."""

    cards: tuple
    boxes: tuple
    link: str | None

    #: The links this kind may name; () = it joins nothing.
    links: tuple[str, ...] = ()
    kind: str = ""

    @property
    def tp(self) -> int:
        return len(self.cards)

    @property
    def nnodes(self) -> int:
        return len(self.boxes)

    def problem(self) -> str:
        """Why this topology cannot be run as the fleet states it ("" when it can)."""
        if self.links and self.link not in self.links:
            return (f"a {self.kind} cluster names its link: one of {', '.join(self.links)} (the fleet file's "
                    f"`link`), not {self.link!r}")
        if not self.links and self.link is not None:
            return f"a {self.kind} cluster joins no cards: it names no link, not {self.link!r}"
        per_host = {b.name: sum(1 for c in self.cards if c.box == b.name) for b in self.boxes}
        if len(set(per_host.values())) > 1:
            return (f"its hosts serve on different numbers of cards ({per_host}): one engine runs the same "
                    f"number of GPUs on every rank")
        return ""

    def gpus(self, box) -> str:
        """docker's --gpus for `box`'s rank: `all` where the rank serves on every card the box has, else the
        serving cards by index — never a card the fleet does not name as serving."""
        serving = [c.index for c in self.cards if c.box == box.name]
        return "all" if len(serving) == len(box.cards) else "device=" + ",".join(map(str, serving))

    def facts(self, box) -> dict[str, str]:
        """What tp-rank is told for `box`'s rank (tp-rank's `need_topology`): every fact, none defaulted there."""
        why = self.problem()
        if why:
            raise ValueError(why)
        return {"CLUSTER_TP_SIZE": str(self.tp), "CLUSTER_NNODES": str(self.nnodes), "CLUSTER_GPUS": self.gpus(box),
                "CLUSTER_LINK": self.link, "CLUSTER_MEMORY": self.cards[0].type.memory.name,
                "CLUSTER_PLATFORM": box.platform} if self.links else self._one_card(box)

    def _one_card(self, box) -> dict[str, str]:
        return {"CLUSTER_TP_SIZE": "1", "CLUSTER_NNODES": "1", "CLUSTER_GPUS": self.gpus(box),
                "CLUSTER_LINK": "pcie", "CLUSTER_MEMORY": self.cards[0].type.memory.name,
                "CLUSTER_PLATFORM": box.platform}


class OneCard(Topology):
    """One card on one host: nothing joined. Its link fact is `pcie` — the card's own bus, no peer, no RDMA."""

    def __init__(self, cards, boxes, link):
        super().__init__(tuple(cards), tuple(boxes), link, (), "one-card")


class TpOneHost(Topology):
    """Several cards of ONE host, one engine across them (vLLM/SGLang tensor parallel within a box)."""

    def __init__(self, cards, boxes, link):
        super().__init__(tuple(cards), tuple(boxes), link, LINKS_ONE_HOST, "tensor-parallel-within-one-host")


class TpAcrossHosts(Topology):
    """One rank on each of several hosts (the DGX Spark pair: two hosts, one GB10 each, RoCE)."""

    def __init__(self, cards, boxes, link):
        super().__init__(tuple(cards), tuple(boxes), link, LINKS_ACROSS_HOSTS, "tensor-parallel-across-hosts")


def topology_of(cluster, boxes: Mapping) -> Topology:
    """The topology a fleet `Cluster` makes on `boxes` (the fleet's, by name): picked by how many hosts its ranks are
    on and how many cards serve — data, never a card type or a control name."""
    rank_boxes = tuple(boxes[r.box] for r in cluster.ranks)
    cards = tuple(cluster.serving_cards)
    link = getattr(cluster, "link", None)
    if len(rank_boxes) > 1:
        return TpAcrossHosts(cards, rank_boxes, link)
    if len(cards) > 1:
        return TpOneHost(cards, rank_boxes, link)
    return OneCard(cards, rank_boxes, link)


def executor_class(control: str):
    """The `ClusterExecutor` class that runs a cluster of `control`, or None for a control whose cluster the LB does
    not run itself (a one-host node: its host's applier; an observed one: nothing)."""
    from vllm_lb.cluster_exec import SshExecutor
    return {"ssh": SshExecutor}.get(control)
