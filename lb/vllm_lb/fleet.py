"""THE FLEET — what one deployment IS, read from its fleet file and validated before anything uses it.

AGENTS.md, operator rule 2026-10-11: a deployment is configuration, never code. Which boxes exist, every card in
each, which cards serve, which clusters run across them and which plain servers the LB dials are a FLEET FILE
(lb/fleet.yml, gitignored like lb/.env; lb/fleet.example.yml and lb/fleet.one-host.example.yml document every key),
mounted read-only into the LB at LB_FLEET. One loader reads it and the hardware types it names, and REFUSES the
start — naming the file and the value — on anything missing, unknown or inconsistent. No defaults, no
auto-detection: a value the file does not state is never filled in.

  boxes:     name → hostname (as the hardware sender reports it), platform (linux/amd64 | linux/arm64: the CPU its
             images are built for), label (its GPU lane: what its servers are called, what a broker claim stamps),
             ssh (how the LB reaches it — for a cluster it drives), and EVERY card: nvidia-smi index, hardware type
             (vllm_lb/hardware/types/<type>.yml), serves true/false.
             A box with two different cards lists both; a card someone else uses is listed with serves: false.
             kv_offload_ram_max_gib (optional): the most GiB of the box's RAM a KV tier of a node on it may take
             (`_ram_tier`) — a SHARED box's bound, stated; absent = no tier there, never a size assumed.
  clusters:  name → control (`CONTROLS`: `ssh` — the tp-rank pair the LB drives; `local` — a one-host node the LB
             requests and its own host applies; `none` — a node it only observes), url (the head rank's API), ranks
             HEAD FIRST (box, and for ssh its interconnect address), run_at_boot (true/false; required when several
             are declared). The clusters are a CATALOG of what the fleet can run: two may share a box — as
             alternatives, or at once where their memory needs fit its pool (`share_problem`) — and the run
             (cluster_run.py) is which of them this LB runs now.
             Its tensor-parallel size is DERIVED — the serving cards under its ranks — never typed beside them.
  backends:  box → url: a plain classic server, on that box's serving cards. One per box: a server is named by its
             box's label.

A one-host node (control local or none) is started by its own host's deployment, not by tp-rank, so the LB cannot
read its launch off a launch record of its own — the fleet names it, once:
  model:     `<catalog id>[@<variant>]`, validated against the model catalog (vllm_lb/library, assembled by catalog.document). Required for
             control local and none; refused for control ssh, whose model is the operator's Start on the panel.
  launch:    image (pinned by digest), served_name, max_model_len, max_num_seqs, gpu_memory_utilization,
             reasoning_parser, tool_call_parser — every key when the block is present, no other but the OPTIONAL
             ones, each absent = vLLM's own, never a number of ours: max_num_batched_tokens (vLLM's prefill chunk,
             `_chunk`), kv_cache_memory (the KV pool pinned in bytes, `_pin`) and kv_offload_ram_gib (a KV tier in the
             host's RAM, `_ram`, bounded by the box's kv_offload_ram_max_gib) and kv_cache_dtype (the KV cache's
             element type, `_kv_dtype`, one the IMAGE accepts); within the model's own catalog limits. Required for control local: its BOOT DEFAULT, which instead of a served_name lists
             `choices` — the models the panel may select, the boot `model` among them, each a catalog row whose
             weights fit the serving card (`_fit`); a choice's served name and checkpoint folder are its row's. The
             panel's intent moves the model and the figures (`MOVABLE`) — `validate_launch`, the one validator for the
             block and every intent, which the host's applier runs too. `python -m vllm_lb.fleet node-env <file> <cluster> [<intent.json>]` renders it,
             with the GPU from the cluster's serving card, as the KEY=VALUE lines a standalone node's compose reads.

Every PEAK, CORE COUNT and MEMORY MODEL the gateway shows comes from the cards that SERVE — `Cluster.serving_cards`,
`Box.serving_cards` — never from "the busiest card" and never from one LB-wide setting (operator, 2026-10-11).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import sys

from vllm_lb import kv_dtype, yamlfile
from vllm_lb.hardware import PLATFORMS, HardwareProfile, compatibility, profile, types


@dataclass(frozen=True)
class Control:
    """A way a cluster is run, and what it therefore needs from the fleet — DATA, so each rule below reads a field of
    the cluster's control and no rule asks which control it is. `ranks`/`cards_per_rank`: what its executor runs,
    None = whatever is named. `ssh`: whether each rank's box names `ssh` (how the LB reaches it): "required",
    "optional" or "forbidden". `interconnect`: each rank names the address its ranks talk over. `model`/`launch`: the
    same three words for the cluster's keys — forbidden where the LB's own Start names them. `drives`: the LB asks for
    its starts and stops (each driven cluster under its own name: worklog/parallel-clusters.md). `selects`: the panel selects its model among the launch block's
    `choices` — each choice's served name and checkpoint folder then its catalog row's, never typed; without it the
    block names the one served name its own deployment uses. `reserves`: with the broker on, its boxes are HELD through a
    broker_parrot reservation per box before anything starts there (the pair's contract, `leases.reserve_for_cluster`
    — a one-host node on a box the broker's own lane serves too), so neither side loads onto the other's card. `frees`:
    its start first frees the boxes' RAM of the classic servers (the pair's `freeing`). `contained`: its executor runs
    the engine as a container on a box others share (deploy/standalone's node), so that box states the container's
    limits (`node_limits`) — never defaulted. `ram_usable`: the cluster states `ram_usable_fraction`, the share of each
rank box's MemTotal a start may take (tp-rank's start gate and memory guard) — "required" or "forbidden". A card whose type does not
    list the control among its `controls` is not one it runs — `none` runs nothing, so any card."""

    name: str
    ranks: int | None
    cards_per_rank: int | None
    executor: str
    ssh: str
    interconnect: bool
    model: str
    launch: str
    drives: bool
    selects: bool
    reserves: bool
    frees: bool
    contained: bool
    ram_usable: str = "forbidden"
    #: The cluster names `iface`, the NIC its ranks bind NCCL/Gloo on (tp-rank's CLUSTER_IFACE): "required" or "forbidden".
    iface: str = "forbidden"
    #: The cluster may list `choices`, the catalog rows its panel offers (`Cluster.offers`): "optional" or "forbidden".
    #: A local node's live under its `launch` block instead (`selects`).
    offers: str = "forbidden"
    #: The cluster states `max_num_batched_tokens`, the prefill budget its ranks launch with when neither the operator's
    #: knob nor the model's own says (vLLM --max-num-batched-tokens; tp-rank's CLUSTER_MAX_NUM_BATCHED_TOKENS):
    #: "required" or "forbidden". A node's chunk is its launch block's own (`CHUNK`, optional there).
    prefill_budget: str = "forbidden"

    def runs(self, card_type) -> bool:
        return not self.drives or self.name in card_type.controls


#: How a cluster may be controlled:
#:   ssh    lb/cluster/tp-rank over SSH: one rank per box on that box's one GPU, TP_SIZE=2, each box's SSH target and
#:          interconnect; its model is the operator's Start on the panel, so the fleet names no model or launch.
#:   local  a one-host node the LB REQUESTS and its own host APPLIES: the LB writes an intent into its state dir, a
#:          systemd applier on the host validates it (`validate_launch`, the launch block's own validator) and runs
#:          deploy/standalone/up.sh node. One rank on one card; no SSH, no peer; its model and its launch (the boot
#:          default the intent moves within) are the fleet's.
#:   none   a node this LB only observes — its own deployment starts it; its model is the fleet's, its launch may be.
CONTROLS: Mapping[str, Control] = MappingProxyType({
    "ssh": Control("ssh", ranks=2, cards_per_rank=1, executor="lb/cluster/tp-rank, over SSH", ssh="required",
                   interconnect=True, model="forbidden", launch="forbidden", drives=True, selects=False,
                   reserves=True, frees=True, contained=False, ram_usable="required", iface="required",
                   offers="optional", prefill_budget="required"),
    "local": Control("local", ranks=1, cards_per_rank=1,
                     executor="its host's applier (an intent in the LB's state dir; deploy/standalone/up.sh node)",
                     ssh="forbidden", interconnect=False, model="required", launch="required", drives=True,
                     selects=True, reserves=True, frees=False, contained=True),
    "none": Control("none", ranks=None, cards_per_rank=None, executor="nothing: its own deployment starts it",
                    ssh="optional", interconnect=False, model="required", launch="optional", drives=False,
                    selects=False, reserves=False, frees=False, contained=False),
})


@dataclass(frozen=True)
class Card:
    """One GPU of a box: nvidia-smi's `index` there, its hardware `type`, and whether a server of this fleet runs on
    it (`serves`)."""

    box: str
    index: int
    type: HardwareProfile
    serves: bool

    def view(self) -> dict:
        return {"box": self.box, "index": self.index, "type": self.type.key, "name": self.type.name,
                "serves": self.serves}


@dataclass(frozen=True)
class Box:
    name: str
    hostname: str
    label: str | None
    ssh: str | None
    cards: tuple[Card, ...]
    #: The CPU platform its images must be built for (`linux/amd64` | `linux/arm64`) — named, never inferred: the
    #: 5090 node needs an x86 image where the GB10 pair's are arm64. Agrees with every card type that states one.
    platform: str
    #: The most GiB of its RAM a KV tier of a node on it may take (`_ram_tier`); None = the fleet states none, so no
    #: tier runs there. A shared box's bound — and at most the host's /dev/shm, where vLLM maps the tier.
    kv_offload_ram_max_gib: int | None = None
    #: The most GiB of its RAM a node's GPU POOL on it may take, where the pool IS the box's RAM (a unified-memory card:
    #: `MemoryModel.pool_problem`) — the operator's bound for a box whose RAM the LB, the chat and the broker share.
    #: None = the fleet states none: a unified box then runs no node launch (refused naming it); a discrete one ignores it.
    gpu_ram_max_gib: int | None = None
    #: The limits of a node container on it (`NODE_LIMITS`: memory_gib, cpus, pids, nice) — required where a contained
    #: node runs (Control.contained), None elsewhere. The measured gui run's guards on a box the LB, chat and broker share.
    node_limits: "NodeLimits | None" = None

    @property
    def serving_cards(self) -> tuple[Card, ...]:
        return tuple(c for c in self.cards if c.serves)


@dataclass(frozen=True)
class Rank:
    box: str
    interconnect: str | None


@dataclass(frozen=True)
class ClusterModel:
    """The catalog row a cluster serves (`spec`, a vllm_lb.catalog ModelSpec) and the variant of its checkpoint
    (`variant`: None = the original, "uncensored" = its twin)."""

    spec: object
    variant: str | None

    @property
    def ref(self) -> str:
        """As the fleet file writes it: `<catalog id>[@<variant>]`."""
        return f"{self.spec.id}@{self.variant}" if self.variant else self.spec.id

    @property
    def checkpoint(self) -> str:
        """The folder its engine loads, under a models directory (MODELS_SEARCH_DIRS; a standalone node's
        EDGE_MODELS_DIR): the catalog's path for this variant — the one place it is written."""
        return self.spec.launched_from(self.variant is not None).path


@dataclass(frozen=True)
class NodeLaunch:
    """How a one-host node is launched by its own host (the fleet's `launch:` block, or an intent moved from it)."""

    image: str
    served_name: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    #: vLLM's prefill chunk (--max-num-batched-tokens); None = not set, vLLM's own default for the card.
    max_num_batched_tokens: int | None
    #: The KV pool PINNED, in bytes (--kv-cache-memory-bytes); None = vLLM sizes it from gpu_memory_utilization.
    kv_cache_memory: int | None
    #: A KV tier in the host's RAM, GiB (--kv-offloading-size, native backend); None = no tier.
    kv_offload_ram_gib: int | None
    #: The KV cache's element type (--kv-cache-dtype), one the image accepts; None = vLLM's own (auto), no flag.
    kv_cache_dtype: str | None
    #: vLLM's reasoning parser (--reasoning-parser); None = the block names none — a model with no thinking format,
    #: whose answers a parser would move into `reasoning` (MAI-UI-2B, 2026-10-17: every `content` empty) — no flag.
    reasoning_parser: str | None
    #: vLLM's tool-call parser (--enable-auto-tool-choice --tool-call-parser) — a CHAT cluster's, required there (the
    #: chat sends tools with tool_choice auto, which vLLM refuses without it); None for a transcription one: no flag.
    tool_call_parser: str | None
    #: The server it runs (`ENGINES`): its launch choice's catalog `node_engine` — which image it is, how it fits.
    engine: str
    #: faster-whisper's: realtime requests queued when every seat is busy (`QUEUE`); None = the service's own (a busy
    #: realtime request is answered 503 at once).
    max_queue: int | None = field(default=None, kw_only=True)
    #: faster-whisper's: CTranslate2's compute type (`COMPUTE`); None = the image's own.
    compute_type: str | None = field(default=None, kw_only=True)

    def model_lines(self) -> list[str]:
        """The node-env lines naming the checkpoint it loads — none here: its folder is its deployment's own wiring."""
        return []


@dataclass(frozen=True)
class SelectedLaunch(NodeLaunch):
    """A launch whose model is SELECTED among the fleet's `choices` (control local): it also names that model
    (`model`, its ref) and the folder it loads (`checkpoint`, the catalog's), and its served name is the catalog
    row's."""

    model: str
    checkpoint: str

    def model_lines(self) -> list[str]:
        return [f"EDGE_NODE_MODEL={self.checkpoint}"]


@dataclass(frozen=True)
class Cluster:
    name: str
    control: str
    url: str
    ranks: tuple[Rank, ...]
    #: The cards it serves on, HEAD FIRST: each rank's box's serving cards in index order.
    serving_cards: tuple[Card, ...]
    #: What it serves — named here for a one-host node (control local, none); None for the pair (its Start names it).
    model: ClusterModel | None = None
    #: How its own host launches it — the boot default (control local: required; none: optional); None = not named.
    launch: NodeLaunch | None = None
    #: The models the panel may select (control local's `launch.choices`, the boot default among them); () elsewhere.
    choices: tuple[ClusterModel, ...] = ()
    #: The boxes of its ranks, head first — what bounds a KV tier in their RAM (`ram_tier_bound`).
    boxes: tuple[Box, ...] = ()
    #: Does this LB run it when it starts? The file's word (`run_at_boot`) where several clusters are declared; the one
    #: cluster of a fleet runs at boot. None only while parsing.
    run_at_boot: bool | None = None
    #: THE DEFAULT CLUSTER (operator, 2026-10-17): the one a request naming no model, or a loose name no cluster serves,
    #: is routed to — never moved to another (router.serving). `default: true` on exactly one when several are
    #: declared; the one cluster of a fleet is it. None only while parsing.
    default: bool | None = None
    #: The OpenAI API family its models answer (catalog `TASKS`) — the router's key: its declared model's and every
    #: choice's, one task (`_task`); a cluster whose Start names its model (the pair) runs the LB's own cluster
    #: profiles, which the catalog holds to `chat`.
    task: str = "chat"
    #: engine → the image it runs (the launch block's `image`: one per engine its choices run); {} with no launch.
    images: Mapping[str, str] = MappingProxyType({})
    #: Why this cluster cannot launch ("" = it can): the fleet's fit refused its launch (e.g. a card type stating no
    #: measured start memory). Kept, not raised, for a cluster NOT in the boot set — it blocks only itself: Activate
    #: refuses it naming this (cluster_run.py) and the + tab shows it; a boot cluster with one refuses the file.
    unlaunchable: str = ""
    #: Are its boxes HELD through broker_parrot (the broker on)? Its control's `reserves`, unless the fleet says
    #: `reserve: false` (a node sharing its box with the broker's lane there).
    reserves: bool = False
    #: The share of each rank box's MemTotal a start may take (fleet `ram_usable_fraction`, ssh control); the rest is
    #: the one reserve the start gate, the KV auto pool and tp-rank's guard keep. None where the control takes none.
    ram_usable_fraction: float | None = None
    #: The NIC its ranks talk over (fleet `iface`, ssh control): NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME on each rank;
    #: its address on each box is that rank's `interconnect`. None where the control binds none.
    iface: str | None = None
    #: How its ranks' GPUs are joined (fleet `link`: roce | ethernet across hosts — executors.py); None where the
    #: control binds no NIC. tp-rank's CLUSTER_LINK.
    link: str | None = None
    #: The catalog rows its panel offers (fleet `choices`, ssh control), as row keys; None = every row.
    offers: tuple[str, ...] | None = None
    #: The KV dtypes it runs (fleet `kv_cache_dtypes`, ssh control); None = every one its images take.
    kv_dtypes: tuple[str, ...] | None = None
    #: The prefill budget its ranks launch with when nothing nearer says (fleet `max_num_batched_tokens`, ssh control:
    #: the operator's knob, then the model's tuned budget or its engine's own, then THIS). None where the control
    #: takes none (a node's is its launch block's).
    max_num_batched_tokens: int | None = None

    @property
    def tp(self) -> int:
        """Its tensor-parallel size: the cards serving under its ranks."""
        return len(self.serving_cards)


@dataclass(frozen=True)
class Backend:
    """A plain classic server: on `box`'s serving cards, named by that box's label, dialled at `url`."""

    box: str
    url: str


@dataclass(frozen=True)
class Fleet:
    source: str
    boxes: Mapping[str, Box]
    clusters: Mapping[str, Cluster]
    backends: Mapping[str, Backend]

    @property
    def hw_hosts(self) -> dict[str, str]:
        """hostname → box: how the hardware listener (hw.py) labels a sender."""
        return {b.hostname: b.name for b in self.boxes.values()}


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SSH = re.compile(r"[^@\s]+@[^@\s]+")
_IFACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,14}")     # a Linux interface name (IFNAMSIZ 16)
_URL = re.compile(r"https?://[^\s/]+(?:/\S*)?")
_MODEL = re.compile(r"(?P<id>[A-Za-z0-9][A-Za-z0-9._-]*)(?:@(?P<variant>[a-z][a-z0-9_-]*))?")
#: An image PINNED: a registry image by its digest, or a local build by its image id (`docker image inspect -f
#: '{{.Id}}'`: content-addressed too) — never a tag, which moves.
_DIGEST = re.compile(r"(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}")
_PINNED = "an image pinned by digest (repo@sha256:<64 hex>) or a local build by its image id (sha256:<64 hex>)"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*")
#: The checkpoint variants a catalog row may have, and whether a row has each.
_VARIANTS = {"uncensored": lambda spec: spec.uncensored is not None}


def _keys(where: str, d: Any, required, optional=()) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping, got {type(d).__name__}")
    missing = [k for k in required if k not in d]
    unknown = [k for k in d if k not in required and k not in optional]
    if missing or unknown:
        allowed = ", ".join([*required, *(f"{o} (optional)" for o in optional)])
        raise ValueError(f"{where} takes {allowed}"
                         + (f"; missing {', '.join(map(str, missing))}" if missing else "")
                         + (f"; unknown {', '.join(map(str, unknown))}" if unknown else ""))
    return d


#: WHERE THE BLUEPRINTS WENT (operator, 2026-10-30: "Blueprints meant to be only for agent to build their own fleet
#: settings, not something user chooses"): a blueprint is a reference setup the setup wizard prefills a deployment's
#: own answers from — never a key the LB reads. One sentence, said by every refusal that meets one (here, the switch).
BLUEPRINTS_MOVED = ("blueprints are the setup wizard's reference setups (lb/vllm_lb/library/blueprints), not a runtime "
                    "object: deploy/setup (SETUP.md; render.py --prefill) writes a blueprint's settings into this "
                    "deployment's own answers, and the fleet carries those settings themselves")
#: Fleet keys an LB no longer reads, each with what replaced it (config.RETIRED_KEYS' rule, for the fleet file): present
#: at all, the file is refused at boot naming the key — never read with it ignored.
RETIRED_FLEET_KEYS = {"blueprints": BLUEPRINTS_MOVED}
RETIRED_CLUSTER_KEYS = {"blueprint": BLUEPRINTS_MOVED}


def _unretired(source: str, raw: Any) -> Any:
    """`raw` (the whole fleet document) — or ValueError naming the first RETIRED key it still carries, at the top or on
    a cluster, and what replaced it. Read before anything else, so every reader of the file (the LB, `node-env`, the
    catalog's targets) refuses it in the same words."""
    if not isinstance(raw, dict):
        return raw
    for k, moved in RETIRED_FLEET_KEYS.items():
        if k in raw:
            raise ValueError(f"{source}: {k} is retired (2026-10-30): {moved} — remove this key")
    clusters = raw.get("clusters")
    for name, c in (clusters.items() if isinstance(clusters, dict) else ()):
        for k, moved in RETIRED_CLUSTER_KEYS.items():
            if isinstance(c, dict) and k in c:
                raise ValueError(f"{source}: clusters.{name}.{k} is retired (2026-10-30): {moved} — remove this key")
    return raw


def _text(where: str, v: Any, pattern: re.Pattern | None = None, shape: str = "") -> str:
    if not isinstance(v, str) or not v.strip() or (pattern is not None and not pattern.fullmatch(v.strip())):
        raise ValueError(f"{where} must be {shape or 'a non-empty string'}, got {v!r}")
    return v.strip()


def _named(where: str, d: Any) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping of names, got {type(d).__name__}")
    for name in d:
        _text(f"{where} name", name, _NAME, "a name of letters, digits, '.', '_' or '-'")
    return d


def _card(where: str, box: str, raw: Any, library) -> Card:
    _keys(where, raw, ("index", "type", "serves"))
    index, serves = raw["index"], raw["serves"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"{where}.index must be nvidia-smi's card number (0, 1, …), got {index!r}")
    if not isinstance(serves, bool):
        raise ValueError(f"{where}.serves must be true or false, got {serves!r}")
    try:
        kind = profile(raw["type"], library)
    except ValueError as exc:
        raise ValueError(f"{where}.type: {exc}") from None
    return Card(box, index, kind, serves)


def _box(src: str, name: str, raw: Any, library) -> Box:
    where = f"{src}: boxes.{name}"
    _keys(where, raw, ("hostname", "platform", "cards"), ("label", "ssh", RAM_BOUND, POOL_BOUND, NODE_LIMITS))
    platform = raw["platform"]
    if platform not in PLATFORMS:
        raise ValueError(f"{where}.platform must be one of {', '.join(PLATFORMS)} (the CPU its images are built for), "
                         f"got {platform!r}")
    cards_raw = raw["cards"]
    if not isinstance(cards_raw, list) or not cards_raw:
        raise ValueError(f"{where}.cards must list every card of the box")
    cards = tuple(sorted((_card(f"{where}.cards[{i}]", name, c, library) for i, c in enumerate(cards_raw)),
                         key=lambda c: c.index))
    indices = [c.index for c in cards]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{where}.cards: an index is listed twice ({indices})")
    families = {type(c.type).__name__ for c in cards}
    if len(families) > 1:
        raise ValueError(f"{where}.cards mix memory families ({', '.join(sorted(families))}): a box's GPU memory "
                         f"is either its own RAM or each card's VRAM")
    for c in cards:
        if c.type.host_platform.known and c.type.host_platform.value != platform:
            raise ValueError(f"{where}.platform is {platform}, but its {c.type.key} carries the box's CPU: "
                             f"{c.type.host_platform.value} ({c.type.host_platform.source})")
        # …and a box whose CPU its card's ACCELERATOR never comes in (accelerators/<key>.yml `platforms`) has no
        # image built for it: refused here, with the accelerator's own source, before any start could ask for one.
        why = compatibility(c.type.accelerator, platform=platform)
        if why:
            raise ValueError(f"{where}.platform is {platform}, but its {c.type.key}'s accelerator: {why}")
    bound = raw.get(RAM_BOUND)
    if RAM_BOUND in raw and (not isinstance(bound, int) or isinstance(bound, bool) or bound < 1):
        raise ValueError(f"{where}.{RAM_BOUND} must be a whole number of GiB from 1 — the most of the box's RAM a KV "
                         f"tier may take — or left out for none, got {bound!r}")
    pool = raw.get(POOL_BOUND)
    if POOL_BOUND in raw and (not isinstance(pool, int) or isinstance(pool, bool) or pool < 1):
        raise ValueError(f"{where}.{POOL_BOUND} must be a whole number of GiB, the most of this box's RAM a node's GPU "
                         f"pool may take — or left out for none, got {pool!r}")
    return Box(name=name, hostname=_text(f"{where}.hostname", raw["hostname"]),
               label=_text(f"{where}.label", raw["label"], _NAME, "a lane name") if "label" in raw else None,
               ssh=_text(f"{where}.ssh", raw["ssh"], _SSH, "user@address") if "ssh" in raw else None,
               cards=cards, platform=platform, kv_offload_ram_max_gib=bound, gpu_ram_max_gib=pool,
               node_limits=_node_limits(f"{where}.{NODE_LIMITS}", raw[NODE_LIMITS]) if NODE_LIMITS in raw else None)


@dataclass(frozen=True)
class NodeLimits:
    """A node container's bounds on its box: memory (GiB, swap none: memswap_limit = mem_limit), CPUs, processes, and
    the niceness its engine runs at. The measured gui run's own guards (2026-10-17): the one live failure there was a
    process explosion only the memcg contained, and a cold torch.compile takes every core for minutes."""
    memory_gib: int
    cpus: float
    pids: int
    nice: int

    def lines(self) -> list[str]:
        return [f"EDGE_NODE_MEMORY={self.memory_gib}g", f"EDGE_NODE_CPUS={self.cpus:g}",
                f"EDGE_NODE_PIDS_LIMIT={self.pids}", f"EDGE_NODE_NICE={self.nice}"]


def _node_limits(where: str, raw: Any) -> NodeLimits:
    """`node_limits` — every key stated, each in range, or ValueError naming it; nothing defaulted."""
    _keys(where, raw, ("memory_gib", "cpus", "pids", "nice"))
    whole = lambda v: isinstance(v, int) and not isinstance(v, bool)     # noqa: E731
    if not whole(raw["memory_gib"]) or raw["memory_gib"] < 1:
        raise ValueError(f"{where}.memory_gib must be a whole number of GiB from 1, got {raw['memory_gib']!r}")
    cpus = raw["cpus"]
    if not isinstance(cpus, (int, float)) or isinstance(cpus, bool) or not cpus > 0:
        raise ValueError(f"{where}.cpus must be a number of CPUs above 0, got {cpus!r}")
    if not whole(raw["pids"]) or raw["pids"] < 1:
        raise ValueError(f"{where}.pids must be a whole number of processes from 1, got {raw['pids']!r}")
    if not whole(raw["nice"]) or not -20 <= raw["nice"] <= 19:
        raise ValueError(f"{where}.nice must be a niceness from -20 to 19, got {raw['nice']!r}")
    return NodeLimits(raw["memory_gib"], cpus, raw["pids"], raw["nice"])


def _model(where: str, raw: Any, catalog) -> ClusterModel:
    m = _MODEL.fullmatch(raw) if isinstance(raw, str) else None
    if m is None:
        raise ValueError(f"{where}.model must be <catalog id>[@<variant>], got {raw!r}")
    if catalog.error:
        raise ValueError(f"{where}.model {raw!r}: the model catalog could not be loaded: {catalog.error}")
    spec = catalog.get(m["id"])
    if spec is None and m["id"] in getattr(catalog, "absent", {}):
        raise ValueError(f"{where}.model: {m['id']} is not staged in this deployment: {catalog.absent[m['id']]}")
    if spec is None:
        raise ValueError(f"{where}.model: no model {m['id']!r} in the catalog; it has "
                         f"{', '.join(s.id for s in catalog.specs)}")
    variant = m["variant"]
    if variant is not None and variant not in _VARIANTS:
        raise ValueError(f"{where}.model: no variant {variant!r}; the catalog's variants are {', '.join(_VARIANTS)}")
    if variant is not None and not _VARIANTS[variant](spec):
        raise ValueError(f"{where}.model: {spec.id} has no {variant} variant in the catalog")
    return ClusterModel(spec=spec, variant=variant)


#: The launch keys an intent may move (`validate_launch`): the model (one of the fleet's `choices`), the figures the
#: catalog bounds, vLLM's prefill chunk (`CHUNK`, which vLLM bounds), the KV pool pinned (`PIN`) and a KV tier in the
#: host's RAM (`RAM_TIER`, which the box bounds). The image and the parsers are the fleet file's alone, and a served
#: name or a checkpoint follows the model — an intent naming one is refused, so nothing the LB writes picks what its
#: host runs.
CHUNK = "max_num_batched_tokens"
#: THE KV POOL PINNED, in bytes — vLLM's --kv-cache-memory-bytes (engine/arg_utils.py:1352 @2eb491e), the pair's
#: knob by the pair's own name (cluster.py `kv_cache_memory`, tp-rank CLUSTER_KV_CACHE_MEMORY). With it vLLM SKIPS its
#: profile-based sizing ("This does not respect the gpu_memory_utilization config", v1/worker/gpu_worker.py:635-660),
#: while its start check on the fraction still runs (`request_memory`, v1/worker/utils.py:716-770, called at
#: gpu_worker.py:521 whatever the pin) — so a fraction the card's free memory cannot meet ("0.99" on a card with
#: 30.85 of 31.36 GiB free) does not start (`_start_check` refuses it in those numbers), and a pinned pool is the
#: honest way to give the KV the whole card. The fit prices a pinned launch on the WHOLE card (`launch_fit`); the
#: fraction stays, for vLLM's start check alone.
PIN = "kv_cache_memory"
#: A KV TIER IN THE HOST'S RAM, GiB — vLLM's --kv-offloading-size with --kv-offloading-backend native (arg_utils.py
#: :1405-1408): the OffloadingConnector (config/vllm.py:1115-1122), which supports the hybrid KV cache manager
#: (offloading_connector.py:53 `SupportsHMA`, so a GDN model keeps it; its linear-attention states are offloaded as
#: align-mode boundary states, offloading/scheduler.py:155-173). What it does: every step it COPIES each scheduled
#: request's completed blocks to the tier (write-through, offloading/scheduler.py:1570-1594 `_build_store_jobs`), a
#: preempted request's pending copies are finished before its blocks are reused (:1856-1863; worker.py:219-244), and
#: when that request is scheduled again its KV is LOADED back into VRAM blocks of its own before it runs
#: (v1/core/sched/scheduler.py:979, :1323 WAITING_FOR_REMOTE_KVS, :3188-3199) instead of recomputed — decode never
#: reads host RAM, and the tier raises no concurrency: the requests decoding at once are still the GPU pool's. vLLM's
#: default `offload_prompt_only` (v1/kv_offload/base.py:620) stores PROMPT blocks only, so the tokens a request
#: generated before its preemption are recomputed (kv_connector_extra_config {"offload_prompt_only": false} through
#: --kv-transfer-config would store them too, config/vllm.py:1112-1123 merging it; not exposed). The tier is an mmap
#: in the host's /dev/shm (v1/kv_offload/cpu/shared_offload_region.py:110), pinned whole at boot (cudaHostRegister,
#: v1/kv_offload/cpu/gpu_worker.py:199-240) — so the box's bound (`RAM_BOUND`) is RAM taken from the start, and vLLM
#: refuses a tier larger than /dev/shm's free space (vllm/distributed/device_communicators/shm_broadcast.py:249-258).
#: It takes no GPU memory: host-pinned buffers (gpu_worker.py:243-252, :806-822), copied by DMA on side streams.
RAM_TIER = "kv_offload_ram_gib"
#: The box key bounding RAM_TIER on it.
RAM_BOUND = "kv_offload_ram_max_gib"
#: The box key bounding a node's GPU pool where the pool is the box's own RAM (unified memory: `Box.gpu_ram_max_gib`).
POOL_BOUND = "gpu_ram_max_gib"
#: A box's node-container limits (`NodeLimits`), required on a box a contained node runs on (Control.contained).
NODE_LIMITS = "node_limits"
#: THE KV CACHE'S ELEMENT TYPE — vLLM's --kv-cache-dtype (nvfp4, fp8_e4m3, … operator 2026-10-22: "implement KV nvfp4
#: to our models"). Which values a node may take is its IMAGE's own CacheDType (vllm/config/cache.py), read out of each
#: image into the ONE KV dtype library (vllm_lb/kv_dtype.py, kv_cache_dtypes.json) keyed by the pinned ref, with the
#: arch a value runs on — the build decides, not the card name: the gui node's ROCm build has no fp4 value at all.
KV_DTYPE = "kv_cache_dtype"
#: FASTER-WHISPER'S OWN KNOBS (`FasterWhisperEngine`): the realtime queue behind its seats — the service's
#: MAX_QUEUE_SIZE with FAIL_FAST_WHEN_BUSY off — and CTranslate2's compute type (COMPUTE_TYPE).
QUEUE, COMPUTE = "max_queue", "compute_type"
MOVABLE = ("model", "max_model_len", "max_num_seqs", "gpu_memory_utilization", CHUNK, PIN, RAM_TIER, KV_DTYPE, QUEUE,
           COMPUTE)
_FIGURES = ("max_model_len", "max_num_seqs", "gpu_memory_utilization")
_FIXED = ("image", "reasoning_parser", "tool_call_parser")
#: The launch keys a TASK requires (a chat node's tool parser) — refused on a cluster of any other task, where they would
#: be flags the engine does not serve with.
TASK_KEYS = {"chat": ("tool_call_parser",), "transcription": ()}
_DERIVED = ("served_name", "checkpoint")
#: The launch keys a block may leave out: absent is the ENGINE's own default, which no value of ours stands in for.
OPTIONAL = (CHUNK, PIN, RAM_TIER, KV_DTYPE, QUEUE, COMPUTE)
#: Fixed keys a block may leave out: absent passes no flag (no reasoning parser: vLLM parses no thinking). The tool parser
#: is its task's (`TASK_KEYS`).
OPTIONAL_FIXED = ("reasoning_parser", "tool_call_parser")
#: The quant methods the node's MFU byte-size patch knows (deploy/standalone/vllm-mfu/sitecustomize.py, tp-rank's
#: heredoc byte for byte): a checkpoint reporting one gets it (EDGE_NODE_MFU_PATCH=on), any other none — the patch's
#: sitecustomize recursed under ROCm's platform probe on a BF16 model that needed nothing of it (2026-10-17).
MFU_PATCHED_QUANTS = ("modelopt_mixed",)
#: The node-env lines (`node_env`) that may be EMPTY — an optional key left to vLLM. Rendered either way: node.env is
#: read after the server file (later wins), so the empty line is what keeps a server file's own value from standing in
#: for vLLM's default. deploy/standalone/.env.example documents each commented out (deploy/tests pins the two alike).
OPTIONAL_NODE_ENV = ("EDGE_NODE_MAX_NUM_BATCHED_TOKENS", "EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_OFFLOAD_RAM_GIB",
                     "EDGE_NODE_REASONING_PARSER", "EDGE_NODE_KV_CACHE_DTYPE", "EDGE_NODE_TOOL_CALL_PARSER",
                     "EDGE_NODE_MAX_QUEUE", "EDGE_NODE_COMPUTE_TYPE")


def _launch_keys(control: Control) -> tuple[str, ...]:
    """The keys a launch block of `control` takes: the image, the figures and the parsers, and either the models the
    panel selects among (`choices`) or the one served name its own deployment uses."""
    return ("image", *(() if control.selects else ("served_name",)), *_FIGURES,
            *(("choices",) if control.selects else ()))


def _figures(w: str, raw: Mapping, model: ClusterModel) -> tuple[int, int, float]:
    """(max_model_len, max_num_seqs, gpu_memory_utilization) within `model`'s catalog limits — or ValueError in the
    words a launch block and an intent share."""
    limits = model.spec.limits
    length, seqs, util = raw["max_model_len"], raw["max_num_seqs"], raw["gpu_memory_utilization"]
    if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= limits.max_model_len:
        raise ValueError(f"{w}.max_model_len must be a whole number of tokens up to {model.spec.id}'s "
                         f"{limits.max_model_len:,}, got {length!r}")
    lo, hi = limits.max_num_seqs
    if not isinstance(seqs, int) or isinstance(seqs, bool) or not lo <= seqs <= hi:
        raise ValueError(f"{w}.max_num_seqs must be a whole number in {model.spec.id}'s {lo}..{hi}, got {seqs!r}")
    if not isinstance(util, (int, float)) or isinstance(util, bool) or not 0 < util <= 1:
        raise ValueError(f"{w}.gpu_memory_utilization must be a fraction in (0, 1], got {util!r}")
    return length, seqs, float(util)


def _chunk(w: str, raw: Mapping, seqs: int) -> int | None:
    """vLLM's prefill chunk (`max_num_batched_tokens`) as a launch asks for it — None when it names none or null, vLLM's
    own default — or ValueError in the words a launch block and an intent share. vLLM's OWN rules, read in its source at
    the image the 5090 node runs (v0.31.1rc1.dev273 = vllm-project/vllm 2eb491e, vllm/config/scheduler.py): the field is
    `Field(ge=1)` (:48), and `verify_max_model_len` raises "max_num_batched_tokens (…) must be greater than or equal to
    max_num_seqs (…)" (:328). Its other check on the chunk — at least max_model_len — binds only with chunked prefill
    off, and vLLM turns it on for every generative model (vllm/config/model.py `is_chunked_prefill_supported`)."""
    value = raw.get(CHUNK)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{w}.{CHUNK} must be a whole number of tokens from 1, or left out for vLLM's own default, "
                         f"got {value!r}")
    if value < seqs:
        raise ValueError(f"{w}.{CHUNK} {value:,} is under max_num_seqs {seqs:,}: vLLM refuses a prefill chunk smaller "
                         f"than its seats (\"must be greater than or equal to max_num_seqs\")")
    return value


def _pin(w: str, raw: Mapping) -> int | None:
    """The KV pool a launch pins (`PIN`), in bytes — None when it names none or null: vLLM sizes the pool from
    gpu_memory_utilization. Whether the card holds it, and one request, is the fit's (`_fit`)."""
    value = raw.get(PIN)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{w}.{PIN} must be a whole number of bytes (vLLM's --kv-cache-memory-bytes), or left out for "
                         f"vLLM to size the pool from gpu_memory_utilization, got {value!r}")
    return value


def _ram(w: str, raw: Mapping) -> int | None:
    """The KV tier a launch asks in the host's RAM (`RAM_TIER`), in GiB — None when it names none or null: no tier. 0 is
    not off — vLLM's spec raises without cpu_bytes_to_use (v1/kv_offload/cpu/spec.py:96-100) — so off is null. Whether
    the box allows it is `_ram_tier`'s."""
    value = raw.get(RAM_TIER)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{w}.{RAM_TIER} must be a whole number of GiB from 1 (vLLM's --kv-offloading-size), or left "
                         f"out for no RAM tier, got {value!r}")
    return value


def _queue(w: str, raw: Mapping) -> int | None:
    """The realtime queue a launch asks (`QUEUE`) — None when it names none or null: the service's own (fail fast)."""
    value = raw.get(QUEUE)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{w}.{QUEUE} must be a whole number of requests from 1, or left out for none (a busy request "
                         f"is answered 503 at once), got {value!r}")
    return value


def _compute(w: str, raw: Mapping) -> str | None:
    """The compute type a launch asks (`COMPUTE`) — None when it names none or null: the image's own. Which values the
    engine runs is its own (`FasterWhisperEngine.COMPUTE_TYPES`)."""
    value = raw.get(COMPUTE)
    if value is None:
        return None
    return _text(f"{w}.{COMPUTE}", value, _TOKEN, "one token")


def _kv_dtype(w: str, value: Any, image: str, cards) -> str | None:
    """The KV cache's element type a launch asks (`KV_DTYPE`) — None when it names none or null: vLLM's own. Otherwise
    one `image` accepts on `cards`' arch (kv_dtype.library); an image the library has no record of takes none."""
    if value is None:
        return None
    why = kv_dtype.library().problem(image, value, [c.type.arch.value for c in cards])
    if why:
        raise ValueError(f"{w}.{KV_DTYPE} {value!r}: {why}")
    return value


def _tier_bound(cards, boxes: Mapping[str, Box]) -> tuple[int | None, str, str]:
    """(the most GiB a KV tier in the host's RAM may take on `cards`, the box that bounds it, "") — or (None, "", why
    none may): the cards' memory family's own refusal (unified memory: hardware/memory.py), else a box that states no
    `RAM_BOUND`, which is never assumed."""
    for card in cards:
        why = card.type.memory.ram_tier_refusal
        if why:
            return None, "", f"{why} ({card.type.key} on box {card.box!r})"
    bounds = []
    for name in dict.fromkeys(c.box for c in cards):
        box = boxes.get(name)
        bound = box.kv_offload_ram_max_gib if box is not None else None
        if bound is None:
            return None, "", (f"box {name!r} states no {RAM_BOUND}: the most of its RAM a KV tier may take is the "
                              f"fleet file's to say, never assumed")
        bounds.append((bound, name))
    if not bounds:
        return None, "", "no card serves it"
    bound, name = min(bounds)
    return bound, name, ""


def ram_tier_bound(cluster: "Cluster") -> tuple[int | None, str]:
    """(the most GiB of host RAM a KV tier of `cluster`'s node may take, "") — or (None, why it may take none) — as the
    validator judges it (`_ram_tier`), for the panel to offer and say."""
    bound, _, why = _tier_bound(cluster.serving_cards, {b.name: b for b in cluster.boxes})
    return bound, why


def _ram_tier(w: str, gib: int | None, cards, boxes: Mapping[str, Box]) -> None:
    """A KV tier of `gib` GiB in the host's RAM on `cards` — refused where their family adds nothing by it, where a box
    states no bound, or past the bound, in words naming it. None (no tier) always passes."""
    if gib is None:
        return
    bound, name, why = _tier_bound(cards, boxes)
    if why:
        raise ValueError(f"{w}.{RAM_TIER} {gib} GiB: {why}")
    if gib > bound:
        raise ValueError(f"{w}.{RAM_TIER} {gib} GiB > the bound of box {name!r} ({RAM_BOUND} {bound} GiB, the fleet "
                         f"file's): vLLM maps the tier in the host's /dev/shm, pinned whole at start, and refuses one "
                         f"larger than its free space (vllm/distributed/device_communicators/shm_broadcast.py:249-258 "
                         f"@2eb491e) — the bound is set at or under it")


GIB = 1 << 30


@dataclass(frozen=True)
class Fit:
    """What one launch needs on its card and what it may take, in GiB: the model's weights, what the engine holds
    beside them and the KV (`overhead_gib`), the KV pool ONE request of its max_model_len needs, and its share of the
    card. A figure the catalog does not state for one card is None, and `why` says which. `overhead_chunk`: the prefill
    chunk the overhead was measured at (None = not stated); `note`: what the fit says of the launch's own chunk where it
    is checked but not priced as itself — a smaller chunk, charged the measured overhead ("" otherwise)."""

    weights_gib: float
    overhead_gib: float | None
    kv_gib: float | None
    why: str
    budget_gib: float
    overhead_chunk: int | None = None
    note: str = ""
    #: The KV pool the launch PINS (`PIN`), GiB — then the budget is the whole card; None = sized from the fraction.
    pool_gib: float | None = None
    #: The KV tier in the host's RAM (`RAM_TIER`), GiB — on the fit line, never in the card's sum; None = none.
    ram_gib: int | None = None


def _chunk_terms(model: ClusterModel, over, chunk: int | None) -> tuple[str, str]:
    """(why, note) for a launch's prefill chunk against the one its overhead was measured at. The overhead holds vLLM's
    PEAK ACTIVATION, which grows with the chunk — 2.18 GiB of the 27B's 3.559 at vLLM's default on the 5090 — so it is
    a figure AT one chunk. vLLM's own default (None) is that one; a chunk up to it is charged the measured figure —
    conservative, a smaller chunk only lowers the activation, and no formula of ours credits it back — and said so; a
    larger one, or any chunk where the catalog names none, is not measured: refused, never guessed."""
    if chunk is None or over is None:
        return "", ""
    if over.chunk_tokens is None:
        return (f"the catalog states no prefill chunk {model.spec.id}'s memory.single_card_overhead was measured at, "
                f"to judge a {chunk:,}-token chunk against", "")
    if chunk > over.chunk_tokens:
        return (f"the catalog measured {model.spec.id}'s engine overhead on one card at a {over.chunk_tokens:,}-token "
                f"prefill chunk; a {chunk:,}-token chunk holds more activation, never measured", "")
    if chunk < over.chunk_tokens:
        return "", (f"engine overhead as measured at a {over.chunk_tokens:,}-token chunk: the {chunk:,}-token chunk's "
                    f"smaller activation is not credited")
    return "", ""


def launch_fit(model: ClusterModel, card: HardwareProfile, util: float, max_model_len: int,
               max_num_batched_tokens: int | None = None, kv_cache_memory: int | None = None,
               kv_offload_ram_gib: int | None = None) -> Fit:
    """The numbers `_fit` judges: the checkpoint's measured on-disk size (`size_gib`), all of which one card loads;
    what the engine holds beside weights and KV on one card (`memory.single_card_overhead`, from its own boot line, at
    the prefill chunk it names — `_chunk_terms` for `max_num_batched_tokens`, None being vLLM's own default); one
    request's KV pool by the engine's block rule ON ONE CARD (`memory.single_card_kv_blocks`, measured there — never
    the pair's per-rank rule scaled); and `util` of the card's memory as its family reads it.

    A PINNED pool (`kv_cache_memory`, bytes) is what vLLM allocates whatever the fraction, so it is priced on the WHOLE
    card: the catalog's single-card overhead was taken so that the fit at the whole card reproduces vLLM's own "to
    fully utilize gpu memory" figure (its source in library/measurements.json) — the card's free memory at start, as the engine
    measured it. The RAM tier (`kv_offload_ram_gib`) is carried for the fit line and is no part of the card's sum."""
    memory, capacity = model.spec.memory, card.gpu_memory()
    blocks = memory.single_card_kv_blocks if memory is not None else None
    over = memory.single_card_overhead if memory is not None else None
    missing = [k for k, v in (("single_card_kv_blocks", blocks), ("single_card_overhead", over)) if v is None]
    chunk_why, note = _chunk_terms(model, over, max_num_batched_tokens)
    return Fit(weights_gib=model.spec.size_gib, overhead_gib=over.gib if over is not None else None,
               kv_gib=blocks.min_pool_bytes(max_model_len) / GIB if blocks is not None else None,
               why=(f"the catalog states no {' or '.join(f'memory.{k}' for k in missing)} for {model.spec.id}"
                    if missing else chunk_why),
               budget_gib=(1.0 if kv_cache_memory is not None else util) * capacity.value if capacity.known else None,
               overhead_chunk=over.chunk_tokens if over is not None else None, note=note,
               pool_gib=kv_cache_memory / GIB if kv_cache_memory is not None else None, ram_gib=kv_offload_ram_gib)


def pin_ceiling(model: ClusterModel, card: HardwareProfile) -> int | None:
    """The largest KV pool (bytes) the fit takes PINNED for `model` on `card`: the whole card less the weights and the
    engine's one-card overhead — `_fit`'s own sum solved for the pool — or None where the catalog or the card states
    one of them not. At or under the measured "fully utilize" figure for the launch the overhead was taken at; more
    seats or a larger chunk than that hold more activation, and vLLM's own boot line says how much less is left."""
    memory, capacity = model.spec.memory, card.gpu_memory()
    over = memory.single_card_overhead if memory is not None else None
    if over is None or model.spec.size_gib is None or not capacity.known:
        return None
    return max(0, int((capacity.value - model.spec.size_gib - over.gib) * GIB))


#: Where vLLM refuses a fraction the card's free memory cannot meet — its start check, read at the image the node runs.
START_CHECK = "v1/worker/utils.py:761 @2eb491e"


def _start_check(w: str, card: HardwareProfile, util: float) -> None:
    """vLLM's START CHECK on the fraction: `request_memory` (v1/worker/utils.py:716-770 @2eb491e, the comparison at :761)
    runs at gpu_worker.py:521 for EVERY launch — a pinned pool too — before anything loads, and raises "Free memory on
    device … on startup is less than desired GPU memory utilization" unless util × the card's VISIBLE memory is within
    its FREE memory at start. Both are the card type's measured facts (`visible_gib`, `free_at_start_gib`: the free
    figure an idle card's with vLLM's own CUDA context already in it), as vLLM printed them, to 0.01 GiB. A type that
    states either not cannot be checked: refused naming the fact and its reason — never the spec's memory_gb in its
    place. The 5090 node, 2026-10-11 23:21Z: 0.99 priced fine on 32 GiB, and the node restart-looped on this check."""
    visible, free = card.visible_gib, card.free_at_start_gib
    missing = [(name, fact) for name, fact in (("visible_gib", visible), ("free_at_start_gib", free))
               if not fact.known]
    if missing:
        raise ValueError(f"{w}: a {card.key} states no {' or '.join(n for n, _ in missing)} ("
                         + "; ".join(f"{n}: {f.source}" for n, f in missing)
                         + f"), so vLLM's start check on gpu_memory_utilization {util:.2f} — its share of the card's "
                           f"visible memory within the memory free at start ({START_CHECK}) — cannot be checked")
    need = util * visible.value
    if need > free.value:
        raise ValueError(f"{w}.gpu_memory_utilization: GPU memory {util:.2f} × {visible.value:g} GiB = {need:.2f} GiB > "
                         f"{free.value:g} GiB free on an idle {card.key} at start: vLLM refuses before loading "
                         f"({START_CHECK})")


def _mfu_patch(w: str, model: ClusterModel, card: HardwareProfile) -> None:
    """A checkpoint whose quant method the MFU byte-size patch serves (`MFU_PATCHED_QUANTS`) runs WITH it, so its card's
    type must say the patch loads there (`mfu_patch_loads`) — false or not stated, the launch is refused naming the
    fact and its source, never rendered with the patch on (on ROCm it recursed until OOM) nor without it (the FFN would
    silently fall out of the FLOP/s and GB/s)."""
    quant = model.spec.quant_method
    loads = card.mfu_patch_loads
    if quant in MFU_PATCHED_QUANTS and loads.value is not True:
        raise ValueError(f"{w}: {model.ref}: its quant method {quant} needs the MFU byte-size patch, and a {card.key} "
                         f"{'cannot load it' if loads.value is False else 'states no'} (mfu_patch_loads: {loads.source})")


def _fit(w: str, model: ClusterModel, card: HardwareProfile, util: float, max_model_len: int,
         max_num_batched_tokens: int | None, kv_cache_memory: int | None = None, *, box: Box) -> None:
    """THE FIT CHECK, one for a launch block, each of its choices and every intent: first vLLM's own start check on the
    fraction (`_start_check`: what it refuses before loading anything), then the model's weights, the engine's
    own overhead on one card (at the launch's prefill chunk: `_chunk_terms`) and ONE request's KV pool within `util` of
    the serving card's memory (`launch_fit`) — or, with the pool PINNED (`kv_cache_memory`), the pinned pool within the
    whole card, and no smaller than one request's, without which vLLM does not start ("To serve at least one request
    with the model's max seq len", v1/core/kv_cache_utils.py:1031 @2eb491e). ValueError with the numbers — and where a
    number is not stated (no size, no card memory, no single-card block rule or overhead, no overhead at the chunk
    asked, no measured start memory), refused, never guessed. Last, THE BOX: where the card's pool is the box's own
    RAM (its family's `pool_problem`), the pool the launch takes — `util` of the card's memory, or a pin's weights +
    overhead + pool — within the box's own bound (`gpu_ram_max_gib`); vLLM's start check does not guard that there.
    Before all of it, the MFU byte-size patch the checkpoint needs, where the card's type says it loads (`_mfu_patch`)."""
    _mfu_patch(w, model, card)
    _start_check(w, card, util)
    memory = card.gpu_memory()
    if model.spec.size_gib is None:
        raise ValueError(f"{w}: {model.ref}: the catalog states no checkpoint size (size_gib), so its fit on a "
                         f"{card.key} cannot be checked")
    if not memory.known:
        raise ValueError(f"{w}: {model.ref}: a {card.key} states no memory to fit it in ({memory.source})")
    fit = launch_fit(model, card, util, max_model_len, max_num_batched_tokens, kv_cache_memory)
    problem = card.memory.pool_problem(box, _pool(fit))
    if problem:
        raise ValueError(f"{w}: {model.ref}: {problem}")
    head = f"{w}: {model.ref}: {fit.weights_gib:g} GiB of weights"
    tail = (f"{util:.2f} × {memory.value:g} GiB on {card.key} ({memory.source})" if fit.pool_gib is None else
            f"the whole {memory.value:g} GiB of {card.key}, a pinned pool being sized by no fraction ({memory.source})")
    if fit.weights_gib > fit.budget_gib:
        raise ValueError(f"{head} > {tail}")
    if fit.why:
        raise ValueError(f"{w}: {model.ref}: {fit.why}, so its launch on a {card.key} cannot be checked")
    if fit.pool_gib is not None:
        blocks = model.spec.memory.single_card_kv_blocks
        least = blocks.min_pool_bytes(max_model_len)
        if kv_cache_memory < least:
            raise ValueError(f"{w}.{PIN} {kv_cache_memory:,} B ({fit.pool_gib:.2f} GiB) < {least / GIB:.2f} GiB: one "
                             f"{max_model_len:,}-token request of {model.ref} needs {blocks.per_request(max_model_len)} "
                             f"blocks × {blocks.block_bytes:,} B, and vLLM does not start without them "
                             f"(v1/core/kv_cache_utils.py:1031 @2eb491e)")
        if fit.weights_gib + fit.overhead_gib + fit.pool_gib > fit.budget_gib:
            raise ValueError(f"{head} + {fit.overhead_gib:.1f} GiB of engine overhead + {fit.pool_gib:.2f} GiB of "
                             f"pinned KV pool > {tail}")
        return
    if fit.weights_gib + fit.overhead_gib + fit.kv_gib > fit.budget_gib:
        raise ValueError(f"{head} + {fit.overhead_gib:.1f} GiB of engine overhead + {fit.kv_gib:.1f} GiB of KV for "
                         f"one {max_model_len:,}-token request > {tail}")


def _pool(fit: Fit) -> float:
    """The GPU pool a vLLM launch takes: `util` of the card (its budget), or a pin's weights + overhead + pool."""
    return fit.budget_gib if fit.pool_gib is None else fit.weights_gib + (fit.overhead_gib or 0.0) + fit.pool_gib


@dataclass(frozen=True)
class Setting:
    """ONE LAUNCH SETTING as the panel offers it: its launch key (`MOVABLE`), its word, the node variable it reaches
    (`env_lines`, which its engine's compose override maps), the values the ENGINE states (empty: the panel's own editor
    bounds it by the fleet's limits and fit), and what null means (None: it is never null)."""

    key: str
    name: str
    env: str
    options: tuple = ()
    unset: str | None = None

    def view(self) -> dict:
        return {"key": self.key, "name": self.name, "env": self.env, "options": list(self.options), "unset": self.unset}


#: Every launch setting, once. An engine names which of them it reads (`NodeEngine.settings`).
SETTINGS = {s.key: s for s in (
    Setting("model", "model", "EDGE_NODE_MODEL"),
    Setting("max_num_seqs", "seats", "EDGE_NODE_MAX_NUM_SEQS"),
    Setting("max_model_len", "context", "EDGE_NODE_MAX_MODEL_LEN"),
    Setting("gpu_memory_utilization", "GPU memory", "EDGE_NODE_GPU_MEM_UTIL"),
    Setting(CHUNK, "prefill budget", "EDGE_NODE_MAX_NUM_BATCHED_TOKENS", unset="vLLM default"),
    Setting(PIN, "KV pool", "EDGE_NODE_KV_CACHE_MEMORY", unset="from GPU memory"),
    Setting(RAM_TIER, "KV offload", "EDGE_NODE_KV_OFFLOAD_RAM_GIB", unset="off"),
    Setting(KV_DTYPE, "KV type", "EDGE_NODE_KV_CACHE_DTYPE", unset="engine auto"),
)}

#: THE PANEL TILES a task's server publishes, by the panel's group names (static/gateway.js `TILE_GROUPS`).
TASK_TILES = {"chat": ("generated", "prefill", "prompt", "queue", "kv", "bandwidth", "compute", "saturation", "swarm"),
              "transcription": ("speech",)}


class NodeEngine:
    """THE SERVER A ONE-HOST NODE RUNS — one class per family (AGENTS.md: one base, one class per behaviour), and the ONE
    place it is declared: its name is the catalog's `node_engine` and the compose override that runs it
    (deploy/standalone/node-engine.<name>.yml, which up.sh picks by EDGE_NODE_ENGINE); the scrape vocabulary it writes
    (`vocabulary`, probe.py's adapter); per task, the settings it reads (`settings`) and the node lines they reach
    (`env_lines`), the tiles it publishes (`tiles`); its memory model (`fit`, `pool_gib`, `memory`); whether it
    runs a whole uploaded file its own way (`whole_file`: "engine" = its scheduler batches it, "lane" = a one-file lane). A launch naming a key its engine does not read is REFUSED,
    never accepted and dropped (catalog.ENGINE_REFUSES' lesson). /lb/status carries `view` (`engine_block`)."""

    name: str = ""
    label: str = ""
    vocabulary: str = ""
    memory: str = ""
    whole_file: str = ""
    #: Does its server load the MFU byte-size patch (deploy/standalone/vllm-mfu: a vLLM sitecustomize) — node-env turns
    #: it on only for an engine that does, and only for a checkpoint whose quant method needs it.
    mfu_patch: bool = False

    def settings(self, task: str) -> tuple[Setting, ...]:
        raise NotImplementedError

    def tiles(self, task: str) -> tuple[str, ...]:
        return TASK_TILES[task]

    def view(self, task: str) -> dict:
        return {"name": self.name, "label": self.label, "task": task, "vocabulary": self.vocabulary,
                "memory": self.memory, "whole_file": self.whole_file,
                "settings": [s.view() for s in self.settings(task)], "tiles": list(self.tiles(task))}

    def env_lines(self, launch: NodeLaunch) -> list[str]:
        """The node-env lines of what it reads — its settings but the model (`NodeLaunch.model_lines`)."""
        raise NotImplementedError

    def fit(self, w: str, model: ClusterModel, card: HardwareProfile, launch: NodeLaunch, *, box: Box) -> None:
        takes = {s.key for s in self.settings(model.spec.task)}
        refused = [k for k in OPTIONAL if getattr(launch, k) is not None and k not in takes]
        if refused:
            raise ValueError(f"{w}: {model.ref} runs on {self.name}, which takes no {', '.join(refused)}")
        self._fit(w, model, card, launch, box=box)

    def _fit(self, w: str, model: ClusterModel, card: HardwareProfile, launch: NodeLaunch, *, box: Box) -> None:
        raise NotImplementedError

    def pool_gib(self, model: ClusterModel, card: HardwareProfile, launch: NodeLaunch) -> float:
        """The GiB of its card's pool a launch takes — what its fit bounds by the box, and what clusters sharing one
        box sum (`pool_need`). ValueError where it is not stated."""
        raise NotImplementedError

    def fit_view(self, model: ClusterModel, cluster: "Cluster", overrides: Mapping[str, Any]) -> dict | None:
        """The panel's fit line for `model` launched by `cluster`'s boot launch moved by `overrides`, in this engine's
        memory model (topology.local_fit) — None where a figure it needs is not a number."""
        raise NotImplementedError


class VllmEngine(NodeEngine):
    """vLLM: sizes its pool as a fraction of the card and its KV in blocks — the fit is `_fit`, its start check first.
    For a transcription model (Whisper, encoder-decoder) it keeps the knobs a speech launch was run with and drops the
    KV tier in host RAM and the KV type: neither was ever started under a Whisper checkpoint here — unmeasured, not
    offered. It batches whole files itself (its scheduler), so it declares no whole-file mode of its own."""

    name = "vllm"
    label = "vLLM"
    vocabulary = "vllm"
    memory = "gpu_memory_utilization × the card (or a pinned KV pool)"
    whole_file = "engine"
    mfu_patch = True
    _CHAT = ("model", "max_num_seqs", "max_model_len", "gpu_memory_utilization", CHUNK, PIN, RAM_TIER, KV_DTYPE)
    _SPEECH = ("model", "max_num_seqs", "max_model_len", "gpu_memory_utilization", CHUNK, PIN)

    def settings(self, task):
        return tuple(SETTINGS[k] for k in (self._SPEECH if task == "transcription" else self._CHAT))

    def env_lines(self, launch):
        chunk = launch.max_num_batched_tokens
        return [f"EDGE_NODE_SERVED_NAME={launch.served_name}", f"EDGE_NODE_MAX_MODEL_LEN={launch.max_model_len}",
                f"EDGE_NODE_MAX_NUM_SEQS={launch.max_num_seqs}",
                # OPTIONAL_NODE_ENV: empty = vLLM's own (the node's entrypoint then passes no flag) — its default chunk,
                # the pool sized from the fraction, no RAM tier.
                f"EDGE_NODE_MAX_NUM_BATCHED_TOKENS={'' if chunk is None else chunk}",
                f"EDGE_NODE_KV_CACHE_MEMORY={'' if launch.kv_cache_memory is None else launch.kv_cache_memory}",
                f"EDGE_NODE_KV_OFFLOAD_RAM_GIB={'' if launch.kv_offload_ram_gib is None else launch.kv_offload_ram_gib}",
                f"EDGE_NODE_KV_CACHE_DTYPE={launch.kv_cache_dtype or ''}",
                f"EDGE_NODE_GPU_MEM_UTIL={launch.gpu_memory_utilization:g}",
                f"EDGE_NODE_REASONING_PARSER={launch.reasoning_parser or ''}",
                f"EDGE_NODE_TOOL_CALL_PARSER={launch.tool_call_parser or ''}"]

    def _fit(self, w, model, card, launch, *, box):
        _fit(w, model, card, launch.gpu_memory_utilization, launch.max_model_len, launch.max_num_batched_tokens,
             launch.kv_cache_memory, box=box)

    def pool_gib(self, model, card, launch):
        return _pool(launch_fit(model, card, launch.gpu_memory_utilization, launch.max_model_len,
                                launch.max_num_batched_tokens, launch.kv_cache_memory))

    def fit_view(self, model, cluster, overrides):
        launch = cluster.launch
        util = overrides.get("gpu_memory_utilization", launch.gpu_memory_utilization)
        length = overrides.get("max_model_len", launch.max_model_len)
        chunk = overrides.get(CHUNK, launch.max_num_batched_tokens)
        pin = overrides.get(PIN, launch.kv_cache_memory)
        ram = overrides.get(RAM_TIER, launch.kv_offload_ram_gib)
        whole = isinstance(length, int) and not isinstance(length, bool) and length > 0
        number = isinstance(util, (int, float)) and not isinstance(util, bool) and util > 0
        counts = all(v is None or (isinstance(v, int) and not isinstance(v, bool) and v > 0) for v in (chunk, pin, ram))
        if not (whole and number and counts):
            return None
        f = launch_fit(model, cluster.serving_cards[0].type, float(util), length, chunk, pin, ram)
        return {"weights_gib": f.weights_gib, "overhead_gib": f.overhead_gib, "kv_gib": f.kv_gib, "why": f.why,
                "budget_gib": f.budget_gib, "overhead_chunk": f.overhead_chunk, "note": f.note,
                "pool_gib": f.pool_gib, "ram_gib": f.ram_gib}


class MeasuredEngine(NodeEngine):
    """An engine whose fit is the WHOLE ENGINE'S MEASURED MEMORY (catalog `engine_memory`), at no more requests than it
    was measured at, within the card and the box's bound — refused, never scaled, where the catalog measured none."""

    memory = "the catalog's measured engine_memory"

    def _fit(self, w, model, card, launch, *, box):
        mem = model.spec.engine_memory
        if mem is None:
            raise ValueError(f"{w}: {model.ref}: the catalog states no engine_memory (the whole engine's GPU memory, "
                             f"measured), so its fit on a {card.key} cannot be checked")
        if launch.max_num_seqs > mem.max_num_seqs:
            raise ValueError(f"{w}.max_num_seqs {launch.max_num_seqs} > the {mem.max_num_seqs} concurrent requests "
                             f"{model.ref}'s {mem.gib:g} GiB was measured at ({mem.source[:60]}…): more is not measured")
        capacity = card.gpu_memory()
        if capacity.known and mem.gib > capacity.value:
            raise ValueError(f"{w}: {model.ref}: {mem.gib:g} GiB measured > {capacity.value:g} GiB on {card.key}")
        problem = card.memory.pool_problem(box, mem.gib)
        if problem:
            raise ValueError(f"{w}: {model.ref}: {problem}")

    def fit_view(self, model, cluster, overrides):
        """Its measured whole-engine memory against the box's pool bound (or the card) — the line says what it is."""
        mem = model.spec.engine_memory
        if mem is None:
            return None
        box, card = cluster.boxes[0], cluster.serving_cards[0].type
        bound, _ = card.memory.shared_pool(box)
        capacity = card.gpu_memory()
        budget = bound if bound is not None else (capacity.value if capacity.known else None)
        return {"engine_gib": mem.gib, "measured_seats": mem.max_num_seqs, "budget_gib": budget,
                "text": f"{mem.gib:g} GiB measured whole engine @ {mem.max_num_seqs} requests"
                        + (f" {'≤' if mem.gib <= budget else '>'} {budget:g} GiB" if budget is not None else "")}

    def pool_gib(self, model, card, launch):
        if model.spec.engine_memory is None:
            raise ValueError(f"{model.ref}: the catalog states no engine_memory (the whole engine's GPU memory, measured)")
        return model.spec.engine_memory.gib


class FasterWhisperEngine(MeasuredEngine):
    """faster-whisper (CTranslate2; a transcription server, docs/design/engines.md's contract): allocates what its
    model and its in-flight requests need — no fraction, no pool, no context window — so vLLM's figures are not its.
    It reads its seats (MAX_ACTIVE_REQUESTS), the realtime queue behind them and the compute type. Its fit is its
    measured engine memory (`MeasuredEngine`). A whole uploaded file is decoded in GPU batches (the service's tier "batch")."""

    name = "faster-whisper"
    label = "faster-whisper (CTranslate2)"
    vocabulary = "faster-whisper"
    whole_file = "lane"
    #: CTranslate2's compute types this engine is offered at: each measured on a faster-whisper ROCm build on a Radeon
    #: 8060S (2026-10-23, large-v3-turbo, one 137 s reading — process GPU memory off DRM fdinfo): float16 2.62 GiB,
    #: bfloat16 2.56, int8_float16 2.42, int8_bfloat16 2.41, int8 2.42 — none above the float16 figure the catalog's
    #: engine_memory was measured at. float32 (3.98 GiB) is above it, so the fit could not vouch for it: not offered.
    COMPUTE_TYPES = ("float16", "bfloat16", "int8_float16", "int8_bfloat16", "int8")
    QUEUE_STEPS = (2, 4, 8, 16, 32)

    def settings(self, task):
        return (SETTINGS["model"], SETTINGS["max_num_seqs"],
                Setting(QUEUE, "queue", "EDGE_NODE_MAX_QUEUE", self.QUEUE_STEPS, "none — busy answers 503"),
                Setting(COMPUTE, "compute type", "EDGE_NODE_COMPUTE_TYPE", self.COMPUTE_TYPES, "the image's own"))

    def env_lines(self, launch):
        return [f"EDGE_NODE_MAX_NUM_SEQS={launch.max_num_seqs}",
                f"EDGE_NODE_MAX_QUEUE={'' if launch.max_queue is None else launch.max_queue}",
                f"EDGE_NODE_COMPUTE_TYPE={launch.compute_type or ''}"]

    def _fit(self, w, model, card, launch, *, box):
        if launch.compute_type is not None and launch.compute_type not in self.COMPUTE_TYPES:
            raise ValueError(f"{w}.{COMPUTE} {launch.compute_type!r}: {self.name} is offered "
                             f"{', '.join(self.COMPUTE_TYPES)} — each measured within the catalog's engine_memory")
        super()._fit(w, model, card, launch, box=box)


class NemoEngine(FasterWhisperEngine):
    """NVIDIA NeMo ASR (a .nemo checkpoint: Parakeet TDT) behind the same transcription-server contract
    (docs/design/engines.md): its seats, its realtime queue, its one-file lane, its /metrics vocabulary and its measured
    whole-engine memory are the server's, as faster-whisper's. CTranslate2's compute type is not its: not offered."""

    name = "nemo"
    label = "NeMo (Parakeet TDT)"
    COMPUTE_TYPES = ()

    def settings(self, task):
        return tuple(s for s in super().settings(task) if s.key != COMPUTE)

    def env_lines(self, launch):
        return [f"EDGE_NODE_MAX_NUM_SEQS={launch.max_num_seqs}",
                f"EDGE_NODE_MAX_QUEUE={'' if launch.max_queue is None else launch.max_queue}"]


class SglangEngine(MeasuredEngine):
    """SGLang (`python -m sglang.launch_server`; the pair's engine since 2026-10-02, tp-rank's): takes a STATIC FRACTION
    of the card at start — weights, the KV pool and the GDN state slots — by ITS OWN rule for the card (the launch's
    gpu_memory_utilization is vLLM's, not read), so its fit is its measured engine memory (`MeasuredEngine`). It reads
    its seats (--max-running-requests), its context (--context-length) and the KV type (--kv-cache-dtype, one its
    image accepts: kv_cache_dtypes.json); vLLM's prefill chunk, pinned pool and RAM tier are not its — refused where
    named. The pair's DFlash2 drafter is the pair's launch (tp-rank); a node runs none."""

    name = "sglang"
    label = "SGLang"
    vocabulary = "sglang"
    whole_file = "engine"

    def settings(self, task):
        return tuple(SETTINGS[k] for k in ("model", "max_num_seqs", "max_model_len", KV_DTYPE))

    def env_lines(self, launch):
        return [f"EDGE_NODE_SERVED_NAME={launch.served_name}", f"EDGE_NODE_MAX_MODEL_LEN={launch.max_model_len}",
                f"EDGE_NODE_MAX_NUM_SEQS={launch.max_num_seqs}",
                f"EDGE_NODE_KV_CACHE_DTYPE={launch.kv_cache_dtype or ''}",
                f"EDGE_NODE_REASONING_PARSER={launch.reasoning_parser or ''}",
                f"EDGE_NODE_TOOL_CALL_PARSER={launch.tool_call_parser or ''}"]


#: engine name (catalog.NODE_ENGINES) → its class.
ENGINES: Mapping[str, NodeEngine] = MappingProxyType({e.name: e for e in (VllmEngine(), FasterWhisperEngine(),
                                                                         NemoEngine(), SglangEngine())})


def _run_by(launch: NodeLaunch, engine: NodeEngine, task: str) -> NodeLaunch:
    """`launch` (a block's boot default) as a choice on `engine` starts from it. On the boot launch's own engine, the
    block as written — a key that engine does not read is refused by its fit. On ANOTHER engine, the block's optional
    settings it does not read are the boot engine's (a vLLM boot block's pinned pool is no SGLang setting): unset, as
    the LB's switch leaves a stored wish's behind (proxy `_read_by`). One an intent NAMES is judged as asked."""
    if engine.name == launch.engine:
        return launch
    reads = {s.key for s in engine.settings(task)}
    return replace(launch, **{k: None for k in OPTIONAL if k not in reads})


def _engine(where: str, model: ClusterModel) -> "NodeEngine":
    """The engine a node runs `model` on — its catalog row's `node_engine` — or ValueError: a row a node launches names
    it, never assumed."""
    name = model.spec.node_engine
    if name is None:
        raise ValueError(f"{where}: {model.ref} names no node_engine in the catalog ({', '.join(ENGINES)}): the server "
                         f"its node runs is the checkpoint's, never assumed")
    return ENGINES[name]


def _task(where: str, models) -> str:
    """The one task `models` (a cluster's model and choices) share — or ValueError: one node serves one API family."""
    tasks = sorted({m.spec.task for m in models})
    if len(tasks) != 1:
        raise ValueError(f"{where}: its models answer {' and '.join(tasks)} — "
                         + "; ".join(f"{m.ref} is {m.spec.task}" for m in models)
                         + ": one cluster serves one task (the router sends it that task's requests alone)")
    return tasks[0]


def _images(w: str, raw: Any, engines: tuple[str, ...]) -> dict[str, str]:
    """engine → its pinned image, from a launch block's `image`: one image where every model runs one engine, else a
    mapping naming exactly the engines its models run — or ValueError naming the gap."""
    if isinstance(raw, str):
        if len(engines) != 1:
            raise ValueError(f"{w}.image names one image, but its models run {', '.join(engines)}: map each engine to "
                             f"its image ({{{', '.join(f'{e}: <image>' for e in engines)}}})")
        return {engines[0]: _text(f"{w}.image", raw, _DIGEST, _PINNED)}
    if not isinstance(raw, dict):
        raise ValueError(f"{w}.image must be {_PINNED}, or a mapping of engine to such an image")
    missing, extra = [e for e in engines if e not in raw], [str(k) for k in raw if k not in engines]
    if missing or extra:
        raise ValueError(f"{w}.image maps the engines its models run ({', '.join(engines)})"
                         + (f"; missing {', '.join(missing)}" if missing else "")
                         + (f"; {', '.join(extra)} runs none of them" if extra else ""))
    return {e: _text(f"{w}.image.{e}", raw[e], _DIGEST, _PINNED) for e in engines}


def _launch(where: str, raw: Any, model: ClusterModel, control: Control, models: tuple[ClusterModel, ...] = ()
            ) -> tuple[NodeLaunch, dict[str, str], str]:
    """(the boot launch, engine → image, the task) of a launch block whose models are `model` and its `models`
    (control local's choices)."""
    w = f"{where}.launch"
    _keys(w, raw, _launch_keys(control), OPTIONAL + OPTIONAL_FIXED)
    every = models or (model,)
    task = _task(f"{where}.launch.choices" if models else where, every)
    engines = tuple(dict.fromkeys(_engine(where, m).name for m in every))
    images = _images(w, raw["image"], engines)
    needs = TASK_KEYS[task]
    for key in ("tool_call_parser",):
        if key in needs and key not in raw:
            raise ValueError(f"{w} needs {key}: a {task} node — vLLM refuses a turn with tools without it")
        if key not in needs and key in raw:
            raise ValueError(f"{w} takes no {key}: its models are {task}, not chat — the flag serves no request here")
    tokens = {k: _text(f"{w}.{k}", raw[k], _TOKEN, "one token (letters, digits, . _ : / + -)")
              for k in (() if control.selects else ("served_name",))}
    for k in OPTIONAL_FIXED:
        tokens[k] = _text(f"{w}.{k}", raw[k], _TOKEN, "one token (letters, digits, . _ : / + -)") if k in raw else None
    length, seqs, util = _figures(w, raw, model)
    chunk, pin, ram, dtype = _chunk(w, raw, seqs), _pin(w, raw), _ram(w, raw), raw.get(KV_DTYPE)
    own = {"max_queue": _queue(w, raw), "compute_type": _compute(w, raw)}
    engine = _engine(where, model).name
    image = images[engine]
    if control.selects:
        return SelectedLaunch(image=image, engine=engine, served_name=model.spec.served_name, max_model_len=length,
                              max_num_seqs=seqs, gpu_memory_utilization=util, max_num_batched_tokens=chunk,
                              kv_cache_memory=pin, kv_offload_ram_gib=ram, kv_cache_dtype=dtype, model=model.ref,
                              checkpoint=model.checkpoint, **own, **tokens), images, task
    return NodeLaunch(image=image, engine=engine, max_model_len=length, max_num_seqs=seqs, gpu_memory_utilization=util,
                      max_num_batched_tokens=chunk, kv_cache_memory=pin, kv_offload_ram_gib=ram, kv_cache_dtype=dtype,
                      **own, **tokens), images, task


def _choices(where: str, raw: Any, boot: ClusterModel, catalog) -> tuple[ClusterModel, ...]:
    w = f"{where}.launch.choices"
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{w} must list the models the panel may select (<catalog id>[@<variant>]), the boot "
                         f"default among them")
    models = tuple(_model(f"{w}[{i}]", ref, catalog) for i, ref in enumerate(raw))
    refs = [m.ref for m in models]
    twice = sorted({r for r in refs if refs.count(r) > 1})
    if twice:
        raise ValueError(f"{w} names {', '.join(twice)} twice")
    if boot.ref not in refs:
        raise ValueError(f"{w}: the boot default {boot.ref} (the cluster's `model`) is not among them: "
                         f"{', '.join(refs)}")
    return models


def _cluster(src: str, name: str, raw: Any, boxes: Mapping[str, Box], catalog) -> Cluster:
    where = f"{src}: clusters.{name}"
    control = CONTROLS.get(raw.get("control")) if isinstance(raw, dict) and isinstance(raw.get("control"), str) else None
    if isinstance(raw, dict) and "control" in raw and control is None:
        raise ValueError(f"{where}.control {raw['control']!r} is no control method: {', '.join(CONTROLS)}")
    usable = ("ram_usable_fraction",) if control is not None and control.ram_usable == "required" else ()
    iface_key = ("iface",) if control is not None and control.iface == "required" else ()
    offers_key = ("choices", "kv_cache_dtypes") if control is not None and control.offers == "optional" else ()
    budget_key = ("max_num_batched_tokens",) if control is not None and control.prefill_budget == "required" else ()
    # THE LINK the ranks are joined by (executors.py: roce or ethernet across hosts) — what tp-rank opens for NCCL, a
    # fact of the deployment (CLUSTER_LINK), never the DGX Spark pair's RoCE assumed. Required wherever a NIC is bound.
    link_key = ("link",) if iface_key else ()
    _keys(where, raw, ("control", "url", "ranks", *usable, *iface_key, *link_key, *budget_key),
          ("model", "launch", "run_at_boot", "default", "reserve", *offers_key))
    budget = None
    if budget_key:
        # THE PREFILL BUDGET (operator, 2026-10-29): the compose default CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 was an env
        # number standing in for the deployment's own. It is the cluster's, in tp-rank's own range; nothing defaults it.
        from vllm_lb.catalog import MAX_NUM_BATCHED_TOKENS_CAP            # the catalog's cap; imported where used
        lo, hi = MAX_NUM_BATCHED_TOKENS_CAP
        budget = raw["max_num_batched_tokens"]
        if isinstance(budget, bool) or not isinstance(budget, int) or not lo <= budget <= hi:
            raise ValueError(f"{where}.max_num_batched_tokens must be a whole number of prompt tokens in {lo}..{hi} "
                             f"(the prefill budget its ranks launch with: vLLM --max-num-batched-tokens, SGLang's "
                             f"chunk), got {budget!r}")
    offers = None
    if "choices" in raw and offers_key:
        # WHAT THIS DEPLOYMENT OFFERS of the shared catalog (operator, 2026-10-26: "remove original censored from our
        # vllm setup"): each row once, as the panel keys it (`<id>` or `<id>@uncensored`).
        if not isinstance(raw["choices"], list) or not raw["choices"]:
            raise ValueError(f"{where}.choices must list the catalog rows its panel offers (<catalog id>[@<variant>]); "
                             f"leave the key out to offer every row")
        offers = tuple(_model(f"{where}.choices[{i}]", ref, catalog()).ref for i, ref in enumerate(raw["choices"]))
        twice = sorted({r for r in offers if offers.count(r) > 1})
        if twice:
            raise ValueError(f"{where}.choices names {', '.join(twice)} twice")
    kv_dtypes = None
    if "kv_cache_dtypes" in raw and offers_key:
        # THE KV ELEMENT TYPES THIS DEPLOYMENT RUNS (operator, 2026-10-26: "skip FP16 everywhere, FP8 is default unless
        # can run nvfp4"): each a dtype the library knows, once — and every offered row's OWN dtype among them, or
        # every Start of that row would be refused.
        listed, known = raw["kv_cache_dtypes"], kv_dtype.library().layouts
        if not isinstance(listed, list) or not listed:
            raise ValueError(f"{where}.kv_cache_dtypes must list the KV dtypes this cluster runs; leave the key out "
                             f"to run every one its images take")
        for d in listed:
            if d not in known:
                raise ValueError(f"{where}.kv_cache_dtypes: no KV dtype {d!r}; the library knows {', '.join(known)}")
        twice = sorted({d for d in listed if listed.count(d) > 1})
        if twice:
            raise ValueError(f"{where}.kv_cache_dtypes names {', '.join(twice)} twice")
        kv_dtypes = tuple(listed)
        from vllm_lb.catalog import CLUSTER_MODE                     # the catalog's mode name; imported where used
        rows = ([r.split("@")[0] for r in offers] if offers is not None
                else [s.id for s in catalog().for_mode(CLUSTER_MODE)])
        for spec_id in dict.fromkeys(rows):
            own = catalog().get(spec_id).profile(CLUSTER_MODE).kv_cache_dtype or "auto"
            if own not in kv_dtypes:
                raise ValueError(f"{where}: {spec_id} runs KV {own}, not one of kv_cache_dtypes "
                                 f"({', '.join(kv_dtypes)}): leave it out of `choices`, or list its dtype")
    fraction = raw.get("ram_usable_fraction")
    if usable and (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1):
        raise ValueError(f"{where}.ram_usable_fraction must be a number between 0 and 1 (the share of each box's "
                         f"MemTotal a start may take, e.g. 0.98), got {fraction!r}")
    iface = _text(f"{where}.iface", raw["iface"], _IFACE,
                  "the interface name NCCL binds on each rank box (NCCL_SOCKET_IFNAME), e.g. bond0") if iface_key else None
    boot = raw.get("run_at_boot")
    if "run_at_boot" in raw and not isinstance(boot, bool):
        raise ValueError(f"{where}.run_at_boot must be true or false, got {boot!r}")
    default = raw.get("default")
    if "default" in raw and not isinstance(default, bool):
        raise ValueError(f"{where}.default must be true or false, got {default!r}")
    control = CONTROLS.get(raw["control"]) if isinstance(raw["control"], str) else None
    if control is None:
        raise ValueError(f"{where}.control {raw['control']!r} is no control method: {', '.join(CONTROLS)}")
    # `reserve: false` — a node that shares its box with the broker's own lane there (allowed by broker_parrot's GPU
    # guard) instead of holding the box: the one word against its control's rule, so only `false`, only where it reserves.
    if "reserve" in raw and (raw["reserve"] is not False or not control.reserves):
        raise ValueError(f"{where}.reserve: only `reserve: false` is said, on a cluster whose control reserves its "
                         f"boxes (control {control.name} {'does' if control.reserves else 'reserves nothing'}); "
                         f"got {raw['reserve']!r}")
    for key, why in (("model", "what the node serves (<catalog id>[@<variant>])"),
                     ("launch", "how the node is launched (its boot default)")):
        rule = getattr(control, key)
        if rule == "forbidden" and key in raw:
            raise ValueError(f"{where}: control {control.name} takes no {key} — the operator's Start on the panel "
                             f"names it and {control.executor} launches it; naming it here would be a second "
                             f"definition")
        if rule == "required" and key not in raw:
            raise ValueError(f"{where}: control {control.name} needs a {key}: {why} — the fleet is the one place "
                             f"that says it")
    model = _model(where, raw["model"], catalog()) if "model" in raw else None
    if "launch" in raw:
        _keys(f"{where}.launch", raw["launch"], _launch_keys(control), OPTIONAL + OPTIONAL_FIXED)
    choices = _choices(where, raw["launch"]["choices"], model, catalog()) if control.selects else ()
    launch, images, task = (_launch(where, raw["launch"], model, control, choices) if "launch" in raw
                            else (None, {}, _task(where, (model,)) if model is not None else "chat"))
    url = _text(f"{where}.url", raw["url"], _URL, "an http(s) URL")
    ranks_raw = raw["ranks"]
    if not isinstance(ranks_raw, list) or not ranks_raw:
        raise ValueError(f"{where}.ranks must list its ranks, head first")
    if control.ranks is not None and len(ranks_raw) != control.ranks:
        raise ValueError(f"{where}: control {control.name} ({control.executor}) runs {control.ranks} ranks, "
                         f"not {len(ranks_raw)}")
    ranks, cards = [], []
    for i, r in enumerate(ranks_raw):
        rw = f"{where}.ranks[{i}]"
        _keys(rw, r, ("box", "interconnect") if control.interconnect else ("box",))
        box = boxes.get(r["box"]) if isinstance(r["box"], str) else None
        if box is None:
            raise ValueError(f"{rw}.box {r['box']!r} is no box of this fleet")
        if any(rank.box == box.name for rank in ranks):
            raise ValueError(f"{rw}.box {box.name!r} holds two ranks of one cluster")
        serving = box.serving_cards
        if not serving:
            raise ValueError(f"{rw}: box {box.name!r} has no card with serves: true for the rank to run on")
        if control.cards_per_rank is not None and len(serving) != control.cards_per_rank:
            raise ValueError(f"{rw}: control {control.name} ({control.executor}) runs a rank on "
                             f"{control.cards_per_rank} card; box {box.name!r} serves on {len(serving)}")
        for card in serving:
            if not control.runs(card.type):
                raise ValueError(f"{rw}: control {control.name} ({control.executor}) does not run a "
                                 f"{card.type.key} — its type lists controls {list(card.type.controls)}")
        if control.ssh == "required" and box.ssh is None:
            raise ValueError(f"{rw}: control {control.name} reaches box {box.name!r} over SSH, but the box names no "
                             f"ssh")
        ranks.append(Rank(box=box.name, interconnect=(_text(f"{rw}.interconnect", r["interconnect"])
                                                      if control.interconnect else None)))
        cards.extend(serving)
    if control.contained:
        for r in ranks:
            if boxes[r.box].node_limits is None:
                raise ValueError(f"{where}: control {control.name} runs its node as a container on box {r.box!r}, "
                                 f"which states no {NODE_LIMITS} (memory_gib, cpus, pids, nice): a node on a shared "
                                 f"box runs bounded, never by a default")
    if launch is not None:
        # A RAM tier the block names, on the cards and boxes it would run on: their family's and the boxes' bound.
        _ram_tier(f"{where}.launch", launch.kv_offload_ram_gib, cards, boxes)
        _kv_dtype(f"{where}.launch", launch.kv_cache_dtype, launch.image, cards)
    unlaunchable = ""
    if control.selects:
        # Every model the panel may select fits the card it serves on at the boot launch's memory fraction (or pin).
        # A refusal is KEPT (`unlaunchable`); `parse_fleet` raises it for a boot cluster, the run refuses the rest.
        try:
            for choice in choices:
                engine = _engine(where, choice)
                engine.fit(f"{where}.launch.choices", choice, cards[0].type, _run_by(launch, engine, task),
                           box=boxes[cards[0].box])
        except ValueError as exc:
            unlaunchable = str(exc)
    return Cluster(name=name, control=control.name, url=url, ranks=tuple(ranks), serving_cards=tuple(cards),
                   model=model, launch=launch, choices=choices, boxes=tuple(boxes[r.box] for r in ranks),
                   run_at_boot=boot, default=default, unlaunchable=unlaunchable, task=task,
                   reserves=control.reserves and raw.get("reserve") is not False,
                   images=MappingProxyType(images),
                   ram_usable_fraction=float(fraction) if usable else None, iface=iface, offers=offers,
                   kv_dtypes=kv_dtypes, max_num_batched_tokens=budget, link=_link(where, raw, link_key, ranks, cards,
                                                                                  boxes))


def _link(where: str, raw: Mapping, link_key, ranks, cards, boxes: Mapping[str, Box]) -> str | None:
    """The cluster's `link` — one its topology may name (executors.py) — or None where it states none."""
    if not link_key:
        return None
    from vllm_lb.executors import topology_of
    link = raw["link"]
    probe = topology_of(type("Probe", (), {"ranks": tuple(ranks), "serving_cards": tuple(cards), "link": link}),
                        boxes)
    why = probe.problem()
    if why:
        raise ValueError(f"{where}: {why}")
    return link


def parse_fleet(text: str, source: str, library: Mapping[str, HardwareProfile] | None = None,
                catalog=None) -> Fleet:
    """The fleet `text` describes, validated against the hardware `library` (default: the shipped types) and, for a
    cluster that names its model, the model `catalog` (default: the shipped one, read only then) — or ValueError
    naming `source` and the value."""
    lib = types() if library is None else library
    loaded: list = []

    def models():
        if catalog is not None:
            return catalog
        if not loaded:
            # The shipped catalog RESOLVED AGAINST THIS FLEET (its own targets), not whichever file LB_FLEET names.
            from vllm_lb.catalog import ModelCatalog
            loaded.append(ModelCatalog.load(deployment=catalog_deployment(raw, source, lib)))
        return loaded[0]

    raw = yamlfile.parse(text, source)
    _keys(f"{source}: the fleet", _unretired(source, raw), ("boxes", "clusters", "backends"), ("checkpoints",))
    boxes_raw = _named(f"{source}: boxes", raw["boxes"])
    if not boxes_raw:
        raise ValueError(f"{source}: boxes must name at least one box")
    boxes = {name: _box(source, name, b, lib) for name, b in boxes_raw.items()}
    for attr in ("hostname", "label"):
        seen: dict[str, str] = {}
        for b in boxes.values():
            v = getattr(b, attr)
            if v is not None and v in seen:
                raise ValueError(f"{source}: boxes {seen[v]!r} and {b.name!r} share the {attr} {v!r}")
            if v is not None:
                seen[v] = b.name
    clusters = {name: _cluster(source, name, c, boxes, models)
                for name, c in _named(f"{source}: clusters", raw["clusters"]).items()}
    # A box's ssh is for a cluster that reaches it over SSH: on a box a node without SSH runs on, an ssh no such
    # cluster uses is refused — a box that also carries a pair's rank (its ssh) may host a local node beside it.
    for c in clusters.values():
        control = CONTROLS[c.control]
        for r in c.ranks:
            box = boxes[r.box]
            if control.ssh == "forbidden" and box.ssh is not None and not any(
                    CONTROLS[o.control].ssh == "required" and any(q.box == box.name for q in o.ranks)
                    for o in clusters.values()):
                raise ValueError(f"{source}: clusters.{c.name}: control {c.control} reaches nothing over SSH — "
                                 f"{control.executor} — but box {box.name!r} names ssh {box.ssh}; no cluster that "
                                 f"reaches it over SSH runs there, so nothing would use it")
    # THE CLUSTERS ARE A CATALOG (operator, 2026-10-17; worklog/parallel-clusters.md): what this fleet CAN run. Two
    # declared clusters may share a box, and which run is the RUN's (cluster_run.py: a box held by a running or starting
    # cluster blocks another unless their memory needs fit its pool, `share_problem`). Which run AT BOOT is the file's
    # (`run_at_boot`), named for every cluster when it names several; the one cluster of a fleet runs at boot.
    if len(clusters) > 1:
        unsaid = [c.name for c in clusters.values() if c.run_at_boot is None]
        if unsaid:
            raise ValueError(f"{source}: clusters {', '.join(unsaid)}: with {len(clusters)} clusters declared, each "
                             f"says run_at_boot: true or false — which of the alternatives this LB runs when it starts "
                             f"(the panel's + tab activates another for one run)")
        # ONE DEFAULT PER TASK: the cluster a request of that task naming no model (or a loose name) goes to — exactly
        # one of the task's clusters says so; a task's one cluster is its default.
        lone = set()
        for task in dict.fromkeys(c.task for c in clusters.values()):
            group = [c for c in clusters.values() if c.task == task]
            defaults = [c.name for c in group if c.default]
            if len(group) == 1:
                if group[0].default is False:
                    raise ValueError(f"{source}: clusters.{group[0].name}.default: false — the one {task} cluster of "
                                     f"a fleet is its default; leave the key out")
                lone.add(group[0].name)
            elif len(defaults) != 1:
                raise ValueError(f"{source}: with {len(group)} {task} clusters declared "
                                 f"({', '.join(c.name for c in group)}), exactly one says default: true — the cluster "
                                 f"a {task} request naming no model, or a name no cluster serves, is sent to; found "
                                 f"{', '.join(defaults) if defaults else 'none'}")
        clusters = {n: replace(c, default=bool(c.default) or n in lone) for n, c in clusters.items()}
        # THE BOOT SET shares a box only as the run does (`share_problem`): their memory needs within its pool.
        on: dict[str, list] = {}
        for c in clusters.values():
            for r in (c.ranks if c.run_at_boot else ()):
                on.setdefault(r.box, []).append(c)
        for box, boot in on.items():
            boot = [c for c in boot if not all(measured_beside(o.control, c.control) for o in boot if o is not c)]
            if len(boot) < 2:
                continue
            needs: dict[str, float | str] = {}
            for c in boot:
                try:
                    needs[c.name] = pool_need(c)
                except ValueError as exc:
                    needs[c.name] = str(exc)
            problem = share_problem(boxes[box], needs)
            if problem:
                raise ValueError(f"{source}: clusters {' and '.join(c.name for c in boot)} all say run_at_boot: true "
                                 f"on box {box}, which they cannot share: {problem} — boot one, activate the other "
                                 f"from the + tab")
    else:
        for only in clusters.values():
            if only.default is False:
                raise ValueError(f"{source}: clusters.{only.name}.default: false — the one cluster of a fleet is its "
                                 f"default; leave the key out")
        clusters = {n: replace(c, run_at_boot=True if c.run_at_boot is None else c.run_at_boot, default=True)
                    for n, c in clusters.items()}
    for c in clusters.values():
        if c.run_at_boot and c.unlaunchable:
            raise ValueError(c.unlaunchable)
    backends: dict[str, Backend] = {}
    for name, b in _named(f"{source}: backends", raw["backends"]).items():
        where = f"{source}: backends.{name}"
        _keys(where, b, ("url",))
        box = boxes.get(name)
        if box is None:
            raise ValueError(f"{where}: {name!r} is no box of this fleet (a backend is keyed by its box)")
        if not box.serving_cards:
            raise ValueError(f"{where}: box {name!r} has no card with serves: true for the server to run on")
        backends[name] = Backend(box=name, url=_text(f"{where}.url", b["url"], _URL, "an http(s) URL"))
    used = {r.box for c in clusters.values() for r in c.ranks} | set(backends)
    for b in boxes.values():
        if b.serving_cards and b.name not in used:
            raise ValueError(f"{source}: boxes.{b.name} has a card with serves: true, but no cluster rank or backend "
                             f"of this fleet runs on it")
        if b.name in used and b.label is None:
            raise ValueError(f"{source}: boxes.{b.name} serves, so it needs a label (its GPU lane: what its servers "
                             f"are called)")
    return Fleet(source=source, boxes=MappingProxyType(boxes), clusters=MappingProxyType(clusters),
                 backends=MappingProxyType(backends))


def start_checker(cluster: Cluster, catalog):
    """A cluster Start's own validator for `cluster` (`cluster.DesiredStore.check`), with the fleet's offers, KV dtypes
    and serving cards' arch — what the setup wizard (deploy/setup/render.py) holds a pair's first Start to before the
    gateway ever boots, so a prefilled or typed first Start is refused in the switch's own words at --check, not at
    Phase 4. Nothing is read or written."""
    from vllm_lb.cluster import DesiredStore
    return DesiredStore("", catalog=catalog, offers=cluster.offers, kv_dtypes=cluster.kv_dtypes,
                        archs=tuple(c.type.arch.value for c in cluster.serving_cards))


def catalog_targets(raw: Any, source: str, library: Mapping[str, HardwareProfile] | None = None) -> dict:
    """WHAT THE MODEL CATALOG'S PROFILES LAUNCH ONTO in this fleet (resolve.py): ServingMode → one `resolve.Target`
    (accelerator × platform × TP), or the reason there is none, as a string.

      cluster  the cards of every cluster whose Start names its model from the catalog's cluster profiles — a control
               whose `model` is "forbidden" (the ssh pair, lb/cluster/tp-rank): its ranks' serving cards
      classic  the serving cards of each box a classic server runs on (`backends`), one server a box

    Several such clusters (or servers) must make the SAME target — one catalog resolves one image a profile; a fleet
    whose two pairs differ is refused naming both (docs/design/engines.md, Open items). Read from the raw document so
    the catalog can be resolved before the fleet's own model references are checked against it (parse_fleet)."""
    from vllm_lb.catalog import CLASSIC, CLUSTER_MODE
    from vllm_lb.resolve import target_of
    lib = types() if library is None else library
    _keys(f"{source}: the fleet", _unretired(source, raw), ("boxes", "clusters", "backends"), ("checkpoints",))
    boxes = {name: _box(source, name, b, lib) for name, b in _named(f"{source}: boxes", raw["boxes"]).items()}

    def ranked(where: str, names) -> "object":
        unknown = [b for b in names if b not in boxes]
        if unknown:
            raise ValueError(f"{where}: no box {unknown[0]!r} in boxes")
        return target_of([c for b in names for c in boxes[b].serving_cards], [boxes[b].platform for b in names],
                         where=where)

    found: dict[str, list] = {CLUSTER_MODE: [], CLASSIC: []}
    for name, c in _named(f"{source}: clusters", raw["clusters"]).items():
        control = CONTROLS.get(c.get("control")) if isinstance(c, dict) else None
        if control is not None and control.model == "forbidden":
            ranks = c.get("ranks") if isinstance(c.get("ranks"), list) else []
            found[CLUSTER_MODE].append(ranked(f"{source}: clusters.{name}",
                                              [r.get("box") for r in ranks if isinstance(r, dict)]))
    for box in _named(f"{source}: backends", raw["backends"]):
        found[CLASSIC].append(ranked(f"{source}: backends.{box}", [box]))
    out: dict = {}
    for mode, targets in found.items():
        keys_ = sorted({t.key for t in targets})
        if not targets:
            out[mode] = (f"{source} names no {'cluster the catalog' if mode == CLUSTER_MODE else 'classic server its'} "
                         f"{mode} profiles launch onto")
        elif len(keys_) > 1:
            raise ValueError(f"{source}: the catalog's {mode} profiles would launch onto {len(keys_)} different "
                             f"targets ({'; '.join(keys_)}): one catalog resolves one image a profile")
        else:
            out[mode] = targets[0]
    return out


def catalog_targets_of(path: str | Path, library: Mapping[str, HardwareProfile] | None = None) -> dict:
    """`catalog_targets` of the fleet file at `path`."""
    p = Path(path)
    return catalog_targets(yamlfile.read(p), p.name, library)


#: A checkpoint entry's keys (`checkpoints.<id>`): where the model sits, its uncensored twin, its drafter checkpoint.
CHECKPOINT_KEYS = ("path", "uncensored", "draft")


def catalog_deployment(raw: Any, source: str, library: Mapping[str, HardwareProfile] | None = None):
    """WHAT THE MODEL CATALOG IS ASSEMBLED FOR in this fleet (`catalog.Deployment`): its targets (`catalog_targets`)
    and `checkpoints` — model id → the folders, relative to the models roots (MODELS_SEARCH_DIRS; a node's
    EDGE_MODELS_DIR), where THIS deployment staged it: `path`, its `uncensored` twin, its `draft` checkpoint. Where a
    checkpoint sits was a catalog row's `path` until 2026-10-30; it is the deployment's, so it lives here. A model the
    block does not name is not staged here: the catalog lists it as absent and refuses a Start of it, naming this key."""
    from vllm_lb.catalog import Checkpoint, Deployment
    targets = catalog_targets(raw, source, library)
    block = raw.get("checkpoints", {}) if isinstance(raw, dict) else {}
    if not isinstance(block, dict):
        raise ValueError(f"{source}: checkpoints must map a model id to {{path, uncensored, draft}}")
    out = {}
    for mid, c in block.items():
        w = f"{source}: checkpoints.{mid}"
        _keys(w, c, ("path",), ("uncensored", "draft"))
        for k in CHECKPOINT_KEYS:
            if k in c and (not isinstance(c[k], str) or not c[k].strip()):
                raise ValueError(f"{w}.{k} must be a folder relative to the models roots, got {c[k]!r}")
        out[str(mid)] = Checkpoint(c["path"], c.get("uncensored"), c.get("draft"))
    return Deployment(targets, MappingProxyType(out), source)


def catalog_deployment_of(path: str | Path, library: Mapping[str, HardwareProfile] | None = None):
    """`catalog_deployment` of the fleet file at `path`."""
    p = Path(path)
    return catalog_deployment(yamlfile.read(p), p.name, library)


def load_fleet(path: str | Path, library: Mapping[str, HardwareProfile] | None = None, catalog=None) -> Fleet:
    """The fleet file at `path`, validated — or ValueError naming the file and the value."""
    p = Path(path)
    return parse_fleet(yamlfile.read_text(p), p.name, library, catalog)


def validate_launch(cluster: Cluster, overrides: Mapping[str, Any]) -> NodeLaunch:
    """The launch an INTENT asks for: `cluster`'s launch block (its boot default) with `overrides` applied, validated
    by the SAME code as the block itself — the model one of its `choices`, the figures within that model's catalog
    limits (`_figures`), the prefill chunk within vLLM's own rules (`_chunk`; null gives it back to vLLM's default),
    its weights within the serving card's share (`_fit`; a pinned pool, `_pin`, within the whole card), a RAM tier
    within its family and its box's bound (`_ram_tier`) — what the LB's panel may request and what the host's
    applier checks before it runs anything. Only `MOVABLE` keys move; `{}` is the boot default; null gives an optional
    key back to vLLM.
    ValueError naming the key for a cluster this LB does not drive or that names no launch, an unknown, fixed or
    derived key, a model the fleet does not offer, a value out of bounds, or a launch that does not fit."""
    where = f"the intent for clusters.{cluster.name}"
    control = CONTROLS[cluster.control]
    if not control.drives:
        raise ValueError(f"{where}: control {control.name} — this LB only observes it ({control.executor}), so no "
                         f"intent of the LB's moves it")
    launch = cluster.launch
    if not isinstance(launch, SelectedLaunch) or not cluster.choices:
        raise ValueError(f"{where}: the cluster names no launch block with choices — nothing to move from")
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{where} must be a mapping of launch keys, got {type(overrides).__name__}")
    unknown = [str(k) for k in overrides if k not in (*MOVABLE, *_FIXED, *_DERIVED, "choices")]
    fixed = [k for k in overrides if k in (*_FIXED, "choices")]
    derived = [k for k in overrides if k in _DERIVED]
    if unknown or fixed or derived:
        raise ValueError(f"{where} moves only {', '.join(MOVABLE)}"
                         + (f"; {', '.join(unknown)} is no launch key" if unknown else "")
                         + (f"; {', '.join(fixed)} is the fleet file's alone" if fixed else "")
                         + (f"; {', '.join(derived)} follows the model (its catalog row's)" if derived else ""))
    ref = overrides.get("model", launch.model)
    model = next((m for m in cluster.choices if m.ref == ref), None)
    if model is None:
        raise ValueError(f"{where}.launch.model {ref!r} is not among the fleet's choices: "
                         f"{', '.join(m.ref for m in cluster.choices)}")
    w = f"{where}.launch"
    engine = _engine(w, model)
    base = _run_by(launch, engine, cluster.task)
    asked = {**{k: getattr(base, k) for k in (*_FIGURES, *OPTIONAL)},
             **{k: v for k, v in overrides.items() if k in (*_FIGURES, *OPTIONAL)}}
    length, seqs, util = _figures(w, asked, model)
    chunk, pin, ram = _chunk(w, asked, seqs), _pin(w, asked), _ram(w, asked)
    image = cluster.images[engine.name]
    dtype = _kv_dtype(w, asked.get(KV_DTYPE), image, cluster.serving_cards)
    moved = SelectedLaunch(image=image, engine=engine.name, served_name=model.spec.served_name, max_model_len=length,
                           max_num_seqs=seqs, gpu_memory_utilization=util, max_num_batched_tokens=chunk,
                           kv_cache_memory=pin, kv_offload_ram_gib=ram, kv_cache_dtype=dtype,
                           reasoning_parser=launch.reasoning_parser, tool_call_parser=launch.tool_call_parser,
                           model=model.ref, checkpoint=model.checkpoint, max_queue=_queue(w, asked),
                           compute_type=_compute(w, asked))
    engine.fit(w, model, cluster.serving_cards[0].type, moved, box=cluster.boxes[0])
    _ram_tier(w, ram, cluster.serving_cards, {b.name: b for b in cluster.boxes})
    return moved


def engine_block(cluster: Cluster, ref: str | None) -> dict:
    """WHAT /lb/status SAYS OF A CLUSTER'S SERVER, from the declarations alone: its task, the engine of the choice `ref`
    (else the boot launch's) with its settings and tiles, every engine its choices run, the tiles the panel draws, and
    the card it serves on as the fleet's hardware type names it — never the engine's own device string (CTranslate2's
    HIP build answers "cuda" on an AMD card). A cluster with no launch (the pair, an observed node) names no engine:
    its tiles are its task's."""
    choice = next((m for m in cluster.choices if m.ref == ref), None)
    name = (ENGINES[choice.spec.node_engine].name if choice is not None and choice.spec.node_engine
            else cluster.launch.engine if cluster.launch is not None else None)
    engine = ENGINES[name].view(cluster.task) if name else None
    card = cluster.serving_cards[0].type
    runtime = card.gpu_runtime
    return {"task": cluster.task, "engine": engine,
            "engines": {e: ENGINES[e].view(cluster.task) for e in cluster.images},
            "tiles": engine["tiles"] if engine else list(TASK_TILES[cluster.task]),
            "hardware": f"{card.name} · {runtime.value}" if runtime.known else card.name}


def pool_need(cluster: Cluster, overrides: Mapping[str, Any] | None = None) -> float:
    """THE GiB OF ITS BOX'S GPU POOL a one-host node's launch takes — its launch block moved by an intent's `overrides`,
    judged first by `validate_launch` — what clusters sharing one box sum (`share_problem`). ValueError where it cannot
    be said: a cluster this LB does not drive, names no launch, or whose launch the fleet refuses."""
    launch = validate_launch(cluster, overrides or {})
    model = next(m for m in cluster.choices if m.ref == launch.model)
    return ENGINES[launch.engine].pool_gib(model, cluster.serving_cards[0].type, launch)


def measured_beside(starting: str, holder: str) -> bool:
    """Is a cluster of control `holder` on a box no obstacle to one of control `starting` — a ONE-HOST node (`none`,
    started by its own deployment, or `local`, by its host's applier) beside a pair (`ssh`), whose start measures the
    box's free memory itself (tp-rank's gate on MemAvailable), the node's use in it. Not the other way: a node starting
    beside a running pair has no such measurement, and beside anything sized from stated needs (`share_problem`) a node
    that states none still holds its box."""
    return holder in ("none", "local") and starting == "ssh"


def share_problem(box: Box, needs: Mapping[str, float | str]) -> str:
    """MAY THESE CLUSTERS HOLD `box` AT ONCE — "" when they may, else why, in their numbers. `needs`: each cluster's
    `pool_need` in GiB, or why it is unknown. A box is shared only where its memory is ONE pool the fleet bounds (its
    serving card's family: `shared_pool`, a unified box's gpu_ram_max_gib), every need is known, and their sum is
    within the bound. Anything else holds the box whole (fail closed) — never an eviction, never a guessed need."""
    bound, why = box.serving_cards[0].type.memory.shared_pool(box)
    if bound is None:
        return f"{why}: one cluster holds it whole"
    unknown = [f"{name}'s memory need is unknown ({need})" for name, need in needs.items() if isinstance(need, str)]
    if unknown:
        return "; ".join(unknown) + f": box {box.name} is shared only by clusters whose needs are known"
    total = sum(needs.values())
    if total > bound:
        return (" + ".join(f"{name} {need:.2f} GiB" for name, need in needs.items())
                + f" = {total:.2f} GiB > box {box.name}'s gpu_ram_max_gib {bound} GiB")
    return ""


def _served_spec(cluster: Cluster, launch: NodeLaunch):
    """The catalog row a node's launch serves: the selected choice's (`SelectedLaunch.model`), else the cluster's own."""
    ref = getattr(launch, "model", None)
    return next((m for m in cluster.choices if m.ref == ref), cluster.model).spec


def node_env(fleet: Fleet, name: str, overrides: Mapping[str, Any] | None = None) -> list[str]:
    """The KEY=VALUE lines a standalone node's compose reads (deploy/standalone), from cluster `name`'s launch block —
    moved by an intent's `overrides` (`validate_launch`) when given — its GPU the cluster's ONE serving card's
    nvidia-smi index, so no server file names the card again. ValueError for a cluster that does not exist, names no
    launch, serves on other than one card, or an intent `validate_launch` refuses."""
    cluster = fleet.clusters.get(name)
    if cluster is None:
        raise ValueError(f"{fleet.source}: no cluster {name!r}; the clusters are {', '.join(fleet.clusters) or 'none'}")
    launch = cluster.launch
    if launch is None:
        raise ValueError(f"{fleet.source}: cluster {name!r} names no launch block — its node is not launched from "
                         f"this fleet file")
    if overrides is not None:
        launch = validate_launch(cluster, overrides)
    if len(cluster.serving_cards) != 1:
        raise ValueError(f"{fleet.source}: cluster {name!r} serves on {len(cluster.serving_cards)} cards; a "
                         f"standalone node runs on one")
    card = cluster.serving_cards[0]
    # HOW THE CONTAINER ATTACHES THE CARD is the card type's fact (`gpu_runtime`): up.sh runs the node with the compose
    # override of that name (deploy/standalone/node-gpu.<runtime>.yml) — selected by data, never by a vendor branch.
    limits = fleet.boxes[card.box].node_limits
    if limits is None:
        raise ValueError(f"{fleet.source}: box {card.box!r} states no {NODE_LIMITS} (memory_gib, cpus, pids, nice): "
                         f"its node container runs bounded, never by a default")
    if not card.type.gpu_runtime.known:
        raise ValueError(f"{fleet.source}: cluster {name!r} serves on a {card.type.key}, whose type states no "
                         f"gpu_runtime ({card.type.gpu_runtime.source}): no container recipe attaches it")
    patched = ENGINES[launch.engine].mfu_patch and _served_spec(cluster, launch).quant_method in MFU_PATCHED_QUANTS
    return [f"EDGE_NODE_IMAGE={launch.image}", f"EDGE_NODE_GPU={card.index}",
            f"EDGE_NODE_GPU_RUNTIME={card.type.gpu_runtime.value}",
            *launch.model_lines(),
            # WHAT ITS ENGINE READS, and nothing another engine would: each line a setting it declares.
            *ENGINES[launch.engine].env_lines(launch),
            # THE MFU BYTE-SIZE PATCH, by the checkpoint's own quant method (the catalog's), on an engine that loads it
            # (`NodeEngine.mfu_patch`): on only where it is needed.
            f"EDGE_NODE_MFU_PATCH={'on' if patched else 'off'}",
            # THE CONTAINER'S LIMITS, the box's (`node_limits`): memory (no swap), CPUs, processes, niceness.
            *limits.lines(),
            # The CPU the image must be built for — compose's `platform:` refuses an image built for another.
            f"EDGE_NODE_PLATFORM={fleet.boxes[cluster.serving_cards[0].box].platform}",
            # THE SERVER IT RUNS (`ENGINES`, the choice's catalog `node_engine`): up.sh adds node-engine.<engine>.yml.
            f"EDGE_NODE_ENGINE={launch.engine}"]


USAGE = "usage: python -m vllm_lb.fleet node-env <fleet file> <cluster> [<intent.json>]"


def _intent(path: str) -> Any:
    """The overrides an intent file holds: one JSON object of launch keys — or ValueError naming the file."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{Path(path).name}: not a readable JSON intent: {exc}") from None


def main(argv: list[str]) -> int:
    """`node-env <fleet file> <cluster> [<intent.json>]`: print `node_env`'s lines — the intent's overrides applied
    when one is named, so the host's applier needs nothing but this image. 64 (EX_USAGE) for anything else, 78
    (EX_CONFIG) — one line naming the value on stderr, nothing on stdout — for a fleet or an intent that refuses."""
    if len(argv) not in (3, 4) or argv[0] != "node-env":
        print(USAGE, file=sys.stderr)
        return 64
    try:
        lines = node_env(load_fleet(argv[1]), argv[2], _intent(argv[3]) if len(argv) == 4 else None)
    except ValueError as exc:
        print(f"fleet: refused: {exc}", file=sys.stderr)
        return 78
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
