"""Configuration — WHAT the fleet is comes from its fleet file (LB_FLEET), HOW this LB is wired from the environment.

AGENTS.md, operator rule 2026-10-11: a deployment is configuration, never code, and a value lives in ONE place. The
fleet file (vllm_lb/fleet.py) names the boxes, their cards, the cluster and the classic servers; this module turns it
into the fields the LB has always read — `cluster_hosts`, `cluster_url`, `cluster_control`, `vllm_boxes` (the LB's
OWN `host_label → box → URL` map: broker_parrot stores no URL, the claim stamps `claimed_by = host_label`) and
`hw_hosts` — and keeps the `Fleet` itself (`fleet`) for whatever asks about cards. The settings those fields used to
come from are RETIRED (`RETIRED_KEYS`): present in the environment at all, they refuse the start naming where their
value lives now. Every other knob is the environment's, with its default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from vllm_lb import topology
from vllm_lb.cluster import ClusterHost, parse_hosts
from vllm_lb.connect import parse_flag, parse_tunnel
from vllm_lb.fleet import CONTROLS, Fleet, load_fleet
from vllm_lb.host_guard import Authority, parse_allowed
from vllm_lb.cluster_defaults import (CLUSTER_KV_CACHE_MEMORY, KV_BYTES_PER_TOKEN_PER_RANK,
                                      recommended_for)


@dataclass(frozen=True)
class VllmBox:
    host_label: str   # what the claim stamps into `claimed_by` (e.g. box-a-gpu)
    box: str          # the physical box id `avoid_box` speaks (e.g. box-a)
    url: str          # reachable from THIS host (e.g. http://192.0.2.2:8000)


def parse_boxes(spec: str) -> tuple[VllmBox, ...]:
    """`label=box=url,label=box=url` → boxes — the RETIRED `LB_VLLM_BOXES` form (the LB reads its fleet file), kept
    as the compact way a test builds `Config.vllm_boxes` and proves the fleet gives the same. Blank → none."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        parts = item.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"LB_VLLM_BOXES entry must be label=box=url, got {item!r}")
        out.append(VllmBox(host_label=parts[0], box=parts[1], url=parts[2].rstrip("/")))
    return tuple(out)


def parse_hw_hosts(spec: str) -> dict[str, str]:
    """`hostname=box,hostname=box` → the map the hardware listener labels senders with — the RETIRED `LB_HW_HOSTS`
    form (the LB reads `boxes.<box>.hostname` from its fleet file), kept for tests. Blank → `{}`."""
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        host, _, box = item.partition("=")
        if not host.strip() or not box.strip():
            raise ValueError(f"LB_HW_HOSTS entry must be hostname=box, got {item!r}")
        out[host.strip()] = box.strip()
    return out


#: Settings whose value moved into the fleet file (AGENTS.md, 2026-10-11). Present in the environment at all — a blank
#: line too — they REFUSE the start, naming where the value lives now: never silently ignored, so no value ever has two
#: definitions (the chat's auth.py RETIRED_KEYS pattern).
RETIRED_KEYS = {
    "LB_VLLM_BOXES": "the fleet file's backends (one classic server per box, named by the box's label)",
    "LB_HW_HOSTS": "the fleet file's boxes.<box>.hostname",
    "LB_CLUSTER_HOSTS": "the fleet file's clusters.<name>.ranks, with each box's label and ssh",
    "LB_CLUSTER_URL": "the fleet file's clusters.<name>.url",
    # The box's prefill budget is the deployment's (operator, 2026-10-29): the fleet's, per cluster — never an env
    # default standing in for it, and never tp-rank's own number.
    "CLUSTER_MAX_NUM_BATCHED_TOKENS": "the fleet file's clusters.<name>.max_num_batched_tokens (the prefill budget a "
                                      "launch uses when neither the operator's knob nor the model's own says)",
    # A GPU's peaks are its CARD's, per box (operator, 2026-10-11): one LB-wide claim cannot describe two cards.
    "LB_GPU_PEAK_GBPS": "each serving card's hardware type (vllm_lb/hardware/types: peak_gbps), named by the "
                        "fleet file's boxes.<box>.cards",
    "LB_GPU_PEAK_TFLOPS": "each serving card's hardware type (vllm_lb/hardware/types: peak_tflops per precision), "
                          "named by the fleet file's boxes.<box>.cards",
    "LB_GPU_PEAK_TFLOPS_PRECISION": "the model's own precision (vllm_lb/cores.py) against each serving card's "
                                    "hardware type",
}
#: A value the MODEL owns is its catalog row's (operator, 2026-10-29: "why do we have LB_CONTEXT_LENGTH? Isn't that set
#: in configs? Same for MAX_NUM_SEQS?"): an LB-wide number standing in for a per-model figure is a fallback — the
#: catalog's `limits` and each cluster row's `tuned` setup are the ONE place — so each such key is refused naming
#: the row field, never read beside it.
class InCatalog(str):
    """A RETIRED key whose value is a field of the model library (lb/vllm_lb/library): refused naming the field."""


RETIRED_KEYS["LB_CONTEXT_LENGTH"] = InCatalog(
    "the model catalog: the lease model's row, limits.max_model_len (the classic default row, or the row LB_LEASE_MODEL "
    "names)")
RETIRED_KEYS["LB_CLUSTER_MAX_NUM_SEQS"] = InCatalog(
    "the model catalog: each cluster row's launch.cluster.tuned.max_num_seqs (the concurrency a Start of that model is "
    "offered)")
RETIRED_KEYS["LB_CLUSTER_MAX_MODEL_LEN"] = InCatalog(
    "the model catalog: each cluster row's launch.cluster.tuned.max_model_len (the context a Start of that model is "
    "offered)")
#: THE FALLBACK TIER IS RETIRED (operator, 2026-10-17: "Why fallback box? If can't run then can't run, no fallback
#: box."): no ollama backend absorbs a request no server can take — it waits for one on its way or is refused naming
#: why and what is live. Every setting of it is refused at boot, naming that, never silently ignored.
class Gone(str):
    """A RETIRED key whose value lives nowhere now: the feature is gone. Refused in its own words."""


_NO_FALLBACK = Gone("the LB's fallback tier is retired (operator, 2026-10-17: a request no server can run waits or is "
                    "refused naming why and what is live, never moved to a fallback)")
RETIRED_KEYS.update({k: _NO_FALLBACK for k in (
    "LB_FALLBACK_URLS", "LB_FALLBACK_MODEL", "LB_FALLBACK_MAX_MODEL_LEN", "LB_FALLBACK_BOXES", "LB_FALLBACK_BOX_LABEL",
    "LB_FALLBACK_IDLE_RELEASE_S", "LB_FALLBACK_LEASE_MAX_HOLD_S")})
#: KEEP LOADED IS RETIRED (operator, 2026-10-17): no operator order pins a classic box's lease; a box's server is held
#: by demand alone. Its state file's setting is refused at boot like the fallback's.
#: THE FLAT MEMORY GATE IS RETIRED (operator rule: no fallbacks, 2026-10-29): a Start is gated on what its selection
#: COSTS on each box (memory_estimate, per rank role); one the LB cannot size is refused naming why, never admitted
#: on a guessed 100 GiB (generation 10 launched onto 77 GiB that way).
RETIRED_KEYS["LB_CLUSTER_MIN_MEM_GIB"] = Gone("the LB's flat memory requirement is retired (2026-10-29): a Start is "
                                              "gated on what its selection costs on each box (the catalog's memory "
                                              "footprint, per rank role), and one the LB cannot size is refused "
                                              "naming what is missing")
RETIRED_KEYS["LB_KEEP_STATE"] = Gone("the LB's keep-loaded order is retired (operator, 2026-10-17): a box's server is "
                                     "held by demand alone")


@dataclass(frozen=True)
class ClusterConfig:
    """ONE cluster as this LB reaches it (worklog/parallel-clusters.md): its fleet NAME — the key of its state dir, its
    loop, its view, its backend and its panel card — its hosts head first, the head's API URL and its control."""

    name: str
    hosts: tuple[ClusterHost, ...]
    url: str
    control: str
    #: Runs when this LB starts (fleet `run_at_boot`); False = declared, activated for one run from the panel.
    boot: bool = True
    #: THE DEFAULT CLUSTER of its task (fleet `default`): where a request naming no model, or a loose name, is routed —
    #: alone.
    default: bool = False
    #: What it answers (fleet `Cluster.task`: chat, or transcription for an audio upload).
    task: str = "chat"
    #: Why it cannot launch (fleet `Cluster.unlaunchable`); "" = it can. Activate refuses it naming this.
    unlaunchable: str = ""
    #: The share of each rank box's MemTotal a start may take (fleet `ram_usable_fraction`); None = this control
    #: takes none, and the LB_CLUSTER_* memory policy applies (`Config.memory_policy`).
    ram_usable_fraction: float | None = None
    #: The NIC its ranks talk over (fleet `iface`): tp-rank's CLUSTER_IFACE; None = this control binds none.
    iface: str | None = None
    #: The catalog rows its panel offers (fleet `choices`), as row keys; None = every row (`DesiredStore.offers`).
    offers: tuple[str, ...] | None = None
    #: The KV dtypes it runs (fleet `kv_cache_dtypes`); None = every one its images take (`DesiredStore.kv_dtypes`).
    kv_dtypes: tuple[str, ...] | None = None
    #: The prefill budget its ranks launch with when nothing nearer says (fleet `max_num_batched_tokens`): tp-rank's
    #: CLUSTER_MAX_NUM_BATCHED_TOKENS on every call (`cluster_run.tp_rank_env`). None = this control takes none.
    max_num_batched_tokens: int | None = None


def topology_of(fleet: Fleet | None) -> dict:
    """The Config fields a fleet declares: EVERY cluster (`clusters`, in the fleet's order — each its hosts head first,
    URL, control), its classic servers and its hardware map. A box whose classic server and unstarted node would share
    its label is refused here."""
    if fleet is None:
        return {"clusters": (), "vllm_boxes": (), "hw_hosts": {}}
    clusters = []
    for cluster in fleet.clusters.values():
        reach = CONTROLS[cluster.control]
        hosts = tuple(ClusterHost(label=fleet.boxes[r.box].label, box=r.box,
                                  ssh_target=(fleet.boxes[r.box].ssh or "") if reach.ssh == "required" else "",
                                  interconnect=r.interconnect or "")
                      for r in cluster.ranks)
        # A one-host node the broker leases (control local) is named by its box's label — its lane, the name a claim
        # stamps — as a classic server there would be; an observed node (none) takes no lease, so it names no lane.
        if cluster.control != topology.SSH and cluster.reserves:
            shared = sorted({h.box for h in hosts} & set(fleet.backends))
            if shared:
                raise ValueError(f"{fleet.source}: box {shared[0]!r} has a classic server (backends) and the node of "
                                 f"cluster {cluster.name!r}: both would be named {fleet.boxes[shared[0]].label!r}")
        clusters.append(ClusterConfig(name=cluster.name, hosts=hosts, url=cluster.url.rstrip("/"),
                                      control=cluster.control, boot=bool(cluster.run_at_boot),
                                      default=bool(cluster.default), unlaunchable=cluster.unlaunchable,
                                      task=cluster.task, ram_usable_fraction=cluster.ram_usable_fraction,
                                      iface=cluster.iface, offers=cluster.offers, kv_dtypes=cluster.kv_dtypes,
                                      max_num_batched_tokens=cluster.max_num_batched_tokens))
    vllm = tuple(VllmBox(host_label=fleet.boxes[box].label, box=box, url=b.url) for box, b in fleet.backends.items())
    return {"clusters": tuple(clusters), "vllm_boxes": vllm, "hw_hosts": dict(fleet.hw_hosts)}


@dataclass(frozen=True)
class Config:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8010
    #: THE HOST ALLOWLIST (vllm_lb/host_guard.py): the names beyond the listener's own loopback names that the gateway
    #: answers to — the Host its reverse proxy / tunnel presents, the docker-network names containers dial. From
    #: LB_ALLOWED_HOSTS, REQUIRED (`from_env` refuses the boot without it; `loopback` = none beyond loopback). The
    #: class default is the strictest set — loopback only — for the tests that build a Config by hand.
    allowed_hosts: tuple[Authority, ...] = ()
    #: The classic vLLM boxes this LB dials: the fleet file's `backends` (`topology_of`). EMPTY by default: the LB
    #: names no machine it was not told about, and with none it serves only through the cluster or answers 503.
    vllm_boxes: tuple[VllmBox, ...] = ()
    #: THE FLEET (vllm_lb/fleet.py), from the file LB_FLEET names: every box, every card and which serve, the cluster
    #: and the classic servers. None = no fleet file: no machine named. What asks about CARDS — a peak, a core count,
    #: a box's memory model — asks this, never an LB-wide setting.
    fleet: Fleet | None = None
    max_leases: int = 2
    idle_release_s: float = 600.0
    #: Operator, 2026-09-12: "queue depth at least 1 for 15 seconds, then ask broker_parrot
    #: for an additional server." The sample is the deepest `waiting` on a leased vLLM or the
    #: number of our own starved admissions in the window, whichever is larger; it must hold
    #: for the whole sustain window before a lease is asked for, and the ask is still capped by
    #: what the broker reports it can start.
    scale_up_waiting: int = 1
    scale_up_sustain_s: float = 15.0
    probe_interval_s: float = 5.0
    probe_timeout_s: float = 3.0
    tick_s: float = 2.0
    #: QUEUE DEPTH — `waiting + running` — past which a vLLM is DEMOTED below every server with room
    #: instead of preferred. Sharing a vLLM with broker_parrot's own llm_chat jobs is
    #: intended; queueing behind them until the first-byte budget expires is not.
    #:
    #: It must count RUNNING, not just waiting. A vLLM at its `--max-num-seqs` reports
    #: `waiting=0` until someone is actually behind, so a waiting-only test calls a completely
    #: full server idle — measured 2026-09-12 on box-b (`wait=0 run=2` with max-num-seqs 2),
    #: where the next request queued for over five minutes.
    #:
    #: `max-num-seqs - 1`. Benchmarked on the deployed server (3 slots, effort=low, 200 tok):
    #: throughput scaled LINEARLY to concurrency 3 (13.0 / 27.0 / 38.9 tok/s = x1.00 / x2.08 /
    #: x3.00) and collapsed past it (conc 4 -> 28.7 tok/s, worst latency 10.3 s -> 22.0 s). So
    #: every slot is worth filling and none beyond. At 1 the LB abandoned a free slot for the
    #: ollama fallback (retired since, 2026-10-17), which the same day took 258116 ms where vLLM took 476 ms.
    vllm_max_depth: int = 2
    #: Reasoning-effort words the served chat template does NOT accept, mapped to one it
    #: does. qwen3.8 takes low/medium/xhigh and raises on anything else; Claude Code sends
    #: `high` (and `max` under --effort max), so without this every session 500s in ~4 ms.
    #: `high` maps to `medium`, which injects NO reasoning instruction — measured 2026-09-12:
    #: with `high` -> `xhigh` the operator typed `ping` and the model "Thought for 6m 47s"
    #: (~5,300 reasoning tokens at 13 tok/s) before answering "pong". `max` still reaches
    #: `xhigh`, so deep reasoning is one flag away rather than the default on every turn.
    effort_map: dict = field(default_factory=lambda: {"high": "medium", "max": "xhigh"})
    #: How many consecutive 5xx responses (no success in between) mark a backend as broken
    #: rather than busy. It is then penalised so the router prefers anything else — the point
    #: of the whole daemon is that a loaded model never sits at 0% while requests die.
    failure_streak_max: int = 3
    #: A LEASED vLLM that takes longer than this to produce its first body byte counts as a
    #: starved admission — the operator's own definition of "waited too long to even reach
    #: inference". Starved admissions feed the same pressure tracker as queue depth.
    starve_after_s: float = 30.0
    #: How long a turn that nothing holds WAITS for a claim before it is refused. The refusal is
    #: still the demand signal; but when a server we could route to is live and only broker_parrot's
    #: claim is missing, that claim takes seconds on an
    #: idle box (median 3 s, 2026-09-15), and a client that does not retry a 503 turned those
    #: seconds into an error. Never waited on: nothing a claim could serve, or a broker we cannot
    #: reach. 0 refuses at once.
    admission_wait_s: float = 20.0
    #: Budget for the FIRST byte. This legitimately covers queueing behind a broker_parrot job
    #: on a shared box, an 18 GB model load, and the
    #: prefill of a long prompt — none of which is a fault, so it is generous. Client-side
    #: API_TIMEOUT_MS should be at least this (fullclaudevllm sets it).
    first_byte_timeout_s: float = 3600.0
    #: Budget for SILENCE once bytes are flowing. Progress, not wall time, is what a timeout
    #: should measure: a turn may stream for an hour and be perfectly healthy, and silence is the
    #: only evidence that a backend is stuck.
    #:
    #: 300 s until 2026-10-03, which was right when a turn was a question and an answer and wrong for
    #: an agentic one: a long tool call, the prefill of a follow-up turn, or an extended reasoning
    #: block is silent for longer than that while being perfectly healthy, and the client saw
    #: "the response stopped arriving". 1800 s now matches the client's own stream-idle watchdog —
    #: `CLAUDE_STREAM_IDLE_TIMEOUT_MS` in fullclaudevllm, 30 min, which is that watchdog's documented
    #: maximum — so neither side gives up before the other, and raising one alone would have changed
    #: nothing. The cost is detection latency: a genuinely wedged engine is now called stalled after
    #: 30 minutes rather than 5. `first_byte_timeout_s` and nginx's proxy_read_timeout both sit above.
    stall_timeout_s: float = 1800.0
    #: SSE KEEPALIVE: after this long with no byte from the backend, a relayed `text/event-stream` gets
    #: `: keepalive` — a comment every parser ignores — at an event boundary, and again for as long as
    #: the silence lasts. Claude Code (measured, 2.1.288) shows "Waiting for API response · will retry
    #: in 29m" after 20 s without a BYTE and aborts after its idle watchdog (30 min with
    #: CLAUDE_STREAM_IDLE_TIMEOUT_MS, 180 s without), while the silences are healthy: a long prefill
    #: before SGLang's adapter emits `message_start`, tool arguments the parser holds until whole. 15 s
    #: is inside the 20 s banner. It is never progress: `stall_timeout_s` still counts the BACKEND's
    #: silence. 0 = off.
    sse_keepalive_s: float = 15.0
    #: THE DRAIN — SIGTERM, i.e. every `docker compose up -d` that recreates this container. Turns in
    #: flight stream to their end for up to this long; a turn arriving meanwhile is held, no byte sent,
    #: and answered 503 `draining` (Retry-After `drain_retry_after_s`) the moment the drain is over, so
    #: it costs the client one retry onto the new container instead of its whole retry budget. Until
    #: 2026-10-03 there was no drain: 2-7 Claude Code sessions at a time lost their stream 1-2 s after a
    #: redeploy (46 on 10-01/10-02). 900 s, measured on 1,179 local-model requests from real sessions
    #: (2026-09-30..10-03, transcripts): p50 82 s, p95 533 s, p99 852 s, max 1,189 s — a turn in flight
    #: at a random SIGTERM is still running 600 s later 3.4 % of the time, 900 s later 0.6 %. A drain
    #: ends as soon as the last turn does, so the window only bounds the tail. compose's
    #: `stop_grace_period` must outlast it (tests/test_drain.py). 0 = exit at once.
    drain_s: float = 900.0
    #: What the 503 `draining` tells the client to wait: the new container's create + boot (~3 s).
    drain_retry_after_s: float = 5.0
    #: THE HOLD: a turn that finds nothing live while the operator WANTS the cluster and it is on its
    #: way (a restart or a model switch: restarting → … → starting → running) waits server-side — no
    #: byte sent — for the pair, up to this long, instead of a 503 its client retries into the ground.
    #: 158 Claude Code errors on 10-01/10-02 were exactly that. 900 s: the boot log's 43 serving
    #: starts took p50 501 s, p90 743 s, 41 of 43 under 900 s; and the client waits up to
    #: API_TIMEOUT_MS (3600 s) for headers with no banner, so the hold leaves 45 min for the
    #: prefill. A failed or blocked start ends the hold at once. 0 = refuse at once.
    cluster_hold_s: float = 900.0
    #: At most this many turns wait at once — held for the cluster, or held through a drain. Past it a
    #: turn is refused at once (503, Retry-After): a wait must never become an unbounded pile of
    #: threads. 64 = twice the most seats the pair has been run with (32) — a whole swarm fits.
    hold_max: int = 64
    #: The `Retry-After` on every other 503 `no_backend`. Claude Code reads only this header on a 503,
    #: and one over 60 s makes it give up after a single request (`api_request_retry_after_too_long`),
    #: so it is sent capped at 60 whatever is set here (proxy.RETRY_AFTER_MAX_S).
    retry_after_s: float = 30.0
    connect_timeout_s: float = 5.0
    penalty_s: float = 10.0
    #: BROKER_PARROT IS OPT-IN (`LB_BROKER_ENABLED`: 1/true/yes; absent or anything else is OFF — operator,
    #: 2026-10-09: most people who run this LB do not run broker_parrot). OFF is the operator's explicit
    #: choice, not a fallback: the engine (`queue_workflows`) is never imported, the broker DB is never read
    #: or written, no lease, reservation or priority node_job is asked for and no worker_controls row is
    #: written. A live configured backend is then routable without a lease, admission never waits for a
    #: claim, and a cluster Start launches on its memory and availability gates alone. ON is the
    #: leased-only gateway every setting below the switch describes.
    broker_enabled: bool = False
    db_url_env: str = "LB_BROKER_DB_URL"
    lease_workflow: str = "vllm_lb_lease"
    lease_node_module: str = "system.vllm_lease"
    #: The model the lease DECLARES it is for, carried in the job's `inputs` — never in
    #: `required_model`. Blank = the catalog's classic default's served name; `LB_LEASE_MODEL` overrides.
    #:
    #: It goes in `inputs` on purpose. `required_model` is a hard capability FILTER in
    #: `claim_next_gpu_job` (broker_parrot `node_queue.py:1064-1075`): a row naming a model is
    #: claimable ONLY where the worker advertised that id, and the advertised list is narrowed
    #: by live memory (`claim_worker.py:2771-2820`) — on a GB10 box `qwen3.8:latest` survives that
    #: gate only while the vLLM the lease exists to START is already serving it. Naming it
    #: would strand the lease in exactly the state the lease is for, arm the unassignable
    #: sweep, and move the box lease slot key off `__llm_server__`. `inputs` is written at
    #: insert and read by no claim predicate, no ORDER BY and no sweep.
    #: See `worklog/vllm-lb-lease-model.md`.
    lease_model: str = ""
    lease_max_hold_s: float = 21600.0
    lease_backoff_max_s: float = 300.0
    #: Serve from a vLLM the LB does NOT hold a broker lease on. FALSE: owning a server is
    #: not holding it, and traffic to an unleased sidecar is capacity taken without telling
    #: the fleet. Set it True only for a box broker_parrot does not manage.
    use_unleased_vllm: bool = False
    #: Anthropic's SERVER-side `web_search` tool, which Claude Code's WebSearch delegates to
    #: the endpoint (see websearch.py). The LB performs it against this SearXNG JSON API;
    #: blank disables the interception and the request forwards to the backend as before.
    search_url: str = "http://searxng:8080/search"
    search_max_results: int = 10
    #: Chars of each result's snippet folded into its title (the client keeps title+url only).
    search_snippet_chars: int = 180
    search_timeout_s: float = 20.0
    search_language: str = "auto"
    #: THE CONNECT DIALOG's routes (vllm_lb/connect.py): the LB's address on its host for an SSH forward
    #: (`LB_CONNECT_TUNNEL`, "" = not offered) and whether the edge in front takes HTTP Basic on the API
    #: (`LB_CONNECT_BASIC`). Neither is guessed: an LB that is not told offers neither.
    connect_tunnel: str = ""
    connect_basic: bool = False
    #: ── hardware off the LAN (each box's fleet_telemetry sender, lb/hw_sender, UDP) ──────
    #: The port every box broadcasts its sample on. 0 turns the listener off entirely.
    hw_port: int = 48291
    #: hostname → box: the fleet file's `boxes.<box>.hostname`. The wire says `<hostname>` (a box's own name);
    #: `avoid_box` says the box. An unmapped sender is IGNORED, never guessed: a wrong box label is worse than a
    #: missing tile.
    hw_hosts: dict = field(default_factory=dict)
    #: Past this, a box's live numbers go blank (its history stays). 6 missed 5 s beats.
    hw_max_age_s: float = 30.0
    #: Samples kept per box per metric — 60 × 5 s ≈ 5 minutes of dots.
    hw_history: int = 60
    #: ── the saturation panel's ceilings: the SERVING CARDS' hardware types (vllm_lb/hardware/types), named by
    #: the fleet file — per box, never one LB-wide claim (operator, 2026-10-11; LB_GPU_PEAK_* are RETIRED_KEYS).
    #: Samples kept per BACKEND per metric, appended once per probe (≈5 s).
    stat_history: int = 60
    #: SAVED BENCHMARK RUNS (bench.py), read by the page at /benchmarks. On the state volume for the
    #: same reason the switches are: it survives an image rebuild and a redeploy, and the harness on
    #: the host writes into it directly (lb/state/bench). A result on a git branch could not be
    #: served at all. The LB only ever READS this directory.
    bench_dir: str = "/data/bench"
    #: THE SWARM CONTROL (benchswarm.py): where the gateway leaves a REQUEST for ONE deep-research
    #: session for `lb/bench_queue_runner.py` on the host, and where that runner reports back. The
    #: LB's OWN state volume and nothing else — since 2026-10-01 the gateway is not mounted into the
    #: harness's directory at all, because the thirty-run matrix is not reachable from the page.
    #: Empty = the control is off and the panel never renders.
    swarm_state: str = "/data/benchq"
    #: THE REAL-TRAFFIC PERFORMANCE LOG (perflog.py): one JSON line per minute in which the cluster
    #: served real requests, written by the LB itself off the probe it already takes. On the state
    #: volume (lb/state/perf) so it survives a rebuild and a redeploy. `perf_enabled` False (LB_PERF=0)
    #: records nothing; `perf_max_mb` is the directory's budget — past it the oldest days go.
    perf_dir: str = "/data/perf"
    perf_enabled: bool = True
    perf_window_s: float = 60.0
    perf_max_mb: float = 256.0
    #: ── the clusters (worklog/parallel-clusters.md; the one-cluster plane: worklog/vllm-cluster-control-plane.md,
    #: worklog/standalone-cluster-panel.md) ────────────────────────────────────────────────────────────────────
    #: EVERY cluster the fleet names, in its order, each keyed by its NAME. The singular `cluster_*` fields below are
    #: the ONE-cluster derivation of this (`__post_init__`: filled from a one-entry tuple, or filling it from a
    #: Config built with them) and EMPTY when several are named — nothing may treat "the cluster" as configured when
    #: two exist; each is reached through `for_cluster(name)`.
    clusters: tuple[ClusterConfig, ...] = ()
    #: The fleet name of THE one cluster this Config speaks of ("" = none, or several: see `clusters`).
    cluster_name: str = ""
    #: `label=box=user@host`, HEAD FIRST (rank 0). Empty = the feature is off; nothing changes. (Derived: `clusters`.)
    cluster_hosts: tuple[ClusterHost, ...] = ()
    #: The head rank's vLLM API — its own port, so it can never be mistaken for a classic sidecar.
    cluster_url: str = ""
    #: HOW THIS LB REACHES THE CLUSTER (vllm_lb/topology.py): `ssh` — tp-rank over SSH drives the TP=2 pair, the only
    #: form LB_CLUSTER_HOSTS can say — or `none` — one node this LB only observes over its HTTP API (a standalone
    #: server's compose `node`). Declared by the deployment's fleet file, never by an environment variable.
    cluster_control: str = topology.SSH
    #: The NIC the ranks talk over (the fleet file's `iface`); "" = none named, and tp-rank refuses the launch naming it.
    cluster_iface: str = ""
    #: The cluster's prefill budget (the fleet file's `clusters.<name>.max_num_batched_tokens`): what a launch uses when
    #: neither the operator's knob nor the model's own says; None = none named, and the executor's env refuses to
    #: build naming it (`cluster_run.tp_rank_env`). Derived from `clusters` like the other singular fields.
    cluster_max_num_batched_tokens: int | None = None
    #: The switch, on disk (a volume): a restarted LB re-reads what the operator last chose.
    cluster_state_path: str = "/data/cluster.json"
    #: The executor's key — accepted on each GB10 box ONLY as the forced `tp-rank` command.
    cluster_ssh_key: str = "/run/cluster-ssh/id_ed25519"
    cluster_known_hosts: str = "/run/cluster-ssh/known_hosts"
    #: broker_parrot's own danger floor: under this a box SHEDS its LLM servers ("[llm-memory] box-a-gpu:
    #: SHEDDING the LLM servers — 10.5 GB available — under the 12 GB danger floor"). A MemoryEstimate that
    #: would leave a box under it is flagged (M8). The fleet's policy, never a model's fact.
    cluster_floor_gib: float = 12.0
    #: How close above that floor still deserves a warning.
    cluster_margin_gib: float = 8.0
    #: WHAT A START MUST LEAVE FREE, beyond what the selection costs — the one number that turns the
    #: estimate into a refusal. Measured boundary on these GB10 boxes: the configuration that has been
    #: serving for weeks (seqs 8, batched 8192) leaves 9.1 GiB and is fine; the one that ran at batched
    #: 16384 left 0.8-4.2 GiB and the kernel log filled with NVRM `Out of memory [NV_ERR_NO_MEMORY]`
    #: until both boxes hung (2026-09-30, lb/tests/fixtures/cluster-b16384-2026-09-30-memory.json). 8.0
    #: sits on the safe side of that boundary and still leaves the known-good configuration 4.5 GiB of
    #: slack. It is NOT `cluster_floor_gib`: that floor is 12.0 and the known-good configuration is
    #: already under it, so gating on the floor would refuse the pair that works. CPU and GPU share one
    #: pool here, so this headroom is the CUDA allocator's room, not spare host RAM.
    cluster_start_headroom_gib: float = 8.0
    #: WHAT LOAD TAKES, at least — the room an `auto` KV pool leaves for it, so that broker_parrot's floor
    #: (`cluster_floor_gib`) still holds UNDER LOAD, not merely at rest. The pool is allocated at start; the
    #: activations of full prefill chunks and the per-request state are not, and the caching allocator keeps
    #: what they grew. Measured (lb/tests/fixtures/cluster-load-2026-10-01-memory.json), the same pair at rest
    #: and then under 16 running sequences: generation 99 (budget 4,096) 19.4 → 13.5 GiB on box-a and 21.8 →
    #: 16.1 on box-b — 5.9 / 5.7 GiB; generations 96/97 (budget 8,192) 9.5-10.2 GiB over hours. The larger
    #: budget is the model's activation slope (`MemoryEstimate.load_gib`), which auto charges when it is the
    #: bigger; this is the floor under it, the operator's measured 6 GiB, above every figure at 4,096.
    cluster_kv_auto_load_gib: float = 6.0
    #: HYSTERESIS. How many consecutive observations must clear the requirement before a rank launches.
    #: While a cluster hold and a chat client fight over ollama, MemAvailable swings by the size of the
    #: model (17.6 GiB) on a 5 s unload/reload cycle, and on 2026-09-16 ONE lucky sample launched 99 GB
    #: of weights onto a box that was really at 77 GiB (worklog/vllm-cluster-hold-robustness.md §3). At
    #: `cluster_observe_s = 5`, three samples is 15 s of "the box really is free".
    cluster_mem_samples: int = 3
    #: How long `freeing` may wait before it FAILS, naming what still holds the RAM. The budget is the
    #: cold start it is protecting (~13 min): waiting longer than that costs more than failing does.
    #: `waiting` has had such a bound since the beginning (REQUEUE_GRACE_S); `freeing` had none.
    cluster_freeing_deadline_s: float = 600.0
    #: How long a restart lets the turns on the pair it replaces finish (cluster.FINISHING_LIMIT_S).
    cluster_finishing_limit_s: float = 900.0
    #: How long a start may say nothing new before it fails, and the free memory under which a rank is
    #: stopped before its box dies (vllm_lb/cluster.py START_STALL_S, KILL_FLOOR_GIB).
    cluster_start_stall_s: float = 600.0
    cluster_kill_floor_gib: float = 0.5
    #: How many times a HUNG start is torn down and relaunched — its evidence saved first — before it is failed
    #: and halted (vllm_lb/cluster.py START_RETRIES). 0 = fail it at once, as before.
    cluster_start_retries: int = 2
    #: The KV cache each rank pins, and what one token of it costs — both MEASURED, for the SHIPPED model
    #: (Flash-Next: worklog/vllm-cluster-ram-headroom.md §4, §5.1). They stand in wherever no ModelSpec says its own: the advice for a
    #: chosen model — `cluster.recommended`, the tight-pool note — is worked out from THAT spec's pin and
    #: `kv_bytes_per_token` (`cluster_defaults.pool_of`), never from these two.
    cluster_kv_cache_memory: int = CLUSTER_KV_CACHE_MEMORY
    cluster_kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_PER_RANK
    #: SSH status reads: this often while anything is in motion, this often at rest.
    cluster_observe_s: float = 5.0
    cluster_idle_observe_s: float = 60.0
    #: How often the head rank's log is read for its loader progress while starting.
    cluster_logs_every_s: float = 15.0
    #: `inputs.max_hold_s` of each GB10 box's reservation: how long it outlives an LB that died holding it. The
    #: operator's Stop is its normal end — the holding node's own 6 h default would end a pair still serving.
    cluster_reservation_max_hold_s: float = 7 * 24 * 3600.0

    def __post_init__(self) -> None:
        self._reconcile_clusters()
        # A topology this LB cannot serve is refused where it is built — at boot — in words naming the rule
        # (topology.problem), never carried into a loop that would half-serve it. Several clusters: each one's.
        for cfg in ([self] if len(self.clusters) <= 1 else [self._narrowed(c) for c in self.clusters]):
            why = topology.problem(cfg)
            if why:
                raise ValueError(why)

    def _reconcile_clusters(self) -> None:
        """ONE definition of the cluster(s): `clusters` is the truth; the singular fields are its one-entry case. A
        Config built with the singular fields (every test that ever did) names one cluster; one built from a fleet of
        one fills the singular fields from it; a fleet of several leaves them empty and refuses any that disagree."""
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 — frozen dataclass
        fleet_names = list(self.fleet.clusters) if self.fleet is not None else []
        if not self.clusters:
            if self.cluster_hosts or self.cluster_url:
                name = self.cluster_name or (fleet_names[0] if len(fleet_names) == 1 else None)
                if not name:
                    # A cluster is keyed by its NAME everywhere (state, view, route): none is made up for it.
                    raise ValueError("cluster_hosts / cluster_url name a cluster but no cluster_name: set cluster_name "
                                     "(the fleet file's clusters.<name>)")
                # The one cluster of a Config is its default (a fleet of one needs no flag).
                set_("clusters", (ClusterConfig(name, tuple(self.cluster_hosts), self.cluster_url, self.cluster_control,
                                                default=True, iface=self.cluster_iface or None,
                                                max_num_batched_tokens=self.cluster_max_num_batched_tokens),))
                set_("cluster_name", name)
            elif not self.cluster_name and len(fleet_names) == 1:
                set_("cluster_name", fleet_names[0])     # a fleet of one, no cluster fields: the hardware still asks it
            return
        if len(self.clusters) == 1:
            (c,) = self.clusters
            singular = (tuple(self.cluster_hosts), self.cluster_url, self.cluster_name)
            if any(singular) and singular != (tuple(c.hosts), c.url, c.name) and singular[:2] != ((), ""):
                raise ValueError(f"cluster {c.name!r} is named by `clusters` and differently by the cluster_* fields: "
                                 f"the singular fields are derived, set `clusters`")
            set_("cluster_hosts", tuple(c.hosts)); set_("cluster_url", c.url)
            set_("cluster_control", c.control); set_("cluster_name", c.name)
            set_("cluster_max_num_batched_tokens", c.max_num_batched_tokens)
            return
        if self.cluster_hosts or self.cluster_url or self.cluster_name:
            raise ValueError(f"{len(self.clusters)} clusters ({', '.join(c.name for c in self.clusters)}): the "
                             f"cluster_* fields name none of them — each is reached through for_cluster(name)")

    @property
    def cluster_reserves(self) -> bool:
        """Does this Config's cluster hold its boxes through broker_parrot (the broker on)? Its fleet cluster's
        `reserves` — the control's rule unless the fleet says `reserve: false`; with no fleet cluster, the control's."""
        c = self.fleet.clusters.get(self.cluster_name) if self.fleet is not None else None
        return c.reserves if c is not None else CONTROLS[self.cluster_control].reserves

    def _narrowed(self, c: ClusterConfig) -> "Config":
        return replace(self, clusters=(c,), cluster_hosts=tuple(c.hosts), cluster_url=c.url, cluster_control=c.control,
                       cluster_name=c.name)

    @property
    def default_cluster(self) -> str | None:
        """THE DEFAULT CHAT CLUSTER's name (fleet `default`; the one chat cluster of a fleet): where a chat request naming
        no model, or a loose name no cluster serves, is routed — alone (router.serving). None: no chat cluster."""
        return self.default_cluster_for("chat")

    def default_cluster_for(self, task: str) -> str | None:
        """THE DEFAULT CLUSTER of `task` (fleet `default`, one per task; a task's one cluster)."""
        return next((c.name for c in self.clusters if c.default and c.task == task), None)

    def for_cluster(self, name: str) -> "Config":
        """THE one-cluster Config for cluster `name` — what every module built for one cluster (the loop, the hub view,
        the switch, the applier's intent) takes. A fleet of one gives this very Config: its state stays where it always
        was (LB_CLUSTER_STATE — live continuity). Several give a copy whose state is `node_control.state_path_for`'s
        `<dir>/clusters/<name>/<file>`; the legacy file beside a missing per-cluster one is REFUSED naming the move
        (unread, the pair's switch would default to `classic` and stop it). ValueError for a name the fleet lacks."""
        from vllm_lb.node_control import state_path_for
        names = [c.name for c in self.clusters]
        if name not in names:
            raise ValueError(f"no cluster {name!r}: the clusters are {', '.join(names) or 'none'}")
        if len(names) == 1:
            return self
        path = state_path_for(self.cluster_state_path, name, names)
        if os.path.exists(self.cluster_state_path) and not os.path.exists(path):
            raise ValueError(f"{self.cluster_state_path} is the switch of ONE cluster, and the fleet names "
                             f"{len(names)} ({', '.join(names)}): move it to the cluster it belongs to — "
                             f"mkdir -p {os.path.dirname(path)} && mv {self.cluster_state_path} {path} (and the "
                             f"node/ dir beside it, for a node the LB requests) — never read as a fresh default")
        cluster = next(c for c in self.clusters if c.name == name)
        return replace(self._narrowed(cluster), cluster_state_path=path)

    def memory_policy(self, mem_total_gib: float | None) -> dict:
        """What a box of THIS cluster keeps free, as `estimate_memory`'s floor / margin / start headroom. A cluster whose
        fleet states `ram_usable_fraction` keeps ONE reserve, MemTotal x (1 - it), for all three — the figure tp-rank's
        guard kills under too (operator, 2026-10-26: "0.98 of ram can be used"); its box's MemTotal unobserved is a
        ValueError, never a guess. Any other cluster: the LB_CLUSTER_* figures."""
        fraction = self.clusters[0].ram_usable_fraction if len(self.clusters) == 1 else None
        if fraction is None:
            return {"floor_gib": self.cluster_floor_gib, "margin_gib": self.cluster_margin_gib,
                    "start_headroom_gib": self.cluster_start_headroom_gib}
        if mem_total_gib is None:
            raise ValueError("the box's MemTotal is not observed yet: ram_usable_fraction has nothing to be a share of")
        reserve = round(mem_total_gib * (1 - fraction), 2)
        return {"floor_gib": reserve, "margin_gib": 0.0, "start_headroom_gib": reserve}

    @property
    def cluster_configured(self) -> bool:
        """The pair (two SSH hosts and a URL) or one one-host node (control local or none: one host, no SSH target, a
        URL) — or the cluster does not exist for this gateway. An incomplete pair is "not configured", as it always
        was."""
        if self.cluster_control in topology.ONE_HOST:
            return len(self.cluster_hosts) == 1 and bool(self.cluster_url)
        return len(self.cluster_hosts) == 2 and bool(self.cluster_url)

    @property
    def cluster_tp(self) -> int:
        """The tensor-parallel size: one rank per host."""
        return len(self.cluster_hosts)

    @property
    def lease_model_id(self) -> str:
        """WHICH model the lease declares. `LB_LEASE_MODEL`, else the model a classic vLLM launches — the catalog's
        classic default's served name (the lease is a classic vLLM's).

        One place owns the rule, so a `Config` built directly (the tests) and one built from
        the environment cannot disagree — and `lease_model` left blank can never mean "no
        model" by accident."""
        if self.lease_model:
            return self.lease_model
        from vllm_lb.catalog import CLASSIC, library_default
        default = library_default(CLASSIC)
        if default is None:
            raise ValueError("no LB_LEASE_MODEL and the catalog has no classic default: a lease would declare no model")
        return default["served_name"]

    @property
    def lease_context_length(self) -> int:
        """THE WINDOW A LEASE DECLARES (broker_parrot migration 0028's `workflow_node_jobs.context_length`): the lease
        model's catalog context — `limits.max_model_len` of the row whose served name `lease_model_id` is. The row owns
        its context; the LB names no window of its own (LB_CONTEXT_LENGTH is retired). A lease model no row serves has
        no window: ValueError naming it, never an unsized lease the 0028 gate would admit on any box."""
        from vllm_lb.catalog import library_models
        model = self.lease_model_id
        # The model's identity and window are the LIBRARY's (library/models.json), staged in this deployment or not.
        try:
            rows = library_models()
        except (OSError, ValueError) as exc:
            raise ValueError(f"the lease model {model!r} has no window: the model library could not be read: "
                             f"{type(exc).__name__}: {exc}") from None
        row = next((m for m in rows if m.get("served_name") == model), None)
        if row is None:
            raise ValueError(f"the lease model {model!r} has no window: no catalog row serves that name (the window "
                             f"is the row's limits.max_model_len; LB_LEASE_MODEL must name a row's served_name)")
        return int(row["limits"]["max_model_len"])

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        e = os.environ if env is None else env
        stale = [k for k in RETIRED_KEYS if k in e]
        if stale:
            raise ValueError("; ".join(
                f"{k} is retired: {RETIRED_KEYS[k]} — remove {k} from the environment (lb/.env)"
                if isinstance(RETIRED_KEYS[k], Gone) else
                f"{k} is retired: its value lives in {RETIRED_KEYS[k]} — remove {k} from the environment (lb/.env)"
                if isinstance(RETIRED_KEYS[k], InCatalog) else
                f"{k} is retired: its value lives in {RETIRED_KEYS[k]} (LB_FLEET names that file) — remove {k} from "
                f"the environment (lb/.env)" for k in stale))
        fleet = load_fleet(e["LB_FLEET"]) if e.get("LB_FLEET", "").strip() else None
        fleet_fields = topology_of(fleet)
        host, _, port = e.get("LB_LISTEN", "0.0.0.0:8010").rpartition(":")

        def f(name: str, default: float) -> float:
            return float(e.get(name, default))

        def secs(default: float, *names: str) -> float:
            """A duration in seconds, from the FIRST name that is set to something non-blank.

            Blank means unset, so blanking a superseded line (the migration an operator
            actually performs when renaming a var) is not a boot-time crash. Non-positive is
            refused HERE rather than at request time: `0` reads as "disable this", but
            `socket.settimeout(0)` means NON-BLOCKING and would make every read fail at once,
            and `-1` raises inside the proxy long after the operator has walked away.
            """
            for n in names:
                raw = e.get(n)
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    raise ValueError(f"{n}={raw!r} is not a number of seconds") from None
                if val <= 0:
                    raise ValueError(
                        f"{n}={raw!r} must be a positive number of seconds; there is no "
                        f"'disable' value — write a large number if you mean 'effectively "
                        f"never' (0 would make every read non-blocking)")
                return val
            return default

        cfg = cls(
            listen_host=host or "0.0.0.0",
            listen_port=int(port or 8010),
            allowed_hosts=parse_allowed("LB_ALLOWED_HOSTS", e.get("LB_ALLOWED_HOSTS")),
            vllm_boxes=fleet_fields["vllm_boxes"],
            fleet=fleet,
            max_leases=int(e.get("LB_MAX_LEASES", cls.max_leases)),
            # LB_VLLM_MAX_WAITING is the old name; it set the same knob when it
            # still meant waiting-only, so it keeps working.
            vllm_max_depth=int(e.get("LB_VLLM_MAX_DEPTH",
                                     e.get("LB_VLLM_MAX_WAITING", cls.vllm_max_depth))),
            idle_release_s=f("LB_IDLE_RELEASE_S", cls.idle_release_s),
            scale_up_waiting=int(e.get("LB_SCALE_UP_WAITING", cls.scale_up_waiting)),
            scale_up_sustain_s=f("LB_SCALE_UP_SUSTAIN_S", cls.scale_up_sustain_s),
            probe_interval_s=f("LB_PROBE_INTERVAL_S", cls.probe_interval_s),
            probe_timeout_s=f("LB_PROBE_TIMEOUT_S", cls.probe_timeout_s),
            tick_s=f("LB_TICK_S", cls.tick_s),
            # LB_BACKEND_TIMEOUT_S is the old name for the first-byte budget; still honoured,
            # and only READ when the new name is absent (a nested f() parsed it eagerly, so a
            # blanked-out old line crashed the boot it was meant to survive).
            first_byte_timeout_s=secs(cls.first_byte_timeout_s,
                                      "LB_FIRST_BYTE_TIMEOUT_S", "LB_BACKEND_TIMEOUT_S"),
            stall_timeout_s=secs(cls.stall_timeout_s, "LB_STALL_TIMEOUT_S"),
            sse_keepalive_s=f("LB_SSE_KEEPALIVE_S", cls.sse_keepalive_s),
            drain_s=f("LB_DRAIN_S", cls.drain_s),
            drain_retry_after_s=f("LB_DRAIN_RETRY_AFTER_S", cls.drain_retry_after_s),
            cluster_hold_s=f("LB_CLUSTER_HOLD_S", cls.cluster_hold_s),
            hold_max=int(e.get("LB_HOLD_MAX", cls.hold_max)),
            retry_after_s=f("LB_RETRY_AFTER_S", cls.retry_after_s),
            starve_after_s=secs(cls.starve_after_s, "LB_STARVE_AFTER_S"),
            admission_wait_s=f("LB_ADMISSION_WAIT_S", cls.admission_wait_s),
            connect_timeout_s=secs(cls.connect_timeout_s, "LB_CONNECT_TIMEOUT_S"),
            penalty_s=f("LB_PENALTY_S", cls.penalty_s),
            broker_enabled=e.get("LB_BROKER_ENABLED", "").strip().lower() in ("1", "true", "yes"),
            db_url_env=e.get("LB_DB_URL_ENV", cls.db_url_env),
            lease_workflow=e.get("LB_LEASE_WORKFLOW", cls.lease_workflow),
            lease_node_module=e.get("LB_LEASE_NODE_MODULE", cls.lease_node_module),
            lease_model=e.get("LB_LEASE_MODEL", cls.lease_model),
            lease_max_hold_s=f("LB_LEASE_MAX_HOLD_S", cls.lease_max_hold_s),
            lease_backoff_max_s=f("LB_LEASE_BACKOFF_MAX_S", cls.lease_backoff_max_s),
            # The CLASS default, not a hardcoded "1": the shipped default is False (no lease,
            # no traffic) and hardcoding the string here silently overrode it in every
            # deployment that had not set the variable.
            use_unleased_vllm=(e["LB_USE_UNLEASED_VLLM"] not in ("0", "false", "no", "")
                               if "LB_USE_UNLEASED_VLLM" in e else cls.use_unleased_vllm),
            search_url=e.get("LB_SEARCH_URL", cls.search_url).strip().rstrip("/"),
            search_max_results=int(e.get("LB_SEARCH_MAX_RESULTS", cls.search_max_results)),
            search_snippet_chars=int(e.get("LB_SEARCH_SNIPPET_CHARS", cls.search_snippet_chars)),
            search_timeout_s=f("LB_SEARCH_TIMEOUT_S", cls.search_timeout_s),
            search_language=e.get("LB_SEARCH_LANGUAGE", cls.search_language),
            connect_tunnel=parse_tunnel(e.get("LB_CONNECT_TUNNEL")),
            connect_basic=parse_flag("LB_CONNECT_BASIC", e.get("LB_CONNECT_BASIC")),
            hw_port=int(e.get("LB_HW_PORT", cls.hw_port)),
            hw_hosts=fleet_fields["hw_hosts"],
            hw_max_age_s=secs(cls.hw_max_age_s, "LB_HW_MAX_AGE_S"),
            hw_history=int(e.get("LB_HW_HISTORY", cls.hw_history)),
            stat_history=int(e.get("LB_STAT_HISTORY", cls.stat_history)),
            bench_dir=e.get("LB_BENCH_DIR", cls.bench_dir),
            swarm_state=e.get("LB_SWARM_STATE", cls.swarm_state).strip(),
            perf_dir=e.get("LB_PERF_DIR", cls.perf_dir),
            perf_enabled=e.get("LB_PERF", "1").strip().lower() not in ("0", "false", "no", "off"),
            perf_window_s=f("LB_PERF_WINDOW_S", cls.perf_window_s),
            perf_max_mb=f("LB_PERF_MAX_MB", cls.perf_max_mb),
            clusters=fleet_fields["clusters"],
            cluster_state_path=e.get("LB_CLUSTER_STATE", cls.cluster_state_path),
            cluster_ssh_key=e.get("LB_CLUSTER_SSH_KEY", cls.cluster_ssh_key),
            cluster_known_hosts=e.get("LB_CLUSTER_KNOWN_HOSTS", cls.cluster_known_hosts),
            cluster_floor_gib=f("LB_CLUSTER_FLOOR_GIB", cls.cluster_floor_gib),
            cluster_margin_gib=f("LB_CLUSTER_MARGIN_GIB", cls.cluster_margin_gib),
            cluster_start_headroom_gib=f("LB_CLUSTER_START_HEADROOM_GIB", cls.cluster_start_headroom_gib),
            cluster_kv_auto_load_gib=f("LB_CLUSTER_KV_AUTO_LOAD_GIB", cls.cluster_kv_auto_load_gib),
            cluster_mem_samples=int(e.get("LB_CLUSTER_MEM_SAMPLES", cls.cluster_mem_samples)),
            cluster_freeing_deadline_s=f("LB_CLUSTER_FREEING_DEADLINE_S", cls.cluster_freeing_deadline_s),
            cluster_finishing_limit_s=f("LB_CLUSTER_FINISHING_S", cls.cluster_finishing_limit_s),
            cluster_start_stall_s=f("LB_CLUSTER_START_STALL_S", cls.cluster_start_stall_s),
            cluster_kill_floor_gib=f("LB_CLUSTER_KILL_FLOOR_GIB", cls.cluster_kill_floor_gib),
            cluster_start_retries=int(e.get("LB_CLUSTER_START_RETRIES", cls.cluster_start_retries)),
            cluster_kv_cache_memory=int(e.get("LB_CLUSTER_KV_CACHE_MEMORY",
                                              cls.cluster_kv_cache_memory)),
            cluster_kv_bytes_per_token=int(e.get("LB_CLUSTER_KV_BYTES_PER_TOKEN",
                                                 cls.cluster_kv_bytes_per_token)),
            cluster_observe_s=f("LB_CLUSTER_OBSERVE_S", cls.cluster_observe_s),
            cluster_idle_observe_s=f("LB_CLUSTER_IDLE_OBSERVE_S", cls.cluster_idle_observe_s),
            cluster_logs_every_s=f("LB_CLUSTER_LOGS_EVERY_S", cls.cluster_logs_every_s),
            cluster_reservation_max_hold_s=secs(cls.cluster_reservation_max_hold_s,
                                                "LB_CLUSTER_RESERVATION_MAX_HOLD_S"),
        )
        return cfg
