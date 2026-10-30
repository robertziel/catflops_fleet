"""THE CLUSTER'S TOPOLOGY: how many hosts serve it, how this LB reaches them, and what that lets it do
(worklog/standalone-cluster-panel.md).

Three shapes exist, and every rule here keys off which one is configured — never off a box's name or its hardware:

  ssh   (2 hosts)  the TP=2 pair. tp-rank over SSH observes, launches and stops each rank (cluster_exec.py), and the
                   planner (`cluster.plan`) frees, reserves, launches and rescues. `Config.cluster_control`'s default,
                   so a deployment configured before the field existed is exactly what it was.
  local (1 host)   a standalone server's node the LB REQUESTS and its host APPLIES (vllm_lb/node_control.py): the
                   panel's Start, Stop and settings become an intent file in the LB's state dir, released once the node
                   has finished what it serves (`cluster.plan_local`), which a systemd applier on the host validates
                   with the fleet's own code and runs `up.sh node` with. No socket, no key, no shell on the host.
  none  (1 host)   a standalone server's node (deploy/standalone): its OWN compose service starts, stops and configures
                   it from its server file. This LB reads the node's HTTP API — the probe every backend gets — and
                   nothing else, so the planner for it (`cluster.plan_observed`) has no actions: `running` while the
                   node answers, `unknown` while it does not. Start, Stop and the settings are refused here, saying
                   why (`OBSERVED_WHY`), never accepted and dropped.

WHERE IT IS DECLARED is the deployment's fleet file (AGENTS.md, "Open-source ready"), whose loader fills
`Config.cluster_hosts`, `cluster_url` and `cluster_control`; no environment variable names a control channel. The
environment's one cluster form, LB_CLUSTER_HOSTS (retiring into that file), can only say the SSH pair.

Both publish ONE view shape (`/lb/status.cluster`): the page draws the same card, the same load figures and the same
strip for either, from `topology` — the TP, the boxes, the control channel and the start steps that apply to it.

WHICH CARD a node runs on, and WHAT that card is, are the fleet file's (vllm_lb/fleet.py: each box's cards, the
`serves: true` ones under the cluster's ranks) and the hardware domain's (vllm_lb.hardware: the card's type). This
module asks through `serving_cards` / `serving_card` / `serving_profile` alone, so the answer changes in one place.
Where the fleet names no serving card there is no reading — never the box's busiest card, which on a shared box can be
another tenant's.
"""
from __future__ import annotations

from dataclasses import asdict

from vllm_lb.cluster import RankObs, limits_view

SSH, LOCAL, NONE = "ssh", "local", "none"
CONTROLS = (SSH, LOCAL, NONE)
#: The one-host shapes: a node on one box, read over HTTP, with no SSH channel.
ONE_HOST = (LOCAL, NONE)

#: Every step a start can pass through, in the planner's order (`cluster.plan`). Which of them APPLY is the
#: topology's (`start_steps`): the page draws only those, so a step that cannot happen is never shown done.
START_STEPS = ("free", "reserve", "drain", "load", "warm up", "serve")

#: What the page and a refused switch say about a node this LB only observes — the one line for both. Data, not prose
#: (operator, 2026-10-01): what it is, and who does what this LB cannot.
OBSERVED_WHY = "observed only — its server file starts, stops and sets it"

#: The NETWORK ceiling's reason at TP=1: the tile measures the link between ranks, and one rank has none.
NO_LINK = "TP=1: one GPU, no link between ranks"


def problem(cfg) -> str:
    """Why this configuration names a cluster topology the LB cannot serve, in the words a boot refusal prints — or
    "" when it can. Strict for the one-host shapes (everything they need is named, nothing is ignored); for `ssh` only
    a host with no SSH target is refused, and a pair that is incomplete stays "not configured", as before."""
    control, hosts, url = cfg.cluster_control, tuple(cfg.cluster_hosts), cfg.cluster_url
    if control not in CONTROLS:
        return (f"cluster control {control!r}: the control channels are {SSH} (tp-rank over SSH, the TP=2 pair), "
                f"{LOCAL} (one node this LB requests and its host applies) and {NONE} (one node it only observes)")
    if control == SSH:
        bare = next((h for h in hosts if not h.ssh_target), None)
        if bare is not None:
            return (f"cluster host {bare.label}={bare.box} has no SSH target: control {SSH} reaches every box over "
                    f"SSH (label=box=user@control=interconnect); a one-host node is control {LOCAL} or {NONE}")
        return ""
    if not hosts and not url:
        return ""                                    # nothing named: there is no cluster
    if len(hosts) != 1:
        return f"control {control} runs ONE node: the cluster names {len(hosts)} hosts"
    host = hosts[0]
    if host.ssh_target or host.interconnect:
        return (f"cluster host {host.label}={host.box} has an SSH target nothing would use: control {control} reads "
                f"the node over HTTP")
    if not url:
        return f"no URL for the node's API, which this LB probes (control {control})"
    return ""


def start_steps(control: str, broker: bool, reserves: bool | None = None) -> tuple[str, ...]:
    """The steps of `START_STEPS` a start of this topology really passes through — read off its control's DATA
    (`fleet.Control`): `free` where its start frees the boxes' RAM (`frees`), `reserve` and `drain` where it holds its
    boxes through broker_parrot (`reserves`) and the broker is on — without it nothing is reserved and no fleet job
    drains — then load, warm up, serve. `reserves`: the cluster's own word (`Config.cluster_reserves`); None = its
    control's."""
    from vllm_lb.fleet import CONTROLS
    c = CONTROLS[control]
    reserves = c.reserves if reserves is None else reserves
    keep = {"load", "warm up", "serve"} | ({"free"} if c.frees else set()) | (
        {"reserve", "drain"} if reserves and broker else set())
    return tuple(s for s in START_STEPS if s in keep)


def view(cfg) -> dict:
    """`cluster.topology` in /lb/status: what the page names the cluster by, and what it may offer."""
    return {"tp": len(cfg.cluster_hosts), "boxes": [h.box for h in cfg.cluster_hosts],
            "control": cfg.cluster_control, "why": OBSERVED_WHY if cfg.cluster_control == NONE else "",
            "steps": list(start_steps(cfg.cluster_control, cfg.broker_enabled, cfg.cluster_reserves))}


# ── the hardware domain's answers, asked in ONE place ───────────────────────────────────────────


def named(cfg) -> str:
    """The cluster in a sentence, from what this LB was configured with: `cluster <its fleet name> (TP=<n> on <box>,
    <box>)`, or `the cluster (TP=<n> on …)` where no fleet names it. Never a topology or a box written in the code —
    "the two-box cluster" was said of a one-GPU node (site-b, 2026-10-11)."""
    cluster = fleet_cluster(cfg)
    boxes = [h.box for h in cfg.cluster_hosts]
    tp = cluster.tp if cluster is not None else len(boxes)
    on = f" on {', '.join(boxes)}" if boxes else ""
    return f"cluster {cluster.name} (TP={tp}{on})" if cluster is not None else f"the cluster (TP={tp}{on})"


def fleet_cluster(cfg):
    """The fleet file's cluster THIS config speaks of (vllm_lb/fleet.py `Cluster`, by `cfg.cluster_name` — a one-cluster
    Config: `Config.for_cluster`), or None where no fleet names it. What the hardware and the declaration below ask of
    the fleet, they ask through here."""
    fleet = getattr(cfg, "fleet", None)
    return fleet.clusters.get(getattr(cfg, "cluster_name", "")) if fleet is not None else None


def serving_cards(cfg) -> tuple:
    """The cards the cluster serves on, HEAD FIRST: the fleet file's `serves: true` cards of its ranks' boxes
    (vllm_lb/fleet.py `Cluster.serving_cards`), each with its hardware profile. () where no fleet names them — then
    no card is claimed, never a guessed one."""
    cluster = fleet_cluster(cfg)
    return cluster.serving_cards if cluster is not None else ()


def serving_card(cfg, hardware: dict, box: str) -> dict | None:
    """The card the node on `box` RUNS ON, as the hardware broadcast read it (`/lb/status.hardware[box].gpus[…]`:
    `{id, use_pct, vram_used_mb, vram_total_mb}`) — the entry whose id is the fleet's serving card's index — or None:
    no fleet card serves there, more than one does, or the broadcast carries no reading for it. Never the box's
    busiest card, which on a shared box can be another tenant's."""
    cards = [c for c in serving_cards(cfg) if c.box == box]
    gpus = ((hardware or {}).get(box) or {}).get("gpus")
    if len(cards) != 1 or not isinstance(gpus, list):
        return None
    return next((g for g in gpus if isinstance(g, dict) and g.get("id") == cards[0].index), None)


def cluster_model(cfg):
    """What a node this LB only OBSERVES serves, as the fleet file names it (`clusters.<name>.model`: a ClusterModel —
    catalog row and variant) — or None: a pair this LB drives names its model at its Start (`effective`), never here."""
    cluster = fleet_cluster(cfg)
    return cluster.model if cluster is not None else None


def serving_cuda_cores(cfg) -> int | None:
    """ONE GPU's CUDA cores, for the CUDA CORES tile: the figure every serving card's hardware type states — None when
    no card is named, a type states none (an AMD iGPU has compute units, not CUDA cores), or the serving cards differ:
    the tile draws one GPU, and two different GPUs are not one count."""
    values = {c.type.cuda_cores.value for c in serving_cards(cfg)}
    return next(iter(values)) if len(values) == 1 and None not in values else None


def serving_profile(cfg, box: str):
    """The hardware profile (`vllm_lb.hardware.HardwareProfile`) of the card serving on `box` — its peaks are the
    saturation row's ceilings and its cores the CUDA CORES tile's — or None where no single fleet card serves there:
    then no peak is claimed."""
    cards = [c for c in serving_cards(cfg) if c.box == box]
    return cards[0].type if len(cards) == 1 else None


# ── the view a node this LB only observes publishes ─────────────────────────────────────────────


def observed_box(host, *, live: bool, error: str) -> dict:
    """One box of an observed node, in the pair's box shape (cluster_loop `_publish`): what the node's HTTP API says
    — it answers, so its engine runs — and null for everything only an SSH channel or the broker could tell."""
    rank = asdict(RankObs(present=True, running=True, gen=None, exit_code=None)) if live else None
    return {"label": host.label, "reservation": None, "park": None, "running_jobs": None,
            "broker_vllm_running": None, "mem_available_gib": None, "mem_stable_gib": None, "mem_total_gib": None,
            "at_rest": None, "mem_rest_gib": None,
            "mem_at_launch": None, "mem_release_target": None, "rank": rank, "availability": None, "memory": None,
            "holds": None, "gpu": None, "network": None,
            "hold": {"reservation_status": None, "claimed_by": None, "waiting_s": 0.0, "deadline_s": None,
                     "blockers": []},
            "heal": None, "error": "" if live else (error or "")}


def observed_effective(head, throughput: dict) -> dict:
    """WHAT A NODE THIS LB ONLY OBSERVES RUNS WITH — `cluster.effective` in the pair's keys, from its own reports alone
    (`head`, the probed backend; `throughput`, its `throughput` block): the window /v1/models states, the checkpoint
    folder its `root` names, what `vllm:cache_config_info` says (GPU memory fraction, KV cache dtype and pool, prefix
    caching), the engine its exposition is written in, and no drafter when it publishes no spec-decode counter — vLLM
    registers them only with a speculative config. Every setting the node does not publish (its seats, its prefill
    budget, the generation, which catalog model it is) is null: never a guess, never a default."""
    config = throughput.get("engine_config") or {}
    pool = throughput.get("kv_pool") or {}
    root = next(iter(head.model_roots), "")
    return {"model_choice": None, "generation": None, "max_num_seqs": None, "max_num_batched_tokens": None,
            "num_speculative_tokens": 0 if throughput.get("spec_decode") is None else None,
            "ple_replicate": None, "kv_cache_memory": pool.get("bytes"), "kv_offload_gib": None,
            # THE KV TIER IN THE HOST'S RAM, in GiB, as the engine states it (throughput.py `kv_offload`); null where
            # it runs none — the fleet's own word for "off" (fleet.RAM_TIER).
            "kv_offload_ram_gib": (throughput.get("kv_offload") or {}).get("tier_gib"),
            "max_model_len": head.max_model_len, "long_prefill_token_threshold": None, "uncensored": None,
            "served_name": next(iter(head.served_models), None),
            "checkpoint": root.rstrip("/").rsplit("/", 1)[-1] or None,
            **{key: config.get(key) for key in ("engine", "gpu_memory_utilization", "kv_cache_dtype",
                                                "prefix_caching")}}


# ── what the fleet file DECLARES a node this LB only observes runs ─────────────────────────────


#: The `launch:` keys the node ALSO reports, each under the same name in `effective` (/v1/models' id, window and the
#: checkpoint folder its `root` names, vllm:cache_config_info's GPU memory fraction and KV tier in RAM): the declaration and the node are
#: compared on these (`cross_check`). A local node's launch names its checkpoint (the catalog's folder of its model).
CROSS_CHECKED = ("served_name", "max_model_len", "gpu_memory_utilization", "checkpoint", "kv_offload_ram_gib")

#: A check's verdict, in the words the page prints.
AGREES, MISMATCH, NOT_EXPOSED = "agrees", "mismatch", "not exposed by the node"


def declaration(cfg, *, launch=None, source: str | None = None) -> dict | None:
    """`cluster.declared` before the node is asked anything: what a one-host node IS — its catalog row and checkpoint
    (`clusters.<name>.model`, a fleet.ClusterModel the loader validated against the catalog) — and is LAUNCHED WITH:
    the fleet's `launch:` block (a fleet.NodeLaunch, which the server file renders into the node's command line; {}
    where the fleet names none), or — for a node the LB requests — `launch`, that block moved by the last intent its
    host applied, with `source` saying so. None for the pair (its Start names its model, its launch record what it
    runs) and where no fleet declares the node."""
    cluster, model = fleet_cluster(cfg), cluster_model(cfg)
    if cfg.cluster_control not in ONE_HOST or model is None:
        return None
    launch = cluster.launch if launch is None else launch
    model = selected_model(cluster, launch) or model
    # A key the launch leaves to the engine (None: vLLM's own prefill chunk) is not declared — nothing to check it by —
    # nor is a setting its engine does not read (fleet.NodeEngine.settings: faster-whisper has no context window).
    from vllm_lb.fleet import ENGINES, MOVABLE
    reads = {st.key for st in ENGINES[launch.engine].settings(cluster.task)} if launch is not None else set()
    return {"source": source or cfg.fleet.source, "cluster": cluster.name, "model": model.ref,
            "model_choice": model.spec.id, "uncensored": model.variant == "uncensored",
            "launch": {k: v for k, v in asdict(launch).items() if v is not None and (k not in MOVABLE or k in reads)}
            if launch is not None else {}}


def selected_model(cluster, launch):
    """The fleet `ClusterModel` a launch SELECTED among its cluster's `choices` (fleet.SelectedLaunch.model, control
    local) — None for a launch that selects nothing (the observed node's, whose model is the cluster's own)."""
    ref = getattr(launch, "model", None)
    return next((m for m in getattr(cluster, "choices", ()) if m.ref == ref), None) if ref else None


def cross_check(declared: dict, effective: dict, served=()) -> dict:
    """{key: {fleet, node, verdict}} for every key the fleet declares, in its order: `agrees` / `mismatch` where the
    node reports the key too (`CROSS_CHECKED`; a served name agrees when it is ANY of the names the node serves), and
    `not exposed by the node` where it does not — this time or ever. A disagreement keeps both values: neither is
    chosen here, and none is ever assumed to agree."""
    out = {}
    for key, fleet in declared["launch"].items():
        node = effective.get(key) if key in CROSS_CHECKED else None
        if key == "served_name" and fleet in served:
            node = fleet
        verdict = NOT_EXPOSED if node is None else (AGREES if node == fleet else MISMATCH)
        out[key] = {"fleet": fleet, "node": node, "verdict": verdict}
    return out


def declared_effective(effective: dict, declared: dict) -> dict:
    """`effective` with what ONLY the declaration can say: which catalog row the node is (`model_choice`,
    `uncensored`), its seats (`max_num_seqs`) and its prefill budget (`max_num_batched_tokens`; null where the launch
    leaves it to vLLM, which publishes the figure it chose nowhere in /metrics) — the node reports none of them. Every
    key the node does report keeps the node's own word; `cross_check` publishes where the declaration disagrees. For a
    node the LB requests, the declaration is its applied intent's, so the performance log records what its host
    applied (perflog `effective_config`)."""
    return {**effective, "model_choice": declared["model_choice"], "uncensored": declared["uncensored"],
            "max_num_seqs": declared["launch"].get("max_num_seqs"),
            "max_num_batched_tokens": declared["launch"].get("max_num_batched_tokens")}


def observed_view(cfg, *, state: str, detail: str, held: bool, since: float, live: bool = False,
                  error: str = "", up_since: float | None = None, loaded_model: dict | None = None,
                  catalog_error: str = "", effective: dict | None = None, served=(), launch=None,
                  source: str | None = None, generation: int | None = None) -> dict:
    """`/lb/status.cluster` for a one-host node: the SAME keys the pair's loop publishes (tests/test_cluster_one_host.py
    pins them equal), each null where the node has nothing to say — for one this LB only observes, no switch
    (`desired`), no start in progress (`loading`, `rescue`, `freeing`), no applier (`applier`; `local_view` fills
    them for a node the LB requests). `effective` is what the node says it runs (`observed_effective`) and `served`
    the names it serves; the declaration (`declared`: the fleet's, or `launch` from `source`, checked against them)
    adds the catalog row and the seats. `max_depth` — what the router holds the node's depth to, as it holds the
    pair's to its concurrency — is those seats; null where none are declared, and then the router counts the node as
    busy (router.py `busy`): the classic boxes' limit is never borrowed for it.

    `generation`: WHICH START the node runs — for a node the LB requests, the id of the intent its host applied (the
    one `launch` comes from), as the pair's is its Start's — in `effective`, so the performance log stamps every minute
    with it and /lb/sessions folds them into that start's session (perflog.sessions). None — a node the LB only
    observes, or one running the fleet's own launch — names none: never a guess."""
    declared = declaration(cfg, launch=launch, source=source)
    seats = (declared or {}).get("launch", {}).get("max_num_seqs")
    if declared is not None:
        checks = cross_check(declared, effective, served) if effective is not None else None
        effective = declared_effective(effective, declared) if effective is not None else None
        declared = {**declared, "checks": checks}
    if effective is not None:
        effective = {**effective, "generation": generation}
    return {"configured": True, "topology": view(cfg), "desired": None, "state": state, "detail": detail,
            "held": held, "loading": {}, "boxes": {h.box: observed_box(h, live=live, error=error)
                                                   for h in cfg.cluster_hosts},
            "rescue": None, "freeing": None, "url": cfg.cluster_url, "limits": limits_view(), "effective": effective,
            "defaults": None, "setups": {}, "since": float(since), "max_depth": seats,
            "up_since": up_since, "recommended": None, "clamp": "", "store_error": "", "models": [], "offers": None,
            # Nothing to choose a KV dtype for on a node this LB only observes (the pair's: per model, per card).
            "kv_cache_dtypes": {},
            "loaded_model": loaded_model, "catalog_error": catalog_error, "declared": declared, "applier": None}


# ── a node the LB requests and its host applies (control local) ────────────────────────────────


def local_desired(cfg, wish: dict | None) -> dict:
    """`cluster.desired` for a node the LB requests: the operator's wish (vllm_lb/node_control.py, the node's switch)
    — or, never set, the fleet's boot default, on exactly when the fleet runs it at boot (`run_at_boot`) — as the
    launch it asks for: the fleet's launch block moved by the
    wish's figures, as stored. Whether the fleet takes it is `local_fit`'s `refused` (fleet.validate_launch's words),
    shown beside it; the model's row and twin are null for a ref that is no longer among the fleet's choices."""
    from vllm_lb.fleet import MOVABLE
    cluster = fleet_cluster(cfg)
    boot, moved = cluster.launch, (wish["launch"] if wish else {})
    # AS STORED: the boot default moved by the wish's own values — validated or not. A wish the fleet took once and
    # refuses now (a stricter fit) is still what the operator asked, and the panel is where it is changed;
    # `local_fit` says why it is refused. The prefill chunk is null where neither names one: vLLM's own default.
    asked = {k: moved.get(k, getattr(boot, k)) for k in MOVABLE}
    model = next((m for m in cluster.choices if m.ref == asked["model"]), None)
    # NEVER ASKED: the fleet's run_at_boot says whether it is meant to run (a declared `run_at_boot: false` node is off
    # until a Start — never "on", which offered Stop for a node nobody started, live 2026-10-17).
    on = wish["state"] == "up" if wish else cluster.run_at_boot
    # ONLY WHAT ITS ENGINE READS (fleet.NodeEngine.settings): a knob the selected model's engine does not read is no
    # setting of this node — faster-whisper has no context window nor memory fraction to show.
    from vllm_lb.fleet import ENGINES
    engine = ENGINES[model.spec.node_engine if model and model.spec.node_engine else boot.engine]
    reads = {st.key for st in engine.settings(cluster.task)}
    return {"mode": "cluster" if on else "off",
            "id": wish["id"] if wish else None, "force": wish["force"] if wish else False,
            "model": asked["model"], "model_choice": model.spec.id if model else None,
            "uncensored": model.variant == "uncensored" if model else None,
            **{k: asked[k] for k in MOVABLE if k != "model" and k in reads}}


def applied_launch(cluster, overrides: dict):
    """The launch a node's host APPLIED: the fleet's boot default moved by the applied intent's model and figures, AS
    APPLIED — never re-judged by today's fleet (a stricter fit since does not change what runs, nor may the boot
    default stand in for it). The model's checkpoint and served name are its catalog row's. None for a model no
    longer among the fleet's choices: what it loaded is then not known here."""
    from dataclasses import replace
    boot = cluster.launch
    ref = overrides.get("model", getattr(boot, "model", None))
    model = next((m for m in getattr(cluster, "choices", ()) if m.ref == ref), None)
    if model is None or boot is None:
        return None
    from vllm_lb.fleet import MOVABLE
    figures = {k: overrides[k] for k in MOVABLE if k != "model" and k in overrides}
    return replace(boot, model=model.ref, checkpoint=model.checkpoint, served_name=model.spec.served_name, **figures)


def released_config(cfg, intent: dict) -> dict | None:
    """A RELEASED INTENT'S LAUNCH in the performance log's own keys (perflog CONFIG_KEYS, and NODE_CONFIG_KEYS: the
    card's share and the KV tier in RAM) — what a node's boot line records, as the pair's records its launch: the
    fleet's boot default moved by the intent's figures (`applied_launch`), its catalog row and twin (`declaration`), and
    the intent's id as its generation. A figure the launch leaves to vLLM is null; one only the engine states (its
    drafter) is absent. None for a model no longer among the fleet's choices."""
    from dataclasses import asdict
    from vllm_lb.perflog import CONFIG_KEYS, NODE_CONFIG_KEYS
    launch = applied_launch(fleet_cluster(cfg), intent.get("launch") or {})
    declared = declaration(cfg, launch=launch) if launch is not None else None
    if declared is None:
        return None
    figures = {k: v for k, v in asdict(launch).items() if k in CONFIG_KEYS + NODE_CONFIG_KEYS}
    return {**figures, "generation": intent["id"], "model_choice": declared["model_choice"],
            "uncensored": declared["uncensored"]}


#: How a node's start is recorded in the boot log (perflog.BootRecorder): the planner's states, but for the RELEASED
#: intent — `waiting` while its host's applier has not taken it up (the page says `starting` from the release), so a
#: refusal before anything ran is a refusal (start_rates.py: no `starting`, not an attempt), not a failed launch.
WAITING = "waiting"


def boot_state(state: str, released: dict, report: dict) -> str:
    """The step of the released intent's start, from the plan's `state` and the applier's report (node_control
    `read_report`). An up intent: `waiting` until the applier applies it, `starting` while it does, the plan's own state
    once it has — `failed` where it REFUSED it, even while the node still serves the intent before (the plan says
    `running` then: what serves is not what was asked). A down intent starts nothing: the plan's state, which stops
    the start it replaces (`unloading`, `off`). `unknown` stays unknown — nothing observed."""
    from vllm_lb.node_control import UP, same_intent
    if released["state"] != UP or state == "unknown":
        return state
    applied, applying = report.get("applied"), report.get("applying")
    if same_intent(released, applied):
        return "failed" if applied.get("refused") else state
    if same_intent(released, applying):
        return "starting"
    return WAITING if state in ("starting", "running") else state


def local_fit(cfg, overrides: dict) -> dict:
    """WHAT A LAUNCH NEEDS ON THE CARD, in the fleet's own numbers (`fleet.launch_fit`: weights, what the engine holds beside
    them, one request's KV, the share of the card) for the boot default moved by `overrides`, and the fleet's verdict on it in its words
    (`refused`, "" when `fleet.validate_launch` takes it). `fit` is null where the model asked is not a choice or a
    figure is not a number — then `refused` says which. A pinned pool (`pool_gib`) is priced on the whole card, and the
    RAM tier (`ram_gib`) is carried beside the card's sum, never in it."""
    from vllm_lb.fleet import ENGINES, validate_launch
    cluster = fleet_cluster(cfg)
    try:
        validate_launch(cluster, overrides)
        refused = ""
    except ValueError as exc:
        refused = str(exc)
    launch = cluster.launch
    model = next((m for m in cluster.choices if m.ref == overrides.get("model", getattr(launch, "model", None))),
                 None)
    # THE ENGINE'S OWN MEMORY MODEL (fleet.NodeEngine.fit_view): vLLM's fraction of the card, faster-whisper's measure.
    engine = ENGINES[model.spec.node_engine if model is not None and model.spec.node_engine else launch.engine]
    fit = engine.fit_view(model, cluster, overrides) if model is not None else None
    return {"fit": fit, "refused": refused}


def local_choices(cfg) -> list[dict]:
    """`cluster.choices`: the models the panel may SELECT for a node the LB requests — the fleet's `launch.choices`
    (each fit-checked against the serving card at load) as the catalog names them: the ref an intent sends, the row, the
    checkpoint it loads, its weights and its own limits — `overhead_chunk`, the prefill chunk its one-card overhead
    was measured at (null where the catalog names none): the largest chunk the fit can check — and `kv_pin_max`, the
    largest pool the fit takes PINNED on the serving card (fleet.pin_ceiling; null where a term is not stated)."""
    from vllm_lb.fleet import pin_ceiling
    cluster = fleet_cluster(cfg)

    def chunk(m):
        over = m.spec.memory.single_card_overhead if m.spec.memory is not None else None
        return over.chunk_tokens if over is not None else None
    return [{"ref": m.ref, "model_choice": m.spec.id, "uncensored": m.variant == "uncensored", "label": m.spec.label,
             "checkpoint": m.checkpoint, "size_gib": m.spec.size_gib, "limits": m.spec.limits.view(),
             "overhead_chunk": chunk(m), "kv_pin_max": pin_ceiling(m, cluster.serving_cards[0].type),
             # ITS ENGINE (fleet.ENGINES; /lb/status `clusters.<name>.engines` declares each): what the panel renders
             # its settings from when it is selected.
             "engine": m.spec.node_engine}
            for m in getattr(cluster, "choices", ())]


def local_kv_cache_dtypes(cluster) -> dict:
    """`cluster.kv_cache_dtypes` for a node the LB requests: per fleet choice (the ref an intent sends), every KV dtype
    and why its launch image cannot run it on the serving cards — the ONE library the fleet's validator asks
    (fleet._kv_dtype), in the pair's shape (`ModelSpec.kv_cache_dtype_choices`). One image serves every choice."""
    from vllm_lb import kv_dtype
    archs = [c.type.arch.value for c in cluster.serving_cards]
    return {m.ref: kv_dtype.library().choices(cluster.launch.image, archs) for m in getattr(cluster, "choices", ())}


#: How many lines of the node's log the view carries while it starts or has failed: the last words, not the log.
LOG_LINES = 12


def applier_view(report: dict, state: str) -> dict:
    """`cluster.applier`: who listens on the node's host and what it last did (node_control.read_report), with the
    node's last log lines while it starts or has failed — its own words for what went wrong."""
    log = report.get("log")
    tail = log.strip().splitlines()[-LOG_LINES:] if log and state in ("starting", "failed") else None
    return {**{k: report.get(k) for k in ("listening", "heartbeat_age_s", "host", "applying", "applied",
                                          "container", "error")}, "log_tail": tail}


def local_view(cfg, *, desired: dict | None, applier: dict, loading: dict, fit: dict | None,
               store_error: str = "", **observed) -> dict:
    """`/lb/status.cluster` for a node the LB requests: `observed_view`'s, with what only this topology has — the
    wish (`desired`, `local_desired`) and what it needs on the card (`fit`, `local_fit`), the models it may select
    (`choices`) with the selected one's catalog limits, the applier (`applier_view`), the start's progress read off
    the node's log by the pair's own reader (`loading`), a stored wish the LB could not read (`store_error`), and how
    much host RAM a KV tier may take there (`ram_tier`: the fleet's bound, or why none — fleet.ram_tier_bound)."""
    from vllm_lb.fleet import ram_tier_bound
    out = observed_view(cfg, **observed)
    choices = local_choices(cfg)
    wanted = next((c for c in choices if desired and c["ref"] == desired.get("model")), None)
    bound, why = ram_tier_bound(fleet_cluster(cfg))
    out.update(desired=desired, applier=applier, loading=loading, store_error=store_error, choices=choices,
               limits=wanted["limits"] if wanted else {}, fit=fit, ram_tier={"max_gib": bound, "refused": why},
               kv_cache_dtypes=local_kv_cache_dtypes(fleet_cluster(cfg)))
    return out
