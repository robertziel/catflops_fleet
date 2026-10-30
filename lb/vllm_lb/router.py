"""Which backend gets this request — a pure function of the snapshot.

Tiers, in order: a LEASED live vLLM or a HELD cluster (ours to use) → any other live vLLM (the fleet's own, if
allowed) → a demoted one (deep queue) → nothing. Within a tier the backend with the fewest of OUR in-flight requests
wins; ties break on URL so the choice is stable. THERE IS NO FALLBACK TIER (operator, 2026-10-17: "If can't run then
can't run, no fallback box"): a request no server can take waits for one on its way, or is refused naming why and
what is live — the LB's ollama fallback, its leases and its keep-loaded are retired (config.RETIRED_KEYS).

With the broker OFF (`Config.broker_enabled`) there is no lease to hold: every configured vLLM sits in the first tier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vllm_lb.catalog import CHAT, EFFORT_OFF, ENGINE_DROPS_PRIORITY, TRANSCRIPTION
from vllm_lb.config import Config
from vllm_lb.state import CLUSTER, VLLM, Backend, Snapshot


def _usable(b: Backend, now: float) -> bool:
    return b.live and b.penalty_until <= now


#: THE PATHS OF THE TRANSCRIPTION TASK (catalog.TASKS): OpenAI's speech-to-text uploads. Every other inference path is
#: the chat task's.
TRANSCRIPTION_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")


def task_of_path(path: str) -> str:
    """The task a request to `path` asks of a model: transcription for an audio upload, chat for everything else."""
    return TRANSCRIPTION if path.split("?", 1)[0] in TRANSCRIPTION_PATHS else CHAT


def task_of(b: Backend, cfg: Config) -> str:
    """The task backend `b` serves: its cluster's (fleet `Cluster.task`, its models' catalog task). A classic server and
    a cluster no fleet declares run the LB's own launch profiles, which the catalog holds to chat."""
    fleet = cfg.fleet
    c = fleet.clusters.get(b.cluster) if fleet is not None and b.kind == CLUSTER and b.cluster else None
    return c.task if c is not None else CHAT


def serving(snap: Snapshot, cfg: Config, model: str | None, *, exact: bool = False, task: str = CHAT
            ) -> list[Backend]:
    """The backends a request naming `model` may go to (worklog/parallel-clusters.md, routing).

    FIRST BY TASK (`task_of_path`): only backends serving the request's task are candidates — a transcription never
    reaches a chat cluster, a chat request never a transcription backend. Then by name, the same rule for both tasks,
    each with ITS default cluster (`Config.default_cluster_for`): a transcription's name is its form's `model` field,
    and a loose one (an OpenAI client's `whisper-1`) is the default speech cluster's, which `multipart.set_field` renames to its
    own.

    A name some server SERVES (`served_by`: its /v1/models) narrows the servers to those — two clusters serving two
    models each get their own requests; the same model on two is picked by depth as ever. A name none serves is, when
    `exact` (a model this fleet's clusters run: `fleet_models`), served by NO server — it names a model this fleet runs
    elsewhere or not now, and rewriting it onto whatever is up would be a silent substitution.

    NO MODEL, or a LOOSE name nobody serves (Claude Code's `qwen3.8:latest`, the chat's unnamed turn): THE DEFAULT
    CLUSTER alone among the clusters (`Config.default_cluster`, the fleet's `default: true`; the one cluster of a fleet
    of one) — under its served name (`rewrite_model`), as today's single cluster takes it — beside the classic servers,
    as ever. Never another cluster: the audit's 2B GUI agent answered Claude Code's turns (AUDIT-FINAL P1-3). With one
    cluster nothing changes. No fallback is ever offered (the tier is retired)."""
    backends = [b for b in snap.backends if task_of(b, cfg) == task]
    if model is not None:
        servers = [b for b in backends if model in served_by(b, cfg)]
        if servers or exact:
            return servers
    default = cfg.default_cluster_for(task)
    return [b for b in backends if b.kind != CLUSTER or b.cluster == default]


def elsewhere(snap: Snapshot, cfg: Config, model: str | None, *, exact: bool = False, task: str = CHAT) -> list[str]:
    """What is LIVE but not offered to this request (`serving` left it out) — `<served name> (cluster <name>)` — for the
    503's words: the client learns what it may name instead of being moved onto it. Only its own task's: another
    task's model is nothing it could name."""
    pool = {b.url for b in serving(snap, cfg, model, exact=exact, task=task)}
    return [f"{m} (cluster {b.cluster})" for b in snap.backends
            if b.live and b.url not in pool and b.kind == CLUSTER and task_of(b, cfg) == task
            for m in served_by(b, cfg)]


@dataclass(frozen=True)
class Eligible:
    """`eligible`'s answer: the backends the request may go to (`serving`), and `why` it is `model_not_served` when
    no server serves the catalog model it names ("" otherwise)."""
    backends: list[Backend]
    why: str = ""


def fleet_models(cfg: Config, catalog) -> frozenset[str]:
    """Every name a CLUSTER of this fleet may serve — the `exact` names (`serving`): each catalog row a cluster can
    launch (a `cluster` launch profile: the pair's Start picks among them) and each row a declared cluster names in
    the fleet file (`model`, `choices`: a one-host node's), by id and served name. A classic-only row is not one: its
    served name (`qwen3.8:latest`) is the classic servers' tag, which clients use loosely."""
    from vllm_lb.catalog import CLUSTER_MODE
    rows = list(catalog.for_mode(CLUSTER_MODE))
    for c in (cfg.fleet.clusters.values() if cfg.fleet is not None else ()):
        rows += [m.spec for m in ((c.model,) if c.model is not None else ()) + tuple(c.choices or ())]
    return frozenset(n for s in rows for n in (s.id, s.served_name))


def cluster_models(cfg: Config, catalog, name: str | None) -> frozenset[str]:
    """The names cluster `name` may serve, by id and served name: its declared `model` and `choices` (a one-host
    node's), or — a cluster whose Start names its model (the pair: no `model` declared) — every catalog row a cluster
    can launch. Empty for no such cluster."""
    from vllm_lb.catalog import CLUSTER_MODE
    c = cfg.fleet.clusters.get(name) if cfg.fleet is not None and name else None
    if c is None:
        return frozenset()
    rows = (catalog.for_mode(CLUSTER_MODE) if c.model is None
            else [m.spec for m in (c.model, *(c.choices or ()))])
    return frozenset(n for s in rows for n in (s.id, s.served_name))


def exact_models(cfg: Config, catalog, task: str = CHAT) -> frozenset[str]:
    """THE names routed EXACTLY on a `task` path — served as named or refused, never moved: what this fleet's clusters
    run, of every task (`fleet_models`: a speech model named on a chat path is refused, not answered by the pair), less
    what that task's DEFAULT cluster can launch (`cluster_models`), which is the default's as a loose name is (a client
    may name a pair profile the pair is not serving). So only a model nothing but ANOTHER
    cluster runs is refused unserved; a fleet of one routes nothing exactly — as before parallel clusters."""
    return fleet_models(cfg, catalog) - cluster_models(cfg, catalog, cfg.default_cluster_for(task))


def eligible(snap: Snapshot, cfg: Config, *, model: str | None = None, catalog=None) -> Eligible:
    """`serving` for a request naming `model`, with `fleet_models` deciding `exact`, and the 503's words when no server
    serves that catalog model — naming the CHAT models that ARE live (`exact` is a chat path's: an upload names none
    the router reads), so the client learns what it may ask for. `catalog`:
    the proxy's (the one its pickers and switch validate against); the shipped catalog file when left out — the same
    file, never another source."""
    from vllm_lb.catalog import ModelCatalog
    if catalog is None:
        catalog = ModelCatalog.load()
    exact = model in exact_models(cfg, catalog)
    pool = serving(snap, cfg, model, exact=exact)
    if exact and not any(model in served_by(b, cfg) for b in pool):
        live = sorted({m for b in snap.backends if b.live and task_of(b, cfg) == CHAT for m in served_by(b, cfg)})
        return Eligible(pool, f"model {model!r} is not served by any cluster on this gateway right now; "
                              f"live: {', '.join(live) if live else 'nothing'}")
    return Eligible(pool)


def choose(snap: Snapshot, cfg: Config, exclude: frozenset[str] = frozenset(),
           *, require_lease: bool = True, model: str | None = None, exact: bool = False,
           task: str = CHAT) -> Backend | None:
    """`exclude` = backends already tried for THIS request (failover walks the tiers).

    `require_lease=False` answers the SCALER's counterfactual — "would this be chosen if we
    held it?" — and is never passed by the proxy.

    `model` = what the request names (worklog/parallel-clusters.md): a name ANY configured backend SERVES
    (`served_models`, its /v1/models) narrows the candidates to the backends serving it — two clusters serving two
    models each get their own requests, and the same model on two is picked by queue depth as ever. A name none
    serves keeps today's contract (any backend; `rewrite_model` renames it to what that backend serves): the fleet
    has always absorbed loose client names, and with one cluster nothing changes. `exact`: the name is a catalog
    model (`eligible`) — served as named or not at all (`serving`). `task`: the request's (`task_of_path`)."""
    pool = serving(snap, cfg, model, exact=exact, task=task)
    # EACH CLUSTER BACKEND ANSWERS TO ITS OWN CLUSTER'S VIEW (`Snapshot.cluster_view_of`, keyed by `Backend.cluster`):
    # HELD is that cluster loop's word — its ranks up on boxes it holds and the head answering
    # (worklog/vllm-cluster-control-plane.md); its SEATS are its own `max_depth` (the pair's concurrency, a node's
    # declared seats — the classic boxes' `vllm_max_depth` never caps it; a cluster whose seats nobody declared has
    # none to be within, `busy`). One view for every cluster backend sent a held pair's traffic to an off node.
    # While the operator WANTS any cluster, no new request goes to classic vLLM: a claimed reservation makes
    # broker_parrot stop those servers, and a turn started on one would be cut off; the turn waits for the cluster.
    classic_allowed = not snap.cluster_wanted()

    def routable(b: Backend) -> bool:
        return b.kind != VLLM or classic_allowed

    leased = snap.leased_labels()
    # BROKER OFF (`LB_BROKER_ENABLED` unset): there is no broker to hold a lease from, so every configured
    # backend is the operator's own and a live one is routable as it stands — the cluster still answers to
    # its loop. The tiers are unchanged: a deep one is demoted.
    broker = cfg.broker_enabled
    if not broker:
        leased = {b.host_label for b in snap.backends if b.kind == VLLM and b.host_label}
    if not require_lease:
        # The counterfactual pretends every vLLM is held, tiers included.
        leased = {b.host_label for b in snap.backends if b.kind == VLLM and b.host_label}

    def held(b: Backend) -> bool:
        """Do we hold a broker lease on this server? NO LEASE, NO TRAFFIC.

        Every backend here is a broker_parrot service, but owning one is not holding it:
        serving from a sidecar the fleet did not give us is capacity taken without telling it.
        The gate lives in `choose` rather than in a tier so that the two tiers which BYPASS
        ordinary tiering — the deep-demotion tier and the penalised retry — cannot become
        exceptions to it.
        """
        if b.kind == CLUSTER:
            # Not a lease the scaler could ask for — held is the loop's word, in the counterfactual too.
            return bool(snap.cluster_view_of(b).get("held"))
        if not broker:
            return True                    # no broker, no lease to hold: the server is ours
        if not require_lease:
            # The COUNTERFACTUAL: "would this be chosen if we held it?" (the claim wait asks it).
            return True
        return b.host_label in leased or cfg.use_unleased_vllm

    def depth(b: Backend) -> int:
        """How many requests are ahead of us on this backend.

        `inflight` is OURS and is counted because the probe is up to `probe_interval_s` stale:
        without it, every turn arriving inside one interval reads the same pre-admission depth
        and they all pile onto the same backend (measured: 8 concurrent picks, one 2-slot box).
        """
        # max(), NOT sum: vLLM's `running` ALREADY counts the request we sent, so adding our
        # own `inflight` counted it twice and one of our turns demoted an idle two-slot box —
        # capping effective concurrency at 1 per box. `inflight` still carries the request in
        # the stale window before the next probe sees it, which is why it is not simply dropped.
        return b.waiting + max(b.running, b.inflight)

    def busy(b: Backend) -> bool:
        # DEPTH, not waiting: a vLLM at its --max-num-seqs reports waiting=0 until someone is
        # actually behind, so a waiting-only test reads a completely full server as idle.
        # UNKNOWN depth counts as busy: a vLLM whose /metrics did not answer has no measured
        # pressure, and guessing "idle" hands it every request precisely when it is loaded.
        if b.kind == CLUSTER:
            # No seats known is no room known: busy, as an unknown depth is — demoted, never banned.
            seats = snap.cluster_view_of(b).get("max_depth")
            return not b.depth_known or not isinstance(seats, int) or depth(b) > seats
        return not b.depth_known or depth(b) > cfg.vllm_max_depth

    def tiers_over(pool: list[Backend]) -> list[list[Backend]]:
        # A vLLM queued deeper than the threshold drops below every server that has room: waiting an hour and
        # then 504ing is not fine. It stays in the list, last, so it is still chosen when nothing else is live
        # (demoted, not banned).
        ok = [b for b in pool if not busy(b)]
        deep = [b for b in pool if busy(b)]
        # `deep` is a DEMOTION, not an escape hatch: it is drawn from the same policy-filtered
        # field as every other tier. Built from the raw pool it did the opposite — a vLLM
        # excluded by `use_unleased_vllm` became reachable exactly when it was overloaded.
        allowed = (lambda b: True) if cfg.use_unleased_vllm else (
            lambda b: b.kind != VLLM or b.host_label in leased)
        return [
            [b for b in ok if b.kind == CLUSTER or (b.kind == VLLM and b.host_label in leased)],
            [b for b in ok if b.kind == VLLM] if cfg.use_unleased_vllm else [],
            [b for b in deep if allowed(b)],
        ]

    def pick(tiers: list[list[Backend]]) -> Backend | None:
        for tier in tiers:
            if tier:
                # Shallowest queue first; inflight and url keep it deterministic.
                return min(tier, key=lambda b: (depth(b), b.inflight, b.url))
        return None

    cands = [b for b in pool
             if _usable(b, snap.now) and b.url not in exclude and held(b) and routable(b)]
    chosen = pick(tiers_over(cands))
    if chosen is not None:
        return chosen
    # Nothing unpenalised is left. A penalty is a PREFERENCE ("try anyone else first"), never
    # a ban: a backend that is merely failing still beats no backend at all, and refusing the
    # turn outright would be the worse of the two outcomes. Only `exclude` — this request has
    # already tried it — is absolute.
    penalised = [b for b in pool
                 if b.live and b.penalty_until > snap.now and b.url not in exclude and held(b)
                 and routable(b)]
    return pick(tiers_over(penalised))


def served_by(backend: Backend, cfg: Config) -> tuple[str, ...]:
    """The model ids this backend answers to: exactly what its /v1/models said (nothing known ⇒ leave requests
    alone)."""
    return tuple(backend.served_models)


def rewrite_effort(body: dict, backend: Backend, cfg: Config) -> dict:
    """Map the client's reasoning-effort word onto one the backend's chat template knows.

    Claude Code sends `output_config: {"effort": "high"}` (captured from the wire); vLLM serves
    the Anthropic Messages API natively and hands that straight to the template. qwen3.8's
    template accepts `low`/`medium`/`xhigh` and raises on anything else — so `high` and `max`,
    which ARE Claude Code's own ladder, made EVERY real session 500 in ~4 ms while a loaded
    model sat at 0% GPU. Hand-written probes carry no `output_config`, which is why three
    deploys looked green.

    Config-driven rather than hardcoded here: the next model will have a different ladder.
    Returns the same dict when nothing changes.
    """
    out = body
    for field, key in (("output_config", "effort"), ("chat_template_kwargs", "reasoning_effort")):
        section = body.get(field)
        if not isinstance(section, dict):
            continue
        have = section.get(key)
        want = cfg.effort_map.get(have) if isinstance(have, str) else None
        if have == EFFORT_OFF:
            # "off" is no effort word: the backend gets thinking DISABLED in the form its API reads —
            # Anthropic's `thinking` block, or the template's own `enable_thinking` switch.
            if out is body:
                out = dict(body)
            rest = {k: v for k, v in section.items() if k != key}
            if field == "chat_template_kwargs":
                out[field] = {**rest, "enable_thinking": False}
            else:
                out["thinking"] = {"type": "disabled"}
                if rest:
                    out[field] = rest
                else:
                    del out[field]
            continue
        if want is None or want == have:
            continue
        if out is body:
            out = dict(body)
        out[field] = {**section, key: want}
    return out


#: THE CALLER'S PRIORITY (operator 2026-10-28): an integer, LOWER = SOONER — vLLM's own direction (config/scheduler.py
#: "lower value means earlier handling", ties by arrival) and SGLang's under --schedule-low-priority-values-first. Every
#: launch turns priority scheduling on (tp-rank, the node's compose), with no header the engine's default 0.
PRIORITY_HEADER = "X-Priority"
#: The paths whose JSON body the engines read `priority` from — their OpenAI request models (vLLM chat_completion/
#: completion protocol, SGLang openai/protocol.py) and the Anthropic /v1/messages adapter of the fleet's patched images
#: (deploy/images/vllm/priority/priority.patch; upstream's adapter drops the field). Anything else — an upload,
#: an embedding — would drop it: refused.
PRIORITY_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages")
_PRIORITY = re.compile(r"-?\d{1,9}")


def priority_of(raw: str | None, path: str, body) -> int | None:
    """The priority `X-Priority` asks for — None with no header — or ValueError: a value that is no integer, a path whose
    engine would DROP it (both engines ignore a field their request model lacks), or a body naming another one."""
    if raw is None:
        return None
    value = raw.strip()
    if not _PRIORITY.fullmatch(value):
        raise ValueError(f"{PRIORITY_HEADER} must be an integer of up to 9 digits (lower = sooner), got {raw!r}")
    if path not in PRIORITY_PATHS:
        raise ValueError(f"{PRIORITY_HEADER} reaches the engine as the body's `priority`, which only "
                         f"{' and '.join(PRIORITY_PATHS)} read — {path} would drop it")
    if not isinstance(body, dict):
        raise ValueError(f"{PRIORITY_HEADER} rides a JSON body's `priority`; this body is not a JSON object")
    if "priority" in body and body["priority"] != int(value):
        raise ValueError(f"{PRIORITY_HEADER} {value} and the body's priority {body['priority']!r} disagree: send one")
    return int(value)


def priority_refusal(engine: str | None) -> str:
    """Why `engine` would DROP an integer priority — "" where it schedules by one (vLLM, SGLang: every launch turns it
    on) or where no engine is known. The proxy refuses the header (400) for a backend running such an engine, once the
    backend is chosen (`proxy.engine_of_backend`), exactly as it refuses a path that would drop it."""
    why = ENGINE_DROPS_PRIORITY.get(engine or "", "")
    return f"{PRIORITY_HEADER} reaches the engine as the body's `priority`, and {why}" if why else ""


#: The page's toggle (the REQUESTS card's square, POST /lb/requests/<id>/priority): one step sooner than a request
#: naming none (0). Under SGLang's own preemption threshold (10), so it orders the queue and never evicts a running turn.
PRIORITY_GUI = -1


def priority_lock(path: str, committed: bool) -> str | None:
    """Why a turn's priority can no longer move — None while it can: the engine would drop the field on its path, or
    the client has its answer's first byte (the engine admitted it into its batch, where neither engine changes it)."""
    if path not in PRIORITY_PATHS:
        return f"{path}: the engine's adapter carries no priority field — it would be dropped"
    if committed:
        return "streaming: the engine is answering it; a running request's priority is fixed"
    return None


def rewrite_priority(body: dict, priority: int | None) -> dict:
    """The body with the caller's `priority` set; the same dict when there is none."""
    return body if priority is None else {**body, "priority": priority}


def rewrite_model(body: dict, backend: Backend, cfg: Config) -> dict:
    """The request's `model` becomes what the backend actually serves. Clients name the
    fleet's model loosely (`qwen3.6`, `qwen3.8:latest`, whatever their config says) and the
    served id changes when the operator reseeds a sidecar; the LB absorbs that here so no
    client needs a config change. Returns the same dict when nothing changes."""
    served = served_by(backend, cfg)
    if not served or body.get("model") in served:
        return body
    out = dict(body)
    out["model"] = served[0]
    return out
