"""ROLES AT THE LB: what a MEMBER sees and may send, from the gate's word in two headers.

The front proxy's sign-in service owns who is what -- rows an admin edits in the panel -- and the proxy forwards its
answer on every door that reaches this LB, REPLACING a client's copy:

  X-Vllm-Role    `admin` | `member`; `unset` when a location forgot to ask or the check answered no role (refused
                 here: 403). nginx DROPS a header set to "", so the proxy must turn an empty answer into `unset` (the
                 README's map): a check that answers 2xx with no role is refused, never read as no header.
  X-Vllm-Models  a member's served model names, comma-joined; empty for an admin (an admin has every model)

NO X-Vllm-Role AT ALL is unrestricted, exactly as before roles existed: the LB has no sign-in of its own, so a caller
that reaches it without the proxy (the bench and swarm runners, the usage MCP, anything on its docker network or
loopback) was always the operator's. The ADDRESS is not checked, and checking it would not help: the proxy dials from
those same internal addresses, so a proxy location that forgot the header looks exactly like an internal caller. What
keeps a public client out is the proxy -- every door it opens must SET the header (a machine's Basic door clears it,
as it clears X-Vllm-User: a machine is the operator's app) -- and the LB reachable only through it.

A MEMBER may not shut a box down; they use the models granted to them and see the boxes that can load them. USE ONLY:
  * /v1/models lists their granted models only;
  * a turn must NAME one of them, and one that is served now: a model they were not granted is 403 naming it, one not
    loaded is 503 naming it -- never routed onto, or renamed to, another model (no substitute: AGENTS.md);
  * /lb/status is narrowed (`member_status`): the servers serving or able to load a granted model, the boxes they run
    on, the catalog rows granted -- and nobody else's requests, leases or runs, no engine address, no error text;
  * nothing else of the LB's: its control POSTs and every other /lb/ route are 403 `admin_only`. The proxy's sign-in
    service may refuse those first; this is the LB's own line, so a door that skipped that check cannot widen a member.

Never forwarded to an engine (proxy.forward drops both headers, as it drops X-Vllm-User).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ROLE_HEADER = "X-Vllm-Role"
MODELS_HEADER = "X-Vllm-Models"
ADMIN, MEMBER = "admin", "member"
#: what forward() strips from what it sends an engine
GATE_HEADERS = frozenset({ROLE_HEADER.lower(), MODELS_HEADER.lower()})


@dataclass(frozen=True)
class Member:
    models: frozenset[str]


class UnknownRole(ValueError):
    pass


def caller(headers: Mapping[str, str]) -> Member | None:
    """The member behind a request, or None for an admin or a caller inside the trust boundary (no header). An unknown
    role -- nginx's `unset` default, a typo, anything else -- raises UnknownRole: refused, never read as either."""
    role = headers.get(ROLE_HEADER)
    if role is None or role == ADMIN:
        return None
    if role != MEMBER:
        raise UnknownRole(f"{ROLE_HEADER} {role!r} is not a role this gateway knows (admin, member)")
    raw = headers.get(MODELS_HEADER) or ""
    return Member(frozenset(m.strip() for m in raw.split(",") if m.strip()))


#: The inference routes a member may POST, by name (a proxy that checks paths itself keeps the same list): an engine
#: route under /v1 that is not listed is the admin's.
MEMBER_POSTS = frozenset({"/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions", "/v1/completions",
                          "/v1/responses", "/v1/audio/transcriptions", "/v1/audio/translations"})
#: What a member reads besides the page's static files: the page itself, the narrowed status and their models.
MEMBER_READS = frozenset({"/", "/index.html", "/lb/status", "/lb/stream", "/v1/models"})
#: Pages that are an admin's even though they are static files (the run history: everyone's requests, every box).
ADMIN_PAGES = ("/benchmarks", "/sessions")


def plain_path(path: str) -> bool:
    """A path as nginx's normalised $uri is -- this LB sees the RAW one nginx forwards: absolute, printable, no %, no
    empty / `.` / `..` segment ("/" itself excepted). A member's request spelled otherwise is refused, not decoded."""
    if not isinstance(path, str) or not path.startswith("/") or not path.isprintable() or "%" in path:
        return False
    return path == "/" or all(s not in ("", ".", "..") for s in path[1:].split("/"))


def member_may(method: str, path: str, is_static) -> bool:
    """The LB's own second line (a proxy's own path check may come first), an ALLOWLIST on the raw path: a member
    reads MEMBER_READS and the page's static files (`is_static(path)`: a file the LB serves, proxy.static_file) except
    the admin's pages, and POSTs MEMBER_POSTS. Everything else is the admin's."""
    if not plain_path(path):
        return False
    if method in ("GET", "HEAD"):
        if path in MEMBER_READS:
            return True
        if path.startswith(("/lb/", "/v1/")) or any(path == p or path.startswith(p + ".") or path.startswith(p + "/")
                                                     for p in ADMIN_PAGES):
            return False
        return bool(is_static(path))
    return method == "POST" and path in MEMBER_POSTS


def turn_refusal(member: Member, model, served: bool) -> tuple[int, dict] | None:
    """A member's turn: (status, body) when it may not go, None when it may. `served`: the model is a served name of
    some live server right now (proxy `_exact_models`)."""
    if not isinstance(model, str) or not model:
        return 400, {"error": {"type": "model_required",
                               "message": "name one of your models in `model`: "
                                          + (", ".join(sorted(member.models)) or "none is granted to you yet")}}
    if model not in member.models:
        return 403, {"error": {"type": "model_not_granted",
                               "message": f"{model} is not one of your models; ask an admin to grant it"}}
    if not served:
        return 503, {"error": {"type": "model_not_loaded",
                               "message": f"{model} is not loaded on any server now; ask an admin to start it"}}
    return None


def routed_elsewhere(model: str) -> tuple[int, dict]:
    """A member's turn whose model IS loaded, but the router picked a server of another model for it: nothing is sent
    there (no substitute), and the refusal says so -- "not loaded" would be false."""
    return 503, {"error": {"type": "model_not_routed",
                           "message": f"no server of {model} could take this turn now; nothing was sent -- retry"}}


def member_models(payload: dict, member: Member) -> dict:
    """/v1/models for a member: the granted models only, as listed for an admin."""
    return {**payload, "data": [d for d in payload.get("data", []) if d.get("id") in member.models]}


def _cluster_models(cluster) -> set[str]:
    """The served names a fleet cluster can load: its model choices."""
    return {getattr(m.spec, "served_name", None) for m in getattr(cluster, "choices", ())} - {None}


#: what a member reads where /lb/status has an error's TEXT: that something failed, not why (an ssh error names a
#: box's address, a catalog error a path) -- the reason is an admin's. Never left out: a failing box must not look
#: healthy.
WITHHELD = "withheld: the reason is an admin's"

#: the keys of a backend row a member reads (an ALLOWLIST): not its `url` / `host_label` (the engine's address and
#: name on the network) nor `last_error` (its text names them)
MEMBER_BACKEND_KEYS = frozenset({
    "kind", "box", "box_alive", "cluster", "live", "loaded", "parked", "penalised", "served_models", "max_model_len",
    "waiting", "running", "inflight", "depth_known", "depth_read_at", "throughput", "tokens_per_s", "history",
    "stalled_s", "failure_streak", "last_probe_at", "last_used_at", "gen_tokens_total", "prompt_tokens_total"})
#: the keys of a server view a member reads (an ALLOWLIST): what it is, its state and its figures -- not its address,
#: applier, switch path, settings ranges, or the catalog's other models (`models` is cut to the granted ones below)
MEMBER_SERVER_KEYS = frozenset({
    "name", "configured", "state", "since", "up_since", "task", "hardware", "topology", "held", "freeing",
    "max_depth", "tiles", "saturation", "throughput_target", "serves", "loaded_model", "loading", "engine", "default",
    "desired", "effective", "models", "boxes", "detail", "catalog_error", "store_error"})
#: a server view's error TEXT, said as WITHHELD
_SERVER_ERROR_KEYS = frozenset({"detail", "catalog_error", "store_error"})
#: the keys of a box (in a server view) a member reads: its memory and its place in the run -- not its name on the
#: network, its holds and heals, its checkpoint search (`availability`: paths and mount errors, said as WITHHELD)
MEMBER_BOX_KEYS = frozenset({
    "reservation", "park", "running_jobs", "broker_vllm_running", "mem_available_gib", "mem_stable_gib",
    "mem_total_gib", "at_rest", "mem_rest_gib", "mem_at_launch", "mem_release_target", "rank", "memory", "gpu",
    "availability", "error"})
_BOX_ERROR_KEYS = frozenset({"availability", "error"})
#: the keys of a box's hardware row a member reads (hw.py builds them): its figures -- not `host`, the machine's name
MEMBER_HARDWARE_KEYS = frozenset({
    "age_s", "stale", "memory", "mem_unified", "mem_total_mb", "mem_used_mb", "ram_percent", "ram_total_mb",
    "ram_used_mb", "swap_used_mb", "cpu_percent", "cpu_per_core", "gpu_count", "gpu_pct", "gpus", "cards", "rows",
    "history", "recorder"})


def _said(value):
    """An error's presence, not its text."""
    return WITHHELD if value else value


def _member_backend(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k in MEMBER_BACKEND_KEYS}
    if "last_error" in row:
        out["last_error"] = _said(row["last_error"])
    return out


#: what a member reads for a parked box: that it is held, not whose hold it is (a park names its holder)
PARKED = "held: by an admin's park"


def _member_box(box: dict) -> dict:
    out = {}
    for k, v in (box or {}).items():
        if k not in MEMBER_BOX_KEYS:
            continue
        if k in _BOX_ERROR_KEYS:
            out[k] = _said(v)
        elif k == "reservation":
            # its status only: a job id and a claimer name the fleet's workers
            out[k] = {"status": v.get("status")} if isinstance(v, dict) else v
        elif k == "park":
            out[k] = PARKED if v else v
        else:
            out[k] = v
    return out


def _member_server(view: dict, models: frozenset) -> dict:
    out = {}
    for k, v in view.items():
        if k not in MEMBER_SERVER_KEYS:
            continue
        if k in _SERVER_ERROR_KEYS:
            out[k] = _said(v)
        elif k == "boxes" and isinstance(v, dict):
            out[k] = {name: _member_box(box) for name, box in v.items()}
        elif k == "models" and isinstance(v, list):
            out[k] = [m for m in v if isinstance(m, dict) and m.get("served_name") in models]
        else:
            out[k] = v
    return out


def member_status(status: dict, member: Member, fleet, cluster_name: str | None = None) -> dict:
    """/lb/status for a member: what their models touch, nothing more. A server is theirs when it SERVES a granted
    model now or CAN LOAD one (a fleet cluster whose choices include it); its boxes are theirs to see. Everyone else's
    requests (the queue's rows carry prompt snippets), leases and runs are not -- those keys are kept, EMPTY, so the
    page renders; every other key of /lb/status is left out (an allowlist). `cluster_name`: the one cluster `status["cluster"]` is (the LB's cluster_name), None for none."""
    models = member.models
    fleet_clusters = getattr(fleet, "clusters", {}) or {}
    can_load = {n for n, c in fleet_clusters.items() if _cluster_models(c) & models}

    def serves_theirs(view: dict) -> bool:
        s = view.get("serves") or {}
        return s.get("served_name") in models

    clusters = {n: v for n, v in (status.get("clusters") or {}).items() if n in can_load or serves_theirs(v)}
    backends = [b for b in status.get("backends", [])
                if set(b.get("served_models") or ()) & models or b.get("cluster") in clusters]
    boxes = {b.get("box") for b in backends if b.get("box")}
    for n in clusters:
        boxes |= {x.name for x in getattr(fleet_clusters.get(n), "boxes", ())}
    one = status.get("cluster") or {}
    queue = status.get("queue") or {}
    models_block = status.get("models") or {}
    # BUILT FROM AN ALLOWLIST: only these keys, so a key /lb/status gains tomorrow is the admin's until
    # it is listed here. The queue keeps its aggregates (how busy the fleet is); its ROWS are other people's turns.
    return {
        "now": status.get("now"),
        "uptime_s": status.get("uptime_s"),
        "db_ok": status.get("db_ok"),
        # whether the hardware listener runs, not its counters or every box's address (hw.status `hosts`)
        "hardware_state": {"state": (status.get("hardware_state") or {}).get("state")},
        "role": MEMBER,
        "granted_models": sorted(models),
        # the two periods the page draws time with (a history's buckets, the stream's "stale") — no deployment detail
        "config": {k: (status.get("config") or {}).get(k) for k in ("probe_interval_s", "tick_s")},
        "backends": [_member_backend(b) for b in backends],
        "clusters": {n: _member_server(v, models) for n, v in clusters.items()},
        "cluster": _member_server(one, models) if cluster_name is not None and cluster_name in clusters
        else {"configured": False},
        "hardware": {k: {hk: hv for hk, hv in (v or {}).items() if hk in MEMBER_HARDWARE_KEYS}
                     for k, v in (status.get("hardware") or {}).items() if k in boxes},
        "models": {"cluster": [m for m in models_block.get("cluster", []) if m.get("served_name") in models],
                   "classic": [m for m in models_block.get("classic", []) if m.get("served_name") in models],
                   "error": _said(models_block.get("error"))},
        "leases": [], "declared": [], "swarm": {"configured": False},
        "queue": {"inflight": [], "inflight_total": 0, "recent": [],
                  "counts": queue.get("counts") or {}, "limits": queue.get("limits") or {}},
    }
