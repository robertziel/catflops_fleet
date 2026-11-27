"""THE OBSERVABILITY ROUTES, answered for the proxy: `GET /metrics` (prom.py), `GET
/lb/perf/daily` (daily.py), `GET /lb/check` (connectivity.py) and `GET /lb/history` (hw_tiers.py).

WHY A MODULE OF ITS OWN. proxy.py is the one file every change to the gateway touches; these four read-only endpoints
need only the proxy's config, its hub snapshot, its hardware listener and its perf log. Kept here, proxy.py carries one
dispatch line and one status key, and each route's argument checking sits beside the module whose data it serves.

All four are GET/HEAD, read-only, and behind the same trust boundary as /lb/status: the host allowlist and the role
check run before the dispatch (proxy.handle), and none of the paths is in a member's allowlist (roles.MEMBER_READS) —
a member is refused before reaching here.
"""
from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

from vllm_lb import connectivity, daily, prom
from vllm_lb.hw_tiers import METRICS, TIERS


def _arg(query: str, name: str) -> str:
    values = urllib.parse.parse_qs(query).get(name) or []
    return values[-1].strip() if values else ""


def _bad(h, message: str, status: int = 400, kind: str = "invalid_request"):
    return h.send_json(status, {"error": {"type": kind, "message": message}})


def metrics(proxy, h, query: str):
    """GET /metrics — the status document as Prometheus text (prom.py). Off unless LB_METRICS=1: then a 404 naming it."""
    if not proxy.cfg.metrics_enabled:
        return _bad(h, "Prometheus /metrics is off on this gateway: LB_METRICS=1 (lb/.env) turns it on", 404,
                    "metrics_off")
    return h.send_own(200, prom.render(proxy.status()).encode(), prom.CONTENT_TYPE)


def perf_daily(proxy, h, query: str):
    """GET /lb/perf/daily?server= — the 30-day per-model rollup of the perf log (daily.py), scoped like /lb/perf."""
    server = proxy._server_param(h, query)
    if server is not None and not isinstance(server, str):
        return None                                     # the 404 naming the declared servers is already sent
    now = time.time()
    got = daily.scan(proxy.perf, now=now, keep=proxy._keep(server))
    out = daily.fold(got["windows"], now=now)
    out["models"] = [{**m, "label": proxy._describe_model(m["id"])["label"] if m["id"] else "(no model recorded)"}
                     for m in out["models"]]
    out.update(server=server, error=got["error"], malformed_total=got["malformed_total"],
               unreadable=got["unreadable"], recorder={"enabled": proxy.cfg.perf_enabled})
    return h.send_json(200, out)


def _executor(cfg):
    """The SSH executor a reachability row runs through: the loop's own class, key and known hosts (executors.py)."""
    from vllm_lb.executors import executor_class
    cls = executor_class("ssh")
    return cls(key_path=cfg.cluster_ssh_key, known_hosts=cfg.cluster_known_hosts,
               script_path=str(Path(__file__).resolve().parent.parent / "cluster" / "tp-rank"))


def check(proxy, h, query: str):
    """GET /lb/check[?box=] — every fleet box's channels (or one box's), probed now (connectivity.py)."""
    from vllm_lb.probe import probe_backend
    now = time.time()
    snap = proxy.hub.snapshot()
    plans = connectivity.plan(proxy.cfg, snap.backends)
    box = _arg(query, "box")
    if box:                                             # one box: the panel's per-card button
        plans = [p for p in plans if p["box"] == box]
        if not plans:
            names = ", ".join(p["box"] for p in connectivity.plan(proxy.cfg, ())) or "none"
            return _bad(h, f"no box {box!r} in the fleet file: it names {names}", 404, "box_unknown")
    ssh = _executor(proxy.cfg) if any(p["ssh"] for p in plans) else None
    hw = proxy.hw
    out = connectivity.check(
        plans, hardware=hw.snapshot(now) if hw is not None else {},
        hw_state=hw.status()["state"] if hw is not None else "not running in this gateway",
        max_age_s=proxy.cfg.hw_max_age_s, now=now,
        probe=lambda b: probe_backend(b, proxy.cfg, now=time.time()),
        reach=lambda host: ssh.reach(host, connectivity.SSH_TIMEOUT_S))
    return h.send_json(200, out)


def tiers_of(proxy):
    hw = proxy.hw
    return getattr(hw, "tiers", None) if hw is not None else None


def tiers_status(proxy) -> dict:
    """/lb/status `hardware_tiers`: the store's own status, or why there is none."""
    tiers = tiers_of(proxy)
    if tiers is None:
        return {"state": "off", "why": "no hardware history in this gateway (LB_HW_TIERS_FILE empty, or no listener)"}
    return tiers.status()


def history(proxy, h, query: str):
    """GET /lb/history?box=&metric=&tier= — one box's coarse history in a tier (hw_tiers.py); `metric` absent = every
    figure, under `series`."""
    tiers = tiers_of(proxy)
    if tiers is None or not tiers.path:
        return _bad(h, "hardware history is off: LB_HW_TIERS_FILE (lb/.env) names its file", 503, "history_off")
    box, metric, tier = _arg(query, "box"), _arg(query, "metric"), _arg(query, "tier")
    if tier not in TIERS:
        return _bad(h, f"`tier` is one of {', '.join(TIERS)}")
    if metric and metric not in METRICS:
        return _bad(h, f"`metric` is one of {', '.join(METRICS)}, or absent for all")
    now = time.time()
    if metric:
        out = tiers.series(box, metric, tier, now=now)
    else:
        each = {m: tiers.series(box, m, tier, now=now) for m in METRICS}
        out = None if any(v is None for v in each.values()) else {
            "box": box, "tier": tier, "bucket_s": TIERS[tier][0], "window_s": TIERS[tier][1],
            "series": {m: {"label": v["label"], "unit": v["unit"], "points": v["points"]} for m, v in each.items()}}
    if out is None:
        return _bad(h, f"no hardware history for box {box!r}: heard {', '.join(tiers.boxes()) or 'none'}", 404,
                    "box_unknown")
    out["state"], out["why"] = tiers.state, tiers.why
    return h.send_json(200, out)


#: path → handler(proxy, h, query). GET/HEAD only (proxy.handle).
ROUTES = {"/metrics": metrics, "/lb/perf/daily": perf_daily, "/lb/check": check, "/lb/history": history}
