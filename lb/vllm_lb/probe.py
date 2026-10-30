"""What is each backend doing right now? One bounded HTTP probe per backend per tick.

vLLM: `/v1/models` says live + which ids it serves + its context window; `/metrics` adds the
queue pressure (`vllm:num_requests_waiting` / `_running`) the scaler reads, and the `scrape` the
hub measures rates from. ollama: live only if the fallback model is actually pulled there — an
ollama that would 404 the model is not a backend — and `/api/ps` says what it holds in memory. Never
raises; the fields come back with `last_error` set instead.

SGLang serves that same HTTP API but exposes a DIFFERENT metrics vocabulary, so it is normalised
here, at the boundary: `parse_metrics` (vllm_lb/engine_metrics.py, the vocabularies' one definition,
re-exported here) reads which vocabulary a scrape speaks off the text itself and maps SGLang's families onto the canonical `vllm:*` keys (`SCRAPE_KEYS`) that throughput.py,
saturation.py, state.py, cluster_defaults.py, cores.py and perflog.py already read. Nothing
downstream learns a second set of names, and a figure SGLang does not publish stays null.

Every result carries `mono_at`, `time.monotonic()` when the probe learned what it reports: for a
scrape, right after its body was READ. A rate's dt is the time between two reads, never between
two tick starts on a wall clock (worklog/vllm-gateway-throughput.md, defect E).
"""
from __future__ import annotations

import email.utils
import http.client
import json
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Callable

from vllm_lb.config import Config
from vllm_lb.state import CLUSTER, VLLM, Backend

Getter = Callable[[str, float], tuple[int, bytes]]

#: The only `/health` answers that mean "this engine cannot serve": vLLM's own not-ready
#: signal, and an erroring one. Every other status means the route is not a usable vLLM
#: health check, which is a statement about the ROUTE, not about the engine.
NOT_READY = frozenset({500, 503})
#: `/health` is trivial and is one of three sequential probes per vLLM per tick.
HEALTH_TIMEOUT_S = 1.5


class Answer(tuple):
    """What a GET answered. It unpacks as `(status, body)` — the getter contract every probe and test
    speaks — and keeps the response headers, lower-cased, for the probe that needs one: ollama's
    `Date`, the server's own clock."""

    headers: dict[str, str]

    def __new__(cls, status: int, body: bytes, headers: dict[str, str] | None = None) -> "Answer":
        answer = super().__new__(cls, (status, body))
        answer.headers = {k.lower(): v for k, v in (headers or {}).items()}
        return answer


def http_get(url: str, timeout: float) -> Answer:
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("GET", u.path or "/", headers={"Host": u.netloc})
        resp = conn.getresponse()
        return Answer(resp.status, resp.read(), dict(resp.getheaders()))
    finally:
        conn.close()


from vllm_lb.engine_metrics import *  # noqa: F401,F403 (the vocabularies, re-exported for the probe's callers)
from vllm_lb.engine_metrics import (_HIT_MODES, _PROMPT_MODES, _labels, _sample, parse_metrics,  # noqa: F401
                                    with_health)




#: Go's `time.Time{}` — ollama has not set an expiry yet: the model is still loading.
_GO_ZERO_YEAR = 1
_RFC3339 = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})")


def _rfc3339(value) -> datetime | None:
    """Go's RFC 3339 time, nanoseconds and offset included, as an aware datetime; None if it is not one."""
    match = _RFC3339.fullmatch(value.strip()) if isinstance(value, str) else None
    if match is None:
        return None
    *parts, fraction, zone = match.groups()
    offset = timedelta(0) if zone == "Z" else timedelta(hours=int(zone[1:3]), minutes=int(zone[4:6]))
    try:
        return datetime(*map(int, parts), int((fraction or "0")[:6].ljust(6, "0")),
                        tzinfo=timezone(-offset if zone[0] == "-" else offset))
    except ValueError:
        return None


def _http_date(value: str | None) -> float | None:
    """An HTTP `Date` header in epoch seconds; None if absent or unreadable."""
    try:
        return email.utils.parsedate_to_datetime(value).timestamp() if value else None
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _nonnegative_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def parse_ollama_ps(body: bytes, date: str | None) -> dict:
    """What ollama's `GET /api/ps` says is resident (api/types.go:845-854), each model's unload
    countdown read against the SERVER's own clock — its `Date` header — never the LB's.

    `expiry`: `counting down` (`expires_in_s` > 0); `in use` (Go's zero time: not set yet, so the
    model is still loading); `unloading` (at or before the server's now); `unknown` (no `Date`, or no
    readable expiry). `offloaded` is broker_parrot's rule: `size_vram < size`. Raises ValueError for a
    body that is not JSON."""
    return _ollama_ps(json.loads(body or b"{}"), _http_date(date))


def ollama_ps_reply(reply: object) -> dict | None:
    """The same, for a BOX's own ollama as `tp-rank status` hands its reply over: `{"date": <the
    server's own Date header>, "ps": <the /api/ps body>}`.

    ONE reader of that route: the gateway's fallback ollama and a GB10 box's own are parsed by the same
    code and described in the same words, so "ollama holds qwen3.8:latest, 17.6 GiB" on the cluster
    panel means exactly what the throughput card means by it.

    None when the box could not be asked — never an empty list for that case: "it did not say" is not
    "nothing is resident", and the planner must be able to tell them apart."""
    if not isinstance(reply, dict) or not isinstance(reply.get("ps"), dict):
        return None
    date = reply.get("date")
    return _ollama_ps(reply["ps"], _http_date(date if isinstance(date, str) else None))


def _ollama_ps(doc: object, server_now: float | None) -> dict:
    models = []
    for entry in (doc.get("models") or []) if isinstance(doc, dict) else []:
        if not isinstance(entry, dict):
            continue
        expires = _rfc3339(entry.get("expires_at"))
        expires_at = expires.timestamp() if expires is not None else None
        if expires is None:
            expiry, left = "unknown", None
        elif expires.year <= _GO_ZERO_YEAR:
            expiry, left = "in use", None
        elif server_now is None:
            expiry, left = "unknown", None
        elif expires_at <= server_now:
            expiry, left = "unloading", None
        else:
            expiry, left = "counting down", expires_at - server_now
        size, vram = _nonnegative_int(entry.get("size")), _nonnegative_int(entry.get("size_vram"))
        models.append({"name": str(entry.get("name") or entry.get("model") or ""),
                       "context": _nonnegative_int(entry.get("context_length")),
                       "size_bytes": size, "gpu_bytes": vram,
                       "offloaded": vram < size if size is not None and vram is not None else None,
                       "expires_at": expires_at, "expires_in_s": left, "expiry": expiry})
    return {"server_now": server_now, "models": models}


def probe_backend(backend: Backend, cfg: Config, *, now: float, get: Getter = http_get) -> dict:
    fields = _probe(backend, cfg, now=now, get=get)
    fields.setdefault("mono_at", time.monotonic())   # no body was read: when the probe ended
    return fields


def _probe(backend: Backend, cfg: Config, *, now: float, get: Getter) -> dict:
    # `depth_known` is False until /metrics actually answers. Pass 13 promoted waiting/running
    # from a scaler hint to the PRIMARY routing key, and a silently-zeroed pair reads as the
    # shallowest backend alive — which a loaded box produces exactly when /metrics goes slow.
    # `depth_read_at`: WHEN running / waiting were read — the tick's wall clock `now` plus the monotonic time to the
    # /metrics body, since /health and /v1/models go first and take seconds under load. None until they are read: the
    # panel says "engine read 3 s ago" off it beside the gateway's own count, and a time it never read is no time.
    started = time.monotonic()
    fields = {"last_probe_at": now, "last_error": "", "live": False,
              "served_models": (), "waiting": 0, "running": 0, "depth_known": False, "depth_read_at": None}
    fields["scrape"] = None              # what /metrics said; None until it answers 200
    base = backend.url.rstrip("/")
    try:
        # The TP=2 cluster speaks the same vLLM HTTP API: its kind is routing, not a protocol.
        if backend.kind in (VLLM, CLUSTER):
            # READINESS FIRST. vLLM binds its API server — and answers /v1/models — while the
            # engine is still loading weights, and every inference request in that window is a
            # 500. `/health` is the distinction vLLM itself draws: 200 once it can serve, 503
            # while it cannot. Without this the router prefers a box that can only 500, and the
            # proxy relays that verbatim (failover is before-headers only, correctly).
            #
            # It BLOCKS only on vLLM's own "cannot serve" answers. Everything else — a redirect
            # (/health -> /health/, or http->https; we do not follow them), an auth-gated or
            # method-restricted route, a proxy hiccup on that route alone, no /health at all —
            # says nothing about the engine, so it is ADVISORY and /v1/models decides alone. A
            # gate that locks out a serving backend is worse than the 500s it prevents.
            # Short budget: this is a trivial endpoint, and it is now one of three sequential
            # probes running against `probe_interval_s`.
            health = None         # its body: what an engine says there that its /metrics needs (`with_health`)
            try:
                status, health = get(f"{base}/health", min(cfg.probe_timeout_s, HEALTH_TIMEOUT_S))
            except (OSError, http.client.HTTPException, ValueError):
                status = 0        # learned nothing about readiness; /v1/models will decide
            if status != 200:
                health = None
            if status in NOT_READY:
                fields["last_error"] = f"/health HTTP {status} — engine not ready to serve"
                return fields
            status, body = get(f"{base}/v1/models", cfg.probe_timeout_s)
            if status != 200:
                fields["last_error"] = f"/v1/models HTTP {status}"
                return fields
            data = json.loads(body or b"{}").get("data") or []
            ids = tuple(str(m["id"]) for m in data if isinstance(m, dict) and m.get("id"))
            if not ids:
                fields["last_error"] = "/v1/models lists no model"
                return fields
            windows = [int(m["max_model_len"]) for m in data
                       if isinstance(m, dict) and isinstance(m.get("max_model_len"), int)]
            # THE CHECKPOINT each served model was loaded from, as the engine names it (`root`): the folder it
            # really holds — what a node this LB did not launch can be identified by. None named, none kept.
            roots = tuple(str(m["root"]) for m in data
                          if isinstance(m, dict) and isinstance(m.get("root"), str) and m["root"].strip())
            fields.update(live=True, served_models=ids, model_roots=roots,
                          max_model_len=min(windows) if windows else None)
            try:
                status, text = get(f"{base}/metrics", cfg.probe_timeout_s)
                fields["mono_at"] = time.monotonic()      # the moment its body was READ
                if status == 200:
                    # Canonical keys whichever engine wrote the exposition: `parse_metrics` reads
                    # the vocabulary off the text itself, so an SGLang pair on the cluster backend
                    # publishes the same figures a vLLM one does.
                    scrape = with_health(parse_metrics(text.decode("utf-8", "replace")), health)
                    fields["scrape"] = scrape
                    # The Backend's counter fields, kept as aliases while the page migrates.
                    fields.update({k: scrape[k] for k in ("gen_tokens_total", "prompt_tokens_total")
                                   if scrape[k] is not None})
                    if scrape["waiting"] is None or scrape["running"] is None:
                        fields["last_error"] = "/metrics exposes no queue depth — depth unknown"
                    else:
                        fields.update(waiting=scrape["waiting"], running=scrape["running"],
                                      depth_known=True, depth_read_at=now + fields["mono_at"] - started)
                else:
                    fields["last_error"] = f"/metrics HTTP {status} — queue depth unknown"
            except (OSError, http.client.HTTPException, ValueError) as exc:
                # NOT "optional information" any more: unknown pressure must never masquerade
                # as an idle backend, so say so and let the router treat it as busy.
                fields["last_error"] = f"/metrics unreadable ({type(exc).__name__}) — depth unknown"
    except (OSError, http.client.HTTPException, ValueError) as exc:
        fields["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return fields
