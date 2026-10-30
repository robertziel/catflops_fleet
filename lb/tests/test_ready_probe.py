"""A vLLM is "live" only when it can SERVE, not merely when it answers `/v1/models`.

vLLM's API server binds and answers `/v1/models` while the engine is still loading weights;
every inference request in that window returns 500. On 2026-09-12 that cost a live session ~35 s
of 500s from box-a while a healthy ollama fallback sat unused (worklog/vllm-lb-ready-probe.md).

`/health` is the distinction vLLM already draws: 200 once the engine can serve, 503 while it
cannot. A server that has no `/health` at all must still work off `/v1/models`, so that a
non-vLLM OpenAI-shaped backend is not locked out.
"""
from __future__ import annotations

import json

from vllm_lb.config import Config
from vllm_lb.state import VLLM, Backend

from vllm_lb.probe import probe_backend

MODELS = json.dumps({"data": [{"id": "qwen3.8:latest", "max_model_len": 262144}]}).encode()


def getter(routes: dict[str, tuple[int, bytes]]):
    """Serve exactly the routes given; anything else is a 404, like a real server."""
    seen = []

    def get(url: str, timeout: float) -> tuple[int, bytes]:
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        seen.append("/" + path)
        return routes.get("/" + path, (404, b"not found"))

    get.seen = seen
    return get


def vllm() -> Backend:
    return Backend(kind=VLLM, url="http://box-a:8000", host_label="box-a-gpu", box="box-a")


def test_a_vllm_that_is_ready_is_live():
    f = probe_backend(vllm(), Config(), now=1.0,
                      get=getter({"/health": (200, b""), "/v1/models": (200, MODELS)}))
    assert f["live"] is True
    assert f["served_models"] == ("qwen3.8:latest",)
    assert f["max_model_len"] == 262144


def test_a_vllm_still_loading_is_not_live_even_though_it_lists_models():
    """The exact 2026-09-12 shape: /v1/models answers, but the engine cannot serve yet."""
    f = probe_backend(vllm(), Config(), now=1.0,
                      get=getter({"/health": (503, b""), "/v1/models": (200, MODELS)}))
    assert f["live"] is False, "listing a model is not the same as being able to serve it"
    assert "health" in f["last_error"].lower() and "503" in f["last_error"], f["last_error"]


def test_a_backend_without_health_still_works_from_v1_models():
    """Not every OpenAI-shaped server is vLLM; a missing /health must not lock one out."""
    g = getter({"/v1/models": (200, MODELS)})          # /health -> 404
    f = probe_backend(vllm(), Config(), now=1.0, get=g)
    assert f["live"] is True
    assert "/health" in g.seen, "we should have asked, and then not held it against them"


def test_health_is_asked_before_models_so_a_loading_box_costs_one_request():
    g = getter({"/health": (503, b""), "/v1/models": (200, MODELS)})
    probe_backend(vllm(), Config(), now=1.0, get=g)
    assert g.seen == ["/health"], f"a loading box should stop at /health, got {g.seen}"


def test_only_vllms_own_not_ready_answers_block_the_backend():
    """503 is vLLM saying 'engine not ready'. 500 is it erroring. Those two block."""
    for status in (503, 500):
        f = probe_backend(vllm(), Config(), now=1.0,
                          get=getter({"/health": (status, b""), "/v1/models": (200, MODELS)}))
        assert f["live"] is False, f"HTTP {status} on /health should block"
        assert str(status) in f["last_error"]


def test_anything_that_is_not_a_usable_health_check_is_advisory():
    """The frame's KILL clause. A redirect (`/health` -> `/health/`, or http->https), an
    auth-gated proxy, a method-restricted route, a proxy hiccup on that route alone — none of
    these say the engine cannot serve, so /v1/models must decide alone. http_get does not
    follow redirects, so treating a 301 as 'not ready' locks out a serving backend entirely."""
    for status in (301, 302, 307, 308, 401, 403, 404, 405, 502):
        f = probe_backend(vllm(), Config(), now=1.0,
                          get=getter({"/health": (status, b""), "/v1/models": (200, MODELS)}))
        assert f["live"] is True, f"HTTP {status} on /health must not lock out a serving backend"
        assert f["served_models"] == ("qwen3.8:latest",)


def test_a_slow_health_check_does_not_kill_a_serving_backend():
    """A timeout on /health teaches us nothing about readiness, so it is advisory too."""
    def get(url: str, timeout: float):
        if url.endswith("/health"):
            raise TimeoutError("timed out")
        return (200, MODELS) if url.endswith("/v1/models") else (404, b"")

    f = probe_backend(vllm(), Config(), now=1.0, get=get)
    assert f["live"] is True, "a slow /health must not take down a backend that serves"


def test_the_health_probe_is_given_a_short_budget():
    """Three sequential probes at the full probe timeout (3s x 3 = 9s) overruns the 5s probe
    interval. /health is a trivial endpoint and gets a short slice."""
    seen: list[float] = []

    def get(url: str, timeout: float):
        seen.append(timeout)
        return (200, b"") if url.endswith("/health") else (200, MODELS)

    cfg = Config()
    probe_backend(vllm(), cfg, now=1.0, get=get)
    assert seen[0] < cfg.probe_timeout_s, f"/health got the full budget: {seen}"
