"""/lb/status per backend: the `throughput` and `loaded` blocks. Null is never zero, a number is a
measurement, and every `why` comes from one closed vocabulary.

worklog/vllm-gateway-throughput.md, Design 3 and 6, edge rows 2, 17-22, 24, 39 and 41. Probe results
are fed to a real Hub exactly as probe_backend returns them (a `scrape` plus both stamps), and every
assertion reads the real `Proxy.status()` — the dict the page renders.
"""
from __future__ import annotations

import pytest

from vllm_lb.cluster import parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.probe import SCRAPE_FAMILIES
from vllm_lb.proxy import Proxy
from vllm_lb.rates import NOT_EXPOSED, WHY

BOX_A = "http://192.0.2.2:8000"
BOX_B = "http://192.0.2.12:8000"
HEAD = "http://192.0.2.2:8100"            # the TP=2 cluster's head: same box as BOX_A, its own port
HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
T0 = 1_789_500_000.0


def fleet_config(**over) -> Config:
    return Config(**{"vllm_boxes": (VllmBox("box-a-gpu", "box-a", BOX_A),
                                    VllmBox("box-b-gpu", "box-b", BOX_B)),
                     "cluster_name": "pair", "cluster_hosts": parse_hosts(HOSTS),
                     "cluster_url": HEAD, "listen_host": "127.0.0.1", "listen_port": 0, **over})


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Fleet:
    """A real Hub fed like the control loop feeds it, and the status the page would read."""

    def __init__(self, cfg: Config | None = None, t: float = T0) -> None:
        self.cfg = cfg or fleet_config()
        self.clock = Clock(t)
        self.hub = Hub(self.cfg, clock=self.clock)

    def probe(self, url: str, at: float, *, live: bool = True, read: bool = True, **families) -> None:
        scrape = dict.fromkeys(key for key, _, _ in SCRAPE_FAMILIES.values())
        scrape.update({"waiting": 0, "running": 0, **families})
        self.clock.t = at
        self.hub.update_backend(
            url, live=live, last_probe_at=at, mono_at=at, depth_known=live and read,
            waiting=scrape["waiting"] or 0, running=scrape["running"] or 0,
            served_models=("qwen3.8:latest",) if live else (),
            max_model_len=262144 if live else None, scrape=scrape if live and read else None)

    def status(self) -> dict[str, dict]:
        return {b["url"]: b for b in Proxy(self.cfg, self.hub).status()["backends"]}


def rated(backends: dict[str, dict]) -> dict[str, float]:
    """What a fleet sum may add: servers measured from metrics whose GENERATED is a number."""
    return {url: b["throughput"]["generated"]["tok_s"] for url, b in backends.items()
            if b["throughput"]["source"] == "metrics"
            and b["throughput"]["generated"]["tok_s"] is not None}


def whys(node) -> set[str]:
    """Every `why` value anywhere inside a status block."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "why" and value is not None:
                found.add(value)
            found |= whys(value)
    elif isinstance(node, list):
        for value in node:
            found |= whys(value)
    return found


def test_null_is_never_zero():
    """Row 2. A figure that was not measured is null WITH its reason — never 0, never a default."""
    f = Fleet()
    f.probe(BOX_A, T0, live=False)
    f.probe(HEAD, T0, gen_tokens_total=500)                  # one sample only
    f.probe(BOX_B, T0)                                      # live, but exposes no token counter
    backends = f.status()
    expected = {BOX_A: "server down", HEAD: "first sample", BOX_B: "not exposed by this server"}
    for url, why in expected.items():
        assert backends[url]["throughput"]["generated"] == {"tok_s": None, "window_s": None,
                                                            "why": why}, url
    for url, b in backends.items():
        for name in ("generated", "prompt", "prefill_speed"):
            figure = b["throughput"][name]
            assert (figure["tok_s"] is None) == (figure["why"] is not None), (url, name, figure)
        assert {"tokens_per_s", "gen_tokens_total", "prompt_tokens_total"} <= b.keys()
        assert "tok_s" in b["history"], "the old fields stay, as aliases, until the page migrates"
    assert rated(backends) == {}


def test_a_measured_zero_is_a_number():
    """Row 1 through the contract: an idle fresh vLLM's GENERATED is 0, and 0 is a measurement."""
    f = Fleet()
    f.probe(BOX_A, T0, gen_tokens_total=0)
    f.probe(BOX_A, T0 + 5, gen_tokens_total=0)
    generated = f.status()[BOX_A]["throughput"]["generated"]
    assert generated == {"tok_s": 0.0, "window_s": 5.0, "why": None}
    assert f.status()[BOX_A]["tokens_per_s"] == 0.0


def test_unleased_server_reports_fleet_work():
    """Rows 17-18. Fleet jobs call a vLLM directly, never through the LB: its metrics count them even
    on a server the LB does not hold, and none of those tokens is ours."""
    f = Fleet()
    for i, gen in enumerate((1000, 1150, 1300)):
        f.probe(BOX_A, T0 + 5 * i, running=2, gen_tokens_total=gen)
    throughput = f.status()[BOX_A]["throughput"]
    assert f.hub.leases() == []
    assert throughput["source"] == "metrics" and throughput["generated"]["tok_s"] == 30.0
    assert throughput["ours"]["counted"] == 0


def test_cluster_counted_once():
    """Row 19. The TP=2 cluster is ONE server: its head's metrics cover both ranks, the headless rank
    is never scraped, and neither GB10 box's classic card carries the cluster's tokens."""
    f = Fleet()
    for i, gen in enumerate((23656, 24346)):
        at = T0 + 10 * i
        f.probe(HEAD, at, running=1, gen_tokens_total=gen)
        f.probe(BOX_A, at, live=False)
        f.probe(BOX_B, at, live=False)
    backends = f.status()
    assert rated(backends) == {HEAD: pytest.approx(69.0)}
    assert list(backends).count(HEAD) == 1


def test_two_servers_on_one_box_both_count():
    """Row 20, seen live at 20:21Z: a classic sidecar and the cluster head up on the same GB10 box are two
    servers. Counted per URL, never per box — both."""
    f = Fleet()
    for i in range(2):
        f.probe(BOX_A, T0 + 5 * i, gen_tokens_total=1000 + 150 * i)
        f.probe(HEAD, T0 + 5 * i, gen_tokens_total=5000 + 345 * i)
    assert rated(f.status()) == {BOX_A: 30.0, HEAD: 69.0}


def test_no_spec_counters_is_null_not_zero():
    """Row 21. Speculative decoding off (the classic sidecar): the spec block is null, never 0 %. On the
    MTP head it says how many drafted tokens were accepted and how many tokens a step emits."""
    f = Fleet()
    f.probe(BOX_A, T0, gen_tokens_total=1000)
    f.probe(BOX_A, T0 + 5, gen_tokens_total=1100)
    f.probe(HEAD, T0, gen_tokens_total=23656, spec_drafts_total=7667,
            spec_draft_tokens_total=30668, spec_accepted_total=16042)
    f.probe(HEAD, T0 + 10, gen_tokens_total=24346, spec_drafts_total=7864,
            spec_draft_tokens_total=31456, spec_accepted_total=16536)
    backends = f.status()
    assert backends[BOX_A]["throughput"]["spec_decode"] is None
    spec = backends[HEAD]["throughput"]["spec_decode"]
    assert spec["acceptance_rate"] == pytest.approx(494 / 788)
    assert spec["mean_accepted_len"] == pytest.approx(1 + 494 / 197)
    assert spec["window_s"] == 10.0 and spec["why"] is None


def test_cached_share_and_prefill_speed():
    """Row 24. Prefix-cache hits inflate prompt tok/s, so PROMPT says what share was cached; prefill
    speed comes only from tokens the engine computed, over the time it spent computing them."""
    f = Fleet()
    f.probe(BOX_A, T0, gen_tokens_total=0, prompt_tokens_total=0, prompt_computed_total=0,
            prompt_cached_total=0, prefill_computed_tokens_sum=0.0, prefill_time_sum=0.0,
            prefill_requests_total=0)
    f.probe(BOX_A, T0 + 5, gen_tokens_total=10, prompt_tokens_total=1000, prompt_computed_total=500,
            prompt_cached_total=500, prefill_computed_tokens_sum=500.0, prefill_time_sum=0.5,
            prefill_requests_total=1)
    throughput = f.status()[BOX_A]["throughput"]
    assert throughput["prompt"] == {"tok_s": 200.0, "computed_tok_s": 100.0, "cached_share": 0.5,
                                    "window_s": 5.0, "credited": "at_first_token", "why": None,
                                    "computed_why": None}
    assert throughput["prefill_speed"] == {"tok_s": 1000.0, "requests": 1, "window_s": 5.0,
                                           "why": None}


def test_a_prompt_rate_with_no_computed_counter_says_why_prefill_is_blank():
    """TensorFold counts every prompt token (prefix-cache hits included) and no computed one: PROMPT has a figure and
    PREFILL — the computed part of it — has none, and says it is not exposed rather than drawing a bare dash."""
    f = Fleet()
    f.probe(BOX_A, T0, gen_tokens_total=0, prompt_tokens_total=0)
    f.probe(BOX_A, T0 + 5, gen_tokens_total=10, prompt_tokens_total=1000)
    prompt = f.status()[BOX_A]["throughput"]["prompt"]
    assert prompt["tok_s"] == 200.0 and prompt["computed_tok_s"] is None
    assert prompt["computed_why"] == NOT_EXPOSED


def test_prefill_speed_needs_a_finished_request():
    """No request finished inside the window: prefill speed is not 0 tok/s, it is not reached yet."""
    f = Fleet()
    for i in range(2):
        f.probe(BOX_A, T0 + 5 * i, gen_tokens_total=i, prompt_tokens_total=0,
                prefill_computed_tokens_sum=900.0, prefill_time_sum=1.0, prefill_requests_total=1)
    speed = f.status()[BOX_A]["throughput"]["prefill_speed"]
    assert speed == {"tok_s": None, "requests": 0, "window_s": 5.0, "why": "window not reached"}


def test_long_prefill_says_credited_at_first_token():
    """Row 22. Running, nothing generated, KV rising: GENERATED is a measured 0; PROMPT is no number
    ("credited at first token"); `prefilling` says since when, and from what KV share. The first
    token ends it."""
    f = Fleet()
    for i, kv in enumerate((0.12, 0.15, 0.18)):
        f.probe(BOX_A, T0 + 5 * i, running=1, gen_tokens_total=10, prompt_tokens_total=10,
                kv_cache_usage=kv)
    throughput = f.status()[BOX_A]["throughput"]
    assert throughput["generated"]["tok_s"] == 0.0
    assert throughput["prompt"]["tok_s"] is None
    assert throughput["prompt"]["why"] == "credited at first token"
    assert throughput["prefilling"] == {"since": T0 + 5, "kv_from": 0.12}
    f.probe(BOX_A, T0 + 15, running=1, gen_tokens_total=11, prompt_tokens_total=250_010,
            kv_cache_usage=0.31)
    throughput = f.status()[BOX_A]["throughput"]
    assert throughput["prefilling"] is None and throughput["prompt"]["tok_s"] is not None


def test_loaded_model_of_a_vllm():
    """The model line's facts for a vLLM: what /v1/models serves, its window, and KV usage from the
    same scrape; nothing about a server that is down."""
    f = Fleet()
    f.probe(BOX_A, T0, gen_tokens_total=0, kv_cache_usage=0.12)
    loaded = f.status()[BOX_A]["loaded"]
    assert loaded["state"] == "resident" and loaded["why"] is None
    (model,) = loaded["models"]
    assert (model["name"], model["context"], model["kv_usage"]) == ("qwen3.8:latest", 262144, 0.12)
    f.probe(BOX_A, T0 + 5, running=1, gen_tokens_total=40, kv_cache_usage=0.2)
    assert f.status()[BOX_A]["loaded"]["state"] == "serving"
    f.probe(BOX_A, T0 + 10, live=False)
    loaded = f.status()[BOX_A]["loaded"]
    assert (loaded["state"], loaded["models"], loaded["why"]) == ("unknown", [], "server down")


def test_fresh_hub_says_since():
    """Row 39. A restarted LB keeps nothing: every figure measures again, `ours` starts at 0, and the
    totals say since when they count."""
    f = Fleet(t=1_789_500_600.0)
    for url, b in f.status().items():
        throughput = b["throughput"]
        assert throughput["totals"] == {"generated": 0, "prompt": 0, "since": 1_789_500_600.0}, url
        assert throughput["ours"]["counted"] == 0 and throughput["last_request"] is None
        assert throughput["generated"]["tok_s"] is None
        assert all(ring == [] for ring in b["history"].values())


def test_totals_add_only_real_deltas():
    """Tokens seen generated since the LB started: the sum of deltas between consecutive reads of one
    process — never a counter's absolute value, never across a probe that read nothing or a restart."""
    f = Fleet()
    timeline = [(1000, True), (1150, True), (None, False), (5000, True), (5075, True), (50, True),
                (80, True)]
    for i, (gen, live) in enumerate(timeline):
        f.probe(BOX_A, T0 + 5 * i, live=live, gen_tokens_total=gen)
    assert f.status()[BOX_A]["throughput"]["totals"]["generated"] == 150 + 75 + 30


def test_duplicate_url_counted_once():
    """Row 41. The same URL configured twice is ONE server: one card, one set of meters."""
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", BOX_A),
                             VllmBox("box-a-gpu-again", "box-a", BOX_A)),
                 listen_host="127.0.0.1", listen_port=0)
    f = Fleet(cfg)
    f.probe(BOX_A, T0, gen_tokens_total=1000)
    f.probe(BOX_A, T0 + 5, gen_tokens_total=1150)
    backends = Proxy(cfg, f.hub).status()["backends"]
    assert [b["url"] for b in backends] == [BOX_A]
    assert backends[0]["throughput"]["generated"]["tok_s"] == 30.0


def test_why_is_a_closed_vocabulary():
    """Every `why` anywhere in `throughput` and `loaded`, across the states that produce one, is a word
    the page knows — and the states below reach most of the vocabulary."""
    f = Fleet()
    seen: set[str] = set()

    def look():
        for b in f.status().values():
            seen.update(whys(b["throughput"]) | whys(b["loaded"]))

    f.probe(BOX_A, T0, live=False)
    f.probe(BOX_B, T0)
    f.probe(HEAD, T0, running=1, gen_tokens_total=100, prompt_tokens_total=10, kv_cache_usage=0.1,
            prefill_computed_tokens_sum=0.0, prefill_time_sum=0.0, spec_drafts_total=1,
            spec_draft_tokens_total=4, spec_accepted_total=2)
    look()
    f.probe(HEAD, T0 + 5, running=1, gen_tokens_total=100, prompt_tokens_total=10,
            kv_cache_usage=0.2, prefill_computed_tokens_sum=0.0, prefill_time_sum=0.0,
            spec_drafts_total=1, spec_draft_tokens_total=4, spec_accepted_total=2)
    look()
    f.probe(HEAD, T0 + 10, read=False)
    look()
    f.probe(HEAD, T0 + 15, gen_tokens_total=5)
    f.probe(HEAD, T0 + 20, gen_tokens_total=1)
    look()
    f.probe(HEAD, T0 + 60, gen_tokens_total=9)
    look()
    assert seen <= WHY, seen - WHY
    assert {"server down", "first sample", "counter reset", "probe gap", "scrape failed",
            "not exposed by this server", "credited at first token", "window not reached"} <= seen
    # The ollama fallback's own words went with it (retired 2026-10-17): no server the LB routes to says them.
    assert not seen & {"no /metrics on ollama", "no request through the LB yet"}, seen


def test_the_page_gets_the_probe_interval():
    """The page buckets history by time; the bucket is the probe interval, and the server says it."""
    f = Fleet()
    assert Proxy(f.cfg, f.hub).status()["config"]["probe_interval_s"] == f.cfg.probe_interval_s
