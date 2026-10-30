"""W3 — the SearXNG the LB searches has to have BREADTH.

`lb/searxng/settings.yml` configures the sidecar that answers every `WebSearch` a client on
:8010 makes. On 2026-09-15 that instance had exactly three engines that could produce a
general result — brave, duckduckgo and google cse — and when all three were rate-limited or
timing out, every query came back empty (worklog/vllm-websearch-resilience.md).

So this file pins the SHAPE of that config rather than any one engine: several independent
general engines, none of which needs an API key, and no per-engine timeout that could outlive
the LB's own search deadline.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vllm_lb.config import Config

SETTINGS = Path(__file__).resolve().parents[1] / "searxng" / "settings.yml"

#: The fewest independent general engines worth calling breadth: with four, two being
#: suspended and one timing out still leaves a query answered.
MIN_GENERAL_ENGINES = 4

#: General-category engines in the pinned image (searxng/searxng@sha256:e084201aa6…,
#: 2026.9.11+61d660276) that need NO API key and are not `inactive`. Taken from the image's own
#: `/usr/local/searxng/searx/settings.yml`: an engine there carrying `api_key:` or
#: `inactive: true` (braveapi, marginalia, startpage, swisscows, deepl, jina, kagi, …) is
#: excluded, and so is anything not in the general category. Enabling an engine outside this
#: set is what this test exists to catch.
KEYLESS_GENERAL_ENGINES = frozenset({
    "360search", "ayo", "baidu", "bing", "brave", "crowdview", "duckduckgo", "duckduckgo web",
    "encyclosearch", "fastbot", "fireball", "fynd", "gabanza", "gmx", "google", "google cse",
    "mojeek", "mwmbl", "mymemory translated", "naver", "privacywall", "quark", "qwant",
    "reloado", "resulthunter", "searchch", "searchmysite", "seznam", "sogou", "tusksearch",
    "vuhuv", "wiby", "wikidata", "wikipedia", "yacy", "yahoo", "yandex", "yep", "zapmeta",
})

#: Engines that fill the infobox only unless `display_type` also asks for `list`, and the
#: special-purpose ones that answer their own query shape (a currency, a translation) — they
#: are enabled by default and contributed NOTHING to `results`, which is how an instance with
#: nine "enabled" general engines had three that could actually answer.
NOT_A_WEB_INDEX = frozenset({"currency", "dictzone", "lingva", "mymemory translated",
                             "wikidata"})


@pytest.fixture(scope="module")
def settings() -> dict:
    return yaml.safe_load(SETTINGS.read_text())


@pytest.fixture(scope="module")
def engines(settings) -> list[dict]:
    return [e for e in (settings.get("engines") or []) if isinstance(e, dict)]


def enabled_web_indexes(engines: list[dict]) -> list[dict]:
    """The engines this file turns ON that can put a row in `results`."""
    return [e for e in engines
            if e.get("disabled") is False and e["name"] not in NOT_A_WEB_INDEX]


def test_several_independent_general_engines_are_enabled(engines):
    """One rate-limited or blocked engine must not be able to zero a query."""
    names = [e["name"] for e in enabled_web_indexes(engines)]
    assert len(names) >= MIN_GENERAL_ENGINES, f"only {names} would answer a query"
    assert len(set(names)) == len(names), f"an engine is listed twice: {names}"


def test_no_engine_this_instance_enables_needs_an_api_key(engines):
    unknown = sorted({e["name"] for e in enabled_web_indexes(engines)} - KEYLESS_GENERAL_ENGINES)
    assert unknown == [], f"not known to be key-less general engines: {unknown}"


def test_no_key_is_configured_anywhere_in_the_file():
    """Belt and braces, and the reason this instance needs no secret at all: the file is in
    git, so a key here would be a key in git."""
    assert "api_key" not in SETTINGS.read_text()


def test_every_engine_timeout_stays_inside_the_lbs_search_deadline(settings, engines):
    """The LB's own deadline is `Config.search_timeout_s` (20 s; LB_SEARCH_TIMEOUT_S is unset in
    docker-compose.yml). SearXNG waits for the slowest SELECTED engine, capped by
    `outgoing.max_request_timeout` — so both have to be under it, or the LB gives up on a
    search that was still coming."""
    deadline = Config().search_timeout_s
    outgoing = settings["outgoing"]
    cap = outgoing["max_request_timeout"]
    assert outgoing["request_timeout"] <= cap < deadline
    for engine in engines:
        if "timeout" in engine:
            assert engine["timeout"] <= cap, engine["name"]


def test_the_json_api_the_lb_calls_stays_reachable(settings):
    """`format=json` is off by default and the limiter would 403 the LB, which is a bot by
    definition (worklog/vllm-lb-websearch.md §3)."""
    assert "json" in settings["search"]["formats"]
    assert settings["server"]["limiter"] is False
