"""No tool result may carry a credential - proved by salting the real documents, not by reading code.

The LB's status document carries no secret today (checked against the real lb/.env: its PG_PW does
not appear in `GET /lb/status`). This file makes that irrelevant. It takes the verbatim capture,
plants credential-shaped values at several depths and under several plausible key names, runs EVERY
tool over it, and asserts none of them come back. That can only pass if the shaping is an allowlist
of named fields - which is the point: a value the LB starts publishing tomorrow cannot leak just by
existing.
"""
from __future__ import annotations

import json

import pytest

from usage_mcp import catalog as catalog_mod
from usage_mcp import inventory, live, server, swarm
from usage_mcp.readonly import Sources

CANARIES = ("SUPERSECRET_PGPW", "sk-ant-CANARY", "hunter2", "CANARY_BEARER")


def _salt(doc: dict) -> dict:
    """Plant credentials where a careless shaping would pick them up."""
    salted = json.loads(json.dumps(doc))
    salted["config"]["APP_DB_URL"] = "postgresql://app:SUPERSECRET_PGPW@postgres:5432/db"
    salted["config"]["PG_PW"] = "SUPERSECRET_PGPW"
    salted["PG_PW"] = "SUPERSECRET_PGPW"
    salted["env"] = {"ANTHROPIC_API_KEY": "sk-ant-CANARY", "PG_PW": "SUPERSECRET_PGPW"}
    salted["cluster"]["url"] = "http://svc:hunter2@192.0.2.2:8000"
    salted["cluster"]["ssh_key"] = "sk-ant-CANARY"
    salted["cluster"]["effective"]["api_key"] = "sk-ant-CANARY"
    salted["cluster"]["loaded_model"]["token"] = "CANARY_BEARER"
    salted["backends"][0]["url"] = "http://admin:hunter2@192.0.2.2:8000"
    salted["backends"][0]["authorization"] = "CANARY_BEARER"
    salted["swarm"]["runner"]["session"]["api_key"] = "sk-ant-CANARY"
    salted["hardware"]["box-a"]["password"] = "SUPERSECRET_PGPW"
    for entry in salted["models"]["cluster"]:
        entry["hf_token"] = "CANARY_BEARER"
    return salted


def _assert_clean(blob: str, where: str) -> None:
    for canary in CANARIES:
        assert canary not in blob, f"{canary} leaked through {where}"


def test_the_live_shaping_drops_every_planted_credential(real_status: dict):
    _assert_clean(json.dumps(live.shape_status(_salt(real_status))), "shape_status")


def test_the_catalog_shaping_drops_planted_credentials(src: Sources, real_status: dict):
    cluster = _salt(real_status)["cluster"]
    _assert_clean(json.dumps(catalog_mod.shape_live(cluster)), "shape_live")
    doc = catalog_mod.load(src)
    poisoned = json.loads(json.dumps(doc))
    for entry in poisoned["models"]:
        entry["hf_token"] = "CANARY_BEARER"
        entry.setdefault("launch", {}).setdefault("cluster", {})["api_key"] = "sk-ant-CANARY"
        entry.setdefault("memory", {})["password"] = "SUPERSECRET_PGPW"
    for entry in poisoned["models"]:
        _assert_clean(json.dumps(catalog_mod.shape_one(entry)), "shape_one")


def test_the_bench_shaping_drops_planted_credentials(real_bench: dict):
    doc = dict(real_bench)
    doc["runs"] = [{"name": "r", "api_key": "sk-ant-CANARY", "env": {"PG_PW": "SUPERSECRET_PGPW"}}]
    doc["credentials"] = "CANARY_BEARER"
    _assert_clean(json.dumps(live.shape_bench(doc)), "shape_bench")


def test_every_tool_result_over_the_real_sources_is_clean(src: Sources, real_status: dict,
                                                          real_bench: dict, monkeypatch):
    """Run all eight tools, with the LB answering the SALTED document, and scan every result."""
    salted = _salt(real_status)

    def fake_get(path: str, s: Sources):
        if path == "/lb/status":
            return salted
        if path == "/lb/bench":
            return real_bench
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr("usage_mcp.readonly.lb_get", fake_get)
    monkeypatch.setattr("usage_mcp.live.lb_get", fake_get)

    built = server.build(src)
    import anyio
    tools = anyio.run(built.list_tools)
    assert len(tools) == 8, [t.name for t in tools]
    for tool in tools:
        result = anyio.run(lambda t=tool.name: built.call_tool(t, {}))
        _assert_clean(json.dumps(result, default=str), f"tool {tool.name}")


def test_the_real_lb_env_file_is_unreachable_and_its_value_absent(src: Sources, state_dir,
                                                                  real_status: dict):
    """The actual PG_PW, read here only to prove it is nowhere in what a tool returns."""
    env = state_dir.parent / ".env"
    if not env.exists():
        pytest.skip("no lb/.env on this host")
    secret = next((ln.split("=", 1)[1].strip() for ln in env.read_text().splitlines()
                   if ln.startswith("PG_PW=")), None)
    if not secret:
        pytest.skip("lb/.env has no PG_PW")
    assert len(secret) >= 6
    assert secret not in json.dumps(real_status), "the LB's own status leaks PG_PW"
    assert secret not in json.dumps(live.shape_status(real_status))
    assert secret not in json.dumps(inventory.take(src))
    assert secret not in json.dumps(swarm.sessions(src), default=str)
