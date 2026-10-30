"""The MCP surface itself: the tools a host sees, and what each one hands back.

`test_a_real_mcp_client_can_handshake_and_call_every_tool` launches `python -m usage_mcp` as a
subprocess and drives it with the official SDK's own stdio client, so the protocol is exercised
rather than assumed - initialize, tools/list, tools/call, over the same transport an MCP host uses.
The LB is deliberately unreachable in these runs (`src` points at 127.0.0.1:1), which is also the
"cluster is off" case: no tool may raise.
"""
from __future__ import annotations

import json
import os
import sys

import anyio
import pytest

from usage_mcp import server
from usage_mcp.readonly import Sources

EXPECTED_TOOLS = {
    "fleet_status", "token_usage", "latency_profile", "cluster_starts",
    "model_configuration", "benchmark_runs", "swarm_sessions", "usage_data_inventory",
}


@pytest.fixture
def built(src: Sources):
    return server.build(src)


def _call(built, name, args=None):
    return anyio.run(lambda: built.call_tool(name, args or {}))


def _payload(result):
    if getattr(result, "structured_content", None):
        return result.structured_content
    text = "".join(getattr(b, "text", "") for b in (result.content or ()))
    return json.loads(text) if text.strip().startswith(("{", "[")) else {"text": text}


def test_the_server_exposes_exactly_the_eight_read_tools(built):
    tools = anyio.run(built.list_tools)
    assert {t.name for t in tools} == EXPECTED_TOOLS, sorted(t.name for t in tools)


def test_every_tool_is_annotated_read_only_and_closed_world(built):
    for tool in anyio.run(built.list_tools):
        ann = tool.annotations
        assert ann is not None, tool.name
        assert ann.read_only_hint is True, tool.name
        assert ann.destructive_hint is False, tool.name
        assert ann.open_world_hint is False, tool.name
        assert tool.description and len(tool.description) > 80, tool.name


def test_no_tool_name_is_an_imperative_that_changes_something(built):
    """`cluster_starts` is a noun - the starts that happened. A verb at the front would not be."""
    verbs = ("start", "stop", "set", "write", "park", "launch", "restart", "delete", "update",
             "post", "create", "kill", "power", "apply", "run", "enable", "disable", "reserve",
             "release", "cap")
    for tool in anyio.run(built.list_tools):
        head = tool.name.lower().split("_")[0]
        assert head not in verbs, tool.name


def test_the_instructions_warn_about_both_measured_caveats(built):
    text = built.instructions or ""
    assert "per TOKEN on SGLang" in text and "per speculative STEP" in text
    assert "No Anthropic" in text or "NO Anthropic" in text
    assert "swarm session's throughput is not its configuration's" in text


def test_every_tool_answers_with_the_lb_unreachable(built):
    """The cluster being off, or the LB down, is an answer - never a tool error."""
    for tool in anyio.run(built.list_tools):
        result = _call(built, tool.name)
        assert result.is_error in (False, None), (tool.name, result)
        got = _payload(result)
        assert "available" in got, (tool.name, sorted(got))
        if got["available"]:
            assert got.get("summary"), tool.name
        else:
            assert got.get("why"), tool.name


def test_the_disk_backed_tools_still_work_with_the_lb_down(built):
    for name in ("token_usage", "latency_profile", "cluster_starts", "swarm_sessions",
                 "usage_data_inventory", "model_configuration"):
        got = _payload(_call(built, name))
        assert got["available"] is True, (name, got.get("why"))


def test_fleet_status_reports_the_lb_being_down_without_pretending(built):
    got = _payload(_call(built, "fleet_status"))
    assert got["available"] is False
    assert "127.0.0.1:1" in got["note"]
    assert "only reads" in got["note"] or "historical" in got["note"]


def test_token_usage_refuses_an_impossible_prefix_hit_rate(built):
    """Real recorded windows carry hits above the queried total (lb defect, fixed in 9c32807)."""
    got = _payload(_call(built, "token_usage", {"days": 99, "group_by": "model"}))
    assert got["available"]
    rates = [got["totals"]["prefix_hit_rate"]] + [g["prefix_hit_rate"] for g in got["groups"]]
    assert all(r is None or 0.0 <= r <= 1.0 for r in rates), rates
    flagged = [g for g in got["groups"] if g.get("prefix_hit_rate_unusable")]
    for g in flagged:
        assert g["prefix_hit_rate"] is None
        assert "9c32807" in g["prefix_hit_rate_unusable"]
        assert g["prefix_hit_tokens"] >= g["prefix_query_tokens"]


def test_token_usage_totals_and_groups(built):
    got = _payload(_call(built, "token_usage", {"days": 99, "group_by": "model"}))
    assert got["available"] and got["totals"]["gen_tokens"] > 0
    assert got["groups"] and got["groups"][0]["gen_tokens"] >= got["groups"][-1]["gen_tokens"]
    assert sum(g["gen_tokens"] for g in got["groups"]) == got["totals"]["gen_tokens"]
    assert any("prefix cache" in c for c in got["caveats"])
    assert any("LOCAL" in c for c in got["caveats"])
    assert "," in got["summary"]


def test_token_usage_refuses_an_unknown_grouping(built):
    got = _payload(_call(built, "token_usage", {"group_by": "colour"}))
    assert got["available"] is False and "group_by" in got["why"]


def test_token_usage_for_a_day_with_nothing_recorded_is_not_an_error(built):
    got = _payload(_call(built, "token_usage", {"day": "1999-01-01"}))
    assert got["available"] is False and "recorded days" in got["why"]


def test_latency_profile_carries_the_measured_itl_semantics(built):
    got = _payload(_call(built, "latency_profile", {"days": 99}))
    assert got["available"]
    sem = got["overall"]["itl_semantics"]
    # the LB's own vocabulary: probe.py's ITL_PER_TOKEN / ITL_PER_STEP are "token" / "step"
    assert sem["means"] in ("token", "step"), sem
    assert sem["tokens_per_sample"] and sem["per_token_divisor"]
    assert "measured" in sem["note"]
    engines = {row["model_choice"]: row["itl_semantics"]["means"] for row in got["by_model"]
               if row["itl_semantics"]["tokens_per_sample"]}
    sglang = [v for k, v in engines.items() if "sglang" in k]
    vllm = [v for k, v in engines.items() if "sglang" not in k]
    assert sglang and all(v == "token" for v in sglang), engines
    assert vllm and all(v == "step" for v in vllm), engines
    assert got["caveats"], "a mixed selection must warn"
    assert any("itl_basis" in c for c in got["caveats"]), got["caveats"]


def test_cluster_starts_tallies_outcomes_and_keeps_the_failure_text(built):
    got = _payload(_call(built, "cluster_starts", {"days": 99, "limit": 100}))
    assert got["available"] and got["tally"]
    assert sum(got["tally"].values()) >= len(got["starts"])
    assert got["starts"][0]["at"] >= got["starts"][-1]["at"], "newest first"
    failed = [s for s in got["starts"] if s["outcome"] not in ("serving", None)]
    assert any(s.get("failure") for s in failed), "no failure text survived"
    assert all(s.get("model_choice") for s in got["starts"] if s["outcome"] == "serving")


def test_model_configuration_lists_then_details(built):
    listed = _payload(_call(built, "model_configuration"))
    assert listed["available"] and len(listed["models"]) >= 4
    pick = next(m["id"] for m in listed["models"] if m["engine"] == "sglang")
    one = _payload(_call(built, "model_configuration", {"model": pick}))
    assert one["available"] and one["recorded"]["id"] == pick
    assert one["recorded"]["engine"] == "sglang"
    assert one["recorded"]["memory"]["calibration"]
    assert any("calibrat" in c for c in one["caveats"])
    assert one["live"] is None, "the LB is down in this run"


def test_model_configuration_names_the_known_models_when_asked_for_a_stranger(built):
    got = _payload(_call(built, "model_configuration", {"model": "gpt-9"}))
    assert got["available"] is False and got["known"]


def test_swarm_sessions_measures_and_carries_both_caveats(built):
    got = _payload(_call(built, "swarm_sessions", {"limit": 50, "days": 99}))
    assert got["available"] and got["sessions"]
    assert any("2.36" in c for c in got["caveats"])
    assert any("FICTION" in c or "fiction" in c for c in got["caveats"])
    newest = got["sessions"][0]
    assert newest["label"].startswith("S-")
    assert "measured" in newest and "client_accounting" in newest


def test_usage_data_inventory_says_what_is_not_there(built):
    got = _payload(_call(built, "usage_data_inventory"))
    assert got["available"]
    assert got["meaning"].startswith("the LOCAL")
    assert got["sources"]["perf_windows"]["files"] >= 1
    assert got["sources"]["swarm_sessions"]["count"] >= 1
    assert got["live_endpoints"] == ["/lb/bench", "/lb/perf", "/lb/status"]
    assert "NO usable Anthropic-account usage" in got["anthropic_usage"]
    assert "wrong-backend" in got["anthropic_usage"], "the one exception must be named"
    assert ".credentials.json" in got["not_wired_in"]


def test_no_tool_result_is_enormous(built):
    """A tool that hands back a megabyte is the thing this server exists to avoid."""
    for tool in anyio.run(built.list_tools):
        blob = json.dumps(_payload(_call(built, tool.name)), default=str)
        assert len(blob) < 120_000, (tool.name, len(blob))


# ── the real wire ──────────────────────────────────────────────────────────────────────────────
def test_a_real_mcp_client_can_handshake_and_call_every_tool(src: Sources):
    """Launch the server as a subprocess and drive it with the SDK's own stdio client."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = dict(os.environ)
    env.update({
        "USAGE_MCP_LB_URL": src.lb_url,
        "USAGE_MCP_STATE_DIR": str(src.state_dir),
        "USAGE_MCP_LIBRARY": str(src.catalog),
        "USAGE_MCP_TIMEOUT_S": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    })
    params = StdioServerParameters(command=sys.executable, args=["-m", "usage_mcp"], env=env)

    async def drive():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                listed = await session.list_tools()
                calls = {}
                for tool in listed.tools:
                    calls[tool.name] = await session.call_tool(tool.name, {})
                return init, listed, calls

    init, listed, calls = anyio.run(drive)
    assert init.server_info.name == "fleet-usage"
    assert init.protocol_version, init
    assert init.capabilities.tools is not None
    assert "per TOKEN on SGLang" in (init.instructions or "")
    assert {t.name for t in listed.tools} == EXPECTED_TOOLS
    for name, result in calls.items():
        assert result.is_error in (False, None), (name, result)
        body = result.structured_content or {}
        if not body:
            body = json.loads("".join(getattr(c, "text", "") for c in result.content))
        assert "available" in body, (name, sorted(body))
    # the handshake also proves the tool annotations survive the wire
    assert all(t.annotations and t.annotations.read_only_hint for t in listed.tools)
