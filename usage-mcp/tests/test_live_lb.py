"""The real LB, when it is reachable - the one check that exercises the actual socket.

Skipped when the LB is not answering, which is the normal case in a hermetic run. When it IS up
(the container needs `--network host`, and `USAGE_MCP_LIVE_LB` names the URL), this proves the
GET path, the shaping and the tools against the live document rather than a capture.
"""
from __future__ import annotations

import dataclasses
import json
import os

import anyio
import pytest

from usage_mcp import live, server
from usage_mcp.readonly import Sources, SourceUnavailable, lb_get


@pytest.fixture(scope="session")
def live_src(request) -> Sources:
    url = os.environ.get("USAGE_MCP_LIVE_LB")
    if not url:
        pytest.skip("set USAGE_MCP_LIVE_LB to run against the live LB")
    src = Sources(lb_url=url, timeout_s=10.0,
                  state_dir=request.getfixturevalue("state_dir"),
                  catalog=request.getfixturevalue("catalog_path"))
    try:
        lb_get("/lb/status", src)
    except SourceUnavailable as exc:
        pytest.skip(f"the LB at {url} is not answering: {exc}")
    return src


def test_the_live_status_shapes(live_src: Sources):
    doc = live.status(live_src)
    shaped = live.shape_status(doc)
    assert isinstance(shaped["serving"], bool)
    assert shaped["cluster_state"]
    assert len(json.dumps(shaped)) < len(json.dumps(doc)) / 4
    if shaped["serving"]:
        assert shaped["model"]["served_name"] and shaped["model"]["engine"] in ("vllm", "sglang")


def test_the_live_perf_endpoint_is_readable_but_not_what_the_tools_use(live_src: Sources):
    """It answers; it is also 3.5 MB, which is why the tools read the jsonl instead."""
    doc = lb_get("/lb/perf", live_src)
    assert isinstance(doc.get("windows"), list)
    assert len(json.dumps(doc)) > 500_000, "expected the raw document to be large"


def test_every_tool_answers_against_the_live_lb(live_src: Sources):
    built = server.build(live_src)
    tools = anyio.run(built.list_tools)
    for tool in tools:
        result = anyio.run(lambda t=tool.name: built.call_tool(t, {}))
        assert result.is_error in (False, None), tool.name
        body = result.structured_content or json.loads(
            "".join(getattr(c, "text", "") for c in result.content))
        assert body.get("available") is True, (tool.name, body.get("why"))
        assert len(json.dumps(body, default=str)) < 120_000, tool.name


def test_a_control_path_is_still_refused_against_the_live_lb(live_src: Sources):
    """The LB really would act on POST /lb/cluster. Nothing here can reach it."""
    from usage_mcp.readonly import ReadOnlyViolation
    for path in ("/lb/cluster", "/lb/power"):
        with pytest.raises(ReadOnlyViolation):
            lb_get(path, live_src)


def test_the_live_bench_directory_matches_what_the_inventory_counted(live_src: Sources):
    from usage_mcp import inventory
    shaped = live.shape_bench(live.bench(live_src))
    counted = inventory.take(live_src)["sources"]["saved_benchmarks"]["files"]
    assert shaped["saved_runs"] == counted, (shaped["saved_runs"], counted)
