"""The /lb/stream patch (vllm_lb/status_delta.py): every case of the shared wire contract
(tests/fixtures/status_delta_cases.json) is diffed to exactly its patch and applied back to exactly its `b` — the
page's applier (static/status_delta.js) is held to the same patches by tests/status_delta.node.js. Then the real
thing: two statuses a probe apart round-trip, and the patch is a small part of the document."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from vllm_lb import status_delta

CASES = json.loads((pathlib.Path(__file__).parent / "fixtures" / "status_delta_cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_diff_is_the_contracts_patch(case):
    assert status_delta.diff(case["a"], case["b"]) == case["patch"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_patch_applied_is_the_next_document(case):
    a = copy.deepcopy(case["a"])
    assert status_delta.apply(case["patch"], a) == case["b"]
    assert a == case["a"], "apply must not change the document it was given"


def _status(now: float, samples: list[float]) -> dict:
    """A status shaped like the live one where it matters: a catalog that never changes, backends with full
    history rings, a queue whose rows change."""
    catalog = {f"model-{i}": {"launch": {"tp": 2, "args": ["--x"] * 20}, "kv": list(range(200))} for i in range(9)}
    backends = [{"url": f"http://192.0.2.{i}:8000", "live": True, "running": len(samples) % 7,
                 "history": {k: list(samples) for k in ("at", "gen_tok_s", "prompt_tok_s", "kv_pct", "depth")}}
                for i in range(5)]
    return {"now": now, "models": catalog, "backends": backends,
            "queue": {"inflight": [{"id": i, "elapsed_s": now - i} for i in range(5)]}}


def test_a_probe_apart_the_patch_is_a_small_part_of_the_document():
    ring = [float(i) for i in range(120)]
    a = _status(1000.0, ring)
    b = _status(1001.0, ring[1:] + [120.0])
    patch = status_delta.diff(a, b)
    assert status_delta.apply(patch, a) == b
    whole, small = len(json.dumps(b)), len(json.dumps(patch))
    assert small * 10 < whole, (small, whole)
    # every history ring went as a one-sample shift, never whole
    for item in patch["o"]["backends"]["i"].values():
        for node in item["o"]["history"]["o"].values():
            assert node == {"r": [1, [120.0]]}


def test_a_node_that_is_not_one_is_refused():
    with pytest.raises(ValueError):
        status_delta.apply({"?": 1}, {})
