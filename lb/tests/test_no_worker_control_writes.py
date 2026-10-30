"""L1 — the vLLM service writes no `worker_controls` row, anywhere.

Operator, 2026-09-15: "We must not interfere with parked anywhere in vllm service." A park is the operator's
lever; the LB may READ one (to say why a box cannot claim) and never write one. This is the source proof;
tests/test_cluster_loop.py records the engine's cursor through start, run, an operator park/unpark, an LB
restart and stop, and finds no write there either.
"""
from __future__ import annotations

import re
from pathlib import Path

import vllm_lb

WRITES = re.compile(r"set_worker_control|(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+worker_controls", re.I)


def test_no_code_in_the_lb_writes_worker_controls():
    root = Path(vllm_lb.__file__).resolve().parent
    hits = [f"{p.relative_to(root)}:{n}: {line.strip()}" for p in sorted(root.rglob("*.py"))
            for n, line in enumerate(p.read_text().splitlines(), 1) if WRITES.search(line)]
    assert hits == []
