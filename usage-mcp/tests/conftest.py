"""Fixtures pointing at the fleet's REAL data.

The state directory is the LB's own `lb/state` (mounted read-only in the container run, or found
next to the repo on the host), so every assertion about perf windows, boot logs and swarm captures
is made against what the fleet actually recorded. The only copied files are the two `/lb/status` and
`/lb/bench` captures in `tests/data/`, because an HTTP document cannot be mounted - and those are
verbatim captures, not hand-written shapes (see tests/data/README.txt).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from usage_mcp.readonly import Sources

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Where the LB's state volume is: /data in the container, ../lb/state when the suite runs on a
#: checkout. The suite skips rather than invents if neither is there.
_CANDIDATES = (Path("/data"), HERE.parent.parent / "lb" / "state")
_CATALOGS = (Path("/library"), HERE.parent.parent / "lb" / "vllm_lb" / "library")


def _first(paths) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="session")
def state_dir() -> Path:
    found = _first(_CANDIDATES)
    if found is None:
        pytest.skip(f"no LB state directory at any of {[str(p) for p in _CANDIDATES]}")
    return found


@pytest.fixture(scope="session")
def catalog_path() -> Path:
    found = _first(_CATALOGS)
    if found is None:
        pytest.skip(f"no model catalog at any of {[str(p) for p in _CATALOGS]}")
    return found


@pytest.fixture
def src(state_dir: Path, catalog_path: Path) -> Sources:
    """Sources pointed at the real data, with the LB URL deliberately unroutable.

    Port 1 on loopback refuses instantly, so every test that must prove "the LB is down is an
    answer, not an error" gets that for free and no test accidentally depends on the live LB.
    """
    return Sources(lb_url="http://127.0.0.1:1", state_dir=state_dir,
                   catalog=catalog_path, timeout_s=1.0)


@pytest.fixture(scope="session")
def real_status() -> dict:
    """A verbatim `GET /lb/status` taken while the pair was serving generation 123."""
    return json.loads((DATA / "lb-status-2026-10-03.json").read_text())


@pytest.fixture(scope="session")
def real_bench() -> dict:
    """A verbatim `GET /lb/bench` from the same minute: zero saved runs."""
    return json.loads((DATA / "lb-bench-2026-10-03.json").read_text())


@pytest.fixture(scope="session")
def closed_perf_day(state_dir: Path) -> str:
    """A perf day-file that is finished appending, so exact totals are stable.

    The newest file is still being written by the live recorder; the one before it is not.
    """
    days = sorted(p.name.split("perf-")[1].removesuffix(".jsonl")
                  for p in (state_dir / "perf").glob("perf-*.jsonl"))
    if len(days) < 2:
        pytest.skip("need at least two recorded perf days to have a closed one")
    return days[-2]
