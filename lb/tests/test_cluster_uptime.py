"""HOW LONG THE PAIR HAS BEEN UP, on the card beside the engine's totals since it loaded.

Operator, 2026-10-03: the stats row ("Σ generated 137.9k · Σ prompt 10.9M · …") was "missing total time since cluster
started". The time is docker's StartedAt of the HEAD's rank container, which tp-rank's status already carries and the
loop already parses (`_started_at`) — on the box's clock, moved onto the LB's by the offset the box's status reports, so
`up_since` is an LB-clock epoch the page subtracts from its own `now`. It survives an LB restart (the container's start
does not move when the LB does), and it is null whenever no pair serves: a stopped or starting pair has no uptime.
"""
from __future__ import annotations

import calendar

from tests.test_cluster_loop import World, writes  # noqa: F401 — `writes` is a fixture
import pytest

STARTED = "2026-10-03T07:25:54.990330365Z"
STARTED_EPOCH = calendar.timegm((2026, 10, 3, 7, 25, 54)) + 0.990330365


@pytest.fixture
def world(broker_db, writes, tmp_path):  # noqa: F811 — `writes` is the fixture imported above
    return World(tmp_path, writes)


def test_a_serving_pair_says_when_its_head_started(world):
    world.executor.started_at = {"box-a": STARTED, "box-b": "2026-10-03T07:25:55.2Z"}
    world.run_to_running()
    v = world.view()
    assert v["state"] == "running"
    assert v["up_since"] == pytest.approx(STARTED_EPOCH, abs=0.01), "the HEAD's container start, not the worker's"


def test_the_uptime_survives_an_lb_restart(world):
    world.executor.started_at = {"box-a": STARTED, "box-b": "2026-10-03T07:25:55.2Z"}
    world.run_to_running()
    world.boot()                                   # a redeployed LB: fresh hub and loop, the same pair
    world.tick()
    v = world.view()
    assert v["state"] == "running" and v["up_since"] == pytest.approx(STARTED_EPOCH, abs=0.01)


def test_no_pair_serving_has_no_uptime(world):
    world.executor.started_at = {"box-a": STARTED, "box-b": STARTED}
    assert world.tick()["up_since"] is None, "off: nothing is up"
    world.store.set(enabled=True)
    assert world.tick()["up_since"] is None, "reserving/starting is not up"
