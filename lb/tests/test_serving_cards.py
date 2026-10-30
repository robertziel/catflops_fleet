"""THE CARD A SERVER RUNS ON — asked in ONE place (vllm_lb/topology.py), answered by the fleet file.

Every peak, core count and per-card reading the panel shows for a server comes from the cards that SERVE it — the
fleet's `serves: true` cards of its ranks' boxes, head first — never from "the busiest card", which on a shared box
can be another tenant's (site-b: the 3090 at index 0 is someone else's; the node runs on the 5090 at index 1).
"""
from __future__ import annotations

import pytest

from tests.fleet_files import ONE_HOST, PAIR, write
from vllm_lb import topology
from vllm_lb.config import Config
from vllm_lb.hardware import profile

#: site-b's broadcast, as hw.py publishes it: the 3090 (card 0) BUSIER than the 5090 (card 1) the node runs on.
HARDWARE = {"solo": {"gpus": [{"id": 0, "use_pct": 96, "vram_used_mb": 20000, "vram_total_mb": 24576},
                              {"id": 1, "use_pct": 12, "vram_used_mb": 28058, "vram_total_mb": 32607}]}}


@pytest.fixture
def pair(tmp_path):
    return Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR)})


@pytest.fixture
def one_host(tmp_path):
    return Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, ONE_HOST)})


def test_the_pair_serves_on_one_gb10_a_box_head_first(pair):
    assert [(c.box, c.index, c.type.key) for c in topology.serving_cards(pair)] == \
        [("node-1", 0, "gb10"), ("node-2", 0, "gb10")]


def test_the_one_host_node_serves_on_its_5090_never_the_3090_beside_it(one_host):
    assert [(c.box, c.index, c.type.key) for c in topology.serving_cards(one_host)] == [("solo", 1, "rtx5090")]
    assert topology.serving_profile(one_host, "solo") is profile("rtx5090")


def test_the_serving_cards_reading_is_its_own_never_the_busiest(one_host):
    card = topology.serving_card(one_host, HARDWARE, "solo")
    assert card == HARDWARE["solo"]["gpus"][1], "card 1 at 12 %, not card 0 at 96 %"


def test_no_reading_for_the_serving_card_is_no_reading(one_host):
    assert topology.serving_card(one_host, {"solo": {"gpus": HARDWARE["solo"]["gpus"][:1]}}, "solo") is None
    assert topology.serving_card(one_host, {}, "solo") is None
    assert topology.serving_card(one_host, {"solo": {"gpus": None}}, "solo") is None, "a stale box"


def test_without_a_fleet_no_card_serves_and_nothing_is_claimed():
    cfg = Config()
    assert topology.serving_cards(cfg) == ()
    assert topology.serving_profile(cfg, "solo") is None
    assert topology.serving_card(cfg, HARDWARE, "solo") is None


def test_a_box_that_serves_nothing_has_no_serving_profile(pair):
    assert topology.serving_profile(pair, "gateway") is None
