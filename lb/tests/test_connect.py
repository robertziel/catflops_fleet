"""THE CONNECT DIALOG's facts from the LB (operator, 2026-10-12: "connect button, top navbar … how to connect agent
to this vllm"). The page builds its snippets from the page's own origin, the LB's live /v1/models and ONE block of
/lb/status, `connect` — what only the deployment can say:

  tunnel  LB_CONNECT_TUNNEL, the LB's address ON ITS HOST (host:port, compose `ports:`): the dialog offers an SSH
          forward to it. Blank = not offered. A value that is not host:port refuses the start, naming it.
  basic   LB_CONNECT_BASIC, the edge in front takes HTTP Basic on the inference API (the edge's vhost,
          deploy/nginx-edge-vllm.conf `$vllm_machine_path`). 1/true/yes or 0/false/no/blank; anything else refuses.
  first_byte_timeout_s / stall_timeout_s   the LB's own budgets, which a client must be at least as patient as
          (fullclaudevllm's API_TIMEOUT_MS and CLAUDE_STREAM_IDLE_TIMEOUT_MS).

And the door rule the dialog reads: the LB itself answers /auth/session 404 (a gate in front answers it)."""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy


@pytest.mark.parametrize("raw, want", [("127.0.0.1:8010", "127.0.0.1:8010"), (" localhost:18010 ", "localhost:18010"),
                                       ("[::1]:8010", "[::1]:8010"), ("", ""), ("  ", "")])
def test_the_tunnel_is_the_lbs_host_and_port_or_nothing(raw, want):
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CONNECT_TUNNEL": raw}).connect_tunnel == want


def test_no_tunnel_is_declared_by_default():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).connect_tunnel == ""
    assert Config().connect_tunnel == ""


@pytest.mark.parametrize("raw", ["8010", "127.0.0.1:", ":8010", "127.0.0.1:abc", "127.0.0.1:0", "127.0.0.1:70000",
                                 "http://127.0.0.1:8010", "127.0.0.1:8010/v1", "a b:8010"])
def test_a_tunnel_that_is_not_host_and_port_refuses_the_start_by_name(raw):
    with pytest.raises(ValueError, match="LB_CONNECT_TUNNEL"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CONNECT_TUNNEL": raw})


@pytest.mark.parametrize("raw, want", [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False),
                                       ("no", False), ("", False)])
def test_basic_is_a_stated_yes_or_no(raw, want):
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CONNECT_BASIC": raw}).connect_basic is want


def test_basic_is_off_unless_stated():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).connect_basic is False


@pytest.mark.parametrize("raw", ["maybe", "2", "on?"])
def test_a_basic_that_is_neither_refuses_the_start_by_name(raw):
    with pytest.raises(ValueError, match="LB_CONNECT_BASIC"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_CONNECT_BASIC": raw})


def test_the_status_carries_the_connect_block():
    cfg = Config(listen_host="127.0.0.1", listen_port=0, connect_tunnel="127.0.0.1:8010", connect_basic=True,
                 first_byte_timeout_s=2400.0, stall_timeout_s=900.0)
    assert Proxy(cfg, Hub(cfg)).status()["connect"] == {
        "tunnel": "127.0.0.1:8010", "basic": True, "first_byte_timeout_s": 2400.0, "stall_timeout_s": 900.0}


def test_an_undeclared_deployment_says_so_in_the_block():
    cfg = Config(listen_host="127.0.0.1", listen_port=0)
    block = Proxy(cfg, Hub(cfg)).status()["connect"]
    assert block["tunnel"] == "" and block["basic"] is False
    assert block["first_byte_timeout_s"] == cfg.first_byte_timeout_s
    assert block["stall_timeout_s"] == cfg.stall_timeout_s


def test_the_lb_itself_answers_auth_session_404_which_the_dialog_reads_as_no_gate():
    cfg = Config(listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("GET", "/auth/session")
        r = c.getresponse()
        assert r.status == 404
        assert json.loads(r.read())["error"]["type"] == "not_found"
    finally:
        p.stop()
