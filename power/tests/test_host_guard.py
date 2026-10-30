"""THE HOST ALLOWLIST on the power panel (fleet_power/host_guard.py — the gateway's, byte for byte).

The panel binds 127.0.0.1, which a DNS-rebinding page reaches like any other loopback port. Its gate secret already
keeps such a page from acting (only the edge's nginx sends X-Power-Gate), but the panel must not answer to a name it is
not reached by either: the same rule as the gateway, from its own REQUIRED variable, POWER_ALLOWED_HOSTS.
"""
from __future__ import annotations

import pytest

from fleet_power.__main__ import settings_from_env
from fleet_power.host_guard import Authority
from tests.fakes import HOST, ORIGIN, audit_lines, get, make_app, post


def test_the_env_is_required_and_its_absence_refuses_the_start_by_name():
    with pytest.raises(ValueError, match="POWER_ALLOWED_HOSTS is required"):
        settings_from_env({})


def test_an_empty_value_is_refused_and_loopback_is_the_stated_choice():
    with pytest.raises(ValueError, match="POWER_ALLOWED_HOSTS is empty"):
        settings_from_env({"POWER_ALLOWED_HOSTS": ""})
    assert settings_from_env({"POWER_ALLOWED_HOSTS": "loopback"}).allowed_hosts == ()


def test_the_names_and_the_bind_come_from_the_environment():
    s = settings_from_env({"POWER_ALLOWED_HOSTS": "vllm.example.com", "POWER_BIND": "127.0.0.1:9016"})
    assert s.allowed_hosts == (Authority("vllm.example.com", None),) and s.bind == "127.0.0.1:9016"


@pytest.mark.parametrize("host", [HOST, f"{HOST}:443", "127.0.0.1:8016", "localhost:8016", "[::1]:8016"])
def test_the_proxy_name_and_the_listeners_loopback_names_are_served(tmp_path, host):
    status, _h, _out = get(make_app(tmp_path), "/power/api/state", Host=host)
    assert status == 200


@pytest.mark.parametrize("host", ["attacker.example", "127.0.0.1:8017", f"{HOST}:8443", None])
def test_any_other_host_is_misdirected_before_the_gate_is_asked(tmp_path, host):
    status, _h, out = get(make_app(tmp_path), "/power/api/state", Host=host)
    assert status == 421 and b"POWER_ALLOWED_HOSTS" in out


def test_a_misdirected_write_is_refused_and_audited(tmp_path):
    status, body = post(make_app(tmp_path), "/power/api/confirm", {"box": "box-a", "action": "poweroff"},
                        Host="attacker.example", Origin="http://attacker.example")
    assert status == 421 and body["error"]["type"] == "misdirected_host"
    assert [(r["event"], r["result"]) for r in audit_lines(tmp_path)] == [("confirm", "refused:misdirected_host")]


def test_a_write_from_a_foreign_origin_keeps_its_one_rule_power_origins(tmp_path):
    status, body = post(make_app(tmp_path), "/power/api/confirm", {"box": "box-a", "action": "poweroff"},
                        Origin="https://attacker.example")
    assert status == 403 and body["error"]["type"] == "origin"


def test_the_allowed_origin_still_reaches_the_route(tmp_path):
    status, body = post(make_app(tmp_path), "/power/api/confirm", {"box": "box-a", "action": "poweroff"},
                        Origin=ORIGIN)
    assert status == 200, body


def test_the_listener_rebinds_the_loopback_names_to_the_port_it_bound(tmp_path):
    app = make_app(tmp_path)
    app.listening_on("127.0.0.1", 18016)
    assert get(app, "/power/api/state", Host="127.0.0.1:18016")[0] == 200
    assert get(app, "/power/api/state", Host="127.0.0.1:8016")[0] == 421
