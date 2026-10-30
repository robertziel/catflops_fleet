"""The box inventory comes from POWER_BOXES, and fails CLOSED when it is absent.

vllm-power is published, and its allowlist of machines it may power off or wake is operator hardware:
addresses, Wake-on-LAN MACs, NIC names. None of it ships in code any more. An unset inventory is an
EMPTY allowlist (every box is a 404), and a malformed one is a boot-time error that names the variable.
"""
from __future__ import annotations

import json

import pytest

from fleet_power.boxes import Box, Wol, load_boxes

# Documentation addresses (RFC 5737) and a locally-administered MAC: test data, not anyone's machine.
ALPHA = {"name": "alpha", "ssh_host": "192.0.2.2", "in_cluster": True, "gpu_clock": True,
         "wol": {"mac": "02:00:00:00:00:02", "iface": "eth-a", "broadcast": "192.0.2.3"},
         "power_on_note": "Wake-on-LAN from the host.", "consequence": "The pair goes down."}
HOST = {"name": "host", "ssh_host": "127.0.0.1", "power_on_note": "Power it on by its plug."}


def env_with(*boxes) -> dict[str, str]:
    return {"POWER_BOXES": json.dumps(list(boxes))}


def test_no_inventory_means_no_box_may_be_acted_on():
    assert load_boxes({}) == {}
    assert load_boxes({"POWER_BOXES": "   "}) == {}


def test_the_inventory_is_read_from_the_environment():
    boxes = load_boxes(env_with(ALPHA, HOST))
    assert boxes["alpha"] == Box(
        "alpha", "192.0.2.2", in_cluster=True, gpu_clock=True,
        wol=Wol("02:00:00:00:00:02", "eth-a", "192.0.2.3"),
        power_on_note="Wake-on-LAN from the host.", consequence="The pair goes down.")
    host = boxes["host"]
    assert host.wol is None and host.ssh_port == 22 and host.ssh_user == "fleet-power"
    assert host.in_cluster is False and host.gpu_clock is False


@pytest.mark.parametrize("bad", [
    "not json",
    "{}",
    "[1]",
    json.dumps([{"name": "x"}]),
    json.dumps([{"name": "x", "ssh_host": "h", "unknown_field": 1}]),
    json.dumps([ALPHA, ALPHA]),
    json.dumps([{"name": "x", "ssh_host": "h", "wol": {"mac": "m"}}]),
])
def test_a_malformed_inventory_fails_loudly_and_names_the_variable(bad):
    with pytest.raises(ValueError, match="POWER_BOXES"):
        load_boxes({"POWER_BOXES": bad})


def test_an_error_never_echoes_a_value():
    secret_mac = "02:00:00:00:00:99"
    bad = json.dumps([{"name": "x", "ssh_host": "h", "wol": {"mac": secret_mac}}])
    with pytest.raises(ValueError) as e:
        load_boxes({"POWER_BOXES": bad})
    assert secret_mac not in str(e.value)


def test_a_default_gpu_clock_is_one_of_the_caps_or_absent():
    boxes = load_boxes(env_with({**ALPHA, "default_gpu_clock_mhz": 2300}, HOST))
    assert boxes["alpha"].default_gpu_clock_mhz == 2300
    assert boxes["host"].default_gpu_clock_mhz is None
    assert load_boxes(env_with(ALPHA))["alpha"].default_gpu_clock_mhz is None


@pytest.mark.parametrize("box", [
    {**ALPHA, "default_gpu_clock_mhz": 2250},          # no key on the box sets this cap
    {**ALPHA, "default_gpu_clock_mhz": 3003},
    {**ALPHA, "default_gpu_clock_mhz": "2200"},
    {**ALPHA, "default_gpu_clock_mhz": 2200.0},
    {**ALPHA, "default_gpu_clock_mhz": True},
    {**HOST, "default_gpu_clock_mhz": 2200},           # no GPU clock this service may set
])
def test_an_unknown_default_gpu_clock_is_refused_at_boot(box):
    with pytest.raises(ValueError, match="POWER_BOXES .*default_gpu_clock_mhz"):
        load_boxes(env_with(box))
