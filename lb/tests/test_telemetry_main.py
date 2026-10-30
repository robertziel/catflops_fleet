"""fleet_telemetry as the sender runs it (`python -m fleet_telemetry`): its configuration, its preflight, the loop.

  * the configuration is the environment the compose files set (HW_SENDER_GPU from a gpu-*.yml, HW_SENDER_NETWORK,
    HW_SENDER_PORT from the server's .env) — an unset or unknown value refuses to start, naming it (sysexits 78);
  * the preflight: the GPU kind the server NAMED answers with at least one card, or nothing is sent (69) — an empty
    GPU row would look exactly like a box without a GPU;
  * the loop broadcasts one envelope per interval to the port it was given, and the gateway's intake takes it.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from fleet_telemetry.__main__ import EX_CONFIG, EX_UNAVAILABLE, Refused, Settings, preflight, run
from fleet_telemetry.broadcast import LanBroadcaster, discover_targets
from tests.test_telemetry_sample import Fixed, apu_sampler, collector

LB = Path(__file__).resolve().parents[1]
LIVE_PORT = 48291


# ── settings ──────────────────────────────────────────────────────────────────

GOOD = {"HW_SENDER_GPU": "amd", "HW_SENDER_NETWORK": "host", "HW_SENDER_PORT": "47999"}


def test_settings_from_the_environment():
    s = Settings.from_env(GOOD)
    assert (s.gpu_kind, s.network, s.port) == ("amd", "host", 47999)


@pytest.mark.parametrize("over,named", [
    ({"HW_SENDER_GPU": None}, "gpu-nvidia.yml"), ({"HW_SENDER_GPU": "intel"}, "intel"),
    ({"HW_SENDER_NETWORK": None}, "HW_SENDER_NETWORK"), ({"HW_SENDER_NETWORK": "macvlan"}, "macvlan"),
    ({"HW_SENDER_PORT": None}, "HW_SENDER_PORT"), ({"HW_SENDER_PORT": "0"}, "HW_SENDER_PORT"),
    ({"HW_SENDER_PORT": "udp"}, "HW_SENDER_PORT"),
])
def test_a_setting_unset_or_unknown_is_refused_by_name(over, named):
    env = {k: v for k, v in {**GOOD, **over}.items() if v is not None}
    with pytest.raises(Refused) as exc:
        Settings.from_env(env)
    assert exc.value.code == EX_CONFIG and named in str(exc.value)


def test_a_retired_engine_name_is_refused_naming_its_replacement():
    with pytest.raises(Refused, match="HW_SENDER_PORT"):
        Settings.from_env({**GOOD, "QUEUE_WORKFLOWS_HW_BROADCAST_PORT": "48291"})


# ── preflight ─────────────────────────────────────────────────────────────────

def test_preflight_passes_a_gpu_that_answers(tmp_path):
    assert "1 GPU" in preflight(apu_sampler(tmp_path), "amd")


@pytest.mark.parametrize("answer,said", [([], "no GPU"), (RuntimeError("couldn't communicate with the driver"),
                                                           "couldn't communicate")])
def test_preflight_refuses_a_named_gpu_that_does_not_answer(tmp_path, answer, said):
    with pytest.raises(Refused) as exc:
        preflight(apu_sampler(tmp_path, gpu=Fixed("nvidia-smi", answer)), "nvidia")
    assert exc.value.code == EX_UNAVAILABLE and said in str(exc.value) and "gpu-nvidia.yml" in str(exc.value)


def test_the_module_refuses_to_start_without_its_mounts():
    """Run as the image runs it, in this container: no cgroup tree at the image's mount point → 78, named."""
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": str(LB), "PYTHONDONTWRITEBYTECODE": "1",
           **GOOD, "HW_SENDER_GPU": "nvidia"}
    run_ = subprocess.run([sys.executable, "-m", "fleet_telemetry", "--once"], env=env, capture_output=True,
                          text=True, timeout=60)
    assert run_.returncode == EX_CONFIG, run_.stderr
    assert "REFUSED" in run_.stderr and "cgroup" in run_.stderr


# ── the loop over UDP ─────────────────────────────────────────────────────────

def test_the_loop_broadcasts_samples_the_gateway_takes(tmp_path):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("", 0))
    port = rx.getsockname()[1]
    assert port != LIVE_PORT
    rx.settimeout(10)
    caster = LanBroadcaster(port=port, targets=("127.0.0.1",))
    t = threading.Thread(target=run, args=(apu_sampler(tmp_path), caster), kwargs=dict(interval_s=0.01, samples=2,
                                                                                         host="box-c-host"))
    t.start()
    try:
        payloads = [rx.recvfrom(65535)[0], rx.recvfrom(65535)[0]]
    finally:
        t.join(10)
        rx.close()
    c = collector("box-c-host", ("radeon8060s",), "linux/amd64")
    for p in payloads:
        c.ingest(p)
    assert c.received == 2 and c.dropped == c.ignored == 0
    assert json.loads(payloads[0])["deep"]["gpu"][0]["temp_c"] == 48.0


def test_targets_are_the_default_routes_broadcast_and_the_limited_one():
    targets = discover_targets()
    assert targets[-1] == "255.255.255.255" and len(set(targets)) == len(targets)
