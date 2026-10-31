"""Boot ORDER: the listener is up even when the engine cannot be imported.

`restart: unless-stopped` + an import above `proxy.start()` = a crash loop with :8010 dead
while the rest of the fleet is perfectly healthy. This runs the real entrypoint as a subprocess WITHOUT the
engine on PYTHONPATH (what a missing /opt/broker_parrot mount looks like) and expects
/healthz to answer and /lb/status to say why the broker side is down.
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time

from tests.fakes import free_port


def _get(port, path, timeout=2):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    r = conn.getresponse()
    return r.status, json.loads(r.read())


def test_listener_is_up_and_status_explains_when_the_engine_is_missing(tmp_path):
    from tests.fleet_files import write
    port = free_port()
    fleet = write(tmp_path, """
panel:
  style: vibe
boxes:
  box-a: {hostname: box-a-host, label: box-a-gpu, platform: linux/arm64, cards: [{index: 0, type: gb10, serves: true}]}
clusters: {}
backends:
  box-a: {url: "http://127.0.0.1:9"}
""")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "/app",                       # NO /opt/broker_parrot
        "LB_LISTEN": f"127.0.0.1:{port}",
        "LB_ALLOWED_HOSTS": "loopback",
        "LB_BROKER_ENABLED": "1",                   # the broker is ON — and its engine is missing
        "LB_FLEET": fleet,                          # one classic box, nothing listening on it
        "LB_PROBE_TIMEOUT_S": "0.5",
        "LB_TICK_S": "0.2",
        "LB_DB_BACKEND": "sqlite",
        "LB_DB_URL_ENV": "LB_SQLITE",
        "LB_SQLITE": "/tmp/boot-test.db",
    }
    proc = subprocess.Popen([sys.executable, "-m", "vllm_lb"], env=env, cwd="/app",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 8
        status = None
        while time.time() < deadline:
            try:
                status, body = _get(port, "/healthz")
                break
            except OSError:
                time.sleep(0.1)
        assert status == 200 and body == {"ok": True}, "listener never came up"
        time.sleep(0.6)                              # a couple of ticks
        st_status, st = _get(port, "/lb/status")
        assert st_status == 200
        assert st["db_ok"] is False and "queue_workflows" in st["db_error"]
        assert st["last_tick_error"] == "", "a missing engine is reported, not a crash"
        assert proc.poll() is None, "the process must still be running"
    finally:
        proc.terminate()
        out = proc.communicate(timeout=5)[0].decode(errors="replace")
    assert "listening on" in out
    assert "Traceback" not in out, out


# ── the fleet file (vllm_lb/fleet.py): the boot a redeploy with lb/fleet.yml in place goes through ──────────────

def test_an_lb_with_a_fleet_file_of_the_box_c_shape_comes_up_and_serves_its_topology(tmp_path):
    """The operator's condition for the migration: box-c's LB comes up cleanly at its next redeploy with its fleet
    file in place and the retired lines gone. Same shape (tests/fleet_files.py PAIR), the LB's own entrypoint."""
    from tests.fleet_files import PAIR, write
    port = free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "/app",
        "LB_LISTEN": f"127.0.0.1:{port}",
        "LB_ALLOWED_HOSTS": "loopback",
        "LB_FLEET": write(tmp_path, PAIR),
        "LB_HW_PORT": "0",                          # no UDP bind in the suite
        "LB_PROBE_TIMEOUT_S": "0.5",
        "LB_TICK_S": "0.2",
        "LB_CLUSTER_STATE": str(tmp_path / "cluster.json"),
        "LB_PERF": "0",
        "LB_SWARM_STATE": "",
    }
    proc = subprocess.Popen([sys.executable, "-m", "vllm_lb"], env=env, cwd="/app",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 8
        status = None
        while time.time() < deadline:
            try:
                status, body = _get(port, "/healthz")
                break
            except OSError:
                time.sleep(0.1)
        assert status == 200 and body == {"ok": True}, "listener never came up"
        st_status, st = _get(port, "/lb/status", timeout=5)
        assert st_status == 200
        assert st["cluster"]["configured"] is True and st["cluster"]["url"] == "http://192.0.2.10:8100"
        assert sorted((b["host_label"], b["box"]) for b in st["backends"] if b["kind"] == "vllm") == \
            [("node-1-gpu", "node-1"), ("node-2-gpu", "node-2")]
        assert st["hardware_state"]["hosts"] == {"node-1-host": "node-1", "node-2-host": "node-2",
                                                 "gateway-host": "gateway"}
        assert proc.poll() is None, "the process must still be running"
    finally:
        proc.terminate()
        out = proc.communicate(timeout=10)[0].decode(errors="replace")
    assert "Traceback" not in out, out


def test_a_retired_setting_refuses_the_boot_naming_where_its_value_went(tmp_path):
    from tests.fleet_files import ENV_OF_PAIR, PAIR, write
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": "/app", "LB_LISTEN": f"127.0.0.1:{free_port()}",
           "LB_FLEET": write(tmp_path, PAIR), "LB_VLLM_BOXES": ENV_OF_PAIR["LB_VLLM_BOXES"]}
    proc = subprocess.run([sys.executable, "-m", "vllm_lb"], env=env, cwd="/app", capture_output=True,
                          timeout=30)
    out = (proc.stdout + proc.stderr).decode(errors="replace")
    assert proc.returncode == 78, out                 # EX_CONFIG: docker ps shows "Restarting (78)"
    assert "LB_VLLM_BOXES" in out and "fleet" in out
    assert "Traceback" not in out, "a refusal is one line naming the setting, not a crash"
