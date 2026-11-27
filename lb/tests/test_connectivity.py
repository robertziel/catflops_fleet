"""PER-BOX CONNECTIVITY CHECK (vllm_lb/connectivity.py): GET /lb/check runs the probes that already
exist, per box the fleet file names, ON DEMAND — never in the tick — and answers pass/fail per capability with a
one-line recovery hint. It reports; it fixes nothing.

Pinned here:
  * each fleet box gets its telemetry row (hw.py's sample age: none heard, stale, fresh);
  * each engine on a box gets the probe's own verdict (probe.py `probe_backend`): live, or its last_error;
  * each rank box of an ssh-controlled cluster gets an SSH reachability row (SshExecutor.reach: `true`, read-only,
    short timeout) — a box only observed over HTTP gets none;
  * a failing row carries a hint, a passing one none; the box is ok only when every row passed;
  * SshExecutor.reach runs `true` and nothing else: no tp-rank script on stdin, exit status is the verdict;
  * the route is the admin's; it is GET only.
"""
from __future__ import annotations

import http.client
import json
import os
import stat

from tests import fleet_files
from vllm_lb import roles
from vllm_lb.cluster import ClusterHost
from vllm_lb.cluster_exec import SshExecutor
from vllm_lb.config import Config
from vllm_lb.connectivity import check, plan

NOW = 1_795_176_000.0


def _cfg(tmp_path):
    return Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": fleet_files.write(tmp_path, fleet_files.PAIR),
                            "LB_CLUSTER_STATE": str(tmp_path / "state.json")})


class B:
    def __init__(self, url, box, kind="vllm", cluster=None):
        self.url, self.box, self.kind, self.cluster = url, box, kind, cluster


def test_plan_names_every_fleet_box_its_engines_and_its_ssh_targets(tmp_path):
    cfg = _cfg(tmp_path)
    backends = [B("http://192.0.2.10:8000", "node-1"), B("http://192.0.2.10:8100", "node-1", "cluster", "pair"),
                B("http://192.0.2.11:8000", "node-2")]
    plans = {p["box"]: p for p in plan(cfg, backends)}
    assert set(plans) == {"node-1", "node-2", "gateway"}
    assert [e.url for e in plans["node-1"]["engines"]] == ["http://192.0.2.10:8000", "http://192.0.2.10:8100"]
    assert [(s["via"], s["host"].ssh_target) for s in plans["node-2"]["ssh"]] == [("fleet · pair", "operator@192.0.2.11")]
    assert plans["gateway"]["engines"] == [] and plans["gateway"]["ssh"] == []
    assert "boxes.gateway.ssh" in plans["gateway"]["ssh_why"]
    assert plans["node-1"]["ssh_why"] is None


ONE_SSH_BOX_NO_CLUSTER = """
panel:
  style: vibe
boxes:
  solo:
    hostname: solo-host
    platform: linux/amd64
    ssh: operator@192.0.2.30
    cards:
      - {index: 0, type: rtx5090, serves: false}
  quiet:
    hostname: quiet-host
    platform: linux/amd64
    cards:
      - {index: 0, type: rtx5090, serves: false}
clusters: {}
backends: {}
"""


def test_a_box_with_an_ssh_address_outside_any_ssh_cluster_gets_its_row(tmp_path):
    """The power panel's boxes: an ssh address in the fleet file, no cluster driving it — still reachable or not."""
    cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback",
                           "LB_FLEET": fleet_files.write(tmp_path, ONE_SSH_BOX_NO_CLUSTER),
                           "LB_CLUSTER_STATE": str(tmp_path / "state.json")})
    plans = {p["box"]: p for p in plan(cfg, [])}
    assert [(s["via"], s["host"].ssh_target, s["host"].box) for s in plans["solo"]["ssh"]] == \
        [("fleet", "operator@192.0.2.30", "solo")]
    assert plans["quiet"]["ssh"] == [] and "boxes.quiet.ssh" in plans["quiet"]["ssh_why"]
    seen = []
    out = check(list(plans.values()), hardware={}, hw_state="listening", max_age_s=30, now=NOW,
                probe=lambda b: {}, reach=lambda host: seen.append(host.ssh_target) or (True, "ssh ok"))
    by = {b["box"]: b for b in out["boxes"]}
    assert seen == ["operator@192.0.2.30"]
    assert [r["cap"] for r in by["solo"]["checks"]] == ["telemetry", "ssh"]
    assert by["solo"]["checks"][1]["target"] == "operator@192.0.2.30 (fleet)"
    assert [r["cap"] for r in by["quiet"]["checks"]] == ["telemetry"], "no address: no SSH row"
    assert "boxes.quiet.ssh" in by["quiet"]["ssh_why"] and by["solo"]["ssh_why"] is None


def _plans():
    return [{"box": "a", "engines": [B("http://e1", "a")], "ssh": [{"via": "pair", "host": ClusterHost("a-gpu", "a", "u@a")}]},
            {"box": "b", "engines": [], "ssh": []},
            {"box": "c", "engines": [], "ssh": []}]


def test_every_capability_passes_or_fails_with_a_hint():
    hw = {"a": {"age_s": 1.0, "stale": False, "host": "a-host"}, "b": {"age_s": 300.0, "stale": True, "host": "b-host"}}
    out = check(_plans(), hardware=hw, hw_state="listening on udp/1", max_age_s=30, now=NOW,
                probe=lambda b: {"live": True, "served_models": ("m",), "last_error": ""},
                reach=lambda host: (False, "ssh exit 255: Permission denied (publickey)"))
    by = {b["box"]: b for b in out["boxes"]}
    rows = {r["cap"]: r for r in by["a"]["checks"]}
    assert rows["telemetry"]["ok"] is True and rows["telemetry"]["hint"] is None
    assert rows["engine"]["ok"] is True and rows["engine"]["target"] == "http://e1"
    assert rows["ssh"]["ok"] is False and "Permission denied" in rows["ssh"]["detail"]
    assert "authorized_keys" in rows["ssh"]["hint"]
    assert by["a"]["ok"] is False
    (tb,) = by["b"]["checks"]
    assert tb["ok"] is False and "300" in tb["detail"] and tb["hint"]
    (tc,) = by["c"]["checks"]
    assert tc["ok"] is False and "no sample" in tc["detail"] and "hostname" in tc["hint"]
    assert out["at"] == NOW and out["passed"] == 2 and out["failed"] == 3


def test_a_dead_engine_says_the_probes_words():
    out = check(_plans()[:1], hardware={"a": {"age_s": 1.0, "stale": False}}, hw_state="listening", max_age_s=30,
                now=NOW, probe=lambda b: {"live": False, "last_error": "/health HTTP 503 — engine not ready to serve"},
                reach=lambda host: (True, "ssh ok"))
    rows = {r["cap"]: r for r in out["boxes"][0]["checks"]}
    assert rows["engine"]["ok"] is False and "503" in rows["engine"]["detail"] and rows["engine"]["hint"]
    assert rows["ssh"]["ok"] is True


def test_without_a_listener_the_telemetry_row_says_so():
    out = check(_plans()[1:2], hardware={}, hw_state="off: LB_HW_PORT=0", max_age_s=30, now=NOW,
                probe=lambda b: {}, reach=lambda h: (True, ""))
    (row,) = out["boxes"][0]["checks"]
    assert row["ok"] is False and "LB_HW_PORT" in row["detail"]


def test_a_probe_that_raises_is_a_failed_row_not_a_500():
    def boom(b):
        raise RuntimeError("kaput")
    out = check(_plans()[:1], hardware={"a": {"age_s": 1.0, "stale": False}}, hw_state="listening", max_age_s=30,
                now=NOW, probe=boom, reach=lambda h: (True, ""))
    rows = {r["cap"]: r for r in out["boxes"][0]["checks"]}
    assert rows["engine"]["ok"] is False and "kaput" in rows["engine"]["detail"]


def _fake_ssh(tmp_path, body):
    path = tmp_path / "ssh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_reach_runs_true_and_nothing_else(tmp_path):
    log = tmp_path / "argv"
    ssh = _fake_ssh(tmp_path, f'echo "$@" > {log}\ncat > {log}.stdin\nexit 0\n')
    ex = SshExecutor(key_path="/k", known_hosts="/kh", script_path="/nonexistent", ssh_bin=ssh)
    ok, detail = ex.reach(ClusterHost("a-gpu", "a", "u@a"))
    assert ok is True
    argv = log.read_text().split()
    assert argv[-2:] == ["u@a", "true"] and "BatchMode=yes" in argv
    assert os.path.getsize(f"{log}.stdin") == 0, "no script is sent"
    assert ex.last_error == {}, "a check never writes the loop's error"


def test_reach_failure_names_the_exit_and_stderr(tmp_path):
    ssh = _fake_ssh(tmp_path, 'echo "Permission denied (publickey)." >&2\nexit 255\n')
    ok, detail = SshExecutor(key_path="/k", known_hosts="/kh", script_path="/x", ssh_bin=ssh).reach(
        ClusterHost("a-gpu", "a", "u@a"))
    assert ok is False and "255" in detail and "Permission denied" in detail


def test_reach_times_out(tmp_path):
    ssh = _fake_ssh(tmp_path, "exec sleep 5\n")
    ok, detail = SshExecutor(key_path="/k", known_hosts="/kh", script_path="/x", ssh_bin=ssh).reach(
        ClusterHost("a-gpu", "a", "u@a"), timeout_s=0.3)
    assert ok is False and "timed out" in detail


# ── the route ──────────────────────────────────────────────────────────────────────────────────

def test_the_route_answers_every_box_and_is_the_admins(tmp_path, monkeypatch):
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    dialled = []
    # The boxes carry ssh addresses (TEST-NET): the route reaches them through SshExecutor.reach, faked here — a test
    # dials no network.
    monkeypatch.setattr(SshExecutor, "reach", lambda self, host, timeout_s=8.0: dialled.append(host.ssh_target)
                        or (False, "ssh exit 255: fake"))
    cfg = Config(fleet=fleet_files.gb10_pair(), listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=30)
        c.request("GET", "/lb/check")
        r = c.getresponse()
        body = json.loads(r.read())
        assert r.status == 200
        assert {b["box"] for b in body["boxes"]} == {"box-a", "box-b"}
        assert sorted(dialled) == ["op@192.0.2.10", "op@192.0.2.11"], "every fleet ssh address, read-only"
        assert all([r["cap"] for r in b["checks"]].count("ssh") == 1 for b in body["boxes"])
        c.request("GET", "/lb/check?box=box-b")
        r = c.getresponse()
        body = json.loads(r.read())
        assert r.status == 200 and [b["box"] for b in body["boxes"]] == ["box-b"]
        c.request("GET", "/lb/check?box=nowhere")
        r = c.getresponse()
        body = json.loads(r.read())
        assert r.status == 404 and "box-a" in body["error"]["message"]
        c.request("GET", "/lb/check", headers={roles.ROLE_HEADER: roles.MEMBER, roles.MODELS_HEADER: "x"})
        r = c.getresponse()
        r.read()
        assert r.status == 403
    finally:
        p.stop()
