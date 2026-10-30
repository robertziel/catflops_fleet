"""broker_parrot is OPT-IN: `LB_BROKER_ENABLED` (1/true/yes) turns it on; absent, or anything else, is OFF.

Operator, 2026-10-09: most people who run this LB will not run broker_parrot. OFF is the operator's explicit
choice, not a fallback: the LB never imports `queue_workflows`, never reads or writes the broker DB, asks for
no lease, reservation or priority node_job and writes no worker_controls. A live configured backend is
routable without a lease, admission never waits for a claim, and a cluster Start launches on its memory and
availability gates alone. /lb/status says so — `"broker": {"enabled": false}` — and publishes no db_ok it
never measured.

Every test here runs WITHOUT broker_parrot mounted; the subprocess one puts /opt/broker_parrot on the path
when it is there, so "never imported" is proven with the engine importable, not merely absent.
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.fleet_files import one_box
from tests.fakes import FakeBackend, FakeClusterExecutor, FakeProber, free_port
from vllm_lb import router
from vllm_lb.catalog import Availability
from vllm_lb.cluster import (BoxObs, Desired, Down, Observed, ReleaseReservation, RankObs, Reserve, Up,
                             blockers, parse_hosts, plan)
from vllm_lb.cluster_defaults import desired_store
from vllm_lb.cluster_loop import ClusterLoop
from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import CLUSTER, VLLM

LB = Path(__file__).resolve().parents[1]
S1, S2, TP = "http://10.0.0.1:8000", "http://10.0.0.2:8000", "http://10.0.0.1:8100"
LIVE_V = {"live": True, "served_models": ("qwen3.8:latest",), "max_model_len": 262144, "waiting": 0, "running": 0,
          "last_error": ""}
LIVE_TP = {"live": True, "served_models": ("Qwen/Qwen3.8-Flash-Next",), "max_model_len": 262144,
           "waiting": 0, "running": 0, "last_error": ""}
DOWN = {"live": False, "last_error": "refused"}
BOXES = ("box-a", "box-b")
FLASH = "qwen3.8-flash-next-hibrid48"


class NoBroker:
    """A tripwire where the broker adapter would be: every attribute read is recorded. OFF must leave it empty."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def __getattr__(self, name):
        self.touched.append(name)
        raise AssertionError(f"the LB touched the broker with LB_BROKER_ENABLED off: {name}")


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ── the switch ────────────────────────────────────────────────────────────────


def test_the_broker_is_off_unless_the_environment_turns_it_on():
    assert Config().broker_enabled is False
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).broker_enabled is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "TRUE", "Yes", " 1 "])
def test_one_true_or_yes_turns_it_on(raw):
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_BROKER_ENABLED": raw}).broker_enabled is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "on", "2", "enabled"])
def test_anything_else_is_off(raw):
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_BROKER_ENABLED": raw}).broker_enabled is False


# ── routing: a live configured backend is ours to use ────────────────────────


def _off_cfg(**over) -> Config:
    return Config(broker_enabled=False, **over)


def _backend(kind, url, **kw):
    from vllm_lb.state import Backend
    b = Backend(kind=kind, url=url, **kw)
    b.live, b.served_models = True, ("qwen3.8:latest",)
    return b


def _snap(backends):
    from vllm_lb.state import Snapshot
    return Snapshot(now=1000.0, backends=list(backends), leases=[], last_request_at=None,
                    last_pressure_at=None, db_ok=True)


def test_a_live_vllm_is_routable_without_a_lease():
    v = _backend(VLLM, S1, host_label="box-a-gpu", box="box-a")
    assert router.choose(_snap([v]), _off_cfg()).url == S1
    assert router.choose(_snap([v]), Config(broker_enabled=True)) is None, "ON keeps the lease gate"


# ── the control loop asks the broker nothing ─────────────────────────────────


def _loop_world(tmp_path, *, cluster: bool):
    cfg = Config(broker_enabled=False,
                 vllm_boxes=(VllmBox("box-a-gpu", "box-a", S1), VllmBox("box-b-gpu", "box-b", S2)),
                 probe_interval_s=0, idle_release_s=600,
                 **({"cluster_name": "pair", "cluster_hosts": parse_hosts("box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,"
                                                   "box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"),
                     "cluster_url": TP, "cluster_state_path": str(tmp_path / "cluster.json"),
                     "cluster_observe_s": 0.0, "cluster_idle_observe_s": 60.0} if cluster else {}))
    clock = Clock()
    hub = Hub(cfg, clock=clock)
    broker = NoBroker()
    executor = FakeClusterExecutor()
    store = desired_store(cfg) if cluster else None
    loop = (ClusterLoop(cfg, hub, broker, executor=executor, store=store, clock=clock) if cluster else None)
    prober = FakeProber({S1: LIVE_V, S2: DOWN, TP: DOWN})
    engine_calls: list[int] = []

    def ensure_engine() -> str:
        engine_calls.append(1)
        return "the engine must not be set up with the broker off"

    ctl = Controller(cfg, hub, broker, prober=prober, clock=clock, ensure_engine=ensure_engine,
                     cluster_loop=loop)
    return cfg, clock, hub, broker, executor, store, prober, ctl, engine_calls


def test_a_tick_probes_and_asks_the_broker_nothing(tmp_path):
    cfg, clock, hub, broker, _, _, _, ctl, engine_calls = _loop_world(tmp_path, cluster=False)
    hub.note_request()                               # demand: ON would ask for a lease now
    for _ in range(12):
        ctl.tick()
        clock.t += 2.0
    snap = hub.snapshot()
    assert broker.touched == [] and engine_calls == []
    assert snap.leases == [] and snap.extra["last_tick_error"] == ""
    assert hub.backend_for(S1).live is True
    assert router.choose(snap, cfg).url == S1


def test_a_cluster_start_launches_on_its_gates_alone_and_reserves_nothing(tmp_path):
    cfg, clock, hub, broker, executor, store, prober, ctl, _ = _loop_world(tmp_path, cluster=True)
    store.set(enabled=True)

    def tick():
        ctl.tick()
        clock.t += 2.0
        return hub.snapshot().extra["cluster"]

    v = tick()
    states = [v["state"]]
    for _ in range(8):
        if v["state"] == "starting":
            break
        v = tick()
        states.append(v["state"])
    assert v["state"] == "starting", (states, v["detail"])
    assert "reserving" not in states and "waiting" not in states
    ups = [c for c in executor.calls if c[0] == "up"]
    assert [(c[1], c[2]) for c in ups] == [("box-a", 0), ("box-b", 1)]
    assert v["held"] is False
    assert all(b["reservation"] is None for b in v["boxes"].values())

    prober.table[TP] = LIVE_TP
    v = tick()
    assert v["state"] == "running" and v["held"] is True
    chosen = router.choose(hub.snapshot(), cfg)
    assert chosen is not None and (chosen.kind, chosen.url) == (CLUSTER, TP)

    store.set(enabled=False)
    for _ in range(4):
        v = tick()
    assert v["state"] == "off", v["detail"]
    assert [c[1] for c in executor.calls if c[0] == "down"] == ["box-a", "box-b"]
    assert not [c for c in executor.calls if c[0] == "heal"]
    assert broker.touched == [] and hub.snapshot().extra["last_tick_error"] == ""


def test_the_memory_gate_still_holds_a_start(tmp_path):
    _, clock, hub, broker, executor, store, _, ctl, _ = _loop_world(tmp_path, cluster=True)
    executor.mem = {"box-a": 30.0, "box-b": 30.0}
    store.set(enabled=True)
    for _ in range(8):
        ctl.tick()
        clock.t += 2.0
    v = hub.snapshot().extra["cluster"]
    assert v["state"] == "freeing" and "GiB free, need" in v["detail"]
    assert not [c for c in executor.calls if c[0] == "up"] and broker.touched == []


# ── the planner, pure ─────────────────────────────────────────────────────────


def _desired(mode="cluster") -> Desired:
    return Desired(mode=mode, generation=3, max_num_seqs=8, max_model_len=262144, model_choice=FLASH)


def _box(*, rank=None, mem=115.0, has=True, broker_vllm=False, needs=100.0) -> BoxObs:
    """`needs`: what the selection costs here (memory_estimate) — a box the LB cannot size is refused, not gated."""
    avail = {FLASH: Availability(name="Qwen3.8-Flash-Next-hibrid48", searched=("~/models",),
                                 dir="~/models" if has else None, image="img", image_present=has,
                                 observed_at=1.0, error="")}
    return BoxObs(reservation=None, rank=rank or RankObs(present=False, running=False, gen=None, exit_code=None),
                  broker_vllm_running=broker_vllm, mem_available_gib=mem, mem_stable_gib=mem, mem_peak_gib=mem,
                  availability=avail, needs_gib=needs)


def _plan(desired, boxes, **kw):
    return plan(desired, Observed(boxes=boxes, **kw), boxes=BOXES, labels=("box-a-gpu", "box-b-gpu"),
                broker=False)


def test_plan_launches_with_no_reservation_no_lane_and_no_fleet_jobs_read():
    p = _plan(_desired(), {b: _box() for b in BOXES})
    assert p.state == "starting"
    assert [type(a) for a in p.actions] == [Up, Up]
    assert not [a for a in p.actions if isinstance(a, (Reserve, ReleaseReservation))]


def test_plan_still_refuses_a_model_a_box_does_not_have():
    p = _plan(_desired(), {"box-a": _box(has=False), "box-b": _box()})
    assert p.state == "blocked" and not p.actions


def test_plan_holds_the_pair_once_both_ranks_run_and_the_head_answers():
    up = RankObs(present=True, running=True, gen=3, exit_code=None, model_choice=FLASH)
    p = _plan(_desired(), {b: _box(rank=up) for b in BOXES}, head_live=True, head_serves=True)
    assert p.state == "running" and p.held is True and not p.actions


def test_plan_stop_takes_the_ranks_down_and_releases_nothing():
    up = RankObs(present=True, running=True, gen=3, exit_code=None, model_choice=FLASH)
    p = _plan(_desired("classic"), {b: _box(rank=up) for b in BOXES})
    assert p.actions == [Down("box-a"), Down("box-b")] and p.state == "unloading"
    p = _plan(_desired("classic"), {b: _box() for b in BOXES})
    assert p.actions == [] and p.state == "off"
    assert "broker_parrot" not in p.detail and "reservation" not in p.detail


def test_a_running_classic_vllm_is_a_blocker_in_the_lbs_own_words():
    p = _plan(_desired(), {"box-a": _box(broker_vllm=True), "box-b": _box()})
    assert p.state == "freeing" and not p.actions
    assert "broker_parrot" not in p.detail and "stopping" not in p.detail
    words = [b["message"] for b in blockers(_box(broker_vllm=True), None, broker=False)]
    assert words and not any("broker_parrot" in w for w in words)


# ── the wire: /lb/status, a turn ─────────────────────────────────────────────


@pytest.fixture
def served():
    started: list[Proxy] = []

    def make(*, broker: bool, vllm: FakeBackend | None = None):
        cfg = Config(broker_enabled=broker,
                     vllm_boxes=(VllmBox("box-a-gpu", "box-a", vllm.url if vllm else "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0, admission_wait_s=30.0,
                     connect_timeout_s=2.0, first_byte_timeout_s=30.0)
        hub = Hub(cfg)
        if vllm is not None:
            b = hub.backend_for(vllm.url)
            b.live, b.served_models, b.max_model_len = True, tuple(vllm.served), vllm.max_model_len
        proxy = Proxy(cfg, hub)
        proxy.start()
        started.append(proxy)
        return cfg, hub, proxy

    yield make
    for p in started:
        p.stop()


def _call(proxy, method, path, body=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=timeout)
    data = None if body is None else json.dumps(body).encode()
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp, json.loads(resp.read() or b"{}")


def test_status_says_the_broker_is_off_and_claims_no_db_state(served):
    _, _, proxy = served(broker=False)
    _, st = _call(proxy, "GET", "/lb/status")
    assert st["broker"] == {"enabled": False}
    for key in ("db_ok", "db_error", "desired_leases", "vllm_capacity", "vllm_boxes_on", "context",
                "run_next", "gpu_slots"):
        assert key not in st, f"{key} is broker state; OFF has none to report"
    assert st["leases"] == []


def test_status_with_the_broker_on_still_carries_its_db_state(served):
    _, _, proxy = served(broker=True)
    _, st = _call(proxy, "GET", "/lb/status")
    assert st["broker"] == {"enabled": True}
    assert st["db_ok"] is True and "desired_leases" in st and "vllm_capacity" in st


def test_a_turn_reaches_a_live_vllm_with_no_lease(served):
    v = FakeBackend()
    v.reply_json = {"ok": "off"}
    try:
        _, _, proxy = served(broker=False, vllm=v)
        r, body = _call(proxy, "POST", "/v1/messages", {"model": "x", "max_tokens": 1})
        assert r.status == 200 and body == {"ok": "off"} and len(v.requests) == 1
    finally:
        v.down()


def test_nothing_live_is_refused_at_once_and_names_no_lease(served):
    _, _, proxy = served(broker=False)
    t0 = time.monotonic()
    r, body = _call(proxy, "POST", "/v1/messages", {"model": "x"})
    assert r.status == 503 and time.monotonic() - t0 < 5.0
    assert "broker_parrot" not in body["error"]["message"] and "lease" not in body["error"]["message"]


# ── the real entrypoint: boots, serves, never imports the engine ─────────────

_PROBE = r"""
import json, os, signal, sys, threading, time, urllib.request
port = int(os.environ["PORT"])

def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
        return json.loads(r.read())

def watch():
    out = {}
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                st = get("/lb/status")
                if st.get("last_tick_at") and any(b["live"] for b in st["backends"]):
                    break
            except OSError:
                pass
            time.sleep(0.1)
        time.sleep(0.6)                                   # a few more ticks
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", method="POST",
                                     data=json.dumps({"model": "x", "max_tokens": 1}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            out["status"] = r.status
            out["reply"] = json.loads(r.read())
        st = get("/lb/status")
        out["broker"] = st.get("broker")
        out["has_db_ok"] = "db_ok" in st
        out["last_tick_error"] = st.get("last_tick_error")
        out["engine_importable"] = __import__("importlib.util").util.find_spec("queue_workflows") is not None
        out["imported"] = sorted(m for m in sys.modules if m.split(".")[0] in ("queue_workflows", "psycopg"))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    print("PROBE " + json.dumps(out), flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

threading.Thread(target=watch, daemon=True).start()
from vllm_lb.__main__ import main
sys.exit(main())
"""


def test_the_entrypoint_boots_serves_and_never_imports_the_engine():
    v = FakeBackend()
    v.reply_json = {"served": "without a lease"}
    port = free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        # The engine on the path WHEN IT IS THERE: OFF must not import it even though it could.
        "PYTHONPATH": f"{LB}:/opt/broker_parrot",
        "PORT": str(port),
        "LB_LISTEN": f"127.0.0.1:{port}",
        "LB_ALLOWED_HOSTS": "loopback",
        "LB_FLEET": one_box(v.url),
        "LB_PROBE_INTERVAL_S": "0.2",
        "LB_PROBE_TIMEOUT_S": "1",
        "LB_TICK_S": "0.2",
        "LB_HW_PORT": "0",
        "LB_PERF": "0",
        "LB_SEARCH_URL": "",
        "LB_DRAIN_S": "1",
        "LB_SWARM_STATE": "",
        # LB_BROKER_ENABLED absent: OFF. And the broker's own settings present, to show they are not read.
        "LB_DB_BACKEND": "pg",
        "LB_BROKER_DB_URL": "postgresql://nobody:nothing@127.0.0.1:9/none",
    }
    try:
        proc = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(LB), capture_output=True,
                              text=True, timeout=40)
    finally:
        v.down()
    lines = [l for l in proc.stdout.splitlines() if l.startswith("PROBE ")]
    assert lines, proc.stdout + proc.stderr
    out = json.loads(lines[-1][len("PROBE "):])
    assert "error" not in out, out
    assert out["status"] == 200 and out["reply"] == {"served": "without a lease"}
    assert out["broker"] == {"enabled": False} and out["has_db_ok"] is False
    assert out["last_tick_error"] == ""
    assert out["imported"] == [], f"OFF imported the broker engine: {out['imported']}"
    assert "Traceback" not in proc.stdout + proc.stderr, proc.stdout + proc.stderr


# ── docker compose: `up` works with no broker_parrot on the host ─────────────


def _compose(name: str) -> dict:
    return yaml.safe_load((LB / name).read_text())


def test_the_base_compose_file_needs_nothing_of_the_broker():
    base = _compose("docker-compose.yml")
    text = (LB / "docker-compose.yml").read_text()
    lb = base["services"]["vllm-lb"]
    assert not any("broker_parrot" in str(v) for v in lb.get("volumes", []))
    assert not any(str(e).startswith(("LB_DB_BACKEND", "LB_DB_URL_ENV", "LB_BROKER_DB_URL"))
                   for e in lb.get("environment", []))
    assert "PG_PW:?" not in text, "a required broker credential makes `docker compose up` fail without it"
    assert not any((n or {}).get("external") for n in (base.get("networks") or {}).values()), \
        "an external network is one only the broker's project creates"
    for svc in base["services"].values():
        assert "broker" not in (svc.get("networks") or [])
    env = {str(e).split("=", 1)[0] for e in lb.get("environment", [])}
    assert "LB_BROKER_ENABLED" in env


def test_the_broker_override_carries_the_mount_the_db_and_the_network():
    over = _compose("docker-compose.broker.yml")
    lb = over["services"]["vllm-lb"]
    assert "${LB_BROKER_CHECKOUT:?set LB_BROKER_CHECKOUT in lb/.env}:/opt/broker_parrot:ro" in lb["volumes"], \
        "the checkout is the deployment's (lb/.env), never a path of one operator's"
    env = {str(e).split("=", 1)[0] for e in lb["environment"]}
    assert {"LB_DB_BACKEND", "LB_DB_URL_ENV", "LB_BROKER_DB_URL"} <= env
    url = next(str(e) for e in lb["environment"] if str(e).startswith("LB_BROKER_DB_URL="))
    for name in ("PG_USER", "PG_PW", "PG_DB"):
        assert f"${{{name}:?" in url, f"{name} has no default: no deployment's database name is compiled in"
    assert "broker" in lb["networks"] and "broker" in over["services"]["searxng"]["networks"]
    assert over["networks"]["broker"] == {"external": True, "name": "${LB_BROKER_NETWORK:?set LB_BROKER_NETWORK in lb/.env}"}


def test_the_env_example_documents_the_switch_and_the_override():
    text = (LB / ".env.example").read_text()
    assert "LB_BROKER_ENABLED" in text and "docker-compose.broker.yml" in text and "COMPOSE_FILE" in text
