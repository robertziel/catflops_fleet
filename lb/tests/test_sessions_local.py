"""A ONE-HOST NODE'S SESSIONS AND STARTS (operator, 2026-10-12: "Why this doesn't aggregate on site-b /sessions").

site-b's live /lb/sessions: `sessions: []`, `by_model: []`, `windows_without_a_session: 310`. Two gaps, one shape:
  * a session is ONE GENERATION (perflog.sessions), and every minute a one-host node recorded carried
    `generation: null` — topology.observed_effective names none, and the declaration laid over it (the applied
    intent's launch) added the model and the seats, never WHICH intent — so every minute was counted out;
  * the boot log was written along the pair's start path only (cluster_loop.tick → `_record_boot`): `_tick_local`
    recorded nothing and `build_cluster_loop` gave a one-host loop no recorder — so the chooser's start bars were
    empty on site-b for the same reason.

A node the LB requests (control `local`) is now recorded as the pair is: its generation is the RELEASED intent's id
(node_control.py — the intent its host's applier applies, (id, at) its identity); one boot line per up intent
released, opened `starting`, every state it then passes, closed `serving`, `failed` (in vLLM's root cause,
cluster.root_cause) or `stopped`; and its minutes carry the APPLIED intent's id, so they join that start's session.
An LB restarted over a node already serving writes no start (it saw none); a node it has seen serve that misses a
probe is READ AGAIN, never a start (perflog.read_again — the rule start_rates.py keeps for the pair, now one rule for
the start rates and the sessions both). A node the LB only observes (control `none`) has no intent: its minutes carry
the engine's own process start (`process_start_time_seconds`), and its session is one engine process — no start
invented. Minutes recorded before any of this carry neither, and are SAID, never guessed into a session.

Fixtures: site-b's own files read 2026-10-12 (tests/fixtures/site-b-2026-10-12) — the wish (cluster.json), the
released intent #14, the applier's report (its host renamed, its log tail dropped: the node's HTTP access log) and
eight of the 310 recorded windows, unedited.
"""
from __future__ import annotations

import http.client
import json
import pathlib

from tests.test_cluster_declared import declared_fleet
from tests.test_cluster_local import START_LOG, TWIN, Local
from tests.test_cluster_local import RAMMED as _RAMMED
from tests.test_cluster_one_host import DOWN, LIVE, NODE, STARTED, Clock, Node, scrape
from vllm_lb import node_control as nc
from vllm_lb.control import Controller
from vllm_lb.perflog import BOOT_PREFIX, PerfLog, PerfRecorder, sessions
from vllm_lb.proxy import Proxy
from vllm_lb.start_rates import start_rates

#: The node the operator STARTS here (each test writes its own wish): not run at boot, so no boot wish of the fleet's
#: is written before it (tests/test_engine_status.py: a down boot node is started once an applier listens).
RAMMED = _RAMMED.replace("    control: local\n", "    control: local\n    run_at_boot: false\n", 1)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "site-b-2026-10-12"
ORIGINAL = "qwen3.8-27b-nvidia-nvfp4"
#: site-b's released intent #14 (2026-10-11 23:26:53Z) and the applier's report of it, as read 2026-10-12.
INTENT = json.loads((FIXTURES / "intent.json").read_text())
REPORT = json.loads((FIXTURES / "report.json").read_text())
#: vLLM refusing site-b's 0.99 at start, 2026-10-11 23:21Z (cluster.root_cause's fixture).
REFUSED_LOG = (pathlib.Path(__file__).resolve().parent / "fixtures"
               / "vllm-2eb491e-start-refused-free-memory.log").read_text()


def busy(node: Local | Node, seconds: float, *, started: float = STARTED) -> None:
    """The node serving for `seconds`: a probe every 5 s, one request running, its tokens growing."""
    for _ in range(int(seconds // 5)):
        node.tokens = getattr(node, "tokens", 7895) + 60
        node.prober.table[NODE] = dict(LIVE, running=1, mono_at=node.clock.t,
                                       scrape=scrape(running=1, gen_tokens_total=node.tokens,
                                                     process_start_time=started))
        node.tick(5.0)


def logged(node) -> tuple[list[dict], list[dict]]:
    """What the LB wrote, read back as /lb/sessions reads it: the minutes, and every boot line, oldest first."""
    if node.recorder is not None:
        node.recorder.close()
        node.recorder._writer.drain()
    perf = PerfLog(node.perf_dir).index(days=None, now=node.clock.t)["windows"]
    boots = PerfLog(node.perf_dir, prefix=BOOT_PREFIX).index(days=None, now=node.clock.t)["windows"]
    return perf, boots


def started_by_the_lb(tmp_path, clock: Clock | None = None) -> Local:
    """site-b's intent #14 launch, released by THIS LB to a node that is down, applied, loaded and serving."""
    node = Local(tmp_path, DOWN, fleet=RAMMED, record=True, clock=clock)
    node.tick()
    nc.IntentStore(node.cfg.cluster_state_path).write("up", INTENT["launch"], now=node.clock.t)
    node.release_at = node.clock.t
    node.tick()                                                             # released: `starting`
    node.beat(applying={**node.released(), "since": node.clock.t})
    node.tick()                                                             # the applier is applying it
    node.beat(applied=node.applied(), container={"status": "running", "exit_code": 0, "restarts": 0},
              log=START_LOG)
    node.tick(30.0)                                                         # applied, loading
    node.serving_at = node.clock.t
    busy(node, 130)                                                         # answering, and busy
    return node


# ── a start the LB released ───────────────────────────────────────────────────────────────────


def test_a_released_intent_is_one_boot_line_and_the_minutes_it_serves_join_its_session(tmp_path):
    node = started_by_the_lb(tmp_path)
    perf, boots = logged(node)
    (line,) = boots
    assert (line["generation"], line["attempt"], line["outcome"]) == (1, 1, "serving"), line
    # Released → `waiting` for the host's applier (the page says `starting`), launched → `starting`, then serving.
    assert list(line["states"]) == ["waiting", "starting", "running"], line["states"]
    assert line["total_s"] == round(node.serving_at - node.release_at, 1), "asked → serving, on the LB's clock"
    # THE LAUNCH IT RELEASED, in the minutes' own keys — the node's two (its card's share, its RAM tier) included.
    config = line["config"]
    assert (config["model_choice"], config["uncensored"], config["generation"]) == (ORIGINAL, True, 1)
    assert (config["max_num_seqs"], config["max_model_len"], config["kv_cache_memory"]) == (16, 131072, 8_501_810_176)
    assert (config["gpu_memory_utilization"], config["kv_offload_ram_gib"]) == (0.95, 32)
    assert config["max_num_batched_tokens"] is None, "left to vLLM: nothing declared"
    assert line["phases"], "the start's progress, read off the node's log by the pair's own reader"
    # ITS MINUTES CARRY THE APPLIED INTENT'S ID, so they are this start's session.
    assert perf and {w["config"]["generation"] for w in perf} == {1}
    out = sessions(perf, boots)
    (s,) = out["sessions"]
    assert (s["generation"], s["model_choice"], s["outcome"], s["boot_attempts"]) == (1, ORIGINAL, "serving", 1)
    assert s["boot_s"] == line["total_s"] and s["windows"] == len(perf) and s["gen_tokens"] > 0
    assert out["windows_without_a_session"] == 0
    # …and THE CHOOSER ROW'S START BARS: one launch of the twin, served.
    row = start_rates(boots, now=node.clock.t)["rows"][TWIN]
    assert (row["ok"], row["n"]) == (1, 1)


def test_an_lb_restarted_over_its_serving_node_keeps_one_session_and_one_start(tmp_path):
    """The redeployed LB's first look may not have the node's answer yet: it says `starting` and opens a line — the
    one the pair writes after every redeploy (generation 177: sixteen of them). An earlier line of the same start
    makes it a READ AGAIN: not an attempt, not a boot time, not a start in the bars."""
    first = started_by_the_lb(tmp_path)
    logged(first)
    again = Local(tmp_path, DOWN, fleet=RAMMED, record=True, clock=first.clock)
    again.beat(applied=again.applied(), container={"status": "running", "exit_code": 0, "restarts": 0})
    again.tick()                                                            # not answered yet: `starting`
    busy(again, 130)
    perf, boots = logged(again)
    assert len(boots) == 2 and boots[1]["states"] == {"starting": 0.0, "running": 2.0}, boots[1]
    (s,) = sessions(perf, boots)["sessions"]
    assert (s["generation"], s["boot_attempts"], s["boot_s"]) == (1, 1, boots[0]["total_s"]), s
    assert s["windows"] == len(perf), "both LBs' minutes, one session"
    row = start_rates(boots, now=again.clock.t)["rows"][TWIN]
    assert (row["ok"], row["n"]) == (1, 1), "one start, not two"


def test_an_lb_deployed_over_site_bs_serving_node_joins_its_minutes_to_intent_14(tmp_path):
    """site-b as it stands: wish #14, released intent #14, applied 2026-10-11 23:27Z and serving since. The LB that
    boots over it saw no start, so it writes none — its minutes are intent #14's session all the same. And a probe it
    then misses (the node answered before) opens a line that SAYS it is a read again: no start, whatever the log
    held before."""
    clock = Clock(REPORT["heartbeat"])
    (tmp_path / "cluster.json").write_text((FIXTURES / "cluster.json").read_text())
    node = Local(tmp_path, LIVE, fleet=RAMMED, record=True, clock=clock, listening=False)
    (node.dir / nc.INTENT).write_text((FIXTURES / "intent.json").read_text())
    node.beat(applied=REPORT["applied"], container=REPORT["container"])
    view = node.tick()
    assert (view["state"], view["effective"]["generation"]) == ("running", 14), view["detail"]
    busy(node, 65)
    node.prober.table[NODE] = DOWN                                          # one probe missed
    assert node.tick()["state"] == "starting"
    busy(node, 65)
    perf, boots = logged(node)
    (line,) = boots
    assert line["read_again"] is True and line["outcome"] == "serving" and line["generation"] == 14, line
    assert {w["config"]["generation"] for w in perf} == {14}
    out = sessions(perf, boots)
    (s,) = out["sessions"]
    assert (s["generation"], s["model_choice"], s["config"]["uncensored"]) == (14, ORIGINAL, True)
    assert (s["boot_attempts"], s["boot_s"]) == (0, None), "no start was seen: none is claimed"
    assert start_rates(boots, now=clock.t)["rows"] == {}, "a read again is no start of the row"


# ── a start that failed, and one that was stopped ─────────────────────────────────────────────


def test_a_failed_intent_is_a_failed_start_in_vllms_root_cause(tmp_path):
    """site-b 2026-10-11 23:21Z replayed: GPU memory 0.99 released, applied (exit 0), and the container restarting —
    vLLM refused the fraction at start. The line closes `failed` in vLLM's own cause, and the twin's row counts it."""
    node = Local(tmp_path, DOWN, fleet=RAMMED, record=True)
    node.tick()
    nc.IntentStore(node.cfg.cluster_state_path).write("up", {**INTENT["launch"], "gpu_memory_utilization": 0.99},
                                                       now=node.clock.t)
    node.tick()
    node.beat(applied=node.applied(restarts_at_apply=0),
              container={"status": "running", "exit_code": 0, "restarts": 1}, log=REFUSED_LOG)
    assert node.tick()["state"] == "failed"
    perf, boots = logged(node)
    (line,) = boots
    assert line["outcome"] == "failed" and line["failed_in"] == "starting", line
    assert "ValueError: Free memory on device cuda:0 (30.85/31.36 GiB)" in line["failure"], line["failure"]
    assert line["config"]["gpu_memory_utilization"] == 0.99, "what was released, as released"
    row = start_rates(boots, now=node.clock.t)["rows"][TWIN]
    assert (row["ok"], row["n"]) == (0, 1)
    (s,) = sessions(perf, boots)["sessions"]
    assert s["outcome"] == "failed" and "Free memory on device" in s["failure"] and s["gen_tokens"] == 0


def test_a_start_stopped_while_it_loads_closes_stopped_in_the_plans_words(tmp_path):
    node = Local(tmp_path, DOWN, fleet=RAMMED, record=True)
    node.tick()
    nc.IntentStore(node.cfg.cluster_state_path).write("up", INTENT["launch"], now=node.clock.t)
    node.tick()
    node.beat(applying={**node.released(), "since": node.clock.t})
    node.tick()                                                             # launched: `starting`
    nc.IntentStore(node.cfg.cluster_state_path).write("down", INTENT["launch"], now=node.clock.t)
    node.tick()
    assert node.released()["state"] == "down"
    _, boots = logged(node)
    (line,) = boots
    assert (line["generation"], line["outcome"], list(line["states"])) == (1, "stopped",
                                                                           ["waiting", "starting", "unloading"]), line
    assert line["failure"] == "intent #2 (down) written; the applier picks it up", \
        "the plan's own words — never “replaced by a start” for a stop"
    assert start_rates(boots, now=node.clock.t)["rows"][TWIN]["stopped"] == 1, "launched, stopped: no verdict"


def test_an_intent_the_applier_refused_is_a_refusal_not_a_failed_launch(tmp_path):
    """Nothing ran: the applier's validator (the fleet's own) said no. Counted apart, as the pair's refusals are."""
    node = Local(tmp_path, DOWN, fleet=RAMMED, record=True)
    node.tick()
    nc.IntentStore(node.cfg.cluster_state_path).write("up", INTENT["launch"], now=node.clock.t)
    node.tick()
    node.beat(applied={**node.released(), "done_at": node.clock.t, "refused": "kv_offload_ram_gib 32 > 16"})
    assert node.tick()["state"] == "failed"
    _, boots = logged(node)
    (line,) = boots
    assert (line["outcome"], list(line["states"])) == ("failed", ["waiting", "failed"]), line
    assert "kv_offload_ram_gib 32 > 16" in line["failure"]
    assert start_rates(boots, now=node.clock.t)["rows"][TWIN] == {"ok": 0, "n": 0, "stopped": 0, "refused": 1,
                                                                   "days": [None] * 7}


# ── a node the LB only observes ───────────────────────────────────────────────────────────────


def test_an_observed_nodes_session_is_one_engine_process_by_its_own_start(tmp_path):
    """No intent, no Start: the LB started nothing, so it writes no boot line. What it has is the engine's own
    `process_start_time_seconds` — one process, one session; a restart, the next."""
    node = Node(LIVE, fleet=declared_fleet(), perf_dir=str(tmp_path))
    node.perf_dir, node.recorder = str(tmp_path), PerfRecorder(str(tmp_path))
    node.ctl = Controller(node.cfg, node.hub, None, prober=node.prober, clock=node.clock, cluster_loop=node.loop,
                          recorder=node.recorder)
    node.tick()
    busy(node, 130)
    busy(node, 130, started=STARTED + 3600.0)                               # its container restarted
    proxy = Proxy(node.cfg, node.hub)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("GET", "/lb/sessions")
        current = json.loads(conn.getresponse().read())["current"]
    finally:
        proxy.stop()
    perf, boots = logged(node)
    assert boots == [] and {w["engine_started_at"] for w in perf} == {STARTED, STARTED + 3600.0}
    out = sessions(perf, boots)
    newest, oldest = out["sessions"]
    assert (oldest["generation"], oldest["engine_started_at"], oldest["started_at"]) == (None, STARTED, STARTED)
    assert newest["engine_started_at"] == STARTED + 3600.0 and newest["model_choice"] == ORIGINAL
    assert out["windows_without_a_session"] == 0
    # THE SESSION RUNNING NOW is the engine process serving, by the same figure its minutes carry.
    assert (current["generation"], current["engine_started_at"], current["model_choice"]) == \
        (None, STARTED + 3600.0, ORIGINAL), current


# ── the minutes recorded before ───────────────────────────────────────────────────────────────


def test_site_bs_minutes_from_before_are_counted_and_said_never_guessed_into_a_session():
    """Eight of site-b's 310 minutes: none names an intent or the engine's start, and the applier's report keeps only
    the LAST intent applied (#14) — so nothing recorded says which start the other 117 served. They are counted as
    minutes from before sessions were recorded, and stay out of every session."""
    perf = [json.loads(line) for line in (FIXTURES / "perf-windows.sample.jsonl").read_text().splitlines()]
    assert len(perf) == 8 and {w["config"]["generation"] for w in perf} == {None}
    out = sessions(perf, [])
    assert out["sessions"] == [] and out["windows_without_a_session"] == 8
    assert out["windows_before_sessions"] == 8
    # one recorded from now on (it names its engine's start) joins a session; the eight still do not
    late = dict(perf[-1], engine_started_at=1_791_761_226.0)
    out = sessions(perf + [late], [])
    assert len(out["sessions"]) == 1 and out["windows_before_sessions"] == 8
