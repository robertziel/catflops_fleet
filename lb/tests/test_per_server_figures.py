"""EACH SERVER'S OWN FIGURES (operator 2026-10-20, "Fix all": the pair's card lost memory bandwidth, compute, CUDA cores,
its throughput target and picker rates, and the performance / sessions pages went blank the moment gui ran beside it).

Before: saturation, the throughput target, the picker rows' rates, the sessions' current session and the perf log were
ONE engine's, built off the LB-wide config, and published EMPTY while several servers ran (`one_cluster_only`). Now each
is the server's own: `/lb/status.clusters[name]` carries its saturation, target and setup targets; every perf window
and boot line names the server that recorded it (`cluster`); `/lb/perf` and `/lb/sessions` take `?server=<name>`. A
line from before the name is the server's whose declared models alone include its model (router.cluster_models) — a
model two servers may run names neither, and the line stays in the unscoped log only."""
from __future__ import annotations

import json

from tests.fleet_files import PARALLEL
from tests.test_parallel_clusters import api  # noqa: F401 (fixture)
from tests.test_parallel_clusters_audit import _cfg
from tests.test_perflog import T0, Engine, cluster
from vllm_lb.perflog import BootRecorder, PerfRecorder, of_cluster


def test_each_server_in_the_run_carries_its_own_saturation_and_targets(tmp_path, api):
    _, call = api(_cfg(tmp_path, PARALLEL, perf_dir=str(tmp_path / "perf")))
    _, body = call("GET", "/lb/status")
    assert "one_cluster_only" not in body, "nothing is published empty for being one engine's any more"
    boxes = {d["name"]: d["boxes"] for d in body["declared"]}
    for name in ("pair", "gui"):
        view = body["clusters"][name]
        sat = view["saturation"]
        assert sat is not None, name
        assert sat["rates_from"] == boxes[name][0] and sat["gpu_count"] == len(boxes[name]), (name, sat)
        assert "throughput_target" in view and isinstance(view["setup_targets"], dict), name
    # The top-level figures stay the ONE server's alias: none of two servers is "the" one.
    assert body["saturation"] is None and body["throughput_target"] is None and body["setup_targets"] == {}


def test_the_recorder_keeps_each_servers_minutes_apart_and_names_them(tmp_path):
    rec = PerfRecorder(str(tmp_path), window_s=60)
    a, b = Engine(), Engine()
    a.running = b.running = 1
    pair, gui = {**cluster(), "name": "pair"}, {**cluster(gen=7), "name": "gui"}
    for t in (0, 6, 12):
        a.work(gen=100)
        b.work(gen=1)
        rec.observe(at=T0 + t, mono=T0 + t, live=True, scrape=a.scrape(), cluster=pair)
        rec.observe(at=T0 + t, mono=T0 + t, live=True, scrape=b.scrape(), cluster=gui)
    rec.close()
    rec._writer.drain()
    lines = [json.loads(x) for x in (tmp_path / "perf-2026-09-30.jsonl").read_text().splitlines()]
    by = {w["cluster"]: w for w in lines}
    assert set(by) == {"pair", "gui"} and len(lines) == 2
    assert by["pair"]["gen_tokens"] == 200 and by["gui"]["gen_tokens"] == 2, "never one engine's minus the other's"
    assert set(rec.status()["open"]) <= {"pair", "gui"}


def test_a_boot_line_names_its_server(tmp_path):
    rec = BootRecorder(str(tmp_path), cluster="gui")
    rec.observe(now=T0, state="starting", detail="", generation=1, config={"model_choice": "m"},
                config_source="launch", loading=None)
    closed = rec.observe(now=T0 + 9, state="running", detail="", generation=1, config={"model_choice": "m"},
                         config_source="launch", loading=None)
    assert closed["cluster"] == "gui"


def test_a_line_belongs_to_its_named_server_and_an_unnamed_one_to_the_one_server_that_may_run_its_model():
    owners = {"pair": frozenset({"big", "both"}), "gui": frozenset({"small", "both"})}
    lines = [{"cluster": "pair", "config": {"model_choice": "small"}},      # named: whatever its model
             {"config": {"model_choice": "big"}},                           # before the name: pair alone runs big
             {"config": {"model_choice": "both"}},                          # two may run it: neither's
             {"config": {"model_choice": "small"}},
             {"cluster": "gui", "config": None}]
    assert of_cluster(lines, "pair", owners) == [lines[0], lines[1]]
    assert of_cluster(lines, "gui", owners) == [lines[3], lines[4]]
    assert of_cluster(lines[1:4], "only", {"only": frozenset()}) == lines[1:4], "a fleet of one: every line is its own"


def _window(name, model, t, gen):
    return {"v": 1, "t_start": t, "t_end": t + 60, "cluster": name,
            "config": {"model_choice": model, "generation": 1 if name == "pair" else 2},
            "gen_tokens": gen, "gen_tok_s": gen / 60}


def test_perf_and_sessions_take_a_server(tmp_path, api):
    import time
    from vllm_lb.perflog import day_of
    perf = tmp_path / "perf"
    perf.mkdir()
    now = time.time() - 600
    (perf / f"perf-{day_of(now)}.jsonl").write_text("\n".join(json.dumps(w) for w in (
        _window("pair", "qwen3.8-27b-nvidia-nvfp4", now, 600),
        _window("gui", "qwen3.8-27b-nvidia-nvfp4", now + 60, 60))) + "\n")
    _, call = api(_cfg(tmp_path, PARALLEL, perf_dir=str(perf)))
    assert len(call("GET", "/lb/perf")[1]["windows"]) == 2, "unscoped: the whole log, as before"
    st, body = call("GET", "/lb/perf?server=gui")
    assert st == 200 and [w["cluster"] for w in body["windows"]] == ["gui"] and body["server"] == "gui"
    st, body = call("GET", "/lb/sessions?server=pair")
    assert st == 200 and body["server"] == "pair"
    assert sum(r["gen_tokens"] for r in body["sessions"]) == 600, body["sessions"]
    st, body = call("GET", "/lb/perf?server=nope")
    assert st == 404 and "pair" in body["error"]["message"] and "gui" in body["error"]["message"]
