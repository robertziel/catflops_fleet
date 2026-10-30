"""The cluster boot log (perflog.BootRecorder): one line per cluster START — the configuration it launched,
the seconds to each state and loading phase, and how it ended — beside the perf windows, read back by the
same reader over the same `days` range, and served on /lb/perf as `boots`.

The ticks here are shaped like `ClusterLoop.tick` feeds the recorder: the planner's state and detail, the
generation, the configuration as the launch resolves it (perflog.CONFIG_KEYS) and `cluster.loading`."""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.perflog import (BOOT_PREFIX, BOOT_RECORD_VERSION, CONFIG_KEYS, BootRecorder, JsonlWriter,
                             PerfLog, day_of)
from vllm_lb.proxy import Proxy

T0 = 1_790_740_020.0          # 2026-09-30 05:47:00 UTC
CONFIG = {"model_choice": "qwen3.8-flash-next-hibrid48", "generation": 57, "max_num_seqs": 8,
          "max_num_batched_tokens": 8192, "num_speculative_tokens": 4, "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
          "max_model_len": 262144, "long_prefill_token_threshold": None, "uncensored": None,
          "kv_cache_dtype": None}


def tick(rec, at, state, detail="", gen=57, config=CONFIG, source="launch", loading=None):
    return rec.observe(now=T0 + at, state=state, detail=detail, generation=gen, config=config,
                       config_source=source, loading=loading)


def lines(directory):
    files = sorted(directory.glob(f"{BOOT_PREFIX}-*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def test_a_start_that_reaches_serving_is_one_line_with_its_configuration_and_every_step_timed(tmp_path):
    rec = BootRecorder(str(tmp_path))
    assert tick(rec, 0, "off", "cluster off") is None and rec.open is None, "rest opens nothing"
    assert tick(rec, 2, "reserving", "asking broker_parrot to reserve box-a and box-b",
                config=None, source="") is None
    tick(rec, 4, "reserving", config={**CONFIG, "max_num_batched_tokens": None}, source="resolved")
    tick(rec, 8, "freeing", "freeing RAM")
    tick(rec, 20, "starting", "both ranks running")
    tick(rec, 35, "starting", loading={"phase": "loading model weights", "passes": []})
    tick(rec, 50, "starting", loading={"phase": "loading model weights"})     # the same phase again: no entry
    tick(rec, 300, "starting", loading={"phase": "initial profiling/warmup run"})
    tick(rec, 420, "starting", loading={"phase": "API server up"})
    assert rec.status()["open"]["generation"] == 57 and rec.status()["written"] == 0
    closed = tick(rec, 431.2, "running", "TP=2 pair up (generation 57)")

    (line,) = lines(tmp_path)
    assert line == closed
    assert line["v"] == BOOT_RECORD_VERSION and line["kind"] == "boot"
    assert (line["t_start"], line["t_end"]) == (T0 + 2, T0 + 431.2), "asked at the first tick on its way up"
    assert line["generation"] == 57 and line["outcome"] == "serving" and line["total_s"] == 429.2
    # The configuration the perf windows are keyed by — the LAST one fed, the launch record's exact values.
    assert set(line["config"]) == set(CONFIG_KEYS) and line["config"] == CONFIG
    assert line["config_source"] == "launch"
    assert line["states"] == {"reserving": 0.0, "freeing": 6.0, "starting": 18.0, "running": 429.2}
    assert line["phases"] == [{"phase": "loading model weights", "at_s": 33.0},
                              {"phase": "initial profiling/warmup run", "at_s": 298.0},
                              {"phase": "API server up", "at_s": 418.0}]
    assert line["failure"] is None and line["failed_in"] is None and line["failed_phase"] is None
    assert rec.open is None and rec.status()["written"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"boots-{day_of(T0)}.jsonl"]


def test_a_failed_start_keeps_the_lbs_own_words_and_where_it_was(tmp_path):
    rec = BootRecorder(str(tmp_path))
    tick(rec, 0, "reserving")
    tick(rec, 15, "starting")
    tick(rec, 30, "starting", loading={"phase": "loading model weights"})
    tick(rec, 400, "starting", loading={"phase": "initial profiling/warmup run"})
    why = "rank on box-b exited (code 1); Stop, then Start again"
    closed = tick(rec, 512.4, "failed", why)
    tick(rec, 520, "failed", why)                            # still failed: nothing more is written
    tick(rec, 530, "off", "cluster off")

    (line,) = lines(tmp_path)
    assert line == closed and line["outcome"] == "failed" and line["failure"] == why
    assert line["failed_in"] == "starting" and line["failed_phase"] == "initial profiling/warmup run"
    assert line["states"] == {"reserving": 0.0, "starting": 15.0, "failed": 512.4} and line["total_s"] == 512.4
    assert line["config"] == CONFIG


def test_a_start_stopped_or_replaced_before_serving_is_recorded_as_stopped(tmp_path):
    rec = BootRecorder(str(tmp_path))
    tick(rec, 0, "reserving", gen=5)
    tick(rec, 10, "unknown", "cannot read broker_parrot's reservations", gen=5)   # skipped, not a step
    tick(rec, 20, "unloading", "removing the rank on box-a", gen=5)
    tick(rec, 60, "restarting", "replacing generation 5 on box-a", gen=6)
    tick(rec, 70, "reserving", gen=6)
    tick(rec, 80, "starting", gen=6)
    tick(rec, 90, "reserving", gen=7)                       # a new Start while 6 was still coming up
    tick(rec, 500, "running", gen=7)

    five, six, seven = lines(tmp_path)
    assert (five["generation"], five["outcome"], five["failure"]) == (5, "stopped", "removing the rank on box-a")
    assert five["states"] == {"reserving": 0.0, "unloading": 20.0}
    assert (six["generation"], six["outcome"]) == (6, "stopped") and "generation 7" in six["failure"]
    assert six["states"] == {"restarting": 0.0, "reserving": 10.0, "starting": 20.0} and six["failed_in"] == "starting"
    assert (seven["generation"], seven["outcome"], seven["total_s"]) == (7, "serving", 410.0)


def test_a_running_pair_seen_without_a_start_records_nothing(tmp_path):
    """An LB that boots under a serving pair, or ticks along beside one, has no start to time."""
    rec = BootRecorder(str(tmp_path))
    for at, state in ((0, "unknown"), (2, "running"), (4, "running"), (6, "failed"), (8, "off")):
        assert tick(rec, at, state) is None
    assert rec.open is None and not list(tmp_path.iterdir())


def test_a_surprise_shape_is_counted_never_raised(tmp_path):
    rec = BootRecorder(str(tmp_path))
    tick(rec, 0, "reserving", config="not a dict", loading="not a dict either")
    rec.observe(now="not a number", state="starting", detail="", generation=57, config=CONFIG,
                config_source="launch", loading=None)
    assert rec.internal_errors == 1 and rec.open is not None
    tick(rec, 100, "running")
    (line,) = lines(tmp_path)
    assert line["outcome"] == "serving" and line["config"] == CONFIG


def test_from_config_follows_the_perf_switch_and_directory(tmp_path):
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),), perf_dir=str(tmp_path))
    assert BootRecorder.from_config(cfg).dir == str(tmp_path)
    assert BootRecorder.from_config(Config(vllm_boxes=cfg.vllm_boxes, perf_enabled=False)) is None


def boot_at(t_start, total=430.0, gen=1):
    return {"v": 1, "kind": "boot", "t_start": t_start, "t_end": t_start + total, "generation": gen,
            "config": CONFIG, "config_source": "launch", "outcome": "serving", "total_s": total,
            "states": {"reserving": 0.0, "starting": 18.0, "running": total}, "phases": [],
            "failure": None, "failed_in": None, "failed_phase": None}


def test_the_reader_keeps_boots_that_ended_inside_the_range_and_never_mixes_the_two_logs(tmp_path):
    boots = JsonlWriter(str(tmp_path), prefix=BOOT_PREFIX, max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    day = 86400.0
    for t in (T0 - 3 * day, T0 - 1.5 * day, T0 - 3600.0):
        boots.write(boot_at(t))
    perf = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    perf.write({"t_start": T0 - 60, "t_end": T0, "config": CONFIG})
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"boots-{day_of(T0 - 3 * day)}.jsonl", f"boots-{day_of(T0 - 1.5 * day)}.jsonl",
         f"boots-{day_of(T0 - 3600)}.jsonl", f"perf-{day_of(T0 - 60)}.jsonl"])

    reader = PerfLog(str(tmp_path), prefix=BOOT_PREFIX)
    assert [b["t_start"] for b in reader.index(days=None, now=T0)["windows"]] == [T0 - 3 * day, T0 - 1.5 * day, T0 - 3600]
    assert [b["t_start"] for b in reader.index(days=2, now=T0)["windows"]] == [T0 - 1.5 * day, T0 - 3600]
    assert [b["t_start"] for b in reader.index(days=1, now=T0)["windows"]] == [T0 - 3600]
    assert [f["file"] for f in reader.index(days=1, now=T0)["files"]] == [f"boots-{day_of(T0 - 3600)}.jsonl"]
    # The perf reader over the same directory sees only its own files, and the other way round.
    assert [w["t_start"] for w in PerfLog(str(tmp_path)).index(days=None, now=T0)["windows"]] == [T0 - 60]


def test_a_boot_file_over_budget_prunes_only_boot_days(tmp_path):
    boots = JsonlWriter(str(tmp_path), prefix=BOOT_PREFIX, max_total_bytes=700, max_file_bytes=1 << 20)
    perf = JsonlWriter(str(tmp_path), max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    perf.write({"t_start": T0 - 5 * 86400, "t_end": T0 - 5 * 86400 + 60})
    for d in (3, 2, 1):
        boots.write(boot_at(T0 - d * 86400))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert f"perf-{day_of(T0 - 5 * 86400)}.jsonl" in names, "the perf log is not the boot log's to prune"
    assert boots.deleted >= 1 and f"boots-{day_of(T0 - 86400)}.jsonl" in names


@pytest.fixture
def served(tmp_path):
    started = []

    def make():
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0, bench_dir=str(tmp_path / "bench"),
                     perf_dir=str(tmp_path / "perf"))
        proxy = Proxy(cfg, Hub(cfg), perf_recorder=None)
        proxy.start()
        started.append(proxy)
        return proxy, tmp_path / "perf"

    yield make
    for p in started:
        p.stop()


def get(proxy, path):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_the_perf_route_carries_the_boots_of_the_same_range(served):
    proxy, perf = served()
    out = json.loads(get(proxy, "/lb/perf")[1])
    assert out["boots"] == [] and out["boots_malformed"] == 0
    rec = BootRecorder(str(perf))
    rec.observe(now=T0, state="reserving", detail="", generation=3, config=CONFIG, config_source="launch",
                loading=None)
    rec.observe(now=T0 + 400, state="running", detail="up", generation=3, config=CONFIG, config_source="launch",
                loading=None)
    (perf / f"boots-{day_of(T0)}.jsonl").open("a").write("{not json\n")
    status, body = get(proxy, "/lb/perf?days=all")
    out = json.loads(body)
    assert status == 200 and [b["generation"] for b in out["boots"]] == [3]
    assert out["boots"][0]["config"] == CONFIG and out["boots"][0]["outcome"] == "serving"
    assert out["boots_malformed"] == 1 and out["windows"] == []
