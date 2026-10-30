"""THE START SUCCESS RATE per model chooser row (vllm_lb/start_rates.py): of the starts the LB LAUNCHED for a
checkpoint over the last 7 UTC days, how many reached serving — per day and over the window — read off the cluster
boot log (`perflog.BootRecorder`, `boots-YYYY-MM-DD.jsonl`).

The operator's question is "does this model fail to start often". So a launch is one attempt (the operator's Start,
and each relaunch the LB makes on its own — a hung start torn down and launched again, a pair whose box rebooted
under it), it SUCCEEDS when it reached serving and FAILS when it ended failed after the launch; a start stopped
before it served has no verdict (`stopped`), one the LB refused before launching anything is not an attempt
(`refused`), and a pair that was already up and is merely read again — an LB redeploy, a /health probe missed by
a serving head — is no start at all. A day without an attempt has no point (null): never 0 %, never filled in.

The real log is pinned too: box-c's 2026-10-05..11 lines (tests/fixtures/boots-2026-10-05-to-10-11.jsonl, box
names and node_job ids sanitized) with generations 183–193 — the vLLM Flash-Next memory-guard kills and the one
hung start whose relaunch served."""
from __future__ import annotations

import http.client
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.perflog import BOOT_PREFIX, JsonlWriter, PerfLog, day_of
from vllm_lb.proxy import Proxy
from vllm_lb.start_rates import WINDOW_DAYS, start_rates

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOW = datetime(2026, 10, 11, 20, 30, tzinfo=timezone.utc).timestamp()
DAY = 86400.0
TODAY = NOW - 3600.0                                  # 19:30 UTC, well inside the day

#: The states a start passes, as BootRecorder stamps them (seconds since it was asked).
SERVED = {"reserving": 0.0, "starting": 6.2, "running": 588.6}
FAILED = {"reserving": 0.0, "starting": 2.2, "failed": 66.5}
HUNG = {"reserving": 0.0, "starting": 2.2, "restarting": 306.0}
RELAUNCH_SERVED = {"freeing": 0.0, "starting": 12.6, "running": 130.4}
READ_AGAIN = {"starting": 0.0, "running": 6.9}                     # an LB redeploy beside a serving pair
KILLED_WHILE_SERVING = {"starting": 0.0, "failed": 6.3}           # a /health miss, then the memory guard
STOPPED_LOADING = {"reserving": 0.0, "starting": 4.2, "unloading": 129.8}
STOPPED_WAITING = {"reserving": 0.0, "freeing": 2.2, "releasing": 14.6}
REFUSED = {"restarting": 0.0, "failed": 2.1}
REBOOT_RELAUNCH = {"restarting": 0.0, "checking": 2.8, "reserving": 5.0, "starting": 48.8, "running": 837.9}
RE_ADOPTED_MID_START = {"starting": 0.0, "running": 452.5}


def boot(at, generation, outcome, states, *, model="m-a", uncensored=None, attempt=1, hang=None) -> dict:
    """A boot line as `BootRecorder._close` writes it — every field, so the rules are tested on the real shape."""
    total = max(states.values())
    return {"v": 1, "kind": "boot", "t_start": at, "t_end": at + total, "generation": generation, "attempt": attempt,
            "config": {"model_choice": model, "generation": generation, "max_num_seqs": 8, "uncensored": uncensored},
            "config_source": "launch", "outcome": outcome, "total_s": total, "states": states, "phases": [],
            "failure": None if outcome == "serving" else "why", "failed_in": None, "failed_phase": None,
            "hang": hang}


def rates(*records, now=NOW):
    return start_rates(list(records), now=now)


def row(out, key="m-a"):
    return out["rows"][key]


def only_today(ok, n):
    return [None] * (WINDOW_DAYS - 1) + [[ok, n]]


# ── what one attempt is, and what success is ──────────────────────────────────────────────────────────────

def test_a_launch_that_served_succeeded_and_one_that_failed_after_its_launch_failed():
    out = rates(boot(TODAY, 140, "failed", FAILED), boot(TODAY + 600, 141, "serving", SERVED))
    assert row(out) == {"ok": 1, "n": 2, "stopped": 0, "refused": 0, "days": only_today(1, 2)}
    assert out["unread"] == 0


def test_a_hung_attempt_and_the_relaunch_that_served_are_two_attempts_one_failed():
    """Generation 193 (2026-10-11): attempt 1 hung in CUDA-graph capture and was torn down, attempt 2 served."""
    hang = {"phase": "capturing CUDA graphs", "quiet_s": 415.8, "limit_s": 400, "attempt": 1, "retries": 2,
            "relaunch": True, "evidence": "start-hangs/gen193-attempt1"}
    out = rates(boot(TODAY, 193, "failed", HUNG, attempt=1, hang=hang),
                boot(TODAY + 1030, 193, "serving", RELAUNCH_SERVED, attempt=2))
    assert row(out)["ok"] == 1 and row(out)["n"] == 2 and row(out)["days"][-1] == [1, 2]


def test_a_pair_already_up_and_read_again_is_no_start_whatever_it_ends_in():
    """An LB redeploy beside a serving pair opens a line in `starting` that is `serving` 7 s later; a serving head
    that missed a /health probe and was then killed opens one that ends `failed`. Neither launched anything: the
    start they belong to is the line before them, with the same generation and attempt."""
    out = rates(boot(TODAY - 7200, 174, "serving", SERVED),
                boot(TODAY - 3600, 174, "serving", READ_AGAIN),
                boot(TODAY - 1800, 174, "serving", READ_AGAIN),
                boot(TODAY, 174, "failed", KILLED_WHILE_SERVING))
    assert row(out) == {"ok": 1, "n": 1, "stopped": 0, "refused": 0, "days": only_today(1, 1)}


def test_a_start_the_lb_was_restarted_in_the_middle_of_is_counted_once_from_the_line_it_left():
    """An LB restart forgets a start in progress (nothing is written for it); the new LB opens its own line in
    `starting` and writes how it ended. With no earlier line of that start, that line IS the start
    (generation 138, 2026-10-05: 452 s of weights and graphs, then serving)."""
    out = rates(boot(TODAY, 138, "serving", RE_ADOPTED_MID_START))
    assert row(out)["ok"] == 1 and row(out)["n"] == 1


def test_a_line_that_says_it_is_a_read_again_is_no_start_even_with_no_earlier_line():
    """A node the LB requests (control local) knows what the pair cannot: this LB saw ITS node serve under that intent,
    so a probe it then misses opens a line marked `read_again` (perflog.BootRecorder). site-b 2026-10-12: intent #14
    started before any boot line existed — without the mark, its first missed probe would be a start of the row."""
    out = rates(boot(TODAY, 14, "serving", READ_AGAIN) | {"read_again": True},
                boot(TODAY + 60, 14, "failed", KILLED_WHILE_SERVING) | {"read_again": True})
    assert out["rows"] == {} and out["unread"] == 0


def test_a_read_again_needs_the_same_start_not_merely_the_same_generation_number():
    """The earlier line must be the same start: same generation, same attempt, same checkpoint."""
    out = rates(boot(TODAY - 600, 9, "serving", SERVED, model="m-b"),
                boot(TODAY, 9, "serving", RE_ADOPTED_MID_START, model="m-a"),
                boot(TODAY + 60, 9, "serving", RE_ADOPTED_MID_START, model="m-a", attempt=2))
    assert row(out, "m-a")["n"] == 2 and row(out, "m-b")["n"] == 1


def test_a_relaunch_after_the_box_rebooted_under_the_pair_is_a_new_attempt():
    """Generation 188 served on 2026-10-10; the next morning its boxes had rebooted under it and the LB launched the
    same generation again through every gate (`restarting` first). That second launch is a start of the model."""
    out = rates(boot(TODAY - DAY, 188, "serving", SERVED, uncensored=True),
                boot(TODAY, 188, "serving", REBOOT_RELAUNCH, uncensored=True))
    assert row(out, "m-a@uncensored")["days"][-2:] == [[1, 1], [1, 1]]
    assert row(out, "m-a@uncensored")["n"] == 2


def test_a_start_stopped_after_its_launch_has_no_verdict_and_is_counted_apart():
    out = rates(boot(TODAY, 142, "stopped", STOPPED_LOADING))
    assert row(out) == {"ok": 0, "n": 0, "stopped": 1, "refused": 0, "days": [None] * WINDOW_DAYS}


def test_a_start_stopped_before_anything_was_launched_says_nothing_about_the_model():
    assert rates(boot(TODAY, 145, "stopped", STOPPED_WAITING))["rows"] == {}


def test_a_start_the_lb_refused_before_launching_is_named_and_is_not_an_attempt():
    out = rates(boot(TODAY, 187, "failed", REFUSED))
    assert row(out) == {"ok": 0, "n": 0, "stopped": 0, "refused": 1, "days": [None] * WINDOW_DAYS}


# ── whose row it is ────────────────────────────────────────────────────────────────────────────────────────

def test_the_twin_is_its_own_row_and_a_line_from_before_the_knob_is_the_original():
    """The picker's own key (`catalog.availability_key`): `<id>` and `<id>@uncensored`. A line written before the
    twins existed (uncensored null) launched the only checkpoint there was — the original."""
    out = rates(boot(TODAY, 1, "serving", SERVED, uncensored=None), boot(TODAY + 1, 2, "failed", FAILED, uncensored=False),
                boot(TODAY + 2, 3, "failed", FAILED, uncensored=True))
    assert set(out["rows"]) == {"m-a", "m-a@uncensored"}
    assert (row(out)["ok"], row(out)["n"]) == (1, 2)
    assert (row(out, "m-a@uncensored")["ok"], row(out, "m-a@uncensored")["n"]) == (0, 1)


def test_two_engine_rows_are_two_catalog_ids_and_never_share_a_rate():
    out = rates(boot(TODAY, 1, "serving", SERVED, model="flash-vllm"),
                boot(TODAY + 1, 2, "failed", FAILED, model="flash-sglang"))
    assert (row(out, "flash-vllm")["ok"], row(out, "flash-vllm")["n"]) == (1, 1)
    assert (row(out, "flash-sglang")["ok"], row(out, "flash-sglang")["n"]) == (0, 1)


# ── the days ───────────────────────────────────────────────────────────────────────────────────────────────

def test_the_window_is_the_seven_utc_days_ending_today_and_a_day_without_an_attempt_has_no_point():
    """UTC days, the boot log's own (`perflog.day_of`, one `boots-<day>.jsonl` a day): today and the six before it.
    A day the model was not launched is null — not 0 %, and not a value carried over from its neighbour."""
    midnight = datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()
    out = rates(boot(midnight - 6 * DAY, 1, "serving", SERVED),             # 2026-10-05 00:00:00: the first day
                boot(midnight - 6 * DAY - 1, 2, "failed", FAILED),        # 2026-10-04 23:59:59: outside
                boot(midnight - 1, 3, "failed", FAILED),                   # 2026-10-10 23:59:59
                boot(midnight, 4, "serving", SERVED))                      # 2026-10-11 00:00:00: today
    assert out["days"] == ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10",
                           "2026-10-11"]
    assert out["tz"] == "UTC"
    assert row(out)["days"] == [[1, 1], None, None, None, None, [0, 1], [1, 1]]
    assert (row(out)["ok"], row(out)["n"]) == (2, 3)


def test_an_earlier_line_outside_the_window_still_marks_a_pair_read_again():
    """A pair that has served for weeks and is read again after an LB redeploy today is not a start today."""
    out = rates(boot(NOW - 30 * DAY, 120, "serving", SERVED), boot(TODAY, 120, "serving", READ_AGAIN))
    assert out["rows"] == {}


def test_a_line_the_rules_cannot_read_is_counted_never_guessed():
    blank = boot(TODAY, 5, "serving", SERVED)
    blank["config"] = None
    odd = boot(TODAY + 1, 6, "rebooted", SERVED)
    out = rates(blank, odd, boot(TODAY + 2, 7, "serving", SERVED))
    assert out["unread"] == 2 and row(out)["n"] == 1


def test_nothing_recorded_is_seven_empty_days_and_no_row():
    out = rates()
    assert out["rows"] == {} and len(out["days"]) == WINDOW_DAYS and out["days"][-1] == day_of(NOW)


# ── box-c's real log, 2026-10-05..11 ─────────────────────────────────────────────────────────────────────

def real_lines() -> list[dict]:
    """The fixture as the LB itself reads it back (PerfLog over a `boots-` file)."""
    path = FIXTURES / "boots-2026-10-05-to-10-11.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_box_c_2026_10_05_to_11_per_model():
    out = start_rates(sorted(real_lines(), key=lambda r: (r["t_start"], r["t_end"])), now=NOW)
    n = None
    assert out["rows"] == {
        "qwen3.8-flash-next-sglang": {"ok": 12, "n": 26, "stopped": 4, "refused": 0,
                                      "days": [[7, 20], [2, 2], n, n, [1, 1], [2, 3], n]},
        "qwen3.8-flash-next-sglang@uncensored": {"ok": 4, "n": 6, "stopped": 1, "refused": 0,
                                                 "days": [n, n, n, n, n, [2, 4], [2, 2]]},
        "qwen3.8-flash-next-hibrid48": {"ok": 1, "n": 2, "stopped": 1, "refused": 0,
                                        "days": [[1, 1], n, n, n, n, n, [0, 1]]},
        "qwen3.8-flash-next-hibrid48@uncensored": {"ok": 1, "n": 6, "stopped": 0, "refused": 0,
                                                   "days": [n, n, n, n, n, [0, 2], [1, 4]]},
        "qwen3.8-27b-sglang-dflash2": {"ok": 1, "n": 1, "stopped": 0, "refused": 0,
                                       "days": [n, n, [1, 1], n, n, n, n]},
        "qwen3.8-27b-sglang-dflash2@uncensored": {"ok": 6, "n": 8, "stopped": 0, "refused": 0,
                                                  "days": [n, n, [1, 2], [1, 2], [3, 3], [1, 1], n]},
        "qwen3.8-27b-dflash2": {"ok": 2, "n": 2, "stopped": 1, "refused": 0,
                                "days": [[1, 1], n, [1, 1], n, n, n, n]},
    }
    assert out["unread"] == 0


def test_box_c_generations_183_to_193_vllm_flash_next():
    """vLLM Flash-Next, 2026-10-10..11: the twin killed by tp-rank's memory guard four times (183, 184, 190, 191),
    hung once in graph capture (193 attempt 1) and served on that hang's relaunch (193 attempt 2); the original
    killed once (192). The SGLang twin beside it in the same hours: one start replaced before it served (187,
    stopped — no verdict), and 186, 188, 188 again after its boxes rebooted, and 189 served."""
    lines = [r for r in real_lines() if 183 <= r["generation"] <= 193]
    out = start_rates(sorted(lines, key=lambda r: (r["t_start"], r["t_end"])), now=NOW)
    assert out["rows"]["qwen3.8-flash-next-hibrid48@uncensored"]["days"][-2:] == [[0, 2], [1, 4]]
    assert out["rows"]["qwen3.8-flash-next-hibrid48"]["days"][-1] == [0, 1]
    assert out["rows"]["qwen3.8-flash-next-sglang@uncensored"] == {
        "ok": 4, "n": 4, "stopped": 1, "refused": 0, "days": [None] * 5 + [[2, 2], [2, 2]]}


def test_the_window_moves_with_the_utc_day():
    """At 00:30 UTC on the 12th the 5th has left the window: the SGLang Flash-Next's twenty starts of that day
    with it."""
    out = start_rates(sorted(real_lines(), key=lambda r: (r["t_start"], r["t_end"])),
                      now=datetime(2026, 10, 12, 0, 30, tzinfo=timezone.utc).timestamp())
    assert out["days"][0] == "2026-10-06" and out["days"][-1] == "2026-10-12"
    sglang = out["rows"]["qwen3.8-flash-next-sglang"]
    assert (sglang["ok"], sglang["n"], sglang["days"]) == (5, 6, [[2, 2], None, None, [1, 1], [2, 3], None, None])


# ── /lb/status ─────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def served(tmp_path):
    started = []

    def make(**over):
        cfg = Config(vllm_boxes=(VllmBox("box-gpu", "box", "http://127.0.0.1:9"),), listen_host="127.0.0.1",
                     listen_port=0, bench_dir=str(tmp_path / "bench"), perf_dir=str(tmp_path / "perf"), **over)
        proxy = Proxy(cfg, Hub(cfg), perf_recorder=None)
        proxy.start()
        started.append(proxy)
        return proxy, tmp_path / "perf"

    yield make
    for p in started:
        p.stop()


def status(proxy) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", "/lb/status")
    return json.loads(conn.getresponse().read())


def test_status_carries_every_recorded_rows_start_rate(served):
    proxy, perf = served()
    now = time.time()
    writer = JsonlWriter(str(perf), prefix=BOOT_PREFIX, max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    writer.write(boot(now - 900, 40, "failed", FAILED))
    writer.write(boot(now - 600, 41, "serving", SERVED))
    writer.write(boot(now - 300, 41, "serving", READ_AGAIN))
    out = status(proxy)["start_rates"]
    assert (out["rows"]["m-a"]["ok"], out["rows"]["m-a"]["n"]) == (1, 2)
    assert out["days"][-1] == day_of(now) and out["tz"] == "UTC"
    assert out["error"] == "" and out["malformed"] == 0


def test_status_names_a_boot_line_it_could_not_read(served):
    proxy, perf = served()
    perf.mkdir(parents=True, exist_ok=True)
    (perf / f"{BOOT_PREFIX}-{day_of(time.time())}.jsonl").write_text("{not json\n")
    out = status(proxy)["start_rates"]
    assert out["malformed"] == 1 and out["rows"] == {}


def test_nothing_recorded_yet_is_no_starts_not_an_error(served):
    """No boot log directory: the recorder makes it with its first line, so nothing has been launched here."""
    proxy, perf = served()
    assert not perf.exists()
    out = status(proxy)["start_rates"]
    assert out["rows"] == {} and out["error"] == "" and len(out["days"]) == WINDOW_DAYS


def test_with_the_boot_log_off_the_rates_are_unknown_never_no_starts(served):
    """LB_PERF=0 records no start: an empty map would read "no starts in 7 d" on every row. It says why instead."""
    proxy, _ = served(perf_enabled=False)
    out = status(proxy)["start_rates"]
    assert out["rows"] == {} and "LB_PERF=0" in out["error"]


def test_the_reader_is_the_lbs_own(tmp_path):
    """The rates are computed over what PerfLog reads back — the same reader /lb/perf and /lb/sessions use."""
    writer = JsonlWriter(str(tmp_path), prefix=BOOT_PREFIX, max_total_bytes=1 << 20, max_file_bytes=1 << 20)
    for rec in real_lines():
        writer.write(rec)
    index = PerfLog(str(tmp_path), prefix=BOOT_PREFIX).index(days=None, now=NOW)
    assert index["malformed_total"] == 0 and len(index["windows"]) == 94
    assert start_rates(index["windows"], now=NOW)["rows"]["qwen3.8-flash-next-hibrid48@uncensored"]["n"] == 6
