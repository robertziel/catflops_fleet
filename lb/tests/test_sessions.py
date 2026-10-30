"""THE HISTORY OF LOADED MODELS (perflog.sessions, GET /lb/sessions, the /sessions page).

Operator, 2026-10-05: "Can we accumulate loaded models sessions history with aggregated data graphs,
time, total tokens generated etc? Make new page index with these data."

A SESSION is one generation of the pair: the model that was loaded, from its first busy minute to its
last. Everything a session says is a SUM over the windows the performance log already keeps — tokens,
requests, seconds, drafts, hits — so a rate is tokens over busy seconds and never a mean of per-minute
means. The fold is server-side on purpose: /lb/perf hands a browser at most MAX_WINDOWS minutes, and
this page has to be right after the log outgrows that.
"""
from __future__ import annotations

import http.client
import json

import pytest

from tests.fakes import FakeBackend
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.perflog import SESSION_VERSION, sessions
from vllm_lb.proxy import Proxy

T0 = 1_790_740_020.0
FLASH = "qwen3.8-flash-next-hibrid48"
SGLANG = "qwen3.8-27b-sglang-dflash2"


def window(gen=52, model=FLASH, *, start, span=60.0, gen_tokens=6000.0, prompt=120_000.0,
           hits=90_000.0, requests=12.0, seqs=8, batched=8192, drafts=4, **over):
    """A window shaped exactly like `perflog.record`'s — the only shape the fold is allowed to read."""
    w = {"v": 1, "t_start": start, "t_end": start + span, "span_s": span, "reads": 11, "pairs": 10,
         "config": {"model_choice": model, "generation": gen, "max_num_seqs": seqs,
                    "max_num_batched_tokens": batched, "num_speculative_tokens": drafts,
                    "max_model_len": 262144, "kv_cache_memory": 28_000_000_000},
         "config_source": "launch", "engine_restarts": 0,
         "gen_tokens": gen_tokens, "gen_tok_s": gen_tokens / span,
         "prompt_tokens": prompt, "prompt_computed_tokens": prompt - hits,
         "prefix_hit_tokens": hits, "prefix_query_tokens": prompt,
         "prefix_hit_rate": hits / prompt if prompt else None,
         "requests_finished": requests,
         "ttft": {"n": 10, "mean": 2.0, "p90": 4.0, "buckets": []},
         "itl": {"n": 100, "mean": 0.04, "p90": 0.05, "buckets": []},
         "running": {"mean": 4.0, "max": 8.0}, "waiting": {"mean": 0.0, "max": 3.0},
         "kv": {"mean": 0.1, "max": 0.2}, "kv_pool_tokens": 1_708_150.0,
         "spec": {"draft_tokens": 1000.0, "accepted": 450.0, "acceptance": 0.45},
         "preemptions": 0.0,
         "boxes": {"box-a": {"sm_mean": 90.0, "sm_max": 99.0, "pwr_mean": 40.0, "pwr_max": 70.0},
                   "box-b": {"sm_mean": 88.0, "sm_max": 97.0, "pwr_mean": 30.0, "pwr_max": 60.0}}}
    w.update(over)
    return w


def boot(gen=52, model=FLASH, *, start, total=120.0, outcome="serving", failure=None):
    return {"v": 1, "kind": "boot", "t_start": start, "t_end": start + total, "generation": gen,
            "attempt": 1, "config": {"model_choice": model, "generation": gen, "max_num_seqs": 8},
            "config_source": "launch", "outcome": outcome, "total_s": total,
            "states": {"starting": total}, "phases": [], "failure": failure}


# ── the fold ──────────────────────────────────────────────────────────────────────────────


def test_a_session_is_one_generation_and_every_figure_is_a_sum_over_its_minutes():
    out = sessions([window(start=T0, gen_tokens=6000.0, requests=12.0),
                    window(start=T0 + 60, gen_tokens=3000.0, requests=4.0, span=30.0)],
                   [boot(start=T0 - 130, total=120.0)])
    (s,) = out["sessions"]
    assert (s["v"], s["generation"], s["model_choice"]) == (SESSION_VERSION, 52, FLASH)
    assert s["gen_tokens"] == 9000 and s["requests"] == 16
    # TWO CLOCKS, and they are not the same number: 90 s of BUSY minutes inside a session that STOOD
    # for 220 — which begins when its START did (the boot record's own t_start), because the two
    # minutes the boxes spent loading the model are time this model held the pair.
    assert (s["active_s"], s["wall_s"]) == (90.0, 220.0)
    assert s["gen_tok_s"] == round(9000 / 90.0, 2), "tokens over BUSY seconds, not over the wall"
    assert s["peak_gen_tok_s"] == 100.0, "the fastest minute it had, not the mean"
    assert s["windows"] == 2 and s["boot_s"] == 120.0 and s["outcome"] == "serving"
    assert s["tokens_per_request"] == 562.5


def test_a_rate_is_the_ratio_of_two_sums_never_an_average_of_ratios():
    """One minute at a 90% hit rate over 200k tokens and one at 10% over 1k is 89.6%, not 50%."""
    out = sessions([window(start=T0, prompt=200_000.0, hits=180_000.0),
                    window(start=T0 + 60, prompt=1_000.0, hits=100.0)])
    (s,) = out["sessions"]
    assert s["prefix_hit_rate"] == round(180_100 / 201_000, 4)
    assert s["acceptance"] == round(900 / 2000, 4), "accepted over drafted, both summed"


def test_what_nobody_measured_is_absent_never_zero():
    plain = window(start=T0, spec=None, boxes={})
    plain["ttft"] = {"n": 0, "mean": None, "p90": None, "buckets": []}
    (s,) = sessions([plain])["sessions"]
    assert s["acceptance"] is None and s["ttft_mean_s"] is None
    assert s["watts_mean"] is None and s["wh"] is None, "a box that reported no power is not 0 W"
    assert s["boxes"] == {}


def test_the_energy_is_the_pairs_draw_over_the_seconds_it_was_busy():
    """70 W across the two boxes for one busy minute is 1.167 Wh — the figure the tiles add up."""
    (s,) = sessions([window(start=T0, span=60.0)])["sessions"]
    assert s["watts_mean"] == 70.0, "both boxes' mean draw, added"
    assert s["wh"] == round(70.0 * 60.0 / 3600.0, 2)


def test_a_generation_that_loaded_and_served_nothing_is_still_a_session():
    """It happened, and the history is the record of what happened — with the failure it ended on."""
    out = sessions([], [boot(gen=131, start=T0, total=90.0, outcome="failed",
                             failure="rank on box-b exited (code 1)")])
    (s,) = out["sessions"]
    assert (s["generation"], s["gen_tokens"], s["active_s"]) == (131, 0, 0.0)
    assert s["outcome"] == "failed" and "box-b" in s["failure"]
    assert s["gen_tok_s"] is None, "no busy second is not a rate of zero"
    assert s["boot_s"] is None, "only a start that SERVED says what a start costs"
    assert s["config"]["model_choice"] == FLASH, "the boot record carries what it tried to launch"


def test_two_generations_of_the_same_model_are_two_sessions_newest_first():
    out = sessions([window(gen=52, start=T0), window(gen=53, start=T0 + 600)])
    assert [s["generation"] for s in out["sessions"]] == [53, 52]


def test_a_generation_the_engine_ran_differently_keeps_the_configuration_its_busiest_minutes_ran():
    """`cluster.effective` takes the ENGINE's own word for a knob it reports differently (generation
    135: 2,048 sent, 8,192 run), so one generation can honestly hold two configurations. The session
    states the one its busiest minutes ran under, and says how many it saw."""
    out = sessions([window(gen=135, start=T0, batched=2048, span=10.0),
                    window(gen=135, start=T0 + 60, batched=8192, span=120.0)])
    (s,) = out["sessions"]
    assert s["config"]["max_num_batched_tokens"] == 8192 and s["config_variants"] == 2


def test_the_totals_and_the_per_model_rollup_are_the_sessions_added_up():
    out = sessions([window(gen=52, model=FLASH, start=T0, gen_tokens=6000.0, requests=12.0),
                    window(gen=53, model=SGLANG, start=T0 + 600, gen_tokens=4000.0, requests=8.0)],
                   [boot(gen=52, start=T0 - 130)],
                   describe=lambda m: {"label": {FLASH: "Flash-Next"}.get(m, "SGLang"), "engine": "vllm"})
    t = out["totals"]
    assert (t["sessions"], t["models"], t["gen_tokens"], t["requests"]) == (2, 2, 10000, 20)
    assert t["active_s"] == 120.0 and t["gen_tok_s"] == round(10000 / 120.0, 2)
    assert t["first_at"] == T0 - 130, "a session starts when its START did, not at its first busy minute"
    by = {m["model_choice"]: m for m in out["by_model"]}
    assert by[FLASH]["gen_tokens"] == 6000 and by[FLASH]["label"] == "Flash-Next"
    assert [m["model_choice"] for m in out["by_model"]] == [FLASH, SGLANG], "most tokens first"


FIXTURES = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


def test_a_pair_read_again_after_every_redeploy_is_no_attempt_and_no_boot_time_of_its_session():
    """box-c's own log (tests/fixtures/boots-2026-10-05-to-10-11.jsonl): generation 177 started in 759.9 s, then
    every LB redeploy that day opened a line `starting` → `running` 7 s later — fifteen of them. The session said 16
    attempts and a 6.8 s boot. ONE rule says which lines are a start (perflog.read_again, the start rates' own)."""
    lines = [json.loads(x) for x in (FIXTURES / "boots-2026-10-05-to-10-11.jsonl").read_text().splitlines()]
    rows = {s["generation"]: s for s in sessions([], lines)["sessions"]}
    assert sum(1 for b in lines if b["generation"] == 177) == 16
    assert (rows[177]["boot_attempts"], rows[177]["boot_s"]) == (1, 759.9)
    assert (rows[179]["boot_attempts"], rows[179]["boot_s"]) == (1, 692.8)


def test_a_read_again_inside_a_range_is_known_by_its_start_before_the_range():
    """/sessions?days=1: the start was two days ago, a redeploy read the pair again today. The start's line makes no
    session of its own outside the range, but it still says today's line is no start."""
    start = boot(gen=60, start=T0 - 2 * 86400, total=700.0)
    again = boot(gen=60, start=T0, total=6.9) | {"states": {"starting": 0.0, "running": 6.9}}
    out = sessions([window(gen=60, start=T0 + 60)], [start, again], since=T0 - 3600)
    (s,) = out["sessions"]
    assert (s["boot_attempts"], s["boot_s"]) == (0, None), "today's line is a read again, not a 6.9 s start"
    assert s["started_at"] == T0, "the start before the range is not this range's"


def test_a_window_with_no_generation_or_model_is_counted_out_loud_not_folded_in():
    out = sessions([window(start=T0), {"t_start": T0, "t_end": T0 + 60, "span_s": 60.0, "config": {}}])
    assert len(out["sessions"]) == 1 and out["windows_without_a_session"] == 1


# ── the endpoint ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def served(tmp_path):
    """A proxy whose performance log is a directory we write the two files into by hand."""
    d = tmp_path / "perf"
    d.mkdir()
    (d / "perf-2026-09-30.jsonl").write_text(
        "\n".join(json.dumps(w) for w in [window(gen=52, start=T0), window(gen=52, start=T0 + 60),
                                          window(gen=53, model=SGLANG, start=T0 + 600)]) + "\n")
    (d / "boots-2026-09-30.jsonl").write_text(json.dumps(boot(gen=52, start=T0 - 130)) + "\n")
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), 
                 listen_host="127.0.0.1", listen_port=0, perf_dir=str(d))
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def get(proxy, path):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    return r.status, json.loads(body)


def test_the_endpoint_folds_the_log_and_names_every_model_from_the_catalog(served):
    status, d = get(served, "/lb/sessions")
    assert status == 200
    assert [s["generation"] for s in d["sessions"]] == [53, 52]
    newest, oldest = d["sessions"]
    # THE CATALOG SAYS WHAT AN ID IS CALLED, and which server runs it: the log stores neither.
    assert newest["label"] == "Qwen3.8 27B · dense NVFP4 · SGLang DFlash2" and newest["engine"] == "sglang"
    assert oldest["label"].startswith("Qwen3.8 Flash-Next") and oldest["engine"] == "vllm"
    assert oldest["gen_tokens"] == 12000 and oldest["boot_s"] == 120.0
    assert d["totals"]["sessions"] == 2 and d["totals"]["gen_tokens"] == 18000
    assert d["recorder"]["enabled"] is False and d["windows_truncated"] == 0


def test_a_model_the_catalog_no_longer_has_keeps_its_id_and_says_so(served):
    import pathlib
    d = pathlib.Path(served.cfg.perf_dir)
    (d / "perf-2026-09-30.jsonl").write_text(
        json.dumps(window(gen=99, model="a-model-we-retired", start=T0)) + "\n")
    (d / "boots-2026-09-30.jsonl").write_text("")
    status, out = get(served, "/lb/sessions")
    assert status == 200
    (s,) = out["sessions"]
    assert (s["label"], s["known"], s["engine"]) == ("a-model-we-retired", False, None)
    assert s["gen_tokens"] == 6000, "its minutes still count: the log is the record, not the catalog"


def test_the_default_range_is_the_whole_history_and_days_is_read_the_same_as_on_perf(served):
    _, everything = get(served, "/lb/sessions")
    assert everything["days"] is None, "the history is the subject: `all` is the default here"
    _, same = get(served, "/lb/sessions?days=all")
    assert [s["generation"] for s in same["sessions"]] == [s["generation"] for s in everything["sessions"]]
    _, recent = get(served, "/lb/sessions?days=1")
    assert recent["days"] == 1.0 and recent["sessions"] == [], "every minute here is older than a day"
    status, bad = get(served, "/lb/sessions?days=soon")
    assert status == 400 and bad["error"]["type"] == "invalid_request"
    # and /lb/perf still reads the word exactly as it did — one parser, two endpoints
    assert get(served, "/lb/perf?days=soon")[0] == 400
    assert get(served, "/lb/perf")[1]["days"] == 7.0


def test_the_page_is_served_at_sessions_and_is_the_only_thing_that_route_does(served):
    c = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    c.request("GET", "/sessions")
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    assert r.status == 200 and "text/html" in (r.getheader("content-type") or "")
    assert 'id="s-tiles"' in body and "/sessions.js" in body
    for method in ("POST", "PUT", "DELETE"):
        c = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
        c.request(method, "/lb/sessions", body=b"{}")
        assert c.getresponse().status in (404, 405), method
        c.close()


def test_a_counter_no_window_carried_is_not_measured_not_zero():
    """A session's `prompt_computed_tokens` is the sum of its windows' — and when NO window carried it
    that is "the log did not measure it", not 0. The difference is a price: 0 computed beside 120,000
    billed prompt tokens reads as "every one was a cache hit". A generation with no windows at all is
    different — it loaded and served nothing, and 0 is true."""
    bare = window(gen=1, start=T0)
    bare["prompt_computed_tokens"] = None
    (s,) = sessions([bare])["sessions"]
    assert s["prompt_computed_tokens"] is None and s["prompt_tokens"] == 120000
    (idle,) = sessions([], [boot(gen=2, start=T0)])["sessions"]
    assert idle["prompt_computed_tokens"] == 0 and idle["gen_tokens"] == 0


# ── tokens per kWh ────────────────────────────────────────────────────────────────────────


def test_tokens_per_kwh_is_generated_tokens_over_the_energy_the_boxes_drew_while_busy():
    """6,000 tokens in a 60 s busy minute at 70 W (both boxes' mean draw added) is 1.1667 Wh, so
    6,000 / (1.1667 / 1000) = 5,142,857 tokens per kWh — from the UNROUNDED energy, not the row's 1.17."""
    (s,) = sessions([window(start=T0, span=60.0, gen_tokens=6000.0)])["sessions"]
    assert s["wh"] == 1.17
    assert s["tok_per_kwh"] == round(6000.0 / (70.0 * 60.0 / 3600.0 / 1000.0)) == 5142857


def test_tokens_per_kwh_needs_both_sides_measured_and_is_never_zero_or_infinite():
    no_power = window(start=T0, boxes={})
    (a,) = sessions([no_power])["sessions"]
    assert a["wh"] is None and a["tok_per_kwh"] is None, "no power reading is not infinite efficiency"
    (b,) = sessions([window(start=T0, gen_tokens=0.0)])["sessions"]
    assert b["tok_per_kwh"] is None, "a session that generated nothing has no ratio, not 0 tok/kWh"
    (c,) = sessions([], [boot(gen=3, start=T0)])["sessions"]
    assert c["tok_per_kwh"] is None


def test_the_aggregate_efficiency_is_a_ratio_of_sums_over_the_sessions_where_both_were_measured():
    """Session A: 6,000 tok on 1.1667 Wh. Session B: 600,000 tok but NO power reading. Counting B's tokens
    over A's energy would be 500x too good; the aggregate uses A alone and says it covered 1 of 2."""
    a = window(gen=1, start=T0, span=60.0, gen_tokens=6000.0)
    b = window(gen=2, start=T0 + 600, span=60.0, gen_tokens=600000.0, boxes={})
    out = sessions([a, b])
    eff = out["totals"]["efficiency"]
    assert eff["measured"] == 1 and eff["of"] == 2
    # equal to the per-session figure to a part in a million (energy is carried to 6 decimals), and
    # nowhere near the 500x-too-good number B's tokens over A's energy would give
    assert eff["tok_per_kwh"] == pytest.approx(5142857, rel=1e-6), "B's tokens must not ride on A's energy"
    (model,) = out["by_model"]
    assert model["efficiency"] == eff, "one model: the rollup is the whole"


def test_current_is_the_serving_pair_effective_launch_and_never_a_wish(served):
    """`current` names the pair that SERVES by its effective launch — the same (model, generation)
    the log stamps its minutes with, which is what lets the page price "this session" alone. No
    cluster published, or one still starting with no effective record: null. (operator 2026-10-06.)"""
    status, d = get(served, "/lb/sessions")
    assert status == 200
    assert d["current"] is None, "nothing published a cluster view — there is no running session"
    served.hub.set_cluster({"configured": True, "state": "running", "desired": {"mode": "cluster"},
                            "effective": {"model_choice": SGLANG, "generation": 53}}, "pair")
    status, d = get(served, "/lb/sessions")
    assert d["current"]["model_choice"] == SGLANG and d["current"]["generation"] == 53
    assert d["current"]["state"] == "running"
    # the label is the catalog's, so the tile can name the model without a table lookup
    assert d["current"]["label"] == "Qwen3.8 27B · dense NVFP4 · SGLang DFlash2"
    # a pair still booting has NO effective record — the wish (`desired`) is not a running session
    served.hub.set_cluster({"configured": True, "state": "starting", "desired": {"mode": "cluster",
                                 "model_choice": SGLANG, "generation": 54}})
    status, d = get(served, "/lb/sessions")
    assert d["current"] is None, "desired without effective is a plan, not a session"
