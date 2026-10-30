"""Saved benchmark runs: what the store reads, what it refuses to invent, and the two read-only
routes the page at /benchmarks calls.

The cases that matter here are the DIRTY ones, because the directory this reads is where a
benchmark harness that can be killed writes: a truncated JSON, a 6 MB corpus INPUT sitting beside
the results, another tool's output in the same shape of file, and a run whose only record of its own
concurrency cap is the text in its label.
"""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.bench import BenchStore, group_of, rep_from_label, seqs_from_label
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy


# ── fixtures: the two shapes the two harnesses really write ────────────────

def synthetic(label="post-seqs24-rep1", *, run=None, seqs_curve=(281.7, 211.2, 19.9, 657.9)):
    """worklog/bench/load100.py's shape, key for key (see worklog/bench/post-seqs24-rep1.json):
    `AGG_TOK_S_overall`, `curve_tok_s`, `requests_ok`, and the series sample key `tok_s`."""
    mean, median, p10, p90 = seqs_curve
    out = {
        "summary": {
            "label": label,
            "args": {"url": "http://192.0.2.2:8100", "model": "Qwen/Qwen3.8-Flash-Next",
                     "requests": 100, "concurrency": 32, "prompt_tokens": 2000,
                     "max_tokens": 512, "seed": 424242, "sample_s": 2.0, "label": label},
            "requests_total": 100, "requests_ok": 100, "errors": 0, "error_sample": None,
            "TOTAL_WALL_S": 182.52, "output_tokens_total": 51200, "AGG_TOK_S_overall": 280.5,
            "engine_agg_tok_s": 280.5, "engine_ms_per_iter": 293.4,
            "curve_tok_s": {"n": 90, "mean": mean, "median": median,
                            "p10_min10pct": p10, "p90_top10pct": p90, "min": 0.0, "max": 720.4},
            "per_request_tok_s": {"mean": 15.4, "p10": 11.97, "p90": 18.58},
            "ttft_s": {"mean": 18.97, "p10": 9.9, "p90": 32.46},
            "preemptions": 0.0, "spec_accept": 0.8, "prefix_hit": 0.0,
            "running_mean": 21.81, "running_max": 24.0, "kv_max": 0.451,
        },
        "series": [{"t": 2.1, "tok_s": 0.5, "iters_s": 0.5, "ms_per_iter": 2009.2,
                    "running": 5.0, "waiting": 27.0, "kv": 0.093},
                   {"t": 4.1, "tok_s": None, "ms_per_iter": None,
                    "running": 24.0, "waiting": 8.0, "kv": 0.31},
                   {"t": 6.1, "tok_s": 640.2, "ms_per_iter": 291.0,
                    "running": 24.0, "waiting": 8.0, "kv": 0.45}],
        "requests": [{"idx": 5, "start_rel": 0.0, "end_rel": 42.51, "ttft_s": 8.63,
                      "e2e_s": 42.51, "out_tokens": 512, "prompt_tokens": 2436,
                      "decode_tok_s": 15.08}],
    }
    if run is not None:
        out["run"] = run
    return out


def corpus(label="real-seqs16"):
    """worklog/bench/load_corpus.py's shape: `AGG_GEN_TOK_S`, `gen_tok_s_curve`, `ok`, two phases,
    and the series sample key `gen_tok_s`. Nothing about it is named as the synthetic harness names
    it, which is the whole reason `view()` exists."""
    return {
        "summary": {
            "label": label, "requests": 120, "ok": 118, "errors": 2,
            "error_sample": "HTTP 500: engine died",
            "args": {"model": "Qwen/Qwen3.8-Flash-Next", "corpus": "corpus_v3.json",
                     "concurrency": 32, "max_tokens": 512, "sample_s": 3.0, "label": label},
            "TOTAL_WALL_S": 900.4, "PHASE1_FILL_WALL_S": 259.8, "PHASE2_SERVE_WALL_S": 640.6,
            "PHASE2_SERVE_TOK_S": 89.4, "output_tokens_total": 57000,
            "AGG_GEN_TOK_S": 63.3, "engine_gen_tok_s": 63.5, "engine_ms_per_iter": 410.2,
            "prefix_hit_rate": 0.926, "preemptions": 3.0, "spec_accept": 0.79,
            "gen_tok_s_curve": {"n": 300, "mean": 63.3, "median": 58.0,
                                "p10_min10pct": 6.1, "p90_top10pct": 141.2, "min": 0.0, "max": 180.0},
            "per_request_decode_tok_s": {"n": 118, "mean": 11.2, "median": 11.0,
                                         "p10_min10pct": 6.4, "p90_top10pct": 15.9, "min": 4.0, "max": 18.0},
            "ttft_s_ALL": {"n": 118, "mean": 40.1, "median": 21.0,
                           "p10_min10pct": 7.0, "p90_top10pct": 130.0, "min": 3.0, "max": 260.0},
            "FILL_cold_ttft_s": {"n": 8, "mean": 180.0, "median": 175.0,
                                 "p10_min10pct": 120.0, "p90_top10pct": 250.0, "min": 110.0, "max": 260.0},
            "WARM_ttft_s": {"n": 110, "mean": 12.0, "median": 9.0,
                            "p10_min10pct": 4.0, "p90_top10pct": 25.0, "min": 3.0, "max": 40.0},
            "running_mean": 12.0, "running_max": 16.0, "kv_max": 0.77,
        },
        "series": [{"t": 3.0, "gen_tok_s": 12.0, "prompt_tok_s": 8100.0, "ms_per_iter": 900.0,
                    "running": 8.0, "waiting": 0.0, "kv": 0.2, "preempt": 0.0}],
        "requests": [{"idx": 0, "start_rel": 0.0, "end_rel": 180.0, "ttft_s": 175.0, "e2e_s": 180.0,
                      "out_tokens": 512, "prompt_tokens": 210000, "decode_tok_s": 10.2,
                      "cold_fill": True}],
    }


#: annotate.py's two provenance phrasings, verbatim.
VERIFIED = "engine argv, verified at launch"
FROM_LABEL = "label (synthetic series; batched and spec were fixed at 8192 / 4)"


def stamp(**over):
    """annotate.py's stamp, key for key (see worklog/bench/C8-s4-dec1.json `summary.config`)."""
    return {"model": "Qwen/Qwen3.8-Flash-Next", "max_num_seqs": 8, "batched": 8192, "spec": 4,
            "generation": 48, "source": VERIFIED, "workload": "realdecode",
            "desc": "100 real prompts x ~2k tokens, 512 out, concurrency 32", **over}


def real_decode(label="C8-s4-dec1", config=None):
    """load100.py --prompts-file realdecode.json: the synthetic harness's shape carrying REAL prompts.
    `args.prompt_tokens` is still load100's default 2000 — which that mode never uses — and the
    summary has `prefix_hit_rate`, which load100.py has written since 2026-09-30 exactly as
    load_corpus.py does."""
    body = synthetic(label)
    s = body["summary"]
    s["args"].update({"prompts_file": "realdecode.json",
                      "out": f"/srv/repo/worklog/bench/{label}.json"})
    s.update({"TOTAL_WALL_S": 348.93, "AGG_TOK_S_overall": 146.7, "engine_agg_tok_s": 146.7,
              "engine_prompt_tok_s": 749.8, "prompt_tokens_total": 261645.0,
              "prefix_hit_rate": 0.0, "prefix_hit": 0.0})
    if config is not None:
        s["config"] = config
    return body


def prefill(label="B0-base-prefill", config=None, **over):
    """prefill_probe.py's FLAT result, key for key as worklog/bench/B0-base-prefill.json — including
    the three iteration fields the probe later withdrew as wrong."""
    body = {"label": label, "requests": 16, "ok": 16, "errors": [], "wall_s": 193.1,
            "prompt_tokens": 593361, "prefix_hit_tokens": 0, "computed_tokens": 593361,
            "PREFILL_TOK_S": 3073.0, "TFLOPS_PER_GPU": 52.83, "GBPS_PER_GPU": 39.3,
            "iterations": 16, "tokens_per_iter": 37085.0, "ms_per_iter": 12067.1,
            "preemptions": 0.0, "PCT_OF_NVFP4_PEAK": 16.3, **over}
    if config is not None:
        body["config"] = config
    return body


@pytest.fixture
def store(tmp_path):
    def make(files: dict):
        for name, body in files.items():
            path = tmp_path / name
            path.write_text(body if isinstance(body, str) else json.dumps(body, indent=1))
        return BenchStore(str(tmp_path))
    return make


# ── the labels, which are all the old results have ─────────────────────────

@pytest.mark.parametrize("label,expected", [
    ("seqs08", 8), ("seqs16", 16), ("seqs16-rep1", 16), ("seqs24-boot", 24),
    ("post-seqs24-rep2", 24), ("real-seqs24", 24), ("seqs_8", 8),
    # Nothing in the name says a concurrency cap: None, and the page shows "—" rather than a guess.
    ("T1", None), ("warmup", None), ("run-2026-09-29", None),
])
def test_the_cap_a_label_claims(label, expected):
    assert seqs_from_label(label) == expected


def test_reps_of_one_setting_share_a_group():
    assert group_of("post-seqs24-rep2") == "post-seqs24"
    assert rep_from_label("post-seqs24-rep2") == 2
    # `-boot` is NOT a rep: it measured a cold cluster and must not average with the warm runs.
    assert group_of("seqs24-boot") == "seqs24-boot"
    assert rep_from_label("seqs24-boot") is None


# ── reading a run ──────────────────────────────────────────────────────────

def test_a_synthetic_run_is_read_and_its_cap_is_marked_as_derived(store):
    idx = store({"post-seqs24-rep1.json": synthetic()}).index()
    assert [r["id"] for r in idx["runs"]] == ["post-seqs24-rep1"]
    r = idx["runs"][0]
    assert r["ok"] and r["kind"] == "synthetic"
    assert r["agg_tok_s"] == 280.5 and r["wall_s"] == 182.52
    assert r["curve"]["p10"] == 19.9 and r["curve"]["p90"] == 657.9
    assert r["requests_ok"] == 100 and r["errors"] == 0
    assert r["model"] == "Qwen/Qwen3.8-Flash-Next"
    # The number IS 24 — and the store says where it came from, because nothing in the file records it.
    assert r["max_num_seqs"] == 24
    assert r["max_num_seqs_source"] == "label"


def test_a_run_that_recorded_its_own_settings_is_not_derived_from_its_name(store):
    """runmeta.py's block wins over the label — and that is the point of shipping it. The label here
    says 24 and the cluster was actually launched at 16; the recorded number is the true one."""
    recorded = {"kind": "synthetic", "harness": "load100.py", "at": 1790700000.0,
                "cluster": {"source": "lb", "max_num_seqs": 16, "max_model_len": 262144,
                            "model_choice": "qwen3.8-flash-next-hibrid48",
                            "served_model": "Qwen/Qwen3.8-Flash-Next", "generation": 41,
                            "state": "serving"}}
    r = store({"post-seqs24-rep1.json": synthetic(run=recorded)}).index()["runs"][0]
    assert r["max_num_seqs"] == 16
    assert r["max_num_seqs_source"] == "recorded"
    assert r["model_source"] == "recorded"
    assert r["generation"] == 41 and r["cluster_state"] == "serving"


def test_a_run_whose_lb_could_not_be_reached_says_it_does_not_know(store):
    """runmeta records the FAILURE rather than a number. With no cap recorded the label is all
    there is, and the source still says so."""
    unavailable = {"kind": "synthetic", "harness": "load100.py",
                   "cluster": {"source": "unavailable", "error": "URLError: connection refused"}}
    r = store({"nameless.json": synthetic("nameless", run=unavailable)}).index()["runs"][0]
    assert r["max_num_seqs"] is None
    assert r["max_num_seqs_source"] == "unknown"


def test_a_corpus_run_is_read_through_its_own_different_key_names(store):
    r = store({"real-seqs16.json": corpus()}).index()["runs"][0]
    assert r["kind"] == "corpus" and r["harness"] == "load_corpus.py"
    # AGG_GEN_TOK_S, gen_tok_s_curve, `ok` — none of which the synthetic harness writes.
    assert r["agg_tok_s"] == 63.3
    assert r["curve"]["p10"] == 6.1 and r["curve"]["mean"] == 63.3
    assert r["requests_ok"] == 118 and r["requests"] == 120 and r["errors"] == 2
    assert r["prefix_hit"] == 0.926
    assert r["phases"]["fill_s"] == 259.8 and r["phases"]["serve_tok_s"] == 89.4
    assert r["corpus_file"] == "corpus_v3.json"


def test_the_series_is_one_shape_whichever_harness_wrote_it(store):
    s = store({"a.json": synthetic("seqs24"), "b.json": corpus("real-seqs16")})
    syn = s.run("a")["series"]
    assert [p["gen_tok_s"] for p in syn] == [0.5, None, 640.2]     # `tok_s` read under one name
    # A sample nobody measured stays a GAP in the list — the page draws a break, not a zero.
    assert syn[1]["gen_tok_s"] is None and syn[1]["running"] == 24.0
    cor = s.run("b")["series"]
    assert cor[0]["gen_tok_s"] == 12.0 and cor[0]["prompt_tok_s"] == 8100.0


# ── what a result was stamped with (annotate.py) ───────────────────────────

def test_the_stamp_is_what_the_run_ran_under_not_what_its_label_says(store):
    """The label says seqs24; the engine's own argv said 16. The stamp is the measurement."""
    body = synthetic("post-seqs24-rep1")
    body["summary"]["config"] = stamp(max_num_seqs=16, spec=6, batched=16384, generation=50,
                                      workload="synthetic", desc="synthetic 2000-token prompts")
    r = store({"post-seqs24-rep1.json": body}).index()["runs"][0]
    assert (r["max_num_seqs"], r["max_num_seqs_source"]) == (16, "verified")
    assert (r["spec"], r["batched"], r["generation"]) == (6, 16384, 50)
    assert r["config_source"] == "verified" and r["config_text"] == VERIFIED
    assert r["workload"] == "synthetic" and r["workload_source"] == "config"
    assert r["workload_desc"] == "synthetic 2000-token prompts"


def test_a_stamp_that_came_from_a_label_is_marked_as_derived_not_verified(store):
    """The older synthetic series were stamped from their names. The number is the same one the
    label parse gives — and it must stay marked as a reading of a name, not a measurement."""
    body = synthetic("seqs16-rep2")
    body["summary"]["config"] = stamp(max_num_seqs=16, generation=None, source=FROM_LABEL,
                                      workload="synthetic")
    r = store({"seqs16-rep2.json": body}).index()["runs"][0]
    assert (r["max_num_seqs"], r["max_num_seqs_source"]) == (16, "label")
    assert (r["spec"], r["batched"]) == (4, 8192)
    assert r["config_source"] == "label" and r["config_text"] == FROM_LABEL


def test_what_runmeta_recorded_beats_a_label_stamp_and_a_verified_stamp_beats_both(store):
    recorded = {"kind": "synthetic", "cluster": {"max_num_seqs": 24, "generation": 41}}
    body = synthetic("seqs16-rep1", run=recorded)
    body["summary"]["config"] = stamp(max_num_seqs=16, source=FROM_LABEL, workload="synthetic")
    s = store({"a.json": body})
    assert (s.run("a")["max_num_seqs"], s.run("a")["max_num_seqs_source"]) == (24, "recorded")
    body["summary"]["config"] = stamp(max_num_seqs=8, workload="synthetic")
    (s.dir / "b.json").write_text(json.dumps(body))
    assert (s.run("b")["max_num_seqs"], s.run("b")["max_num_seqs_source"]) == (8, "verified")


def test_real_prompts_are_neither_2000_token_prompts_nor_a_corpus_run(store):
    """load100.py with --prompts-file: `args.prompt_tokens` is a default that mode ignores, and
    `prefix_hit_rate` no longer means load_corpus.py wrote the file. Both used to be read as facts:
    every real-prompt decode run was shown as a 2,000-token run AND as a corpus run."""
    r = store({"C8-s4-dec1.json": real_decode(config=stamp())}).index()["runs"][0]
    assert r["kind"] == "real" and r["harness"] == "load100.py"
    assert r["prompt_tokens"] is None and r["prompts_file"] == "realdecode.json"
    assert r["prompt_tok_s"] == 749.8 and r["agg_tok_s"] == 146.7
    assert r["workload"] == "realdecode" and (r["max_num_seqs"], r["spec"]) == (8, 4)


def test_an_unstamped_run_gets_no_workload_and_no_settings_from_its_name(store):
    """C8-s6-dec3.json was saved after the last annotate.py pass. Its name spells a cap and a
    speculation depth, but a name is not a stamp: it goes in NO comparison until it is stamped."""
    r = store({"C8-s6-dec3.json": real_decode("C8-s6-dec3")}).index()["runs"][0]
    assert r["ok"] and r["agg_tok_s"] == 146.7
    assert r["workload"] is None and r["workload_source"] == "unknown"
    assert (r["max_num_seqs"], r["max_num_seqs_source"]) == (None, "unknown")
    assert (r["spec"], r["batched"], r["config_source"]) == (None, None, "unknown")


def test_the_filters_offer_the_stamped_workloads_and_settings(store):
    f = store({"a.json": real_decode("C8-s4-dec1", config=stamp()),
               "b.json": real_decode("C16-s6-mix2", config=stamp(
                   max_num_seqs=16, spec=6, workload="realmixed-warm")),
               "c.json": prefill(config=stamp(max_num_seqs=24, batched=16384,
                                              workload="prefill-probe"))}).index()["facets"]
    assert f["workloads"] == ["prefill-probe", "realdecode", "realmixed-warm"]
    assert f["spec"] == [4, 6] and f["batched"] == [8192, 16384]
    assert f["max_num_seqs"] == [8, 16, 24]


# ── the prefill probe: a first-class run ──────────────────────────────────

def test_a_prefill_probe_is_a_run_not_a_skipped_file(store):
    """prefill_probe.py writes a FLAT object with no `summary` — which the store used to list as
    "not a benchmark result" beside the corpus inputs."""
    body = prefill(config=stamp(max_num_seqs=24, generation=46, workload="prefill-probe",
                                desc="16 real prompts x ~32k tokens, max_tokens 1: pure prefill"))
    idx = store({"B0-base-prefill.json": body}).index()
    assert idx["skipped"] == []
    r = idx["runs"][0]
    assert r["ok"] and r["kind"] == "prefill" and r["harness"] == "prefill_probe.py"
    assert r["workload"] == "prefill-probe" and r["workload_source"] == "config"
    assert (r["prefill_tok_s"], r["tflops_per_gpu"], r["pct_nvfp4_peak"]) == (3073.0, 52.83, 16.3)
    assert (r["gbps_per_gpu"], r["wall_s"]) == (39.3, 193.1)
    assert (r["requests"], r["requests_ok"], r["errors"]) == (16, 16, 0)
    assert (r["prompt_tokens_total"], r["prefix_hit_tokens"], r["computed_tokens"]) == (593361, 0, 593361)
    assert (r["max_num_seqs"], r["max_num_seqs_source"]) == (24, "verified")
    assert (r["spec"], r["batched"], r["generation"]) == (4, 8192, 46)
    assert r["model"] == "Qwen/Qwen3.8-Flash-Next"


def test_a_prefill_probe_claims_nothing_it_did_not_measure(store):
    """No generated tokens, no curve, no TTFT — each is None, never 0. And the iteration figures
    the probe itself withdrew (mid-prefill chunks emit nothing, so vLLM never counted them) are
    not read back in: B0's 12,067 ms "per iteration" was one sample per finished REQUEST."""
    r = store({"p.json": prefill(config=stamp(workload="prefill-probe"))}).run("p")
    assert r["agg_tok_s"] is None and r["prompt_tok_s"] is None and r["ms_per_iter"] is None
    assert r["curve"] == {} and r["ttft"] == {} and r["per_request"] == {}
    assert r["concurrency"] is None and r["prompt_tokens"] is None     # not saved by the probe
    assert r["series"] == [] and r["requests_detail"] == [] and r["samples"] == 0


def test_an_unstamped_prefill_probe_is_still_one_but_knows_no_settings(store):
    """The shape names the workload — nothing else writes PREFILL_TOK_S — but not the cap."""
    r = store({"C16-s6-prefill.json": prefill("C16-s6-prefill")}).index()["runs"][0]
    assert r["workload"] == "prefill-probe" and r["workload_source"] == "shape"
    assert (r["max_num_seqs"], r["max_num_seqs_source"]) == (None, "unknown")
    assert r["spec"] is None and r["model"] is None


def test_a_prefill_probes_failures_are_counted_from_its_requests(store):
    """The probe saves the first two error MESSAGES, not a count."""
    body = prefill(ok=13, errors=["HTTP 500: engine died", "timed out"])
    r = store({"p.json": body}).index()["runs"][0]
    assert r["errors"] == 3 and r["requests_ok"] == 13
    assert r["error_sample"] == "HTTP 500: engine died"


# ── the dirty directory ────────────────────────────────────────────────────

def test_a_truncated_run_is_listed_with_the_parsers_own_words_not_dropped(store):
    """Several results in worklog/bench are exactly this: json.dump was killed mid-write. The run
    must still appear — a killed run that vanishes looks like a bug in the page."""
    good = json.dumps(synthetic("seqs16-rep1"), indent=1)
    idx = store({"seqs16-rep1.json": good,
                 "seqs24-rep2.json": json.dumps(synthetic("seqs24-rep2"), indent=1)[:900]}).index()
    by_id = {r["id"]: r for r in idx["runs"]}
    assert by_id["seqs16-rep1"]["ok"] is True
    broken = by_id["seqs24-rep2"]
    assert broken["ok"] is False
    assert "truncated or malformed JSON" in broken["error"]
    # Even unreadable, what its NAME says is still offered — that is how the operator finds it.
    assert broken["max_num_seqs"] == 24 and broken["max_num_seqs_source"] == "label"


def test_a_corpus_input_beside_the_results_is_skipped_and_named(store):
    """corpus_v3.json (6 MB of prompts) and T1.json (moe_sweep's `test`/`args`/`rows` table) live in
    the same directory. Neither is a run; neither may be dropped in silence."""
    idx = store({"seqs08.json": synthetic("seqs08"),
                 "corpus_v3.json": {"families": [{"family": "f", "prefix": "x"}], "requests": []},
                 "T1.json": {"test": "moe", "args": {}, "rows": [[1, 2]]}}).index()
    assert [r["id"] for r in idx["runs"]] == ["seqs08"]
    assert sorted(s["file"] for s in idx["skipped"]) == ["T1.json", "corpus_v3.json"]
    assert all("not a benchmark result" in s["error"] for s in idx["skipped"])


def test_a_json_that_is_not_an_object_does_not_raise(store):
    idx = store({"weird.json": "[1, 2, 3]"}).index()
    assert idx["runs"] == []
    assert idx["skipped"][0]["error"].startswith("no `summary`") or "not an object" in idx["skipped"][0]["error"]


def test_a_statistic_that_is_null_stays_null_and_never_becomes_zero(store):
    """A run that generated nothing has `mean: null`, and 0 tok/s is a different claim from "not
    measured". The page prints an em dash for one and a number for the other."""
    body = synthetic("dead")
    body["summary"]["curve_tok_s"] = {"n": 0, "mean": None, "median": None,
                                      "p10_min10pct": None, "p90_top10pct": None, "min": None, "max": None}
    body["summary"]["AGG_TOK_S_overall"] = None
    r = store({"dead.json": body}).index()["runs"][0]
    assert r["ok"] and r["agg_tok_s"] is None and r["curve"]["mean"] is None


def test_an_empty_directory_is_an_empty_page_not_an_error(store):
    idx = store({}).index()
    assert idx["runs"] == [] and idx["skipped"] == [] and idx["error"] == ""
    assert idx["facets"]["models"] == [] and idx["facets"]["max_num_seqs"] == []


def test_a_missing_directory_says_so_instead_of_showing_a_blank_page(tmp_path):
    idx = BenchStore(str(tmp_path / "not-there")).index()
    assert idx["runs"] == []
    assert "nothing has been saved" in idx["error"]


def test_newest_first(store):
    import os
    s = store({"old.json": synthetic("seqs08"), "new.json": synthetic("seqs24")})
    os.utime(s.dir / "old.json", (1, 1))
    os.utime(s.dir / "new.json", (2_000_000_000, 2_000_000_000))
    assert [r["id"] for r in s.index()["runs"]] == ["new", "old"]


def test_the_filters_only_offer_values_that_exist(store):
    f = store({"a.json": synthetic("seqs08"), "b.json": synthetic("seqs24"),
               "c.json": corpus("real-seqs16")}).index()["facets"]
    assert f["max_num_seqs"] == [8, 16, 24]
    assert f["kinds"] == ["corpus", "synthetic"]
    assert f["models"] == ["Qwen/Qwen3.8-Flash-Next"]
    assert f["concurrency"] == [32]


def test_a_reread_after_a_rewrite_sees_the_new_content(store):
    """The index caches on (mtime, size) — a harness overwriting a label must not be served the
    previous run's numbers."""
    import os
    s = store({"a.json": synthetic("seqs08")})
    assert s.index()["runs"][0]["curve"]["p10"] == 19.9
    (s.dir / "a.json").write_text(json.dumps(synthetic("seqs08", seqs_curve=(9.0, 8.0, 1.5, 20.0))))
    os.utime(s.dir / "a.json", (3_000_000_000, 3_000_000_000))
    assert s.index()["runs"][0]["curve"]["p10"] == 1.5


# ── the run id is a file name, so it is contained ──────────────────────────

@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "..%2f..%2fetc%2fpasswd", "/etc/passwd", "a/b", "a\\b", "", "   ",
    "x" * 300, "nul\x00l",
])
def test_no_run_id_escapes_the_results_directory(store, bad):
    s = store({"seqs08.json": synthetic("seqs08")})
    assert s.run(bad) is None


def test_a_symlink_out_of_the_directory_is_not_a_run(store, tmp_path):
    outside = tmp_path.parent / "secret.json"
    outside.write_text(json.dumps(synthetic("secret")))
    s = store({"seqs08.json": synthetic("seqs08")})
    (s.dir / "link.json").symlink_to(outside)
    assert s.run("link") is None


# ── the two routes ─────────────────────────────────────────────────────────

@pytest.fixture
def served(tmp_path):
    started: list[Proxy] = []

    def make(files: dict):
        bench = tmp_path / "bench"
        bench.mkdir(exist_ok=True)
        for name, body in files.items():
            (bench / name).write_text(body if isinstance(body, str) else json.dumps(body, indent=1))
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0, bench_dir=str(bench))
        proxy = Proxy(cfg, Hub(cfg))
        proxy.start()
        started.append(proxy)
        return proxy

    yield make
    for proxy in started:
        proxy.stop()


def call(proxy: Proxy, path: str, method: str = "GET"):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request(method, path)
    resp = conn.getresponse()
    body = resp.read()
    return resp.status, resp.getheader("Content-Type"), body


def test_the_index_route_answers_the_runs(served):
    proxy = served({"post-seqs24-rep1.json": synthetic()})
    status, ctype, body = call(proxy, "/lb/bench")
    assert status == 200 and "application/json" in ctype
    idx = json.loads(body)
    assert [r["id"] for r in idx["runs"]] == ["post-seqs24-rep1"]
    assert idx["facets"]["max_num_seqs"] == [24]
    # The index is a LIST; the series belongs to the run route, not to every row of the list.
    assert "series" not in idx["runs"][0]


def test_the_run_route_answers_one_run_with_its_series(served):
    proxy = served({"post-seqs24-rep1.json": synthetic()})
    status, _, body = call(proxy, "/lb/bench/run?id=post-seqs24-rep1")
    assert status == 200
    run = json.loads(body)["run"]
    assert len(run["series"]) == 3 and run["requests_detail"][0]["out_tokens"] == 512


def test_the_run_route_answers_a_prefill_probe(served):
    proxy = served({"B0-base-prefill.json": prefill(config=stamp(workload="prefill-probe"))})
    status, _, body = call(proxy, "/lb/bench/run?id=B0-base-prefill")
    assert status == 200
    run = json.loads(body)["run"]
    assert run["kind"] == "prefill" and run["tflops_per_gpu"] == 52.83 and run["series"] == []
    status, _, body = call(proxy, "/lb/bench")
    assert [r["id"] for r in json.loads(body)["runs"]] == ["B0-base-prefill"]


def test_a_run_that_does_not_exist_is_a_404_not_a_500(served):
    proxy = served({})
    status, _, body = call(proxy, "/lb/bench/run?id=nope")
    assert status == 404 and json.loads(body)["error"]["type"] == "not_found"


def test_no_id_is_the_callers_mistake_and_says_what_is_missing(served):
    proxy = served({})
    status, _, body = call(proxy, "/lb/bench/run")
    assert status == 400 and "`id`" in json.loads(body)["error"]["message"]


def test_a_traversing_id_over_the_wire_is_a_404(served):
    proxy = served({"seqs08.json": synthetic("seqs08")})
    for path in ("/lb/bench/run?id=../../../etc/passwd",
                 "/lb/bench/run?id=%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                 "/lb/bench/run?id=/etc/passwd"):
        status, _, body = call(proxy, path)
        assert status == 404, path


def test_the_page_is_served_at_the_operators_url_without_an_extension(served):
    proxy = served({})
    for path in ("/benchmarks", "/benchmarks/"):
        status, ctype, body = call(proxy, path)
        assert status == 200, path
        assert "text/html" in ctype
        assert b'<link rel="stylesheet" href="/benchmarks.css">' in body
        assert b'src="/benchmarks.js"' in body


def test_the_pages_own_assets_are_served(served):
    proxy = served({})
    for path, kind in (("/benchmarks.js", "javascript"), ("/benchmarks.css", "text/css")):
        status, ctype, _ = call(proxy, path)
        assert status == 200 and kind in ctype, path


def test_the_benchmark_routes_are_read_only(served):
    """Nothing under /lb/bench writes, and nothing there may be reached with a method that suggests
    it does. A POST ends at the gateway's own 405 and is never proxied to a backend."""
    proxy = served({})
    for method in ("POST", "PUT", "DELETE"):
        status, _, body = call(proxy, "/lb/bench", method)
        assert status == 405, method
        assert json.loads(body)["error"]["type"] == "method_not_allowed"


def test_the_gateway_page_links_to_the_benchmarks_page(served):
    """The operator was given a URL, but a page nobody can reach from the gateway is a page nobody
    opens twice."""
    proxy = served({})
    status, _, body = call(proxy, "/")
    assert status == 200 and b'href="/benchmarks"' in body
