"""The live shaping, against a verbatim `GET /lb/status` taken while the pair was serving.

The capture is from 2026-10-03 03:01, generation 123, qwen3.8-27b-sglang-dflash2 on SGLang at 16
seats - so these assertions are about what the LB really published, including that the shaped
answer is a small fraction of the 58 KB document it came from.
"""
from __future__ import annotations

import json

from usage_mcp import live


def test_the_capture_is_the_serving_state_it_claims_to_be(real_status: dict):
    assert real_status["cluster"]["state"] == "running"
    assert real_status["cluster"]["loaded_model"]["spec"] == "qwen3.8-27b-sglang-dflash2"
    assert real_status["cluster"]["effective"]["generation"] == 123


def test_the_shaped_snapshot_answers_the_question(real_status: dict):
    out = live.shape_status(real_status)
    assert out["serving"] is True
    assert out["model"]["served_name"] == "Qwen/Qwen3.8-27B-SGLang"
    assert out["model"]["engine"] == "sglang"
    assert out["model"]["generation"] == 123
    assert out["configuration"]["max_num_seqs"] == 16
    assert out["configuration"]["num_speculative_tokens"] == 4
    assert out["configuration"]["max_model_len"] == 262144
    assert out["rate"]["gen_tok_s"] > 0
    assert set(out["boxes"]) >= {"box-a", "box-b"}
    assert out["saturation"]["verdict"]
    assert out["swarm"]["last_session"]["label"].startswith("S-")


def test_the_engine_comes_from_the_lbs_own_models_view(real_status: dict):
    engines = live.engines_from_status(real_status)
    assert engines["qwen3.8-27b-sglang-dflash2"] == "sglang"
    # Entries whose catalog has no `engine` key are vLLM - the same default the LB fills in.
    assert engines["qwen3.8-flash-next-hibrid48"] == "vllm"
    assert set(engines.values()) <= {"vllm", "sglang"}, engines


def test_the_shaped_answer_is_far_smaller_than_the_document(real_status: dict):
    """A tool that returns /lb/status whole is worse than curl; this one must not."""
    raw = len(json.dumps(real_status))
    shaped = len(json.dumps(live.shape_status(real_status)))
    assert raw > 50_000, raw
    assert shaped < raw / 8, (shaped, raw)


def test_nothing_loaded_is_an_answer_not_an_error(real_status: dict):
    """The same shaping over an off cluster: `serving` false, a reason, and no exception."""
    off = json.loads(json.dumps(real_status))
    off["cluster"].update({"state": "unloaded", "detail": "the pair is down",
                           "loaded_model": None, "effective": {}})
    off["throughput_target"] = {}
    out = live.shape_status(off)
    assert out["serving"] is False
    assert out["model"] is None and out["configuration"] is None and out["rate"] is None
    assert out["cluster_state"] == "unloaded"
    assert any("Nothing is loaded" in c for c in out["caveats"]), out["caveats"]
    assert out["boxes"], "hardware is still reportable with the cluster off"


def test_a_status_missing_whole_sections_still_shapes(real_status: dict):
    for drop in ("cluster", "hardware", "backends", "saturation", "swarm", "throughput_target"):
        partial = {k: v for k, v in real_status.items() if k != drop}
        out = live.shape_status(partial)
        assert isinstance(out, dict) and "serving" in out, drop


def test_an_empty_bench_directory_is_reported_as_such(real_bench: dict):
    out = live.shape_bench(real_bench)
    assert out["saved_runs"] == 0
    assert out["runs"] == []
    assert out["error"] is None
    assert out["dir"]


def test_bench_rows_stay_compact_when_there_are_runs(real_bench: dict):
    doc = dict(real_bench)
    doc["runs"] = [{"name": f"run{i}", "model": "m", "kind": "decode", "concurrency": 8,
                    "gen_tok_s": 100.0 + i, "secret": "x" * 5000,
                    "config": {"max_num_seqs": 16}} for i in range(40)]
    out = live.shape_bench(doc, limit=5)
    assert out["saved_runs"] == 40 and out["shown"] == 5
    assert "secret" not in json.dumps(out)
    assert out["runs"][0]["max_num_seqs"] == 16


def test_every_url_that_leaves_has_no_credentials(real_status: dict):
    salted = json.loads(json.dumps(real_status))
    salted["cluster"]["url"] = "http://user:hunter2@192.0.2.2:8000"
    salted["backends"][0]["url"] = "http://admin:letmein@192.0.2.2:8000"
    blob = json.dumps(live.shape_status(salted))
    assert "hunter2" not in blob and "letmein" not in blob
    assert "192.0.2.2:8000" in blob, "the host itself is useful and should survive"
