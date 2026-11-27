"""THE FIDELITY BENCH (vllm_lb/fidelity.py): prompt-logprob NLL / perplexity of the LOADED model on a
frozen corpus, and the paired ΔNLL of two runs with its 95 % bootstrap interval.

A fake engine stands in for the model: it answers `/v1/completions` with echo + logprobs in the OpenAI legacy shape
vLLM and SGLang both return. The math, the storage, the refusals (an engine with no token logprobs, a second run while
one runs, nothing loaded, a cross-site POST) and the HTTP surface are pinned here."""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import threading
import time
from types import SimpleNamespace

import pytest

from tests.fakes import FakeBackend
from vllm_lb import fidelity, roles
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy


# ── the corpus ────────────────────────────────────────────────────────────────


def test_the_corpus_is_frozen_by_its_pinned_sha256():
    corpus = fidelity.load_corpus()
    raw = fidelity.CORPUS_PATH.read_bytes()
    assert corpus.sha256 == hashlib.sha256(raw).hexdigest() == fidelity.CORPUS_SHA256


def test_the_corpus_states_its_source_and_licence_in_its_header():
    head = fidelity.CORPUS_PATH.read_text().split("@@", 1)[0].lower()
    assert "public domain" in head and "source" in head


def test_every_document_names_its_source_and_has_text():
    corpus = fidelity.load_corpus()
    assert len(corpus.docs) >= 10, "a bootstrap over documents needs documents"
    ids = [d.id for d in corpus.docs]
    assert len(set(ids)) == len(ids)
    for d in corpus.docs:
        assert d.source and d.text.strip() and d.bytes == len(d.text.encode())
        assert "@@" not in d.text and not d.text.startswith("\n") and not d.text.endswith("\n")


def test_a_changed_corpus_is_refused_naming_both_hashes(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(fidelity.CORPUS_PATH.read_text() + "x")
    with pytest.raises(ValueError) as exc:
        fidelity.load_corpus(p)
    assert fidelity.CORPUS_SHA256[:12] in str(exc.value)


def test_a_corpus_of_no_documents_is_refused(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# only a header\n")
    with pytest.raises(ValueError):
        fidelity.load_corpus(p, expect_sha=hashlib.sha256(p.read_bytes()).hexdigest())


# ── one document's score ──────────────────────────────────────────────────────


def reply(lps, tokens=None, *, model="m", prompt_tokens=None, generated=True):
    """The legacy completions shape with echo: every prompt position, then the one generated token."""
    tokens = tokens or [f"t{i}" for i in range(len(lps))]
    lp = list(lps) + ([-0.25] if generated else [])
    toks = list(tokens) + (["<gen>"] if generated else [])
    return {"model": model, "choices": [{"text": "x", "logprobs": {"tokens": toks, "token_logprobs": lp,
                                                                       "top_logprobs": None, "text_offset": None}}],
            "usage": {"prompt_tokens": len(lps) if prompt_tokens is None else prompt_tokens, "completion_tokens": 1}}


def test_the_score_sums_the_prompt_positions_and_drops_the_first_and_the_generated():
    s = fidelity.score_reply(reply([None, -1.0, -2.0]), engine="vllm", served_name="m", doc_id="d")
    assert s["tokens"] == 2 and s["sum_nll"] == pytest.approx(3.0)
    assert len(s["tokens_sha"]) == 16


def test_a_bos_scored_position_zero_is_still_dropped():
    """Position 0 has no context: a number there (an engine that scores it against nothing) is not a prediction."""
    s = fidelity.score_reply(reply([-9.0, -1.0]), engine="vllm", served_name="m", doc_id="d")
    assert s["tokens"] == 1 and s["sum_nll"] == pytest.approx(1.0)


def test_the_tokenisation_hash_tells_two_tokenisations_apart():
    a = fidelity.score_reply(reply([None, -1.0], ["a", "b"]), engine="vllm", served_name="m", doc_id="d")
    b = fidelity.score_reply(reply([None, -1.0], ["a", "c"]), engine="vllm", served_name="m", doc_id="d")
    assert a["tokens_sha"] != b["tokens_sha"]


@pytest.mark.parametrize("bad, words", [
    ({"choices": [{"text": "x", "logprobs": None}], "usage": {"prompt_tokens": 3}}, "no token logprobs"),
    ({"choices": [{"text": "x"}], "usage": {"prompt_tokens": 3}}, "no token logprobs"),
    ({"choices": []}, "no choices"),
    ({"choices": [{"logprobs": {"tokens": ["a"], "token_logprobs": [None]}}], "usage": {"prompt_tokens": 1}},
     "fewer than two"),
])
def test_an_engine_that_returns_no_token_logprobs_is_an_error_naming_it(bad, words):
    with pytest.raises(fidelity.FidelityError) as exc:
        fidelity.score_reply(bad, engine="sglang", served_name="m", doc_id="genesis")
    msg = str(exc.value)
    assert "sglang" in msg and "genesis" in msg and words in msg


def test_a_null_inside_the_prompt_is_an_error_not_a_skipped_token():
    with pytest.raises(fidelity.FidelityError) as exc:
        fidelity.score_reply(reply([None, -1.0, None, -2.0]), engine="vllm", served_name="m", doc_id="d")
    assert "position 2" in str(exc.value)


def test_no_prompt_token_count_is_an_error_not_a_guess_at_where_the_prompt_ends():
    r = reply([None, -1.0])
    del r["usage"]
    with pytest.raises(fidelity.FidelityError) as exc:
        fidelity.score_reply(r, engine="vllm", served_name="m", doc_id="d")
    assert "prompt_tokens" in str(exc.value)


def test_an_answer_from_another_model_is_refused_never_scored():
    with pytest.raises(fidelity.FidelityError) as exc:
        fidelity.score_reply(reply([None, -1.0], model="other"), engine="vllm", served_name="m", doc_id="d")
    assert "other" in str(exc.value) and "m" in str(exc.value)


# ── a whole run, against a fake engine ────────────────────────────────────────


def fake_engine(shift=0.0, *, model="m", calls=None, gate=None):
    """A model whose per-token logprob is a function of the document: deterministic, distinct per doc."""
    def post(body, timeout):
        if calls is not None:
            calls.append(body)
        if gate is not None:
            gate.wait(10)
        words = body["prompt"].split()
        lps = [None] + [-(1.0 + (len(w) % 5) * 0.1) - shift for w in words[1:]]
        return 200, reply(lps, words, model=model)
    return post


META = {"server": "pair", "served_name": "m", "model": "qwen-x", "model_label": "Qwen X", "engine": "vllm",
        "accelerator": "acc-a", "tp": 2, "quant": "modelopt", "kv_cache_dtype": "fp8_e4m3", "why": {}}


def bench(tmp_path, post, meta=META):
    return fidelity.FidelityBench(fidelity.FidelityStore(str(tmp_path / "fid")), post=post,
                                  describe=lambda server=None: dict(meta))


def wait_done(b, timeout=10):
    t0 = time.time()
    while b.view()["active"] is not None:
        assert time.time() - t0 < timeout, "the run did not finish"
        time.sleep(0.02)
    return b.view()["runs"][0]


def test_a_run_scores_every_document_and_stores_the_totals(tmp_path):
    calls = []
    b = bench(tmp_path, fake_engine(calls=calls))
    started = b.start()
    assert started["status"] == "running" and started["meta"]["engine"] == "vllm"
    run = wait_done(b)
    corpus = fidelity.load_corpus()
    assert run["status"] == "done" and run["error"] is None
    assert len(run["docs"]) == len(corpus.docs) == len(calls)
    assert all(c["echo"] is True and c["logprobs"] == 1 and c["max_tokens"] == 1 and c["model"] == "m"
               and c["stream"] is False for c in calls)
    tokens = sum(d["tokens"] for d in run["docs"])
    nll = sum(d["sum_nll"] for d in run["docs"])
    assert run["tokens"] == tokens
    assert run["mean_nll"] == pytest.approx(nll / tokens, abs=1e-6)
    assert run["ppl"] == pytest.approx(math.exp(nll / tokens), rel=1e-6)
    assert run["bits_per_byte"] == pytest.approx(nll / math.log(2) / sum(d.bytes for d in corpus.docs), rel=1e-6)
    assert run["corpus"]["sha256"] == fidelity.CORPUS_SHA256
    assert run["meta"] == META and run["done_docs"] == len(corpus.docs)


def test_a_run_survives_a_restart_on_disk(tmp_path):
    b = bench(tmp_path, fake_engine())
    b.start()
    run = wait_done(b)
    again = fidelity.FidelityStore(str(tmp_path / "fid"))
    assert again.get(run["id"]) == run
    assert again.runs()[0]["id"] == run["id"]


def test_the_engine_with_no_logprobs_fails_the_run_naming_it_and_stores_the_failure(tmp_path):
    def post(body, timeout):
        return 200, {"model": "m", "choices": [{"text": "x", "logprobs": None}], "usage": {"prompt_tokens": 9}}
    b = bench(tmp_path, post, {**META, "engine": "tensorfold"})
    b.start()
    run = wait_done(b)
    assert run["status"] == "failed"
    assert "tensorfold" in run["error"] and "no token logprobs" in run["error"]
    assert run["mean_nll"] is None and run["ppl"] is None, "a failed run carries no figure"


def test_an_http_error_from_the_gateway_fails_the_run_with_its_status_and_words(tmp_path):
    b = bench(tmp_path, lambda body, timeout: (503, {"error": {"message": "no backend"}}))
    b.start()
    run = wait_done(b)
    assert run["status"] == "failed" and "503" in run["error"] and "no backend" in run["error"]


def test_a_second_run_while_one_runs_is_refused(tmp_path):
    gate = threading.Event()
    b = bench(tmp_path, fake_engine(gate=gate))
    b.start()
    try:
        with pytest.raises(fidelity.Busy):
            b.start()
    finally:
        gate.set()
    wait_done(b)
    b.start()                                         # free again once it ended
    wait_done(b)


def test_nothing_loaded_is_refused_before_anything_is_written(tmp_path):
    def describe(server=None):
        raise fidelity.NothingLoaded("no vLLM is serving on this gateway")
    b = fidelity.FidelityBench(fidelity.FidelityStore(str(tmp_path / "fid")), post=fake_engine(), describe=describe)
    with pytest.raises(fidelity.NothingLoaded):
        b.start()
    assert b.view()["runs"] == [] and b.view()["active"] is None


def test_a_run_left_running_by_a_restart_reads_as_interrupted(tmp_path):
    store = fidelity.FidelityStore(str(tmp_path / "fid"))
    store.save({"id": "fid-20261127T000000Z-abcd", "status": "running", "started_at": 1.0, "error": None})
    run = fidelity.FidelityBench(store, post=fake_engine(), describe=lambda server=None: META).view()["runs"][0]
    assert run["status"] == "interrupted" and "restart" in run["error"]


def test_an_unreadable_run_file_is_listed_with_its_reason_not_dropped(tmp_path):
    store = fidelity.FidelityStore(str(tmp_path / "fid"))
    (tmp_path / "fid").mkdir()
    (tmp_path / "fid" / "fid-20261127T000000Z-abcd.json").write_text("{trunc")
    runs = store.runs()
    assert runs[0]["status"] == "unreadable" and runs[0]["error"]


@pytest.mark.parametrize("bad", ["../x", "fid-1/../../etc", "", "x" * 300, "fid-20261127T000000Z-ABCD"])
def test_a_run_id_that_is_not_one_names_nothing(tmp_path, bad):
    assert fidelity.FidelityStore(str(tmp_path)).get(bad) is None


# ── the paired comparison ─────────────────────────────────────────────────────


def run_of(docs, sha=fidelity.CORPUS_SHA256, status="done"):
    return {"id": "r", "status": status, "corpus": {"sha256": sha}, "docs": docs}


def doc(i, tokens, sum_nll, sha="s"):
    return {"id": f"d{i}", "tokens": tokens, "sum_nll": sum_nll, "tokens_sha": sha, "bytes": tokens * 4}


def test_identical_runs_have_zero_delta_and_same_quality():
    docs = [doc(i, 100 + i, 150.0 + i * 3) for i in range(12)]
    c = fidelity.compare(run_of(docs), run_of(docs))
    assert c["comparable"] and c["d_nll"] == 0 and c["ci95"] == [0, 0] and c["same_quality"] is True
    assert c["matched_docs"] == 12 and c["mismatched_docs"] == 0


def test_a_consistent_shift_is_measured_and_its_interval_excludes_zero():
    a = [doc(i, 100, 150.0 + i) for i in range(12)]
    b = [doc(i, 100, 150.0 + i + 2.0 + (i % 3) * 0.5) for i in range(12)]     # B worse by 0.02-0.03 nat/token
    c = fidelity.compare(run_of(a), run_of(b))
    assert c["d_nll"] == pytest.approx((sum(d["sum_nll"] for d in b) - sum(d["sum_nll"] for d in a)) / 1200, abs=1e-4)
    lo, hi = c["ci95"]
    assert 0 < lo <= c["d_nll"] <= hi
    assert c["same_quality"] is False


def test_a_noisy_difference_spans_zero_and_reads_as_same_quality():
    a = [doc(i, 100, 150.0) for i in range(12)]
    b = [doc(i, 100, 150.0 + (3.0 if i % 2 else -3.0)) for i in range(12)]
    c = fidelity.compare(run_of(a), run_of(b))
    lo, hi = c["ci95"]
    assert lo < 0 < hi and c["same_quality"] is True


def test_the_bootstrap_is_reproducible():
    a = [doc(i, 90 + i, 150.0 + i * 1.7) for i in range(12)]
    b = [doc(i, 90 + i, 151.0 + i * 1.5) for i in range(12)]
    assert fidelity.compare(run_of(a), run_of(b)) == fidelity.compare(run_of(a), run_of(b))


def test_documents_tokenised_differently_are_left_out_and_counted():
    a = [doc(i, 100, 150.0) for i in range(12)]
    b = [doc(i, 100, 150.0, sha="other" if i < 3 else "s") for i in range(12)]
    c = fidelity.compare(run_of(a), run_of(b))
    assert c["matched_docs"] == 9 and c["mismatched_docs"] == 3


@pytest.mark.parametrize("a, b, words", [
    (run_of([doc(0, 1, 1.0)]), run_of([doc(0, 1, 1.0)], sha="other"), "different corpus"),
    (run_of([doc(0, 1, 1.0)], status="failed"), run_of([doc(0, 1, 1.0)]), "not a finished run"),
    (run_of([doc(0, 1, 1.0, sha="x")]), run_of([doc(0, 1, 1.0, sha="y")]), "tokenised identically"),
])
def test_runs_that_cannot_be_paired_say_why(a, b, words):
    c = fidelity.compare(a, b)
    assert c["comparable"] is False and words in c["reason"]


# ── what a run records about the loaded model ─────────────────────────────────


def spec(*, engine="sglang", accelerator="acc-a", tp=2, kv="fp8_e4m3", resolved=True):
    target = SimpleNamespace(accelerator=SimpleNamespace(key=accelerator), tp=tp)
    profile = SimpleNamespace(engine=engine, kv_cache_dtype=kv,
                              resolution=SimpleNamespace(engine=engine, target=target) if resolved else None)
    return SimpleNamespace(id="qwen-x", label="Qwen X", served_name="m", quant_format="modelopt",
                           profile=lambda mode: profile)


def test_a_cluster_is_described_by_its_engine_target_quant_and_running_kv_dtype():
    view = {"state": "running", "loaded_model": {"spec": "qwen-x", "served_name": "m"},
            "effective": {"kv_cache_dtype": "fp4_e2m1"}}
    m = fidelity.describe_cluster("pair", view, spec())
    assert m == {"server": "pair", "served_name": "m", "model": "qwen-x", "model_label": "Qwen X", "engine": "sglang",
                 "accelerator": "acc-a", "tp": 2, "quant": "modelopt", "kv_cache_dtype": "fp4_e2m1", "why": {}}


def test_a_run_that_kept_the_profiles_kv_dtype_records_that_dtype():
    view = {"loaded_model": {"spec": "qwen-x", "served_name": "m"}, "effective": {"kv_cache_dtype": None}}
    assert fidelity.describe_cluster("pair", view, spec())["kv_cache_dtype"] == "fp8_e4m3"


def test_what_cannot_be_read_is_absent_with_why_never_a_default():
    view = {"loaded_model": {"spec": "gone", "served_name": "m"}, "effective": None}
    m = fidelity.describe_cluster("pair", view, None)
    assert m["served_name"] == "m"
    for key in ("engine", "accelerator", "tp", "quant", "kv_cache_dtype"):
        assert m[key] is None and m["why"][key]
    m = fidelity.describe_cluster("pair", view, spec(resolved=False))
    assert m["engine"] == "sglang" and m["accelerator"] is None and m["tp"] is None and m["why"]["accelerator"]


# ── over HTTP ─────────────────────────────────────────────────────────────────


@pytest.fixture
def backend():
    b = FakeBackend()
    b.reply_json = reply([None, -1.0, -1.5, -0.5], model="qwen3.8:latest")
    yield b
    b.close()


def serving(backend, tmp_path, *, live=True):
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", backend.url),),
                 listen_host="127.0.0.1", listen_port=0, perf_dir=str(tmp_path / "perf"))
    hub = Hub(cfg)
    if live:
        hub.update_backend(backend.url, live=True, served_models=("qwen3.8:latest",), last_probe_at=time.time())
    p = Proxy(cfg, hub)
    p.start()
    return p


def call(proxy, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    c.request(method, path, json.dumps(body) if body is not None else None, hdrs)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"null")


def poll_done(p, timeout=20):
    t0 = time.time()
    while True:
        _, view = call(p, "GET", "/lb/fidelity")
        if view["active"] is None:
            return view
        assert time.time() - t0 < timeout
        time.sleep(0.05)


def test_a_click_runs_the_corpus_through_the_gateway_against_the_loaded_model(backend, tmp_path):
    p = serving(backend, tmp_path)
    try:
        status, body = call(p, "POST", "/lb/fidelity/run", {})
        assert status == 202 and body["run"]["status"] == "running"
        view = poll_done(p)
        run = view["runs"][0]
        assert run["status"] == "done", run
        assert run["meta"]["served_name"] == "qwen3.8:latest" and run["meta"]["engine"] == "vllm"
        assert run["meta"]["accelerator"] is None and run["meta"]["why"]["accelerator"]
        sent = [json.loads(r["body"]) for r in backend.requests if r["path"] == "/v1/completions"]
        assert len(sent) == len(fidelity.load_corpus().docs)
        assert all(s["model"] == "qwen3.8:latest" and s["echo"] is True for s in sent)
        assert view["corpus"]["docs"] == len(sent) and view["corpus"]["sha256"] == fidelity.CORPUS_SHA256
    finally:
        p.stop()


def test_a_second_click_while_a_run_runs_is_a_409(backend, tmp_path):
    backend.first_byte_delay_s = 0.3
    p = serving(backend, tmp_path)
    try:
        assert call(p, "POST", "/lb/fidelity/run", {})[0] == 202
        status, body = call(p, "POST", "/lb/fidelity/run", {})
        assert status == 409 and body["error"]["type"] == "fidelity_busy"
    finally:
        backend.first_byte_delay_s = 0
        poll_done(p, timeout=30)
        p.stop()


def test_nothing_loaded_is_a_409_and_writes_nothing(backend, tmp_path):
    p = serving(backend, tmp_path, live=False)
    try:
        status, body = call(p, "POST", "/lb/fidelity/run", {})
        assert status == 409 and body["error"]["type"] == "fidelity_nothing_loaded"
        assert call(p, "GET", "/lb/fidelity")[1]["runs"] == []
    finally:
        p.stop()


def test_a_cross_site_shaped_post_is_refused(backend, tmp_path):
    p = serving(backend, tmp_path)
    try:
        c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=10)
        c.request("POST", "/lb/fidelity/run", "{}", {"Content-Type": "text/plain"})
        assert c.getresponse().status == 415
        assert call(p, "GET", "/lb/fidelity")[1]["active"] is None
    finally:
        p.stop()


def test_the_start_is_post_only_and_admin_only(backend, tmp_path):
    p = serving(backend, tmp_path)
    try:
        assert call(p, "GET", "/lb/fidelity/run")[0] in (404, 405)
        status, _ = call(p, "POST", "/lb/fidelity/run", {}, {roles.ROLE_HEADER: roles.MEMBER,
                                                              roles.MODELS_HEADER: "qwen3.8:latest"})
        assert status == 403
        assert call(p, "GET", "/lb/fidelity")[1]["active"] is None
    finally:
        p.stop()


def test_the_compare_route_pairs_two_runs(backend, tmp_path):
    p = serving(backend, tmp_path)
    try:
        call(p, "POST", "/lb/fidelity/run", {})
        a = poll_done(p)["runs"][0]["id"]
        call(p, "POST", "/lb/fidelity/run", {})
        runs = poll_done(p)["runs"]
        b = runs[0]["id"]
        assert b != a
        status, c = call(p, "GET", f"/lb/fidelity/compare?a={a}&b={b}")
        assert status == 200 and c["comparable"] and c["d_nll"] == 0 and c["same_quality"] is True
        assert call(p, "GET", f"/lb/fidelity/compare?a={a}&b=fid-nope")[0] == 404
        assert call(p, "GET", "/lb/fidelity/compare?a=")[0] == 400
    finally:
        p.stop()


# ── the bootstrap's fixed seed, pinned ────────────────────────────────────────

#: A noisy pair: per-document deltas that disagree in size and sign, so every resample set moves the interval.
#: compare(SEED_A, SEED_B)["ci95"] under BOOTSTRAP_SEED 20261105 and 2000 reps.
PINNED_CI95 = [-0.0295, 0.0307]
SEED_A = [doc(i, 80 + 7 * i, 120.0 + 11.3 * i) for i in range(12)]
SEED_B = [doc(i, 80 + 7 * i, 120.0 + 11.3 * i + ((i * 37) % 11 - 4.5) * 1.9) for i in range(12)]


def test_two_compares_of_the_same_pair_give_the_identical_interval():
    """The page re-asks on every tick; an interval that moved between two asks of the same two runs would be noise
    posing as a measurement. Distinct compares, so nothing is cached between them."""
    first = fidelity.compare(run_of(SEED_A), run_of(SEED_B))["ci95"]
    for _ in range(3):
        assert fidelity.compare(run_of(list(SEED_A)), run_of(list(SEED_B)))["ci95"] == first


def test_the_interval_of_a_fixed_input_is_the_pinned_one():
    """BOOTSTRAP_SEED and BOOTSTRAP_REPS make the interval a function of the two runs alone: changing either (or
    dropping the seed) changes these digits, deliberately."""
    c = fidelity.compare(run_of(SEED_A), run_of(SEED_B))
    assert c["ci95"] == PINNED_CI95
