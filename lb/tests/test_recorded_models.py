"""THE MODELS /benchmarks OFFERS ARE THE ONES THIS DEPLOYMENT RECORDED (GET /lb/perf `models`).

Operator, 2026-10-12, on a one-model deployment's page — five model segments, four of them "0 MIN", on
a box that has only ever served one model: "should not be hardcoded, added on the way when new model
loaded". The selector was built from the CATALOG (every model this code knows of) and
the log's ids appended to it, so every deployment showed every catalog model whether it had ever run it.

The rule: a model is offered when, and only when, the performance log holds a recorded minute of it in
the range asked — read off the same files the page reads its minutes from. It appears the first time a
window of it is written. The catalog only NAMES a model that has data; a model the catalog does not know
keeps its raw id and is never dropped; a log with no minute says so with an empty list.
"""
from __future__ import annotations

import http.client
import json
import time

import pytest

from vllm_lb import perflog
from vllm_lb.catalog import ModelCatalog
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.perflog import JsonlWriter, PerfLog
from vllm_lb.proxy import Proxy

T0 = 1_790_740_020.0          # 2026-09-30 05:47:00 UTC
DENSE = "qwen3.8-27b-nvidia-nvfp4"
FLASH = "qwen3.8-flash-next-hibrid48"
SGLANG = "qwen3.8-27b-sglang-dflash2"


def minute(t, model=None):
    w = {"v": 1, "t_start": t, "t_end": t + 60, "span_s": 60, "gen_tok_s": 10.0}
    if model is not None:
        w["config"] = {"model_choice": model, "max_num_seqs": 8}
    return w


def boot(t, model):
    return {"v": 1, "kind": "boot", "t_start": t, "t_end": t + 90, "generation": 1,
            "config": {"model_choice": model}, "outcome": "serving"}


def write(directory, records, prefix="perf"):
    w = JsonlWriter(str(directory), max_total_bytes=1 << 22, max_file_bytes=1 << 22, prefix=prefix)
    for r in records:
        w.write(r)


@pytest.fixture
def served(tmp_path):
    started = []

    def make():
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0, bench_dir=str(tmp_path / "bench"),
                     perf_dir=str(tmp_path / "perf"))
        proxy = Proxy(cfg, Hub(cfg))
        proxy.start()
        started.append(proxy)
        return proxy, tmp_path / "perf"

    yield make
    for p in started:
        p.stop()


def perf(proxy, query="days=all"):
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", f"/lb/perf?{query}")
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    assert resp.status == 200, body
    return body


def ids(body):
    return [m["id"] for m in body["models"]]


# ── the endpoint ──────────────────────────────────────────────────────────────────────────


def test_one_recorded_model_is_exactly_one_model(served):
    """307 minutes of the dense 27B and nothing else — one segment, never the catalog's six."""
    proxy, d = served()
    write(d, [minute(T0 + 60 * i, DENSE) for i in range(307)])
    body = perf(proxy)
    (m,) = body["models"]
    assert (m["id"], m["minutes"], m["known"]) == (DENSE, 307, True)
    assert m["label"] == ModelCatalog.load().get(DENSE).label, "the catalog NAMES a model that has data"
    assert (m["first"], m["last"]) == (T0, T0 + 60 * 306 + 60)


def test_three_recorded_models_are_three_in_the_order_each_was_first_recorded(served):
    proxy, d = served()
    write(d, [minute(T0, FLASH), minute(T0 + 60, SGLANG), minute(T0 + 120, FLASH),
              minute(T0 + 180, DENSE), minute(T0 + 240, SGLANG)])
    body = perf(proxy)
    assert ids(body) == [FLASH, SGLANG, DENSE], "a model is added the first time a minute of it is written"
    assert [m["minutes"] for m in body["models"]] == [2, 2, 1]


def test_a_catalog_model_without_a_recorded_minute_is_absent(served):
    """The catalog lists six models; a deployment that served one offers one. A model that only BOOTED
    (a boot record, no busy minute) has no minute to show either: it is not offered."""
    proxy, d = served()
    write(d, [minute(T0, DENSE)])
    write(d, [boot(T0 - 600, FLASH), boot(T0 - 300, DENSE)], prefix=perflog.BOOT_PREFIX)
    body = perf(proxy)
    catalog = {s.id for s in ModelCatalog.load().specs}
    assert len(catalog) > 1 and ids(body) == [DENSE]
    assert not (catalog - {DENSE}) & set(ids(body))
    assert all(m["minutes"] > 0 for m in body["models"]), "no 0-minute model, ever"


def test_a_model_the_catalog_does_not_know_keeps_its_raw_id_and_is_never_dropped(served):
    proxy, d = served()
    write(d, [minute(T0, "someones-own-finetune"), minute(T0 + 60, "someones-own-finetune"), minute(T0 + 120, DENSE)])
    body = perf(proxy)
    assert ids(body) == ["someones-own-finetune", DENSE]
    unknown = body["models"][0]
    assert (unknown["label"], unknown["known"], unknown["engine"], unknown["minutes"]) == (
        "someones-own-finetune", False, None, 2)


def test_a_recorded_model_names_its_engine_for_the_page_to_say_its_default(served):
    """A local node's launch that leaves the prefill budget unset runs the ENGINE's default, and the page says
    whose ("batched vLLM default") — the display name the panel already uses (saturation.ENGINE_NAMES), from
    the catalog's profile; a model the catalog does not know has no engine to name."""
    proxy, d = served()
    write(d, [minute(T0, DENSE), minute(T0 + 60, SGLANG), minute(T0 + 120, "someones-own-finetune")])
    by = {m["id"]: m for m in perf(proxy)["models"]}
    assert (by[DENSE]["engine"], by[DENSE]["engine_name"]) == ("vllm", "vLLM")
    assert (by[SGLANG]["engine"], by[SGLANG]["engine_name"]) == ("sglang", "SGLang")
    assert (by["someones-own-finetune"]["engine"], by["someones-own-finetune"]["engine_name"]) == (None, None)


def test_nothing_recorded_is_an_empty_list_not_the_catalog(served):
    proxy, _ = served()
    body = perf(proxy)
    assert body["models"] == [] and "nothing has been recorded" in body["error"]


def test_a_minute_that_named_no_model_is_still_recorded_and_still_offered(served):
    """It happened: the page files it under "model not recorded" (id ""), never drops it."""
    proxy, d = served()
    write(d, [minute(T0), minute(T0 + 60, DENSE)])
    body = perf(proxy)
    assert ids(body) == ["", DENSE] and body["models"][0]["known"] is False


def test_the_range_decides_which_models_were_recorded_in_it(served):
    proxy, d = served()
    now = time.time()                          # the endpoint's range is read against the real clock
    write(d, [minute(now - 5 * 86400, FLASH), minute(now - 600, DENSE)])
    assert ids(perf(proxy, "days=all")) == [FLASH, DENSE]
    assert ids(perf(proxy, "days=1")) == [DENSE], "a model with no minute in the range is no segment of it"


# ── the index under it: counted before the hand-over cut ───────────────────────────────────


def test_a_model_whose_every_minute_was_cut_is_still_a_recorded_model(tmp_path, monkeypatch):
    """/lb/perf hands over at most MAX_WINDOWS minutes; the cut ones are minutes this log really holds,
    so their model is listed with its real count — a long history keeps every model it recorded."""
    monkeypatch.setattr(perflog, "MAX_WINDOWS", 2)
    write(tmp_path, [minute(T0, FLASH), minute(T0 + 60, FLASH), minute(T0 + 120, DENSE), minute(T0 + 180, DENSE)])
    idx = PerfLog(str(tmp_path)).index(days=None, now=T0)
    assert [w["config"]["model_choice"] for w in idx["windows"]] == [DENSE, DENSE]
    assert [(m["id"], m["minutes"]) for m in idx["models"]] == [(FLASH, 2), (DENSE, 2)]
    assert idx["truncated_by_model"] == {FLASH: 2}
