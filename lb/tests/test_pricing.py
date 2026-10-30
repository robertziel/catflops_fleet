"""WHAT THE TOKENS WOULD HAVE COST (pricing.py, pricing.json, GET /lb/sessions `cost`).

Operator, 2026-10-05: "include price estimate based on model official pricing ... to know how much saved".

The rules that make a dollar figure safe to read: it is the vendor's published number with its date and
its source; a model with NO published price is unpriced (None) — never 0, never another model's price;
and a total that skips unpriced sessions says how many it covered.
"""
from __future__ import annotations

import copy
import http.client
import json
import pathlib

import pytest

from tests.fakes import FakeBackend
from vllm_lb.catalog import ModelCatalog
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.perflog import sessions
from vllm_lb.pricing import PRICING_PATH, Pricing, PricingError, apply
from vllm_lb.proxy import Proxy
from tests.test_sessions import SGLANG, T0, boot, window

FLASH_NEXT = "qwen3.8-flash-next-hibrid48"
OPUS, QWEN = "anthropic-claude-opus-5-5", "alibaba-qwen3.8-27b-sg"


def doc():
    return json.loads(pathlib.Path(PRICING_PATH).read_text())


def priced(windows, **kw):
    out = sessions(windows, [], describe=lambda m: {"label": m, "engine": "vllm"})
    return apply(out, kw.get("pricing") or Pricing.load())


# ── the arithmetic, to the cent ───────────────────────────────────────────────────────────


def test_a_session_is_priced_by_what_the_engine_computed_what_its_cache_served_and_what_it_generated():
    """1,000,000 prompt tokens of which 900,000 came from the prefix cache, and 100,000 generated, on
    Claude Opus 5.5 ($4 in / $20 out / $0.20 cache read per MTok):
        computed 100,000 x $4  = $0.40      cached 900,000 x $0.20 = $0.18      output 100,000 x $20 = $2.00
    = $2.58. On Qwen3.8-27B ($0.50 / $3.00 / $0.10): $0.05 + $0.09 + $0.30 = $0.44."""
    out = priced([window(gen=1, model=SGLANG, start=T0, prompt=1_000_000.0, hits=900_000.0,
                         gen_tokens=100_000.0)])
    (row,) = out["sessions"]
    assert row["cost"][OPUS] == 2.58
    assert row["cost"][QWEN] == 0.44
    assert row["own_price"] == {"list": QWEN, "why": ""}


def test_totals_and_rollups_are_sums_over_the_sessions_and_say_how_many_they_cover():
    out = priced([window(gen=1, model=SGLANG, start=T0, prompt=1_000_000.0, hits=900_000.0, gen_tokens=100_000.0),
                  window(gen=2, model=SGLANG, start=T0 + 600, prompt=1_000_000.0, hits=900_000.0,
                         gen_tokens=100_000.0)])
    c = out["totals"]["cost"]
    assert c[OPUS] == {"usd": 5.16, "priced": 2, "of": 2}
    assert c["own"] == {"usd": 0.88, "priced": 2, "of": 2}
    assert out["by_model"][0]["cost"][OPUS] == {"usd": 5.16, "priced": 2, "of": 2}


# ── NO FALLBACKS: an unpriced model is None, and a total says it left it out ──────────────


def test_a_model_with_no_published_price_is_unpriced_never_zero_and_never_another_models():
    """Qwen3.8-Flash-Next is not on Alibaba's list. Its OWN cost is None — not 0 (that would claim it
    cost nothing), not qwen3.8-flash's (a different model with an unpublished cache price)."""
    out = priced([window(gen=1, model=FLASH_NEXT, start=T0, gen_tokens=100_000.0)])
    (row,) = out["sessions"]
    assert QWEN not in row["cost"], "no list prices it: it must not be there at all, let alone as 0"
    assert row["cost"][OPUS] is not None, "the REFERENCE still prices it: that list applies to every model"
    assert row["own_price"]["list"] is None and "not on Alibaba" in row["own_price"]["why"]
    assert out["totals"]["cost"]["own"] == {"usd": None, "priced": 0, "of": 1}, "a sum of nothing is not $0"
    assert FLASH_NEXT in out["price_lists"]["unpriced"]


def test_a_total_over_a_mix_counts_only_what_it_priced_and_says_so():
    out = priced([window(gen=1, model=SGLANG, start=T0, gen_tokens=100_000.0),
                  window(gen=2, model=FLASH_NEXT, start=T0 + 600, gen_tokens=100_000.0)])
    own = out["totals"]["cost"]["own"]
    assert (own["priced"], own["of"]) == (1, 2), "one of two sessions priced — the page must say so"
    assert out["totals"]["cost"][OPUS]["priced"] == 2


def test_a_session_whose_log_carries_no_tokens_is_unpriced_not_free():
    bare = window(gen=1, model=SGLANG, start=T0)
    bare["prompt_computed_tokens"] = None
    (row,) = priced([bare])["sessions"]
    assert row["cost"][OPUS] is None and row["cost"][QWEN] is None


def test_a_hit_counter_that_ran_ahead_of_its_prompt_never_makes_a_negative_price():
    """Generation 123 recorded a 949% prefix hit rate: more hits than prompt tokens. The cached
    population is clamped at 0, so the price can be wrong-high but never negative."""
    w = window(gen=1, model=SGLANG, start=T0, prompt=1_000.0, hits=9_490.0)
    w["prompt_computed_tokens"] = 1_100.0                          # computed > billed, the same fault
    (row,) = priced([w])["sessions"]
    assert row["cost"][OPUS] >= 0 and row["cost"][QWEN] >= 0


# ── the document is strict, and a bad one is loud ─────────────────────────────────────────


def test_every_list_states_its_date_its_source_and_where_each_number_came_from():
    for pid, item in doc()["lists"].items():
        assert item["as_of"] and item["source"].startswith("https://") and item["basis"], pid
        assert set(item["usd_per_mtok"]) == {"input", "output", "cache_read"}, pid


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["lists"][OPUS].pop("as_of"), "as_of is required"),
    (lambda d: d["lists"][OPUS].pop("source"), "source is required"),
    (lambda d: d["lists"][OPUS]["usd_per_mtok"].pop("cache_read"), "must state exactly"),
    (lambda d: d["lists"][OPUS]["usd_per_mtok"].update(cache_read=9.0), "dearer than the input"),
    (lambda d: d["lists"][OPUS]["usd_per_mtok"].update(input=-1), "non-negative"),
    (lambda d: d["references"].append("no-such-list"), "references must name"),
    (lambda d: d["own"].update({"x": "no-such-list"}), "does not exist"),
    (lambda d: d["own"].update({"x": {"unpriced": ""}}), "a list id, or"),
])
def test_a_price_document_that_breaks_a_rule_is_refused_whole(mutate, match):
    d = copy.deepcopy(doc())
    mutate(d)
    with pytest.raises(PricingError, match=match):
        Pricing.from_dict(d)


def test_a_price_file_that_cannot_be_read_prices_nothing_and_says_why(tmp_path):
    (tmp_path / "pricing.json").write_text("{not json")
    p = Pricing.load(tmp_path / "pricing.json")
    assert p.error and p.reference_lists() == []
    out = apply(sessions([window(gen=1, model=SGLANG, start=T0)]), p)
    assert out["sessions"][0]["cost"] == {}, "no list, no figure — not a default"
    assert out["price_lists"]["error"] == p.error


def test_every_model_the_catalog_can_launch_has_a_pricing_decision():
    """A new model must be PRICED or declared unpriced WITH A REASON — otherwise it shows a dash for ever
    and nobody ever decided that. Also: no stale entry for a model that is gone."""
    catalog = ModelCatalog.load()
    ids = {m.id for m in catalog.specs}
    decided = set(Pricing.load().own)
    assert ids - decided == set(), f"models with no pricing decision: {sorted(ids - decided)}"
    assert decided - ids == set(), f"pricing entries for models the catalog does not have: {sorted(decided - ids)}"


# ── through the endpoint the page reads ───────────────────────────────────────────────────


@pytest.fixture
def served(tmp_path):
    d = tmp_path / "perf"
    d.mkdir()
    (d / "perf-2026-09-30.jsonl").write_text(json.dumps(
        window(gen=1, model=SGLANG, start=T0, prompt=1_000_000.0, hits=900_000.0, gen_tokens=100_000.0)) + "\n")
    (d / "boots-2026-09-30.jsonl").write_text(json.dumps(boot(gen=1, model=SGLANG, start=T0 - 130)) + "\n")
    b = FakeBackend()
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", b.url),), 
                 listen_host="127.0.0.1", listen_port=0, perf_dir=str(d))
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()
    b.close()


def test_the_endpoint_prices_the_sessions_and_carries_the_lists_it_used(served):
    c = http.client.HTTPConnection("127.0.0.1", served.port, timeout=5)
    c.request("GET", "/lb/sessions")
    d = json.loads(c.getresponse().read())
    c.close()
    (row,) = d["sessions"]
    assert row["cost"] == {QWEN: 0.44, OPUS: 2.58}
    assert d["totals"]["cost"][OPUS]["usd"] == 2.58
    lists = d["price_lists"]["lists"]
    assert lists[OPUS]["as_of"] and lists[OPUS]["source"] and lists[QWEN]["as_of"] == "2026-10-03"
    assert d["price_lists"]["references"] == [OPUS] and d["price_lists"]["error"] == ""
