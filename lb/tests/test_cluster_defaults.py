"""C1/C2/C4 — the pair's measured ceiling, and the setup the page offers when nothing is stored.

The numbers are NOT derived here. They are quoted from `worklog/vllm-cluster-ram-headroom.md`, the
read-only audit of box-a and box-b taken while generation 5 was serving (merged `1812ed0`):

  * the KV pool is PINNED by `--kv-cache-memory 28000000000` per rank and does not grow with
    `max_num_seqs` — vLLM logs the identical `GPU KV cache size: 1,708,119 tokens, Maximum
    concurrency for 262,144 tokens per request: 6.52x` at seqs=1 (generation 1) and seqs=2
    (generation 5);
  * the per-token cost is 16,392 B per rank (§5.1), measured: 12,288 B architectural x 1.334 for
    the hybrid's mamba page padding;
  * so the ceiling is floor(kv_tokens / max_model_len) — §5.2's table, reproduced row for row below.

The point of the pass: today the shipped default is 16, which needs 4,194,304 KV tokens against a
pool of 1,708,150. The page must offer what FITS, and must never disturb a pair already running.
"""
from __future__ import annotations

import json
import re

from pathlib import Path

import pytest

from vllm_lb.cluster import CLASSIC, CLUSTER_MODE, MAX_NUM_SEQS_RANGE, Desired, DesiredStore
from vllm_lb.cluster_defaults import (CLUSTER_KV_CACHE_MEMORY, KV_BYTES_PER_TOKEN_PER_RANK, desired_store,
                                      kv_tokens_per_rank, max_num_seqs_that_fits, recommend)
from vllm_lb.config import Config

#: vLLM's own line for the shipped pool, quoted from the audit §4.
VLLM_REPORTED_TOKENS = 1_708_119
#: What the shipped pool holds by the measured per-token cost: 28e9 // 16392.
POOL_TOKENS = 1_708_150


# ── C1: the ceiling is derived, not magic ─────────────────────────────────────


def test_the_shipped_pool_reproduces_vllms_own_token_count():
    """The calibration that makes every row below trustworthy: our arithmetic must land on the
    number vLLM printed, not near it. 1,708,150 vs 1,708,119 is 0.002 %."""
    tokens = kv_tokens_per_rank(CLUSTER_KV_CACHE_MEMORY)
    assert tokens == POOL_TOKENS
    assert abs(tokens - VLLM_REPORTED_TOKENS) / VLLM_REPORTED_TOKENS < 0.0001


def test_the_measured_per_token_cost_is_the_audits():
    assert (KV_BYTES_PER_TOKEN_PER_RANK, CLUSTER_KV_CACHE_MEMORY) == (16_392, 28_000_000_000)


@pytest.mark.parametrize("max_model_len, fits", [
    # worklog/vllm-cluster-ram-headroom.md §5.2, every row. The three the frame names are 262144,
    # 131072 and 65536; the rest guard the arithmetic between them.
    (262_144, 6),      # vLLM's own "6.52x" — the full trained window, 3x what generation 5 runs
    (196_608, 8),      # 8.69x
    (131_072, 13),     # 13.03x
    (98_304, 17),      # 17.38x
    (65_536, 26),      # 26.06x
])
def test_the_audits_table_row_for_row(max_model_len, fits):
    assert max_num_seqs_that_fits(POOL_TOKENS, max_model_len) == fits


def test_the_ratio_is_floored_never_rounded_up():
    """6.52x is six seats, not seven: the seventh does not fit, and vLLM would queue it."""
    assert POOL_TOKENS / 262_144 == pytest.approx(6.52, abs=0.01)
    assert max_num_seqs_that_fits(POOL_TOKENS, 262_144) == 6
    assert max_num_seqs_that_fits(199, 100) == 1, "1.99 seats is one seat"
    assert max_num_seqs_that_fits(200, 100) == 2, "exactly two fit"


def test_a_context_larger_than_the_whole_pool_still_offers_one_seat():
    """Not zero: tp-rank refuses max_num_seqs below 1, and vLLM QUEUES past the pool rather than
    failing to launch (audit §5.2). One seat is the honest floor."""
    assert max_num_seqs_that_fits(POOL_TOKENS, POOL_TOKENS + 1) == 1
    assert max_num_seqs_that_fits(1_000, 262_144) == 1


def test_a_pool_of_zero_offers_one_seat_and_does_not_crash():
    assert kv_tokens_per_rank(0) == 0
    assert max_num_seqs_that_fits(0, 262_144) == 1


def test_the_ceiling_never_exceeds_what_tp_rank_accepts():
    """tp-rank validates max_num_seqs in 1..48 (`cmd_up`), and the audit's own table marks the
    32,768 and 16,384 rows 'over the cap'. The ceiling is clamped, never posted out of range."""
    assert max_num_seqs_that_fits(POOL_TOKENS, 32_768) == MAX_NUM_SEQS_RANGE[1] == 48
    assert max_num_seqs_that_fits(POOL_TOKENS, 16_384) == 48, "104 fits the pool but not tp-rank"


@pytest.mark.parametrize("bad", [0, -1, -262_144])
def test_a_non_positive_context_is_refused_rather_than_dividing_by_zero(bad):
    with pytest.raises(ValueError):
        max_num_seqs_that_fits(POOL_TOKENS, bad)


def test_the_recommendation_carries_its_own_evidence():
    """The page shows a number the operator did not choose, so it must be able to say where it came
    from: the pool it was computed against, and the audit that measured it."""
    r = recommend(CLUSTER_KV_CACHE_MEMORY, 262_144)
    assert (r.max_num_seqs, r.max_model_len, r.kv_tokens) == (6, 262_144, POOL_TOKENS)
    assert "vllm-cluster-ram-headroom" in r.source
    assert r.view() == {"max_num_seqs": 6, "max_model_len": 262_144, "kv_tokens": POOL_TOKENS,
                        "kv_cache_memory": CLUSTER_KV_CACHE_MEMORY, "source": r.source}


# ── C2: the page comes up on it ───────────────────────────────────────────────


def test_the_cold_start_switch_offers_the_default_rows_own_seats_and_context_not_a_gateway_number():
    """With nothing stored, the switch offers the catalog's default cluster row its own `tuned` seats and context —
    6 at 262,144 for Flash-Next (its row's measurement). The LB has no gateway-wide concurrency or window: a number
    from the environment cannot know which model is about to be launched (operator, 2026-10-29)."""
    from vllm_lb.catalog import document, ModelCatalog
    catalog = ModelCatalog.load()
    tuned = catalog.default(CLUSTER_MODE).profile(CLUSTER_MODE).tuned
    s = desired_store(Config(cluster_state_path="/nonexistent/cluster.json"), catalog=catalog)
    assert (s.load().max_num_seqs, s.load().max_model_len) == (tuned.max_num_seqs, tuned.max_model_len) == (6, 262_144)
    for gone in ("cluster_max_num_seqs", "cluster_max_model_len", "cluster_default_max_num_seqs"):
        assert not hasattr(Config(), gone), gone


@pytest.mark.parametrize("key, field", [("LB_CLUSTER_MAX_NUM_SEQS", "tuned.max_num_seqs"),
                                        ("LB_CLUSTER_MAX_MODEL_LEN", "tuned.max_model_len")])
def test_the_gateway_wide_seats_and_context_are_retired_naming_the_row_field(key, field):
    """Set to anything — a blank line too — the boot is refused naming the catalog row field that owns the value."""
    for value in ("12", "131072", "   ", ""):
        with pytest.raises(ValueError, match=key) as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: value})
        assert "catalog" in str(e.value) and field in str(e.value), str(e.value)


def test_the_recommendation_follows_the_models_own_context():
    """Halve the window and nearly twice as many sessions fit — the ceiling is not a constant. 11, not the
    token rule's 13: by the engine's blocks 1,225 // (82 + 25) — the 25 fixed blocks a request takes at any
    context are why halving the window does not quite double the seats. The context is the ROW's tuned one
    (or the caller's), never a gateway setting."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import recommended_for
    doc = document()
    next(m for m in doc["models"] if m["id"] == FLASH)["launch"]["cluster"]["tuned"]["max_model_len"] = 131072
    halved = ModelCatalog.from_dict(doc).get(FLASH)
    assert recommended_for(Config(), halved).max_num_seqs == 11
    assert recommended_for(Config(), ModelCatalog.load().get(FLASH)).max_num_seqs == 6
    assert recommended_for(Config(), ModelCatalog.load().get(FLASH), max_model_len=131072).max_num_seqs == 11


HOSTS = ("box-a-gpu=box-a=user@192.0.2.2=192.0.2.201,"
         "box-b-gpu=box-b=user@192.0.2.12=192.0.2.202")


def test_before_the_first_tick_the_status_already_offers_the_recommendation():
    """hub.py publishes a cluster view for the ~2 s between the proxy listening and the cluster
    loop's first tick. Until this pass it carried the hardcoded 16 as `max_depth`, so a page loaded
    in that window offered — and a router admitted — a concurrency the pool cannot hold."""
    from vllm_lb.hub import Hub

    from vllm_lb.cluster import parse_hosts
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url="http://192.0.2.2:8100")
    from vllm_lb.catalog import ModelCatalog, document
    view = Hub(cfg, cluster_spec=ModelCatalog.load().default(CLUSTER_MODE)).snapshot().extra["cluster"]
    assert view["configured"] is True
    assert view["max_depth"] == 6, "the pre-tick admission depth is the default row's own tuned seats"
    assert view["recommended"]["max_num_seqs"] == 6
    # No catalog row to offer (the catalog did not load): no seats are claimed, so the router admits nothing there.
    view = Hub(cfg).snapshot().extra["cluster"]
    assert view["max_depth"] is None and view["recommended"] is None


def test_the_pre_tick_recommendation_is_for_the_model_the_switch_holds():
    """Audit #7, the Hub's half: its placeholder view recommended for one configured constant whatever model
    the switch held. `__main__` hands it the stored ModelSpec; with none it is the configured pool, as before."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.hub import Hub

    from vllm_lb.cluster import parse_hosts
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url="http://192.0.2.2:8100")
    dense = ModelCatalog.load().get("qwen3.8-27b-nvidia-nvfp4")
    r = Hub(cfg, cluster_spec=dense).snapshot().extra["cluster"]["recommended"]
    assert (r["kv_cache_memory"], r["kv_tokens"], r["max_num_seqs"]) == (48_000_000_000, 1_302_224, 4)
    assert Hub(cfg).snapshot().extra["cluster"]["recommended"] is None, "no model held: nothing recommended"


def test_the_deployment_offers_each_model_its_own_measured_best_start_input():
    """SUPERSEDES the 2026-09-30 rule for a model that has been measured. That rule — operator: "Set as default
    variables that were set at the beginning before benchmarks" — made the deployment's own three settings the
    default for EVERY model: concurrency 8 (LB_CLUSTER_MAX_NUM_SEQS), prefill budget 8192
    (CLUSTER_MAX_NUM_BATCHED_TOKENS), the full 262,144 window (LB_CLUSTER_MAX_MODEL_LEN), and the catalog
    LaunchProfile's 4 speculative tokens. They are all still there, still read from the mechanism that owns
    them and never hardcoded in the page or the store, and they are still exactly what a model with nothing
    measured is offered (the dense 27B, below).

    What changed on 2026-10-05, at the operator's request: a model whose best start input has been MEASURED is
    offered that instead — one number from the environment cannot know which model is about to be launched.
    Flash-Next: 6 seats, a 4,096 budget, depth 4 (`catalog.TunedSetup`, whose `source` names the runs)."""
    import re

    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    from vllm_lb.cluster_exec import resolved_launch

    catalog = ModelCatalog.load()
    flash = catalog.default(CLUSTER_MODE)
    box = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"}
    store = desired_store(Config(cluster_state_path="/nonexistent/cluster.json"), catalog=catalog)
    base = store.defaults_for(flash.id)
    offered = resolved_launch(flash, box, max_num_seqs=base.max_num_seqs, max_model_len=base.max_model_len,
                              uncensored=base.uncensored)
    assert offered == {"max_num_seqs": 6, "max_model_len": 262_144, "max_num_batched_tokens": 4096,
                       # The PLE table split, the profile's own placement since 2026-10-11
                       # (tests/test_flash_next_ple_split.py).
                       "num_speculative_tokens": 4, "ple_replicate": False, "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                       "long_prefill_token_threshold": 0,
                       # Its uncensored twin: the checkpoint a Start of it launches (operator, 2026-10-07).
                       "uncensored": True,
                       # vLLM's own: the profile names no KV dtype, so none is sent.
                       "kv_cache_dtype": "fp8_e4m3"}
    assert store.load().max_num_seqs == 6, "the cold-start switch offers the default model's own"
    # A ROW NOTHING WAS BENCHED FOR states what it is offered too — the dense 27B's row carries the figures the
    # gateway's former defaults gave it (6 seats at 262,144: the default row's pool rule at the full window, written
    # into the row with that source on 2026-10-29) and no budget, so the box's 8,192 stands.
    dense = catalog.get("qwen3.8-27b-nvidia-nvfp4")
    assert dense.profile(CLUSTER_MODE).tuned.max_num_batched_tokens is None
    untuned_base = store.defaults_for(dense.id)
    untuned = resolved_launch(dense, box, max_num_seqs=untuned_base.max_num_seqs,
                              max_model_len=untuned_base.max_model_len)
    assert (untuned["max_num_seqs"], untuned["max_num_batched_tokens"], untuned["max_model_len"]) == (
        6, 8192, 262_144)


# ── C4: the operator's own choice wins — a value past what fits is stored, and warned about ──


def store(tmp_path, **kw) -> DesiredStore:
    return desired_store(Config(cluster_state_path=str(tmp_path / "state" / "cluster.json"), **kw))


def test_with_nothing_stored_the_store_offers_the_recommendation(tmp_path):
    # `model_choice` joined `Desired` in pass 4 (`vllm-model-selection`): with nothing stored, the
    # store offers the ModelCatalog's own cluster default, exactly as it offers the measured
    # concurrency here. That is this test's subject — what an unwritten switch offers — extended to
    # the model, not a weakened expectation: every other field is asserted as before.
    assert store(tmp_path).load() == Desired(mode=CLASSIC, generation=0, max_num_seqs=6,
                                             max_model_len=262_144,
                                             model_choice="qwen3.8-flash-next-hibrid48", uncensored=True)


def test_a_value_the_operator_posted_survives_a_restart_and_beats_the_default(tmp_path):
    """C4 both directions: the default applies ONLY when nothing is stored."""
    store(tmp_path).set(enabled=True, max_num_seqs=3, max_model_len=262_144)
    again = store(tmp_path)                       # a fresh process — a restarted container
    assert again.load().max_num_seqs == 3, "the default overwrote the operator's choice"
    assert again.load().mode == CLUSTER_MODE and again.load().generation == 1


def test_a_stored_value_above_what_fits_is_started_as_asked_and_only_warned_about(tmp_path):
    """The operator asked for 8 at the full window and the gateway started 6 — its own recommendation
    overruling the person running it. That is the same failure as ignoring them, in the other direction.

    Past the KV pool vLLM ADMITS fewer sequences at a time and queues the rest: it does not OOM and it
    does not refuse to launch, because the pool is paged and shared
    (worklog/vllm-cluster-ram-headroom.md §5.2). So what fits is ADVICE. A cluster.json from an older LB
    carrying 16 is started at 16, with the reason put where the page can read it."""
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mode": "classic", "generation": 4, "max_num_seqs": 16, '
                    '"max_model_len": 262144}')
    s = store(tmp_path)
    assert s.load().max_num_seqs == 16, "a read never rewrites what is stored"
    started = s.set(enabled=True, kv_cache_dtype="bfloat16")          # bf16's arithmetic (Flash-Next runs nvfp4 by default)
    assert started.max_num_seqs == 16, "the operator's stored concurrency was overruled"
    assert "16" in s.last_clamp and "6" in s.last_clamp, s.last_clamp
    assert store(tmp_path).load().max_num_seqs == 16, "what was started is not what was written"


def test_the_operator_may_ask_for_more_concurrency_than_the_measurement_recommends(tmp_path):
    """8 was silently reduced to 6 on the live gateway; 30 is what the operator asked to be able to
    choose; 48 is the most tp-rank itself accepts (`MAX_NUM_SEQS_RANGE`), and the catalog allows it for
    this model. Each is stored exactly as asked, and each still gets its warning."""
    for asked in (8, 30, 48):
        s = store(tmp_path / str(asked))
        got = s.set(enabled=True, max_num_seqs=asked, kv_cache_dtype="bfloat16", max_model_len=262_144)
        assert got.max_num_seqs == asked, f"asked for {asked}, was given {got.max_num_seqs}"
        assert store(tmp_path / str(asked)).load().max_num_seqs == asked, f"{asked} was not written"
        assert str(asked) in s.last_clamp, f"a tight choice must still be explained: {s.last_clamp!r}"


def test_a_choice_that_fits_is_never_clamped_and_leaves_no_note(tmp_path):
    s = store(tmp_path)
    assert s.set(enabled=True, max_num_seqs=4, kv_cache_dtype="bfloat16", max_model_len=262_144).max_num_seqs == 4
    assert s.last_clamp == ""


def test_the_clamp_follows_the_chosen_context_not_the_configured_one(tmp_path):
    """18 seats do not fit 262,144, but they are exactly right at 65,536 (1,225 blocks // (41 + 25))."""
    s = store(tmp_path)
    assert s.set(enabled=True, max_num_seqs=18, kv_cache_dtype="bfloat16", max_model_len=65_536).max_num_seqs == 18
    assert s.last_clamp == ""
    assert s.set(enabled=True, restart=True, max_num_seqs=19, max_model_len=65_536).max_num_seqs == 19
    assert "holds 18 x 65,536-token requests" in s.last_clamp, s.last_clamp


def test_a_start_on_an_already_running_cluster_stays_idempotent_under_the_clamp(tmp_path):
    """C3's edge: the clamp lives on the WRITE path, after the idempotent-Start branch. A second
    Start on a pair running at a now-unfittable 16 must return that pair unchanged — not raise
    'stop the cluster', and not rewrite its generation."""
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mode": "cluster", "generation": 5, "max_num_seqs": 16, '
                    '"max_model_len": 262144, "start_id": "abc"}')
    s = store(tmp_path)
    again = s.set(enabled=True)
    assert (again.max_num_seqs, again.generation, again.mode) == (16, 5, CLUSTER_MODE)
    assert store(tmp_path).load().max_num_seqs == 16, "a running pair was rewritten on disk"


# ── per MODEL: the pool and its cost per token are the chosen spec's, never one constant ──────────────
# Audit 2026-10-02 (#2, #7, #3): every figure above was Flash-Next's — 28 GB at 16,392 B/token — and was
# applied to whichever model was chosen. The dense 27B pins 48 GB at 36,860 B/token (measured, generation 103:
# vLLM logged 1,302,238 tokens and "Maximum concurrency for 262,144 tokens per request: 4.97x").

FLASH, DENSE = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4"
DENSE_TOKENS = 48_000_000_000 // 36_860
VLLM_REPORTED_DENSE_TOKENS = 1_302_238


def test_the_dense_models_pool_is_its_own_arithmetic():
    assert DENSE_TOKENS == 1_302_224
    assert kv_tokens_per_rank(48_000_000_000, bytes_per_token=36_860) == DENSE_TOKENS
    assert abs(DENSE_TOKENS - VLLM_REPORTED_DENSE_TOKENS) / VLLM_REPORTED_DENSE_TOKENS < 0.0001
    assert max_num_seqs_that_fits(DENSE_TOKENS, 262_144) == 4, "vLLM's own 4.97x, floored"


@pytest.mark.parametrize("spec_id, tokens, pool, seats", [
    (FLASH, POOL_TOKENS, 28_000_000_000, 6),
    (DENSE, DENSE_TOKENS, 48_000_000_000, 4),      # vLLM's own 4.97x — not Flash-Next's 6
])
def test_the_recommendation_is_made_for_the_model_it_is_given(spec_id, tokens, pool, seats):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import recommended_for
    r = recommended_for(Config(), ModelCatalog.load().get(spec_id))
    assert (r.kv_tokens, r.kv_cache_memory, r.max_num_seqs, r.max_model_len) == (tokens, pool, seats, 262_144)
    with pytest.raises(ValueError, match="no model to recommend for"):
        recommended_for(Config(), None)          # no spec, no context of its own: nothing to recommend


@pytest.mark.parametrize("args, seats", [
    ((262_144,), 6),                                   # nothing named: the configured pool and cost
    ((262_144, 35_000_000_000), 8),                    # the operator's pin, at the configured cost
    ((262_144, 48_000_000_000, 36_860), 4),            # the dense model's pool at the dense model's cost
    ((262_144, 28_000_000_000, 36_860), 2),            # …and what Flash-Next's pool would hold of ITS tokens
    ((65_536, None, 36_860), 11),                      # a spec with no pin of its own: the configured pool
])
def test_fits_takes_the_pool_and_the_cost_per_token(args, seats):
    from vllm_lb.cluster_defaults import fits_for
    assert fits_for(Config())(*args) == seats


def test_the_tight_note_is_worked_out_for_the_chosen_model(tmp_path):
    """6 seats at the full window fit Flash-Next's pool and not the 27B's (4): the note must say 4, for the
    model that would launch — and follow the operator's pin at that model's cost per token."""
    s = store(tmp_path)
    s.set(enabled=False, max_num_seqs=6, kv_cache_dtype="bfloat16", max_model_len=262_144)
    assert s.last_clamp == ""
    s.set(enabled=False, model_choice=DENSE, kv_cache_dtype="bfloat16", max_num_seqs=6, max_model_len=262_144)   # bf16's arithmetic
    assert "KV pool holds 4 x 262,144" in s.last_clamp and "concurrency 6" in s.last_clamp, s.last_clamp
    s.set(enabled=False, kv_cache_memory=64_000_000_000)          # 1,736,299 tokens: 6 seats
    assert s.last_clamp == ""
    s.set(enabled=False, kv_cache_memory=28_000_000_000)          # 759,631 of the dense model's tokens: 2
    assert "KV pool holds 2 x" in s.last_clamp, s.last_clamp


@pytest.mark.parametrize("fits, said", [
    (lambda ctx: 2, "holds 2 x"),
    (lambda ctx, pool: 3, "holds 3 x"),
    (lambda ctx, pool, per_token: 4, "holds 4 x"),
    (lambda ctx, pool, per_token, spec: 5, "holds 5 x"),
    (lambda *a: len(a), "holds 4 x"),
])
def test_a_store_still_takes_a_fits_of_fewer_arguments(tmp_path, fits, said):
    s = DesiredStore(str(tmp_path / "state" / "cluster.json"), fits=fits)
    s.set(enabled=True, max_num_seqs=16)
    assert said in s.last_clamp, s.last_clamp


def test_a_fits_that_raises_typeerror_of_its_own_is_not_mistaken_for_a_shorter_signature(tmp_path):
    def broken(ctx, pool, per_token):
        raise TypeError("a real bug")
    with pytest.raises(TypeError, match="a real bug"):
        DesiredStore(str(tmp_path / "state" / "cluster.json"), fits=broken).set(enabled=True)


# ── auto is never sized from numbers nobody measured ─────────────────────────────────────────────────




def uncalibrated() -> "ModelCatalog":
    """The shipped catalog with the dense 27B's footprint UNMEASURED again (`calibration: ""`, no anchors) — the
    model it was until 2026-10-02, and the case these tests are about: a spec nobody calibrated."""
    from vllm_lb.catalog import ModelCatalog, document
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["memory"].update(
        calibration="", calibrated_batched_tokens=0, calibrated_max_num_seqs=0)
    return ModelCatalog.from_dict(doc)


def _auto(spec_id, view=None, catalog=None):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.cluster_defaults import kv_auto
    catalog = catalog or ModelCatalog.load()
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url="http://192.0.2.2:8100")
    view = view or {"boxes": {b: {"mem_stable_gib": 115.0, "rank": None} for b in ("box-a", "box-b")}}
    return kv_auto(cfg, catalog, view, catalog.get(spec_id), max_num_seqs=8, max_model_len=262_144,
                   max_num_batched_tokens=8192)


def test_auto_is_refused_for_a_model_whose_footprint_was_never_calibrated():
    """Audit #3: the dense 27B's MemoryFootprint is arithmetic (`calibration: ""`), and on two idle boxes auto
    sized a ~79 GB pool from it. It is refused in words the page can show; the calibrated model still sizes."""
    dense = _auto(DENSE, catalog=uncalibrated())
    assert dense.kv_cache_memory is None
    assert dense.refusal == "Qwen3.8 27B · dense NVFP4 is not calibrated: choose a size"
    assert dense.baseline == 48_000_000_000 and dense.view()["boxes"] == []
    flash = _auto(FLASH, {"boxes": {b: {"mem_stable_gib": 127.0, "rank": None} for b in ("box-a", "box-b")}})
    assert flash.refusal == "" and flash.kv_cache_memory >= 28_000_000_000 and len(flash.boxes) == 2


def test_a_start_at_auto_for_an_uncalibrated_model_is_the_switchs_refusal(tmp_path):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster import KV_AUTO, parse_hosts
    from vllm_lb.cluster_defaults import kv_auto_resolver
    cfg = Config(cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url="http://192.0.2.2:8100",
                 cluster_state_path=str(tmp_path / "state" / "cluster.json"))
    view = {"boxes": {b: {"mem_stable_gib": 115.0, "rank": None} for b in ("box-a", "box-b")}}
    s = desired_store(cfg)
    with pytest.raises(ValueError, match="is not calibrated: choose a size"):
        s.set(enabled=True, model_choice=DENSE, kv_cache_memory=KV_AUTO,
              resolve_kv_auto=kv_auto_resolver(cfg, uncalibrated(), lambda: view))
    assert s.load().mode == CLASSIC


# ── C3: seats by the ENGINE's rule — blocks per request, not tokens over context ──────────────────
#
# vLLM holds the pool in BLOCKS and a request takes ceil(context / block_tokens) + F of them, F the
# model's fixed per-request blocks (its linear-attention state, padded to pages). Measured off the
# engine's own cache_config_info (lb/tests/fixtures/cluster-27b-2026-10-02-memory.json): the 27B at
# 48 GB holds 1,689 blocks of 816 tokens and reports 4.97x at 262,144 = 1689 / (322 + 18); Flash-Next
# at 28 GB holds 1,225 blocks of 1,616 tokens, 6.52x = 1225 / (163 + 25) — and the 35 GB pin of
# generation 99 (1,531 blocks) gave the 8.14x the engine printed then. Tokens over context said 19 at
# 65,536 on the 27B where the engine seats 17.

BLOCKS = json.loads((Path(__file__).resolve().parent / "fixtures"
                     / "cluster-27b-2026-10-02-memory.json").read_text())


@pytest.mark.parametrize("spec_id, pool, ctx, seats", [
    (DENSE, 48_000_000_000, 262_144, 4),       # 1689 // (322 + 18) — vLLM's own 4.97x
    (DENSE, 48_000_000_000, 65_536, 17),       # 1689 // (81 + 18): the LB said 19
    (FLASH, 28_000_000_000, 262_144, 6),       # 1225 // (163 + 25) — 6.52x
    (FLASH, 35_000_000_000, 262_144, 8),       # 1531 // 188 — generation 99's 8.14x
    (FLASH, 28_000_000_000, 65_536, 18),       # 1225 // (41 + 25): tokens over context said 26
])
def test_seats_follow_the_engines_block_rule_for_the_model(spec_id, pool, ctx, seats):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import seats_for
    spec = ModelCatalog.load().get(spec_id)
    assert seats_for(spec, pool, ctx) == seats


def test_the_engines_own_block_count_is_preferred_while_it_serves():
    """`vllm:cache_config_info` says how many blocks the pool really holds; given, it replaces the
    catalog's arithmetic for the count — the per-request blocks stay the model's rule."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import seats_for
    dense = ModelCatalog.load().get(DENSE)
    live = BLOCKS["cache_config_info_generation_105"]
    assert seats_for(dense, live["kv_cache_memory_bytes"], 262_144, pool_blocks=live["num_gpu_blocks"]) == 4
    assert seats_for(dense, 48_000_000_000, 65_536, pool_blocks=1980) == 20, "20 = 1980 // 99"
    assert dense.memory.kv_blocks.blocks_in(48_000_000_000) == live["num_gpu_blocks"]
    assert round(live["num_gpu_blocks"] / dense.memory.kv_blocks.per_request(262_144), 3) == \
        round(live["kv_cache_max_concurrency"], 3)


def test_a_model_with_no_block_rule_recorded_still_seats_by_tokens():
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import seats_for
    doc = document()
    del next(m for m in doc["models"] if m["id"] == DENSE)["memory"]["kv_blocks"]
    spec = ModelCatalog.from_dict(doc).get(DENSE)
    assert spec.memory.kv_blocks is None
    assert seats_for(spec, 48_000_000_000, 65_536) == 19, "1,302,224 // 65,536 — the old rule, honestly"
    assert seats_for(None, 28_000_000_000, 262_144) == 6, "no spec: the gateway's configured pool and cost"


def test_the_recommendation_and_the_clamp_seat_by_the_block_rule(tmp_path):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import recommended_for
    cfg = Config()
    catalog = ModelCatalog.load()
    assert recommended_for(cfg, catalog.get(DENSE), max_model_len=65_536).max_num_seqs == 17
    assert recommended_for(cfg, catalog.get(FLASH), max_model_len=65_536).max_num_seqs == 18
    s = desired_store(Config(cluster_state_path=str(tmp_path / "state" / "cluster.json")), catalog=catalog)
    s.set(enabled=True, model_choice=DENSE, max_num_seqs=18, kv_cache_dtype="bfloat16", max_model_len=65_536)
    assert "holds 17 x 65,536-token requests" in s.last_clamp, s.last_clamp


def test_the_cold_start_default_is_the_default_models_own_seats_not_the_gateways_constants(tmp_path):
    """The cold-start seats used to be derived from 28 GB / 16,392 — the gateway's two Flash-Next constants —
    whatever the catalog's default model was. They are that model's own `tuned` seats now, and the gateway's
    pool constant does not move them."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    from vllm_lb.cluster_defaults import recommended_for
    catalog = ModelCatalog.load()
    default = catalog.default(CLUSTER_MODE)
    path = str(tmp_path / "cluster.json")
    assert desired_store(Config(cluster_state_path=path), catalog=catalog).load().max_num_seqs \
        == recommended_for(Config(), default).max_num_seqs == default.profile(CLUSTER_MODE).tuned.max_num_seqs == 6
    assert desired_store(Config(cluster_state_path=path, cluster_kv_cache_memory=14_000_000_000),
                         catalog=catalog).load().max_num_seqs == 6, \
        "the gateway's constant does not size the default model's seats"


# ── the engine's own pool beats the arithmetic over the pin ───────────────────────────────────────
#
# Everything above derives seats from a pin: tokens over context at a measured cost per token, or blocks
# over blocks-per-request at a measured block size. Both extrapolate constants taken at ONE pin to any
# other, and on generation 109 that broke: the dense 27B's dflash profile pages at 832 tokens a block
# (its own start log) where the catalog records 864, and reuses the MTP profile's 28,418,000 B block,
# measured at the 48 GB pin. The arithmetic said 3 seats at 262,144. The engine said 2.40x.
#
# vLLM prints its answer once per start and republishes it as `vllm:cache_config_info`, and the two
# numbers on that line are ONE number: the engine derives the token count from its block count as
# max_concurrency x max_model_len, so tokens / max_model_len IS the concurrency it printed.

#: The engine's own line for generation 109, read out of the head's log rather than typed here.
DFLASH2_POOL = re.search(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens "
                         r"per request: ([\d.]+)x",
                         (Path(__file__).resolve().parent / "fixtures"
                          / "qwen38-dflash2-start-2026-10-02-rank0.log").read_text())
DFLASH = "qwen3.8-27b-dflash2"


def test_the_engines_line_for_generation_109_is_what_the_fixture_recorded():
    """The numbers below are the engine's, quoted — not derived here and not remembered."""
    tokens, ctx, conc = (int(DFLASH2_POOL.group(1).replace(",", "")),
                         int(DFLASH2_POOL.group(2).replace(",", "")), float(DFLASH2_POOL.group(3)))
    assert (tokens, ctx, conc) == (629_682, 262_144, 2.40)


@pytest.mark.parametrize("tokens, ctx, printed", [
    (1_708_119, 262_144, 6.52),      # Flash-Next, 28 GB — the audit's §4 line, and the 2026-09-15 start
    (1_302_238, 262_144, 4.97),      # the 27B at 48 GB, generation 105
    (629_682, 262_144, 2.40),        # the 27B DFlash2 at 28 GB, generation 109
])
def test_the_pools_tokens_over_its_context_are_the_concurrency_the_engine_printed(tokens, ctx, printed):
    from vllm_lb.cluster_defaults import EnginePool
    assert round(EnginePool(tokens=tokens, max_model_len=ctx).concurrency, 2) == printed


@pytest.mark.parametrize("spec_id, pool, blocks, conc", [
    (FLASH, 28_000_000_000, 1225, 6.51595744680851),       # fixtures/cluster-27b-…json, Flash-Next 2026-09-30
    (DENSE, 48_000_000_000, 1689, 4.9676470588235295),     # …and generation 105
])
def test_the_engines_concurrency_recovers_its_own_block_count_through_the_models_rule(spec_id, pool, blocks, conc):
    """`EnginePool.blocks` is the measurement turned back into blocks, so a what-if at another context
    stays anchored on it. Where the catalog's block rule matches the engine it lands on `num_gpu_blocks`
    exactly — which is what makes it safe to prefer over `blocks_in(pin)`."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import EnginePool
    rule = ModelCatalog.load().get(spec_id).memory.kv_blocks
    pool_tokens = int(round(conc * 262_144))
    assert EnginePool(tokens=pool_tokens, max_model_len=262_144).blocks(rule) == blocks
    assert rule.blocks_in(pool) == blocks, "the arithmetic agrees here — these constants were measured here"


@pytest.mark.parametrize("spec_id, pin, engine_tokens, arithmetic, engine_says", [
    # Flash-Next at its own 28 GB pin: every source already agreed, and the page does not move.
    (FLASH, 28_000_000_000, 1_708_119, 6, 6),
    # The dense 27B's dflash profile at its own 28 GB pin. Since the catalog's block figure was corrected
    # to the 832 the engine printed at THIS profile's defaults, the arithmetic agrees with the measurement
    # — 2 either way. That agreement is a property of one corrected constant, not of the method, which is
    # the point of the row below.
    (DFLASH, 28_000_000_000, 629_682, 2, 2),
    # The method, with nothing coincidental left in it: a pool the engine really measured somewhere the
    # catalog's anchor does not describe. Half the tokens, so half the seats, and the arithmetic's own
    # answer is NOT what is published.
    (DFLASH, 28_000_000_000, 314_841, 2, 1),
])
def test_the_engines_own_pool_beats_the_arithmetic_over_the_pin(spec_id, pin, engine_tokens, arithmetic, engine_says):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import EnginePool, seats_for
    spec = ModelCatalog.load().get(spec_id)
    assert seats_for(spec, pin, 262_144) == arithmetic, "the arithmetic is unchanged — it is the fallback"
    engine = EnginePool(tokens=engine_tokens, max_model_len=262_144)
    assert seats_for(spec, pin, 262_144, engine=engine) == engine_says


def test_a_pool_measured_at_one_context_still_answers_for_another_through_the_block_rule():
    """The engine divided by the window it was launched for, so 4.97x at 262,144 is that context's. The
    recovered block count carries it to 65,536: 1,689 // (81 + 18) = 17, the figure the engine itself
    reports there — where tokens over context said 19."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import EnginePool, seats_for
    dense = ModelCatalog.load().get(DENSE)
    engine = EnginePool(tokens=1_302_238, max_model_len=262_144)
    assert engine.seats(262_144, dense.memory.kv_blocks) == 4
    assert engine.seats(65_536, dense.memory.kv_blocks) == 17
    assert seats_for(dense, 48_000_000_000, 65_536, engine=engine) == 17
    assert engine.seats(65_536) == 19, "with no block rule to carry it, tokens over context — honestly"


def test_the_clamp_the_page_shows_is_the_engines_figure_for_the_stored_selection(tmp_path):
    """`published_clamp` is what /lb/status carries: the stored selection, measured where the engine has
    measured it. 8 seats were asked for, the engine seats 2 — and the arithmetic's 3 is not published."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import EnginePool
    catalog = ModelCatalog.load()
    s = desired_store(Config(cluster_state_path=str(tmp_path / "state" / "cluster.json")), catalog=catalog)
    s.set(enabled=True, model_choice=DFLASH, max_num_seqs=8, max_model_len=262_144)
    assert "holds 2 x" in s.last_clamp, s.last_clamp
    assert "holds 2 x" in s.published_clamp(s.load()), "with nothing measured, the arithmetic still warns"
    # A pool the engine measured where the catalog's anchor does not reach: the published sentence is the
    # engine's, and it is NOT the arithmetic's answer, which is what makes this a test of the source.
    engine = EnginePool(tokens=314_841, max_model_len=262_144)
    assert s.published_clamp(s.load(), engine=engine) == (
        "KV pool holds 1 x 262,144-token requests: concurrency 8 runs 1 at a time and queues the rest")
    assert s.load().max_num_seqs == 8, "a read never rewrote what is stored"


def test_a_selection_that_fits_the_engines_pool_leaves_no_sentence(tmp_path):
    from vllm_lb.cluster_defaults import EnginePool
    s = store(tmp_path)
    s.set(enabled=True, model_choice=DFLASH, max_num_seqs=2, max_model_len=262_144)
    assert s.published_clamp(s.load(), engine=EnginePool(tokens=629_682, max_model_len=262_144)) == ""


@pytest.mark.parametrize("view, pool, why", [
    ({"held": False, "effective": {"model_choice": DFLASH, "kv_cache_memory": 28_000_000_000,
                                   "max_model_len": 262_144}}, {"tokens": 629_682}, "no pair is held"),
    ({"held": True, "effective": None}, {"tokens": 629_682}, "nothing is launched"),
    ({"held": True, "effective": {"model_choice": DENSE, "kv_cache_memory": 28_000_000_000,
                                  "max_model_len": 262_144}}, {"tokens": 629_682}, "another model runs"),
    ({"held": True, "effective": {"model_choice": DFLASH, "kv_cache_memory": 48_000_000_000,
                                  "max_model_len": 262_144}}, {"tokens": 629_682}, "another pin runs"),
    ({"held": True, "effective": {"model_choice": DFLASH, "kv_cache_memory": 28_000_000_000,
                                  "max_model_len": 262_144}}, None, "the head reported no pool"),
    ({"held": True, "effective": {"model_choice": DFLASH, "kv_cache_memory": 28_000_000_000,
                                  "max_model_len": 262_144}}, {"tokens": 0}, "the pool is not a figure"),
])
def test_a_measurement_is_only_used_for_the_pool_it_measured(view, pool, why):
    """The guards are `proxy._engine_blocks`'s. Anything a measurement cannot answer falls back to the
    arithmetic rather than describing one pool with another's number."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import engine_pool
    assert engine_pool(view, ModelCatalog.load().get(DFLASH), pool) is None, why


def test_the_measurement_is_taken_for_the_pair_that_is_really_serving_it():
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import EnginePool, engine_pool
    view = {"held": True, "effective": {"model_choice": DFLASH, "kv_cache_memory": 28_000_000_000,
                                        "max_model_len": 262_144}}
    got = engine_pool(view, ModelCatalog.load().get(DFLASH), {"tokens": 629_682, "blocks": 3286})
    assert got == EnginePool(tokens=629_682, max_model_len=262_144)
    assert got.view() == {"tokens": 629_682, "max_model_len": 262_144, "max_concurrency": 2.402}


# ── not enough RAM is said at the Start, never minutes later at the launch ──────────────────────────
#
# Generation 249 (2026-10-26): the Start was stored, the pair reserved both GB10 boxes, and only then did the launch gate
# refuse it — box-b at rest at 111.0 GiB free against 115.1 needed. The operator: "if not enough ram tell before I hit
# start". `memory_admission` asks the launch gate's own question at the Start, of every box that is AT REST.

def _admission_cfg(tmp_path, fraction=None):
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import ClusterConfig
    return Config(clusters=(ClusterConfig(name="pair", hosts=tuple(parse_hosts(HOSTS)), url="http://192.0.2.2:8100",
                                          control="ssh", ram_usable_fraction=fraction, max_num_batched_tokens=8192),),
                  cluster_state_path=str(tmp_path / "state" / "cluster.json"))


def _need(cfg, box, role, total=121.7):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.memory_estimate import estimate_memory
    catalog = ModelCatalog.load()
    return estimate_memory(catalog.get(FLASH), box=box, role=role, max_num_seqs=8, max_model_len=131_072, tp=2,
                           available_gib=None,
                           max_num_batched_tokens=cfg.cluster_max_num_batched_tokens, **cfg.memory_policy(total))


def _admit_view(free: dict, at_rest: bool = True, total: float = 121.7) -> dict:
    return {"boxes": {b: {"mem_stable_gib": f, "mem_total_gib": total, "at_rest": at_rest, "rank": None}
                      for b, f in free.items()}}


def test_a_start_a_box_at_rest_cannot_hold_is_refused_with_its_figures_and_nothing_stored(tmp_path):
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import memory_admission
    cfg = _admission_cfg(tmp_path)
    worker = _need(cfg, "box-b", "worker")
    view = _admit_view({"box-a": 127.0, "box-b": worker.needed_now_gib - 4.0})
    s = desired_store(cfg)
    with pytest.raises(ValueError, match=r"not enough RAM: box-b: .* short 4\.0 GiB"):
        s.set(enabled=True, model_choice=FLASH, max_num_seqs=8, max_model_len=131_072,
              admit=memory_admission(cfg, ModelCatalog.load(), lambda: view))
    assert s.load().mode == CLASSIC


def test_a_box_not_at_rest_is_the_loops_to_wait_for_not_the_starts_to_refuse(tmp_path):
    """A fleet job or ollama holding RAM there is a wait the handover ends — refusing it would refuse every Start."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import memory_admission
    cfg = _admission_cfg(tmp_path)
    view = _admit_view({"box-a": 40.0, "box-b": 40.0}, at_rest=False)
    s = desired_store(cfg)
    s.set(enabled=True, model_choice=FLASH, max_num_seqs=8, max_model_len=131_072,
          admit=memory_admission(cfg, ModelCatalog.load(), lambda: view))
    assert s.load().mode == CLUSTER_MODE


def test_the_fleets_ram_usable_fraction_is_the_one_reserve_a_start_keeps(tmp_path):
    """ram_usable_fraction 0.98 of a 121.7 GiB box keeps 2.43 GiB — floor, start headroom, no margin — and admits
    the Start that the LB_CLUSTER_* 8 GiB headroom refused."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_defaults import memory_admission
    usable = _admission_cfg(tmp_path, fraction=0.98)
    assert usable.memory_policy(121.7) == {"floor_gib": 2.43, "margin_gib": 0.0, "start_headroom_gib": 2.43}
    with pytest.raises(ValueError, match="MemTotal"):
        usable.memory_policy(None)
    worker = _need(_admission_cfg(tmp_path), "box-b", "worker")
    free = worker.total_gib + 4.0                       # under total + 8, over total + 2.43
    view = _admit_view({"box-a": 127.0, "box-b": free})
    s = desired_store(usable)
    s.set(enabled=True, model_choice=FLASH, max_num_seqs=8, max_model_len=131_072,
          admit=memory_admission(usable, ModelCatalog.load(), lambda: view))
    assert s.load().mode == CLUSTER_MODE


# ── what THIS deployment offers (fleet `choices` on the pair) ───────────────────────────────────────────

def test_a_start_of_a_row_the_deployment_does_not_offer_is_refused_and_nothing_stored(tmp_path):
    """Operator, 2026-10-26: "remove original censored from our vllm setup". The pair's fleet `choices` names the
    rows it offers; a Start of any other — the original of a model whose twin alone is offered — is refused (409 at
    POST /lb/cluster), naming what is offered. The twin row starts."""
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import ClusterConfig
    cfg = Config(clusters=(ClusterConfig(name="pair", hosts=tuple(parse_hosts(HOSTS)), url="http://192.0.2.2:8100",
                                          control="ssh", offers=(f"{FLASH}@uncensored",)),),
                 cluster_state_path=str(tmp_path / "state" / "cluster.json"))
    s = desired_store(cfg)
    with pytest.raises(ValueError, match=rf"{re.escape(FLASH)} is not offered here: {re.escape(FLASH)}@uncensored"):
        s.set(enabled=True, model_choice=FLASH, uncensored=False)
    assert s.load().mode == CLASSIC
    s.set(enabled=True, model_choice=FLASH, uncensored=True)
    assert s.load().mode == CLUSTER_MODE


# ── what KV dtypes THIS deployment runs (fleet `kv_cache_dtypes` on the pair) ─────────────────────────────

def test_a_start_at_a_kv_dtype_the_deployment_does_not_run_is_refused(tmp_path):
    """Operator, 2026-10-26: "skip FP16 everywhere, FP8 is default unless can run nvfp4". A Start that would run any
    other KV dtype — asked for, or the row's own — is refused naming the list; the row's own fp8_e4m3 starts."""
    from vllm_lb.cluster import parse_hosts
    from vllm_lb.config import ClusterConfig
    cfg = Config(clusters=(ClusterConfig(name="pair", hosts=tuple(parse_hosts(HOSTS)), url="http://192.0.2.2:8100",
                                          control="ssh", kv_dtypes=("fp8", "fp8_e4m3", "nvfp4")),),
                 cluster_state_path=str(tmp_path / "state" / "cluster.json"))
    s = desired_store(cfg)
    with pytest.raises(ValueError, match="KV dtype bfloat16 is not run here: fp8, fp8_e4m3, nvfp4"):
        s.set(enabled=True, model_choice=FLASH, kv_cache_dtype="bfloat16")
    with pytest.raises(ValueError, match="KV dtype auto is not run here"):
        s.set(enabled=True, model_choice="qwen3.8-27b-dflash2")                # its own: vLLM's auto (bf16)
    assert s.load().mode == CLASSIC
    assert s.set(enabled=True, model_choice=FLASH).mode == CLUSTER_MODE        # its own fp8_e4m3


def test_the_published_kv_choices_are_the_deployments():
    from vllm_lb.cluster_defaults import kv_choices_run
    choices = [{"dtype": "bfloat16", "refused": ""}, {"dtype": "fp8_e4m3", "refused": ""}, {"dtype": "nvfp4", "refused": "x"}]
    assert kv_choices_run(choices, None) == choices
    assert [c["dtype"] for c in kv_choices_run(choices, ("fp8_e4m3", "nvfp4"))] == ["fp8_e4m3", "nvfp4"]
