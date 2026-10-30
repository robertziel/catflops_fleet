"""M8 — a MemoryEstimate per box for the selection: model, concurrency, context (worklog/vllm-gateway-program.md,
pass 4's frame amendment; worklog/vllm-model-selection.md).

  MemoryFootprint  a model's memory per rank, catalog DATA on its ModelSpec: the weights, its PLE table, its MTP
                   drafter, KV bytes per token and the per-sequence state of its linear-attention layers — with
                   the run it was calibrated on ("" = not measured yet)
  MemoryEstimate   what one box needs for a selection and the MemAvailable that leaves — marked against
                   broker_parrot's 12 GB danger floor and the planner's 100 GiB start gate — and, once the pair
                   runs, next to what it actually used

Every constant checked here is READ off the committed evidence of the 2026-09-15 start — the head and worker
logs, and the MemAvailable the LB's own loop observed at launch and at `running` — never typed in. The three
numbers the logs cannot hold are cited where they come from.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from vllm_lb.catalog import document, CLUSTER_MODE, ModelCatalog
from vllm_lb.memory_estimate import (BUDGET, CLASSIC_MEMORY, CONCURRENCY, DEVIATION, DEVIATION_PCT, FLOOR, GATE,
                                    NEAR_FLOOR, NO_HEADROOM, PIN_CAPS, KvAutoBox, estimate_memory, resolve_kv_auto)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEAD_LOG = (FIXTURES / "qwen38-start-2026-09-15-rank0.log").read_text()
WORKER_LOG = (FIXTURES / "qwen38-start-2026-09-15-rank1.log").read_text()
RUN = json.loads((FIXTURES / "cluster-e2e-2026-09-15-memory.json").read_text())
GIB = 1024 ** 3
FLASH, DENSE, SEEDED = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4", "qwen3.8-27b-nvfp4"
POLICY = {"floor_gib": 12.0, "margin_gib": 8.0}

# What the logs cannot hold, and where each is read:
#: The MTP layer's tensors in the checkpoint's own safetensors headers (worklog/vllm-cluster-load-speed.md §1,
#: "MTP layer (1 layer, 512 experts) | 6,173 | 1.49"), sharded like the target's experts across TP=2.
MTP_CHECKPOINT_GIB = 1.49
#: NVFP4 as the checkpoint's README maps it: e2m1 codes (4 bits) plus an fp8 scale per 16 values, per value.
NVFP4_BYTES_PER_VALUE = 0.5 + 1 / 16
#: The checkpoint's config.json: 36 `linear_attention` of 48 `layer_types`; `num_key_value_heads` 2 and `head_dim`
#: 256, so ONE KV head per rank at TP=2 and 2 × 1 × 256 × 2 bytes (K and V, bf16) of attention KV per token per layer.
LINEAR_LAYERS = 36
ATTENTION_BYTES_PER_TOKEN_PER_LAYER = 2 * 1 * 256 * 2


def logged(pattern: str, text: str = HEAD_LOG) -> str:
    found = re.search(pattern, text)
    assert found, f"the committed log no longer says {pattern!r}"
    return found.group(1)


def shipped():
    return ModelCatalog.load()


#: The PLE placement Flash-Next's measurements in this file were taken at: the committed 2026-09-15 start and the
#: 2026-09-30/10-01 runs all launched the table REPLICATED (`MBX_PLE_REPLICATE=1`, "REPLICATED (full table" in both
#: ranks' logs). The profile launches it SPLIT since 2026-10-11 (tests/test_flash_next_ple_split.py), so `estimate`
#: states the measurement's own placement, and a test about the profile's says `ple_replicate=None`.
MEASURED_PLACEMENT = {FLASH: True}


def estimate(spec_id: str = FLASH, *, catalog=None, role="head", seqs=1, ctx=262144, available=114.8, **kw):
    spec = (catalog or shipped()).get(spec_id)
    if spec_id in MEASURED_PLACEMENT:
        kw.setdefault("ple_replicate", MEASURED_PLACEMENT[spec_id])
    return estimate_memory(spec, box="box-a" if role == "head" else "box-b", role=role, max_num_seqs=seqs,
                           max_model_len=ctx, tp=2, available_gib=available, **{**POLICY, **kw})


# ── calibrated on the real start ──────────────────────────────────────────────

def test_the_constants_are_read_off_the_real_start_not_typed_in():
    m = shipped().get(FLASH).memory
    loaded = float(logged(r"Model loading took ([\d.]+) GiB memory"))                         # 64.46, head
    assert float(logged(r"Model loading took ([\d.]+) GiB memory", WORKER_LOG)) == loaded
    rows, dim = (int(v) for v in re.search(r"PLE: GPU-resident partition (\d+) rows x (\d+)", HEAD_LOG).groups())
    assert abs(m.ple_gib - rows * dim * NVFP4_BYTES_PER_VALUE / GIB) < 0.01                  # 26.82 GiB, replicated
    assert "REPLICATED (full table" in HEAD_LOG and "REPLICATED (full table" in WORKER_LOG
    assert abs(m.mtp_gib - MTP_CHECKPOINT_GIB / 2) < 0.01
    assert abs(m.weights_gib + m.ple_gib + m.mtp_gib - loaded) < 0.02, "the three add up to what vLLM loaded"
    pin = int(logged(r"'kv_cache_memory_bytes': (\d+)"))
    tokens = int(logged(r"GPU KV cache size: ([\d,]+) tokens").replace(",", ""))
    assert abs(m.kv_bytes_per_token - pin / tokens) < 1                                       # 16,392 B/token
    block = int(logged(r"Setting attention block size to (\d+) tokens"))
    padding = float(logged(r"Padding mamba page size by ([\d.]+)%")) / 100
    state = LINEAR_LAYERS * block * ATTENTION_BYTES_PER_TOKEN_PER_LAYER / (1 + padding) / GIB
    assert abs(m.seq_state_gib - state) < 0.001, "one mamba page per linear layer per running sequence"
    profile = shipped().get(FLASH).profile(CLUSTER_MODE)
    assert profile.kv_cache_memory == pin
    assert abs(float(logged(r"reserved ([\d.]+) GiB memory for KV Cache")) - pin / GIB) < 0.01
    for box, role in RUN["roles"].items():
        used = RUN["launched"]["mem_available_gib"][box] - RUN["running"]["mem_available_gib"][box]
        assert abs(profile.overhead_gib[role] - (used - loaded - pin / GIB)) < 0.01, \
            f"{role} overhead is what the {box} run used beyond weights and the KV pin"
    assert m.calibration and "2026-09-15" in m.calibration


def test_the_estimate_matches_the_measured_run_within_ten_percent():
    for box, role in RUN["roles"].items():
        before = RUN["launched"]["mem_available_gib"][box]
        after = RUN["running"]["mem_available_gib"][box]
        e = estimate(role=role, seqs=RUN["selection"]["max_num_seqs"], ctx=RUN["selection"]["max_model_len"],
                     available=before)
        used = before - after
        assert abs(e.total_gib - used) <= 0.10 * used, (box, e.total_gib, used)
        assert abs(e.leaves_gib - after) <= 0.10 * after, (box, e.leaves_gib, after)


# ── the formula: data, not branches ───────────────────────────────────────────

def test_the_kv_pin_is_what_the_box_holds_and_the_need_stands_beside_it():
    e = estimate(seqs=1, ctx=262144)
    pin_gib = 28_000_000_000 / GIB
    assert abs(e.kv_pinned_gib - pin_gib) < 1e-9 and e.kv_gib == e.kv_pinned_gib
    m = shipped().get(FLASH).memory
    assert abs(e.kv_needed_gib - (262144 * m.kv_bytes_per_token / GIB + m.seq_state_gib)) < 1e-9
    assert PIN_CAPS not in e.codes
    full = estimate(seqs=48, ctx=262144)
    assert full.kv_gib == full.kv_pinned_gib < full.kv_needed_gib and PIN_CAPS in full.codes
    message = full.message(PIN_CAPS)
    assert "12.58M" in message and "fits 6" in message, message


def test_without_a_pin_the_kv_cache_is_sized_from_concurrency_and_context():
    doc = document()
    next(s for s in doc["models"] if s["id"] == FLASH)["launch"]["cluster"]["kv_cache_memory"] = None
    unpinned = ModelCatalog.from_dict(doc)
    m = unpinned.get(FLASH).memory
    e = estimate(catalog=unpinned, seqs=2, ctx=65536)
    assert e.kv_pinned_gib is None
    assert abs(e.kv_gib - 2 * (65536 * m.kv_bytes_per_token / GIB + m.seq_state_gib)) < 1e-9
    assert estimate(catalog=unpinned, seqs=4, ctx=65536).total_gib > e.total_gib


def test_splitting_the_ple_table_takes_half_of_it_off_each_box():
    split = estimate(ple_replicate=None)                  # the profile's own placement: split since 2026-10-11
    replicated = estimate()                               # the measurement's: the whole table on each rank
    ple = shipped().get(FLASH).memory.ple_gib
    assert abs((replicated.weights_gib - split.weights_gib) - ple / 2) < 1e-9, "−13.4 GiB per box (load-speed §C)"
    # The profile's placement is what an unchosen one costs, whichever it is.
    doc = document()
    next(s for s in doc["models"] if s["id"] == FLASH)["launch"]["cluster"]["ple_replicate"] = True
    assert estimate(catalog=ModelCatalog.from_dict(doc), ple_replicate=None).view() == replicated.view()


def test_the_mtp_drafter_counts_only_with_speculative_decoding():
    flash, dense = estimate(FLASH), estimate(DENSE)
    assert abs(flash.mtp_gib - shipped().get(FLASH).memory.mtp_gib) < 1e-9
    assert abs(dense.mtp_gib - shipped().get(DENSE).memory.mtp_gib) < 1e-9, "its own drafter, its own size"
    assert estimate(DENSE, speculative=False).mtp_gib == 0.0, "speculation off: no drafter loaded"


@pytest.mark.parametrize("available, codes", [
    (114.8, {FLOOR}),                 # leaves ~11.6: under 12
    (118.0, {NEAR_FLOOR}),            # leaves ~14.8: under 12 + 8
    (130.0, set()),                   # leaves ~26.8
    (99.0, {GATE, FLOOR}),            # under the start gate — and what it would leave is under the floor
])
def test_what_a_box_leaves_is_marked_against_the_danger_floor_and_the_start_gate(available, codes):
    e = estimate(available=available)
    assert set(e.codes) & {FLOOR, NEAR_FLOOR, GATE} == codes, e.codes


def test_the_start_gate_is_what_this_selection_NEEDS_not_a_constant():
    """H3 (worklog/vllm-cluster-handover.md). `need 100` was the same number for every model, every
    context and every concurrency — while the last successful load finished at 10.5 GiB free
    (worklog/vllm-cluster-ram-headroom.md §2.1) and this selection actually costs 103.2 GiB on the head.
    A box with 101 GiB free clears the constant and still cannot hold the rank."""
    tight = estimate(available=101.0)
    assert tight.total_gib > 101.0, "the real need, not a constant"
    assert GATE in tight.codes, "the estimate said the box was fine for a rank it cannot hold"
    assert f"{tight.total_gib:.1f}" in tight.message(GATE), tight.message(GATE)
    roomy = estimate(available=tight.total_gib + 0.5)
    assert GATE not in roomy.codes, "a box that can hold the selection is under no gate"


def test_a_box_that_cannot_be_observed_gets_an_estimate_but_no_verdict():
    e = estimate(available=None)
    assert e.total_gib > 0 and e.leaves_gib is None
    assert not set(e.codes) & {FLOOR, NEAR_FLOOR, GATE}


def test_once_running_the_estimate_stands_next_to_the_actual_and_a_deviation_over_ten_percent_is_flagged():
    close = estimate(actual_used_gib=103.2)
    assert close.deviation_pct is not None and abs(close.deviation_pct) < 10 and DEVIATION not in close.codes
    off = estimate(actual_used_gib=90.0)
    assert off.deviation_pct > 10 and DEVIATION in off.codes
    assert "90.0" in off.message(DEVIATION)
    assert estimate().deviation_pct is None, "before running there is nothing to compare"


def test_a_model_not_measured_yet_says_so():
    doc = document()
    next(s for s in doc["models"] if s["id"] == DENSE)["memory"].update(
        calibration="", calibrated_batched_tokens=0, calibrated_max_num_seqs=0)
    e = estimate(DENSE, catalog=ModelCatalog.from_dict(doc))
    assert e.calibration == "" and e.view()["calibration"] == ""


def test_classic_vllm_memory_is_set_by_broker_parrot_and_never_estimated():
    with pytest.raises(ValueError, match=CLASSIC_MEMORY):
        estimate(SEEDED)
    assert CLASSIC_MEMORY == "set by broker_parrot"


def test_the_view_is_the_contract_the_page_reads():
    v = estimate(role="head", available=114.8).view()
    assert set(v) == {"box", "role", "model_choice", "calibration", "weights_gib", "mtp_gib", "kv_gib",
                      "kv_needed_gib", "kv_pinned_gib", "overhead_gib", "total_gib", "available_gib", "leaves_gib",
                      "floor_gib", "actual_used_gib", "deviation_pct", "warnings",
                      # What the PREFILL BUDGET costs, the budget it is charged from, the anchor it is
                      # measured against, and the requirement the planner actually gates on — all four so
                      # a refusal on the page can be read without guessing which knob caused it.
                      "activation_gib", "max_num_batched_tokens", "calibrated_batched_tokens",
                      "needed_now_gib", "start_headroom_gib",
                      # …and the same four for the CONCURRENCY, which costs 1.2 GiB per seat above the
                      # anchor and rested box-a at 1.7 GiB free on 2026-10-01 at 24 of them.
                      "concurrency_gib", "max_num_seqs", "calibrated_max_num_seqs",
                      # THE SLOPES the page writes its notes from, per model — and what a split PLE table frees.
                      "activation_gib_per_1k_tokens", "concurrency_gib_per_seq", "ple_split_frees_gib",
                      # What the box had when its rank launched: once launched, `leaves_gib` is measured from it.
                      "launched_gib", "max_model_len",
                      # The engine's rule: the pool under which one request does not start, the seats by blocks.
                      "kv_floor_gib", "pin_seats"}
    assert (v["box"], v["role"], v["model_choice"]) == ("box-a", "head", FLASH)
    assert v["total_gib"] == round(v["total_gib"], 1) and v["leaves_gib"] == 11.6
    assert v["warnings"] and all(set(w) == {"code", "message", "scope"} for w in v["warnings"])
    assert json.loads(json.dumps(v)) == v


def test_the_view_carries_each_models_own_slopes_anchors_and_calibration():
    """The page's notes ("above 16 · ~1.2 GiB per seat") were Flash-Next's constants shown for every model. They
    are the estimate's now: the measured model's figures, and — for the dense 27B, calibrated 2026-10-02 at
    one point — its anchors with slopes of 0: not measured, not counted. It has no PLE table to split."""
    flash, dense = estimate(FLASH).view(), estimate(DENSE).view()
    assert (flash["calibrated_batched_tokens"], flash["activation_gib_per_1k_tokens"]) == (8192, 1.1)
    assert (flash["calibrated_max_num_seqs"], flash["concurrency_gib_per_seq"]) == (16, 1.2)
    assert flash["ple_split_frees_gib"] == 13.4 and flash["calibration"]
    assert (dense["calibrated_batched_tokens"], dense["activation_gib_per_1k_tokens"]) == (8192, 0.0)
    assert (dense["calibrated_max_num_seqs"], dense["concurrency_gib_per_seq"]) == (16, 0.0)
    assert dense["ple_split_frees_gib"] == 0.0 and "2026-10-02" in dense["calibration"]


def test_a_pre_flight_warning_stops_once_the_rank_has_actually_launched():
    """The danger floor, its margin and the start gate answer ONE question: should this be started on
    this box? Once a rank HAS launched there they are moot — `available_gib` is then what is free with
    the model already loaded, so `leaves_gib` goes negative and "under broker_parrot's danger floor"
    about a pair that is serving happily would be pure noise. What survives is the comparison that only
    exists after a launch: the estimate against what the box really gave up."""
    before = estimate(available=114.8)
    assert FLOOR in before.codes, before.codes

    # The same box once its rank is up: 11.6 GiB free, 103.2 GiB gone (2026-09-15, run 1).
    after = estimate(available=11.6, actual_used_gib=103.2)
    assert not ({GATE, FLOOR, NEAR_FLOOR} & set(after.codes)), after.codes
    assert after.leaves_gib is not None, "what it leaves is still reported, just not warned about"

    off = estimate(available=11.6, actual_used_gib=60.0)
    assert DEVIATION in off.codes and "actual 60.0" in off.message(DEVIATION)


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        estimate(role="boss")


def test_the_catalog_supplies_the_policy_free_constants_only():
    """The floor, its margin and the gate are the fleet's policy (config), not a model's facts."""
    doc = document()
    text = json.dumps(doc)
    for policy in ("floor", "gate", "margin"):
        assert policy not in text


# ── the prefill budget (max_num_batched_tokens) ────────────────────────────────
# The estimator was calibrated at ONE prefill budget — 8,192, the value in the committed start log's
# non-default args — and then the operator was given a control that raises it to 65,536 while the
# estimate stayed blind: `batched 8192` and `batched 16384` produced the identical verdict 103.2 /
# 97.7 GiB. On 2026-09-30 the pair ran at 16,384 down to 0.8 GiB free and both GB10 boxes hung; the
# kernel log is NVRM `Out of memory [NV_ERR_NO_MEMORY]` — the CUDA allocator, on a box whose CPU and
# GPU share one pool — recurring all evening and last at the generation-80 handover.

B16K = json.loads((FIXTURES / "cluster-b16384-2026-09-30-memory.json").read_text())
CALIBRATED_BUDGET = 8192
#: What a start must LEAVE free, from the measured boundary in that fixture: 9.1 GiB left is a
#: configuration that served for weeks, 0.8-4.2 GiB left is the one that hung the boxes.
HEADROOM = 8.0


def test_the_budget_the_estimator_was_calibrated_at_is_the_one_in_the_committed_start_log():
    """The whole term hangs off this anchor: `overhead_gib` was measured on a run whose prefill budget
    was 8,192, so 8,192 is where the extra term must be exactly zero."""
    assert "'max_num_batched_tokens': 8192" in HEAD_LOG, "the head's own non-default args"
    for text in (HEAD_LOG, WORKER_LOG):                      # and the engine config both ranks printed
        assert int(logged(r"max_num_batched_tokens=(\d+)", text)) == CALIBRATED_BUDGET
    assert shipped().get(FLASH).memory.calibrated_batched_tokens == CALIBRATED_BUDGET


def test_the_calibrated_budget_costs_nothing_extra_so_nothing_that_runs_today_is_refused():
    """The configuration the operator restored as the original — seqs 8, batched 8192, spec 4, ctx
    262144 — must be estimated at exactly what it was before this term existed, or a fix for a knob
    nobody touched would start refusing the pair that has been serving for weeks."""
    plain = estimate(role="head", seqs=8)
    budgeted = estimate(role="head", seqs=8, max_num_batched_tokens=CALIBRATED_BUDGET)
    assert budgeted.activation_gib == 0.0
    assert budgeted.total_gib == plain.total_gib == pytest.approx(103.2, abs=0.05)


def test_a_budget_under_the_calibration_is_never_credited_back():
    """Conservative on purpose: we measured what a BIGGER budget costs, never what a smaller one
    releases, so 2,048 is charged as 8,192 is. A credit we have not measured could let a start
    through that should have waited."""
    small = estimate(role="head", seqs=8, max_num_batched_tokens=2048)
    assert small.activation_gib == 0.0
    assert small.total_gib == estimate(role="head", seqs=8, max_num_batched_tokens=CALIBRATED_BUDGET).total_gib


def test_the_prefill_budget_is_counted_at_the_slope_measured_on_the_real_boxes():
    """(115.2 - 106.6) GiB over (16384 - 8192) tokens on the same box in the same role = 1.075 GiB per
    1,024 tokens from the low-water mark, 0.65 from the single sample. The catalog carries a
    coefficient inside that measured range, at the low-water end: a gate must hold at the box's worst
    moment, not its average one."""
    m = shipped().get(FLASH).memory
    low, high = B16K["slope"]["gib_per_1024_tokens_from_the_sample"], \
        B16K["slope"]["gib_per_1024_tokens_from_the_low_water"]
    assert low <= m.activation_gib_per_1k_tokens
    assert m.activation_gib_per_1k_tokens >= high, "a start gate is sized on the low-water mark"
    assert m.activation_gib_per_1k_tokens < 2 * high, "and it is not a number nobody measured"
    big = estimate(role="head", seqs=8, max_num_batched_tokens=16384)
    assert big.activation_gib == pytest.approx(m.activation_gib_per_1k_tokens * 8, abs=1e-6)
    assert big.total_gib == pytest.approx(103.19 + big.activation_gib, abs=0.05)
    assert big.total_gib > estimate(role="head", seqs=8, max_num_batched_tokens=CALIBRATED_BUDGET).total_gib


def test_16384_is_REFUSED_on_the_memory_the_boxes_really_had_and_8192_is_ALLOWED():
    """THE TEST THIS EXISTS FOR, on the real recorded numbers. box-a had 115.7 GiB free when
    generation 52 launched on it (worklog/fixtures/gateway-knobs/lb-status-live-pre-knobs.json) and
    ~116 when generation 68 did; the box recovered from the hang reads 117 available. At 8,192 the
    pair served for weeks. At 16,384 it ran down to 0.8 GiB and the boxes hung."""
    had = B16K["batched_8192"]["mem_at_launch_gib"]["box-a"]                  # 115.7, measured
    assert had == 115.7
    for free in (had, B16K["batched_16384"]["mem_at_launch_gib"]["box-a"], 117.0):
        ok = estimate(role="head", seqs=8, available=free, max_num_batched_tokens=8192,
                      start_headroom_gib=HEADROOM)
        assert free >= ok.needed_now_gib, f"8192 must start on {free} GiB: needs {ok.needed_now_gib}"
        assert not set(ok.codes) & {GATE, NO_HEADROOM, BUDGET}, ok.codes

        no = estimate(role="head", seqs=8, available=free, max_num_batched_tokens=16384,
                      start_headroom_gib=HEADROOM)
        assert free < no.needed_now_gib, f"16384 must be refused on {free} GiB: needs {no.needed_now_gib}"
        assert NO_HEADROOM in no.codes and BUDGET in no.codes, no.codes


def test_the_refusal_names_what_it_needs_what_the_box_has_and_which_knob_is_responsible():
    no = estimate(role="head", seqs=8, available=115.7, max_num_batched_tokens=16384,
                  start_headroom_gib=HEADROOM)
    said = no.message(NO_HEADROOM) + " " + no.message(BUDGET)
    assert "115.7" in said, "what the box has"
    assert f"{no.total_gib:.1f}" in said, "what this configuration needs"
    assert f"{HEADROOM:.0f}" in said, "and the headroom a start must leave"
    assert "batched 16,384" in said, "which knob is responsible"
    assert "@ 8,192" in said, "and what it would fit at"
    assert "headroom" in said, said


def test_the_budget_is_only_blamed_when_lowering_it_would_actually_help():
    """`budget` is an instruction to the operator, so it must not appear on a box that could not hold
    the selection at the calibrated budget either — there, the budget is not the problem."""
    hopeless = estimate(role="head", seqs=8, available=40.0, max_num_batched_tokens=16384,
                        start_headroom_gib=HEADROOM)
    assert GATE in hopeless.codes and BUDGET not in hopeless.codes, hopeless.codes


def test_a_start_must_leave_the_allocator_room_and_that_is_not_the_danger_floor():
    """The measured boundary is between 4.2 GiB left (hung) and 9.1 GiB left (served for weeks), so a
    start leaves 8. broker_parrot's 12.0 GiB danger floor cannot be the gate: the configuration that
    has been serving is already under it, and gating on 12 would refuse it."""
    served = B16K["batched_8192"]["mem_available_gib"]["box-a"]               # 9.1, and safe
    assert served < POLICY["floor_gib"], "the known-good configuration is already under the floor"
    assert HEADROOM < served, "so the start headroom sits below what has been shown to work"
    fits = estimate(role="head", seqs=8, available=115.7, max_num_batched_tokens=8192,
                    start_headroom_gib=HEADROOM)
    assert fits.needed_now_gib == pytest.approx(fits.total_gib + HEADROOM, abs=1e-9)
    tight = estimate(role="head", seqs=8, available=fits.total_gib + 1.0, max_num_batched_tokens=8192,
                     start_headroom_gib=HEADROOM)
    assert NO_HEADROOM in tight.codes and GATE not in tight.codes, "it fits, it just may not start"


def test_the_headroom_is_off_unless_the_gateway_sets_it():
    """Nothing in the catalog decides policy. With no headroom configured the requirement is exactly
    what the selection costs, as it was before."""
    e = estimate(role="head", seqs=8, max_num_batched_tokens=16384)
    assert e.start_headroom_gib == 0.0 and e.needed_now_gib == e.total_gib


def test_a_serving_pair_at_16384_is_never_warned_about_and_never_refused():
    """40b34b2's property, kept: the gate, the headroom and the budget blame answer ONE question —
    should this be STARTED here? The pair that ran at 16,384 down to 0.8 GiB free was serving, and
    telling it so afterwards is noise. Only the estimate-against-actual survives a launch."""
    low = B16K["batched_16384"]["mem_available_gib_low_water"]["box-a"]       # 0.8
    used = B16K["batched_16384"]["actual_used_gib_low_water"]["box-a"]        # 115.2
    running = estimate(role="head", seqs=16, available=low, max_num_batched_tokens=16384,
                       start_headroom_gib=HEADROOM, actual_used_gib=used)
    assert not set(running.codes) & {GATE, NO_HEADROOM, BUDGET, FLOOR, NEAR_FLOOR}, running.codes
    assert running.leaves_gib is not None, "what it leaves is still reported, just not warned about"


def test_the_estimate_now_predicts_the_16384_run_it_used_to_miss_by_twelve_gib():
    """Blind, it said 103.2 GiB for a rank that took 115.2 — a 12 GiB miss that read as a 10 %
    deviation nobody saw. With the budget counted the miss is inside the catalog's own tolerance."""
    used = B16K["batched_16384"]["actual_used_gib_low_water"]["box-a"]
    blind = estimate(role="head", seqs=16, available=0.8, actual_used_gib=used)
    assert blind.total_gib == pytest.approx(103.2, abs=0.05) and abs(blind.deviation_pct) > 10
    seeing = estimate(role="head", seqs=16, available=0.8, max_num_batched_tokens=16384,
                      actual_used_gib=used)
    assert abs(seeing.deviation_pct) < abs(blind.deviation_pct), "counting the budget closes the gap"
    assert abs(seeing.total_gib - used) < abs(blind.total_gib - used)


def test_the_view_carries_the_budget_so_the_page_can_show_what_it_costs():
    v = estimate(role="head", seqs=8, available=115.7, max_num_batched_tokens=16384,
                 start_headroom_gib=HEADROOM).view()
    assert v["max_num_batched_tokens"] == 16384
    assert v["activation_gib"] == round(v["activation_gib"], 1) and v["activation_gib"] > 0
    assert v["needed_now_gib"] == round(v["total_gib"] + HEADROOM, 1)
    assert v["start_headroom_gib"] == HEADROOM
    assert json.loads(json.dumps(v)) == v
    assert {w["code"] for w in v["warnings"]} >= {NO_HEADROOM, BUDGET}


# ── the concurrency (max_num_seqs) ─────────────────────────────────────────────
# The budget above was charged and `max_num_seqs` deliberately was not: the only figures then were
# CUDA-graph captures at 1 and 2 seats (0.29 / 0.59 GiB, worklog/vllm-cluster-ram-headroom.md §4), and
# a slope through two points out to 48 seats would have refused configurations that work. On 2026-10-01
# the live pair supplied the missing figures — same model, same context, same prefill budget, 16 seats
# against 24 — and they are not small: at 24 the head RESTS at 1.7 GiB free with no traffic at all,
# inside the 0.8-4.2 GiB band that hung both GB10 boxes the night before
# (lb/tests/fixtures/cluster-seqs24-2026-10-01-memory.json holds every number).

S24 = json.loads((FIXTURES / "cluster-seqs24-2026-10-01-memory.json").read_text())
#: The concurrency the charge is measured FROM — the highest one measured to rest safely, not the
#: 2026-09-15 calibration's single seat. The fixture's `per_seat.anchor_note` says why.
CALIBRATED_SEQS = 16
#: What these boxes have free when a rank launches on them (the fixture's `boxes`).
AT_LAUNCH = 115.5


def sampled(generation: int) -> dict:
    return next(s for s in S24["samples"] if s["generation"] == generation)


def test_the_concurrency_anchor_is_the_highest_concurrency_MEASURED_to_rest_safely():
    """A charge needs a zero point, and this one is not the calibration's. At 16 seats the head rests
    at 11.3 GiB free — above what a start must leave and clear of the band that hung the boxes — so a
    charge there would refuse what demonstrably works, which is exactly what f1c8491's author refused
    to do. Charging from 1 seat would need that two-point CUDA-graph slope (~4.4 GiB over 15 seats) and
    the live pair contradicts it: at 16 the real footprint is ~1.0-1.5 GiB above the 1-seat estimate."""
    m = shipped().get(FLASH).memory
    anchor = S24["per_seat"]["anchor_max_num_seqs"]
    assert anchor == CALIBRATED_SEQS and m.calibrated_max_num_seqs == anchor
    rests = sampled(84)["mem_available_gib"]                                   # 11.3, idle at 16 seats
    assert sampled(84)["max_num_seqs"] == anchor
    assert rests > HEADROOM, "the anchor is a concurrency measured to leave more than a start must"
    assert rests > max(S24["hang_band"]["fatal_free_gib"]), "and clear of what hung the boxes"
    assert "'max_num_seqs': 1" in HEAD_LOG, "the calibration's own seat count, and not the anchor"
    assert m.calibrated_max_num_seqs > 1


def test_the_per_seat_cost_is_the_difference_between_two_idle_measurements():
    """(11.3 - 1.7) GiB over (24 - 16) seats, both read IDLE on the same box in the same role = 1.2 GiB
    per seat. The two samples differ in `num_speculative_tokens` as well (4 and 3), so this is a BOUND:
    the confound has a known sign — fewer proposals is a smaller uniform decode batch, so less captured
    graph and less per-step activation, never more — so the step 4 -> 3 released memory while the eight
    extra seats took it, and 1.2 understates a seat if anything."""
    idle16, idle24 = sampled(84), sampled(90)
    assert idle16["num_speculative_tokens"] != idle24["num_speculative_tokens"], "the confound is real"
    assert "idle" in idle16["load"].lower() and "IDLE" in idle24["load"], "and both are resting reads"
    measured = ((idle16["mem_available_gib"] - idle24["mem_available_gib"])
                / (idle24["max_num_seqs"] - idle16["max_num_seqs"]))
    bound = S24["per_seat"]["gib_per_seat_between_16_and_24"]
    assert measured == pytest.approx(bound, abs=0.01)
    m = shipped().get(FLASH).memory
    assert m.concurrency_gib_per_seq >= bound, "a start gate is sized on the bound, never under it"
    assert m.concurrency_gib_per_seq < 2 * bound, "and it is not a number nobody measured"


def test_the_anchor_costs_nothing_extra_and_nothing_below_it_is_credited_back():
    """Everything the pair has banked must be estimated exactly as it was before this term existed —
    and a concurrency UNDER the anchor is charged as the anchor is, because what a seat releases was
    never measured and a credit nobody measured could let a start through that should have waited."""
    for seqs in (1, 8, 16):
        e = estimate(role="head", seqs=seqs, max_num_batched_tokens=CALIBRATED_BUDGET)
        assert e.concurrency_gib == 0.0, seqs
        assert e.total_gib == pytest.approx(103.2, abs=0.05), seqs
    m = shipped().get(FLASH).memory
    big = estimate(role="head", seqs=24, max_num_batched_tokens=CALIBRATED_BUDGET)
    assert big.concurrency_gib == pytest.approx(8 * m.concurrency_gib_per_seq, abs=1e-9)
    assert big.total_gib == pytest.approx(103.19 + big.concurrency_gib, abs=0.05)


def test_24_SEQUENCES_is_REFUSED_on_the_memory_the_boxes_really_have_and_16_is_ALLOWED():
    """THE TEST THIS EXISTS FOR, on the real recorded numbers. At 16 seats and batched 8192 the pair
    has banked generation after generation and rests at 11.3 GiB free. At 24 it rests at 1.7 GiB with
    NO traffic — inside the band that hung both boxes, and with nothing left for the handover that is
    the minute they died."""
    assert S24["boxes"]["mem_at_launch_gib"] == AT_LAUNCH
    for free in (AT_LAUNCH, 115.7, 116.0, 117.0):
        ok = estimate(role="head", seqs=16, available=free, max_num_batched_tokens=8192,
                      start_headroom_gib=HEADROOM)
        assert free >= ok.needed_now_gib, f"16 seats must start on {free} GiB: needs {ok.needed_now_gib}"
        assert not set(ok.codes) & {GATE, NO_HEADROOM, CONCURRENCY, BUDGET}, ok.codes

        no = estimate(role="head", seqs=24, available=free, max_num_batched_tokens=8192,
                      start_headroom_gib=HEADROOM)
        assert free < no.needed_now_gib, f"24 seats must be refused on {free} GiB: needs {no.needed_now_gib}"
        assert NO_HEADROOM in no.codes and CONCURRENCY in no.codes, no.codes


def test_the_16384_refusal_is_not_weakened_at_any_concurrency():
    """f1c8491's property, kept: the prefill budget stays refused, and stays NAMED, at every seat count."""
    for seqs in (8, 16, 24):
        no = estimate(role="head", seqs=seqs, available=115.7, max_num_batched_tokens=16384,
                      start_headroom_gib=HEADROOM)
        assert 115.7 < no.needed_now_gib, (seqs, no.needed_now_gib)
        assert BUDGET in no.codes and "batched 16,384" in no.message(BUDGET), (seqs, no.codes)


def test_the_estimate_at_24_seats_matches_what_the_box_really_rested_at():
    """Blind to concurrency the estimator said 103.2 GiB for a rank that really took 113.8 — a 10.6 GiB
    miss that read as 9.3 % and so sat just under the 10 % tolerance, unflagged. With the seats charged
    the miss is 1.0 GiB."""
    used = S24["implied_footprint"]["actual_used_gib_at_24_seats"]["box-a"]
    rest = sampled(90)["mem_available_gib"]
    assert used == pytest.approx(AT_LAUNCH - rest, abs=0.05), "what the box gave up, from its own reads"
    e = estimate(role="head", seqs=24, available=rest, max_num_batched_tokens=8192, actual_used_gib=used)
    assert e.calibrated_seqs_total_gib == pytest.approx(103.2, abs=0.05), "what it used to say"
    assert abs(e.calibrated_seqs_total_gib - used) == pytest.approx(S24["implied_footprint"]["miss_gib"],
                                                                   abs=0.1)
    assert e.total_gib == pytest.approx(
        S24["implied_footprint"]["estimated_total_gib_after_this_fix"]["box-a"], abs=0.05)
    assert abs(e.deviation_pct) < DEVIATION_PCT and DEVIATION not in e.codes


def test_the_refusal_names_the_concurrency_what_it_adds_and_what_it_would_fit_at():
    """The operator reads a dropdown, so the refusal has to name that dropdown: "the 24-sequence
    concurrency adds 9.6 GiB; at 16 this selection fits" tells them which one to turn."""
    no = estimate(role="head", seqs=24, available=AT_LAUNCH, max_num_batched_tokens=8192,
                  start_headroom_gib=HEADROOM)
    said = no.message(NO_HEADROOM) + " " + no.message(CONCURRENCY)
    assert "115.5" in said, "what the box has"
    assert f"{no.total_gib:.1f}" in said, "what this configuration needs"
    assert f"{HEADROOM:.0f}" in said, "and the headroom a start must leave"
    assert "seqs 24" in said, "which knob is responsible"
    assert f"{no.concurrency_gib:.1f}" in said, "how much of the requirement is that knob"
    assert "@ 16" in said and "fits" in said, "and what it would fit at"
    assert "seqs 24" in no.needs_note, no.needs_note


def test_the_concurrency_is_only_blamed_when_lowering_it_would_actually_help():
    """Same rule as the budget: `concurrency` is an instruction, so it must not appear on a box that
    could not hold the selection at the anchor either. There, the concurrency is not the problem."""
    hopeless = estimate(role="head", seqs=24, available=40.0, max_num_batched_tokens=8192,
                        start_headroom_gib=HEADROOM)
    assert GATE in hopeless.codes and CONCURRENCY not in hopeless.codes, hopeless.codes


def test_both_knobs_are_named_when_both_are_adding_to_the_requirement():
    """24 seats AND a 16,384 budget: neither knob alone brings this back, so neither may claim to —
    and the operator must be told both, instead of being left with a bare "need 130"."""
    no = estimate(role="head", seqs=24, available=115.7, max_num_batched_tokens=16384,
                  start_headroom_gib=HEADROOM)
    assert GATE in no.codes, no.codes
    assert {BUDGET, CONCURRENCY} <= set(no.codes), no.codes
    assert "batched 16,384" in no.needs_note and "seqs 24" in no.needs_note, no.needs_note
    assert "fits" not in no.message(BUDGET) and "fits" not in no.message(CONCURRENCY), \
        "neither knob alone fixes it, so neither says it does"


def test_a_serving_pair_at_24_sequences_is_never_warned_about_and_never_refused():
    """40b34b2's property, carried through the new term. The pair resting at 1.7 GiB free IS serving;
    telling it that it should not have started is noise. Only the estimate-against-actual survives."""
    rest = sampled(90)["mem_available_gib"]
    used = S24["implied_footprint"]["actual_used_gib_at_24_seats"]["box-a"]
    running = estimate(role="head", seqs=24, available=rest, max_num_batched_tokens=8192,
                       start_headroom_gib=HEADROOM, actual_used_gib=used)
    assert not set(running.codes) & {GATE, NO_HEADROOM, BUDGET, CONCURRENCY, FLOOR, NEAR_FLOOR}, \
        running.codes
    assert running.leaves_gib is not None, "what it leaves is still reported, just not warned about"


def test_the_concurrency_is_charged_on_both_ranks():
    """Only the head was sampled at 24 seats. Captured decode graphs and per-step activations are a
    per-rank cost, so both ranks are charged — the conservative side, and free in practice: the head
    needs 5.5 GiB more either way, so the head is the rank that gates."""
    head = estimate(role="head", seqs=24, max_num_batched_tokens=8192)
    worker = estimate(role="worker", seqs=24, max_num_batched_tokens=8192)
    assert head.concurrency_gib == worker.concurrency_gib > 0
    assert head.total_gib > worker.total_gib


def test_a_model_whose_concurrency_was_never_measured_is_not_charged_for_it():
    e = estimate(DENSE, role="head", seqs=48, max_num_batched_tokens=8192)
    assert e.concurrency_gib_per_seq == 0.0 and e.concurrency_gib == 0.0, "an anchor with no slope charges nothing"


def test_the_view_carries_the_concurrency_so_the_page_can_show_what_it_costs():
    v = estimate(role="head", seqs=24, available=AT_LAUNCH, max_num_batched_tokens=8192,
                 start_headroom_gib=HEADROOM).view()
    assert v["max_num_seqs"] == 24 and v["calibrated_max_num_seqs"] == CALIBRATED_SEQS
    assert v["concurrency_gib"] == round(v["concurrency_gib"], 1) and v["concurrency_gib"] > 0
    assert v["needed_now_gib"] == round(v["total_gib"] + HEADROOM, 1)
    assert json.loads(json.dumps(v)) == v
    assert {w["code"] for w in v["warnings"]} >= {NO_HEADROOM, CONCURRENCY}


# ── the pool knobs the operator can now set (cluster.KV_CACHE_MEMORY_RANGE, Desired.ple_replicate) ──────

def test_the_gate_charges_for_the_operators_ple_placement_not_the_profiles():
    """Where the PLE table lives moves 13.4 GiB on EVERY box, and the operator sets it now. If the estimate
    kept reading the profile, a split that fits would be refused — and worse, a replicated table the box
    cannot hold would be admitted, which is how both GB10 boxes hung on 2026-09-30."""
    profile = estimate(ple_replicate=None)                # hibrid48's profile: half the table on each rank
    replicated = estimate(ple_replicate=True)             # the operator's: the whole table on each rank
    whole = CATALOG_PLE_GIB = 26.82                       # catalog.json, read in the constants test above
    assert replicated.weights_gib - profile.weights_gib == pytest.approx(whole / 2, abs=0.01), \
        "a split table costs ple_gib / TP on each rank, so the saving is half the table"
    assert profile.total_gib < replicated.total_gib
    # Asking for the profile's own placement is the profile's own estimate, to the byte.
    assert estimate(ple_replicate=False).view() == profile.view()


def test_the_gate_charges_for_the_operators_kv_pin():
    """The pin is the single largest number the gate adds up, and it is settable now. The estimate has to
    follow the operator's, or the start gate is deciding on a pool the launch will not use."""
    default = estimate()
    assert default.kv_pinned_gib == pytest.approx(28_000_000_000 / GIB, abs=0.01)
    bigger = estimate(kv_cache_memory=36_000_000_000)
    assert bigger.kv_pinned_gib == pytest.approx(36_000_000_000 / GIB, abs=0.01)
    assert bigger.total_gib - default.total_gib == pytest.approx(8_000_000_000 / GIB, abs=0.01), \
        "every extra byte of pool is an extra byte the box must hold"
    assert estimate(kv_cache_memory=28_000_000_000).view() == default.view()


def test_a_bigger_pin_can_be_refused_where_the_default_one_fits():
    """The point of charging for it: the gate must be able to say no. A box that affords the shipped pool
    and not a larger one has to refuse the larger one, rather than discover it with an NVRM out-of-memory."""
    free = estimate().needed_now_gib + 0.5               # just enough for the configuration as it ships
    assert GATE not in estimate(available=free).codes and NO_HEADROOM not in estimate(available=free).codes
    tight = estimate(available=free, kv_cache_memory=40_000_000_000)
    assert GATE in tight.codes or NO_HEADROOM in tight.codes, "a pool the box cannot hold has to be refused"
    # And the saving goes the other way: the split table buys room the replicated one does not have.
    assert GATE not in estimate(available=free, kv_cache_memory=40_000_000_000,
                                ple_replicate=False).codes


# ── auto: the KV pool sized to what the boxes have left ───────────────────────
#
# Every figure below is read off lb/tests/fixtures/cluster-load-2026-10-01-memory.json — `sar -r` on both GB10 boxes,
# the LB's own reads, and generation 99's vLLM logs — never typed in.

LOAD = json.loads((FIXTURES / "cluster-load-2026-10-01-memory.json").read_text())
GB = 1_000_000_000
RANGE = (8 * GB, 96 * GB)


def generation_99(role: str):
    """The selection generation 99 ran — what it costs, and what auto would restart it at."""
    c = LOAD["generation_99"]["config"]
    return estimate(role=role, seqs=c["max_num_seqs"], ctx=c["max_model_len"], available=None,
                    max_num_batched_tokens=c["max_num_batched_tokens"], ple_replicate=c["ple_replicate"],
                    kv_cache_memory=c["kv_cache_memory"])


def auto_box(box="box-a", *, available, rest=None, returned=0.0, beside=60.0, load=6.0, floor=12.0) -> KvAutoBox:
    return KvAutoBox(box=box, available_gib=available, rest_gib=rest, returned_gib=returned,
                     beside_pool_gib=beside, floor_gib=floor, load_gib=load)


def test_auto_sizes_the_one_pool_both_ranks_pin_to_the_tighter_box():
    """TP=2 ranks take ONE pin, so the box with less room sizes both — whichever order they come in."""
    roomy = auto_box("box-b", available=110.0)          # 110 − 6 − 12 − 60 = 32 GiB of pool
    tight = auto_box("box-a", available=100.0)           # 100 − 6 − 12 − 60 = 22 GiB
    for boxes in ([roomy, tight], [tight, roomy]):
        out = resolve_kv_auto(boxes, baseline=None, step=GB, pin_range=RANGE)
        assert (out.kv_cache_memory, out.binding, out.refusal) == (23 * GB, "box-a", "")
    assert tight.pool_gib == pytest.approx(22.0) and roomy.pool_gib == pytest.approx(32.0)


def test_auto_keeps_the_floor_under_load_and_charges_the_load_only_once():
    """The floor is kept UNDER LOAD. A pair at rest has paid none of it, so all of it is charged; the same pair
    after its busy minutes has paid part of it already — the caching allocator keeps what load grew — and
    charging it all again on that lower figure would pay it twice (generation 99: 19.4 at rest, 13.5 loaded)."""
    g = LOAD["generation_99"]
    rest, loaded = g["at_rest_gib"]["box-a"], g["loaded_gib"]["box-a"]
    at_rest = auto_box(available=rest, rest=rest)
    assert at_rest.loaded_gib == pytest.approx(rest - 6.0)
    busy = auto_box(available=loaded, rest=rest)
    assert busy.loaded_gib == pytest.approx(min(loaded, rest - 6.0)), "the resting figure, minus the load, once"
    assert busy.loaded_gib > loaded - 6.0
    # Load that took MORE than the headroom is what the box really has, and that is what counts.
    heavy = auto_box(available=rest - 7.0, rest=rest)
    assert heavy.loaded_gib == pytest.approx(rest - 7.0)
    # With no resting figure, free now is taken as one — the whole load is charged on it.
    assert auto_box(available=loaded).loaded_gib == pytest.approx(loaded - 6.0)
    # And the pool is the one that leaves exactly the floor at that figure.
    pool = auto_box(available=rest, rest=rest, returned=96.3, beside=63.7)
    assert pool.pool_gib == pytest.approx(rest - 6.0 - 12.0 + 96.3 - 63.7)
    assert pool.left_gib(int(pool.pool_gib * GIB)) == pytest.approx(12.0, abs=1e-6)


def test_auto_rounds_down_to_the_panels_whole_gb_never_up():
    box = auto_box(available=100.0)                      # 22 GiB = 23.62 GB of room
    out = resolve_kv_auto([box], baseline=None, step=GB, pin_range=RANGE)
    assert out.kv_cache_memory == 23 * GB
    assert out.kv_cache_memory <= box.pool_gib * GIB < out.kv_cache_memory + GB
    # …but never under a pool the box CAN hold that is not on the step: rounding is not a reason to shrink.
    kept = resolve_kv_auto([box], baseline=int(23.5 * GB), step=GB, pin_range=RANGE)
    assert kept.kv_cache_memory == int(23.5 * GB)


def test_auto_refuses_in_the_gates_words_when_even_the_current_pool_would_go_under_the_floor():
    """Auto fills what is left; it never quietly shrinks the pool a pair already runs. A box that cannot keep the
    floor even at the current pool is a refusal, with its numbers — here the figure an LB has that began watching
    generation 99 under load, with no resting figure to credit."""
    loaded = LOAD["generation_99"]["loaded_gib"]["box-a"]
    box = auto_box("box-a", available=loaded, returned=96.3, beside=63.7)
    out = resolve_kv_auto([box], baseline=35 * GB, step=GB, pin_range=RANGE)
    left = loaded - 6.0 + 96.3 - 63.7 - 35 * GB / GIB
    assert (out.kv_cache_memory, out.binding) == (None, "box-a")
    assert out.refusal == (f"box-a: {loaded:.1f} GiB free now, {loaded - 6.0:.1f} under load — even the current "
                           f"35 GB pool would leave it {left:.1f} GiB, under broker_parrot's 12 GB danger floor: "
                           f"short {12.0 - left:.1f} GiB")
    # With no pool running, the floor of the range is the least it may resolve to.
    low = resolve_kv_auto([auto_box(available=80.0, beside=63.7)], baseline=None, step=GB, pin_range=RANGE)
    assert low.kv_cache_memory is None and "the smallest pool (8 GB)" in low.refusal and "short" in low.refusal


def test_auto_never_resolves_past_the_range_and_never_sizes_a_box_it_has_not_observed():
    out = resolve_kv_auto([auto_box(available=200.0, beside=10.0)], baseline=None, step=GB, pin_range=RANGE)
    assert out.kv_cache_memory == RANGE[1]
    blind = resolve_kv_auto([auto_box("box-b", available=None), auto_box(available=110.0)], baseline=None,
                            step=GB, pin_range=RANGE)
    assert blind.kv_cache_memory is None and blind.refusal.startswith("box-b: free memory not observed yet")


def test_the_load_is_the_models_activation_slope_up_to_the_calibrated_budget_and_never_charged_twice():
    at = lambda budget: estimate(max_num_batched_tokens=budget)          # noqa: E731
    assert at(4096).load_gib == pytest.approx(1.1 * 4)
    assert at(8192).load_gib == pytest.approx(1.1 * 8)
    # Past the calibration M8 already charges it as `activation_gib`: together the slope over the whole budget.
    assert at(16384).load_gib == pytest.approx(1.1 * 8)
    assert at(16384).load_gib + at(16384).activation_gib == pytest.approx(1.1 * 16)
    assert estimate().load_gib == 0.0, "no budget given, none charged"


def test_the_gateways_load_headroom_is_never_under_what_load_took_at_the_budget_that_ships():
    """The operator's rule for the minimum: never less than measured. At budget 4,096 (generations 98 and 99) the
    configured minimum is what auto charges, since the slope says less; at 8,192 the slope is what it charges,
    and what that misses of the worst measured load comes out of the floor it keeps — leaving the box far above
    the 0.8-4.2 GiB band that hung both GB10 boxes on 2026-09-30."""
    from vllm_lb.config import Config

    cfg = Config()
    at_4096 = [LOAD[g]["load_gib"][b] for g in ("generation_99", "generation_98") for b in ("box-a", "box-b")]
    assert cfg.cluster_kv_auto_load_gib >= max(at_4096) > estimate(max_num_batched_tokens=4096).load_gib
    at_8192 = [v for g in LOAD["budget_8192"]["load_gib"].values() for v in g.values()]
    charged = max(cfg.cluster_kv_auto_load_gib, estimate(max_num_batched_tokens=8192).load_gib)
    assert cfg.cluster_floor_gib - (max(at_8192) - charged) > 4.2


def test_beside_the_pool_is_everything_the_selection_costs_but_its_pool():
    head = generation_99("head")
    assert head.beside_pool_gib == pytest.approx(head.total_gib - 35 * GB / GIB)
    assert head.beside_pool_gib == pytest.approx(head.weights_gib + head.mtp_gib + head.overhead_gib)


def test_generation_99s_own_figures_resolve_auto_to_the_pin_this_change_recommends():
    """The recommendation, as arithmetic anyone can re-run: generation 99's resting and loaded figures, its own
    estimate returned and charged again for the same selection — so the difference is exactly its 35 GB pin —
    and the gateway's policy (12 GB floor, 6 GiB load at least)."""
    g = LOAD["generation_99"]
    boxes = []
    for box, role in (("box-a", "head"), ("box-b", "worker")):
        running = generation_99(role)
        boxes.append(KvAutoBox(box=box, available_gib=g["loaded_gib"][box], rest_gib=g["at_rest_gib"][box],
                               returned_gib=running.total_gib, beside_pool_gib=running.beside_pool_gib,
                               floor_gib=12.0, load_gib=max(6.0, running.load_gib)))
    out = resolve_kv_auto(boxes, baseline=g["config"]["kv_cache_memory"], step=GB, pin_range=RANGE)
    assert (out.kv_cache_memory, out.binding) == (36 * GB, "box-a")
    # box-a: min(13.5, 19.4 − 6) − 12 + 32.6 = 34.0 GiB = 36.5 GB, rounded down
    assert boxes[0].pool_gib == pytest.approx(19.4 - 6.0 - 12.0 + 35 * GB / GIB)


# ── UX audit 2026-10-02 (1-3): a launched rank's figures, and figures instead of sentences ──

#: Generation 105 as /lb/status published it (2026-10-02 12:0xZ): the dense 27B at 32 seats, batched 4,096,
#: 48 GB pin — box-a had 115.5 GiB free at launch, 44.6 now; the estimate said 68.0 and the box gave up 70.9.
GEN_105 = {"box-a": {"at_launch": 115.5, "now": 44.6, "actual": 70.9}}


def test_once_launched_what_the_box_leaves_is_what_it_had_at_launch_minus_the_cost():
    """`leaves` subtracted the pair from what is free WITH the pair loaded — twice: the page read
    `leaves -22.2 GiB` beside 45.8 GiB free. A launched rank leaves (free at launch − total)."""
    g = GEN_105["box-a"]
    e = estimate(DENSE, seqs=32, ctx=262144, available=g["now"], actual_used_gib=g["actual"],
                 launched_gib=g["at_launch"], max_num_batched_tokens=4096)
    assert e.leaves_gib == pytest.approx(g["at_launch"] - e.total_gib, abs=1e-9)
    assert e.leaves_gib > 40, e.leaves_gib
    assert e.view()["leaves_gib"] == round(g["at_launch"] - e.total_gib, 1)
    # Before a launch nothing changes: what is free now is what it would leave from.
    assert estimate(DENSE, available=g["at_launch"]).leaves_gib == pytest.approx(
        g["at_launch"] - estimate(DENSE, available=g["at_launch"]).total_gib)


def test_every_warning_is_a_figure_line_in_gib_and_the_pin_cap_is_the_pairs_once():
    """Operator rule: figures, not prose. One unit (GiB) in every line, the cap said once for the pair
    (scope `pair`, which the page shows once), every other warning is this box's (`box`)."""
    tight = estimate(role="head", seqs=48, ctx=262144, available=100.0, max_num_batched_tokens=16384,
                     start_headroom_gib=8.0)
    for code in tight.codes:
        line = tight.message(code)
        assert len(line) <= 72, (code, line)
        assert "GB" not in line.replace("GiB", ""), ("one unit", code, line)
        assert not line.endswith("."), (code, line)
        assert "this selection" not in line and "broker_parrot" not in line, (code, line)
    pin = estimate(seqs=48, ctx=262144)
    assert pin.message(PIN_CAPS) == "pin 1.71M < 12.58M tokens · fits 6 @ 262,144", pin.message(PIN_CAPS)
    scopes = {w["code"]: w["scope"] for w in pin.view()["warnings"]}
    assert scopes[PIN_CAPS] == "pair"
    floor = estimate(available=114.8)
    assert floor.message(FLOOR) == "leaves 11.6 < floor 12 GiB", floor.message(FLOOR)
    assert {w["code"]: w["scope"] for w in floor.view()["warnings"]}[FLOOR] == "box"
    gate = estimate(role="head", seqs=8, available=40.0, max_num_batched_tokens=8192, start_headroom_gib=8.0)
    assert gate.message(GATE) == f"free 40.0 < need {gate.needed_now_gib:.1f} GiB", gate.message(GATE)
    off = estimate(available=11.6, actual_used_gib=60.0, launched_gib=114.8)
    assert off.message(DEVIATION) == f"est {off.total_gib:.1f} vs actual 60.0 GiB (+72 %)", off.message(DEVIATION)


# ── the dense 27B, calibrated on generations 103 and 104 (2026-10-02) ─────────

DENSE_RUN = json.loads((FIXTURES / "cluster-27b-2026-10-02-memory.json").read_text())


def test_the_dense_27b_constants_are_read_off_generations_103_and_104_not_flash_nexts():
    """Audit 2026-10-02 (2): the 27B's catalog entry carried Flash-Next's per-role overhead (12.66 / 7.16) and
    an empty calibration. Measured on two generations of the real pair: head 14.8 GiB, worker 11.2."""
    spec = shipped().get(DENSE)
    m, profile = spec.memory, spec.profile(CLUSTER_MODE)
    lines = DENSE_RUN["engine_log_lines_generation_104"]
    loaded = float(re.search(r"Model loading took ([\d.]+) GiB", lines["rank0"][0]).group(1))         # 10.68
    assert abs(m.weights_gib + m.mtp_gib - loaded) < 0.01, "weights + drafter = what vLLM loaded, per rank"
    pin = DENSE_RUN["config"]["kv_cache_memory"] / GIB
    for gen in ("generation_103", "generation_104"):
        for box, role in DENSE_RUN["roles"].items():
            used = DENSE_RUN[gen]["actual_used_gib"][box]
            measured = used - m.weights_gib - m.mtp_gib - pin
            assert profile.overhead_gib[role] >= measured - 0.01, (gen, role, measured)
            assert profile.overhead_gib[role] - measured < 0.5, (gen, role, "the catalog is not padded past the run")
    assert "2026-10-02" in m.calibration and "103" in m.calibration and "104" in m.calibration
    assert spec.calibrated
    assert (m.calibrated_max_num_seqs, m.calibrated_batched_tokens) == (16, 8192), "the anchors the run was at"
    assert (m.concurrency_gib_per_seq, m.activation_gib_per_1k_tokens) == (0.0, 0.0), "slopes nobody measured"
    for box, role in DENSE_RUN["roles"].items():
        e = estimate(DENSE, role=role, seqs=16, ctx=262144, max_num_batched_tokens=8192,
                     available=DENSE_RUN["generation_104"]["mem_at_launch_gib"][box],
                     actual_used_gib=DENSE_RUN["generation_104"]["actual_used_gib"][box])
        assert abs(e.deviation_pct) < 2.0, (role, e.total_gib, e.actual_used_gib)
        assert e.view()["calibration"] != ""


# ── the KV floor: a pin under one full-context request kills the engine at init ──

def test_a_pin_under_one_full_context_request_is_a_warning_of_its_own_and_the_seats_are_the_engines():
    """Audit 2026-10-02 (3, 10): the engine needs ceil(ctx / block_tokens) + F blocks for ONE request —
    340 × 28.4 MB = 9.66 GB on the 27B at 262,144 — so an 8-9 GB pin passed the LB and vLLM died at init.
    Below that the pin is refused as `pin_floor`; above it the seats beside the cap are the engine's."""
    from vllm_lb.memory_estimate import PIN_FLOOR
    dead = estimate(DENSE, seqs=1, ctx=262144, kv_cache_memory=8_000_000_000)
    assert PIN_FLOOR in dead.codes and PIN_CAPS not in dead.codes, dead.codes
    assert dead.message(PIN_FLOOR) == "pin 7.5 < 9.0 GiB for one 262,144 request (340 blocks)", \
        dead.message(PIN_FLOOR)
    assert dead.view()["kv_floor_gib"] == 9.0 and dead.scope(PIN_FLOOR) == "pair"
    alive = estimate(DENSE, seqs=1, ctx=65536, kv_cache_memory=8_000_000_000)
    assert PIN_FLOOR not in alive.codes, alive.codes
    capped = estimate(DENSE, seqs=32, ctx=65536)
    assert capped.pin_seats == 17, "1689 // (81 + 18), not 1,302,224 // 65,536 = 19"
    assert capped.message(PIN_CAPS).endswith("fits 17 @ 65,536"), capped.message(PIN_CAPS)
    served = estimate(DENSE, seqs=32, ctx=65536, engine_blocks=1980)
    assert served.pin_seats == 20, "the engine's own block count while it serves"


# ── SGLang: a GDN checkpoint slot is paid for by the KV pool, never by the box ─────────────────────────
#
# Two committed SGLang starts of the same model at the same --mem-fraction-static 0.80: generation 123's
# (16 seats, --max-mamba-cache-size 80) and generation 131's (32 seats, 160). SGLang subtracts the mamba
# pool — and the per-seat verify scratch — from the fraction's budget and gives the KV pools the REST
# (srt/mem_cache/kv_cache_configurator.py `_profile_available_bytes`: rest = free − slack, then
# `_handle_max_mamba_cache`), so what `mamba_slots_per_req` buys comes out of KV tokens, and the start
# gate, which adds weights + pin + overhead, must not charge it a second time on top.
SGLANG = "qwen3.8-27b-sglang-dflash2"
SGLANG_STARTS = {80: FIXTURES / "sglang-lb-start-2026-10-03-rank0.log",
                 160: FIXTURES / "sglang-hung-start-2026-10-03-gen131-rank0.log"}
#: One GDN state slot on one rank, off the checkpoint's config.json (48 linear-attention layers, 48 value
#: heads x 128 x 128, conv channels 2 x 16 x 128 + 48 x 128 over kernel 4) at TP=2 and --mamba-ssm-dtype
#: bfloat16: the ssm state plus the conv state's 3 taps, 2 bytes each.
GDN_SLOT_BYTES = 48 * ((48 // 2) * 128 * 128 + (2 * 16 * 128 + 48 * 128) // 2 * 3) * 2


def sglang_pools(slots: int) -> dict:
    text = SGLANG_STARTS[slots].read_text()
    m = re.search(r"max_mamba_cache_size: (\d+), conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB "
                  r"intermediate_ssm_state_cache size: ([\d.]+)GB intermediate_conv_window_cache size: ([\d.]+)GB",
                  text)
    kv = re.findall(r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB",
                    text)
    left = re.findall(r"Memory pool end\. avail mem=([\d.]+) GB", text)
    assert m and len(kv) == 2 and len(left) == 2, f"{SGLANG_STARTS[slots].name} no longer logs its pools"
    assert int(m.group(1)) == slots and kv[0][0] == kv[1][0], "one token count for the target and draft pools"
    return {"slots": (float(m.group(2)) + float(m.group(3))) * GIB,
            "scratch": (float(m.group(4)) + float(m.group(5))) * GIB,
            "kv": sum(float(k) + float(v) for _, k, v in kv) * GIB, "tokens": int(kv[0][0]),
            "left": float(left[-1]) * GIB}


def test_an_sglang_gdn_slot_costs_kv_tokens_and_the_box_holds_the_same_whatever_the_slot_count():
    small, big = sglang_pools(80), sglang_pools(160)
    # The slot is what the config says it is: the engine allocates slots + 1 (its padding slot).
    for pools, allocated in ((small, 81), (big, 161)):
        assert pools["slots"] / allocated == pytest.approx(GDN_SLOT_BYTES, rel=0.005)
    # The mamba side grew 5.2 GiB (80 more slots, and 16 more seats of verify scratch) …
    grew = (big["slots"] + big["scratch"]) - (small["slots"] + small["scratch"])
    assert 5.0 * GIB < grew < 5.4 * GIB, grew / GIB
    # … the KV pools gave nearly all of it back, and the box kept the SAME memory beside the pools:
    # 21.74 GiB free after them at 80 slots, 21.83 at 160. The fraction, not the slot count, sets the total.
    assert abs(big["left"] - small["left"]) < 0.2 * GIB
    assert abs((small["kv"] - big["kv"]) - grew) < 0.6 * GIB
    # So a slot costs this many KV tokens, and the pool the shipped setting leaves at 32 seats is predicted
    # from the 160-slot start: the number to read back off the next start's own 'KV Cache is allocated'.
    kv_bytes_per_token = small["kv"] / small["tokens"]
    assert round(GDN_SLOT_BYTES / kv_bytes_per_token) == 1824
    p = shipped().get(SGLANG).profile(CLUSTER_MODE)
    slots = 32 * p.engine_args["mamba_slots_per_req"]
    predicted = big["tokens"] - (slots - 160) * GDN_SLOT_BYTES / kv_bytes_per_token
    assert predicted >= 10 * 262144, f"{predicted:,.0f} tokens: under ten whole contexts is too much KV given up"
    # And the start gate does not charge the slots on top: the estimate is the same whatever the count.
    doc = document()
    entry = next(e for e in doc["models"] if e["id"] == SGLANG)
    entry["launch"]["cluster"]["engine_args"]["mamba_slots_per_req"] = 5
    five = estimate(SGLANG, catalog=ModelCatalog.from_dict(doc), seqs=32)
    assert estimate(SGLANG, seqs=32).total_gib == five.total_gib


def test_a_default_that_runs_the_drafter_off_charges_no_drafter_weights():
    """None is the profile's DEFAULT depth. DeepSeek-V4-Flash keeps its dspark config but is tuned to 0 (its drafter
    cannot load): a launch naming no depth loads no drafter — launch_env sends none — so the gate must not charge its
    5.1 GiB. Generation 249 (2026-10-26) was refused 4 GiB short on box-b partly for weights that never load."""
    spec = shipped().get("deepseek-v4-flash-nvfp4")
    profile = spec.profile("cluster")
    assert profile.speculative_config is not None and profile.num_speculative_tokens == 0 and spec.memory.mtp_gib > 0
    assert estimate("deepseek-v4-flash-nvfp4", seqs=8, ctx=131072).mtp_gib == 0.0
    assert estimate("deepseek-v4-flash-nvfp4", seqs=8, ctx=131072, speculative=True).mtp_gib == spec.memory.mtp_gib
    assert estimate(seqs=1, ctx=131072).mtp_gib == shipped().get(FLASH).memory.mtp_gib, "Flash-Next's default is MTP 4"
