"""S3 — the cluster switch lives on disk. A restarted LB re-reads it; no process remembers it.

Pass 4 (worklog/vllm-model-selection.md, M2): the switch also holds the cluster's ModelChoice — validated by
`ModelCatalog.choice`, bounded by that ModelSpec's ModelLimits, frozen while the cluster is on, carried across a
Stop and an LB restart."""
from __future__ import annotations

import json

import pytest

from vllm_lb.catalog import document, ModelCatalog, UnknownModelChoice
from vllm_lb.cluster import CLASSIC, CLUSTER_MODE, KV_AUTO, Desired, DesiredStore

FLASH, DENSE, SEEDED = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4", "qwen3.8-27b-nvfp4"
#: The SGLang pair — the other model with a measured-best start input of its own.
SGLANG = "qwen3.8-27b-sglang-dflash2"


def store(tmp_path, **kw) -> DesiredStore:
    return DesiredStore(str(tmp_path / "state" / "cluster.json"), **kw)


def catalog_with(**dense) -> ModelCatalog:
    """The shipped catalog with Qwen3.8-27B's ModelSpec changed — its stated start input kept inside new limits."""
    doc = document()
    row = next(m for m in doc["models"] if m["id"] == DENSE)
    row.update(dense)
    tuned = row["launch"]["cluster"]["tuned"]
    tuned["max_model_len"] = min(tuned["max_model_len"], row["limits"]["max_model_len"])
    tuned["max_num_seqs"] = min(tuned["max_num_seqs"], row["limits"]["max_num_seqs"][1])
    return ModelCatalog.from_dict(doc)


def uncalibrated() -> ModelCatalog:
    """The shipped catalog with the dense 27B's footprint UNMEASURED again — the model it was until 2026-10-02."""
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["memory"].update(
        calibration="", calibrated_batched_tokens=0, calibrated_max_num_seqs=0)
    return ModelCatalog.from_dict(doc)


def without_drafter() -> ModelCatalog:
    """The shipped catalog with the dense model's drafter taken away: a ModelSpec with no speculative
    decoding, which is what a checkpoint without an MTP head is (the fleet's own llm-compressor build of the
    27B lost its head; NVIDIA's, the one shipped, keeps it)."""
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["launch"]["cluster"].update(speculative_config=None,
                                                                                 mtp_prefix="")
    return ModelCatalog.from_dict(doc)


def write_file(tmp_path, **fields) -> None:
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def test_no_file_is_the_classic_default(tmp_path):
    """…at the DEFAULT MODEL'S OWN measured-best concurrency and context (`catalog.TunedSetup`): a number from the
    environment cannot know which model is about to be launched, and the store takes none (2026-10-29)."""
    s = store(tmp_path)
    tuned = ModelCatalog.load().get(FLASH).profile("cluster").tuned
    assert s.load() == Desired(mode=CLASSIC, generation=0, max_num_seqs=tuned.max_num_seqs,
                               max_model_len=tuned.max_model_len, model_choice=FLASH, uncensored=True)
    assert s.last_error == ""


def test_every_model_is_offered_its_own_rows_seats_and_context(tmp_path):
    """The other half of the rule above: there is no gateway pair to fall back on, so EVERY cluster row states its
    own (the dense 27B: the 6 at 262,144 the gateway's former default gave it, written into its row with that
    source) — `defaults_for` is the row's `tuned`, and the store cannot be handed another number."""
    import pytest
    s = store(tmp_path)
    for spec in ModelCatalog.load().for_mode(CLUSTER_MODE):
        tuned = spec.profile("cluster").tuned
        offered = s.defaults_for(spec.id)
        assert (offered.max_num_seqs, offered.max_model_len, offered.model_choice) == (
            tuned.max_num_seqs, tuned.max_model_len, spec.id), spec.id
    with pytest.raises(TypeError):
        store(tmp_path, max_num_seqs=12, max_model_len=131072)


def test_start_bumps_the_generation_and_survives_a_new_process(tmp_path):
    first = store(tmp_path).set(enabled=True, max_num_seqs=8, max_model_len=65536)
    assert first == Desired(mode=CLUSTER_MODE, generation=1, max_num_seqs=8, max_model_len=65536,
                            model_choice=FLASH, uncensored=True)
    assert store(tmp_path).load() == first, "a fresh store — a restarted LB — reads the same switch"
    assert not list((tmp_path / "state").glob("*.tmp")), "writes are atomic: no temp file is left"


def test_stop_keeps_the_generation_and_the_next_start_is_a_new_one(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True)
    stopped = s.set(enabled=False)
    assert stopped.mode == CLASSIC and stopped.generation == 1
    assert s.set(enabled=True).generation == 2


def test_start_while_on_is_idempotent(tmp_path):
    s = store(tmp_path)
    on = s.set(enabled=True)
    assert s.set(enabled=True) == on


@pytest.mark.parametrize("kw", [{"max_num_seqs": 0}, {"max_num_seqs": 49},
                                {"max_model_len": 1023}, {"max_model_len": 262145}])
def test_settings_outside_the_limits_are_refused_and_nothing_is_written(tmp_path, kw):
    with pytest.raises(ValueError):
        store(tmp_path).set(enabled=True, **kw)
    assert store(tmp_path).load().mode == CLASSIC


def test_settings_cannot_change_under_a_running_cluster_but_may_ride_on_a_stop(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True, max_num_seqs=8)
    with pytest.raises(ValueError, match="stop the cluster"):
        s.set(enabled=True, max_num_seqs=16)
    assert s.set(enabled=False, max_num_seqs=16).max_num_seqs == 16


def test_every_start_names_its_reservations_afresh_and_stop_keeps_the_name(tmp_path):
    """The reservations' identity (leases.cluster_reservation_id) is generation + this id + box. A
    generation number reused after the file is lost must never name an old reservation — that
    would read as "the reservation ended" and fail a fresh Start."""
    s = store(tmp_path)
    first = s.set(enabled=True)
    _, start_1 = s.current()
    assert start_1, "a Start writes a fresh id"
    assert store(tmp_path).current() == (first, start_1), "a restarted LB reads the same pair"
    stopped = s.set(enabled=False)
    assert s.current() == (stopped, start_1), "Stop releases THIS start's reservations: it keeps the id"
    s.set(enabled=True)
    assert s.current()[1] not in (None, start_1)
    reset = store(tmp_path / "elsewhere")
    reset.set(enabled=True)
    assert reset.current()[0].generation == first.generation and reset.current()[1] != start_1


def test_a_file_written_before_start_ids_existed_still_loads(tmp_path):
    write_file(tmp_path, mode="cluster", generation=2, max_num_seqs=1, max_model_len=262144)
    desired, start_id = store(tmp_path).current()
    assert (desired.mode, desired.generation, start_id) == (CLUSTER_MODE, 2, None)


def test_an_unreadable_file_is_the_classic_default_and_says_why(tmp_path):
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir()
    path.write_text("{not json")
    s = store(tmp_path)
    assert s.load().mode == CLASSIC and s.last_error


# ── the ModelChoice (M2) ──────────────────────────────────────────────────────

def test_a_model_choice_is_written_by_start_and_survives_a_new_process(tmp_path):
    on = store(tmp_path).set(enabled=True, model_choice=DENSE)
    assert on.model_choice == DENSE and on.mode == CLUSTER_MODE
    assert store(tmp_path).load() == on, "an LB restart reads the same ModelChoice"


@pytest.mark.parametrize("bad", ["nope", SEEDED, 7], ids=["no-such-model", "a-classic-only-model", "not-an-id"])
def test_an_unknown_model_choice_is_refused_as_unknown_and_nothing_is_written(tmp_path, bad):
    with pytest.raises(UnknownModelChoice):
        store(tmp_path).set(enabled=True, model_choice=bad)
    assert not (tmp_path / "state" / "cluster.json").exists()


def test_the_model_choice_cannot_change_under_a_running_cluster_but_may_ride_on_a_stop(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True, model_choice=FLASH)
    with pytest.raises(ValueError, match="stop the cluster before changing its model choice"):
        s.set(enabled=True, model_choice=DENSE)
    assert s.set(enabled=True, model_choice=FLASH).generation == 1, "the same choice again is no change"
    assert s.set(enabled=False, model_choice=DENSE).model_choice == DENSE
    again = s.set(enabled=True)
    assert (again.generation, again.model_choice) == (2, DENSE), "the next Start loads what rode on the Stop"


def test_a_restart_bumps_the_generation_without_ever_passing_through_classic(tmp_path):
    """D1: a Stop is what hands the GB10 boxes back to the fleet, so a restart at new settings must never be
    one. `restart=True` is the operator saying "reload the pair at these settings": the generation is
    bumped, the mode stays `cluster`, and the start id is new so the next generation's reservation ids
    can never name the old rows (leases.cluster_reservation_id)."""
    s = store(tmp_path)
    on = s.set(enabled=True, max_num_seqs=2)
    _, first_start = s.current()
    again = s.set(enabled=True, restart=True, max_num_seqs=8, max_model_len=131072)
    after, start_id = s.current()
    assert (again.mode, again.generation) == (CLUSTER_MODE, on.generation + 1)
    assert (again.max_num_seqs, again.max_model_len) == (8, 131072)
    assert after == again, "a restarted LB reads the restart back"
    assert start_id and start_id != first_start, "the new generation needs its own start id"


def test_a_restart_needs_no_change_and_a_change_still_needs_the_restart_flag(tmp_path):
    """Two halves of the same rule. Asking for a restart with the same settings is still a restart — the
    operator may simply want the pair reloaded. Changing the settings WITHOUT asking for one is still
    refused, so no stray Start can replace a serving pair by accident."""
    s = store(tmp_path)
    s.set(enabled=True, max_num_seqs=2)
    assert s.set(enabled=True, restart=True).generation == 2, "a restart at the same settings is a restart"
    with pytest.raises(ValueError, match="stop the cluster before changing its model choice"):
        s.set(enabled=True, max_num_seqs=8)
    assert s.load().max_num_seqs == 2, "the refused change was not written"


def test_a_file_written_before_model_choices_existed_loads_the_catalogs_cluster_default(tmp_path):
    write_file(tmp_path, mode="cluster", generation=2, max_num_seqs=1, max_model_len=262144, start_id="ab")
    assert store(tmp_path).load().model_choice == FLASH, "the model tp-rank launched before model choices existed"


def test_context_and_concurrency_are_bounded_by_the_chosen_models_own_limits(tmp_path):
    small = catalog_with(limits={"max_model_len": 131072, "max_num_seqs": [1, 8]})
    for bad in ({"max_model_len": 262144}, {"max_num_seqs": 9}):     # a switch given nothing takes the row's own
        with pytest.raises(ValueError, match="Qwen3.8 27B"):
            store(tmp_path, catalog=small).set(enabled=True, model_choice=DENSE, **bad)
    ok = store(tmp_path, catalog=small).set(enabled=True, model_choice=DENSE, max_model_len=131072, max_num_seqs=8)
    assert (ok.max_model_len, ok.max_num_seqs) == (131072, 8)


def test_an_unsupported_model_is_refused_with_its_evidence(tmp_path):
    marked = catalog_with(unsupported="vLLM 0.29 on v5: Qwen3_5ForConditionalGeneration cannot run at TP=2")
    with pytest.raises(ValueError, match="cannot run at TP=2") as info:
        store(tmp_path, catalog=marked).set(enabled=True, model_choice=DENSE)
    assert not isinstance(info.value, UnknownModelChoice)


def test_stop_never_needs_a_valid_choice(tmp_path):
    """A choice the catalog no longer has — an entry removed by a redeploy — must never block a Stop."""
    write_file(tmp_path, mode="cluster", generation=3, max_num_seqs=16, max_model_len=262144,
               model_choice="a-removed-model", start_id="cd")
    s = store(tmp_path)
    stopped = s.set(enabled=False)
    assert (stopped.mode, stopped.model_choice) == (CLASSIC, "a-removed-model")
    with pytest.raises(UnknownModelChoice):
        s.set(enabled=True)


def test_a_catalog_that_could_not_be_loaded_refuses_start_with_the_reason_and_still_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("LB_FLEET", str(tmp_path / "missing.yml"))
    broken = ModelCatalog.load()
    assert broken.error
    s = store(tmp_path, catalog=broken)
    assert s.load().model_choice is None
    with pytest.raises(ValueError, match="catalog"):
        s.set(enabled=True)
    assert s.set(enabled=False).mode == CLASSIC


# ── the engine knobs: the prefill budget and the drafter's depth ──────────────

def test_a_file_from_before_the_knobs_loads_as_their_defaults(tmp_path):
    """Backwards compatibility: the live switch on 2026-09-30 (generation 52, serving) has neither field. It
    must load unchanged, with both knobs at null — the defaults, i.e. exactly what that pair was launched with."""
    write_file(tmp_path, mode="cluster", generation=52, max_num_seqs=8, max_model_len=262144, model_choice=FLASH,
               start_id="dd20fa7af3c44ee9bdbc3bc1811fe377", updated_at=1790739322.7488)
    s = store(tmp_path)
    desired, start_id = s.current()
    assert desired == Desired(mode=CLUSTER_MODE, generation=52, max_num_seqs=8, max_model_len=262144,
                              model_choice=FLASH, max_num_batched_tokens=None, num_speculative_tokens=None)
    assert start_id == "dd20fa7af3c44ee9bdbc3bc1811fe377" and s.last_error == ""
    assert s.set(enabled=True) == desired, "a Start on it is still the idempotent no-op it was"


def test_a_knob_in_the_file_is_read_as_leniently_as_every_other_field(tmp_path):
    """An unreadable switch reads as classic — which takes a running pair down — so the knobs must add no new
    way for the file to become unreadable: "16384" in a hand-edited file reads like "8" does in max_num_seqs."""
    write_file(tmp_path, mode="cluster", generation=3, max_num_seqs="8", max_model_len=262144,
               max_num_batched_tokens="16384", num_speculative_tokens=None, start_id="ab")
    s = store(tmp_path)
    d = s.load()
    assert (d.mode, d.max_num_seqs, d.max_num_batched_tokens, d.num_speculative_tokens) == (CLUSTER_MODE, 8, 16384, None)
    assert s.last_error == ""


def test_the_knobs_are_written_by_a_start_and_survive_a_new_process(tmp_path):
    on = store(tmp_path).set(enabled=True, max_num_batched_tokens=16384, num_speculative_tokens=2)
    assert (on.max_num_batched_tokens, on.num_speculative_tokens) == (16384, 2)
    assert store(tmp_path).load() == on, "a restarted LB reads the operator's knobs back"
    raw = json.loads((tmp_path / "state" / "cluster.json").read_text())
    assert (raw["max_num_batched_tokens"], raw["num_speculative_tokens"]) == (16384, 2)


def test_null_is_the_default_and_a_knob_left_out_keeps_what_is_stored(tmp_path):
    """None is a VALUE for a knob — the default — so "not given" is UNCHANGED: a Stop that says nothing about
    them must not reset the operator's choice, and an explicit null must."""
    s = store(tmp_path)
    s.set(enabled=True, max_num_batched_tokens=4096, num_speculative_tokens=0)
    kept = s.set(enabled=False)
    assert (kept.max_num_batched_tokens, kept.num_speculative_tokens) == (4096, 0), "a Stop reset the knobs"
    again = s.set(enabled=True)
    assert (again.max_num_batched_tokens, again.num_speculative_tokens) == (4096, 0)
    s.set(enabled=False)
    reset = s.set(enabled=True, max_num_batched_tokens=None, num_speculative_tokens=None)
    assert (reset.max_num_batched_tokens, reset.num_speculative_tokens) == (None, None)


@pytest.mark.parametrize("kw", [{"max_num_batched_tokens": 2047}, {"max_num_batched_tokens": 65537},
                                {"max_num_batched_tokens": True}, {"max_num_batched_tokens": "8192"},
                                {"num_speculative_tokens": -1}, {"num_speculative_tokens": 9},
                                {"num_speculative_tokens": False}, {"num_speculative_tokens": 2.0}],
                         ids=repr)
def test_knobs_outside_what_tp_rank_and_the_page_take_are_refused_and_nothing_is_written(tmp_path, kw):
    """2048..65536 is tp-rank's own bound on CLUSTER_MAX_NUM_BATCHED_TOKENS; 0..8 is what the page offers for
    the drafter, 0 being speculation off. A bool is not a number here, whatever Python says."""
    with pytest.raises(ValueError, match=next(iter(kw))):
        store(tmp_path).set(enabled=True, **kw)
    assert not (tmp_path / "state" / "cluster.json").exists()


@pytest.mark.parametrize("value", [2048, 65536])
def test_the_prefill_budgets_own_bounds_are_accepted(tmp_path, value):
    assert store(tmp_path).set(enabled=True, max_num_batched_tokens=value).max_num_batched_tokens == value


def test_a_drafter_depth_for_a_model_without_a_drafter_is_refused_but_off_is_not(tmp_path):
    """A LaunchProfile with no speculative decoding has no drafter to set: asking it for 2 tokens would launch
    something the catalog never described. 0 — and the default — are what it runs."""
    no_drafter = without_drafter()
    with pytest.raises(ValueError, match=r"Qwen3\.8 27B .*has no speculative decoding"):
        store(tmp_path, catalog=no_drafter).set(enabled=True, model_choice=DENSE, num_speculative_tokens=2)
    assert not (tmp_path / "state" / "cluster.json").exists()
    assert store(tmp_path, catalog=no_drafter).set(enabled=True, model_choice=DENSE,
                                                   num_speculative_tokens=0).num_speculative_tokens == 0


def test_a_stored_depth_is_checked_against_the_model_it_would_launch_with(tmp_path):
    """CHANGED 2026-10-02 (audit #6). This test used to say the depth OUTLIVES a change of model, and that the
    Start of a drafterless model is then refused. A model-scoped knob no longer rides across: changing the
    model resets it to the new model's default, so that Start launches. What still holds is the check itself —
    a depth stored for a model that cannot take it (a redeployed catalog), or named with the change, is refused."""
    s = store(tmp_path, catalog=without_drafter())
    s.set(enabled=True, num_speculative_tokens=2)
    assert s.set(enabled=False, model_choice=DENSE).num_speculative_tokens is None, "2 rode into another model"
    assert s.set(enabled=True).model_choice == DENSE
    s.set(enabled=False)
    s.set(enabled=False, model_choice=FLASH)                        # back to the model with a drafter …
    with pytest.raises(ValueError, match="no speculative decoding"):
        s.set(enabled=False, model_choice=DENSE, num_speculative_tokens=2)   # … and across WITH the depth named
    write_file(tmp_path, mode="classic", generation=7, max_num_seqs=8, max_model_len=262144, model_choice=DENSE,
               num_speculative_tokens=2, start_id="ab")             # a file from before the catalog changed
    with pytest.raises(ValueError, match="no speculative decoding"):
        store(tmp_path, catalog=without_drafter()).set(enabled=True)


def test_a_changed_model_takes_its_own_pool_placement_and_depth_and_keeps_the_engine_wide_knobs(tmp_path):
    """Audit #6: the live switch held kv 30 GB, a split PLE table and a 2,048 budget, all chosen for Flash-Next;
    a change to the 27B carried them into its first start. The knobs sized for ONE model — the pool (and auto),
    the PLE placement, the drafter's depth, and since 2026-10-05 the PREFILL BUDGET and the concurrency and
    context too, every one of them something a model is measured at — go back to the new model's defaults;
    only the two that belong to no model in particular (the long-prefill threshold, the disk tier) stay."""
    s = store(tmp_path)
    s.set(enabled=False, kv_cache_memory=30_000_000_000, ple_replicate=False, num_speculative_tokens=2,
          max_num_batched_tokens=2048, long_prefill_token_threshold=4096, kv_offload_gib=64)
    moved = s.set(enabled=False, model_choice=DENSE)
    assert (moved.kv_cache_memory, moved.kv_cache_auto, moved.ple_replicate, moved.num_speculative_tokens) == (
        None, False, None, None)
    assert (moved.max_num_batched_tokens, moved.long_prefill_token_threshold, moved.kv_offload_gib) == (
        None, 4096, 64), "the budget is the new model's own now; the threshold and the disk tier are the pair's"
    assert store(tmp_path).load() == moved, "the reset is what is on disk"
    # Naming the model it already has changes nothing.
    assert s.set(enabled=False, model_choice=DENSE, kv_cache_memory=40_000_000_000).kv_cache_memory == 40_000_000_000
    assert s.set(enabled=False, model_choice=DENSE).kv_cache_memory == 40_000_000_000
    # A knob NAMED with the change is the operator's, and is kept.
    back = s.set(enabled=False, model_choice=FLASH, kv_cache_memory=32_000_000_000, num_speculative_tokens=3)
    assert (back.kv_cache_memory, back.num_speculative_tokens, back.ple_replicate) == (32_000_000_000, 3, None)


def test_a_changed_model_takes_its_own_measured_best_start_input(tmp_path):
    """THE 2026-10-05 ASK, in the switch: "set specific defaults per model, best we got so far, so switching
    model changes automatically values to best defaults hardcoded in the system".

    The SGLang pair's own figures are stored (32 seats, a 2,048 budget — generation 137's); a change to
    Flash-Next must leave its own instead (6 seats, 4,096, which `resolved_launch` reads from the same
    `TunedSetup` the page shows), with nothing of the other model's left behind. The operator naming a knob
    with the change still wins over both."""
    from vllm_lb.catalog import CLUSTER_MODE
    from vllm_lb.cluster_exec import resolved_launch

    catalog = ModelCatalog.load()
    sglang, flash = (catalog.get(SGLANG).profile(CLUSTER_MODE).tuned,
                     catalog.get(FLASH).profile(CLUSTER_MODE).tuned)
    assert (sglang.max_num_seqs, sglang.max_num_batched_tokens) == (32, 2048)
    assert (flash.max_num_seqs, flash.max_num_batched_tokens) == (6, 4096)

    s = store(tmp_path)
    at_sglang = s.set(enabled=False, model_choice=SGLANG)
    assert (at_sglang.max_num_seqs, at_sglang.max_model_len) == (32, 262144), "the SGLang pair's own"
    moved = s.set(enabled=False, model_choice=FLASH)
    assert (moved.max_num_seqs, moved.max_model_len) == (6, 262144), "Flash-Next's own, not the pair's 32"
    # The nullable knobs follow the model through the launch rather than being copied into the file.
    assert moved.max_num_batched_tokens is None and moved.num_speculative_tokens is None
    launched = resolved_launch(catalog.get(FLASH), {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"},
                               max_num_seqs=moved.max_num_seqs, max_model_len=moved.max_model_len,
                               max_num_batched_tokens=moved.max_num_batched_tokens,
                               num_speculative_tokens=moved.num_speculative_tokens)
    assert (launched["max_num_seqs"], launched["max_num_batched_tokens"],
            launched["num_speculative_tokens"]) == (6, 4096, 4)
    # Back the other way, and the operator's own seat count given WITH the change is kept.
    assert s.set(enabled=False, model_choice=SGLANG).max_num_seqs == 32
    assert s.set(enabled=False, model_choice=FLASH, max_num_seqs=12).max_num_seqs == 12
    # And a model with nothing benched takes its OWN row's stated seats, not the last model's.
    assert s.set(enabled=False, model_choice=DENSE).max_num_seqs == s.defaults_for(DENSE).max_num_seqs == 6


def test_auto_does_not_ride_across_a_model_change_either(tmp_path):
    s = store(tmp_path)
    s.set(enabled=False, kv_cache_memory=KV_AUTO)
    assert s.load().kv_cache_auto is True
    moved = s.set(enabled=True, model_choice=DENSE)           # no resolver given: auto would have refused this
    assert (moved.kv_cache_auto, moved.kv_cache_memory) == (False, None)


def test_a_restart_onto_another_model_resets_its_knobs_too(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True, kv_cache_memory=30_000_000_000, ple_replicate=False, max_num_batched_tokens=2048)
    with pytest.raises(ValueError, match="stop the cluster before changing its model choice"):
        s.set(enabled=True, model_choice=DENSE)
    again = s.set(enabled=True, restart=True, model_choice=DENSE)
    assert (again.generation, again.model_choice, again.kv_cache_memory, again.ple_replicate,
            again.max_num_batched_tokens) == (2, DENSE, None, None, None)


def test_a_ple_placement_for_a_model_without_a_ple_table_is_refused(tmp_path):
    """Audit #13: the dense 27B has no PLE table (`ple_replicate` null in its LaunchProfile), so there is
    nothing to place — true and false are both refused, on a Start and on a save, like a depth for a model
    with no drafter. null, the model's own, always passes."""
    s = store(tmp_path)
    for value in (True, False):
        with pytest.raises(ValueError, match="no PLE table"):
            s.set(enabled=True, model_choice=DENSE, ple_replicate=value)
        with pytest.raises(ValueError, match="no PLE table"):
            s.set(enabled=False, model_choice=DENSE, ple_replicate=value)
    assert not (tmp_path / "state" / "cluster.json").exists()
    assert s.set(enabled=True, model_choice=DENSE, ple_replicate=None).ple_replicate is None
    # …and Flash-Next, which has one, still takes both.
    assert store(tmp_path / "f").set(enabled=True, ple_replicate=False).ple_replicate is False


def test_a_stored_placement_the_model_cannot_take_refuses_the_start_and_never_a_stop(tmp_path):
    write_file(tmp_path, mode="cluster", generation=4, max_num_seqs=8, max_model_len=262144, model_choice=DENSE,
               ple_replicate=False, start_id="ef")
    assert store(tmp_path).set(enabled=False).mode == CLASSIC
    with pytest.raises(ValueError, match="no PLE table"):
        store(tmp_path).set(enabled=True)
    assert store(tmp_path).set(enabled=True, ple_replicate=None).mode == CLUSTER_MODE


def test_a_pair_already_running_with_such_a_value_is_left_alone_until_it_is_restarted(tmp_path):
    """A cluster.json from before the refusal: the 27B RUNNING with a placement and an `auto` pool stored. A
    second Start is idempotent — it launches nothing, so it must return that pair, not a 409 — a Stop passes,
    and only what would make a NEW generation is refused, until the value is sent back to null."""
    write_file(tmp_path, mode="cluster", generation=4, max_num_seqs=8, max_model_len=262144, model_choice=DENSE,
               ple_replicate=False, kv_cache_memory=48_000_000_000, kv_cache_auto=True, start_id="ef")
    s = store(tmp_path, catalog=uncalibrated())
    running = s.load()
    assert s.set(enabled=True) == running, "an idempotent Start was refused, or rewrote a running pair"
    with pytest.raises(ValueError, match="no PLE table"):
        s.set(enabled=True, restart=True)
    with pytest.raises(ValueError, match="is not calibrated: choose a size"):
        s.set(enabled=True, restart=True, ple_replicate=None)
    assert s.load() == running, "a refused restart was written"
    again = s.set(enabled=True, restart=True, ple_replicate=None, kv_cache_memory=None)
    assert (again.generation, again.ple_replicate, again.kv_cache_auto, again.kv_cache_memory) == (5, None, False, None)


def test_a_save_is_checked_against_the_model_the_switch_holds(tmp_path):
    """A save that names no model is for the stored one: a placement, or `auto`, that model cannot take is
    refused there too — not stored to be refused by the next Start."""
    s = store(tmp_path, catalog=uncalibrated())
    s.set(enabled=False, model_choice=DENSE)
    before = s.load()
    with pytest.raises(ValueError, match="no PLE table"):
        s.set(enabled=False, ple_replicate=True)
    with pytest.raises(ValueError, match='kv_cache_memory "auto": Qwen3.8 27B · dense NVFP4 is not calibrated'):
        s.set(enabled=False, kv_cache_memory=KV_AUTO)
    assert s.load() == before
    # The calibrated model still takes both.
    flash = store(tmp_path / "f").set(enabled=False, ple_replicate=False, kv_cache_memory=KV_AUTO)
    assert (flash.ple_replicate, flash.kv_cache_auto) == (False, True)


def test_a_stop_is_never_blocked_by_a_knob_it_did_not_touch_and_keeps_it(tmp_path):
    """A Stop checks only what it is given: a stored depth the model cannot take (a redeployed catalog, say)
    must not keep the pair running — and the operator's choice rides through the Stop, untouched."""
    write_file(tmp_path, mode="cluster", generation=4, max_num_seqs=8, max_model_len=262144, model_choice=DENSE,
               max_num_batched_tokens=4096, num_speculative_tokens=2, start_id="ef")
    stopped = store(tmp_path).set(enabled=False)
    assert (stopped.mode, stopped.max_num_batched_tokens, stopped.num_speculative_tokens) == (CLASSIC, 4096, 2)


def test_the_knobs_are_frozen_under_a_running_pair_and_a_restart_applies_them(tmp_path):
    """Both are engine arguments: changing one reloads both ranks. So like the rest of the selection they are
    refused under a running pair — and `restart` is the operator saying "reload it at these", which bumps the
    generation without ever passing through classic (worklog/vllm-cluster-handover.md D1)."""
    s = store(tmp_path)
    on = s.set(enabled=True)
    for kw in ({"max_num_batched_tokens": 16384}, {"num_speculative_tokens": 2}):
        with pytest.raises(ValueError, match="stop the cluster before changing its model choice"):
            s.set(enabled=True, **kw)
    assert s.load() == on, "a refused change was written"
    again = s.set(enabled=True, restart=True, max_num_batched_tokens=16384, num_speculative_tokens=2)
    assert (again.mode, again.generation) == (CLUSTER_MODE, on.generation + 1)
    assert (again.max_num_batched_tokens, again.num_speculative_tokens) == (16384, 2)
    assert s.set(enabled=True, max_num_batched_tokens=16384) == again, "the same values again are no change"


# ── the KV pool at `auto`: sized by each Start, stored as the number ─────────

class Resolver:
    """A stand-in for the LB's `kv_auto_resolver`: hands out the sizes it is given, in order, and remembers
    every selection it was asked to size."""

    def __init__(self, *sizes: int) -> None:
        self.sizes = list(sizes)
        self.asked: list[Desired] = []

    def __call__(self, selection: Desired) -> int:
        self.asked.append(selection)
        return self.sizes.pop(0)


def test_a_start_at_auto_stores_the_size_it_resolved_and_the_choice_beside_it(tmp_path):
    """Everything downstream — the gate, tp-rank, `cluster.effective`, the performance log — reads ONE number
    per generation, so the switch stores the number; the choice that it was sized, not typed, rides beside it."""
    s, sized = store(tmp_path), Resolver(36_000_000_000)
    on = s.set(enabled=True, max_num_batched_tokens=4096, ple_replicate=False, kv_cache_memory=KV_AUTO,
               resolve_kv_auto=sized)
    assert (on.kv_cache_memory, on.kv_cache_auto, on.generation) == (36_000_000_000, True, 1)
    (asked,) = sized.asked
    assert (asked.generation, asked.max_num_batched_tokens, asked.ple_replicate, asked.kv_cache_auto) == \
        (1, 4096, False, True), "it sizes the selection this Start stores, not the one before it"
    assert store(tmp_path).load() == on, "a restarted LB reads both back"
    raw = json.loads((tmp_path / "state" / "cluster.json").read_text())
    assert (raw["kv_cache_memory"], raw["kv_cache_auto"]) == (36_000_000_000, True)


def test_auto_is_sized_afresh_by_every_start_while_it_is_the_choice(tmp_path):
    """"Always match the RAM left": a restart that leaves the pool out keeps AUTO, not the last size — and so does
    a Stop and the Start after it. A Start that changes nothing under a running pair sizes nothing."""
    s, sized = store(tmp_path), Resolver(36_000_000_000, 37_000_000_000, 34_000_000_000)
    s.set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=sized)
    assert s.set(enabled=True, resolve_kv_auto=sized).kv_cache_memory == 36_000_000_000, "an idempotent Start"
    assert len(sized.asked) == 1
    again = s.set(enabled=True, restart=True, max_num_seqs=8, resolve_kv_auto=sized)
    assert (again.kv_cache_memory, again.kv_cache_auto, again.generation) == (37_000_000_000, True, 2)
    assert sized.asked[-1].max_num_seqs == 8
    stopped = s.set(enabled=False, resolve_kv_auto=sized)
    assert (stopped.kv_cache_auto, len(sized.asked)) == (True, 2), "a Stop keeps the choice and sizes nothing"
    assert s.set(enabled=True, resolve_kv_auto=sized).kv_cache_memory == 34_000_000_000


def test_a_size_or_null_ends_auto(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver(36_000_000_000))
    typed = s.set(enabled=True, restart=True, kv_cache_memory=28_000_000_000)
    assert (typed.kv_cache_memory, typed.kv_cache_auto) == (28_000_000_000, False)
    s.set(enabled=True, restart=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver(36_000_000_000))
    default = s.set(enabled=True, restart=True, kv_cache_memory=None)
    assert (default.kv_cache_memory, default.kv_cache_auto) == (None, False)


def test_a_start_at_auto_that_cannot_be_sized_is_refused_and_nothing_is_written(tmp_path):
    """The resolver's refusal is the Start's refusal, in its words; and auto with no resolver to ask is a refusal
    too — a pool is never guessed."""
    def refuse(selection):
        raise ValueError("kv_cache_memory \"auto\" cannot be sized now: box-a: 14.4 GiB free now, short 3.6 GiB")

    with pytest.raises(ValueError, match="short 3.6 GiB"):
        store(tmp_path).set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=refuse)
    with pytest.raises(ValueError, match="nothing was given to ask them"):
        store(tmp_path).set(enabled=True, kv_cache_memory=KV_AUTO)
    assert not (tmp_path / "state" / "cluster.json").exists()
    # A resolver handing back a size tp-rank refuses is not a launch either.
    with pytest.raises(ValueError, match="kv_cache_memory must be a whole number of bytes"):
        store(tmp_path).set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver(97_000_000_000))
    assert not (tmp_path / "state" / "cluster.json").exists()


def test_auto_is_frozen_under_a_running_pair_like_every_other_setting(tmp_path):
    s = store(tmp_path)
    on = s.set(enabled=True, kv_cache_memory=35_000_000_000)
    with pytest.raises(ValueError, match="stop the cluster before changing"):
        s.set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver(36_000_000_000))
    assert s.load() == on
    auto = s.set(enabled=True, restart=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver(36_000_000_000))
    assert s.set(enabled=True, kv_cache_memory=KV_AUTO, resolve_kv_auto=Resolver()) == auto, "the same choice"


@pytest.mark.parametrize("value", ["Auto", "36GB", 7_999_999_999, True])
def test_a_pool_that_is_neither_a_size_auto_nor_null_is_refused(tmp_path, value):
    with pytest.raises(ValueError, match="kv_cache_memory must be a whole number of bytes"):
        store(tmp_path).set(enabled=True, kv_cache_memory=value, resolve_kv_auto=Resolver(36_000_000_000))


def test_a_file_from_before_auto_reads_as_a_size(tmp_path):
    write_file(tmp_path, mode="cluster", generation=99, max_num_seqs=16, max_model_len=262144, model_choice=FLASH,
               max_num_batched_tokens=4096, ple_replicate=False, kv_cache_memory=35_000_000_000, start_id="ab")
    d = store(tmp_path).load()
    assert (d.kv_cache_memory, d.kv_cache_auto) == (35_000_000_000, False)


def test_a_pin_under_one_full_context_request_is_the_switchs_refusal(tmp_path):
    """The engine needs ceil(ctx / block_tokens) + F blocks for one request and dies at init without them
    (audit 2026-10-02): 9.66 GB on the 27B at 262,144. Refused where it is given and where it would launch;
    the same pin at a context it holds is taken."""
    s = store(tmp_path)
    bf16 = dict(kv_cache_dtype="bfloat16")          # the audit's arithmetic; the 27B's own int4 holds ~4x (2026-10-26)
    with pytest.raises(ValueError, match=r"kv_cache_memory 8 GB < 9.7 GB: one 262,144-token request"):
        s.set(enabled=True, model_choice=DENSE, max_model_len=262144, kv_cache_memory=8_000_000_000, **bf16)
    assert s.load().mode == CLASSIC
    with pytest.raises(ValueError, match="one 262,144-token request"):
        s.set(enabled=False, model_choice=DENSE, max_model_len=262144, kv_cache_memory=8_000_000_000, **bf16)
    ok = s.set(enabled=True, model_choice=DENSE, max_model_len=65536, kv_cache_memory=8_000_000_000, **bf16)
    assert (ok.mode, ok.kv_cache_memory) == (CLUSTER_MODE, 8_000_000_000)
    with pytest.raises(ValueError, match="one 262,144-token request"):
        s.set(enabled=True, restart=True, max_model_len=262144)


# ── a pin past the profile's safe ceiling: refused where it is given, before any generation is made ──────────────────

FLASH_SGLANG = "qwen3.8-flash-next-sglang"
CEILING_WORDS = r"safe 0.85 ceiling \(maximum 19,918,363,626 B\)"


def test_a_pin_past_the_profiles_safe_ceiling_is_the_switchs_refusal_before_a_generation_is_made(tmp_path):
    """GENERATION 187 (2026-10-10 20:55:20Z): POST /lb/cluster restart with KV pool 24,000,000,000 on the SGLang
    Flash-Next row was ACCEPTED (202, a generation bumped, the serving pair torn down), and only the launch refused
    it — `SGLang KV pin 24,000,000,000 B needs mem_fraction_static 0.885, above this profile's safe 0.85 ceiling
    (maximum 19,918,363,626 B)` — 131 times in six minutes. The ceiling is the profile's; the switch asks it."""
    s = store(tmp_path)
    with pytest.raises(ValueError, match=CEILING_WORDS) as refused:
        s.set(enabled=True, model_choice=FLASH_SGLANG, kv_cache_memory=24_000_000_000)
    assert "Flash-Next" in str(refused.value) and "kv_cache_memory" in str(refused.value), str(refused.value)
    d = s.load()
    assert (d.mode, d.generation) == (CLASSIC, 0), "nothing was written, no generation was made"
    # Refused on a SAVE too — a value the next Start would be refused on is not stored for it.
    with pytest.raises(ValueError, match=CEILING_WORDS):
        s.set(enabled=False, model_choice=FLASH_SGLANG, kv_cache_memory=24_000_000_000)
    assert s.load().kv_cache_memory is None
    # The largest pin the profile takes is taken; so is its own.
    ok = s.set(enabled=True, model_choice=FLASH_SGLANG, kv_cache_memory=19_918_363_626)
    assert (ok.mode, ok.generation, ok.kv_cache_memory) == (CLUSTER_MODE, 1, 19_918_363_626)
    ok = s.set(enabled=True, restart=True, kv_cache_memory=18_000_000_000)
    assert (ok.generation, ok.kv_cache_memory) == (2, 18_000_000_000)
    # A restart carrying the pin past the ceiling is refused and the generation stands: no pair is torn down.
    with pytest.raises(ValueError, match=CEILING_WORDS):
        s.set(enabled=True, restart=True, kv_cache_memory=24_000_000_000)
    assert s.load().generation == 2


def test_a_stored_pin_past_the_ceiling_refuses_the_start_that_would_launch_it(tmp_path):
    """A file an older LB wrote (or a catalog whose ceiling moved under it): the Start that would launch the stored
    pin is refused with the same words, never launched to fail on the box — and the pin named is the stored one."""
    write_file(tmp_path, mode=CLASSIC, generation=4, max_num_seqs=16, max_model_len=262144,
               model_choice=FLASH_SGLANG, kv_cache_memory=24_000_000_000, start_id="ab")
    s = store(tmp_path)
    with pytest.raises(ValueError, match=CEILING_WORDS):
        s.set(enabled=True)
    assert s.load().generation == 4
    # Naming a pin the profile takes — or none, the model's own — starts.
    assert s.set(enabled=True, kv_cache_memory=None).generation == 5
