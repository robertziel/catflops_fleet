"""THE PANEL'S KNOBS ON AN SGLANG PAIR: each one reaches SGLang with its own flag, or is refused where it is given and
locked on the page — never accepted and silently dropped.

THE FAILURE, OBSERVED 2026-10-03. At 07:12:30 the operator restarted the SGLang pair with prefill budget 2048
(POST /lb/cluster `max_num_batched_tokens: 2048`; the LB logged "prefill budget 2048" and `cluster.effective` said
2048), but the engine's own server_args said `chunked_prefill_size 8192 / max_prefill_tokens 16384`, and every
`Prefill batch` line was `#new-token: 8192`: the catalog's `engine_args.chunked_prefill_size` won over the operator's
budget in tp-rank's `sglang_cmd`. The same audit found the pool knobs dropped the same way: tp-rank's SGLang command
never reads CLUSTER_KV_CACHE_MEMORY (generation 135 ran with a stored 40 GB pin while SGLang took its 0.80 fraction),
the disk tier and the long-prefill threshold are vLLM flags, and the PLE placement is a vLLM-image patch.

    knob                           SGLang
    max_num_batched_tokens         --chunked-prefill-size; and --max-prefill-tokens with it past SGLang's 16384
    num_speculative_tokens         --speculative-num-draft-tokens (0: no drafter flags at all)
    max_num_seqs                   --max-running-requests (and the mamba pool sized by it)
    max_model_len                  --context-length
    kv_cache_memory / auto         refused: SGLang sizes its pools by engine_args.mem_fraction_static
    kv_offload_gib                 refused: vLLM's LMCache connector
    long_prefill_token_threshold   refused: no per-request cap in SGLang
    ple_replicate                  refused: a vLLM-image patch (and the 27B has no PLE table)

SGLang's own code, image myllmbox/sglang-27b-tp2:708f51e-fleet.2: `PrefillAdder` (srt/managers/schedule_policy.py)
starts a prefill batch with `rem_input_tokens = max_prefill_tokens` and `rem_chunk_tokens = chunked_prefill_size`,
takes every admitted request's tokens off BOTH (`_update_prefill_budget`), and stops admitting when either reaches 0
(`budget_state`). Under 16384 (`max_prefill_tokens`' default, srt/arg_groups/fields/schedule.py) the chunk binds and
the default needs no flag; past it the default would stop a batch short of the operator's budget, so tp-rank raises it
to the budget. `max_prefill_buffer_tokens` (srt/arg_groups/overrides.py) sizes the prefill buffers off the chunk alone.
"""
from __future__ import annotations

import json

import pytest

from tests.test_cluster_api import call, served  # noqa: F401 — `served` is a fixture
from tests.test_cluster_defaults import _auto
from tests.test_cluster_loop import World, writes  # noqa: F401 — `writes` is a fixture
from tests.test_cluster_store import store, write_file
from tests.test_start_rescue import HEALTHY_HEAD, sglang_launched, whole
from tests.test_tp_rank import LIVE_BOX_ENV, last_run, box_a, ssh  # noqa: F401 — fixtures
from vllm_lb.catalog import document, CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster import CLUSTER_MODE as ON
from vllm_lb.cluster_exec import resolved_launch
from vllm_lb.memory_estimate import estimate_memory

SGLANG = "qwen3.8-27b-sglang-dflash2"
FLASH = "qwen3.8-flash-next-hibrid48"
#: Flash-Next on SGLang: the row generation 187 was restarted on with a 24 GB pin (2026-10-10).
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
#: A model with NO tuned start input, for the "the box's budget, as before" half of the precedence.
DENSE = "qwen3.8-27b-nvidia-nvfp4"
REFUSED = {"kv_offload_gib", "long_prefill_token_threshold", "ple_replicate"}


def shipped() -> ModelCatalog:
    return ModelCatalog.from_dict(document())


def sglang_up(box_a, ssh, box_env=None, **knobs) -> tuple[list[str], dict]:  # noqa: F811 — fixtures
    spec = shipped().get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, **(box_env or {}), "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=135, max_num_seqs=32, max_model_len=262144, spec=spec, **knobs)
    assert ok is True, msg
    run = last_run(box_a)
    return run, dict(zip(run, run[1:]))


# ── the prefill budget reaches SGLang ─────────────────────────────────────────

def test_the_operators_prefill_budget_is_sglangs_chunk(box_a, ssh):  # noqa: F811
    run, flags = sglang_up(box_a, ssh, max_num_batched_tokens=2048)
    assert flags["--chunked-prefill-size"] == "2048", "generation 135 asked for 2048 and ran 8192"
    assert "--max-prefill-tokens" not in run, "under 16384 SGLang's own default bounds nothing: no flag"


def test_an_operators_kv_pin_becomes_sglangs_exact_token_pool():
    """The cluster KV control is an engine input on SGLang too, never a saved-but-ignored value.

    `--max-total-tokens` is the pool SGLang's scheduler actually exposes.  Its byte equivalent is
    model-specific, including the target and speculative pools, so this conversion belongs beside
    the single profile-to-launch translation rather than in the panel.
    """
    from vllm_lb.cluster_exec import launch_env

    spec = shipped().get(SGLANG)
    requested = 78_000_000_000
    env = launch_env(spec, kv_cache_memory=requested)
    args = json.loads(env["CLUSTER_ENGINE_ARGS"])
    assert args["max_total_tokens"] == requested // spec.memory.kv_bytes_per_token
    assert args["mem_fraction_static"] > spec.profile(CLUSTER_MODE).engine_args["mem_fraction_static"]


def test_tp_rank_forwards_sglangs_kv_token_pool_to_the_engine(box_a, ssh):  # noqa: F811
    requested = 78_000_000_000
    run, flags = sglang_up(box_a, ssh, kv_cache_memory=requested)
    spec = shipped().get(SGLANG)
    assert flags["--max-total-tokens"] == str(requested // spec.memory.kv_bytes_per_token)
    assert float(flags["--mem-fraction-static"]) > 0.80


def test_with_no_budget_given_the_models_measured_chunk_is_the_default_whatever_the_box_says(box_a, ssh):  # noqa: F811
    """The catalog's own figure beats the box's, as it always did — and since 2026-10-05 that figure is the
    MEASURED one (`catalog.TunedSetup`: 2,048, generation 137's, the only place this profile states a budget)."""
    run, flags = sglang_up(box_a, ssh, box_env={"CLUSTER_MAX_NUM_BATCHED_TOKENS": "4096"})
    tuned = shipped().get(SGLANG).profile("cluster").tuned.max_num_batched_tokens
    assert flags["--chunked-prefill-size"] == str(tuned) == "2048"
    assert "--max-prefill-tokens" not in run


def test_the_gdn_prefill_kernel_the_profile_names_reaches_sglang(box_a, ssh):  # noqa: F811
    run, flags = sglang_up(box_a, ssh)
    assert flags["--linear-attn-prefill-backend"] == "flashinfer", \
        "SGLang's auto-selector leaves SM 12.1 on Triton; only the explicit flag moves it"
    assert "--linear-attn-backend" not in run, "decode and verify stay on the engine's own choice"


def test_a_budget_past_sglangs_prefill_ceiling_raises_the_ceiling_with_it(box_a, ssh):  # noqa: F811
    run, flags = sglang_up(box_a, ssh, max_num_batched_tokens=32768)
    assert flags["--chunked-prefill-size"] == "32768"
    assert flags["--max-prefill-tokens"] == "32768", "16384 would stop a batch at half the operator's budget"


def test_what_the_lb_says_was_launched_is_what_sglang_was_given():
    """`resolved_launch` is the launch record, `cluster.effective`, the boot log's config and the start gate's
    budget: the operator's budget, else the MODEL's measured one (`catalog.TunedSetup`) — and the box's only
    for a model that has none, which is every model that was never A/B'd."""
    spec, untuned = shipped().get(SGLANG), shipped().get(DENSE)
    box = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "4096"}
    knobs = dict(max_num_seqs=32, max_model_len=262144)
    assert resolved_launch(spec, box, **knobs)["max_num_batched_tokens"] == 2048, "generation 137's"
    assert resolved_launch(spec, box, **knobs, max_num_batched_tokens=16384)["max_num_batched_tokens"] == 16384
    assert untuned.profile(CLUSTER_MODE).tuned.max_num_batched_tokens is None
    assert resolved_launch(untuned, box, **knobs)["max_num_batched_tokens"] == 4096, "no budget measured: the box's"
    # SGLang takes the KV pin too; the other engine-specific knobs remain safely refused.
    got = resolved_launch(spec, box, **knobs, kv_cache_memory=40_000_000_000, kv_offload_gib=64,
                          long_prefill_token_threshold=1024)
    profile = spec.profile(CLUSTER_MODE)
    assert (got["kv_cache_memory"], got["kv_offload_gib"], got["long_prefill_token_threshold"],
            got["ple_replicate"]) == (40_000_000_000, 0, 0, None)


def test_the_engines_own_words_for_its_prefill_budget_are_read():
    from vllm_lb.cluster_startup import stated_prefill_budget
    assert stated_prefill_budget(whole(HEALTHY_HEAD)) == 8192, "`max_total_num_tokens=…, chunked_prefill_size=8192`"
    head = whole(HEALTHY_HEAD)
    assert stated_prefill_budget(head[:head.index("max_total_num_tokens=")]) == 8192, "server_args, as given"
    assert stated_prefill_budget(whole("qwen38-dflash2-start-2026-10-02-rank0.log")) == 4096, \
        "vLLM: `Chunked prefill is enabled with max_num_batched_tokens=4096.`"
    assert stated_prefill_budget("") is None


# ── what SGLang does not take is refused, never dropped ───────────────────────

def test_the_catalog_says_which_knobs_each_engine_refuses_and_why():
    view = shipped().get(SGLANG).profile(CLUSTER_MODE).view()
    assert set(view["refuses"]) == REFUSED
    assert all("SGLang" in why for why in view["refuses"].values()), view["refuses"]
    assert shipped().get(FLASH).profile(CLUSTER_MODE).view()["refuses"] == {}


@pytest.mark.parametrize("knob,value", [("kv_offload_gib", 64), ("long_prefill_token_threshold", 1024)])
def test_a_knob_sglang_does_not_take_is_refused_where_it_is_given(tmp_path, knob, value):
    s = store(tmp_path)
    with pytest.raises(ValueError, match=f"SGLang.*{knob}|{knob}.*SGLang"):
        s.set(enabled=True, model_choice=SGLANG, **{knob: value})
    with pytest.raises(ValueError, match="SGLang"):
        s.set(enabled=False, model_choice=SGLANG, **{knob: value})
    assert not (tmp_path / "state" / "cluster.json").exists(), "a refused selection was written"
    off = None if knob == "kv_cache_memory" else 0
    assert s.set(enabled=True, model_choice=SGLANG, **{knob: off}).mode == ON


def test_a_value_kept_from_a_vllm_model_refuses_the_sglang_start_until_it_is_cleared(tmp_path):
    """The disk tier and the long-prefill threshold are the pair's, not a model's: a model change keeps them. Onto
    SGLang they would be dropped — so the Start is refused, saying which, until they are sent back to off."""
    write_file(tmp_path, mode="classic", generation=7, max_num_seqs=8, max_model_len=262144, model_choice=FLASH,
               kv_offload_gib=64, long_prefill_token_threshold=4096)
    with pytest.raises(ValueError, match="SGLang"):
        store(tmp_path).set(enabled=True, model_choice=SGLANG, max_num_seqs=32)
    started = store(tmp_path).set(enabled=True, model_choice=SGLANG, max_num_seqs=32, kv_offload_gib=None,
                                  long_prefill_token_threshold=None)
    assert (started.mode, started.model_choice, started.kv_offload_gib) == (ON, SGLANG, None)


def test_a_stored_sglang_pin_is_honoured_on_restart(tmp_path):
    """A pin from an older generation is no longer decorative: a restart keeps and applies it."""
    write_file(tmp_path, mode="cluster", generation=135, max_num_seqs=32, max_model_len=262144,
               model_choice=SGLANG, max_num_batched_tokens=2048, kv_cache_memory=40_000_000_000,
               start_id="60df20af71dd49b1a0cccc073dda2d4d")
    s = store(tmp_path)
    running = s.load()
    assert s.set(enabled=True) == running
    again = s.set(enabled=True, restart=True)
    assert (again.generation, again.max_num_batched_tokens, again.kv_cache_memory) == (136, 2048, 40_000_000_000)


def test_the_start_gate_charges_the_sglang_pool_the_operator_selected():
    spec = shipped().get(SGLANG)
    common = dict(box="box-a", role="head", max_num_seqs=32, max_model_len=262144, tp=2, available_gib=115.0,
                  floor_gib=4.0, margin_gib=2.0, max_num_batched_tokens=2048)
    own = estimate_memory(spec, **common)
    pinned = estimate_memory(spec, **common, kv_cache_memory=40_000_000_000)
    assert pinned.kv_gib == 40_000_000_000 / (1 << 30)
    assert pinned.kv_gib < own.kv_gib
    assert pinned.total_gib < own.total_gib


def test_auto_uses_the_same_memory_estimator_for_sglang():
    a = _auto(SGLANG)
    assert "SGLang sizes its KV pools" not in (a.refusal or ""), a.refusal


# ── the loop: the gate, and what the panel says the pair runs ─────────────────

@pytest.fixture
def world(broker_db, writes, tmp_path):  # noqa: F811
    return World(tmp_path, writes)


def test_the_start_gate_weighs_the_budget_sglang_is_given(world):
    world.executor.env = {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "4096"}
    world.store.set(enabled=False, model_choice=SGLANG, max_num_seqs=32)
    v = world.tick()
    assert v["boxes"]["box-a"]["memory"]["max_num_batched_tokens"] == 2048, \
        "the model's MEASURED chunk (TunedSetup), not the box's"
    assert v["defaults"]["max_num_batched_tokens"] == 2048
    world.store.set(enabled=False, max_num_batched_tokens=16384)
    assert world.tick()["boxes"]["box-a"]["memory"]["max_num_batched_tokens"] == 16384, \
        "and the operator's own value still wins over both"


def test_the_panel_states_the_budget_the_engine_says_it_runs(world, caplog):
    """`cluster.effective` is what the pair RUNS: once the head has said its budget, its word — so a launch the
    engine did not honour reads "running 8,192" beside the 2,048 the operator stored, and the LB says so."""
    world.store.set(enabled=True, model_choice=SGLANG, max_num_seqs=32, max_num_batched_tokens=2048)
    world.tick()
    world.claim()
    world.settle(until="starting")
    assert world.executor.knobs["box-a"]["max_num_batched_tokens"] == 2048
    assert world.tick()["effective"]["max_num_batched_tokens"] == 2048, "before the head has said anything"
    world.executor.log_text["box-a"] = whole(HEALTHY_HEAD)          # an engine that ran 8192 regardless
    with caplog.at_level("WARNING", logger="vllm_lb.cluster_loop"):
        for _ in range(9):
            v = world.tick(seconds=15)
    assert v["effective"]["max_num_batched_tokens"] == 8192, v["effective"]
    assert any("8192" in r.getMessage() and "2048" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_a_launch_record_says_which_sglang_pin_it_was_started_with(world):
    """`cluster.effective` reports the requested pin, which this version sends to SGLang as a real pool cap."""
    world.store.set(enabled=True, model_choice=SGLANG, max_num_seqs=32)
    world.tick()
    world.claim()
    world.settle(until="starting")
    world.loop._launched = {**world.loop._launched, "kv_cache_memory": 40_000_000_000, "kv_offload_gib": 64}
    eff = world.tick()["effective"]
    assert eff["kv_cache_memory"] == 40_000_000_000
    assert eff["kv_offload_gib"] == 0 and eff["max_num_seqs"] == 32


def test_a_pin_the_launch_cannot_build_is_one_refusal_for_the_executor_and_the_loops_own_check(box_a, ssh):  # noqa: F811
    """GENERATION 187: the executor answered `(False, "up: ValueError: SGLang KV pin … above this profile's safe 0.85
    ceiling …")` — it never raises for a box, and the loop only knew a refusal that was RAISED. The build is one function
    now: `up` refuses before dialing with its words, and `launch_problem` gives the loop the same words for the stored
    selection before any Up is planned."""
    from vllm_lb.cluster_exec import launch_problem

    spec = shipped().get(FLASH_SGLANG)
    ex, host = ssh.executor(env=LIVE_BOX_ENV)
    knobs = dict(rank=0, gen=187, max_num_seqs=16, max_model_len=262144)
    ok, msg = ex.up(host, spec=spec, kv_cache_memory=24_000_000_000, **knobs)
    assert ok is False and "maximum 19,918,363,626 B" in msg, msg
    assert ssh.calls() == [], "refused before dialing: nothing reached the box"
    problem = launch_problem(spec, kv_cache_memory=24_000_000_000, **knobs)
    assert problem and problem in msg, (problem, msg)
    assert launch_problem(spec, kv_cache_memory=18_000_000_000, **knobs) == ""
    assert launch_problem(spec, **knobs) == "", "the model's own pin builds"


def test_the_api_accepts_sglangs_real_kv_control_and_refuses_the_others(served):  # noqa: F811
    cfg, proxy = served()
    status, reply = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": SGLANG, "max_num_seqs": 32,
                                                        "max_num_batched_tokens": 2048,
                                                        "kv_cache_memory": 40_000_000_000})
    assert status == 202 and reply["desired"]["kv_cache_memory"] == 40_000_000_000, reply
