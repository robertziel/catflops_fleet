"""NO ENV NUMBER STANDS IN FOR A MODEL'S OR A FLEET'S OWN (operator, 2026-10-29: "why do we have LB_CONTEXT_LENGTH?
Isn't that set in configs? Same for MAX_NUM_SEQS?").

Five compose defaults used to stand in for figures the catalog and the fleet file own: LB_CONTEXT_LENGTH (the lease
window), LB_CLUSTER_MIN_MEM_GIB (a flat memory gate), LB_CLUSTER_MAX_NUM_SEQS and LB_CLUSTER_MAX_MODEL_LEN (the seats
and context a row with nothing stated was offered) and CLUSTER_MAX_NUM_BATCHED_TOKENS (the box's prefill budget).
Each was a fallback with a second definition. This file pins, for the example fleet and the shipped catalog, that the
figures the LB resolves WITHOUT them are exactly the ones those defaults produced on the live pair — so retiring
them changed no running number — and that each name now refuses the boot.
"""
from __future__ import annotations

import pytest

from vllm_lb.catalog import CLASSIC, CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster_defaults import desired_store
from vllm_lb.config import RETIRED_KEYS, Config

RETIRED = ("LB_CONTEXT_LENGTH", "LB_CLUSTER_MIN_MEM_GIB", "LB_CLUSTER_MAX_NUM_SEQS", "LB_CLUSTER_MAX_MODEL_LEN",
           "CLUSTER_MAX_NUM_BATCHED_TOKENS")

#: (seats, context) each cluster row was offered under the old defaults — LB_CLUSTER_MAX_MODEL_LEN=262144 and
#: LB_CLUSTER_MAX_NUM_SEQS unset (the default row's pool rule at that window: 6) for a row with no tuned setup, and
#: the row's own tuned setup otherwise — computed on the old code at e4f6ca9 with lb/fleet.example.yml.
OFFERED_BEFORE = {
    "qwen3.8-flash-next-hibrid48": (6, 262_144),
    "qwen3.8-flash-next-sglang": (16, 262_144),
    "qwen3.8-flash-next-tensorfold-nvfp4": (6, 262_144),
    "qwen3.8-flash-next-tensorfold-mlx-mtp": (6, 262_144),
    "qwen3.8-27b-nvidia-nvfp4": (6, 262_144),
    "qwen3.8-27b-dflash2": (6, 262_144),
    "qwen3.8-27b-sglang-dflash2": (32, 262_144),
    "glm-5.3-flash-nvfp4": (8, 262_144),
    "deepseek-v4-flash-nvfp4": (8, 131_072),
}


#: The prefill budget each row launched with under the old defaults — the operator's knob unset: the row's tuned
#: budget, else its engine's own (SGLang's chunked_prefill_size), else the box's CLUSTER_MAX_NUM_BATCHED_TOKENS=8192;
#: None for an engine that sizes its own chunk (TensorFold refuses the knob).
BUDGET_BEFORE = {
    "qwen3.8-flash-next-hibrid48": 4096,
    "qwen3.8-flash-next-sglang": 4096,
    "qwen3.8-flash-next-tensorfold-nvfp4": None,
    "qwen3.8-flash-next-tensorfold-mlx-mtp": None,
    "qwen3.8-27b-nvidia-nvfp4": 8192,
    "qwen3.8-27b-dflash2": 8192,
    "qwen3.8-27b-sglang-dflash2": 2048,
    "glm-5.3-flash-nvfp4": 4096,
    "deepseek-v4-flash-nvfp4": 8192,
}


def _cfg(tmp_path) -> Config:
    from tests.fleet_files import PAIR, write
    return Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})


def test_every_cluster_row_is_offered_the_seats_and_context_the_old_defaults_gave_it(tmp_path):
    catalog = ModelCatalog.load()
    store = desired_store(_cfg(tmp_path), catalog=catalog)
    offered = {s.id: (store.defaults_for(s.id).max_num_seqs, store.defaults_for(s.id).max_model_len)
               for s in catalog.for_mode(CLUSTER_MODE)}
    assert offered == OFFERED_BEFORE
    cold = store.load()
    assert (cold.model_choice, cold.max_num_seqs, cold.max_model_len) == ("qwen3.8-flash-next-hibrid48", 6, 262_144)


def test_every_cluster_row_launches_with_the_prefill_budget_the_old_default_gave_it(tmp_path):
    """The fleet file's `clusters.pair.max_num_batched_tokens: 8192` (lb/fleet.example.yml, the live file's one new
    line) reaches every launch through the executor's env exactly as CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 did."""
    from vllm_lb.cluster_exec import resolved_launch
    from vllm_lb.cluster_run import tp_rank_env
    cfg = _cfg(tmp_path)
    catalog = ModelCatalog.load()
    store = desired_store(cfg, catalog=catalog)
    env = tp_rank_env(cfg)
    assert env["CLUSTER_MAX_NUM_BATCHED_TOKENS"] == "8192"
    launched = {}
    for spec in catalog.for_mode(CLUSTER_MODE):
        base = store.defaults_for(spec.id)
        launched[spec.id] = resolved_launch(spec, env, max_num_seqs=base.max_num_seqs,
                                            max_model_len=base.max_model_len)["max_num_batched_tokens"]
    assert launched == BUDGET_BEFORE


def test_the_lease_window_is_what_the_old_default_declared(tmp_path):
    """262144 — LB_CONTEXT_LENGTH's compose default — is the classic default row's own context."""
    cfg = _cfg(tmp_path)
    assert cfg.lease_context_length == 262_144
    assert ModelCatalog.load().default(CLASSIC).limits.max_model_len == 262_144


@pytest.mark.parametrize("key", RETIRED)
def test_each_retired_name_refuses_the_boot_naming_its_replacement(tmp_path, key):
    from tests.fleet_files import PAIR, write
    assert key in RETIRED_KEYS, key
    for value in ("8192", ""):
        with pytest.raises(ValueError, match=key) as e:
            Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR), key: value})
        assert "retired" in str(e.value), str(e.value)
