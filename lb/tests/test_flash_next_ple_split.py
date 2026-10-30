"""The vLLM Flash-Next row launches its PLE table SPLIT across the two ranks by default (2026-10-11).

THE FAILURE IT ENDS. From the day tp-rank's memory guard landed (bfbf893, 2026-10-05) to 2026-10-11, no Start of
`qwen3.8-flash-next-hibrid48` served. Every one ran the profile's placement — the whole 26.82 GiB per-layer-embedding
table on EACH rank (`MBX_PLE_REPLICATE=1`, 'Model loading took 63.97 GiB' a rank) — and the guard killed every one in
the phases between the weights and the KV pool (encoder-cache profiling, torch.compile, the warm-up forward):
generations 183/184 (the uncensored twin in its publisher's compressed-tensors layout), 190/191 (the twin rewritten to
ModelOpt) and 192 (the ORIGINAL checkpoint). So it was never the twin: the operator's "loads censored, fails
uncensored" was 192 still STARTING when seen, and it died nine minutes later exactly like the twin. Each kill sits
inside the guard's swap rule — MemAvailable under twice its floor while the box has swapped more than its allowance
since the load began — and each came BEFORE the pool was allocated, which is why 191 and 192 at a 20 GB pin died as
190 did at 28: no pin could have saved them.

THE PLACEMENT THAT STARTS. Split, each rank holds half the table — 13.41 GiB less a box, and the profile/compile/
warm-up dip no longer reaches the guard's line. That is what the row ran when it last served (generations 99-138,
2026-10-01..05, before the guard existed) and what generation 193 — the twin, every other setting the catalog's
default — loaded with: never under 24 GiB available, served at 2026-10-11T20:08:35Z, and answered. The box swapped
3.3-3.5 GiB during that load too (the 100 GiB checkpoint streams through the page cache either way); with 24+ GiB
still available, that is ordinary reclaim and the guard rightly let it be.

The guard is the protection and stays exactly as it is; the placement is the profile's to choose. Every figure is
read from lb/tests/fixtures/flash-next-vllm-ple-2026-10-11.json, which cites where each came from, and the guard's
thresholds are read off tp-rank itself, so a changed guard re-judges this evidence here rather than in production.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster_exec import launch_env
from vllm_lb.memory_estimate import estimate_memory

FLASH = "qwen3.8-flash-next-hibrid48"
EVIDENCE = json.loads((Path(__file__).resolve().parent / "fixtures" / "flash-next-vllm-ple-2026-10-11.json").read_text())
TP_RANK = Path(__file__).resolve().parents[1] / "cluster" / "tp-rank"
MIB_PER_GIB = 1024


def tp_rank_default(var: str) -> int:
    """A tp-rank setting's own default, as the script declares it (`: "${VAR:=N}"`)."""
    found = re.search(rf': "\${{{var}:=(\d+)}}"', TP_RANK.read_text())
    assert found, f"tp-rank no longer declares a default for {var}"
    return int(found.group(1))


#: The guard's floor WHEN THESE STARTS RAN (2026-10-11): tp-rank's then-default 8 GiB. Since 2026-10-26 it is the
#: fleet's MemTotal x (1 - ram_usable_fraction); the evidence is judged by the line it was recorded under.
GUARD_FLOOR_GIB_2026_10_11 = 8


def guard_line_mib() -> int:
    """Where the guard's swap rule started counting: MemAvailable under TWICE its floor (`arm_mem_guard`)."""
    return 2 * GUARD_FLOOR_GIB_2026_10_11 * MIB_PER_GIB


def test_the_vllm_flash_next_row_splits_its_ple_table_by_default():
    """The profile's placement is split, so a Start that leaves PLE at its default — every Start the panel makes
    unless the operator picks one — sends MBX_PLE_REPLICATE=0, and the gate charges half the table a rank."""
    spec = ModelCatalog.load().get(FLASH)
    profile = spec.profile(CLUSTER_MODE)
    assert profile.engine == "vllm" and profile.ple_replicate is False
    assert launch_env(spec)["CLUSTER_PLE_REPLICATE"] == "0"
    assert launch_env(spec, ple_replicate=True)["CLUSTER_PLE_REPLICATE"] == "1", "replicated stays the operator's to ask for"
    m = spec.memory
    default = estimate_memory(spec, box="box-a", role="head", max_num_seqs=6, max_model_len=262144, tp=2,
                              available_gib=115.5, floor_gib=12.0, margin_gib=8.0)
    assert default.weights_gib == pytest.approx(m.weights_gib + m.ple_gib / 2, abs=0.05)


def test_every_replicated_start_since_the_guard_died_inside_its_swap_rule_before_the_kv_pool():
    replicated = EVIDENCE["replicated"]
    assert [r["generation"] for r in replicated] == [183, 184, 190, 191, 192]
    allowance = tp_rank_default("CLUSTER_GUARD_SWAP_GIB") * MIB_PER_GIB
    before_pool = EVIDENCE["phases_before_the_kv_pool"]["order"]
    assert before_pool[-1] == "allocating the KV cache"
    for r in replicated:
        for kill in (r, r.get("other_box") or r):
            assert kill["available_mib"] < guard_line_mib() and kill["swapped_mib"] > allowance, \
                f"generation {r['generation']} on {kill['box']} is not the guard's swap rule"
        assert r["failed_phase"] in before_pool[:-1], f"generation {r['generation']} died after the pool existed"
    # It was never the checkpoint: the original died as the twin did, in either of its layouts.
    assert {r["checkpoint"].split(",")[0] for r in replicated} == {"uncensored twin", "the ORIGINAL hibrid48"}
    # And never the pin: the kill comes before the pool, so 20 GB died as 28 did.
    assert {r["kv_cache_memory"] for r in replicated} == {20_000_000_000, 28_000_000_000}


def test_the_split_start_never_reached_the_guards_line_and_served():
    split = EVIDENCE["split"]
    assert split["config"]["ple_replicate"] is False
    for attempt in split["attempts"]:
        for box, low in attempt["min_available_mib"].items():
            assert low > guard_line_mib(), f"{box} went under the guard's line at split"
    assert split["attempts"][-1]["outcome"].startswith("running at ")
    assert "403" in split["serving"]["completion"]


def test_the_split_saved_exactly_half_the_table_a_rank():
    """'Model loading took' replicated (the original, 192) minus split (the twin, 193) is half the catalog's table
    less what the twin's BF16 linear-attention projections add a rank — to 0.1 GiB: the placement, not the
    checkpoint, is what moved."""
    spec = ModelCatalog.load().get(FLASH)
    loaded = lambda lines: float(re.search(r"Model loading took ([\d.]+) GiB", lines[0]).group(1))  # noqa: E731
    replicated = loaded(EVIDENCE["replicated"][-1]["vllm"])
    split = loaded(EVIDENCE["split"]["vllm"])
    twin_a_rank = EVIDENCE["twin_over_original_gib"]["whole"] / 2
    assert replicated - split == pytest.approx(spec.memory.ple_gib / 2 - twin_a_rank, abs=0.1)
