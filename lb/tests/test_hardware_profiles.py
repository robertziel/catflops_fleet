"""THE HARDWARE TYPES (vllm_lb/hardware/types/*.yml, loaded by vllm_lb.hardware): one file per type, every number
with its source.

A type is what the gateway may CLAIM about a GPU it did not measure: its CUDA cores (the CUDA CORES tile's
denominator), its memory bandwidth (MEMORY BANDWIDTH's ceiling), its dense TFLOP/s per precision (COMPUTE's), whether
a DRAM-traffic counter can be read on it, its arch, its memory and its behaviour family. A figure nobody can trace is
the invention this gateway exists to refuse, so a type cannot hold a number without its source — and where no source
says, the figure is absent and the type says why (`Fact(None, why)`). The facts are DATA (AGENTS.md: a deployment is
configuration, never code); the classes keep only the behaviour (unified vs discrete memory).

The expectations below are the SOURCES' own figures, typed from them, not from the files:
  * NVIDIA RTX Blackwell GPU Architecture whitepaper v1.1, Appendix A, Table 3 (RTX 5090 vs 4090 vs 3090): dense
    peaks at boost clock (the `/Y` figure is with sparsity) — 5090: 21760 cores, 1792 GB/sec, BF16 209.5 / FP8 419
    / FP4 1676 Tensor TFLOPS with FP32 accumulate; 3090: 10496 cores, 936 GB/sec, BF16 71.2, FP8 and FP4 N/A.
  * NVIDIA DGX Spark User Guide, Hardware Overview: 6,144 CUDA cores; 128 GB LPDDR5x unified, 273 GB/s.
  * developer.nvidia.com/cuda-gpus: compute capability GB10 12.1, RTX 5090 12.0, RTX 3090 8.6.
  * AMD ROCm GPU hardware specifications, Ryzen APUs: Radeon 8060S, RDNA3.5, gfx1151, VRAM "Dynamic + carveout".
  * this project's pair, 2026-09-30, ONE idle GB10 (lb/docker-compose.yml, worklog/bench/prefill_probe.py): bf16
    96.5, fp8 188.3, nvfp4 323.2 TFLOP/s per GPU; and 2026-09-29 (saturation.py): nvidia-smi's memory duty reads 0
    at 96 % SM fill on both GB10 boxes — a GB10 has no DRAM counter to read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_lb import hardware
from vllm_lb.hardware import (DISCRETE, TYPES_DIR, UNIFIED, DiscreteProfile, Fact, HardwareProfile, UnifiedProfile,
                              load_types, profile, types)


def _facts(p: HardwareProfile):
    yield from (p.arch, p.memory_gb, p.cuda_cores, p.peak_gbps, p.bandwidth_counter, p.host_platform, p.visible_gib,
                p.free_at_start_gib)
    yield from p.peak_tflops.values()


# ── the facts, as the sources state them ────────────────────────────────────────────────────────────

def test_the_gb10_is_the_dgx_boxes_unified_memory_gpu():
    gb10 = profile("gb10")
    assert isinstance(gb10, UnifiedProfile) and gb10.memory is UNIFIED and gb10.key == "gb10"
    assert (gb10.arch.value, gb10.cuda_cores.value, gb10.memory_gb.value, gb10.peak_gbps.value) == \
        ("sm_121", 6144, 128, 273)
    assert {p: f.value for p, f in gb10.peak_tflops.items()} == {"bf16": 96.5, "fp8": 188.3, "nvfp4": 323.2}
    assert gb10.bandwidth_counter.value is False
    assert gb10.controls == ("ssh", "local"), "tp-rank drives DGX Spark pairs; a one-host node runs on one (Parakeet)"


def test_the_323_nvfp4_peak_is_one_gpus_measured_on_one_box_not_the_pairs():
    """The operator's question, 2026-10-11: per box, or for the pair? lb/docker-compose.yml: 'What ONE GPU can
    compute … MEASURED on an idle GB10 2026-09-30 (the pair must be stopped …)'; worklog/bench/prefill_probe.py:
    '323.2 TFLOP/s per GPU'. One GB10 a box, so it is each box's own ceiling."""
    source = profile("gb10").peak_tflops["nvfp4"].source
    assert "ONE idle GB10" in source and "per GPU" in source


def test_the_rtx_5090_is_a_discrete_card_at_its_whitepapers_dense_peaks():
    card = profile("rtx5090")
    assert isinstance(card, DiscreteProfile) and card.memory is DISCRETE
    assert (card.arch.value, card.cuda_cores.value, card.memory_gb.value, card.peak_gbps.value) == \
        ("sm_120", 21760, 32, 1792)
    assert {p: f.value for p, f in card.peak_tflops.items()} == {"bf16": 209.5, "fp8": 419, "nvfp4": 1676}
    assert card.controls == ("local",), "the standalone node: the LB requests, its host applies (up.sh node)"
    assert profile("rtx3090").controls == ()


def test_the_rtx_3090_has_no_fp8_or_fp4_peak_and_says_so():
    card = profile("rtx3090")
    assert isinstance(card, DiscreteProfile)
    assert (card.arch.value, card.cuda_cores.value, card.memory_gb.value, card.peak_gbps.value) == \
        ("sm_86", 10496, 24, 936)
    assert card.peak_tflops["bf16"].value == 71.2
    for precision in ("fp8", "nvfp4"):
        absent = card.peak_tflops[precision]
        assert absent.value is None and "N/A" in absent.source, precision


def test_the_amd_igpu_is_unified_memory_and_claims_no_cuda_figure():
    igpu = profile("radeon8060s")
    assert isinstance(igpu, UnifiedProfile) and igpu.arch.value == "gfx1151"
    assert igpu.cuda_cores.value is None and "compute units" in igpu.cuda_cores.source
    assert igpu.memory_gb.value is None and "Dynamic + carveout" in igpu.memory_gb.source


def test_a_bandwidth_counter_nobody_established_is_absent_with_its_reason():
    for key in ("rtx5090", "rtx3090", "radeon8060s"):
        c = profile(key).bandwidth_counter
        assert c.value is None and c.source.startswith("not established"), key


def test_the_5090_states_the_memory_vllm_finds_on_it_at_start_measured_with_its_cuda_context():
    """vLLM's START CHECK (v1/worker/utils.py:761 @2eb491e, `request_memory`, run at gpu_worker.py:521 after the CUDA
    context and NCCL exist): gpu_memory_utilization × the card's VISIBLE memory must be within its FREE memory then, or
    it refuses before loading. The one-card 5090 node's own boot, 2026-10-11 23:21Z: "Free memory on device cuda:0
    (30.85/31.36 GiB) on startup is less than desired GPU memory utilization (0.99, 31.04 GiB)". Both measured, both
    sourced; of the rest only the radeon8060s has since (its own test) — the others are absent and say why."""
    card = profile("rtx5090")
    assert (card.visible_gib.value, card.free_at_start_gib.value) == (31.36, 30.85)
    for fact in (card.visible_gib, card.free_at_start_gib):
        assert "utils.py:761" in fact.source and "2026-10-11" in fact.source, fact.source
    assert "CUDA context" in card.free_at_start_gib.source, "the free figure is an idle card's WITH vLLM's own context"
    for key in ("gb10", "rtx3090"):              # (the radeon8060s is measured: its own test)
        t = profile(key)
        assert (t.visible_gib.value, t.free_at_start_gib.value) == (None, None), key
        assert t.free_at_start_gib.source and t.visible_gib.source, key
    v = card.view()
    assert v["free_at_start_gib"] == {"value": 30.85, "source": card.free_at_start_gib.source}
    assert v["visible_gib"]["value"] == 31.36


def test_a_peak_is_dense_never_the_sparsity_figure():
    """NVIDIA's `X/Y` is dense/sparse; the gateway's FLOPs are dense, so its ceiling is X."""
    for key in ("rtx5090", "rtx3090"):
        for precision, fact in profile(key).peak_tflops.items():
            if fact.value is not None:
                assert "dense" in fact.source, (key, precision)


# ── no number without its source ────────────────────────────────────────────────────────────────────

def test_every_fact_of_every_shipped_type_names_its_source():
    assert set(types()) == {"gb10", "rtx5090", "rtx3090", "radeon8060s"}
    for p in types().values():
        for fact in _facts(p):
            assert isinstance(fact, Fact) and len(fact.source) >= 20, (p.key, fact)


def test_a_fact_without_a_source_cannot_be_made():
    with pytest.raises(ValueError, match="source"):
        Fact(6144, "")
    with pytest.raises(ValueError, match="source"):
        Fact(None, "  ")


# ── the library refuses what it cannot read, naming the file ────────────────────────────────────────

GOOD = (TYPES_DIR / "rtx5090.yml").read_text()


def _library(tmp_path: Path, text: str, name: str = "card.yml") -> Path:
    (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize("broken, says", [
    (GOOD.replace("memory: discrete", "memory: hybrid"), "hybrid"),                       # no such family
    (GOOD.replace("name: NVIDIA GeForce RTX 5090\n", ""), "name"),                       # a required key missing
    (GOOD + "colour: green\n", "colour"),                                                # an unknown key
    (GOOD.replace("  cuda_cores: {value: 21760, source: ", "  cores: {value: 21760, source: "), "cores"),
    (GOOD.replace('{value: 21760, source: ', '{value: 21760, origin: '), "source"),           # a fact with no source
    (GOOD.replace("memory: discrete\n", "memory: discrete\narch: sm_120\n"), "arch"),   # arch is its accelerator's
    (GOOD.replace("{value: 1792,", "{value: fast,"), "peak_gbps"),                       # not a number
    (GOOD.replace("{value: 21760,", "{value: true,"), "cuda_cores"),                     # a boolean is no count
    (GOOD.replace("controls: [local]", "controls: local"), "controls"),                         # not a list
    (GOOD + "name: again\n", "name"),                                                    # a key twice in YAML
    ("- just\n- a list\n", "mapping"),
])
def test_a_broken_type_file_is_refused_naming_file_and_value(tmp_path, broken, says):
    assert broken != GOOD, "the mutation matched nothing in the type file"
    with pytest.raises(ValueError) as e:
        load_types(_library(tmp_path, broken))
    assert "card.yml" in str(e.value) and says in str(e.value)


def test_a_type_is_named_by_its_file_and_nothing_else(tmp_path):
    lib = load_types(_library(tmp_path, GOOD, "my-card.yml"))
    assert list(lib) == ["my-card"] and lib["my-card"].key == "my-card"


def test_an_empty_library_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no hardware type"):
        load_types(tmp_path)


# ── how a type is chosen ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["", "GB10", "rtx 5090", "5090", "nvidia", "auto"])
def test_an_unknown_key_is_refused_naming_the_ones_there_are_never_guessed(key):
    with pytest.raises(ValueError) as e:
        profile(key)
    for known in types():
        assert known in str(e.value)


# ── what a reader of /lb/status gets ────────────────────────────────────────────────────────────────

def test_the_view_carries_every_figure_with_its_source_and_is_json():
    v = profile("rtx3090").view()
    json.dumps(v)
    assert v["key"] == "rtx3090" and v["name"] == "NVIDIA GeForce RTX 3090" and v["memory"] == "discrete"
    assert v["cuda_cores"] == {"value": 10496, "source": profile("rtx3090").cuda_cores.source}
    assert v["peak_tflops"]["nvfp4"]["value"] is None and v["peak_tflops"]["nvfp4"]["source"]
    assert profile("gb10").view()["memory"] == "unified"


def test_a_precision_the_type_does_not_list_is_absent_with_its_reason():
    fact = profile("rtx5090").peak_tflops_for("fp64")
    assert fact.value is None and all(p in fact.source for p in ("bf16", "fp8", "nvfp4"))
    assert profile("rtx5090").peak_tflops_for("nvfp4") is profile("rtx5090").peak_tflops["nvfp4"]
    assert "lists none" in profile("radeon8060s").peak_tflops_for("bf16").source


def test_the_package_is_the_one_door():
    for name in ("Fact", "HardwareProfile", "UnifiedProfile", "DiscreteProfile", "TYPES_DIR", "load_types",
                 "types", "profile"):
        assert hasattr(hardware, name), name
    for gone in ("GB10", "RTX_5090", "RTX_3090", "CLUSTER", "Gb10Unified", "NvidiaDiscrete", "PROFILES"):
        assert not hasattr(hardware, gone), f"{gone}: a hardware literal in code — the facts are YAML now"


def test_the_memory_a_launch_takes_a_fraction_of_is_the_familys_reading_of_the_type():
    """What gpu_memory_utilization is a fraction of, as its FAMILY reads the type: a discrete card's own VRAM (its spec
    `memory_gb`); on unified memory the GPU POOL vLLM itself measures and takes the fraction of (`visible_gib`, its start
    check's total — the spec states no size for an APU, and a GB10's 128 GB is not what vLLM divides) — each with its
    source, or absent with its reason (a GB10's pool is not measured: no one-card start check runs on it here)."""
    card, chip, apu = profile("rtx5090").gpu_memory(), profile("gb10").gpu_memory(), profile("radeon8060s").gpu_memory()
    assert (card.value, chip.value, apu.value) == (32, None, 124.0)
    assert card.source.startswith("its own VRAM") and chip.source.startswith("the GPU pool vLLM measures")
    assert "boot-startcheck.log:58" in apu.source


def test_the_radeon_8060s_states_the_start_memory_measured_on_this_fleet():
    """2026-10-17, rocm/vllm@sha256:394194d3… (vLLM 0.19.1+rocm7.13.0rc2, AMD's gfx1151 build), vLLM's own start check
    refusing a deliberate 0.99: 'Free memory on device cuda:0 (119.26/124.0 GiB)'. The free figure is the GPU pool's,
    NOT the box's RAM (MemAvailable 90.39 GiB then) — which is why the fit also bounds the pool by the box (fleet)."""
    t = profile("radeon8060s")
    assert (t.visible_gib.value, t.free_at_start_gib.value) == (124.0, 119.26)
    assert "394194d3" in t.visible_gib.source and "boot-startcheck.log:58" in t.free_at_start_gib.source
    assert "NOT the box's RAM" in t.free_at_start_gib.source


def test_a_kv_tier_in_the_hosts_ram_is_the_familys_to_allow():
    """A KV tier in the HOST's RAM (vLLM --kv-offloading-size): beside a discrete card's own VRAM it is a second,
    larger pool; on unified memory the host's RAM IS the GPU's pool, so the tier would take the memory the KV pool lives
    in — refused in the family's own words, for every type of it, never by a caller asking which kind of box it is."""
    for t in types().values():
        assert (t.memory.ram_tier_refusal == "") is (t.memory is DISCRETE), (t.key, t.memory.ram_tier_refusal)
    assert DISCRETE.ram_tier_refusal == ""
    assert UNIFIED.ram_tier_refusal.startswith("unified memory: the host RAM is the GPU's own pool"), \
        UNIFIED.ram_tier_refusal
