"""THE ACCELERATORS (vllm_lb/hardware/accelerators/*.yml): what an engine's KERNELS target — one file per target, every
figure {value, source} like a card type's.

A card TYPE is one product (its memory, cores, peaks); its ACCELERATOR is what its kernels are compiled for and what
they can run: the arch (sm_121, sm_120, sm_86, gfx1151), how a container attaches it (gpu_runtime), the CPU platforms
its hosts come in, the precisions its tensor units take and the quantisation formats served on it. Two products of one
accelerator share one file (one place per value: `arch` and `gpu_runtime` left the type files on 2026-10-30).

What these tests pin, from the sources' own words:
  * developer.nvidia.com/cuda-gpus: GB10 12.1, RTX 5090 12.0, RTX 3090 8.6 — so the GB10 and the 5090 are TWO
    accelerators: this repo's GB10 images are compiled for sm_121a (deploy/images/vllm/Dockerfile (stage glm53)
    TORCH_CUDA_ARCH_LIST=12.1a; deploy/images/vllm/Dockerfile (stage deepseek-v4-b12x) CUTE_DSL_ARCH=sm_121a), and an
    arch-specific (`a`) cubin runs only on that exact compute capability (CUDA C++ Programming Guide, "Feature
    Availability": arch-specific features are not forward- or cross-compatible);
  * the RTX Blackwell whitepaper, Table 3: FP8 and FP4 tensor TFLOPS N/A on the 3090 — Ampere has neither;
  * AMD's ROCm GPU specification table: Radeon 8060S = gfx1151.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_lb.hardware import (ACCELERATORS_DIR, TYPES_DIR, Accelerator, accelerator, accelerators, compatibility,
                              load_accelerators, load_types, profile, types)


# ── the shipped library ─────────────────────────────────────────────────────────────────────────────

def test_every_shipped_card_type_names_an_accelerator_in_the_library():
    assert {t.key: t.accelerator.key for t in types().values()} == {
        "gb10": "nvidia-sm121", "rtx5090": "nvidia-sm120", "rtx3090": "nvidia-sm86", "radeon8060s": "amd-gfx1151"}
    assert set(accelerators()) == {"nvidia-sm121", "nvidia-sm120", "nvidia-sm86", "amd-gfx1151"}


def test_the_arch_and_runtime_of_a_card_are_its_accelerators_and_nowhere_else():
    """ONE place per value: a type file no longer states `arch` or `gpu_runtime`; the profile reads them through."""
    for t in types().values():
        assert t.arch is t.accelerator.arch and t.gpu_runtime is t.accelerator.gpu_runtime, t.key
    for f in TYPES_DIR.glob("*.yml"):
        text = f.read_text()
        assert "\n  arch:" not in text and "\n  gpu_runtime:" not in text, f.name
        assert "\naccelerator: " in text, f.name


def test_the_gb10_and_the_5090_are_two_accelerators_because_their_kernels_differ():
    gb10, card = accelerator("nvidia-sm121"), accelerator("nvidia-sm120")
    assert (gb10.arch.value, card.arch.value) == ("sm_121", "sm_120")
    assert "12.1a" in gb10.arch.source or "sm_121a" in gb10.arch.source, "the images' own arch-specific target"
    assert gb10.platforms.value == ("linux/arm64",), "the GB10 superchip carries the box's Arm CPU"


def test_ampere_has_no_fp8_or_fp4_and_its_accelerator_says_so():
    a = accelerator("nvidia-sm86")
    assert a.arch.value == "sm_86"
    assert "fp8" not in a.precisions.value and "nvfp4" not in a.precisions.value
    assert "N/A" in a.precisions.source


def test_the_amd_igpu_is_gfx1151_attached_through_rocm():
    a = accelerator("amd-gfx1151")
    assert (a.arch.value, a.gpu_runtime.value) == ("gfx1151", "rocm")


def test_every_fact_of_every_accelerator_names_its_source():
    for a in accelerators().values():
        for fact in (a.arch, a.gpu_runtime, a.platforms, a.precisions, a.quant_formats):
            assert len(fact.source) >= 20, (a.key, fact)


def test_a_types_listed_peaks_are_precisions_its_accelerator_takes():
    for t in types().values():
        known = {p for p, f in t.peak_tflops.items() if f.known}
        assert known <= set(t.accelerator.precisions.value), (t.key, known)


def test_the_view_carries_the_accelerator_and_stays_json():
    import json
    v = profile("gb10").view()
    json.dumps(v)
    assert v["accelerator"]["key"] == "nvidia-sm121" and v["arch"]["value"] == "sm_121"
    assert v["gpu_runtime"]["value"] == "nvidia"


# ── the loader refuses what it cannot mean ──────────────────────────────────────────────────────────

GOOD = (ACCELERATORS_DIR / "nvidia-sm120.yml").read_text()


def _lib(tmp_path: Path, text: str, name: str = "acc.yml") -> Path:
    (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize("broken, says", [
    (GOOD.replace("gpu_runtime: {value: nvidia,", "gpu_runtime: {value: cuda,"), "gpu_runtime"),
    (GOOD.replace("platforms: {value: [linux/amd64]", "platforms: {value: [linux/s390x]"), "platforms"),
    (GOOD + "colour: green\n", "colour"),
    (GOOD.replace("precisions: {value: [", "precisions: {value: [7, "), "precisions"),
    (GOOD.replace("name: ", "label: ", 1), "name"),
])
def test_a_broken_accelerator_file_is_refused_naming_file_and_value(tmp_path, broken, says):
    assert broken != GOOD, "the mutation matched nothing"
    with pytest.raises(ValueError) as e:
        load_accelerators(_lib(tmp_path, broken))
    assert "acc.yml" in str(e.value) and says in str(e.value)


def test_a_type_naming_an_accelerator_there_is_not_is_refused(tmp_path):
    text = (TYPES_DIR / "rtx5090.yml").read_text().replace("accelerator: nvidia-sm120", "accelerator: nvidia-sm999")
    (tmp_path / "rtx5090.yml").write_text(text)
    with pytest.raises(ValueError) as e:
        load_types(tmp_path)
    assert "nvidia-sm999" in str(e.value) and "nvidia-sm120" in str(e.value), "names the ones there are"


def test_a_type_whose_peak_its_accelerator_cannot_run_is_refused(tmp_path):
    text = (TYPES_DIR / "rtx3090.yml").read_text()
    text = text.replace('fp8: {value: null,', 'fp8: {value: 142.0,')
    (tmp_path / "rtx3090.yml").write_text(text)
    with pytest.raises(ValueError) as e:
        load_types(tmp_path)
    assert "fp8" in str(e.value) and "nvidia-sm86" in str(e.value)


def test_a_type_on_a_platform_its_accelerator_never_comes_in_is_refused(tmp_path):
    text = (TYPES_DIR / "gb10.yml").read_text().replace("host_platform: {value: linux/arm64,",
                                                         "host_platform: {value: linux/amd64,")
    (tmp_path / "gb10.yml").write_text(text)
    with pytest.raises(ValueError) as e:
        load_types(tmp_path)
    assert "linux/amd64" in str(e.value) and "nvidia-sm121" in str(e.value)


@pytest.mark.parametrize("key", ["", "sm_121", "blackwell", "auto"])
def test_an_unknown_accelerator_is_refused_naming_the_ones_there_are(key):
    with pytest.raises(ValueError) as e:
        accelerator(key)
    for known in accelerators():
        assert known in str(e.value)


# ── the compatibility check (validation only: no launch reads it yet) ───────────────────────────────

def test_a_quant_format_the_accelerator_has_no_kernel_for_is_a_refusal_naming_both():
    why = compatibility(accelerator("nvidia-sm86"), quant_format="modelopt_mixed", platform="linux/amd64")
    assert "modelopt_mixed" in why and "nvidia-sm86" in why


def test_a_platform_the_accelerator_never_comes_in_is_a_refusal():
    why = compatibility(accelerator("nvidia-sm121"), quant_format="modelopt_mixed", platform="linux/amd64")
    assert "linux/amd64" in why and "nvidia-sm121" in why


def test_what_the_accelerator_runs_passes_with_no_reason():
    assert compatibility(accelerator("nvidia-sm121"), quant_format="modelopt_mixed", platform="linux/arm64") == ""
    assert compatibility(accelerator("nvidia-sm120"), quant_format="none", platform="linux/amd64") == ""


def test_an_accelerator_that_states_no_quant_formats_refuses_every_one_with_its_reason(tmp_path):
    text = GOOD.replace("quant_formats: {value: [", "quant_formats: {value: null, was: [", 1)
    text = text.split("quant_formats:")[0] + 'quant_formats: {value: null, source: "not established on this card"}\n'
    lib = load_accelerators(_lib(tmp_path, text, "nvidia-sm120.yml"))
    why = compatibility(lib["nvidia-sm120"], quant_format="none", platform="linux/amd64")
    assert "not established on this card" in why


def test_a_box_whose_platform_its_cards_accelerator_never_comes_in_is_refused():
    """The fleet file: a box's CPU must be one its cards' accelerator comes in — no image is built for any other."""
    from vllm_lb.fleet import parse_fleet
    text = ("panel: {style: vibe}\nboxes:\n  solo:\n    hostname: solo-host\n    platform: linux/arm64\n    cards:\n"
            "      - {index: 0, type: rtx3090, serves: false}\nclusters: {}\nbackends: {}\n")
    with pytest.raises(ValueError) as e:
        parse_fleet(text, "fleet.yml")
    said = str(e.value)
    assert "linux/arm64" in said and "nvidia-sm86" in said and "fleet.yml" in said, said


def test_an_accelerator_is_a_frozen_value():
    a = accelerator("nvidia-sm121")
    assert isinstance(a, Accelerator)
    with pytest.raises(Exception):
        a.key = "other"           # type: ignore[misc]
