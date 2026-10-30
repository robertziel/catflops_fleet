"""THE HARDWARE DOMAIN (vllm_lb/hardware): what the gateway knows about a KIND of GPU box, in one place.

A box's GPU memory is one of two THINGS, and each is a class of its own (AGENTS.md, "Hardware"):

  * UNIFIED — a GB10 (DGX Spark): CPU and GPU share 128 GB of LPDDR5X. nvidia-smi has no VRAM of its own to
    report there ([N/A]), the sender turns that into `vram_used_mb: 0, vram_total_mb: 0`, and the GPU's memory
    is the box's RAM. "0 GB VRAM" would be a number the wire never meant.
  * DISCRETE — a card with VRAM of its own (an RTX 5090's 32 GB): the box's figure is the cards' VRAM summed, the
    aggregate every reader of /lb/status already takes, and each card is a row of its own.

Which one a box is, is NAMED by its cards' types in the fleet file — never inferred from the wire. The samples below
are the wire's own shapes (tests/test_hw.py holds the verbatim captures).
"""
from __future__ import annotations

import pytest

from vllm_lb import hardware
from vllm_lb.hardware import DISCRETE, UNIFIED, GpuRow, MemoryModel

#: box-a's card, verbatim off the wire (tests/test_hw.py BOX_A): a GB10 reports no VRAM of its own.
GB10_CARD = {"id": 0, "use_pct": 96, "vram_used_mb": 0, "vram_total_mb": 0}
BOX_A = {"host": "box-a-host", "ram_used_mb": 92921, "ram_total_mb": 124610, "gpus": [GB10_CARD]}
#: A discrete card with VRAM of its own.
CARD = {"id": 0, "use_pct": 3, "vram_used_mb": 900, "vram_total_mb": 16384}


# ── how each model draws a box's GPUs ──────────────────────────────────────────────────────────────

def test_a_unified_box_is_drawn_as_one_gpu_row_whose_memory_is_the_mem_row():
    assert UNIFIED.rows((0,)) == (GpuRow("GPU", None, False),)
    assert UNIFIED.unified is True


def test_each_memory_model_names_itself_so_no_caller_has_to_ask_which_it_is():
    assert (UNIFIED.name, DISCRETE.name) == ("unified", "discrete")


def test_a_discrete_box_is_drawn_as_one_row_per_card_each_with_vram_of_its_own():
    assert DISCRETE.rows((1, 0)) == (GpuRow("GPU 0", 0, True), GpuRow("GPU 1", 1, True)), "index order"
    assert DISCRETE.unified is False


# ── what each model reads ───────────────────────────────────────────────────────────────────────────

def test_unified_memory_is_the_boxs_own_ram():
    assert UNIFIED.read(BOX_A, BOX_A["gpus"]) == (92921, 124610)


def test_unified_memory_reports_nothing_the_sample_did_not_send():
    assert UNIFIED.read({"gpus": []}, []) == (None, None)
    assert UNIFIED.read({"ram_used_mb": True, "ram_total_mb": "124610"}, []) == (None, None), \
        "a boolean or a string is not a figure"


def test_discrete_memory_sums_the_cards_that_report_a_total():
    big = {"id": 1, "use_pct": 5, "vram_used_mb": 12000, "vram_total_mb": 32607}
    assert DISCRETE.read({}, [CARD, big]) == (900 + 12000, 16384 + 32607)
    assert DISCRETE.read({}, [CARD, GB10_CARD, "junk"]) == (900, 16384), \
        "a card with no total of its own adds nothing to the box's VRAM"


# ── the abstraction itself ──────────────────────────────────────────────────────────────────────────

def test_every_memory_model_is_one_class_of_one_base():
    """A new kind of memory is a new subclass with its own tests — never an `if unified:` at a caller."""
    assert isinstance(UNIFIED, MemoryModel) and isinstance(DISCRETE, MemoryModel)
    assert type(UNIFIED) is not type(DISCRETE)
    with pytest.raises(NotImplementedError):
        MemoryModel().read({}, [])
    with pytest.raises(NotImplementedError):
        MemoryModel().rows((0,))


def test_the_public_surface_is_the_package():
    """Callers import from `vllm_lb.hardware`, never from a submodule: the package is the one door."""
    for name in ("MemoryModel", "UNIFIED", "DISCRETE", "GpuRow"):
        assert hasattr(hardware, name), name
    assert not hasattr(hardware, "memory_model_of"), "a box's memory model is named by its fleet, never inferred"
