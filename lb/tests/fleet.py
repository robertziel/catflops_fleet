"""Hardware figures the tests were written against, named once.

The SHIPPED defaults claim no hardware: a GPU's peaks are its card's hardware type's (vllm_lb/hardware/types),
named by a fleet file. A test that reasons about a GB10's share of its bandwidth says so here, so the assumption
is visible.
"""
from __future__ import annotations

#: The GB10's published LPDDR5X figure, GB/s: the peak the saturation and bandwidth tests were written against.
GB10_PEAK_GBPS = 273.0
