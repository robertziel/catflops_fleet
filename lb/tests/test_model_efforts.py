"""A MODEL'S THINKING LEVELS ARE ITS OWN DATA — `efforts` in catalog.json, published on `loaded_model`.

Operator, 2026-10-14: "add all thinking levels on site-b vllm" — the four REAL levels the Qwen3.8 27B chat
template supports, off / low / medium / xhigh. The template (chat_template.jinja lines 46-55, md5 519239a4908b
on the 27B's original, its uncensored twin and the classic seed) takes `reasoning_effort` in xhigh (default),
medium, low and raises on anything else; `enable_thinking` false turns reasoning off. So the ladder is stated
on the rows whose template that is, default first, and a row with no stated ladder publishes none (null) —
the chat and the panel then say so instead of inventing one.
"""
from __future__ import annotations

import copy

import pytest

from tests.test_model_catalog import DENSE, DFLASH, FLASH, FLASH_SGLANG, SEEDED, SGLANG, raw, shipped, spec_raw
from vllm_lb.catalog import ModelCatalog

LADDER = ["xhigh", "medium", "low", "off"]


def test_the_rows_on_the_27b_template_state_its_four_levels_default_first():
    c = shipped()
    for spec_id in (DENSE, SGLANG, SEEDED):
        assert c.get(spec_id).efforts == tuple(LADDER), spec_id
        assert c.get(spec_id).view()["efforts"] == LADDER
    # Not read off a template on this fleet (the checkpoints were unreachable): no ladder, never a guess.
    for spec_id in (FLASH, FLASH_SGLANG, DFLASH):
        assert c.get(spec_id).efforts is None and c.get(spec_id).view()["efforts"] is None, spec_id
    assert "chat_template.jinja" in spec_raw(raw(), DENSE)["efforts"]["source"]


def test_the_loaded_model_publishes_its_ladder_and_a_stranger_none():
    c = shipped()
    assert c.loaded_model(("Qwen/Qwen3.8-27B",), 131072).view()["efforts"] == LADDER
    assert c.loaded_model(("Qwen/Qwen3.8-Flash-Next",), 131072).view()["efforts"] is None
    assert c.loaded_model(("someone/else",), None).view()["efforts"] is None


@pytest.mark.parametrize("efforts, says", [
    ({"ladder": ["xhigh", "loud"], "source": "x"}, "unknown effort 'loud'"),
    ({"ladder": ["low", "low"], "source": "x"}, "repeats"),
    ({"ladder": [], "source": "x"}, "non-empty list"),
    ({"ladder": "xhigh", "source": "x"}, "non-empty list"),
    ({"ladder": ["xhigh"]}, "missing \\['source'\\]"),
    ({"ladder": ["xhigh"], "source": " "}, "source"),
    ({"ladder": ["xhigh"], "source": "x", "extra": 1}, "unknown \\['extra'\\]"),
])
def test_a_ladder_is_validated_and_refused_loudly(efforts, says):
    doc = copy.deepcopy(raw())
    spec_raw(doc, DENSE)["efforts"] = efforts
    with pytest.raises(ValueError, match=says):
        ModelCatalog.from_dict(doc)
