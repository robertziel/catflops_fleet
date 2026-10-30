"""Claude Code's effort ladder is not qwen3.8's, and vLLM 500s on the mismatch.

Captured from the wire (one real Claude Code request, 86,405 bytes): the body carries

    "output_config": {"effort": "high"}

vLLM serves the Anthropic Messages API natively and maps that into the chat template's
`reasoning_effort`. Verified against the live backend, one value per request:

    high   -> 500 Unexpected reasoning effort high. Supported types are xhigh, medium, low.
    max    -> 500 Unexpected reasoning effort max.
    xhigh  -> 200      medium -> 200      low -> 200

`high` and `max` ARE Claude Code's ladder, so every real session failed while the hand-written
curl probes used in earlier passes — which carry no output_config — all passed.
"""
from __future__ import annotations

import pytest

from vllm_lb.config import Config
from vllm_lb.router import rewrite_effort
from vllm_lb.state import VLLM, Backend

CFG = Config()
V = Backend(kind=VLLM, url="http://box-a:8000", host_label="box-a-gpu", box="box-a")


@pytest.mark.parametrize("sent,want", [("high", "medium"), ("max", "xhigh")])
def test_an_unsupported_effort_becomes_one_the_template_accepts(sent, want):
    """CHANGED 2026-09-12 for `high` (worklog/vllm-lb-effort-default.md): it used to map to
    `xhigh`, the model's most verbose reasoning tier, and since `high` is what Claude Code
    sends by DEFAULT the operator typed `ping` and got "Thought for 6m 47s". `medium` injects
    no reasoning instruction at all. `max` still reaches `xhigh`."""
    body = {"model": "m", "output_config": {"effort": sent}}
    out = rewrite_effort(body, V, CFG)
    assert out["output_config"]["effort"] == want
    assert body["output_config"]["effort"] == sent, "the caller's dict must not be mutated"


@pytest.mark.parametrize("sent", ["low", "medium", "xhigh"])
def test_a_supported_effort_is_left_alone(sent):
    body = {"model": "m", "output_config": {"effort": sent}}
    assert rewrite_effort(body, V, CFG) is body, "no copy when nothing changes"


def test_the_same_mapping_applies_to_chat_template_kwargs():
    """The field vLLM actually hands the template — reproduced as a 500 directly."""
    body = {"model": "m", "chat_template_kwargs": {"reasoning_effort": "high"}}
    assert rewrite_effort(body, V, CFG)["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_a_body_with_no_effort_is_untouched():
    body = {"model": "m", "messages": []}
    assert rewrite_effort(body, V, CFG) is body


def test_other_output_config_keys_survive_the_rewrite():
    body = {"output_config": {"effort": "high", "verbosity": "low"}}
    out = rewrite_effort(body, V, CFG)
    assert out["output_config"] == {"effort": "medium", "verbosity": "low"}


# ── "off": thinking disabled, never an effort word ────────────────────────────
# Measured 2026-10-14 on site-b's vLLM (Qwen3.8 27B): output_config.effort "none"/"minimal" -> 400
# literal_error, while `"thinking": {"type": "disabled"}` -> 200 with text only (also on box-c's SGLang).
# The template's own switch is chat_template_kwargs.enable_thinking false. So "off" — the fourth level the
# catalog publishes (catalog.json `efforts`) — is translated here, never forwarded.

def test_off_in_output_config_becomes_thinking_disabled():
    body = {"model": "m", "output_config": {"effort": "off"}, "max_tokens": 9}
    out = rewrite_effort(body, V, CFG)
    assert out == {"model": "m", "thinking": {"type": "disabled"}, "max_tokens": 9}, "the empty section goes too"
    assert body["output_config"] == {"effort": "off"} and "thinking" not in body, "the caller's dict is untouched"


def test_off_keeps_the_rest_of_output_config():
    out = rewrite_effort({"output_config": {"effort": "off", "verbosity": "low"}}, V, CFG)
    assert out == {"output_config": {"verbosity": "low"}, "thinking": {"type": "disabled"}}


def test_off_in_chat_template_kwargs_becomes_enable_thinking_false():
    body = {"chat_template_kwargs": {"reasoning_effort": "off", "x": 1}}
    out = rewrite_effort(body, V, CFG)
    assert out == {"chat_template_kwargs": {"enable_thinking": False, "x": 1}}
    assert body == {"chat_template_kwargs": {"reasoning_effort": "off", "x": 1}}


def test_off_overrides_a_thinking_block_the_client_also_sent():
    out = rewrite_effort({"output_config": {"effort": "off"}, "thinking": {"type": "adaptive"}}, V, CFG)
    assert out == {"thinking": {"type": "disabled"}}
