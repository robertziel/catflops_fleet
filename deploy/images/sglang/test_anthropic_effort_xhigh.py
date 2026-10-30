"""/v1/messages must pass Claude Code's reasoning effort through to a word the served chat
template accepts.

Runs inside the SGLang image (`python -m pytest`), against the image's own code: the real
Anthropic request models and the real adapter. Upstream's adapter harness is imported, not copied.

The defect (2026-10-08, 27B SGLang on /v1/messages?beta=true): the adapter mapped
`output_config.effort == "xhigh"` onto the OpenAI word `"max"` — a collapse it did "because the
OpenAI Literal does not include the Anthropic-only xhigh", which is no longer true (the image's
`ReasoningEffortType` carries `xhigh` and `max` both). qwen3.8's chat template accepts
`low`/`medium`/`xhigh` and `raise_exception`s on anything else, so the minted `max` died at
template render: `jinja2.exceptions.TemplateError: Unexpected reasoning effort max` → HTTP 500,
79 bytes, ~50 ms, before any GPU work. Claude Code retried it ten times (`attempt 6/10`). It hit
BOTH client settings — `xhigh` directly, and `max` after the LB's own `max -> xhigh` rewrite — so
every high-effort turn 500'd regardless of what the user picked.

The fix passes `output_config.effort` through verbatim; the template then accepts every word the
fleet actually sends (the LB maps the client's `high`/`max` onto `medium`/`xhigh` upstream of here).
"""

from fleet_harness import upstream  # stubs sgl_kernel first, as upstream's own run does

# The reasoning-effort words qwen3.8's chat template renders without raising (chat_template.jinja:
# `resolved_reasoning_effort not in ('xhigh', 'medium', 'low')` -> raise_exception). `max` is NOT
# in it — which is exactly why the old xhigh->max collapse 500'd.
_TEMPLATE_ACCEPTS = {"low", "medium", "xhigh"}

# Anthropic `output_config.effort` Literal (protocol.py): what a client may send.
_ANTHROPIC_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def _body(effort):
    return {"model": "m", "max_tokens": 16, "stream": False,
            "messages": [{"role": "user", "content": "go"}],
            "output_config": {"effort": effort}}


def _reasoning_effort_for(effort):
    serving = upstream.AnthropicServing(upstream._FakeOpenAIServingChat())
    request = upstream.AnthropicMessagesRequest.model_validate(_body(effort))
    chat_request = serving._convert_to_chat_completion_request(request)
    return chat_request.reasoning_effort


def test_output_config_effort_passes_through_verbatim():
    """Every Anthropic effort word reaches `reasoning_effort` unchanged — no remapping in between."""
    for effort in _ANTHROPIC_EFFORTS:
        assert _reasoning_effort_for(effort) == effort, effort


def test_xhigh_is_not_collapsed_to_max():
    """The regression: `xhigh` minted `max`, which the chat template rejects. It must stay `xhigh`,
    and `xhigh` is a word the template renders."""
    got = _reasoning_effort_for("xhigh")
    assert got == "xhigh"
    assert got in _TEMPLATE_ACCEPTS
    assert "max" not in _TEMPLATE_ACCEPTS, "max is unrenderable — the collapse's target was fatal"
