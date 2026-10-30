"""/v1/messages must accept `thinking.display: "updates"`, the value Claude Code sends by default.

Runs inside the SGLang image (`python -m pytest`), against the image's own code: the real Anthropic
request models, the real adapter and the real FastAPI app (route class + validation handler).
Upstream's adapter harness is imported, not copied.

The defect (2026-10-03, 27B SGLang): Claude Code 2.1.288 in a terminal sends
`thinking: {"type": "adaptive", "display": "updates"}` with the beta header
`thinking-display-updates-2026-08-18` on the first request of every process (a start, a resume). The
image's `AnthropicThinkingParam.display` is `Literal["summarized", "omitted"]`, so the request died in
pydantic before reaching the adapter: HTTP 400 in ~7 ms, 129 bytes —
`{"type":"error","error":{"type":"invalid_request_error","message":"thinking.display: Input should be
'summarized' or 'omitted'"}}`. Claude Code heals by retrying once without `display` and latching it
off, but only once a retry is answered — interrupt the retry during prefill and the next turn sends
it (and 400s) again, which is why the 400s sat on the first request after an Esc.

Anthropic's meaning of "updates": reasoning hidden like "omitted", short progress notes returned as
`thinking` blocks. This backend cannot hide or summarise reasoning mid-stream — it already accepts
"omitted" and streams the full reasoning with a WARNING — so "updates" gets exactly that treatment:
accepted, reasoning on, nothing dropped.
"""

import asyncio
import logging

import pytest
from pydantic import ValidationError

from fleet_harness import upstream  # stubs sgl_kernel first, as upstream's own run does

from sglang.srt.entrypoints.anthropic.protocol import AnthropicCountTokensRequest  # noqa: E402

# The request fields Claude Code 2.1.288 sent on its first turn (captured 2026-10-03 through a
# recording stub, interactive session) — identifiers and the classifier payload left out.
_BETAS = ("claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
          "context-management-2025-06-27,prompt-caching-scope-2026-01-05,"
          "mid-conversation-system-2026-04-07,mid-conversation-tool-changes-2026-07-01,"
          "effort-2025-11-24,dangerous-tool-use-2026-09-03,thinking-display-updates-2026-08-18,"
          "afk-mode-2026-01-31")


# "updates" is its default (beta thinking-display-updates-2026-08-18); "highlights" is the other value
# its thinking_display setting takes (`jWe=["summarized","omitted","highlights"]` in the 2.1.288 binary,
# beside "the API has already rejected \"highlights\""). Neither was in the image's Literal.
_SENT = ("updates", "highlights")


def _claude_code_body(display="updates", **overrides):
    body = {
        "model": "qwen3.8:latest",
        "metadata": {"user_id": "{\"device_id\":\"d\",\"account_uuid\":\"\",\"session_id\":\"s\"}"},
        "max_tokens": 16384,
        "thinking": {"type": "adaptive", "display": display},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "output_config": {"effort": "high"},
        "stream": True,
        "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "Run a command",
                   "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "launch it"}]},
            {"role": "system", "content": "# Environment\n - Platform: linux"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Now launch the Workflow.", "signature": ""},
                {"type": "text", "text": "\n\nResuming — launching Wave 1 now."}]},
            {"role": "user", "content": [{"type": "text", "text": "[Request interrupted by user]\n"},
                                         {"type": "text", "text": "resume"}]},
        ],
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize("display", _SENT)
def test_claude_codes_request_validates(display):
    request = upstream.AnthropicMessagesRequest.model_validate(_claude_code_body(display))
    assert request.thinking.display == display


@pytest.mark.parametrize("display", _SENT)
def test_the_route_no_longer_answers_400(display):
    """Through the image's real FastAPI app: the route class, the body parse and the validation
    handler that wrote the 129-byte 400. A stub stands in for the adapter behind the route."""
    from fastapi.responses import JSONResponse
    from starlette.testclient import TestClient

    from sglang.srt.entrypoints import http_server

    reached = []

    class _Serving:
        async def handle_messages(self, request, raw_request):
            reached.append(request.thinking.display)
            return JSONResponse({"reached": "adapter"})

    app = http_server.app
    had = getattr(app.state, "anthropic_serving", None)
    app.state.anthropic_serving = _Serving()
    try:
        r = TestClient(app).post("/v1/messages?beta=true", json=_claude_code_body(display),
                                 headers={"anthropic-beta": _BETAS, "anthropic-version": "2023-06-01"})
    finally:
        app.state.anthropic_serving = had
    assert (r.status_code, len(r.content), r.content) == (200, len(b'{"reached":"adapter"}'),
                                                          b'{"reached":"adapter"}')
    assert reached == [display]


@pytest.mark.parametrize("display", _SENT)
def test_reasoning_stays_on_and_the_unhonoured_display_is_logged(display):
    """Same contract as upstream's "omitted": accepted, reasoning enabled, a WARNING saying the
    reasoning text is still emitted."""
    serving = upstream.AnthropicServing(upstream._FakeOpenAIServingChat())
    request = upstream.AnthropicMessagesRequest.model_validate(_claude_code_body(display, stream=False))
    logger = logging.getLogger("sglang.srt.entrypoints.anthropic.serving")
    records = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        serving._convert_to_chat_completion_request(request)
    finally:
        logger.removeHandler(handler)
    assert serving.openai_serving_chat.apply_reasoning_calls == [True]
    assert any(display in r.getMessage() and "still be emitted" in r.getMessage() for r in records)


def test_the_reasoning_still_reaches_the_client_as_a_thinking_block():
    """Nothing is dropped: the full reasoning streams back as a thinking block, then the answer."""
    serving = upstream.AnthropicServing(upstream._FakeOpenAIServingChat([
        upstream._chunk([upstream._choice({"role": "assistant", "content": ""})]),
        upstream._chunk([upstream._choice({"reasoning_content": "check the queue first"})]),
        upstream._chunk([upstream._choice({"content": "done"})]),
        upstream._chunk([upstream._choice({}, finish_reason="stop")],
                        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        "data: [DONE]\n\n",
    ]))
    request = upstream.AnthropicMessagesRequest.model_validate(_claude_code_body())
    events = asyncio.run(upstream._collect_anthropic_events(serving, request))
    blocks = [(e["content_block"]["type"]) for e in events if e["type"] == "content_block_start"]
    deltas = [e["delta"].get("thinking", e["delta"].get("text")) for e in events
              if e["type"] == "content_block_delta"]
    assert blocks == ["thinking", "text"]
    assert deltas == ["check the queue first", "done"]


@pytest.mark.parametrize("display", _SENT)
def test_count_tokens_accepts_it_too(display):
    body = _claude_code_body(display)
    for key in ("max_tokens", "stream", "metadata", "context_management"):
        body.pop(key)
    assert AnthropicCountTokensRequest.model_validate(body).thinking.display == display


@pytest.mark.parametrize("thinking", [
    {"type": "adaptive", "display": "bogus"},      # an unknown value is still a 400
    {"type": "disabled", "display": "updates"},    # disabled still takes no display
])
def test_the_sdk_shape_rules_still_hold(thinking):
    with pytest.raises(ValidationError):
        upstream.AnthropicMessagesRequest.model_validate(_claude_code_body(thinking=thinking))
