"""The real SGLang path from decode steps to what a /v1/messages client reassembles, for the fleet's
patch tests. Runs inside the SGLang image against the image's own code: the qwen3_coder detector,
the OpenAI stream builder (`_process_tool_call_stream` with the terminal flush, then
`_check_for_unstreamed_tool_args`, as `_generate_stream_content` calls them) and the Anthropic
adapter. Upstream's adapter harness is imported, not copied."""

import asyncio
import importlib.util
import inspect
import json
from types import SimpleNamespace

_UPSTREAM = "/sgl-workspace/sglang/test/registered/unit/entrypoints/anthropic/test_serving.py"
_spec = importlib.util.spec_from_file_location("upstream_anthropic_test_serving", _UPSTREAM)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)   # stubs sgl_kernel first, as upstream's own run does

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402


def _created(method):
    """The `created` argument where this tree's method takes one (upstream added it after 708f51e — v0.5.22 has it on
    both stream methods), none where it does not: the harness drives each tree the fleet builds by its own signature."""
    return {"created": 0} if "created" in inspect.signature(method).parameters else {}


def engine_stream(steps, tools, finish="stop"):
    """The OpenAI SSE lines SGLang's own chat stream emits for these decode steps."""
    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    chat.tool_call_parser = "qwen3_coder"
    chat.tokenizer_manager = SimpleNamespace(tokenizer=None)
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "go"}],
                                    tools=tools, stream=True)
    parser_dict, has_tool_calls = {}, {}
    content = {"meta_info": {"id": "chatcmpl-test"}}

    async def run():
        lines = [upstream._chunk([upstream._choice({"role": "assistant", "content": ""})])]
        for n, step in enumerate(steps):
            last = n == len(steps) - 1
            async for line in chat._process_tool_call_stream(
                    0, step, parser_dict, content, request, has_tool_calls, flush=last,
                    **_created(OpenAIServingChat._process_tool_call_stream)):
                lines.append(line)
            if last and 0 in parser_dict:   # as _generate_stream_content does at the finish
                rest = chat._check_for_unstreamed_tool_args(
                    parser_dict[0], content, request, 0, **_created(OpenAIServingChat._check_for_unstreamed_tool_args))
                if rest:
                    lines.append(rest)
        # _generate_chat_stream: "stop" becomes "tool_calls" when calls were made; "length" stays
        final = "tool_calls" if finish == "stop" and has_tool_calls.get(0) else finish
        lines.append(upstream._chunk([upstream._choice({}, finish_reason=final)]))
        lines.append("data: [DONE]\n\n")
        return lines

    return asyncio.run(run())


def messages_events(lines):
    """The /v1/messages events the Anthropic adapter makes of these OpenAI SSE lines."""
    serving = upstream.AnthropicServing(upstream._FakeOpenAIServingChat(lines))
    request = upstream.AnthropicMessagesRequest.model_validate(
        {"model": "m", "max_tokens": 16, "stream": True,
         "messages": [{"role": "user", "content": "go"}]})
    return asyncio.run(upstream._collect_anthropic_events(serving, request))


def client(steps, tools, finish="stop"):
    """What a client reassembles from /v1/messages: (blocks, stop_reason). A text block is
    ("text", text); a tool_use block is ("tool_use", name, raw input JSON, parsed input or None
    when it does not parse) — no input delta at all means `{}`."""
    blocks, stop = {}, None
    for ev in messages_events(engine_stream(steps, tools, finish)):
        if ev["type"] == "content_block_start":
            cb = ev["content_block"]
            blocks[ev["index"]] = {"type": cb["type"], "name": cb.get("name"), "buf": ""}
        elif ev["type"] == "content_block_delta":
            d = ev["delta"]
            blocks[ev["index"]]["buf"] += d.get("partial_json", d.get("text", ""))
        elif ev["type"] == "message_delta":
            stop = ev["delta"]["stop_reason"]
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] != "tool_use":
            out.append((b["type"], b["buf"]))
            continue
        raw = b["buf"] or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        out.append(("tool_use", b["name"], raw, parsed))
    return out, stop


def tool_calls(blocks):
    """(name, parsed input) of each tool_use block."""
    return [(b[1], b[3]) for b in blocks if b[0] == "tool_use"]
