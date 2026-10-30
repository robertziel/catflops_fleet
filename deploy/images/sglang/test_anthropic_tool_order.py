"""A tool call's closing `}` must reach its own tool_use block on /v1/messages.

Runs inside the SGLang image (`python -m pytest`) through fleet_harness: the image's own
detector, OpenAI stream builder and Anthropic adapter.

The defect (2026-10-03, 27B SGLang + DFlash2): when one decode step carries the end of a tool call
AND the text after it — `</function>\\n</tool_call>\\n<tool_call>` arrives as one speculative
step — the detector returns the step's text and its tool fragments separately, and
`_process_tool_call_stream` yields the text chunk FIRST. The adapter closed the open tool_use
block for that text, then logged "Dropping tool_call argument delta with no open tool_use block:
'}'" and dropped the closing brace, so Claude Code got `{"project": "x"` and answered
"InputValidationError ... could not be parsed as JSON".
"""

import json

import pytest

from fleet_harness import client, engine_stream, tool_calls

_TOOLS = [
    {"type": "function", "function": {"name": "Read", "parameters": {
        "type": "object", "properties": {"file_path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "Bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"},
                                         "description": {"type": "string"}}}}},
]

# Two parallel tool calls in the model's own format (chat_template.jinja of Qwen3.8-27B), cut into
# the steps a speculative decoder delivers. Step 3 holds the first call's end AND the newline that
# separates it from the second call — the shape that lost the brace.
_STEPS = [
    "<tool_call>\n<function=Read>\n<parameter=file_path>\n",
    "/tmp/a.txt\n</parameter>\n",
    "</function>\n</tool_call>\n<tool_call>\n<function=Bash>\n",
    "<parameter=command>\nls -la\n</parameter>\n<parameter=description>\nList files\n</parameter>\n",
    "</function>\n</tool_call>",
]
_EXPECTED = [("Read", {"file_path": "/tmp/a.txt"}),
             ("Bash", {"command": "ls -la", "description": "List files"})]


def test_the_engine_releases_a_steps_text_before_its_closing_fragment():
    """The cause, pinned: if this ordering ever changes upstream, this test says so."""
    lines = [json.loads(l[6:]) for l in engine_stream(_STEPS, _TOOLS) if l.startswith("data: {")]
    deltas = [c["choices"][0]["delta"] for c in lines if c["choices"]]
    text_at = next(i for i, d in enumerate(deltas) if d.get("content") == "\n")
    assert deltas[text_at + 1]["tool_calls"][0]["function"]["arguments"] == "}"


def test_every_tool_call_arrives_whole_when_a_step_ends_one_and_starts_the_next():
    blocks, _ = client(_STEPS, _TOOLS)
    assert tool_calls(blocks) == _EXPECTED


def test_the_text_between_the_calls_is_kept_in_model_order():
    """Nothing is dropped to fix it: the newline the model wrote between the calls still arrives,
    after the first call and before the second, exactly as it was generated."""
    blocks, _ = client(_STEPS, _TOOLS)
    assert [b[:2] for b in blocks] == [("tool_use", "Read"), ("text", "\n"), ("tool_use", "Bash")]


@pytest.mark.parametrize("cut", range(1, 40))
def test_no_step_boundary_loses_a_fragment(cut):
    """Same output, cut at every early boundary into two steps: every cut parses to the same calls."""
    text = "".join(_STEPS)
    blocks, _ = client([text[:cut * 4], text[cut * 4:]], _TOOLS)
    assert tool_calls(blocks) == _EXPECTED
