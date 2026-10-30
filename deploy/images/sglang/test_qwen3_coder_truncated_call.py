"""A tool call cut off by max_tokens must not reach /v1/messages as a complete call, and a long
string argument must stream while the model writes it.

Runs inside the SGLang image (`python -m pytest`), against the image's own code: the real
qwen3_coder detector, the real OpenAI stream builder (`_process_tool_call_stream` with the
terminal flush, then `_check_for_unstreamed_tool_args`, as `_generate_stream_content` calls them)
and the real Anthropic adapter. Upstream's adapter harness is imported, not copied.

The defect (2026-10-03, 27B SGLang + DFlash2, deep-research session 9ec2c985 rows 210-217): the
model thought for ~10k tokens, opened `<function=Workflow>\\n<parameter=script>` and wrote the
script for 172 s until the client's CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384 cut it. The detector emits
a parameter only once `</parameter>` arrives and its finish() hook was the base no-op, so nothing
of the arguments ever left: the client got tool_use `Workflow` with input `{}` and stop_reason
max_tokens. Claude Code 2.1.288 RUNS a well-formed call even under max_tokens ("Must provide
script, ..."), and the model concluded "The call went empty" and wrote the same script again.
Measured against Claude Code 2.1.288 on a local fake endpoint: input `{}` -> tool runs with no
arguments; unclosed JSON -> "could not be parsed as JSON. You sent ...: {"command": "echo cut-he
... truncated output", the tool does not run; closed JSON of a cut value -> the tool RUNS the cut
value. So a cut call goes out as the JSON written so far, never closed: what Anthropic's API
streams for a cut tool_use, and never a call the client would execute.

The same hold-until-`</parameter>` made a long `script`/`content` argument send NO event for
minutes (47 sessions hit the client's 180 s idle abort on SGLang; vLLM streams values and did
not). A string-typed parameter (per the tool's schema; unknown or untyped counts as string, as in
_convert_param_value) now streams as JSON-escaped fragments, holding back only what may still be
an end marker, the one trailing "\n" the one-shot parser strips, and a value that may still be
`null`. The fragments must add up to exactly what detect_and_parse returns for the same text.
"""

import json
import random
import zlib

import pytest

from fleet_harness import client, engine_stream, messages_events

from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: E402
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector  # noqa: E402

_TOOLS = [
    {"type": "function", "function": {"name": "Workflow", "parameters": {
        "type": "object", "properties": {"name": {"type": "string"}, "script": {"type": "string"},
                                         "runId": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "Bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"},
                                         "timeout": {"type": "integer"}}}}},
]

# A Workflow script with what the brief lists: `<` and `>`, quotes, backticks, a `</parameter`
# with no `>`, `<function=` text, JSON, and runs of newlines.
_SCRIPT = (
    "const agents = [`a<b`, 'c>d', \"e<=f\"];\n"
    "if (n < 3 && m > 1) { log(`<div class=\"x\">${n}</div>`); }\n\n\n"
    "// a literal </parameter with no closing angle, and <function=fake> text\n"
    "const cfg = {\"k\": [1, 2, {\"n\": \"v\"}], \"html\": \"<p>\\n</p>\"};\n"
    "for (const a of agents) {\n  await spawn(a, cfg);\n}\n\n"
    "return `done <${agents.length}>`;"
)


def _call(name, params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


# The reported call: `script` is the first parameter, so nothing of the arguments had gone out.
_REPORTED = _call("Workflow", [("script", _SCRIPT)])
_REPORTED_INPUT = {"script": _SCRIPT}
_WORKFLOW = _call("Workflow", [("name", "sweep"), ("script", _SCRIPT)])
_WORKFLOW_INPUT = {"name": "sweep", "script": _SCRIPT}


def _client(steps, finish):
    return client(steps, _TOOLS, finish)


def _tools(blocks):
    return [b[1:] for b in blocks if b[0] == "tool_use"]


def _steps(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


# ---------------------------------------------------------------- 1. the stream ends mid-call


def test_the_reported_cut_inside_the_script_is_not_an_empty_call():
    """Rows 210-217: name sent, script being written, max_tokens. The client must not get `{}`."""
    cut = _REPORTED.index("for (const a")
    blocks, stop = _client(_steps(_REPORTED[:cut], 4), "length")
    assert stop == "max_tokens"
    [(name, raw, parsed)] = _tools(blocks)
    assert name == "Workflow"
    assert parsed is None, f"a cut call reached the client as the valid call {raw!r}"


@pytest.mark.parametrize("cut_after", ["<parameter=script>", "<parameter=script>\n", "agents = [",
                                       "a literal </para", "</parameter", "return `done <"])
def test_a_cut_parameter_goes_out_as_the_json_written_so_far_and_never_closed(cut_after):
    """Anthropic's API streams a cut tool_use's input as far as it got; so does this. The client
    shows the model what it sent ("could not be parsed as JSON ... truncated output")."""
    start = _REPORTED.index("<parameter=script>") + len("<parameter=script>")
    cut = _REPORTED.index(cut_after, start - len(cut_after)) + len(cut_after)
    blocks, stop = _client(_steps(_REPORTED[:cut], 4), "length")
    [(name, raw, parsed)] = _tools(blocks)
    written = _REPORTED[start:cut]
    written = written[1:] if written.startswith("\n") else written
    assert (stop, parsed) == ("max_tokens", None)
    assert raw == '{"script": ' + json.dumps(written, ensure_ascii=False)[:-1]


@pytest.mark.parametrize("text,expected_raw", [
    ("<tool_call>\n<function=Workflow>\n", "{"),                       # before any parameter
    ("<tool_call>\n<function=Workflow>\n<parameter=scr", "{"),          # inside a parameter name
    ("<tool_call>\n<function=Workflow>\n<parameter=name>\nsweep\n</parameter>\n",
     '{"name": "sweep"'),                                                 # between parameters
    ("<tool_call>\n<function=Bash>\n<parameter=timeout>\n12", '{"timeout": "12'),  # a non-string
])
def test_every_cut_point_inside_a_call_leaves_json_that_does_not_parse(text, expected_raw):
    blocks, stop = _client(_steps(text, 4), "length")
    [(name, raw, parsed)] = _tools(blocks)
    assert (stop, raw, parsed) == ("max_tokens", expected_raw, None)


def test_a_cut_second_call_leaves_the_first_one_whole():
    second = _call("Bash", [("command", "ls -la")])
    text = _WORKFLOW + "\n" + second[:second.index("ls -") + 1]
    blocks, stop = _client(_steps(text, 4), "length")
    (n1, _, p1), (n2, raw2, p2) = _tools(blocks)
    assert (stop, n1, p1, n2, p2) == ("max_tokens", "Workflow", _WORKFLOW_INPUT, "Bash", None)
    assert raw2 == '{"command": "l'


def test_a_call_cut_after_its_function_end_is_complete():
    """`</function>` arrived: the call is whole even though `</tool_call>` did not."""
    text = _WORKFLOW[:_WORKFLOW.index("</function>") + len("</function>")]
    blocks, stop = _client(_steps(text, 4), "length")
    assert (stop, [(n, p) for n, _, p in _tools(blocks)]) == (
        "max_tokens", [("Workflow", _WORKFLOW_INPUT)])


@pytest.mark.parametrize("text,full", [(_REPORTED, _REPORTED_INPUT), (_WORKFLOW, _WORKFLOW_INPUT)],
                         ids=["script-only", "name-then-script"])
@pytest.mark.parametrize("cut", range(len(_WORKFLOW) + 1))
def test_no_cut_point_makes_a_call_the_client_would_run_with_other_arguments(cut, text, full):
    """The invariant, at every character: a cut call either parses to the call the model wrote
    in full, or does not parse at all. Never `{}`, never a shortened script that would run."""
    blocks, stop = _client(_steps(text[:cut], 4), "length")
    for name, raw, parsed in _tools(blocks):
        assert parsed in (None, full), f"cut {cut}: client would run {raw!r}"


def test_text_held_back_for_a_tag_that_never_came_is_released_at_the_end():
    """Outside a call a trailing `<` (or `</`, `<tool_c`) waits for a tag; the stream ending means
    it never comes, and the base finish() contract says that text is released, not dropped."""
    for tail in ("<", "</", "<tool_c"):
        text = "Compare the two: a " + tail
        blocks, stop = _client(_steps(text, 4), "stop")
        assert (blocks, stop) == ([("text", text)], "end_turn"), tail


# --------------------------------------------- 2. values with tag-like text, JSON and newlines


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 16, len(_WORKFLOW)])
def test_a_value_with_angles_quotes_json_and_newline_runs_arrives_exactly(size):
    blocks, stop = _client(_steps(_WORKFLOW, size), "stop")
    assert stop == "tool_use"
    assert [(n, p) for n, _, p in _tools(blocks)] == [("Workflow", _WORKFLOW_INPUT)]


@pytest.mark.parametrize("value", [
    "x <parameter=y> z", "x </function> z", "x </parameter> z"])
def test_a_value_holding_the_formats_own_markers_never_runs_as_another_call(value):
    """The format has no escaping: a value carrying `<parameter=`, `</function>` or `</parameter>`
    ends there, in the one-shot parser too. Streaming may give up instead (a value holding
    `</function>` closes the call twice: `{"script": "x "}}`), but it never hands the client a
    runnable call the one-shot parser would not have produced."""
    text = _call("Workflow", [("script", value)])
    tools = [Tool.model_validate(t) for t in _TOOLS]
    one_shot = json.loads(Qwen3CoderDetector().detect_and_parse(text, tools).calls[0].parameters)
    assert one_shot["script"] == "x "
    for size in (1, 4, len(text)):
        blocks, _ = _client(_steps(text, size), "stop")
        assert [p in (None, one_shot) for _, _, p in _tools(blocks)] == [True], size


# --------------------------------------------- 3. `</parameter>` split at every step boundary


@pytest.mark.parametrize("cut", range(len(_WORKFLOW) + 1))
def test_every_two_step_split_parses_to_the_same_call(cut):
    blocks, stop = _client([_WORKFLOW[:cut], _WORKFLOW[cut:]], "stop")
    assert [(n, p) for n, _, p in _tools(blocks)] == [("Workflow", _WORKFLOW_INPUT)]


@pytest.mark.parametrize("k", range(1, len("</parameter>")))
def test_the_end_marker_split_three_ways_around_each_inner_boundary(k):
    end = _WORKFLOW.index("</parameter>", _WORKFLOW.index("<parameter=script>"))
    steps = [_WORKFLOW[:end + k], _WORKFLOW[end + k:end + k + 1], _WORKFLOW[end + k + 1:]]
    blocks, _ = _client([s for s in steps if s], "stop")
    assert [(n, p) for n, _, p in _tools(blocks)] == [("Workflow", _WORKFLOW_INPUT)]


# ------------------------------------------- 4. string values stream, and add up to one-shot

_MIXED_TOOL = {"type": "function", "function": {"name": "Mixed", "parameters": {
    "type": "object", "properties": {
        "path": {"type": "string"}, "count": {"type": "integer"}, "ratio": {"type": "number"},
        "force": {"type": "boolean"}, "opts": {"type": "object"}, "tags": {"type": "array"},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}}
_TYPED = [Tool.model_validate(t) for t in _TOOLS + [_MIXED_TOOL]]

_VALUES = [
    "", "\n", "\n\n", "x", "null", "NULL", "Null\n", "nul", "nullable", "none",
    'say "hi" and \\"escaped\\"', "C:\\path\\to\\file\\", "\\", "\\n is not a newline",
    "unicode: zażółć 東京 🦀 é\u0301 \u2028 \u00a0", "tab\tcr\rnul\x00us\x1f",
    "a<b", "<", "<<", "</", "</p", "</parameter", "</parameter >", "<parameter", "</function",
    "<function=x>", "<tool_call>", "x\n", "\nx", "\n\nx\n\n", "line1\n\n\nline3\n", "ends with <",
    '{"json": [1, 2, {"k": "v"}], "s": "<p>\\n</p>"}', "`backticks` ${template} 'q'", _SCRIPT,
]


def _texts(value):
    return {
        "string": _call("Workflow", [("script", value)]),
        "mixed": _call("Mixed", [("count", "3"), ("path", value), ("opts", '{"a": [1, "<b>"]}'),
                                  ("mode", "a"), ("maybe", value), ("force", "true"),
                                  ("undeclared", value), ("ratio", "0.5"), ("tags", '["x"]')]),
        "parallel": _call("Workflow", [("name", value), ("script", value)]) + "\n"
                    + _call("Bash", [("command", value), ("timeout", "30")]),
    }


def _one_shot(text):
    return [(c.name, c.parameters) for c in Qwen3CoderDetector().detect_and_parse(text, _TYPED).calls]


def _streamed(chunks):
    """(name, concatenated arguments) per call, from the detector fed these chunks, then finish()."""
    detector, calls = Qwen3CoderDetector(), {}
    results = [detector.parse_streaming_increment(c, _TYPED) for c in chunks]
    results.append(detector.finish(_TYPED))
    for result in results:
        for item in result.calls:
            name, args = calls.get(item.tool_index, (None, ""))
            calls[item.tool_index] = (item.name or name, args + (item.parameters or ""))
    return [calls[i] for i in sorted(calls)]


def _splits(text, seed):
    rng = random.Random(seed)
    yield [text]
    for size in (1, 2, 3, 4, 5, 7, 13):
        yield _steps(text, size)
    for _ in range(12):
        cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(1, 40))))
        yield [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]


@pytest.mark.parametrize("shape", ["string", "mixed", "parallel"])
@pytest.mark.parametrize("value", _VALUES, ids=[repr(v)[:24] for v in _VALUES])
def test_streamed_arguments_equal_the_one_shot_parse_exactly(value, shape):
    text = _texts(value)[shape]
    expected = _one_shot(text)
    assert expected and all(json.loads(args) is not None for _, args in expected)
    for n, chunks in enumerate(_splits(text, seed=zlib.crc32(f"{shape}:{value}".encode()))):
        assert _streamed(chunks) == expected, f"split #{n}: {chunks!r}"


@pytest.mark.parametrize("cut", range(len(_texts(_SCRIPT)["mixed"]) + 1))
def test_every_two_step_split_of_a_mixed_call_equals_the_one_shot_parse(cut):
    text = _texts(_SCRIPT)["mixed"]
    assert _streamed([text[:cut], text[cut:]]) == _one_shot(text)


def test_a_long_string_argument_goes_out_while_it_is_written():
    """The idle-abort cause: 20 kB of script must not wait for `</parameter>`. After every step
    the client holds the value written so far, short of at most a possible end marker."""
    value = _SCRIPT * 40
    text = _call("Workflow", [("script", value)])
    start = text.index("<parameter=script>\n") + len("<parameter=script>\n")
    detector, args = Qwen3CoderDetector(), ""
    for chunk_end in range(4, start + len(value), 4):
        for item in detector.parse_streaming_increment(text[chunk_end - 4:chunk_end], _TYPED).calls:
            args += item.parameters or ""
        if chunk_end > start + 1:
            sent = json.loads(args + '"}')["script"]
            assert value.startswith(sent)
            assert len(sent) >= chunk_end - start - len("</parameter>"), chunk_end


def test_the_client_sees_the_script_grow_and_receives_the_one_shot_arguments():
    text = _call("Workflow", [("script", _SCRIPT * 5)])
    events = messages_events(engine_stream(_steps(text, 4), _TOOLS))
    deltas = [e["delta"]["partial_json"] for e in events
              if e["type"] == "content_block_delta" and "partial_json" in e["delta"]]
    assert len(deltas) > len(text) // 8
    assert "".join(deltas) == _one_shot(text)[0][1]
