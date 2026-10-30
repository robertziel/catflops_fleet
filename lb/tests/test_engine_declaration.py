"""ONE DECLARATION PER ENGINE (fleet.ENGINES; operator 2026-10-23: "build special engines for STT, it's no longer vllm
only server"). Each engine says, for the task it serves: the settings it reads (and the node variable each reaches),
the panel tiles it publishes, its memory model and whether it decodes a whole file in batches. /lb/status carries the
declaration; the panel renders from it — a new engine is a new class, never a panel branch.

A knob an engine does not read does not exist for it: not in its settings, not in the wish the panel shows, not in the
node env rendered for it — and one it shows reaches the node (`env_lines`, the compose override maps the variable).
"""
from __future__ import annotations

import pytest

from tests.test_stt import FW_IMAGE, STT, _env
from vllm_lb.fleet import ENGINES, MOVABLE, engine_block, node_env, parse_fleet, validate_launch

FW = "faster-whisper-large-v3-turbo"


def keys(engine: str, task: str) -> tuple[str, ...]:
    return tuple(s["key"] for s in ENGINES[engine].view(task)["settings"])


def test_faster_whisper_reads_seats_queue_and_compute_type_only():
    assert keys("faster-whisper", "transcription") == ("model", "max_num_seqs", "max_queue", "compute_type")


def test_vllm_for_speech_keeps_its_encoder_decoder_knobs_and_drops_the_chat_ones():
    speech, chat = keys("vllm", "transcription"), keys("vllm", "chat")
    assert speech == ("model", "max_num_seqs", "max_model_len", "gpu_memory_utilization", "max_num_batched_tokens",
                      "kv_cache_memory")
    assert set(chat) - set(speech) == {"kv_offload_ram_gib", "kv_cache_dtype"}
    assert set(chat) <= {"model", *MOVABLE}


def test_each_setting_names_the_node_variable_it_reaches():
    fw = {s["key"]: s for s in ENGINES["faster-whisper"].view("transcription")["settings"]}
    assert fw["max_num_seqs"]["env"] == "EDGE_NODE_MAX_NUM_SEQS"
    assert fw["max_queue"]["env"] == "EDGE_NODE_MAX_QUEUE" and fw["max_queue"]["unset"]
    assert "float32" not in fw["compute_type"]["options"], "more memory than the measured engine_memory"
    assert {"float16", "int8_float16"} <= set(fw["compute_type"]["options"])


def test_the_tiles_are_the_tasks():
    assert ENGINES["faster-whisper"].view("transcription")["tiles"] == ["speech"]
    assert ENGINES["vllm"].view("transcription")["tiles"] == ["speech"]
    assert "generated" in ENGINES["vllm"].view("chat")["tiles"]


def test_whole_file_batching_is_declared():
    # a one-file lane on the sequential decoder: faster-whisper's batched pipeline lost text on real files
    assert ENGINES["faster-whisper"].view("transcription")["whole_file"] == "lane"
    assert ENGINES["vllm"].view("transcription")["whole_file"] == "engine"


def test_the_faster_whisper_node_env_carries_its_knobs_and_none_of_vllms():
    fleet = parse_fleet(STT, "fleet.yml")
    env = _env(node_env(fleet, "stt", {"model": FW, "max_queue": 8, "compute_type": "int8_float16"}))
    assert (env["EDGE_NODE_MAX_NUM_SEQS"], env["EDGE_NODE_MAX_QUEUE"], env["EDGE_NODE_COMPUTE_TYPE"]) == \
        ("4", "8", "int8_float16")
    for vllm_only in ("EDGE_NODE_MAX_MODEL_LEN", "EDGE_NODE_GPU_MEM_UTIL", "EDGE_NODE_SERVED_NAME",
                      "EDGE_NODE_MAX_NUM_BATCHED_TOKENS", "EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_CACHE_DTYPE",
                      "EDGE_NODE_TOOL_CALL_PARSER", "EDGE_NODE_REASONING_PARSER", "EDGE_NODE_KV_OFFLOAD_RAM_GIB"):
        assert vllm_only not in env, vllm_only
    unset = _env(node_env(fleet, "stt", {"model": FW}))
    assert unset["EDGE_NODE_MAX_QUEUE"] == "" and unset["EDGE_NODE_COMPUTE_TYPE"] == "", "the service's own"


def test_the_vllm_node_env_carries_none_of_faster_whispers():
    env = _env(node_env(parse_fleet(STT, "fleet.yml"), "stt"))
    assert "EDGE_NODE_MAX_QUEUE" not in env and "EDGE_NODE_COMPUTE_TYPE" not in env
    assert env["EDGE_NODE_MAX_MODEL_LEN"] == "448"


def test_a_knob_the_selected_engine_does_not_read_is_refused():
    stt = parse_fleet(STT, "fleet.yml").clusters["stt"]
    with pytest.raises(ValueError, match="takes no max_queue"):
        validate_launch(stt, {"max_queue": 4})
    with pytest.raises(ValueError, match="compute_type"):
        validate_launch(stt, {"model": FW, "compute_type": "float32"})
    with pytest.raises(ValueError, match="max_queue"):
        validate_launch(stt, {"model": FW, "max_queue": 0})
    assert validate_launch(stt, {"model": FW, "max_queue": 8}).max_queue == 8


def test_the_fleet_launch_block_may_name_them():
    text = STT.replace("      max_num_seqs: 4\n", "      max_num_seqs: 4\n      max_queue: 4\n")
    with pytest.raises(ValueError, match="takes no max_queue"):
        parse_fleet(text, "fleet.yml")          # the boot model is vLLM's
    fw = text.replace("model: whisper-large-v3-turbo\n", f"model: {FW}\n").replace(
        "choices: [whisper-large-v3-turbo, faster-whisper-large-v3-turbo]", f"choices: [{FW}]").replace(
        f'{{vllm: "vllm-audio@sha256:{"3" * 64}", faster-whisper: "{FW_IMAGE}"}}', f'"{FW_IMAGE}"')
    assert parse_fleet(fw, "fleet.yml").clusters["stt"].launch.max_queue == 4


def test_the_engine_block_of_a_cluster():
    stt = parse_fleet(STT, "fleet.yml").clusters["stt"]
    block = engine_block(stt, FW)
    assert block["task"] == "transcription" and block["tiles"] == ["speech"]
    assert block["engine"]["name"] == "faster-whisper"
    assert set(block["engines"]) == {"vllm", "faster-whisper"}
    assert block["hardware"] == "AMD Radeon 8060S (Ryzen AI Max+ 395) · rocm"
    assert engine_block(stt, None)["engine"]["name"] == "vllm", "the boot default's engine"


def test_the_panels_declaration_fixture_is_the_declaration():
    """tests/fixtures/engine-declarations.json feeds the node tests the declarations /lb/status sends: never stale."""
    import json
    import pathlib
    got = json.loads((pathlib.Path(__file__).parent / "fixtures" / "engine-declarations.json").read_text())
    assert got == {f"{e}/{t}": ENGINES[e].view(t) for e in ENGINES for t in ("chat", "transcription")}
