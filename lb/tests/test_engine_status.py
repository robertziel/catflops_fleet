"""/lb/status says, per cluster, what its server IS — its task, the engine of the model it is set to serve (fleet.ENGINES'
declaration: settings, tiles), every engine its choices run, and its card as the fleet's hardware type names it — and
a local node's wish carries only what that engine reads. A `run_at_boot` node nobody has asked anything of is STARTED:
its boot wish is written once an applier listens (live 2026-10-23: after an LB redeploy the stt node sat
`desired.mode: cluster` with no intent until a manual POST)."""
from __future__ import annotations

from tests.test_cluster_local import FLEET, LIVE, Local, get, post, served  # noqa: F401  (fixture)
from tests.test_cluster_one_host import DOWN, NODE
from tests.test_stt import STT
from vllm_lb import node_control as nc

FW = "faster-whisper-large-v3-turbo"
#: The speech cluster of tests/test_stt.py on the box, card and URL the Local harness drives.
SPEECH = (STT.replace("apu:", "node-a:").replace("apu-host", "node-a-host").replace("apu-gpu", "node-a-gpu")
          .replace("{box: apu}", "{box: node-a}").replace("http://stt-node:8000", NODE).replace("  stt:", "  node:"))


def status(node, served) -> dict:
    proxy = served(node)
    node.tick()
    return get(proxy, "/lb/status")[1]["clusters"]["node"]


def test_a_chat_node_declares_vllm_for_chat(tmp_path, served):
    c = status(Local(tmp_path, LIVE), served)
    assert c["task"] == "chat" and c["engine"]["name"] == "vllm"
    assert "generated" in c["tiles"] and "kv_cache_dtype" in [s["key"] for s in c["engine"]["settings"]]
    assert c["hardware"].endswith("· nvidia")


def test_a_speech_node_declares_its_selected_engine_and_both_choices(tmp_path, served):
    node = Local(tmp_path, LIVE, fleet=SPEECH)
    proxy = served(node)
    node.tick()
    c = get(proxy, "/lb/status")[1]["clusters"]["node"]
    assert (c["task"], c["engine"]["name"], c["tiles"]) == ("transcription", "vllm", ["speech"])
    assert set(c["engines"]) == {"vllm", "faster-whisper"}
    assert c["hardware"] == "AMD Radeon 8060S (Ryzen AI Max+ 395) · rocm", "the fleet's card, never the engine's 'cuda'"
    assert {ch["ref"]: ch["engine"] for ch in c["choices"]}[FW] == "faster-whisper"
    assert post(proxy, {"enabled": True, "model": FW, "max_queue": 8})[0] == 200
    node.tick()
    c = get(proxy, "/lb/status")[1]["clusters"]["node"]
    assert c["engine"]["name"] == "faster-whisper"
    assert {k for k in c["desired"] if k not in ("mode", "id", "force", "model", "model_choice", "uncensored")} == \
        {"max_num_seqs", "max_queue", "compute_type"}, "no context window, no memory fraction: not its knobs"
    assert c["desired"]["max_queue"] == 8


def test_a_boot_node_never_asked_is_started_once_an_applier_listens(tmp_path):
    node = Local(tmp_path, DOWN, listening=False)
    node.tick()
    assert nc.IntentStore(node.cfg.cluster_state_path).read() is None, "nobody to apply it: nothing written"
    node.beat()
    node.tick()
    wish = nc.IntentStore(node.cfg.cluster_state_path).read()
    assert wish is not None and wish["state"] == nc.UP and wish["launch"] == {}
    assert node.released() is not None and node.released()["state"] == nc.UP
    node.tick()
    assert nc.IntentStore(node.cfg.cluster_state_path).read()["id"] == wish["id"], "written once"


def test_a_node_already_answering_is_left_as_it_runs(tmp_path):
    node = Local(tmp_path, LIVE)
    node.tick()
    assert nc.IntentStore(node.cfg.cluster_state_path).read() is None and node.released() is None


def test_a_node_not_run_at_boot_is_left_alone(tmp_path):
    node = Local(tmp_path, DOWN, fleet=FLEET.replace("    control: local\n", "    control: local\n    run_at_boot: false\n"))
    node.tick()
    assert nc.IntentStore(node.cfg.cluster_state_path).read() is None and node.released() is None


def test_switching_engines_carries_no_knob_of_the_other_engine(tmp_path, served):
    """The stored wish's faster-whisper queue is not a vLLM setting: a Start of the vLLM choice keeps none of it — while
    a body that NAMES a knob its engine does not read is still refused."""
    node = Local(tmp_path, LIVE, fleet=SPEECH)
    proxy = served(node)
    assert post(proxy, {"enabled": True, "model": FW, "max_queue": 8, "compute_type": "int8"})[0] == 200
    code, body = post(proxy, {"enabled": True, "model": "whisper-large-v3-turbo"})
    assert code == 200, body
    assert "max_queue" not in body["desired"]["launch"] and "compute_type" not in body["desired"]["launch"]
    code, body = post(proxy, {"enabled": True, "max_queue": 4})
    assert code == 400 and "takes no max_queue" in body["error"]["message"]


def test_the_estimate_prices_a_faster_whisper_selection_with_its_own_knobs(tmp_path, served):
    node = Local(tmp_path, LIVE, fleet=SPEECH)
    proxy = served(node)
    node.tick()
    code, body = get(proxy, f"/lb/cluster/estimate?model={FW}&max_num_seqs=4&max_queue=8&compute_type=int8")
    assert code == 200 and not body.get("refused"), body
    assert "GiB measured whole engine @" in body["fit"]["text"] and "max_model_len" not in str(body["fit"]), \
        "its own memory model: the measured engine, never vLLM's fraction of the card"
    code, body = get(proxy, f"/lb/cluster/estimate?model={FW}&compute_type=float32")
    assert code == 200 and "compute_type" in body["refused"]


def test_what_the_fleet_declares_a_faster_whisper_node_runs_is_its_engines_settings(tmp_path):
    node = Local(tmp_path, LIVE, fleet=SPEECH.replace("    model: whisper-large-v3-turbo\n", f"    model: {FW}\n"))
    launch = node.tick()["declared"]["launch"]
    assert launch["engine"] == "faster-whisper" and launch["max_num_seqs"] == 4
    assert not {"max_model_len", "gpu_memory_utilization"} & set(launch), launch
