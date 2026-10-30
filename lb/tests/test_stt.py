"""SPEECH-TO-TEXT NODES — a model's TASK is catalog data, a node's ENGINE is its launch choice's (operator, 2026-10-23).

  * TASK ({value, source}, every row): the OpenAI API family a model answers — `chat` (/v1/chat/completions,
    /v1/completions, /v1/messages, …) or `transcription` (/v1/audio/transcriptions, /v1/audio/translations). The LB
    routes BY IT: an audio path goes only to a live backend whose model transcribes, whatever model name the client
    sends (OpenAI clients send `whisper-1`); a chat path never reaches a transcription backend. The multipart `model` field is
    rewritten to the serving backend's own name, every other byte kept. No transcription backend live: 503, said so.
  * ENGINE (`node_engine`, a row a fleet node launches): the server that loads the checkpoint — `vllm` (a HF Whisper
    checkpoint, vLLM's own /v1/audio/transcriptions) or `faster-whisper` (a CTranslate2 checkpoint). One class per
    family in fleet.py (`ENGINES`): its image (the launch block's `image`, one per engine), its fit, its node-env.
"""
from __future__ import annotations

import copy
import http.client
import json
from dataclasses import replace

import pytest

from tests.fleet_files import PAIR, write
from vllm_lb import catalog as catalog_mod
from vllm_lb.catalog import document, NODE_ENGINES, TASKS, ModelCatalog
from vllm_lb.config import Config
from vllm_lb.fleet import ENGINES, node_env, parse_fleet, validate_launch
from vllm_lb.router import TRANSCRIPTION_PATHS, choose, serving, task_of_path
from vllm_lb.state import CLUSTER, Backend, Snapshot

VLLM_IMAGE = "vllm-audio@sha256:" + "3" * 64
FW_IMAGE = "sha256:" + "4" * 64           # a local build, pinned by its image id
STT = f"""
boxes:
  apu:
    hostname: apu-host
    label: apu-gpu
    platform: linux/amd64
    gpu_ram_max_gib: 32
    node_limits: {{memory_gib: 24, cpus: 16, pids: 4096, nice: 10}}
    cards:
      - {{index: 0, type: radeon8060s, serves: true}}
clusters:
  stt:
    control: local
    model: whisper-large-v3-turbo
    url: http://stt-node:8000
    launch:
      image: {{vllm: "{VLLM_IMAGE}", faster-whisper: "{FW_IMAGE}"}}
      max_model_len: 448
      max_num_seqs: 4
      gpu_memory_utilization: 0.05
      choices: [whisper-large-v3-turbo, faster-whisper-large-v3-turbo]
    ranks:
      - {{box: apu}}
backends: {{}}
"""


def _env(lines) -> dict:
    return dict(line.split("=", 1) for line in lines)


# ── the catalog: task and engine are a row's data ──────────────────────────────────────────────────────────────────

def test_every_shipped_row_states_its_task_with_a_source():
    doc = document()
    for row in doc["models"]:
        fact = row.get("task")
        assert isinstance(fact, dict) and fact.get("value") in TASKS and fact.get("source", "").strip(), row["id"]
    cat = ModelCatalog.load()
    assert cat.get("mai-ui-2b").task == "chat"
    assert cat.get("whisper-large-v3-turbo").task == "transcription"
    assert cat.get("faster-whisper-large-v3-turbo").task == "transcription"


def test_a_row_without_a_task_is_refused():
    doc = document()
    del doc["models"][0]["task"]
    with pytest.raises(ValueError, match="task"):
        ModelCatalog.from_dict(doc, roots=["/models"])


def test_a_row_the_lbs_own_launchers_start_is_a_chat_model():
    """tp-rank and broker_parrot's classic servers run chat engines: a row with a launch profile that says
    transcription is a document error."""
    doc = document()
    row = next(r for r in doc["models"] if r["launch"])
    row["task"] = {"value": "transcription", "source": "test"}
    with pytest.raises(ValueError, match="transcription"):
        ModelCatalog.from_dict(doc, roots=["/models"])


def test_the_whisper_rows_name_their_engine_and_a_fleet_only_window():
    cat = ModelCatalog.load()
    v, fw = cat.get("whisper-large-v3-turbo"), cat.get("faster-whisper-large-v3-turbo")
    assert (v.node_engine, fw.node_engine) == ("vllm", "faster-whisper")
    assert v.limits.max_model_len == 448, "Whisper's decoder window (max_target_positions), under tp-rank's floor"
    assert fw.engine_memory is not None and fw.engine_memory.source.strip()
    assert v.memory.single_card_kv_blocks is not None and v.memory.single_card_overhead is not None


def test_an_unknown_node_engine_is_refused():
    doc = document()
    row = next(r for r in doc["models"] if r["id"] == "whisper-large-v3-turbo")
    row["node_engine"] = "whisper.cpp"
    with pytest.raises(ValueError, match="node_engine"):
        ModelCatalog.from_dict(doc, roots=["/models"])


def test_the_fleets_engine_classes_are_the_catalogs_engine_names():
    assert tuple(ENGINES) == NODE_ENGINES


# ── the fleet: one cluster, two engines ─────────────────────────────────────────────────────────────────────────────

def test_a_transcription_cluster_parses_with_one_image_per_engine():
    f = parse_fleet(STT, "fleet.yml")
    stt = f.clusters["stt"]
    assert stt.task == "transcription" and stt.unlaunchable == ""
    assert stt.launch.image == VLLM_IMAGE, "the boot model's engine's image"


def test_a_chat_cluster_needs_its_tool_parser_and_a_transcription_one_takes_none():
    with pytest.raises(ValueError, match="tool_call_parser"):
        parse_fleet(STT.replace("      max_model_len: 448", "      tool_call_parser: hermes\n      max_model_len: 448"),
                    "fleet.yml")
    with pytest.raises(ValueError, match="tool_call_parser"):
        parse_fleet(STT.replace("model: whisper-large-v3-turbo", "model: mai-ui-2b").replace(
            "[whisper-large-v3-turbo, faster-whisper-large-v3-turbo]", "[mai-ui-2b]").replace(
            f'{{vllm: "{VLLM_IMAGE}", faster-whisper: "{FW_IMAGE}"}}', VLLM_IMAGE).replace(
            "max_model_len: 448", "max_model_len: 32768").replace("0.05", "0.10"), "fleet.yml")


def test_choices_of_two_tasks_are_refused():
    with pytest.raises(ValueError, match="task"):
        parse_fleet(STT.replace("faster-whisper-large-v3-turbo]", "faster-whisper-large-v3-turbo, mai-ui-2b]"),
                    "fleet.yml")


@pytest.mark.parametrize("image, why", [
    (f'{{vllm: "{VLLM_IMAGE}"}}', "faster-whisper"),                                      # an engine with no image
    (f'{{vllm: "{VLLM_IMAGE}", faster-whisper: "{FW_IMAGE}", sglang: "{FW_IMAGE}"}}', "sglang"),   # an extra one
    (f'"{VLLM_IMAGE}"', "faster-whisper"),                    # one image for two engines
    (f'{{vllm: "{VLLM_IMAGE}", faster-whisper: "vexa-stt-rocm:dev"}}', "digest"),         # not pinned
])
def test_the_images_name_exactly_the_choices_engines_each_pinned(image, why):
    text = STT.replace(f'{{vllm: "{VLLM_IMAGE}", faster-whisper: "{FW_IMAGE}"}}', image)
    with pytest.raises(ValueError, match=why):
        parse_fleet(text, "fleet.yml")


def test_node_env_renders_the_vllm_engine_for_the_boot_model():
    env = _env(node_env(parse_fleet(STT, "fleet.yml"), "stt"))
    assert env["EDGE_NODE_ENGINE"] == "vllm" and env["EDGE_NODE_IMAGE"] == VLLM_IMAGE
    assert env["EDGE_NODE_MODEL"] == "whisper-large-v3-turbo"
    assert env["EDGE_NODE_SERVED_NAME"] == "openai/whisper-large-v3-turbo"
    assert env["EDGE_NODE_TOOL_CALL_PARSER"] == "", "no tool parser: vLLM gets no chat-only tool flags"
    assert env["EDGE_NODE_MAX_MODEL_LEN"] == "448" and env["EDGE_NODE_MFU_PATCH"] == "off"


def test_an_intent_moves_the_node_to_the_other_engine_and_its_image():
    fleet = parse_fleet(STT, "fleet.yml")
    env = _env(node_env(fleet, "stt", {"model": "faster-whisper-large-v3-turbo"}))
    assert env["EDGE_NODE_ENGINE"] == "faster-whisper" and env["EDGE_NODE_IMAGE"] == FW_IMAGE
    assert env["EDGE_NODE_MODEL"] == "faster-whisper-large-v3-turbo"
    assert "EDGE_NODE_SERVED_NAME" not in env, "vLLM's --served-model-name: the service names its model itself"
    assert validate_launch(fleet.clusters["stt"], {"model": "faster-whisper-large-v3-turbo"}).image == FW_IMAGE


def test_the_faster_whisper_fit_is_its_measured_engine_memory():
    """A CTranslate2 engine sizes no pool from a fraction: its fit is the whole engine's measured memory, at no more
    concurrent requests than it was measured at, within the box's bound."""
    fleet = parse_fleet(STT, "fleet.yml")
    seats = ModelCatalog.load().get("faster-whisper-large-v3-turbo").engine_memory.max_num_seqs
    with pytest.raises(ValueError, match=rf"1\.\.{seats}|measured"):
        validate_launch(fleet.clusters["stt"], {"model": "faster-whisper-large-v3-turbo", "max_num_seqs": seats + 1})
    stt = fleet.clusters["stt"]
    fw = next(m for m in stt.choices if m.spec.node_engine == "faster-whisper")
    launch = validate_launch(stt, {"model": fw.ref})
    with pytest.raises(ValueError, match="gpu_ram_max_gib 1"):
        ENGINES["faster-whisper"].fit("w", fw, stt.serving_cards[0].type, launch,
                                      box=replace(stt.boxes[0], gpu_ram_max_gib=1))
    with pytest.raises(ValueError, match="kv_cache_memory"):
        ENGINES["faster-whisper"].fit("w", fw, stt.serving_cards[0].type, replace(launch, kv_cache_memory=1 << 30),
                                      box=stt.boxes[0])


def test_the_nemo_engine_is_the_services_but_for_ctranslate2s_compute_type():
    """Parakeet runs behind the same transcription-server contract (engine nemo): its seats and queue reach the node, its fit
    is its measured engine memory; CTranslate2's compute type is not offered, so a launch naming one is refused."""
    fleet = parse_fleet(STT.replace("model: whisper-large-v3-turbo", "model: parakeet-tdt-0.6b-v3")
                        .replace(f'{{vllm: "{VLLM_IMAGE}", faster-whisper: "{FW_IMAGE}"}}', f'"{FW_IMAGE}"')
                        .replace("max_model_len: 448", "max_model_len: 1")
                        .replace("[whisper-large-v3-turbo, faster-whisper-large-v3-turbo]", "[parakeet-tdt-0.6b-v3]"),
                        "fleet.yml")
    nemo = ENGINES["nemo"]
    assert [s.key for s in nemo.settings("transcription")] == ["model", "max_num_seqs", "max_queue"]
    env = _env(node_env(fleet, "stt", {"model": "parakeet-tdt-0.6b-v3", "max_queue": 8}))
    assert (env["EDGE_NODE_ENGINE"], env["EDGE_NODE_MODEL"], env["EDGE_NODE_MAX_QUEUE"]) == (
        "nemo", "parakeet-tdt-0.6b-v3", "8")
    assert "EDGE_NODE_COMPUTE_TYPE" not in env
    with pytest.raises(ValueError, match="takes no compute_type"):
        validate_launch(fleet.clusters["stt"], {"model": "parakeet-tdt-0.6b-v3", "compute_type": "int8"})


def test_a_chat_nodes_env_names_the_vllm_engine_and_its_tool_parser():
    from tests.test_node_guards import APU
    env = _env(node_env(parse_fleet(APU.replace(", qwen3.8-27b-nvidia-nvfp4]", "]"), "fleet.yml"), "gui", {}))
    assert env["EDGE_NODE_ENGINE"] == "vllm" and env["EDGE_NODE_TOOL_CALL_PARSER"] == "hermes"


# ── routing by task ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_audio_paths_are_the_transcription_task():
    for path in TRANSCRIPTION_PATHS:
        assert task_of_path(path) == "transcription"
    assert task_of_path("/v1/audio/transcriptions?x=1") == "transcription"
    for path in ("/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/embeddings"):
        assert task_of_path(path) == "chat"


PAIR_STT = PAIR.replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n    default: true\n").replace(
    "  gateway:\n    hostname: gateway-host\n    platform: linux/amd64\n    cards:\n"
    "      - {index: 0, type: radeon8060s, serves: false}\n",
    "  gateway:\n    hostname: gateway-host\n    label: gateway-gpu\n    platform: linux/amd64\n"
    "    gpu_ram_max_gib: 32\n    node_limits: {memory_gib: 24, cpus: 16, pids: 4096, nice: 10}\n    cards:\n"
    "      - {index: 0, type: radeon8060s, serves: true}\n").replace("backends:", STT.split("clusters:\n", 1)[1].split(
    "backends:", 1)[0].replace("    control: local\n", "    control: local\n    run_at_boot: true\n").replace(
    "{box: apu}", "{box: gateway}") + "backends:")
HELD = {"configured": True, "held": True, "state": "running", "desired": {"mode": "cluster"}, "max_depth": 4}


def _cfg(tmp_path) -> Config:
    return replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PAIR_STT),
                                    "LB_CLUSTER_STATE": str(tmp_path / "c.json")}), broker_enabled=False)


def _snap(*backends) -> Snapshot:
    return Snapshot(now=100.0, backends=list(backends), leases=[], last_request_at=99.0, last_pressure_at=None,
                    db_ok=True, extra={"cluster": HELD, "clusters": {"pair": HELD, "stt": HELD}})


PAIR_HEAD = Backend(kind=CLUSTER, url="http://pair:8100", live=True, served_models=("Qwen/Qwen3.8-27B",),
                    depth_known=True, cluster="pair")
STT_HEAD = Backend(kind=CLUSTER, url="http://stt:8000", live=True, served_models=("openai/whisper-large-v3-turbo",),
                   depth_known=True, cluster="stt")


def test_a_transcription_goes_to_the_transcription_node_whatever_model_it_names(tmp_path):
    cfg = _cfg(tmp_path)
    s = _snap(PAIR_HEAD, STT_HEAD)
    for model in ("whisper-1", None, "Qwen/Qwen3.8-27B"):
        assert choose(s, cfg, model=model, task="transcription").url == STT_HEAD.url, model
    assert choose(_snap(PAIR_HEAD), cfg, task="transcription") is None, "never a chat cluster"


#: A second speech cluster: Parakeet on the pair's second box, started by its own deployment (control none).
SPEECH = """  speech:
    control: none
    run_at_boot: true
    model: parakeet-tdt-0.6b-v3
    url: http://192.0.2.11:18000
    ranks:
      - {box: node-2}
"""
PARAKEET = "/models/parakeet-tdt-0.6b-v3"
SPEECH_HEAD = Backend(kind=CLUSTER, url="http://192.0.2.11:18000", live=True, served_models=(PARAKEET,),
                      depth_known=True, cluster="speech")


def _two_speech(tmp_path, stt_default: bool = True) -> Config:
    text = PAIR_STT.replace("    control: local\n    run_at_boot: true\n",
                            "    control: local\n    run_at_boot: true\n" + ("    default: true\n" if stt_default else "")
                            ).replace("backends:", SPEECH + "backends:")
    return replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                   broker_enabled=False)


def test_two_speech_clusters_a_loose_name_goes_to_the_default_a_served_one_to_its_server(tmp_path):
    from vllm_lb.router import exact_models
    cfg = _two_speech(tmp_path)
    assert (cfg.default_cluster, cfg.default_cluster_for("transcription")) == ("pair", "stt")
    s = _snap(PAIR_HEAD, STT_HEAD, SPEECH_HEAD)
    for model in ("whisper-1", None):
        assert serving(s, cfg, model, task="transcription") == [STT_HEAD], model
    exact = exact_models(cfg, ModelCatalog.load(), "transcription")
    assert PARAKEET in exact and "whisper-1" not in exact
    assert serving(s, cfg, PARAKEET, exact=True, task="transcription") == [SPEECH_HEAD]
    assert serving(_snap(PAIR_HEAD, STT_HEAD), cfg, PARAKEET, exact=True, task="transcription") == [], \
        "the named model's server is down: nothing, never the default's whisper in its place"


def test_two_speech_clusters_need_one_default_between_them(tmp_path):
    with pytest.raises(ValueError, match="2 transcription clusters declared .* exactly one says default: true"):
        _two_speech(tmp_path, stt_default=False)


def test_the_model_field_is_read_off_the_form():
    from vllm_lb.multipart import get_field
    body = _form(("file", AUDIO, "a.wav"), ("model", PARAKEET.encode(), None))
    assert get_field(body, CTYPE, "model") == PARAKEET
    assert get_field(_form(("file", AUDIO, "a.wav")), CTYPE, "model") is None
    assert get_field(_form(("file", b"x", "model")), CTYPE, "file") is None, "a file part is no text field"


def test_a_chat_request_never_reaches_a_transcription_node(tmp_path):
    cfg = _cfg(tmp_path)
    s = _snap(PAIR_HEAD, STT_HEAD)
    assert choose(s, cfg).url == PAIR_HEAD.url
    assert choose(s, cfg, model="openai/whisper-large-v3-turbo") in (None, PAIR_HEAD)
    assert choose(_snap(STT_HEAD), cfg, model="openai/whisper-large-v3-turbo") is None


def test_a_lone_transcription_node_with_no_metrics_still_serves(tmp_path):
    """faster-whisper exposes no /metrics: its depth is unknown, so it is BUSY — demoted, never banned — and alone in
    its task it is still the one chosen."""
    head = replace(STT_HEAD, depth_known=False)
    assert choose(_snap(PAIR_HEAD, head), _cfg(tmp_path), task="transcription").url == head.url


# ── the multipart body: the model field rewritten, every other byte kept ────────────────────────────────────────────

BOUNDARY = "----lbtest7MA4YWxk"
CTYPE = f"multipart/form-data; boundary={BOUNDARY}"


def _form(*parts: tuple[str, bytes, str | None]) -> bytes:
    out = b""
    for name, value, filename in parts:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{BOUNDARY}\r\nContent-Disposition: {disp}\r\n".encode()
        out += b"Content-Type: audio/wav\r\n" if filename else b""
        out += b"\r\n" + value + b"\r\n"
    return out + f"--{BOUNDARY}--\r\n".encode()


AUDIO = bytes(range(256)) * 8 + b"\r\n--not-the-boundary\r\n"


def test_the_model_field_is_replaced_and_nothing_else_moves():
    from vllm_lb.multipart import set_field
    body = _form(("file", AUDIO, "a.wav"), ("model", b"whisper-1", None), ("language", b"en", None))
    out = set_field(body, CTYPE, "model", "openai/whisper-large-v3-turbo")
    assert out == _form(("file", AUDIO, "a.wav"), ("model", b"openai/whisper-large-v3-turbo", None),
                        ("language", b"en", None))


def test_a_form_without_a_model_field_gets_one():
    from vllm_lb.multipart import set_field
    out = set_field(_form(("file", AUDIO, "a.wav")), CTYPE, "model", "m")
    assert out == _form(("file", AUDIO, "a.wav"), ("model", b"m", None))


@pytest.mark.parametrize("ctype", ["application/json", "multipart/form-data", None])
def test_a_body_that_is_not_a_form_is_refused(ctype):
    from vllm_lb.multipart import set_field
    with pytest.raises(ValueError):
        set_field(b"{}", ctype, "model", "m")


# ── through the proxy ───────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def stt_live(tmp_path):
    from tests.fakes import FakeBackend
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    pair, stt = FakeBackend(served=("Qwen/Qwen3.8-27B",)), FakeBackend(served=("openai/whisper-large-v3-turbo",))
    stt.reply_json = {"text": "she sells seashells"}
    text = PAIR_STT.replace("http://192.0.2.10:8100", pair.url).replace("http://stt-node:8000", stt.url)
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                  listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, first_byte_timeout_s=10.0,
                  broker_enabled=False, cluster_hold_s=0.0)
    hub = Hub(cfg)
    for name, head in (("pair", pair), ("stt", stt)):
        hub.update_backend(head.url, live=True, served_models=tuple(head.served), max_model_len=4096,
                           depth_known=True)
        hub.set_cluster({**hub.cluster_views[name], **HELD}, name)
    proxy = Proxy(cfg, hub)
    proxy.start()

    def post(path, body, ctype):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
        conn.request("POST", path, body=body, headers={"Content-Type": ctype})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    yield hub, pair, stt, post
    proxy.stop()
    pair.close()
    stt.close()


def _posts(head):
    return [r for r in head.requests if r["method"] == "POST"]


def test_a_transcription_upload_reaches_the_stt_node_under_its_served_name(stt_live):
    _, pair, stt, post = stt_live
    body = _form(("file", AUDIO, "a.wav"), ("model", b"whisper-1", None))
    status, answer = post("/v1/audio/transcriptions", body, CTYPE)
    assert status == 200 and answer == {"text": "she sells seashells"}
    sent = _posts(stt)
    assert len(sent) == 1 and _posts(pair) == []
    assert sent[0]["body"] == _form(("file", AUDIO, "a.wav"), ("model", b"openai/whisper-large-v3-turbo", None))
    assert int(sent[0]["headers"]["content-length"]) == len(sent[0]["body"])
    assert sent[0]["headers"]["content-type"] == CTYPE


def test_chat_still_goes_to_the_pair(stt_live):
    _, pair, stt, post = stt_live
    status, _ = post("/v1/chat/completions", json.dumps({"model": "x", "messages": []}).encode(), "application/json")
    assert status == 200 and len(_posts(pair)) == 1 and _posts(stt) == []


def test_no_transcription_node_live_is_a_503_that_says_so(stt_live):
    hub, pair, stt, post = stt_live
    hub.update_backend(stt.url, live=False)
    status, answer = post("/v1/audio/transcriptions", _form(("file", AUDIO, "a.wav"), ("model", b"whisper-1", None)),
                          CTYPE)
    assert status == 503 and "transcription" in answer["error"]["message"], answer
    assert _posts(pair) == [], "never a chat cluster"


def test_an_upload_that_is_not_a_form_is_a_400_never_sent(stt_live):
    _, pair, stt, post = stt_live
    status, answer = post("/v1/audio/transcriptions", b"{}", "application/json")
    assert status == 400 and "multipart" in answer["error"]["message"], answer
    assert _posts(stt) == [] and _posts(pair) == []


def test_a_chat_request_naming_the_speech_model_is_refused_naming_only_chat_models_live(tmp_path):
    """The speech model is a catalog model only the stt cluster runs: on a chat path it is refused (never moved onto
    the pair), and the refusal lists what a CHAT request may name — not the speech model it cannot use."""
    from vllm_lb.router import eligible
    verdict = eligible(_snap(PAIR_HEAD, STT_HEAD), _cfg(tmp_path), model="openai/whisper-large-v3-turbo")
    assert verdict.backends == [] and "Qwen/Qwen3.8-27B" in verdict.why
    assert "live: Qwen/Qwen3.8-27B" in verdict.why and "live: " + "openai" not in verdict.why
    assert verdict.why.count("openai/whisper-large-v3-turbo") == 1, "named once: as the model asked for"


@pytest.fixture
def two_speech_live(tmp_path):
    from tests.fakes import FakeBackend
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    pair, stt, speech = (FakeBackend(served=("Qwen/Qwen3.8-27B",)), FakeBackend(served=("openai/whisper-large-v3-turbo",)),
                         FakeBackend(served=(PARAKEET,)))
    stt.reply_json, speech.reply_json = {"text": "whisper"}, {"text": "parakeet"}
    text = (PAIR_STT.replace("    control: local\n    run_at_boot: true\n", "    control: local\n    run_at_boot: true\n"
                             "    default: true\n").replace("backends:", SPEECH + "backends:")
            .replace("http://192.0.2.10:8100", pair.url).replace("http://stt-node:8000", stt.url)
            .replace("http://192.0.2.11:18000", speech.url))
    cfg = replace(Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, text), "LB_CLUSTER_STATE": str(tmp_path / "c.json")}),
                  listen_host="127.0.0.1", listen_port=0, connect_timeout_s=2.0, first_byte_timeout_s=10.0,
                  broker_enabled=False, cluster_hold_s=0.0)
    hub = Hub(cfg)
    for name, head in (("pair", pair), ("stt", stt), ("speech", speech)):
        hub.update_backend(head.url, live=True, served_models=tuple(head.served), max_model_len=4096,
                           depth_known=True)
        hub.set_cluster({**hub.cluster_views[name], **HELD}, name)
    proxy = Proxy(cfg, hub)
    proxy.start()

    def post(model: bytes):
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=20)
        conn.request("POST", "/v1/audio/transcriptions", body=_form(("file", AUDIO, "a.wav"), ("model", model, None)),
                     headers={"Content-Type": CTYPE})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    yield hub, stt, speech, post
    proxy.stop()
    for b in (pair, stt, speech):
        b.close()


def test_an_upload_naming_parakeet_reaches_it_and_a_loose_one_the_default(two_speech_live):
    hub, stt, speech, post = two_speech_live
    assert post(PARAKEET.encode()) == (200, {"text": "parakeet"})
    assert post(b"whisper-1") == (200, {"text": "whisper"})
    assert [len(_posts(stt)), len(_posts(speech))] == [1, 1]
    hub.update_backend(speech.url, live=False)
    status, answer = post(PARAKEET.encode())
    assert status == 503 and len(_posts(stt)) == 1, "Parakeet down: refused, never Whisper in its place"
