"""The page's contract: the switch (`POST /lb/cluster`) and what `/lb/status` says about it."""
from __future__ import annotations

import http.client
import json

import pytest

from vllm_lb.catalog import document, ModelCatalog
from vllm_lb.cluster import DesiredStore, parse_hosts
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy
from vllm_lb.state import LEASE_RUNNING, Lease

HOSTS = "box-a-gpu=box-a=user@10.0.0.1=10.9.200.1,box-b-gpu=box-b=user@10.0.0.2=10.9.200.2"
#: The ModelChoices in vllm_lb/library: two for the cluster, one for classic vLLM.
FLASH, DENSE, NVFP4 = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4", "qwen3.8-27b-nvfp4"
DFLASH = "qwen3.8-27b-dflash2"
SGLANG = "qwen3.8-27b-sglang-dflash2"
#: Flash-Next on SGLang: the same weights as FLASH, the same engine as SGLANG (catalog.json, 2026-10-05).
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
GLM = "glm-5.3-flash-nvfp4"
DSV4 = "deepseek-v4-flash-nvfp4"


def free_memory(box_a: float = 114.8, box_b: float = 114.3):
    """A published cluster view carrying only what an estimate needs: each box's MemAvailable now."""
    def setup(hub):
        hub.set_cluster({"configured": True, "state": "off", "held": False, "max_depth": 16,
                         "desired": {"mode": "classic"},
                         "boxes": {"box-a": {"mem_available_gib": box_a},
                                   "box-b": {"mem_available_gib": box_b}}})
    return setup


@pytest.fixture
def served(tmp_path):
    started: list[Proxy] = []

    def make(*, configured: bool = True, hub_setup=None, catalog=None):
        cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),),
                     listen_host="127.0.0.1", listen_port=0,
                     cluster_name="pair", cluster_hosts=parse_hosts(HOSTS) if configured else (),
                     cluster_url="http://127.0.0.1:8100" if configured else "",
                     cluster_max_num_batched_tokens=8192 if configured else None,
                     cluster_state_path=str(tmp_path / "cluster.json"))
        hub = Hub(cfg)
        if hub_setup:
            hub_setup(hub)
        proxy = Proxy(cfg, hub, cluster_switch=(DesiredStore(cfg.cluster_state_path, catalog=catalog)
                                                if configured else None))
        proxy.start()
        started.append(proxy)
        return cfg, proxy

    yield make
    for proxy in started:
        proxy.stop()


def call(proxy: Proxy, method: str, path: str, body=None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request(method, path, body=None if body is None else json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read() or b"{}")


def test_start_is_accepted_and_written_as_the_desired_state(served):
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_seqs": 8, "max_model_len": 65536})
    assert status == 202
    assert body["desired"] == {"mode": "cluster", "generation": 1, "max_num_seqs": 8, "max_model_len": 65536,
                               "model_choice": "qwen3.8-flash-next-hibrid48",
                               # Which checkpoint: resolved by the Start to the model's default, its twin.
                               "uncensored": True,
                               # The KV dtype nobody set: null, the profile's own.
                               "kv_cache_dtype": None,
                               # Sized by its measurement, not the estimate: the operator chose none.
                               "estimated": False,
                               # The engine knobs nobody set: null, the defaults.
                               "max_num_batched_tokens": None, "num_speculative_tokens": None,
                               "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None,
                               # …and the pool is a size (or the model's), not sized at each Start.
                               "kv_cache_auto": False,
                               # …and no long-prefill threshold: null, vLLM's own default (off).
                               "long_prefill_token_threshold": None}
    assert DesiredStore(cfg.cluster_state_path).load().mode == "cluster"


def test_stop_is_accepted(served):
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False})
    assert status == 202 and body["desired"]["mode"] == "classic"


@pytest.mark.parametrize("body", [{}, {"enabled": "yes"}, {"enabled": True, "max_num_seqs": "8"},
                                  {"enabled": True, "max_num_seqs": True}])
def test_a_malformed_switch_is_a_400(served, body):
    _, proxy = served()
    status, _ = call(proxy, "POST", "/lb/cluster", body)
    assert status == 400


def test_a_refused_switch_is_a_409_with_the_reason(served):
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_seqs": 8})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_seqs": 16})
    assert status == 409
    assert body["error"]["type"] == "cluster_rejected" and "stop the cluster" in body["error"]["message"]


def test_the_switch_is_logged_with_who_asked(served, caplog):
    """A Start at 2026-09-15 21:12:34Z took both GB10 boxes from the fleet, and its log line could not say
    whether a browser behind the tunnel or a script on the host had sent it. The switch is the most
    consequential write this LB takes, so its line names the caller: address, User-Agent, X-Forwarded-For."""
    import logging

    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    _, proxy = served()
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("POST", "/lb/cluster", body=json.dumps({"enabled": True}),
                 headers={"Content-Type": "application/json", "User-Agent": "operator-browser/1.0",
                          "X-Forwarded-For": "203.0.113.9"})
    assert conn.getresponse().status == 202
    (line,) = [r.getMessage() for r in caplog.records if "cluster switch" in r.getMessage()]
    assert "127.0.0.1" in line and "operator-browser/1.0" in line and "203.0.113.9" in line, line


def test_a_start_names_the_model_choice_and_it_is_what_gets_written(served):
    """The operator CHOOSES; the switch records it, and the loop launches that and nothing else."""
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster",
                        {"enabled": True, "model_choice": DENSE, "max_num_seqs": 4})
    assert status == 202
    assert body["desired"]["model_choice"] == DENSE and body["desired"]["max_num_seqs"] == 4
    assert DesiredStore(cfg.cluster_state_path).load().model_choice == DENSE


def test_a_start_without_a_model_choice_keeps_the_catalogs_default(served):
    _, proxy = served()
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": True})
    assert body["desired"]["model_choice"] == FLASH


def test_a_model_choice_the_catalog_cannot_launch_is_a_400_that_says_what_it_has(served):
    _, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": "gpt-5"})
    assert status == 400 and body["error"]["type"] == "unknown_model_choice"
    assert FLASH in body["error"]["message"] and DENSE in body["error"]["message"]


def test_a_classic_only_model_is_not_a_cluster_choice(served):
    _, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": NVFP4})
    assert status == 400 and body["error"]["type"] == "unknown_model_choice"


@pytest.mark.parametrize("value", [7, True, "", {"id": FLASH}])
def test_a_model_choice_that_is_not_a_name_is_a_400(served, value):
    _, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": value})
    assert status == 400 and body["error"]["type"] == "invalid_request"


def test_changing_the_model_while_the_cluster_runs_is_refused_with_the_reason(served):
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": FLASH})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE})
    assert status == 409 and "stop the cluster" in body["error"]["message"]


def test_a_restart_is_accepted_while_the_cluster_runs_and_bumps_the_generation(served):
    """D1: the operator's "reload the pair at these settings" must not have to go through a Stop, which
    is what hands the GB10 boxes back to the fleet (worklog/vllm-cluster-handover.md)."""
    cfg, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_seqs": 2})
    status, body = call(proxy, "POST", "/lb/cluster",
                        {"enabled": True, "restart": True, "max_num_seqs": 8})
    assert status == 202, body
    assert body["desired"]["mode"] == "cluster" and body["desired"]["generation"] == 2
    assert body["desired"]["max_num_seqs"] == 8
    assert DesiredStore(cfg.cluster_state_path).load().max_num_seqs == 8


@pytest.mark.parametrize("value", ["yes", 1, None], ids=["a-word", "a-number", "null"])
def test_a_restart_flag_that_is_not_a_boolean_is_a_400(served, value):
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": value})
    assert status == 400 and body["error"]["type"] == "invalid_request"


def test_a_stop_never_needs_a_valid_choice(served):
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False})
    assert status == 202 and body["desired"]["mode"] == "classic"


def test_the_switch_line_names_the_model_it_started(served, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE})
    (line,) = [r.getMessage() for r in caplog.records if "cluster switch" in r.getMessage()]
    assert DENSE in line, line


# ── what the page may choose from ─────────────────────────────────────────────

def test_status_offers_every_model_the_gateway_can_launch_in_each_serving_mode(served):
    """ONE ModelCatalog behind both pickers: the cluster's models, and classic vLLM's."""
    _, proxy = served()
    _, body = call(proxy, "GET", "/lb/status")
    models = body["models"]
    assert {m["id"] for m in models["cluster"]} == {FLASH, DENSE, DFLASH, SGLANG, FLASH_SGLANG, GLM, DSV4,
                                                   "qwen3.8-flash-next-tensorfold-nvfp4",
                                                   "qwen3.8-flash-next-tensorfold-mlx-mtp"}
    assert [m["id"] for m in models["classic"]] == [NVFP4]
    assert models["classic_choice"] == NVFP4
    assert models["error"] == ""
    assert models["classic_memory"] == "set by broker_parrot"


def test_the_classic_picker_says_plainly_that_it_cannot_be_chosen_yet(served):
    """Choosing the classic model needs a broker_parrot engine change (M7); until it ships, the page
    must say so rather than offer a control that does nothing."""
    _, proxy = served()
    _, body = call(proxy, "GET", "/lb/status")
    assert body["models"]["classic_selectable"] is False
    assert body["models"]["classic_detail"]


def test_the_catalog_is_offered_even_where_no_cluster_is_configured(served):
    _, proxy = served(configured=False)
    _, body = call(proxy, "GET", "/lb/status")
    assert [m["id"] for m in body["models"]["classic"]] == [NVFP4]


# ── the estimate, before anything is started (M8) ─────────────────────────────

def test_the_estimate_sizes_a_selection_per_box_before_anything_is_started(served):
    """The operator's question: what will this model, at this concurrency and context, take per box?"""
    _, proxy = served(hub_setup=free_memory())
    # The 2026-09-15 start's own placement, the table replicated: what its 103.2 GiB and 11.6 left were measured at.
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=2&max_model_len=262144"
                                      f"&ple_replicate=true")
    assert status == 200
    assert body["model_choice"] == FLASH
    assert [b["box"] for b in body["boxes"]] == ["box-a", "box-b"]
    head, worker = body["boxes"]
    assert (head["role"], worker["role"]) == ("head", "worker")
    assert head["total_gib"] == 103.2 and head["leaves_gib"] == 11.6
    assert head["available_gib"] == 114.8 and head["actual_used_gib"] is None
    assert any(w["code"] == "floor" for w in head["warnings"]), "11.6 GiB is under the 12 GB danger floor"


def test_the_estimate_follows_the_selection_it_is_given(served):
    _, proxy = served(hub_setup=free_memory())
    _, small = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={DENSE}&max_num_seqs=1"
                                  f"&max_model_len=32768")
    _, big = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=8"
                                f"&max_model_len=262144")
    assert small["boxes"][0]["total_gib"] != big["boxes"][0]["total_gib"]
    assert small["model_choice"] == DENSE


def test_the_estimate_without_a_box_to_size_against_still_says_what_the_model_needs(served):
    _, proxy = served()
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}")
    assert status == 200
    assert body["boxes"][0]["available_gib"] is None and body["boxes"][0]["leaves_gib"] is None
    need = body["boxes"][0]
    assert need["total_gib"] > need["weights_gib"] + need["kv_gib"] > 0, "at least its weights and its pool"


@pytest.mark.parametrize("query", [
    f"model_choice={NVFP4}",                     # classic: broker_parrot sizes it, not us
    "model_choice=nope",
    f"model_choice={FLASH}&max_num_seqs=0",
    f"model_choice={FLASH}&max_num_seqs=49",
    f"model_choice={FLASH}&max_model_len=999",
    f"model_choice={FLASH}&max_num_seqs=two",
    "",
], ids=repr)
def test_the_estimate_refuses_what_it_cannot_size(served, query):
    _, proxy = served(hub_setup=free_memory())
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?{query}")
    assert status == 400, body
    assert body["error"]["message"]


def test_an_unconfigured_cluster_refuses_the_switch(served):
    _, proxy = served(configured=False)
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True})
    assert status == 409 and "not configured" in body["error"]["message"]


def test_status_carries_the_cluster_block_and_its_backend(served):
    _, proxy = served()
    _, body = call(proxy, "GET", "/lb/status")
    assert body["cluster"]["configured"] is True
    assert [b["url"] for b in body["backends"] if b["kind"] == "cluster"] == ["http://127.0.0.1:8100"]


def test_an_unconfigured_status_says_so_and_has_no_cluster_backend(served):
    _, proxy = served(configured=False)
    _, body = call(proxy, "GET", "/lb/status")
    assert body["cluster"] == {"configured": False}
    assert not [b for b in body["backends"] if b["kind"] == "cluster"]


def test_an_answer_from_the_cluster_says_which_boxes_served_it(tmp_path):
    """The page stamps every answer with X-LB-Box; the pair has no single box, so it names both."""
    from tests.fakes import FakeBackend

    fake = FakeBackend(served=("Qwen/Qwen3.8-Flash-Next",), max_model_len=262144)
    cfg = Config(vllm_boxes=(), listen_host="127.0.0.1", listen_port=0,
                 cluster_name="pair", cluster_hosts=parse_hosts(HOSTS), cluster_url=fake.url,
                 cluster_state_path=str(tmp_path / "cluster.json"))
    hub = Hub(cfg)
    hub.update_backend(fake.url, live=True, served_models=("Qwen/Qwen3.8-Flash-Next",), depth_known=True,
                       last_probe_at=1.0)
    hub.set_cluster({"configured": True, "held": True, "state": "running",
                     "desired": {"mode": "cluster"}, "max_depth": 16})
    proxy = Proxy(cfg, hub)
    proxy.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
        conn.request("POST", "/v1/messages", body=json.dumps({"model": "qwen3.8:latest", "max_tokens": 8,
                                                              "messages": [{"role": "user", "content": "hi"}]}),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
    finally:
        proxy.stop()
        fake.close()
    assert resp.status == 200
    assert resp.getheader("X-LB-Kind") == "cluster"
    assert resp.getheader("X-LB-Box") == "box-a + box-b"


# ── the engine knobs: the prefill budget and the drafter's depth ──────────────

def test_the_knobs_ride_on_a_start_and_come_back_in_the_desired_state(served):
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_batched_tokens": 16384,
                                                       "num_speculative_tokens": 2})
    assert status == 202, body
    assert (body["desired"]["max_num_batched_tokens"], body["desired"]["num_speculative_tokens"]) == (16384, 2)
    stored = DesiredStore(cfg.cluster_state_path).load()
    assert (stored.max_num_batched_tokens, stored.num_speculative_tokens) == (16384, 2)


def test_null_resets_a_knob_to_its_default_and_leaving_it_out_keeps_it(served):
    """`desired` says null for "the default", and what the page reads it can post back: null is a value."""
    cfg, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_batched_tokens": 4096, "num_speculative_tokens": 0})
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": False})
    assert (body["desired"]["max_num_batched_tokens"], body["desired"]["num_speculative_tokens"]) == (4096, 0), \
        "a Stop that said nothing about the knobs reset them"
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_batched_tokens": None,
                                                  "num_speculative_tokens": None, "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None})
    assert (body["desired"]["max_num_batched_tokens"], body["desired"]["num_speculative_tokens"]) == (None, None)


@pytest.mark.parametrize("key, value, range_words", [
    ("max_num_batched_tokens", 1024, "from 2048 to 65536"),
    ("max_num_batched_tokens", 2047, "from 2048 to 65536"),
    ("max_num_batched_tokens", 65537, "from 2048 to 65536"),
    ("max_num_batched_tokens", "8192", "from 2048 to 65536"),
    ("max_num_batched_tokens", 8192.0, "from 2048 to 65536"),
    ("max_num_batched_tokens", True, "from 2048 to 65536"),
    ("max_num_batched_tokens", [8192], "from 2048 to 65536"),
    ("num_speculative_tokens", -1, "from 0 to 8"),
    ("num_speculative_tokens", 9, "from 0 to 8"),
    ("num_speculative_tokens", "4", "from 0 to 8"),
    ("num_speculative_tokens", False, "from 0 to 8"),
    ("num_speculative_tokens", {"n": 4}, "from 0 to 8"),
    ("long_prefill_token_threshold", -1, "from 0 to 65536"),
    ("long_prefill_token_threshold", 65537, "from 0 to 65536"),
    ("long_prefill_token_threshold", "2048", "from 0 to 65536"),
    ("long_prefill_token_threshold", 2048.0, "from 0 to 65536"),
    ("long_prefill_token_threshold", True, "from 0 to 65536"),
], ids=repr)
def test_a_knob_of_the_wrong_type_or_out_of_range_is_a_400_naming_the_field_and_its_range(served, key, value,
                                                                                           range_words):
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, key: value})
    assert status == 400 and body["error"]["type"] == "invalid_request", body
    assert f"`{key}`" in body["error"]["message"] and range_words in body["error"]["message"], body
    assert DesiredStore(cfg.cluster_state_path).load().mode == "classic", "a refused switch wrote nothing"


def test_a_ple_placement_for_a_model_without_a_ple_table_is_the_switchs_refusal(served):
    """Audit #13: the dense 27B has no PLE table, so `ple_replicate` true or false is a 409 in the switch's
    words — like a depth for a model with no drafter — and nothing is written."""
    cfg, proxy = served()
    for value in (True, False):
        status, reply = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE,
                                                            "ple_replicate": value})
        assert status == 409 and reply["error"]["type"] == "cluster_rejected", reply
        assert "no PLE table" in reply["error"]["message"]
    assert DesiredStore(cfg.cluster_state_path).load().mode == "classic", "a refused switch wrote nothing"
    status, reply = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE,
                                                        "ple_replicate": None})
    assert status == 202 and reply["desired"]["ple_replicate"] is None, reply


def uncalibrated() -> ModelCatalog:
    """The shipped catalog with the dense 27B's footprint UNMEASURED again — the model it was until 2026-10-02."""
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["memory"].update(
        calibration="", calibrated_batched_tokens=0, calibrated_max_num_seqs=0)
    return ModelCatalog.from_dict(doc)


def test_auto_for_an_uncalibrated_model_is_refused_and_the_estimate_says_why(served):
    cfg, proxy = served(catalog=uncalibrated())
    status, reply = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE,
                                                        "kv_cache_memory": "auto"})
    assert status == 409 and "is not calibrated: choose a size" in reply["error"]["message"], reply
    status, est = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={DENSE}")
    assert status == 200 and est["auto"]["kv_cache_memory"] is None
    assert est["auto"]["refusal"] == "Qwen3.8 27B · dense NVFP4 is not calibrated: choose a size"
    assert all(b["calibration"] == "" and b["concurrency_gib_per_seq"] == 0.0 for b in est["boxes"])


def test_a_drafter_depth_for_a_model_without_a_drafter_is_the_switchs_refusal(served):
    # The shipped catalog with the dense model's drafter taken away: a ModelSpec with no speculative decoding.
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["launch"]["cluster"].update(speculative_config=None,
                                                                                 mtp_prefix="")
    _, proxy = served(catalog=ModelCatalog.from_dict(doc))
    status, reply = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": DENSE,
                                                        "num_speculative_tokens": 2})
    assert status == 409 and reply["error"]["type"] == "cluster_rejected", reply
    assert "no speculative decoding" in reply["error"]["message"]


def test_changing_a_knob_under_a_running_pair_needs_the_restart_that_applies_it(served):
    """The panel's "apply & restart": without `restart` the change is refused, so no stray Start can reload a
    serving pair; with it the generation moves on in `cluster` mode and the new values are what is stored."""
    cfg, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True})
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_batched_tokens": 16384})
    assert status == 409 and "stop the cluster" in body["error"]["message"]
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True,
                                                       "max_num_batched_tokens": 16384, "num_speculative_tokens": 3})
    assert status == 202, body
    assert (body["desired"]["mode"], body["desired"]["generation"]) == ("cluster", 2)
    assert (body["desired"]["max_num_batched_tokens"], body["desired"]["num_speculative_tokens"]) == (16384, 3)


def test_the_long_prefill_threshold_rides_a_restart_is_kept_by_a_stop_and_null_or_0_is_off(served):
    """The long-prefill threshold is an engine knob like the budget: it rides a Start, a change under a running
    pair needs the restart that applies it, a Stop that says nothing keeps it, and null resets it. One past
    max_model_len is vLLM's own refusal, said here as the switch's 409 before two boxes load a model."""
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_num_batched_tokens": 8192})
    assert status == 202 and body["desired"]["long_prefill_token_threshold"] is None, body
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "long_prefill_token_threshold": 2048})
    assert status == 409 and "long-prefill threshold" in body["error"]["message"], body
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True,
                                                       "long_prefill_token_threshold": 2048})
    assert status == 202, body
    assert (body["desired"]["generation"], body["desired"]["max_num_batched_tokens"],
            body["desired"]["long_prefill_token_threshold"]) == (2, 8192, 2048)
    assert DesiredStore(cfg.cluster_state_path).load().long_prefill_token_threshold == 2048
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": False})
    assert body["desired"]["long_prefill_token_threshold"] == 2048, "a Stop that said nothing reset it"
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "long_prefill_token_threshold": 0})
    assert body["desired"]["long_prefill_token_threshold"] == 0
    _, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "long_prefill_token_threshold": None})
    assert body["desired"]["long_prefill_token_threshold"] is None
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "max_model_len": 4096,
                                                       "long_prefill_token_threshold": 8192})
    assert status == 409 and "cannot be greater than max_model_len" in body["error"]["message"], body


def test_the_switch_line_names_the_knobs_it_started_with(served, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    _, proxy = served()
    call(proxy, "POST", "/lb/cluster", {"enabled": True, "num_speculative_tokens": 2})
    (line,) = [r.getMessage() for r in caplog.records if "cluster switch" in r.getMessage()]
    assert "prefill budget default" in line and "speculative tokens 2" in line, line


def test_the_estimate_leaves_the_drafter_out_when_speculation_is_off(served):
    """0 launches no drafter, so its weights are not on the box — 0.745 GiB of Flash-Next's per rank."""
    _, proxy = served(hub_setup=free_memory())
    base = f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=8&max_model_len=262144"
    _, on = call(proxy, "GET", base)
    _, off = call(proxy, "GET", base + "&num_speculative_tokens=0")
    _, four = call(proxy, "GET", base + "&num_speculative_tokens=4")
    assert on["boxes"][0]["mtp_gib"] == 0.7 and four["boxes"][0]["mtp_gib"] == 0.7
    assert off["boxes"][0]["mtp_gib"] == 0.0
    assert off["boxes"][0]["total_gib"] < on["boxes"][0]["total_gib"]
    status, bad = call(proxy, "GET", base + "&num_speculative_tokens=9")
    assert status == 400 and "num_speculative_tokens" in bad["error"]["message"]


def test_the_estimate_sizes_the_prefill_budget_so_a_refusal_is_visible_before_Start(served):
    """The operator changes the prefill budget in the panel and the panel asks THIS. It has to answer
    that 16,384 will not start on 114.8 GiB, while the budget can still be changed — the alternative is
    finding out from two hung boxes, which is what happened on 2026-09-30."""
    _, proxy = served(hub_setup=free_memory())
    # As measured on 2026-09-30: the table replicated.
    base = f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=8&max_model_len=262144&ple_replicate=true"
    _, small = call(proxy, "GET", base + "&max_num_batched_tokens=8192")
    _, big = call(proxy, "GET", base + "&max_num_batched_tokens=16384")

    head_small, head_big = small["boxes"][0], big["boxes"][0]
    assert head_small["max_num_batched_tokens"] == 8192 and head_big["max_num_batched_tokens"] == 16384
    assert head_small["activation_gib"] == 0.0 and head_big["activation_gib"] > 0
    assert head_big["total_gib"] > head_small["total_gib"]
    # 8,192 fits on what the box really had; 16,384 does not, and the answer names the knob.
    assert head_small["needed_now_gib"] <= head_small["available_gib"]
    assert head_big["needed_now_gib"] > head_big["available_gib"]
    codes = {w["code"] for w in head_big["warnings"]}
    assert {"no_headroom", "budget"} <= codes, head_big["warnings"]
    assert not {"no_headroom", "budget"} & {w["code"] for w in head_small["warnings"]}
    said = " ".join(w["message"] for w in head_big["warnings"])
    assert "batched 16,384" in said and "@ 8,192" in said, said


def test_the_estimate_sizes_the_concurrency_so_a_refusal_is_visible_before_Start(served):
    """The operator changes the concurrency dropdown and the panel asks THIS. It has to answer that 24
    seats will not start on 114.8 GiB — measured 2026-10-01, that configuration rests the head at
    1.7 GiB free with no traffic at all — while the dropdown can still be changed."""
    _, proxy = served(hub_setup=free_memory())
    # As measured on 2026-10-01: the table replicated.
    base = (f"/lb/cluster/estimate?model_choice={FLASH}&max_model_len=262144&max_num_batched_tokens=8192"
            f"&ple_replicate=true")
    _, small = call(proxy, "GET", base + "&max_num_seqs=16")
    _, big = call(proxy, "GET", base + "&max_num_seqs=24")

    head_small, head_big = small["boxes"][0], big["boxes"][0]
    assert head_small["max_num_seqs"] == 16 and head_big["max_num_seqs"] == 24
    assert head_small["concurrency_gib"] == 0.0 and head_big["concurrency_gib"] > 0
    assert head_big["total_gib"] > head_small["total_gib"]
    # 16 fits on what the box really has; 24 does not, and the answer names the knob.
    assert head_small["needed_now_gib"] <= head_small["available_gib"]
    assert head_big["needed_now_gib"] > head_big["available_gib"]
    codes = {w["code"] for w in head_big["warnings"]}
    assert {"no_headroom", "concurrency"} <= codes, head_big["warnings"]
    assert not {"no_headroom", "concurrency"} & {w["code"] for w in head_small["warnings"]}
    said = " ".join(w["message"] for w in head_big["warnings"])
    assert "seqs 24" in said and "@ 16" in said, said


def test_the_estimate_defaults_the_prefill_budget_to_what_the_launch_would_resolve(served):
    """Nothing given is the budget the pair would really run — the model's own tuned one (Flash-Next: 4,096), else
    the FLEET's `max_num_batched_tokens` (the dense 27B: 8,192) — never "no budget" and never tp-rank's own number
    (it has none since 2026-10-29)."""
    _, proxy = served(hub_setup=free_memory())
    _, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=8")
    assert body["boxes"][0]["max_num_batched_tokens"] == 4096
    assert body["max_num_batched_tokens"] == 4096
    _, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={DENSE}&max_num_seqs=8")
    assert body["boxes"][0]["max_num_batched_tokens"] == body["max_num_batched_tokens"] == 8192, "the fleet's"
    for bad in ("2047", "65537", "lots"):
        status, refused = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}"
                                             f"&max_num_batched_tokens={bad}")
        assert status == 400 and "max_num_batched_tokens" in refused["error"]["message"], refused


# ── the KV pool at `auto`: what the estimate shows is what the Start pins ─────

LOAD = json.loads((__import__("pathlib").Path(__file__).resolve().parent / "fixtures"
                   / "cluster-load-2026-10-01-memory.json").read_text())
GEN_99 = LOAD["generation_99"]["config"]
#: The Start that put generation 99 up, as the page posts it.
START_99 = {"enabled": True, "max_num_seqs": GEN_99["max_num_seqs"], "max_model_len": GEN_99["max_model_len"],
            "max_num_batched_tokens": GEN_99["max_num_batched_tokens"], "ple_replicate": GEN_99["ple_replicate"],
            "kv_cache_memory": GEN_99["kv_cache_memory"]}


def serving_99(*, held: bool = True, now: dict | None = None, rest: dict | None = None):
    """The cluster view the loop published while generation 99 SERVED on 2026-10-01: each box's lowest recent
    MemAvailable (`now`) and the most it had at rest, both read off the committed fixture, and what the pair
    was launched with (`effective`)."""
    now = now if now is not None else LOAD["generation_99"]["loaded_gib"]
    rest = rest if rest is not None else LOAD["generation_99"]["at_rest_gib"]
    rank = {"present": True, "running": True, "gen": 99, "exit_code": None, "model_choice": FLASH}

    def setup(hub):
        hub.set_cluster({"configured": True, "state": "running" if held else "starting", "held": held,
                         "max_depth": 16, "desired": {"mode": "cluster", "generation": 99},
                         "effective": {"model_choice": FLASH, "generation": 99, "num_speculative_tokens": 4,
                                       "kv_offload_gib": 0, **{k: GEN_99[k] for k in GEN_99}},
                         "defaults": {"max_num_batched_tokens": 8192},
                         "boxes": {b: {"mem_available_gib": now[b], "mem_stable_gib": now[b],
                                       "mem_rest_gib": rest[b] if held else None, "rank": rank}
                                   for b in ("box-a", "box-b")}})
    return setup


def test_a_restart_at_auto_pins_the_size_the_estimate_showed_and_says_so(served, caplog):
    """The page shows `auto` from GET /lb/cluster/estimate; the Start at auto is sized from the same published
    figures, so the number the operator was shown is the number stored — and logged."""
    import logging

    caplog.set_level(logging.INFO, logger="vllm_lb.proxy")
    cfg, proxy = served(hub_setup=serving_99())
    assert call(proxy, "POST", "/lb/cluster", START_99)[0] == 202
    query = (f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=16&max_model_len=262144"
             f"&max_num_batched_tokens=4096&ple_replicate=false")
    _, shown = call(proxy, "GET", query)
    assert (shown["auto"]["kv_cache_memory"], shown["auto"]["binding"]) == (36_000_000_000, "box-a")
    assert shown["auto"]["baseline"] == 35_000_000_000 and shown["auto"]["refusal"] == ""
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True, "kv_cache_memory": "auto"})
    assert status == 202, body
    assert (body["desired"]["kv_cache_memory"], body["desired"]["kv_cache_auto"], body["desired"]["generation"]) == \
        (36_000_000_000, True, 2)
    assert DesiredStore(cfg.cluster_state_path).load().kv_cache_memory == 36_000_000_000
    (line,) = [r.getMessage() for r in caplog.records if "cluster restart" in r.getMessage()]
    assert "KV pool 36000000000 (auto)" in line, line


def test_the_estimate_can_be_asked_at_auto_and_sizes_every_box_at_the_pool_it_resolves_to(served):
    _, proxy = served(hub_setup=serving_99())
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=16"
                                      f"&max_num_batched_tokens=4096&ple_replicate=false&kv_cache_memory=auto")
    assert status == 200, body
    assert (body["kv_cache_memory"], body["kv_cache_auto"]) == (36_000_000_000, True)
    assert body["boxes"][0]["kv_pinned_gib"] == round(36_000_000_000 / 1024 ** 3, 1)


def test_a_restart_at_auto_that_would_take_a_box_under_the_floor_is_the_switchs_409_in_the_gates_words(served):
    """An LB that began watching generation 99 only under load has no resting figure to credit: from 13.5 GiB it
    charges the whole load again, and not even the current 35 GB pool keeps the floor. A refusal, never a
    smaller pool — and nothing is written."""
    loaded = LOAD["generation_99"]["loaded_gib"]
    cfg, proxy = served(hub_setup=serving_99(rest=loaded))
    call(proxy, "POST", "/lb/cluster", START_99)
    before = DesiredStore(cfg.cluster_state_path).load()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True, "kv_cache_memory": "auto"})
    assert status == 409 and body["error"]["type"] == "cluster_rejected", body
    message = body["error"]["message"]
    assert message.startswith(f"kv_cache_memory \"auto\" cannot be sized now: box-a: {loaded['box-a']:.1f} GiB "
                              f"free now"), message
    assert "even the current 35 GB pool" in message and "danger floor: short" in message, message
    assert DesiredStore(cfg.cluster_state_path).load() == before


def test_auto_is_not_sized_while_the_pair_is_still_starting(served):
    """A starting pair's memory is still moving — weights loading, the pool not yet allocated — so it is neither a
    serving pair to measure nor a free box to size."""
    _, proxy = served(hub_setup=serving_99(held=False))
    call(proxy, "POST", "/lb/cluster", START_99)
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True, "kv_cache_memory": "auto"})
    assert status == 409, body
    assert "the pair on box-a and box-b is starting" in body["error"]["message"]


@pytest.mark.parametrize("value", ["AUTO", "36GB", 7_999_999_999, True, [36_000_000_000]], ids=repr)
def test_a_pool_that_is_not_a_size_auto_or_null_is_a_400_that_says_all_three(served, value):
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "kv_cache_memory": value})
    assert status == 400 and body["error"]["type"] == "invalid_request", body
    assert "from 8000000000 to 96000000000" in body["error"]["message"]
    assert "\"auto\"" in body["error"]["message"] and "null" in body["error"]["message"]
    assert DesiredStore(cfg.cluster_state_path).load().mode == "classic"


def test_auto_charges_the_load_a_selections_own_prefill_budget_takes_and_never_less_than_the_minimum(served):
    """The load is not one number: a full 8,192-token chunk's activations take twice what a 4,096 one does
    (generations 96/97 lost 9.5-10.2 GiB to load, 98/99 3.8-5.9). Auto charges the model's slope for the budget
    the selection launches with, and the gateway's measured minimum where that is the larger."""
    _, proxy = served(hub_setup=serving_99())
    base = f"/lb/cluster/estimate?model_choice={FLASH}&max_num_seqs=16&ple_replicate=false"
    _, small = call(proxy, "GET", base + "&max_num_batched_tokens=4096")
    _, large = call(proxy, "GET", base + "&max_num_batched_tokens=8192")
    assert {b["load_gib"] for b in small["auto"]["boxes"]} == {6.0}
    assert {b["load_gib"] for b in large["auto"]["boxes"]} == {8.8}
    # …which on generation 99's boxes is more than even its 35 GB pool leaves room for: refused, not shrunk.
    assert small["auto"]["kv_cache_memory"] == 36_000_000_000
    assert large["auto"]["kv_cache_memory"] is None and "even the current 35 GB pool" in large["auto"]["refusal"]


def test_a_restart_preview_credits_the_running_pairs_own_memory_back(served):
    """UX audit 2026-10-02: with generation 105 serving, every Apply preview read `45.7 GiB free now: this
    selection needs 76.0 GiB` — a refusal of the very selection that was running. A restart replaces the pair,
    so the box's memory for it is what it had when the pair launched (`mem_at_launch`, published), else what
    is free now plus what our rank there costs."""
    now = LOAD["generation_99"]["loaded_gib"]
    launch = {"box-a": 115.5, "box-b": 115.6}

    def with_launch(hub):
        serving_99()(hub)
        view = hub.snapshot().extra["cluster"]
        for box, gib in launch.items():
            view["boxes"][box]["mem_at_launch"] = gib
        hub.set_cluster(view)

    _, proxy = served(hub_setup=with_launch)
    same = f"/lb/cluster/estimate?model_choice={FLASH}&" + "&".join(
        f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in START_99.items()
        if k != "enabled" and v is not None)
    status, body = call(proxy, "GET", same)
    assert status == 200, body
    head = body["boxes"][0]
    assert head["available_gib"] == launch["box-a"] and head["available_basis"] == "launch", head
    assert not [w for w in head["warnings"] if w["code"] in ("gate", "no_headroom")], head["warnings"]
    assert head["leaves_gib"] == round(launch["box-a"] - head["total_gib"], 1)

    _, proxy = served(hub_setup=serving_99())                   # no launch figure: free + what our rank holds
    status, body = call(proxy, "GET", same)
    head = body["boxes"][0]
    assert head["available_basis"] == "free+running" and head["available_gib"] > now["box-a"] + 50, head
    assert head["credited_gib"] > 50 and head["available_gib"] == round(now["box-a"] + head["credited_gib"], 1)

    _, proxy = served(hub_setup=free_memory())                  # nothing runs: free is free
    _, body = call(proxy, "GET", same)
    assert body["boxes"][0]["available_basis"] == "free" and body["boxes"][0]["credited_gib"] == 0


# ── a KV pin past the profile's safe ceiling: refused at the switch and in the estimate, never launched ──────────────

CEILING_MAX = "19,918,363,626"


def test_a_kv_pin_past_the_profiles_ceiling_is_refused_at_the_switch_with_the_profiles_words_and_the_maximum(served):
    """GENERATION 187 (2026-10-10 20:55:20Z): this exact POST was answered 202; the launch then refused the pin on every
    tick. The switch answers 409 with the profile's own words and the maximum, and makes no generation."""
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": FLASH_SGLANG,
                                                       "kv_cache_memory": 24_000_000_000})
    assert status == 409, body
    message = body["error"]["message"]
    assert body["error"]["type"] == "cluster_rejected"
    assert "mem_fraction_static 0.885" in message and "safe 0.85 ceiling" in message and CEILING_MAX in message, message
    assert DesiredStore(cfg.cluster_state_path).load().mode == "classic"
    # A serving pair is not torn down for it either: the restart is refused and the generation stands.
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": FLASH_SGLANG})
    assert status == 202 and body["desired"]["generation"] == 1, body
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True, "kv_cache_memory": 24_000_000_000})
    assert status == 409 and CEILING_MAX in body["error"]["message"], body
    assert DesiredStore(cfg.cluster_state_path).load().generation == 1
    # The maximum itself is taken.
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "restart": True, "kv_cache_memory": 19_918_363_626})
    assert status == 202 and body["desired"]["kv_cache_memory"] == 19_918_363_626, body


def test_the_estimate_refuses_a_kv_pin_past_the_profiles_ceiling_in_the_profiles_words(served):
    """The estimate is where a refusal has to be READABLE while the knob can still be turned."""
    _, proxy = served(hub_setup=free_memory())
    base = f"/lb/cluster/estimate?model_choice={FLASH_SGLANG}&max_num_seqs=16&max_model_len=262144"
    status, body = call(proxy, "GET", f"{base}&kv_cache_memory=24000000000")
    assert status == 400, body
    assert body["error"]["type"] == "cluster_rejected" and CEILING_MAX in body["error"]["message"], body
    assert "safe 0.85 ceiling" in body["error"]["message"]
    status, body = call(proxy, "GET", f"{base}&kv_cache_memory=18000000000")
    assert status == 200 and body["kv_cache_memory"] == 18_000_000_000, body


def test_auto_for_a_profile_with_a_ceiling_never_sizes_the_pool_past_it(served):
    """`auto` is the largest pool that fits EVERY bound the LB knows: the range, the boxes, and — on a profile that
    has one — its safe ceiling, rounded down to the panel's step. Boxes with room for 27 GB are sized at 19 GB."""
    def roomy(hub):                                   # both boxes observed at 125 GiB on every recent sample
        hub.set_cluster({"configured": True, "state": "off", "held": False, "max_depth": 16,
                         "desired": {"mode": "classic"},
                         "boxes": {b: {"mem_available_gib": 125.0, "mem_stable_gib": 125.0} for b in ("box-a", "box-b")}})
    cfg, proxy = served(hub_setup=roomy)
    base = f"/lb/cluster/estimate?model_choice={FLASH_SGLANG}&max_num_seqs=16&max_model_len=262144"
    status, body = call(proxy, "GET", f"{base}&kv_cache_memory=auto")
    assert status == 200, body
    assert body["auto"]["kv_cache_memory"] == 19_000_000_000, body["auto"]
    assert body["kv_cache_memory"] == 19_000_000_000 and body["kv_cache_auto"] is True
    # …and a Start at auto stores exactly that, so the gate, the launch and the page see one number under the ceiling.
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "model_choice": FLASH_SGLANG,
                                                       "kv_cache_memory": "auto"})
    assert status == 202, body
    assert (body["desired"]["kv_cache_memory"], body["desired"]["kv_cache_auto"]) == (19_000_000_000, True)
    assert DesiredStore(cfg.cluster_state_path).load().kv_cache_memory == 19_000_000_000
