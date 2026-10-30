"""S2 — `/v1/models` is synthesized: what a client would get from the backend it would hit."""
from vllm_lb.config import Config
from vllm_lb.models_view import models_payload
from vllm_lb.state import Backend, Lease, Snapshot

NOW = 10.0


def test_a_live_vllm_reports_its_own_ids_and_window():
    v = Backend(kind="vllm", url="http://a:8000", host_label="box-a-gpu", box="box-a", live=True,
                served_models=("qwen3.8:latest",), max_model_len=32768)
    s = Snapshot(now=NOW, backends=[v],
                 leases=[Lease(lease_id="r", job_id="j", created_at=0, status="running",
                               host_label="box-a-gpu", box="box-a")],
                 last_request_at=None, last_pressure_at=None, db_ok=True)
    out = models_payload(s, Config())
    assert out["object"] == "list"
    assert [m["id"] for m in out["data"]] == ["qwen3.8:latest"]
    assert out["data"][0]["max_model_len"] == 32768


def test_nothing_live_lists_no_model_never_a_stand_in():
    """No fallback (retired 2026-10-17): with nothing live there is nothing a client could be served — an empty list."""
    s = Snapshot(now=NOW, backends=[], leases=[], last_request_at=None,
                 last_pressure_at=None, db_ok=False)
    out = models_payload(s, Config())
    assert out["object"] == "list" and out["data"] == []


def test_every_model_a_running_server_serves_is_listed_once_the_defaults_first(tmp_path):
    """Live 2026-10-17 (box-c): /v1/models listed only the default route's models, so a client discovering models
    never saw MAI-UI while the gui node served it. Every model a request naming it would be ROUTED to now (the router's
    own rule: held, its own cluster view) is listed once, with its server and that server's window — the DEFAULT
    route's first (fullclaudevllm sizes Claude Code from data[0]; the chat budgets an unnamed turn to it). A server not
    held lists nothing."""
    from dataclasses import replace

    from tests.test_parallel_clusters import HELD, _cluster, _snap, _two
    gui = replace(_cluster("http://gui:8000", "gui", "Tongyi-MAI/MAI-UI-2B"), max_model_len=32768)
    pair = replace(_cluster("http://pair:8100", "pair", "Qwen/Qwen3.8-27B"), max_model_len=131072)
    cfg = replace(_two(tmp_path), broker_enabled=False)
    out = models_payload(_snap(gui, pair, extra={"cluster": HELD, "clusters": {"pair": HELD, "gui": HELD}}), cfg)
    assert [(m["id"], m["max_model_len"], m["server"]) for m in out["data"]] == [
        ("Qwen/Qwen3.8-27B", 131072, "pair"), ("Tongyi-MAI/MAI-UI-2B", 32768, "gui")]
    assert all(m["object"] == "model" and m["owned_by"] == "vllm-lb" for m in out["data"])
    off = {"configured": True, "held": False, "max_depth": 8, "desired": {"mode": "off"}}
    out = models_payload(_snap(gui, pair, extra={"cluster": HELD, "clusters": {"pair": HELD, "gui": off}}), cfg)
    assert [m["id"] for m in out["data"]] == ["Qwen/Qwen3.8-27B"], "a server not held: nothing a client could reach"
    both = replace(gui, served_models=("Tongyi-MAI/MAI-UI-2B", "Qwen/Qwen3.8-27B"))
    out = models_payload(_snap(both, pair, extra={"cluster": HELD, "clusters": {"pair": HELD, "gui": HELD}}), cfg)
    assert [m["id"] for m in out["data"]].count("Qwen/Qwen3.8-27B") == 1, "each model once"
