"""REFERENCE SETUPS — the blueprints (vllm_lb/blueprints.py; docs/design/engines.md "Blueprints") — and that they are
NOT a runtime object.

Operator, 2026-10-30: "Blueprints meant to be only for agent to build their own fleet settings, not something user
chooses." A blueprint is one of the project's real, proven setups by hardware type, shipped in
library/blueprints/<id>.yml, which a coding agent running the setup wizard (SETUP.md; deploy/setup) uses to PREFILL a
deployment's own fleet answers. The LB never starts one, the panel never offers one, the API never takes one.

The pins:
  * THE LIBRARY: the shipped blueprints are the project's two production setups, each checked against the library —
    explicit-complete (every knob present, null only where the engine does not take it; a node's null is vLLM's own),
    the model and its twin, the image matrix, measured or explicitly estimated.
  * NOT THE RUNTIME: no LB module imports blueprints.py; a fleet naming `clusters.<name>.blueprint` or `blueprints:` is
    refused at boot naming what replaced it (and so is `node-env`, the host applier's entry point); a switch body naming
    `blueprint` is a 400; GET /lb/clusters/<name>/blueprints is gone; /lb/status carries no blueprint, drift or list;
    the panel ships no picker.
  * THE STORED STATE an LB of the blueprint era wrote — a pair's switch file or a node's wish carrying the provenance
    field `blueprint` — still loads: the field is removed from the file ONCE, with a log line naming it, and nothing
    else changes (it never launched anything).
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path

import pytest

from vllm_lb import blueprints as bps
from vllm_lb.blueprints import Blueprint, check_library, parse

PAIR_BP = "2x-gb10-qwen3.8-flash-next-hibrid48"
NODE_BP = "1x-rtx5090-qwen3.8-27b-nvfp4"
LB_DIR = Path(__file__).resolve().parents[1]


def raw_of(bid: str) -> dict:
    from vllm_lb import yamlfile
    return copy.deepcopy(yamlfile.read(bps.BLUEPRINTS_DIR / f"{bid}.yml"))


def checked(bid: str, raw: dict) -> Blueprint:
    bp = parse(bid, raw, f"test {bid}")
    check_library(bp)
    return bp


# ── the shipped library ──────────────────────────────────────────────────────────────────────────


def test_the_library_ships_the_two_production_setups():
    lib = bps.read_library()
    assert set(lib) == {PAIR_BP, NODE_BP}
    pair, node = lib[PAIR_BP], lib[NODE_BP]
    assert (pair.target.accelerator, pair.target.tp, pair.target.control) == ("nvidia-sm121", 2, "ssh")
    assert (pair.model, pair.uncensored, pair.estimated) == ("qwen3.8-flash-next-hibrid48", True, False)
    assert dict(pair.knobs) == {"max_num_seqs": 12, "max_model_len": 131072, "max_num_batched_tokens": 4096,
                                "num_speculative_tokens": 4, "kv_cache_dtype": "fp8_e4m3",
                                "kv_cache_memory": 28_000_000_000, "ple_replicate": False, "kv_offload_gib": 0,
                                "long_prefill_token_threshold": 0}
    assert (node.target.accelerator, node.target.tp, node.target.control) == ("nvidia-sm120", 1, "local")
    assert node.ref == "qwen3.8-27b-nvidia-nvfp4@uncensored"
    assert dict(node.knobs) == {"max_num_seqs": 32, "max_model_len": 98304, "gpu_memory_utilization": 0.98,
                                "max_num_batched_tokens": None, "kv_cache_memory": 8_501_810_176,
                                "kv_offload_ram_gib": 32, "kv_cache_dtype": "fp8_e4m3"}
    assert dict(node.fixed) == {"reasoning_parser": "qwen3", "tool_call_parser": "qwen3_coder"}
    for bp in lib.values():
        assert "production" in bp.source and bp.label


def test_the_pair_blueprint_writes_what_the_row_resolves_to(monkeypatch, tmp_path):
    """The pair's stored intent left every engine knob null; the blueprint writes each as what it resolved to — so a
    deployment prefilled from it starts what generation 285 launched, and does not move when a library row does."""
    from tests.fleet_files import PAIR, write
    from vllm_lb.catalog import ModelCatalog
    from vllm_lb.cluster_exec import resolved_launch
    monkeypatch.setenv("LB_FLEET", write(tmp_path, PAIR))
    monkeypatch.setenv("MODELS_SEARCH_DIRS", "/models")
    spec = ModelCatalog.load().get("qwen3.8-flash-next-hibrid48")
    nulls = resolved_launch(spec, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"}, max_num_seqs=12, max_model_len=131072)
    knobs = dict(bps.read_library()[PAIR_BP].knobs)
    assert {k: nulls[k] for k in knobs} == knobs


def test_the_node_blueprints_image_is_in_the_matrix_for_its_accelerator():
    from vllm_lb.images import image_for
    entry = image_for("vllm", "priority", "nvidia-sm120", "linux/amd64")
    assert "7b35cbcf2a880e0e61233848f038417a04b1a94950ac691ae901948abff7f360" in entry.source


@pytest.mark.parametrize("bid, edit, says", [
    (PAIR_BP, lambda r: r["knobs"].pop("kv_offload_gib"), "missing kv_offload_gib"),
    (PAIR_BP, lambda r: r["knobs"].update(gpu_memory_utilization=0.9), "gpu_memory_utilization is no knob"),
    (PAIR_BP, lambda r: r["knobs"].update(max_num_batched_tokens=None), "is null — write the value"),
    (PAIR_BP, lambda r: r.pop("source"), "missing source"),
    (PAIR_BP, lambda r: r["target"].update(accelerator="nvidia-sm999"), "is no accelerator"),
    (PAIR_BP, lambda r: r["target"].update(control="none"), "control the LB starts"),
    (PAIR_BP, lambda r: r.update(model="no-such-model"), "no model 'no-such-model'"),
    (PAIR_BP, lambda r: r.update(model="glm-5.3-flash-nvfp4"), "no uncensored twin"),
    (PAIR_BP, lambda r: r["knobs"].update(max_num_seqs=49), "max_num_seqs must be from 1 to 48"),
    (PAIR_BP, lambda r: r["target"].update(tp=4), "not measured on vllm × nvidia-sm121 × 4"),
    (PAIR_BP, lambda r: r.update(estimated=True), "estimated must be false"),
    # A reference setup is generic: it names a hardware TYPE, never one deployment's cluster.
    (PAIR_BP, lambda r: r["target"].update(cluster="pair"), "unknown cluster"),
    (NODE_BP, lambda r: r["knobs"].pop("kv_offload_ram_gib"), "missing kv_offload_ram_gib"),
    (NODE_BP, lambda r: r.pop("fixed"), "missing fixed"),
    (NODE_BP, lambda r: r["fixed"].pop("tool_call_parser"), "fixed names exactly reasoning_parser, tool_call_parser"),
    (NODE_BP, lambda r: r["target"].update(accelerator="amd-gfx1151"), "no vllm image for amd-gfx1151"),
])
def test_a_bad_blueprint_is_refused_naming_it_and_why(bid, edit, says):
    raw = raw_of(bid)
    edit(raw)
    with pytest.raises(ValueError, match=f"{bid}|test {bid}") as exc:
        checked(bid, raw)
    assert says in str(exc.value)


def test_a_knob_the_engine_does_not_take_must_be_null():
    """SGLang reads no PLE placement: a blueprint saying false there would be prefilled and dropped (catalog.ENGINE_REFUSES)."""
    raw = raw_of(PAIR_BP)
    raw.update(model="qwen3.8-flash-next-sglang")
    raw["knobs"].update(kv_cache_memory=18_000_000_000)
    with pytest.raises(ValueError, match="knobs.ple_replicate must be null: the PLE placement"):
        checked("sglang-try", raw)
    raw["knobs"].update(kv_offload_gib=None, long_prefill_token_threshold=None, ple_replicate=None)
    checked("sglang-try", raw)                                 # every refused knob null: it is complete


def test_an_unmeasured_target_is_taken_only_with_the_estimate_said():
    raw = raw_of(PAIR_BP)
    raw["target"]["tp"] = 4
    raw["estimated"] = True
    assert checked("four", raw).estimated is True


# ── not the runtime ───────────────────────────────────────────────────────────────────────────────


def test_no_runtime_module_imports_the_blueprints():
    """The LB reads no blueprint: only the library check (and the tests) import blueprints.py."""
    hits = [p.relative_to(LB_DIR).as_posix() for p in (LB_DIR / "vllm_lb").rglob("*.py")
            if p.name != "blueprints.py" and re.search(r"\bblueprints\b", p.read_text(encoding="utf-8"))
            and re.search(r"^\s*(from vllm_lb import .*\bblueprints\b|from vllm_lb\.blueprints|import vllm_lb\.blueprints)",
                          p.read_text(encoding="utf-8"), re.M)]
    assert hits == []


@pytest.mark.parametrize("edit, says", [
    (lambda t: t.replace("    control: ssh\n", f"    control: ssh\n    blueprint: {PAIR_BP}\n", 1),
     "clusters.pair.blueprint is retired"),
    (lambda t: t.replace("    control: local\n", f"    control: local\n    blueprint: {NODE_BP}\n", 1),
     "clusters.gui.blueprint is retired"),
    (lambda t: t + "blueprints:\n  mine: {label: x}\n", "blueprints is retired"),
])
def test_a_fleet_still_naming_a_blueprint_is_refused_at_boot_naming_what_replaced_it(edit, says):
    from tests.fleet_files import PARALLEL
    from vllm_lb.fleet import parse_fleet
    with pytest.raises(ValueError) as exc:
        parse_fleet(edit(PARALLEL), "fleet.yml")
    msg = str(exc.value)
    assert says in msg and "deploy/setup" in msg and "remove this key" in msg, msg


def test_the_node_env_command_refuses_a_fleet_naming_a_blueprint(tmp_path):
    """`python -m vllm_lb.fleet node-env` — the host applier's entry point — reads the same file: refused the same way."""
    import os
    import subprocess
    import sys
    from tests.fleet_files import PARALLEL, write
    path = write(tmp_path, PARALLEL.replace("    control: local\n", f"    control: local\n    blueprint: {NODE_BP}\n", 1))
    r = subprocess.run([sys.executable, "-m", "vllm_lb.fleet", "node-env", path, "gui"],
                       capture_output=True, text=True, cwd=LB_DIR, env=dict(os.environ))
    assert r.returncode != 0
    assert "clusters.gui.blueprint is retired" in r.stderr


@pytest.fixture
def lb(tmp_path):
    """An LB booted on fleet_files.PARALLEL (the pair and a local node `gui`) with its run, its proxy listening, and the
    node's applier listening (its report written by hand). `call(method, path, body)` → (status, json)."""
    import http.client
    from dataclasses import replace
    from tests.fleet_files import PARALLEL, write
    from vllm_lb import node_control as nc
    from vllm_lb.cluster_run import ClusterRun
    from vllm_lb.config import Config
    from vllm_lb.hub import Hub
    from vllm_lb.proxy import Proxy
    started = []

    def make(*, state: dict | None = None, wish: dict | None = None):
        cfg = Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_FLEET": write(tmp_path, PARALLEL),
                               "LB_CLUSTER_STATE": str(tmp_path / "cluster.json")})
        cfg = replace(cfg, listen_host="127.0.0.1", listen_port=0)
        for name, doc in {**(state or {}), **({"gui": wish} if wish else {})}.items():
            path = Path(cfg.for_cluster(name).cluster_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc))
        hub = Hub(cfg)
        run = ClusterRun(cfg, hub, None)
        run.start(None)
        gui = cfg.for_cluster("gui")
        node_dir = nc.node_dir(gui.cluster_state_path)
        Path(node_dir).mkdir(parents=True, exist_ok=True)
        nc.write_report(node_dir, nc.report_doc(heartbeat=hub.clock(), host="solo-host", applied=None,
                                                container=None, log=None))
        proxy = Proxy(cfg, hub, cluster_switch=None, run=run)
        proxy.start()
        started.append(proxy)

        def call(method, path, body=None):
            conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
            conn.request(method, path, body=None if body is None else json.dumps(body),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        return cfg, hub, run, call
    yield make
    for p in started:
        p.stop()


@pytest.mark.parametrize("cluster", ["pair", "gui"])
def test_a_switch_body_naming_a_blueprint_is_a_400_saying_where_blueprints_went(lb, cluster):
    """Accepted-and-ignored is not allowed (no fallbacks): the key is refused, naming the wizard, and nothing is stored."""
    cfg, _, _, call = lb()
    status, body = call("POST", f"/lb/clusters/{cluster}", {"enabled": True, "blueprint": PAIR_BP})
    assert status == 400, body
    assert body["error"]["type"] == "blueprint_retired"
    assert "deploy/setup" in body["error"]["message"]
    assert not Path(cfg.for_cluster(cluster).cluster_state_path).exists()


def test_the_blueprints_route_and_the_status_fields_are_gone(lb):
    _, _, _, call = lb()
    assert call("GET", "/lb/clusters/pair/blueprints")[0] == 404
    clusters = call("GET", "/lb/status")[1]["clusters"]
    assert clusters, "the status names its clusters"
    for name, c in clusters.items():
        assert not {"blueprint", "blueprints", "drift", "blueprint_missing"} & set(c), name
        assert "blueprint" not in (c.get("desired") or {}), name


# ── the stored state of the blueprint era ──────────────────────────────────────────────────────────


def test_a_stored_pair_selection_naming_a_blueprint_loads_and_the_field_is_removed_once(tmp_path, caplog):
    from vllm_lb.cluster import DesiredStore
    path = tmp_path / "cluster.json"
    stored = {"mode": "cluster", "generation": 285, "max_num_seqs": 12, "max_model_len": 131072,
              "model_choice": "qwen3.8-flash-next-hibrid48", "uncensored": True, "kv_cache_memory": 28_000_000_000,
              "blueprint": PAIR_BP, "start_id": "abc", "updated_at": 1.0}
    path.write_text(json.dumps(stored))
    store = DesiredStore(str(path))
    with caplog.at_level(logging.INFO):
        d = store.load()
    assert (d.mode, d.generation, d.max_num_seqs, d.max_model_len, d.kv_cache_memory) == \
        ("cluster", 285, 12, 131072, 28_000_000_000)
    assert store.current()[1] == "abc", "the start id is kept: the running generation is the same Start"
    on_disk = json.loads(path.read_text())
    assert on_disk == {k: v for k, v in stored.items() if k != "blueprint"}, "only the field goes"
    said = [r.getMessage() for r in caplog.records if "blueprint" in r.getMessage()]
    assert len(said) == 1 and PAIR_BP in said[0] and str(path) in said[0]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        store.load()
    assert not [r for r in caplog.records if "blueprint" in r.getMessage()], "once"


def test_a_stored_node_wish_naming_a_blueprint_loads_and_the_field_is_removed_once(tmp_path, caplog):
    from vllm_lb import node_control as nc
    path = tmp_path / "gui.json"
    wish = {"id": 7, "at": 1.0, "state": "up", "force": False,
            "launch": {"model": "qwen3.8-27b-nvidia-nvfp4@uncensored", "max_num_seqs": 32}, "blueprint": NODE_BP}
    path.write_text(json.dumps(wish))
    with caplog.at_level(logging.INFO):
        got = nc.IntentStore(str(path)).read()
    assert got == {k: v for k, v in wish.items() if k != "blueprint"}, "the same wish: same id, nothing re-applied"
    assert json.loads(path.read_text()) == got
    said = [r.getMessage() for r in caplog.records if "blueprint" in r.getMessage()]
    assert len(said) == 1 and NODE_BP in said[0]


def test_an_intent_with_any_other_unknown_key_is_still_refused():
    from vllm_lb import node_control as nc
    with pytest.raises(ValueError, match="unknown origin"):
        nc.parse_intent({"id": 1, "at": 1.0, "state": "up", "force": False, "launch": {}, "origin": "x"})


def test_an_lb_restarted_on_blueprint_era_state_serves_it(lb):
    stored = {"mode": "classic", "generation": 7, "max_num_seqs": 6, "max_model_len": 262144,
              "model_choice": "qwen3.8-flash-next-hibrid48", "blueprint": PAIR_BP}
    wish = {"id": 3, "at": 1.0, "state": "down", "force": False,
            "launch": {"model": "qwen3.8-27b-nvidia-nvfp4"}, "blueprint": NODE_BP}
    from vllm_lb import node_control as nc
    cfg, _, run, _ = lb(state={"pair": stored}, wish=wish)
    d = run.switches["pair"].load()
    assert (d.mode, d.generation, d.max_num_seqs) == ("classic", 7, 6)
    assert nc.IntentStore(cfg.for_cluster("gui").cluster_state_path).read()["id"] == 3
    for name in ("pair", "gui"):
        assert "blueprint" not in json.loads(Path(cfg.for_cluster(name).cluster_state_path).read_text())


# ── the panel ships no picker ─────────────────────────────────────────────────────────────────────


def test_the_panel_ships_no_blueprint_picker():
    static = LB_DIR / "vllm_lb" / "static"
    assert not (static / "blueprints.js").exists()
    assert not (LB_DIR / "tests" / "blueprints.node.js").exists()
    for f in sorted(static.glob("*")):
        if f.suffix in (".html", ".js", ".css"):
            assert not re.search(r"blueprint", f.read_text(encoding="utf-8"), re.I), f.name
    assert "blueprint" not in (LB_DIR / "Dockerfile").read_text().lower()
