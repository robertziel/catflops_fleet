"""THE MODEL LIBRARY SPLIT (2026-10-30; vllm_lb/catalog.py `document`, docs/design/engines.md).

What one catalog.json row held is four things with four owners now:
  library/models.json        what the checkpoint IS (its config.json's facts, label, served name)
  library/recipes.json       how it is launched, independent of hardware (engine, build, parsers, flags; overrides per
                             accelerator where one needs its own)
  library/measurements.json  what was measured, keyed by what it was measured ON (engine × accelerator × TP)
  the fleet file             where THIS deployment staged it (`checkpoints.<id>`: path, uncensored twin, drafter)

The regression: for the reference fleet the assembled document IS the catalog the pair ran before the split
(tests/fixtures/catalog-9a58e08a.json, that commit's catalog.json, its provenance text kept in step with the library) —
the same rows, the same values; the only additions are each row's `measured_on` (what its measurement covers) and the document's `absent` (nothing, here).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_lb import catalog as catalog_mod
from vllm_lb.catalog import (CLUSTER_MODE, LIBRARY_DIR, ModelCatalog, current_deployment, document)
from vllm_lb.fleet import catalog_deployment_of, parse_fleet

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BEFORE = json.loads((FIXTURES / "catalog-9a58e08a.json").read_text())
#: Facts CORRECTED since that commit by reading the checkpoint (python -m vllm_lb.model_facts --check, 2026-10-30): a
#: model fact the resolver checks, never a launch setting. DeepSeek-V4-Flash's config.json says `fp8`; `deepseek_v4_fp8`
#: is the B12X image's own name for it (its quant_method, unchanged).
CORRECTED = {("deepseek-v4-flash-nvfp4", "quant_format"): "fp8"}
#: Traffic since that commit (2026-10-30, operator "Fix bandwidth in tensor fold"): every `traffic` names its family
#: (catalog `TRAFFIC_FAMILIES`), TensorFold's NVFP4 row gained a routed one (tests/test_tensorfold_bandwidth_model.py),
#: and that row's `architectures` — the same value — is now READ off its config.json rather than inferred.
TRAFFIC_FAMILY = {"qwen3.8-27b-sglang-dflash2": "dense", "qwen3.8-flash-next-tensorfold-nvfp4": "routed"}
READ_SINCE = {("qwen3.8-flash-next-tensorfold-nvfp4", "architectures")}


def _rows(doc: dict) -> dict:
    return {m["id"]: m for m in doc["models"]}


# ── the regression: the pair's catalog is unchanged ─────────────────────────────────────────────────

def test_the_reference_fleets_document_is_the_catalog_the_pair_ran_before_the_split():
    doc = document()
    assert doc["absent"] == {}
    got, want = _rows(doc), _rows(BEFORE)
    assert list(got) == list(want), "the same models, in the same order"
    for mid, row in got.items():
        row = dict(row)
        covers = row.pop("measured_on", None)
        row.pop("shape", None)          # the config.json layer shapes the estimate reads: new, a model fact
        for (fixed, key), value in CORRECTED.items():
            if fixed == mid:
                assert row.pop(key)["value"] == value, (mid, key)
                want[mid] = {k: v for k, v in want[mid].items() if k != key}
        if mid in TRAFFIC_FAMILY:
            traffic = dict(row.pop("traffic"))
            assert traffic.pop("family") == TRAFFIC_FAMILY[mid], mid
            if "traffic" in want[mid]:
                assert traffic == want[mid]["traffic"], mid
            want[mid] = {k: v for k, v in want[mid].items() if k != "traffic"}
        for fixed, key in READ_SINCE:
            if fixed == mid:
                assert row.pop(key)["value"] == want[mid][key]["value"], (mid, key)
                want[mid] = {k: v for k, v in want[mid].items() if k != key}
        assert row == want[mid], mid
        if "memory" in want[mid] or "engine_memory" in want[mid]:
            assert covers and covers["covers"], mid


def test_every_cluster_row_is_measured_on_the_reference_pair():
    catalog = ModelCatalog.load()
    for s in catalog.for_mode(CLUSTER_MODE):
        r = s.profile(CLUSTER_MODE).resolution
        assert (r.sizing, r.refused) == ("measured", ""), (s.id, r.refused)
        assert r.measurement.endswith("× nvidia-sm121 × TP=2"), (s.id, r.measurement)


# ── each file holds only its own axis ───────────────────────────────────────────────────────────────

def test_models_hold_facts_and_no_deployment_or_launch():
    for m in json.loads((LIBRARY_DIR / "models.json").read_text())["models"]:
        assert not set(m) & {"path", "memory", "engine_memory", "launch", "node_engine"}, m["id"]
        assert {"format", "architectures", "quant_format", "quant_method"} <= set(m), m["id"]
        assert "path" not in (m.get("uncensored") or {}), m["id"]


def test_recipes_name_a_build_never_an_image_a_folder_or_a_measurement():
    for r in json.loads((LIBRARY_DIR / "recipes.json").read_text())["recipes"]:
        for mode, p in r["launch"].items():
            assert "build" in p and not set(p) & {"image", "draft_path", "tuned", "overhead_gib"}, (r["model"], mode)


def test_measurements_say_what_they_were_measured_on():
    for e in json.loads((LIBRARY_DIR / "measurements.json").read_text())["measurements"]:
        assert e["covers"] and all(set(c) == {"engine", "accelerator", "tp"} for c in e["covers"]), e["model"]
        assert e["covers_source"], e["model"]


def test_deepseeks_b12x_kernels_are_its_gb10_override_alone():
    r = next(r for r in json.loads((LIBRARY_DIR / "recipes.json").read_text())["recipes"]
             if r["model"] == "deepseek-v4-flash-nvfp4")
    assert "attention_backend" not in r["launch"]["cluster"]
    assert r["overrides"]["nvidia-sm121"]["attention_backend"] == "B12X"
    dep = current_deployment()
    other = replace(dep, targets={**dep.targets, CLUSTER_MODE: "no cluster"})
    row = _rows(document(other))["deepseek-v4-flash-nvfp4"]
    assert "attention_backend" not in row["launch"]["cluster"], "applied on its accelerator only"
    assert _rows(document(dep))["deepseek-v4-flash-nvfp4"]["launch"]["cluster"]["attention_backend"] == "B12X"


# ── where a checkpoint sits is the deployment's ─────────────────────────────────────────────────────

NO_CHECKPOINTS = ("panel: {style: vibe}\nboxes:\n  solo: {hostname: s, platform: linux/amd64, cards: [{index: 0, type: rtx5090, "
                  "serves: true}]}\nclusters: {}\nbackends: {}\ncheckpoints: {}\n")


def test_a_model_the_fleet_names_no_folder_for_is_absent_and_its_start_refused_naming_the_key(tmp_path):
    f = tmp_path / "fleet.yml"
    f.write_text(NO_CHECKPOINTS)
    catalog = ModelCatalog.load(deployment=catalog_deployment_of(f))
    assert not catalog.error and catalog.specs == ()
    assert "checkpoints.qwen3.8-27b-dflash2.path" in catalog.absent["qwen3.8-27b-dflash2"]
    with pytest.raises(ValueError) as e:
        catalog.choice(CLUSTER_MODE, "qwen3.8-27b-dflash2")
    assert "checkpoints.qwen3.8-27b-dflash2.path" in str(e.value)


def test_a_node_serving_a_model_its_fleet_did_not_stage_is_refused_naming_the_key():
    text = NO_CHECKPOINTS.replace("clusters: {}", "clusters:\n  node:\n    control: none\n    model: mai-ui-2b\n"
                                  "    url: http://127.0.0.1:8000\n    ranks: [{box: solo}]")
    with pytest.raises(ValueError) as e:
        parse_fleet(text, "fleet.yml")
    assert "checkpoints.mai-ui-2b.path" in str(e.value), str(e.value)


def test_a_drafter_checkpoint_the_recipe_needs_must_be_staged(tmp_path):
    dep = current_deployment()
    staged = dict(dep.checkpoints)
    staged["qwen3.8-27b-dflash2"] = replace(staged["qwen3.8-27b-dflash2"], draft=None)
    with pytest.raises(ValueError, match="checkpoints.qwen3.8-27b-dflash2.draft"):
        document(replace(dep, checkpoints=staged))


def test_a_checkpoint_for_a_model_the_library_does_not_know_is_refused():
    dep = current_deployment()
    staged = {**dep.checkpoints, "qwen9": catalog_mod.Checkpoint("qwen9")}
    with pytest.raises(ValueError, match="qwen9"):
        document(replace(dep, checkpoints=staged))


# ── a moved key is refused naming where it lives now ────────────────────────────────────────────────

def test_a_leftover_catalog_json_is_refused_naming_the_split(tmp_path, monkeypatch):
    old = tmp_path / "catalog.json"
    old.write_text("{}")
    monkeypatch.setattr(catalog_mod, "RETIRED_CATALOG", old)
    c = ModelCatalog.load()
    assert "retired" in c.error and "models.json" in c.error


@pytest.mark.parametrize("file, edit, says", [
    ("models.json", lambda d: d["models"][0].update(path="x"), "checkpoints.<id>.path"),
    ("models.json", lambda d: d["models"][0].update(memory={}), "measurements.json"),
    ("recipes.json", lambda d: d["recipes"][0]["launch"]["cluster"].update(image="x:1"), "image matrix"),
    ("recipes.json", lambda d: d["recipes"][0]["launch"]["cluster"].update(draft_path="x"), "checkpoints.<id>.draft"),
    ("recipes.json", lambda d: d["recipes"][0]["launch"]["cluster"].update(tuned={}), "measurements.json"),
])
def test_a_moved_key_written_in_the_library_is_refused_naming_its_new_home(tmp_path, monkeypatch, file, edit, says):
    lib = tmp_path / "library"
    shutil.copytree(LIBRARY_DIR, lib)
    doc = json.loads((lib / file).read_text())
    edit(doc)
    (lib / file).write_text(json.dumps(doc))
    monkeypatch.setattr(catalog_mod, "LIBRARY_DIR", lib)
    with pytest.raises(ValueError) as e:
        document()
    assert says in str(e.value), str(e.value)


def test_every_committed_example_fleet_stages_every_model_it_names():
    """An example a user copies must boot: each names a checkpoint folder for every model its clusters serve or offer
    (read from the file itself — the suite's scaffolding, conftest `reference_checkpoints`, never reaches this)."""
    from vllm_lb import yamlfile
    lb = Path(__file__).resolve().parent.parent
    for f in sorted(lb.glob("fleet*.example.yml")):
        raw = yamlfile.read(f)
        staged = set(raw.get("checkpoints") or {})
        named = set()
        for c in raw["clusters"].values():
            for ref in [c.get("model"), *((c.get("launch") or {}).get("choices") or ()), *(c.get("choices") or ())]:
                if isinstance(ref, str):
                    named.add(ref.split("@")[0])
        assert named <= staged, (f.name, sorted(named - staged))
        if any(c.get("control") == "ssh" for c in raw["clusters"].values()):
            assert staged, f.name
