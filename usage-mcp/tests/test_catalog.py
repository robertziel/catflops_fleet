"""The recorded configuration, read from the fleet's own model library (lb/vllm_lb/library)."""
from __future__ import annotations

import json

import pytest

from usage_mcp import catalog as catalog_mod
from usage_mcp.readonly import ReadOnlyViolation, Sources, SourceUnavailable


def test_the_real_catalog_loads_and_names_its_engines(src: Sources):
    doc = catalog_mod.load(src)
    engines = catalog_mod.engines(doc)
    assert engines, doc.keys()
    assert set(engines.values()) <= {"vllm", "sglang", "tensorfold"}, engines
    assert any(v == "sglang" for v in engines.values()), engines
    assert any(v == "vllm" for v in engines.values()), engines


def test_every_catalog_entry_shapes_without_losing_the_calibration(src: Sources):
    entries = catalog_mod.index(catalog_mod.load(src))
    assert len(entries) >= 4, sorted(entries)
    calibrated = 0
    for name, entry in entries.items():
        shaped = catalog_mod.shape_one(entry)
        assert shaped["id"] == name
        assert shaped["engine"] in ("vllm", "sglang", "tensorfold")
        assert shaped["limits"]["max_model_len"], name
        if shaped["memory"]["calibration"]:
            calibrated += 1
            assert shaped["memory"]["kv_bytes_per_token"], name
    assert calibrated >= 2, "the catalog records calibration runs; none survived the shaping"


def test_exactly_one_entry_is_the_cluster_default(src: Sources):
    entries = catalog_mod.index(catalog_mod.load(src))
    defaults = [catalog_mod.shape_one(e)["id"] for e in entries.values()
                if catalog_mod.shape_one(e)["is_cluster_default"]]
    assert len(defaults) == 1, defaults


def test_the_sglang_entry_carries_its_engine_arguments(src: Sources):
    entries = catalog_mod.index(catalog_mod.load(src))
    sglang = [catalog_mod.shape_one(e) for e in entries.values()
              if catalog_mod.shape_one(e)["engine"] == "sglang"]
    assert sglang, sorted(entries)
    one = sglang[0]
    assert one["launch"]["build"], one          # the image is per deployment: /lb/status resolves it
    assert one["launch"]["num_speculative_tokens"], one


def test_the_live_block_picks_only_configuration_keys(real_status: dict):
    shaped = catalog_mod.shape_live(real_status["cluster"])
    assert shaped["loaded_model_choice"] == "qwen3.8-27b-sglang-dflash2"
    assert shaped["effective"]["max_num_seqs"] == 16
    assert shaped["effective"]["generation"] == 123
    assert shaped["defaults"] and shaped["limits"] and shaped["recommended"]
    for block in ("effective", "desired", "defaults", "recommended", "limits"):
        extra = set(shaped[block]) - set(catalog_mod._LIVE_KEYS)
        assert not extra, (block, extra)


def test_the_live_block_is_none_when_the_lb_said_nothing():
    assert catalog_mod.shape_live(None) is None
    assert catalog_mod.shape_live({}) is None


def test_the_catalog_is_not_reachable_outside_the_roots(src: Sources, state_dir):
    import dataclasses
    elsewhere = dataclasses.replace(src, catalog=state_dir.parent / ".env")
    # The catalog path itself is allowlisted by identity, so a repointed catalog is read - what is
    # structural is that it must still be JSON a model catalog, and nothing else is reachable.
    with pytest.raises(SourceUnavailable):
        catalog_mod.load(elsewhere)
    with pytest.raises(ReadOnlyViolation):
        from usage_mcp import readonly
        readonly.read_text(state_dir.parent / ".env", src)


def test_a_non_catalog_json_is_refused(src: Sources, tmp_path):
    import dataclasses
    bogus = tmp_path / "library"
    bogus.mkdir()
    for name in ("models.json", "recipes.json", "measurements.json"):
        (bogus / name).write_text(json.dumps({"not": "a catalog"}))
    with pytest.raises(SourceUnavailable):
        catalog_mod.load(dataclasses.replace(src, catalog=bogus))


def test_the_retired_catalog_env_is_refused_naming_its_replacement():
    with pytest.raises(ValueError, match="USAGE_MCP_LIBRARY"):
        Sources.from_env({"USAGE_MCP_CATALOG": "/catalog.json"})
