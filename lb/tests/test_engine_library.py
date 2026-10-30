"""THE ENGINE LIBRARY (vllm_lb/library/engines/<engine>.yml, read by vllm_lb/engine_library.py): what each server
software SERVES — checkpoint formats, model architectures, quant formats — as data with sources. Its behaviour stays
code (catalog.ENGINE_REFUSES, fleet.NodeEngine, engine_metrics); this is the half the resolver checks a model
against before it asks for an image (docs/design/engines.md, "Resolution at Start")."""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_lb.catalog import CLUSTER_ENGINES, NODE_ENGINES
from vllm_lb.engine_library import ENGINES_DIR, engine, engines, load_engines, model_problem


def test_every_engine_the_catalog_and_the_fleet_name_has_a_capability_file():
    assert set(engines()) == set(CLUSTER_ENGINES) | set(NODE_ENGINES)


def test_every_fact_names_its_source():
    for e in engines().values():
        for f in (e.formats, e.architectures, e.quant_formats):
            assert len(f.source) >= 20, (e.key, f)


def test_a_model_of_an_architecture_the_engine_serves_passes():
    assert model_problem(engine("vllm"), fmt="hf", architectures=("Qwen3_5ForConditionalGeneration",),
                         quant_format="modelopt_mixed") == ""


def test_an_architecture_the_engine_never_served_is_a_refusal_naming_both():
    why = model_problem(engine("sglang"), fmt="hf", architectures=("DeepseekV4ForCausalLM",),
                        quant_format="modelopt_mixed")
    assert "DeepseekV4ForCausalLM" in why and "sglang" in why


def test_a_quant_format_the_engine_has_no_method_for_is_a_refusal():
    why = model_problem(engine("tensorfold"), fmt="hf", architectures=("Qwen4ExpForConditionalGeneration",),
                        quant_format="compressed-tensors")
    assert "compressed-tensors" in why and "tensorfold" in why


def test_a_checkpoint_format_the_engine_cannot_load_is_a_refusal():
    why = model_problem(engine("faster-whisper"), fmt="hf", architectures=("WhisperForConditionalGeneration",),
                        quant_format="none")
    assert "hf" in why and "ctranslate2" in why


def test_a_format_with_no_architectures_is_checked_by_its_format_alone():
    """A CTranslate2 conversion names no architecture: the engine's own file says the format is what it is checked
    against (its `architectures` is an empty list WITH that source) — not an unknown waved through."""
    assert model_problem(engine("faster-whisper"), fmt="ctranslate2", architectures=(), quant_format="none") == ""


def test_unknown_architectures_of_an_hf_model_are_a_refusal_naming_the_tool():
    """A model whose config.json was never read cannot be checked: refused, naming how to read it — never passed."""
    why = model_problem(engine("vllm"), fmt="hf", architectures=None, quant_format="modelopt")
    assert "architectures" in why and "model_facts" in why


GOOD = (ENGINES_DIR / "sglang.yml").read_text()


@pytest.mark.parametrize("broken, says", [
    (GOOD + "colour: green\n", "colour"),
    (GOOD.replace("formats: {value: [hf]", "formats: {value: [HF Folder]"), "formats"),
    (GOOD.replace("quant_formats: {value: [modelopt_mixed], source: ", "quant_formats: {value: [modelopt_mixed], "
                  "origin: "), "source"),
])
def test_a_broken_engine_file_is_refused_naming_file_and_value(tmp_path: Path, broken, says):
    assert broken != GOOD
    (tmp_path / "sglang.yml").write_text(broken)
    with pytest.raises(ValueError) as e:
        load_engines(tmp_path)
    assert "sglang.yml" in str(e.value) and says in str(e.value)


def test_an_unknown_engine_is_refused_naming_the_ones_there_are():
    with pytest.raises(ValueError) as e:
        engine("llama.cpp")
    for known in engines():
        assert known in str(e.value)
