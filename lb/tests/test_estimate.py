"""AN UNMEASURED COMBINATION IS REFUSED BY DEFAULT; THE ESTIMATE IS THE OPERATOR'S EXPLICIT CHOICE (docs/design/engines.md).

A model's memory, tuned input and overhead are measured on an engine × accelerator × TP (library/measurements.json
`covers`). A cluster whose cards are another target refuses its Start, "not measured on …" — the default. The operator
may still start it, EXPLICITLY, sized by the estimate from its config.json (catalog.estimated_footprint): the API's
`estimated: true`, the panel's control — stored per Start (`Desired.estimated`), reset by a model change, labelled
"estimated" wherever it shows, never chosen by the LB on anyone's behalf (that would be a fallback).
"""
from __future__ import annotations

import pytest

from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog, document
from vllm_lb.cluster import DesiredStore
from vllm_lb.cluster_exec import launch_problem
from vllm_lb.hardware import accelerator
from vllm_lb.resolve import Target
from tests.test_cluster_api import served  # noqa: F401  (the fixture)

FLASH, DENSE, GLM, TF = ("qwen3.8-flash-next-hibrid48", "qwen3.8-27b-dflash2", "glm-5.3-flash-nvfp4",
                         "qwen3.8-flash-next-tensorfold-nvfp4")
#: Four GB10s at TP=4: the images exist (nvidia-sm121 × linux/arm64), the measurements are TP=2's.
TP4 = Target(accelerator("nvidia-sm121"), "linux/arm64", 4, source="test: four GB10s")


def _catalog(target=TP4) -> ModelCatalog:
    c = ModelCatalog.from_dict(document(), targets={CLUSTER_MODE: target, "classic": "none here"})
    assert not c.error, c.error
    return c


def test_an_unmeasured_target_is_refused_by_default_naming_what_was_measured():
    p = _catalog().get(DENSE).profile(CLUSTER_MODE)
    assert p.resolution.sizing == "unmeasured"
    assert "not measured on nvidia-sm121 at TP=4" in p.refused and "TP=2" in p.refused, p.refused
    assert p.image == "myllmbox/qwen38-flash-next-cluster-vllm:v5", "the image resolves; only the sizing is missing"
    with pytest.raises(ValueError, match="not measured on nvidia-sm121 at TP=4"):
        _catalog().choice(CLUSTER_MODE, DENSE)


def test_the_operator_may_start_it_explicitly_sized_by_the_estimate():
    spec = _catalog().choice(CLUSTER_MODE, FLASH, estimated=True)
    m = spec.memory
    # 98.1 GiB on disk / 4; 2 x 12 full-attention layers x max(1, 2 KV heads // 4) x 256 head_dim x fp8_e4m3's 1 B.
    assert m.weights_gib == round(98.1 / 4, 3)
    assert m.kv_bytes_per_token == 2 * 12 * 1 * 256 * 1
    assert m.calibration == "", "an estimate is never a calibration: nothing is SIZED from it as if measured"
    assert not spec.calibrated
    how = spec.profile(CLUSTER_MODE).resolution.estimate
    assert how.startswith("ESTIMATED, NOT MEASURED") and "/ TP 4" in how


def test_the_estimate_is_offered_only_where_no_measurement_covers_the_target():
    measured = Target(accelerator("nvidia-sm121"), "linux/arm64", 2, source="test: the pair")
    with pytest.raises(ValueError, match="measured on these cards"):
        _catalog(measured).choice(CLUSTER_MODE, DENSE, estimated=True)


@pytest.mark.parametrize("model, says", [
    (DENSE, "names no KV dtype"),                  # vLLM's auto: the estimate prices a dtype the launch names
    (GLM, "shapes are not in the library"),       # its MLA shapes are not stated: model_facts first
    (TF, "shapes are not in the library"),
])
def test_a_model_whose_shapes_are_not_known_cannot_be_estimated_and_says_how_to_read_them(model, says):
    with pytest.raises(ValueError) as e:
        _catalog().choice(CLUSTER_MODE, model, estimated=True)
    assert "cannot be estimated" in str(e.value) and says in str(e.value), str(e.value)


def test_an_mla_model_is_not_estimated():
    from vllm_lb.catalog import estimated_footprint
    est, why = estimated_footprint({"full_attention_layers": 45, "num_key_value_heads": 64, "head_dim": 0,
                                    "kv_lora_rank": 512}, size_gib=184.29, tp=2, kv_cache_dtype="fp8_e4m3")
    assert est is None and "MLA" in why


# ── the switch stores the choice; the launch gate holds it ──────────────────────────────────────────

def test_a_start_of_an_unmeasured_model_needs_the_explicit_choice_and_stores_it(tmp_path):
    store = DesiredStore(str(tmp_path / "desired.json"), catalog=_catalog())
    with pytest.raises(ValueError, match="not measured"):
        store.set(enabled=True, model_choice=FLASH)
    d = store.set(enabled=True, model_choice=FLASH, estimated=True)
    assert d.estimated is True and store.load().estimated is True
    stopped = store.set(enabled=False)
    assert stopped.estimated is True, "a Stop keeps the stored choice"
    # A MODEL CHANGE hands it back: it was chosen for ONE model (on the pair, where the other is measured).
    measured = DesiredStore(str(tmp_path / "desired.json"), catalog=ModelCatalog.load())
    other = measured.set(enabled=False, model_choice="qwen3.8-27b-nvidia-nvfp4")
    assert other.estimated is False


def test_estimated_must_be_a_bool(tmp_path):
    store = DesiredStore(str(tmp_path / "desired.json"), catalog=_catalog())
    with pytest.raises(ValueError, match="true or false"):
        store.set(enabled=True, model_choice=DENSE, estimated="yes")


def test_the_launch_gate_refuses_an_unmeasured_model_without_the_choice():
    spec = _catalog().get(DENSE)
    settings = dict(rank=0, gen=1, max_num_seqs=4, max_model_len=65536)
    assert "not measured" in launch_problem(spec, **settings)
    assert launch_problem(spec, estimated=True, **settings) == ""


def test_the_view_labels_the_sizing():
    v = _catalog().get(FLASH).profile(CLUSTER_MODE).view()["resolution"]
    assert v["sizing"] == "unmeasured" and v["estimate"].startswith("ESTIMATED")
    measured = ModelCatalog.load().get(DENSE).profile(CLUSTER_MODE).view()["resolution"]
    assert (measured["sizing"], measured["measurement"]) == ("measured", "vllm × nvidia-sm121 × TP=2")


def test_the_api_takes_estimated_as_a_bool_only(served):
    from tests.test_cluster_api import call
    _, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "estimated": "yes"})
    assert status == 400 and "estimated" in body["error"]["message"]


def test_the_boot_log_names_what_each_profile_resolved_to_or_why_not():
    pair = ModelCatalog.load().resolution_lines()
    assert any(ln.startswith("glm-5.3-flash-nvfp4 cluster: vllm/glm53 on nvidia-sm121 × linux/arm64 × TP=2 → "
                             "myllmbox/glm53-flash-cluster-vllm:v2 [") and ln.endswith("sized by vllm × nvidia-sm121 × TP=2")
               for ln in pair), pair
    four = _catalog().resolution_lines()
    assert any(ln.startswith(f"{FLASH} cluster: vllm/fleet on nvidia-sm121 × linux/arm64 × TP=4 REFUSED: not measured")
               for ln in four), four
