"""THE UNCENSORED COUNTERPART: a catalog model that has an abliterated twin on disk can be started from it.

Operator, 2026-10-07: the four cluster entries over two checkpoints (Flash-Next hibrid48 on vLLM and SGLang,
the 27B NVFP4 on vLLM and SGLang DFlash2) get an `uncensored` option, DEFAULT ON; the two entries whose
checkpoint has no twin (the bf16-head DFlash2 build, the classic seed) do not. The twins are arch-exact
(same architectures, quant_method, layer count, 262,144 window), so the whole launch block applies
unchanged and the ONE thing the flag selects is the `path` the engine loads. What it must never do is fall
back: a twin that is not on the boxes FAILS the start naming the missing dir, exactly as a missing original
does — the planner blocks on the box's availability answer for THAT checkpoint, and tp-rank's own
`find_checkpoint` refuses the launch, so there is no path by which the censored weights load in its place.

The memory block of each entry was measured on the ORIGINAL. The twin inherits the numbers (same shapes) but
its footprint says so in its own `calibration`: not a second measurement, an unverified inheritance.

2026-10-10: the Flash-Next twin is no longer edp1096's QAD build — it never loaded (generations 180 and 181,
`a Tensor with 3840 elements cannot be converted to Scalar`: 384 MXFP8 layers our SGLang has no param for).
It is myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored, the abliterated sibling of our hibrid48 from the same
publisher. The publisher ships it in compressed-tensors (FP8 weight-only GDN/QSA/shared experts, NVFP4 weight-only
routed experts and lm_head), which SGLang's fleet image cannot load (`Unsupported FusedMoe scheme`). vLLM did load
it: generations 183/184 were killed by tp-rank's memory guard in torch.compile, before the KV pool, with the PLE
table REPLICATED — not by the layout, since the ModelOpt rewrite (190/191) and the original hibrid48 (192) died the
same way and split PLE served (193; 264b09c, tests/fixtures/flash-next-vllm-ple-2026-10-11.json). What the boxes
hold under the twin's name is that checkpoint REWRITTEN into the censored one's ModelOpt MIXED_PRECISION layout by
deploy/sglang-27b-tp2/ct_to_modelopt.py — the same tensors, the same kernels, config path and tp-rank views as
the original — so BOTH Flash-Next rows offer the twin, and its note says what the rewrite copied from the
original (the experts' activation scales) rather than measured.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from tests.fakes import FakeClusterExecutor
from tests.test_cluster_api import call, served  # noqa: F401 — the proxy fixture
from tests.test_cluster_loop import world, writes  # noqa: F401 — the loop fixture and what it writes through
from vllm_lb.catalog import document, CLUSTER_MODE, ModelCatalog, availability_key
from vllm_lb.cluster import CLASSIC, Desired, DesiredStore, limits_view
from vllm_lb.cluster_exec import _availabilities, launch_env, resolved_launch
from vllm_lb.cluster_loop import EFFECTIVE_KEYS

FLASH = "qwen3.8-flash-next-hibrid48"
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
DENSE = "qwen3.8-27b-nvidia-nvfp4"
SGLANG = "qwen3.8-27b-sglang-dflash2"
DFLASH = "qwen3.8-27b-dflash2"
SEEDED = "qwen3.8-27b-nvfp4"
#: Where the twins land (the operator's download, box-a:~/models, which every cluster root reaches).
UNCENSORED = {FLASH: "qwen38-flash-next-cluster/Qwen3.8-Flash-Next-hibrid48_uncensored",
              FLASH_SGLANG: "qwen38-flash-next-cluster/Qwen3.8-Flash-Next-hibrid48_uncensored",
              DENSE: "Qwen3.8-27B-NVFP4_uncensored",
              SGLANG: "Qwen3.8-27B-NVFP4_uncensored"}
WITHOUT = (DFLASH, SEEDED)


def shipped() -> ModelCatalog:
    return ModelCatalog.load()


def raw() -> dict:
    return document()


def spec_raw(doc: dict, spec_id: str) -> dict:
    return next(m for m in doc["models"] if m["id"] == spec_id)


# ── the catalog ───────────────────────────────────────────────────────────────

def test_the_models_with_a_twin_offer_it_uncensored_by_default_and_the_ones_without_do_not():
    cat = shipped()
    assert cat.error == ""
    for spec_id, path in UNCENSORED.items():
        spec = cat.get(spec_id)
        assert spec.uncensored is not None, spec_id
        assert spec.uncensored.path == path
        assert spec.uncensored.default is True, f"{spec_id}: uncensored is the default where a twin exists"
        assert spec.uncensored_default is True
        assert spec.view()["uncensored"] == {"path": path, "default": True,
                                             "calibration": spec.uncensored.calibration}
    for spec_id in WITHOUT:
        spec = cat.get(spec_id)
        assert spec.uncensored is None, f"{spec_id} has no twin on disk and must not claim one"
        assert spec.uncensored_default is False
        assert spec.view()["uncensored"] is None


def test_the_flag_selects_the_checkpoint_the_engine_loads_and_changes_nothing_else():
    cat = shipped()
    for spec_id, path in UNCENSORED.items():
        spec = cat.get(spec_id)
        twin = spec.launched_from(True)
        assert twin.path == path and twin.launched_uncensored is True
        assert twin.id == spec.id and twin.name == path.rsplit("/", 1)[-1]
        assert twin.cluster_dirs == spec.cluster_dirs, "the twin sits beside the original: one mount"
        assert twin.launch == spec.launch and twin.limits == spec.limits
        assert spec.launched_from(False) is spec
        before, after = launch_env(spec), launch_env(twin)
        assert after["CLUSTER_MODEL_NAME"] == path.rsplit("/", 1)[-1]
        assert {k: v for k, v in after.items() if k != "CLUSTER_MODEL_NAME"} == \
            {k: v for k, v in before.items() if k != "CLUSTER_MODEL_NAME"}, spec_id
        if before.get("CLUSTER_DRAFT_NAME"):
            assert after["CLUSTER_DRAFT_NAME"] == before["CLUSTER_DRAFT_NAME"]


def test_the_twins_footprint_is_an_unverified_inheritance_not_a_second_measurement():
    cat = shipped()
    for spec_id in UNCENSORED:
        spec, twin = cat.get(spec_id), cat.get(spec_id).launched_from(True)
        assert twin.memory.calibration.startswith("NOT MEASURED ON THIS CHECKPOINT")
        assert spec.path in twin.memory.calibration, "it names the checkpoint the numbers were measured on"
        assert spec.uncensored.calibration in twin.memory.calibration
        assert "NOT CALIBRATION-VERIFIED" in spec.uncensored.calibration, "the catalog says so in its own words"
        for key in ("weights_gib", "ple_gib", "mtp_gib", "kv_bytes_per_token", "seq_state_gib"):
            assert getattr(twin.memory, key) == getattr(spec.memory, key)
        # Arch-exact: the shapes the gate sizes from are the same tensors, so the start gate still has a
        # figure to weigh — what it must not do is call it measured, and the note above is what says so.
        assert twin.calibrated == spec.calibrated


def test_the_flash_next_twin_is_the_myllmbox_build_converted_to_modelopt_on_both_engines():
    """One checkpoint dir, two rows: the twin the boxes hold is the publisher's compressed-tensors build rewritten
    into the original's ModelOpt layout, and each row's note says so — what was copied (the experts' activation
    scales, from the original), what is exact, and that none of it is a measurement."""
    cat = shipped()
    assert cat.get(FLASH).launch["cluster"].engine == "vllm"
    assert cat.get(FLASH_SGLANG).launch["cluster"].engine == "sglang"
    for spec_id in (FLASH, FLASH_SGLANG):
        note = cat.get(spec_id).uncensored.calibration
        assert cat.get(spec_id).uncensored.path == UNCENSORED[FLASH]
        assert "myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored" in note
        assert "ct_to_modelopt" in note and "modelopt" in note.lower(), spec_id
        assert "input_scale" in note, "the one thing that is neither the publisher's nor measured is named"
        assert "edp1096" not in note, "the QAD build that never loaded is gone from the boxes and the catalog"
    assert cat.get(FLASH_SGLANG).uncensored.default is True


def test_a_model_without_a_twin_cannot_be_launched_uncensored():
    cat = shipped()
    for spec_id in WITHOUT:
        with pytest.raises(ValueError, match=f"{spec_id}.*no uncensored"):
            cat.get(spec_id).launched_from(True)


def _broken(fn):
    doc = copy.deepcopy(raw())
    fn(doc)
    return doc


BROKEN = {
    "a twin that is the original": lambda d: spec_raw(d, DENSE)["uncensored"].update(path="Qwen3.8-27B-NVFP4"),
    "a twin outside the one dir tp-rank mounts": lambda d: spec_raw(d, FLASH)["uncensored"].update(
        path="Qwen3.8-Flash-Next-hibrid48_uncensored"),
    "a twin that climbs out": lambda d: spec_raw(d, DENSE)["uncensored"].update(path="../x_uncensored"),
    "a twin with no stated default": lambda d: spec_raw(d, DENSE)["uncensored"].pop("default"),
    "a default that is not a bool": lambda d: spec_raw(d, DENSE)["uncensored"].update(default="yes"),
    "a twin with no calibration note": lambda d: spec_raw(d, DENSE)["uncensored"].pop("calibration"),
    "an empty calibration note": lambda d: spec_raw(d, DENSE)["uncensored"].update(calibration=" "),
    "a twin with no path": lambda d: spec_raw(d, DENSE)["uncensored"].pop("path"),
    "a key nobody reads": lambda d: spec_raw(d, DENSE)["uncensored"].update(fallback="Qwen3.8-27B-NVFP4"),
}


@pytest.mark.parametrize("why", sorted(BROKEN))
def test_a_twin_that_breaks_a_rule_refuses_the_whole_catalog(why):
    with pytest.raises(ValueError):
        ModelCatalog.from_dict(_broken(BROKEN[why]))


def test_every_checkpoint_the_cluster_may_launch_is_asked_about_by_its_own_name():
    """The box is asked about the ORIGINAL and the TWIN as two checkpoints, keyed apart, so a planner that
    needs the twin can read the twin's answer and not the original's."""
    cat = shipped()
    asked = cat.checkpoints(CLUSTER_MODE)
    assert [s.id for s in cat.for_mode(CLUSTER_MODE)] == [s.id for s in asked[:len(cat.for_mode(CLUSTER_MODE))]]
    twins = [s for s in asked if s.launched_uncensored]
    assert sorted(s.id for s in twins) == sorted(UNCENSORED)
    assert all(s.name.endswith("_uncensored") for s in twins)
    assert len({s.availability_key for s in asked}) == len(asked)
    assert availability_key(FLASH, True) == cat.get(FLASH).launched_from(True).availability_key
    assert availability_key(FLASH, False) == FLASH == cat.get(FLASH).availability_key
    assert availability_key(FLASH, None) == FLASH, "a switch from before the knob launched the original"
    reply = {"images": {s.profile(CLUSTER_MODE).image: True for s in asked},
             "models": [{"name": s.name, "dir": None if s.launched_uncensored else s.cluster_dirs[0],
                         "searched": list(s.cluster_dirs), "unusable": ""} for s in asked]}
    seen = _availabilities(reply, asked, 1.0)
    assert seen[FLASH].present is True
    assert seen[availability_key(FLASH, True)].present is False
    assert "Qwen3.8-Flash-Next-hibrid48_uncensored/config.json" in seen[availability_key(FLASH, True)].reason()


# ── the switch ────────────────────────────────────────────────────────────────

def store(tmp_path, **kw) -> DesiredStore:
    return DesiredStore(str(tmp_path / "state" / "cluster.json"), **kw)


def write_file(tmp_path, **fields) -> None:
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def test_a_start_stores_the_models_own_default_for_the_checkpoint_visibly(tmp_path):
    """The default is RESOLVED by the Start and written, not left null for a launch to guess at: the file,
    `cluster.desired` and the planner all read the same true/false."""
    s = store(tmp_path)
    assert s.defaults.uncensored is True, "the catalog's cluster default has a twin, and the twin is the default"
    d = s.set(enabled=True)
    assert d.uncensored is True
    assert json.loads((tmp_path / "state" / "cluster.json").read_text())["uncensored"] is True
    assert s.current()[0].uncensored is True
    s.set(enabled=False)
    d = s.set(enabled=True, model_choice=DFLASH, restart=True)
    assert d.uncensored is False, "no twin: the only checkpoint it has"
    assert s.defaults_for(DFLASH).uncensored is False and s.defaults_for(FLASH).uncensored is True


def test_the_operator_overrides_the_default_and_null_hands_it_back(tmp_path):
    s = store(tmp_path)
    d = s.set(enabled=True, uncensored=False)
    assert d.uncensored is False
    s.set(enabled=False)
    assert s.current()[0].uncensored is False, "a Stop keeps the choice"
    d = s.set(enabled=True, uncensored=None)
    assert d.uncensored is True, "null is the model's default, resolved again by this Start"
    with pytest.raises(ValueError, match="uncensored must be true, false or null"):
        s.set(enabled=False, uncensored="yes")


def test_uncensored_is_refused_for_a_model_that_has_no_twin_where_it_is_given_and_at_a_start(tmp_path):
    s = store(tmp_path)
    with pytest.raises(ValueError, match=f"{DFLASH}.*no uncensored"):
        s.set(enabled=False, model_choice=DFLASH, uncensored=True)
    with pytest.raises(ValueError, match=f"{DFLASH}.*no uncensored"):
        s.set(enabled=True, model_choice=DFLASH, uncensored=True)
    # A hand-edited file that names a twin for a model with none: the Start is refused, never launched censored.
    write_file(tmp_path, mode="classic", generation=3, max_num_seqs=8, max_model_len=262144,
               model_choice=DFLASH, uncensored=True)
    with pytest.raises(ValueError, match=f"{DFLASH}.*no uncensored"):
        store(tmp_path).set(enabled=True)


def test_a_changed_model_takes_its_own_checkpoint_default(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True, model_choice=DENSE, uncensored=False)
    d = s.set(enabled=True, model_choice=FLASH, restart=True)
    assert d.uncensored is True, "the override was for the 27B; Flash-Next starts at its own default"
    d = s.set(enabled=True, model_choice=DFLASH, restart=True)
    assert d.uncensored is False


def test_flipping_the_checkpoint_under_a_running_pair_needs_a_restart(tmp_path):
    s = store(tmp_path)
    s.set(enabled=True)
    with pytest.raises(ValueError, match="stop the cluster before changing"):
        s.set(enabled=True, uncensored=False)
    d = s.set(enabled=True, uncensored=False, restart=True)
    assert d.uncensored is False and d.generation == 2


def test_a_file_from_before_the_knob_reads_as_null_and_launches_what_it_launched(tmp_path):
    write_file(tmp_path, mode="cluster", generation=7, max_num_seqs=8, max_model_len=262144, model_choice=FLASH,
               start_id="abc")
    s = store(tmp_path)
    d, _ = s.current()
    assert d.uncensored is None
    assert s.set(enabled=True) == d, "an idempotent Start of the running pair changes nothing"
    assert resolved_launch(s.catalog.get(FLASH), {}, max_num_seqs=8, max_model_len=262144)["uncensored"] is False
    assert resolved_launch(s.catalog.get(FLASH), {}, max_num_seqs=8, max_model_len=262144,
                           uncensored=True)["uncensored"] is True
    assert "uncensored" in EFFECTIVE_KEYS and limits_view()["uncensored"] == [0, 1]
    assert Desired().uncensored is None and Desired(mode=CLASSIC).uncensored is None


# ── the planner and the loop ──────────────────────────────────────────────────

def test_a_box_a_without_the_twin_blocks_the_start_naming_the_missing_dir(world):  # noqa: F811
    """NO FALLBACK: the original is on both boxes, the twin is not on box-b — the start is blocked, nothing is
    reserved, and the reason names the twin's dir, not the original's."""
    world.executor.have["box-b"] = {availability_key(FLASH, True): False}
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] == "blocked" and v["held"] is False, v
    assert "Qwen3.8-Flash-Next-hibrid48_uncensored/config.json" in v["detail"], v["detail"]
    assert world.reservation_jobs() == [] and not world.calls("up")


def test_the_twin_on_both_boxes_starts_even_where_the_original_is_gone(world):  # noqa: F811
    gone = {availability_key(FLASH, False): False, availability_key(FLASH, True): True}
    world.executor.have["box-a"] = dict(gone)
    world.executor.have["box-b"] = dict(gone)
    world.store.set(enabled=True)
    v = world.tick()
    assert v["state"] != "blocked", v
    asked = [c for c in world.executor.calls if c[0] == "availability"]
    assert asked and availability_key(FLASH, True) in asked[0][2], "the boxes are asked about the twin by name"


def test_the_rank_is_launched_from_the_twin_and_the_launch_record_says_so(world):  # noqa: F811
    world.executor.broker_vllm = {"box-a": True, "box-b": True}
    world.executor.mem = {"box-a": 30.0, "box-b": 30.0}
    world.store.set(enabled=True)
    world.tick()                                         # reserving
    world.tick()                                         # broker_parrot has not claimed yet
    world.claim()
    world.tick()                                         # freeing RAM
    world.executor.broker_vllm = {"box-a": False, "box-b": False}
    world.executor.mem = {"box-a": 115.0, "box-b": 115.0}
    v = world.settle(until="starting")
    assert world.calls("up"), v
    assert all(k["uncensored"] is True for k in world.executor.knobs.values()), world.executor.knobs
    record = json.loads(pathlib.Path(world.cfg.cluster_state_path).with_name("cluster-launch.json").read_text())
    assert record["uncensored"] is True and record["model_choice"] == FLASH, record


def test_the_fake_executor_launches_the_twin_the_real_one_would(tmp_path):
    """The real executor swaps the checkpoint INSIDE `up`, from the spec it is handed, so that a loop that
    resolved the flag once never has to resolve it twice (`ClusterExecutor.up`)."""
    from vllm_lb.cluster import ClusterHost
    from vllm_lb.cluster_exec import SshExecutor

    sent = {}

    class Probe(SshExecutor):
        def _action(self, host, words, env=None):
            sent.update(words=words, env=env)
            return True, "started"

    ex = Probe(key_path=str(tmp_path / "key"), known_hosts=str(tmp_path / "kh"), script_path=str(tmp_path / "tp-rank"))
    spec = shipped().get(DENSE)
    host = ClusterHost("box-a", "box-a", "u@box-a", "pair-link")
    ok, why = ex.up(host, rank=0, gen=1, max_num_seqs=8, max_model_len=262144, spec=spec, uncensored=True)
    assert ok, why
    assert sent["env"]["CLUSTER_MODEL_NAME"] == "Qwen3.8-27B-NVFP4_uncensored"
    ok, why = ex.up(host, rank=0, gen=1, max_num_seqs=8, max_model_len=262144, spec=shipped().get(DFLASH),
                    uncensored=True)
    assert not ok and "no uncensored" in why


# ── the API ───────────────────────────────────────────────────────────────────

def test_the_switch_takes_the_flag_and_refuses_a_value_that_is_not_a_bool(served):  # noqa: F811
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "uncensored": "yes"})
    assert status == 400 and "uncensored" in body["error"]["message"]
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "uncensored": False})
    assert status == 202 and body["desired"]["uncensored"] is False
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": True, "uncensored": None, "restart": True})
    assert status == 202 and body["desired"]["uncensored"] is True
    status, body = call(proxy, "GET", "/lb/status")
    assert status == 200
    model = next(m for m in body["models"]["cluster"] if m["id"] == FLASH)
    assert model["uncensored"]["default"] is True and model["uncensored"]["path"] == UNCENSORED[FLASH]
