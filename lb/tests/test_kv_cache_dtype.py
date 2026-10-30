"""THE KV CACHE DTYPE AS A RUN KNOB (operator 2026-10-22: "Implement kv on fp4 then, allow set in model params on run,
don't switch yet, dispatch and implement to all models").

Null is the profile's own — a run that sets nothing launches byte for byte as before. A value is checked against what
the profile's ENGINE IMAGE accepts (vllm_lb/kv_cache_dtypes.json, read in each image's own source) on this server's
cards, refused where it is given in the engine's own words, and re-prices every KV figure: the memory model's bytes per
token, the SGLang token pool the byte pin becomes, the block rules — measured at one dtype, derived at another."""
from __future__ import annotations

import copy
import json

import pytest

from tests.test_cluster_api import call, served  # noqa: F401 — fixture
from tests.test_cluster_store import store
from tests.test_tp_rank import LIVE_BOX_ENV, last_run, box_a, ssh  # noqa: F401 — fixtures
from vllm_lb import cluster_exec, kv_dtype
from vllm_lb.catalog import document, CLUSTER_MODE, ModelCatalog
from vllm_lb.cluster import CLUSTER_MODE as ON
from vllm_lb.cluster_exec import launch_env, resolved_launch
from vllm_lb.memory_estimate import estimate_memory

SGLANG = "qwen3.8-27b-sglang-dflash2"
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
DENSE = "qwen3.8-27b-nvidia-nvfp4"
GLM = "glm-5.3-flash-nvfp4"
GB10, RTX5090 = ("sm_121", "sm_121"), ("sm_120",)


def shipped() -> ModelCatalog:
    return ModelCatalog.from_dict(document())


# ── the engines' own facts ─────────────────────────────────────────────────────

def test_every_cluster_image_the_catalog_launches_is_recorded_with_its_profiles_own_dtype():
    lib, cat = kv_dtype.library(), shipped()
    for spec in cat.for_mode(CLUSTER_MODE):
        p = spec.profile(CLUSTER_MODE)
        assert p.image in lib.images, p.image
        assert lib.images[p.image].engine == p.engine
        if p.kv_cache_dtype is not None:
            assert p.kv_cache_dtype in lib.images[p.image].accepted, (spec.id, p.kv_cache_dtype)
        if spec.memory is not None:
            assert spec.memory.kv_cache_dtype in lib.layouts


def test_sglang_reads_each_fp4_pool_with_the_backends_its_access_rules_name():
    """SGLang 708f51e reads an NVFP4 pool with flashinfer (prefill, FP8 dequant workspace) and trtllm_mha (decode,
    native FP4) only, and an fp4_mx_block16 pool with triton (fp4_kv_cache_quant_method.py:766-771, 840-845)."""
    lib = kv_dtype.library()
    for tag in ("fleet.2", "fleet.3"):
        took = lib.accepted(f"myllmbox/sglang-27b-tp2:708f51e-{tag}", "nvfp4")
        # THE DRAFTER'S OWN POOL is not FP4 (--speculative-draft-kv-cache-dtype, arg_groups/fields/spec.py:144-156):
        # an FP4 one killed generations 214 (flashinfer cannot decode it), 215 (trtllm_mha cannot run the drafter's
        # block pass) and 217 (the DFLASH writer copies BF16 into it, dflash_worker_v2.py:1554).
        assert dict(took.attention_backends) == {"prefill": "flashinfer", "decode": "trtllm_mha"}
        assert took.draft_kv_cache_dtype == "fp8_e4m3"
        assert lib.accepted(f"myllmbox/sglang-27b-tp2:708f51e-{tag}", "fp4_mx_block16").draft_kv_cache_dtype == "fp8_e4m3"
        assert took.workspace_layers == 1
        mx = lib.accepted(f"myllmbox/sglang-27b-tp2:708f51e-{tag}", "fp4_mx_block16")
        assert dict(mx.attention_backends) == {"prefill": "triton", "decode": "triton"}
        assert "kv_cache_hook.py" in took.source and "NOT MEASURED" in took.source
    assert lib.problem("myllmbox/sglang-27b-tp2:708f51e-fleet.2", "nvfp4", GB10) == ""
    assert "float16" in lib.problem("myllmbox/sglang-27b-tp2:708f51e-fleet.2", "float16", GB10)


def test_vllm_nvfp4_is_refused_on_gb10_by_the_engines_own_rule_and_taken_on_a_5090():
    lib = kv_dtype.library()
    why = lib.problem("myllmbox/qwen38-flash-next-cluster-vllm:v5", "nvfp4", GB10)
    assert "sm_121" in why and "flashinfer.py" in why
    five = next(i for i in lib.images if i.startswith("vllm/vllm-openai@sha256:514cd427"))
    assert lib.problem(five, "nvfp4", RTX5090) == ""
    rocm = next(i for i in lib.images if i.startswith("rocm/vllm@"))
    assert "no FP4" in lib.problem(rocm, "nvfp4", ("gfx1151",))
    assert "no record" in lib.problem("someone/else:1", "bfloat16", GB10)
    assert "one of" in lib.problem(five, "fp8;rm", RTX5090)


def test_the_library_refuses_what_it_cannot_mean():
    raw = json.loads(kv_dtype.LIBRARY_PATH.read_text())
    bad = copy.deepcopy(raw)
    next(iter(bad["images"].values()))["dtypes"]["int3"] = {"source": "x"}
    with pytest.raises(ValueError, match="no layout"):
        kv_dtype.parse(bad)
    bad = copy.deepcopy(raw)
    del bad["layouts"]["nvfp4"]["source"]
    with pytest.raises(ValueError, match="missing source"):
        kv_dtype.parse(bad)


# ── the profile's own, and what a run may set ─────────────────────────────────

def test_each_profile_states_its_own_dtype():
    cat = shipped()
    assert cat.get(GLM).profile(CLUSTER_MODE).kv_cache_dtype == "fp8_e4m3"
    assert cat.get(SGLANG).profile(CLUSTER_MODE).kv_cache_dtype == "fp8_e4m3", "tp-rank sglang_cmd's default"
    assert cat.get(FLASH_SGLANG).profile(CLUSTER_MODE).kv_cache_dtype == "fp8_e4m3", "-fleet.4's qsa-fp8-kv patch"
    # The lowest KV the vLLM image runs on GB10 (nvfp4 needs sm_10x; operator 2026-10-26): no longer vLLM's auto.
    # …and the dense 27B 4-bit on Triton since generation 263 (healthy, 17x23 = 391): int4_per_token_head.
    assert cat.get(DENSE).profile(CLUSTER_MODE).kv_cache_dtype == "int4_per_token_head"
    assert cat.get("qwen3.8-flash-next-hibrid48").profile(CLUSTER_MODE).kv_cache_dtype == "fp8_e4m3"


def test_flash_next_on_sglang_takes_fp8_kv_on_the_patched_image_and_no_fp4():
    """Operator 2026-10-26, "implement FP8 there": the SM121 QSA kernel reads BF16 K/V only (generation 153), and
    -fleet.4's qsa-fp8-kv patch widens an FP8 pool's gathered rows to BF16 for it. An FP4 pool is not widened."""
    spec = shipped().get(FLASH_SGLANG)
    assert spec.profile(CLUSTER_MODE).image.endswith("708f51e-fleet.4")
    assert spec.kv_cache_dtype_problem("fp8_e4m3", GB10) == ""
    assert spec.kv_cache_dtype_problem("bfloat16", GB10) == ""
    assert "qsa-fp8-kv" in spec.kv_cache_dtype_problem("nvfp4", GB10)


# ── the memory model at another dtype ─────────────────────────────────────────

def test_sglang_fp4_pool_is_priced_with_its_dequant_workspace():
    """Per rank, TP=2: 16 target layers x 2 KV heads x 256, K and V, at 9/16 B plus ONE layer at 1 B (the FP4 dequant
    workspace, pool_configurator.py:409-417); the drafter's 5 layers x 4 x 128 at ITS OWN dtype, fp8: 1 B, no workspace."""
    spec = shipped().get(SGLANG)
    fp4 = spec.at_kv_cache_dtype("nvfp4")
    assert fp4.memory.kv_bytes_per_token == 16 * 1024 * 9 // 16 + 1024 + 5 * 1024 == 15360
    assert spec.at_kv_cache_dtype("bfloat16").memory.kv_bytes_per_token == 2 * spec.memory.kv_bytes_per_token
    assert spec.at_kv_cache_dtype(None) is spec


def test_a_block_rule_measured_at_one_dtype_is_derived_at_another():
    spec = shipped().get(DENSE)
    m = spec.memory
    assert (m.kv_cache_dtype, m.kv_blocks.kv_cache_dtype, m.single_card_kv_blocks.kv_cache_dtype) == \
           ("bfloat16", "bfloat16", "fp8_e4m3")
    fp8 = spec.at_kv_cache_dtype("fp8_e4m3").memory
    # hybrid (state blocks): the page is the GDN state's, so a narrower element holds more tokens per block
    assert (fp8.kv_blocks.block_tokens, fp8.kv_blocks.block_bytes) == (1632, m.kv_blocks.block_bytes)
    assert "DERIVED" in fp8.kv_blocks.source
    assert fp8.single_card_kv_blocks is m.single_card_kv_blocks, "measured at fp8_e4m3 already"
    mai = shipped().get("mai-ui-2b")
    fp8_mai = mai.at_kv_cache_dtype("fp8_e4m3").memory.single_card_kv_blocks
    assert (fp8_mai.block_tokens, fp8_mai.block_bytes) == (16, 1835008 // 2), "pure attention: the page halves"


def test_the_estimate_counts_the_pool_at_the_runs_dtype():
    spec = shipped().get(SGLANG)
    kw = dict(box="box-a", role="head", max_num_seqs=8, max_model_len=262144, tp=2, available_gib=None,
              floor_gib=8, margin_gib=4)
    assert estimate_memory(spec, kv_cache_dtype="nvfp4", **kw).kv_bytes_per_token == 15360
    assert estimate_memory(spec, **kw).kv_bytes_per_token == spec.memory.kv_bytes_per_token


# ── the launch ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", [SGLANG, FLASH_SGLANG, DENSE, GLM])
def test_a_run_that_sets_nothing_launches_byte_for_byte_as_before(model):
    spec = shipped().get(model)
    assert launch_env(spec, kv_cache_dtype=None) == launch_env(spec)
    env = launch_env(spec)
    if model == DENSE:                                  # its own int4, with the one backend that reads it
        assert (env["CLUSTER_KV_CACHE_DTYPE"], env["CLUSTER_ATTENTION_BACKEND"]) == ("int4_per_token_head", "TRITON_ATTN")
    elif model == GLM:
        assert env["CLUSTER_KV_CACHE_DTYPE"] == "fp8_e4m3"
    else:
        assert "CLUSTER_KV_CACHE_DTYPE" not in env
        assert "attention_backend" not in env.get("CLUSTER_ENGINE_ARGS", "")


def test_sglang_nvfp4_sends_the_dtype_its_backend_and_a_token_pool_sized_at_fp4():
    spec = shipped().get(SGLANG)
    pin = spec.profile(CLUSTER_MODE).kv_cache_memory
    env = launch_env(spec, kv_cache_dtype="nvfp4")
    args = json.loads(env["CLUSTER_ENGINE_ARGS"])
    assert env["CLUSTER_KV_CACHE_DTYPE"] == "nvfp4"
    assert (args["prefill_attention_backend"], args["decode_attention_backend"]) == ("flashinfer", "trtllm_mha")
    assert args["speculative_draft_kv_cache_dtype"] == "fp8_e4m3" and "speculative_draft_attention_backend" not in args
    assert args["max_total_tokens"] == pin // 15360
    assert args["mem_fraction_static"] == json.loads(launch_env(spec)["CLUSTER_ENGINE_ARGS"])["mem_fraction_static"]


def test_a_vllm_run_dtype_overrides_the_profiles_flag():
    assert launch_env(shipped().get(GLM), kv_cache_dtype="bfloat16")["CLUSTER_KV_CACHE_DTYPE"] == "bfloat16"
    assert launch_env(shipped().get(DENSE), kv_cache_dtype="fp8_e4m3")["CLUSTER_KV_CACHE_DTYPE"] == "fp8_e4m3"


def test_resolved_launch_names_the_dtype_that_runs():
    spec = shipped().get(SGLANG)
    kw = dict(max_num_seqs=8, max_model_len=262144)
    assert resolved_launch(spec, {}, **kw)["kv_cache_dtype"] == "fp8_e4m3"
    assert resolved_launch(spec, {}, kv_cache_dtype="nvfp4", **kw)["kv_cache_dtype"] == "nvfp4"
    assert resolved_launch(shipped().get(DENSE), {}, **kw)["kv_cache_dtype"] == "int4_per_token_head"


def _sglang_up(box_a, ssh, **knobs):  # noqa: F811
    spec = shipped().get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=135, max_num_seqs=8, max_model_len=262144, spec=spec, **knobs)
    return ok, msg, (last_run(box_a) if ok else None)


def test_tp_rank_launches_sglang_at_fp8_and_flashinfer_unless_told(box_a, ssh):  # noqa: F811
    ok, msg, run = _sglang_up(box_a, ssh)
    assert ok, msg
    flags = dict(zip(run, run[1:]))
    assert (flags["--kv-cache-dtype"], flags["--attention-backend"]) == ("fp8_e4m3", "flashinfer")
    assert "--prefill-attention-backend" not in run and "--decode-attention-backend" not in run


def test_tp_rank_launches_sglang_nvfp4_with_flashinfer_prefill_and_trtllm_decode(box_a, ssh):  # noqa: F811
    """Generation 218: NVFP4 with the drafter off — correct on every check, the one way it serves."""
    ok, msg, run = _sglang_up(box_a, ssh, kv_cache_dtype="nvfp4", num_speculative_tokens=0)
    assert ok, msg
    flags = dict(zip(run, run[1:]))
    assert (flags["--kv-cache-dtype"], flags["--attention-backend"]) == ("nvfp4", "flashinfer")
    assert (flags["--prefill-attention-backend"], flags["--decode-attention-backend"]) == ("flashinfer", "trtllm_mha")
    assert "--speculative-algorithm" not in run and "--speculative-draft-kv-cache-dtype" not in run


@pytest.mark.parametrize("dtype", ["nvfp4", "fp4_mx_block16"])
def test_tp_rank_refuses_fp4_kv_with_a_drafter(box_a, ssh, dtype):  # noqa: F811
    """Generation 219 (2026-10-22): NVFP4 + DFlash2 with the drafter on its own fp8 pool STARTED and served wrong output
    — 17x23 as "22 by 5", no tool call, a code missed at 8k — while the drafter's acceptance looked healthy. Before it,
    214/215/217 died on the drafter. Refused before two boxes load anything."""
    with pytest.raises(cluster_exec.LaunchRefused) as refused:
        _sglang_up(box_a, ssh, kv_cache_dtype=dtype)
    msg = str(refused.value)
    assert "219" in msg and "num_speculative_tokens 0" in msg and "tp-rank on box-a refused" in msg, msg
    assert not [c for c in box_a.calls() if c[0] == "run"]


def test_tp_rank_launches_sglang_mx_fp4_on_triton(box_a, ssh):  # noqa: F811
    ok, msg, run = _sglang_up(box_a, ssh, kv_cache_dtype="fp4_mx_block16", num_speculative_tokens=0)
    assert ok, msg
    flags = dict(zip(run, run[1:]))
    assert (flags["--prefill-attention-backend"], flags["--decode-attention-backend"]) == ("triton", "triton")


@pytest.mark.parametrize("dtype, word", [("nvfp4", "attention"), ("float16", "kv_cache_dtype")])
def test_tp_rank_refuses_what_sglang_would_refuse_at_startup(box_a, ssh, dtype, word):  # noqa: F811
    """tp-rank's own guard, the LB aside: CLUSTER_KV_CACHE_DTYPE from the box's env, no attention backend named."""
    spec = shipped().get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models), "CLUSTER_KV_CACHE_DTYPE": dtype})
    with pytest.raises(cluster_exec.LaunchRefused, match=word):     # a refusal of the plan: never retried
        ex.up(host, rank=0, gen=135, max_num_seqs=8, max_model_len=262144, spec=spec)
    assert not [c for c in box_a.calls() if c[0] == "run"]


# ── the switch ─────────────────────────────────────────────────────────────────

def test_the_switch_stores_a_dtype_the_image_takes_and_refuses_one_it_does_not(tmp_path):
    s = store(tmp_path, archs=GB10)
    d = s.set(enabled=True, model_choice=SGLANG, kv_cache_dtype="nvfp4", num_speculative_tokens=0)
    assert (d.mode, d.kv_cache_dtype) == (ON, "nvfp4")
    assert s.load().kv_cache_dtype == "nvfp4"
    with pytest.raises(ValueError, match="stop the cluster"):
        s.set(enabled=True, kv_cache_dtype="bfloat16")
    assert s.set(enabled=True, restart=True, kv_cache_dtype=None).kv_cache_dtype is None
    with pytest.raises(ValueError, match="sm_121"):
        store(tmp_path / "b", archs=GB10).set(enabled=False, model_choice=DENSE, kv_cache_dtype="nvfp4")
    with pytest.raises(ValueError, match="qsa-fp8-kv"):
        store(tmp_path / "c", archs=GB10).set(enabled=True, model_choice=FLASH_SGLANG, kv_cache_dtype="nvfp4")


def test_a_changed_model_takes_its_own_dtype_again(tmp_path):
    s = store(tmp_path, archs=GB10)
    s.set(enabled=False, model_choice=SGLANG, kv_cache_dtype="nvfp4", num_speculative_tokens=0)
    assert s.set(enabled=False, model_choice=DENSE).kv_cache_dtype is None


def test_an_arch_bound_dtype_with_no_card_named_is_refused(tmp_path):
    with pytest.raises(ValueError, match="states no arch"):
        store(tmp_path).set(enabled=True, model_choice=SGLANG, kv_cache_dtype="nvfp4")


def test_the_api_takes_the_dtype_and_refuses_a_malformed_one(served):  # noqa: F811
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "model_choice": SGLANG,
                                                        "kv_cache_dtype": "bfloat16"})
    assert status == 202 and body["desired"]["kv_cache_dtype"] == "bfloat16", body
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "kv_cache_dtype": 4})
    assert status == 400 and "kv_cache_dtype" in body["error"]["message"]
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "kv_cache_dtype": "int3"})
    assert status == 409 and "one of" in body["error"]["message"]


def test_the_pair_publishes_each_models_dtype_choices_for_its_cards():
    spec = shipped().get(SGLANG)
    choices = {c["dtype"]: c for c in spec.kv_cache_dtype_choices(GB10)}
    assert choices["nvfp4"]["refused"] == ""
    assert choices["nvfp4"]["attention_backends"] == {"prefill": "flashinfer", "decode": "trtllm_mha"}
    assert choices["float16"]["refused"]
    dense = {c["dtype"]: c for c in shipped().get(DENSE).kv_cache_dtype_choices(GB10)}
    assert "sm_121" in dense["nvfp4"]["refused"]


def test_the_catalogs_sglang_default_is_tp_ranks_own():
    from tests.test_tp_rank import TP_RANK
    from vllm_lb.catalog import SGLANG_KV_CACHE_DTYPE
    assert f"$(engine_arg kv_cache_dtype {SGLANG_KV_CACHE_DTYPE})" in TP_RANK.read_text()


def test_the_estimate_prices_the_selection_at_its_dtype_and_refuses_one_it_cannot_run(served):  # noqa: F811
    cfg, proxy = served()
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={SGLANG}&kv_cache_dtype=bfloat16")
    assert status == 200 and body["kv_cache_dtype"] == "bfloat16", body
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={SGLANG}&kv_cache_dtype=float16")
    assert status != 200 and "float16" in json.dumps(body)


# ── FP4 KV WITH A DRAFTER: refused at the switch and on the page, from one fact ────────────────────────────────────
# 2026-10-22: the panel's KV type row took nvfp4 at the drafter's default depth, the switch ACCEPTED it, and generation
# 221's tp-rank refused it on both ranks — retried every ~2 s while the page said "launching both ranks". The fact is
# the image's (kv_cache_dtypes.json `with_drafter`), read by the switch, the estimate and the page's choices alike.

@pytest.mark.parametrize("dtype", ["nvfp4", "fp4_mx_block16"])
def test_sglang_fp4_with_a_drafter_is_the_images_own_refusal(dtype):
    lib = kv_dtype.library()
    for tag in ("fleet.2", "fleet.3"):
        why = lib.accepted(f"myllmbox/sglang-27b-tp2:708f51e-{tag}", dtype).with_drafter
        assert "214" in why and "219" in why and "set num_speculative_tokens 0" in why, why
    spec = shipped().get(SGLANG)
    assert spec.profile(CLUSTER_MODE).num_speculative_tokens > 0, "the profile's own depth drafts"
    for asked in (None, 4):                                  # None: the profile's own depth
        why = spec.kv_cache_dtype_problem(dtype, GB10, num_speculative_tokens=asked)
        assert "219" in why and "set num_speculative_tokens 0" in why, (asked, why)
    assert spec.kv_cache_dtype_problem(dtype, GB10, num_speculative_tokens=0) == ""
    assert spec.kv_cache_dtype_problem("fp8_e4m3", GB10, num_speculative_tokens=4) == ""


def test_the_pair_publishes_each_dtypes_refusal_with_a_drafter_beside_its_own():
    choices = {c["dtype"]: c for c in shipped().get(SGLANG).kv_cache_dtype_choices(GB10)}
    assert choices["nvfp4"]["refused"] == "", "at depth 0 it runs"
    assert "219" in choices["nvfp4"]["refused_with_drafter"]
    assert choices["fp8_e4m3"]["refused_with_drafter"] == ""
    assert choices["float16"]["refused_with_drafter"] == choices["float16"]["refused"] != ""
    dense = {c["dtype"]: c for c in shipped().get(DENSE).kv_cache_dtype_choices(RTX5090)}
    assert all(c["refused_with_drafter"] == c["refused"] for c in dense.values()), "vLLM: no drafter rule"


def test_the_switch_refuses_fp4_kv_at_a_drafting_depth_and_stores_nothing(tmp_path):
    s = store(tmp_path, archs=GB10)
    for kw in ({}, {"num_speculative_tokens": 4}):
        for enabled in (True, False):
            with pytest.raises(ValueError, match="set num_speculative_tokens 0"):
                s.set(enabled=enabled, model_choice=SGLANG, kv_cache_dtype="nvfp4", **kw)
    assert s.load().kv_cache_dtype is None, "nothing stored"
    d = s.set(enabled=False, model_choice=SGLANG, kv_cache_dtype="nvfp4", num_speculative_tokens=0)
    assert (d.kv_cache_dtype, d.num_speculative_tokens) == ("nvfp4", 0)
    with pytest.raises(ValueError, match="219"):             # the depth turned back on under a stored FP4 pool
        s.set(enabled=False, num_speculative_tokens=4)
    with pytest.raises(ValueError, match="219"):             # …or back to the profile's own
        s.set(enabled=False, num_speculative_tokens=None)
    assert s.set(enabled=True).mode == ON, "nvfp4 at depth 0 starts"


def test_a_start_of_a_stored_fp4_pool_with_a_drafter_is_refused(tmp_path):
    s = store(tmp_path, archs=GB10)
    path = tmp_path / "state" / "cluster.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": "classic", "generation": 220, "max_num_seqs": 8, "max_model_len": 262144,
                                "model_choice": SGLANG, "kv_cache_dtype": "nvfp4"}))
    with pytest.raises(ValueError, match="219"):
        s.set(enabled=True)
    assert s.load().mode != ON and s.load().generation == 220


def test_the_api_refuses_fp4_kv_with_a_drafter_by_name(served):  # noqa: F811
    cfg, proxy = served()
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "model_choice": SGLANG,
                                                        "kv_cache_dtype": "fp4_mx_block16", "num_speculative_tokens": 4})
    assert status == 409 and "set num_speculative_tokens 0" in body["error"]["message"], body
    status, body = call(proxy, "POST", "/lb/cluster", {"enabled": False, "model_choice": SGLANG,
                                                        "kv_cache_dtype": "fp4_mx_block16", "num_speculative_tokens": 0})
    assert status == 202, body
    status, body = call(proxy, "GET", f"/lb/cluster/estimate?model_choice={SGLANG}&kv_cache_dtype=fp4_mx_block16"
                                      f"&num_speculative_tokens=4")
    assert status != 200 and "219" in json.dumps(body), body


def test_tp_ranks_refusal_raises_and_a_box_failure_does_not(box_a, ssh, monkeypatch):  # noqa: F811
    """A refusal of the PLAN (tp-rank `fail 2`) is the same on every retry: the executor raises it as LaunchRefused, the
    loop's refusal path. A failure of the BOX (docker run, ssh) still answers `(False, why)` for the loop to retry."""
    with pytest.raises(cluster_exec.LaunchRefused):
        _sglang_up(box_a, ssh, kv_cache_dtype="nvfp4", num_speculative_tokens=4)
    box_a.set(run_error="docker: Error response from daemon: transient")
    ok, msg, _ = _sglang_up(box_a, ssh, kv_cache_dtype="nvfp4", num_speculative_tokens=0)
    assert ok is False and "docker run failed" in msg, msg


def test_a_launch_with_no_depth_given_runs_the_profiles_tuned_depth_not_its_configs():
    """GENERATION 249 (2026-10-26): DeepSeek-V4-Flash's profile keeps its dspark speculative_config (depth 5, the
    drafter is in the checkpoint) but its TUNED depth is 0 because the drafter cannot load. The launch record said '0
    speculative tokens' (resolved_launch reads the tuned depth) while launch_env, given no depth, sent the config's 5 —
    and both ranks died loading the drafter ('strided dtype conversion: mtp.0.ffn.experts.0.w1.scale'). No depth given
    is the profile's default depth: the one resolved_launch records, so the launch and its record cannot differ."""
    import dataclasses
    from vllm_lb.cluster_exec import resolved_launch
    spec = shipped().get("qwen3.8-flash-next-hibrid48")
    profile = spec.profile(CLUSTER_MODE)
    assert profile.speculative_config and int(profile.speculative_config["num_speculative_tokens"]) > 0
    off = dataclasses.replace(profile, tuned=dataclasses.replace(profile.tuned, num_speculative_tokens=0))
    spec = dataclasses.replace(spec, launch={**spec.launch, CLUSTER_MODE: off})
    assert resolved_launch(spec, {}, max_num_seqs=8, max_model_len=131_072)["num_speculative_tokens"] == 0
    assert launch_env(spec)["CLUSTER_SPECULATIVE_CONFIG"] == ""
    assert json.loads(launch_env(spec, num_speculative_tokens=3)["CLUSTER_SPECULATIVE_CONFIG"])[
        "num_speculative_tokens"] == 3


# ── 4-bit KV on TRITON for the vLLM rows (operator 2026-10-26: "switch all to triton … one DRY") ─────────────────

V5 = "myllmbox/qwen38-flash-next-cluster-vllm:v5"


def test_the_vllm_image_takes_int4_per_token_head_read_by_triton_on_gb10():
    """vLLM 0.29.0 in the v5 image (read on box-a 2026-10-26): config/cache.py CacheDType names int4_per_token_head, and
    only the Triton backend's supported_kv_cache_dtypes does (triton_attn.py:301) — no arch rule, so GB10's sm_121 runs
    it where nvfp4 (FlashInfer trtllm-gen, sm_10x only) cannot. Its element: two nibbles a byte plus one fp32 scale per
    token and head for K and V (TritonAttentionBackend.customize_spec) — 264 B against bf16's 1,024 at head size 256."""
    lib = kv_dtype.library()
    took = lib.accepted(V5, "int4_per_token_head")
    assert took is not None and dict(took.attention_backends) == {"prefill": "triton_attn", "decode": "triton_attn"}
    assert lib.problem(V5, "int4_per_token_head", GB10) == ""
    assert lib.layouts["int4_per_token_head"].bytes_per_element == 264 / 512
    assert lib.problem(V5, "nvfp4", GB10) != "", "nvfp4 stays refused on GB10"


def test_one_table_names_the_backend_and_each_engine_gets_its_own_flag():
    """DRY: the library's `attention_backends` for the run's dtype is the ONE statement of which kernels read the pool;
    launch_env translates it per engine — vLLM's one --attention-backend (TRITON_ATTN), SGLang's prefill/decode pair.
    A dtype that names none sends no backend: the profile's own run as before."""
    dense = shipped().get(DENSE)                  # the dense 27B: standard attention, the image's backends read it
    env = launch_env(dense, kv_cache_dtype="int4_per_token_head")
    assert (env["CLUSTER_KV_CACHE_DTYPE"], env["CLUSTER_ATTENTION_BACKEND"]) == ("int4_per_token_head", "TRITON_ATTN")
    fp8 = launch_env(dense, kv_cache_dtype="fp8_e4m3")
    assert "CLUSTER_ATTENTION_BACKEND" not in fp8, "fp8_e4m3: no backend named, vLLM picks its own"
    sglang = launch_env(shipped().get(SGLANG), kv_cache_dtype="fp4_mx_block16", num_speculative_tokens=0)
    args = json.loads(sglang["CLUSTER_ENGINE_ARGS"])
    assert (args["prefill_attention_backend"], args["decode_attention_backend"]) == ("triton", "triton")


def test_tp_rank_launches_the_vllm_row_at_int4_kv_on_triton(box_a, ssh):  # noqa: F811
    """End to end through the real script: the image's CacheDType takes int4_per_token_head, and the rank is launched
    with it AND the backend that reads it — the one table's word, never two flags set apart."""
    dense = shipped().get(DENSE)                  # generation 263 ran exactly this: healthy, 17x23 = 391
    models = box_a.place_checkpoint(box_a.home / "models", name=dense.name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=131072, spec=dense,
                    kv_cache_dtype="int4_per_token_head")
    assert ok is True, msg
    run = last_run(box_a)
    flags = dict(zip(run, run[1:]))
    assert (flags["--kv-cache-dtype"], flags["--attention-backend"]) == ("int4_per_token_head", "TRITON_ATTN")


# ── a model whose OWN attention backend reads the pool: Flash-Next's QSA on Triton ─────────────────────────────

def test_flash_next_reads_nvfp4_on_its_own_triton_qsa_backend_on_gb10():
    """Operator 2026-10-26 ("switch all to triton"): in the v5 image Flash-Next's attention is QWEN4_EXP_QSA_TRITON —
    its main cache read ONLY by the Triton kernel in ops/qsa.py, which dequantizes in registers ("the FlashAttention
    capability is irrelevant here", qsa.py), supported bfloat16/fp8/fp8_e4m3/nvfp4. So the image's generic nvfp4 rule
    (FlashInfer trtllm-gen, sm_10x) is not this model's: NVFP4 runs on GB10's sm_121. And int4_per_token_head, which
    the image takes for TRITON_ATTN, is refused by the model's own layer before any launch (generations 262/265:
    "Qwen4Exp QSA requires a BF16, FP8-E4M3 or NVFP4 main KV cache")."""
    flash = shipped().get("qwen3.8-flash-next-hibrid48")
    assert flash.kv_cache_dtype_problem("nvfp4", GB10) == ""
    int4 = flash.kv_cache_dtype_problem("int4_per_token_head", GB10)
    assert "QWEN4_EXP_QSA_TRITON" in int4 and "nvfp4" in int4, int4
    dense = shipped().get(DENSE)                           # the same image, standard attention: the image's rule stands
    assert "sm_10" in dense.kv_cache_dtype_problem("nvfp4", GB10)
    assert dense.kv_cache_dtype_problem("int4_per_token_head", GB10) == ""
    by = {c["dtype"]: c["refused"] for c in flash.profile(CLUSTER_MODE).kv_cache_dtype_choices(GB10)}
    assert by["nvfp4"] == "" and by["int4_per_token_head"] != "" and by["fp8_e4m3"] == ""


def test_a_model_with_its_own_backend_is_sent_no_library_backend():
    """vLLM selects QWEN4_EXP_QSA_TRITON for the model itself: sending --attention-backend would override it."""
    env = launch_env(shipped().get("qwen3.8-flash-next-hibrid48"), kv_cache_dtype="nvfp4")
    assert env["CLUSTER_KV_CACHE_DTYPE"] == "nvfp4" and "CLUSTER_ATTENTION_BACKEND" not in env
