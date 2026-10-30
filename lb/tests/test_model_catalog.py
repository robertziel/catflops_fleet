"""M1 — ONE ModelCatalog for both pickers (worklog/vllm-model-selection.md; worklog/vllm-gateway-program.md
pass 4).

The domain words, the same here as in the code, `/lb/status`, the page and the other tests:

  ModelCatalog   every model this gateway knows how to launch — the single source of ids and launch settings
  ModelSpec      one launchable model: id, label, path, served name, ModelLimits, a LaunchProfile per mode
  ServingMode    `classic` (broker_parrot's per-box vLLM) or `cluster` (the TP=2 pair)
  LaunchProfile  a model's launch SETTINGS in one ServingMode
  ModelLimits    the context window and the concurrency range a spec may be started with
  ModelChoice    the spec id the operator chose for a ServingMode
  Availability   whether a spec's files and image are on a box — observed, never assumed
  LoadedModel    what an endpoint actually serves, from its own /v1/models — never the ModelChoice
"""
from __future__ import annotations

import ast
import copy
import json
import tempfile
from pathlib import Path

import pytest

from vllm_lb.catalog import (document, CLASSIC, CLUSTER_MODE, MAX_NUM_SEQS_CAP, Availability,
                             LoadedModel, ModelCatalog, UnknownModelChoice)

FLASH = "qwen3.8-flash-next-hibrid48"
DENSE = "qwen3.8-27b-nvidia-nvfp4"
SEEDED = "qwen3.8-27b-nvfp4"
#: The dense 27B again, with a drafter that is its OWN checkpoint (draft_path) rather than tensors inside
#: the target's — so the cluster picker lists three, and `dflash` is the second drafter method in the file.
DFLASH = "qwen3.8-27b-dflash2"
#: The same 27B, served by SGLang instead of vLLM — the catalog's first non-vLLM cluster profile.
SGLANG = "qwen3.8-27b-sglang-dflash2"
#: Flash-Next on SGLang: FLASH's weights, SGLANG's engine, and the only NEXTN profile (catalog.json, 2026-10-05).
FLASH_SGLANG = "qwen3.8-flash-next-sglang"
#: zai-org's GLM-5.3-Flash (321B MoE, 18B active) in RedHatAI's compressed-tensors NVFP4 (operator 2026-10-20).
GLM = "glm-5.3-flash-nvfp4"
#: DeepSeek-V4-Flash-0731 (304B MoE, 13B active) in nvidia's NVFP4, on the B12X stack (operator 2026-10-23).
DSV4 = "deepseek-v4-flash-nvfp4"
TF_NATIVE = "qwen3.8-flash-next-tensorfold-nvfp4"
TF_MLX = "qwen3.8-flash-next-tensorfold-mlx-mtp"
TP_RANK = Path(__file__).resolve().parents[1] / "cluster" / "tp-rank"


def shipped() -> ModelCatalog:
    return ModelCatalog.load()


def raw() -> dict:
    return document()


def spec_raw(doc: dict, spec_id: str) -> dict:
    return next(m for m in doc["models"] if m["id"] == spec_id)


# ── what ships ────────────────────────────────────────────────────────────────

def test_the_shipped_catalog_loads_with_one_default_per_serving_mode():
    c = shipped()
    assert c.error == ""
    assert [s.id for s in c.for_mode(CLUSTER_MODE)] == [FLASH, FLASH_SGLANG, TF_NATIVE, TF_MLX, DENSE, DFLASH, SGLANG, GLM, DSV4]
    assert [s.id for s in c.for_mode(CLASSIC)] == [SEEDED]
    assert c.default(CLUSTER_MODE).id == FLASH and c.default(CLASSIC).id == SEEDED


def test_the_cluster_default_is_exactly_the_rank_tp_rank_launched_before_model_choices_existed():
    """Upgrading must change nothing that runs: the default ModelSpec's LaunchProfile is tp-rank's own
    defaults, read off the script, and the folders are the search path it has always used."""
    spec = shipped().get(FLASH)
    p = spec.profile(CLUSTER_MODE)
    script = TP_RANK.read_text()
    for var, value in (("CLUSTER_IMAGE", p.image), ("CLUSTER_MODEL_NAME", spec.name),
                       ("CLUSTER_SERVED_MODEL_NAME", spec.served_name),
                       ("CLUSTER_GPU_MEMORY_UTILIZATION", f"{p.gpu_memory_utilization:.2f}"),
                       ("CLUSTER_KV_CACHE_MEMORY", str(p.kv_cache_memory))):
        assert f': "${{{var}:={value}}}"' in script, f"{var} is not tp-rank's default {value!r}"
    assert (p.tool_call_parser, p.reasoning_parser) == ("qwen3_xml", "qwen3")
    assert p.speculative_config == {"method": "mtp", "num_speculative_tokens": 4}
    assert spec.cluster_dirs == ("~/models/qwen38-flash-next-cluster",
                                 "/mnt/box-a-home/models/qwen38-flash-next-cluster",
                                 "/srv/models/qwen38-flash-next-cluster")
    assert spec.limits.max_model_len == 262144 and spec.limits.max_num_seqs == (1, 48)
    assert p.default is True and spec.unsupported == ""
    # pass 6's per-model tp-rank settings travel in the LaunchProfile: the drafter view on, and the PLE table SPLIT —
    # tp-rank's own default (1, replicated) is what this profile launched until 2026-10-11, when every replicated
    # start since the memory guard had died in it (tests/test_flash_next_ple_split.py)
    assert (p.ple_replicate, p.mtp_prefix) == (False, "mtp.")
    assert spec.memory is not None and spec.memory.calibration, "the default is calibrated on a real start"


def test_glm_53_flash_is_a_vllm_choice_launched_the_way_the_measured_2x_box_a_recipe_launches_it():
    """GLM-5.3-Flash on the pair (operator 2026-10-20: "prepare engine that will run GLM-5.3-Flash on our cluster"):
    the flags each published 2x DGX Spark recipe found necessary on GB10, the model card's parsers, its OWN MTP layer
    as the drafter (no view: `model_mtp.safetensors` is read by vLLM's mtp method in place), and the image that boots
    it on sm_121 — never the default (the pair keeps serving what it serves until the operator starts this).
    MULTIMODAL as MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark @aed98a1 serves it (operator 2026-10-21): the vision tower
    loaded, its max-size profile skipped and the items per prompt bounded instead, util 0.84 (0.86 lost rank 0 to the
    image warmup there), and its 1.05 GiB tower in the weights."""
    spec = shipped().get(GLM)
    p = spec.profile(CLUSTER_MODE)
    assert p.engine == "vllm" and p.default is False
    assert p.vllm_flags == {"kv_cache_dtype": "fp8_e4m3", "block_size": 2304, "moe_backend": "marlin",
                            "enforce_eager": True, "skip_mm_profiling": True,
                            "limit_mm_per_prompt": {"image": 4, "video": 1}}
    assert spec.vision is True and p.gpu_memory_utilization == 0.84
    assert (p.tool_call_parser, p.reasoning_parser) == ("glm47", "glm45")
    assert p.speculative_config == {"method": "mtp", "num_speculative_tokens": 4} and p.mtp_prefix == ""
    assert p.image.startswith("myllmbox/glm53-flash-cluster-vllm:")
    assert spec.cluster_dirs[0] == "~/models/glm53-flash-cluster" and spec.name == "GLM-5.3-Flash-NVFP4"
    assert spec.served_name == "zai-org/GLM-5.3-Flash" and spec.quant_method == "compressed-tensors"
    assert spec.limits.max_model_len == 262144 and spec.limits.max_num_seqs == (1, 8)
    m = spec.memory
    assert (m.weights_gib, m.mtp_gib, m.ple_gib) == (88.57, 3.55, 0.0) and m.kv_bytes_per_token == 6973
    assert m.calibration == "", "uncalibrated until its first start on the pair"


def test_deepseek_v4_flash_nvfp4_is_a_vllm_choice_launched_the_way_the_b12x_2x_box_a_recipe_launches_0731():
    """DeepSeek-V4-Flash-0731 in nvidia's NVFP4 (operator 2026-10-23, chosen over the native checkpoint): the flags of
    eugr/spark-vllm-docker's 2x DGX Spark recipe for 0731 on the B12X stack, the checkpoint's own DSpark drafter (no
    view: its heads ride in the shards), the only KV dtype DeepSeek-V4's sparse MLA takes (fp8, the fp8_ds_mla alias),
    the operator's 131,072 context cap — never the default, and uncalibrated until its first start."""
    spec = shipped().get(DSV4)
    p = spec.profile(CLUSTER_MODE)
    assert p.engine == "vllm" and p.default is False and spec.vision is False
    assert p.vllm_flags == {
        "kv_cache_dtype": "fp8", "block_size": 256, "moe_backend": "b12x", "trust_remote_code": True,
        "tokenizer_mode": "deepseek_v4", "load_format": "b12x", "linear_backend": "b12x", "attention_backend": "B12X",
        "max_cudagraph_capture_size": 48,
        "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]},
        "reasoning_config": {"reasoning_parser": "deepseek_v4", "reasoning_start_str": "", "reasoning_end_str": ""},
        "default_chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}
    assert (p.tool_call_parser, p.reasoning_parser) == ("deepseek_v4", "deepseek_v4")
    assert p.speculative_config == {"method": "dspark", "num_speculative_tokens": 5,
                                    "draft_sample_method": "probabilistic", "attention_backend": "B12X"}
    assert p.mtp_prefix == "" and p.image == "myllmbox/deepseek-v4-flash-cluster-vllm:v1"
    assert spec.name == "DeepSeek-V4-Flash-0731-NVFP4" and spec.quant_method == "deepseek_v4_fp8"
    assert spec.cluster_dirs[0] == "~/models/deepseek-v4-flash-cluster"
    t = p.tuned
    # Depth 0: the drafter's MXFP4 experts do not load into this vLLM's NVFP4 method (generation 233).
    assert (t.max_num_seqs, t.max_model_len, t.max_num_batched_tokens, t.num_speculative_tokens) == (8, 131072, 8192, 0)
    m = spec.memory
    # Measured on generations 234/235: vLLM's own load and KV-pool lines.
    assert (m.weights_gib, m.mtp_gib, m.kv_bytes_per_token, m.kv_cache_dtype) == (79.29, 5.06, 55011, "fp8")
    assert m.calibration.startswith("MEASURED 2026-10-24, generation 235")
    assert p.kv_cache_memory == 10_000_000_000, "16e9 was killed by the memory guard during b12x warmup (generation 234)"


def test_the_second_cluster_choice_is_nvidias_qwen38_27b_nvfp4_with_the_drafter_it_ships():
    """nvidia/Qwen3.8-27B-NVFP4 (deploy/stage-qwen38-27b-nvfp4.sh): the dense 27B in ModelOpt mixed precision,
    with the model's own MTP head in the checkpoint — so it launches with a drafter and a drafter view, as
    Flash-Next does. It replaced the bf16 build, which was never started on the pair."""
    spec = shipped().get(DENSE)
    p = spec.profile(CLUSTER_MODE)
    assert (spec.served_name, spec.path, spec.name) == ("Qwen/Qwen3.8-27B", "Qwen3.8-27B-NVFP4",
                                                        "Qwen3.8-27B-NVFP4")
    assert p.image == "myllmbox/qwen38-flash-next-cluster-vllm:v5"
    assert p.speculative_config == {"method": "mtp", "num_speculative_tokens": 4}
    assert (p.tool_call_parser, p.reasoning_parser) == ("qwen3_xml", "qwen3")
    assert spec.cluster_dirs == ("~/models", "/mnt/box-a-home/models", "/srv/models"), \
        "box-a's NVMe copy first — box-b reaches it through box-a's share — and the HDD copy last"
    assert p.default is False and spec.profile(CLASSIC) is None
    assert p.mtp_prefix == "mtp." and p.ple_replicate is None, "the drafter view on; no PLE table at all"
    assert p.kv_cache_memory == 48_000_000_000, "its own pool: a 10 GiB rank leaves room Flash-Next never had"
    assert (spec.memory.weights_gib, spec.memory.ple_gib, spec.memory.kv_bytes_per_token) == (10.28, 0.0, 36860)
    assert "2026-10-02" in spec.memory.calibration and "103" in spec.memory.calibration, \
        "measured on generations 103 and 104 (lb/tests/fixtures/cluster-27b-2026-10-02-memory.json)"


def test_the_classic_default_is_what_the_broker_sidecar_is_seeded_with():
    """seed-broker-sidecar.sh:32-35: /models/base.qwen3.8-dense-nvfp4-vllm served as qwen3.8:latest on v0.23.0."""
    spec = shipped().get(SEEDED)
    p = spec.profile(CLASSIC)
    assert (spec.path, spec.served_name, p.image) == ("base.qwen3.8-dense-nvfp4-vllm", "qwen3.8:latest",
                                                      "vllm/vllm-openai:v0.23.0")
    assert p.default is True and spec.profile(CLUSTER_MODE) is None
    assert spec.memory is None, "classic vLLM's memory is set by broker_parrot, not estimated here"


# ── the one validator ─────────────────────────────────────────────────────────

def test_choice_is_the_one_validator_for_both_serving_modes():
    c = shipped()
    assert c.choice(CLUSTER_MODE, DENSE).id == DENSE
    assert c.choice(CLASSIC, SEEDED).id == SEEDED
    for mode, bad in ((CLUSTER_MODE, "nope"), (CLUSTER_MODE, SEEDED), (CLASSIC, FLASH), (CLUSTER_MODE, ""),
                      ("solo", FLASH)):
        with pytest.raises(UnknownModelChoice) as info:
            c.choice(mode, bad)
        assert isinstance(info.value, ValueError)
    with pytest.raises(UnknownModelChoice, match=f"{FLASH}.*{DENSE}"):
        c.choice(CLUSTER_MODE, "nope")                         # the message names what can be chosen


def test_a_separate_drafter_is_named_RELATIVE_to_the_one_dir_tp_rank_mounts():
    """The defect this test exists for: `draft_path` is written under the models root, like `path`, but
    tp-rank mounts a root joined with the checkpoint's own folder and names everything inside it relative
    to THAT. Sending the catalog's path verbatim made both ranks refuse the start with
    `the drafter dflash2-27b-tp2/Qwen3.8-27B-DFlash2 has no config.json under
    /home/user/models/dflash2-27b-tp2` — the folder counted twice (2026-10-02, generation 106)."""
    spec = shipped().get(DFLASH)
    assert spec.profile(CLUSTER_MODE).draft_path == "dflash2-27b-tp2/Qwen3.8-27B-DFlash2"
    assert spec.cluster_dirs[0].endswith("/dflash2-27b-tp2"), spec.cluster_dirs
    # What tp-rank is sent, and what it will therefore stat: $MODELS_DIR/$CLUSTER_DRAFT_NAME.
    assert spec.draft_name == "Qwen3.8-27B-DFlash2"
    assert "/" not in spec.draft_name, "a drafter beside the checkpoint is a NAME, not a path"
    for d in spec.cluster_dirs:
        assert f"{d}/{spec.draft_name}" == f"{d}/Qwen3.8-27B-DFlash2"


def test_dflash2_offers_the_48_seats_the_gateway_allows_and_the_catalog_says_what_the_gate_will_not_charge():
    """The operator's decision of 2026-10-02: this model offers the SAME concurrency range as the other two
    cluster choices — 1..48 — and not the 8 its footprint happened to be measured at.

    48 and not more, because the ceiling is arithmetic rather than preference. `MAX_NUM_SEQS_CAP` mirrors
    tp-rank's own `1..48` (`cmd_up`), and a catalog ceiling the launcher refuses would be a worse bug than a
    low one; below that, the 28 GB pin holds 985 blocks of 864 tokens, and one request costs
    ceil(ctx / 864) + 18 of them — so the pool admits 49 requests at the 1,024-token floor and 3 at the
    262,144 default. 64 seats were asked for and are impossible at this pin at ANY context: their
    `fixed_blocks` alone are 64 x 18 x 28,418,000 = 32.7 GB, more than the whole pool.

    What 48 seats do NOT come with is an estimate that can refuse them. `concurrency_gib_per_seq` is 0.0
    here, so seats 9..48 are charged nothing and the start gate cannot stop a start that will not fit. The
    catalog is where the next person meets that, so `calibration` has to say it.
    """
    spec = shipped().get(DFLASH)
    assert spec.limits.max_num_seqs == (1, 48)
    assert spec.limits.max_num_seqs == MAX_NUM_SEQS_CAP, "the gateway's own ceiling, not a number of its own"
    # The ceiling itself did not move, and is still tp-rank's: a catalog that asks for more is refused.
    assert MAX_NUM_SEQS_CAP == (1, 48)
    assert f'int_in "$seqs" {MAX_NUM_SEQS_CAP[0]} {MAX_NUM_SEQS_CAP[1]}' in TP_RANK.read_text(), \
        "the cap mirrors tp-rank's `cmd_up` bound — the launcher is what really refuses a seat count"
    doc = raw()
    spec_raw(doc, DFLASH)["limits"]["max_num_seqs"] = [1, 64]
    with pytest.raises(ValueError, match=r"max_num_seqs must be \[low, high\] within 1\.\.48"):
        ModelCatalog.from_dict(doc)
    # The arithmetic the decision rests on, read off this model's own block rule and pin.
    kb, pin = spec.memory.kv_blocks, spec.profile(CLUSTER_MODE).kv_cache_memory
    assert (kb.blocks_in(pin), kb.per_request(262144), kb.per_request(1024)) == (985, 334, 20)
    assert (kb.seats(pin, 1024), kb.seats(pin, 262144)) == (49, 2)
    assert 64 * kb.fixed_blocks * kb.block_bytes > pin, "64 seats cannot hold even their fixed blocks"
    # A dflash step submits 1 + K queries a seat; at 48 and K=7 that is inside the prefill budget.
    assert 48 * (1 + spec.profile(CLUSTER_MODE).num_speculative_tokens) <= spec.memory.calibrated_batched_tokens
    # Seats above the measured anchor cost the estimate nothing here, and the catalog must not be silent.
    assert (spec.memory.concurrency_gib_per_seq, spec.memory.calibrated_max_num_seqs) == (0.0, 8)
    assert "charges nothing for seats 9..48" in spec.memory.calibration, \
        "the catalog states the blindness, so it is not learned from an agent report"


def test_the_status_view_tells_the_cluster_models_apart_by_engine_and_drafter():
    """What the cluster card needs to name the model it is loading. Three of the five cluster models are
    the SAME 27B checkpoint family and their labels all begin "Qwen3.8 27B", while TWO MORE are the same
    Flash-Next weights on different engines and their labels both begin "Qwen3.8 Flash-Next hibrid48", so
    `engine` and `speculative_method` are what distinguishes them on the wire."""
    view = {v["id"]: v["launch"]["cluster"] for v in shipped().view() if "cluster" in v["launch"]}
    assert {k: (v["engine"], v["speculative_method"]) for k, v in view.items()} == {
        FLASH: ("vllm", "mtp"), FLASH_SGLANG: ("sglang", "nextn"), DENSE: ("vllm", "mtp"),
        DFLASH: ("vllm", "dflash"), SGLANG: ("sglang", "dflash"), GLM: ("vllm", "mtp"), DSV4: ("vllm", "dspark"),
        # TensorFold drafts with the checkpoint's MTP head by its own rule: no speculative_config the LB shapes.
        TF_NATIVE: ("tensorfold", None), TF_MLX: ("tensorfold", None)}
    # And the served names stay distinct, which is the page's last resort when a label is ambiguous.
    served = [v["served_name"] for v in shipped().view()]
    assert len(set(served)) == len(served), served


#: What each engine's own `--help` accepts, read off the images this catalog pins (2026-10-03). Only the
#: names a profile here could plausibly want are listed; the point is that the two vocabularies DIFFER.
_PARSERS = {
    "vllm": {"tool": {"qwen3_xml", "qwen3_coder", "hermes", "mistral", "llama3_json"},
             "reasoning": {"qwen3", "deepseek_r1", "granite"}},
    # `python3 -m sglang.launch_server --help` inside lmsysorg/sglang@sha256:00205b89…
    "sglang": {"tool": {"auto", "qwen", "qwen25", "qwen3_coder", "glm", "glm45", "hermes", "mistral",
                        "pythonic", "llama3", "deepseekv3"},
               "reasoning": {"auto", "qwen3", "qwen3-thinking", "deepseek-r1", "glm45", "kimi"}},
    # TensorFold takes no parser flag at all (server/cli.zig, 0.6.6 cli_args.py): its server renders tool calls and
    # reasoning itself, so a TensorFold profile names none.
    "tensorfold": {"tool": set(), "reasoning": set()},
}
#: An image whose vLLM is not the pinned v0.29 above carries its own vocabulary, keyed by the image's repository.
#: The GLM-5.3-Flash base (ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6…, image id 35c6f70f…), read
#: 2026-10-20 off vllm/tool_parsers/__init__.py and vllm/reasoning/__init__.py inside it.
_IMAGE_PARSERS = {
    "myllmbox/glm53-flash-cluster-vllm": {"tool": {"glm45", "glm47", "hermes", "qwen3_xml", "qwen3_coder"},
                                          "reasoning": {"glm45", "glm47", "deepseek_r1", "qwen3"}},
    # eugr/box-a-vllm-b12x@sha256:03a95654… (vLLM 0.1.dev21830), read 2026-10-23 off the same two files.
    "myllmbox/deepseek-v4-flash-cluster-vllm": {"tool": {"deepseek_v4", "deepseek_v3", "deepseek_v31"},
                                                "reasoning": {"deepseek_v4", "deepseek_r1"}},
}


def test_every_profile_names_a_parser_ITS_OWN_engine_accepts():
    """The two engines do not share a parser vocabulary, and the catalog field does not say which engine
    a name belongs to. The SGLang profile shipped with vLLM's `qwen3_xml`, and the first start that
    reached SGLang died on it:

        sglang serve: error: argument --tool-call-parser: invalid choice: 'qwen3_xml'
        (choose from 'auto', …, 'qwen', 'qwen25', 'qwen3_coder', …)

    exit code 2, before a single weight was read. `qwen3` happens to be valid for BOTH engines'
    reasoning parsers, which is exactly why only the tool parser failed and why reading one name as
    proof of the other would be wrong."""
    for spec in shipped().for_mode(CLUSTER_MODE):
        p = spec.profile(CLUSTER_MODE)
        ok = _IMAGE_PARSERS.get(p.image.split(":")[0]) or _PARSERS[p.engine]
        assert not p.tool_call_parser or p.tool_call_parser in ok["tool"], (
            f"{spec.id} runs on {p.engine}, which does not take tool parser {p.tool_call_parser!r}")
        assert not p.reasoning_parser or p.reasoning_parser in ok["reasoning"], (
            f"{spec.id} runs on {p.engine}, which does not take reasoning parser {p.reasoning_parser!r}")
    # And the specific pair that caught us: the two engines really do disagree about this one name.
    assert "qwen3_xml" in _PARSERS["vllm"]["tool"] and "qwen3_xml" not in _PARSERS["sglang"]["tool"]


def test_the_sglang_profile_names_its_engine_and_keeps_the_one_client_address():
    """A fourth cluster model, served by a different program. The ServingMode is still `cluster` — the two
    GB10 boxes, TP=2, one engine, reached at :8010 — and `engine` is the only thing that differs, which is how
    the operator's rule ("only one and always use 8010 port") survives a second engine existing at all."""
    spec = shipped().get(SGLANG)
    p = spec.profile(CLUSTER_MODE)
    # The fleet's build of the digest-pinned upstream image (deploy/images/sglang: FROM
    # lmsysorg/sglang@sha256:00205b89…, plus the patch series its Dockerfile lists). Upstream alone is
    # not enough: Claude Code's parallel tool calls lost a brace, a call cut by max_tokens arrived as
    # `{}`, and its default `thinking.display` was a 400. The tag names the series, so a box holding
    # an older build can never pass for this one.
    assert p.engine == "sglang" and p.image == "myllmbox/sglang-27b-tp2:708f51e-fleet.2", p.image
    assert spec.draft_name == "Qwen3.8-27B-DFlash2", "the DFlash2 drafter, beside the checkpoint"
    assert p.speculative_config == {"method": "dflash", "num_speculative_tokens": 4}, \
        "4 drafts, measured: 278.69 tok/s against 8's 254.73 at 16 seats (worklog/sglang-27b-tp2-c16.md)"
    # TEN GDN state slots a seat: FIVE are the engine's own bound for a running request (it caps
    # max_running_requests at pool // 5 — at four a seat, 8 seats became 6), and the other five are where
    # the radix cache keeps each session's checkpoint between its turns. At five a seat, 30 running
    # requests held ~120 of 160 and the cache deleted live prefixes with 2.4 M KV tokens free.
    assert p.engine_args["mamba_slots_per_req"] == 10, \
        "5 a running request (pool // 5 caps the seats) + 5 checkpoints a seat for the prefix cache"
    # Every other cluster model is vLLM, and says so by saying nothing: `engine` defaults to it.
    assert [s.profile(CLUSTER_MODE).engine for s in shipped().for_mode(CLUSTER_MODE)] == \
        ["vllm", "sglang", "tensorfold", "tensorfold", "vllm", "vllm", "sglang", "vllm", "vllm"]


def test_the_sglang_pin_and_its_mem_fraction_describe_the_same_measurement():
    """SGLang sizes weights AND the KV pools as a fraction of the box; every other profile here states an
    absolute pin, and the memory estimate, the start gate and the operator's knob are all written against
    that pin. The entry therefore carries both, and they must not drift apart — a pin that says one thing
    while the engine is told another is how a start gate passes a launch that cannot fit."""
    spec = shipped().get(SGLANG)
    p = spec.profile(CLUSTER_MODE)
    box_gib = 121.0                       # what a GB10 box reports to `free -g`, both of them
    weights_gib = spec.memory.weights_gib + spec.memory.mtp_gib
    pin_gib = p.kv_cache_memory / (1 << 30)
    implied = (weights_gib + pin_gib) / box_gib
    # The relation is `<=`, not `==`, and the difference is a start that was REFUSED. The fraction is
    # what the engine MAY take for weights and KV together; the pin is the KV pool it actually built.
    # Writing the fraction's whole share into the pin counts the weights twice — this estimate adds
    # weights and overhead ON TOP of the pin — and generation 117 was refused at "need 117" on a 115 GiB
    # box for exactly that. Under-stating the pin is the dangerous direction and the ceiling catches it.
    assert implied <= p.engine_args["mem_fraction_static"] + 0.005, (
        f"the pin implies mem-fraction {implied:.3f}, above the {p.engine_args['mem_fraction_static']} "
        f"the engine is told — the pin cannot claim more than the fraction allows")
    # The lower bound is deliberately loose, and generation 123 is why: with the pin corrected to the
    # engine's OWN two pools (68.4 GiB) the implied fraction is 0.680 against the 0.80 it is told, so
    # SGLang left a sixth of its allowance unused. The fraction is a ceiling the engine may take, not a
    # figure it fills, and how much it leaves depends on the seats and the mamba cache. The real guard
    # against a pin that under-states the box is the total below, which is checked against a measurement.
    assert implied > p.engine_args["mem_fraction_static"] - 0.20, (
        f"the pin implies {implied:.3f}, far below the fraction: a pool that small would under-state "
        f"what the box must hold")
    # And the whole estimate reproduces what the boxes really held: 107 (box-a) / 105 (box-b) GiB.
    # Generation 123's own reading: 106 GiB held with 15.05 GB free, on a 121 GiB box.
    whole = weights_gib + pin_gib + p.overhead_gib["head"]
    assert 104.0 <= whole <= 109.0, f"the head rank estimates {whole:.1f} GiB against a measured 106"
    # And the pin divided by the per-token cost must give back the engine's own pool, not a figure 36%
    # under it — the defect that printed two token counts for one pool on the same screen.
    assert round(p.kv_cache_memory / spec.memory.kv_bytes_per_token / 1000) == 3415, (
        f"pin/bytes-per-token is {p.kv_cache_memory / spec.memory.kv_bytes_per_token:,.0f} tokens, "
        f"against the engine's own max_total_num_tokens of 3,415,240")
    # And the derivation the profile offers agrees with what is written down.
    # The derivation the profile offers is the SAME quantity the assertions above bound: what weights
    # plus this pin would ask of the box, as a fraction. It sits under what the engine is told, because
    # the engine did not fill the share it was allowed.
    # Since the KV-pool calibration (81fa32d) the fraction the engine is TOLD comes from a measured launch, not
    # from weights over the box: the shipped pin IS that launch's pool (`reference_bytes`), so it is told exactly
    # the measured mem_fraction_static — and a pin past the profile's safe ceiling is refused, never shrunk.
    assert p.engine_args["kv_pool_calibration"]["reference_bytes"] == p.kv_cache_memory
    assert p.sglang_mem_fraction(p.kv_cache_memory) == p.engine_args["mem_fraction_static"]
    with pytest.raises(ValueError, match="ceiling"):
        p.sglang_mem_fraction(p.kv_cache_memory + 40 * (1 << 30))
    for nonsense in (None, 0, -1, True):                   # no pin is no pool: refused, never a 0.0 fraction
        with pytest.raises(ValueError):
            p.sglang_mem_fraction(nonsense)


def test_every_cluster_row_states_the_seats_and_context_it_is_offered():
    """THE ROW OWNS ITS START INPUT (operator, 2026-10-29: "why do we have LB_CONTEXT_LENGTH? Isn't that set in configs?
    Same for MAX_NUM_SEQS?"): a cluster LaunchProfile must state `tuned.max_num_seqs` and `tuned.max_model_len` — the
    concurrency and context a Start of it is offered — with the source they came from. The gateway-wide
    LB_CLUSTER_MAX_NUM_SEQS / LB_CLUSTER_MAX_MODEL_LEN that stood in for a row with nothing stated are retired, so a
    row without them is a document error, never a launch at a number nobody chose for that model."""
    for spec in shipped().for_mode(CLUSTER_MODE):
        tuned = spec.profile(CLUSTER_MODE).tuned
        assert tuned is not None and tuned.max_num_seqs is not None and tuned.max_model_len is not None, spec.id
        assert tuned.source.strip(), spec.id
    doc = raw()
    del spec_raw(doc, DENSE)["launch"]["cluster"]["tuned"]
    with pytest.raises(ValueError, match=rf"model {DENSE}: .*tuned\.max_num_seqs and tuned\.max_model_len.*"
                                         r"LB_CLUSTER_MAX_NUM_SEQS"):
        ModelCatalog.from_dict(doc)
    for knob in ("max_num_seqs", "max_model_len"):
        doc = raw()
        del spec_raw(doc, FLASH)["launch"]["cluster"]["tuned"][knob]
        with pytest.raises(ValueError, match=rf"model {FLASH}: .*tuned\.max_num_seqs and tuned\.max_model_len"):
            ModelCatalog.from_dict(doc)


def test_a_tuned_start_input_is_validated_as_strictly_as_the_rest_of_the_document():
    """A model's MEASURED-BEST start input (`TunedSetup`, 2026-10-05) is a default the page fills in for the
    operator and a launch really uses, so the document must not be able to state a nonsense one: every knob
    inside tp-rank's own caps AND the model's own ModelLimits, a `source` that names the run it came from, no
    depth for a model with no drafter, no empty block, and the prefill budget stated in ONE place."""
    def refused(where, match, **tuned):
        doc = raw()
        spec_raw(doc, FLASH)["launch"]["cluster"]["tuned"] = {**TUNED, **tuned} if tuned else where
        with pytest.raises(ValueError, match=match):
            ModelCatalog.from_dict(doc)

    TUNED = dict(shipped().get(FLASH).profile(CLUSTER_MODE).tuned.view())
    TUNED.pop("max_num_batched_tokens")             # a block may leave the budget and the depth unstated
    TUNED = {k: v for k, v in TUNED.items() if v is not None}
    assert ModelCatalog.from_dict({**raw(), "models": raw()["models"]}).error == ""

    refused(None, "tuned max_num_seqs must be a whole number in 1..48", max_num_seqs=0)
    refused(None, "tuned max_num_seqs must be a whole number in 1..48", max_num_seqs=49)
    refused(None, "tuned max_num_batched_tokens must be a whole number in 2048..65536",
            max_num_batched_tokens=1024)
    refused(None, "tuned num_speculative_tokens must be a whole number in 0..8", num_speculative_tokens=9)
    refused(None, "tuned max_model_len must be a whole number in 1024..262144", max_model_len=262145)
    refused({"max_num_seqs": 6}, r"tuned: missing \['source'\]")
    refused({"source": "", "max_num_seqs": 6}, "tuned source must be a non-empty string")
    refused({"source": "a run"}, "tuned must state at least one of")
    refused({"source": "a run", "num_speculative_tokens": 4, "parallel": 8}, r"unknown \['parallel'\]")

    # Inside the MODEL's own limits, not just tp-rank's: the switch would refuse a selection outside them,
    # so a default there could never be applied.
    doc = raw()
    spec_raw(doc, FLASH)["limits"]["max_num_seqs"] = [1, 4]
    with pytest.raises(ValueError, match="tuned max_num_seqs 6 is outside this model's own limits 1..4"):
        ModelCatalog.from_dict(doc)
    doc = raw()
    spec_raw(doc, FLASH)["limits"]["max_model_len"] = 131072
    with pytest.raises(ValueError, match="tuned max_model_len 262144 is outside this model's own limits"):
        ModelCatalog.from_dict(doc)

    # A depth for a model with no drafter, and a budget stated twice.
    doc = raw()
    spec_raw(doc, FLASH)["launch"]["cluster"]["speculative_config"] = None
    spec_raw(doc, FLASH)["launch"]["cluster"]["mtp_prefix"] = ""
    with pytest.raises(ValueError, match="0 .off. is the only depth such a model can be tuned to"):
        ModelCatalog.from_dict(doc)
    doc = raw()
    spec_raw(doc, SGLANG)["launch"]["cluster"]["engine_args"]["chunked_prefill_size"] = 8192
    with pytest.raises(ValueError, match="the prefill budget is stated twice"):
        ModelCatalog.from_dict(doc)

    # And a classic profile has no cluster start input to tune.
    doc = raw()
    spec_raw(doc, SEEDED)["launch"]["classic"]["tuned"] = {"source": "a run", "max_num_seqs": 4}
    with pytest.raises(ValueError, match="tuned states the CLUSTER's start input"):
        ModelCatalog.from_dict(doc)


def test_a_drafter_outside_the_checkpoints_own_folder_is_refused_by_the_catalog():
    """tp-rank reaches exactly one dir. A drafter anywhere else cannot be mounted, so the catalog says so
    at load rather than letting two ranks fail on a path the page cannot show."""
    doc = raw()
    spec_raw(doc, DFLASH)["launch"]["cluster"]["draft_path"] = "somewhere-else/Qwen3.8-27B-DFlash2"
    with pytest.raises(ValueError, match="must sit beside the checkpoint"):
        ModelCatalog.from_dict(doc)
    # And through `load`, which never raises, the same thing is the status line the page shows — the drafter's folder
    # being the deployment's (the fleet file's checkpoints.<id>.draft).
    from dataclasses import replace
    from vllm_lb.catalog import current_deployment
    dep = current_deployment()
    staged = dict(dep.checkpoints)
    staged[DFLASH] = replace(staged[DFLASH], draft="somewhere-else/Qwen3.8-27B-DFlash2")
    assert "must sit beside the checkpoint" in ModelCatalog.load(deployment=replace(dep, checkpoints=staged)).error


def test_a_drafter_needs_a_speculative_config_and_excludes_an_in_checkpoint_prefix():
    """The two drafter sources are exclusive, and a drafter with nothing to point at it is a mistake."""
    doc = raw()
    spec_raw(doc, DFLASH)["launch"]["cluster"]["mtp_prefix"] = "mtp."
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelCatalog.from_dict(doc)
    doc = raw()
    del spec_raw(doc, DFLASH)["launch"]["cluster"]["speculative_config"]
    with pytest.raises(ValueError, match="no speculative_config to point at it"):
        ModelCatalog.from_dict(doc)


def test_an_unsupported_spec_is_listed_but_cannot_be_chosen_and_says_why():
    """M6's way out: a model that cannot load on the image is marked unsupported with the log line."""
    doc = raw()
    evidence = "vLLM 0.29 on v5: ValueError: Qwen3_5ForConditionalGeneration has no TP=2 MTP support"
    spec_raw(doc, DENSE)["unsupported"] = evidence
    c = ModelCatalog.from_dict(doc)
    assert [s.id for s in c.for_mode(CLUSTER_MODE)] == [FLASH, FLASH_SGLANG, TF_NATIVE, TF_MLX, DENSE, DFLASH, SGLANG, GLM, DSV4], \
        "still listed, so the page can say why"
    with pytest.raises(ValueError, match="unsupported") as info:
        c.choice(CLUSTER_MODE, DENSE)
    assert not isinstance(info.value, UnknownModelChoice) and evidence in str(info.value)
    assert next(v for v in c.view() if v["id"] == DENSE)["unsupported"] == evidence


# ── the rules a catalog file must keep ────────────────────────────────────────

def _mutate(fn):
    doc = copy.deepcopy(raw())
    fn(doc)
    return doc


BROKEN = {
    "duplicate id": lambda d: spec_raw(d, DENSE).update(id=FLASH),
    "duplicate served name": lambda d: spec_raw(d, DENSE).update(served_name="Qwen/Qwen3.8-Flash-Next"),
    "an id tp-rank's argv refuses": lambda d: spec_raw(d, DENSE).update(id="Qwen 27B"),
    "a path that climbs out": lambda d: spec_raw(d, DENSE).update(path="../Qwen3.8-27B"),
    "an absolute path": lambda d: spec_raw(d, DENSE).update(path="/home/user/models/Qwen3.8-27B"),
    "an empty path segment": lambda d: spec_raw(d, DENSE).update(path="models//Qwen3.8-27B"),
    "a window past tp-rank's cap": lambda d: spec_raw(d, DENSE)["limits"].update(max_model_len=262145),
    "a window under tp-rank's floor": lambda d: spec_raw(d, DENSE)["limits"].update(max_model_len=512),
    "zero seats": lambda d: spec_raw(d, DENSE)["limits"].update(max_num_seqs=[0, 8]),
    "more seats than tp-rank allows": lambda d: spec_raw(d, DENSE)["limits"].update(max_num_seqs=[1, 49]),
    "two cluster defaults": lambda d: spec_raw(d, DENSE)["launch"]["cluster"].update(default=True),
    "no cluster default": lambda d: spec_raw(d, FLASH)["launch"]["cluster"].update(default=False),
    "a speculative config that is not an object": lambda d: spec_raw(d, DENSE)["launch"]["cluster"].update(
        speculative_config="mtp"),
    "a ServingMode nobody has": lambda d: spec_raw(d, DENSE)["launch"].update(solo={"image": "x"}),
    "a profile without a build": lambda d: spec_raw(d, DENSE)["launch"]["cluster"].pop("build"),
    "an image written in the row": lambda d: spec_raw(d, DENSE)["launch"]["cluster"].update(image="x:1"),
    "a row without its architectures": lambda d: spec_raw(d, DENSE).pop("architectures"),
    "a typo in a key": lambda d: spec_raw(d, DENSE).update(servedname="Qwen/Qwen3.8-27B"),
    "a launch that maps nothing": lambda d: spec_raw(d, DENSE).update(launch=["cluster"]),
    "a cluster spec with no memory footprint": lambda d: spec_raw(d, DENSE).pop("memory"),
    "a negative weight": lambda d: spec_raw(d, DENSE)["memory"].update(weights_gib=-1),
    "an overhead for a role no rank has": lambda d: spec_raw(d, DENSE)["launch"]["cluster"].update(
        overhead_gib={"head": 12.0, "boss": 7.0}),
    "a PLE mode that is not a bool": lambda d: spec_raw(d, FLASH)["launch"]["cluster"].update(ple_replicate="1"),
}


@pytest.mark.parametrize("why", sorted(BROKEN))
def test_a_catalog_that_breaks_a_rule_is_refused_whole_with_the_reason(why):
    with pytest.raises(ValueError):
        ModelCatalog.from_dict(_mutate(BROKEN[why]))


# ── where the cluster's checkpoints live: the deployment's, from the environment ─────────────────────────────

def test_the_folders_the_cluster_searches_come_from_the_environment_not_the_catalog(monkeypatch):
    """MODELS_SEARCH_DIRS (lb/.env), best first, `~` the BOX's home (operator, 2026-10-09: "models dir also should be
    in .env defined"). A public install keeps its models elsewhere, so the shipped catalog names no folder at all."""
    assert "cluster_roots" not in document(), "the catalog names no deployment's folders"
    monkeypatch.setenv("MODELS_SEARCH_DIRS", "~/fast:/srv/models")
    spec = ModelCatalog.load().get(FLASH)
    group = spec.path.rsplit("/", 1)[0]
    assert spec.cluster_dirs == (f"~/fast/{group}", f"/srv/models/{group}")


def test_no_search_folders_is_a_catalog_error_naming_the_variable(monkeypatch):
    """Nothing to fall back to: an LB with no MODELS_SEARCH_DIRS says so where the page reads the catalog's error,
    and offers no cluster model to start."""
    monkeypatch.delenv("MODELS_SEARCH_DIRS", raising=False)
    c = ModelCatalog.load()
    assert c.error and "MODELS_SEARCH_DIRS" in c.error, c.error
    assert c.for_mode(CLUSTER_MODE) == ()


def test_a_catalog_that_still_names_cluster_roots_is_refused_pointing_at_the_variable():
    raw = document()
    raw["cluster_roots"] = ["~/models"]
    with pytest.raises(ValueError, match="MODELS_SEARCH_DIRS"):
        ModelCatalog.from_dict(raw)


def test_a_broken_catalog_file_is_reported_not_fatal(tmp_path, monkeypatch):
    """The LB listens first (test_main_boot.py): a catalog it cannot read is a status line and a refused
    Start, never a crash — and there is no hidden second copy of the default to fall back to. The catalog is assembled
    for the fleet file LB_FLEET names: a fleet file it cannot read is such a catalog."""
    for name, text in (("garbled.yml", "boxes: ["), ("missing.yml", None)):
        path = tmp_path / name
        if text is not None:
            path.write_text(text)
        monkeypatch.setenv("LB_FLEET", str(path))
        c = ModelCatalog.load()
        assert c.error and name in c.error, c.error
        assert c.for_mode(CLUSTER_MODE) == () and c.default(CLUSTER_MODE) is None
        with pytest.raises(ValueError, match="catalog"):
            c.choice(CLUSTER_MODE, FLASH)


# ── observations, named ───────────────────────────────────────────────────────

def test_the_loaded_model_is_what_the_endpoint_serves_mapped_back_to_its_spec():
    c = shipped()
    assert c.loaded_model(("Qwen/Qwen3.8-27B",), 131072) == LoadedModel(
        served_name="Qwen/Qwen3.8-27B", spec=DENSE, label=c.get(DENSE).label, max_model_len=131072,
        efforts=("xhigh", "medium", "low", "off"))
    stranger = c.loaded_model(("someone/else",), None)
    assert (stranger.served_name, stranger.spec, stranger.label) == ("someone/else", None, None)
    assert c.loaded_model((), 262144) is None, "nothing answering is no LoadedModel — not the ModelChoice"
    assert c.loaded_model(("qwen3.8:latest",), 262144).view() == {
        "served_name": "qwen3.8:latest", "spec": SEEDED, "label": c.get(SEEDED).label, "max_model_len": 262144,
        "efforts": ["xhigh", "medium", "low", "off"]}


def test_availability_is_present_only_when_the_files_and_the_image_are_both_on_the_box():
    searched = ("/home/user/models", "/mnt/box-a-home/models", "/srv/models")

    def seen(**over) -> Availability:
        return Availability(**{"name": "Qwen3.8-27B", "searched": searched, "dir": "/home/user/models",
                               "image": "myllmbox/qwen38-flash-next-cluster-vllm:v5", "image_present": True,
                               "observed_at": 100.0, "error": "", **over})

    assert seen().present is True
    missing = seen(dir=None)
    assert missing.present is False
    assert missing.reason() == ("no Qwen3.8-27B/config.json in /home/user/models, "
                                "/mnt/box-a-home/models, /srv/models")
    no_image = seen(image_present=False)
    assert no_image.present is False and no_image.reason() == "no image myllmbox/qwen38-flash-next-cluster-vllm:v5"
    assert seen(image_present=None).present is None, "a box that cannot say is unknown, not missing"
    assert seen(dir=None, error="/srv/models: timed out after 5 s").present is None
    assert seen().view() == {"name": "Qwen3.8-27B", "searched": list(searched), "dir": "/home/user/models",
                             "image": "myllmbox/qwen38-flash-next-cluster-vllm:v5", "image_present": True,
                             "present": True, "observed_at": 100.0, "error": "", "unusable": ""}
    # Found, but not readable from the container (generation 279): certainly not launchable, and it says which files.
    broken = seen(unusable="Qwen3.8-27B/tokenizer.json -> /x/blobs/abc (outside /home/user/models)")
    assert broken.present is False
    assert broken.reason() == ("Qwen3.8-27B in /home/user/models cannot be read from the container: "
                               "Qwen3.8-27B/tokenizer.json -> /x/blobs/abc (outside /home/user/models)")


def test_the_status_view_carries_what_both_pickers_need():
    view = {v["id"]: v for v in shipped().view()}
    # WHAT EACH PROFILE RESOLVED TO on this deployment's cards (resolve.py; the suite's fleet is the reference pair):
    # its image, the matrix entry's source, the target — checked here, then set aside for the rest of the view.
    for v in view.values():
        for p in v["launch"].values():
            r = p.pop("resolution")
            assert r["refused"] == "" and r["image"] == p["image"] and r["image_source"], (v["id"], r)
            assert r["target"]["accelerator"] == "nvidia-sm121" and r["build"] == p["build"]
    assert list(view) == [FLASH, FLASH_SGLANG, TF_NATIVE, TF_MLX, DENSE, SEEDED, DFLASH, SGLANG, GLM, DSV4, "mai-ui-2b", "whisper-large-v3-turbo",
                          "faster-whisper-large-v3-turbo", "faster-whisper-large-v3",
                          "parakeet-tdt-0.6b-v3"]
    assert view[DENSE] == {
        "id": DENSE, "label": "Qwen3.8 27B · dense NVFP4", "served_name": "Qwen/Qwen3.8-27B", "size_gib": 20.4,
        "limits": {"max_model_len": 262144, "max_num_seqs": [1, 48]},
        "launch": {"cluster": {"default": False, "image": "myllmbox/qwen38-flash-next-cluster-vllm:v5", "build": "fleet",
                               "speculative": True, "num_speculative_tokens": 4,
                               # Which server, and which drafter. The page had neither before 2026-10-03:
                               # it called every drafter "MTP" and could not tell two profiles of ONE
                               # checkpoint apart, which is what the operator saw on the cluster card.
                               "engine": "vllm", "speculative_method": "mtp",
                               # The MODEL's own pool, and no PLE table to place: the picker's defaults.
                               "kv_cache_memory": 48_000_000_000, "ple_replicate": None,
                               # vLLM pins what the gate affords: no ceiling of its own (SGLang's: the
                               # calibrated bound, test_a_profile_states_the_largest_kv_pin_it_can_take…).
                               "kv_cache_memory_max": None,
                               # NOTHING BENCHED FOR THIS ONE, but the row still states the seats and context it
                               # is offered (the figures the gateway's former defaults gave it, said so in the
                               # source); no budget and no depth, so the box's and the profile's own stand.
                               "tuned": {"max_num_seqs": 6, "max_model_len": 262144, "max_num_batched_tokens": None,
                                         "num_speculative_tokens": None,
                                         "source": shipped().get(DENSE).profile(CLUSTER_MODE).tuned.source},
                               # vLLM takes every knob the panel has (SGLang's: tests/test_sglang_knobs.py).
                               "refuses": {},
                               # The lowest the image runs on GB10 (tests/test_kv_cache_dtype.py; 2026-10-26).
                               "kv_cache_dtype": "int4_per_token_head",       # 4-bit on Triton, generation 263
                               # Its depth is the LB's to set: only an engine that drafts by its own rule
                               # (TensorFold, tests/test_tensorfold_engine.py) says whether it drafts.
                               "engine_drafts": None, "pool_per_seat": False}},
        # What turns a pool into tokens is the model's, and so is whether anyone measured it — and the engine's
        # block rule for its seats (816-token blocks of 28.4 MB, 18 fixed per request; generation 105).
        "kv_bytes_per_token": 36860, "calibrated": True,
        "kv_blocks": {"block_tokens": 816, "block_bytes": 28_418_000, "fixed_blocks": 18,
                      "source": view[DENSE]["kv_blocks"]["source"], "kv_cache_dtype": "bfloat16"},
        # Its uncensored twin, the default checkpoint since 2026-10-07 (tests/test_uncensored.py).
        "uncensored": {"path": "Qwen3.8-27B-NVFP4_uncensored", "default": True,
                       "calibration": view[DENSE]["uncensored"]["calibration"]},
        # Its template's thinking levels, default first (tests/test_model_efforts.py).
        "efforts": ["xhigh", "medium", "low", "off"],
        "unsupported": ""}
    assert "cache_config_info 2026-10-02" in view[DENSE]["kv_blocks"]["source"]
    assert view[FLASH]["kv_blocks"]["block_tokens"] == 1616 and view[FLASH]["kv_blocks"]["fixed_blocks"] == 25
    assert view[SEEDED]["kv_blocks"] is None
    # The drafter's depth each model launches with when the operator sets none: the panel shows it.
    assert view[FLASH]["launch"] == {"cluster": {"default": True, "image": "myllmbox/qwen38-flash-next-cluster-vllm:v5", "build": "fleet",
                                                 "speculative": True, "num_speculative_tokens": 4,
                                                 "engine": "vllm", "speculative_method": "mtp",
                                                 "kv_cache_memory": 28_000_000_000, "ple_replicate": False,
                                                 "kv_cache_memory_max": None,
                                                 # ITS MEASURED-BEST START INPUT, which the page pre-fills
                                                 # the controls with the moment this model is picked — and
                                                 # the run it was measured on, which the note can show.
                                                 "tuned": {"max_num_seqs": 6, "max_model_len": 262144,
                                                           "max_num_batched_tokens": 4096,
                                                           "num_speculative_tokens": 4,
                                                           "source": view[FLASH]["launch"]["cluster"]["tuned"]["source"]},
                                                 "refuses": {}, "kv_cache_dtype": "fp8_e4m3",
                                                 "engine_drafts": None, "pool_per_seat": False}}
    assert "OPERATOR'S CHOICE 2026-10-05" in view[FLASH]["launch"]["cluster"]["tuned"]["source"]
    # The drafter depth the profile PUBLISHES is the one a default launch really uses: the tuned one where a
    # model has it (`LaunchProfile.num_speculative_tokens`), so the page can never offer a depth the launch
    # would not take.
    assert view[SGLANG]["launch"]["cluster"]["tuned"] == {
        "max_num_seqs": 32, "max_model_len": 262144, "max_num_batched_tokens": 2048,
        "num_speculative_tokens": 4, "source": view[SGLANG]["launch"]["cluster"]["tuned"]["source"]}
    assert "GENERATION 137" in view[SGLANG]["launch"]["cluster"]["tuned"]["source"]
    assert (view[FLASH]["kv_bytes_per_token"], view[FLASH]["calibrated"]) == (16392, True)
    assert view[SEEDED]["launch"] == {"classic": {"default": True, "image": "vllm/vllm-openai:v0.23.0", "build": "upstream-0.23",
                                                  "speculative": False, "num_speculative_tokens": 0,
                                                  "engine": "vllm", "speculative_method": None,
                                                  "kv_cache_memory": None, "ple_replicate": None,
                                                  "kv_cache_memory_max": None,
                                                  # A classic profile has no start input to tune: the panel's
                                                  # knobs are the CLUSTER's (`_tuned` refuses one here).
                                                  "tuned": None,
                                                  "refuses": {}, "kv_cache_dtype": None, "engine_drafts": None, "pool_per_seat": False}}
    # A classic spec has no MemoryFootprint: no token figure is offered for it, and nothing was measured.
    assert (view[SEEDED]["kv_bytes_per_token"], view[SEEDED]["calibrated"]) == (None, False)
    # The picker's "48 GB · 1,302,224 tokens" is this arithmetic, per model. 36,860 B is MEASURED: generation 103
    # logged `GPU KV cache size: 1,302,238 tokens` for the 48 GB pin (48e9 / 1,302,238 = 36,859.6) — not the
    # architectural 32,768, which would have promised 1,464,843.
    dense_tokens = view[DENSE]["launch"]["cluster"]["kv_cache_memory"] // view[DENSE]["kv_bytes_per_token"]
    assert dense_tokens == 1_302_224 and abs(dense_tokens - 1_302_238) / 1_302_238 < 0.0001
    assert view[FLASH]["launch"]["cluster"]["kv_cache_memory"] // view[FLASH]["kv_bytes_per_token"] == 1_708_150


# ── one name, one meaning ─────────────────────────────────────────────────────

def test_serving_modes_are_defined_once_and_the_cluster_switch_uses_them():
    import vllm_lb.catalog as catalog
    import vllm_lb.cluster as cluster
    assert (cluster.CLASSIC, cluster.CLUSTER_MODE) == (catalog.CLASSIC, catalog.CLUSTER_MODE) == ("classic", "cluster")
    tree = ast.parse(Path(cluster.__file__).read_text())
    assigned = {t.id for node in ast.walk(tree) if isinstance(node, ast.Assign) for t in node.targets
                if isinstance(t, ast.Name)}
    assert not {"CLASSIC", "CLUSTER_MODE"} & assigned, "cluster.py must import the ServingModes, not redefine them"


def test_the_lb_never_imports_broker_parrots_modelspec():
    """broker_parrot has its own `model_registry.ModelSpec` (an in-process loader). Here ModelSpec means one
    thing: vllm_lb.catalog.ModelSpec."""
    package = Path(__file__).resolve().parents[1] / "vllm_lb"
    offenders = [p.name for p in package.glob("*.py") if "model_registry" in p.read_text()]
    assert offenders == []


def test_the_sglang_moe_row_defaults_to_an_18_gb_kv_pool_inside_its_measured_ceiling():
    """Operator, 2026-10-10: raise the SGLang Flash-Next row's default KV pool from 7 GB (535,565 tokens: two
    full-context requests). 28 GB does not fit: with generation 186 serving the uncensored twin, the LB's own
    estimate (GET /lb/cluster/estimate, `auto`) put the largest pool at 18 GB — box-a binds, 27.1 GiB available
    at the 7 GB pool against its 12 GiB floor. 18 GB is also under the 19 GB generations 177-179 served at and
    inside the profile's 0.85 mem_fraction ceiling. The calibration's reference stays the 7 GB MEASUREMENT: it
    anchors the arithmetic, it is not the default."""
    spec = shipped().get(FLASH_SGLANG)
    profile = spec.profile(CLUSTER_MODE)
    assert profile.kv_cache_memory == 18_000_000_000
    assert profile.kv_cache_memory <= profile.sglang_max_kv_cache_memory()
    assert profile.sglang_mem_fraction(profile.kv_cache_memory) <= profile.engine_args["kv_pool_calibration"]["max_fraction"]
    assert profile.engine_args["kv_pool_calibration"]["reference_bytes"] == 6_993_411_168
    assert spec.launched_from(True).profile(CLUSTER_MODE).kv_cache_memory == 18_000_000_000, "the twin shares the row"


# ── the largest KV pin a profile can take, stated once and refused in one set of words ───────────────────────────────

def test_a_profile_states_the_largest_kv_pin_it_can_take_and_refuses_one_past_it_in_its_own_words():
    """GENERATION 187, 2026-10-10 20:55Z: the switch ACCEPTED a 24 GB pin on the SGLang Flash-Next row, and the launch
    refused it — `sglang_mem_fraction` raised on every attempt (131 times in six minutes) while the page flapped between
    `starting` and `freeing`. The ceiling is the profile's one fact; it is read from here by the switch, the estimate,
    the loop's launch check and the page, so no caller can accept what the launch will refuse."""
    sglang = shipped().get(FLASH_SGLANG).profile(CLUSTER_MODE)
    assert sglang.kv_cache_memory_max() == sglang.sglang_max_kv_cache_memory() == 19_918_363_626
    problem = sglang.kv_pin_problem(24_000_000_000)
    assert "mem_fraction_static 0.885" in problem and "safe 0.85 ceiling" in problem, problem
    assert "maximum 19,918,363,626 B" in problem, problem
    assert sglang.kv_pin_problem(18_000_000_000) == "" and sglang.kv_pin_problem(19_918_363_626) == ""
    # The bound carries the calibration's own 1e-9 tolerance (~117 B): a megabyte over it is over.
    assert sglang.kv_pin_problem(19_918_363_626 + 1_000_000), "past the maximum is refused"
    # The same words the launch raises: ONE definition of the bound.
    with pytest.raises(ValueError) as raised:
        sglang.sglang_mem_fraction(24_000_000_000)
    assert str(raised.value) == problem
    # A vLLM profile pins what the gate affords: no ceiling of its own, nothing to refuse here.
    vllm = shipped().get(FLASH).profile(CLUSTER_MODE)
    assert vllm.kv_cache_memory_max() is None and vllm.kv_pin_problem(96_000_000_000) == ""
    # …and the page reads the ceiling off the profile's view, beside the pool it pins.
    assert sglang.view()["kv_cache_memory_max"] == 19_918_363_626 and vllm.view()["kv_cache_memory_max"] is None


# ── the dense 27B on ONE card: the block rule a standalone node pages at ──────────────────────────────

def test_the_dense_27b_states_its_single_card_block_rule_measured_on_the_5090_node():
    """`memory.kv_blocks` is per rank of the TP=2 pair (816-token blocks). One card pages differently: the live 5090
    node, 2026-10-11, vLLM v0.31.1rc1, fp8_e4m3 KV — 1,568-token blocks, 6 fixed a request (116 blocks, 1.2889x at
    131,072 = 116 / 90). The page's bytes from the engine's own exact figure: `--kv-cache-memory=8633930752` fully uses
    the card and is 168 blocks (≤ 254,016 tokens), so a block is at most 8,633,930,752 / 168 B."""
    kb = shipped().get(DENSE).memory.single_card_kv_blocks
    assert (kb.block_tokens, kb.block_bytes, kb.fixed_blocks) == (1568, 51_392_444, 6)
    assert kb.per_request(131072) == 90 and round(116 / kb.per_request(131072), 4) == 1.2889
    assert kb.blocks_in(8_633_930_752) == 168 and (168 - kb.fixed_blocks) * kb.block_tokens == 254_016
    assert "--kv-cache-memory=8633930752" in kb.source and "num_gpu_blocks=\"116\"" in kb.source
    assert shipped().get(DFLASH).memory.single_card_kv_blocks is None, "measured for no other row: absent, never scaled"


def test_the_dense_27b_states_what_one_card_loses_beyond_weights_and_kv():
    """What vLLM's own boot line implies: at gpu_memory_utilization 1.0 the 5090 leaves 8,633,930,752 B (8.04 GiB) for
    KV, so everything else the engine holds — non-torch memory, peak activation, CUDA graphs, the card's own reserve —
    is the card's 32 GiB less the weights' 20.4 GiB less that: 3.559 GiB. With it, util 1.0 reproduces the engine's
    own ceiling exactly (fleet tests)."""
    over = shipped().get(DENSE).memory.single_card_overhead
    assert over.gib == 3.559 and round(32 - 20.4 - 8_633_930_752 / (1 << 30), 3) == over.gib
    assert "Free memory on device (30.85/31.36 GiB)" in over.source
    assert shipped().get(DFLASH).memory.single_card_overhead is None


def test_a_malformed_single_card_block_rule_is_refused():
    doc = raw()
    spec_raw(doc, DENSE)["memory"]["single_card_kv_blocks"] = {"block_tokens": 0, "block_bytes": 1, "fixed_blocks": 0,
                                                              "kv_cache_dtype": "fp8_e4m3"}
    with pytest.raises(ValueError, match="single_card_kv_blocks block_tokens"):
        ModelCatalog.from_dict(doc)


def test_a_malformed_single_card_overhead_is_refused():
    doc = raw()
    spec_raw(doc, DENSE)["memory"]["single_card_overhead"] = {"gib": -1, "source": "x"}
    with pytest.raises(ValueError, match="single_card_overhead"):
        ModelCatalog.from_dict(doc)


def test_the_dense_27b_overhead_states_the_prefill_chunk_it_was_measured_at():
    """The engine's peak activation — part of the overhead — grows with the prefill chunk, so the figure is a figure AT
    one chunk. The 5090 node passed no --max-num-batched-tokens, so it ran vLLM's own default for its image (v0.31.1rc1
    .dev273 = 2eb491e): vllm/engine/arg_utils.py get_batch_defaults gives the API server 2,048 on a card under 70 GiB
    (the boot line's 31.36 GiB)."""
    over = shipped().get(DENSE).memory.single_card_overhead
    assert over.chunk_tokens == 2048
    for cited in ("2eb491e", "get_batch_defaults", "70 GiB", "31.36 GiB", "no --max-num-batched-tokens"):
        assert cited in over.chunk_source, cited


@pytest.mark.parametrize("chunk, says", [
    ({"max_num_batched_tokens": 0, "source": "s"}, "chunk max_num_batched_tokens"),
    ({"max_num_batched_tokens": True, "source": "s"}, "chunk max_num_batched_tokens"),
    ({"max_num_batched_tokens": 2048}, "chunk"),                      # no source: a number nobody can trace
    ({"max_num_batched_tokens": 2048, "source": "s", "x": 1}, "chunk"),
])
def test_a_malformed_overhead_chunk_is_refused(chunk, says):
    doc = raw()
    spec_raw(doc, DENSE)["memory"]["single_card_overhead"]["chunk"] = chunk
    with pytest.raises(ValueError, match=says):
        ModelCatalog.from_dict(doc)


def test_a_row_only_a_fleet_launches_has_no_serving_mode_and_states_its_facts():
    """`launch: {}` (worklog/parallel-clusters.md, "The gui node"): MAI-UI-2B is launched by a one-host node's own
    launch block in the fleet file, never by a ServingMode of the LB — so no picker offers it, and no profile is
    invented for it. Its facts are {value, source}: unquantised (no MFU patch), takes images."""
    cat = shipped()
    mai = cat.get("mai-ui-2b")
    assert mai.launch == {} and all(mai not in cat.for_mode(m) for m in ("cluster", "classic"))
    assert (mai.served_name, mai.quant_method, mai.vision, mai.size_gib) == ("Tongyi-MAI/MAI-UI-2B", None, True, 3.963)
    assert mai.memory.single_card_overhead.gib == 3.087 and "DERIVED" in mai.memory.single_card_overhead.source
    assert cat.get("qwen3.8-27b-nvidia-nvfp4").quant_method == "modelopt_mixed"
