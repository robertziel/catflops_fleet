"""TENSORFOLD ON THE PAIR — a third cluster engine (operator 2026-10-29: "Can we implement same engine as additional
option in our local setup?", then "waiting for TensorFold in LB").

The spike (worklog/tensorfold-spike.md, worklog/bench/tensorfold/) ran Qwen3.8 Flash-Next on the box-a/box-b pair two
ways: the mia-ai recipe's native Zig engine (`tensorfold-native serve`, NVIDIA's NVFP4 checkpoint, the ranks' small
all-gathers as one-shot RoCE writes) and upstream's Python 0.6.6 engine (`tensorfold serve`, the MLX 4-bit MTP
checkpoint, NCCL). Both become catalog rows the panel picks like any other: `launch.cluster.engine: "tensorfold"`, and
tp-rank builds TensorFold's command on the same container name, mounts and CLUSTER_PORT as vLLM's and SGLang's, so :8010
answers whichever engine is behind it.

What TensorFold does NOT take is refused where it is given, in words (the no-fallback rule; precedent ENGINE_REFUSES and
the X-Priority refusal of 412af46/c311fe0): a KV pool pin (it budgets its sequence memory itself), a prefill budget (its
prompt chunk is its own), a fixed drafter depth (its MTP chain stops by confidence), the PLE placement, the disk tier,
the long-prefill threshold — and an integer X-Priority, which its server reads only as the word "background"."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_tp_rank import LIVE_BOX_ENV, last_run, box_a, ssh  # noqa: F401 — the fake GB10 box fixtures
from vllm_lb.catalog import LIBRARY_DIR, document, CLUSTER_ENGINES, CLUSTER_MODE, ENGINE_REFUSES, TENSORFOLD, ModelCatalog

FIXTURES = Path(__file__).parent / "fixtures"
#: The fleet's build of the mia-ai image (deploy/images/tensorfold): the same TensorFold tree and recipe patch set
#: (f3e1300f4f62, recipe be3ed85) plus the fleet's patches. The tag names the series, so a box holding the plain
#: mia-ai image can never pass for it.
FLEET_IMAGE = "tensorfold-qwen38fn:zig-db28187-f3e1300f4f62-fleet.2"
NATIVE = "qwen3.8-flash-next-tensorfold-nvfp4"
PY066 = "qwen3.8-flash-next-tensorfold-mlx-mtp"
#: The box settings the RoCE all-gather needs (lb/.env, sent to every rank by cluster_run): the bond's slave HCAs and
#: the GID index their RoCE v2 IPv4 GID sits at. Documentation values: the test asserts they travel, not what they are.
FABRIC = {"CLUSTER_IB_HCA": "mlx_a,mlx_b", "CLUSTER_IB_GID_INDEX": "3"}


def catalog() -> ModelCatalog:
    return ModelCatalog.from_dict(document())


def spec(model_id: str):
    return catalog().get(model_id)


# ── the catalog ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_tensorfold_is_a_cluster_engine_and_both_spike_legs_are_picker_rows():
    assert TENSORFOLD in CLUSTER_ENGINES
    for model_id in (NATIVE, PY066):
        profile = spec(model_id).profile(CLUSTER_MODE)
        assert profile is not None and profile.engine == TENSORFOLD, model_id
        assert profile.view()["engine"] == TENSORFOLD


def test_the_images_and_checkpoints_are_the_spikes_and_live_in_the_catalog_not_in_code():
    native, py = spec(NATIVE), spec(PY066)
    assert native.profile(CLUSTER_MODE).image == FLEET_IMAGE
    assert py.profile(CLUSTER_MODE).image == "tensorfold-py066:cb2ebf0"
    # The recipe's pin of NVIDIA's checkpoint, and the MLX revision the spike downloaded — HF cache snapshot folders.
    assert native.path.endswith("models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
                                "fc694b54fb0174e0913e6adf86691ef85a4ead47")
    assert py.path.endswith("models--TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/"
                            "2b170fa6309d5d1ee380b35636075fac7945f286")
    code = "\n".join(p.read_text() for p in (LIBRARY_DIR.parent).glob("*.py"))
    for literal in ("tensorfold-qwen38fn", "tensorfold-py066", "fc694b54fb01", "2b170fa6309d"):
        assert literal not in code, f"{literal} is config, never code"


def test_tensorfold_takes_sixteen_streams_at_most():
    """--parallel is 1..16 in both engines (the recipe's start.sh refuses more; 0.6.6 serves "up to 16 streams")."""
    for model_id in (NATIVE, PY066):
        assert spec(model_id).limits.max_num_seqs == (1, 16)


@pytest.mark.parametrize("knob", ["kv_cache_memory", "max_num_batched_tokens", "num_speculative_tokens",
                                  "ple_replicate", "kv_offload_gib", "long_prefill_token_threshold"])
def test_every_knob_tensorfold_does_not_take_is_refused_with_why(knob):
    why = ENGINE_REFUSES[TENSORFOLD][knob]
    assert len(why) > 30, "a reason in words"
    assert spec(NATIVE).profile(CLUSTER_MODE).view()["refuses"][knob] == why


def test_the_kv_dtype_is_the_profiles_and_each_image_records_what_it_takes():
    from vllm_lb.kv_dtype import library
    assert spec(NATIVE).profile(CLUSTER_MODE).kv_cache_dtype == "fp8_e4m3"
    assert spec(PY066).profile(CLUSTER_MODE).kv_cache_dtype == "bfloat16"
    lib = library()
    native_image = spec(NATIVE).profile(CLUSTER_MODE).image
    py_image = spec(PY066).profile(CLUSTER_MODE).image
    assert set(lib.images[native_image].accepted) == {"bfloat16", "fp8_e4m3"}
    assert set(lib.images[py_image].accepted) == {"bfloat16"}
    assert lib.problem(py_image, "fp8_e4m3"), "0.6.6 serves bf16, int8 or int4 KV — no fp8"
    assert lib.problem(native_image, "nvfp4")


# ── the switch ──────────────────────────────────────────────────────────────────────────────────────────────────────

def store(tmp_path):
    from vllm_lb.cluster import DesiredStore
    return DesiredStore(str(tmp_path / "cluster.json"), catalog=catalog())


@pytest.mark.parametrize("knob, value", [("kv_cache_memory", 20_000_000_000), ("max_num_batched_tokens", 4096),
                                         ("num_speculative_tokens", 4), ("kv_offload_gib", 10),
                                         ("long_prefill_token_threshold", 4096)])
def test_a_start_naming_a_knob_tensorfold_does_not_take_is_refused_in_its_words(tmp_path, knob, value):
    s = store(tmp_path)
    with pytest.raises(ValueError, match=knob):
        s.set(enabled=True, model_choice=NATIVE, max_num_seqs=16, max_model_len=131072, **{knob: value})


def test_a_start_with_none_of_them_is_taken(tmp_path):
    s = store(tmp_path)
    d = s.set(enabled=True, model_choice=NATIVE, max_num_seqs=16, max_model_len=131072)
    assert d.model_choice == NATIVE and d.max_num_seqs == 16


def test_seventeen_streams_are_past_the_models_limits(tmp_path):
    with pytest.raises(ValueError):
        store(tmp_path).set(enabled=True, model_choice=NATIVE, max_num_seqs=17, max_model_len=131072)


# ── the launch ──────────────────────────────────────────────────────────────────────────────────────────────────────

def test_launch_env_names_the_engine_and_its_own_settings_and_no_vllm_or_sglang_one():
    from vllm_lb.cluster_exec import launch_env
    env = launch_env(spec(NATIVE))
    assert env["CLUSTER_ENGINE"] == TENSORFOLD
    assert env["CLUSTER_IMAGE"] == FLEET_IMAGE
    args = json.loads(env["CLUSTER_ENGINE_ARGS"])
    assert args["cli"] == "tensorfold-native" and args["kv_cache_dtype"] == "fp8_e4m3"
    assert args["roce"] == 1, "the native leg's RoCE all-gather is the reason it is the primary"
    for absent in ("CLUSTER_KV_CACHE_MEMORY", "CLUSTER_GPU_MEMORY_UTILIZATION", "CLUSTER_SPECULATIVE_CONFIG_x"):
        assert absent not in env
    assert env["CLUSTER_SPECULATIVE_CONFIG"] == "" and env["CLUSTER_MTP_PREFIX"] == ""


def test_a_launch_given_a_knob_tensorfold_does_not_take_is_refused_never_dropped():
    from vllm_lb.cluster_exec import launch_problem
    s = spec(NATIVE)
    assert launch_problem(s, rank=0, gen=1, max_num_seqs=16, max_model_len=131072) == ""
    for knob, value in (("max_num_batched_tokens", 4096), ("num_speculative_tokens", 3),
                        ("kv_cache_memory", 20_000_000_000)):
        why = launch_problem(s, rank=0, gen=1, max_num_seqs=16, max_model_len=131072, **{knob: value})
        assert knob in why and ENGINE_REFUSES[TENSORFOLD][knob] in why, why


def test_the_launch_record_says_the_engine_runs_its_own_budget_and_drafter_not_a_number_it_never_got():
    from vllm_lb.cluster_exec import resolved_launch
    r = resolved_launch(spec(NATIVE), LIVE_BOX_ENV, max_num_seqs=16, max_model_len=131072)
    assert r["max_num_batched_tokens"] is None and r["num_speculative_tokens"] is None
    assert r["kv_cache_memory"] is None and r["kv_cache_dtype"] == "fp8_e4m3"


def place(box_a, s) -> Path:  # noqa: F811
    """The HF cache's snapshot folder, at the catalog's path under the box's models root (what MODELS_SEARCH_DIRS names)."""
    parent = Path(s.path).parent
    return box_a.place_checkpoint(box_a.home / "models" / parent, name=s.name)


def _up(box_a, ssh, model_id: str, *, rank: int, env: dict | None = None, **kw):  # noqa: F811
    s = spec(model_id)
    models = place(box_a, s)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, **FABRIC, "CLUSTER_MODELS_DIRS": str(models), **(env or {})})
    ok, msg = ex.up(host, rank=rank, gen=301, max_num_seqs=16, max_model_len=131072, spec=s, **kw)
    return ok, msg, s


def envs(run: list[str]) -> dict[str, str]:
    return dict(run[i + 1].split("=", 1) for i, w in enumerate(run[:-1]) if w == "-e" and "=" in run[i + 1])


@pytest.mark.parametrize("rank", [0, 1])
def test_the_native_rank_runs_tensorfold_native_serve_on_the_heads_port(box_a, ssh, rank):  # noqa: F811
    ok, msg, s = _up(box_a, ssh, NATIVE, rank=rank)
    assert ok is True, msg
    run = last_run(box_a)
    i = run.index("tensorfold-native")
    assert run[i + 1] == "serve" and run[i + 2] == f"/models/{s.name}"
    flags = dict(zip(run, run[1:]))
    assert flags["--name"] == s.served_name and flags["--host"] == "0.0.0.0" and flags["--port"] == "8100"
    assert flags["--parallel"] == "16" and flags["--context"] == "131072"
    assert flags["--kv-dtype"] == "fp8", "TensorFold's word for the e4m3 cache"
    assert flags["--tp"] == "2" and flags["--rank"] == str(rank)
    assert flags["--master"] == "192.0.2.201" and flags["--master-port"] == "25000"
    assert flags["--max-tokens"] == "32768" and "--thinking" in run and "--no-update-check" in run
    assert "--vision" in run, "the spike's native leg served image and video input"
    assert "--no-drafts" not in run, "the checkpoint's own MTP drafts, as measured"
    e = envs(run)
    assert e["TF_FLASHNEXT_ROCE"] == "1" and e["TF_FLASHNEXT_L2PF"] == "1" and e["TF_FLASHNEXT_YARN"] == "0"
    assert e["NCCL_IB_HCA"] == FABRIC["CLUSTER_IB_HCA"] and e["NCCL_IB_GID_INDEX"] == "3"
    assert e["NCCL_MIN_NCHANNELS"] == e["NCCL_MAX_NCHANNELS"] == "4"
    assert e["TENSORFOLD_MEMORY_RESERVE_GIB"] == "10"
    for other in ("--headless", "--model", "--max-num-seqs", "--kv-cache-memory", "--tensor-parallel-size",
                  "sglang.launch_server", "--enable-mfu-metrics", "--scheduling-policy"):
        assert other not in run, f"{other} is another engine's"
    assert not any(w.startswith("PYTHONPATH=/mfu-patch") for w in run), "the MFU patch is vLLM's"


def test_the_066_rank_runs_the_python_engine_on_nccl(box_a, ssh):  # noqa: F811
    ok, msg, s = _up(box_a, ssh, PY066, rank=0)
    assert ok is True, msg
    run = last_run(box_a)
    i = run.index("tensorfold")
    assert run[i + 1] == "serve" and run[i + 2] == f"/models/{s.name}"
    flags = dict(zip(run, run[1:]))
    assert flags["--kv-dtype"] == "bf16" and flags["--parallel"] == "16" and flags["--tp"] == "2"
    e = envs(run)
    assert "TF_FLASHNEXT_ROCE" not in e, "0.6.6's ranks talk over NCCL; the RoCE gather is the native engine's"
    assert "--vision" not in run


def test_a_roce_launch_without_the_fabric_settings_is_refused_before_docker(box_a, ssh):  # noqa: F811
    """TensorFold leaves BOTH ranks on NCCL when a rank cannot open RoCE (no NCCL_IB_HCA) — the engine's own silent
    fallback. So the launch that asks for RoCE and has no HCA list to give is refused, naming the setting."""
    s = spec(NATIVE)
    models = place(box_a, s)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=302, max_num_seqs=16, max_model_len=131072, spec=s)
    assert ok is False and "CLUSTER_IB_HCA" in msg, msg
    assert not any(c[:1] == ["run"] for c in box_a.calls())


def test_a_run_kv_dtype_reaches_the_engine_in_its_own_word(box_a, ssh):  # noqa: F811
    ok, msg, _ = _up(box_a, ssh, NATIVE, rank=0, kv_cache_dtype="bfloat16")
    assert ok is True, msg
    assert dict(zip(last_run(box_a), last_run(box_a)[1:]))["--kv-dtype"] == "bf16"


def test_tp_rank_refuses_a_kv_dtype_tensorfold_has_no_word_for(box_a):  # noqa: F811
    s = spec(NATIVE)
    models = place(box_a, s)
    from vllm_lb.cluster_exec import launch_env
    cfg = {**LIVE_BOX_ENV, **FABRIC, **launch_env(s), "CLUSTER_MODELS_DIRS": str(models),
           "CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_KV_CACHE_DTYPE": "fp8_e5m2"}
    proc = box_a.run("up", "0", "303", "16", "131072", NATIVE, config=cfg)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "fp8_e5m2" in proc.stdout


# ── what the engine says: its metrics, its start ────────────────────────────────────────────────────────────────────

def test_the_probe_reads_tensorfolds_own_vocabulary():
    from vllm_lb.probe import TENSORFOLD_UNAVAILABLE, parse_metrics
    out = parse_metrics((FIXTURES / "tensorfold-metrics-native-db28187.txt").read_text())
    assert out["engine"] == TENSORFOLD
    assert out["running"] == 3 and out["waiting"] == 2
    # Finished requests' tokens PLUS the live streams': the only count that moves while the work happens.
    assert out["gen_tokens_total"] == 12800 + 300
    assert out["prompt_tokens_total"] == 5120
    assert out["spec_draft_tokens_total"] == 13900 and out["spec_accepted_total"] == 7800
    assert out["requests_finished_total"] == 50 and out["request_seconds_sum"] == 251.5
    assert out["ttft_sum"] == 27.25 and out["ttft_count"] == 50
    assert out["prefill_time_sum"] == 4.5 and out["prefill_requests_total"] == 50
    # Its kv_cache_usage_perc is ONE STREAM's occupancy of its window, not the pool's: never read as the pool. The
    # streams' fills are kept, summed, for /health's seat count to turn into the pool in use (`with_health`).
    assert out["kv_cache_usage"] is None and "kv_cache_usage" not in TENSORFOLD_UNAVAILABLE
    assert out["kv_stream_fill_sum"] == pytest.approx(0.002 + 0.0031 + 0.0007)
    for key in ("est_flops_total", "prefix_hits_total", "process_start_time"):
        assert out[key] is None and key in TENSORFOLD_UNAVAILABLE


# ── patch 0014: the engine's own work, so COMPUTE and PREFILL have a source ─────────────────────────────────────────
#
# The #pair COMPUTE, PREFILL and prefill/request tiles were empty while TensorFold served (operator 2026-10-29: "Explain
# why other stats are still empty"): its /metrics had no pass, computed-prompt or busy-time family. Image -fleet.2's
# patch 0014 (deploy/images/tensorfold) counts them; this fixture is that patch's metrics.zig `render` run on a stub
# status (its own unit-test seam), so the exposition is the new image's byte for byte — not yet a live scrape.

FLEET2 = FIXTURES / "tensorfold-metrics-fleet2-rendered.txt"


def _later(text: str, **counters) -> str:
    """The same scrape with some counters moved on, as the next probe would read them."""
    import re
    for name, value in counters.items():
        text, n = re.subn(rf"^tensorfold:{name} .*$", f"tensorfold:{name} {value}", text, flags=re.M)
        assert n == 1, name
    return text


def test_patch_0014s_work_reads_as_the_canonical_pass_and_prompt_keys():
    from vllm_lb.probe import parse_metrics
    out = parse_metrics(FLEET2.read_text())
    assert out["engine"] == TENSORFOLD
    # Every forward, and the decode ones: the same two counters SGLang's pass families give the LB.
    assert out["decode_steps_total"] == 4210
    assert out["steps_total"] == 4210 + 21
    assert out["prompt_computed_total"] == 38192 and out["prompt_cached_total"] == 26000
    # The last decode forward: its streams' summed context and the rows a stream carried.
    assert out["decode_context_tokens"] == 39500 and out["spec_verify_width"] == 4.5
    assert out["busy_seconds_total"] == 312.75
    # prefill / request: the computed prompt of exactly the requests request_prefill_seconds timed.
    assert out["prefill_computed_tokens_sum"] == 12192
    assert out["prefill_time_sum"] == 12.5 and out["prefill_requests_total"] == 2


def test_the_live_fleet2_scrape_carries_every_work_key():
    """Generation 284's own /metrics (2026-10-30 01:24, image sha256:b9d1b0869fc9, after a few requests): every key
    patch 0014 feeds is a number, and the steps are its prompt chunks plus its decode forwards."""
    from vllm_lb.probe import parse_metrics
    text = (FIXTURES / "tensorfold-metrics-fleet2-live-gen284.txt").read_text()
    out = parse_metrics(text)
    for key in ("steps_total", "decode_steps_total", "prompt_computed_total", "prompt_cached_total",
                "decode_context_tokens", "spec_verify_width", "busy_seconds_total", "prefill_computed_tokens_sum"):
        assert isinstance(out[key], (int, float)), key
    chunks = next(int(line.split()[1]) for line in text.splitlines() if line.startswith("tensorfold:prompt_passes_total "))
    assert out["steps_total"] == chunks + out["decode_steps_total"]


def test_an_image_before_0014_reads_its_pass_keys_as_unknown_never_zero():
    from vllm_lb.probe import parse_metrics
    out = parse_metrics((FIXTURES / "tensorfold-metrics-native-db28187.txt").read_text())
    for key in ("steps_total", "decode_steps_total", "prompt_computed_total", "prompt_cached_total",
                "decode_context_tokens", "spec_verify_width", "busy_seconds_total", "prefill_computed_tokens_sum"):
        assert out[key] is None, key


def test_steps_need_both_kinds_of_pass():
    """A server that counts decode forwards but not its prompt chunks has no `steps_total`: half a sum is no count."""
    from vllm_lb.probe import parse_metrics
    text = "\n".join(line for line in FLEET2.read_text().splitlines()
                      if not line.startswith(("tensorfold:prompt_passes_total", "tensorfold:prompt_tokens_computed_total")))
    out = parse_metrics(text)
    assert out["decode_steps_total"] == 4210
    assert out["steps_total"] is None and out["prompt_computed_total"] is None


def _two_reads():
    from vllm_lb.probe import parse_metrics
    a = FLEET2.read_text()
    # 6 s on: 100 decode forwards, 2 prompt chunks of 2,048, 5.5 s inside the backend.
    b = _later(a, decode_passes_total=4310, prompt_passes_total=23, prompt_tokens_computed_total=42288,
               verify_rows_total=21950, busy_seconds_total=318.25)
    return [(0.0, parse_metrics(a)), (6.0, parse_metrics(b))]


def test_compute_is_the_lbs_flop_model_over_tensorfolds_own_passes():
    from tests.test_sglang_compute_model import _status
    from vllm_lb import cores
    shape = cores.SHAPES[NATIVE]
    sat = _status(NATIVE, _two_reads())["saturation"]
    model = sat["compute_model"]
    assert model is not None and model["why"] is None
    assert model["prefill_passes"] == 2 and model["prefill_tokens"] == 4096 and model["decode_passes"] == 100
    prefill = 4096 * shape["flops_per_token"]
    # a verify: 2 streams x 4.5 rows through the layers and the lm_head (Flash-Next's sparse attention is not charged)
    decode = 100 * 2 * 4.5 * (shape["flops_per_token"] + shape["flops_per_logit"])
    assert model["tflops_per_gpu"] == pytest.approx((prefill + decode) / 6.0 / 1e12, abs=0.005)   # /lb/status: 2 decimals
    assert sat["compute"]["rate"] == pytest.approx(model["tflops_per_gpu"], abs=0.05)   # the tile's one decimal
    assert sat["compute"]["why"] is None


def test_a_vllm_or_sglang_scrape_is_still_read_as_itself():
    from vllm_lb.probe import parse_metrics
    assert parse_metrics("vllm:num_requests_running 1\n")["engine"] != TENSORFOLD
    assert parse_metrics("sglang:num_running_reqs 1\n")["engine"] == "sglang"


def test_the_native_start_reads_in_the_pages_words():
    from vllm_lb.cluster_startup import StartupPhase, read_startup, stated_prefill_budget
    log = (FIXTURES / "tensorfold-native-2026-10-29-rank0.log").read_text()
    got = read_startup(log, now=1.9e9)
    assert got is not None
    assert got.steps == (StartupPhase.STARTING_TENSORFOLD, StartupPhase.CONNECTING_RANKS,
                         StartupPhase.loading("model weights"), StartupPhase.KV_CACHE_ALLOCATED,
                         StartupPhase.API_SERVER_UP), got.steps
    assert stated_prefill_budget(log) == 2048, "its own words: prompt chunks of up to 2,048 tokens"


def test_the_066_start_reads_in_the_pages_words():
    from vllm_lb.cluster_startup import StartupPhase, read_startup
    got = read_startup((FIXTURES / "tensorfold-py066-2026-10-29-rank0.log").read_text(), now=1.9e9)
    assert got is not None
    assert got.steps == (StartupPhase.loading("model weights"), StartupPhase.BUILDING_KERNELS,
                         StartupPhase.WARMING_UP_KERNELS, StartupPhase.KV_CACHE_ALLOCATED,
                         StartupPhase.API_SERVER_UP), got.steps


def test_a_partial_native_start_is_loading_while_the_weights_stream():
    from vllm_lb.cluster_startup import StartupPhase, read_startup
    log = (FIXTURES / "tensorfold-native-2026-10-29-rank0.log").read_text()
    upto = log[:log.index("NCCL joined") + len("NCCL joined\n")]
    phase = read_startup(upto, now=1.9e9).phase
    assert StartupPhase.is_loading(phase), "the memory the box gives up is the load's progress (cluster_loop._track)"


# ── the caller's priority ───────────────────────────────────────────────────────────────────────────────────────────

def test_an_integer_priority_to_tensorfold_is_refused_never_forwarded_to_be_ignored():
    from vllm_lb.router import priority_refusal
    why = priority_refusal(TENSORFOLD)
    assert "background" in why and "TensorFold" in why
    assert priority_refusal("vllm") == "" and priority_refusal("sglang") == "" and priority_refusal(None) == ""


def test_the_proxy_answers_400_for_a_priority_bound_for_a_tensorfold_pair():
    from types import SimpleNamespace

    from vllm_lb.proxy import engine_of_backend
    from vllm_lb.state import CLUSTER, Backend
    b = Backend(kind=CLUSTER, url="http://head:8100", cluster="pair")
    snap = SimpleNamespace(cluster_view_of=lambda _b: {"effective": {"model_choice": NATIVE}})
    assert engine_of_backend(snap, b, catalog()) == TENSORFOLD
    nothing = SimpleNamespace(cluster_view_of=lambda _b: {})
    assert engine_of_backend(nothing, b, catalog()) is None, "nothing launched: no engine to name"


def test_the_engine_is_named_for_a_reader():
    from vllm_lb.saturation import ENGINE_NAMES
    assert ENGINE_NAMES[TENSORFOLD] == "TensorFold"


def test_cluster_run_sends_the_fabric_settings_to_every_rank():
    from vllm_lb.cluster_run import TP_RANK_SETTINGS
    assert {"CLUSTER_IB_HCA", "CLUSTER_IB_GID_INDEX"} <= set(TP_RANK_SETTINGS)


def test_the_page_is_told_the_engine_drafts_by_itself():
    assert spec(NATIVE).profile(CLUSTER_MODE).view()["engine_drafts"] is True
    assert spec("qwen3.8-flash-next-sglang").profile(CLUSTER_MODE).view()["engine_drafts"] is None


def test_the_start_gate_charges_the_sequence_memory_tensorfold_will_take_for_the_seats_and_context():
    """No pin: TensorFold sizes its caches to seats x context itself, so the estimate charges exactly that — the
    spike's 30.76 GiB a rank at 16 x 131,072 — beside the weights and the head's vision workspace."""
    from vllm_lb.memory_estimate import estimate_memory
    est = estimate_memory(spec(NATIVE), box="box-a", role="head", max_num_seqs=16, max_model_len=131072, tp=2,
                          available_gib=110.0, floor_gib=12.0, margin_gib=8.0)
    assert est.kv_pinned_gib is None
    assert est.kv_needed_gib == pytest.approx(30.76, abs=0.01)
    assert est.total_gib == pytest.approx(62.3 + 30.76 + 6.6, abs=0.05)


# ── the fleet's image ───────────────────────────────────────────────────────────────────────────────────────────────

def test_the_native_row_runs_the_fleet_build_whose_messages_route_takes_claude_codes_reminders():
    """Claude Code (beta mid-conversation-system-2026-04-07) sends its reminders as a trailing `role: system` message.
    The mia-ai image's native server answered every such turn 400 "the model's chat template does not support
    mid-conversation system messages" (generation 280): Qwen3.8's template renders no later
    system message. SGLang's and vLLM's Anthropic adapters merge them into the leading system message on such a
    template (`_merge_inline_system`); the fleet's patch makes TensorFold's Zig translator do the same, and the build
    gates on TensorFold's own server unit tests, the patch's among them."""
    assert spec(NATIVE).profile(CLUSTER_MODE).image == FLEET_IMAGE, "deploy/images/tensorfold builds it (deploy/tests)"
    # The KV library knows the image the row names: the patch touches no KV code, so its record is the base's.
    library = json.loads((LIBRARY_DIR.parent / "kv_cache_dtypes.json").read_text())
    images = library["images"] if "images" in library else library
    assert set(images[FLEET_IMAGE]["dtypes"]) == set(images["tensorfold-qwen38fn:zig-db28187-ghcr"]["dtypes"])


# ── #pair's figures while TensorFold serves (operator 2026-10-29: "still not all hardware stats are shown") ──────────
#: TensorFold's own /health with no API key (native server/status_routes.zig `health`): `max_batch_size` is its lanes,
#: the `--parallel` it serves.
TF_HEALTH = (b'{"status": "ok", "model": "Qwen/Qwen3.8-Flash-Next-TensorFold", "model_ids": '
             b'["Qwen/Qwen3.8-Flash-Next-TensorFold"], "max_batch_size": 16, "warming": false, "memory": {}, '
             b'"live": {"connections": 0, "waiting": 0, "decode_tokens_per_second": 0.0, '
             b'"prefill_tokens_per_second": 0.0}}')


def test_the_kv_pool_in_use_is_the_streams_windows_over_the_seats_tensorfold_reserves():
    """Its kv_cache_usage_ratio{pool=i} is stream i's tokens over ITS window (metrics.zig `pools`), and its sequence
    memory IS seats x window (rank 0: 'sequence memory: 30.76 GiB' = 16 x 131,072 x 15,749 B): so the pool in use is
    the streams' ratios summed over the seats — read with /health's own seat count, never the LB's setting."""
    from vllm_lb.engine_metrics import parse_metrics, with_health
    scrape = parse_metrics((FIXTURES / "tensorfold-metrics-native-db28187.txt").read_text())
    assert scrape["kv_cache_usage"] is None, "the scrape alone holds no seat count"
    out = with_health(scrape, TF_HEALTH)
    assert out["seats"] == 16
    assert out["kv_cache_usage"] == pytest.approx((0.002 + 0.0031 + 0.0007) / 16)
    # No seat count (a keyed server's health says only "ok"; none read) -> no pool figure, never a guess.
    assert with_health(scrape, b'{"status": "ok"}')["kv_cache_usage"] is None
    assert with_health(scrape, None)["kv_cache_usage"] is None
    # An idle server prints one stream at 0: the pool is empty, a measured 0.
    idle = parse_metrics((FIXTURES / "tensorfold-metrics-2026-10-29-gen280.txt").read_text())
    assert with_health(idle, TF_HEALTH)["kv_cache_usage"] == 0.0


def test_another_engines_scrape_takes_nothing_from_a_health_body():
    from vllm_lb.engine_metrics import parse_metrics, with_health
    vllm = parse_metrics("vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0.25\n")
    assert with_health(vllm, TF_HEALTH) == vllm


def test_the_probe_reads_the_pool_through_the_engines_own_health():
    from vllm_lb.config import Config
    from vllm_lb.probe import probe_backend
    from vllm_lb.state import CLUSTER, Backend
    metrics = (FIXTURES / "tensorfold-metrics-native-db28187.txt").read_bytes()

    def get(url, timeout):
        if url.endswith("/health"):
            return 200, TF_HEALTH
        if url.endswith("/v1/models"):
            return 200, b'{"data": [{"id": "Qwen/Qwen3.8-Flash-Next-TensorFold"}]}'
        return 200, metrics

    out = probe_backend(Backend(kind=CLUSTER, url="http://head:8000"), Config(probe_timeout_s=0.5), now=1.0, get=get)
    assert out["scrape"]["kv_cache_usage"] == pytest.approx(0.0058 / 16)


def test_the_pool_is_sized_per_seat_so_no_seat_is_ever_said_to_queue():
    """The page said 'KV pool 28 GB · 1,777,890 tokens · fits 13 — concurrency 16 runs 13 at a time and queues the
    rest' for generation 281: the gateway's configured 28 GB pin, which TensorFold never takes. Its pool is the seats
    x the context at the model's bytes a token, so every seat fits by construction."""
    from vllm_lb.cluster import Desired, DesiredStore
    cat = ModelCatalog.load()
    for row in ("qwen3.8-flash-next-tensorfold-nvfp4", "qwen3.8-flash-next-tensorfold-mlx-mtp"):
        profile = cat.get(row).profile(CLUSTER_MODE)
        assert profile.pool_per_seat is True and profile.view()["pool_per_seat"] is True
    assert cat.get("qwen3.8-flash-next-hibrid48").profile(CLUSTER_MODE).pool_per_seat is False
    store = DesiredStore.__new__(DesiredStore)
    store.catalog, store._fits = cat, (lambda *a: 13)
    d = Desired(mode="cluster", max_num_seqs=16, max_model_len=131072,
                model_choice="qwen3.8-flash-next-tensorfold-nvfp4")
    assert store.published_clamp(d) == ""
