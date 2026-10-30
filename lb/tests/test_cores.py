"""cores.estimate: one GPU's cores split into multiplying / waiting on memory / other, from step counters."""
import pytest

from vllm_lb import cores

M = "qwen3.8-flash-next-hibrid48"
DENSE = "qwen3.8-27b-nvidia-nvfp4"
#: The GB10's CUDA cores (vllm_lb/hardware/types/gb10.yml): the serving card's, passed by the caller.
GB10_CORES = 6144
KW = dict(running=16, speculative=4, budget=2048, peak_tflops=323.2, peak_gbps=273.0, cuda_cores=GB10_CORES)
#: The dense 27B as generation 103 first served it (2026-10-02): 3 running, 10.7 steps a second.
LIVE = dict(steps_s=10.7, tokens_step=2.5, running=3, speculative=4, budget=8192, peak_tflops=323.2,
            peak_gbps=273.0, cuda_cores=GB10_CORES)


def test_a_decode_only_load_is_memory_not_maths():
    c = cores.estimate(M, steps_s=5.5, tokens_step=80, **KW)          # 180 ms steps of 80 tokens
    assert c["cores"] == 6144 and c["mixed_share"] == 0.0
    assert c["multiplying"] < 60, c                                   # under 1 % of the cores
    assert c["memory"] > 3000, c                                      # 79 % of 34 GB a step, 5.5 a second
    assert c["multiplying"] + c["memory"] + c["other"] == 6144


def test_prefill_chunks_touch_every_expert_and_add_tokens():
    c = cores.estimate(M, steps_s=3.17, tokens_step=489, **KW)        # the live mix of 2026-10-02
    assert c["mixed_share"] == pytest.approx(0.2, abs=0.01)
    assert c["tflops"] == pytest.approx(6.35, abs=0.1) and 100 < c["multiplying"] < 140, c
    assert c["gbps"] == pytest.approx(93, abs=3), c


def test_the_shares_never_exceed_the_cores():
    c = cores.estimate(M, steps_s=50, tokens_step=20000, **KW)
    assert c["multiplying"] + c["memory"] + c["other"] == 6144
    assert min(c[k] for k in ("multiplying", "memory", "other")) >= 0


@pytest.mark.parametrize("kw", [dict(steps_s=None, tokens_step=80), dict(steps_s=0, tokens_step=80),
                                dict(steps_s=3, tokens_step=None)])
def test_no_step_measured_is_no_estimate(kw):
    assert cores.estimate(M, **kw, **KW) is None


def test_a_model_with_no_recorded_shape_gets_no_figure():
    assert cores.estimate("other-model", steps_s=3, tokens_step=80, **KW) is None
    assert cores.estimate(None, steps_s=3, tokens_step=80, **KW) is None


def test_the_two_kinds_of_step_are_split_and_add_up_to_the_wall_clock():
    c = cores.estimate(M, steps_s=3.17, tokens_step=489, **KW)
    d, p = c["cases"]["decode"], c["cases"]["prefill"]
    assert d["time_share"] + p["time_share"] == pytest.approx(1.0, abs=0.01)
    assert p["step_ms"] > 3 * d["step_ms"], (d, p)                    # a chunk makes the step several times longer
    assert p["multiplying"] > d["multiplying"] and d["memory"] > p["memory"], (d, p)
    for k in (d, p):
        assert k["multiplying"] + k["memory"] + k["other"] == 6144


def test_a_window_with_no_prefill_keeps_the_prefill_case_at_zero_time():
    c = cores.estimate(M, steps_s=5.5, tokens_step=80, **KW)
    assert c["cases"]["prefill"]["time_share"] == 0.0
    assert c["cases"]["decode"]["time_share"] == pytest.approx(0.991, abs=0.002) and c["idle_share"] < 0.01
    assert c["cases"]["prefill"]["step_ms"] > c["cases"]["decode"]["step_ms"]
    assert c["cases"]["decode"]["step_ms"] == 180, "the fit's own step: 153 ms + 0.34 ms x 80 tokens"


def test_the_step_time_fit_falls_back_when_the_probes_do_not_vary():
    assert cores.fit_step_time([(5.5, 80)] * 20) == cores.STEP_FIT
    assert cores.fit_step_time([]) == cores.STEP_FIT
    pairs = [(1 / (0.15 + 0.0004 * t), t) for t in range(80, 1280, 100)]
    a, b = cores.fit_step_time(pairs)
    assert a == pytest.approx(0.15, abs=1e-6) and b == pytest.approx(0.0004, abs=1e-8)


# ── what a step holds: the tokens VERIFIED, not the tokens emitted ────────────

def test_the_decode_side_is_charged_for_every_position_verified_not_for_what_was_emitted():
    """`iteration_tokens` counts what a step PRODUCED: 16 sequences emitting ~2.7 tokens each read 44, while
    the target verified 16 x (1 + 4) = 80 positions. Charging 44 under-counted the maths and the experts
    touched — and read the 36 missing tokens as nothing at all."""
    emitted = cores.estimate(M, steps_s=5.66, tokens_step=43.9, **KW)
    verified = cores.estimate(M, steps_s=5.66, tokens_step=80, **KW)
    assert (emitted["tflops"], emitted["gbps"], emitted["mixed_share"]) == (
        verified["tflops"], verified["gbps"], 0.0)
    assert emitted["tflops"] == pytest.approx(5.66 * 80 * (3.995e9 + 0.636e9) / 1e12, abs=0.01)


def test_a_quiet_engine_is_idle_not_slow():
    """Few steps a second is an engine that stood idle between requests. Its steps are still the fit's steps;
    the rest of the wall clock is idle, and the two kinds' shares say so by not adding up to it."""
    quiet = cores.estimate(DENSE, **{**LIVE, "steps_s": 0.47, "tokens_step": 5.0, "running": 1})
    d = quiet["cases"]["decode"]
    assert d["step_ms"] == 91 and d["time_share"] == pytest.approx(0.47 * 0.09125, abs=0.002)
    assert quiet["idle_share"] == pytest.approx(1 - 0.47 * 0.09125, abs=0.002)
    assert quiet["other"] > 6000, "idle cores are not multiplying and not waiting on memory"
    assert d["memory"] > 2500, "…while DURING a step they wait on memory as any decode step does"


# ── the dense 27B: every weight, every step ───────────────────────────────────

def test_the_dense_model_has_a_shape_so_its_tile_is_drawn():
    c = cores.estimate(DENSE, **LIVE)
    assert c is not None and c["cores"] == 6144
    assert c["multiplying"] + c["memory"] + c["other"] == 6144
    # 15 positions verified a step; 8.80e9 B of layers + lm_head and 4 drafter passes of 0.782e9 B.
    assert c["gbps"] == pytest.approx(10.7 * (8.80e9 + 4 * 0.782e9) / 1e9, abs=0.2)
    assert c["tflops"] == pytest.approx(10.7 * (15 * (24.32e9 + 1.271e9) + 4 * 3 * 1.696e9) / 1e12, abs=0.02)
    assert c["memory"] > 20 * c["multiplying"], "decode on a dense model is the memory bus too"


def test_a_dense_step_reads_every_weight_however_few_tokens_it_holds():
    one = cores.estimate(DENSE, **{**LIVE, "running": 1, "tokens_step": 1.0})
    many = cores.estimate(DENSE, **{**LIVE, "running": 16, "tokens_step": 40.0})
    assert one["gbps"] == many["gbps"], "no experts to leave untouched: the bytes do not depend on the batch"
    assert many["tflops"] > 5 * one["tflops"], "…and the maths is all the batch buys"


def test_the_drafter_is_charged_only_when_it_runs():
    on, off = cores.estimate(DENSE, **LIVE), cores.estimate(DENSE, **{**LIVE, "speculative": 0})
    assert off["gbps"] == pytest.approx(10.7 * 8.80, abs=0.1) and on["gbps"] > off["gbps"]
    assert cores.estimate(DENSE, **{**LIVE, "speculative": None})["gbps"] == off["gbps"]


def test_a_dense_prefill_step_is_where_the_cores_multiply():
    c = cores.estimate(DENSE, **{**LIVE, "steps_s": 4.0, "tokens_step": 700.0, "running": 16})
    d, p = c["cases"]["decode"], c["cases"]["prefill"]
    assert p["multiplying"] > 1000 > d["multiplying"], (d, p)        # ~29 % of the cores against ~2 %
    assert d["memory"] > 10 * p["memory"], (d, p)
    assert d["time_share"] + p["time_share"] + c["idle_share"] == pytest.approx(1.0, abs=0.01)


def test_context_adds_attention_only_where_attention_runs_over_all_of_it():
    """The 27B's 16 full-attention layers read the whole context for every token; Flash-Next's indexer bounds
    it at 2,048 keys, so its figure does not move with context."""
    bare, long = cores.estimate(DENSE, **LIVE), cores.estimate(DENSE, **LIVE, context=60_000)
    assert long["tflops"] == pytest.approx(bare["tflops"] + 10.7 * 15 * 60_000 * 196_608 / 1e12, abs=0.02)
    assert long["gbps"] == bare["gbps"]
    assert cores.estimate(M, steps_s=5.5, tokens_step=80, context=60_000, **KW) == \
        cores.estimate(M, steps_s=5.5, tokens_step=80, **KW)
    for junk in (None, 0, -5, True, "60000"):
        assert cores.estimate(DENSE, **LIVE, context=junk)["tflops"] == bare["tflops"]


def test_each_model_brings_its_own_prior_for_a_steps_time():
    """With nothing to fit, a step is the MODEL's own measured step — not Flash-Next's 153 ms for every model."""
    assert cores.estimate(DENSE, **LIVE)["cases"]["decode"]["step_ms"] == 93      # 90 ms + 0.25 ms x 15, at 10.7 a second
    explicit = cores.estimate(DENSE, **LIVE, fit=(0.05, 0.0001))
    assert explicit["cases"]["decode"]["step_ms"] == 52, "a real fit wins over the prior"
    assert cores.estimate(M, steps_s=1.0, tokens_step=80, **KW)["cases"]["decode"]["step_ms"] == 180


def test_the_fit_is_not_bent_by_a_probe_the_engine_idled_through():
    """The first hour of generation 103: thirteen probes of real steps and one in which the engine mostly
    stood idle (0.47 steps a second = a "2,123 ms step"). Least squares answered a = 184 ms."""
    pairs = [(10.79, 2.4), (10.71, 2.1), (10.70, 2.7), (10.85, 2.6), (10.71, 2.6), (10.71, 2.4), (10.71, 2.5),
             (10.55, 2.5), (10.70, 3.0), (10.55, 3.9), (0.47, 5.0), (3.66, 988.1), (1.27, 2155.0), (9.23, 8.6)]
    a, b = cores.fit_step_time(pairs)
    assert a == pytest.approx(0.093, abs=0.002) and b == pytest.approx(0.00018, abs=0.00002)


def test_the_context_a_running_request_holds_leaves_the_fixed_blocks_out():
    """The KV gauge counts the blocks running requests reference, F of which per request are the
    model's fixed state, not context: counting them added 18 × 816 = 14.7k tokens of attention per
    running request on the 27B (audit 2026-10-02, 10)."""
    from vllm_lb.catalog import ModelCatalog
    from vllm_lb.cores import mean_context
    dense = ModelCatalog.load().get(DENSE)
    # 17 running, gauge 36 % of 1,689 blocks = 608 blocks → 35.8 a request, 18 of them fixed → 17.8 × 816.
    assert mean_context(0.36, running=17, pool_blocks=1689, blocks=dense.memory.kv_blocks) == \
        pytest.approx((0.36 * 1689 / 17 - 18) * 816, abs=1.0)
    assert mean_context(0.36, running=17, pool_blocks=1689, blocks=dense.memory.kv_blocks) < 15_000
    assert mean_context(0.0, running=3, pool_blocks=1689, blocks=dense.memory.kv_blocks) == 0.0, "never negative"
    assert mean_context(0.5, running=0, pool_blocks=1689, blocks=dense.memory.kv_blocks) is None
    assert mean_context(0.5, running=2, pool_blocks=None, blocks=dense.memory.kv_blocks) is None


# ── the catalog is the authority: every model it ships must have a shape ──────

DFLASH = "qwen3.8-27b-dflash2"
#: The dflash2 profile as the catalog ships it: 7 drafts a step, 8 seats, the 8,192-token budget.
DFL = dict(steps_s=5.0, tokens_step=200.0, running=8, speculative=7, budget=8192, peak_tflops=323.2,
           peak_gbps=273.0, cuda_cores=GB10_CORES)
#: One probe window that is valid for ANY model, for the tile-renders guard below.
PROBE = dict(steps_s=5.0, tokens_step=200.0, running=4, budget=8192, peak_tflops=323.2, peak_gbps=273.0,
             cuda_cores=GB10_CORES)
#: Every field `coresTile` and `coreCase` read off the estimate (static/gateway.js).
TILE_FIELDS = ("cores", "multiplying", "memory", "other", "tflops", "gbps", "steps_s", "tokens_step",
               "mixed_share", "idle_share", "cases")
CASE_FIELDS = ("multiplying", "memory", "other", "step_ms", "time_share")

HOW_TO_ADD_A_ROW = """
`SHAPES` in vllm_lb/cores.py is keyed by MODEL ID, so a cluster model the catalog ships with no row
here estimates None, `coresTile` returns '' (static/gateway.js) and the CUDA CORES tile leaves the page
entirely — taking the steps_s and tokens_step rings, which reach the screen through no other path, with
it. That is finding 3 of the 2026-10-02 gateway audit: `qwen3.8-27b-dflash2` was added to the catalog
and the tile the operator asked for silently disappeared, with every other input to it present.

To add the row, per GPU (one rank of TP=2), and say in a comment where each number came from:
  cores, experts, top_k, expert_bytes, flops_per_token, flops_per_logit, attn_flops_per_ctx_token
      THE ARCHITECTURE. A FLOP does not depend on a dtype, so a checkpoint of a family already in the
      table carries these over unchanged (`_DENSE_27B` below is exactly that, shared by two rows);
      otherwise count the shard headers' matmul elements the way each row's comment records.
  fixed_bytes
      The weights EVERY step reads: the layers' plus the lm_head's tensor bytes, weight AND scale
      tensors, halved for TP=2. PER CHECKPOINT, not per architecture — the same 27B layers under a BF16
      head read 0.91e9 B a step more than under an NVFP4 one.
  draft_bytes, draft_flops, draft_block
      The drafter: bytes a rank reads per PASS, FLOPs per drafted POSITION per sequence, and the
      positions one pass drafts (1 for a sequential MTP head, its block for a block drafter).
  step_fit
      PER ENGINE, and the only field in the row that is a MEASUREMENT rather than arithmetic off a
      checkpoint: `a + b x tokens` from that engine's own (steps_s, tokens_step) probes on this
      hardware. vLLM's step is not SGLang's step for the same weights.
Weakening or deleting this test to land a model is the bug, not the fix: the tile is what the operator
asked for ("real GPU usage, GPU cores fill rate, teraflops/s rate in real time", saturation.py).
"""


def _cluster_ids() -> list[str]:
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    cat = ModelCatalog.load()
    assert not cat.error, cat.error
    return [s.id for s in cat.for_mode(CLUSTER_MODE)]


def test_every_cluster_model_the_catalog_ships_has_a_recorded_shape():
    """THE ALARM for the class of bug, not for one model: a hard-coded table keyed by model id loses a
    tile SILENTLY when a model is added, so the catalog — not this table — is the authority, and a
    cluster model without a row fails here instead of on the operator's screen."""
    missing = sorted(set(_cluster_ids()) - set(cores.SHAPES))
    assert not missing, f"cluster models with no CUDA CORES shape: {missing}\n{HOW_TO_ADD_A_ROW}"


def test_the_shape_table_keeps_no_row_for_a_model_the_catalog_no_longer_ships():
    """The same drift the other way: a row left behind by a renamed or dropped model id reads as
    coverage while the model that replaced it has none."""
    stale = sorted(set(cores.SHAPES) - set(_cluster_ids()))
    assert not stale, (f"shapes for models the catalog does not ship as cluster models: {stale} — "
                       "a stale row is not coverage; rename it with the model or drop it")


def test_a_classic_only_model_needs_no_shape():
    """The estimate is only ever asked for the CLUSTER's loaded model: `proxy._cores(head, cluster)`
    passes `cluster["effective"]["model_choice"]`, which `cluster.py` validates against CLUSTER_MODE.
    A classic-only model (`qwen3.8-27b-nvfp4`, served by broker_parrot's own engine) has no tile on
    this page to lose, so the guard above asks for no row for it — and must not."""
    from vllm_lb.catalog import CLASSIC, CLUSTER_MODE, ModelCatalog
    cat = ModelCatalog.load()
    classic_only = [s.id for s in cat.for_mode(CLASSIC) if s.profile(CLUSTER_MODE) is None]
    assert classic_only, "the catalog used to ship a classic-only model; if it no longer does, say so here"
    assert not set(classic_only) & set(cores.SHAPES)
    assert not set(classic_only) & set(_cluster_ids())


def test_every_cluster_model_renders_a_complete_tile():
    """A row that EXISTS but is missing a field raises inside `estimate`, `_cores` swallows it ("an
    estimate is never allowed to break /lb/status") and the tile vanishes the same way a missing row
    loses it. So every cluster model is estimated here, at its own catalog drafter depth, and checked
    for every field the page reads off it."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    cat = ModelCatalog.load()
    for mid in _cluster_ids():
        drafts = cat.get(mid).profile(CLUSTER_MODE).num_speculative_tokens
        c = cores.estimate(mid, **PROBE, speculative=drafts)
        assert c is not None, f"{mid}: no estimate, so no tile\n{HOW_TO_ADD_A_ROW}"
        assert all(c.get(f) is not None for f in TILE_FIELDS), (mid, c)
        assert c["multiplying"] + c["memory"] + c["other"] == c["cores"], (mid, c)
        for kind in ("decode", "prefill"):
            case = c["cases"][kind]
            assert all(case.get(f) is not None for f in CASE_FIELDS), (mid, kind, case)
            assert case["multiplying"] + case["memory"] + case["other"] == c["cores"], (mid, kind, case)
            assert case["step_ms"] > 0, (mid, kind, case)


# ── the dense 27B drafted by its own DFlash2 checkpoint ───────────────────────

def test_the_two_27b_rows_share_one_architecture():
    """A FLOP does not depend on a dtype: the same 64 layers under either lm_head do the same maths, so the
    architecture is recorded ONCE and a row states only what its own checkpoint and engine make different.
    Two hand-typed copies of the same figures is how they drift."""
    arch = ("experts", "top_k", "expert_bytes", "flops_per_token", "flops_per_logit", "attn_flops_per_ctx_token")
    for k in arch:
        assert cores.SHAPES[DENSE][k] == cores.SHAPES[DFLASH][k] == cores._DENSE_27B[k], k


def test_the_bf16_head_is_read_every_step_where_the_nvfp4_one_was_not():
    """The two rows differ in one weight: the head's dtype. 248,320 x 5,120 BF16 is 1.271e9 B a rank
    against the NVFP4 head's 0.36e9 — 0.91e9 B more on the bus EVERY step, which is the +1.70 GiB the
    catalog records between the two checkpoint paths (size_gib 22.1 against 20.4)."""
    assert cores.SHAPES[DFLASH]["fixed_bytes"] - cores.SHAPES[DENSE]["fixed_bytes"] == \
        pytest.approx(1.271e9 - 0.36e9, abs=0.02e9)


def test_the_block_drafter_is_charged_one_pass_a_step_not_one_a_draft_token():
    """DFlash 2 proposes the whole block in a single pass, so its 3.85e9 B of weights cross the bus ONCE
    a step. Charged like an MTP head — once per draft token — the tile would have read 26.9 GB a step off
    a 273 GB/s bus and called a quiet engine memory-bound."""
    c = cores.estimate(DFLASH, **DFL)
    assert c["gbps"] == pytest.approx(5.0 * (9.72e9 + 5.12e9) / 1e9, abs=0.2)
    assert c["gbps"] < 0.5 * 5.0 * (9.72e9 + 7 * 5.12e9) / 1e9, "seven passes would be most of the bus"
    # 8 seats x (1 + 7) positions verified + 136 prefill tokens, and the drafter's own block of 8 a seat.
    assert c["tflops"] == pytest.approx(
        5.0 * (64 * (24.32e9 + 1.271e9) + 136 * 24.32e9 + 8 * 8 * 5.12e9) / 1e12, abs=0.05)


def test_a_block_drafters_bytes_do_not_move_with_the_operators_draft_count():
    """The panel lets the operator choose K. A block drafter re-reads its weights once for any K its block
    covers; a sequential MTP head re-reads them K times. That is the whole of `draft_block`, and the reason
    the drafter's bytes are not `K x draft_bytes` for every model."""
    few, many = (cores.estimate(DFLASH, **{**DFL, "speculative": k}) for k in (3, 7))
    assert few["gbps"] == many["gbps"]
    mtp_few, mtp_many = (cores.estimate(DENSE, **{**DFL, "speculative": k}) for k in (3, 7))
    assert mtp_many["gbps"] > mtp_few["gbps"]
    # …and a K past the block is a second pass, not a fraction of one.
    assert cores.estimate(DFLASH, **{**DFL, "speculative": 9})["gbps"] == \
        pytest.approx(5.0 * (9.72e9 + 2 * 5.12e9) / 1e9, abs=0.2)


def test_the_mtp_rows_arithmetic_is_untouched_by_the_block_field():
    """`draft_block` 1 is exactly the old `K x draft_bytes` / `K x draft_flops`: the two rows that shipped
    before it must estimate the same figures they did."""
    c = cores.estimate(DENSE, **LIVE)
    assert c["gbps"] == pytest.approx(10.7 * (8.80e9 + 4 * 0.782e9) / 1e9, abs=0.2)
    assert c["tflops"] == pytest.approx(10.7 * (15 * (24.32e9 + 1.271e9) + 4 * 3 * 1.696e9) / 1e12, abs=0.02)


def test_the_dflash_prior_is_this_profiles_own_measured_step():
    """A model's prior is a MEASUREMENT, so with nothing to fit the tile shows generation 109's own step —
    106.0 ms at one seat over 5,023 of them — not Flash-Next's 153 ms and not the MTP profile's 93."""
    quiet = cores.estimate(DFLASH, **{**DFL, "steps_s": 2.0, "tokens_step": 8.0, "running": 1})
    assert quiet["cases"]["decode"]["step_ms"] == 108, "106.0 ms + 0.298 ms x 8 verified positions"
    assert cores.estimate(DENSE, **{**DFL, "steps_s": 2.0, "tokens_step": 8.0,
                                    "running": 1})["cases"]["decode"]["step_ms"] == 92



# ── the GPU is the serving card's, never the model's ─────────────────────────────────────────────

def test_a_models_shape_carries_no_hardware():
    """A shape is the CHECKPOINT's arithmetic; how many cores do it is the serving card's (its hardware type). The
    6,144 a GB10 has was written into three rows here — on a 5090 the tile would have counted a GB10's cores."""
    for mid, shape in cores.SHAPES.items():
        assert "cores" not in shape, mid
    assert "cores" not in cores._DENSE_27B


def test_the_tile_counts_the_cores_it_is_given_and_none_without_them():
    assert cores.estimate(M, steps_s=5.5, tokens_step=80, **{**KW, "cuda_cores": 21760})["cores"] == 21760
    assert cores.estimate(M, steps_s=5.5, tokens_step=80, **{**KW, "cuda_cores": None}) is None, \
        "no card named: no tile, never a stand-in count"
    with pytest.raises(TypeError):
        cores.estimate(M, steps_s=5.5, tokens_step=80, running=16, speculative=4, budget=2048, peak_tflops=323.2,
                       peak_gbps=273.0)


def test_every_shape_names_the_precision_its_gemms_run_in():
    """The COMPUTE tile's ceiling is the serving card's dense peak AT the model's precision (the card's hardware
    type lists one per precision). Every cluster model here is an NVFP4 checkpoint whose GEMMs — the MoE experts,
    the dense MLPs — run nvfp4 (lb/docker-compose.yml: "the MoE GEMMs dominate the FLOPs and run FLASHINFER_CUTLASS
    nvfp4"), so the LB-wide LB_GPU_PEAK_TFLOPS_PRECISION=nvfp4 it replaces is what each row now says for itself."""
    from vllm_lb.hardware import profile
    # The one exception, said here: TensorFold 0.6.6 dequantises the MLX 4-bit checkpoint's affine weights to bf16
    # for its GEMMs ("prompts: bf16 activations", the spike's rank 0), so its ceiling is bf16's.
    own = {"qwen3.8-flash-next-tensorfold-mlx-mtp": ("bf16", 96.5)}
    for mid in cores.SHAPES:
        p = cores.precision_of(mid)
        want, peak = own.get(mid, ("nvfp4", 323.2))
        assert p == want, (mid, p)
        assert profile("gb10").peak_tflops_for(p).value == peak
    assert cores.precision_of("no-such-model") is None and cores.precision_of(None) is None
