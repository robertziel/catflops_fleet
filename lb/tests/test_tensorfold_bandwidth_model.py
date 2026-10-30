"""MEMORY BANDWIDTH for a TensorFold generation: the LB's model of each forward pass, priced from the row's
`RoutedStepTraffic` at TensorFold's own pass counters (the fleet image's patch 0014) — operator 2026-10-30: "Fix
bandwidth in tensor fold". Until this the tile read "— · no per-pass byte model (StepTraffic) is recorded for this
model" beside a modeled COMPUTE figure, because the one StepTraffic family (`StepTraffic`, dense: the 27B on SGLang)
streams every weight on every pass and Flash-Next does not.

WHAT A PASS MOVES on one rank (TP=2), and where each figure is read:

    decode  = D + R x (1 - (1 - k/E)^(seqs x width))                         weights
            + min(ctx, seqs x B) x L x row  +  ctx x L x I                   selected K/V + the indexer's pooled keys
            + seqs x S x 2                                                    GDN state, read and written in place
    prefill = D + R x (1 - (1 - k/E)^rows)                                    weights; its attention not charged

  D     5,098,281,344 B  every non-routed weight a pass streams, off NVIDIA's own shard headers
                         (fixtures/shape/Qwen3.8-Flash-Next-NVFP4.header-groups.json) under TensorFold's loader
                         rules (`_loader_share` below, cuda_weights.zig): GDN and attention projections and the
                         lm_head split over the ranks as stored (bf16), hyper-connections, router, indexer and
                         PLE projections replicated, the shared expert as bf16 tiles with unit fp32 scales.
  R     33,973,862,400 B the routed experts, NVFP4 as shipped (U8 codes + F8 scales), half the width a rank.
  k/E   10 / 512         config.json num_experts_per_tok / num_experts: a row's routed experts.
  B, L  2,048, 12        indexer_budget (keys one row attends to) and the full-attention layers.
  row   528 B            K (head_dim fp8 codes + a 16-byte scale trailer) + V a position a layer a rank — the
                         engine's own "528 bytes a position a layer" in its start log.
  I     64 B             the indexer's pooled keys a position: 128 bf16 dims per block of 4 positions.
  S     57,729,024 B     the GDN recurrent state (36 x 24 x 128 x 128, f32) + conv window (36 x 3 x 5,120, bf16).

NOT CHARGED, and the tile says so (`omits`, a floor): the MTP drafter's passes — TensorFold counts no draft pass, so
the drafter head's weights and its own attention layer have no count to multiply."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from tests.fleet import GB10_PEAK_GBPS
from tests.test_tensorfold_engine import FIXTURES, NATIVE, PY066, _two_reads, catalog, spec
from vllm_lb.catalog import RoutedStepTraffic

SHAPE = FIXTURES / "shape"
CONFIG = json.loads((SHAPE / "Qwen3.8-Flash-Next-NVFP4.config.json").read_text())["text_config"]
HEADERS = json.loads((SHAPE / "Qwen3.8-Flash-Next-NVFP4.header-groups.json").read_text())
#: TensorFold's own split (cuda_weights.zig, recipe patch 0003: "DeltaNet key/value heads and attention q/kv heads by
#: rank, out_proj / o_proj input columns, every expert and the shared expert at the rank's half of the intermediate
#: width, lm_head and the draft head by vocabulary, the rest replicated").
_SPLIT = re.compile(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_a|in_proj_b|conv1d|A_log|dt_bias)"
                    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$|^lm_head\.")
#: Gathered by token (the embedding, the n-gram table), idle on text (the vision tower), or the drafter's (mtp.*,
#: not charged): resident, never streamed by a target pass.
_NOT_STREAMED = re.compile(r"embed_tokens|ngram_embedding|ple_embedding\.(layer_multipliers|ngram_heads)"
                           r"|^model\.visual|^mtp\.")
_ROUTED = re.compile(r"mlp\.experts\.\*\.(gate|up|down)_proj\.(weight|weight_scale)$")
_ROUTED_SCALARS = re.compile(r"mlp\.experts\.\*\..*(input_scale|weight_scale_2)$")
_SHARED = re.compile(r"mlp\.shared_expert\.(gate|up|down)_proj\.weight$")


def _loader_share() -> dict[str, int]:
    """One rank's bytes of NVIDIA's checkpoint as TensorFold's loader lays them out, by kind."""
    out = dict.fromkeys(("split", "replicated", "shared", "routed", "not_streamed"), 0)
    layers = CONFIG["num_hidden_layers"]
    for name, g in HEADERS["groups"].items():
        if _NOT_STREAMED.search(name) or _ROUTED_SCALARS.search(name):
            out["not_streamed"] += g["bytes"]
        elif _ROUTED.search(name):
            out["routed"] += g["bytes"] // 2
        elif _SHARED.search(name):
            # fp4Table: the rank's half of the bf16 rows as stored, plus one fp32 unit scale per 16 weights
            weights = g["bytes"] // 2 // 2
            out["shared"] += weights * 2 + weights // 16 * 4
        elif _SPLIT.search(name):
            out["split"] += g["bytes"] // 2
        else:
            out["replicated"] += g["bytes"]
    # each table's fp32 scale2, one per output row: gate|up (2 x 320 rows a rank) and down (2,560)
    out["shared"] += layers * (2 * CONFIG["moe_intermediate_size"] // 2 + CONFIG["hidden_size"]) * 4
    return out


def traffic() -> RoutedStepTraffic:
    t = spec(NATIVE).traffic
    assert isinstance(t, RoutedStepTraffic)
    return t


# ───────────────────────── the row's figures, against their sources ─────────────────────────

def test_the_header_fixture_is_the_whole_checkpoint():
    """The grouped headers add up to the index's own total_size (model.safetensors.index.json @ fc694b54)."""
    assert sum(g["bytes"] for g in HEADERS["groups"].values()) == HEADERS["total_bytes"] == 132_639_846_394


def test_weights_a_pass_streams_are_the_checkpoints_own_bytes_under_tensorfolds_split():
    share = _loader_share()
    t = traffic()
    assert t.dense_read_bytes == share["split"] + share["replicated"] + share["shared"] == 5_098_281_344
    assert t.routed_expert_bytes == share["routed"] == 33_973_862_400


def test_the_resident_share_agrees_with_the_recipes_weights_a_rank():
    """Cross-check, not the figure: streamed + the embedding (replicated) + the rank's half of the n-gram table + the
    drafter is ~62 GiB a rank — the recipe's own 62.3 GiB (measurements.json weights_gib)."""
    share = _loader_share()
    g = HEADERS["groups"]
    ngram = g["model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*.weight"]["bytes"]
    mtp = sum(v["bytes"] for k, v in g.items() if k.startswith("mtp."))
    resident = (share["split"] + share["replicated"] + share["shared"] + share["routed"]
                + g["model.language_model.embed_tokens.weight"]["bytes"] + ngram // 2 + mtp // 2)
    assert resident / 2**30 == pytest.approx(62.3, abs=1.0)


def test_the_shapes_are_config_jsons():
    t = traffic()
    layer_types = CONFIG["layer_types"]
    assert (t.experts, t.top_k) == (CONFIG["num_experts"], CONFIG["num_experts_per_tok"]) == (512, 10)
    assert t.attention_layers == layer_types.count("full_attention") == 12
    assert t.linear_attention_layers == layer_types.count("linear_attention") == 36
    assert (t.num_key_value_heads, t.head_dim) == (CONFIG["num_key_value_heads"], CONFIG["head_dim"])
    assert (t.index_budget, t.index_ratio, t.index_head_dim) == (
        CONFIG["indexer_budget"], CONFIG["indexer_compress_ratio"], CONFIG["indexer_head_dim"])
    assert (t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim,
            t.linear_conv_kernel) == (CONFIG["linear_num_key_heads"], CONFIG["linear_num_value_heads"],
                                      CONFIG["linear_key_head_dim"], CONFIG["linear_value_head_dim"],
                                      CONFIG["linear_conv_kernel_dim"])


def test_a_position_a_layer_is_the_engines_own_528_bytes():
    log = (FIXTURES / "tensorfold-native-2026-10-29-rank0.log").read_text()
    said = int(re.search(r"fp8 KV cache, (\d+) bytes a position a layer", log).group(1))
    assert traffic().kv_row_bytes == said == 528
    # bf16 (--kv-dtype bf16): no scale trailer — kRow + vRow = 2 x 1 x 256 x 2
    assert spec(NATIVE).at_kv_cache_dtype("bfloat16").traffic.kv_row_bytes == 1024


def test_the_gdn_state_a_sequence_is_the_engines_fixed_part():
    """cuda_state.zig fixedSizes: conv [36, 3, 2·8·128 + 24·128] bf16 + recurrent [36, 24, 128, 128] f32, a rank."""
    assert traffic().state_bytes == 36 * 3 * 5120 * 2 + 36 * 24 * 128 * 128 * 4 == 57_729_024


def test_every_figure_names_its_source_and_the_omission_is_stated():
    t = traffic()
    for needle in ("fc694b54", "cuda_weights.zig", "528 bytes a position a layer", "fixedSizes"):
        assert needle in t.source, needle
    assert "drafter" in t.omits and "not charged" in t.omits


# ───────────────────────── one pass ─────────────────────────

def test_a_decode_pass_reads_the_experts_its_rows_touch():
    t = traffic()
    weights, kv, state = t.decode_pass_bytes(context=39_500, seqs=2, verify_width=4.5)
    touched = 1 - (1 - 10 / 512) ** 9
    assert weights == pytest.approx(5_098_281_344 + 33_973_862_400 * touched, rel=1e-9)
    assert kv == min(39_500, 2 * 2048) * 12 * 528 + 39_500 * 12 * 64
    assert state == 2 * 57_729_024 * 2


def test_a_prompt_chunk_of_2048_rows_reads_every_expert_and_a_short_one_does_not():
    t = traffic()
    assert t.prefill_bytes(2048) == pytest.approx(5_098_281_344 + 33_973_862_400, rel=1e-12)
    assert t.prefill_bytes(28) == pytest.approx(5_098_281_344 + 33_973_862_400 * (1 - (1 - 10 / 512) ** 28))
    # rows unknown: a full chunk, never a guess below it
    assert t.prefill_bytes(None) == t.prefill_bytes(2048)


# ───────────────────────── the tile ─────────────────────────

def test_the_bandwidth_tile_is_modeled_in_gbs_against_the_gb10s_273():
    from tests.test_sglang_compute_model import _status
    sat = _status(NATIVE, _two_reads())["saturation"]
    t = traffic()
    decode = sum(t.decode_pass_bytes(context=39_500, seqs=2, verify_width=4.5))
    want = (100 * decode + 2 * t.prefill_bytes(2048)) / 6.0 / 1e9
    bw = sat["bandwidth"]
    assert bw["why"] is None
    assert bw["rate"] == pytest.approx(want, abs=0.05)
    assert bw["peak"] == GB10_PEAK_GBPS
    assert bw["pct"] == pytest.approx(want / GB10_PEAK_GBPS * 100, abs=0.1)
    model = sat["bandwidth_model"]
    assert model["why"] is None
    # a floor twice over: the prompt chunks' attention, and the drafter nobody counts
    assert t.omits in model["floor"]
    for box in bw["boxes"]:
        assert box["rate"] == pytest.approx(bw["rate"], abs=0.05) and box["derived"]


def test_the_live_gen284_scrape_prices_its_last_pass():
    """The live head of generation 284 (fixtures/tensorfold-metrics-fleet2-live-gen284.txt): one stream, 958 tokens
    of context, 4 rows — and its lifetime pass rate over its own busy clock."""
    from vllm_lb.probe import parse_metrics
    s = parse_metrics((FIXTURES / "tensorfold-metrics-fleet2-live-gen284.txt").read_text())
    t = traffic()
    weights, kv, state = t.decode_pass_bytes(context=s["decode_context_tokens"], seqs=1,
                                             verify_width=s["spec_verify_width"])
    assert weights / 1e9 == pytest.approx(7.68, abs=0.01)
    assert (kv + state) / 1e9 == pytest.approx(0.13, abs=0.01)


def test_the_mlx_row_states_none_and_says_so():
    """The Python 0.6.6 engine carries no patch 0014: no pass counter to price, so no traffic is recorded for it."""
    from tests.test_sglang_compute_model import _status
    from vllm_lb.proxy import NO_STEP_TRAFFIC
    assert spec(PY066).traffic is None
    sat = _status(PY066, _two_reads())["saturation"]
    assert sat["bandwidth"]["rate"] is None and sat["bandwidth"]["why"] == NO_STEP_TRAFFIC


def test_a_traffic_names_its_family_and_an_unknown_one_is_refused():
    from vllm_lb.catalog import ModelCatalog, document
    doc = document()
    row = next(m for m in doc["models"] if m["id"] == NATIVE)
    row["traffic"] = dict(row["traffic"], family="guessed")
    with pytest.raises(ValueError, match="family"):
        ModelCatalog.from_dict(doc)
