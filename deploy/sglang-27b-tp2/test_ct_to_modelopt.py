"""ct_to_modelopt: a compressed-tensors Flash-Next checkpoint comes out in the ModelOpt layout, exactly.

Run (the SGLang image has torch, safetensors and pytest; nothing here needs a GPU):
    rsync -a deploy/sglang-27b-tp2/{ct_to_modelopt.py,test_ct_to_modelopt.py} box-a:/tmp/ct2mo/
    ssh box-a docker run --rm -v /tmp/ct2mo:/work:ro -w /tmp --entrypoint python3 \
        myllmbox/sglang-27b-tp2:708f51e-fleet.3 -m pytest -q -p no:cacheprovider /work/test_ct_to_modelopt.py

The fixtures are tiny synthetic checkpoints with the REAL tensor families of the two formats (routed experts,
an MTP drafter with unequal gate/up globals, FP8-channel GDN/QSA/shared-expert linears, the NVFP4 lm_head, a
PLE shard, BF16 pass-through tensors) so that every rule the converter applies is exercised on something
whose dequantised value the test computes INDEPENDENTLY (its own E2M1 table, its own fp8 decode).
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ct_to_modelopt as c2m  # noqa: E402

HID, INTER, VOCAB, EXPERTS = 64, 32, 100, 4
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ── independent dequantisation (the test's own, not the module's) ─────────────

def fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """U8 [N, K/2] -> float32 [N, K]: low nibble first, sign bit 3, E2M1 magnitude."""
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
    for src, dst in ((lo, out[:, 0::2]), (hi, out[:, 1::2])):
        mag = E2M1[src & 7]
        dst.copy_(torch.where((src >> 3).bool(), -mag, mag))
    return out


def dequant_ct(packed, block_scale, global_scale) -> torch.Tensor:
    """compressed-tensors: fp4 * block_scale / weight_global_scale (the global is a DIVISOR)."""
    bs = block_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return fp4_values(packed) * bs / float(global_scale.reshape(-1)[0])


def dequant_modelopt(packed, block_scale, weight_scale_2) -> torch.Tensor:
    """ModelOpt: fp4 * block_scale * weight_scale_2 (the global is a MULTIPLIER)."""
    bs = block_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return fp4_values(packed) * bs * float(weight_scale_2.reshape(-1)[0])


def silu(x):
    return x * torch.sigmoid(x)


# ── the synthetic checkpoints ──────────────────────────────────────────────────

def nvfp4_proj(gen, out_features, in_features, global_scale, scale_max=448.0):
    packed = torch.randint(0, 256, (out_features, in_features // 2), generator=gen, dtype=torch.uint8)
    scales = (torch.rand(out_features, in_features // 16, generator=gen) * scale_max + 1).to(torch.float8_e4m3fn)
    return {"weight_packed": packed, "weight_scale": scales,
            "weight_global_scale": torch.tensor([global_scale], dtype=torch.float32)}


def fp8_channel_linear(gen, out_features, in_features):
    w = (torch.randn(out_features, in_features, generator=gen) * 2).to(torch.float8_e4m3fn)
    s = (torch.rand(out_features, 1, generator=gen) * 1e-3 + 1e-5).to(torch.bfloat16)
    return {"weight": w, "weight_scale": s}


def expert_tensors(gen, prefix, globals_):
    """{prefix}.{proj}.{leaf} for gate/up/down with the given (gate, up, down) weight_global_scale."""
    out = {}
    for proj, g in zip(("gate", "up", "down"), globals_):
        n, k = (INTER, HID) if proj != "down" else (HID, INTER)
        for leaf, t in nvfp4_proj(gen, n, k, g).items():
            out[f"{prefix}.{proj}_proj.{leaf}"] = t
    return out


def _save(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


def make_source(root: pathlib.Path, *, unequal_mtp=True) -> dict:
    """A compressed-tensors checkpoint with the families of the real one. Returns the tensors by file."""
    gen = torch.Generator().manual_seed(7)
    root.mkdir()
    L = "model.language_model.layers"
    shard1, shard2 = {}, {}
    # layer 0: experts, one GDN block, shared expert, router, hyper-connection, BF16 pass-through
    for e in range(EXPERTS):
        shard1.update(expert_tensors(gen, f"{L}.0.mlp.experts.{e}", (15000.0 + e, 15000.0 + e, 16000.0 + e)))
    for name, (n, k) in {"in_proj_qkv": (4 * HID, HID), "in_proj_z": (2 * HID, HID), "out_proj": (HID, 2 * HID)}.items():
        for leaf, t in fp8_channel_linear(gen, n, k).items():
            shard1[f"{L}.0.linear_attn.{name}.{leaf}"] = t
    shard1[f"{L}.0.linear_attn.in_proj_a.weight"] = torch.randn(8, HID, generator=gen).to(torch.bfloat16)
    shard1[f"{L}.0.linear_attn.A_log"] = torch.randn(8, generator=gen).to(torch.bfloat16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        n, k = (INTER, HID) if name != "down_proj" else (HID, INTER)
        for leaf, t in fp8_channel_linear(gen, n, k).items():
            shard1[f"{L}.0.mlp.shared_expert.{name}.{leaf}"] = t
    shard1[f"{L}.0.mlp.gate.weight"] = torch.randn(EXPERTS, HID, generator=gen).to(torch.bfloat16)
    shard1[f"{L}.0.mlp.shared_expert_gate.weight"] = torch.randn(1, HID, generator=gen).to(torch.bfloat16)
    shard1["model.language_model.embed_tokens.weight"] = torch.randn(VOCAB, HID, generator=gen).to(torch.bfloat16)
    shard1[f"{L}.1.ple.ple_embedding.layer_multipliers"] = torch.tensor([1, 2, 3], dtype=torch.int64)
    # layer 1 (a full-attention layer): experts split ACROSS the two shards, QSA in FP8-channel
    for e in range(EXPERTS):
        t = expert_tensors(gen, f"{L}.1.mlp.experts.{e}", (12000.0 + e, 12000.0 + e, 13000.0 + e))
        for k_, v in t.items():
            (shard2 if (e == 3 and ".down_proj." in k_) else shard1)[k_] = v
    for name, (n, k) in {"q_proj": (2 * HID, HID), "k_proj": (HID, HID), "v_proj": (HID, HID), "o_proj": (HID, 2 * HID)}.items():
        for leaf, t in fp8_channel_linear(gen, n, k).items():
            # the real checkpoint splits layer 35's q_proj weight and weight_scale over two shards: o_proj's scale
            # goes in the OTHER shard here so the converter has to fetch it from where the index says it is
            (shard1 if (name == "o_proj" and leaf == "weight_scale") else shard2)[f"{L}.1.self_attn.{name}.{leaf}"] = t
    shard2[f"{L}.1.self_attn.q_norm.weight"] = torch.ones(HID, dtype=torch.bfloat16)
    shard2[f"{L}.1.self_attn.indexer.index_qk_proj.weight"] = torch.randn(16, HID, generator=gen).to(torch.bfloat16)
    # the MTP drafter: experts 1 and 2 carry UNEQUAL gate/up globals (the real drafter has 507 of 512)
    mtp = {}
    for e in range(EXPERTS):
        g = 9000.0 + e
        u = g if (e in (0, 3) or not unequal_mtp) else g * (0.4 if e == 1 else 2.5)
        mtp.update(expert_tensors(gen, f"mtp.layers.0.mlp.experts.{e}", (g, u, 9500.0 + e)))
    mtp["mtp.layers.0.mlp.gate.weight"] = torch.randn(EXPERTS, HID, generator=gen).to(torch.bfloat16)
    mtp["mtp.fc_embedding.weight"] = torch.randn(HID, 2 * HID, generator=gen).to(torch.bfloat16)
    lmhead = {f"lm_head.{k}": v for k, v in nvfp4_proj(gen, VOCAB, HID, 6500.0).items()}
    lmhead["lm_head.weight_global_scale"] = lmhead["lm_head.weight_global_scale"].reshape(())
    ple = {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_global": torch.tensor(3.3e-5),
           "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_shard_0.packed":
               torch.randint(0, 256, (64, 80), generator=gen, dtype=torch.uint8),
           "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_shard_0.scales":
               torch.rand(64, 10, generator=gen).to(torch.float8_e4m3fn)}
    files = {"model-00001-of-00002.safetensors": shard1, "model-00002-of-00002.safetensors": shard2,
             "model-mtp.safetensors": mtp, "model-lmhead-nvfp4.safetensors": lmhead,
             "ple-nvfp4-00001-of-00001.safetensors": ple}
    weight_map, total = {}, 0
    for fname, tensors in files.items():
        _save(root / fname, tensors)
        for name, t in tensors.items():
            weight_map[name] = fname
            total += t.numel() * t.element_size()
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "dtype": "bfloat16",
        "transformers_version": "5.16.0",
        "text_config": {"hidden_size": HID, "num_experts": EXPERTS, "moe_intermediate_size": INTER, "hidden_act": "silu",
                        "norm_topk_prob": True, "seed": 1234},
        "ple_quantization": {"format": "nvfp4", "block": 16, "rows": 64, "dim": 160, "shards": 1, "shard_rows": 64},
        "mbx_provenance": [{"tool": "quantize-lm-head.py", "from": "hibrid47-uncensored", "format": "compressed-tensors"}],
        "quantization_config": {"quant_method": "compressed-tensors", "format": "mixed-precision", "version": "0.18.0",
                                "kv_cache_scheme": None, "ignore": [f"{L}.0.mlp.gate", "model.language_model.embed_tokens"],
                                "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
                                                  "group_1": {"weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                                                          "group_size": 16}, "targets": ["re:.*mlp\\.experts\\..*proj$"]}}}}))
    for small in ("tokenizer.json", "chat_template.jinja", "README.md", "README-orcarouter.md", "LICENSE", "generation_config.json"):
        (root / small).write_text(f"{small} of the source\n")
    return files


def make_reference(root: pathlib.Path, *, drop_input_scale_of: str | None = None) -> dict:
    """The censored checkpoint's shape: a ModelOpt MIXED_PRECISION config and per-expert input_scale tensors."""
    gen = torch.Generator().manual_seed(11)
    root.mkdir()
    L = "model.language_model.layers"
    quantized = {}
    for n in (0, 1):
        for alias in (f"model.language_model.layers.{n}", f"language_model.model.layers.{n}", f"model.layers.{n}"):
            quantized[f"{alias}.mlp.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
        if n == 0:
            for alias in (f"model.language_model.layers.{n}", f"language_model.model.layers.{n}", f"model.layers.{n}"):
                for mod in ("in_proj_qkv", "in_proj_qkvz", "in_proj_z", "out_proj"):
                    quantized[f"{alias}.linear_attn.{mod}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    # the drafter is announced under BOTH its own index 0 and num_hidden_layers (2 here, 48 in the real one): the
    # engines name the MTP layer by whichever convention they use, and the censored config covers both
    for n in (0, 2):
        for alias in (f"mtp.layers.{n}", f"mtp.model.layers.{n}", f"model.mtp.layers.{n}", f"layers.{n}"):
            quantized[f"{alias}.mlp.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    for alias in ("lm_head", "model.lm_head", "language_model.lm_head", "mtp.lm_head"):
        quantized[alias] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    ignore = [f"{L}.0.linear_attn.A_log", f"{L}.0.linear_attn.in_proj_a", f"{L}.0.mlp.gate", f"{L}.0.mlp.shared_expert.gate_proj",
              f"{L}.0.mlp.shared_expert.up_proj", f"{L}.0.mlp.shared_expert.down_proj", f"{L}.0.mlp.shared_expert_gate",
              f"{L}.1.self_attn.q_proj", f"{L}.1.self_attn.k_proj", f"{L}.1.self_attn.v_proj", f"{L}.1.self_attn.o_proj",
              f"{L}.1.self_attn.q_norm", f"{L}.1.self_attn.indexer.index_qk_proj", "model.language_model.embed_tokens",
              f"{L}.1.ple.ple_embedding.ngram_embedding", f"{L}.1.ple.ple_embedding.layer_multipliers",
              "mtp.fc_embedding", "mtp.layers.0.mlp.gate"]
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "transformers_version": "5.8.0.dev0",
        "text_config": {"hidden_size": HID, "num_experts": EXPERTS, "moe_intermediate_size": INTER, "hidden_act": "silu",
                        "num_hidden_layers": 2},
        "ple_quantization": {"format": "nvfp4", "block": 16, "rows": 64, "dim": 160, "shards": 1, "shard_rows": 64},
        "mbx_provenance": [{"tool": "quantize-lm-head.py", "from": "hibrid47"}],
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": 16,
                                "with_input_scale": True, "input_scale_quantile": 0.9999, "ignore": ignore,
                                "quantized_layers": quantized}}))
    scales = {}
    for n in (0, 1):
        for e in range(EXPERTS):
            for proj in ("gate", "up", "down"):
                scales[f"{L}.{n}.mlp.experts.{e}.{proj}_proj.input_scale"] = torch.tensor(1e-3 * (1 + n) + 1e-5 * e + (2e-4 if proj == "down" else 0.0))
                scales[f"{L}.{n}.mlp.experts.{e}.{proj}_proj.weight_scale_2"] = torch.tensor(6e-5)
    for e in range(EXPERTS):
        for proj in ("gate", "up", "down"):
            scales[f"mtp.layers.0.mlp.experts.{e}.{proj}_proj.input_scale"] = torch.tensor(2e-3 + 1e-5 * e + (3e-4 if proj == "down" else 0.0))
    scales[f"{L}.0.linear_attn.in_proj_qkv.weight_scale_2"] = torch.tensor(5e-5)   # the reference's own W4A16 GDN, unused
    scales["lm_head.weight_scale_2"] = torch.tensor(1.5e-4)
    if drop_input_scale_of:
        del scales[drop_input_scale_of]
    files = {"model-00001-of-00002.safetensors": {k: v for k, v in scales.items() if "layers.0." in k and not k.startswith("mtp")},
             "model-00002-of-00002.safetensors": {k: v for k, v in scales.items() if not ("layers.0." in k and not k.startswith("mtp"))}}
    weight_map = {}
    for fname, tensors in files.items():
        _save(root / fname, tensors)
        weight_map.update({k: fname for k in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return scales


@pytest.fixture
def ckpt(tmp_path):
    src, ref = tmp_path / "Src_uncensored", tmp_path / "Ref"
    files = make_source(src)
    scales = make_reference(ref)
    return src, ref, tmp_path / "Src_uncensored.modelopt", files, scales


def load_all(d: pathlib.Path) -> dict:
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    out = {}
    for fname in sorted(set(idx["weight_map"].values())):
        with safe_open(str(d / fname), "pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out, idx


def convert(src, ref, dst):
    return c2m.main(["convert", str(src), str(ref), str(dst)])


# ── the layout ────────────────────────────────────────────────────────────────

def test_expert_tensors_take_the_modelopt_names_shapes_and_dtypes(ckpt):
    src, ref, dst, _, _ = ckpt
    convert(src, ref, dst)
    out, idx = load_all(dst)
    for n in (0, 1):
        for e in range(EXPERTS):
            for proj in ("gate", "up", "down"):
                mod = f"model.language_model.layers.{n}.mlp.experts.{e}.{proj}_proj"
                rows, cols = (INTER, HID) if proj != "down" else (HID, INTER)
                assert out[f"{mod}.weight"].dtype == torch.uint8 and tuple(out[f"{mod}.weight"].shape) == (rows, cols // 2)
                assert out[f"{mod}.weight_scale"].dtype == torch.float8_e4m3fn and tuple(out[f"{mod}.weight_scale"].shape) == (rows, cols // 16)
                assert out[f"{mod}.weight_scale_2"].dtype == torch.float32 and out[f"{mod}.weight_scale_2"].shape == ()
                assert out[f"{mod}.input_scale"].dtype == torch.float32 and out[f"{mod}.input_scale"].shape == ()
                for leaf in ("weight_packed", "weight_global_scale"):
                    assert f"{mod}.{leaf}" not in out
    assert not [k for k in out if k.endswith(("weight_packed", "weight_global_scale"))]
    # every tensor the index names is in its file with the stated dtype/shape, and nothing is unindexed
    assert set(idx["weight_map"]) == set(out)
    assert idx["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in out.values())
    # the shard layout is the source's: the same file names, the drafter in its own file, lm_head in its own
    assert idx["weight_map"]["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == "model-mtp.safetensors"
    assert idx["weight_map"]["lm_head.weight"] == "model-lmhead-nvfp4.safetensors"
    assert idx["weight_map"]["model.language_model.layers.1.mlp.experts.3.down_proj.weight"] == "model-00002-of-00002.safetensors"


def test_experts_dequantise_to_the_same_weights(ckpt):
    """fp4 * block_scale / global == fp4 * block_scale * weight_scale_2: the bytes are kept, only the global inverts."""
    src, ref, dst, files, _ = ckpt
    convert(src, ref, dst)
    out, _ = load_all(dst)
    src_t = {k: v for f in files.values() for k, v in f.items()}
    for n in (0, 1):
        for e in range(EXPERTS):
            for proj in ("gate", "up", "down"):
                mod = f"model.language_model.layers.{n}.mlp.experts.{e}.{proj}_proj"
                want = dequant_ct(src_t[f"{mod}.weight_packed"], src_t[f"{mod}.weight_scale"], src_t[f"{mod}.weight_global_scale"])
                got = dequant_modelopt(out[f"{mod}.weight"], out[f"{mod}.weight_scale"], out[f"{mod}.weight_scale_2"])
                torch.testing.assert_close(got, want, rtol=1e-6, atol=0)
                assert torch.equal(out[f"{mod}.weight"], src_t[f"{mod}.weight_packed"])
                assert torch.equal(out[f"{mod}.weight_scale"].view(torch.uint8), src_t[f"{mod}.weight_scale"].view(torch.uint8))
                assert out[f"{mod}.weight_scale_2"].item() == pytest.approx(1.0 / src_t[f"{mod}.weight_global_scale"].item(), rel=1e-7)


def test_expert_input_scales_are_the_references_for_the_same_expert(ckpt):
    src, ref, dst, _, scales = ckpt
    convert(src, ref, dst)
    out, _ = load_all(dst)
    for n in (0, 1):
        for e in range(EXPERTS):
            for proj in ("gate", "up", "down"):
                mod = f"model.language_model.layers.{n}.mlp.experts.{e}.{proj}_proj"
                assert out[f"{mod}.input_scale"].item() == scales[f"{mod}.input_scale"].item(), mod
    # the drafter's experts with equal gate/up globals carry the reference's scales untouched
    for e in (0, 3):
        for proj in ("gate", "up", "down"):
            mod = f"mtp.layers.0.mlp.experts.{e}.{proj}_proj"
            assert out[f"{mod}.input_scale"].item() == scales[f"{mod}.input_scale"].item(), mod


def test_unequal_gate_up_globals_are_absorbed_into_down_proj_exactly(ckpt):
    """The engines' MoE kernels scale the fused gate|up GEMM by ONE global (the gate's). An expert whose up global
    differs would come out wrong by that ratio, so the converter sets up := gate and folds the ratio into
    down_proj's weight_scale_2 (and divides its input_scale by it): silu(g) * u is linear in u and down is linear,
    so the expert's function is unchanged, to float32 rounding."""
    src, ref, dst, files, scales = ckpt
    convert(src, ref, dst)
    out, _ = load_all(dst)
    src_t = {k: v for f in files.values() for k, v in f.items()}
    x = torch.randn(5, HID, generator=torch.Generator().manual_seed(3))
    for e in range(EXPERTS):
        mod = f"mtp.layers.0.mlp.experts.{e}"
        g_src = dequant_ct(src_t[f"{mod}.gate_proj.weight_packed"], src_t[f"{mod}.gate_proj.weight_scale"], src_t[f"{mod}.gate_proj.weight_global_scale"])
        u_src = dequant_ct(src_t[f"{mod}.up_proj.weight_packed"], src_t[f"{mod}.up_proj.weight_scale"], src_t[f"{mod}.up_proj.weight_global_scale"])
        d_src = dequant_ct(src_t[f"{mod}.down_proj.weight_packed"], src_t[f"{mod}.down_proj.weight_scale"], src_t[f"{mod}.down_proj.weight_global_scale"])
        want = (silu(x @ g_src.T) * (x @ u_src.T)) @ d_src.T
        # what a one-global kernel computes from the converted tensors: up decoded with the GATE's weight_scale_2
        gate_ws2 = out[f"{mod}.gate_proj.weight_scale_2"]
        assert torch.equal(out[f"{mod}.up_proj.weight_scale_2"], gate_ws2), "up carries the gate's global after conversion"
        g_out = dequant_modelopt(out[f"{mod}.gate_proj.weight"], out[f"{mod}.gate_proj.weight_scale"], gate_ws2)
        u_out = dequant_modelopt(out[f"{mod}.up_proj.weight"], out[f"{mod}.up_proj.weight_scale"], gate_ws2)
        d_out = dequant_modelopt(out[f"{mod}.down_proj.weight"], out[f"{mod}.down_proj.weight_scale"], out[f"{mod}.down_proj.weight_scale_2"])
        got = (silu(x @ g_out.T) * (x @ u_out.T)) @ d_out.T
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        # down's activation scale follows the same ratio, so its quantisation window still fits the rescaled input
        r = src_t[f"{mod}.gate_proj.weight_global_scale"].item() / src_t[f"{mod}.up_proj.weight_global_scale"].item()
        want_is = scales[f"{mod}.down_proj.input_scale"].item() / r
        assert out[f"{mod}.down_proj.input_scale"].item() == pytest.approx(want_is, rel=1e-6)
        if e in (0, 3):
            assert r == 1.0 and out[f"{mod}.down_proj.weight_scale_2"].item() == pytest.approx(1 / src_t[f"{mod}.down_proj.weight_global_scale"].item(), rel=1e-7)


def test_fp8_channel_linears_become_exact_bf16_and_lose_their_scale(ckpt):
    src, ref, dst, files, _ = ckpt
    convert(src, ref, dst)
    out, _ = load_all(dst)
    src_t = {k: v for f in files.values() for k, v in f.items()}
    mods = [f"model.language_model.layers.0.linear_attn.{m}" for m in ("in_proj_qkv", "in_proj_z", "out_proj")] + \
           [f"model.language_model.layers.0.mlp.shared_expert.{m}" for m in ("gate_proj", "up_proj", "down_proj")] + \
           [f"model.language_model.layers.1.self_attn.{m}" for m in ("q_proj", "k_proj", "v_proj", "o_proj")]
    for mod in mods:
        want = (src_t[f"{mod}.weight"].to(torch.float32) * src_t[f"{mod}.weight_scale"].to(torch.float32)).to(torch.bfloat16)
        assert out[f"{mod}.weight"].dtype == torch.bfloat16
        assert torch.equal(out[f"{mod}.weight"], want), mod
        assert f"{mod}.weight_scale" not in out
    assert not [k for k, v in out.items() if v.dtype == torch.float8_e4m3fn and not k.endswith((".weight_scale", ".scales"))]


def test_lm_head_is_w4a16_nvfp4_with_the_inverted_global(ckpt):
    src, ref, dst, files, _ = ckpt
    convert(src, ref, dst)
    out, _ = load_all(dst)
    lm = files["model-lmhead-nvfp4.safetensors"]
    assert torch.equal(out["lm_head.weight"], lm["lm_head.weight_packed"])
    assert out["lm_head.weight_scale_2"].shape == () and out["lm_head.weight_scale_2"].item() == pytest.approx(1 / 6500.0, rel=1e-7)
    assert "lm_head.input_scale" not in out, "W4A16: the censored head has no activation scale either"
    torch.testing.assert_close(dequant_modelopt(out["lm_head.weight"], out["lm_head.weight_scale"], out["lm_head.weight_scale_2"]),
                               dequant_ct(lm["lm_head.weight_packed"], lm["lm_head.weight_scale"], lm["lm_head.weight_global_scale"]),
                               rtol=1e-6, atol=0)


def test_everything_else_passes_through_untouched_and_the_ple_shards_are_hardlinks(ckpt):
    src, ref, dst, files, _ = ckpt
    convert(src, ref, dst)
    out, idx = load_all(dst)
    src_t = {k: v for f in files.values() for k, v in f.items()}
    for name in ("model.language_model.embed_tokens.weight", "model.language_model.layers.0.mlp.gate.weight",
                 "model.language_model.layers.0.linear_attn.in_proj_a.weight", "model.language_model.layers.0.linear_attn.A_log",
                 "model.language_model.layers.1.ple.ple_embedding.layer_multipliers", "mtp.fc_embedding.weight",
                 "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight"):
        assert out[name].dtype == src_t[name].dtype and torch.equal(out[name], src_t[name]), name
    ple = "ple-nvfp4-00001-of-00001.safetensors"
    assert os.stat(dst / ple).st_ino == os.stat(src / ple).st_ino, "identical bytes are shared, not copied"
    assert idx["weight_map"]["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_shard_0.packed"] == ple
    for small in ("tokenizer.json", "chat_template.jinja", "README.md", "README-orcarouter.md", "LICENSE", "generation_config.json"):
        assert (dst / small).read_text() == (src / small).read_text()
    assert (dst / "README-modelopt-conversion.md").exists()
    assert not (dst.parent / f"{dst.name}.partial").exists(), "the partial dir was renamed in one step"


# ── the config ────────────────────────────────────────────────────────────────

def test_the_config_is_the_references_modelopt_scheme_with_the_bf16_modules_moved_to_ignore(ckpt):
    src, ref, dst, _, _ = ckpt
    convert(src, ref, dst)
    cfg = json.loads((dst / "config.json").read_text())
    ref_cfg = json.loads((ref / "config.json").read_text())
    src_cfg = json.loads((src / "config.json").read_text())
    q = cfg["quantization_config"]
    assert q["quant_method"] == "modelopt" and q["quant_algo"] == "MIXED_PRECISION" and q["group_size"] == 16
    assert q["with_input_scale"] is True
    L = "model.language_model.layers"
    # the experts (both aliases sets), the drafter's experts and lm_head keep the reference's entries verbatim
    for key, info in ref_cfg["quantization_config"]["quantized_layers"].items():
        if ".linear_attn." in key:
            assert key not in q["quantized_layers"], f"{key}: dequantised to BF16, must not be announced as W4A16"
        else:
            assert q["quantized_layers"][key] == info, key
    assert not [k for k in q["quantized_layers"] if ".linear_attn." in k]
    assert set(q["quantized_layers"]) == {k for k in ref_cfg["quantization_config"]["quantized_layers"] if ".linear_attn." not in k}
    for alias in ("mtp.layers.2", "mtp.model.layers.2", "model.mtp.layers.2", "layers.2", "mtp.layers.0", "layers.0"):
        assert q["quantized_layers"][f"{alias}.mlp.experts.up_proj"]["quant_algo"] == "NVFP4", "the drafter under both indices"
    # every module that came out BF16 is excluded, the fused GDN name included, and nothing the reference excluded is lost
    for mod in ("in_proj_qkv", "in_proj_z", "in_proj_qkvz", "out_proj"):
        assert f"{L}.0.linear_attn.{mod}" in q["ignore"]
    for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"{L}.1.self_attn.{mod}" in q["ignore"]
    assert set(ref_cfg["quantization_config"]["ignore"]) <= set(q["ignore"])
    assert len(q["ignore"]) == len(set(q["ignore"]))
    # the architecture keys are the reference's (the config both engines already serve), the provenance the source's plus ours
    assert cfg["text_config"] == ref_cfg["text_config"] and cfg["transformers_version"] == ref_cfg["transformers_version"]
    assert cfg["ple_quantization"] == ref_cfg["ple_quantization"]
    assert cfg["mbx_provenance"][:-1] == src_cfg["mbx_provenance"]
    note = cfg["mbx_provenance"][-1]
    assert note["tool"] == "ct_to_modelopt.py" and note["from"] == src.name and note["input_scale_from"] == ref.name
    assert note["absorbed_experts"] == 2 and note["fp8_channel_to_bf16"] == 10
    assert "hf_quant_config.json" not in os.listdir(dst), "tp-rank builds the SGLang view from config.json, as for the original"


# ── refusals: anything the rules do not cover stops the conversion, before a byte of weights is written ──

def test_refuses_a_reference_missing_an_experts_input_scale(tmp_path):
    src, ref = tmp_path / "Src_uncensored", tmp_path / "Ref"
    make_source(src)
    make_reference(ref, drop_input_scale_of="model.language_model.layers.1.mlp.experts.2.down_proj.input_scale")
    with pytest.raises(SystemExit, match="layers.1.mlp.experts.2.down_proj.input_scale"):
        convert(src, ref, tmp_path / "out")
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob("out.partial*"))


def test_refuses_a_source_that_is_not_compressed_tensors(ckpt):
    src, ref, dst, _, _ = ckpt
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"]["quant_method"] = "modelopt"
    (src / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="compressed-tensors"):
        convert(src, ref, dst)


def test_refuses_a_reference_that_is_not_modelopt_mixed_precision(ckpt):
    src, ref, dst, _, _ = ckpt
    cfg = json.loads((ref / "config.json").read_text())
    cfg["quantization_config"]["quant_algo"] = "NVFP4"
    (ref / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="MIXED_PRECISION"):
        convert(src, ref, dst)


@pytest.mark.parametrize("name,tensor,why", [
    ("model.language_model.layers.0.linear_attn.in_proj_a.weight_packed", torch.zeros(8, 32, dtype=torch.uint8), "weight_packed"),
    ("model.language_model.layers.0.mlp.some_proj.weight", torch.zeros(4, 64).to(torch.float8_e4m3fn), "F8_E4M3"),
    ("model.language_model.layers.0.linear_attn.in_proj_a.weight_scale", torch.ones(8, 1, dtype=torch.bfloat16), "weight_scale"),
    ("model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale", torch.tensor(1.0), "input_scale"),
])
def test_refuses_a_tensor_no_rule_covers(ckpt, name, tensor, why):
    src, ref, dst, _, _ = ckpt
    extra = src / "model-00003-of-00002.safetensors"
    _save(extra, {name: tensor})
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    idx["weight_map"][name] = extra.name
    (src / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(SystemExit, match=why):
        convert(src, ref, dst)
    assert not dst.exists()


def test_refuses_an_fp8_weight_whose_scale_is_not_per_channel(ckpt):
    src, ref, dst, files, _ = ckpt
    shard = dict(files["model-00002-of-00002.safetensors"])
    shard["model.language_model.layers.1.self_attn.q_proj.weight_scale"] = torch.tensor(2.0, dtype=torch.bfloat16)
    _save(src / "model-00002-of-00002.safetensors", shard)
    with pytest.raises(SystemExit, match="q_proj.weight_scale"):
        convert(src, ref, dst)


def test_refuses_an_expert_layer_the_reference_does_not_list_and_a_listed_layer_the_source_lacks(ckpt):
    src, ref, dst, _, _ = ckpt
    cfg = json.loads((ref / "config.json").read_text())
    for alias in ("model.language_model.layers.1", "language_model.model.layers.1", "model.layers.1"):
        del cfg["quantization_config"]["quantized_layers"][f"{alias}.mlp.experts.up_proj"]
    (ref / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="layers.1"):
        convert(src, ref, dst)
    cfg = json.loads((ref / "config.json").read_text())
    for alias in ("model.language_model.layers.1", "language_model.model.layers.1", "model.layers.1"):
        cfg["quantization_config"]["quantized_layers"][f"{alias}.mlp.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
        cfg["quantization_config"]["quantized_layers"][f"{alias}.mlp.experts.gate_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    cfg["quantization_config"]["quantized_layers"]["model.language_model.layers.7.mlp.experts.up_proj"] = {"quant_algo": "NVFP4", "group_size": 16}
    (ref / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="layers.7"):
        convert(src, ref, dst)


def test_refuses_to_write_over_an_existing_destination(ckpt):
    src, ref, dst, _, _ = ckpt
    dst.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        convert(src, ref, dst)


# ── verify: the converted checkpoint checked against its source, and a corruption caught ──

def test_verify_passes_on_the_conversion_and_catches_a_corrupted_scale(ckpt, capsys):
    src, ref, dst, _, _ = ckpt
    convert(src, ref, dst)
    assert c2m.main(["verify", str(src), str(ref), str(dst), "--experts", "all"]) == 0
    report = capsys.readouterr().out
    assert "absorbed" in report and "fp8" in report.lower()
    # flip the drafter's expert 2 down_proj weight_scale_2 and the function check must fail
    path = dst / "model-mtp.safetensors"
    with safe_open(str(path), "pt") as fh:
        tensors = {k: fh.get_tensor(k) for k in fh.keys()}
    tensors["mtp.layers.0.mlp.experts.2.down_proj.weight_scale_2"] = tensors["mtp.layers.0.mlp.experts.2.down_proj.weight_scale_2"] * 1.01
    _save(path, tensors)
    with pytest.raises(SystemExit, match="experts.2"):
        c2m.main(["verify", str(src), str(ref), str(dst), "--experts", "all"])


def test_verify_catches_a_tensor_the_index_names_but_no_file_holds(ckpt):
    src, ref, dst, _, _ = ckpt
    convert(src, ref, dst)
    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    idx["weight_map"]["model.language_model.layers.9.mlp.gate.weight"] = "model-00001-of-00002.safetensors"
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(SystemExit, match="layers.9.mlp.gate.weight"):
        c2m.main(["verify", str(src), str(ref), str(dst)])


# ── the streaming writer ──────────────────────────────────────────────────────

def test_the_writer_produces_a_safetensors_file_the_library_reads_with_an_8_byte_aligned_header(ckpt):
    src, ref, dst, _, _ = ckpt
    convert(src, ref, dst)
    for fname in ("model-00001-of-00002.safetensors", "model-mtp.safetensors", "model-lmhead-nvfp4.safetensors"):
        with open(dst / fname, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        assert (8 + n) % 8 == 0
        assert header["__metadata__"] == {"format": "pt"}
        offsets = sorted(v["data_offsets"] for k, v in header.items() if k != "__metadata__")
        assert offsets[0][0] == 0 and all(a[1] == b[0] for a, b in zip(offsets, offsets[1:])), "contiguous, no gaps"
        assert os.path.getsize(dst / fname) == 8 + n + offsets[-1][1]
