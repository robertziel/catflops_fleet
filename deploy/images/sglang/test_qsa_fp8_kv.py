"""Fleet patch qsa-fp8-kv: Flash-Next (Qwen3.8 QSA) on SGLang with an FP8 KV pool. The SM121 QSA decode kernel and the
chunk-prefill kernel read BF16 K/V only ("unsupported SM121 QSA call: expected BF16 …", generation 153), so the rows
gathered from an FP8 pool are widened to BF16 — times the layer's KV scale, which the pool divided by on write.
Operator 2026-10-26: "implement FP8 there". CPU-only, like the image's other build-gate tests."""
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa


def test_an_fp8_pool_is_gathered_as_bf16_and_any_other_as_itself():
    assert qsa._qsa_gather_dtype(torch.float8_e4m3fn) is torch.bfloat16
    assert qsa._qsa_gather_dtype(torch.float8_e5m2) is torch.bfloat16
    assert qsa._qsa_gather_dtype(torch.bfloat16) is torch.bfloat16
    assert qsa._qsa_gather_dtype(torch.float16) is torch.float16


def test_fp8_rows_widen_to_bf16_with_the_scale_multiplied_back():
    values = torch.tensor([[0.5, -1.0, 2.0, 448.0]], dtype=torch.float32)   # all exact in e4m3
    stored = (values / 2.0).to(torch.float8_e4m3fn)                         # the pool divided by k_scale 2.0
    out = qsa._qsa_widen(stored, 2.0)
    assert out.dtype is torch.bfloat16 and torch.equal(out.float(), values)
    plain = qsa._qsa_widen(values.to(torch.float8_e4m3fn), None)
    assert plain.dtype is torch.bfloat16 and torch.equal(plain.float(), values)


def test_a_bf16_pool_is_passed_through_untouched():
    t = torch.ones(2, 3, dtype=torch.bfloat16)
    assert qsa._qsa_widen(t, 2.0) is t, "no copy, no scale: the path before the patch"
