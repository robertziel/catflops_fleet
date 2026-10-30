#!/usr/bin/env python3
"""Build a copy of an NVFP4 Qwen3.8 checkpoint whose lm_head is dequantized to BF16.

WHY: the DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) ships 81 tensors and NO lm_head and NO
embed_tokens -- it borrows both from the target. Its candidate path runs the target's head
through LogitsProcessor.get_top_k_tokens(), which needs a plain head; nvidia/Qwen3.8-27B-NVFP4
quantizes lm_head to NVFP4 (config group_1, group_size 16), which that path cannot use.
The target's embed_tokens is already BF16, so only lm_head is rewritten.

WHAT IT TOUCHES: in nvidia/Qwen3.8-27B-NVFP4 all four lm_head tensors
(weight, weight_scale, weight_scale_2, input_scale) live in ONE shard,
model-00003-of-00003.safetensors. So only that shard is rewritten. The other files are
hard-linked, which costs no space and is safe because every rewrite here goes to a temporary
file and lands with os.replace() -- a hard link is never written through.

THE SOURCE DIRECTORY IS NEVER WRITTEN. Verify it afterwards with
  deploy/stage-qwen38-27b-nvfp4.sh check

THE DEQUANTIZATION IS vLLM'S OWN, not arithmetic written here:
vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils.dequantize_to_dtype
with swizzle=False (our weight_scale is [out, in/16] = the linear layout, not the 128x4
swizzled one). Each row chunk is then re-checked on the CPU against vLLM's break_fp4_bytes
through an independent scale multiply, and any mismatch is fatal. Run it in the cluster image,
which is where that module lives.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
    dequantize_to_dtype,
)

HEAD_TENSORS = ("lm_head.weight", "lm_head.weight_scale",
                "lm_head.weight_scale_2", "lm_head.input_scale")


def dequantize_head(packed: torch.Tensor, scale: torch.Tensor,
                    global_scale: torch.Tensor, block: int,
                    device: str, rows: int) -> torch.Tensor:
    """packed [out, in/2] uint8 -> bf16 [out, in], in row chunks, vLLM's own kernel."""
    if packed.dtype != torch.uint8:
        raise SystemExit(f"lm_head.weight is {packed.dtype}, expected uint8 (packed NVFP4)")
    out_features, half_in = packed.shape
    in_features = half_in * 2
    if in_features % block:
        raise SystemExit(f"in_features={in_features} is not a multiple of block={block}")
    want_sf = (out_features, in_features // block)
    if tuple(scale.shape) != want_sf:
        raise SystemExit(f"weight_scale shape {tuple(scale.shape)} != {want_sf}: "
                         "this is not the linear (unswizzled) scale layout")
    gs = global_scale.to(torch.float32).reshape(())
    out = torch.empty((out_features, in_features), dtype=torch.bfloat16)
    worst = 0.0
    for start in range(0, out_features, rows):
        stop = min(start + rows, out_features)
        p = packed[start:stop].to(device)
        s = scale[start:stop].to(device)
        chunk = dequantize_to_dtype(p, s, gs.to(device), torch.bfloat16,
                                    block_size=block, swizzle=False)
        out[start:stop] = chunk.to("cpu")
        # Independent CPU recomputation from vLLM's nibble decode: values * sf * global_scale.
        ref = break_fp4_bytes(packed[start:stop], torch.float32)
        ref = ref.reshape(stop - start, in_features // block, block)
        ref = ref * (scale[start:stop].to(torch.float32) * gs).unsqueeze(-1)
        ref = ref.reshape(stop - start, in_features).to(torch.bfloat16)
        bad = int((ref != out[start:stop]).sum())
        if bad:
            diff = (ref.to(torch.float32) - out[start:stop].to(torch.float32)).abs().max()
            worst = max(worst, float(diff))
            raise SystemExit(f"rows {start}:{stop}: {bad} elements disagree with the CPU "
                             f"recomputation, max |diff| {diff:.6g}")
        print(f"  rows {start}:{stop} ok (cross-check exact)", flush=True)
    del worst
    return out


def strip_lm_head_quant(cfg: dict) -> int:
    """Drop lm_head from quantization_config: remove it from every group's `targets` and add it
    to `ignore`. `targets` is what vLLM honours, so an entry left there would re-quantize the
    head whatever `ignore` says."""
    removed = 0
    qc = cfg.get("quantization_config")
    if not qc:
        return 0
    for group in (qc.get("config_groups") or {}).values():
        targets = group.get("targets")
        if not targets:
            continue
        kept = [t for t in targets if t != "lm_head" and not t.endswith(".lm_head")]
        removed += len(targets) - len(kept)
        group["targets"] = kept
    ignore = list(qc.get("ignore") or [])
    if "lm_head" not in ignore:
        ignore.append("lm_head")
    qc["ignore"] = ignore
    ql = qc.get("quantized_layers")
    if isinstance(ql, dict) and ql.pop("lm_head", None) is not None:
        removed += 1
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=16384)
    args = ap.parse_args()

    src, dest = os.path.abspath(args.src), os.path.abspath(args.dest)
    if os.path.realpath(src) == os.path.realpath(dest):
        raise SystemExit("src and dest are the same directory")
    if os.path.exists(dest):
        raise SystemExit(f"{dest} already exists; remove it or pick another name")
    part = dest + ".partial"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)

    index_name = "model.safetensors.index.json"
    with open(os.path.join(src, index_name), encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    present = [t for t in HEAD_TENSORS if t in weight_map]
    if "lm_head.weight" not in present:
        raise SystemExit("no lm_head.weight in the index; nothing to do")
    shards = {weight_map[t] for t in present}
    if len(shards) != 1:
        raise SystemExit(f"lm_head tensors span {sorted(shards)}; this script rewrites one shard")
    head_shard = shards.pop()
    print(f"lm_head lives in {head_shard}; rewriting that shard only", flush=True)

    linked = copied = 0
    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        if not os.path.isfile(s) or name == head_shard:
            continue
        d = os.path.join(part, name)
        try:
            os.link(s, d)
            linked += 1
        except OSError:
            shutil.copy2(s, d)
            copied += 1
    print(f"{linked} files hard-linked, {copied} copied", flush=True)

    src_shard = os.path.join(src, head_shard)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src_shard, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        packed = f.get_tensor("lm_head.weight")
        scale = f.get_tensor("lm_head.weight_scale")
        gscale = f.get_tensor("lm_head.weight_scale_2")
        iscale = f.get_tensor("lm_head.input_scale") if "lm_head.input_scale" in keys else None
        print(f"lm_head.weight {tuple(packed.shape)} {packed.dtype}; "
              f"weight_scale {tuple(scale.shape)} {scale.dtype}; "
              f"weight_scale_2 {tuple(gscale.shape)} {gscale.dtype} = {gscale.item():.8g}; "
              f"input_scale = {None if iscale is None else iscale.item()}", flush=True)
        deq = dequantize_head(packed, scale, gscale, args.block, args.device, args.rows)
        f32 = deq.to(torch.float32)
        print(f"dequantized lm_head.weight -> {tuple(deq.shape)} {deq.dtype}; "
              f"absmax={f32.abs().max().item():.6g} mean={f32.mean().item():.6g} "
              f"std={f32.std().item():.6g} zeros={int((f32 == 0).sum())}", flush=True)
        del packed, scale, gscale, f32
        for key in keys:
            if key in HEAD_TENSORS:
                continue
            tensors[key] = f.get_tensor(key)
    tensors["lm_head.weight"] = deq

    tmp_shard = os.path.join(part, head_shard + ".tmp")
    save_file(tensors, tmp_shard, metadata=metadata or None)
    os.replace(tmp_shard, os.path.join(part, head_shard))
    new_size = os.path.getsize(os.path.join(part, head_shard))
    print(f"wrote {head_shard}: {len(tensors)} tensors, {new_size} bytes "
          f"(was {os.path.getsize(src_shard)})", flush=True)
    del tensors, deq

    for t in HEAD_TENSORS:
        if t != "lm_head.weight":
            weight_map.pop(t, None)
    total = sum(os.path.getsize(os.path.join(part, s)) for s in sorted(set(weight_map.values())))
    index.setdefault("metadata", {})["total_size"] = total
    tmp = os.path.join(part, index_name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, os.path.join(part, index_name))
    print(f"index: {len(weight_map)} tensors, total_size {total}", flush=True)

    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    n = strip_lm_head_quant(cfg)
    tmp = os.path.join(part, "config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(part, "config.json"))
    print(f"config.json: dropped {n} lm_head quant entries, "
          f"ignore={cfg['quantization_config']['ignore']}", flush=True)

    hq_path = os.path.join(src, "hf_quant_config.json")
    if os.path.isfile(hq_path):
        with open(hq_path, encoding="utf-8") as f:
            hq = json.load(f)
        q = hq.get("quantization", hq)
        ex = list(q.get("exclude_modules") or [])
        if "lm_head" not in ex:
            ex.append("lm_head")
        q["exclude_modules"] = ex
        if isinstance(q.get("quantized_layers"), dict):
            q["quantized_layers"].pop("lm_head", None)
        tmp = os.path.join(part, "hf_quant_config.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hq, f, indent=2)
        os.replace(tmp, os.path.join(part, "hf_quant_config.json"))
        print(f"hf_quant_config.json: exclude_modules={ex}", flush=True)

    # The source manifest describes the source bytes, not these: left in place it would make
    # stage-hf-model.sh's verifier call a whole checkpoint broken.
    stale = os.path.join(part, ".hf-manifest.json")
    if os.path.exists(stale):
        os.unlink(stale)
        print("removed .hf-manifest.json (it describes the source, not this copy)", flush=True)

    os.chmod(part, 0o755)
    for name in os.listdir(part):
        p = os.path.join(part, name)
        if os.path.isfile(p):
            os.chmod(p, 0o644)
    os.rename(part, dest)
    print(f"OK {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
