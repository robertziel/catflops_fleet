"""A compressed-tensors Flash-Next checkpoint -> the ModelOpt MIXED_PRECISION layout this fleet already serves.

    python3 ct_to_modelopt.py convert <src: compressed-tensors dir> <ref: the censored ModelOpt dir> <dst>
    python3 ct_to_modelopt.py verify  <src> <ref> <dst> [--experts N|all] [--fp8 N|all]

WHY A CONVERSION AND NOT AN ENGINE PATCH. myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored is the abliterated
twin of the hibrid48 both engines serve, but it ships in compressed-tensors (CT) 0.18 "mixed-precision":
NVFP4 weight-only (W4A16) routed experts and lm_head, FP8 per-channel weight-only GDN / QSA / shared-expert
linears. Our SGLang image has no CT scheme for a W4A16 NVFP4 MoE or linear (`Unsupported FusedMoe scheme`),
and vLLM takes the experts down a Marlin W4A16 path that lost 24 GiB a box at the first warm-up forward
(generations 183/184). The censored checkpoint is the SAME body in ModelOpt MIXED_PRECISION — NVFP4 (W4A4)
experts with per-expert activation scales, W4A16_NVFP4 lm_head — and that layout is what both engines have
served for weeks. So the twin is rewritten INTO that layout, tensor for tensor, and launches through the
identical kernels, config path and tp-rank views as the original. Nothing in either image changes.

WHAT IS EXACT AND WHAT IS NOT — the whole of it, so the catalog note can say the same:
  routed experts, main 48 layers (24,576 of them) and the MTP drafter's 512:
      weight_packed  -> weight          bytes unchanged (same nibble packing, same E2M1 table)
      weight_scale   -> weight_scale    bytes unchanged (F8_E4M3 [N, K/16], linear on disk in both formats)
      weight_global_scale [1] -> weight_scale_2 []  = 1 / global   (CT divides by its global, ModelOpt
                                        multiplies by weight_scale_2: numerically the reciprocal, confirmed on
                                        the real pair — f32(1/global) == weight_scale_2 bit-exact where the
                                        two checkpoints share a projection)
      input_scale    NEW, COPIED from the reference checkpoint's tensor of the SAME expert and projection.
                     The CT checkpoint is weight-only: it has no activation statistics. The ModelOpt MoE path
                     is W4A4 and quantises the expert inputs with a per-expert static scale; the engine fills a
                     missing one with 1.0, which flushes every activation block whose amax is under ~0.012 to
                     zero in the FP8 block scale (e4m3's floor) — silently. The censored checkpoint's scales
                     were calibrated (quantile 0.9999, 128 samples) on the same base model; abliteration edits
                     weights, and the residual stream's magnitudes are unchanged to the precision these scales
                     carry. They are the best measurement that exists for these inputs. They are NOT a
                     calibration of these weights, and the config/README say so.
  MTP drafter experts whose gate and up globals differ (507 of 512 in the real checkpoint; 0 in the main layers):
      every MoE kernel path in both engines that fuses gate|up decodes BOTH halves with the gate's global
      (SGLang flashinfer_cutlass/cutedsl/marlin, vLLM CT and ModelOpt MoE; only TRT-LLM keeps two). So up is
      given the gate's weight_scale_2 and the ratio r = ws2_up/ws2_gate is folded into down_proj:
      ws2_down *= r and input_scale_down /= r. silu(g)*u is linear in u and down is linear with one
      per-expert scalar, so the expert's function is unchanged (the test shows 1e-7 relative). Exact.
  FP8 per-channel linears (300: linear_attn in_proj_qkv/in_proj_z/out_proj x36, self_attn q/k/v/o x12,
  shared_expert gate/up/down x48):
      ModelOpt MIXED_PRECISION has no per-channel FP8 algorithm (FP8 is per-tensor static, FP8_PB_WO is
      128x128 blocks); a per-tensor requant costs 2.6% relative per weight. Dequantising to BF16 costs one
      bf16 rounding (2^-8) and +2.71 GiB whole-model (+1.36 GiB a rank at TP=2) over the FP8 bytes. The
      modules come out as plain BF16 `weight`, their `weight_scale` is DROPPED (a stray scale beside an
      unquantised module trips qwen4_exp's `Expected 1.0` assert), and they are listed under `ignore` —
      the GDN entries the reference announces as W4A16_NVFP4 are removed from `quantized_layers`, the fused
      `in_proj_qkvz` name with them.
  lm_head: as the experts, without input_scale (W4A16_NVFP4 — the reference's head has none either).
  PLE n-gram table: the 8 ple-nvfp4 shards are byte-identical in the two checkpoints (HF LFS sha256), so
      they are HARDLINKED, not copied. Everything else (embeddings, norms, router, hyper-connections, the
      drafter's BF16 parts, tokenizer files) passes through unchanged.
  config.json: the REFERENCE's (the architecture keys both engines already serve — the CT config carries
      transformers-5.16 default fills the base model's config does not), with the source's `mbx_provenance`
      plus one entry for this conversion, and the quantization_config edited exactly as above.

IT STREAMS AND EVICTS. A GB10's 128 GB is one pool for CPU and GPU: 100 GB through the page cache leaves
MemFree at ~1 GiB and the NVIDIA driver then fails allocations for anything serving beside it. Every tensor is
read with pread and dropped (posix_fadvise DONTNEED), every 512 MiB written is fdatasync'd and dropped, and no
shard is ever held in memory: the output header is computed from the input header (sizes are known before a
byte moves) and tensors are written in the source's own order. Peak RSS is one tensor (the 1.27 GB embedding).

IT REFUSES WHAT IT DOES NOT UNDERSTAND. Every tensor must match one rule — expert leaf, lm_head leaf, FP8
weight with its [N,1] BF16 scale, that scale, or a pass-through dtype — or the run stops naming it, before the
destination exists. The reference must carry an input_scale for every expert projection the source has, and
announce exactly the expert layers the source carries. No defaults, no best effort.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import shutil
import struct
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

INDEX = "model.safetensors.index.json"
README = "README-modelopt-conversion.md"
CHUNK = 64 << 20
EVICT_EVERY = 512 << 20
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}
PASS_THROUGH = {"BF16", "F16", "F32", "F64", "I64", "I32", "I16", "I8", "BOOL"}
EXPERT = re.compile(r"^(?P<mod>(?P<expert>.+\.mlp\.experts\.\d+)\.(?P<proj>gate|up|down)_proj)\.(?P<leaf>weight_packed|weight_scale|weight_global_scale)$")
LM_HEAD = re.compile(r"^(?P<mod>lm_head)\.(?P<leaf>weight_packed|weight_scale|weight_global_scale)$")
EXPERT_KEY = re.compile(r"^(?P<layer>(?:mtp\.)?layers\.\d+)\.mlp\.experts\.(?:gate|up|down)_proj$")
PLE_FILE = re.compile(r"^ple-nvfp4-\d+-of-\d+\.safetensors$")
CT_LEAVES = ("weight_packed", "weight_global_scale", "input_scale")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def die(msg: str) -> None:
    raise SystemExit(msg)


# ── safetensors, by hand: headers with pread, tensors as byte ranges, nothing resident ─────────────

class Shard:
    """One safetensors file: its header (name -> dtype/shape/offsets), its metadata and where the data starts."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        fd = os.open(path, os.O_RDONLY)
        try:
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, n, 8))
        finally:
            os.close(fd)
        self.metadata = header.pop("__metadata__", None)
        self.tensors: Dict[str, dict] = header
        self.data_start = 8 + n
        for name, t in header.items():
            a, b = t["data_offsets"]
            want = int(np.prod(t["shape"])) * DTYPE_BYTES[t["dtype"]] if t["shape"] else DTYPE_BYTES[t["dtype"]]
            if b - a != want:
                die(f"{path}: {name}: {t['dtype']} {t['shape']} spans {b - a} bytes, expected {want}")

    def nbytes(self, name: str) -> int:
        a, b = self.tensors[name]["data_offsets"]
        return b - a

    def in_order(self) -> List[str]:
        return sorted(self.tensors, key=lambda k: self.tensors[k]["data_offsets"][0])


class Reader:
    """pread of a shard's tensors, each range evicted from the page cache once it has been read."""

    def __init__(self, shard: Shard):
        self.shard = shard
        self.fd = os.open(shard.path, os.O_RDONLY)

    def chunks(self, name: str) -> Iterator[bytes]:
        a, b = self.shard.tensors[name]["data_offsets"]
        start, pos = self.shard.data_start + a, 0
        while pos < b - a:
            n = min(CHUNK, b - a - pos)
            buf = os.pread(self.fd, n, start + pos)
            if len(buf) != n:
                die(f"{self.shard.path}: short read at {start + pos}: {len(buf)} of {n} bytes ({name})")
            yield buf
            pos += n
        os.posix_fadvise(self.fd, start, b - a, os.POSIX_FADV_DONTNEED)

    def read(self, name: str) -> bytes:
        return b"".join(self.chunks(name))

    def scalar(self, name: str) -> float:
        t = self.shard.tensors[name]
        if t["dtype"] != "F32" or (t["shape"] not in ([], [1])):
            die(f"{self.shard.path}: {name}: expected an F32 scalar ([] or [1]), got {t['dtype']} {t['shape']}")
        return float(np.frombuffer(self.read(name), dtype="<f4")[0])

    def close(self) -> None:
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(self.fd)


class Writer:
    """A safetensors file written front to back: the header first (every size is known from the plan), then
    the tensors in header order, fdatasync'd and evicted every EVICT_EVERY bytes."""

    def __init__(self, path: pathlib.Path, metadata: Optional[dict], entries: Sequence[Tuple[str, str, list, int]]):
        header: Dict[str, object] = {}
        if metadata is not None:
            header["__metadata__"] = metadata
        off = 0
        for name, dtype, shape, nbytes in entries:
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
            off += nbytes
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((8 - (8 + len(raw)) % 8) % 8)
        self.path, self.expected, self.done, self.synced = path, off, 0, 0
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self._put(struct.pack("<Q", len(raw)) + raw)

    def _put(self, data) -> None:
        mv = memoryview(data)
        while mv:
            n = os.write(self.fd, mv)
            mv = mv[n:]

    def write(self, data) -> None:
        self._put(data)
        self.done += len(data)
        if self.done - self.synced >= EVICT_EVERY:
            os.fdatasync(self.fd)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
            self.synced = self.done

    def close(self) -> None:
        os.fsync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(self.fd)
        if self.done != self.expected:
            die(f"{self.path}: wrote {self.done} data bytes, the header promised {self.expected}")


def load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        die(f"{path}: {exc}")


def checkpoint_files(root: pathlib.Path) -> Tuple[dict, Dict[str, Shard]]:
    """The index and a Shard for every file it names; every .safetensors on disk must be named by it."""
    index = load_json(root / INDEX)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        die(f"{root / INDEX}: no weight_map")
    shards: Dict[str, Shard] = {}
    for fname in sorted(set(weight_map.values())):
        if not (root / fname).is_file():
            die(f"{root}: {INDEX} names {fname}, which is not a file there")
        shards[fname] = Shard(root / fname)
    for path in root.glob("*.safetensors"):
        if path.name not in shards:
            die(f"{root}: {path.name} is not named by {INDEX} — not a checkpoint this converter understands")
    for name, fname in weight_map.items():
        if name not in shards[fname].tensors:
            die(f"{root}: {INDEX} names {name} in {fname}, which does not hold it")
    for fname, shard in shards.items():
        for name in shard.tensors:
            if weight_map.get(name) != fname:
                die(f"{root}: {fname} holds {name}, which {INDEX} does not name there")
    return index, shards


# ── dequantisation, for the checks ────────────────────────────────────────────

def fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """U8 [N, K/2] -> float32 [N, K]: element 2i in the low nibble, 2i+1 in the high, sign bit 3, E2M1 magnitude."""
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
    for code, dst in ((lo, out[:, 0::2]), (hi, out[:, 1::2])):
        mag = E2M1[code & 7]
        dst.copy_(torch.where((code >> 3).bool(), -mag, mag))
    return out


def dequant_nvfp4(packed: bytes, shape: list, scales: bytes, global_mult: float) -> torch.Tensor:
    """fp4 * block_scale (F8_E4M3, 16 a block) * global_mult — CT passes 1/weight_global_scale, ModelOpt weight_scale_2."""
    p = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).reshape(shape).copy())
    s = torch.from_numpy(np.frombuffer(scales, dtype=np.uint8).reshape(shape[0], -1).copy()).view(torch.float8_e4m3fn)
    return fp4_values(p) * s.to(torch.float32).repeat_interleave(16, dim=1) * global_mult


def fp8_to_bf16(weight: bytes, shape: list, scale: bytes) -> torch.Tensor:
    """fp8 x bf16 per-channel scale, the product in float32 (exact) then ONE rounding to bf16."""
    w = torch.from_numpy(np.frombuffer(weight, dtype=np.uint8).reshape(shape).copy()).view(torch.float8_e4m3fn)
    s = torch.from_numpy(np.frombuffer(scale, dtype=np.int16).reshape(shape[0], 1).copy()).view(torch.bfloat16)
    return (w.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)


def bf16_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.int16).numpy().tobytes()


def f32_bytes(x: float) -> bytes:
    return struct.pack("<f", x)


# ── the plan: every source tensor classified, every refusal before the destination exists ───────

class Plan:
    def __init__(self, src: pathlib.Path, ref: pathlib.Path):
        self.src, self.ref = src, ref
        self.src_cfg, self.ref_cfg = load_json(src / "config.json"), load_json(ref / "config.json")
        sq = self.src_cfg.get("quantization_config") or {}
        if sq.get("quant_method") != "compressed-tensors":
            die(f"{src / 'config.json'}: quant_method is {sq.get('quant_method')!r}, not compressed-tensors — "
                "this converter only knows the CT mixed-precision layout")
        rq = self.ref_cfg.get("quantization_config") or {}
        if rq.get("quant_method") != "modelopt" or str(rq.get("quant_algo", "")).upper() != "MIXED_PRECISION" \
                or not isinstance(rq.get("quantized_layers"), dict) or not rq["quantized_layers"]:
            die(f"{ref / 'config.json'}: the reference must be a modelopt MIXED_PRECISION checkpoint with "
                f"quantized_layers (got quant_method={rq.get('quant_method')!r}, quant_algo={rq.get('quant_algo')!r})")
        self.index, self.shards = checkpoint_files(src)
        self.ref_index, self.ref_shards = checkpoint_files(ref)
        self.where: Dict[str, str] = dict(self.index["weight_map"])
        self.ple_files = sorted(f for f in self.shards if PLE_FILE.match(f))
        self.classify()
        self.read_scalars()
        self.absorb()
        self.config = derive_config(self.src_cfg, self.ref_cfg, self.fp8_modules, self.expert_layers(),
                                    "lm_head" in self.heads, self.note())

    def announced_layer(self, key: str) -> Optional[str]:
        """The source expert layer a quantized_layers key announces, or None when it is not an expert key."""
        return announced_layer(key, self.ref_cfg, self.expert_layers())

    def entry(self, name: str) -> dict:
        return self.shards[self.where[name]].tensors[name]

    def classify(self) -> None:
        self.ops: Dict[str, Tuple[str, str]] = {}             # tensor -> (op, module)
        self.expert_mods: Dict[str, Dict[str, str]] = {}       # module -> leaf -> tensor
        self.heads: Dict[str, Dict[str, str]] = {}
        self.fp8_modules: List[str] = []
        fp8_weights, bf16_scales = {}, {}
        for name, fname in self.where.items():
            if fname in self.ple_files:
                continue
            t = self.entry(name)
            m = EXPERT.match(name) or LM_HEAD.match(name)
            if m:
                leaf, mod = m.group("leaf"), m.group("mod")
                if leaf == "weight_packed" and not (t["dtype"] == "U8" and len(t["shape"]) == 2):
                    die(f"{name}: expected U8 [N, K/2], got {t['dtype']} {t['shape']}")
                if leaf == "weight_scale" and not (t["dtype"] == "F8_E4M3" and len(t["shape"]) == 2):
                    die(f"{name}: expected F8_E4M3 [N, K/16], got {t['dtype']} {t['shape']}")
                if leaf == "weight_global_scale" and not (t["dtype"] == "F32" and t["shape"] in ([], [1])):
                    die(f"{name}: expected an F32 scalar, got {t['dtype']} {t['shape']}")
                (self.expert_mods if m.re is EXPERT else self.heads).setdefault(mod, {})[leaf] = name
                self.ops[name] = ("expert" if m.re is EXPERT else "head", mod)
                continue
            if name.endswith(CT_LEAVES):
                die(f"{name}: a {name.rsplit('.', 1)[-1]} tensor outside the routed experts and lm_head — no rule "
                    "converts it (and an input_scale in the source would be overwritten by the reference's)")
            if name.endswith(".weight") and t["dtype"] == "F8_E4M3":
                fp8_weights[name[:-len(".weight")]] = name
                continue
            if name.endswith(".weight_scale") and t["dtype"] == "BF16":
                bf16_scales[name[:-len(".weight_scale")]] = name
                continue
            if t["dtype"] not in PASS_THROUGH:
                die(f"{name}: {t['dtype']} {t['shape']} matches no rule — not an expert/lm_head leaf, not an "
                    "FP8-channel weight, not a pass-through dtype")
            self.ops[name] = ("copy", "")
        for mod, wname in sorted(fp8_weights.items()):
            w = self.entry(wname)
            if mod not in bf16_scales:
                die(f"{wname}: F8_E4M3 weight with no BF16 [N,1] weight_scale beside it (not an FP8-channel linear)")
            s = self.entry(bf16_scales[mod])
            if len(w["shape"]) != 2 or s["shape"] != [w["shape"][0], 1]:
                die(f"{mod}.weight_scale: FP8 weight {w['shape']} needs a BF16 [N,1] per-channel scale, got "
                    f"{s['dtype']} {s['shape']}")
            self.ops[wname] = ("fp8", mod)
            self.ops[bf16_scales[mod]] = ("drop", mod)
            self.fp8_modules.append(mod)
        for mod, sname in bf16_scales.items():
            if mod not in fp8_weights:
                die(f"{sname}: a weight_scale whose weight is not FP8 — nothing to apply it to")
        for mod, leaves in list(self.expert_mods.items()) + list(self.heads.items()):
            missing = [l for l in ("weight_packed", "weight_scale", "weight_global_scale") if l not in leaves]
            if missing:
                die(f"{mod}: missing {', '.join(missing)}")
            packed, scale = self.entry(leaves["weight_packed"])["shape"], self.entry(leaves["weight_scale"])["shape"]
            if scale != [packed[0], packed[1] * 2 // 16]:
                die(f"{mod}: weight_scale {scale} does not block weight_packed {packed} in 16s")
        self.experts: Dict[str, Dict[str, str]] = {}           # expert prefix -> proj -> module
        for mod in self.expert_mods:
            m = EXPERT.match(mod + ".weight_packed")
            self.experts.setdefault(m.group("expert"), {})[m.group("proj")] = mod
        for expert, projs in self.experts.items():
            if set(projs) != {"gate", "up", "down"}:
                die(f"{expert}: has {sorted(projs)} of gate/up/down")
        if set(self.heads) - {"lm_head"}:
            die(f"unexpected head modules: {sorted(set(self.heads) - {'lm_head'})}")

    def read_scalars(self) -> None:
        """Every global (source) and every expert input_scale (reference), 4 bytes each, before anything is written."""
        self.gs: Dict[str, float] = {}
        by_file: Dict[str, List[str]] = {}
        for mod, leaves in list(self.expert_mods.items()) + list(self.heads.items()):
            by_file.setdefault(self.where[leaves["weight_global_scale"]], []).append(mod)
        for fname, mods in by_file.items():
            r = Reader(self.shards[fname])
            for mod in mods:
                self.gs[mod] = r.scalar(self.expert_mods.get(mod, self.heads.get(mod))["weight_global_scale"])
                if not (self.gs[mod] > 0 and np.isfinite(self.gs[mod])):
                    die(f"{mod}.weight_global_scale = {self.gs[mod]}: not a positive finite scale")
            r.close()
        self.input_scale: Dict[str, float] = {}
        ref_map = self.ref_index["weight_map"]
        by_file = {}
        for mod in self.expert_mods:
            name = f"{mod}.input_scale"
            if name not in ref_map:
                die(f"{self.ref}: no {name} — the reference must carry an activation scale for every expert "
                    "projection the source has; nothing is invented in its place")
            by_file.setdefault(ref_map[name], []).append(mod)
        for fname, mods in by_file.items():
            r = Reader(self.ref_shards[fname])
            for mod in mods:
                self.input_scale[mod] = r.scalar(f"{mod}.input_scale")
                if not (self.input_scale[mod] > 0 and np.isfinite(self.input_scale[mod])):
                    die(f"{self.ref}: {mod}.input_scale = {self.input_scale[mod]}: not a positive finite scale")
            r.close()

    def absorb(self) -> None:
        """Per expert: weight_scale_2 and input_scale as written — up takes gate's global, down absorbs the ratio."""
        self.ws2: Dict[str, float] = {}
        self.isc: Dict[str, float] = {}
        self.absorbed: Dict[str, float] = {}
        for mod in self.heads:
            self.ws2[mod] = float(np.float32(1.0 / self.gs[mod]))
        for expert, projs in self.experts.items():
            g, u, d = (self.gs[projs[p]] for p in ("gate", "up", "down"))
            ws2_g = float(np.float32(1.0 / g))
            self.ws2[projs["gate"]] = ws2_g
            self.isc[projs["gate"]] = self.input_scale[projs["gate"]]
            self.isc[projs["up"]] = self.input_scale[projs["up"]]
            if g == u:
                self.ws2[projs["up"]] = ws2_g
                self.ws2[projs["down"]] = float(np.float32(1.0 / d))
                self.isc[projs["down"]] = self.input_scale[projs["down"]]
            else:
                r = g / u                                      # = ws2_up / ws2_gate
                self.absorbed[expert] = r
                self.ws2[projs["up"]] = ws2_g
                self.ws2[projs["down"]] = float(np.float32((1.0 / d) * r))
                self.isc[projs["down"]] = float(np.float32(self.input_scale[projs["down"]] / r))

    def expert_layers(self) -> List[str]:
        return sorted({canonical(e.rsplit(".mlp.experts.", 1)[0]) for e in self.experts})

    def note(self) -> dict:
        return {"tool": "ct_to_modelopt.py", "from": self.src.name, "input_scale_from": self.ref.name,
                "date": _dt.date.today().isoformat(),
                "experts": "compressed-tensors NVFP4 W4A16 -> modelopt NVFP4 (W4A4): weight_packed->weight, "
                           "weight_scale kept, weight_scale_2 = 1/weight_global_scale, input_scale COPIED from "
                           f"{self.ref.name}'s tensor of the same expert and projection (NOT calibrated on these weights)",
                "routed_expert_modules": len(self.expert_mods),
                "absorbed_experts": len(self.absorbed),
                "absorbed": "experts whose up global != gate global: up takes gate's weight_scale_2, down_proj's "
                            "weight_scale_2 *= r and input_scale /= r, r = ws2_up/ws2_gate (exact for SwiGLU)",
                "absorbed_ratio_range": [min(self.absorbed.values()), max(self.absorbed.values())] if self.absorbed else None,
                "fp8_channel_to_bf16": len(self.fp8_modules),
                "fp8": "FP8 E4M3 x BF16 [N,1] per-channel scale -> BF16 weight (one bf16 rounding), weight_scale dropped, "
                       "module moved to ignore (modelopt has no per-channel FP8 algorithm)",
                "lm_head": "W4A16_NVFP4, weight_scale_2 = 1/weight_global_scale, no input_scale",
                "ple": "ple-nvfp4 shards hardlinked (byte-identical to the reference's)"}

    def output_entries(self, fname: str) -> List[Tuple[str, str, list, int, Tuple[str, str]]]:
        """(name, dtype, shape, nbytes, (op, source tensor)) for one shard, in the source's byte order."""
        out = []
        shard = self.shards[fname]
        for name in shard.in_order():
            t = shard.tensors[name]
            op, mod = self.ops[name]
            if op == "copy":
                out.append((name, t["dtype"], t["shape"], shard.nbytes(name), ("copy", name)))
            elif op in ("expert", "head"):
                leaf = name.rsplit(".", 1)[-1]
                if leaf == "weight_packed":
                    out.append((f"{mod}.weight", "U8", t["shape"], shard.nbytes(name), ("copy", name)))
                elif leaf == "weight_scale":
                    out.append((name, "F8_E4M3", t["shape"], shard.nbytes(name), ("copy", name)))
                else:
                    out.append((f"{mod}.weight_scale_2", "F32", [], 4, ("ws2", mod)))
                    if op == "expert":
                        out.append((f"{mod}.input_scale", "F32", [], 4, ("isc", mod)))
            elif op == "fp8":
                out.append((name, "BF16", t["shape"], 2 * shard.nbytes(name), ("fp8", mod)))
            elif op == "drop":
                continue
            else:
                die(f"{name}: unknown op {op}")
        return out


def canonical(key: str) -> str:
    """A quantized_layers / ignore key without its alias prefix: model.language_model.X, language_model.model.X,
    model.X -> X; mtp.model.X, model.mtp.X -> mtp.X."""
    parts = key.split(".")
    while parts and parts[0] in ("model", "language_model"):
        parts.pop(0)
    if len(parts) > 1 and parts[0] == "mtp" and parts[1] == "model":
        del parts[1]
    return ".".join(parts)


def announced_layer(key: str, ref_cfg: dict, expert_layers: Sequence[str]) -> Optional[str]:
    """The expert layer a quantized_layers key announces, in the source's naming — `layers.N` and `mtp.layers.0`.
    The drafter is announced under its own index 0 AND under num_hidden_layers (`mtp.layers.48`, `layers.48`:
    the index engines that count it after the body use); both mean the one drafter layer the source has."""
    m = EXPERT_KEY.match(canonical(key))
    if not m:
        return None
    layer = m.group("layer")
    n = (ref_cfg.get("text_config") or ref_cfg).get("num_hidden_layers")
    if n is not None and layer in (f"mtp.layers.{n}", f"layers.{n}") and "mtp.layers.0" in expert_layers:
        return "mtp.layers.0"
    return layer


def derive_config(src_cfg: dict, ref_cfg: dict, fp8_modules: Sequence[str], expert_layers: Sequence[str],
                  has_lm_head: bool, note: dict) -> dict:
    cfg = json.loads(json.dumps(ref_cfg))
    q = cfg["quantization_config"]
    bf16 = {canonical(m) for m in fp8_modules}
    fused = set()
    for mod in sorted(bf16):
        if mod.endswith(".linear_attn.in_proj_qkv") or mod.endswith(".linear_attn.in_proj_z"):
            base = mod.rsplit(".", 1)[0]
            if {f"{base}.in_proj_qkv", f"{base}.in_proj_z"} <= bf16:
                fused.add(f"{base}.in_proj_qkvz")
            else:
                die(f"{mod}: dequantised to BF16 while its partner in the fused in_proj_qkvz is not — half a fused "
                    "module cannot be excluded")
    removed = {k for k in q["quantized_layers"] if canonical(k) in bf16 | fused}
    kept = {k: v for k, v in q["quantized_layers"].items() if k not in removed}
    announced = set()
    for key in kept:
        c = canonical(key)
        layer = announced_layer(key, ref_cfg, expert_layers)
        if layer:
            if layer not in expert_layers:
                die(f"{key}: the reference announces NVFP4 experts in {layer}, which the source does not "
                    "carry as packed NVFP4 experts")
            announced.add(layer)
        elif c == "lm_head":
            if not has_lm_head:
                die(f"{key}: the reference announces an NVFP4 lm_head, which the source does not carry")
        elif c == "mtp.lm_head":
            if not has_lm_head:
                die(f"{key}: mtp.lm_head is shared from the target's lm_head, which the source does not carry")
        else:
            die(f"{key}: the reference announces {kept[key].get('quant_algo')!r} for a module the conversion did not "
                "produce as NVFP4 — nothing in the source corresponds to it")
    for layer in expert_layers:
        if layer not in announced:
            die(f"{layer}: the source carries packed NVFP4 experts here and the reference's quantized_layers does "
                "not announce them — the engines would build that MoE unquantised")
    ignore = list(q.get("ignore") or [])
    seen = set(ignore)
    for mod in sorted(fp8_modules):
        for name in (mod, mod.rsplit(".", 1)[0] + ".in_proj_qkvz" if canonical(mod).rsplit(".", 1)[0] + ".in_proj_qkvz" in fused else None):
            if name and name not in seen:
                ignore.append(name)
                seen.add(name)
    q["quantized_layers"] = kept
    q["ignore"] = ignore
    cfg["mbx_provenance"] = list(src_cfg.get("mbx_provenance") or []) + [note]
    return cfg


# ── convert ───────────────────────────────────────────────────────────────────

def meminfo_gib(key: str = "MemFree") -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        return None
    return None


def log(msg: str) -> None:
    free = meminfo_gib()
    print(f"[{_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')}] {msg}" + (f"  (MemFree {free:.1f} GiB)" if free is not None else ""), flush=True)


def convert(src: pathlib.Path, ref: pathlib.Path, dst: pathlib.Path) -> int:
    for p in (dst, dst.parent / f"{dst.name}.partial"):
        if p.exists() or p.is_symlink():
            die(f"{p} exists — this converter never writes over a checkpoint; remove it first")
    plan = Plan(src, ref)
    log(f"plan: {len(plan.expert_mods)} expert modules in {len(plan.experts)} experts over {len(plan.expert_layers())} layers, "
        f"{len(plan.absorbed)} absorbed, {len(plan.fp8_modules)} FP8-channel -> BF16, {len(plan.ple_files)} PLE shards linked")
    partial = dst.parent / f"{dst.name}.partial"
    partial.mkdir(mode=0o755)
    try:
        weight_map: Dict[str, str] = {}
        total = 0
        for fname in plan.ple_files:
            os.link(src / fname, partial / fname)
            for name in plan.shards[fname].tensors:
                weight_map[name] = fname
                total += plan.shards[fname].nbytes(name)
        readers: Dict[str, Reader] = {}          # an FP8 weight's scale may sit in ANOTHER shard (layer 35 q_proj does)
        for fname in sorted(f for f in plan.shards if f not in plan.ple_files):
            entries = plan.output_entries(fname)
            reader = readers.setdefault(fname, Reader(plan.shards[fname]))
            writer = Writer(partial / fname, plan.shards[fname].metadata, [e[:4] for e in entries])
            for name, dtype, shape, nbytes, (op, arg) in entries:
                if op == "copy":
                    for chunk in reader.chunks(arg):
                        writer.write(chunk)
                elif op == "ws2":
                    writer.write(f32_bytes(plan.ws2[arg]))
                elif op == "isc":
                    writer.write(f32_bytes(plan.isc[arg]))
                elif op == "fp8":
                    sfile = plan.where[f"{arg}.weight_scale"]
                    scale = readers.setdefault(sfile, Reader(plan.shards[sfile])).read(f"{arg}.weight_scale")
                    writer.write(bf16_bytes(fp8_to_bf16(reader.read(f"{arg}.weight"), shape, scale)))
                weight_map[name] = fname
                total += nbytes
            writer.close()
            log(f"{fname}: {len(entries)} tensors, {writer.done / 2**30:.2f} GiB")
        for r in readers.values():
            r.close()
        for path in sorted(src.iterdir()):
            if path.is_file() and not path.name.endswith(".safetensors") and path.name not in (INDEX, "config.json", README):
                shutil.copy2(path, partial / path.name)
        index = {"metadata": dict(plan.index.get("metadata") or {}), "weight_map": weight_map}
        index["metadata"]["total_size"] = total
        (partial / INDEX).write_text(json.dumps(index, indent=2, sort_keys=True))
        (partial / "config.json").write_text(json.dumps(plan.config, indent=2, sort_keys=True))
        (partial / README).write_text(readme(plan))
        for p in partial.iterdir():
            os.chmod(p, 0o644)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    os.rename(partial, dst)
    log(f"done: {dst} ({total / 2**30:.2f} GiB of tensors, {len(weight_map)} in the index)")
    return 0


def readme(plan: Plan) -> str:
    n = plan.note()
    rng = n["absorbed_ratio_range"]
    return f"""# ModelOpt conversion of {plan.src.name}

Converted by `deploy/sglang-27b-tp2/ct_to_modelopt.py` (catflops_fleet) on {n['date']}, from the compressed-tensors
checkpoint `{plan.src.name}` with `{plan.ref.name}` as the reference ModelOpt checkpoint.
The weights are the publisher's; only their encoding changed, except where stated.

| what | how | exact? |
|---|---|---|
| routed experts ({n['routed_expert_modules']} modules, {len(plan.experts)} experts, {len(plan.expert_layers())} layers) | weight_packed -> weight, weight_scale kept, weight_scale_2 = 1 / weight_global_scale | yes (float32 reciprocal) |
| expert `input_scale` | COPIED from `{plan.ref.name}`'s tensor of the same expert and projection | NO — a calibration of the censored weights, not these; the CT checkpoint is weight-only and has no activation statistics |
| {n['absorbed_experts']} experts with up global != gate global{' (ratio ' + f'{rng[0]:.4f}..{rng[1]:.4f}' + ')' if rng else ''} | up takes gate's weight_scale_2; down_proj weight_scale_2 *= r, input_scale /= r | yes (SwiGLU is linear in up; down is linear) |
| {n['fp8_channel_to_bf16']} FP8 per-channel linears (GDN in_proj_qkv/in_proj_z/out_proj, QSA q/k/v/o, shared_expert) | fp8 x bf16 scale in float32, rounded once to BF16; scale dropped; module under `ignore` | one bf16 rounding (2^-8); +2.71 GiB whole-model |
| lm_head | as the experts, no input_scale (W4A16_NVFP4) | yes |
| PLE n-gram table ({len(plan.ple_files)} shards) | hardlinked, byte-identical to the reference's | yes |
| everything else | copied | yes |

`config.json` is the reference's (the architecture keys both engines already serve) with this checkpoint's
`mbx_provenance` plus one entry for the conversion, the GDN modules moved from `quantized_layers` to `ignore`,
and nothing else changed. Verify with `ct_to_modelopt.py verify <src> <ref> <this dir>`.

NOT CALIBRATION-VERIFIED: nothing here was measured on a running engine when it was written.
"""


# ── verify ────────────────────────────────────────────────────────────────────

def sample(items: Sequence, n) -> List:
    if n == "all" or len(items) <= int(n):
        return list(items)
    n = int(n)
    return [items[round(i * (len(items) - 1) / max(n - 1, 1))] for i in range(n)]


def verify(src: pathlib.Path, ref: pathlib.Path, dst: pathlib.Path, experts="4", fp8="8") -> int:
    plan = Plan(src, ref)
    index, shards = checkpoint_files(dst)
    cfg = load_json(dst / "config.json")
    problems: List[str] = []
    total = sum(sh.nbytes(n) for sh in shards.values() for n in sh.tensors)
    if index.get("metadata", {}).get("total_size") != total:
        problems.append(f"index total_size {index.get('metadata', {}).get('total_size')} != {total} bytes of tensors")
    for fname, sh in shards.items():
        for name, t in sh.tensors.items():
            if name.endswith(CT_LEAVES) and not (name.endswith(".input_scale") and EXPERT.match(name.replace(".input_scale", ".weight_scale"))):
                problems.append(f"{name}: a compressed-tensors leaf survived")
            if t["dtype"] == "F8_E4M3" and not (name.endswith(".weight_scale") and (EXPERT.match(name) or LM_HEAD.match(name)) or (fname in plan.ple_files)):
                problems.append(f"{name}: F8_E4M3 outside an NVFP4 block scale")
            if t["dtype"] == "U8" and not ((name.endswith(".weight") and (EXPERT.match(name + "_packed") or LM_HEAD.match(name + "_packed"))) or fname in plan.ple_files):
                problems.append(f"{name}: U8 outside a packed NVFP4 weight")
            if t["dtype"] == "BF16" and name.endswith(".weight_scale"):
                problems.append(f"{name}: a BF16 per-channel scale survived")
    if problems:
        die("verify: " + "; ".join(problems[:20]))
    q = cfg.get("quantization_config") or {}
    if q.get("quant_method") != "modelopt" or str(q.get("quant_algo", "")).upper() != "MIXED_PRECISION":
        die(f"verify: config quantization_config is {q.get('quant_method')}/{q.get('quant_algo')}, not modelopt MIXED_PRECISION")
    names = {n for sh in shards.values() for n in sh.tensors}
    for key, info in q["quantized_layers"].items():
        c = canonical(key)
        layer = plan.announced_layer(key)
        if layer:
            probe = next((n for n in names if canonical(n).startswith(layer + ".mlp.experts.") and n.endswith(".weight_scale_2")), None)
            if probe is None:
                die(f"verify: quantized_layers {key} ({info.get('quant_algo')}) but no expert weight_scale_2 in {layer}")
        elif c == "lm_head" and "lm_head.weight_scale_2" not in names:
            die(f"verify: quantized_layers {key} but no lm_head.weight_scale_2")
        elif c not in ("lm_head", "mtp.lm_head"):
            die(f"verify: quantized_layers announces {key} ({info.get('quant_algo')}), a module the checkpoint holds as plain tensors")
    ignore = set(q.get("ignore") or [])
    for mod in plan.fp8_modules:
        if mod not in ignore:
            die(f"verify: {mod} came out BF16 but is not under ignore")
        if f"{mod}.weight" not in names or shards[index["weight_map"][f"{mod}.weight"]].tensors[f"{mod}.weight"]["dtype"] != "BF16":
            die(f"verify: {mod}.weight is not a BF16 tensor in the destination")
    # numerics: experts (all absorbed ones, a sample of the rest), FP8 linears, lm_head
    readers: Dict[Tuple[str, str], Reader] = {}

    def rd(root: pathlib.Path, where: Dict[str, str], shs: Dict[str, Shard], name: str) -> Reader:
        key = (str(root), where[name])
        if key not in readers:
            readers[key] = Reader(shs[where[name]])
        return readers[key]

    def src_read(name):
        return rd(src, plan.where, plan.shards, name).read(name)

    def dst_read(name):
        return rd(dst, index["weight_map"], shards, name).read(name)

    def dst_scalar(name):
        return rd(dst, index["weight_map"], shards, name).scalar(name)

    by_layer: Dict[str, List[str]] = {}
    for expert in sorted(plan.experts, key=lambda e: (e.rsplit(".", 1)[0], int(e.rsplit(".", 1)[1]))):
        by_layer.setdefault(expert.rsplit(".mlp.experts.", 1)[0], []).append(expert)
    chosen = set(plan.absorbed)
    for layer, es in by_layer.items():
        chosen.update(sample(es, experts))
    worst_dq = worst_fn = worst_is = 0.0
    x = torch.randn(4, 1, generator=torch.Generator().manual_seed(1))
    for expert in sorted(chosen):
        projs = plan.experts[expert]
        dq_src, dq_dst = {}, {}
        for proj, mod in projs.items():
            shape = plan.entry(plan.expert_mods[mod]["weight_packed"])["shape"]
            dq_src[proj] = dequant_nvfp4(src_read(f"{mod}.weight_packed"), shape, src_read(f"{mod}.weight_scale"), 1.0 / plan.gs[mod])
            ws2 = dst_scalar(f"{mod}.weight_scale_2")
            if proj == "up":
                ws2 = dst_scalar(f"{projs['gate']}.weight_scale_2")          # what a one-global kernel applies
            dq_dst[proj] = dequant_nvfp4(dst_read(f"{mod}.weight"), shape, dst_read(f"{mod}.weight_scale"), ws2)
            got_is = dst_scalar(f"{mod}.input_scale")
            want_is = plan.input_scale[mod] / (plan.absorbed.get(expert, 1.0) if proj == "down" else 1.0)
            worst_is = max(worst_is, abs(got_is - want_is) / want_is)
            if abs(got_is - want_is) / want_is > 1e-6:
                die(f"verify: {mod}.input_scale = {got_is}, expected {want_is} (reference{'/r' if proj == 'down' and expert in plan.absorbed else ''})")
        if expert in plan.absorbed:
            xx = torch.randn(5, dq_src["gate"].shape[1], generator=torch.Generator().manual_seed(2))
            want = (torch.nn.functional.silu(xx @ dq_src["gate"].T) * (xx @ dq_src["up"].T)) @ dq_src["down"].T
            got = (torch.nn.functional.silu(xx @ dq_dst["gate"].T) * (xx @ dq_dst["up"].T)) @ dq_dst["down"].T
            err = float(((got - want).norm() / want.norm().clamp_min(1e-30)).item())
            worst_fn = max(worst_fn, err)
            if err > 1e-5:
                die(f"verify: {expert}: absorbed expert function differs by {err:.3e} relative")
        else:
            for proj in projs:
                err = float(((dq_dst[proj] - dq_src[proj]).abs().max() / dq_src[proj].abs().max().clamp_min(1e-30)).item())
                worst_dq = max(worst_dq, err)
                if err > 1e-6:
                    die(f"verify: {projs[proj]}: dequantised weights differ by {err:.3e} of the max")
    fp8_worst = 0
    chosen_fp8 = sample(plan.fp8_modules, fp8)
    for mod in chosen_fp8:
        shape = plan.entry(f"{mod}.weight")["shape"]
        want = fp8_to_bf16(src_read(f"{mod}.weight"), shape, src_read(f"{mod}.weight_scale"))
        got = torch.from_numpy(np.frombuffer(dst_read(f"{mod}.weight"), dtype=np.int16).reshape(shape).copy()).view(torch.bfloat16)
        bad = int((got != want).sum().item())
        fp8_worst = max(fp8_worst, bad)
        if bad:
            die(f"verify: {mod}.weight: {bad} BF16 elements differ from fp8 x scale")
    head_err = None
    if "lm_head" in plan.heads:
        shape = plan.entry("lm_head.weight_packed")["shape"]
        rows = min(shape[0], 2048)
        a = dequant_nvfp4(src_read("lm_head.weight_packed")[: rows * shape[1]], [rows, shape[1]], src_read("lm_head.weight_scale")[: rows * (shape[1] * 2 // 16)], 1.0 / plan.gs["lm_head"])
        b = dequant_nvfp4(dst_read("lm_head.weight")[: rows * shape[1]], [rows, shape[1]], dst_read("lm_head.weight_scale")[: rows * (shape[1] * 2 // 16)], dst_scalar("lm_head.weight_scale_2"))
        head_err = float(((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item())
        if head_err > 1e-6:
            die(f"verify: lm_head dequantised weights differ by {head_err:.3e} of the max")
    for r in readers.values():
        r.close()
    for fname in plan.ple_files:
        if not (dst / fname).is_file() or os.path.getsize(dst / fname) != os.path.getsize(src / fname):
            die(f"verify: {fname}: missing or not the source's size in the destination")
    print(f"verify OK: {len(names)} tensors in {len(shards)} files, {total / 2**30:.2f} GiB; "
          f"{len(chosen)} experts checked ({len(plan.absorbed)} absorbed, worst function err {worst_fn:.2e}; "
          f"worst dequant err {worst_dq:.2e}; worst input_scale err {worst_is:.2e}); "
          f"{len(chosen_fp8)} of {len(plan.fp8_modules)} fp8-channel modules bit-exact as BF16; "
          f"lm_head err {head_err if head_err is None else f'{head_err:.2e}'}; "
          f"PLE shards {'hardlinked' if all(os.path.samefile(dst / f, src / f) for f in plan.ple_files) else 'copied'}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("convert", "verify"):
        p = sub.add_parser(cmd)
        p.add_argument("src", type=pathlib.Path)
        p.add_argument("ref", type=pathlib.Path)
        p.add_argument("dst", type=pathlib.Path)
        if cmd == "verify":
            p.add_argument("--experts", default="4", help="experts checked per layer (N or all); absorbed ones are always all checked")
            p.add_argument("--fp8", default="8", help="FP8-channel modules checked (N or all)")
    args = ap.parse_args(argv)
    if args.cmd == "convert":
        return convert(args.src, args.ref, args.dst)
    return verify(args.src, args.ref, args.dst, args.experts, args.fp8)


if __name__ == "__main__":
    sys.exit(main())
