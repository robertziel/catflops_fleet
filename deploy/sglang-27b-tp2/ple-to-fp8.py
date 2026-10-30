"""nvfp4 PLE shards -> the fp8 shard layout SGLang's own loader accepts.

WHY. The checkpoint stores the n-gram table as 8 x {packed U8 [R,80], scales F8_E4M3 [R,10]}
plus one F32 global - a layout only the MBX vLLM overlay reads. SGLang's qwen4_exp has no
case for it, so every one of those tensors falls through load_weights' final else and is
SKIPPED with a warning; the table then stays at its bf16 allocation, 95.37 GiB, which is
what took both GB10 boxes down. SGLang DOES have a sharded PLE path, for tensors named
    <prefix>.ngram_embedding.shard_<i>.weight
and it auto-switches the table's storage to fp8 when those arrive as float8_e4m3fn.

SHARD COUNT IS NOT OURS TO CHOOSE. The loader places shard i at
    shard_size = ceil(org_vocab_size / text_config.split_ngram_parts); start = i * shard_size
and split_ngram_parts is 128 here, so 2,500,012 rows per shard. 8 source shards of
40,000,192 rows each therefore split into exactly 16 outputs apiece (40,000,192 = 16 x
2,500,012), 128 in all. Emitting 8 would silently scatter them across the wrong rows.

SCALE. One per-tensor scale for the whole table, into the weight_scale buffer the gather
already multiplies by (qwen4_exp.py: embeddings = embeddings * ngram_embedding.weight_scale).
Without it the values, all <= 0.051, fall into fp8_e4m3's subnormal floor and small ones
flush to zero. With it, measured on 32M real values: 2.46% mean / 4.76% worst relative
error - exactly fp8's half-ULP bound - and nothing flushed.
"""
import glob, json, os, struct, sys, time
import numpy as np, torch
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
PARTS, FP8_MAX, LUT = 128, 448.0, (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
lut = torch.tensor(LUT, dtype=torch.float32)

def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n

def read(path, hdr, base, key, rows=None):
    t = hdr[key]; a, b = t["data_offsets"]; shape = t["shape"]
    nb = {"U8": 1, "F8_E4M3": 1, "F32": 4}[t["dtype"]]
    if rows is not None and shape:
        per = int(np.prod(shape[1:])) * nb
        a += rows[0] * per; b = a + (rows[1] - rows[0]) * per
        shape = [rows[1] - rows[0]] + shape[1:]
    with open(path, "rb") as fh:
        fh.seek(base + a); raw = fh.read(b - a)
    dt = np.float32 if t["dtype"] == "F32" else np.uint8
    arr = np.frombuffer(raw, dtype=dt).reshape(shape if shape else ())
    out = torch.from_numpy(arr.copy())
    return out.view(torch.float8_e4m3fn) if t["dtype"] == "F8_E4M3" else out

files = sorted(glob.glob(os.path.join(SRC, "ple-nvfp4-*.safetensors")))
assert files, f"no ple-nvfp4-*.safetensors in {SRC}"
metas = [header(f) for f in files]
PREFIX = next(k for k, _ in metas[0][0].items() if k.endswith("nvfp4_global")).rsplit(".", 1)[0]
print(f"{len(files)} source shards, prefix {PREFIX}", flush=True)

# PASS 1 - one scale for the whole table. max|v| = max_lut(6.0) * max block scale * global.
gl = None; smax = 0.0
for f, (h, b) in zip(files, metas):
    for k in h:
        if k.endswith("nvfp4_global"): gl = float(read(f, h, b, k))
        if k.endswith(".scales"): smax = max(smax, float(read(f, h, b, k).float().abs().max()))
maxabs = 6.0 * smax * gl
scale = maxabs / FP8_MAX
print(f"global={gl:.8g} max_block_scale={smax:.8g} -> max|v|={maxabs:.8g} scale={scale:.8g}", flush=True)

os.makedirs(DST, exist_ok=True)
rows_total = 0
for si, (f, (h, b)) in enumerate(zip(files, metas)):
    pk = next(k for k in h if k.endswith(".packed")); sk = next(k for k in h if k.endswith(".scales"))
    R = h[pk]["shape"][0]
    assert R % 16 == 0, R
    sub = R // 16
    for k in range(16):
        idx = si * 16 + k
        r0, r1 = k * sub, (k + 1) * sub
        packed = read(f, h, b, pk, (r0, r1)); scl = read(f, h, b, sk, (r0, r1))
        lo, hi = packed & 0xF, (packed >> 4) & 0xF
        codes = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1).long()
        sign = torch.where(((codes >> 3) & 1) == 1, -1.0, 1.0)
        vals = sign * lut[codes & 0x7] * scl.float().repeat_interleave(16, 1) * gl
        q = (vals / scale).to(torch.float8_e4m3fn).contiguous()
        tensors = {f"{PREFIX}.shard_{idx}.weight": q}
        if idx == 0:
            tensors[f"{PREFIX}.weight_scale"] = torch.tensor([scale], dtype=torch.bfloat16)
        save_file(tensors, os.path.join(DST, f"ple-fp8-{idx:05d}-of-{PARTS:05d}.safetensors"))
        rows_total += q.shape[0]
        if idx % 8 == 0 or idx == PARTS - 1:
            print(f"  shard {idx+1}/{PARTS}  rows {rows_total:,}  {time.strftime('%H:%M:%S')}", flush=True)
print(f"DONE rows={rows_total:,} expected={PARTS*2500012:,} scale={scale:.8g}", flush=True)
