#!/usr/bin/env python3
"""Measure the DFlash2 pair: decode tok/s, TTFT, prefill tok/s, and acceptance from the engine.

Every configuration is measured on the SAME prompts, so a speedup is this harness's own baseline
vs its own speculative run, not a comparison across corpora. Acceptance comes from vLLM's
`vllm:spec_decode_*` counters, read before and after each phase, never from the client's guess.

  bench.py --base http://192.0.2.201:8021 --model Qwen/Qwen3.8-27B-DFlash2 \
           --mode single --tag K16 --out /path/run.json
  bench.py ... --mode sweep --concurrency 2 4 8
  bench.py ... --mode ab --max-tokens 400        # greedy capture for the correctness A/B

--mode ab streams with temperature 0 and thinking ON and records the ordered delta chunks, which
is the token sequence as the server emitted it; two runs' files are compared with --compare.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request

# Real work, not a synthetic repeat: a synthetic prompt overstates speculative acceptance (measured
# on this fleet, 0.80 synthetic against 0.45 real). EDIT is upstream's own shape -- a file to
# rewrite with a mechanical change, where a block drafter should do best. PROSE and REASON are the
# other two ends, so a headline figure cannot come from the friendliest prompt alone.
EDIT = """Here is a Python module:

""" + "\n\n".join(
    f"""@dataclass
class Item{i}:
    name: str = "item{i}"
    qty: int = {i}
    unit_price: int = {100 + i}

    def total(self) -> int:
        return self.qty * self.unit_price"""
    for i in range(1, 31)
) + """

Add a `discount(self, pct: int) -> int` method to EVERY Item class above that returns the
total reduced by pct percent, using integer arithmetic. Output the COMPLETE modified file."""

PROSE = ("Write a detailed technical explanation of why memory bandwidth, rather than peak FLOP/s, "
         "usually limits single-stream decoding throughput for a 27-billion-parameter transformer "
         "on one accelerator. Cover the arithmetic intensity of a decode step, why batching helps, "
         "and what speculative decoding changes about that arithmetic.")

REASON = ("A train leaves station A at 14:05 travelling at 72 km/h. A second train leaves station B, "
          "243 km away, at 14:20 travelling toward A at 90 km/h. At what clock time do they meet, "
          "and how far from station A? Show your working, then state the answer on one line.")

PROMPTS = {"edit": EDIT, "prose": PROSE, "reason": REASON}

# The counters carry a `_total` suffix, and each also has a `_created` GAUGE holding a unix
# timestamp (~1.79e9). Match the counters by their exact names so the epoch gauges can never be
# summed into a rate, and so a missing name shows up as absent rather than as zero.
SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_KEYS = (SPEC_DRAFTS, SPEC_DRAFT_TOKENS, SPEC_ACCEPTED,
             "vllm:spec_decode_num_accepted_tokens_per_pos_total")


def post(url: str, body: dict, timeout: float = 1800.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(base: str) -> dict[str, float]:
    """Every counter whose name starts with one of SPEC_KEYS, summed over label sets."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=20) as r:
            text = r.read().decode()
    except Exception as exc:
        return {"_error": repr(exc)}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if not m:
            continue
        name, labels, val = m.groups()
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = re.search(r'position="(\d+)"', labels or "")
            if pos:
                name = f"accepted_at_pos_{int(pos.group(1)):02d}"
        if name in SPEC_KEYS or name in (
                "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                "vllm:num_preemptions_total"):
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return out


def one_stream(base: str, model: str, prompt: str, max_tokens: int, temperature: float,
               thinking: bool, seed: int | None, capture: bool) -> dict:
    """One streamed chat completion. Returns timings and, with capture, the ordered deltas."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    n_chunks = 0
    usage = None
    finish = None
    with post(base + "/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                # vLLM 0.29.0 names the thinking delta "reasoning"; older/other builds use
                # "reasoning_content". Take whichever is present so a thinking-on run is not
                # silently measured as zero tokens.
                piece = ((d.get("content") or "") + (d.get("reasoning_content") or "")
                         + (d.get("reasoning") or ""))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
                    if capture:
                        chunks.append(piece)
    t_end = time.perf_counter()
    out = {
        "wall_s": t_end - t0,
        "ttft_s": ttft,
        "decode_s": (t_end - t0 - ttft) if ttft is not None else None,
        "stream_chunks": n_chunks,
        "usage": usage,
        "finish_reason": finish,
    }
    if usage:
        ct = usage.get("completion_tokens") or 0
        pt = usage.get("prompt_tokens") or 0
        out["completion_tokens"] = ct
        out["prompt_tokens"] = pt
        if ttft and ttft > 0:
            out["prefill_tok_s"] = pt / ttft
        if out["decode_s"] and out["decode_s"] > 0 and ct:
            out["decode_tok_s"] = (ct - 1) / out["decode_s"]
        if out["wall_s"] > 0 and ct:
            out["e2e_tok_s"] = ct / out["wall_s"]
    if capture:
        out["text"] = "".join(chunks)
        out["deltas"] = chunks
    return out


def phase(base: str, model: str, runs: list[dict], label: str) -> dict:
    """Run `runs` concurrently, bracketing them with the engine's spec counters."""
    before = metrics(base)
    results: list[dict | None] = [None] * len(runs)
    errors: list[str] = []

    def work(i: int, spec: dict):
        try:
            results[i] = one_stream(base, model, **spec)
        except Exception as exc:
            errors.append(f"[{i}] {type(exc).__name__}: {exc}")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(i, r)) for i, r in enumerate(runs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    after = metrics(base)

    ok = [r for r in results if r]
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after if not k.startswith("_")}
    draft = delta.get(SPEC_DRAFT_TOKENS, 0.0)
    acc = delta.get(SPEC_ACCEPTED, 0.0)
    drafts = delta.get(SPEC_DRAFTS, 0.0)
    toks = sum(r.get("completion_tokens") or 0 for r in ok)
    summary = {
        "label": label,
        "concurrency": len(runs),
        "wall_s": wall,
        "streams_ok": len(ok),
        "errors": errors,
        "completion_tokens_total": toks,
        "aggregate_tok_s": (toks / wall) if wall > 0 else None,
        "per_stream_tok_s": [r.get("decode_tok_s") for r in ok],
        "ttft_s": [r.get("ttft_s") for r in ok],
        "prefill_tok_s": [r.get("prefill_tok_s") for r in ok],
        "spec_delta": delta,
        "acceptance": (acc / draft) if draft else None,
        "mean_accepted_per_draft": (acc / drafts) if drafts else None,
        "tokens_per_pass": (1 + acc / drafts) if drafts else None,
        "draft_tokens_per_draft": (draft / drafts) if drafts else None,
        "accepted_per_pos": {k: v for k, v in sorted(delta.items())
                             if k.startswith("accepted_at_pos_") and v},
    }
    return {"summary": summary, "runs": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("single", "sweep", "ab"), default="single")
    ap.add_argument("--prompt", choices=tuple(PROMPTS) + ("all",), default="all")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[2, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", choices=("on", "off"), default="on")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a = json.load(open(args.compare[0]))
        b = json.load(open(args.compare[1]))
        print(f"A = {args.compare[0]}  tag={a.get('tag')}")
        print(f"B = {args.compare[1]}  tag={b.get('tag')}")
        worst = 0
        for name in sorted(set(a["ab"]) & set(b["ab"])):
            ta, tb = a["ab"][name]["text"], b["ab"][name]["text"]
            da, db = a["ab"][name]["deltas"], b["ab"][name]["deltas"]
            if ta == tb:
                print(f"  {name:7s} IDENTICAL  ({len(ta)} chars, {len(da)} chunks)")
                continue
            i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]),
                     min(len(ta), len(tb)))
            j = next((k for k in range(min(len(da), len(db))) if da[k] != db[k]),
                     min(len(da), len(db)))
            worst += 1
            print(f"  {name:7s} DIVERGES at char {i}, stream chunk {j} "
                  f"(A {len(da)} chunks/{len(ta)} chars, B {len(db)}/{len(tb)})")
            print(f"    A ...{ta[max(0,i-60):i]!r} >>> {ta[i:i+60]!r}")
            print(f"    B ...{tb[max(0,i-60):i]!r} >>> {tb[i:i+60]!r}")
        print(f"VERDICT: {'DIVERGENCE' if worst else 'no divergence'} "
              f"({worst} of {len(set(a['ab']) & set(b['ab']))} prompts differ)")
        return 1 if worst else 0

    names = list(PROMPTS) if args.prompt == "all" else [args.prompt]
    doc: dict = {"tag": args.tag, "base": args.base, "model": args.model, "mode": args.mode,
                 "max_tokens": args.max_tokens, "temperature": args.temperature,
                 "thinking": args.thinking, "seed": args.seed,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "metrics_at_start": metrics(args.base)}

    if args.mode == "ab":
        doc["ab"] = {}
        for name in names:
            r = one_stream(args.base, args.model, PROMPTS[name], args.max_tokens,
                           args.temperature, args.thinking == "on", args.seed, capture=True)
            doc["ab"][name] = r
            print(f"[ab] {name}: {r.get('completion_tokens')} tok, "
                  f"{r.get('stream_chunks')} chunks, {r['wall_s']:.1f}s, "
                  f"finish={r.get('finish_reason')}", flush=True)
    elif args.mode == "single":
        doc["phases"] = []
        for name in names:
            spec = dict(prompt=PROMPTS[name], max_tokens=args.max_tokens,
                        temperature=args.temperature, thinking=args.thinking == "on",
                        seed=args.seed, capture=False)
            p = phase(args.base, args.model, [spec], f"single/{name}")
            s = p["summary"]
            print(f"[single/{name}] decode {fmt(s['per_stream_tok_s'][0] if s['per_stream_tok_s'] else None)} tok/s "
                  f"| ttft {fmt(s['ttft_s'][0] if s['ttft_s'] else None)}s "
                  f"| prefill {fmt(s['prefill_tok_s'][0] if s['prefill_tok_s'] else None)} tok/s "
                  f"| accept {fmt(s['acceptance'])} | tok/pass {fmt(s['tokens_per_pass'])} "
                  f"| {s['completion_tokens_total']} tok", flush=True)
            doc["phases"].append(p)
    else:
        doc["phases"] = []
        for c in args.concurrency:
            runs = [dict(prompt=PROMPTS[names[i % len(names)]], max_tokens=args.max_tokens,
                         temperature=args.temperature, thinking=args.thinking == "on",
                         seed=args.seed, capture=False) for i in range(c)]
            p = phase(args.base, args.model, runs, f"sweep/c{c}")
            s = p["summary"]
            per = [v for v in s["per_stream_tok_s"] if v]
            print(f"[sweep/c{c}] aggregate {fmt(s['aggregate_tok_s'])} tok/s "
                  f"| per stream {fmt(sum(per)/len(per) if per else None)} tok/s "
                  f"| accept {fmt(s['acceptance'])} | tok/pass {fmt(s['tokens_per_pass'])} "
                  f"| {s['streams_ok']}/{c} ok | {s['completion_tokens_total']} tok "
                  f"| errors {len(s['errors'])}", flush=True)
            doc["phases"].append(p)

    doc["metrics_at_end"] = metrics(args.base)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {args.out}")
    return 0


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    sys.exit(main())
