#!/usr/bin/env python3
"""Load generator for the TP=2 SGLang pair: streamed chat completions, N at a time.

Standard library only, so it runs anywhere ssh reaches and installs nothing.

What it measures, per request, from the SSE stream itself:
  ttft_s        wall seconds to the first token that carries content
  decode_s      wall seconds from that first token to the last
  out_tokens    the server's own usage.completion_tokens when the stream reports it, else the
                number of content deltas seen (said in the output, so the two are never confused)
  tok_s         out_tokens / decode_s           — the per-stream decode rate
and, per level, aggregate = sum(out_tokens) / (last finish - first start), the throughput a user
of the whole box would see.

Why every prompt carries a fresh uuid: with a shared prefix the radix cache answers the prefill of
every stream after the first, which flatters TTFT and makes a concurrency sweep measure the cache
instead of the engine. Each stream gets its own prefix, so each pays its own prefill.
"""
import argparse, json, statistics, sys, threading, time, urllib.error, urllib.request, uuid

PROMPTS = {
    "code": (
        "Write a complete, self-contained Python module that implements a thread-safe LRU cache with "
        "a time-to-live per entry. Include type hints, docstrings, the eviction logic, and a short "
        "block of unit tests using unittest. Explain the locking choices in comments."
    ),
    "prose": (
        "Write a detailed technical essay on the history of computing, from Babbage's difference "
        "engine to modern GPU accelerators. Cover the key architectural shifts and why each one "
        "mattered."
    ),
    "chat": (
        "I have a two-node GPU cluster joined by a 200 Gb RoCE fabric and I am seeing poor "
        "all-reduce bandwidth during tensor-parallel decoding. Walk me through how you would "
        "diagnose it, in order, and what numbers you would expect at each step."
    ),
}


def one(url, model, prompt_key, max_tokens, thinking, timeout, out, idx):
    """Run one streamed request and append its record to `out`."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"Session {uuid.uuid4()}. {PROMPTS[prompt_key]}"}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not thinking:
        # The model card's non-thinking sampling, and the template switch that turns the trace off.
        body.update(temperature=0.7, top_p=0.8, presence_penalty=1.5)
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    rec = {"i": idx, "prompt": prompt_key, "thinking": thinking, "error": None,
           "ttft_s": None, "decode_s": None, "out_tokens": None, "token_source": None}
    t0 = time.monotonic()
    first = last = None
    deltas = 0
    usage_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage_tokens = ev["usage"].get("completion_tokens")
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    # reasoning_content is what --reasoning-parser qwen3 puts the <think> block in:
                    # it is generated text and must count, or thinking mode reads as 0 tok/s.
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                        deltas += 1
    except Exception as exc:                                     # noqa: BLE001 - reported, not raised
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["t_start"], rec["t_end"] = t0, time.monotonic()
    if first is not None:
        rec["ttft_s"] = round(first - t0, 3)
        rec["decode_s"] = round((last - first), 3) or None
    if usage_tokens is not None:
        rec["out_tokens"], rec["token_source"] = usage_tokens, "usage"
    elif deltas:
        rec["out_tokens"], rec["token_source"] = deltas, "deltas"
    if rec["out_tokens"] and rec["decode_s"]:
        rec["tok_s"] = round(rec["out_tokens"] / rec["decode_s"], 2)
    out.append(rec)


def level(url, model, n, max_tokens, thinking, timeout):
    """n streams at once; returns the records plus the aggregate over the whole batch."""
    recs, threads = [], []
    keys = list(PROMPTS)
    lock = threading.Lock()
    sink = []

    def run(i):
        local = []
        one(url, model, keys[i % len(keys)], max_tokens, thinking, timeout, local, i)
        with lock:
            sink.extend(local)

    wall0 = time.monotonic()
    for i in range(n):
        t = threading.Thread(target=run, args=(i,), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - wall0
    recs = sorted(sink, key=lambda r: r["i"])
    good = [r for r in recs if r.get("tok_s")]
    toks = sum(r["out_tokens"] for r in good)
    # The aggregate spans from the first request's start to the last one's end, so queueing counts.
    span = (max(r["t_end"] for r in recs) - min(r["t_start"] for r in recs)) if recs else wall
    return {
        "concurrency": n,
        "thinking": thinking,
        "ok": len(good),
        "failed": [r["error"] for r in recs if r["error"]],
        "wall_s": round(span, 2),
        "out_tokens_total": toks,
        "agg_tok_s": round(toks / span, 2) if span else None,
        "per_stream_tok_s_mean": round(statistics.fmean(r["tok_s"] for r in good), 2) if good else None,
        "per_stream_tok_s_min": min((r["tok_s"] for r in good), default=None),
        "per_stream_tok_s_max": max((r["tok_s"] for r in good), default=None),
        "ttft_s_mean": round(statistics.fmean(r["ttft_s"] for r in good), 3) if good else None,
        "ttft_s_max": max((r["ttft_s"] for r in good), default=None),
        "token_source": sorted({r["token_source"] for r in good if r["token_source"]}),
        "requests": recs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="e.g. http://192.0.2.201:8020")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1])
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--out", help="write the full record as JSON here")
    a = p.parse_args()
    url = a.base.rstrip("/") + "/v1/chat/completions"
    results = []
    for n in a.concurrency:
        if n > 8:
            print(f"refusing concurrency {n}: 8 is the thermal ceiling for this work", file=sys.stderr)
            return 2
        r = level(url, a.model, n, a.max_tokens, not a.no_thinking, a.timeout)
        results.append(r)
        print(f"c{r['concurrency']:<3} ok={r['ok']:<3} agg={r['agg_tok_s']!s:>8} tok/s  "
              f"per_stream={r['per_stream_tok_s_mean']!s:>7} (min {r['per_stream_tok_s_min']}, "
              f"max {r['per_stream_tok_s_max']})  ttft={r['ttft_s_mean']}s/{r['ttft_s_max']}s  "
              f"tokens={r['out_tokens_total']} via {','.join(r['token_source']) or '?'}", flush=True)
        for e in r["failed"]:
            print(f"     FAILED: {e}", file=sys.stderr, flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
