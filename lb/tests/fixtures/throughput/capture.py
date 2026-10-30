#!/usr/bin/env python3
"""Capture, THROUGH the deployed LB, the exact response bytes each accounted shape produces.

Step 1 of worklog/vllm-gateway-throughput.md. `UsageTap` is tested on these bytes, not on bytes
shaped from reading upstream source: a stream written by hand only confirms what its author
already believes a server sends (the verify-with-captured-client-bodies lesson).

The rules this script keeps, from the operator's go-ahead:
  * every inference request goes to the LB (`--url`), never to ollama or a vLLM directly;
  * at most MAX_POSTS POSTs in total, a retried 503 included, each asking for <= MAX_TOKENS;
  * ollama's `/api/ps` is read with a GET (no inference): the model line is built from that route.

The body is read with `read1`, the call the LB's relay itself makes, so the saved bytes are the
relayed bytes. Each read's size and arrival time are kept too, so a test can replay the real
chunking and the real pacing. Per case it writes `<name>.body` (the bytes) and `<name>.json` (the
request, the status line, the headers, who served it, the reads).

  docker run --rm --network host --user "$(id -u):$(id -g)" \
    -v "$PWD/lb/tests/fixtures/throughput":/out vllm-lb:throughput \
    python /out/capture.py --out /out
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path

MAX_POSTS = 12
MAX_TOKENS = 64
MODEL = "qwen3.8:latest"
PROMPT = "Count from 1 to 20, separated by single spaces. Reply with the numbers only."
MESSAGES = [{"role": "user", "content": PROMPT}]

#: (name, path, body, abandon). `abandon` hangs up once the first content delta is in.
CASES = [
    # Claude Code's own path carries `?beta=true`: the tap must strip the query to classify it.
    ("anthropic-stream", "/v1/messages?beta=true",
     {"model": MODEL, "max_tokens": 64, "stream": True, "messages": MESSAGES}, False),
    ("anthropic-json", "/v1/messages",
     {"model": MODEL, "max_tokens": 32, "stream": False, "messages": MESSAGES}, False),
    ("chat-stream-usage", "/v1/chat/completions",
     {"model": MODEL, "max_tokens": 48, "stream": True, "stream_options": {"include_usage": True},
      "messages": MESSAGES}, False),
    ("chat-stream-no-usage", "/v1/chat/completions",
     {"model": MODEL, "max_tokens": 32, "stream": True, "messages": MESSAGES}, False),
    ("chat-json", "/v1/chat/completions",
     {"model": MODEL, "max_tokens": 32, "stream": False, "messages": MESSAGES}, False),
    # The LB forwards every POST path to the chosen backend, so native /api/chat reaches ollama
    # through it too.
    ("native-chat-stream", "/api/chat",
     {"model": MODEL, "stream": True, "options": {"num_predict": 48}, "messages": MESSAGES}, False),
    ("anthropic-abandoned", "/v1/messages",
     {"model": MODEL, "max_tokens": 64, "stream": True, "messages": MESSAGES}, True),
]


def asked_tokens(body: dict) -> int:
    return int(body.get("max_tokens") or (body.get("options") or {}).get("num_predict") or 0)


def wait_for_lease(base: str, timeout_s: float = 300.0) -> None:
    """GET /lb/status until a lease is running, so a retried POST is not spent on the same refusal."""
    u = urllib.parse.urlsplit(base)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        try:
            conn.request("GET", "/lb/status")
            status = json.loads(conn.getresponse().read())
        finally:
            conn.close()
        if any(lease.get("status") == "running" for lease in status.get("leases") or []):
            return
        time.sleep(3)
    raise SystemExit("no lease was claimed in time; stopping without spending another POST")


def post(base: str, path: str, body: dict, *, abandon: bool, budget: dict) -> tuple[bytes, dict]:
    u = urllib.parse.urlsplit(base)
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    refused = []
    while True:
        if budget["posts"] >= budget["max"]:
            raise SystemExit(f"stopping: the budget of {budget['max']} POSTs is spent")
        budget["posts"] += 1
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
        t0 = time.monotonic()
        conn.request("POST", path, body=payload, headers=headers)
        resp = conn.getresponse()
        head_s = time.monotonic() - t0
        if resp.status == 503:                 # nothing held yet: the LB asked for a lease
            refused.append(resp.read().decode("utf-8", "replace")[:300])
            conn.close()
            print("  503: waiting (GET /lb/status) for the fallback lease to be claimed", flush=True)
            wait_for_lease(base)
            continue
        reads, data = [], bytearray()
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            reads.append([round(time.monotonic() - t0, 6), len(chunk)])
            data += chunk
            if abandon and b"content_block_delta" in data:
                break
        resp.close()                            # for `abandon`, the client hangs up mid-stream here
        conn.close()
        meta = {
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "via": base,
            "request": {"method": "POST", "path": path, "headers": headers, "body": body},
            "status": resp.status, "reason": resp.reason, "headers": resp.getheaders(),
            "served_by": {k: resp.getheader(k) for k in ("X-LB-Backend", "X-LB-Kind", "X-LB-Box")},
            "head_s": round(head_s, 6), "reads": reads, "bytes": len(data),
            "abandoned": abandon, "refused_before": refused, "posts_so_far": budget["posts"],
        }
        return bytes(data), meta


def snapshot_ps(ollama: str, out: Path, label: str) -> None:
    """GET /api/ps with its headers: the resident models, and the server's own clock (`Date`)."""
    u = urllib.parse.urlsplit(ollama)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    try:
        conn.request("GET", "/api/ps")
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()
    (out / f"ollama-ps-{label}.json").write_text(json.dumps({
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "url": f"{ollama}/api/ps", "status": resp.status, "headers": resp.getheaders(),
        "body": body.decode("utf-8", "replace")}, indent=2) + "\n")
    print(f"ps {label}: {resp.status} Date={resp.getheader('Date')} {body[:200]!r}", flush=True)


def head_idle(metrics_url: str) -> bool:
    """True only when the vLLM answers /metrics and reports 0 running and 0 waiting. A head that is
    gone, refuses, or does not expose depth counts as busy: a turn of ours must never queue behind
    someone else's work."""
    u = urllib.parse.urlsplit(metrics_url)
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
        try:
            conn.request("GET", u.path or "/metrics")
            resp = conn.getresponse()
            text = resp.read().decode("utf-8", "replace")
        finally:
            conn.close()
    except (OSError, http.client.HTTPException):
        return False
    if resp.status != 200:
        return False
    depth: dict[str, float] = {}
    for line in text.splitlines():
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(name + "{") or line.startswith(name + " "):   # never `_by_reason`
                try:
                    depth[name] = depth.get(name, 0.0) + float(line.rsplit(None, 1)[1])
                except ValueError:
                    return False
    return depth.get("vllm:num_requests_running") == 0 and depth.get("vllm:num_requests_waiting") == 0


def wait_until_idle(metrics_url: str, wait_s: float, every_s: float) -> bool:
    deadline = time.monotonic() + wait_s
    while not head_idle(metrics_url):
        if time.monotonic() + every_s > deadline:
            return False
        print(f"  head busy (or gone); re-checking in {every_s:.0f} s", flush=True)
        time.sleep(every_s)
    return True


def capture_metrics(url: str, out: Path, name: str, count: int, every_s: float) -> None:
    """GET a vLLM's /metrics `count` times, `every_s` apart (GET only, no inference): real exposition
    text, and a real pair of samples whose counter deltas a test can turn into a rate. Each body is
    saved as sent; `<name>.json` keeps the status, headers and monotonic read time of each."""
    u = urllib.parse.urlsplit(url)
    t0 = time.monotonic()
    samples = []
    for i in range(count):
        time.sleep(max(0.0, t0 + i * every_s - time.monotonic()))
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        try:
            conn.request("GET", u.path or "/metrics")
            resp = conn.getresponse()
            body = resp.read()
            read_at = time.monotonic() - t0            # when the body was READ, as the probe stamps it
        finally:
            conn.close()
        file = f"{name}-{i + 1}.metrics"
        (out / file).write_bytes(body)
        samples.append({"file": file, "status": resp.status, "headers": resp.getheaders(),
                        "read_at_s": round(read_at, 6), "bytes": len(body)})
        print(f"{file}: {resp.status} {len(body)} B read at {read_at:.3f} s", flush=True)
    (out / f"{name}.json").write_text(json.dumps({
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "url": url, "samples": samples}, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8010", help="the deployed LB")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="GET /api/ps only")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="case names to run (default: all)")
    ap.add_argument("--metrics", help="instead of the turns: GET this vLLM /metrics URL, no POST at all")
    ap.add_argument("--name", default="vllm-metrics", help="file stem for --metrics")
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--every", type=float, default=10.0, help="seconds between --metrics reads")
    ap.add_argument("--gate", help="send each case only while this vLLM /metrics shows 0 running, 0 waiting")
    ap.add_argument("--gate-wait", type=float, default=300.0, help="then skip the case")
    ap.add_argument("--gate-every", type=float, default=30.0)
    ap.add_argument("--expect-kind", help="keep a capture only if the LB's X-LB-Kind says this")
    ap.add_argument("--prefix", default="", help="prepended to each saved case name")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="cap on every case")
    ap.add_argument("--max-posts", type=int, default=MAX_POSTS, help="POST budget for this run")
    args = ap.parse_args()
    if args.metrics:
        capture_metrics(args.metrics, args.out, args.name, args.count, args.every)
        return 0
    cap = min(args.max_tokens, MAX_TOKENS)
    budget = {"posts": 0, "max": min(args.max_posts, MAX_POSTS)}
    if not args.gate:
        snapshot_ps(args.ollama, args.out, "before")
    for name, path, body, abandon in CASES:
        if args.only and name not in args.only:
            continue
        if "max_tokens" in body:
            body = {**body, "max_tokens": min(body["max_tokens"], cap)}
        else:
            body = {**body, "options": {**body["options"], "num_predict": min(asked_tokens(body), cap)}}
        assert asked_tokens(body) <= cap, body
        if args.gate and not wait_until_idle(args.gate, args.gate_wait, args.gate_every):
            print(f"{args.prefix}{name}: SKIPPED — the head stayed busy for {args.gate_wait:.0f} s",
                  flush=True)
            continue
        print(f"{args.prefix}{name}: POST {path}", flush=True)
        data, meta = post(args.url, path, body, abandon=abandon, budget=budget)
        print(f"  {meta['status']} via {meta['served_by']} {meta['bytes']} B in "
              f"{len(meta['reads'])} reads, last at {meta['reads'][-1][0] if meta['reads'] else '-'} s",
              flush=True)
        if args.expect_kind and meta["served_by"].get("X-LB-Kind") != args.expect_kind:
            print(f"  DISCARDED — not served by {args.expect_kind}", flush=True)
            continue
        (args.out / f"{args.prefix}{name}.body").write_bytes(data)
        (args.out / f"{args.prefix}{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not args.gate:
        snapshot_ps(args.ollama, args.out, "after")
    print(f"POSTs sent: {budget['posts']} of {budget['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
