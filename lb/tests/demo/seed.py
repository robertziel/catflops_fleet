"""DEMO SEED: the history the gateway would have had, written into the demo's state directory
before it starts — so the daily rollup, the Activity tile and the 48 h / 30 d history dialog have days to show.

Written with the gateway's OWN writers, so the files are exactly what a running gateway leaves:
  perf-YYYY-MM-DD.jsonl   per-minute windows of the `node` server (perflog.py's line shape: v, t_start, t_end, span_s,
                          config.model_choice, gen_tokens, prompt_tokens, cluster), two models over 12 days, plus one
                          day of minutes no engine could count (null tokens: an unmeasured day, never 0);
  events-YYYY-MM-DD.jsonl one event of each family over the last three days (events.EventLog with a fixed clock);
  hw-tiers.json           48 h of one-minute and 30 d of fifteen-minute buckets per box (hw_tiers.HwTiers.ingest);
  perf/fidelity/fid-*.json  three finished fidelity runs on the frozen corpus (FP8 KV, NVFP4 KV — measurably worse —,
                          and a second FP8 run whose launch resolved to no target: accelerator / TP null with their why)
                          and one failed run (fidelity.FidelityStore).

THE EXPECTED FILE (`expected.json` beside them) holds the figures computed HERE, independently of the gateway's fold:
each model's last-24 h generated tokens and its best minute's tok/s, the 24 h total, each box's seeded maximum GPU
temperature, each fidelity run's mean NLL and perplexity and the A/B ΔNLL of the FP8 and NVFP4 runs. The audit (audit.mjs) checks the panel against these numbers.

Usage (in the gateway image): python -I seed.py <state dir>
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, "/app")

from vllm_lb.events import EventLog  # noqa: E402
from vllm_lb.fidelity import FidelityStore, load_corpus  # noqa: E402
from vllm_lb.hw_tiers import HwTiers  # noqa: E402
from vllm_lb.perflog import day_of  # noqa: E402

DAY = 86400.0
#: model id → (generated tokens per minute on day 0, the change per day back, minutes per day, days it ran).
MODELS = {"qwen3.8-27b-nvidia-nvfp4": (1_830, -22, 90, range(0, 12)),
          "qwen3.8-27b-dflash2": (3_310, 41, 35, range(0, 12, 3))}
#: Box → (the seeded GPU temperature's base and swing, °C): box-c peaks over its live 87 °C so its maximum is seeded.
TEMPS = {"box-a": (52.0, 9.0), "box-b": (50.0, 8.0), "box-c": (70.0, 20.0)}


def perf(state: str, now: float) -> dict:
    out = {m: {"last_24h_gen": 0, "peak_tok_s": 0.0} for m in MODELS}
    by_day: dict[str, list[str]] = {}
    for model, (per_min, slope, minutes, days) in MODELS.items():
        for d in days:
            # Minutes run in one block, ending 2 h before `now` minus d days: day 0 sits inside the last 24 h.
            end = now - 2 * 3600 - d * DAY
            for i in range(minutes):
                t_end = end - i * 60.0
                gen = per_min + slope * d + (i % 7) * 11
                rec = {"v": 1, "t_start": t_end - 60.0, "t_end": t_end, "span_s": 60.0,
                       "config": {"model_choice": model}, "gen_tokens": gen, "prompt_tokens": gen * 3,
                       "cluster": "node"}
                by_day.setdefault(day_of(t_end), []).append(json.dumps(rec))
                if t_end > now - DAY:
                    out[model]["last_24h_gen"] += gen
                out[model]["peak_tok_s"] = max(out[model]["peak_tok_s"], gen / 60.0)
    # One unmeasured day of the first model: minutes whose engine exposed no token counter.
    first = next(iter(MODELS))
    for i in range(20):
        t_end = now - 13 * DAY - i * 60.0
        rec = {"v": 1, "t_start": t_end - 60.0, "t_end": t_end, "span_s": 60.0, "config": {"model_choice": first},
               "gen_tokens": None, "prompt_tokens": None, "cluster": "node"}
        by_day.setdefault(day_of(t_end), []).append(json.dumps(rec))
    for day, lines in by_day.items():
        with open(os.path.join(state, "perf", f"perf-{day}.jsonl"), "a") as fh:
            fh.write("\n".join(sorted(lines, key=lambda s: json.loads(s)["t_end"])) + "\n")
    return {"models": out, "total_24h_gen": sum(m["last_24h_gen"] for m in out.values())}


def seeded_events(state: str, now: float) -> int:
    t = [now]
    plan = [
        (3 * DAY, "cluster.state", "node", "starting → running: 2 ranks up", {"state": "running"}),
        (2.5 * DAY, "request.killed", "box-a", "request 9f3c killed: 412 s queued", {"id": "9f3c", "by": "admin"}),
        (2 * DAY, "cluster.state", "node", "running → failed: rank 1 exited 137", {"state": "failed"}),
        (2 * DAY - 300, "cluster.state", "node", "failed → running: 2 ranks up", {"state": "running"}),
        (1 * DAY, "power.action", "box-c", "clock box-c: ok by admin",
         {"action": "clock", "result": "ok", "user": "admin", "dry_run": False, "rc": 0}),
        (20 * 3600, "finding.firing", "box-b", "RAM AVAIL on box-b: RAM available 7.9 GiB < 11.9 GiB (10 % of 119 GiB)",
         {"id": "box-b:low_ram", "finding": "low_ram", "value": 7.9, "threshold": 11.9, "unit": "GiB"}),
        (19 * 3600, "finding.cleared", "box-b", "RAM AVAIL cleared on box-b · 31.2 GiB vs 11.9 GiB",
         {"id": "box-b:low_ram", "finding": "low_ram", "value": 31.2, "threshold": 11.9, "unit": "GiB"}),
        (19 * 3600 - 1, "alert.sent", "box-b", "webhook: CLEARED low_ram on box-b", {"channel": "webhook"}),
    ]
    lg = EventLog(os.path.join(state, "perf"), clock=lambda: t[0])
    for ago, kind, subject, summary, fields in plan:
        t[0] = now - ago
        lg.record(kind, subject, summary, **fields)
    lg.drain()
    return len(plan)


def tiers(state: str, now: float) -> dict:
    tf = HwTiers(os.path.join(state, "hw-tiers.json"), clock=lambda: now)
    maxima = {}
    for box, (base, swing) in TEMPS.items():
        top = -math.inf
        steps = [(now - 30 * DAY + i * 900.0) for i in range(int((30 * DAY - 48 * 3600) / 900))]
        steps += [(now - 48 * 3600 + i * 60.0) for i in range(48 * 60 - 5)]
        for t in steps:
            phase = (t % DAY) / DAY * 2 * math.pi
            temp = round(base + swing * math.sin(phase), 1)
            top = max(top, temp)
            tf.ingest(box, {"gpu_temp_c": temp, "gpu_power_w": round(30 + 0.4 * temp, 1),
                            "gpu_pct": round(40 + 30 * math.sin(phase), 1), "mem_used_mb": 60_000 + int(9_000 * math.cos(phase)),
                            "net_in_mbps": round(400 + 350 * math.sin(phase * 3), 1),
                            "net_out_mbps": round(250 + 200 * math.cos(phase * 2), 1)}, t)
        maxima[box] = top
    tf.save(now=now)
    return maxima


#: Fidelity runs: (days ago, KV dtype, NLL per token, the per-document wobble's seed offset, status).
FIDELITY = (("fp8", 5.0, 2.1031, 0, "done"), ("nvfp4", 3.0, 2.1342, 7, "done"), ("fp8", 1.0, 2.1036, 3, "done"),
            ("nvfp4", 0.5, None, 0, "failed"))


#: describe_cluster's words for a launch that resolved to no target (vllm_lb/fidelity.py).
UNRESOLVED = "Qwen3.8 27B · dense NVFP4's cluster launch did not resolve to a target on this fleet"


def fidelity(state: str, now: float) -> dict:
    corpus = load_corpus()
    store = FidelityStore(os.path.join(state, "perf", "fidelity"))
    os.makedirs(store.dir, exist_ok=True)
    out = []
    for i, (kv, ago, nll, wob, status) in enumerate(FIDELITY):
        t = now - ago * DAY
        rid = f"fid-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t))}-{i:04x}"
        meta = {"server": "node", "served_name": "Qwen/Qwen3.8-27B", "model": "qwen3.8-27b-nvidia-nvfp4",
                "model_label": "Qwen3.8 27B · dense NVFP4", "engine": "vllm", "accelerator": "nvidia-sm121", "tp": 1,
                "quant": "modelopt_mixed", "kv_cache_dtype": kv, "why": {}}
        if i == 2:      # a run whose cluster launch resolved to no target: the figures are null with their why
            meta.update(accelerator=None, tp=None, why={"accelerator": UNRESOLVED, "tp": UNRESOLVED})
        docs = []
        for j, d in enumerate(corpus.docs):
            tokens = d.bytes // 4
            # A per-document wobble (±1.5 %), so the bootstrap interval has a width; the tokenizer is one (same sha).
            w = 1 + 0.015 * math.sin(1.7 * j + wob)
            docs.append({"id": d.id, "bytes": d.bytes, "tokens": tokens, "sum_nll": round(tokens * (nll or 2.1) * w, 6),
                         "tokens_sha": f"{abs(hash(d.id)) % (1 << 60):016x}"[:16]})
        run = {"id": rid, "status": status, "error": None, "started_at": t, "finished_at": t + 41.5, "meta": meta,
               "corpus": corpus.view(), "total_docs": len(corpus.docs), "done_docs": len(docs), "docs": docs}
        if status == "done":
            tok = sum(x["tokens"] for x in docs)
            s_nll = sum(x["sum_nll"] for x in docs)
            run.update(tokens=tok, mean_nll=s_nll / tok, ppl=math.exp(s_nll / tok),
                       bits_per_byte=s_nll / math.log(2) / sum(x.bytes for x in corpus.docs))
        else:
            run.update(docs=[], done_docs=0, tokens=0, mean_nll=None, ppl=None, bits_per_byte=None,
                       error=f"vllm (Qwen/Qwen3.8-27B), document {corpus.docs[0].id}: returned no token logprobs "
                             f"(choices[0].logprobs.token_logprobs is absent) — this engine does not score the prompt "
                             f"under echo + logprobs")
        store.save(run)
        out.append({"id": rid, "status": status, "kv": kv, "why": meta["why"], "mean_nll": run["mean_nll"], "ppl": run["ppl"],
                    "tokens": run["tokens"], "sums": [x["sum_nll"] for x in docs], "toks": [x["tokens"] for x in docs]})
    a, b = out[0], out[1]
    tok = sum(a["toks"])
    d_nll = round((sum(b["sums"]) - sum(a["sums"])) / tok, 4)
    for r in out:
        r.pop("sums"), r.pop("toks")
    return {"runs": out, "ab": {"a": a["id"], "b": b["id"], "d_nll": d_nll, "nll_a": round(a["mean_nll"], 4),
                                "nll_b": round(b["mean_nll"], 4), "docs": len(corpus.docs), "tokens": tok}}


def main() -> None:
    state = sys.argv[1]
    os.makedirs(os.path.join(state, "perf"), exist_ok=True)
    now = time.time()
    expected = {"seeded_at": now, "perf": perf(state, now), "events": seeded_events(state, now),
                "max_gpu_temp_c": tiers(state, now), "fidelity": fidelity(state, now)}
    with open(os.path.join(state, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
    print(f"[seed] {state}: perf {expected['perf']['total_24h_gen']:,} tok in 24 h · {expected['events']} events · "
          f"tiers {', '.join(f'{b} max {v} °C' for b, v in expected['max_gpu_temp_c'].items())} · "
          f"fidelity {len(expected['fidelity']['runs'])} runs, A/B ΔNLL {expected['fidelity']['ab']['d_nll']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
