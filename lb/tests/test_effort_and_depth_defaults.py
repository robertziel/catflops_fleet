"""Two defaults, both settled by measurement on the deployed box-a vLLM (2026-09-12).

1. EFFORT. `high` is what Claude Code sends by default, and mapping it to `xhigh` injects the
   model's most verbose reasoning instruction ("think carefully through the task, validate key
   assumptions, consider plausible alternatives"). The operator typed `ping` and got:

       Thought for 6m 47s
       pong

   ~5,300 reasoning tokens at the measured 13 tok/s, for a one-word answer. `medium` injects NO
   instruction at all — the model's natural behaviour — which is the right default for an
   agentic coding tool. `max` still maps to `xhigh`, so deep reasoning stays one flag away.

2. DEPTH. Benchmarked against the deployed server (--max-num-seqs 3), effort=low, 200 tokens:

       conc  wall_s  tok/s   scaling
          1    10.3   13.0   x1.00
          2    14.8   27.0   x2.08
          3    15.4   38.9   x3.00   <- linear to exactly max-num-seqs
          4    22.0   28.7   x2.22   <- throughput drops, worst latency doubles
          6    30.6   33.1   x2.55

   So concurrency is worth having right up to the slot count and not beyond, and the demotion
   threshold belongs at `max-num-seqs - 1` = 2. At 1 the LB abandoned a free slot for the ollama
   fallback, which the same day measured 258116 ms against vLLM's 476 ms on the same path.
"""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.router import rewrite_effort
from vllm_lb.state import VLLM, Backend

V = Backend(kind=VLLM, url="http://box-a:8000", host_label="box-a-gpu", box="box-a")


def test_claude_codes_default_effort_does_not_trigger_maximal_reasoning():
    body = {"output_config": {"effort": "high"}}
    assert rewrite_effort(body, V, Config())["output_config"]["effort"] == "medium"


def test_max_still_reaches_the_models_top_tier():
    body = {"output_config": {"effort": "max"}}
    assert rewrite_effort(body, V, Config())["output_config"]["effort"] == "xhigh"


def test_the_depth_threshold_matches_the_measured_knee():
    assert Config().vllm_max_depth == 2, "linear scaling held to conc=3 on a 3-slot server"
