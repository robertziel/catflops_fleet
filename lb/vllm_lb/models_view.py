"""`GET /v1/models`, synthesized: EVERY model a request naming it would be routed to right now, each once, with the
server it would reach (`server`: its cluster, or a classic server's box label) and that server's context window
(`max_model_len`). THE DEFAULT ROUTE'S MODELS COME FIRST — what a request naming no model reaches: fullclaudevllm sizes
Claude Code's context from `data[0]`, and the chat budgets an unnamed turn to the first entry. "Would be routed" is the
router's own rule (`choose`: held, its own cluster view, leases), never a second definition of what is running.
Nothing a request would reach: no model is listed — never a stand-in (the fallback is retired). Live 2026-10-17: the
list was the default route's alone, so a client discovering models never saw the gui node's MAI-UI."""
from __future__ import annotations

from vllm_lb.config import Config
from vllm_lb.router import choose, task_of
from vllm_lb.state import CLUSTER, VLLM, Snapshot


def models_payload(snap: Snapshot, cfg: Config) -> dict:
    listed: dict[str, object] = {}                  # model id -> the backend a request naming it reaches

    def add(ids, backend) -> None:
        for i in ids:
            listed.setdefault(i, backend)

    default = choose(snap, cfg)
    if default is not None and default.kind in (VLLM, CLUSTER):
        add(default.served_models, default)
    for b in snap.backends:
        if b.kind in (VLLM, CLUSTER):
            for m in b.served_models:
                if m not in listed and (reached := choose(snap, cfg, model=m, task=task_of(b, cfg))) is not None:
                    add((m,), reached)
    return {
        "object": "list",
        "data": [{"id": i, "object": "model", "owned_by": "vllm-lb", "max_model_len": b.max_model_len,
                  "server": b.cluster or b.host_label}
                 for i, b in listed.items()],
    }
