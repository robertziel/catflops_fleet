"""THE SEALED SWARM RUNNER: the gateway's "Start test swarm" button, run where it cannot spend money.

WHY IT EXISTS. Until 2026-10-11 the button's runner (lb/bench_queue_runner.py) was a systemd user unit on
the LB's host, and every session it launched was the operator's own `claude` with the operator's own
HOME. bench_session.py's four guards keep the SESSION on the local LB, but everything around it was the
operator's: ~/.claude/.credentials.json (an OAuth subscription) within reach of a `--permission-mode
bypassPermissions` agent that reads arbitrary web pages; the user's MCP servers (40 bench sessions'
transcripts show agents calling a personal notes MCP and the browser farm's `fetch`); the global
CLAUDE.md shaping the workload; the operator's auto-memory; a CLI that auto-updated under the benchmark.
A prompt-injected `env -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL claude -p …` from one of those agents
would have billed the subscription, and no guard looks at a NESTED claude.

WHAT IS SEALED, AND HOW (one container per role, both from this directory's image):

    runner   lb/bench_queue_runner.py + lb/bench_session.py, UNMODIFIED in behaviour, as user `bench`.
             On ONE docker network, `internal: true` — no route to anything outside it. Its HOME is an
             empty directory of its own; no credential is mounted, copied or created (entrypoint.py
             refuses to start if one appears). Claude Code is a pinned binary (claude_pin.py: version,
             tarball and binary sha256, checked at build AND at every start).
    egress   the only other member of that network. It relays two fixed ports to the LB and to the
             engine's /metrics (private addresses, checked), and is the HTTPS proxy the runner's WebFetch
             uses — refusing Anthropic's names and published ranges and every private, CGNAT,
             link-local, metadata and loopback address (policy.py).

The seal is PROVEN, not assumed: egress_check.py tries to reach Anthropic directly and through the proxy,
and a LAN address through the proxy, and must be refused every time. The entrypoint runs it before the
runner starts, and the runner runs it again before EVERY session (`--seal-check`): a failure is the
request's answer (`unsealed`), and claude is never started.

Every value that differs between deployments is `${VAR:?}` in docker-compose.yml (.env.example documents
each); the wiring inside the sealed network (`lb`, `engine`, `egress`) is this directory's own.
"""
