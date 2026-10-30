# vllm-bench-runner: the gateway's swarm runner, SEALED (bench_runner/__init__.py says why and how).
# One image, two containers (docker-compose.yml): `runner` (bench_queue_runner.py + bench_session.py +
# the pinned Claude Code, as user bench) and `egress` (the proxy and the two relays, as nobody).
#
# Built from lb/ (the context is the directory above), and Dockerfile.dockerignore admits only the files
# COPY'd below: lb/.env, lb/cluster-ssh/ and lb/state/ never reach the builder.
#
# Pinned by digest: the base (python:3.12-slim as of 2026-08) and, in claude_pin.py, Claude Code itself.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
ENV PYTHONPATH=/opt/bench PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The one network step: npm's linux-x64 Claude Code tarball, refused unless both its sha256 and the
# binary's match claude_pin.py. A layer of its own, so a code change does not download 115 MB again.
COPY bench_runner/__init__.py bench_runner/claude_pin.py bench_runner/install_claude.py /opt/bench/bench_runner/
RUN python -m bench_runner.install_claude
# Managed settings: what no user setting can override. WebFetch's domain preflight asks Anthropic, which the
# egress refuses, so it is skipped; the auto-updater and the nonessential traffic would only be refused too.
COPY bench_runner/managed-settings.json /etc/claude-code/managed-settings.json
# The account the sessions run as: uid 1000, no shell, no home of its own in the image (the entrypoint
# gives it one inside the mounted state directory).
RUN groupadd --gid 1000 bench \
    && useradd --uid 1000 --gid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin bench
COPY bench_queue_runner.py bench_session.py /opt/bench/
# The engines' metric vocabularies, which bench_session reads /metrics through: the one module of the LB's here
# (stdlib only, no config or state), so the swarm and the probe never disagree on what an engine is.
COPY vllm_lb/__init__.py vllm_lb/engine_metrics.py /opt/bench/vllm_lb/
COPY bench_runner/ /opt/bench/bench_runner/
# The seal check is exec'd by the runner (`--seal-check`), so it must be executable; and the build fails
# if any module the two containers run does not import.
RUN chmod 0755 /opt/bench/bench_runner/egress_check.py \
    && python -c "import bench_queue_runner, bench_session, bench_runner.egress, bench_runner.egress_check, bench_runner.entrypoint, bench_runner.policy" \
    && /opt/claude/bin/claude --version
ENTRYPOINT ["python", "-m", "bench_runner.entrypoint"]
