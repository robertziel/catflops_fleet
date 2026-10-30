# vllm-lb — the always-on LB, leasing vLLM from broker_parrot when LB_BROKER_ENABLED=1.
# The engine (queue_workflows) is NOT baked: with the broker on it is bind-mounted read-only
# at /opt/broker_parrot by docker-compose.broker.yml (the same NFS checkout the GB10 boxes'
# box-agents run), so the LB always speaks the schema the fleet is on. Off — the default —
# nothing imports it and /opt/broker_parrot on PYTHONPATH is simply absent. psycopg is the
# engine's one hard dep.
FROM python:3.12-slim
# openssh-client: the cluster loop runs this image's cluster/tp-rank on each GB10 box over SSH.
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*
# PyYAML is a RUNTIME dependency (since 2026-10-11): what a deployment IS — its fleet file and the hardware-type
# library — is YAML (AGENTS.md: a deployment is configuration, never code), read strictly by vllm_lb/yamlfile.py.
# PINNED: a parser that changes what a configuration MEANS is chosen, never pulled. 6.0.3 is the version this image
# already resolved for the suite (tests/test_search_engines.py reads lb/searxng/settings.yml with it).
RUN pip install --no-cache-dir pyyaml==6.0.3
# psycopg is the engine's one hard dep. pytest is the SUITE's.
# pytest-xdist: the suite runs on every core (`-n auto`, pytest.ini) — each test's broker DB is its own
# tempfile and every listener binds port 0, so nothing is shared between workers.
RUN pip install --no-cache-dir "psycopg[binary,pool]>=3.1" pytest pytest-xdist
WORKDIR /app
COPY . /app
# CACHE BUSTING: every build stamps the pages' script and stylesheet references with the files' content
# hash ("/gateway.js?v=<sha256[:12]>"), so a browser — or the vhost in front of the tunnel — that cached
# /gateway.js under the old URL fetches the new build under a new one. The static handler serves by path
# and ignores the query (proxy.py `handle`: `urlsplit(h.path).path`). The operator saw the previous
# build's gateway.js after a deploy despite `Cache-Control: no-cache` on every file (2026-10-01). The
# build FAILS if a stamp did not land, so a renamed reference cannot quietly ship unversioned.
RUN cd /app/vllm_lb/static \
    && for f in clusters.js gateway.js gateway.css layout.js layout.css benchmarks.js benchmarks.css sessions.js sessions.css server-scope.js recorder.js recorder.css; do \
         v=$(sha256sum "$f" | cut -c1-12) \
         && sed -i "s#\"/$f\"#\"/$f?v=$v\"#g" index.html benchmarks.html sessions.html; \
       done \
    && grep -q 'gateway.js?v=' index.html && grep -q 'gateway.css?v=' index.html \
    && grep -q 'clusters.js?v=' index.html \
    && grep -q 'recorder.js?v=' index.html && grep -q 'recorder.css?v=' index.html \
    && grep -q 'layout.js?v=' index.html && grep -q 'layout.css?v=' index.html \
    && grep -q 'benchmarks.js?v=' benchmarks.html && grep -q 'benchmarks.css?v=' benchmarks.html \
    && grep -q 'gateway.css?v=' benchmarks.html \
    && grep -q 'sessions.js?v=' sessions.html && grep -q 'sessions.css?v=' sessions.html \
    && grep -q 'gateway.css?v=' sessions.html && grep -q 'benchmarks.css?v=' sessions.html \
    && grep -q 'server-scope.js?v=' benchmarks.html && grep -q 'server-scope.js?v=' sessions.html
ENV PYTHONPATH=/app:/opt/broker_parrot PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8010
CMD ["python", "-m", "vllm_lb"]
