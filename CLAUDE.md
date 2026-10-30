# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**catflops_fleet** is the client side of a self-hosted LLM fleet: an always-on gateway (`lb/`, `:8010`) that
routes Anthropic `/v1/messages` and OpenAI `/v1/chat/completions` traffic to vLLM / SGLang / TensorFold
servers you run and can start a two-box tensor-parallel cluster over SSH; a power panel (`power/`, `:8016`); a
read-only usage MCP (`usage-mcp/`); image recipes, a standalone one-box fleet and the publishability guards
(`deploy/`); and a legacy `dashboard/`. This repository runs no inference engine itself. A chat front-end is a
separate project (catflops_chat): any OpenAI- or Anthropic-compatible client works against the gateway. The protocol
between the gateway, the optional broker (broker_parrot, a separate project) and the cluster is in README.md.

## Commands

```bash
# Every service is its own compose project and its own image; tests run hermetically inside that image.
cd lb && docker build -t vllm-lb:dev . && docker run --rm -v "$PWD":/app -w /app vllm-lb:dev python -m pytest
cd power && docker build -t vllm-power:dev . && docker run --rm -v "$PWD":/app -w /app vllm-power:dev python -m pytest
cd usage-mcp && docker build -t vllm-usage-mcp:dev . && docker run --rm -v "$PWD":/app -w /app vllm-usage-mcp:dev python -m pytest
docker run --rm -v "$PWD":/repo:ro -w /repo python:3.12-slim python -m unittest discover -s deploy/tests
# Browser-side tests (*.node.js): the images have no node.
docker run --rm -v "$PWD":/repo:ro -w /repo/lb node:22-slim node tests/<file>.node.js
# The lb broker tests need a broker_parrot checkout mounted read-only; without it they skip, naming why.
docker run --rm -v "$PWD/lb":/app -v /path/to/broker_parrot:/opt/broker_parrot:ro -w /app vllm-lb:dev python -m pytest
# Run a service
cd lb && cp .env.example .env && cp fleet.example.yml fleet.yml && docker compose up -d --build --wait
```

`pytest.ini` defaults to xdist `-n auto`; add `-n 0` for a serial run, a path for one test. The lb tests use a
throwaway SQLite broker DB (`tests/conftest.py` `broker_db`), never Postgres.

## How the code is organised

- Each service is a package run as `python -m <pkg>` (`lb/vllm_lb`, `power/fleet_power`, `usage-mcp/usage_mcp`,
  `lb/fleet_telemetry` — the box-side hardware sender, image + compose in `lb/hw_sender/`; the LB imports its
  `figures.py`); stdlib only at runtime. A third-party dependency goes in that
  service's Dockerfile with a comment saying why.
- **A deployment is configuration, never code.** What a fleet IS lives in YAML (`lb/fleet.yml`, gitignored;
  the hardware-type library `lb/vllm_lb/hardware/types/*.yml`, one file per type with every figure sourced).
  How it is WIRED (ports, domains, secrets) is env: each service's gitignored `.env`, `${VAR:?}` in compose,
  every name documented in its `.env.example`. One place per value. The loader fails loudly on a missing or
  unknown value: no defaults, no auto-detection.
- **No fallbacks.** When the thing asked for is unavailable the code fails with the reason (a 503, an
  exception, an empty state naming what is missing) — never a substitute model, source or number.
  A retired env name is refused at boot naming its replacement (`lb/vllm_lb/config.py` `RETIRED_KEYS`).
- **Five axes, five libraries** (docs/design/engines.md): model (`lb/vllm_lb/library/models.json`, read by
  `python -m vllm_lb.model_facts`), engine (`library/engines/*.yml` + its behaviour code), accelerator
  (`hardware/accelerators/*.yml`), card type (`hardware/types/*.yml`), topology (`executors.py`). Images are the
  matrix's (`library/images/*.yml`), measurements are keyed engine × accelerator × TP, checkpoint folders are the
  fleet file's `checkpoints`. `deploy/images/<engine>/` builds one Dockerfile per engine, BASE per accelerator.
- **Hardware behaviour is a class per family** (`lb/vllm_lb/hardware`: unified vs discrete memory). Code
  outside `hardware/` never branches on a memory family or names a hardware type; a new type is a new YAML
  file plus a test pinning its sources, and a class only for a new behaviour family.

## Conventions

- TDD: the failing test first, then the code. DRY: one abstraction before a second implementation.
- Publishability is tested: `deploy/tests/test_deploy.py`, `lb/tests/test_publishable.py` and
  `lb/tests/test_hardware_guard.py` fail on an operator host, address, domain, home path, session name or
  hardware literal in shipped code, compose, examples, tests or fixtures. Real deployment files sit next to a
  committed `*.example.*` and are gitignored.
- Commits: conventional, lowercase, scope = service or subsystem — `feat(lb/cluster): …`, `fix(power): …`.
- Heavy WHY-docstrings on every module; snake_case everywhere.

## Pitfalls

- `lb/Dockerfile` rewrites static asset refs with content hashes and fails if a stamp misses: rename a JS/CSS
  ref in both the HTML and the stamp loop.
- `vllm-lb` has `stop_grace_period: 960s`: a stop or redeploy drains in-flight streams by design.
- `lb/state/` is runtime state (cluster intent, snapshots), not source.
- The gateway's backends are whatever the fleet file routes to (SGLang in the reference fleet), not necessarily vLLM.

## Setting a deployment up

Follow `SETUP.md`: every question in `deploy/setup/questions.yml` is asked before any file is written; the answer sheet
(`deploy/setup/answers.yml`, gitignored) is rendered by `deploy/setup/render.py`. A new env var or fleet key needs a
question there (`deploy/tests/test_setup.py` fails without one).

