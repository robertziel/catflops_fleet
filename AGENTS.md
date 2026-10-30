# catflops_fleet

The client side of a self-hosted LLM fleet: the always-on load balancer (`lb/`, :8010) that routes to and
can start vLLM / SGLang / TensorFold servers on the boxes a fleet file names, power control (`power/`, :8016),
a read-only usage MCP (`usage-mcp/`), image recipes and the standalone one-box fleet (`deploy/`), and a legacy
`dashboard/`. Every service binds 127.0.0.1: the authenticating proxy in front is the deployment's own, never
this repository's. This repo runs no inference engine itself, and ships no chat front-end: that is a separate
project (catflops_chat), and any OpenAI- or Anthropic-compatible client works against the LB. See README.md for
the lease/cluster protocol.

## Services & deploy
Each service is its own compose project: `cd lb && docker compose build && docker compose up -d --wait`
(same in `power/`). All images are `python:3.12-slim`, stdlib-first; runtime deps are
installed in the Dockerfile, test tools too — a full suite runs hermetically inside the image.
All published ports bind 127.0.0.1 only (one deliberate UDP exception in lb/ — see its compose
comments). `deploy/` holds example edge vhosts + frpc configs; don't widen a bind or route without
checking them.

## Sibling projects (optional)
- **broker_parrot** — OPT-IN: `LB_BROKER_ENABLED=1` + `COMPOSE_FILE=docker-compose.yml:docker-compose.broker.yml`
  in lb/.env, the checkout mounted read-only at `LB_BROKER_CHECKOUT`. Run lb compose FROM `lb/` with no `-f`,
  or the override is ignored. Off (the default), the engine is never imported.

## Test — docker, exact invocation (README.md "Tests")
```
# lb — mount broker_parrot to run the broker tests (LB_BROKER_ENABLED=1); without it they SKIP, naming why
cd lb && docker build -t vllm-lb:dev .
docker run --rm -v "$PWD/lb":/app \
  -v /path/to/broker_parrot:/opt/broker_parrot:ro \
  -w /app vllm-lb:dev python -m pytest
# power / usage-mcp: no broker mount needed
docker run --rm -v "$PWD/power":/app -w /app vllm-power:dev python -m pytest
# the publishability + deploy guards
docker run --rm -v "$PWD":/repo:ro -w /repo python:3.12-slim python -m unittest discover -s deploy/tests
```
pytest.ini defaults to xdist `-n auto`; add `-n 0` for serial, a path for one test. lb tests
use a throwaway sqlite broker DB via the `broker_db` fixture (tests/conftest.py) — never
Postgres. `*.node.js` tests: the image has no node; run under
`docker run --rm -v "$PWD":/lb:ro -w /lb node:22-slim node tests/<f>.node.js` (the paired
python test skips and names the command when node is absent).

## Conventions (observed)
- TDD, test-first: write the failing test, watch it fail, then the code.
- DRY: one engine abstraction before adding a second implementation. Engine code = an
  `EngineCore` base class; per-backend engines inherit it.
- Stdlib only at runtime; a new third-party dep goes in the Dockerfile with a comment saying
  why (see lb/Dockerfile's psycopg line).
- Module layout: each service is a package (`lb/vllm_lb`, `power/fleet_power`, `lb/fleet_telemetry`, ...) run as
  `python -m <pkg>`; snake_case everywhere; heavy WHY-docstrings on every module.
- Commits: Conventional-ish single-line, lowercase, scope = service or subsystem —
  `feat(lb/cluster): ...`, `fix(power): ...`. What changed and why, terse.

## A deployment is configuration, never code
This software must run on anyone's hardware and topology. Nothing about one deployment is hardcoded
— not a host, IP, domain, port, card, cluster size or model choice. Every new feature follows this;
an existing hardcoded case is a bug to migrate, not a precedent.
- What a fleet IS lives in YAML: hardware TYPES (one file per type: memory model, VRAM, peaks, cores,
  arch — every number with its source, or absent and said so) and one FLEET file per deployment
  (its boxes, their cards and which card serves, its clusters and their topology/TP, how each is
  controlled). One loader validates them and FAILS LOUDLY on a missing or unknown value: no
  defaults, no auto-detection, no fallbacks.
- How a deployment is WIRED (ports, domains, issuer, secrets) is env: per-service `.env` files,
  `${VAR:?}` in compose, every name documented in `.env.example`. A value lives in exactly ONE place
  (YAML or env, never both).
- Code works for ANY hardware type through the hardware profile abstraction: one base, one class per
  behaviour family (e.g. unified vs discrete memory). A new type is a new YAML file — plus a class
  only if it behaves differently — never an `if <model>:` branch.
- Commit generic examples (`*.example.*`, the hardware-type library); real deployment files are
  gitignored next to them. No host, IP, domain, home path or person's name in code, compose, tests,
  fixtures or examples — the publishability guards enforce it: `deploy/tests/test_deploy.py`,
  `lb/tests/test_publishable.py`, `lb/tests/test_hardware_guard.py`.
  The guards are digest-based (they name no real token); a new class of leak gets a planted violation
  that fails first.
- A retired env name is REFUSED at boot naming its replacement (lb/vllm_lb/config.py `RETIRED_KEYS`),
  never silently ignored.
- Two fleet shapes ship as examples: a pair fleet (an LB box + two boxes running one TP=2 cluster,
  `lb/fleet.example.yml`) and a standalone fleet (LB + one discrete-GPU node; `deploy/standalone/`, one
  server file + `up.sh`). Neither depends on the other.
- TDD, DRY and DDD for every feature: the failing test first; one definition per fact; the domain
  (hardware, fleet, model catalog) kept apart from transport and UI.

### Hardware — where it lives, how to add one (lb/vllm_lb/hardware, lb/vllm_lb/fleet.py)
- FACTS are data: `lb/vllm_lb/hardware/types/<key>.yml`, one per type (gb10, rtx5090, rtx3090, radeon8060s):
  `memory` family, its `accelerator`, `controls`, and `facts` (memory_gb, cuda_cores, peak_gbps, bandwidth_counter,
  host_platform, peak_tflops per precision), each `{value, source}` — a vendor spec page or a measurement
  with its date and record; `value: null` + why where no source states it. Peaks are DENSE.
- WHAT KERNELS TARGET is an ACCELERATOR: `lb/vllm_lb/hardware/accelerators/<key>.yml` (nvidia-sm121, nvidia-sm120,
  nvidia-sm86, amd-gfx1151): arch, gpu_runtime, platforms, precisions, quant_formats — one place per value; a type
  reads `arch`/`gpu_runtime` through it. Images, measurements and recipe overrides are keyed by accelerator, never
  by product (docs/design/engines.md).
- MODELS are a library, not a catalog file: `lb/vllm_lb/library/models.json` (facts read from config.json —
  `python -m vllm_lb.model_facts`), `recipes.json` (engine + build, never an image or a path), `measurements.json`
  (keyed engine × accelerator × TP), `images/<engine>.yml` (the image matrix), `engines/<engine>.yml`. WHERE a
  checkpoint sits is the fleet file's `checkpoints.<id>`. A moved key (`path`, `image`, `memory`, `tuned` …) is
  refused naming its new home; a leftover `catalog.json` refuses the catalog. A profile resolves per deployment or
  is refused naming the missing piece; an unmeasured target is refused unless the operator chooses the estimate.
- TOPOLOGY is facts, not tp-rank constants: `lb/vllm_lb/executors.py` derives TP, nodes, GPUs, `link` (fleet
  `clusters.<name>.link`), memory family and platform; the pair's argv is pinned by `tests/test_pair_launch_golden.py`.
- BEHAVIOUR is a class per family: `UnifiedProfile`/`DiscreteProfile` + `UnifiedMemory`/`DiscreteMemory`
  (how a box's GPU memory is read, its `name`, and its machine-tile `rows`). Callers ask the model;
  nobody outside `hardware/` branches on `.unified` or names a type.
- WHICH box has which cards and which SERVES is the fleet file (`lb/fleet.yml`, gitignored, mounted
  read-only at `LB_FLEET`; examples `lb/fleet.example.yml`, `lb/fleet.one-host.example.yml`), and each box's
  `platform` (linux/amd64 | linux/arm64 — the image's CPU, never inferred; a type stating `host_platform` must
  agree). A cluster's `control` is data (`fleet.CONTROLS`: ssh pair / local node / none) — its rules read the
  control's fields, never its name; a local node's intent is checked by `fleet.validate_launch`. The card a
  server runs on is asked in ONE place — `topology.serving_cards` / `serving_card` / `serving_profile` /
  `serving_cuda_cores` / `cluster_model` — never "the busiest card". Peaks are per serving box
  (`saturation.per_box`); a box's share of the engine's one rate is DERIVED and labelled so. A model's
  compute precision is its shape's (`cores.SHAPES` `precision`), never an LB-wide setting.
- A NEW HARDWARE TYPE: (1) `types/<key>.yml` with every figure sourced or null + why; (2) a test in
  `lb/tests/test_hardware_profiles.py` pinning the source's own figures; (3) a new class ONLY for a new
  behaviour family (`FAMILIES`), with its tests — never an `if <type>:`; (4) name it on a box in the fleet
  file. `python -m vllm_lb.fleet node-env <file> <cluster>` renders a standalone node's compose values.
- GUARDS (`lb/tests/test_hardware_guard.py`) fail on a hardware identity or figure as a literal, a branch
  on a memory family outside `hardware/`, or a panel that asks which kind of box it draws. Its figures are
  derived from the type library; prove a new scanner rule against a planted violation, as it does.

## Pitfalls
- `lb/.env` is gitignored and holds secrets (PG creds, the SearXNG secret); never print or commit it.
- `lb/state/` is runtime state (cluster intent, snapshots) — not source; `state-audit/` is
  scratch and gitignored.
- A mounted broker engine (`/opt/broker_parrot`) is read-only by design — never edit it from here.
- lb/Dockerfile build rewrites static asset refs with content hashes and FAILS if a stamp
  misses; rename a JS/CSS ref in BOTH the html and the stamp loop.
- vllm-lb `stop_grace_period: 960s` (compose): a stop/redeploy drains in-flight streams —
  don't "fix slowness" by killing the grace period.
- The gateway's backends are whatever the fleet file routes to (SGLang in the reference fleet), not necessarily vLLM.

## Setting a deployment up: SETUP.md
An agent asked to set the fleet up follows `SETUP.md`: ask every question of `deploy/setup/questions.yml`
(read-only probes only PROPOSE answers), show the answer sheet with secrets masked, wait for a yes, then render the
gitignored files with `deploy/setup/render.py` (it never overwrites without `--diff` + `--replace`) and boot. A new
setting in any `.env.example`, compose `${KEY:?}` or fleet key needs its question: `deploy/tests/test_setup.py`
fails without one.

