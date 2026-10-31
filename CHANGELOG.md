# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-beta.2] — 2026-10-31

### Added
- **README screenshots in the Glass style**; the hardware view full height with every container listed.
- **Layout gallery and guide.** README shows the four layout templates side by side; `docs/guides/layout-templates.md` gives the template format, the element ids and the rules for vibe coding a template with an agent.
- **Panel styles with a deployment default.** The panel's Style menu offers `vibe` (Vibe coded), `clean` (light,
  quiet), `mono` (monospace), `glass` (translucent; follows the system light / dark) and `terminal` (green-phosphor
  CRT; reduced motion stops its cursor). Every colour, font and radius is a token; a style is one scoped sheet. Each browser's pick is kept in its `localStorage` only. The fleet file sets the default for every browser
  that has not picked one: `panel: {style: <key>}`. The LB writes it onto each page's `<html>`, so it applies before
  first paint with no extra request, and the menu tags it `default`. Picking the default clears the browser's
  choice. Setup wizard: question `panel.style`. Guide: `docs/guides/panel-styles.md`.

### Changed
- **Glass is the shipped default style.** The fleet examples, the setup answer examples and the wizard's recommendation use `panel.style: glass`; a fleet file without `panel` is refused with `style: glass` as the line to add (`vibe` keeps the original look).
- **`lb/fleet.yml` requires `panel.style`.** An existing fleet file is refused at boot until it gets these lines
  (`vibe` keeps today's look):
  ```yaml
  panel:
    style: vibe
  ```
  An unknown key is refused too, and the refusal lists the keys in `lb/vllm_lb/static/styles/index.json`.

## [1.0.0-beta] — 2026-10-30

The first public release, under Apache-2.0. It is a beta: the services are in daily production use, but the
configuration surface (fleet YAML, hardware types, the model library, the per-service `.env` files) is still settling
and may change between beta releases without a deprecation period.

### Added
- **`lb/` — the gateway (`:8010`).** One always-on load balancer in front of self-hosted vLLM / SGLang /
  TensorFold servers, speaking the Anthropic `/v1/messages` and the OpenAI `/v1/chat/completions` APIs.
  Every request is routed to a live backend; with none live it waits for one and answers 503 — never a
  stand-in model. Speech-to-text (`/v1/audio/transcriptions`) is routed by task to transcription backends only.
- **A fleet is YAML, not code.** `lb/fleet.yml` (gitignored; `lb/fleet.example.yml`, `fleet.one-host.example.yml`,
  `fleet.parallel.example.yml`, `fleet.amd-node.example.yml`) names the boxes, their cards, which card serves,
  the clusters and how each is controlled. The loader refuses a missing or unknown value by name; a retired env
  name is refused at boot naming its replacement.
- **No env number stands in for a model's or the fleet's own.** The window a broker lease declares, the seats and
  context a model is offered and the prefill budget a launch falls to come from the model library and the fleet
  file; a Start the gateway cannot size is refused naming why.
- **Engines × accelerators** (docs/design/engines.md): five axes, five libraries — model
  (`lb/vllm_lb/library/models.json`, read from config.json by `python -m vllm_lb.model_facts`), engine
  (`library/engines/*.yml` + behaviour code), accelerator (`hardware/accelerators/*.yml`: nvidia-sm121, nvidia-sm120,
  nvidia-sm86, amd-gfx1151), card type (`hardware/types/*.yml`: GB10, RTX 5090, RTX 3090, Radeon 8060S, every
  figure with its source), topology (`executors.py`: one card · TP within one host · TP across hosts). Launch
  recipes (`library/recipes.json`) and measurements keyed by engine × accelerator × TP (`library/measurements.json`)
  are separate from where a deployment staged each checkpoint (the fleet file's `checkpoints.<id>`).
- **Image matrix** `library/images/<engine>.yml`: engine × build × accelerator × platform → image. Every profile is
  resolved on the fleet's cards at load, or refused naming the missing piece; the boot log prints one `[catalog]`
  line a profile; `/lb/status` carries each profile's `resolution`. One Dockerfile per engine, BASE per accelerator
  (`deploy/images/<engine>/`, `deploy/images/build.sh`, `deploy/image-tag.sh`).
- **Unmeasured combination**: refused by default; the operator may start it explicitly sized by the config.json
  estimate (`POST /lb/cluster {"estimated": true}`, the panel's checkbox), labelled "estimated".
- **A tensor-parallel cluster from a panel.** Start / Stop a two-box TP=2 cluster over SSH (`lb/cluster/tp-rank`,
  told the topology: `CLUSTER_TP_SIZE`, `CLUSTER_NNODES`, `CLUSTER_GPUS`, `CLUSTER_LINK`, `CLUSTER_MEMORY`,
  `CLUSTER_PLATFORM`), pick the model from the library, see the per-box memory estimate before starting, and watch
  the load passes. Engines: vLLM, SGLang (+ DFlash2 draft), TensorFold native; one-host nodes also run the
  transcription engines `faster-whisper` and `nemo` behind a documented transcription-server contract.
- **Hardware telemetry on the gateway page:** per-box memory, bandwidth, compute and KV-pool tiles, modeled from
  the engine's own counters where it exposes them and labelled "not exposed" where it does not; TensorFold's
  bandwidth from a routed-expert byte model, shown as a lower bound (`≥`). GPU cards on each box (the fan turns at
  the GPU's utilisation, VRAM chips lit by used/total) and a per-box flight recorder (power, temperatures, clocks,
  CPU, network, containers; an hour of charts; `GET /lb/hardware/<box>/recorder`).
- **`lb/fleet_telemetry`**: the box-side hardware sender (stdlib only): one reader per source (nvidia-smi, amdgpu
  sysfs, /proc, thermal zones, fans, net, cgroup v2 containers), one figure table shared with the gateway's
  recorder. `lb/hw_sender` runs it.
- **Requests vs engine**: the REQUESTS card shows the gateway's and the engine's counts side by side, says why they
  differ and when the engine was read; requests that just ended stay 10 s, muted.
- **Panel layout templates** (`lb/vllm_lb/static/layout.js`, `layouts/*.json`): `default`, `compact`, `wall`,
  `column`; an edit mode, export/import as JSON. Kept in the browser's localStorage only.
- **Setup wizard for coding agents** (`SETUP.md`, `deploy/setup/questions.yml`, `render.py`): every deployment
  question asked before any file is written, a confirmed answer sheet, deterministic rendering of the config files;
  a coverage test fails when a new setting has no question.
- **Reference setups (blueprints)** (docs/design/engines.md "Blueprints", `lb/vllm_lb/blueprints.py`,
  `lb/vllm_lb/library/blueprints/*.yml`): proven setups by hardware type — a model and every knob its Start takes,
  with their source — that the setup wizard proposes as answers (`render.py --prefill`). They are reference material
  for building fleet settings, not a runtime choice: the rendered fleet holds plain settings.
- **Server-side web search for Claude Code.** The gateway answers the `web_search` server tool itself through a
  bundled SearXNG, so the native `WebSearch` works through the gateway with no client configuration.
- **A benchmark runner and swarms** (`lb/bench_runner`, `/lb/swarm/*`) with an egress policy that refuses
  LAN, CGNAT and metadata addresses.
- **No chat front-end.** A web chat is a separate project (catflops_chat); any OpenAI- or Anthropic-compatible
  client works against the gateway (`/v1/chat/completions`, `/v1/messages`).
- **`power/` — a power panel (`:8016`)**: wake-on-LAN and shutdown of the fleet's boxes, dry-run by default,
  answering only the front proxy with its secret.
- **`usage-mcp/`** — the fleet's own usage data as read-only MCP tools.
- **`deploy/`** — image recipes, staging scripts for checkpoints, a standalone one-box fleet
  (`deploy/standalone/`, systemd applier), example nginx vhosts and frpc tunnels, and `deploy/tests` (the
  publishability guards).
- **Publishability guards.** Each service's tests fail on an operator host, address, domain, home path or
  session name in shipped code, compose, examples, tests or fixtures (digest-based, so the guard itself names
  none), and on a real deployment file tracked next to its `.example`.
- `docs/guides/bring-your-own-hardware.md`.

### Security
- Every service binds `127.0.0.1` only (one documented UDP exception for display-only hardware telemetry).
- The gateway has no authentication of its own and is documented as requiring an authenticating proxy.
  Its write routes take the media type `application/json` exactly (anything else is a 415), so a cross-site page
  cannot send one as a simple request.
- Nothing ships a credential, a key or a default secret; a missing one stops the service by name.
- **Host allowlist (DNS rebinding).** Every request's `Host` must be one of the listener's loopback names at its port
  or a name in the required `LB_ALLOWED_HOSTS` (comma-separated `host[:port]`, or `loopback`), else 421; a write whose
  `Origin` is not such a name is 403. The power panel applies the same rule with `POWER_ALLOWED_HOSTS`.
