# Setup wizard — the rules a coding agent follows

For Claude Code, Cursor, Codex or any coding agent asked to set catflops_fleet up. Read this whole file
before you run anything.

This project has no defaults and no auto-detection (AGENTS.md, "A deployment is configuration, never code").
Every deployment value is a deliberate choice: in YAML (`lb/fleet.yml`, the hardware types) or in env (each
service's `.env`, documented in its `.env.example`). So you **collect every choice from the user before you
write a single file**, from a fixed list of questions, and render the files from the answers with a script.
You never type a config file by hand and never fill a gap with a guess.

| file | what it is |
|---|---|
| `deploy/setup/questions.yml` | **the question library**: 192 questions in 25 sections, in asking order. Each one has `ask`, `why`, `kind`, `choices` (or the library they are read from), `validate`, `when` (the condition on earlier answers), `writes` (the exact file and key it fills), `secret` and an optional read-only `probe`. |
| `deploy/setup/answers.yml` | **the answer sheet** (gitignored, mode 600): one key per question id. Start from `answers.example.yml` (the pair shape) or `answers.standalone.example.yml` (one box). |
| `deploy/setup/render.py` | **the renderer**: validates the sheet against the library and writes the files. Standard library only. |
| `deploy/tests/test_setup.py` | the guard: every required key of every `.env.example` and compose `${KEY:?}`, and every key of every fleet example, is written by some question. A new setting without a question fails the suite. |

## Phase 0: read, then probe (read-only)

1. Read `README.md`, `AGENTS.md`, this file, `deploy/setup/questions.yml` and the examples its `files:`
   section names (`lb/.env.example`, `lb/fleet.example.yml` and its siblings, `power/.env.example`, …).
2. You may run a question's `probe` (on the machine the question is about) to **propose** an answer.
   Probes only read: `nvidia-smi`, `df`, `uname -m`, `docker info`, `ss -ltn`, `ip -brief addr`. Never
   install, pull, start, write, or `ssh` into a machine the user has not named.
3. A proposal is not an answer. If a detected value were written without the user confirming it,
   auto-detection would become config, and that is a fallback.

## Phase 1: ASK

- Ask section by section, in the library's order. Skip a question when its `when` is false. A `per`
  question is asked once for each instance (each box, cluster, power box, sender, new accelerator, new
  hardware type, new model or staged model).
- Put each question as its `ask` says. Give the `why` when the user wants it, list the `choices` (read from
  the library each one names: `render.py` `SOURCES`, never a list typed from memory), and show a probe's
  output as "proposed: … — confirm?".
- **Never invent a value.** An unanswered required question stops the parts that need it. Say which, e.g.
  "no models directory for box-b: its rank cannot start". Then carry on with the rest.
- A `secret` (`search.searxng_secret`, `broker.pg_pw`, `download.hf_token`, `exposure.frp_token`,
  `crawl.token`): you may offer to generate one (`openssl rand -hex 32`). Never echo it, never put it in a
  commit, a log or a chat summary.
- Write the answers into `deploy/setup/answers.yml` as you go. On a re-run, read it first and ask only what
  is missing or what the user wants to change.

The sections (`questions.yml` `sections`, in asking order):

| section | asks | n |
|---|---|---|
| `shape` | gateway in front of existing servers / one-box standalone / two-box TP pair / a mix; where the gateway runs | 2 |
| `machines` | each machine: name, roles, hostname, label, CPU platform, address, SSH login, a node's container limits, RAM bounds | 13 |
| `cards` | every GPU by index: its hardware type from `lb/vllm_lb/hardware/types/`, and whether it serves (its accelerator is the type's: shown, not asked) | 1 |
| `accel` | a kernel target the library lacks: the accelerator sub-wizard (arch, GPU runtime, platforms, precisions, quant formats, each sourced) | 7 |
| `hwtype` | a card the library lacks: the hardware-type sub-wizard (its accelerator, every figure with its source) | 14 |
| `runtime` | GPU driver stack (nvidia / rocm; checked against the serving cards' accelerator), Docker with Compose v2, the NVIDIA Container Toolkit | 3 |
| `broker` | broker_parrot on/off; its checkout, network and Postgres; a client network | 7 |
| `models` | task (chat / transcription), engine (`library/engines`), quantisation, speculative decoding, models the library lacks, **which models this deployment stages** | 6 |
| `newmodel` | a model the library lacks: its facts (`model_facts`), recipe (engine, build), measured or estimate-only, limits | 16 |
| `clusters` | each cluster: control (ssh / local / none), ranks, **start from a reference blueprint?** (prefills what follows), URL, boot and default; **which model** it serves or the pair's first Start and **its figures**, and **an unmeasured first Start: measure first or start on the estimate** | 13 |
| `pair` | the TP pair: NIC, **link** (roce / ethernet), RAM share, prefill budget, KV dtypes, SSH key, host keys, images, RoCE devices, worker unit | 12 |
| `node` | each one-host node: pinned image, model choices, **context length**, **concurrency**, memory share, parsers, **KV dtype**, its server file, and a standalone server's own lines | 29 |
| `backends` | the URL of each classic server already running | 1 |
| `storage` | **the models directory** on each machine, its free space against the models' sizes, ownership, NFS shared or own copy, the search path, and **each staged model's checkpoint folder** (its uncensored twin, its drafter) | 8 |
| `download` | staged or download now, which checkpoints, the download directory, gated, **the Hugging Face token** | 5 |
| `network` | the ports `lb/` fixes are free | 1 |
| `exposure` | localhost / LAN / internet; the proxy, **its sign-in** (there is no "none"), domain, TLS, frp | 10 |
| `search` | the SearXNG secret; web search on/off | 2 |
| `power` | power panel on/off, origin, dry run, the gate secret, and per machine: SSH, Wake-on-LAN, clock cap, the panel's words | 16 |
| `telemetry` | hardware senders: port, which machines, GPU kind, bridged network | 6 |
| `bench` | the sealed swarm runner and its networks | 8 |
| `extras` | usage-mcp (build / register), the Crawl4AI service, the legacy dashboard | 8 |
| `panel` | the gateway panel's default style (`lb/fleet.yml` `panel.style`: one of `lb/vllm_lb/static/styles/index.json`; docs/guides/panel-styles.md) | 1 |
| `tuning` | any gateway knob changed from what `lb/.env.example` documents | 1 |
| `clients` | which clients connect and from where (sets up the smoke test) | 2 |

There is no time-zone question: no service reads one (usage-mcp mounts the host's `/etc/localtime`).

### The three sub-wizards

The libraries are split by owner (docs/design/engines.md): a card TYPE is one product, its ACCELERATOR what its
kernels target; a MODEL is its facts, its recipe and its measurement, while where its checkpoint sits is this
deployment's (`checkpoint.*`, rendered into `lb/fleet.yml` `checkpoints`).

- **An accelerator** (`accel`): a kernel target that is not in `lb/vllm_lb/hardware/accelerators/` (a new
  architecture; a new card of a known one needs none). Arch, GPU runtime, platforms, precisions and quant
  formats, each `{value, source}`. Written by hand with its test; images, measurements and recipe overrides are
  keyed by it.
- **A hardware type** (`hwtype`): a card that is not in `lb/vllm_lb/hardware/types/`. Its answers are a new
  type file naming its accelerator, and every figure needs a source: a vendor page, or a measurement with its
  date. When no source states a figure, it is `null` plus the reason. You write it **by hand** together with its
  test in `lb/tests/test_hardware_profiles.py` (AGENTS.md "Hardware"). It is a contribution, so the renderer
  never writes it.
- **A model** (`newmodel`): a model that is not in `lb/vllm_lb/library/models.json`. Its facts are read from
  its own files (`python -m vllm_lb.model_facts <folder>`), its recipe names an engine and a build (the image is
  the matrix's), and it is either measured on your engine × accelerator × TP (a `measurements.json` entry) or
  started on the estimate, explicitly, at each Start. You write the library entries by hand;
  `lb/tests/test_model_catalog.py` must pass. Its folder is a `checkpoint.path` answer like any staged model's.

### Reference blueprints: a starting point, prefilled

A **reference blueprint** (`lb/vllm_lb/library/blueprints/<id>.yml`; docs/design/engines.md "Blueprints") is one of
this project's own production setups by hardware type: a model and every knob its Start takes, with the source of its
numbers. It is yours to start from, never a key of the files you render, and the gateway never reads one.

1. Right after a cluster's control and ranks, ask `cluster.blueprint` ("start from a reference blueprint?"). List the
   ones that fit with `render.py <answers.yml> --prefill`: a blueprint fits where its target's control, TP (the ranks'
   serving cards) and accelerator are the cluster's answered ones; it names why each other one does not.
2. On a choice, run `--prefill` again. It prints each question the blueprint answers (`questions.yml` `prefill`) as an
   answer-sheet line, beside what the sheet says now: the model and choices, a node's launch block or a pair's
   `cluster.first_start_settings`, the KV dtypes, the parsers, the box's RAM tier (raised, never lowered), the model
   added to `models.staged`. A knob the blueprint leaves `null` is the engine's own: leave that question unanswered.
3. Put each line to the user as "proposed: … — confirm?", exactly like a probe's proposal, and write only what they
   confirm or edit. The image (`cluster.image`) and every folder (`checkpoint.*`) stay their answers: a blueprint
   names neither, and a node's KV dtype is taken only on an image `lb/vllm_lb/kv_cache_dtypes.json` records.

`--prefill` reads a sheet that is still being asked and writes nothing. A blueprint that does not fit its cluster is
refused at `--check`, naming why; in the vllm-lb image `--check` also holds a pair's first Start (its model and
`cluster.first_start_settings`) to the switch's own validator.

### Shown, never asked

`--check` prints, under "DERIVED FROM THE LIBRARIES": each card's **accelerator** (its type's: arch, GPU
runtime), and, in the vllm-lb image, each ssh cluster's **first Start as the gateway's resolver reads it**: the
recipe's engine and build, the target (accelerator × platform × TP), the image matrix's image, and whether the
model is measured there. `cluster.first_start` is answered exactly where that says `unmeasured`.

## Phase 2: CONFIRM

```sh
docker run --rm -v "$PWD":/repo:ro -w /repo python:3.12-slim \
  python deploy/setup/render.py deploy/setup/answers.yml --out . --check
```

`--check` validates the sheet and writes nothing. It refuses, naming each problem: a missing required
answer (it names the question and the instance), an id the library does not have, a choice outside its
library, an answer to a question its `when` skips, or a box whose driver stack is not its serving cards'
accelerator's. In the vllm-lb image it also reads the rendered fleet with the gateway's loader and resolves each
pair's first Start (no image for it, or a sizing answer where it does not apply, is refused). Fix the sheet with
the user until it passes.

Then show the user what it prints: the **whole answer sheet with every secret masked**, the list of files
it will write with their keys, and the hand-written files from the sub-wizards. **Wait for an explicit
yes.**

## Phase 3: WRITE

1. Build the gateway's image: `docker build -t vllm-lb:dev lb`. The renderer runs in it, so it can read
   the rendered `lb/fleet.yml` with the gateway's own loader, and so a node's lines come from
   `python -m vllm_lb.fleet node-env`. Outside that image it refuses a node's server file and names this
   command:
   ```sh
   docker run --rm -v "$PWD":/repo -w /repo vllm-lb:dev \
     python deploy/setup/render.py deploy/setup/answers.yml --out .
   ```
2. It writes `lb/.env`, `lb/fleet.yml`, `power/.env`, `deploy/standalone/<name>.env`,
   `lb/bench_runner/.env`, `crawl/.env` and the root `.env`, whichever the answers fill. Env files get
   mode 600. Every one is gitignored. **Never commit them.**
3. **It never overwrites a file.** For a file that exists, run with `--diff` (secrets masked), show the user
   the diff, and only on their yes re-run with `--replace <file>` for that file.
4. `hosts/<box>/…` (gitignored) holds files that belong in **another machine's** checkout, e.g. each
   telemetry sender's `lb/hw_sender/.env`. Copying them there touches another machine, so ask first.
5. The pair (`pair.ssh_key`, `pair.known_hosts`): generate `lb/cluster-ssh/id_ed25519` (gitignored) or copy
   the key the user names. Pin the host keys the way they chose; you never read `~/.ssh` on your own.
   Authorising the key on the boxes (`ssh-copy-id`) is the user's step, or yours after their yes.
6. Power (`power.gate_secret`): `~/.config/fleet-power/` is outside the repo, so write there only after the
   user's yes. `power/install/host-setup.sh` runs with sudo on each box: the user runs it.
7. The sub-wizards' files, by hand, each with its test.

## Phase 4: VALIDATE, BOOT, VERIFY

1. **Loaders fail loudly; let them.** Phase 3 already read the fleet with the gateway's loader.
   `cd lb && docker compose config -q` checks every `${KEY:?}`. A standalone server runs
   `deploy/standalone/up.sh --check`. A refusal goes back into the sheet; never edit around it.
2. **Checkpoints** (`download.mode: download-now`): `deploy/stage-<model>.sh fetch`, then `stage` for a
   pair. Every file is verified against the pinned commit. Check `box.models_free_gib` against the
   model library's `size_gib` first.
3. **Boot**: `cd lb && docker compose up -d --build --wait`. A standalone server runs
   `deploy/standalone/up.sh`, then `install-node-applier.sh --cluster <name> --env <its server file>`.
   Each sender: `cd lb/hw_sender && docker compose up -d --build` on its box. Power:
   `cd power && docker compose up -d --build`. usage-mcp: build it, and register it only if
   `extras.usage_mcp` says so.
4. **Verify**:
   - `curl -s http://127.0.0.1:8010/lb/status`: every backend and cluster the fleet names.
   - The pair: `GET /lb/clusters/<name>/estimate?model=<id>&max_num_seqs=<n>`, then
     `POST /lb/clusters/<name>` with the first model from `cluster.first_model` (`model_choice`, and `uncensored`:
     true for an `@uncensored` ref, false otherwise), every `cluster.first_start_settings` key as it is, and
     `"estimated": true` only where `cluster.first_start` is `start-on-estimate` (`measure-first` means no Start
     until it is measured); wait for `running`.
   - **One streamed completion per API** the clients use: `POST /v1/messages` with `"stream": true`
     (Claude Code, Anthropic SDK) and `POST /v1/chat/completions` with `"stream": true` (OpenAI SDK).
     Show the user the first chunks.
5. **Tests**: the lb suite and the deploy guards in Docker (CLAUDE.md "Commands"), both green.
6. **The reverse proxy, last**, and only when `exposure.scope` is not `localhost`. Configure it with the
   sign-in from `exposure.auth`, then show the user that a request without credentials is refused from
   outside. Until then everything stays on 127.0.0.1.

## Never

- Fill in a default, a detected value or a "sensible" number the user did not confirm. That is a fallback.
- Echo, log or commit a secret, or the answer sheet.
- Touch a file outside the repository, or another machine, without asking.
- Expose a port beyond localhost without an authentication layer in front. The gateway has none of its
  own, and its control endpoints change what runs on your GPUs.
- Write a RETIRED setting (`lb/vllm_lb/config.py` `RETIRED_KEYS`): the gateway refuses to boot with one,
  and `lb.tuning` refuses it too.
- Overwrite an existing file without the user approving its `--diff`.

## Open items

- **A one-host node's image** (`cluster.image`) is still a pinned answer, not the image matrix's: node
  launches keep their fleet `launch.image` until the matrix covers them (docs/design/engines.md, Open items).
  So is a node's fit against `covers`: `cluster.first_start` applies to an ssh cluster's first Start only.
- **The first Start's resolution needs the vllm-lb image**: in python:3.12-slim `--check` cannot say whether
  the pair's first model is measured on its cards, so whether `cluster.first_start` applies is decided there.
- **Fixed ports**: `lb/docker-compose.yml` publishes 8010 and 8811, and `power/` publishes 8016. They are
  not configurable yet, so `network.lb_ports_free` can only check that they are free.
- **Gated downloads**: `deploy/stage-checkpoint.sh` fetches anonymously, so `download.hf_token` has no
  consumer yet. A gated checkpoint stops at the download step until the script sends the token.
- **Cross-answer rules the renderer leaves to the loaders**: whether a box's roles match the ranks, a model
  fits its card, a parser suits its model, a staged model's folder exists. The gateway's loader refuses these,
  naming the value, when it reads the rendered fleet (or, for a folder, at Start).
- **A tree that also ships the signed-in edge** asks for the edge's lines below `deploy/standalone/.env`'s
  `GATE-BELOW` marker on a full `up.sh`. This wizard does not cover them: put your own authenticating proxy
  in front (`exposure`).
