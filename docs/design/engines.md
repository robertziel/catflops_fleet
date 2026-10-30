# Engines × accelerators — the design

Status: implemented on `engine-accelerators` (operator-approved 2026-10-30). Goal: catflops_fleet runs on anyone's
hardware out of the box — a user writes only `lb/fleet.yml` (boxes, cards, how each is reached, where checkpoints sit).

## The five axes

What used to be glued together in one catalog row (and in tp-rank's DGX Spark constants) is five independent axes,
each its own committed library:

| axis | what | where | form |
|---|---|---|---|
| Model | what the checkpoint is | `lb/vllm_lb/library/models.json` | facts READ from its `config.json` (`python -m vllm_lb.model_facts`): format, architectures, quant format, layer shapes, window — each `{value, source}` |
| Engine | the server software | `lb/vllm_lb/library/engines/<engine>.yml` + code | capabilities as data (formats, architectures, quant formats); behaviour as code (`catalog.ENGINE_REFUSES`/`VLLM_FLAGS`/`TENSORFOLD_*`, `fleet.NodeEngine`, `engine_metrics`, tp-rank `*_cmd`); KV dtypes per image in `kv_cache_dtypes.json` |
| Accelerator | what kernels target | `lb/vllm_lb/hardware/accelerators/<key>.yml` | arch, gpu_runtime, platforms, precisions, quant formats — each `{value, source}` |
| Card type | one product | `lb/vllm_lb/hardware/types/<key>.yml` | memory family, its accelerator, memory, cores, bandwidth, peaks; `arch`/`gpu_runtime` read through the accelerator |
| Topology | cards and how joined | `lb/vllm_lb/executors.py` | one class per kind: `OneCard`, `TpOneHost`, `TpAcrossHosts`; facts tp-rank is told |

Accelerators shipped:

| key | arch | runtime | platforms | precisions | quant formats | products |
|---|---|---|---|---|---|---|
| `nvidia-sm121` | sm_121 | nvidia | linux/arm64 | bf16 fp16 fp8 nvfp4 | none modelopt modelopt_mixed compressed-tensors fp8 mlx | gb10 |
| `nvidia-sm120` | sm_120 | nvidia | linux/amd64 | bf16 fp16 fp8 nvfp4 | none modelopt_mixed | rtx5090 |
| `nvidia-sm86` | sm_86 | nvidia | linux/amd64 | bf16 fp16 | none | rtx3090 |
| `amd-gfx1151` | gfx1151 | rocm | linux/amd64 | bf16 fp16 | none | radeon8060s |

Why sm_121 and sm_120 are two keys: this repo's GB10 images compile the arch-specific target 12.1a
(`TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a`), and arch-specific cubins run only on that exact compute
capability (CUDA C++ Programming Guide, Feature Availability). One key would hand a 5090 an image that cannot load.

## The glue

| piece | where | keyed by |
|---|---|---|
| Image matrix | `lb/vllm_lb/library/images/<engine>.yml` | engine × build × accelerator × platform → image (one entry a line; `deploy/image-tag.sh` greps it) |
| Recipes | `lb/vllm_lb/library/recipes.json` | model → engine + build, parsers, drafter, KV dtype, flags; `overrides.<accelerator>` (DeepSeek-V4's B12X kernels on `nvidia-sm121`) |
| Measurements | `lb/vllm_lb/library/measurements.json` | model → memory, engine memory, tuned input, overhead; `covers`: engine × accelerator × TP |
| Deployment | `lb/fleet.yml` `checkpoints.<id>` | path, uncensored twin, drafter — where THIS deployment staged it |
| Image builds | `deploy/images/<engine>/Dockerfile` + `<build>.<accelerator>.env` | one Dockerfile per engine, BASE per accelerator (`deploy/images/build.sh`) |

`catalog.document(deployment)` assembles the four library/deployment parts into the row shape the catalog has always
validated; `ModelCatalog.from_dict` then RESOLVES every profile against the deployment's targets.

## Resolution at Start

A deployment's TARGET per ServingMode is derived from the fleet file (`fleet.catalog_targets`): the cluster mode's
target is the serving cards of every cluster whose Start names its model from the catalog (the ssh pair), as ONE
accelerator × ONE platform × TP; the classic mode's, each backend box's serving cards. Mixed accelerators or platforms
under one target are refused.

```
model + target cards
  └─ engine: the recipe's (engine, build)
       1. target exists                    else  "<fleet> names no cluster …"
       2. engine serves the model          else  "<engine> serves none of the architectures … / no quant format …"
          (format, architectures, quant)          (engine_library.model_problem; unknown architectures → run model_facts)
       3. accelerator runs it              else  "<acc> has no quant format / platform …" (hardware.compatibility)
       4. image exists                     else  "no image of <engine> build <b> for <acc> × <platform>: it is built for …"
       5. KV dtype runs on image + arch    else  the image record's own words (kv_cache_dtypes.json)
       6. measured on engine × acc × TP    else  "not measured on <acc> at TP=<n> … — start it explicitly sized by the
                                                  estimate (estimated: true), or measure it there"
  └─ memory fits the boxes                 at Start, as before (DesiredStore.set, memory_admission, memory_estimate)
```

Every refusal is a string shown verbatim (Start → 409; `/lb/status` `models.cluster[].launch.cluster.resolution`;
the boot log, one `[catalog]` line a profile). Never a substitute: no nearest accelerator, no default image.

## The unmeasured combination: refuse, or the operator's explicit estimate

Default: REFUSE. A model whose `covers` does not include the target's engine × accelerator × TP is refused, naming what
it was measured on.

The operator may start it anyway, EXPLICITLY, sized by the estimate: `POST /lb/cluster {"enabled": true,
"model_choice": "<id>", "estimated": true}` or the panel's checkbox (shown only for such a model, unchecked on every
pick). Stored per Start (`Desired.estimated`): kept by a Stop and a restart of the same model, reset by a model change —
chosen each time, never carried to another model, never set by the LB. The launch gate (`cluster_exec.launch_problem`)
refuses an unmeasured model whose stored selection did not choose it.

The estimate (`catalog.estimated_footprint`, from the model's `shape` facts): weights = size on disk / TP; KV bytes a
token = 2 × full-attention layers × max(1, KV heads / TP) × head_dim × the run's KV element size. Not estimated (0):
PLE, drafter, linear-attention state, activations, the per-seat slope; overhead is the measured target's. Its
`calibration` is empty, so nothing is sized from it as if measured (`auto` pools refuse it). Refused where it cannot be
made: MLA attention, shapes not in the library, a recipe that names no KV dtype. Labelled: the picker row says "not
measured here · estimate only", the control quotes `resolution.refused` and `resolution.estimate`, `/lb/status` carries
`resolution.sizing` (`measured` | `unmeasured`) and `resolution.estimate`.

## Blueprints

Operator, 2026-10-30, asked whether the pair's live choice (12 seats × 131,072) should become a library default: "Maybe
instead of library default, let's implement 'blueprints' for specific setups". The library stays MEASURED FACT; a
blueprint is a REFERENCE SETUP — one of the project's real, proven setups by hardware type, said as one
(`lb/vllm_lb/blueprints.py`). The same day, on seeing a runtime picker of them: "Blueprints meant to be only for agent
to build their own fleet settings, not something user chooses." So a blueprint is the setup wizard's starting point
for a deployment's own answers — never a runtime object.

| piece | where | what |
|---|---|---|
| The reference setups | `lb/vllm_lb/library/blueprints/<id>.yml` | one file a blueprint, named by hardware type + model, never a host: the project's production setups |
| The wizard's question | `deploy/setup/questions.yml` `cluster.blueprint` | "start from a reference blueprint?" — offered where its target (accelerator × TP × control) is the cluster's answered cards and control |
| The proposals | `render.py <answers.yml> --prefill` | what fits each cluster, and each question's proposed answer (its `prefill`) beside the sheet's; the user confirms or edits |
| What is rendered | `lb/fleet.yml`, a node's server file | plain settings: the model, choices, launch block, KV dtypes, the box's RAM tier — never a blueprint id |

A blueprint: `label`, `source` (why these numbers: a measurement / generation, or "operator's choice <date>"), `target`
(`accelerator`, `tp`, `control`: ssh | local), `model`, `uncensored`, `estimated`, `knobs`, and for a node `fixed` (its
parsers). **Explicit-complete**: every knob of the target's Start is present — a missing one is refused, never filled in.
A pair's knobs are the Start's (`CLUSTER_KNOBS`): a value for each its engine takes, written as what the row resolves to
(so a blueprint never moves when a library row does); `null` for one its engine does not take (`ENGINE_REFUSES`) or a
PLE placement with no table. A node's knobs are its engine's settings (`fleet.NodeEngine.settings`), where `null` is the
engine's own, as in the fleet's launch block.

What holds them, and where:

```
the library (every test run:       keys complete · model + twin known · knob set = the engine's · null rule ·
 lb/tests/test_blueprints.py)      image matrix has engine × build × accelerator · seats/context within the model's
                                   limits · measured on engine × accelerator × TP, or estimated: true (and only then)
the wizard (render.py)             cluster.blueprint: its target's control × TP (the ranks' serving cards) × their
                                   accelerator = the cluster's, else refused naming why · every blueprint value is
                                   some question's `prefill` (deploy/tests/test_setup.py)
the deployment (--check in the     the rendered fleet by the LB's loader (offers, KV dtypes on the pinned image, the
 vllm-lb image)                    node's fit and RAM tier) · a pair's first Start (`cluster.first_model` +
                                   `cluster.first_start_settings`) by DesiredStore.check — the switch's own validator,
                                   nothing written
```

Prefill (`questions.yml` `prefill`): `cluster.model`, `cluster.first_model` ← the model (`@uncensored` where it runs the
twin); `cluster.choices`, `cluster.launch_choices` ← [it]; `models.staged` ← it, added; a pair's
`cluster.first_start_settings` ← every knob, `cluster.kv_cache_dtypes` ← [its dtype], `cluster.first_start` ←
`start-on-estimate` only where it is `estimated`; a node's launch questions ← each knob and parser (a null proposes
nothing: the engine's own); its box's `kv_offload_ram_max_gib` ← its RAM tier, raised, never lowered. The image is the
deployment's answer: a node's KV dtype is taken only on an image `kv_cache_dtypes.json` records.

The runtime side is gone (2026-10-30): no panel picker, no `{"blueprint": …}` switch key (a 400 `blueprint_retired`), no
`/lb/clusters/<name>/blueprints`, no `/lb/status` `blueprint` / `drift`, no boot blueprint. A fleet naming
`clusters.<name>.blueprint` or `blueprints:` is refused at boot naming this (`fleet.RETIRED_FLEET_KEYS`,
`RETIRED_CLUSTER_KEYS`); a pair switch file or node wish that recorded a `blueprint` loses the field once, logged
(`atomic.retire_blueprint_field`) — it was provenance only, so what runs is unchanged.

## Topology executors

`executors.topology_of(cluster)` picks the class from the cluster's ranks and serving cards; `Topology.facts(box)` are
what `lb/cluster/tp-rank` is told on every call (`cluster_run.tp_rank_env`), with no default there:

| fact | GB10 pair | meaning |
|---|---|---|
| `CLUSTER_TP_SIZE` | 2 | GPUs one engine spans |
| `CLUSTER_NNODES` | 2 | hosts, one rank each; a rank is 0..NNODES-1 |
| `CLUSTER_GPUS` | all | docker `--gpus`: `all`, or `device=<i>,<j>` (serving cards) |
| `CLUSTER_LINK` | roce | fleet `clusters.<name>.link`: roce / ethernet across hosts, pcie / nvlink within one |
| `CLUSTER_MEMORY` | unified | the cards' memory family; the RAM guard runs only on unified |
| `CLUSTER_PLATFORM` | linux/arm64 | checked against the box's `uname -m` |

The pair's 18 launches (9 profiles × 2 ranks) are pinned byte for byte against their pre-redesign argv
(`lb/tests/test_pair_launch_golden.py`, fixture recorded at 65b24de1). `ClusterExecutor` is the verb interface;
`SshExecutor` (tp-rank over SSH) the one the LB drives; `executors.executor_class` picks it by control.

## Transcription servers — the contract

The node engines `faster-whisper` and `nemo` (`fleet.FasterWhisperEngine`, `fleet.NemoEngine`) run no server this
repository builds: the image is the operator's choice, named per engine in the fleet file's `launch.image` and pinned
by id. Any image honouring this contract serves; the gateway reads nothing else.

**Process.** The node compose (`deploy/standalone/node-engine.faster-whisper.yml`, `node-engine.nemo.yml`) replaces
the vLLM entrypoint with `python -m transcription` (under `nice`), listening on port 8000 with the checkpoint mounted
at `/models`. Its environment, mapped from the node's values:

| variable | value | engines |
|---|---|---|
| `MODEL_SIZE` | `/models/<checkpoint folder>`; also the id `GET /v1/models` lists (the catalog row's `served_name`) | both |
| `DEVICE` | `cuda` (HIP answers to it under ROCm) | both |
| `MAX_ACTIVE_REQUESTS` | the launch's `max_num_seqs`: requests decoded at once (seats) | both |
| `WORKER_ID` | the node's name, for its logs | both |
| `MAX_QUEUE_SIZE` + `FAIL_FAST_WHEN_BUSY=false` | set only with the launch's `max_queue`: realtime requests queued behind busy seats; unset, a busy request is answered 503 at once | both |
| `COMPUTE_TYPE` | set only with the launch's `compute_type` (CTranslate2's); unset, the image's own | faster-whisper |

**HTTP.** `POST /v1/audio/transcriptions` (OpenAI-compatible multipart, word timestamps), `GET /health`,
`GET /v1/models`, `GET /metrics` (Prometheus text). The gateway rewrites the form's `model` field to the served id,
so the server may refuse any other name.

**Metrics** — the `transcription_*` family `engine_metrics.parse_transcription_metrics` reads (a family absent is
null on the panel, never a substitute):

| metric | type | canonical key |
|---|---|---|
| `transcription_requests_running` | gauge | running |
| `transcription_requests_waiting` | gauge | waiting |
| `transcription_seats` | gauge | seats |
| `transcription_queue_capacity` | gauge | queue_capacity |
| `transcription_requests_total{outcome}` | counter; outcome ∈ ok, silence, failed, rejected_busy, rejected_queue_full, rejected_deferred | finished / silence / failed / rejected |
| `transcription_audio_seconds_total` | counter | audio_seconds_total |
| `transcription_processing_seconds_sum` / `_count` | summary | request_seconds_sum / _count |
| `transcription_gpu_memory_bytes{region}` | gauge, summed over regions (vram, gtt) | gpu_memory_bytes |

## Open items

| item | why open | next step |
|---|---|---|
| TensorFold rows' `architectures` are INFERRED (Qwen4ExpForConditionalGeneration) | the checkpoints are staged on the pair only | `python -m vllm_lb.model_facts <folder> --check <id>` on box-a |
| Image matrix entries are TAGS, as the catalog pinned them | a digest needs the image on a box or registry | `docker image inspect` on the pair; replace each `image:` with `repo@sha256:…` |
| vLLM `fleet` build (myllmbox/qwen38-flash-next-cluster-vllm:v5) has no recipe in the repo | built outside this repo | add its Dockerfile + pin under `deploy/images/vllm/` |
| No `deploy/images` pin for vllm × nvidia-sm120 (`priority`, in the matrix since 2026-10-30), sglang fleet.2 | the full base digest / patch list is not recorded | read them off the 5090 node and the pair |
| A node blueprint names no image | a node's image is its launch block's pinned id; the matrix names the build's tag | match a node's `launch.image` to the matrix entry of its engine × accelerator, then prefill it |
| No measurement off `nvidia-sm121` × TP=2 for any cluster row | needs the hardware | run a Start on another accelerator; add a `covers` entry with its calibration |
| Node launches (control local/none) keep their fleet `launch.image`, and node fits are not held to `covers` | the image/measurement keying is applied to catalog cluster/classic profiles | move node images to the matrix; check node fits against `covers` |
| One target per ServingMode | one catalog resolves one image a profile | a catalog per cluster when two ssh clusters differ |
| One measurement entry per model | the shipped rows each have one | select among entries by target |
| TP within one host: classes + tp-rank support only | the planner assumes a two-rank pair (`config.cluster_configured`, `RANK_ROLES`); deploy/standalone reserves one GPU id | wire one-host TP into the loop; a multi-GPU device reservation |
| One tp-rank env for every rank | per-rank facts (GPU indices) must agree across boxes | send per-host facts with `CLUSTER_HOST_IP` |
| Estimate: PLE, drafter, linear-attention state, activations not modelled; MLA not estimated; GLM / DeepSeek / TensorFold shapes not in the library | not derivable from config.json alone / not read | extend `estimated_footprint`; run model_facts on box-a |
| Quant formats on sm86 / gfx1151: `none` only; sm120: `none`, `modelopt_mixed` only; sm120 / sm86 hosts: linux/amd64 only | nothing else served there in this project | add each with its source when it runs |
