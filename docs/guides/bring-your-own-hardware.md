# Bring your own hardware

The design: [docs/design/engines.md](../design/engines.md). What you write for a deployment is `lb/fleet.yml` only;
everything below is a committed library you extend once and every deployment shares.

| you have | you add | tests |
|---|---|---|
| a new card of a known architecture (another sm_120 card) | `lb/vllm_lb/hardware/types/<key>.yml` naming `accelerator: nvidia-sm120` | a pin of its sources in `lb/tests/test_hardware_profiles.py` |
| a new architecture | `hardware/accelerators/<key>.yml` + an image build + measurements | `lb/tests/test_accelerators.py`, `test_images.py` |
| a new engine | `library/engines/<engine>.yml` + its behaviour class + its image matrix | `lb/tests/test_engine_library.py` |
| a new model | `library/models.json` row (read by `model_facts`) + `recipes.json` + a measurement | `lb/tests/test_library.py` |
| a new deployment | `lb/fleet.yml` from `lb/fleet.example.yml` | `python -m vllm_lb.fleet node-env …`, the boot log's `[catalog]` lines |

## 1. A new card (same architecture): one YAML

```yaml
# lb/vllm_lb/hardware/types/<key>.yml — every figure {value, source}; null + why where no source states it
name: <product name>
memory: discrete            # or unified: the behaviour family (profiles.py)
accelerator: nvidia-sm120   # what its kernels target: arch and gpu_runtime come from there
controls: [local]
facts:
  memory_gb: {value: 32, source: "<spec page>"}
  cuda_cores: {value: …, source: "…"}
  peak_gbps: {value: …, source: "…"}
  host_platform: {value: null, source: "a discrete card: the box's platform says"}
  bandwidth_counter: {value: null, source: "not established: …"}
  visible_gib: {value: null, source: "not measured: …"}
  free_at_start_gib: {value: null, source: "not measured: …"}
  mfu_patch_loads: {value: null, source: "not established: …"}
  peak_tflops: {bf16: {value: …, source: "… dense …"}}
```

The loader refuses a peak at a precision the accelerator has no tensor unit for, and a host CPU it never comes in.
Name it on a box in `lb/fleet.yml`. Every image, recipe and measurement of its accelerator applies unchanged.

## 2. A new architecture: accelerator + image + measurements

1. `lb/vllm_lb/hardware/accelerators/<key>.yml` — `arch`, `gpu_runtime` (nvidia | rocm), `platforms`, `precisions`,
   `quant_formats` (only those served on it, each sourced; nothing established = `[none]` or null + why).
2. An image per engine build you run there: a pin file `deploy/images/<engine>/<build>.<key>.env`
   (`BASE=<repo>@sha256:…`, `PLATFORM=linux/…`, the engine's patch args), then
   `deploy/images/build.sh <engine> <build> <key> build`, and the matrix entry, one line:
   `  - {build: <build>, accelerator: <key>, platform: linux/…, image: "<tag or digest>", source: "…"}` in
   `lb/vllm_lb/library/images/<engine>.yml`.
3. Measurements: until a model is measured on `<key>`, a Start of it is refused "not measured on <key>". Start it once
   explicitly sized by the estimate (panel checkbox, or `"estimated": true`), read the engine's own figures off its
   boot log, and add a `covers` entry `{engine, accelerator: <key>, tp}` with its `memory` and `calibration` to
   `lb/vllm_lb/library/measurements.json`.

## 3. A new engine: class + YAML

1. `lb/vllm_lb/library/engines/<engine>.yml` — `formats`, `architectures`, `quant_formats` it serves, sourced.
2. Its behaviour: the knobs it refuses (`catalog.ENGINE_REFUSES`), its command (`lb/cluster/tp-rank` `<engine>_cmd`
   for a cluster, a `fleet.NodeEngine` subclass for a one-host node), its metrics vocabulary (`engine_metrics.py`).
3. `lb/vllm_lb/library/images/<engine>.yml` and `deploy/images/<engine>/Dockerfile` (`ARG BASE`, per-accelerator pins).
4. Its KV dtypes: one record per image in `lb/vllm_lb/kv_cache_dtypes.json`, read from the image's own source.

## 4. A new model

```bash
python -m vllm_lb.model_facts /models/<folder>            # format, architectures, quant format, shapes — with sources
python -m vllm_lb.model_facts /models/<folder> --check <id>   # exits 1 naming each fact that differs from the library
```

Add the facts to `library/models.json`, its engine settings to `library/recipes.json` (engine, build — never an image,
never a path), and where it sits to your `lb/fleet.yml` `checkpoints.<id>`.

## 5. Topology

| your cards | fleet | tp-rank is told |
|---|---|---|
| one card | a node (`control: local` / `none`) | — (deploy/standalone) |
| two hosts, one GPU each, RDMA | `control: ssh`, `iface`, `link: roce`, two ranks | TP 2 · NNODES 2 · GPUS all · LINK roce |
| two hosts over plain Ethernet | `link: ethernet` | NCCL sockets, no `/dev/infiniband`, `NCCL_IB_DISABLE=1` |
| several GPUs in one host | `TpOneHost` (`link: pcie` / `nvlink`) | TP n · NNODES 1 · GPUS device=… — wiring into the planner is an open item |
