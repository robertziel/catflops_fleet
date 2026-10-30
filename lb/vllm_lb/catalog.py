"""The models this gateway can launch: ONE ModelCatalog behind both pickers (worklog/vllm-model-selection.md;
worklog/vllm-gateway-program.md pass 4).

The domain words, the same in this module, in `/lb/status` (`models`, `cluster`), on the page
(`static/gateway.js`) and in the tests (`tests/test_model_catalog.py`, `tests/test_memory_estimate.py`):

  ModelCatalog     every model this gateway knows how to launch — the single source of model ids and launch
                   settings. Data, in `catalog.json` beside this module. Not derived from broker_parrot, whose
                   config knows no image, parser or KV pin.
  ModelSpec        one launchable model: id, label, path, served name, ModelLimits, a MemoryFootprint, and one
                   LaunchProfile per ServingMode it can be launched in.
  ServingMode      `classic` (broker_parrot's per-box vLLM) or `cluster` (the TP=2 pair). Defined here, once.
  LaunchProfile    a model's launch SETTINGS in one ServingMode: image, parsers, speculative config, KV pin, PLE
                   mode, MTP drafter view, per-role overhead. The cache identity tp-rank derives from them is its
                   `profile_key` — a different thing, with a different name.
  ModelLimits      the context window and the concurrency range a spec may be started with.
  MemoryFootprint  a model's memory per rank, calibrated on a real start ("" = not measured yet).
  UncensoredVariant the abliterated twin of a ModelSpec's checkpoint, beside it on disk: the ONE thing that
                   differs is the `path` the engine loads (`ModelSpec.launched_from`). Where a model has one
                   it is the DEFAULT (operator, 2026-10-07); a model without one cannot be asked for it.
  ModelChoice      the spec id the operator chose for a ServingMode. `ModelCatalog.choice` is its one validator.
  Availability     whether a spec's files and image are on a box — observed by the box, never assumed.
  LoadedModel      what an endpoint actually serves, from its own /v1/models — never the ModelChoice.

In this repo `ModelSpec` means this one. broker_parrot's engine has a class of the same name — an in-process
loader registration keyed by a role tag — and the LB never imports it (tests/test_model_catalog.py).
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from vllm_lb.kv_dtype import library as kv_dtypes
from vllm_lb.resolve import Resolution, Target, resolve

CLASSIC = "classic"
CLUSTER_MODE = "cluster"
SERVING_MODES = (CLASSIC, CLUSTER_MODE)
#: Which server a CLUSTER profile launches on the pair. The ServingMode stays `cluster` for both — it means
#: "the two GB10 boxes, TP=2, one engine, reached at :8010" — and the ENGINE is what tp-rank builds a different
#: command for. One engine runs at a time by construction: a start is one generation of one profile.
VLLM, SGLANG, TENSORFOLD = "vllm", "sglang", "tensorfold"
CLUSTER_ENGINES = (VLLM, SGLANG, TENSORFOLD)
#: A MODEL'S TASK — the OpenAI API family it answers, the LB's routing key (router.task_of_path): `chat`
#: (/v1/chat/completions, /v1/completions, /v1/messages, …) or `transcription` (/v1/audio/transcriptions,
#: /v1/audio/translations: a multipart upload, answered with text). Every row states it ({value, source}); a row the
#: LB's own launchers start (a classic or cluster LaunchProfile — tp-rank, broker_parrot's servers) is `chat`: they
#: run chat engines.
CHAT, TRANSCRIPTION = "chat", "transcription"
TASKS = (CHAT, TRANSCRIPTION)
#: THE SERVER A FLEET NODE RUNS a checkpoint on (`node_engine`, a row's — its format decides it: a Hugging Face Whisper
#: checkpoint is vLLM's, a CTranslate2 one faster-whisper's, a .nemo one nemo's; a row whose launch is SGLang's runs on
#: SGLang). Each is one class in fleet.py (`ENGINES`): its image, its fit and its node-env. Required on a row a fleet's
#: one-host node launches (fleet.py `_engine`).
NODE_ENGINES = (VLLM, "faster-whisper", "nemo", SGLANG)
#: THE OPERATOR'S KNOBS AN ENGINE DOES NOT TAKE, and why. Each is refused by the switch where it is given and at a
#: Start that would launch a stored one (`DesiredStore.set`), locked on the page (`refuses` in the profile's view),
#: and resolved to what the engine really runs (`cluster_exec.resolved_launch`, `memory_estimate.estimate_memory`) —
#: never accepted and dropped, which is what the SGLang pair did with all four until 2026-10-03 (generation 135 ran
#: with a stored 40 GB pin SGLang never read). Every knob NOT named reaches the engine — on SGLang the prefill
#: budget as --chunked-prefill-size, the drafter depth as --speculative-num-draft-tokens, the concurrency as
#: --max-running-requests, the context as --context-length (tp-rank `sglang_cmd`).
ENGINE_REFUSES: dict[str, dict[str, str]] = {
    VLLM: {},
    SGLANG: {
        "kv_offload_gib": "the disk tier is vLLM's LMCache connector, which SGLang is launched without",
        "long_prefill_token_threshold": "SGLang has no per-request prefill cap: its prefill budget bounds every chunk",
        "ple_replicate": "the PLE placement is a patch in the vLLM image, which SGLang does not read",
    },
    # TENSORFOLD (operator 2026-10-29; worklog/tensorfold-spike.md). Each read in its own source: the recipe's
    # scripts/config.sh and patches (mia-ai @ dc05c91, on TensorFold zig-flashnext db281878) and upstream 0.6.6
    # (cb2ebf05) src/tensorfold/cli_args.py. Every knob NOT named reaches it: the seats as --parallel, the context as
    # --context, the KV dtype as --kv-dtype (tp-rank `tensorfold_cmd`).
    TENSORFOLD: {
        "kv_cache_memory": "TensorFold budgets its sequence memory itself — MemAvailable at load less "
                           "TENSORFOLD_MEMORY_RESERVE_GIB, sized to the seats x context it serves ('sequence memory: "
                           "30.76 GiB (43.35 GiB free, 10.0 GiB kept free)', the spike's rank 0); it takes no pin",
        "max_num_batched_tokens": "TensorFold's prompt chunk is its own (the native engine: 'prompt chunks of up to "
                                  "2,048 tokens'; 0.6.6: 2,048-row pieces) and neither CLI takes a budget",
        "num_speculative_tokens": "TensorFold drafts with the checkpoint's own MTP head and stops each round's chain by "
                                  "confidence (native: up to TF_FLASHNEXT_DEPTH 15; 0.6.6: up to 6, under 70%); a "
                                  "fixed depth is not a setting it takes — the profile's engine_args.drafts turns "
                                  "them on or off",
        "ple_replicate": "the PLE placement is a patch in the vLLM image, which TensorFold does not read",
        "kv_offload_gib": "the disk tier is vLLM's LMCache connector, which TensorFold is launched without",
        "long_prefill_token_threshold": "TensorFold has no per-request prefill cap: its own prompt chunk bounds every "
                                        "pass",
    },
}
#: THE CALLER'S X-Priority, per engine that would IGNORE it, and why (router.priority_refusal): the proxy refuses the
#: header (400) for a backend running one of these, never forwards it to be dropped. vLLM and SGLang are started to
#: schedule by it (tp-rank); TensorFold's server reads `priority` only as the word "background" (native
#: server/chat.zig `prepare`, 0.6.6 server/app.py:239) and an integer is dropped.
ENGINE_DROPS_PRIORITY: dict[str, str] = {
    TENSORFOLD: "TensorFold reads a request's `priority` only as the word \"background\" (title and batch jobs after "
                "the foreground); an integer priority is dropped, so it cannot order this request",
}
#: THE ENGINES THAT RESERVE THEIR KV POOL ONE WINDOW A SEAT: the pool IS seats x context at the model's bytes a token,
#: so every seat fits at the context it was started with — no pin is read and no "queues the rest" can be true.
#: TensorFold's sequence memory (rank 0, 2026-10-29: 'sequence memory: 30.76 GiB' = 16 x 131,072 x 15,749 B; native
#: 'caches in reserved address space'). vLLM and SGLang page ONE shared pool sized by a pin or a memory fraction.
ENGINE_POOL_PER_SEAT = frozenset({TENSORFOLD})
#: THE ENGINES WHOSE COMPUTE THE LB MODELS from their own pass counters (cores.py `modeled_flops`), and whose memory
#: bandwidth it models where the catalog records a `StepTraffic`: SGLang, whose own FLOP and byte estimates are
#: withheld (engine_metrics `SGLANG_FAMILIES`), and TensorFold, which publishes none (its pass counters are the fleet
#: image's patch 0014). vLLM's own `--enable-mfu-metrics` estimate is drawn instead.
ENGINE_MODELED_PERF = frozenset({SGLANG, TENSORFOLD})
#: THE TENSORFOLD SETTINGS a profile's engine_args MUST state — no default in tp-rank either: `cli` (which server the
#: image carries: `tensorfold-native` for the Zig engine, `tensorfold` for the Python one), `kv_cache_dtype` (the LB's
#: own name for the cache's element type; tp-rank turns it into TensorFold's word), `drafts` (the MTP head on or off).
TENSORFOLD_REQUIRED_ARGS = ("cli", "kv_cache_dtype", "drafts")
TENSORFOLD_CLIS = ("tensorfold-native", "tensorfold")
#: A vLLM PROFILE'S ENGINE FLAGS — settings a model needs that no other names, each the profile's own field (vLLM's
#: settings are never a free-form engine_args): {field: (kind, what it means)}. `cluster_exec.launch_env` sends each
#: as CLUSTER_<FIELD> (a bool as 1/0) and tp-rank turns it into `--<field-with-dashes>` (a bool: the bare flag at 1).
#: Named for GLM-5.3-Flash on GB10 (operator 2026-10-20), each measured necessary on 2x DGX Spark by the published
#: recipes (himorishige/glm53-flash-2x-dgx-spark-recipe docs/pitfalls.md); a profile that names none sends none.
VLLM_FLAGS: dict[str, tuple[type, str]] = {
    "kv_cache_dtype": (str, "the KV cache's element type, e.g. fp8_e4m3"),
    "block_size": (int, "tokens per KV page — a model whose kernels expect one page size states it"),
    "moe_backend": (str, "the MoE kernel family, e.g. marlin"),
    "enforce_eager": (bool, "no CUDA graphs"),
    "language_model_only": (bool, "skip a multimodal checkpoint's vision tower and its warmup profile"),
    # Serving images and video on GB10 instead (MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark, operator 2026-10-21):
    "skip_mm_profiling": (bool, "no max-size image/video dummy forward at init (it OOMs a unified-memory rank)"),
    "limit_mm_per_prompt": (dict, "the most items of each modality one prompt may carry, e.g. {image: 4, video: 1}"),
    # DeepSeek-V4-Flash on GB10's B12X stack (operator 2026-10-23; eugr/spark-vllm-docker @0bdc9c94
    # recipes/deepseek-v4-flash-0731.yaml, the one published 2x DGX Spark launch of this model family):
    "trust_remote_code": (bool, "load the checkpoint's own model code"),
    "tokenizer_mode": (str, "the tokenizer family, e.g. deepseek_v4"),
    "load_format": (str, "the weight loader, e.g. b12x"),
    "linear_backend": (str, "the dense GEMM kernel family, e.g. b12x"),
    "attention_backend": (str, "the attention kernel family, e.g. B12X"),
    "max_cudagraph_capture_size": (int, "the largest batch a CUDA graph is captured for"),
    "compilation_config": (object, "vLLM's compilation config, one flat JSON object"),
    "reasoning_config": (object, "the reasoning parser's start/end markers, one flat JSON object"),
    "default_chat_template_kwargs": (object, "what the chat template gets when a request names nothing"),
}
#: THE KV DTYPE AN SGLANG PROFILE RUNS when its engine_args name none: tp-rank `sglang_cmd`'s own default
#: (`engine_arg kv_cache_dtype fp8_e4m3`, pinned against the script by tests/test_kv_cache_dtype.py).
SGLANG_KV_CACHE_DTYPE = "fp8_e4m3"
#: The pair's two ranks: rank 0 serves the API on the head; rank 1 runs headless on the worker.
RANK_ROLES = ("head", "worker")
#: The pair's tensor-parallel size: one rank per role. Every PER-RANK figure in this catalog is at this TP —
#: the MemoryFootprint's GiB and a StepTraffic's bytes alike.
CLUSTER_TP = len(RANK_ROLES)

#: WHERE THE CLUSTER'S CHECKPOINTS LIVE on each box, best first, `:`-separated, `~` the BOX's home — a deployment's
#: own (lb/.env), never the catalog's: e.g. "~/models:/mnt/box-a-home/models:/srv/models" (NVMe copy, the
#: same copy over the pair-link NFS, then the HDD). Unset is an error naming it, not a guessed folder (2026-10-09).
MODELS_SEARCH_DIRS = "MODELS_SEARCH_DIRS"


def models_search_dirs(env=None) -> list[str] | None:
    """MODELS_SEARCH_DIRS from the environment as a list, or None when it names no folder."""
    raw = (os.environ if env is None else env).get(MODELS_SEARCH_DIRS, "")
    dirs = [d.strip() for d in raw.split(":") if d.strip()]
    return dirs or None


#: The fleet file the LB reads (config.py, compose's mount): the deployment a catalog is ASSEMBLED for and RESOLVED
#: against — its cards (the targets) and where its checkpoints sit (`checkpoints`).
FLEET_ENV = "LB_FLEET"

#: THE MODEL LIBRARY (operator-approved split, 2026-10-30; docs/design/engines.md). What `catalog.json` held in one row
#: is four things with four owners, and each lives in its own place now:
#:   models.json        the MODEL — facts its config.json states (format, architectures, quant format, window, layer
#:                      shapes), its label and served name. `python -m vllm_lb.model_facts` reads them from a folder.
#:   recipes.json       the ENGINE SETTINGS for it, independent of hardware: engine + build (the image is the matrix's,
#:                      library/images), parsers, speculative decoding, KV dtype, flags; `overrides.<accelerator>` for
#:                      the few that apply on one accelerator only.
#:   measurements.json  what was MEASURED, keyed by what it was measured on (`covers`: engine × accelerator × TP):
#:                      memory footprint, engine memory, tuned start input, per-rank overhead.
#:   the fleet file     WHERE each checkpoint sits in THIS deployment (`checkpoints.<id>`: path, its uncensored twin,
#:                      its drafter) — gitignored, like every deployment file; lb/fleet.example.yml names the reference.
#: `document` assembles them into the one row shape `from_dict` validates, so every rule the catalog kept still holds.
LIBRARY_DIR = Path(__file__).resolve().parent / "library"
#: The retired single file: a checkout still holding it (an operator's local edit) is refused naming the split, never
#: silently ignored — its edit would otherwise vanish from what the LB runs.
RETIRED_CATALOG = Path(__file__).resolve().with_name("catalog.json")
#: Keys a library row may no longer carry, each with where its value lives now (config.RETIRED_KEYS' rule for files).
RETIRED_ROW_KEYS = {
    "path": "the fleet file's checkpoints.<id>.path (where a checkpoint sits is the deployment's)",
    "memory": "library/measurements.json, keyed by engine × accelerator × TP",
    "engine_memory": "library/measurements.json, keyed by engine × accelerator × TP",
    "launch": "library/recipes.json",
    "node_engine": "library/recipes.json",
}
RETIRED_PROFILE_KEYS = {
    "image": "the image matrix (library/images/<engine>.yml): a recipe names its `build`",
    "draft_path": "the fleet file's checkpoints.<id>.draft",
    "tuned": "library/measurements.json (`launch.<mode>.tuned`)",
    "overhead_gib": "library/measurements.json (`launch.<mode>.overhead_gib`)",
}


@dataclass(frozen=True)
class Checkpoint:
    """Where one model's files sit in a deployment, relative to its models roots (MODELS_SEARCH_DIRS): the checkpoint,
    its uncensored twin (None: not staged), its drafter checkpoint (None: the model has none)."""

    path: str
    uncensored: str | None = None
    draft: str | None = None


@dataclass(frozen=True)
class Deployment:
    """What a catalog is assembled for (`document`) and resolved against (`from_dict`): ServingMode → its `Target`, or
    why there is none; model id → its `Checkpoint`; and the file both came from."""

    targets: Mapping[str, object]
    checkpoints: Mapping[str, Checkpoint]
    source: str


def current_deployment(env=None) -> Deployment:
    """THIS LB's deployment: the fleet file LB_FLEET names (`fleet.catalog_deployment_of`). No fleet file: no cards and
    no checkpoint folders are named, so every model is `absent` and says so — the catalog never resolves against cards
    nobody named (AGENTS.md: no auto-detection)."""
    path = (os.environ if env is None else env).get(FLEET_ENV, "").strip()
    if not path:
        why = f"{FLEET_ENV} names no fleet file"
        return Deployment({m: f"{why}, so no cards are named to resolve its image against" for m in SERVING_MODES},
                          {}, why)
    from vllm_lb.fleet import catalog_deployment_of
    return catalog_deployment_of(path)


def deployment_targets(env=None) -> dict:
    """ServingMode → the `resolve.Target` its profiles launch onto in this deployment, or why there is none."""
    return dict(current_deployment(env).targets)


def _library(name: str, key: str) -> list:
    raw = json.loads((LIBRARY_DIR / name).read_text())
    _keys(raw, f"library/{name}", required={"about", key})
    if not isinstance(raw[key], list):
        raise ValueError(f"library/{name}: `{key}` must be a list")
    return raw[key]


def _by_model(rows: list, name: str) -> dict:
    out = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("model"), str):
            raise ValueError(f"library/{name}: every entry names its `model`")
        if r["model"] in out:
            raise ValueError(f"library/{name}: {r['model']} is stated twice (one entry a model; docs/design/engines.md "
                             f"Open items: several measurements of one model)")
        out[r["model"]] = r
    return out


def _covers(entry: dict, where: str) -> list:
    covers = entry.get("covers")
    if not isinstance(covers, list) or not covers:
        raise ValueError(f"{where}: `covers` must list what it was measured on: [{{engine, accelerator, tp}}, …]")
    from vllm_lb.hardware import accelerators
    for c in covers:
        _keys(c, f"{where}: covers", required={"engine", "accelerator", "tp"})
        if c["accelerator"] not in accelerators():
            raise ValueError(f"{where}: covers names no accelerator {c['accelerator']!r}: the accelerators are "
                             f"{', '.join(sorted(accelerators()))}")
        if not (_int(c["tp"]) and c["tp"] >= 1):
            raise ValueError(f"{where}: covers tp must be a whole number from 1")
    return covers


def library_models() -> list:
    """Every model the LIBRARY knows (library/models.json rows), staged in this deployment or not."""
    return _library("models.json", "models")


def library_default(mode: str) -> dict | None:
    """The LIBRARY's default model for `mode` (its recipe says `default: true` there), as its models.json row — whether
    or not this deployment staged it. What a lease DECLARES (config.lease_model_id: the classic default's served name
    and window) is the model's identity, not where its folder sits: broker_parrot's classic server loads it, never this
    LB, so a fleet that stages no checkpoint still leases under its name."""
    recipes = _by_model(_library("recipes.json", "recipes"), "recipes.json")
    for m in _library("models.json", "models"):
        launch = (recipes.get(m.get("id")) or {}).get("launch") or {}
        if (launch.get(mode) or {}).get("default") is True:
            return m
    return None


def document(dep: Deployment | None = None) -> dict:
    """THE CATALOG DOCUMENT FOR A DEPLOYMENT: each library model with its recipe, its measurement and this deployment's
    checkpoint folders, in the one row shape `from_dict` validates. A model the deployment names no checkpoint for is
    `absent`, with why — it cannot launch here and is not offered. Any moved key still written in the library is
    refused naming where it lives now (`RETIRED_ROW_KEYS`, `RETIRED_PROFILE_KEYS`)."""
    if RETIRED_CATALOG.exists():
        raise ValueError(f"{RETIRED_CATALOG.name} is retired (2026-10-30): its rows are split into library/models.json, "
                         f"recipes.json, measurements.json and the fleet file's checkpoints — port any local edit "
                         f"there and remove {RETIRED_CATALOG}")
    dep = current_deployment() if dep is None else dep
    models = _library("models.json", "models")
    recipes = _by_model(_library("recipes.json", "recipes"), "recipes.json")
    measured = _by_model(_library("measurements.json", "measurements"), "measurements.json")
    ids = [m.get("id") for m in models if isinstance(m, dict)]
    for name, table in (("recipes.json", recipes), ("measurements.json", measured), ("the fleet's checkpoints",
                                                                                     dep.checkpoints)):
        stray = sorted(set(table) - set(ids))
        if stray:
            raise ValueError(f"{name} names {', '.join(stray)}, which no model in library/models.json is "
                             f"(the models are {', '.join(ids)})")
    rows, absent = [], {}
    for m in models:
        if not isinstance(m, dict):
            raise ValueError("library/models.json: every model is an object")
        mid = m.get("id")
        for k, where in RETIRED_ROW_KEYS.items():
            if k in m:
                raise ValueError(f"library/models.json: {mid}: `{k}` moved to {where}")
        ck = dep.checkpoints.get(mid)
        if ck is None:
            absent[mid] = (f"{dep.source} names no checkpoint folder for it (checkpoints.{mid}.path): it is not "
                           f"staged in this deployment")
            continue
        recipe = recipes.get(mid) or {"model": mid, "launch": {}}
        _keys(recipe, f"library/recipes.json: {mid}", required={"model", "launch"},
              optional={"node_engine", "overrides"})
        row = {k: v for k, v in m.items()}
        row["path"] = ck.path
        if "uncensored" in row:
            if ck.uncensored is None:
                del row["uncensored"]           # the twin is not staged here: the model launches from its original
            else:
                row["uncensored"] = {"path": ck.uncensored, **row["uncensored"]}
        elif ck.uncensored is not None:
            raise ValueError(f"{dep.source}: checkpoints.{mid}.uncensored names a twin, but library/models.json states "
                             f"none for {mid}")
        if "node_engine" in recipe:
            row["node_engine"] = recipe["node_engine"]
        entry = measured.get(mid)
        measured_launch = {}
        if entry is not None:
            w = f"library/measurements.json: {mid}"
            _keys(entry, w, required={"model", "covers", "covers_source"},
                  optional={"memory", "engine_memory", "launch"})
            row["measured_on"] = {"covers": _covers(entry, w), "source": _text(entry["covers_source"],
                                                                                f"{w}: covers_source")}
            for k in ("memory", "engine_memory"):
                if k in entry:
                    row[k] = entry[k]
            measured_launch = entry.get("launch") or {}
        launch = {}
        overrides = recipe.get("overrides") or {}
        for mode, p in recipe["launch"].items():
            for k, where in RETIRED_PROFILE_KEYS.items():
                if k in p:
                    raise ValueError(f"library/recipes.json: {mid}: {mode}: `{k}` moved to {where}")
            profile = dict(p)
            # A drafter that is its OWN checkpoint (DFlash2): the recipe says it needs one, the deployment says where.
            if profile.pop("draft_checkpoint", False) and ck.draft is None:
                raise ValueError(f"{dep.source}: {mid}'s {mode} recipe drafts with a separate checkpoint: name its "
                                 f"folder at checkpoints.{mid}.draft")
            profile.update(measured_launch.get(mode) or {})
            target = dep.targets.get(mode)
            acc = getattr(getattr(target, "accelerator", None), "key", None)
            if acc is not None and acc in overrides:
                profile.update(overrides[acc])
            if ck.draft is not None and mode == CLUSTER_MODE:
                profile["draft_path"] = ck.draft
            launch[mode] = profile
        if ck.draft is not None and CLUSTER_MODE not in launch:
            raise ValueError(f"{dep.source}: checkpoints.{mid}.draft names a drafter, but {mid} has no cluster recipe")
        row["launch"] = launch
        rows.append(row)
    return {"models": rows, "absent": absent}

#: tp-rank's absolute caps (`cmd_up`), which `cluster_exec` mirrors. No ModelLimits may exceed them.
MAX_NUM_SEQS_CAP = (1, 48)
MAX_MODEL_LEN_CAP = (1024, 262144)
#: And tp-rank's bounds for the two ENGINE knobs a TunedSetup may state (`check_launch_settings`), mirrored
#: for the same reason as the two above: a catalog value outside them is a document error caught on load, not
#: a start that dies on two boxes. `cluster.MAX_NUM_BATCHED_TOKENS_RANGE` and `NUM_SPECULATIVE_TOKENS_RANGE`
#: ARE these — the switch reads them from here, so the page, the catalog and the launch bound one thing once.
MAX_NUM_BATCHED_TOKENS_CAP = (2048, 65536)
NUM_SPECULATIVE_TOKENS_CAP = (0, 8)
#: A spec id travels on tp-rank's argv and in a docker label.
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")


class UnknownModelChoice(ValueError):
    """A ModelChoice that names no ModelSpec launchable in that ServingMode — the caller's mistake."""


# ── value objects ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelLimits:
    max_model_len: int
    max_num_seqs: tuple[int, int]

    def view(self) -> dict:
        return {"max_model_len": self.max_model_len, "max_num_seqs": list(self.max_num_seqs)}


@dataclass(frozen=True)
class KvBlocks:
    """THE ENGINE'S RULE for seats, per model: vLLM holds the KV pool in BLOCKS of `block_tokens` tokens,
    `block_bytes` each (every KV-carrying layer's page for one block), and ONE request takes
    ceil(context / block_tokens) + `fixed_blocks` of them — the fixed ones its linear-attention state,
    padded to pages, the same at any context. Read off `vllm:cache_config_info` (num_gpu_blocks,
    kv_cache_max_concurrency) and the start log's "Setting attention block size": the 27B at 48 GB holds
    1,689 blocks of 816 tokens and reports 4.97x at 262,144 = 1689 / (322 + 18); Flash-Next at 28 GB
    1,225 of 1,616 tokens, 6.52x = 1225 / (163 + 25). A pool under one request's blocks kills the engine
    at init (2026-10-02: an 8 GB pin on the 27B at the full window)."""
    block_tokens: int
    block_bytes: int
    fixed_blocks: int
    source: str = ""
    #: The KV element type the rule was MEASURED at (vllm_lb/kv_cache_dtypes.json `layouts`): `at` derives it at another.
    kv_cache_dtype: str = ""

    def at(self, dtype: str) -> "KvBlocks":
        """THIS RULE AT ANOTHER KV DTYPE — derived, never measured, and its source says so. A rule with no fixed (state)
        blocks is pure attention: the engine's block is a fixed count of tokens, so its bytes scale with the element.
        A rule WITH state blocks is a hybrid one: vLLM pads the attention page to the linear-attention state's page
        (the block size is the smallest one whose page holds the state), so the page bytes stay and the tokens a
        block holds scale inversely — floored, within the engine's page alignment, which is not modelled."""
        if dtype == self.kv_cache_dtype:
            return self
        r = kv_dtypes().ratio(dtype, self.kv_cache_dtype)
        why = (f"DERIVED at {dtype} from the rule measured at {self.kv_cache_dtype} (x{r:g} bytes an element), not "
               f"measured: {self.source}")
        if self.fixed_blocks == 0:
            return replace(self, block_bytes=math.ceil(self.block_bytes * r), source=why, kv_cache_dtype=dtype)
        return replace(self, block_tokens=max(1, math.floor(self.block_tokens / r)), source=why, kv_cache_dtype=dtype)

    def blocks_in(self, pool_bytes: int) -> int:
        return max(0, int(pool_bytes)) // self.block_bytes

    def per_request(self, max_model_len: int) -> int:
        return -(-int(max_model_len) // self.block_tokens) + self.fixed_blocks

    def seats(self, pool_bytes: int, max_model_len: int, *, pool_blocks: int | None = None) -> int:
        """How many requests of `max_model_len` the pool holds at once — the engine's own figure, floored.
        `pool_blocks`: the count the engine itself reports while it serves, preferred over the arithmetic."""
        blocks = pool_blocks if isinstance(pool_blocks, int) and pool_blocks > 0 else self.blocks_in(pool_bytes)
        return blocks // self.per_request(max_model_len)

    def min_pool_bytes(self, max_model_len: int) -> int:
        """The smallest pool that holds ONE request of this context: under it the engine does not start."""
        return self.per_request(max_model_len) * self.block_bytes

    def view(self) -> dict:
        return {"block_tokens": self.block_tokens, "block_bytes": self.block_bytes,
                "fixed_blocks": self.fixed_blocks, "source": self.source, "kv_cache_dtype": self.kv_cache_dtype}


@dataclass(frozen=True)
class CardOverhead:
    """What ONE card running this model alone loses beyond its weights and its KV pool — non-torch memory, peak
    activation, CUDA graphs, the card's own reserve — in GiB, as the engine's own boot line implies it (`source`).
    The peak activation grows with the prefill chunk, so the figure is one AT a chunk: `chunk_tokens`, the
    max_num_batched_tokens the engine ran at, and where that number comes from (`chunk_source`). None = not stated, and
    no launch that names a chunk can be judged against the figure (fleet.launch_fit)."""
    gib: float
    source: str = ""
    chunk_tokens: int | None = None
    chunk_source: str = ""


@dataclass(frozen=True)
class MemoryFootprint:
    """Per rank, in GiB unless named otherwise. `weights_gib` leaves out the PLE table and the MTP drafter, which
    depend on the LaunchProfile (PLE mode, speculative decoding) and are counted by `memory_estimate`."""
    weights_gib: float
    ple_gib: float
    mtp_gib: float
    kv_bytes_per_token: int
    seq_state_gib: float
    calibration: str = ""
    #: What ONE RANK loses per 1,024 tokens of prefill budget (`max_num_batched_tokens`) ABOVE
    #: `calibrated_batched_tokens` — activations, the caching allocator's high-water mark, the inductor
    #: compile range and the captured prefill graphs, which all scale with that one knob. 0.0 = never
    #: measured for this model, and then the budget is not counted (the blindness this replaces).
    activation_gib_per_1k_tokens: float = 0.0
    #: The prefill budget `overhead_gib` was measured at, so the term above is exactly zero there.
    #: 0 = unknown, and then nothing is charged: a slope needs a point to start from.
    calibrated_batched_tokens: int = 0
    #: What ONE RANK loses per SEQUENCE SEAT (`max_num_seqs`) above `calibrated_max_num_seqs` — the
    #: captured decode CUDA graphs, the per-step activation and sampler buffers sized on the batch, and
    #: the caching allocator's high-water mark, which torch never returns to the driver. Measured on the
    #: live pair, 2026-10-01: 1.2 GiB per seat between 16 seats and 24. 0.0 = never measured for this
    #: model, and then the concurrency is not counted.
    concurrency_gib_per_seq: float = 0.0
    #: The concurrency the term above is charged FROM — the highest one MEASURED to rest safely on these
    #: boxes (16: 11.3 GiB free, idle), not the single seat `overhead_gib` was taken at. Charging from 1
    #: would need a slope extrapolated from two CUDA-graph captures, and the live pair contradicts it.
    #: 0 = unknown, and then nothing is charged: a slope needs a point to start from.
    calibrated_max_num_seqs: int = 0
    #: The engine's block rule for this model (`KvBlocks`); None = not recorded, and seats are tokens over
    #: context — the rule before 2026-10-02, which overstated them (19 for 17 at 65,536 on the 27B).
    kv_blocks: KvBlocks | None = None
    #: The same rule on ONE card (TP=1, a standalone node) — measured there, never the pair's scaled: one card pages
    #: at its own block size (the 27B: 1,568 tokens on a 5090 against the pair's 816). None = not measured, and a
    #: one-card launch's fit cannot be checked (fleet._fit refuses it).
    single_card_kv_blocks: KvBlocks | None = None
    #: The rest of what that one card holds (`CardOverhead`); None = not measured, and the fit cannot be checked.
    single_card_overhead: CardOverhead | None = None
    #: The KV element type `kv_bytes_per_token` was measured at — what the launch's `auto` resolved to where it named
    #: none (vllm_lb/kv_cache_dtypes.json `layouts`). Required: a byte figure is only a figure at one dtype.
    kv_cache_dtype: str = ""

    def at(self, dtype: str, kv_bytes_per_token: int | None = None) -> "MemoryFootprint":
        """This footprint at another KV dtype: the bytes per token scaled by the element (or `kv_bytes_per_token`, where
        the caller prices the engine's own extra), each block rule derived (`KvBlocks.at`). Everything else is not KV."""
        per_token = (kv_bytes_per_token if kv_bytes_per_token is not None else
                     math.ceil(self.kv_bytes_per_token * kv_dtypes().ratio(dtype, self.kv_cache_dtype)))
        return replace(self, kv_bytes_per_token=per_token, kv_cache_dtype=dtype,
                       kv_blocks=self.kv_blocks.at(dtype) if self.kv_blocks is not None else None,
                       single_card_kv_blocks=(self.single_card_kv_blocks.at(dtype)
                                              if self.single_card_kv_blocks is not None else None))


@dataclass(frozen=True)
class StepTraffic:
    """WHAT ONE FORWARD PASS MOVES THROUGH ONE RANK'S MEMORY, for the modeled MEMORY BANDWIDTH and the CUDA
    CORES tile (`throughput.Throughput.bandwidth_model`, `cores.estimate`) — the `dense` family: every weight
    of the target on every pass. Recorded for the SGLang 27B profile, whose engine publishes no byte figure worth
    drawing (probe.py `SGLANG_FAMILIES`); vLLM's own estimate is drawn for every vLLM row, and a routed-expert
    model is `RoutedStepTraffic`.

    Two figures are measured off the checkpoints' shard headers, because a forward streams only some of what is
    resident: `target_read_bytes` is the layers + the lm_head shard (the embedding shard is gathered row by
    row, the vision tower never runs on text), `draft_read_bytes` the drafter's own per-rank share + the
    target's lm_head shard its selector projects onto. The rest is config.json, so the derivations below are
    the model's own arithmetic, each checked against what SGLang itself allocated
    (tests/test_sglang_bandwidth_model.py). The two dtypes are the LAUNCH's, not the checkpoint's: tp-rank's
    `sglang_cmd` passes `--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16` on every SGLang start."""
    target_read_bytes: int
    draft_read_bytes: int
    full_attention_layers: int
    num_key_value_heads: int
    head_dim: int
    kv_cache_dtype_bytes: int
    linear_attention_layers: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_state_dtype_bytes: int
    draft_attention_layers: int
    draft_num_key_value_heads: int
    draft_head_dim: int
    draft_sliding_window: int
    source: str = ""

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every full-attention layer, this rank's KV heads: 2 x 16 x 4/2 x 256 x 1 B = 16,384."""
        return math.ceil(2 * self.full_attention_layers * (self.num_key_value_heads // CLUSTER_TP) * self.head_dim
                         * self.kv_cache_dtype_bytes)

    @property
    def draft_kv_bytes_per_token(self) -> int:
        """The drafter's own KV pool, same dtype: 2 x 5 x 8/2 x 128 x 1 B = 5,120."""
        return math.ceil(2 * self.draft_attention_layers * (self.draft_num_key_value_heads // CLUSTER_TP)
                         * self.draft_head_dim * self.kv_cache_dtype_bytes)

    def pool_bytes_per_token(self, bytes_per_element: float, workspace_layers: int,
                             draft_bytes_per_element: float | None = None) -> int:
        """What one token costs the TARGET's and the DRAFTER's KV pools at `bytes_per_element`, plus `workspace_layers`
        of each FP4 pool's K/V at 1 B an element — SGLang's FP4 pools add one shared FP8 dequant layer per token
        (model_executor/pool_configurator.py:409-417 @708f51e: cell // 2 + scales / 16 + n·k·2). The drafter's pool at
        its own dtype where one is given (`draft_bytes_per_element`: --speculative-draft-kv-cache-dtype), with no
        workspace; else at the target's, workspace and all (the same configurator builds it)."""
        target = 2 * (self.num_key_value_heads // CLUSTER_TP) * self.head_dim
        draft = 2 * (self.draft_num_key_value_heads // CLUSTER_TP) * self.draft_head_dim
        own = draft_bytes_per_element is not None
        return math.ceil(self.full_attention_layers * target * bytes_per_element + workspace_layers * target
                         + self.draft_attention_layers * draft * (draft_bytes_per_element if own else bytes_per_element)
                         + (0 if own else workspace_layers * draft))

    @property
    def linear_state_bytes(self) -> int:
        """One sequence's recurrent GDN state on this rank: 48 x 48/2 x 128 x 128 x 2 B = 37,748,736."""
        return (self.linear_attention_layers * (self.linear_num_value_heads // CLUSTER_TP)
                * self.linear_key_head_dim * self.linear_value_head_dim * self.linear_state_dtype_bytes)

    @staticmethod
    def state_transfers(verify_width: int | None) -> int:
        """How many times a decode pass moves one sequence's GDN state. A VERIFY of D tokens reads it once and
        writes D snapshots (SGLang gdn_triton.py:201-241, `disable_state_update=True`,
        `cache_steps=draft_token_num`), then the commit copies the accepted one back, a read and a write
        (hybrid_linear_attn_backend.py `scatter_mamba_states_after_mtp_verify`): D + 3. A plain decode reads it
        and writes it in place: 2."""
        return verify_width + 3 if _drafting(verify_width) else 2

    def decode_pass_bytes(self, *, context: float, seqs: float, verify_width: int | None) -> tuple[int, int, int]:
        """(weights, KV, GDN state) bytes one DECODE pass moves on one rank. `context`: the tokens summed over
        the running sequences, every one of which a full-attention layer reads; `seqs`: those sequences;
        `verify_width`: the tokens a verify carries per sequence (SGLang's `spec_num_draft_tokens`; 0 or 1 =
        no drafter, which then reads nothing). The drafter's attention reads its 2,048-token sliding window,
        so its KV is charged min(context, seqs x window) — exact when every context is at least the window,
        an upper bound otherwise."""
        drafting = _drafting(verify_width)
        weights = self.target_read_bytes + (self.draft_read_bytes if drafting else 0)
        kv = int(context) * self.kv_bytes_per_token
        if drafting:
            kv += int(min(context, seqs * self.draft_sliding_window)) * self.draft_kv_bytes_per_token
        return weights, kv, int(seqs) * self.linear_state_bytes * self.state_transfers(verify_width)

    def prefill_bytes(self, rows: float | None = None) -> int:
        """A PREFILL pass, as far as it is modeled: the target's weights, once — a DENSE model streams every one
        whatever the chunk's `rows`. A floor — its attention also reads the cached prefix's KV, and the drafter
        only projects the new tokens into its own KV."""
        return self.target_read_bytes

    @property
    def omits(self) -> str:
        """What a pass moves that this model does not charge, beyond the prefill floor: nothing — the drafter's
        passes are the verify's own (one DFlash2 block a pass), and they are priced."""
        return ""


#: THE TRAFFIC FAMILIES a catalog row's `traffic.family` names — one class per way a forward pass moves bytes, as the
#: hardware has one per memory family: a new model of an existing family is new data, a new family is a new class.
#: `dense`: every weight on every pass (`StepTraffic`, the 27B under SGLang). `routed`: a MoE whose routed experts are
#: read as the pass's rows touch them, over sparse attention (`RoutedStepTraffic`, Flash-Next under TensorFold).
TRAFFIC_FAMILIES = ("dense", "routed")


@dataclass(frozen=True)
class RoutedStepTraffic:
    """WHAT ONE FORWARD PASS MOVES THROUGH ONE RANK'S MEMORY for a ROUTED-EXPERT model over sparse attention — Qwen3.8
    Flash-Next as TensorFold's native engine lays it out (operator 2026-10-30: "Fix bandwidth in tensor fold"). The same
    interface as `StepTraffic` (`decode_pass_bytes`, `prefill_bytes`, `kv_cache_dtype_bytes`), so the MEMORY BANDWIDTH
    tile and the CUDA CORES tile price it unchanged.

    WHY A SECOND FAMILY. The dense model streams its whole target on every pass. Here 87 % of a rank's weights are 512
    routed experts of which a row reads 10, so what a pass streams depends on how many rows it carries: a 1-stream
    verify of 4 rows touches 7.6 % of them, a 2,048-row prompt chunk all of them. Its attention is SPARSE — past
    `index_budget` keys a row reads only the blocks its indexer selects, after scanning every block's pooled key —
    and its GDN state is folded in place (one read, one write a pass), not snapshotted per draft as SGLang's is.

    Every figure is per rank at `CLUSTER_TP` and sourced in the row: the weights off the checkpoint's own shard headers
    under the engine's loader split, the shapes off its config.json, the layouts off the engine's source. `omits` is
    what a pass moves that NOTHING here can count, said on the tile as a floor — never priced from a guess."""
    #: Every non-routed weight a pass streams: projections, the shared expert, router, hyper-connections, lm_head.
    dense_read_bytes: int
    #: All routed experts a rank holds; a pass reads the share its rows touch (`touched`).
    routed_expert_bytes: int
    experts: int
    top_k: int
    attention_layers: int
    num_key_value_heads: int
    head_dim: int
    kv_cache_dtype_bytes: float
    #: The FP8 cache's per key row trailer (two fp32 scales + 8 pad bytes); none in a bf16 cache.
    fp8_key_trailer_bytes: int
    index_budget: int
    index_ratio: int
    index_head_dim: int
    index_dtype_bytes: int
    linear_attention_layers: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_state_dtype_bytes: int
    linear_conv_kernel: int
    linear_conv_dtype_bytes: int
    #: Times a decode pass moves one sequence's GDN state: read, then written in place.
    state_transfers: int
    #: What a pass moves that is not charged, and why (a floor on the tile). Required: "" would claim completeness.
    omits: str
    source: str = ""

    def touched(self, rows: float | None) -> float:
        """The share of the routed experts a pass of `rows` rows reads: each row picks `top_k` of `experts`, taken as
        independent draws — 1 − (1 − k/E)^rows. `rows` None (not known) is a full chunk: every expert."""
        if rows is None:
            return 1.0
        return 1.0 - (1.0 - self.top_k / self.experts) ** max(0.0, float(rows))

    @property
    def kv_row_bytes(self) -> int:
        """One position's K and V in one attention layer on this rank (cuda_state.zig kRow + vRow): fp8 = 1 x (256 +
        16) + 1 x 256 = 528, the engine's own start-log figure; bf16 = 2 x 1 x 256 x 2 = 1,024."""
        heads = self.num_key_value_heads // CLUSTER_TP
        trailer = self.fp8_key_trailer_bytes if self.kv_cache_dtype_bytes == 1 else 0
        return math.ceil(heads * (2 * self.head_dim * self.kv_cache_dtype_bytes + trailer))

    @property
    def index_bytes_per_position(self) -> float:
        """The indexer's pooled keys a position: one `index_head_dim` key per `index_ratio` positions — 128 x 2 / 4."""
        return self.index_head_dim * self.index_dtype_bytes / self.index_ratio

    @property
    def state_bytes(self) -> int:
        """One sequence's GDN state on this rank (cuda_state.zig fixedSizes): the recurrent state
        36 x 24 x 128 x 128 x 4 B + the conv window 36 x 3 x (2·8·128 + 24·128) x 2 B = 57,729,024."""
        nk, nv = self.linear_num_key_heads // CLUSTER_TP, self.linear_num_value_heads // CLUSTER_TP
        recurrent = nv * self.linear_value_head_dim * self.linear_key_head_dim * self.linear_state_dtype_bytes
        conv = ((self.linear_conv_kernel - 1) * (2 * nk * self.linear_key_head_dim + nv * self.linear_value_head_dim)
                * self.linear_conv_dtype_bytes)
        return self.linear_attention_layers * (recurrent + conv)

    def decode_pass_bytes(self, *, context: float, seqs: float, verify_width) -> tuple[int, int, int]:
        """(weights, KV, GDN state) one DECODE pass moves on this rank. The pass carries `seqs` x `verify_width`
        rows (1 a stream without a drafter), which pick the routed experts it reads. Each stream's attention reads
        the keys it selects — at most `index_budget` a stream, so min(context, seqs x budget), exact when every
        context is past the budget and its rows select the same blocks (a floor where they select different ones) —
        and its indexer scans every position's pooled key. The GDN state is read and written in place."""
        rows = float(seqs) * (float(verify_width) if _drafting(verify_width) else 1.0)
        weights = round(self.dense_read_bytes + self.routed_expert_bytes * self.touched(rows))
        kv = (int(min(context, seqs * self.index_budget)) * self.attention_layers * self.kv_row_bytes
              + round(int(context) * self.attention_layers * self.index_bytes_per_position))
        return weights, kv, int(seqs) * self.state_bytes * self.state_transfers

    def prefill_bytes(self, rows: float | None = None) -> int:
        """A PROMPT CHUNK of `rows` rows (the window's mean, None = a full chunk): the non-routed weights and the routed
        experts its rows touch, once — a 2,048-row chunk reads them all. A floor, as the dense model's: the chunk's
        attention over its prefix is not charged."""
        return round(self.dense_read_bytes + self.routed_expert_bytes * self.touched(rows))


def _drafting(verify_width) -> bool:
    return isinstance(verify_width, (int, float)) and not isinstance(verify_width, bool) and verify_width >= 2


#: The knobs a TunedSetup may state: THE START INPUT a benchmark can be run at and compared on — the
#: concurrency, the context, the prefill budget and the drafter's depth. Deliberately not the KV pool, the PLE
#: placement or the disk tier: a profile already states its own pool and placement, and the disk tier is off
#: and unproven on this fleet. Every one is a field of `cluster.Desired` and a control on the panel.
TUNED_KNOBS = ("max_num_seqs", "max_model_len", "max_num_batched_tokens", "num_speculative_tokens")


@dataclass(frozen=True)
class TunedSetup:
    """THE BEST START INPUT MEASURED FOR ONE MODEL: what the panel offers, and what a launch uses, when the
    operator has chosen nothing — per ModelChoice, not per gateway.

    The default used to be ONE setup for every model: the gateway's configured concurrency and context
    (LB_CLUSTER_MAX_NUM_SEQS / LB_CLUSTER_MAX_MODEL_LEN, retired 2026-10-29) plus the box's prefill budget, which is
    the arithmetic maximum a pool holds and the environment's one number — neither of them a measurement of
    what runs FASTEST. So switching the model in the picker left the seats, the context and the budget that
    were tuned for the model before it, and the operator retyped three knobs from memory every time
    (2026-10-05). A TunedSetup is the answer to "what did we measure as best for THIS model", stated once,
    in the catalog, beside the rest of what the model is. Since 2026-10-29 every cluster row states at least its
    `max_num_seqs` and `max_model_len` (`_spec` refuses one without): the LB has no gateway-wide seats or window.

    It is ADVICE WITH EVIDENCE, like every other number in this file: `source` is required and must name the
    run it comes from, because a default nobody can trace is indistinguishable from a guess. It is not a
    limit — ModelLimits and tp-rank's caps still bind, the start gate still refuses a selection the boxes
    cannot hold, and the operator may set anything in range over it.

    PRECEDENCE, wherever a knob is left at its default: this setup, then the profile's own structural value
    (its `speculative_config`'s depth, SGLang's `engine_args.chunked_prefill_size`), then the fleet's
    (the cluster's prefill budget). One order, in `LaunchProfile.num_speculative_tokens` and `prefill_budget`
    for the two nullable knobs; the two that are always numbers are this setup's (`cluster.DesiredStore`).
    """
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    num_speculative_tokens: int | None = None
    #: WHICH RUN measured it. Required by `_tuned`: no source, no tuned setup.
    source: str = ""

    def get(self, knob: str) -> int | None:
        """This setup's value for one knob, or None where it states none."""
        if knob not in TUNED_KNOBS:
            raise KeyError(f"{knob!r} is not a tuned knob; they are {', '.join(TUNED_KNOBS)}")
        return getattr(self, knob)

    @property
    def stated(self) -> dict:
        """Only the knobs this setup really states — what a model switch applies, and nothing else."""
        return {knob: getattr(self, knob) for knob in TUNED_KNOBS if getattr(self, knob) is not None}

    def view(self) -> dict:
        """`models[].launch.cluster.tuned` — what the page pre-fills a model's controls from, and the
        evidence it can show for them."""
        return {**{knob: getattr(self, knob) for knob in TUNED_KNOBS}, "source": self.source}


@dataclass(frozen=True)
class LaunchProfile:
    #: THE IMAGE THIS PROFILE RUNS IN THIS DEPLOYMENT — RESOLVED, never written in the row (since 2026-10-30): the
    #: image matrix's entry for its engine's `build` on the target's accelerator and platform (`resolution`,
    #: vllm_lb/resolve.py). None where it does not resolve, and then `resolution.refused` says why and no Start of it
    #: is taken (`ModelCatalog.choice`).
    image: str | None
    #: Which BUILD of its engine it runs (library/images/<engine>.yml `builds`): an image recipe — base + patches.
    build: str = ""
    #: What it resolved to on this deployment's target (`resolve.Resolution`), or why it cannot run there.
    resolution: "Resolution | None" = None
    default: bool = False
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    speculative_config: dict | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_memory: int | None = None
    #: The PLE table on each rank (True) or split across the ranks (False); None = the model has none.
    ple_replicate: bool | None = None
    #: The MTP drafter's own tensor prefix for tp-rank's drafter view; "" = no drafter view; None = tp-rank's default.
    mtp_prefix: str | None = None
    #: A SEPARATE drafter checkpoint under the same models root (DFlash2 and anything else whose drafter is
    #: its own checkpoint rather than tensors inside the target's). "" = the model has none. Mutually
    #: exclusive with mtp_prefix: tp-rank splices exactly one "model" into the speculative config.
    draft_path: str = ""
    #: The server tp-rank launches for this profile: `vllm` (the default, and every profile before
    #: 2026-10-02) or `sglang`. It changes the command, the log wording the startup phases are read from,
    #: and the metric names the probe scrapes — not the address, which stays :8010 for every engine.
    engine: str = VLLM
    #: THE START INPUT for this model in this mode (`TunedSetup`): a cluster profile always has one, stating at
    #: least the seats and the context (`_spec`); a budget or depth it leaves out is the profile's own, then the
    #: fleet's. None only on a classic profile (broker_parrot's own vLLM).
    tuned: TunedSetup | None = None
    #: Engine settings that have no vLLM equivalent, passed through to tp-rank as it understands them.
    #: For SGLang, `kv_pool_calibration` is LB-only launch metadata: it turns the operator's byte pin into
    #: `--max-total-tokens` and the smallest safe `--mem-fraction-static`, then is not sent to the engine.
    engine_args: dict = field(default_factory=dict)
    #: Memory each rank role takes beyond weights and the KV cache: processes, CUDA graphs, activations.
    overhead_gib: dict = field(default_factory=dict)
    #: The vLLM engine flags this profile names (`VLLM_FLAGS`), {field: value}; empty = none, so none is sent.
    vllm_flags: dict = field(default_factory=dict)
    #: KV DTYPES THIS MODEL CANNOT RUN though its image takes them, {dtype: why} — a model x kernel fact (Flash-Next's
    #: SM121 QSA kernel takes bf16 KV only; GLM's MLA has no NVFP4 page). Refused at the switch in these words.
    kv_cache_dtype_refuses: dict = field(default_factory=dict)
    #: THE MODEL'S OWN ATTENTION BACKEND, where it brings one (`{name, reads, source}`): it alone reads the KV pool, so
    #: the image's generic backend rules do not apply and only the dtypes it `reads` run (vllm_lb/kv_dtype.py `problem`).
    #: None: the image's backends read it, as for every model before (2026-10-26, Flash-Next's QSA on Triton).
    kv_cache_dtype_backend: dict | None = None

    @property
    def kv_cache_dtype(self) -> str | None:
        """The KV dtype this profile launches when the run names none: a vLLM profile's own flag (None = no flag,
        vLLM's `auto`), an SGLang profile's engine_args, else tp-rank's SGLang default."""
        if self.engine == SGLANG:
            return str(self.engine_args.get("kv_cache_dtype", SGLANG_KV_CACHE_DTYPE))
        if self.engine == TENSORFOLD:
            return str(self.engine_args["kv_cache_dtype"])     # required (`_profile`): tp-rank has no default
        return self.vllm_flags.get("kv_cache_dtype")

    @property
    def refused(self) -> str:
        """Why this profile cannot launch in this deployment ("" when it can): its resolution's refusal."""
        return self.resolution.refused if self.resolution is not None else ""

    def kv_cache_dtype_choices(self, archs=()) -> list[dict]:
        """Every dtype, and why this profile cannot run it on cards of `archs` ("" where it can) — the page's options.
        An unresolved profile runs no image, so every dtype is refused with its resolution's reason."""
        if self.image is None:
            return [{"dtype": d, "refused": self.refused, "source": "", "attention_backends": None}
                    for d in kv_dtypes().vocabulary]
        return kv_dtypes().choices(self.image, archs, self.kv_cache_dtype_refuses, backend=self.kv_cache_dtype_backend)

    def _sglang_kv_calibration(self) -> tuple[int, float, float, float]:
        """`(reference bytes, reference fraction, pre-load GiB, safe fraction ceiling)` for SGLang.

        SGLang accepts an exact token-pool ceiling, but still reserves CUDA working memory through
        `mem_fraction_static`.  These are measurements from a successful launch of this profile, not a
        generic fraction of GPU RAM: the non-KV working set changes materially with the model, drafter and
        Mamba-seat count.  A profile without this calibration must not pretend a custom pin is safe.
        """
        raw = self.engine_args.get("kv_pool_calibration")
        base = self.engine_args.get("mem_fraction_static")
        if not isinstance(raw, dict) or not isinstance(base, (int, float)) or isinstance(base, bool):
            raise ValueError("SGLang profile has no KV-pool calibration")
        ref, available, ceiling = raw.get("reference_bytes"), raw.get("pre_load_gib"), raw.get("max_fraction")
        if (not isinstance(ref, int) or isinstance(ref, bool) or ref <= 0
                or not isinstance(available, (int, float)) or isinstance(available, bool) or available <= 0
                or not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool)
                or not 0 < float(base) <= float(ceiling) < 1):
            raise ValueError("SGLang KV-pool calibration is invalid")
        return ref, float(base), float(available), float(ceiling)

    def sglang_mem_fraction(self, pin_bytes: int) -> float:
        """The smallest measured-safe SGLang fraction that can honour `pin_bytes`.

        The calculation is anchored at a real pool, then spends only the additional requested bytes from
        the memory SGLang reported before it loaded the model.  It never lowers the measured baseline and
        it refuses a request past the profile's proven safety ceiling rather than silently shrinking it.
        """
        if not isinstance(pin_bytes, int) or isinstance(pin_bytes, bool) or pin_bytes <= 0:
            raise ValueError(f"SGLang KV pin must be a positive whole number of bytes, not {pin_bytes!r}")
        reference, base, pre_load_gib, ceiling = self._sglang_kv_calibration()
        fraction = base + max(0, pin_bytes - reference) / (pre_load_gib * (1 << 30))
        if fraction > ceiling + 1e-9:
            maximum = reference + int((ceiling - base) * pre_load_gib * (1 << 30))
            raise ValueError(f"SGLang KV pin {pin_bytes:,} B needs mem_fraction_static {fraction:.3f}, above "
                             f"this profile's safe {ceiling:.2f} ceiling (maximum {maximum:,} B)")
        return round(fraction, 4)

    def sglang_max_kv_cache_memory(self) -> int:
        """Largest absolute SGLang KV pin this profile can request without crossing its safe ceiling."""
        reference, base, pre_load_gib, ceiling = self._sglang_kv_calibration()
        return reference + int((ceiling - base) * pre_load_gib * (1 << 30))

    def kv_cache_memory_max(self) -> int | None:
        """THE LARGEST KV PIN THIS PROFILE CAN TAKE, in bytes — or None where the profile sets no bound of its own
        (vLLM pins what the start gate affords; a SGLang profile without a calibration cannot say).

        One fact, read by everything that could otherwise accept what the launch refuses: the switch
        (`cluster._check_pin_ceiling`), `auto` (`cluster_defaults.kv_auto` sizes under it), the estimate, the loop's
        launch check and the page (`launch.cluster.kv_cache_memory_max`, which greys the pins past it). Generation 187
        (2026-10-10): the switch accepted a 24 GB pin on the SGLang Flash-Next row, the launch refused it 131 times."""
        if self.engine != SGLANG:
            return None
        try:
            return self.sglang_max_kv_cache_memory()
        except ValueError:
            return None

    def kv_pin_problem(self, pin_bytes: int) -> str:
        """Why this profile's LAUNCH would refuse `pin_bytes` as its KV pin, in the launch's own words — "" when it
        would build. For SGLang that is `sglang_mem_fraction`'s refusal (the pin past the calibrated ceiling, with the
        maximum; or no calibration to size it by); a vLLM profile pins what it is told and leaves the box's memory to
        the start gate. The switch and the estimate refuse on these words so the launch never has to."""
        if self.engine != SGLANG:
            return ""
        try:
            self.sglang_mem_fraction(pin_bytes)
        except ValueError as exc:
            return str(exc)
        return ""

    @property
    def refuses(self) -> dict[str, str]:
        """The operator's knobs this profile's ENGINE does not take, each with why (`ENGINE_REFUSES`)."""
        return ENGINE_REFUSES.get(self.engine, {})

    @property
    def pool_per_seat(self) -> bool:
        """Whether this profile's engine reserves its KV pool one window a seat (`ENGINE_POOL_PER_SEAT`)."""
        return self.engine in ENGINE_POOL_PER_SEAT

    @property
    def prefill_budget(self) -> int | None:
        """The prefill budget this profile launches with when the operator gives none: the MEASURED-BEST one
        (`TunedSetup`) for whichever engine, else SGLang's `engine_args.chunked_prefill_size`, which tp-rank
        takes over the fleet's CLUSTER_MAX_NUM_BATCHED_TOKENS. None: the fleet's budget stands.

        A tuned budget reaches a vLLM pair too (`cluster_exec.up` sends it as CLUSTER_MAX_NUM_BATCHED_TOKENS),
        which is the point: the SGLang pair ran a day at 2,048 while its catalog entry still said 8,192, and
        Flash-Next's best is 4,096 against a box default of 8,192."""
        if self.tuned is not None and self.tuned.max_num_batched_tokens is not None:
            return self.tuned.max_num_batched_tokens
        value = self.engine_args.get("chunked_prefill_size") if self.engine == SGLANG else None
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def num_speculative_tokens(self) -> int:
        """What this profile's drafter proposes per step — the catalog's DEFAULT for the ModelChoice, which only
        the operator's own stored choice may override (`cluster.Desired.num_speculative_tokens`). 0: the model
        has no speculative decoding. `_profile` refuses a speculative config that does not say its count.

        The MEASURED-BEST depth (`TunedSetup`) comes first, so a model whose A/B chose another depth than its
        speculative config states launches at the one that was measured."""
        if self.tuned is not None and self.tuned.num_speculative_tokens is not None:
            return self.tuned.num_speculative_tokens
        return int(self.speculative_config["num_speculative_tokens"]) if self.speculative_config else 0

    def view(self) -> dict:
        # The MODEL-SCOPED defaults the page shows for the model IN THE PICKER — the pool this profile pins and
        # where its PLE table sits (null: the model has none, and there is nothing to place).
        return {"default": self.default, "image": self.image, "build": self.build,
                # WHAT IT RESOLVED TO HERE (engine × build × accelerator × platform × TP → image, with the matrix
                # entry's source) or why it cannot run on this deployment's cards: the page shows the refusal.
                "resolution": self.resolution.view() if self.resolution is not None else None,
                "speculative": self.speculative_config is not None,
                "num_speculative_tokens": self.num_speculative_tokens,
                # WHICH server, and WHICH drafter: the page draws both, and before 2026-10-03 it had
                # neither — so it called every drafter "MTP", including the two that draft with DFlash2,
                # and it could not tell two profiles of the same checkpoint apart at all.
                "engine": self.engine,
                "speculative_method": (self.speculative_config or {}).get("method") or None,
                "kv_cache_memory": self.kv_cache_memory, "ple_replicate": self.ple_replicate,
                # The largest pin this profile can take (null: no bound of its own) — the page greys the pins past
                # it and refuses a stored one before the Start, in the figure the switch would refuse with.
                "kv_cache_memory_max": self.kv_cache_memory_max(),
                # THE MEASURED-BEST START INPUT for this model, with its evidence: what the page pre-fills
                # the controls with the moment this model is picked (null: nothing measured for it).
                "tuned": self.tuned.view() if self.tuned is not None else None,
                # The knobs this engine does not take, and why: the page locks them for this model.
                "refuses": dict(self.refuses),
                # A POOL OF seats x context (`pool_per_seat`): the page sizes it from those two, never from a pin.
                "pool_per_seat": self.pool_per_seat,
                # The KV dtype it runs when the run names none (null: vLLM's auto, no flag sent).
                "kv_cache_dtype": self.kv_cache_dtype,
                # AN ENGINE THAT DRAFTS BY ITS OWN RULE (TensorFold: the checkpoint's MTP head, its chain stopped by
                # confidence) says whether it drafts at all; null for every profile whose depth the LB sets.
                "engine_drafts": bool(self.engine_args["drafts"]) if self.engine == TENSORFOLD else None}


@dataclass(frozen=True)
class UncensoredVariant:
    """A model's abliterated twin, as the catalog states it: a checkpoint BESIDE the original (same parent, so
    tp-rank's one mount reaches it), arch-exact with it (same architectures, quant_method, layer count and
    window — which is why the whole launch block applies unchanged), and which of the two is the default.

    `default` says WHICH OF THE MODEL'S TWO ROWS IS PRESELECTED — on the page each checkpoint is its own row in
    the model picker (the original, then its twin), and a Start that names no checkpoint (`uncensored: null`)
    resolves to this one. It does not mean "this row launches the twin": the row picked does that
    (`ModelSpec.launched_from`), and `default: true` leaves the original row startable as before.

    `calibration` is REQUIRED and says, in words the operator reads on the page, that the memory block it
    inherits was measured on the ORIGINAL: the twin's footprint is the same tensors in the same shapes, but
    nobody has watched it load, and an estimate must never pass for a measurement (`MemoryFootprint`)."""
    path: str
    default: bool
    calibration: str

    def view(self) -> dict:
        return {"path": self.path, "default": self.default, "calibration": self.calibration}


#: What the twin's MemoryFootprint says of itself, before the original's own calibration.
_INHERITED = "NOT MEASURED ON THIS CHECKPOINT"


def availability_key(spec_id: str, uncensored) -> str:
    """How a box's `Availability` answers are keyed: a ModelSpec's id for its original checkpoint, and
    `<id>@uncensored` for its twin — two checkpoints, two answers (`cluster_exec._availabilities`), so the
    planner reads the answer for the one it is about to launch. None, a switch from before the knob, launched
    the original."""
    return f"{spec_id}@uncensored" if uncensored else spec_id


@dataclass(frozen=True)
class EngineMemory:
    """What a whole engine holds in GPU memory with `max_num_seqs` requests in flight, in GiB — measured on a real boot
    (`source`: date, card, image, method). An engine with no pool of its own to size (CTranslate2) is fitted by it: no
    more concurrent requests than it was measured at, never a figure scaled from it."""
    gib: float
    max_num_seqs: int
    source: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    path: str
    served_name: str
    limits: ModelLimits
    launch: dict
    memory: MemoryFootprint | None = None
    size_gib: float | None = None
    #: "" when launchable; otherwise why not, with its evidence (M6: a model that cannot load on the image).
    unsupported: str = ""
    #: Where tp-rank looks for `<name>/config.json` on a box, best first: the catalog's roots joined with `path`.
    cluster_dirs: tuple[str, ...] = ()
    #: What one forward pass moves through a rank's memory (`StepTraffic` or `RoutedStepTraffic`, by its family);
    #: None = not recorded, and the page draws the engine's own byte estimate where it has one.
    traffic: StepTraffic | RoutedStepTraffic | None = None
    #: The abliterated twin this model may be launched from instead (`launched_from`); None = it has none,
    #: and asking for one is refused — never answered with the original.
    uncensored: UncensoredVariant | None = None
    #: True on the ModelSpec `launched_from(True)` returns: its `path` IS the twin's.
    launched_uncensored: bool = False
    #: The thinking levels its chat template takes, its default FIRST (`_efforts`); None = none stated, and
    #: nothing downstream invents one.
    efforts: tuple[str, ...] | None = None
    #: The checkpoint's QUANTISATION METHOD as vLLM reports it (`quant_method`, {value, source}, every row
    #: states it); None = unquantised, the source says so. What selects the node's MFU byte-size patch (fleet.MFU_PATCHED_QUANTS).
    quant_method: str | None = None
    #: It takes images (`vision`, {value, source}); None = not stated.
    vision: bool | None = None
    #: A chat can talk to it (`chat`, {value, source}, every row states it): the chat's server picker lists only
    #: servers whose model says True — a GUI-grounding model answers click(x,y), not prose.
    chat: bool | None = None
    #: The OpenAI API family it answers (`TASKS`, {value, source}, every row states it): what the LB routes by.
    task: str = CHAT
    #: The server a fleet node runs it on (`NODE_ENGINES`); None = no fleet node launches it.
    node_engine: str | None = None
    #: THE MODEL AXIS (docs/design/engines.md): what the checkpoint IS, read from its config.json — the format of its
    #: folder (`hf`, `ctranslate2`, `nemo`), its `architectures` (None = never read: every launch of it is refused,
    #: naming vllm_lb.model_facts), and its quant FORMAT (`quant_method`, `mlx` for MLX's own, `none` unquantised).
    #: The resolver checks each against the engine and the accelerator (resolve.py).
    format: str | None = None
    architectures: tuple[str, ...] | None = None
    quant_format: str | None = None
    #: WHAT ITS MEASUREMENT COVERS (library/measurements.json `covers`): (engine, accelerator, TP) triples its memory,
    #: tuned input and overhead were measured on. A cluster launch on a target outside them is refused, "not measured
    #: on …" (`_resolved`), unless the operator explicitly starts it sized by the estimate. () = never measured.
    measured_on: tuple[tuple[str, str, int], ...] = ()
    #: Where those figures were measured, in the measurement's own words.
    measured_source: str = ""
    #: THE WHOLE ENGINE'S GPU MEMORY, measured (`EngineMemory`) — the fit of an engine that sizes no pool from a
    #: fraction (faster-whisper); None = not measured.
    engine_memory: "EngineMemory | None" = None

    @property
    def uncensored_default(self) -> bool:
        """What a Start that names no checkpoint launches: the twin where there is one (the operator's rule),
        the only checkpoint there is where there is not."""
        return self.uncensored is not None and self.uncensored.default

    def launched_from(self, uncensored: bool) -> "ModelSpec":
        """This model as the engine loads it: itself, or — `uncensored` — the same ModelSpec with the twin's
        `path`, and a MemoryFootprint that SAYS its numbers are the original's. Everything else (id, launch
        block, limits, roots) is the same object's, which is the point: arch-exact twins share one launch.
        ValueError for a model with no twin — the one answer that is never "the original, then"."""
        if not uncensored:
            return self
        if self.uncensored is None:
            raise ValueError(f"{self.id} has no uncensored counterpart in the catalog: it launches only from "
                             f"{self.path}")
        memory = self.memory
        if memory is not None:
            memory = replace(memory, calibration=f"{_INHERITED}: {self.uncensored.calibration} The figures "
                                                 f"below were measured on {self.path} — {memory.calibration}")
        return replace(self, path=self.uncensored.path, memory=memory, launched_uncensored=True)

    @property
    def availability_key(self) -> str:
        return availability_key(self.id, self.launched_uncensored)

    def kv_cache_dtype_problem(self, dtype, archs=(), num_speculative_tokens: int | None = None) -> str:
        """Why this model's cluster launch cannot run KV `dtype` on cards of `archs` at drafter depth
        `num_speculative_tokens` (None: the profile's own) — "" when it can. The image's word (vllm_lb/kv_dtype.py, its
        refusal `with_drafter` among them), the profile's own refusals, and a dtype whose pool needs layer shapes this
        row lacks."""
        profile = self.profile(CLUSTER_MODE)
        if profile is None:
            return f"{self.id} has no {CLUSTER_MODE} LaunchProfile"
        if profile.image is None:
            return f"{self.id} runs no image on this deployment's cards: {profile.refused}"
        depth = profile.num_speculative_tokens if num_speculative_tokens is None else num_speculative_tokens
        problem = kv_dtypes().problem(profile.image, dtype, archs, profile.kv_cache_dtype_refuses, drafting=depth > 0,
                                      backend=profile.kv_cache_dtype_backend)
        took = kv_dtypes().accepted(profile.image, dtype) if not problem else None
        if took is not None and took.workspace_layers and self.traffic is None:
            return (f"{dtype}: its pool adds a per-token dequant workspace this row cannot price — it states no layer "
                    f"shapes (traffic)")
        return problem

    def kv_cache_dtype_choices(self, archs=()) -> list[dict]:
        """`kv_cache_dtype_choices` with this ROW's own refusals (`kv_cache_dtype_problem`) — what the page offers:
        `refused` with the drafter off (depth 0), `refused_with_drafter` at any depth above 0, so the page picks by the
        depth it shows and never re-derives the rule."""
        profile = self.profile(CLUSTER_MODE)
        out = profile.kv_cache_dtype_choices(archs) if profile is not None else []
        return [{**c, "refused": self.kv_cache_dtype_problem(c["dtype"], archs, 0),
                 "refused_with_drafter": self.kv_cache_dtype_problem(c["dtype"], archs, 1)} for c in out]

    def at_kv_cache_dtype(self, dtype: str | None) -> "ModelSpec":
        """THIS MODEL AT THE RUN'S KV DTYPE: every KV figure re-priced — `memory` (`MemoryFootprint.at`) and the bytes a
        forward pass streams (`traffic`). None, or the profile's own measured dtype, is this spec unchanged. A dtype
        whose pool carries a workspace (SGLang FP4) is priced from the layer shapes, exactly as the engine sizes it;
        without them it is refused (`kv_cache_dtype_problem`), never approximated."""
        if dtype is None or self.memory is None:
            return self
        lib = kv_dtypes()
        profile = self.profile(CLUSTER_MODE)
        took = lib.accepted(profile.image, dtype) if profile is not None else None
        b = lib.layouts[dtype].bytes_per_element
        workspace = took.workspace_layers if took is not None else 0
        if workspace and self.traffic is None:
            raise ValueError(self.kv_cache_dtype_problem(dtype))
        draft_b = (lib.layouts[took.draft_kv_cache_dtype].bytes_per_element
                   if took is not None and took.draft_kv_cache_dtype else None)
        per_token = self.traffic.pool_bytes_per_token(b, workspace, draft_b) if workspace else None
        traffic = replace(self.traffic, kv_cache_dtype_bytes=b) if self.traffic is not None else None
        return replace(self, memory=self.memory.at(dtype, per_token), traffic=traffic)

    @property
    def name(self) -> str:
        """The checkpoint folder itself — tp-rank's CLUSTER_MODEL_NAME."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def parent(self) -> str:
        """The path under the roots that `cluster_dirs` already carries; "" when the checkpoint sits at a root."""
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    @property
    def draft_name(self) -> str:
        """The drafter as tp-rank must be told it — RELATIVE to the dir it mounts, which is a root joined with
        `parent`, not the root itself. "" when the model's profile brings no separate drafter. `_spec` refuses a
        draft_path outside `parent`, so this is always a name beside the checkpoint."""
        draft = (self.profile(CLUSTER_MODE).draft_path if self.profile(CLUSTER_MODE) else "") or ""
        if not draft:
            return ""
        return draft[len(self.parent) + 1:] if self.parent else draft

    def profile(self, mode: str) -> LaunchProfile | None:
        return self.launch.get(mode)

    @property
    def calibrated(self) -> bool:
        """Its MemoryFootprint was measured on a real start (`calibration` names the run). False: the numbers
        are arithmetic nobody has checked, and nothing may be SIZED from them (`cluster_defaults.kv_auto`)."""
        return self.memory is not None and bool(self.memory.calibration.strip())

    def view(self) -> dict:
        return {"id": self.id, "label": self.label, "served_name": self.served_name, "size_gib": self.size_gib,
                "limits": self.limits.view(),
                "launch": {mode: self.launch[mode].view() for mode in SERVING_MODES if mode in self.launch},
                # What one token of KV costs one rank — the figure that turns a pool into tokens, per MODEL;
                # null without a MemoryFootprint. `calibrated`: that footprint was measured on a real start.
                "kv_bytes_per_token": self.memory.kv_bytes_per_token if self.memory is not None else None,
                "kv_blocks": (self.memory.kv_blocks.view()
                              if self.memory is not None and self.memory.kv_blocks is not None else None),
                "calibrated": self.calibrated,
                # The twin this model may be launched from, and whether it is the default; null: it has none.
                "uncensored": self.uncensored.view() if self.uncensored is not None else None,
                "efforts": list(self.efforts) if self.efforts is not None else None,
                "unsupported": self.unsupported}


@dataclass(frozen=True)
class Availability:
    """Whether one ModelSpec can be launched from a box, as the box answered. What was looked for travels with the
    answer, so the page and the planner can say exactly what is missing and where it was searched."""
    name: str
    searched: tuple[str, ...]
    dir: str | None
    image: str
    image_present: bool | None
    observed_at: float | None = None
    error: str = ""
    #: Found, but the rank's container could not read it (tp-rank `links_escaping`: a link out of the mounted folder,
    #: dangling or absolute — generation 279), file by file; "" when it can.
    unusable: str = ""

    @property
    def present(self) -> bool | None:
        """True: files and image both there. False: one of them is certainly missing, or the files are there and the
        container cannot read them. None: the box could not say."""
        if self.dir is None:
            return None if self.error else False
        if self.unusable:
            return False
        return self.image_present

    def reason(self) -> str:
        """Why this checkpoint cannot be launched from the box (`present` False), or why the box could not say
        (`present` None), in words the planner quotes; "" when it can. A folder the box could not LOOK in
        (`error`: its share not mounted, a stale handle, a hang) is never worded as the file being absent —
        "no …/config.json" is said only of folders that were read."""
        if self.dir is None:
            if self.error:
                return self.error
            return f"no {self.name}/config.json in {', '.join(self.searched) or 'any folder'}"
        if self.unusable:
            return f"{self.name} in {self.dir} cannot be read from the container: {self.unusable}"
        if self.image_present is False:
            return f"no image {self.image}"
        if self.image_present is None:
            return (f"docker on the box could not say whether the image {self.image} is there"
                    + (f" ({self.error})" if self.error else ""))
        return ""

    def view(self) -> dict:
        return {"name": self.name, "searched": list(self.searched), "dir": self.dir, "image": self.image,
                "image_present": self.image_present, "present": self.present, "observed_at": self.observed_at,
                "error": self.error, "unusable": self.unusable}


@dataclass(frozen=True)
class LoadedModel:
    served_name: str
    spec: str | None
    label: str | None
    max_model_len: int | None
    #: The served spec's `efforts`; None for a stranger or a spec that states none.
    efforts: tuple[str, ...] | None = None

    def view(self) -> dict:
        return {"served_name": self.served_name, "spec": self.spec, "label": self.label,
                "max_model_len": self.max_model_len,
                "efforts": list(self.efforts) if self.efforts is not None else None}


# ── the catalog ───────────────────────────────────────────────────────────────


class ModelCatalog:
    def __init__(self, specs=(), *, cluster_roots=(), error: str = "", absent=None) -> None:
        self.specs: tuple[ModelSpec, ...] = tuple(specs)
        self.cluster_roots: tuple[str, ...] = tuple(cluster_roots)
        #: The library's models this deployment cannot launch at all, each with why (no checkpoint folder named in
        #: its fleet file): not specs here, and a Start naming one is refused in these words (`choice`).
        self.absent: dict[str, str] = dict(absent or {})
        #: "" when the catalog loaded; otherwise why it did not — and then nothing can be chosen.
        self.error = error
        self._by_id = {s.id: s for s in self.specs}
        self._by_served = {s.served_name: s for s in self.specs}

    @classmethod
    def load(cls, *, deployment: "Deployment | None" = None, roots=None) -> "ModelCatalog":
        """The catalog — the library ASSEMBLED for a deployment (`document`) and resolved against its targets — never
        raising: the LB listens first, and a catalog it cannot read is a status line. `deployment`: default this LB's
        own, from the fleet file LB_FLEET names (`deployment`)."""
        try:
            dep = current_deployment() if deployment is None else deployment
            return cls.from_dict(document(dep), roots=roots, targets=dep.targets)
        except (OSError, ValueError) as exc:
            return cls(error=f"the model catalog: {type(exc).__name__}: {exc}"[:300])

    @classmethod
    def from_dict(cls, raw, roots=None, *, targets=None) -> "ModelCatalog":
        """A catalog document, validated strictly: any rule broken refuses the whole document with the reason.
        `roots`: the folders the cluster searches, MODELS_SEARCH_DIRS from the environment unless given.
        `targets`: ServingMode → the `resolve.Target` its profiles launch onto in this deployment, or why there is
        none (a string) — every profile's image is RESOLVED against it (`LaunchProfile.resolution`). Default: the
        deployment's own, read from the fleet file LB_FLEET names (`deployment_targets`)."""
        targets = deployment_targets() if targets is None else targets
        if isinstance(raw, dict) and "cluster_roots" in raw:
            raise ValueError(f"`cluster_roots` moved out of the catalog: set {MODELS_SEARCH_DIRS} in lb/.env")
        _keys(raw, "the catalog", required={"models"}, optional={"absent"})
        if not isinstance(raw["models"], list) or (not raw["models"] and not raw.get("absent")):
            raise ValueError("the catalog's `models` must be a non-empty list")
        absent = raw.get("absent") or {}
        if not isinstance(absent, dict) or not all(isinstance(v, str) and v for v in absent.values()):
            raise ValueError("the catalog's `absent` must map a model id to why this deployment cannot launch it")
        roots = models_search_dirs() if roots is None else list(roots)
        specs = [_spec(item, roots, targets) for item in raw["models"]]
        for attr in ("id", "served_name"):
            seen = [getattr(s, attr) for s in specs]
            dupes = sorted({v for v in seen if seen.count(v) > 1})
            if dupes:
                raise ValueError(f"the catalog repeats {attr} {', '.join(dupes)}")
        for mode in SERVING_MODES:
            launchable = [s for s in specs if s.profile(mode) is not None]
            defaults = [s.id for s in launchable if s.profile(mode).default]
            # Exactly one default — unless the library's default is a model this deployment has not staged (`absent`):
            # then none, and a Start must name its model (DesiredStore: no default, nothing preselected).
            if launchable and len(defaults) != 1 and not (not defaults and absent):
                raise ValueError(f"the {mode} ServingMode needs exactly one default, found {defaults or 'none'}")
        return cls(specs, cluster_roots=tuple(roots or ()), absent=absent)

    # ── reads ────────────────────────────────────────────────────────────

    def get(self, spec_id) -> ModelSpec | None:
        return self._by_id.get(spec_id) if isinstance(spec_id, str) else None

    def for_mode(self, mode: str) -> tuple[ModelSpec, ...]:
        return tuple(s for s in self.specs if s.profile(mode) is not None)

    def checkpoints(self, mode: str) -> tuple[ModelSpec, ...]:
        """Every CHECKPOINT a ServingMode may launch: each model, then each model's twin where it has one —
        what a box is asked about (`cluster_loop._observe_availability`), each under its own
        `availability_key`."""
        models = self.for_mode(mode)
        return models + tuple(s.launched_from(True) for s in models if s.uncensored is not None)

    def default(self, mode: str) -> ModelSpec | None:
        return next((s for s in self.for_mode(mode) if s.profile(mode).default), None)

    def choice(self, mode: str, spec_id, *, estimated: bool = False) -> ModelSpec:
        """THE validator of a ModelChoice, for both ServingModes. UnknownModelChoice: the caller named nothing
        launchable there. ValueError: the catalog itself cannot serve it (not loaded, or marked unsupported)."""
        if self.error:
            raise ValueError(f"the model catalog could not be loaded: {self.error}")
        spec = self.get(spec_id)
        if spec is None and isinstance(spec_id, str) and spec_id in self.absent:
            raise ValueError(f"{spec_id} cannot start in this deployment: {self.absent[spec_id]}")
        if mode not in SERVING_MODES or spec is None or spec.profile(mode) is None:
            known = ", ".join(s.id for s in self.for_mode(mode)) or "nothing"
            raise UnknownModelChoice(f"unknown model_choice {spec_id!r} for the {mode} ServingMode; "
                                     f"the catalog has: {known}")
        if spec.unsupported:
            raise ValueError(f"{spec.label} is marked unsupported: {spec.unsupported}")
        # RESOLUTION AT START: a profile whose engine, accelerator, image or KV dtype does not resolve on this
        # deployment's cards is refused here, naming the missing piece (resolve.py) — never launched on a substitute.
        # ONE refusal the operator may lift, explicitly, for this Start: "not measured on …" (`estimated`), and only
        # where the estimate can be made (`estimated_footprint`).
        resolution = spec.profile(mode).resolution
        refused = spec.profile(mode).refused
        unmeasured = resolution is not None and resolution.sizing == "unmeasured"
        if estimated and not unmeasured:
            raise ValueError(f"{spec.label} is not sized by the estimate here: " + (
                "it is measured on these cards" if not refused else refused))
        if refused and not (unmeasured and estimated):
            raise ValueError(f"{spec.label} cannot start on this deployment's cards: {refused}")
        if unmeasured and spec.memory is None:
            raise ValueError(f"{spec.label} is not measured on these cards and cannot be estimated: {resolution.estimate}")
        return spec

    def loaded_model(self, served_models, max_model_len) -> LoadedModel | None:
        """What an endpoint serves, as its /v1/models said it (`Backend.served_models`); None when it said nothing."""
        ids = [str(i) for i in (served_models or ()) if i]
        if not ids:
            return None
        spec = next((self._by_served[i] for i in ids if i in self._by_served), None)
        return LoadedModel(served_name=spec.served_name if spec else ids[0], spec=spec.id if spec else None,
                           label=spec.label if spec else None, max_model_len=max_model_len,
                           efforts=spec.efforts if spec else None)

    def view(self) -> list[dict]:
        return [s.view() for s in self.specs]

    def resolution_lines(self) -> list[str]:
        """What each profile RESOLVED TO on this deployment's cards, one line each — the boot log's record
        (__main__): `<id> <mode>: <engine>/<build> on <accelerator × platform × TP> → <image> [<source>]; sized by
        <measurement>`, or `… REFUSED: <why>`; and every model this deployment did not stage."""
        out = []
        for s in self.specs:
            for mode, p in s.launch.items():
                r = p.resolution
                where = r.target.key if r is not None and r.target is not None else "no target"
                head = f"{s.id} {mode}: {p.engine}/{p.build} on {where}"
                if p.refused:
                    out.append(f"{head} REFUSED: {p.refused}")
                else:
                    sized = f"; sized by {r.measurement}" if r.measurement else ""
                    out.append(f"{head} → {p.image} [{r.image_source}]{sized}")
        out += [f"{mid} absent: {why}" for mid, why in self.absent.items()]
        return out


# ── the rules a catalog document keeps ────────────────────────────────────────

#: `quant_method` is REQUIRED ({value, source}; null = unquantised, said so): an absent one would read as "no patch" and
#: drop the FFN from a node's FLOP/s and GB/s unseen (AUDIT-FINAL-5 P2-5).
_SPEC_REQUIRED = {"id", "label", "path", "served_name", "limits", "launch", "quant_method", "chat", "task",
                  "format", "architectures", "quant_format"}
_SPEC_OPTIONAL = {"memory", "size_gib", "unsupported", "traffic", "uncensored", "efforts", "vision", "node_engine",
                  "engine_memory", "measured_on", "shape"}
_PROFILE_OPTIONAL = {"default", "tool_call_parser", "reasoning_parser", "speculative_config",
                     "gpu_memory_utilization", "kv_cache_memory", "ple_replicate", "mtp_prefix", "overhead_gib",
                     "draft_path", "engine", "engine_args", "tuned", "kv_cache_dtype_refuses",
                     "kv_cache_dtype_backend", *VLLM_FLAGS}
_MEMORY_REQUIRED = {"weights_gib", "ple_gib", "mtp_gib", "kv_bytes_per_token", "seq_state_gib", "kv_cache_dtype"}


def _keys(obj, where: str, *, required: set, optional: set = frozenset()) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    missing, unknown = required - set(obj), set(obj) - required - set(optional)
    if missing or unknown:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{where} must be {'a' if empty else 'a non-empty'} string")
    return value


def _architectures(raw, where: str) -> tuple[str, ...] | None:
    """A row's `architectures` fact: the names its config.json lists (an empty list where its format has none), or
    None stated null with why — never read."""
    v = _fact_of(raw, where, list, required=True)
    if v is not None and not all(isinstance(a, str) and a.strip() == a and a for a in v):
        raise ValueError(f"{where}.value must list architecture names, got {v!r}")
    return tuple(v) if v is not None else None


def _resolved(launch: dict, *, targets, where: str, fmt, architectures, quant_format,
              measured_on=(), measured_source: str = "") -> dict:
    """Each LaunchProfile with its image RESOLVED for this deployment's target of its mode (resolve.py) — or its
    refusal recorded, image None. `targets`: mode → Target, or the reason there is none.

    A CLUSTER profile is also held to its MEASUREMENT: its memory, tuned input and overhead were measured on the
    (engine, accelerator, TP) triples `measured_on` names, and a target outside them is REFUSED by default, "not
    measured on …" (`Resolution.sizing` "unmeasured") — the operator may still start it, explicitly, sized by the
    estimate (`ModelCatalog.choice(estimated=True)`). A classic profile is broker_parrot's to size: no such check."""
    out = {}
    for mode, p in launch.items():
        t = targets.get(mode)
        res = resolve(engine=p.engine, build=p.build, target=t if isinstance(t, Target) else None, fmt=fmt,
                      architectures=architectures, quant_format=quant_format, kv_cache_dtype=p.kv_cache_dtype,
                      kv_refuses=p.kv_cache_dtype_refuses, kv_backend=p.kv_cache_dtype_backend,
                      drafting=p.num_speculative_tokens > 0,
                      no_target=t if isinstance(t, str) else f"this deployment has no {mode} target")
        if mode == CLUSTER_MODE and not res.refused:
            key = (p.engine, res.target.accelerator.key, res.target.tp)
            if key in measured_on:
                res = replace(res, sizing="measured", measurement=f"{p.engine} × {key[1]} × TP={key[2]}")
            else:
                done = "; ".join(f"{e} × {a} × TP={n}" for e, a, n in measured_on) or "nothing"
                res = replace(res, sizing="unmeasured", refused=(
                    f"not measured on {key[1]} at TP={key[2]} with {p.engine}: its memory, tuned input and overhead "
                    f"were measured on {done}" + (f" ({measured_source})" if measured_source else "")
                    + " — start it explicitly sized by the estimate (estimated: true), or measure it there"))
        out[mode] = replace(p, image=res.image, resolution=res)
    return out


#: The layer shapes a model's config.json states (vllm_lb/model_facts.py `shape`), each {value, source}: what the
#: memory ESTIMATE is computed from.
SHAPE_KEYS = ("num_hidden_layers", "full_attention_layers", "num_key_value_heads", "head_dim",
              "max_position_embeddings", "kv_lora_rank")


def _shape(raw, where: str) -> dict:
    """A row's `shape` facts as {key: value} (None where stated null) — {} when the row states none."""
    if raw is None:
        return {}
    _keys(raw, f"{where}: shape", required=set(), optional=set(SHAPE_KEYS))
    out = {}
    for k, f in raw.items():
        v = _fact_of(f, f"{where}: shape.{k}", int, required=True)
        if v is not None and v < 0:
            raise ValueError(f"{where}: shape.{k} must be a whole number from 0, got {v!r}")
        out[k] = v
    return out


def estimated_footprint(shape: dict, *, size_gib, tp: int, kv_cache_dtype: str | None) -> tuple:
    """THE ESTIMATE (docs/design/engines.md, "Unmeasured combination"): a model's per-rank footprint computed from
    its config.json's shapes, for a target nobody measured it on — `(MemoryFootprint, how)`, or `(None, why not)`.

      weights_gib         the checkpoint on disk (`size_gib`) over the TP size — weights split evenly across ranks
      kv_bytes_per_token  2 (K and V) × full-attention layers × KV heads a rank holds (heads / TP, at least 1) ×
                          head_dim × the bytes of an element at the KV dtype the launch runs (kv_cache_dtypes.json)
      the rest            0, and NOT MEASURED: the PLE table, the drafter, linear-attention state, activations and
                          the per-seat slope are not estimated; the profile's overhead is the measured target's

    Its `calibration` is EMPTY on purpose: nothing may be SIZED from it as if measured (an `auto` pool refuses it,
    `ModelSpec.calibrated`), and every view labels it "estimated". An MLA model (kv_lora_rank) is not estimated: its
    latent cache is not heads × head_dim. Never chosen by the LB: only a Start that asks (`estimated: true`)."""
    missing = [k for k in ("full_attention_layers", "num_key_value_heads", "head_dim") if shape.get(k) is None]
    if shape.get("kv_lora_rank"):
        return None, ("its attention is MLA (kv_lora_rank " + str(shape["kv_lora_rank"]) + "): a latent KV cache is "
                      "not heads × head_dim, and the estimate does not model it")
    if missing:
        return None, (f"its config.json shapes are not in the library ({', '.join(missing)}): run "
                      f"`python -m vllm_lb.model_facts <checkpoint>` and add them to library/models.json `shape`")
    if not isinstance(size_gib, (int, float)) or isinstance(size_gib, bool) or size_gib <= 0:
        return None, "its size on disk (size_gib) is not stated, so its weights cannot be estimated"
    if kv_cache_dtype is None or kv_cache_dtype not in kv_dtypes().layouts:
        return None, (f"its recipe names no KV dtype with a known element size ({kv_cache_dtype or 'none'}): the "
                      f"estimate prices the cache at the dtype the launch runs")
    heads = max(1, shape["num_key_value_heads"] // tp)
    per_token = math.ceil(2 * shape["full_attention_layers"] * heads * shape["head_dim"]
                          * kv_dtypes().layouts[kv_cache_dtype].bytes_per_element)
    weights = round(float(size_gib) / tp, 3)
    how = (f"ESTIMATED, NOT MEASURED: weights {weights} GiB a rank = {size_gib} GiB on disk / TP {tp}; KV "
           f"{per_token:,} B a token a rank = 2 × {shape['full_attention_layers']} full-attention layers × {heads} KV "
           f"heads × {shape['head_dim']} head_dim × {kv_cache_dtype}; PLE, drafter, sequence state and activations "
           f"not estimated (0)")
    return MemoryFootprint(weights_gib=weights, ple_gib=0.0, mtp_gib=0.0, kv_bytes_per_token=per_token,
                           seq_state_gib=0.0, calibration="", kv_cache_dtype=kv_cache_dtype), how


def _measured_on(raw, where: str) -> tuple[tuple, str]:
    """A row's `measured_on` ({covers: [{engine, accelerator, tp}], source}, assembled from
    library/measurements.json) as (engine, accelerator, tp) triples and its source; ((), "") for none."""
    if raw is None:
        return (), ""
    _keys(raw, f"{where}: measured_on", required={"covers", "source"})
    return tuple((c["engine"], c["accelerator"], int(c["tp"])) for c in raw["covers"]), raw["source"]


def _spec(raw, roots, targets) -> ModelSpec:
    _keys(raw, "a model", required=_SPEC_REQUIRED, optional=_SPEC_OPTIONAL)
    spec_id = _text(raw["id"], "a model's id")
    if not _ID.fullmatch(spec_id):
        raise ValueError(f"model id {spec_id!r} must match {_ID.pattern}")
    where = f"model {spec_id}"
    path = _checkpoint_path(raw["path"], f"{where}: path")
    uncensored = _uncensored(raw["uncensored"], where, path) if raw.get("uncensored") is not None else None
    launch_raw = raw["launch"]
    # tp-rank's floor on the window binds a row it can launch; a row only a fleet launches (`launch: {}`) has its
    # engine's own (Whisper's decoder: 448 tokens).
    limits = _limits(raw["limits"], where, floor=MAX_MODEL_LEN_CAP[0] if launch_raw else 1)
    # `{}`: a row only a FLEET launches — a one-host node whose launch block (image, figures, parsers) is the fleet
    # file's (fleet.py `_launch`): no ServingMode of the LB's own starts it, so no profile is invented for it.
    if not isinstance(launch_raw, dict):
        raise ValueError(f"{where}: `launch` must map ServingModes to profiles ({{}} for a row only a fleet launches)")
    unknown = set(launch_raw) - set(SERVING_MODES)
    if unknown:
        raise ValueError(f"{where}: no ServingMode {sorted(unknown)}; the modes are {', '.join(SERVING_MODES)}")
    launch = {mode: _profile(launch_raw[mode], f"{where}: {mode}", cluster=mode == CLUSTER_MODE)
              for mode in SERVING_MODES if mode in launch_raw}
    memory = None
    if "memory" in raw:
        memory = _memory(raw["memory"], where)
    # A cluster recipe whose measurement states no footprint is a document error; a model with NO measurement at all is
    # "not measured" on every target (`_resolved`) — refused per row, sized by the estimate only when the operator asks.
    if CLUSTER_MODE in launch and memory is None and raw.get("measured_on") is not None:
        raise ValueError(f"{where}: a cluster LaunchProfile needs a memory footprint")
    dirs: tuple[str, ...] = ()
    if CLUSTER_MODE in launch:
        if not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r.strip() for r in roots):
            raise ValueError(f"{MODELS_SEARCH_DIRS} is not set: the folders the cluster's checkpoints live under on "
                             f"each box, best first (lb/.env, e.g. ~/models:/mnt/models)")
        group = path.rsplit("/", 1)[0] if "/" in path else ""
        dirs = tuple(f"{r.rstrip('/')}/{group}" if group else r.rstrip("/") for r in roots)
        # tp-rank mounts ONE dir — a root joined with `group` — and names everything inside it relative to
        # that. A drafter anywhere else is not reachable through that mount, so it is a catalog error here
        # rather than a start that fails on both boxes with a path nobody can see from the page.
        draft = launch[CLUSTER_MODE].draft_path
        if draft:
            draft_group = draft.rsplit("/", 1)[0] if "/" in draft else ""
            if draft_group != group:
                raise ValueError(f"{where}: draft_path {draft!r} must sit beside the checkpoint, under "
                                 f"{group or 'the models root'} — tp-rank mounts that one dir and reaches "
                                 f"nothing outside it")
    # A TUNED SETUP MUST BE STARTABLE: the switch bounds every selection by the ModelSpec's own ModelLimits,
    # so a tuned default outside them could never be applied — a document error, not a refused Start.
    tuned = launch[CLUSTER_MODE].tuned if CLUSTER_MODE in launch else None
    # …AND A CLUSTER ROW STATES THE SEATS AND CONTEXT IT IS OFFERED (operator, 2026-10-29): the gateway-wide
    # LB_CLUSTER_MAX_NUM_SEQS / LB_CLUSTER_MAX_MODEL_LEN that stood in for a row with nothing stated are retired, so a
    # row without them would launch at a number nobody chose for that model. A figure nobody benched is still stated,
    # with that said in its `source`.
    if CLUSTER_MODE in launch and (tuned is None or tuned.max_num_seqs is None or tuned.max_model_len is None):
        raise ValueError(f"{where}: a cluster LaunchProfile must state tuned.max_num_seqs and tuned.max_model_len — "
                         f"the concurrency and context a Start of it is offered, with their source (a measured sweep, "
                         f"or the figures it has run at, said so); the LB has no gateway-wide default "
                         f"(LB_CLUSTER_MAX_NUM_SEQS and LB_CLUSTER_MAX_MODEL_LEN are retired)")
    if tuned is not None:
        lo_seqs, hi_seqs = limits.max_num_seqs
        if tuned.max_num_seqs is not None and not lo_seqs <= tuned.max_num_seqs <= hi_seqs:
            raise ValueError(f"{where}: tuned max_num_seqs {tuned.max_num_seqs} is outside this model's own "
                             f"limits {lo_seqs}..{hi_seqs}")
        if tuned.max_model_len is not None and not MAX_MODEL_LEN_CAP[0] <= tuned.max_model_len <= limits.max_model_len:
            raise ValueError(f"{where}: tuned max_model_len {tuned.max_model_len} is outside this model's own "
                             f"limits {MAX_MODEL_LEN_CAP[0]}..{limits.max_model_len}")
    size = raw.get("size_gib")
    if size is not None and not (_number(size) and size >= 0):
        raise ValueError(f"{where}: size_gib must be a number of GiB")
    traffic = _traffic(raw["traffic"], where) if raw.get("traffic") is not None else None
    task = _fact_of(raw["task"], f"{where}: task", str, required=True)
    if task not in TASKS:
        raise ValueError(f"{where}: task must be one of {', '.join(TASKS)}, got {task!r}")
    if launch and task != CHAT:
        raise ValueError(f"{where}: task {task} with a {'/'.join(launch)} LaunchProfile — the LB's own launchers run "
                         f"chat engines; a {task} model is launched by a fleet node (`launch: {{}}`)")
    engine = raw.get("node_engine")
    if engine is not None and engine not in NODE_ENGINES:
        raise ValueError(f"{where}: node_engine must be one of {', '.join(NODE_ENGINES)}, got {engine!r}")
    fmt = _fact_of(raw["format"], f"{where}: format", str, required=True)
    architectures = _architectures(raw["architectures"], f"{where}: architectures")
    quant_format = _fact_of(raw["quant_format"], f"{where}: quant_format", str, required=True)
    measured_on, measured_source = _measured_on(raw.get("measured_on"), where)
    launch = _resolved(launch, targets=targets, where=where, fmt=fmt, architectures=architectures,
                       quant_format=quant_format, measured_on=measured_on, measured_source=measured_source)
    shape = _shape(raw.get("shape"), where)
    # NOT MEASURED ON THIS TARGET: the footprint another target measured is not this one's. What the row carries is
    # the ESTIMATE from its config.json instead (`estimated_footprint`), for the one Start that explicitly asks for it
    # (`ModelCatalog.choice(estimated=True)`) — or none, and the refusal says why the estimate cannot be made either.
    cluster = launch.get(CLUSTER_MODE)
    if cluster is not None and cluster.resolution.sizing == "unmeasured":
        estimate, how = estimated_footprint(shape, size_gib=raw.get("size_gib"), tp=cluster.resolution.target.tp,
                                            kv_cache_dtype=cluster.kv_cache_dtype)
        memory = estimate
        launch = {**launch, CLUSTER_MODE: replace(cluster, resolution=replace(cluster.resolution, estimate=how))}
    return ModelSpec(format=fmt, architectures=architectures, quant_format=quant_format, measured_on=measured_on,
                     measured_source=measured_source,id=spec_id, label=_text(raw["label"], f"{where}: label"), path=path,
                     served_name=_text(raw["served_name"], f"{where}: served_name"), limits=limits, launch=launch,
                     memory=memory, size_gib=size, unsupported=_text(raw.get("unsupported", ""), f"{where}: "
                                                                     "unsupported", empty=True),
                     cluster_dirs=dirs, traffic=traffic, uncensored=uncensored,
                     efforts=_efforts(raw["efforts"], where) if "efforts" in raw else None,
                     quant_method=_fact_of(raw["quant_method"], f"{where}: quant_method", str, required=True),
                     vision=_fact_of(raw.get("vision"), f"{where}: vision", bool),
                     chat=_fact_of(raw["chat"], f"{where}: chat", bool, required=True), task=task,
                     node_engine=engine,
                     engine_memory=_engine_memory(raw["engine_memory"], where) if "engine_memory" in raw else None)


def _engine_memory(raw, where: str) -> EngineMemory:
    _keys(raw, f"{where}: engine_memory", required={"gib", "max_num_seqs", "source"})
    if not (_number(raw["gib"]) and raw["gib"] > 0):
        raise ValueError(f"{where}: engine_memory gib must be a number of GiB above 0, not {raw['gib']!r}")
    if not (_int(raw["max_num_seqs"]) and raw["max_num_seqs"] >= 1):
        raise ValueError(f"{where}: engine_memory max_num_seqs must be a whole number from 1, "
                         f"not {raw['max_num_seqs']!r}")
    return EngineMemory(gib=float(raw["gib"]), max_num_seqs=raw["max_num_seqs"],
                        source=_text(raw["source"], f"{where}: engine_memory source"))


def _fact_of(raw, where: str, kind: type, required: bool = False):
    """A row's `{value, source}` fact — its value (None absent, or stated null with its reason) — or ValueError; a
    `required` fact must be the mapping (null as a VALUE, with its source — never the key's absence)."""
    if raw is None:
        if required:
            raise ValueError(f"{where} must be stated as {{value, source}} (value null with why, for none)")
        return None
    _keys(raw, where, required={"value", "source"})
    _text(raw["source"], f"{where}.source")
    v = raw["value"]
    if v is not None and not (isinstance(v, kind) and (kind is not str or v.strip())):
        raise ValueError(f"{where}.value must be a {kind.__name__} or null, got {v!r}")
    return v


#: The word that turns thinking OFF: never an effort the backend sees — the LB sends thinking disabled
#: (router.rewrite_effort), since vLLM refuses "none"/"minimal" as an effort (400 literal_error, 2026-10-14).
EFFORT_OFF = "off"
#: Every word a ladder may hold: the Anthropic effort words (Claude Code's own ladder) and EFFORT_OFF.
EFFORT_WORDS = (EFFORT_OFF, "low", "medium", "high", "xhigh", "max")


def _efforts(raw, where: str) -> tuple[str, ...]:
    """A model's `efforts` block: the thinking levels its chat template takes, its default FIRST, and the
    `source` that says where they were read (the template's own lines). Stated, never inferred."""
    _keys(raw, f"{where}: efforts", required={"ladder", "source"})
    ladder = raw["ladder"]
    if not isinstance(ladder, list) or not ladder:
        raise ValueError(f"{where}: efforts ladder must be a non-empty list")
    for word in ladder:
        if word not in EFFORT_WORDS:
            raise ValueError(f"{where}: unknown effort {word!r}; the words are {', '.join(EFFORT_WORDS)}")
    if len(set(ladder)) != len(ladder):
        raise ValueError(f"{where}: efforts ladder repeats a word: {ladder}")
    _text(raw["source"], f"{where}: efforts source")
    return tuple(ladder)


def _checkpoint_path(value, where: str) -> str:
    path = _text(value, where)
    segments = path.split("/")
    if any(s in ("", ".", "..") or not _SEGMENT.fullmatch(s) for s in segments):
        raise ValueError(f"{where} {path!r} must be relative, of [A-Za-z0-9_.-] folders")
    return path


def _uncensored(raw, where: str, path: str) -> UncensoredVariant:
    """A model's `uncensored` block: the twin's path (beside the original — tp-rank mounts ONE dir, a root
    joined with the original's parent, and names everything inside it), whether it is the default (stated,
    never assumed) and the note that says its footprint is inherited (stated, never empty)."""
    _keys(raw, f"{where}: uncensored", required={"path", "default", "calibration"})
    twin = _checkpoint_path(raw["path"], f"{where}: uncensored path")
    if twin == path:
        raise ValueError(f"{where}: uncensored path {twin!r} is the original itself")
    group = path.rsplit("/", 1)[0] if "/" in path else ""
    twin_group = twin.rsplit("/", 1)[0] if "/" in twin else ""
    if twin_group != group:
        raise ValueError(f"{where}: uncensored path {twin!r} must sit beside the checkpoint, under "
                         f"{group or 'the models root'} — tp-rank mounts that one dir and reaches nothing "
                         f"outside it")
    if not isinstance(raw["default"], bool):
        raise ValueError(f"{where}: uncensored default must be true or false")
    return UncensoredVariant(path=twin, default=raw["default"],
                             calibration=_text(raw["calibration"], f"{where}: uncensored calibration"))


#: StepTraffic's whole-number fields; the head counts must split evenly over the pair's ranks.
_TRAFFIC_INTS = ("target_read_bytes", "draft_read_bytes", "full_attention_layers", "num_key_value_heads",
                 "head_dim", "kv_cache_dtype_bytes", "linear_attention_layers", "linear_num_value_heads",
                 "linear_key_head_dim", "linear_value_head_dim", "linear_state_dtype_bytes",
                 "draft_attention_layers", "draft_num_key_value_heads", "draft_head_dim", "draft_sliding_window")
_TRAFFIC_HEADS = ("num_key_value_heads", "linear_num_value_heads", "draft_num_key_value_heads")


#: RoutedStepTraffic's whole-number fields, and the head counts that must split over the pair's ranks.
_ROUTED_INTS = ("dense_read_bytes", "routed_expert_bytes", "experts", "top_k", "attention_layers",
                "num_key_value_heads", "head_dim", "kv_cache_dtype_bytes", "fp8_key_trailer_bytes", "index_budget",
                "index_ratio", "index_head_dim", "index_dtype_bytes", "linear_attention_layers", "linear_num_key_heads",
                "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_state_dtype_bytes",
                "linear_conv_kernel", "linear_conv_dtype_bytes", "state_transfers")
_ROUTED_HEADS = ("num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads")


def _traffic(raw, where: str) -> StepTraffic | RoutedStepTraffic:
    """A row's `traffic`, of the family it names (`TRAFFIC_FAMILIES`) — no family is no traffic, never a default."""
    if not isinstance(raw, dict) or raw.get("family") not in TRAFFIC_FAMILIES:
        family = raw.get("family") if isinstance(raw, dict) else None
        raise ValueError(f"{where}: traffic family must be one of {', '.join(TRAFFIC_FAMILIES)}, not {family!r}")
    routed = raw["family"] == "routed"
    ints, heads = (_ROUTED_INTS, _ROUTED_HEADS) if routed else (_TRAFFIC_INTS, _TRAFFIC_HEADS)
    _keys(raw, f"{where}: traffic", required={"family", *ints, *(("omits", "source") if routed else ())},
          optional=() if routed else {"source"})
    for key in ints:
        if not (_int(raw[key]) and raw[key] > 0):
            raise ValueError(f"{where}: traffic {key} must be a positive integer, not {raw[key]!r}")
    for key in heads:
        if raw[key] % CLUSTER_TP:
            raise ValueError(f"{where}: traffic {key} {raw[key]} does not split over {CLUSTER_TP} ranks")
    if routed and raw["top_k"] > raw["experts"]:
        raise ValueError(f"{where}: traffic top_k {raw['top_k']} exceeds its {raw['experts']} experts")
    source = _text(raw.get("source", ""), f"{where}: traffic source", empty=not routed)
    if routed:
        return RoutedStepTraffic(**{key: raw[key] for key in ints}, source=source,
                                 omits=_text(raw["omits"], f"{where}: traffic omits"))
    return StepTraffic(**{key: raw[key] for key in ints}, source=source)


def _limits(raw, where: str, *, floor: int) -> ModelLimits:
    _keys(raw, f"{where}: limits", required={"max_model_len", "max_num_seqs"})
    ctx, seqs = raw["max_model_len"], raw["max_num_seqs"]
    lo_ctx, hi_ctx = floor, MAX_MODEL_LEN_CAP[1]
    if not (_int(ctx) and lo_ctx <= ctx <= hi_ctx):
        raise ValueError(f"{where}: max_model_len must be an integer in {lo_ctx}..{hi_ctx}")
    lo, hi = MAX_NUM_SEQS_CAP
    if not (isinstance(seqs, list) and len(seqs) == 2 and all(_int(v) for v in seqs)
            and lo <= seqs[0] <= seqs[1] <= hi):
        raise ValueError(f"{where}: max_num_seqs must be [low, high] within {lo}..{hi}")
    return ModelLimits(max_model_len=ctx, max_num_seqs=(seqs[0], seqs[1]))


def _profile(raw, where: str, *, cluster: bool) -> LaunchProfile:
    if isinstance(raw, dict) and "image" in raw:
        raise ValueError(f"{where}: `image` moved out of the catalog (2026-10-30): a profile names its engine's "
                         f"`build`, and the image is the matrix's for this deployment's accelerator and platform "
                         f"(vllm_lb/library/images/<engine>.yml)")
    _keys(raw, where, required={"build"}, optional=_PROFILE_OPTIONAL)
    spec_cfg = raw.get("speculative_config")
    if spec_cfg is not None and not isinstance(spec_cfg, dict):
        raise ValueError(f"{where}: speculative_config must be an object or null")
    # The count is the one part of it the operator may change (0 turns it off), so the catalog's default
    # must say it: a drafter whose depth is vLLM's guess could not be shown, nor overridden honestly.
    drafts = (spec_cfg or {}).get("num_speculative_tokens")
    if spec_cfg is not None and not (_int(drafts) and drafts >= 1):
        raise ValueError(f"{where}: speculative_config must say its num_speculative_tokens, a whole number "
                         f"from 1 (null, not 0, is a model without speculative decoding)")
    util = raw.get("gpu_memory_utilization")
    if util is not None and not (_number(util) and 0 < util <= 1):
        raise ValueError(f"{where}: gpu_memory_utilization must be in (0, 1]")
    pin = raw.get("kv_cache_memory")
    if pin is not None and not (_int(pin) and pin > 0):
        raise ValueError(f"{where}: kv_cache_memory must be a positive number of bytes or null")
    ple = raw.get("ple_replicate")
    if ple is not None and not isinstance(ple, bool):
        raise ValueError(f"{where}: ple_replicate must be true, false or absent")
    prefix = raw.get("mtp_prefix")
    if prefix is not None:
        _text(prefix, f"{where}: mtp_prefix", empty=True)
    # A separate drafter checkpoint. tp-rank is the one that names the drafter to vLLM — from the view it
    # builds under mtp_prefix, or from this path — so a profile may say one or the other, never both, and
    # never a "model" of its own.
    draft = _text(raw.get("draft_path", ""), f"{where}: draft_path", empty=True)
    if draft:
        if spec_cfg is None:
            raise ValueError(f"{where}: draft_path names a drafter checkpoint but the profile has no "
                             f"speculative_config to point at it")
        if prefix:
            raise ValueError(f"{where}: draft_path and mtp_prefix are mutually exclusive — a drafter is "
                             f"either its own checkpoint or tensors inside the target's, not both")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", draft) or ".." in draft.split("/"):
            raise ValueError(f"{where}: draft_path must be a relative path of [A-Za-z0-9_.-] names under the "
                             f"models root, not {draft!r}")
    engine = _text(raw.get("engine", VLLM), f"{where}: engine")
    if engine not in CLUSTER_ENGINES:
        raise ValueError(f"{where}: engine must be one of {', '.join(CLUSTER_ENGINES)}, not {engine!r}")
    if engine != VLLM and not cluster:
        raise ValueError(f"{where}: engine {engine!r} is a cluster server; a classic profile is "
                         f"broker_parrot's own vLLM")
    tuned = (_tuned(raw["tuned"], where, cluster=cluster, speculative=spec_cfg is not None)
             if raw.get("tuned") is not None else None)
    engine_args = raw.get("engine_args", {})
    if not isinstance(engine_args, dict) or not all(isinstance(k, str) for k in engine_args):
        raise ValueError(f"{where}: engine_args must be an object keyed by setting name")
    if engine_args and engine == VLLM:
        raise ValueError(f"{where}: engine_args is for a non-vLLM engine; vLLM's settings are the "
                         f"profile's own fields")
    if engine == TENSORFOLD:
        _tensorfold_profile(raw, engine_args, spec_cfg, where)
    vllm_flags = {k: raw[k] for k in VLLM_FLAGS if k in raw}
    if vllm_flags and engine != VLLM:
        raise ValueError(f"{where}: {', '.join(vllm_flags)} {'is a' if len(vllm_flags) == 1 else 'are'} vLLM "
                         f"flag(s); a {engine} profile states its settings in engine_args")
    for k, v in vllm_flags.items():
        kind = VLLM_FLAGS[k][0]
        if kind is bool:
            ok = isinstance(v, bool)
        elif kind is int:
            ok = isinstance(v, int) and not isinstance(v, bool) and v > 0
        elif kind is dict:
            ok = isinstance(v, dict) and bool(v) and all(
                re.fullmatch(r"[a-z]+", m) and isinstance(n, int) and not isinstance(n, bool) and 0 < n < 10000
                for m, n in v.items())
        elif kind is object:
            # Flat, as tp-rank's FLAT_JSON takes it: a scalar or a list of scalars, never a brace inside a string.
            scalar = (lambda x: isinstance(x, (bool, int, float)) or isinstance(x, str) and not set(x) & set("{}"))
            ok = isinstance(v, dict) and bool(v) and all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", m) and (
                    scalar(n) or isinstance(n, list) and all(scalar(x) for x in n))
                for m, n in v.items())
        else:
            ok = isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", v) is not None
        if not ok:
            raise ValueError(f"{where}: {k} must be {'true or false' if kind is bool else 'a positive whole number' if kind is int else '{modality: count 1..9999}' if kind is dict else 'one flat JSON object (scalars or lists of scalars)' if kind is object else 'a name of [A-Za-z0-9_.-]'}, not {v!r}")
    refuses = raw.get("kv_cache_dtype_refuses", {})
    if not isinstance(refuses, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip()
                                                for k, v in refuses.items()):
        raise ValueError(f"{where}: kv_cache_dtype_refuses must map a dtype to why this model cannot run it")
    own = raw.get("kv_cache_dtype_backend")
    if own is not None:
        if (not isinstance(own, dict) or set(own) != {"name", "reads", "source"}
                or not (isinstance(own["name"], str) and re.fullmatch(r"[A-Za-z0-9_]+", own["name"]))
                or not (isinstance(own["reads"], list) and own["reads"] and all(isinstance(d, str) for d in own["reads"]))
                or not (isinstance(own["source"], str) and own["source"].strip())):
            raise ValueError(f"{where}: kv_cache_dtype_backend must be {{name, reads: [dtype, …], source}} — the model's "
                             f"own attention backend and the KV dtypes it reads")
        own = {"name": own["name"], "reads": tuple(own["reads"]), "source": own["source"]}
    if tuned is not None and tuned.max_num_batched_tokens is not None and "chunked_prefill_size" in engine_args:
        raise ValueError(f"{where}: the prefill budget is stated twice — engine_args.chunked_prefill_size and "
                         f"tuned.max_num_batched_tokens. State it once, in tuned: the measured one wins "
                         f"(`LaunchProfile.prefill_budget`), so the other is data that can only go stale")
    if spec_cfg is not None and "model" in spec_cfg:
        raise ValueError(f"{where}: speculative_config must not name a model — tp-rank splices the drafter "
                         f"it resolves (draft_path, or the view it builds from mtp_prefix)")
    default = raw.get("default", False)
    if not isinstance(default, bool):
        raise ValueError(f"{where}: default must be true or false")
    overhead = raw.get("overhead_gib", {})
    if cluster or overhead:
        if (not isinstance(overhead, dict) or set(overhead) != set(RANK_ROLES)
                or not all(_number(v) and v >= 0 for v in overhead.values())):
            raise ValueError(f"{where}: overhead_gib must give GiB for exactly the rank roles {', '.join(RANK_ROLES)}")
    return LaunchProfile(image=None, build=_text(raw["build"], f"{where}: build"), default=default,
                         tool_call_parser=_text(raw.get("tool_call_parser", ""), f"{where}: tool_call_parser",
                                                empty=True),
                         reasoning_parser=_text(raw.get("reasoning_parser", ""), f"{where}: reasoning_parser",
                                                empty=True),
                         speculative_config=dict(spec_cfg) if spec_cfg is not None else None,
                         gpu_memory_utilization=util, kv_cache_memory=pin, ple_replicate=ple, mtp_prefix=prefix,
                         draft_path=draft, engine=engine, engine_args=dict(engine_args),
                         tuned=tuned, overhead_gib=dict(overhead), vllm_flags=vllm_flags,
                         kv_cache_dtype_refuses=dict(refuses), kv_cache_dtype_backend=own)


def _tensorfold_profile(raw: dict, engine_args: dict, spec_cfg, where: str) -> None:
    """A TensorFold profile states what tp-rank must not guess (`TENSORFOLD_REQUIRED_ARGS`), and none of what the
    engine sizes or chooses itself (`ENGINE_REFUSES[TENSORFOLD]`): no KV pin, no memory fraction, no speculative config
    (its MTP chain is the engine's — engine_args.drafts), no prefill budget. A flat object of words and numbers, as
    tp-rank's `engine_arg` reads it."""
    missing = [k for k in TENSORFOLD_REQUIRED_ARGS if k not in engine_args]
    if missing:
        raise ValueError(f"{where}: a tensorfold profile's engine_args must state {', '.join(missing)}")
    if engine_args["cli"] not in TENSORFOLD_CLIS:
        raise ValueError(f"{where}: engine_args.cli must be one of {', '.join(TENSORFOLD_CLIS)}, "
                         f"not {engine_args['cli']!r}")
    if not isinstance(engine_args["drafts"], bool):
        raise ValueError(f"{where}: engine_args.drafts must be true or false")
    for key, value in engine_args.items():
        if not (isinstance(value, (bool, int)) or isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value)):
            raise ValueError(f"{where}: engine_args.{key} must be a number, true/false or a word of [A-Za-z0-9_.-] "
                             f"(tp-rank reads them so), not {value!r}")
    for key, why in (("kv_cache_memory", "kv_cache_memory"), ("gpu_memory_utilization", "kv_cache_memory"),
                     ("ple_replicate", "ple_replicate")):
        if raw.get(key) is not None:
            raise ValueError(f"{where}: {key}: {ENGINE_REFUSES[TENSORFOLD][why]}")
    if spec_cfg is not None:
        raise ValueError(f"{where}: speculative_config: {ENGINE_REFUSES[TENSORFOLD]['num_speculative_tokens']}")
    if "chunked_prefill_size" in engine_args or (raw.get("tuned") or {}).get("max_num_batched_tokens") is not None:
        raise ValueError(f"{where}: a prefill budget: {ENGINE_REFUSES[TENSORFOLD]['max_num_batched_tokens']}")


#: Each tuned knob's absolute bound — tp-rank's own, mirrored at the top of this module.
_TUNED_CAPS = {"max_num_seqs": MAX_NUM_SEQS_CAP, "max_model_len": MAX_MODEL_LEN_CAP,
               "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS_CAP,
               "num_speculative_tokens": NUM_SPECULATIVE_TOKENS_CAP}


def _tuned(raw, where: str, *, cluster: bool, speculative: bool) -> TunedSetup:
    """A profile's `tuned` block (`TunedSetup`): the measured-best start input, validated as strictly as the
    rest of the document. `source` is REQUIRED — a default nobody can trace is a guess — and at least one knob
    must be stated, else the block says nothing and should not be there."""
    _keys(raw, f"{where}: tuned", required={"source"}, optional=set(TUNED_KNOBS))
    if not cluster:
        raise ValueError(f"{where}: tuned states the CLUSTER's start input; a classic profile is "
                         f"broker_parrot's own vLLM and has none")
    stated = {knob: raw[knob] for knob in TUNED_KNOBS if knob in raw}
    if not stated:
        raise ValueError(f"{where}: tuned must state at least one of {', '.join(TUNED_KNOBS)}")
    for knob, value in stated.items():
        lo, hi = _TUNED_CAPS[knob]
        if not (_int(value) and lo <= value <= hi):
            raise ValueError(f"{where}: tuned {knob} must be a whole number in {lo}..{hi}, not {value!r}")
    if stated.get("num_speculative_tokens") and not speculative:
        raise ValueError(f"{where}: tuned num_speculative_tokens {stated['num_speculative_tokens']}, but the "
                         f"profile has no speculative_config — 0 (off) is the only depth such a model can be "
                         f"tuned to")
    return TunedSetup(**stated, source=_text(raw["source"], f"{where}: tuned source"))


def _memory(raw, where: str) -> MemoryFootprint:
    _keys(raw, f"{where}: memory", required=_MEMORY_REQUIRED,
          optional={"calibration", "activation_gib_per_1k_tokens", "calibrated_batched_tokens",
                    "concurrency_gib_per_seq", "calibrated_max_num_seqs", "kv_blocks", "single_card_kv_blocks",
                    "single_card_overhead"})
    for key in ("weights_gib", "ple_gib", "mtp_gib", "seq_state_gib"):
        if not (_number(raw[key]) and raw[key] >= 0):
            raise ValueError(f"{where}: memory {key} must be a number of GiB, not {raw[key]!r}")
    if not (_int(raw["kv_bytes_per_token"]) and raw["kv_bytes_per_token"] > 0):
        raise ValueError(f"{where}: memory kv_bytes_per_token must be a positive integer")
    dtype = _layout(raw["kv_cache_dtype"], f"{where}: memory kv_cache_dtype")
    slope = raw.get("activation_gib_per_1k_tokens", 0.0)
    if not (_number(slope) and slope >= 0):
        raise ValueError(f"{where}: memory activation_gib_per_1k_tokens must be a number of GiB, not {slope!r}")
    anchor = raw.get("calibrated_batched_tokens", 0)
    if not (_int(anchor) and anchor >= 0):
        raise ValueError(f"{where}: memory calibrated_batched_tokens must be a whole number of tokens, "
                         f"not {anchor!r}")
    if slope and not anchor:
        raise ValueError(f"{where}: memory activation_gib_per_1k_tokens needs the "
                         f"calibrated_batched_tokens it is measured from")
    seat = raw.get("concurrency_gib_per_seq", 0.0)
    if not (_number(seat) and seat >= 0):
        raise ValueError(f"{where}: memory concurrency_gib_per_seq must be a number of GiB, not {seat!r}")
    seats = raw.get("calibrated_max_num_seqs", 0)
    if not (_int(seats) and seats >= 0):
        raise ValueError(f"{where}: memory calibrated_max_num_seqs must be a whole number of sequences, "
                         f"not {seats!r}")
    if seat and not seats:
        raise ValueError(f"{where}: memory concurrency_gib_per_seq needs the calibrated_max_num_seqs it "
                         f"is measured from")
    return MemoryFootprint(weights_gib=float(raw["weights_gib"]), ple_gib=float(raw["ple_gib"]),
                           mtp_gib=float(raw["mtp_gib"]), kv_bytes_per_token=raw["kv_bytes_per_token"],
                           seq_state_gib=float(raw["seq_state_gib"]),
                           calibration=_text(raw.get("calibration", ""), f"{where}: calibration", empty=True),
                           activation_gib_per_1k_tokens=float(slope), calibrated_batched_tokens=int(anchor),
                           concurrency_gib_per_seq=float(seat), calibrated_max_num_seqs=int(seats),
                           kv_blocks=_kv_blocks(raw.get("kv_blocks"), where, "kv_blocks"),
                           single_card_kv_blocks=_kv_blocks(raw.get("single_card_kv_blocks"), where,
                                                            "single_card_kv_blocks"),
                           single_card_overhead=_card_overhead(raw.get("single_card_overhead"), where),
                           kv_cache_dtype=dtype)


def _layout(value, where: str) -> str:
    """A dtype a KV figure was measured at: one with a layout (vllm_lb/kv_cache_dtypes.json), never `auto`."""
    if value not in kv_dtypes().layouts:
        raise ValueError(f"{where} must be the dtype the figure was measured at, one of "
                         f"{', '.join(kv_dtypes().layouts)} (what `auto` resolved to), not {value!r}")
    return value


def _card_overhead(raw, where: str) -> CardOverhead | None:
    if raw is None:
        return None
    _keys(raw, f"{where}: memory single_card_overhead", required={"gib", "source"}, optional={"chunk"})
    if not (_number(raw["gib"]) and raw["gib"] >= 0):
        raise ValueError(f"{where}: memory single_card_overhead gib must be a number of GiB, not {raw['gib']!r}")
    chunk = raw.get("chunk")
    if chunk is not None:
        # The prefill chunk the figure was measured at, and its source: a number nobody can trace is a guess.
        _keys(chunk, f"{where}: memory single_card_overhead chunk", required={"max_num_batched_tokens", "source"})
        if not (_int(chunk["max_num_batched_tokens"]) and chunk["max_num_batched_tokens"] >= 1):
            raise ValueError(f"{where}: memory single_card_overhead chunk max_num_batched_tokens must be a whole number "
                             f"of tokens from 1, not {chunk['max_num_batched_tokens']!r}")
    return CardOverhead(gib=float(raw["gib"]),
                        source=_text(raw["source"], f"{where}: memory single_card_overhead source"),
                        chunk_tokens=chunk["max_num_batched_tokens"] if chunk is not None else None,
                        chunk_source=(_text(chunk["source"], f"{where}: memory single_card_overhead chunk source")
                                      if chunk is not None else ""))


def _kv_blocks(kb, where: str, name: str) -> KvBlocks | None:
    """One engine block rule (`memory.<name>`), or None where the row records none."""
    if kb is None:
        return None
    _keys(kb, f"{where}: memory {name}", required={"block_tokens", "block_bytes", "fixed_blocks", "kv_cache_dtype"},
          optional={"source"})
    for key in ("block_tokens", "block_bytes"):
        if not (_int(kb[key]) and kb[key] > 0):
            raise ValueError(f"{where}: memory {name} {key} must be a positive integer, not {kb[key]!r}")
    if not (_int(kb["fixed_blocks"]) and kb["fixed_blocks"] >= 0):
        raise ValueError(f"{where}: memory {name} fixed_blocks must be a whole number, not {kb['fixed_blocks']!r}")
    return KvBlocks(block_tokens=kb["block_tokens"], block_bytes=kb["block_bytes"], fixed_blocks=kb["fixed_blocks"],
                    source=_text(kb.get("source", ""), f"{where}: memory {name} source", empty=True),
                    kv_cache_dtype=_layout(kb["kv_cache_dtype"], f"{where}: memory {name} kv_cache_dtype"))
