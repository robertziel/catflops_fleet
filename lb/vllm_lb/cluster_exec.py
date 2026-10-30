"""How a rank is started, stopped and observed on a GB10 box: one SSH call per verb, nothing installed.

The planner (`vllm_lb.cluster`) decides WHEN a rank runs. This module carries a verb to a GB10 box and
brings back what it said. Nothing lives on the box: every call sends this LB's own copy of
`lb/cluster/tp-rank` on stdin and runs it there as

    env CLUSTER_…=… CLUSTER_HOST_IP=<the box's interconnect> bash -s -- <verb> <args…>

so the script that runs is always the one the LB was built with, and there is nothing on a GB10 box to
install, update or drift. The rules this side keeps:

* **Never raise.** ssh missing, a refused key, a timeout, garbage on stdout, an unreadable script:
  each one becomes `None`, `(False, why)` or `""`, with the reason in `last_error[box]`. The planner
  has a state for "could not observe"; an exception would take the whole tick down, and with it the
  classic and fallback planners that run in the same tick.
* **Quote what the remote shell parses.** With a plain key, sshd hands the command line to the box's
  login shell. Every value is `shlex.quote`d; a variable NAME cannot be quoted, so an env key that is
  not one is refused before dialing.
* **Refuse before dialing.** `up`'s ranges mirror tp-rank's, which refuses on its own anyway; a bad
  plan should fail here, with a reason, not as an SSH round trip. The build is ONE function
  (`build_launch`, raising `LaunchRefused`): `up` answers it as `(False, "up: …")`, and the cluster loop
  asks `launch_problem` about the stored selection before it plans a launch — a refusal of the SETTINGS
  is the same on every retry, and must fail the start, never be re-sent every tick (generation 187).
* **Pin the connection, and reuse it.** Only the given key (`IdentitiesOnly`), only a host key already
  in the given known_hosts (`StrictHostKeyChecking=yes`), never a prompt (`BatchMode`), and a link
  that dies mid-call noticed within ~15 s (`ServerAlive*`). `ControlMaster=auto` +
  `ControlPersist=120` carry every call to a box over one connection that outlives the call by two
  minutes, so a status poll per tick is a new channel, not a new sshd session — a previous design ran
  box-a's sshd into MaxStartups. The socket is `<control_dir>/cm-%C` (`%C`: a hash of user, host and
  port); keep that path short, since a unix socket path cannot exceed ~104 bytes.

This is a plain login key, not a forced command: whoever holds it, or can open the control socket,
holds that account on the GB10 boxes. tp-rank's validation guards against a bad PLAN, not a hostile
caller — the key and the socket belong to the LB container alone.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
import re
import shlex
import subprocess
import time
from pathlib import Path

from vllm_lb.catalog import CLUSTER_MODE, SGLANG, TENSORFOLD, VLLM, Availability, ModelSpec
from vllm_lb.cluster import LONG_PREFILL_TOKEN_THRESHOLD_RANGE, NUM_SPECULATIVE_TOKENS_RANGE, ClusterHost
from vllm_lb.kv_dtype import library as kv_dtypes

#: tp-rank's own limits, mirrored so a bad plan fails before it costs a round trip.
RANK_RANGE = (0, 1)
GEN_RANGE = (1, 999_999)
MAX_NUM_SEQS_RANGE = (1, 48)
MAX_MODEL_LEN_RANGE = (1024, 262_144)
MAX_NUM_BATCHED_TOKENS_RANGE = (2048, 65_536)
LOG_LINES_RANGE = (1, 2000)
#: What tp-rank launches with when it is told nothing: the drafter depth of its own MTP_DRAFTER, used only when no
#: CLUSTER_SPECULATIVE_CONFIG is sent at all — i.e. a launch with no ModelChoice. Mirrored so the LB can say what a
#: rank was launched WITH (`resolved_launch`). The prefill budget has NO such default anywhere (2026-10-29): the
#: fleet's travels with every call (`cluster_run.tp_rank_env`), and tp-rank refuses an `up` without one.
TP_RANK_NUM_SPECULATIVE_TOKENS = 4
#: How much of a failure `last_error` keeps: a docker daemon's reason fits, a whole log does not.
ERROR_MAX_CHARS = 500
#: A name `env` sets as given and the remote shell cannot misread.
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


#: What `up` writes the LMCache config to on the box, and mounts into the engine container. Only ever
#: touched when the operator turned the disk tier on; off, no directory is made and no flag is passed.
OFFLOAD_DIR = "~/.cache/vllm-cluster-kv"


def offload_yaml(gib: int) -> str:
    """The LMCache engine config for a DISK-ONLY tier, as one YAML document.

    `local_cpu: false` is the load-bearing line. On a GB10 the host's RAM and the GPU's are ONE unified
    LPDDR5X pool, so a CPU tier does not add a tier — it takes the pool's own memory to cache the pool.
    That is also why vLLM's `--kv-offloading-backend lmcache` is not used: it wires LMCacheMPConnector to
    a standalone `lmcache_server` whose storage factory accepts only "cpu" (LMCache 0.5.4 has the disk
    branch commented out as a TODO), which on this hardware is strictly worse than nothing.

    NOT PROVEN ON THIS FLEET. The config parses and the NVMe is writable from inside the container, both
    checked 2026-10-01; the engine has never been started with it. `LocalDiskBackend` takes a
    `LocalCPUBackend` as a constructor argument, and what that costs with `max_local_cpu_size: 0` is
    unmeasured — which is why the knob defaults OFF and the page says so."""
    return ("# Written by lb/cluster/tp-rank for CLUSTER_KV_OFFLOAD_GIB. Disk only — see cluster_exec.py.\n"
            "local_cpu: false\n"
            "max_local_cpu_size: 0\n"
            'local_disk: "file:///kvcache/"\n'
            f"max_local_disk_size: {int(gib)}\n"
            "chunk_size: 256\n")


def _taken(spec: ModelSpec | None, knob: str, value):
    """`value`, unless the ModelChoice's engine does not take `knob` (`LaunchProfile.refuses`): then None — the
    engine's own, which is what it runs whatever was stored. The switch refuses such a value before it is ever
    stored for a Start; this is what keeps one from an older file honest about the launch it rides."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    return None if profile is not None and knob in profile.refuses else value


def _sglang_kv_args(spec: ModelSpec, pin: int, args: dict) -> None:
    """Turn the cluster's byte pin into the two SGLang inputs that actually size its pool.

    `--max-total-tokens` is the engine's hard, token-addressable cache limit.  The companion fraction
    pays for that allocation while retaining the model-specific CUDA/Mamba reserve measured in the catalog.
    Keep the conversion here, beside `launch_env`: the switch, memory gate, launch record and panel all
    already speak the byte pin, so teaching each of them SGLang's token arithmetic would drift.
    """
    profile = spec.profile(CLUSTER_MODE)
    memory = spec.memory
    if profile is None or memory is None or memory.kv_bytes_per_token <= 0:
        raise ValueError(f"{spec.id} has no measured SGLang KV bytes per token")
    page = args.get("page_size", 1)
    if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
        raise ValueError(f"{spec.id}: SGLang page_size must be a positive whole number, not {page!r}")
    tokens = (int(pin) // memory.kv_bytes_per_token) // page * page
    if tokens <= 0:
        raise ValueError(f"{spec.id}: KV pin {pin:,} B cannot hold one SGLang page of {page} tokens")
    args.pop("kv_pool_calibration", None)       # LB metadata, never an engine argument.
    args["max_total_tokens"] = tokens
    args["mem_fraction_static"] = profile.sglang_mem_fraction(int(pin))


def launch_env(spec: ModelSpec, *, num_speculative_tokens: int | None = None,
               ple_replicate: bool | None = None, kv_cache_memory: int | None = None,
               kv_offload_gib: int | None = None, max_num_batched_tokens: int | None = None,
               kv_cache_dtype: str | None = None) -> dict[str, str]:
    """A ModelChoice's cluster LaunchProfile as tp-rank's settings: the ONE place the one becomes the
    other (DRY). Every key emitted is a setting `lb/cluster/tp-rank` declares, and a value left EMPTY is
    how a profile says the model has none of it — tp-rank then passes no flag at all.

    `num_speculative_tokens` is the OPERATOR's drafter depth (`cluster.Desired`), the one thing that may
    override a profile's speculation. None sends the profile's own config byte for byte; 0 sends none at all
    (speculation off); N replaces only its `num_speculative_tokens`, so the method and every other key stay
    the catalog's. A profile with no speculative decoding has no drafter to set, and refuses N > 0.

    `max_num_batched_tokens` is the operator's prefill budget. vLLM takes it as CLUSTER_MAX_NUM_BATCHED_TOKENS (`up`
    sends that, over the fleet's in the env); SGLang as its `chunked_prefill_size`, which tp-rank reads from engine_args BEFORE the box's budget —
    so it is written there, over the catalog's, or the catalog's 8192 runs whatever the operator chose (generation
    135, 2026-10-03: 2048 asked, `#new-token: 8192` served). A knob the engine does not take is dropped (`_taken`).
    None is the model's own budget — its measured-best (`LaunchProfile.prefill_budget`, a `TunedSetup`), else the
    profile's engine_args — resolved here as the pool and the PLE placement already are, so a default start
    launches at the figure the catalog measured rather than at the box's one number for every model.

    `kv_cache_dtype` is the run's KV element type (`cluster.Desired.kv_cache_dtype`); None is the profile's own, and
    sends exactly what was sent before the knob. A value goes as CLUSTER_KV_CACHE_DTYPE for either engine (tp-rank
    takes it over the profile's), the attention backends the image reads that dtype's pool with beside it (SGLang's
    NVFP4: flashinfer prefill + trtllm_mha decode; fp4_mx_block16: triton — vllm_lb/kv_cache_dtypes.json), and every KV figure is priced at it (`ModelSpec.at_kv_cache_dtype`) — the
    SGLang token pool the byte pin becomes most of all."""
    profile = spec.profile(CLUSTER_MODE)
    # THE DTYPE THAT RUNS — the run's, else the profile's own — and the image's record of it: which attention backends
    # read its pool (kv_cache_dtypes.json `attention_backends`, the ONE statement of it for every engine).
    runs_kv = kv_cache_dtype or (profile.kv_cache_dtype if profile is not None else None)
    took = kv_dtypes().accepted(profile.image, runs_kv) if profile is not None and runs_kv else None
    spec = spec.at_kv_cache_dtype(kv_cache_dtype)
    if profile is None:
        raise ValueError(f"{spec.id} has no {CLUSTER_MODE} LaunchProfile to launch from")
    if max_num_batched_tokens is None:
        max_num_batched_tokens = profile.prefill_budget
    kv_cache_memory = _taken(spec, "kv_cache_memory", kv_cache_memory)
    ple_replicate = _taken(spec, "ple_replicate", ple_replicate)
    kv_offload_gib = _taken(spec, "kv_offload_gib", kv_offload_gib)
    env = {"CLUSTER_IMAGE": profile.image,
           # The catalog's roots, joined with the model's own folder; `~` is the BOX's home, which only
           # the box can expand.
           "CLUSTER_MODELS_DIRS": ":".join(spec.cluster_dirs),
           "CLUSTER_MODEL_NAME": spec.name,
           "CLUSTER_SERVED_MODEL_NAME": spec.served_name}
    if profile.gpu_memory_utilization is not None:
        env["CLUSTER_GPU_MEMORY_UTILIZATION"] = f"{profile.gpu_memory_utilization:.2f}"
    pin = profile.kv_cache_memory if kv_cache_memory is None else kv_cache_memory
    if pin is not None:
        env["CLUSTER_KV_CACHE_MEMORY"] = str(pin)
    env["CLUSTER_TOOL_CALL_PARSER"] = profile.tool_call_parser
    env["CLUSTER_REASONING_PARSER"] = profile.reasoning_parser
    speculative = profile.speculative_config
    # No depth given is the profile's DEFAULT depth (its tuned one first), as `resolved_launch` records it — never the
    # config's own: generation 249 recorded 0 and launched DeepSeek's drafter at 5, which cannot load.
    if num_speculative_tokens is None and speculative:
        num_speculative_tokens = profile.num_speculative_tokens
    if num_speculative_tokens == 0:
        speculative = None
    elif num_speculative_tokens is not None:
        if not speculative:
            raise ValueError(f"{spec.id} has no speculative decoding to set {num_speculative_tokens} tokens for")
        speculative = {**speculative, "num_speculative_tokens": num_speculative_tokens}
    env["CLUSTER_SPECULATIVE_CONFIG"] = (json.dumps(speculative, separators=(",", ":"))
                                         if speculative else "")
    if profile.mtp_prefix is not None:
        env["CLUSTER_MTP_PREFIX"] = profile.mtp_prefix
    # The drafter's own checkpoint, when the model's drafter is one. Sent only with speculation ON: at
    # num_speculative_tokens 0 there is no drafter to name, and tp-rank passes no flag at all.
    if profile.draft_path and speculative:
        env["CLUSTER_DRAFT_NAME"] = spec.draft_name
    # Which server tp-rank launches, and the settings only that server has. `vllm` is left unsent: it is
    # tp-rank's own default, so a vLLM profile's launch is byte for byte the one sent before engines
    # existed. The depth the operator chose reaches a non-vLLM engine through engine_args, since its flag
    # is not vLLM's — 0 means the drafter is not passed at all.
    if profile.engine == TENSORFOLD:
        # TENSORFOLD'S OWN SETTINGS, as tp-rank `tensorfold_cmd` reads them: the profile's engine_args, true/false as
        # 1/0 (its `engine_arg` reads words and numbers). Nothing else of the operator's reaches it here: the seats and
        # the context ride on `up`'s argv, a run KV dtype as CLUSTER_KV_CACHE_DTYPE below, and every knob it does not
        # take was refused before this (`build_launch`, `DesiredStore.set`).
        env["CLUSTER_ENGINE"] = profile.engine
        env["CLUSTER_ENGINE_ARGS"] = json.dumps(
            {k: (int(v) if isinstance(v, bool) else v) for k, v in profile.engine_args.items()}, separators=(",", ":"))
    elif profile.engine != VLLM:
        env["CLUSTER_ENGINE"] = profile.engine
        args = dict(profile.engine_args)
        if profile.engine == SGLANG:
            if pin is None:
                raise ValueError(f"{spec.id}: SGLang needs a measured KV pin")
            _sglang_kv_args(spec, int(pin), args)
        if num_speculative_tokens is not None:
            args["draft_tokens"] = num_speculative_tokens
        if took is not None and took.attention_backends:
            args["prefill_attention_backend"] = took.attention_backends["prefill"]
            args["decode_attention_backend"] = took.attention_backends["decode"]
        if took is not None and took.draft_kv_cache_dtype:
            args["speculative_draft_kv_cache_dtype"] = took.draft_kv_cache_dtype
        if max_num_batched_tokens is not None and profile.engine == SGLANG:
            args["chunked_prefill_size"] = int(max_num_batched_tokens)
        if args:
            env["CLUSTER_ENGINE_ARGS"] = json.dumps(args, separators=(",", ":"))
    # The profile's own vLLM engine flags (catalog.VLLM_FLAGS), one setting each; none named, none sent.
    for key, value in profile.vllm_flags.items():
        env[f"CLUSTER_{key.upper()}"] = (("1" if value else "0") if isinstance(value, bool)
                                         else json.dumps(value, separators=(",", ":")) if isinstance(value, dict)
                                         else str(value))
    if kv_cache_dtype is not None:
        env["CLUSTER_KV_CACHE_DTYPE"] = kv_cache_dtype
    # …in vLLM's ONE flag (SGLang's prefill/decode pair is set above): int4_per_token_head is read by Triton only
    # (TRITON_ATTN), so the backend travels with the dtype. A profile naming another backend is a contradiction, refused.
    # A model that brings its OWN backend (`kv_cache_dtype_backend`, Flash-Next's QSA) is never sent one: vLLM picks it.
    if profile.engine == VLLM and took is not None and took.attention_backends and not profile.kv_cache_dtype_backend:
        prefill, decode = took.attention_backends["prefill"], took.attention_backends["decode"]
        if prefill != decode:
            raise ValueError(f"{spec.id}: KV {runs_kv} is read by {prefill} prefill and {decode} decode, and vLLM takes "
                             f"one --attention-backend")
        backend = prefill.upper()
        if env.get("CLUSTER_ATTENTION_BACKEND", backend) != backend:
            raise ValueError(f"{spec.id}: KV {runs_kv} is read by {backend} only, and the profile names attention "
                             f"backend {env['CLUSTER_ATTENTION_BACKEND']}")
        env["CLUSTER_ATTENTION_BACKEND"] = backend
    ple = profile.ple_replicate if ple_replicate is None else ple_replicate
    if ple is not None:
        env["CLUSTER_PLE_REPLICATE"] = "1" if ple else "0"
    # OFF adds nothing at all: no directory, no YAML, no flag — the launch is the one sent before the
    # disk tier existed, which is what keeps a default start byte-identical.
    if kv_offload_gib:
        env["CLUSTER_KV_OFFLOAD_GIB"] = str(int(kv_offload_gib))
        env["CLUSTER_KV_OFFLOAD_YAML"] = offload_yaml(int(kv_offload_gib))
    return env


def resolved_launch(spec: ModelSpec | None, box_env, *, max_num_seqs: int, max_model_len: int,
                    max_num_batched_tokens: int | None = None,
                    num_speculative_tokens: int | None = None, ple_replicate: bool | None = None,
                    kv_cache_memory: int | None = None, kv_offload_gib: int | None = None,
                    long_prefill_token_threshold: int | None = None, uncensored: bool | None = None,
                    kv_cache_dtype: str | None = None) -> dict:
    """What `up` with these arguments LAUNCHES, every default resolved the way the launch resolves it — the
    operator's stored value, else the ModelChoice's own (its LaunchProfile's budget and drafter), else the box's
    (`box_env`: the settings every call carries — the fleet's CLUSTER_MAX_NUM_BATCHED_TOKENS, `cluster_run.tp_rank_env`)
    or tp-rank's own depth. One function, beside the one that builds the launch, so the two cannot drift. A budget of
    None: no number travels (an engine that chunks for itself), or the env carries one tp-rank would refuse.

    A profile's OWN prefill budget (SGLang's `engine_args.chunked_prefill_size`, `LaunchProfile.prefill_budget`)
    comes before the box's, as tp-rank takes it; a knob the engine does not take resolves to what it runs."""
    kv_cache_memory = _taken(spec, "kv_cache_memory", kv_cache_memory)
    ple_replicate = _taken(spec, "ple_replicate", ple_replicate)
    kv_offload_gib = _taken(spec, "kv_offload_gib", kv_offload_gib)
    long_prefill_token_threshold = _taken(spec, "long_prefill_token_threshold", long_prefill_token_threshold)
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    own_budget = profile.prefill_budget if profile is not None else None
    out = {"max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": (max_num_batched_tokens if max_num_batched_tokens is not None
                                       else own_budget if own_budget is not None
                                       else _box_batched_tokens((box_env or {}).get("CLUSTER_MAX_NUM_BATCHED_TOKENS"))),
            "num_speculative_tokens": (num_speculative_tokens if num_speculative_tokens is not None
                                       else _profile_drafts(spec)),
            "ple_replicate": (ple_replicate if ple_replicate is not None else _profile_ple(spec)),
            "kv_cache_memory": (kv_cache_memory if kv_cache_memory is not None else _profile_pin(spec)),
            "kv_offload_gib": int(kv_offload_gib or 0),
            "max_model_len": max_model_len,
            # None and 0 are both OFF — vLLM's own default, the launch with no flag.
            "long_prefill_token_threshold": int(long_prefill_token_threshold or 0),
            # Which checkpoint: the switch resolved it before the launch (`Desired.uncensored`); None launched
            # the original, exactly as `up` reads it.
            "uncensored": bool(uncensored),
            # The KV element type: the run's, else the profile's own — `auto` where a vLLM profile names none.
            "kv_cache_dtype": (kv_cache_dtype if kv_cache_dtype is not None
                               else (profile.kv_cache_dtype if profile is not None else None) or "auto")}
    # An engine that chooses its prefill chunk and its drafter depth ITSELF (TensorFold) ran with no number the LB sent:
    # the record says so (None), never the box's budget or a depth of 0 it did not run with.
    for knob in ("max_num_batched_tokens", "num_speculative_tokens"):
        if profile is not None and knob in profile.refuses:
            out[knob] = None
    return out


def _box_batched_tokens(raw) -> int | None:
    """The cluster's prefill budget as tp-rank reads it from the env every call carries (the fleet's): a plain number
    in `int_in 2048 65536`'s range, else None — unset and empty are NOTHING, not a default (tp-rank refuses an `up`
    without one by name)."""
    if raw is None or str(raw) == "":
        return None
    text = str(raw)
    lo, hi = MAX_NUM_BATCHED_TOKENS_RANGE
    if re.fullmatch(r"0|[1-9][0-9]{0,6}", text) and lo <= int(text) <= hi:
        return int(text)
    return None


def _profile_drafts(spec: ModelSpec | None) -> int:
    """The drafter depth a launch with no operator value gets: its LaunchProfile's, or tp-rank's own drafter
    when there is no ModelChoice to take a profile from."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    return profile.num_speculative_tokens if profile is not None else TP_RANK_NUM_SPECULATIVE_TOKENS


def _profile_ple(spec: ModelSpec | None) -> bool | None:
    """Where the PLE table lives when the operator set nothing: the LaunchProfile's placement, or None when
    there is no ModelChoice — tp-rank then applies its own default (replicated)."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    return profile.ple_replicate if profile is not None else None


def _profile_pin(spec: ModelSpec | None) -> int | None:
    """The KV pool a launch with no operator value gets: its LaunchProfile's pin, else None — tp-rank's own
    CLUSTER_KV_CACHE_MEMORY default then stands."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    return profile.kv_cache_memory if profile is not None else None


class ClusterExecutor(ABC):
    """THE EXECUTOR INTERFACE (vllm_lb/executors.py): the verbs the cluster loop runs a cluster by — each answering
    for ONE host (`ClusterHost`) and never raising for a box or a link (`last_error` keeps why). One class per way a
    cluster is run; `executors.executor_class` picks it by the fleet's control. The topology it runs is FACTS it is
    given (`executors.Topology.facts`), never its own assumption."""

    #: box -> the text of its last failure; "" once a call to that box succeeds again.
    last_error: dict[str, str]

    @abstractmethod
    def status(self, host: ClusterHost) -> dict | None: ...

    @abstractmethod
    def availability(self, host: ClusterHost, specs, *, observed_at: float | None = None) -> dict | None: ...

    @abstractmethod
    def up(self, host: ClusterHost, *, rank: int, gen: int, max_num_seqs: int, max_model_len: int,
           **settings) -> tuple[bool, str]: ...

    @abstractmethod
    def halt(self, host: ClusterHost) -> tuple[bool, str]: ...

    @abstractmethod
    def down(self, host: ClusterHost) -> tuple[bool, str]: ...


class SshExecutor(ClusterExecutor):
    """tp-rank's verbs over SSH. Holds nothing but `last_error`, so one instance serves every tick."""

    def __init__(self, *, key_path: str, known_hosts: str, script_path: str,
                 env: dict[str, str] | None = None, ssh_bin: str = "ssh",
                 status_timeout_s: float = 15.0, action_timeout_s: float = 90.0,
                 control_dir: str = "/tmp") -> None:
        self.key_path = key_path
        self.known_hosts = known_hosts
        #: Read on every call, so a redeployed script is the one the next call sends.
        self.script_path = script_path
        #: What every box gets (CLUSTER_PORT, CLUSTER_MASTER_ADDR, …); each call adds its box's
        #: CLUSTER_HOST_IP.
        self.env: dict[str, str] = dict(env or {})
        self.ssh_bin = ssh_bin
        self.status_timeout_s = status_timeout_s
        self.action_timeout_s = action_timeout_s
        self.control_dir = control_dir
        #: box -> the text of its last failure; "" once a call to that box succeeds again.
        self.last_error: dict[str, str] = {}

    # ── the verbs ──────────────────────────────────────────────────────────

    def status(self, host: ClusterHost) -> dict | None:
        """tp-rank's status reply, parsed — or None when the box could not be observed, for ANY
        reason. None is not "no rank": the planner reads it as unknown and blocks on it."""
        proc = self._run(host, ["status"], self.status_timeout_s)
        if proc is None:
            return None
        data = _last_json(proc.stdout)
        if proc.returncode != 0:
            self._fail(host, "status", _why(proc, data))
            return None
        if not _is_status(data):
            self._fail(host, "status", f"not a status reply: {proc.stdout.strip()!r}")
            return None
        self.last_error[host.box] = ""
        return data

    def up(self, host: ClusterHost, *, rank: int, gen: int, max_num_seqs: int, max_model_len: int,
           spec: ModelSpec | None = None, max_num_batched_tokens: int | None = None,
           num_speculative_tokens: int | None = None, ple_replicate: bool | None = None,
           kv_cache_memory: int | None = None, kv_offload_gib: int | None = None,
           long_prefill_token_threshold: int | None = None, uncensored: bool | None = None,
           kv_cache_dtype: str | None = None) -> tuple[bool, str]:
        """Start this box's rank of generation `gen`; `(True, "already-running")` if it is up. It
        returns once docker has started the container — the model load is watched via `status`.

        `spec` is the ModelChoice: its id goes on argv, so the rank carries it as a label and a rank
        launched for another choice is replaced; its cluster LaunchProfile goes as this call's settings.
        `uncensored` picks WHICH CHECKPOINT of it the rank loads (`ModelSpec.launched_from`): the twin's dir
        name goes as CLUSTER_MODEL_NAME and tp-rank's `find_checkpoint` fails the launch naming it when no
        root holds it — the original is never loaded in its place. None launched the original.

        The operator's engine knobs ride as this call's settings too: a prefill budget replaces the fleet's
        CLUSTER_MAX_NUM_BATCHED_TOKENS (in every call's env) for this launch, and a drafter depth the profile's
        (`launch_env`). A budget left at None is the MODEL's own — its measured-best (`LaunchProfile.prefill_budget`)
        — and rides exactly the same way; a model with none measured adds nothing, so the fleet's stands. The long-prefill threshold rides as CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD only when
        it is ON: None and 0 are vLLM's own default, and add nothing either.

        tp-rank's REFUSAL of the plan (exit 2) is raised as `LaunchRefused`, never answered: it is not a box failing."""
        try:
            words, env = build_launch(spec, rank=rank, gen=gen, max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                                      max_num_batched_tokens=max_num_batched_tokens,
                                      num_speculative_tokens=num_speculative_tokens, ple_replicate=ple_replicate,
                                      kv_cache_memory=kv_cache_memory, kv_offload_gib=kv_offload_gib,
                                      long_prefill_token_threshold=long_prefill_token_threshold, uncensored=uncensored,
                                      kv_cache_dtype=kv_cache_dtype)
        except LaunchRefused as exc:
            self._fail(host, "up", str(exc))
            return False, self.last_error[host.box]
        return self._action(host, words, env=env)

    def halt(self, host: ClusterHost) -> tuple[bool, str]:
        """Stop this box's rank and KEEP it: the memory is freed, the exit code and the log stay for the
        post-mortem until `down`. Nothing running is a success."""
        return self._action(host, ["halt"])

    def down(self, host: ClusterHost) -> tuple[bool, str]:
        """Remove this box's rank, whatever its generation. Nothing to remove is a success."""
        return self._action(host, ["down"])

    def heal(self, host: ClusterHost) -> tuple[bool, str]:
        """Bring this box's broker worker back: remount every fstab NFS mount that is not mounted, then
        restart the worker's boot backstop unless it is already waiting. Idempotent, harmless on a
        healthy box, and never touches a rank. `(False, why)` names what is still missing."""
        return self._action(host, ["heal"])

    def logs(self, host: ClusterHost, n: int = 160) -> str:
        """The rank's last `n` log lines, or "" (no rank, or they could not be read). Bounded by the
        STATUS timeout: it is an observation, and must not hold its caller for an action's budget."""
        if not _is_int(n, *LOG_LINES_RANGE):
            lo, hi = LOG_LINES_RANGE
            self._fail(host, "logs", f"n must be an integer in {lo}..{hi}, not {n!r}")
            return ""
        proc = self._run(host, ["logs", str(n)], self.status_timeout_s)
        if proc is None:
            return ""
        if proc.returncode != 0:
            self._fail(host, "logs", _why(proc, _last_json(proc.stdout)))
            return ""
        self.last_error[host.box] = ""
        return proc.stdout

    def stacks(self, host: ClusterHost) -> str:
        """What every engine process of this box's rank is doing NOW — py-spy's Python and native stacks, every
        thread's state and wchan, every thread's native stack where the image has eu-stack or gdb — and what the
        box saw meanwhile (its GPU, /proc/meminfo, its kernel log), as tp-rank `stacks` prints them; or "" (no
        rank running, or the box could not be asked). The post-mortem of a HUNG start, taken before it is torn
        down (`cluster_loop`). tp-rank bounds it at ~30 s; the ACTION timeout is the outer bound."""
        proc = self._run(host, ["stacks"], self.action_timeout_s)
        if proc is None:
            return ""
        if proc.returncode != 0:
            self._fail(host, "stacks", _why(proc, _last_json(proc.stdout)))
            return ""
        self.last_error[host.box] = ""
        return proc.stdout

    def availability(self, host: ClusterHost, specs, *,
                     observed_at: float | None = None) -> dict | None:
        """What this box HAS: one `Availability` per ModelSpec, asked for in ONE call — or None when the
        box could not be asked, or did not answer for every one of them. Read-only, so the planner may
        ask before it reserves anything."""
        # A profile that resolved no image on this deployment's cards (resolve.py) has nothing to look for: its Start
        # is refused naming why (`ModelCatalog.choice`), so the box is never asked about it.
        specs = [s for s in specs if s is not None and s.profile(CLUSTER_MODE) is not None
                 and s.profile(CLUSTER_MODE).image is not None]
        if not specs:
            return {}
        images: list[str] = []
        for spec in specs:
            image = spec.profile(CLUSTER_MODE).image
            if image not in images:
                images.append(image)
        proc = self._run(host, ["availability"], self.status_timeout_s,
                         {"CLUSTER_AVAILABILITY_MODELS":
                          "\n".join(f"{s.name}\t" + ":".join(s.cluster_dirs) for s in specs),
                          "CLUSTER_AVAILABILITY_IMAGES": " ".join(images)})
        if proc is None:
            return None
        data = _last_json(proc.stdout)
        if proc.returncode != 0:
            self._fail(host, "availability", _why(proc, data))
            return None
        seen = _availabilities(data, specs, time.time() if observed_at is None else observed_at)
        if seen is None:
            self._fail(host, "availability", f"not an availability reply: {proc.stdout.strip()!r}")
            return None
        self.last_error[host.box] = ""
        return seen

    # ── plumbing ───────────────────────────────────────────────────────────

    def _argv(self, host: ClusterHost, command: str) -> list[str]:
        return [self.ssh_bin, "-i", self.key_path,
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2",
                "-o", "ControlMaster=auto", "-o", f"ControlPath={self.control_dir}/cm-%C",
                "-o", "ControlPersist=120",
                host.ssh_target, command]

    def _command(self, host: ClusterHost, words: list[str], extra: dict | None = None) -> str:
        """`env K=V … bash -s -- <words>`, quoted for the box's login shell, which parses it. It names
        no path on the box: bash comes from the box's PATH, the script from stdin. `extra` is this
        call's own settings: a ModelChoice's LaunchProfile, or what `availability` is to look for."""
        env = {**self.env, **(extra or {}), "CLUSTER_HOST_IP": host.interconnect}
        bad = [key for key in env if not _ENV_KEY.fullmatch(key)]
        if bad:
            raise ValueError(f"env keys must be variable names, not {bad!r}")
        assignments = [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]
        return " ".join(["env", *assignments, "bash", "-s", "--", *map(shlex.quote, words)])

    def _run(self, host: ClusterHost, words: list[str], timeout: float,
             env: dict | None = None) -> subprocess.CompletedProcess[str] | None:
        verb = words[0]
        try:
            script = Path(self.script_path).read_bytes()
            argv = self._argv(host, self._command(host, words, env))
            # input=: the script's bytes travel on stdin, so nothing on a box is ever run by path.
            proc = subprocess.run(argv, input=script, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            self._fail(host, verb, f"timed out after {timeout:g} s")
            return None
        except (OSError, ValueError) as exc:
            self._fail(host, verb, f"{type(exc).__name__}: {exc}")
            return None
        # errors="replace": one stray byte in a vLLM log must not cost the whole reply.
        return subprocess.CompletedProcess(argv, proc.returncode,
                                           proc.stdout.decode("utf-8", "replace"),
                                           proc.stderr.decode("utf-8", "replace"))

    def _action(self, host: ClusterHost, words: list[str],
                env: dict | None = None) -> tuple[bool, str]:
        verb = words[0]
        proc = self._run(host, words, self.action_timeout_s, env)
        if proc is None:
            return False, self.last_error[host.box]
        data = _last_json(proc.stdout)
        if proc.returncode != 0 or not isinstance(data, dict) or data.get("ok") is not True:
            self._fail(host, verb, _why(proc, data))
            # tp-rank REFUSED THE PLAN (`fail 2`, its own JSON reason): the same on every retry, so it is raised as the
            # loop's refusal — generation 221 retried one every ~2 s under "launching both ranks". A box or a link
            # failure (exit 1, ssh's 255, a timeout) still answers (False, why), and is retried.
            if verb == "up" and proc.returncode == REFUSED_EXIT and isinstance(data, dict) and data.get("error"):
                raise LaunchRefused(f"tp-rank on {host.box} refused the launch: {self.last_error[host.box]}")
            return False, self.last_error[host.box]
        self.last_error[host.box] = ""
        return True, _done(data)

    def _fail(self, host: ClusterHost, verb: str, why: str) -> None:
        self.last_error[host.box] = f"{verb}: {why}"[:ERROR_MAX_CHARS]


def _is_int(value: object, lo: int, hi: int) -> bool:
    """An int within [lo, hi] — and not a bool, which Python counts as an int and vLLM would not."""
    return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi


#: tp-rank's exit for a request it refuses (`fail 2`): for `up`, the launch plan — never retried.
REFUSED_EXIT = 2


class LaunchRefused(ValueError):
    """These SETTINGS cannot be launched — the words are the launch's own, the same on every retry. Raised by
    `build_launch` before anything is dialled; `up` turns it into `(False, "up: …")` for its callers, and the loop asks
    `launch_problem` for the stored selection so the start FAILS with these words instead of being planned again."""


def build_launch(spec: ModelSpec | None, *, rank: int, gen: int, max_num_seqs: int, max_model_len: int,
                 max_num_batched_tokens: int | None = None, num_speculative_tokens: int | None = None,
                 ple_replicate: bool | None = None, kv_cache_memory: int | None = None,
                 kv_offload_gib: int | None = None, long_prefill_token_threshold: int | None = None,
                 uncensored: bool | None = None, kv_cache_dtype: str | None = None) -> tuple[list[str], dict | None]:
    """THE ONE BUILD of a rank's launch: tp-rank's argv words and its settings, or `LaunchRefused` saying why these
    settings cannot be launched — tp-rank's own ranges, a knob the ModelChoice cannot take, a twin it does not have,
    a KV pin past its profile's ceiling (`launch_env`). `up` sends what this builds; the cluster loop asks it about the
    stored selection BEFORE planning a launch (`launch_problem`), so a refusal that used to come back from `up` as
    `(False, …)` on every tick — generation 187, 131 times in six minutes — fails the start on the first plan.

    THE BUDGET THIS LAUNCH REALLY USES is resolved once for both places it is written: tp-rank's
    CLUSTER_MAX_NUM_BATCHED_TOKENS and, for a non-vLLM engine, its own setting in `launch_env`."""
    words = ["up", str(rank), str(gen), str(max_num_seqs), str(max_model_len)]
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    if max_num_batched_tokens is None and profile is not None:
        max_num_batched_tokens = profile.prefill_budget
    problem = (_up_problem(rank, gen, max_num_seqs, max_model_len)
               or _knob_problem(max_num_batched_tokens, num_speculative_tokens, spec,
                                long_prefill_token_threshold, max_model_len))
    if problem:
        raise LaunchRefused(problem)
    # A KNOB THE ENGINE CHOOSES ITSELF (TensorFold's pool, prefill budget, drafter depth: `catalog.ENGINE_REFUSES`) is
    # refused at the launch too, never dropped on the way to it — the switch refuses it first (`DesiredStore.set`).
    given = {"kv_cache_memory": kv_cache_memory, "max_num_batched_tokens": max_num_batched_tokens,
             "num_speculative_tokens": num_speculative_tokens or None}
    for knob, value in given.items():
        if value is not None and profile is not None and knob in profile.refuses:
            raise LaunchRefused(f"{knob}: {profile.refuses[knob]}")
    env = None
    if spec is not None:
        try:
            spec = spec.launched_from(bool(uncensored))
            env = launch_env(spec, num_speculative_tokens=num_speculative_tokens,
                             ple_replicate=ple_replicate, kv_cache_memory=kv_cache_memory,
                             kv_offload_gib=kv_offload_gib, max_num_batched_tokens=max_num_batched_tokens,
                             kv_cache_dtype=kv_cache_dtype)
        except (AttributeError, TypeError, ValueError) as exc:
            raise LaunchRefused(f"{type(exc).__name__}: {exc}") from exc
        words.append(spec.id)
    if max_num_batched_tokens is not None:
        env = {**(env or {}), "CLUSTER_MAX_NUM_BATCHED_TOKENS": str(max_num_batched_tokens)}
    if _taken(spec, "long_prefill_token_threshold", long_prefill_token_threshold):
        env = {**(env or {}), "CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD": str(long_prefill_token_threshold)}
    return words, env


def launch_problem(spec: ModelSpec | None, *, estimated: bool = False, **settings) -> str:
    """Why `build_launch(spec, **settings)` would refuse, in its words — "" when the launch builds. And THE
    MEASUREMENT GATE, at the last step too: a model never measured on these cards (`Resolution.sizing` "unmeasured")
    launches only when the operator chose the estimate for this Start (`estimated`, `Desired.estimated`) — a stored
    selection from before the catalog lost its measurement is refused here, never launched on a guess."""
    profile = spec.profile(CLUSTER_MODE) if spec is not None else None
    resolution = profile.resolution if profile is not None else None
    if resolution is not None and resolution.sizing == "unmeasured" and not estimated:
        return resolution.refused
    if resolution is not None and resolution.refused and resolution.sizing != "unmeasured":
        return resolution.refused
    try:
        build_launch(spec, **settings)
    except LaunchRefused as exc:
        return str(exc)
    return ""


def _up_problem(rank: object, gen: object, max_num_seqs: object, max_model_len: object) -> str:
    """Why tp-rank would refuse this launch, or "" when it would not."""
    for name, value, (lo, hi) in (("rank", rank, RANK_RANGE), ("gen", gen, GEN_RANGE),
                                  ("max_num_seqs", max_num_seqs, MAX_NUM_SEQS_RANGE),
                                  ("max_model_len", max_model_len, MAX_MODEL_LEN_RANGE)):
        if not _is_int(value, lo, hi):
            return f"{name} must be an integer in {lo}..{hi}, not {value!r}"
    return ""


def _knob_problem(max_num_batched_tokens: object, num_speculative_tokens: object, spec,
                  long_prefill_token_threshold: object = None, max_model_len: object = None) -> str:
    """Why the operator's engine knobs cannot be launched, or "". None is the default and always can be. A
    drafter depth needs the ModelChoice's own drafter to set: tp-rank's fallback drafter is never re-shaped."""
    for name, value, (lo, hi) in (("max_num_batched_tokens", max_num_batched_tokens, MAX_NUM_BATCHED_TOKENS_RANGE),
                                  ("num_speculative_tokens", num_speculative_tokens, NUM_SPECULATIVE_TOKENS_RANGE),
                                  ("long_prefill_token_threshold", long_prefill_token_threshold,
                                   LONG_PREFILL_TOKEN_THRESHOLD_RANGE)):
        if value is not None and not _is_int(value, lo, hi):
            return f"{name} must be an integer in {lo}..{hi}, not {value!r}"
    if (long_prefill_token_threshold and isinstance(max_model_len, int)
            and long_prefill_token_threshold > max_model_len):
        return (f"long_prefill_token_threshold ({long_prefill_token_threshold}) cannot be greater than "
                f"max_model_len ({max_model_len})")
    if num_speculative_tokens is not None and spec is None:
        return "num_speculative_tokens needs a ModelChoice whose LaunchProfile it sets"
    return ""


def _last_json(stdout: str) -> object:
    """tp-rank's reply is its ONE line, taken as the last non-empty one: sshd runs the command
    through the box's login shell, and a chatty rc file may print before it."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError:
        return None


def _int_or_none(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _is_status(data: object) -> bool:
    """The fields the planner reads, with the types it reads them as. Anything else is "not
    observed" — never a guess filled in with defaults."""
    if not isinstance(data, dict) or not isinstance(data.get("rank"), dict):
        return False
    rank, mem, choice = data["rank"], data.get("mem_available_gib"), data["rank"].get("model_choice")
    return (isinstance(rank.get("present"), bool) and isinstance(rank.get("running"), bool)
            and _int_or_none(rank.get("gen")) and _int_or_none(rank.get("exit_code"))
            and (choice is None or isinstance(choice, str))
            and isinstance(data.get("broker_vllm_running"), bool)
            and isinstance(mem, (int, float)) and not isinstance(mem, bool))


def _availabilities(data: object, specs, observed_at: float) -> dict | None:
    """tp-rank's availability reply as one `Availability` per ModelSpec, keyed by spec id, in the order
    they were asked about — or None unless the box answered for every one of them, and for its image."""
    if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
        return None
    models, images = data.get("models"), data["images"]
    if not isinstance(models, list) or len(models) != len(specs):
        return None
    seen = {}
    for spec, item in zip(specs, models):
        if not isinstance(item, dict) or item.get("name") != spec.name:
            return None
        where, searched, error = item.get("dir"), item.get("searched"), item.get("error", "")
        unusable = item.get("unusable")         # required: a reply without it did not say the files are readable
        image = spec.profile(CLUSTER_MODE).image
        if image not in images:
            return None
        present = images[image]
        if (not (where is None or isinstance(where, str)) or not isinstance(error, str) or not isinstance(unusable, str)
                or not (present is None or isinstance(present, bool))
                or not isinstance(searched, list) or not all(isinstance(d, str) for d in searched)):
            return None
        # Keyed per CHECKPOINT: a model and its uncensored twin are two answers (`catalog.availability_key`).
        seen[spec.availability_key] = Availability(name=spec.name, searched=tuple(searched), dir=where,
                                                   image=image, image_present=present, observed_at=observed_at,
                                                   error=error, unusable=unusable)
    return seen


def _why(proc: subprocess.CompletedProcess[str], data: object) -> str:
    """tp-rank's own reason when it gave one, else whatever ssh said — exit 255 is ssh's own failure
    (refused key, unknown host key, no route), and only its stderr says which."""
    if isinstance(data, dict) and isinstance(data.get("error"), str) and data["error"]:
        text = data["error"]
    else:
        text = proc.stderr.strip() or proc.stdout.strip() or "no output"
    return f"exit {proc.returncode}: {text}"


#: tp-rank's reply key for each verb that ends a rank → what it did, and what it found nothing to do to.
_ENDED = {"removed": ("removed", "nothing to remove"), "halted": ("halted", "nothing to halt")}


def _done(data: dict) -> str:
    """A success in words: `started 0123456789ab`, `already-running`, `removed`, `nothing to halt`,
    `remounted /srv/hdd; worker backstop restarted`."""
    if "worker_unit" in data:                            # heal
        remounted = [str(m) for m in data.get("remounted") or []]
        return (f"remounted {', '.join(remounted)}" if remounted else "every NFS mount was up") \
            + ("; no worker backstop named" if data["worker_unit"] == "none" else f"; worker backstop {data['worker_unit']}")
    for key, (did, nothing) in _ENDED.items():
        if key in data:
            return did if data[key] else nothing
    action = str(data.get("action") or "ok")
    return f"{action} {data['container']}" if data.get("container") else action
