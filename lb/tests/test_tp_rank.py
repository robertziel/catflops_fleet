"""S4 — the executor: `tp-rank` does four verbs and validates every argument, and `SshExecutor`
runs it on a GB10 box over SSH with nothing installed there.

`lb/cluster/tp-rank` ships in the LB image. `SshExecutor` sends its bytes on stdin to
`env CLUSTER_…=… bash -s -- <verb> <args…>`, so these tests feed the real script to `bash -s` the
same way. A fake `docker` on PATH keeps its containers in a JSON file and logs every argv before it
does anything else — so "refused" is asserted as "docker never ran", and a launch as the exact argv
docker would have received.

`SshExecutor` is tested end to end through a fake `ssh` that insists on the pinned options, then does
what sshd does with a plain key: hands the command line to the box's shell (here a local `bash -c`,
in the box's own environment and home) with stdin passed through.

Nothing here needs SSH, Docker, a GPU or a broker.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest import mock

import pytest

from tests.fakes import FakeBackend

TP_RANK = Path(__file__).resolve().parents[1] / "cluster" / "tp-rank"
NAME = "qwen38-cluster-vllm"
MODEL = "Qwen3.8-Flash-Next-hibrid48"
SPEC = '{"method":"mtp","num_speculative_tokens":4}'
#: The same drafter, pointed at its view beside the checkpoint.
VIEW_SPEC = ('{"method":"mtp","num_speculative_tokens":4,'
             '"model":"/models/Qwen3.8-Flash-Next-hibrid48-mtp"}')
TARGET = "user@192.0.2.2"
STARTED = "2026-09-15T19:58:01.123456789Z"
#: When the fake box booted (`btime` in its /proc/stat): 2026-09-15T06:00:00Z, well before STARTED.
BOOTED = 1789452000
#: The GB10 pair's topology as the LB sends it (vllm_lb/executors.py `TpAcrossHosts.facts` for lb/fleet.example.yml).
GB10_PAIR_FACTS = {"CLUSTER_TP_SIZE": "2", "CLUSTER_NNODES": "2", "CLUSTER_GPUS": "all", "CLUSTER_LINK": "roce",
                   "CLUSTER_MEMORY": "unified", "CLUSTER_PLATFORM": "linux/arm64"}
EMPTY_RANK = {"present": False, "running": False, "gen": None, "exit_code": None,
              "started_at": None, "model_choice": None, "exit_reason": ""}
#: Pass 4's ModelChoices for the cluster (the model library, vllm_lb/library): today's model, and the dense 27B.
FLASH, DENSE = "qwen3.8-flash-next-hibrid48", "qwen3.8-27b-nvidia-nvfp4"

#: The default image as the ONE path component its caches live under, spelled out rather than
#: recomputed — `/` → `_S`, `:` → `_C` (and `_` → `__`, `@` → `_A`) — because a new layout is a cold
#: start on every box.
IMAGE_DIR = "myllmbox_Sqwen38-flash-next-cluster-vllm_Cv5"


#: Where a rank keeps FlashInfer's autotune results: a tmpfs of the container's own, so they NEVER outlive
#: the start that made them (the 2026-09-18 hang: a cache written by rank 0 alone, keyed by tp_rank).
AUTOTUNE_DIR = "/flashinfer-autotune"
AUTOTUNE_TMPFS = f"{AUTOTUNE_DIR}:rw,nosuid,nodev,noexec,size=16m"


def cache_of(home: Path, *, image_dir: str = IMAGE_DIR, model: str = MODEL, ple: str = "1") -> Path:
    """A profile key's cache dir on a box with this home: image / model / TP size and PLE mode."""
    return home / ".cache" / "vllm-cluster" / image_dir / model / f"tp2-ple{ple}"


#: Today's checkpoint as its index lays it out, from model.safetensors.index.json of the copy on
#: box-c's HDD (read 2026-09-15). The shard names are real, and so is which KIND of tensor sits in
#: which shard: the drafter's `mtp.` tensors in 00018 and 00019, the embedding in 00001, lm_head in
#: model-lmhead-nvfp4 — beside what the drafter never reads, look-alike names included (the vision
#: tower's patch_embed, the PLE table's ngram_embedding). The tensor names themselves are samples.
INDEX = "model.safetensors.index.json"
REAL_SHARDS = {
    "model-00001-of-00021.safetensors": ["model.language_model.embed_tokens.weight",
                                         "model.language_model.layers.0.self_attn.q_proj.weight"],
    "model-00002-of-00021.safetensors": ["model.visual.patch_embed.proj.weight",
                                         "model.language_model.layers.3.mlp.experts.0.up_proj.weight"],
    **{f"model-{i:05d}-of-00021.safetensors": [
        f"model.language_model.layers.{i * 2}.mlp.experts.511.down_proj.weight_scale"]
       for i in range(3, 18)},
    "model-00018-of-00021.safetensors": [
        "model.language_model.layers.47.mlp.experts.9.gate_proj.input_scale",
        "mtp.layers.0.mlp.experts.0.up_proj.weight"],
    "model-00019-of-00021.safetensors": ["mtp.fc_embedding.weight", "mtp.pre_fc_norm_embedding.weight"],
    "model-lmhead-nvfp4.safetensors": ["lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2"],
    **{f"ple-nvfp4-{i:05d}-of-00008.safetensors": [
        f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_shard_{i - 1}.packed"]
       for i in range(1, 9)},
}
#: The rest of its top level: config, tokenizer, processor and template files, a licence, READMEs.
REAL_FILES = ["LICENSE", "README-base.md", "README.md", "chat_template.jinja", "config.json",
              "generation_config.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
              "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"]
#: The four shards research §4 B saw the drafter read in the load log: what the index must yield.
DRAFTER_SHARDS = ["model-00001-of-00021.safetensors", "model-00018-of-00021.safetensors",
                  "model-00019-of-00021.safetensors", "model-lmhead-nvfp4.safetensors"]


def lay_out_checkpoint(checkpoint: Path, *, shards: dict[str, list[str]] | None = None,
                       index: bool = True) -> Path:
    """Fill a checkpoint dir the way the real one is laid out: its files, (empty) shards, what a
    download leaves hidden beside them — never part of a view — and, unless told not to, the index."""
    shards = REAL_SHARDS if shards is None else shards
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name in REAL_FILES:
        (checkpoint / name).write_text("{}\n")
    for shard in shards:
        (checkpoint / shard).write_bytes(b"")
    (checkpoint / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (checkpoint / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    if index:
        weight_map = {tensor: shard for shard, tensors in shards.items() for tensor in tensors}
        (checkpoint / INDEX).write_text(json.dumps(
            {"metadata": {"total_size": 105280623656}, "weight_map": weight_map}, indent=2))
    return checkpoint

FAKE_DOCKER = r'''#!/usr/bin/env python3
"""A docker CLI that knows four command shapes and keeps its containers in ../docker-state.json.

Every argv is logged BEFORE anything else, so a test can prove docker was never reached.
Anything tp-rank has no business sending exits 97.
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
STATE, CALLS = ROOT / "docker-state.json", ROOT / "docker-calls.jsonl"
RANK_FORMAT = ('{{.State.Running}}|{{.State.ExitCode}}|{{.State.StartedAt}}|'
               '{{index .Config.Labels "lb.cluster.gen"}}|{{index .Config.Labels "lb.cluster.model_choice"}}')
# The memory guard's poll: is our rank still running, and which generation is it.
GUARD_FORMAT = '{{.State.Running}}|{{index .Config.Labels "lb.cluster.gen"}}'

args = sys.argv[1:]
with CALLS.open("a") as f:
    f.write(json.dumps(args) + "\n")
state = json.loads(STATE.read_text()) if STATE.exists() else {}
containers = state.setdefault("containers", {})


def die(msg, code=1):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


if state.get("daemon_down"):
    die("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
        "Is the docker daemon running?")

if len(args) == 5 and args[:4] == ["image", "inspect", "-f", '{{index .Config.Labels "lb.io_uring"}}']:
    print(state.get("image_labels", {}).get(args[4], {}).get("lb.io_uring", ""))   # docker prints "" for no label
elif len(args) == 5 and args[:4] == ["image", "inspect", "-f", "{{.Id}}"]:
    if args[4] not in state.get("images", []):
        die(f"Error: No such image: {args[4]}")
    print("sha256:" + hashlib.sha256(args[4].encode()).hexdigest())
elif len(args) == 4 and args[:2] == ["inspect", "-f"]:
    fmt, name = args[2], args[3]
    c = containers.get(name)
    if c is None:
        die(f"error: no such object: {name}")                   # docker 29.1's exact words
    running = "true" if c["running"] else "false"
    if fmt == RANK_FORMAT:
        gen = c["labels"].get("lb.cluster.gen", "")
        choice = c["labels"].get("lb.cluster.model_choice", "")
        print(f"{running}|{c['exit_code']}|{c['started_at']}|{gen}|{choice}")
    elif fmt == "{{.State.Running}}":
        print(running)
    elif fmt == GUARD_FORMAT:
        print(f"{running}|{c['labels'].get('lb.cluster.gen', '')}")
    else:
        die(f"fake docker: unexpected format {fmt!r}", 97)
elif len(args) == 4 and args[:3] == ["kill", "-s", "SIGKILL"]:
    c = containers.get(args[3])
    if c is None or not c["running"]:
        die(f"Error response from daemon: cannot kill container: {args[3]}: container is not running")
    c["running"], c["exit_code"] = False, 137
    STATE.write_text(json.dumps(state))
    print(args[3])
elif len(args) == 3 and args[:2] == ["rm", "-f"]:
    containers.pop(args[2], None)
    STATE.write_text(json.dumps(state))
elif len(args) == 4 and args[:2] == ["stop", "-t"] and args[2].isdigit():
    c = containers.get(args[3])
    if c is None:
        die(f"Error response from daemon: No such container: {args[3]}")
    if state.get("stop_error"):
        die(state["stop_error"])
    if c["running"]:                         # SIGTERM, then SIGKILL: the container stays, exited
        c["running"], c["exit_code"] = False, 137
    STATE.write_text(json.dumps(state))
    print(args[3])
elif len(args) > 2 and args[:2] == ["run", "-d"]:
    if state.get("run_error"):
        die(state["run_error"], 125)
    name, labels = None, {}
    for flag, value in zip(args, args[1:]):
        if flag == "--name":
            name = value
        elif flag == "--label":
            key, _, val = value.partition("=")
            labels[key] = val
    if name in containers:                   # the real daemon refuses a name in use, and so do we
        die(f'docker: Error response from daemon: Conflict. The container name "/{name}" '
            "is already in use.", 125)
    containers[name] = {"running": True, "exit_code": 0, "started_at": "2026-09-15T20:00:00Z",
                        "labels": labels, "logs": []}
    STATE.write_text(json.dumps(state))
    print(hashlib.sha256(json.dumps(args).encode()).hexdigest())
elif args == ["images", "--format", "{{.Repository}}:{{.Tag}}"]:
    for ref in state.get("images", []):
        print(ref)
elif len(args) > 20 and args[:5] == ["run", "--rm", "--privileged", "--pid=host", "--net=host"]:
    # heal's root shell: <image> nsenter -t 1 -m -u -i -n -p -- sh -c <script> sh <timeout> <mount…>
    if args[5] not in state.get("images", []):
        die(f"Unable to find image '{args[5]}' locally", 125)
    if args[6:17] != ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "sh", "-c"] or args[18] != "sh":
        die(f"fake docker: unexpected root shell {args[6:19]!r}", 97)
    rc = 0
    for mp in args[20:]:
        if mp in state.get("unreachable", []):
            print(f"mount.nfs: Network is unreachable for 192.0.2.11:{mp} on {mp}")
            rc = 32
        elif mp not in state.setdefault("mounted", []):
            state["mounted"].append(mp)
    STATE.write_text(json.dumps(state))
    sys.exit(rc)
elif (len(args) == 19 and args[:5] == ["run", "--rm", "--privileged", "--pid=host", "--net=host"]
      and args[6:17] == ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "sh", "-c"] and args[18] == "sh"):
    # a root shell with no further arguments (`stacks` reading the kernel log): run its script here, as root
    import os
    import subprocess
    if args[5] not in state.get("images", []):
        die(f"Unable to find image '{args[5]}' locally", 125)
    sys.exit(subprocess.run(["sh", "-c", args[17], "sh"], env={**os.environ, "FAKE_ROOT": "1"}).returncode)
elif len(args) == 12 and args[:8] == ["run", "--rm", "--pull", "never", "--network", "none",
                                      "--entrypoint", "python3"] and args[9:11] == ["-I", "-c"]:
    # `up` reading the image's own files (its KV cache dtypes): the script runs HERE against the site dir a
    # test laid out for that image (`vllm_site`), never against a real vLLM.
    import os
    import subprocess
    site = state.get("vllm_site", {}).get(args[8])
    if site is None:
        die(f"Error response from daemon: No such image: {args[8]}", 125)
    proc = subprocess.run([sys.executable, "-c", args[11]], env={**os.environ, "PYTHONPATH": site},
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)
elif len(args) == 4 and args[:2] == ["logs", "--tail"]:
    c = containers.get(args[3])
    if c is None:
        die(f"Error response from daemon: No such container: {args[3]}")
    for line in c.get("logs", [])[-int(args[2]):]:
        sys.stderr.write(line + "\n")                   # vLLM logs to stderr, like the real one
elif len(args) == 7 and args[0] == "exec" and args[2:4] == ["sh", "-c"] and args[5] == "sh":
    # `docker exec NAME sh -c <script> sh <arg>`: run INSIDE the container — here this box, whose process
    # table stands in for the container's, on the container's own PATH (`exec_path`) when a test gives one.
    import os
    import subprocess
    c = containers.get(args[1])
    if c is None:
        die(f"Error response from daemon: No such container: {args[1]}")
    if not c["running"]:
        die(f"Error response from daemon: container {args[1]} is not running")
    env = {**os.environ, "PATH": state["exec_path"]} if state.get("exec_path") else None
    sys.exit(subprocess.run(args[2:], env=env).returncode)
else:
    die(f"fake docker: unsupported command {args!r}", 97)
'''

FAKE_MOUNTPOINT = r'''#!/usr/bin/env python3
"""`mountpoint -q <path>`: 0 when docker-state.json lists it as mounted, 32 when it does not."""
import json
import sys
from pathlib import Path
from unittest import mock

STATE = Path(__file__).resolve().parent.parent / "docker-state.json"
state = json.loads(STATE.read_text()) if STATE.exists() else {}
if sys.argv[1:2] != ["-q"] or len(sys.argv) != 3:
    sys.exit(97)
sys.exit(0 if sys.argv[2] in state.get("mounted", []) else 32)
'''

#: `nvidia-smi` as a GB10's really answers (box-a, 2026-09-29 19:02Z), for the two forms `status` uses.
#: The `-` columns are real: mtemp and mclk have no value on this card, and dmon says so with a dash.
#: docker-state.json decides what it reports, or that it fails like a wedged driver.
FAKE_NVIDIA_SMI = r"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from unittest import mock

STATE = Path(__file__).resolve().parent.parent / "docker-state.json"
state = json.loads(STATE.read_text()) if STATE.exists() else {}
gpu = state.get("gpu", {})
if gpu.get("broken"):
    sys.exit(15)
rows = gpu.get("rows") or [{"pwr": 57, "gtemp": 71, "sm": 96, "mem": 0, "pclk": 2424}]
if sys.argv[1:2] == ["dmon"]:
    print("# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk ")
    print("# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz ")
    for i, r in enumerate(rows):
        print("    %d %6d %6d      - %6d %6d      0      0      0      0      - %6d"
              % (i, r["pwr"], r["gtemp"], r["sm"], r["mem"], r["pclk"]))
    sys.exit(0)
if "--query-gpu=clocks.max.sm" in sys.argv:
    for _ in rows:
        print(gpu.get("max_clock", 3003))
    sys.exit(0)
if "--query-gpu=utilization.gpu,power.draw,memory.used" in sys.argv and "--format=csv" in sys.argv:
    print("utilization.gpu [%], power.draw [W], memory.used [MiB]")
    for r in rows:
        print(f"{r['sm']} %, {r['pwr']:.2f} W, [N/A]")
    sys.exit(0)
sys.exit(97)
"""

FAKE_SYSTEMCTL = r'''#!/usr/bin/env python3
"""`systemctl --user` for the worker's backstop: `show -p <LoadState|SubState> --value <unit>` and
`restart <unit>`, with the units in docker-state.json. Every argv is logged first."""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
STATE, CALLS = ROOT / "docker-state.json", ROOT / "systemctl-calls.jsonl"
args = sys.argv[1:]
with CALLS.open("a") as f:
    f.write(json.dumps(args) + "\n")
state = json.loads(STATE.read_text()) if STATE.exists() else {}
units = state.setdefault("units", {})
if len(args) == 6 and args[:3] == ["--user", "show", "-p"] and args[4] == "--value":
    unit = units.get(args[5])
    if args[3] == "LoadState":
        print(unit["load"] if unit else "not-found")
    elif args[3] == "SubState":
        print(unit["sub"] if unit else "dead")
    else:
        sys.exit(97)
elif len(args) == 3 and args[:2] == ["--user", "restart"]:
    if state.get("restart_error"):
        sys.stderr.write(state["restart_error"] + "\n")
        sys.exit(1)
    units.setdefault(args[2], {"load": "loaded"})["sub"] = "running"
    STATE.write_text(json.dumps(state))
else:
    sys.stderr.write(f"fake systemctl: unsupported {args!r}\n")
    sys.exit(97)
'''

FAKE_CURL = r'''#!/usr/bin/env python3
"""curl as the memory guard asks a /health: `curl -s -o /dev/null -w %{http_code} --max-time <s> <url>`.
Every argv is logged first. A URL docker-state.json's `http` maps answers with that status code; any other URL
is refused the way real curl reports it, `000` and exit 7. Any other shape exits 97."""
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
STATE, CALLS = ROOT / "docker-state.json", ROOT / "curl-calls.jsonl"
args = sys.argv[1:]
with CALLS.open("a") as f:
    f.write(json.dumps(args) + "\n")
if len(args) != 8 or args[:6] != ["-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time"]:
    sys.exit(97)
state = json.loads(STATE.read_text()) if STATE.exists() else {}
code = state.get("http", {}).get(args[7])
sys.stdout.write("000" if code is None else str(code))
sys.exit(7 if code is None else 0)
'''

#: a box's /etc/fstab (the 2026-09-28 shape): four NFS mounts among local disks, and a retired one.
FSTAB = """# /etc/fstab: static file system information.
UUID=3b1f2c1e-0000-4000-8000-000000000001  /          ext4  defaults    0 1
UUID=3B1F-0002                             /boot/efi  vfat  umask=0077  0 1
192.0.2.11:/mnt/nas-a  /mnt/nas-a  nfs  vers=4,_netdev,nofail,bg,x-systemd.automount,hard,noatime  0 0
192.0.2.11:/mnt/nas-b  /mnt/nas-b  nfs  vers=4,_netdev,nofail,bg,x-systemd.automount,hard,noatime  0 0
192.0.2.11:/srv/hdd  /srv/hdd  nfs  vers=4,_netdev,nofail,bg,x-systemd.automount,hard,noatime  0 0
192.0.2.201:/  /mnt/box-a-home  nfs  vers=4,_netdev,nofail,bg,x-systemd.automount,hard,noatime,nconnect=4  0 0    # pair-link-nfs
#192.0.2.11:/mnt/retired  /mnt/retired  nfs  defaults  0 0
"""

FAKE_SSH = r'''#!/usr/bin/env python3
"""An ssh that is not a network. It refuses to "connect" unless the executor pinned the key, the
host keys and connection reuse; then it does what sshd does with a plain key — hands the command
line to the box's shell (a local `bash -c`, in the box's environment and home) with stdin passed
through — and keeps a copy of that stdin, so a test can see what travelled."""
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "ssh.json").read_text())
args = sys.argv[1:]
with (ROOT / "ssh-calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\n")

options = {value for flag, value in zip(args, args[1:]) if flag == "-o"}
missing = [o for o in cfg["options"] if o not in options]
if missing or "-i" not in args or args[args.index("-i") + 1] != cfg["key"]:
    sys.stderr.write(f"fake ssh: will not connect without the pinned key and {missing}\n")
    sys.exit(255)
if cfg["mode"] == "refuse":
    sys.stderr.write(f"{args[-2]}: Permission denied (publickey).\n")
    sys.exit(255)
if cfg["mode"] == "hang":
    time.sleep(30)
    sys.exit(0)
if cfg["mode"] == "reply":
    sys.stdout.write(cfg["reply"])
    sys.exit(0)
stdin = sys.stdin.buffer.read()
(ROOT / "ssh-stdin.bin").write_bytes(stdin)
shell = subprocess.run(["bash", "-c", args[-1]], input=stdin, env=cfg["env"], cwd=cfg["home"])
sys.exit(shell.returncode)
'''


def _install(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class FakeBox:
    """One fake GB10 box: a fake `docker` on PATH, a meminfo, a home with the checkpoint on its NVMe —
    and no tp-rank. As on the real boxes, every run feeds the script's bytes to `bash -s` on stdin."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "bin").mkdir()
        _install(root / "bin" / "docker", FAKE_DOCKER)
        _install(root / "bin" / "mountpoint", FAKE_MOUNTPOINT)
        _install(root / "bin" / "systemctl", FAKE_SYSTEMCTL)
        #: NOT installed: a box with no nvidia-smi is the default, and `with_gpu()` gives it one. So
        #: every existing test asserts the no-driver case without saying so.
        self.fstab = root / "fstab"
        self.fstab.write_text(FSTAB)
        self.home = root / "home"
        self.home.mkdir()
        # box-a's own NVMe copy: where the default search path looks first
        self.models = self.place_checkpoint(self.home / "models" / "qwen38-flash-next-cluster")
        self.meminfo = root / "meminfo"
        self.meminfo.write_text("MemTotal:       121690000 kB\n"
                                "MemFree:          3000000 kB\n"
                                "MemAvailable:   117000000 kB\n")
        # The box's /proc/stat, for its boot time (`btime`): this fake box booted at BOOTED.
        self.proc_stat = root / "proc_stat"
        self.proc_stat.write_text("cpu  2255 34 2290 22625563 6290 127 456 0 0 0\n"
                                  f"btime {BOOTED}\n"
                                  "processes 12345\n")
        self.guard_file = root / "tp-rank-memguard.log"
        # THE FLASH-NEXT vLLM IMAGE'S OWN CacheDType, as the real boxes hold it (vllm_lb/kv_cache_dtypes.json, read
        # in the image on box-a 2026-10-22): its shipped profile launches fp8_e4m3 KV (2026-10-26), which `up` checks
        # against this list before anything starts. test_vllm_engine_flags.image_accepts lays out its own instead.
        site = root / "image-site-default"
        (site / "vllm" / "config").mkdir(parents=True)
        (site / "vllm" / "__init__.py").write_text("raise SystemExit('vllm was imported')\n")
        (site / "vllm" / "config" / "cache.py").write_text(
            'from typing import Literal\n\nCacheDType = Literal["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", '
            '"fp8_e5m2", "fp8_inc", "fp8_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc", "turboquant_k3v4_nc", '
            '"turboquant_3bit_nc", "int4_per_token_head", "int8_per_token_head", "fp8_per_token_head", "nvfp4", '
            '"nvfp4_4over6"]\n')                  # the v5 image's own list, read on box-a 2026-10-26
        self._save({**self._state(), "vllm_site": {"myllmbox/qwen38-flash-next-cluster-vllm:v5": str(site)}})

    def memory(self, *, available_mib: int, swap_free_mib: int) -> None:
        """This box's /proc/meminfo as the memory guard reads it — MemAvailable and SwapFree — replaced in ONE
        rename, so a poll never reads half a file."""
        new = self.root / "meminfo.new"
        new.write_text("MemTotal:       121690000 kB\n"
                       "MemFree:          3000000 kB\n"
                       f"MemAvailable:   {available_mib * 1024} kB\n"
                       "SwapTotal:       62914556 kB\n"
                       f"SwapFree:       {swap_free_mib * 1024} kB\n")
        os.replace(new, self.meminfo)

    def with_curl(self, http: dict[str, int]) -> None:
        """Give this box a curl: each URL in `http` answers with its status code, every other is refused."""
        _install(self.root / "bin" / "curl", FAKE_CURL)
        self.set(http=http)

    def curl_calls(self) -> list[list[str]]:
        return _jsonl(self.root / "curl-calls.jsonl")

    def guard_log(self) -> str:
        return self.guard_file.read_text() if self.guard_file.exists() else ""

    def with_gpu(self, **gpu) -> None:
        """Give this box a driver that answers. `gpu` goes into docker-state.json: `rows` (one per
        card), `max_clock`, or `broken` for a driver that fails the way a wedged one does."""
        _install(self.root / "bin" / "nvidia-smi", FAKE_NVIDIA_SMI)
        if gpu:
            self.set(gpu=gpu)

    @staticmethod
    def place_checkpoint(models_dir: Path, *, name: str = MODEL, partial: bool = False) -> Path:
        """A checkpoint as the copy leaves it: `<name>.partial` while it lands, `<name>` once whole."""
        checkpoint = models_dir / (name + ".partial" if partial else name)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text("{}\n")
        return models_dir

    def env(self) -> dict[str, str]:
        """The box's own environment — built, not inherited, so nothing of the test runner's leaks
        in."""
        # TMPDIR is the box's own, so the MFU sitecustomize `up` writes lands inside the test's
        # tree instead of the runner's real /tmp.
        tmp = self.root / "tmp"
        tmp.mkdir(exist_ok=True)
        # The memory guard's log is this box's, never the runner's /var/tmp — and the guard is OFF unless a test
        # arms it and watches it (`arm`): it polls every 2 s for as long as its rank runs, and a fake rank never
        # stops on its own.
        return {"PATH": f"{self.root / 'bin'}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": str(self.home), "TP_RANK_MEMINFO": str(self.meminfo),
                "TP_RANK_PROC_STAT": str(self.proc_stat),
                "TMPDIR": str(tmp), "TP_RANK_FSTAB": str(self.fstab),
                "TP_RANK_GUARD_FILE": str(self.guard_file), "CLUSTER_GUARD": "0",
                # What the LB always sends from the fleet file — the head's interconnect and the NIC the ranks talk
                # over — set here on the box's side (tp-rank has no default for either), so an executor test can
                # assert the env IT sends verbatim; a test that means "none sent" passes "".
                "CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_IFACE": "bond0",
                # …and the cluster's prefill budget (the fleet file's `max_num_batched_tokens`), likewise.
                "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192",
                # …and THE TOPOLOGY, as facts (vllm_lb/executors.py; tp-rank has no default for any): the GB10 pair's.
                **GB10_PAIR_FACTS,
                # This fake box is an Arm one, as a GB10 is, whatever CPU runs the suite (tp-rank reads `uname -m`).
                "TP_RANK_MACHINE": "aarch64"}

    def run(self, *argv: str, config: dict[str, str] | None = None,
            extra_env: dict[str, str] | None = None,
            script: bytes | None = None) -> subprocess.CompletedProcess:
        """`bash -s -- <argv>` with the script on stdin, as SshExecutor runs it. `config` is the
        CLUSTER_* environment the executor would pass; by default, just this box's host IP."""
        return self.finish(self.start(*argv, config=config, extra_env=extra_env, script=script))

    def start(self, *argv: str, config: dict[str, str] | None = None,
              extra_env: dict[str, str] | None = None,
              script: bytes | None = None) -> subprocess.Popen:
        """`run` without waiting for it — for two boxes acting at once. `finish` collects it."""
        env = {**self.env(), **(extra_env or {}),
               **({"CLUSTER_HOST_IP": "192.0.2.201"} if config is None else config)}
        # The search path the LB SENDS for this model (catalog folder under each MODELS_SEARCH_DIRS entry, conftest's
        # layout) — tp-rank has no default of its own since 2026-10-09; a test that means "none sent" passes "".
        env.setdefault("CLUSTER_MODELS_DIRS", ":".join(f"{root}/qwen38-flash-next-cluster" for root in (
            f"{self.home}/models", "/mnt/box-a-home/models", "/srv/models")))
        proc = subprocess.Popen(["bash", "-s", "--", *argv], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                cwd=self.home)
        try:
            proc.stdin.write(TP_RANK.read_bytes() if script is None else script)
            proc.stdin.close()
        except BrokenPipeError:                 # bash stopped reading: its exit status says why
            pass
        proc.stdin = None                       # sent and closed: `finish` must not flush it again
        return proc

    @staticmethod
    def finish(proc: subprocess.Popen) -> subprocess.CompletedProcess:
        stdout, stderr = proc.communicate(timeout=30)
        return subprocess.CompletedProcess(proc.args, proc.returncode,
                                           stdout.decode("utf-8", "replace"),
                                           stderr.decode("utf-8", "replace"))

    # ── the fake docker's world ──────────────────────────────────────────────

    def _state(self) -> dict:
        path = self.root / "docker-state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _save(self, state: dict) -> None:
        (self.root / "docker-state.json").write_text(json.dumps(state))

    def seed(self, name: str = NAME, *, running: bool = True, exit_code: int = 0,
             gen: int | None = 7, model_choice: str | None = None, logs=()) -> None:
        state = self._state()
        labels = {} if gen is None else {"lb.cluster.gen": str(gen)}
        if model_choice is not None:
            labels["lb.cluster.model_choice"] = model_choice
        state.setdefault("containers", {})[name] = {
            "running": running, "exit_code": exit_code, "started_at": STARTED,
            "labels": labels, "logs": list(logs)}
        self._save(state)

    def set(self, **flags) -> None:
        self._save({**self._state(), **flags})

    def containers(self) -> dict:
        return self._state().get("containers", {})

    def calls(self) -> list[list[str]]:
        return _jsonl(self.root / "docker-calls.jsonl")

    def systemctl_calls(self) -> list[list[str]]:
        return _jsonl(self.root / "systemctl-calls.jsonl")


def cluster_host(**fields):
    """box-a's ClusterHost unless told otherwise — by keyword only, so that vllm_lb.cluster's own
    ClusterHost can replace this one without touching a call site."""
    from vllm_lb.cluster_exec import ClusterHost
    return ClusterHost(**{"label": "box-a-gpu", "box": "box-a", "ssh_target": TARGET,
                          "interconnect": "192.0.2.201", **fields})


class FakeSsh:
    """The `ssh_bin` for SshExecutor: a script beside the fake GB10 box, steered by ssh.json."""

    def __init__(self, box_a: FakeBox) -> None:
        self.root = box_a.root
        self.path = box_a.root / "ssh"
        self.key = str(box_a.root / "id_ed25519")
        self.known_hosts = str(box_a.root / "known_hosts")
        _install(self.path, FAKE_SSH)
        # The BOX's own environment holds a search path for a launch that sends none (no ModelSpec — the LB always
        # has one; these tests drive the plumbing). What the LB sends is unchanged and still overrides it.
        box_env = {**box_a.env(), "CLUSTER_MODELS_DIRS": ":".join(f"{root}/qwen38-flash-next-cluster" for root in (
            f"{box_a.home}/models", "/mnt/box-a-home/models", "/srv/models"))}
        self.cfg = {"mode": "forward", "reply": "", "env": box_env, "home": str(box_a.home),
                    "key": self.key,
                    "options": ["BatchMode=yes", "IdentitiesOnly=yes", "StrictHostKeyChecking=yes",
                                f"UserKnownHostsFile={self.known_hosts}", "ConnectTimeout=5",
                                "ServerAliveInterval=5", "ServerAliveCountMax=2",
                                "ControlMaster=auto", f"ControlPath={self.root}/cm-%C",
                                "ControlPersist=120"]}
        self.set()

    def set(self, **changes) -> None:
        self.cfg.update(changes)
        (self.root / "ssh.json").write_text(json.dumps(self.cfg))

    def calls(self) -> list[list[str]]:
        return _jsonl(self.root / "ssh-calls.jsonl")

    def stdin(self) -> bytes:
        """What the last forwarded call carried on stdin."""
        return (self.root / "ssh-stdin.bin").read_bytes()

    def executor(self, **kw):
        # Imported here, not at module level: the script's contract must not wait on the client.
        from vllm_lb.cluster_exec import SshExecutor
        ex = SshExecutor(**{"key_path": self.key, "known_hosts": self.known_hosts,
                            "script_path": str(TP_RANK), "ssh_bin": str(self.path),
                            "control_dir": str(self.root), **kw})
        return ex, cluster_host()


@pytest.fixture
def box_a(tmp_path) -> FakeBox:
    return FakeBox(tmp_path)


@pytest.fixture
def ssh(box_a) -> FakeSsh:
    return FakeSsh(box_a)


def reply(proc: subprocess.CompletedProcess) -> dict:
    """tp-rank's contract for every verb but `logs`: ONE line on stdout, and it is a JSON object."""
    lines = proc.stdout.splitlines()
    assert len(lines) == 1, f"want one JSON line; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    data = json.loads(lines[0])
    assert isinstance(data, dict), data
    return data


def mfu_of(box_a) -> str:
    """Where `up` writes the sitecustomize that teaches vLLM's MFU model this checkpoint's
    quantisation. Under the box's TMPDIR, not $HOME/.cache — the revert path must leave that
    directory untouched, and the patch is rewritten on every launch anyway."""
    return f"{box_a.root / 'tmp'}/vllm-cluster-mfu"


def launch_argv(*, rank: int, gen: int, seqs: int, ctx: int, models_dir: Path | str,
                cache_dir: Path | str | None, host_ip: str = "192.0.2.201", ple: str = "1",
                spec: str = SPEC, model_choice: str | None = None,
                mfu_patch_dir: Path | str | None = None, batched: int = 8192, threshold: int = 0,
                kv_dtype: str | None = None) -> list[str]:
    """The rank launch, element by element — every configuration default spelled out, `models_dir`
    wherever the search path found the checkpoint, `cache_dir` the profile key's cache dir (None:
    no cache), `spec` the drafter as vLLM is told it, `model_choice` the ModelChoice label, and
    `threshold` the long-prefill threshold (0: off, no flag at all), `kv_dtype` the KV dtype flag (the shipped
    profile's fp8_e4m3 since 2026-10-26; None: no flag)."""
    cache = [] if cache_dir is None else [
        "-v", f"{cache_dir}:/cache",
        "-e", "VLLM_CACHE_ROOT=/cache/vllm", "-e", "FLASHINFER_WORKSPACE_BASE=/cache/flashinfer",
        "--tmpfs", AUTOTUNE_TMPFS, "-e", f"VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR={AUTOTUNE_DIR}"]
    choice = [] if model_choice is None else ["--label", f"lb.cluster.model_choice={model_choice}"]
    argv = ["run", "-d", "--restart", "no", "--name", NAME,
            "--label", f"lb.cluster.gen={gen}", "--label", f"lb.cluster.rank={rank}", *choice,
            "--gpus", "all", "--ipc=host", "--network", "host", "--cap-add", "SYS_PTRACE",
            "--device", "/dev/infiniband",
            "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1",
            "-v", f"{models_dir}:/models:ro", *cache,
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", f"MBX_PLE_REPLICATE={ple}",
            "-e", "TORCH_NCCL_ASYNC_ERROR_HANDLING=1", "-e", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180",
            "-e", "NCCL_SOCKET_IFNAME=bond0", "-e", "GLOO_SOCKET_IFNAME=bond0",
            "-e", f"VLLM_HOST_IP={host_ip}", "-e", "NCCL_IB_DISABLE=0",
            *([] if mfu_patch_dir is None
              else ["-v", f"{mfu_patch_dir}:/mfu-patch:ro", "-e", "PYTHONPATH=/mfu-patch"]),
            "myllmbox/qwen38-flash-next-cluster-vllm:v5",
            "--model", "/models/Qwen3.8-Flash-Next-hibrid48", "--host", "0.0.0.0", "--port", "8100",
            "--nnodes", "2", "--node-rank", str(rank), "--master-addr", "192.0.2.201",
            "--master-port", "25000", "--tensor-parallel-size", "2",
            "--served-model-name", "Qwen/Qwen3.8-Flash-Next", "--max-model-len", str(ctx),
            "--max-num-seqs", str(seqs), "--max-num-batched-tokens", str(batched),
            *([] if not threshold else ["--long-prefill-token-threshold", str(threshold)]),
            "--gpu-memory-utilization", "0.70", "--kv-cache-memory", "28000000000",
            "--enable-prefix-caching", "--scheduling-policy", "priority", "--enable-mfu-metrics",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_xml", "--reasoning-parser", "qwen3",
            "--speculative-config", spec, *([] if kv_dtype is None else ["--kv-cache-dtype", kv_dtype])]
    return argv + ["--headless"] if rank == 1 else argv


# ── the script itself ─────────────────────────────────────────────────────────

def test_tp_rank_is_a_bash_script_that_never_evals_and_acts_only_on_its_last_line():
    text = TP_RANK.read_text()
    lines = text.splitlines()
    assert lines[0] in ("#!/usr/bin/env bash", "#!/bin/bash")
    assert "set -euo pipefail" in text
    code = [line for line in lines if not line.lstrip().startswith("#")]
    assert not [line for line in code if re.search(r"\beval\b", line)], "arguments are data, never code"
    assert [line for line in lines if line.strip()][-1] == 'main "$@"', \
        "nothing may run before the whole script has arrived"
    assert os.access(TP_RANK, os.X_OK), "executable too, so an operator can run it on a box by hand"


def test_a_script_cut_off_in_transit_does_nothing(box_a):
    """The script crosses the network on stdin, and a dropped connection hands bash a PREFIX of it.
    A prefix only defines things — `main "$@"` is the last line — so no prefix may reach docker."""
    box_a.seed(gen=7)
    script = TP_RANK.read_bytes()
    whole = len(script.rstrip())
    cuts = {i for i, byte in enumerate(script) if byte == 0x0A} | set(range(0, whole, 101))
    for cut in sorted(c for c in cuts if c < whole):
        box_a.run("down", script=script[:cut])
        assert box_a.calls() == [], f"a {cut}-byte prefix of the script reached docker"
    assert reply(box_a.run("down")) == {"ok": True, "removed": True}, "the whole script still acts"


def test_the_request_is_argv_and_the_config_is_the_environment_nothing_else(box_a):
    """No forced command and no config file any more: a stray SSH_ORIGINAL_COMMAND or env file on a
    box must change neither what runs nor with what."""
    box_a.seed(gen=7)
    stray = box_a.home / ".config" / "tp-rank.env"
    stray.parent.mkdir()
    stray.write_text("CLUSTER_PORT=9999\n")
    proc = box_a.run("status", config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_PORT": "8200"},
                     extra_env={"SSH_ORIGINAL_COMMAND": "down", "TP_RANK_ENV": str(stray)})
    out = reply(proc)
    assert out["rank"]["present"] is True, "argv said status; SSH_ORIGINAL_COMMAND said down"
    assert out["port"] == 8200, "the port comes from the environment, not from a file"


# ── status ────────────────────────────────────────────────────────────────────

def test_status_of_an_empty_box_a_reports_no_rank_no_broker_vllm_and_free_memory(box_a):
    proc = box_a.run("status")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # 117000000 kB / 1048576 = 111.58 GiB, printed with one decimal. `ollama` is null: nothing is
    # listening on this fake box's ollama address, and a box that could not be asked has not said.
    # `gpu` is null for the same reason: this box has no nvidia-smi, and a GPU that could not be
    # observed says nothing rather than 0 %.
    assert reply(proc) == {"rank": EMPTY_RANK, "broker_vllm_running": False,
                           "mem_available_gib": 111.6, "port": 8100, "ollama": None, "gpu": None,
                           # MemTotal (121690000 kB = 116.05 GiB): what the fleet's ram_usable_fraction is OF.
                           "mem_total_gib": 116.1,
                           # No bond0 on the test host, so no TP-link reading: null, never 0.
                           "net": None,
                           # When THIS boot began (/proc/stat btime): what tells a rank that died with a
                           # reboot from one that exited.
                           "boot_at": BOOTED}


def test_status_reports_when_the_box_booted_or_null_when_it_cannot_tell(box_a):
    """A rank container that started BEFORE the box's current boot did not exit — the box rebooted under
    it (2026-09-19, 2026-10-10). `boot_at` is the box's own `btime`, so the LB can tell the two apart;
    a /proc/stat that cannot be read is null, never a guess."""
    assert reply(box_a.run("status"))["boot_at"] == BOOTED
    out = reply(box_a.run("status", extra_env={"TP_RANK_PROC_STAT": str(box_a.root / "no-such-stat")}))
    assert out["boot_at"] is None and out["rank"] == EMPTY_RANK, "and status still works"


def test_status_reports_what_ollama_still_holds_on_this_box(box_a):
    """L8 (worklog/vllm-cluster-hold-robustness.md §6). `freeing RAM; box-a: 77 GiB free, need 100` never
    said WHY the RAM was not coming back — and the answer, every time, was ollama holding 17.6 GiB of
    qwen3.8 because a chat job on that box reloaded it. The box's own `/api/ps` is the one thing that
    knows; nothing else in the status reply can tell."""
    ollama = FakeBackend(kind="ollama")
    ollama.ps = {"models": [{"name": "qwen3.8:latest", "model": "qwen3.8:latest",
                             "size": 18_882_431_548, "size_vram": 18_882_431_548,
                             "context_length": 262144, "expires_at": "2026-09-16T01:06:31.0Z"}]}
    try:
        out = reply(box_a.run("status", config={"CLUSTER_HOST_IP": "192.0.2.201",
                                                "CLUSTER_OLLAMA_URL": ollama.url}))
    finally:
        ollama.close()
    assert out["ollama"]["ps"]["models"][0]["name"] == "qwen3.8:latest"
    assert out["ollama"]["ps"]["models"][0]["size_vram"] == 18_882_431_548
    assert out["ollama"]["date"], "the SERVER's own clock, so a countdown is never read on ours"


def test_an_ollama_that_cannot_be_asked_is_null_never_an_empty_list(box_a):
    """A box that did not answer has NOT said "nothing is resident": the planner must be able to tell
    those apart, exactly as it does for every other observation."""
    closed = FakeBackend(kind="ollama")
    url = closed.url
    closed.close()                                   # nothing is listening there any more
    out = reply(box_a.run("status", config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_OLLAMA_URL": url}))
    assert out["ollama"] is None
    assert out["rank"] == EMPTY_RANK and out["mem_available_gib"] == 111.6, "and status still works"


def test_status_reports_a_running_rank_with_its_generation_and_no_exit_code(box_a):
    box_a.seed(gen=7)
    assert reply(box_a.run("status"))["rank"] == {"present": True, "running": True, "gen": 7,
                                                  "exit_code": None, "started_at": STARTED,
                                                  "model_choice": None, "exit_reason": ""}


def test_status_reports_an_exited_rank_with_its_exit_code(box_a):
    box_a.seed(gen=7, running=False, exit_code=137)
    # A rank that stopped says WHY where the box can tell: the memory guard's verdict, else the
    # newest fault line in the engine's log. The fake box logs neither, so the field is empty and the
    # planner's line is the one it has always been.
    assert reply(box_a.run("status"))["rank"] == {"present": True, "running": False, "gen": 7,
                                                  "exit_code": 137, "started_at": STARTED,
                                                  "model_choice": None, "exit_reason": ""}


def test_status_reports_the_model_choice_a_rank_was_launched_with(box_a):
    """What the rank was LAUNCHED for, off its own label — never what the LB wants now. A rank launched before
    model choices existed has no label, and says null."""
    box_a.seed(gen=7, model_choice=DENSE)
    assert reply(box_a.run("status"))["rank"]["model_choice"] == DENSE
    box_a.seed(gen=7, model_choice="Not A Choice")                 # not one of ours: no choice at all
    assert reply(box_a.run("status"))["rank"]["model_choice"] is None


def test_a_rank_container_without_our_generation_label_has_no_generation(box_a):
    """The rejected attempt launched this very name with no labels: not a rank of ANY generation."""
    box_a.seed(gen=None)
    rank = reply(box_a.run("status"))["rank"]
    assert rank["present"] is True and rank["running"] is True and rank["gen"] is None


def test_status_sees_the_brokers_classic_vllm_container(box_a):
    box_a.seed("vllm", gen=None)
    out = reply(box_a.run("status"))
    assert out["broker_vllm_running"] is True and out["rank"] == EMPTY_RANK
    box_a.seed("vllm", gen=None, running=False, exit_code=0)
    assert reply(box_a.run("status"))["broker_vllm_running"] is False


def test_a_docker_that_cannot_answer_is_a_failure_not_an_empty_box(box_a):
    """"No such object" means absent. Any OTHER failure means unobserved, and must say so: a dead
    daemon read as an empty box would let the planner unpark a GB10 box whose rank it cannot see."""
    box_a.set(daemon_down=True)
    for argv in (["status"], ["down"], ["up", "0", "7", "16", "262144"]):
        proc = box_a.run(*argv)
        assert proc.returncode == 1, (argv, proc.stdout, proc.stderr)
        out = reply(proc)
        assert out["ok"] is False and "Cannot connect to the Docker daemon" in out["error"], argv
    assert {c[0] for c in box_a.calls()} == {"inspect"}, "nothing is removed or started blind"


# ── up ────────────────────────────────────────────────────────────────────────

def test_the_mfu_patch_sizes_every_nvfp4_expert_method_the_pair_serves(box_a):
    """vLLM's MFU model drops the FFN of a quant method it has no byte size for: Qwen's `modelopt_mixed` and
    DeepSeek-V4's `deepseek_v4_fp8` (generation 237: only `unembed` instantiated, 14 GB/s beside 0.0 TFLOP/s)."""
    assert box_a.run("up", "0", "7", "16", "262144").returncode == 0
    table: dict = {}
    perf = types.SimpleNamespace(_QUANT_WEIGHT_BYTE_SIZE=table)
    mods = {"vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.metrics": types.SimpleNamespace(perf=perf), "vllm.v1.metrics.perf": perf}
    with mock.patch.dict(sys.modules, mods):
        exec((Path(mfu_of(box_a)) / "sitecustomize.py").read_text(), {})
    assert table == {"modelopt_mixed": 0.5625, "deepseek_v4_fp8": 0.5625}


def test_up_rank_0_launches_the_head_with_the_exact_flags(box_a):
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    runs = [c for c in box_a.calls() if c[0] == "run"]
    assert runs == [launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                cache_dir=cache_of(box_a.home))]
    run = runs[0]
    assert run[run.index("--node-rank") + 1] == "0" and "--headless" not in run
    assert SPEC in run, "the speculative config must reach docker as ONE argv element"
    network = run.index("--network")
    assert run[network:network + 4] == ["--network", "host", "--cap-add", "SYS_PTRACE"], \
        "NCCL's cross-process CUDA memory sharing needs SYS_PTRACE; the one healthy launch had it"
    assert run[run.index("--restart") + 1] == "no" and run[run.index("--name") + 1] == NAME
    assert reply(proc) == {"ok": True, "action": "started",
                           "container": hashlib.sha256(json.dumps(run).encode()).hexdigest()[:12]}
    assert box_a.containers()[NAME]["labels"] == {"lb.cluster.gen": "7", "lb.cluster.rank": "0"}


def test_an_image_that_declares_io_uring_launches_without_dockers_seccomp_filter(box_a):
    """b12x's weight loader reads through io_uring, which Docker >= 25's default seccomp profile refuses
    ("io_uring bounce initialization failed: Operation not permitted", generation 232). The image says so in its
    own label; every other image keeps the filter (the exact-flags test above)."""
    box_a.set(image_labels={"myllmbox/qwen38-flash-next-cluster-vllm:v5": {"lb.io_uring": "1"}})
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("--security-opt") + 1] == "seccomp=unconfined"
    assert run.index("--security-opt") < run.index("myllmbox/qwen38-flash-next-cluster-vllm:v5"), "a docker flag"


def test_up_rank_1_launches_the_worker_headless(box_a):
    proc = box_a.run("up", "1", "7", "16", "262144")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = [c for c in box_a.calls() if c[0] == "run"][-1]
    assert run == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=1, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                              cache_dir=cache_of(box_a.home))
    assert run[-1] == "--headless" and run[run.index("--node-rank") + 1] == "1"


def test_a_second_up_of_the_same_generation_leaves_the_running_rank_alone(box_a):
    assert reply(box_a.run("up", "0", "7", "16", "262144"))["action"] == "started"
    before = len(box_a.calls())
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0 and reply(proc) == {"ok": True, "action": "already-running"}
    assert [c[0] for c in box_a.calls()[before:]] == ["inspect"], "no rm, no second run"


def test_up_replaces_an_exited_rank_even_of_the_same_generation(box_a):
    box_a.seed(gen=7, running=False, exit_code=137)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0 and reply(proc)["action"] == "started"
    assert box_a.containers()[NAME]["running"] is True


def test_up_of_a_new_generation_removes_the_old_rank_before_starting(box_a):
    box_a.seed(gen=7)
    proc = box_a.run("up", "0", "8", "16", "262144")
    assert proc.returncode == 0 and reply(proc)["action"] == "started", (proc.stdout, proc.stderr)
    verbs = [c[0] for c in box_a.calls()]
    assert ["rm", "-f", NAME] in box_a.calls() and verbs.index("rm") < verbs.index("run")
    assert box_a.containers()[NAME]["labels"]["lb.cluster.gen"] == "8"


def test_a_failed_launch_is_reported_as_valid_json_in_dockers_own_words(box_a):
    message = ('docker: Error response from daemon: could not select device driver "" with '
               "capabilities: [[gpu]].\n\tSee 'docker run --help'. C:\\path \x1b[0m")
    box_a.set(run_error=message)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)           # quotes, a backslash, a newline, a tab and an ESC — still JSON
    assert out["ok"] is False and message in out["error"]


def test_up_without_the_boxes_host_ip_is_refused_before_docker(box_a):
    proc = box_a.run("up", "0", "7", "16", "262144", config={"CLUSTER_PORT": "8100"})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and "CLUSTER_HOST_IP" in out["error"]
    assert box_a.calls() == []


# ── the memory guard: armed while the pair loads, disarmed once the HEAD serves ─

#: The head's /health, where the PAIR says it serves: the address both ranks meet at (CLUSTER_MASTER_ADDR, which the
#: LB sends from the fleet file) and the port the head serves on (CLUSTER_PORT). Rank 1 runs headless — vLLM's
#: `--headless`, SGLang's node-rank 1 — and answers nothing on its own box.
HEAD_HEALTH = "http://192.0.2.201:8100/health"
#: box-b under the serving generation 193, read-only on 2026-10-11 (its /proc/meminfo): MemAvailable 29,953,076 kB,
#: SwapFree 58,430,860 kB.
LIVE_AVAILABLE_MIB, LIVE_SWAP_FREE_MIB = 29_251, 57_061
#: The fleet's ram_usable_fraction as the LB sends it to an armed guard: its floor is MemTotal x (1 - it) —
#: 0.93 of the fake box's 121,690,000 kB leaves 8,318 MiB, about the 8 GiB the guard used to hard-code.
GUARD_FRACTION = "0.93"
GUARD_FLOOR_MIB = int(121_690_000 * (1 - float(GUARD_FRACTION))) // 1024
#: The guard's one poll: our rank, its generation.
GUARD_POLL = ["inspect", "-f", '{{.State.Running}}|{{index .Config.Labels "lb.cluster.gen"}}', NAME]


def guard_default(var: str) -> int:
    """A threshold as tp-rank itself defaults it — read, never restated, so these tests move with the script."""
    return int(re.search(rf': "\${{{var}:=(\d+)}}"', TP_RANK.read_text()).group(1))


def wait_until(predicate, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def guard_polls(box_a: FakeBox) -> int:
    return sum(1 for call in box_a.calls() if call == GUARD_POLL)


def guard_running(box_a: FakeBox) -> bool:
    """Whether this box's memory guard is still alive: a process whose argv carries this box's guard log, which is
    one of the guard's own arguments and unique to the test's tmp dir. A guard that has exited reads as an empty
    cmdline even before anyone reaps it (a zombie under the container's pid 1), so it counts as gone. Waiting for
    THIS, not for a line in its log, is waiting for the guard to be done: its verdict, its `docker kill`, its exit."""
    mine = os.fsencode(box_a.guard_file)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:                         # it exited between the listing and the read
            continue
        if mine in argv:
            return True
    return False


@pytest.fixture
def guarded(box_a):
    """A box whose test arms the memory guard. Teardown takes the rank away — the one thing that ends a guard that
    never disarmed, since it exits when its rank is gone — so no test leaves one polling."""
    yield box_a
    state = box_a._state()
    state.get("containers", {}).pop(NAME, None)
    box_a._save(state)


def arm(box_a: FakeBox, *, rank: int, host_ip: str, master: str = "192.0.2.201",
        script: bytes | None = None) -> None:
    """Launch `rank` of generation 193 with the guard ARMED, as the LB sends it (master address and port), on the
    memory box-b had under generation 193 — and return once the guard has polled, so whatever the test does to
    the box next happens to a guard that is watching."""
    box_a.memory(available_mib=LIVE_AVAILABLE_MIB, swap_free_mib=LIVE_SWAP_FREE_MIB)
    proc = box_a.run("up", str(rank), "193", "6", "262144", script=script,
                     config={"CLUSTER_HOST_IP": host_ip, "CLUSTER_MASTER_ADDR": master, "CLUSTER_PORT": "8100",
                             "CLUSTER_GUARD": "1", "CLUSTER_RAM_USABLE_FRACTION": GUARD_FRACTION})
    assert proc.returncode == 0 and reply(proc)["action"] == "started", (proc.stdout, proc.stderr)
    assert wait_until(lambda: guard_polls(box_a) >= 1), f"the guard never polled; calls={box_a.calls()}"


def distress(box_a: FakeBox) -> None:
    """Where a bigger KV pin or more seats would take box-b under generation 193: MemAvailable a GiB under twice
    the floor — where the swap rule starts counting — and swapped half a GiB past the allowance since launch."""
    box_a.memory(available_mib=2 * GUARD_FLOOR_MIB - 1024,
                 swap_free_mib=LIVE_SWAP_FREE_MIB - guard_default("CLUSTER_GUARD_SWAP_GIB") * 1024 - 512)


@pytest.mark.parametrize("fraction", [None, "", "1", "0", "0.98x", "1.5"])
def test_an_armed_up_refuses_without_a_ram_usable_fraction(box_a, fraction):
    """THE GUARD'S FLOOR IS THE FLEET'S (operator, 2026-10-26: "0.98 of ram can be used, make it settable"): MemTotal x
    (1 - CLUSTER_RAM_USABLE_FRACTION), never a floor tp-rank makes up. Missing or not a fraction in (0, 1): the
    request was wrong — exit 2, nothing started, the variable named."""
    config = {"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_PORT": "8100",
              "CLUSTER_GUARD": "1"}
    if fraction is not None:
        config["CLUSTER_RAM_USABLE_FRACTION"] = fraction
    proc = box_a.run("up", "0", "193", "6", "262144", config=config)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "CLUSTER_RAM_USABLE_FRACTION" in proc.stdout + proc.stderr
    assert NAME not in box_a.containers()


def test_the_guard_kills_under_the_floor_the_fraction_leaves(guarded):
    """The floor is MemTotal x (1 - fraction), in the KILLED line in MiB — the fleet's figure, not 8 GiB."""
    box_a = guarded
    arm(box_a, rank=0, host_ip="192.0.2.201")
    box_a.memory(available_mib=GUARD_FLOOR_MIB - 256, swap_free_mib=LIVE_SWAP_FREE_MIB)
    assert wait_until(lambda: "KILLED" in box_a.guard_log()), box_a.guard_log()
    assert f"below the {GUARD_FLOOR_MIB} MiB floor" in box_a.guard_log()


def status_reason(box_a: FakeBox) -> str:
    return reply(box_a.run("status", config={"CLUSTER_HOST_IP": "192.0.2.202", "CLUSTER_OLLAMA_URL": ""}))[
        "rank"]["exit_reason"]


@pytest.mark.parametrize("rank, host_ip", [(1, "192.0.2.202"), (0, "192.0.2.201")], ids=["rank1", "rank0"])
def test_every_ranks_guard_disarms_when_the_heads_health_answers_200(guarded, rank, host_ip):
    """THE BUG, generation 193 (2026-10-11): the guard disarmed when ITS OWN box answered /health, and rank 1 runs
    headless, so box-b's guard was still armed under the SERVING pair. box-b had swapped past the 2 GiB allowance
    since launch, and only its ~29 GiB available kept the swap rule from firing: a bigger KV pin or more seats would
    have killed a pair in service. The pair serves when the HEAD answers, so every rank asks the head — and the
    head itself asks the same address, the one both ranks meet at."""
    box_a = guarded
    box_a.with_curl({HEAD_HEALTH: 200})
    arm(box_a, rank=rank, host_ip=host_ip)
    distress(box_a)
    assert wait_until(lambda: "DISARMED" in box_a.guard_log() or "KILLED" in box_a.guard_log()), (
        f"the guard is still armed under a serving head: log={box_a.guard_log()!r} asked={box_a.curl_calls()}")
    polls = guard_polls(box_a)
    time.sleep(2.5)                                     # one more poll interval: a disarmed guard has stopped
    log = box_a.guard_log()
    assert "KILLED" not in log and box_a.containers()[NAME]["running"] is True, log
    assert [call for call in box_a.calls() if call[0] == "kill"] == []
    assert guard_polls(box_a) == polls, "disarmed means it stopped watching"
    assert {call[-1] for call in box_a.curl_calls()} == {HEAD_HEALTH}, "the head, and only the head, is asked"
    armed, disarmed = log.splitlines()
    assert "ARMED for generation 193" in armed and HEAD_HEALTH in armed, armed
    assert "DISARMED generation 193" in disarmed and HEAD_HEALTH in disarmed and "200" in disarmed, disarmed
    # The guard's log says when it disarmed, and that is not why a rank later stops: the engine's own fault is.
    box_a.seed(gen=193, running=False, exit_code=1,
               logs=["INFO serving", "RuntimeError: NCCL communicator was aborted on rank 1."])
    assert status_reason(box_a) == "RuntimeError: NCCL communicator was aborted on rank 1."


@pytest.mark.parametrize("head", [{}, {HEAD_HEALTH: 503}], ids=["head-refuses", "head-503"])
def test_until_the_head_answers_200_the_guard_stays_armed_and_kills_a_load_eating_the_box(guarded, head):
    """The window the guard exists for — weights streaming in, nothing serving yet — stays guarded on the headless
    rank: a head that is not up, or answers anything but 200, disarms nothing, and the swap rule still fires.

    Each step waits on what the guard DID, never on a span of time. A second poll on a healthy box means the first
    one asked the head, heard no 200 and went back to watching. The guard's process gone means its verdict, its
    `docker kill` and its exit are all behind it — tp-rank logs KILLED BEFORE it kills, so the log line alone
    (what this test waited for until 2026-10-12) raced the kill: under host load ~13 the fake docker had not yet
    logged `kill` when the test looked, 3 runs in 5."""
    box_a = guarded
    box_a.with_curl(head)
    arm(box_a, rank=1, host_ip="192.0.2.202")
    assert wait_until(lambda: guard_polls(box_a) >= 2), (
        f"the guard stopped watching after it asked the head: log={box_a.guard_log()!r} asked={box_a.curl_calls()}")
    assert guard_running(box_a) and "DISARMED" not in box_a.guard_log(), f"log={box_a.guard_log()!r}"
    assert {call[-1] for call in box_a.curl_calls()} == {HEAD_HEALTH}, "a full poll asked the head and stayed armed"
    distress(box_a)
    assert wait_until(lambda: not guard_running(box_a)), f"the guard never acted: log={box_a.guard_log()!r}"
    assert "KILLED" in box_a.guard_log() and "DISARMED" not in box_a.guard_log(), f"log={box_a.guard_log()!r}"
    assert ["kill", "-s", "SIGKILL", NAME] in box_a.calls()
    assert box_a.containers()[NAME]["running"] is False and box_a.containers()[NAME]["exit_code"] == 137
    killed = box_a.guard_log().splitlines()[-1]
    assert f"KILLED generation 193: the box swapped 2560 MiB with only {2 * GUARD_FLOOR_MIB - 1024} MiB available" in killed, killed
    assert {call[-1] for call in box_a.curl_calls()} == {HEAD_HEALTH}
    assert status_reason(box_a) == killed, "the guard's verdict is why the rank stopped"


def test_up_without_the_interface_the_ranks_talk_over_refuses_naming_it(box_a):
    """CLUSTER_IFACE is the fleet file's `iface`, sent by the LB: a script that defaulted it to one fleet's bond would
    put another deployment's NCCL on a NIC it does not have. None sent: exit 1, the variable named, nothing started."""
    assert ': "${CLUSTER_IFACE:=}"' in TP_RANK.read_text(), "the NIC has no default"
    proc = box_a.run("up", "0", "7", "16", "262144", extra_env={"CLUSTER_IFACE": ""})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and "CLUSTER_IFACE is not set" in out["error"]
    assert box_a.calls() == [], "nothing launched"


def test_a_guard_that_does_not_know_the_heads_address_says_so_asks_nobody_and_stays_armed(guarded):
    """No guessing. The LB always sends CLUSTER_MASTER_ADDR (the fleet file's head interconnect) and tp-rank has no
    default of its own (the open-source rule, AGENTS.md: a script that named one address described one fleet). A guard
    with no head to ask must not ask its own box or loopback — a headless rank answers nothing there — but say in its
    log that it cannot disarm, and keep guarding."""
    box_a = guarded
    assert ': "${CLUSTER_MASTER_ADDR:=}"' in TP_RANK.read_text(), "the head's address has no default"
    box_a.with_curl({HEAD_HEALTH: 200, "http://127.0.0.1:8100/health": 200, "http://192.0.2.202:8100/health": 200})
    arm(box_a, rank=1, host_ip="192.0.2.202", master="")
    assert wait_until(lambda: guard_polls(box_a) >= 2)
    log = box_a.guard_log()
    assert "the head's address is not known (CLUSTER_MASTER_ADDR is empty)" in log and "DISARMED" not in log, log
    assert box_a.curl_calls() == [], "nothing is asked in the head's place"
    distress(box_a)
    assert wait_until(lambda: "KILLED" in box_a.guard_log()), f"log={box_a.guard_log()!r}"


# ── where the checkpoint is ───────────────────────────────────────────────────

SEARCH = ("nvme", "nfs", "hdd")     # box-a's own copy · box-a's home over pair-link · box-c's HDD


def search_path(box_a: FakeBox) -> tuple[dict[str, Path], dict[str, str]]:
    """Three candidate dirs under the fake box, and the config that searches them in that order."""
    dirs = {name: box_a.root / name for name in SEARCH}
    return dirs, {"CLUSTER_HOST_IP": "192.0.2.201",
                  "CLUSTER_MODELS_DIRS": ":".join(str(dirs[name]) for name in SEARCH)}


@pytest.mark.parametrize("whole, partial, mounted", [
    ({"nvme"}, set(), "nvme"),
    ({"nfs"}, set(), "nfs"),
    ({"nvme", "hdd"}, set(), "nvme"),
    ({"hdd"}, {"nvme"}, "hdd"),
], ids=["only-in-the-1st", "only-in-the-2nd", "the-first-of-several", "past-a-partial-copy"])
def test_up_mounts_the_first_dir_on_the_search_path_that_holds_the_checkpoint(box_a, whole, partial,
                                                                              mounted):
    """A copy lands as `<name>.partial` and is renamed only once complete, so a dir counts only when
    `<name>/config.json` is in it."""
    dirs, config = search_path(box_a)
    for name in whole:
        box_a.place_checkpoint(dirs[name])
    for name in partial:
        box_a.place_checkpoint(dirs[name], partial=True)
    proc = box_a.run("up", "0", "7", "16", "262144", config=config)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = [c for c in box_a.calls() if c[0] == "run"][-1]
    assert run[run.index("-v") + 1] == f"{dirs[mounted]}:/models:ro"
    assert run[run.index("--model") + 1] == f"/models/{MODEL}"


def test_up_with_no_checkpoint_on_the_search_path_names_every_dir_and_changes_nothing(box_a):
    dirs, config = search_path(box_a)
    box_a.place_checkpoint(dirs["nvme"], partial=True)       # still copying: not a checkpoint yet
    box_a.seed(gen=6)                                        # an older rank, which must survive
    proc = box_a.run("up", "0", "7", "16", "262144", config=config)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    error = reply(proc)["error"]
    assert MODEL in error and all(str(d) in error for d in dirs.values()), error
    assert [c[0] for c in box_a.calls()] == ["inspect"], "no rm and no run: nothing was touched"
    assert box_a.containers()[NAME]["labels"]["lb.cluster.gen"] == "6"


def test_with_no_search_path_a_start_fails_naming_it_and_touches_nothing(box_a):
    """No default: a deployment's model folders are its own (the LB sends CLUSTER_MODELS_DIRS, built from MODELS_SEARCH_DIRS
    in lb/.env), and a guessed one is how a start loads a stale copy from the wrong disk."""
    proc = box_a.run("up", "0", "7", "16", "262144", extra_env={"CLUSTER_MODELS_DIRS": ""})   # none sent
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "CLUSTER_MODELS_DIRS" in reply(proc)["error"] and "MODELS_SEARCH_DIRS" in reply(proc)["error"]
    assert not [c for c in box_a.calls() if c[0] == "run"]


# ── the profile key's caches (LS1) and the PLE mode (LS3) ────────────────────
#
# The LaunchProfile is a model's launch SETTINGS (vllm_lb/catalog.py); the PROFILE KEY is the identity
# derived from them — image, model, TP size, PLE mode — under which this box keeps vLLM's caches.

def last_run(box_a: FakeBox) -> list[str]:
    runs = [c for c in box_a.calls() if c[0] == "run"]
    assert runs, "docker run was never reached"
    return runs[-1]


def mounted_cache(run: list[str]) -> str | None:
    """The host dir the launch mounts at /cache, or None."""
    mounts = [v for flag, v in zip(run, run[1:]) if flag == "-v" and v.endswith(":/cache")]
    assert len(mounts) <= 1, mounts
    return mounts[0].removesuffix(":/cache") if mounts else None


def test_up_mounts_the_profile_keys_own_cache_dir_with_vllm_and_flashinfer_caching_in_it(box_a):
    """One host dir per profile key under the box's own home, made by `up` itself — the SSH user's,
    not a dir docker would create as root — mounted at /cache, with VLLM_CACHE_ROOT (compile/AOT,
    Triton, model-info) and FLASHINFER_WORKSPACE_BASE inside it. FlashInfer's AUTOTUNE results are the
    one cache that is NOT kept there: see the test below."""
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run, cache = last_run(box_a), cache_of(box_a.home)
    assert run == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models, cache_dir=cache)
    at = run.index(f"{cache}:/cache")
    assert run[at + 1:at + 5] == ["-e", "VLLM_CACHE_ROOT=/cache/vllm",
                                  "-e", "FLASHINFER_WORKSPACE_BASE=/cache/flashinfer"]
    assert cache.is_dir()
    assert cache.name == f"tp{run[run.index('--tensor-parallel-size') + 1]}-ple1", \
        "the TP size in the key is the TP size launched"


def env_of(run: list[str], name: str) -> list[str]:
    """Every value the launch gives the environment variable `name`."""
    return [v.partition("=")[2] for flag, v in zip(run, run[1:]) if flag == "-e" and v.startswith(name + "=")]


def test_the_flashinfer_autotune_cache_never_outlives_the_start_that_made_it(box_a):
    """THE HANG, 2026-09-18 (tests/fixtures/qwen38-start-2026-09-18-hung-rank{0,1}.log). Three starts in a
    row sat in "autotuning FlashInfer kernels" for 53 minutes and then failed, 62.6-63.0 minutes after
    launch each time; the one start that had worked (2026-09-16 01:06, 11 m 49 s) was the one that CREATED
    the persistent cache.

    vLLM 0.29.0 lets only rank 0 save `autotune_configs.json`, and FlashInfer keys every entry by the
    runner's tp_rank: all 42 entries in the file both GB10 boxes held said tp_rank=0. On the next start rank 0
    read it, broadcast it, and hit all 42 ("Config cache hit … source=config file"); rank 1 loaded the same
    bytes, hit NONE, and profiled alone — and the profiler all-reduces every tactic's timing over the
    ranks' gloo group, which the peer that skipped profiling never joins. The first profile took 30:09,
    gloo's 30-minute timeout. FlashInfer states the contract itself (autotuner.py,
    set_autotune_process_group): "every rank must enter _profile_single_kernel the same number of times …
    identical (ideally empty) starting caches".

    So the autotune results live on a tmpfs of the container's own: empty at every start, on both ranks,
    whatever was tuned before — a fresh tune is 35 s (worklog/vllm-cluster-load-speed.md 9f), a poisoned one
    was an hour and both boxes. The compile/AOT, Triton and JIT caches stay persistent: they are per
    box and per rank, never negotiated between the ranks."""
    for rank in ("0", "1"):
        proc = box_a.run("up", rank, "7", "16", "262144")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        run = last_run(box_a)
        assert env_of(run, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR") == [AUTOTUNE_DIR], \
            f"rank {rank} would read a FlashInfer autotune cache an earlier start left behind"
        tmpfs = [v for flag, v in zip(run, run[1:]) if flag == "--tmpfs"]
        assert tmpfs == [AUTOTUNE_TMPFS], f"rank {rank}: the autotune dir must die with the container: {tmpfs}"
        mounts = [v for flag, v in zip(run, run[1:]) if flag == "-v"]
        assert not [m for m in mounts if m.split(":")[1].startswith(AUTOTUNE_DIR)], \
            f"rank {rank} keeps its autotune results on the host: {mounts}"
        assert not AUTOTUNE_DIR.startswith("/cache"), "inside the persistent mount it would outlive the start"
        # …and everything that IS safe to keep is still kept.
        assert mounted_cache(run) == str(cache_of(box_a.home))
        assert env_of(run, "VLLM_CACHE_ROOT") == ["/cache/vllm"]
        assert env_of(run, "FLASHINFER_WORKSPACE_BASE") == ["/cache/flashinfer"]
        box_a.run("down")


#: A profile key per entry: the defaults, then ONE component changed at a time.
PROFILE_CHANGES = {
    "defaults": {},
    "image-tag": {"CLUSTER_IMAGE": "myllmbox/qwen38-flash-next-cluster-vllm:v6"},
    "image-digest": {"CLUSTER_IMAGE": "myllmbox/qwen38-flash-next-cluster-vllm@sha256:" + "0f" * 32},
    # an image NAMED like the default image's dir: the mapping must be reversible, not merely mangled
    "image-look-alike": {"CLUSTER_IMAGE": IMAGE_DIR},
    "model": {"CLUSTER_MODEL_NAME": "Qwen3.8-Flash-Next-hibrid48-v2"},
    "ple": {"CLUSTER_PLE_REPLICATE": "0"},
}


def test_each_profile_key_component_changes_the_cache_dir_and_no_two_keys_share_or_nest(box_a):
    """What vLLM's own hash cannot see is the key: the image, the model, the TP size, the PLE mode."""
    FakeBox.place_checkpoint(box_a.models, name="Qwen3.8-Flash-Next-hibrid48-v2")
    dirs = {}
    for label, change in PROFILE_CHANGES.items():
        assert reply(box_a.run("down"))["ok"] is True
        proc = box_a.run("up", "0", "7", "16", "262144",
                         config={"CLUSTER_HOST_IP": "192.0.2.201", **change})
        assert proc.returncode == 0, (label, proc.stdout, proc.stderr)
        dirs[label] = mounted_cache(last_run(box_a))
        assert dirs[label] and Path(dirs[label]).is_dir(), label
    assert dirs["defaults"] == str(cache_of(box_a.home))
    assert dirs["image-look-alike"] == str(
        cache_of(box_a.home, image_dir="myllmbox__Sqwen38-flash-next-cluster-vllm__Cv5"))
    assert dirs["model"] == str(cache_of(box_a.home, model="Qwen3.8-Flash-Next-hibrid48-v2"))
    assert dirs["ple"] == str(cache_of(box_a.home, ple="0"))
    labels = list(dirs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            x, y = dirs[a], dirs[b]
            assert x != y and not x.startswith(y + "/") and not y.startswith(x + "/"), (a, b)


@pytest.mark.parametrize("ple", ["0", "1"])
def test_the_ple_mode_reaches_docker_as_MBX_PLE_REPLICATE_and_keys_the_cache(box_a, ple):
    proc = box_a.run("up", "1", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.202", "CLUSTER_PLE_REPLICATE": ple})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=1, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home, ple=ple),
                                          host_ip="192.0.2.202", ple=ple)


def test_the_ple_mode_defaults_to_todays_whole_table_on_each_rank(box_a):
    """The split table is adopted only on a measured A/B, and not by this script's default."""
    assert box_a.run("up", "0", "7", "16", "262144").returncode == 0
    run = last_run(box_a)
    assert "MBX_PLE_REPLICATE=1" in run and not [e for e in run if e.startswith("MBX_PLE_REPLICATE=0")]


def test_an_empty_cache_dir_launches_with_no_cache_and_leaves_nothing_on_the_box(box_a):
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_CACHE_DIR": ""})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=None)
    assert not (box_a.home / ".cache").exists()


def test_the_off_switches_give_back_the_launch_from_before_the_load_speed_pass(box_a):
    """K3's revert without a rebuild: CLUSTER_CACHE_DIR= and CLUSTER_MTP_PREFIX= launch argv for argv as
    before this pass — with an index a view could have been built from — and leave nothing on disk."""
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_CACHE_DIR": "",
                             "CLUSTER_MTP_PREFIX": ""})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=None, ple="1", spec=SPEC)
    assert not (box_a.home / ".cache").exists() and not list(box_a.models.glob(f"{MODEL}-mtp*"))


#: Launch settings `up` must refuse, by name, before docker runs at all. {models} is the box's own
#: models dir, first on its search path.
BAD_SETTINGS = [
    ("CLUSTER_MAX_NUM_BATCHED_TOKENS", "1024"),
    ("CLUSTER_MAX_NUM_BATCHED_TOKENS", "131072"),
    ("CLUSTER_MAX_NUM_BATCHED_TOKENS", "16k"),
    ("CLUSTER_MAX_NUM_BATCHED_TOKENS", "8192; rm -rf /"),
    ("CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD", "-1"),
    ("CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD", "65537"),
    ("CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD", "2k"),
    ("CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD", "02048"),
    ("CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD", "2048 --enforce-eager"),
    ("CLUSTER_MODEL_NAME", "../Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_MODEL_NAME", ".Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_MODEL_NAME", "Qwen 3.8"),
    # a path under a models root is a model name too (pass 4's catalog), but only a plain one
    ("CLUSTER_MODEL_NAME", "/home/user/models/Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_MODEL_NAME", "qwen38-flash-next-cluster/../Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_MODEL_NAME", "qwen38-flash-next-cluster//Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_MODEL_NAME", "qwen38-flash-next-cluster/"),
    ("CLUSTER_MODEL_NAME", "qwen38-flash-next-cluster/.Qwen3.8-Flash-Next-hibrid48"),
    ("CLUSTER_IMAGE", "myllmbox/qwen38 v5"),
    ("CLUSTER_IMAGE", "myllmbox/qwen38;reboot"),
    ("CLUSTER_IMAGE", "--privileged"),
    ("CLUSTER_IMAGE", "a" * 256),
    ("CLUSTER_PLE_REPLICATE", "2"),
    ("CLUSTER_PLE_REPLICATE", "01"),
    ("CLUSTER_PLE_REPLICATE", "true"),
    ("CLUSTER_MTP_PREFIX", "mtp/"),
    ("CLUSTER_MTP_PREFIX", "mtp.*"),
    # pass 4: a LaunchProfile's parsers and speculative config are flags `up` makes, so they are checked too
    ("CLUSTER_TOOL_CALL_PARSER", "qwen3 xml"),
    ("CLUSTER_TOOL_CALL_PARSER", "--privileged"),
    ("CLUSTER_REASONING_PARSER", "qwen3;reboot"),
    ("CLUSTER_SPECULATIVE_CONFIG", "mtp"),
    ("CLUSTER_SPECULATIVE_CONFIG", '{"method":{"name":"mtp"}}'),
    # the drafter's own model is tp-rank's to name (the drafter view), never the profile's
    ("CLUSTER_SPECULATIVE_CONFIG", '{"method":"mtp","model":"/models/elsewhere"}'),
    ("CLUSTER_CACHE_DIR", "relative/cache"),
    ("CLUSTER_CACHE_DIR", "/home/user/../../etc"),
    ("CLUSTER_CACHE_DIR", "/home/user/./cache"),
    ("CLUSTER_CACHE_DIR", "/cache:/etc"),
    ("CLUSTER_CACHE_DIR", "/cache dir"),
    ("CLUSTER_CACHE_DIR", "/cache/"),
    ("CLUSTER_CACHE_DIR", "{models}"),
    ("CLUSTER_CACHE_DIR", "{models}/.cache"),
    # box-b's hazard: box-a's home, over NFS, is on box-b's search path
    ("CLUSTER_CACHE_DIR", "/mnt/box-a-home/models/qwen38-flash-next-cluster/cache"),
]


@pytest.mark.parametrize("key, value", BAD_SETTINGS, ids=repr)
def test_a_bad_launch_setting_refuses_up_by_name_before_docker_runs(box_a, key, value):
    # A literal substitution, not str.format: a JSON setting's own braces are not placeholders.
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201",
                             key: value.replace("{models}", str(box_a.models))})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and key in out["error"], out
    assert box_a.calls() == [], "a launch that names a bad path or flag must never reach docker"


def test_a_cache_dir_that_cannot_be_made_refuses_up_and_changes_nothing(box_a):
    blocker = box_a.root / "a-file"
    blocker.write_text("not a dir\n")
    box_a.seed(gen=6)
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_CACHE_DIR": f"{blocker}/cache"})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "cache dir" in reply(proc)["error"]
    assert [c[0] for c in box_a.calls()] == ["inspect"], "no rm and no run: nothing was touched"
    assert box_a.containers()[NAME]["labels"]["lb.cluster.gen"] == "6"


# ── the drafter view (LS2) ────────────────────────────────────────────────────

def view_of(models_dir: Path) -> Path:
    return models_dir / f"{MODEL}-mtp"


def assert_whole_view(models_dir: Path) -> None:
    """The checkpoint's own files and exactly the drafter's shards, each a RELATIVE link resolving
    inside the checkpoint — readable by a squashed root — and no builder's partial left beside it."""
    view, checkpoint = view_of(models_dir), models_dir / MODEL
    assert sorted(p.name for p in view.iterdir()) == sorted(REAL_FILES + DRAFTER_SHARDS)
    for entry in view.iterdir():
        assert entry.is_symlink() and os.readlink(entry) == f"../{MODEL}/{entry.name}", entry
        assert entry.resolve().parent == checkpoint.resolve() and entry.resolve().is_file(), entry
    assert view.stat().st_mode & 0o777 == 0o755, "box-b's container reads it as root over NFS"
    assert not list(models_dir.glob("*.partial*"))


def test_up_points_the_drafter_at_a_view_of_only_the_shards_the_index_names_for_it(box_a):
    """Today's layout yields research §4 B's four shards: the two holding `mtp.` tensors, and the
    embedding's and lm_head's, which the drafter loads before taking the target's — never an expert
    shard, the vision tower or the PLE table, look-alike names and all."""
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert_whole_view(box_a.models)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home), spec=VIEW_SPEC)


def test_the_drafter_view_is_built_once_and_then_used_as_it_is(box_a):
    lay_out_checkpoint(box_a.models / MODEL)
    assert box_a.run("up", "0", "7", "16", "262144").returncode == 0
    view = view_of(box_a.models)
    built = (view.stat().st_ino, {p.name: p.lstat().st_ino for p in view.iterdir()})
    for gen in ("8", "9"):
        proc = box_a.run("up", "0", gen, "16", "262144")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert (view.stat().st_ino, {p.name: p.lstat().st_ino for p in view.iterdir()}) == built, \
            "the view was rebuilt"
        run = last_run(box_a)
        assert run[run.index("--speculative-config") + 1] == VIEW_SPEC
    assert_whole_view(box_a.models)


def two_boxes_on_one_export(tmp_path: Path) -> tuple[FakeBox, FakeBox, Path, dict[str, str]]:
    """box-a and box-b — each its own docker and home — searching ONE models dir, as box-b reads
    box-a's over pair-link."""
    boxes = []
    for name in ("box-a", "box-b"):
        (tmp_path / name).mkdir()
        boxes.append(FakeBox(tmp_path / name))
    export = tmp_path / "export" / "models"
    lay_out_checkpoint(export / MODEL)
    return boxes[0], boxes[1], export, {"CLUSTER_MODELS_DIRS": str(export)}


def wait_for(path: Path, proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert proc.poll() is None, f"the held rank ended before {path.name}: {proc.communicate()}"
        assert time.monotonic() < deadline, f"no {path.name} within {timeout_s} s"
        time.sleep(0.02)


def test_two_ranks_building_the_view_at_once_leave_one_whole_view_and_both_launch_with_it(tmp_path):
    """box-a's rank is held just before its rename until box-b's rank has built the view and
    launched. box-a therefore loses the rename, for certain: it must remove its own partial and take
    box-b's view."""
    box_a, box_b, export, search = two_boxes_on_one_export(tmp_path)
    hold = tmp_path / "hold"
    (hold / "bin").mkdir(parents=True)
    _install(hold / "bin" / "mv", f'#!/bin/sh\n: > "{hold}/waiting"\n'
                                  f'while [ ! -e "{hold}/go" ]; do sleep 0.02; done\n'
                                  f'exec {shutil.which("mv")} "$@"\n')
    held = box_a.start("up", "0", "7", "16", "262144",
                       config={"CLUSTER_HOST_IP": "192.0.2.201", **search},
                       extra_env={"PATH": f"{hold / 'bin'}:{box_a.env()['PATH']}"})
    wait_for(hold / "waiting", held)
    assert len(list(export.glob(f"{MODEL}-mtp.partial.*"))) == 1 and not view_of(export).exists()
    other = box_b.run("up", "1", "7", "16", "262144",
                       config={"CLUSTER_HOST_IP": "192.0.2.202", **search})
    assert other.returncode == 0, (other.stdout, other.stderr)
    (hold / "go").touch()
    first = FakeBox.finish(held)
    assert first.returncode == 0, (first.stdout, first.stderr)
    assert_whole_view(export)
    for box in (box_a, box_b):
        run = last_run(box)
        assert run[run.index("--speculative-config") + 1] == VIEW_SPEC, box.root.name


def test_ranks_racing_freely_to_build_the_view_always_leave_one_whole_view(tmp_path):
    box_a, box_b, export, search = two_boxes_on_one_export(tmp_path)
    for gen in range(1, 7):
        shutil.rmtree(view_of(export), ignore_errors=True)
        procs = [box.start("up", str(rank), str(gen), "16", "262144",
                           config={"CLUSTER_HOST_IP": ip, **search})
                 for rank, (box, ip) in enumerate([(box_a, "192.0.2.201"), (box_b, "192.0.2.202")])]
        for proc in procs:
            done = FakeBox.finish(proc)
            assert done.returncode == 0, (gen, done.stdout, done.stderr)
        assert_whole_view(export)
        for box in (box_a, box_b):
            run = last_run(box)
            assert run[run.index("--speculative-config") + 1] == VIEW_SPEC, (gen, box.root.name)


#: The layout with no `mtp.` tensor in its index: nothing for a view to hold.
NO_DRAFTER = {shard: [t for t in tensors if not t.startswith("mtp.")] for shard, tensors in REAL_SHARDS.items()}


@pytest.mark.parametrize("case", ["no-index", "no-drafter-in-the-index", "prefix-switched-off"])
def test_with_nothing_to_build_a_view_from_the_drafter_walks_the_checkpoint_as_today(box_a, case):
    lay_out_checkpoint(box_a.models / MODEL, index=case != "no-index",
                       shards=NO_DRAFTER if case == "no-drafter-in-the-index" else None)
    config = {"CLUSTER_HOST_IP": "192.0.2.201",
              **({"CLUSTER_MTP_PREFIX": ""} if case == "prefix-switched-off" else {})}
    proc = box_a.run("up", "0", "7", "16", "262144", config=config)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home), spec=SPEC)
    assert not list(box_a.models.glob(f"{MODEL}-mtp*"))


def test_a_checkpoint_without_a_shard_its_index_names_for_the_drafter_refuses_up_by_name(box_a):
    checkpoint = lay_out_checkpoint(box_a.models / MODEL)
    (checkpoint / "model-00019-of-00021.safetensors").unlink()
    box_a.seed(gen=6)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    error = reply(proc)["error"]
    assert "drafter view" in error and "model-00019-of-00021.safetensors" in error, error
    assert [c[0] for c in box_a.calls()] == ["inspect"], "no rm and no run: nothing was touched"
    assert box_a.containers()[NAME]["labels"]["lb.cluster.gen"] == "6"
    assert not list(box_a.models.glob(f"{MODEL}-mtp*"))


#: An index that must not be trusted, and the words that say why.
BAD_INDEXES = {
    "cut-short": ('{"metadata": {}, "weight_map": ', "JSONDecodeError"),
    "no-weight-map": ('{"metadata": {"total_size": 1}}', "KeyError"),
    "a-shard-outside-the-checkpoint": (
        json.dumps({"weight_map": {"mtp.fc.weight": "../../etc/model.safetensors"}}), "not a file name"),
}


@pytest.mark.parametrize("case", list(BAD_INDEXES))
def test_an_index_that_cannot_be_trusted_refuses_up_by_name(box_a, case):
    text, why = BAD_INDEXES[case]
    (lay_out_checkpoint(box_a.models / MODEL) / INDEX).write_text(text)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    error = reply(proc)["error"]
    assert "drafter view" in error and why in error, error
    assert [c[0] for c in box_a.calls()] == ["inspect"]
    assert not list(box_a.models.glob(f"{MODEL}-mtp*"))


def view_entries(view: Path) -> dict[str, str]:
    if view.is_symlink():
        return {".": os.readlink(view)}
    return {p.name: os.readlink(p) if p.is_symlink() else "" for p in view.iterdir()}


@pytest.mark.parametrize("case", ["links-a-shard-the-index-does-not-name", "lacks-the-shards",
                                  "a-link-to-the-checkpoint"])
def test_something_else_in_the_views_place_refuses_up_and_is_left_alone(box_a, case):
    """Not built by this index — stale, half made by hand, or a link: never mistaken for the view,
    never replaced behind an operator's back."""
    lay_out_checkpoint(box_a.models / MODEL)
    view = view_of(box_a.models)
    if case == "a-link-to-the-checkpoint":
        view.symlink_to(MODEL)
    else:
        view.mkdir()
        (view / "config.json").symlink_to(f"../{MODEL}/config.json")
        if case == "links-a-shard-the-index-does-not-name":
            for shard in DRAFTER_SHARDS + ["model-00002-of-00021.safetensors"]:
                (view / shard).symlink_to(f"../{MODEL}/{shard}")
    before = view_entries(view)
    box_a.seed(gen=6)
    proc = box_a.run("up", "0", "7", "16", "262144")
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    error = reply(proc)["error"]
    assert "drafter view" in error and str(view) in error, error
    assert [c[0] for c in box_a.calls()] == ["inspect"]
    assert view_entries(view) == before and not list(box_a.models.glob("*.partial*"))


# ── a model named by its path under a models root (pass 4's catalog) ─────────

#: How pass 4's catalog names today's model: by its path under a models root
#: (worklog/vllm-model-selection.md, the catalog's `path`).
MODEL_PATH = f"qwen38-flash-next-cluster/{MODEL}"


def test_a_model_named_by_its_path_under_a_root_launches_from_the_root_with_its_view_beside_it(box_a):
    """The same launch, one level up: the root mounted, the path under /models, the drafter view beside
    the checkpoint linking into it by the checkpoint's own dir name, and the path as ONE name in the key."""
    root = box_a.home / "models"
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MODELS_DIRS": str(root),
                             "CLUSTER_MODEL_NAME": MODEL_PATH})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("-v") + 1] == f"{root}:/models:ro"
    assert run[run.index("--model") + 1] == f"/models/{MODEL_PATH}"
    assert run[run.index("--speculative-config") + 1] == (
        '{"method":"mtp","num_speculative_tokens":4,"model":"/models/' + MODEL_PATH + '-mtp"}')
    assert mounted_cache(run) == str(cache_of(box_a.home, model=f"qwen38-flash-next-cluster_S{MODEL}"))
    assert_whole_view(box_a.models)


def test_a_model_path_and_a_name_spelled_like_its_key_never_share_a_cache_dir(box_a):
    """A model path enters the key as one name, by the image's reversible mapping — so a model whose own
    name reads like that encoding still gets a dir of its own."""
    root = box_a.home / "models"
    look_alike = f"qwen38-flash-next-cluster_S{MODEL}"
    FakeBox.place_checkpoint(root, name=look_alike)
    dirs = {}
    for model in (MODEL_PATH, look_alike):
        assert reply(box_a.run("down"))["ok"] is True
        proc = box_a.run("up", "0", "7", "16", "262144",
                         config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MODELS_DIRS": str(root),
                                 "CLUSTER_MODEL_NAME": model})
        assert proc.returncode == 0, (model, proc.stdout, proc.stderr)
        dirs[model] = mounted_cache(last_run(box_a))
    assert dirs == {MODEL_PATH: str(cache_of(box_a.home, model=f"qwen38-flash-next-cluster_S{MODEL}")),
                    look_alike: str(cache_of(box_a.home, model=f"qwen38-flash-next-cluster__S{MODEL}"))}


# ── the ModelChoice and its LaunchProfile (pass 4) ───────────────────────────

def dense_env(box_a: FakeBox, **over) -> dict[str, str]:
    """A LaunchProfile for a model WITHOUT a drafter, as `cluster_exec.launch_env` would send it: found by its
    own dir name under the models ROOTS, served under its own name, with no speculative decoding. The 27B's
    bf16 build was such a profile; the catalog's dense model today (NVIDIA's NVFP4 build) carries its MTP head."""
    return {"CLUSTER_HOST_IP": "192.0.2.201",
            "CLUSTER_IMAGE": "myllmbox/qwen38-flash-next-cluster-vllm:v5",
            "CLUSTER_MODELS_DIRS": "~/models:/mnt/box-a-home/models:/srv/models",
            "CLUSTER_MODEL_NAME": "Qwen3.8-27B", "CLUSTER_SERVED_MODEL_NAME": "Qwen/Qwen3.8-27B",
            "CLUSTER_GPU_MEMORY_UTILIZATION": "0.70", "CLUSTER_KV_CACHE_MEMORY": "28000000000",
            "CLUSTER_TOOL_CALL_PARSER": "qwen3_xml", "CLUSTER_REASONING_PARSER": "qwen3",
            "CLUSTER_SPECULATIVE_CONFIG": "", "CLUSTER_MTP_PREFIX": "", **over}


def test_up_labels_the_rank_with_the_model_choice_it_was_given(box_a):
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16, ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home), model_choice=FLASH)
    assert box_a.containers()[NAME]["labels"] == {"lb.cluster.gen": "7", "lb.cluster.rank": "0",
                                                  "lb.cluster.model_choice": FLASH}
    assert reply(box_a.run("status"))["rank"]["model_choice"] == FLASH


def test_a_rank_of_the_same_generation_but_another_model_choice_is_replaced(box_a):
    """A generation alone cannot tell two models apart: a rank launched for another ModelChoice is stale."""
    assert reply(box_a.run("up", "0", "7", "16", "262144", FLASH))["action"] == "started"
    before = len(box_a.calls())
    assert reply(box_a.run("up", "0", "7", "16", "262144", FLASH)) == {"ok": True, "action": "already-running"}
    assert [c[0] for c in box_a.calls()[before:]] == ["inspect"], "the same choice again touches nothing"
    FakeBox.place_checkpoint(box_a.home / "models", name="Qwen3.8-27B")
    proc = box_a.run("up", "0", "7", "16", "262144", DENSE, config=dense_env(box_a))
    assert proc.returncode == 0 and reply(proc)["action"] == "started", (proc.stdout, proc.stderr)
    assert box_a.containers()[NAME]["labels"]["lb.cluster.model_choice"] == DENSE
    assert ["rm", "-f", NAME] in box_a.calls()


def test_up_launches_the_launch_profile_it_is_given(box_a):
    """A second model, entirely from its LaunchProfile: its own models root (`~` is this box's home), its
    checkpoint, its served name — and no speculative decoding, so no drafter flag and no view."""
    FakeBox.place_checkpoint(box_a.home / "models", name="Qwen3.8-27B")
    proc = box_a.run("up", "0", "9", "8", "131072", DENSE, config=dense_env(box_a))
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("-v") + 1] == f"{box_a.home}/models:/models:ro", "`~/models` is this box's home"
    assert run[run.index("--model") + 1] == "/models/Qwen3.8-27B"
    assert run[run.index("--served-model-name") + 1] == "Qwen/Qwen3.8-27B"
    assert run[run.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert run[run.index("--reasoning-parser") + 1] == "qwen3"
    assert "--speculative-config" not in run and not list((box_a.home / "models").glob("*-mtp"))
    assert mounted_cache(run) == str(cache_of(box_a.home, model="Qwen3.8-27B"))


@pytest.mark.parametrize("off, gone", [
    ("CLUSTER_TOOL_CALL_PARSER", ("--enable-auto-tool-choice", "--tool-call-parser")),
    ("CLUSTER_REASONING_PARSER", ("--reasoning-parser",)),
    ("CLUSTER_SPECULATIVE_CONFIG", ("--speculative-config",)),
], ids=["no-tool-call-parser", "no-reasoning-parser", "no-speculative-decoding"])
def test_a_launch_setting_left_empty_passes_no_flag_at_all(box_a, off, gone):
    """Empty is how a LaunchProfile says a model has none of it — never a flag it never asked for."""
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", off: ""})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert not [flag for flag in gone if flag in run], run
    if off == "CLUSTER_SPECULATIVE_CONFIG":
        assert not list(box_a.models.glob(f"{MODEL}-mtp*")), "no drafter, no drafter view"


def test_a_speculative_config_from_the_launch_profile_is_what_reaches_vllm(box_a):
    """The profile's own drafter, with tp-rank's drafter view added to it — and nothing else changed."""
    lay_out_checkpoint(box_a.models / MODEL)
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201",
                             "CLUSTER_SPECULATIVE_CONFIG": '{"method":"mtp","num_speculative_tokens":2}'})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a)[last_run(box_a).index("--speculative-config") + 1] == (
        '{"method":"mtp","num_speculative_tokens":2,"model":"/models/' + MODEL + '-mtp"}')


def test_a_models_dir_under_the_boxs_own_home_is_written_with_a_tilde(box_a):
    """The LB cannot know a box's HOME — and a quoted value never expands — so the catalog's roots say
    `~/models` and the box itself expands it."""
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201",
                             "CLUSTER_MODELS_DIRS": "~/models/qwen38-flash-next-cluster"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    run = last_run(box_a)
    assert run[run.index("-v") + 1] == f"{box_a.models}:/models:ro"


def test_a_cache_dir_inside_a_tilde_models_dir_is_still_refused(box_a):
    """The cache must stay on the box's own disk, whichever way its models dir was written."""
    proc = box_a.run("up", "0", "7", "16", "262144", FLASH,
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MODELS_DIRS": "~/models",
                             "CLUSTER_CACHE_DIR": f"{box_a.home}/models/cache"})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "CLUSTER_CACHE_DIR" in reply(proc)["error"]
    assert box_a.calls() == []


# ── availability: what a box has, asked before anything is reserved ──────────

IMAGE = "myllmbox/qwen38-flash-next-cluster-vllm:v5"


def availability_env(models: dict, images=(IMAGE,), **over) -> dict[str, str]:
    """The LB asks for several models at once: one line each, `<name>TAB<dirs>`, best dir first."""
    lines = "\n".join(f"{name}\t" + ":".join(str(d) for d in dirs) for name, dirs in models.items())
    return {"CLUSTER_AVAILABILITY_MODELS": lines, "CLUSTER_AVAILABILITY_IMAGES": " ".join(images), **over}


def test_availability_reports_the_first_folder_holding_each_model_and_whether_its_image_is_there(box_a):
    box_a.set(images=[IMAGE])
    hdd = box_a.root / "hdd"
    FakeBox.place_checkpoint(hdd, name="Qwen3.8-27B")
    out = reply(box_a.run("availability", config=availability_env(
        {MODEL: ["~/models/qwen38-flash-next-cluster", "/mnt/box-a-home/models/qwen38-flash-next-cluster"],
         "Qwen3.8-27B": [f"{box_a.home}/models", hdd]},
        images=(IMAGE, "vllm/vllm-openai:v0.23.0"))))
    assert out["models"] == [
        {"name": MODEL, "dir": str(box_a.models), "searched": [str(box_a.models)], "error": "", "unusable": ""},
        {"name": "Qwen3.8-27B", "dir": str(hdd), "searched": [f"{box_a.home}/models", str(hdd)], "error": "",
         "unusable": ""}]
    assert out["images"] == {IMAGE: True, "vllm/vllm-openai:v0.23.0": False}


def test_availability_says_missing_only_after_looking_in_every_folder_and_names_them_all(box_a):
    hdd = box_a.root / "hdd"
    hdd.mkdir()
    out = reply(box_a.run("availability", config=availability_env(
        {"Qwen3.8-27B": [f"{box_a.home}/models", hdd]}, images=())))
    assert out["models"] == [{"name": "Qwen3.8-27B", "dir": None,
                              "searched": [f"{box_a.home}/models", str(hdd)], "error": "", "unusable": ""}]
    assert out["images"] == {}


# GENERATION 279, 2026-10-29: box-b's TensorFold snapshot was a Hugging Face cache snapshot — every file a link
# `../../blobs/<sha>`, out of the folder `up` mounts as /models. The box said "present" (config.json stats fine on the
# host, through the link), the rank started, and TensorFold died on `cannot read /models/fc694b54…/tokenizer.json
# (FileNotFound)`. A checkpoint is launchable only if the CONTAINER can read it: a link that leaves the mounted folder,
# dangles, or is absolute (the host's path, which /models does not have) makes it unusable, named file by file —
# never dereferenced on the operator's behalf.
def test_a_checkpoint_whose_links_leave_the_mounted_folder_is_unusable_and_says_which(box_a):
    hf = box_a.root / "hf"
    snap = hf / "snapshots"
    (hf / "blobs").mkdir(parents=True)
    (hf / "blobs" / "abc").write_text("{}")
    FakeBox.place_checkpoint(snap, name="rev1")
    (snap / "rev1" / "tokenizer.json").symlink_to("../../blobs/abc")                 # resolves, OUTSIDE /models
    (snap / "rev1" / "vocab.json").symlink_to("../../blobs/gone")                    # dangles
    (snap / "shared.json").write_text("{}")
    (snap / "rev1" / "absolute.json").symlink_to(snap / "shared.json")              # inside, but the host's path
    (snap / "rev1" / "relative.json").symlink_to("../shared.json")                  # inside, relative: fine
    clean = box_a.root / "clean"
    FakeBox.place_checkpoint(clean, name="rev2")
    (clean / "rev2" / "tokenizer.json").symlink_to("config.json")                   # inside, relative: fine
    out = reply(box_a.run("availability", config=availability_env({"rev1": [snap], "rev2": [clean]}, images=())))
    bad, good = out["models"]
    assert bad["dir"] == str(snap), "the checkpoint was found: what it cannot be is launched"
    for name in ("rev1/tokenizer.json", "rev1/vocab.json", "rev1/absolute.json"):
        assert name in bad["unusable"], bad["unusable"]
    assert "relative.json" not in bad["unusable"] and "outside" in bad["unusable"], bad["unusable"]
    assert good == {"name": "rev2", "dir": str(clean), "searched": [str(clean)], "error": "", "unusable": ""}


def test_executor_availability_marks_a_checkpoint_with_escaping_links_absent_with_the_files_named(box_a, ssh):
    """The box's word reaches the planner: present False, and the reason the blocked start quotes."""
    from vllm_lb.catalog import document, ModelCatalog

    box_a.set(images=[IMAGE], mounted=["/mnt/box-a-home", "/srv/hdd"])
    outside = box_a.root / "blobs"
    outside.mkdir()
    (outside / "tok").write_text("{}")
    (box_a.models / MODEL / "tokenizer.json").symlink_to(outside / "tok")
    ex, host = ssh.executor()
    flash = ex.availability(host, [ModelCatalog.load().get(FLASH)], observed_at=1.0)[FLASH]
    assert flash.present is False and flash.dir == str(box_a.models)
    assert f"{MODEL}/tokenizer.json" in flash.reason(), flash.reason()


def test_a_folder_that_does_not_answer_is_unknown_never_missing(box_a):
    """A hung NFS mount must never read as "the files are gone": each look is bounded, and says so."""
    _install(box_a.root / "bin" / "stat",
             '#!/bin/sh\ncase "$*" in *hung*) sleep 30;; esac\nexec /usr/bin/stat "$@"\n')
    hung = box_a.root / "hung"
    hung.mkdir()
    out = reply(box_a.run("availability", config=availability_env(
        {MODEL: [hung, "~/models/qwen38-flash-next-cluster"], "Qwen3.8-27B": [hung]},
        images=(), TP_RANK_PROBE_TIMEOUT_S="1")))
    found, unknown = out["models"]
    assert found["dir"] == str(box_a.models), "a later folder still answers for it"
    assert str(hung) in found["error"] and "1 s" in found["error"]
    assert unknown["dir"] is None and str(hung) in unknown["error"], "unknown, not missing"


# THE FALSE FLAG OF 2026-10-10 20:15:22Z. Both GB10 boxes were rebooted while generation 186 started; box-b came
# back at 20:12:54, box-a at 20:14:03, and box-b reads the checkpoint over NFS from box-a (/mnt/box-a-home,
# an fstab automount of 192.0.2.201:/). Asked at 20:15:22, before that share was mounted, tp-rank stat'ed
# `/mnt/box-a-home/models/…/config.json`, got ENOENT off the EMPTY MOUNT POINT and answered "not here" —
# and the LB said `blocked: box-b: no …/config.json in …; choose another model, or copy it there` about a
# checkpoint that was exactly where it always is. Two seconds later the LB healed box-b's worker (which
# remounted the share) and the pair launched fine. Only a folder that can be READ and lacks the file is
# "missing"; a share that is not mounted, a stale handle, an I/O error or a hang is "cannot look", and says why.

NFS_DIR = "/mnt/box-a-home/models/qwen38-flash-next-cluster"          # under FSTAB's pair-link-nfs entry


def test_a_folder_under_an_nfs_share_that_is_not_mounted_is_unknown_never_missing(box_a):
    out = reply(box_a.run("availability", config=availability_env({MODEL: [NFS_DIR]}, images=())))
    (m,) = out["models"]
    assert m["dir"] is None and m["searched"] == [NFS_DIR]
    assert "/mnt/box-a-home" in m["error"] and "not mounted" in m["error"], m["error"]
    assert "192.0.2.201:/" in m["error"], "names the share, so the operator knows which box to look at"


def test_a_folder_under_a_mounted_nfs_share_that_lacks_the_model_is_missing(box_a):
    """The share is up and the folder is simply not on it: that IS "not here", with no error."""
    box_a.set(mounted=["/mnt/box-a-home"])
    out = reply(box_a.run("availability", config=availability_env({MODEL: [NFS_DIR]}, images=())))
    (m,) = out["models"]
    assert m["dir"] is None and m["error"] == "", m


def test_a_folder_this_box_does_not_have_at_all_is_missing_not_unknown(box_a):
    """A search path is sent to both boxes alike; a folder that is no NFS share of this box's and does
    not exist here is a folder with nothing in it, not a share that is still coming up."""
    out = reply(box_a.run("availability", config=availability_env(
        {MODEL: [box_a.root / "no-such-disk" / "models"]}, images=())))
    (m,) = out["models"]
    assert m["dir"] is None and m["error"] == "", m


def test_a_folder_that_cannot_be_read_is_unknown_and_says_why(box_a):
    """A stale NFS handle or an I/O error on the folder: the box cannot look, and the error is stat's own."""
    _install(box_a.root / "bin" / "stat",
             '#!/bin/sh\ncase "$*" in *stale*) echo "stat: cannot statx \'$2\': Stale file handle" >&2; exit 1;; esac\n'
             'exec /usr/bin/stat "$@"\n')
    stale = box_a.root / "stale"
    stale.mkdir()
    out = reply(box_a.run("availability", config=availability_env({MODEL: [stale]}, images=())))
    (m,) = out["models"]
    assert m["dir"] is None and str(stale) in m["error"] and "Stale file handle" in m["error"], m


def test_a_config_json_that_cannot_be_read_in_a_readable_folder_is_unknown_too(box_a):
    _install(box_a.root / "bin" / "stat",
             '#!/bin/sh\ncase "$*" in *broken/*config.json) echo "stat: cannot statx \'$2\': Input/output error" >&2; exit 1;; esac\n'
             'exec /usr/bin/stat "$@"\n')
    broken = box_a.root / "broken"
    broken.mkdir()
    out = reply(box_a.run("availability", config=availability_env({MODEL: [broken]}, images=())))
    (m,) = out["models"]
    assert m["dir"] is None and "Input/output error" in m["error"] and f"{broken}/{MODEL}/config.json" in m["error"], m


def test_availability_asks_docker_only_about_images_and_changes_nothing_on_the_box(box_a):
    box_a.seed(gen=7)
    out = reply(box_a.run("availability", config=availability_env({MODEL: [box_a.models]})))
    assert out["images"] == {IMAGE: False}
    assert [c[:2] for c in box_a.calls()] == [["image", "inspect"]]
    assert box_a.containers()[NAME]["labels"]["lb.cluster.gen"] == "7", "no rank was touched"
    assert not (box_a.home / ".cache").exists() and not list(box_a.models.glob("*-mtp*"))


def test_an_image_docker_cannot_speak_about_is_unknown_and_the_files_still_answer(box_a):
    box_a.set(daemon_down=True)
    proc = box_a.run("availability", config=availability_env({MODEL: [box_a.models]}))
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["images"] == {IMAGE: None} and out["models"][0]["dir"] == str(box_a.models)


@pytest.mark.parametrize("bad", [
    {"CLUSTER_AVAILABILITY_MODELS": "../x\t/models"},
    {"CLUSTER_AVAILABILITY_MODELS": ".hidden\t/models"},
    {"CLUSTER_AVAILABILITY_MODELS": ""},
    {"CLUSTER_AVAILABILITY_IMAGES": "an image"},
], ids=repr)
def test_availability_refuses_what_it_would_not_launch(box_a, bad):
    proc = box_a.run("availability", config={**availability_env({MODEL: [box_a.models]}), **bad})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and "CLUSTER_AVAILABILITY" in out["error"], out
    assert box_a.calls() == []


# ── down and logs ─────────────────────────────────────────────────────────────

def test_down_removes_our_rank_and_leaves_the_brokers_vllm_alone(box_a):
    box_a.seed(gen=7)
    box_a.seed("vllm", gen=None)
    proc = box_a.run("down")
    assert proc.returncode == 0 and reply(proc) == {"ok": True, "removed": True}
    assert NAME not in box_a.containers() and "vllm" in box_a.containers()
    assert [c for c in box_a.calls() if c[0] == "rm"] == [["rm", "-f", NAME]]


def test_down_with_no_rank_succeeds_and_removes_nothing(box_a):
    proc = box_a.run("down")
    assert proc.returncode == 0 and reply(proc) == {"ok": True, "removed": False}
    assert not [c for c in box_a.calls() if c[0] == "rm"]


def test_halt_stops_our_rank_and_keeps_it_and_its_log_for_the_post_mortem(box_a):
    """A FAILED pair is stopped, never removed. `down` is `docker rm -f`: it frees the box and destroys the
    only record of why the rank died — generation 10's exit was unrecoverable for exactly that reason
    (worklog/vllm-gateway-program.md, L5), and on 2026-09-18 the WORKER's log was the whole diagnosis while
    the head's said only "check the logs of the remote worker". `halt` is `docker stop`: the process and
    its ~100 GiB are gone, the container, its labels, its exit code and its log stay until the operator's
    Stop removes them."""
    box_a.seed(gen=7, model_choice=FLASH, logs=["Autotuning process starts ...", "RuntimeError: boom"])
    box_a.seed("vllm", gen=None)
    proc = box_a.run("halt")
    assert proc.returncode == 0 and reply(proc) == {"ok": True, "halted": True}
    assert [c for c in box_a.calls() if c[0] in ("stop", "rm", "kill")] == [["stop", "-t", "10", NAME]]
    assert box_a.containers()["vllm"]["running"] is True, "the broker's vLLM is never ours to stop"
    status = reply(box_a.run("status"))["rank"]
    assert (status["present"], status["running"], status["gen"], status["exit_code"],
            status["model_choice"]) == (True, False, 7, 137, FLASH)
    assert box_a.run("logs").stdout.splitlines() == ["Autotuning process starts ...", "RuntimeError: boom"]


@pytest.mark.parametrize("seeded", [None, "exited"])
def test_halt_with_no_rank_running_succeeds_and_stops_nothing(box_a, seeded):
    """Idempotent, like every verb: the planner halts on every tick until it SEES the rank stopped."""
    if seeded:
        box_a.seed(gen=7, running=False, exit_code=1)
    proc = box_a.run("halt")
    assert proc.returncode == 0 and reply(proc) == {"ok": True, "halted": False}
    assert not [c for c in box_a.calls() if c[0] in ("stop", "rm", "kill")]
    if seeded:
        assert box_a.containers()[NAME]["exit_code"] == 1, "an exited rank keeps the exit code it died with"


def test_a_halt_docker_refuses_is_a_failure_in_dockers_own_words(box_a):
    box_a.seed(gen=7)
    box_a.set(stop_error="Error response from daemon: cannot stop container: permission denied")
    proc = box_a.run("halt")
    assert proc.returncode == 1
    out = reply(proc)
    assert out["ok"] is False and "docker stop -t 10 " + NAME in out["error"] and "permission denied" in out["error"]
    assert box_a.containers()[NAME]["running"] is True


def test_logs_tails_the_rank_with_its_stderr_folded_in(box_a):
    box_a.seed(gen=7, logs=[f"line {i}" for i in range(300)])
    proc = box_a.run("logs", "2")
    assert proc.returncode == 0 and proc.stdout.splitlines() == ["line 298", "line 299"]
    proc = box_a.run("logs")
    assert proc.stdout.splitlines() == [f"line {i}" for i in range(140, 300)]
    assert ["logs", "--tail", "160", NAME] in box_a.calls()


def test_logs_with_no_rank_prints_nothing(box_a):
    proc = box_a.run("logs")
    assert proc.returncode == 0 and proc.stdout == ""


# ── stacks: what a hung rank is doing, before the LB takes it down ───────────

#: py-spy as an image would carry it: it says which process it was pointed at — by the title that process
#: set for itself, as /proc has it — and whether it was asked for the native frames too.
FAKE_PY_SPY = r"""#!/bin/sh
native="" pid=""
while [ $# -gt 0 ]; do
  case $1 in --native) native=" (native)" ;; --pid) pid=$2; shift ;; esac
  shift
done
echo "Process $pid: $(tr '\0' ' ' </proc/$pid/cmdline)$native"
echo "    capture_one_shape (runner/decode_cuda_graph_runner.py:1182)"
"""
#: pip with no index to reach: what an image without py-spy says when asked to install it offline.
FAKE_PIP_PYTHON3 = "#!/bin/sh\necho 'ERROR: No matching distribution found for py-spy'\nexit 1\n"


def container_bin(box_a, **tools: str) -> Path:
    """The rank container's own PATH, holding `tools` (name → script) before the box's /usr/bin:/bin."""
    where = box_a.root / "container-bin"
    where.mkdir(exist_ok=True)
    for name, text in tools.items():
        _install(where / name.replace("_", "-"), text)
    box_a.set(exec_path=f"{where}:/usr/bin:/bin")
    return where


def scheduler(title: str = "sglang::scheduler_TP0") -> subprocess.Popen:
    """A process by the title SGLang's scheduler gives itself (setproctitle rewrites /proc/<pid>/cmdline)."""
    proc = subprocess.Popen(["bash", "-c", f"exec -a {title} sleep 30"])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if Path(f"/proc/{proc.pid}/cmdline").read_bytes().startswith(title.encode()):
                return proc
        except OSError:
            pass
        time.sleep(0.01)
    proc.kill()
    raise AssertionError(f"{title} never showed up in /proc")


#: A native stack walker as an image might carry one (eu-stack, else gdb): it says which process it walked.
FAKE_WALKER = "#!/bin/sh\necho \"walked every thread of: $*\"\n"
#: journalctl as the box answers it: the kernel's lines only to root (`-q` hides the refusal from anyone else).
FAKE_JOURNALCTL = ("#!/bin/sh\n[ \"$FAKE_ROOT\" = 1 ] || [ -n \"$JOURNAL_READABLE\" ] || exit 0\n"
                   "echo 'Oct 03 06:28:41 box-a kernel: NVRM: Xid (PCI:000f:01:00): 13, pid=4242'\n")


def test_stacks_dumps_every_engine_process_of_the_running_rank_python_then_native(box_a):
    """Generation 131's diagnosis was taken by hand, inside each rank, before anyone removed it: py-spy's
    stacks (rank 0 inside libcuda on a 128-byte copy in FlashInfer's plan, rank 1 at the gloo barrier), and
    from the box, a GPU at idle power. `stacks` takes all of it for the LB, which keeps it under its state dir
    before it tears a hung start down (cluster_loop `_save_hang_evidence`) — Python stacks first, every
    thread's state and wchan, native stacks, then the host's GPU, memory and kernel log."""
    box_a.seed(gen=131)
    box_a.with_gpu(rows=[{"pwr": 11, "gtemp": 48, "sm": 0, "mem": 0, "pclk": 2424}])
    container_bin(box_a, py_spy=FAKE_PY_SPY)
    sched = scheduler()
    try:
        proc = box_a.run("stacks")
    finally:
        sched.kill()
        sched.wait()
    out = proc.stdout
    assert proc.returncode == 0, out + proc.stderr
    assert f"=== py-spy, pid {sched.pid}: sglang::scheduler_TP0 30" in out, out
    assert f"Process {sched.pid}: sglang::scheduler_TP0 30 \n" in out, out
    assert f"=== threads, pid {sched.pid}: sglang::scheduler_TP0 30" in out, out
    assert re.search(rf"^{sched.pid} \S* {sched.pid} \(sleep\) [RSD] ", out, re.M), "the thread's wchan and stat"
    assert f"Process {sched.pid}: sglang::scheduler_TP0 30  (native)" in out, out
    assert out.index("=== py-spy, pid") < out.index("=== threads, pid") < out.index("=== py-spy --native, pid")
    assert "=== host: GPU\nutilization.gpu [%], power.draw [W], memory.used [MiB]\n0 %, 11.00 W, [N/A]\n" in out
    assert "=== host: meminfo\nMemTotal:" in out and "MemAvailable:   117000000 kB" in out
    assert "=== host: kernel log, last 15 minutes\n(not readable here, or nothing in the last 15 minutes)" in out
    (call,) = [c for c in box_a.calls() if c[0] == "exec"]
    assert call[:4] == ["exec", NAME, "sh", "-c"] and call[5:] == ["sh", "8"], call


@pytest.mark.parametrize("walker", ["eu_stack", "gdb"])
def test_stacks_walks_every_native_thread_when_the_image_can(box_a, walker):
    """py-spy cannot see libcuda's or NCCL's own threads; eu-stack (else gdb) can, when the image has one."""
    box_a.seed(gen=131)
    container_bin(box_a, py_spy=FAKE_PY_SPY, **{walker: FAKE_WALKER})
    sched = scheduler()
    try:
        out = box_a.run("stacks").stdout
    finally:
        sched.kill()
        sched.wait()
    name = walker.replace("_", "-")
    args = f"-p {sched.pid}" if walker == "eu_stack" else f"-p {sched.pid} -batch -ex thread apply all bt"
    assert f"=== {name}, pid {sched.pid}: every thread\nwalked every thread of: {args}\n" in out, out


def test_stacks_without_a_native_walker_skips_it_silently(box_a):
    box_a.seed(gen=131)
    container_bin(box_a, py_spy=FAKE_PY_SPY)
    sched = scheduler()
    try:
        out = box_a.run("stacks").stdout
    finally:
        sched.kill()
        sched.wait()
    assert f"=== threads, pid {sched.pid}" in out, out
    assert "eu-stack" not in out and "=== gdb" not in out, out


@pytest.mark.parametrize("access", ["readable", "root shell", "neither"])
def test_stacks_reads_the_kernel_log_as_this_account_or_through_the_root_shell_heal_uses(box_a, access):
    box_a.seed(gen=131)
    _install(box_a.root / "bin" / "journalctl", FAKE_JOURNALCTL)
    if access == "root shell":
        box_a.set(images=["ubuntu:24.04"])
    out = box_a.run("stacks", extra_env={"JOURNAL_READABLE": "1"} if access == "readable" else None).stdout
    tail = out[out.index("=== host: kernel log, last 15 minutes"):]
    if access == "neither":
        assert "(not readable here, or nothing in the last 15 minutes)" in tail, tail
    else:
        assert "NVRM: Xid (PCI:000f:01:00): 13" in tail, tail
    roots = [c for c in box_a.calls() if c[:2] == ["run", "--rm"]]
    assert bool(roots) == (access == "root shell"), "the root shell only when this account cannot read it"


def test_stacks_on_an_image_without_py_spy_tries_one_bounded_install_and_says_what_happened(box_a):
    box_a.seed(gen=131)
    container_bin(box_a, python3=FAKE_PIP_PYTHON3)
    sched = scheduler()
    try:
        proc = box_a.run("stacks", extra_env={"TP_RANK_STACKS_PIP_S": "5"})
    finally:
        sched.kill()
        sched.wait()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "py-spy is not in the image" in proc.stdout and "No matching distribution" in proc.stdout
    assert "py-spy: not available" in proc.stdout, proc.stdout
    assert f"=== threads, pid {sched.pid}" in proc.stdout, "the thread states need no py-spy"


@pytest.mark.parametrize("seeded", [None, "exited"])
def test_stacks_with_no_rank_running_prints_nothing_and_enters_nothing(box_a, seeded):
    if seeded:
        box_a.seed(gen=131, running=False, exit_code=137)
    proc = box_a.run("stacks")
    assert proc.returncode == 0 and proc.stdout == ""
    assert not [c for c in box_a.calls() if c[0] == "exec"]


@pytest.mark.parametrize("name, value", [("TP_RANK_STACKS_TIMEOUT_S", "0"), ("TP_RANK_STACKS_PIP_S", "x")])
def test_stacks_refuses_a_bound_it_cannot_honour(box_a, name, value):
    box_a.seed(gen=131)
    proc = box_a.run("stacks", extra_env={name: value})
    assert proc.returncode == 1 and name in reply(proc)["error"]
    assert not [c for c in box_a.calls() if c[0] == "exec"]


def test_executor_stacks_brings_the_dump_back_and_says_why_when_it_cannot(box_a, ssh):
    box_a.seed(gen=131)
    container_bin(box_a, py_spy=FAKE_PY_SPY)
    ex, host = ssh.executor()
    sched = scheduler()
    try:
        text = ex.stacks(host)
    finally:
        sched.kill()
        sched.wait()
    assert f"Process {sched.pid}: sglang::scheduler_TP0" in text and ex.last_error["box-a"] == ""
    assert ssh.calls()[-1][-1].endswith("bash -s -- stacks")
    ssh.set(mode="refuse")
    assert ex.stacks(host) == "" and ex.last_error["box-a"].startswith("stacks: exit 255")


# ── what it refuses ───────────────────────────────────────────────────────────

REFUSED = [
    ["up", "2", "7", "16", "262144"],
    ["up", "-1", "7", "16", "262144"],
    ["up", "0", "0", "16", "262144"],
    ["up", "0", "1000000", "16", "262144"],
    ["up", "0", "7", "0", "262144"],
    ["up", "0", "7", "49", "262144"],
    ["up", "0", "7", "16", "999"],
    ["up", "0", "7", "16", "262145"],
    ["up", "0", "7", "16", "262144;rm"],
    ["up", "0", "7", "16", "262144 && reboot"],
    ["up", "0", "07", "16", "262144"],                     # would never equal the label "7"
    ["up", "18446744073709551617", "7", "16", "262144"],   # 2**64 + 1 wraps to 1 in bash
    ["up", "0", "7", "16"],
    ["up", "0", "7", "16", "262144", FLASH, "extra"],
    # a ModelChoice is a catalog id: lower-case, and a label docker will take
    ["up", "0", "7", "16", "262144", "Qwen3.8-27B"],
    ["up", "0", "7", "16", "262144", "-rf"],
    ["up", "0", "7", "16", "262144", ".hidden"],
    ["status", "now"],
    ["availability", "now"],
    ["status\nreboot"],
    ["logs", "$(reboot)"],
    ["down", "`reboot`"],
    ["down", "hard"],
    ["halt", "now"],
    ["halt", "$(reboot)"],
    ["logs", "0"],
    ["logs", "2001"],
    ["logs", "10", "20"],
    [""],
    ["bash"],
    [],
]


@pytest.mark.parametrize("argv", REFUSED, ids=repr)
def test_invalid_requests_are_refused_as_json_before_docker_runs(box_a, argv):
    proc = box_a.run(*argv)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and out["error"]
    assert box_a.calls() == [], "a refused request must never reach docker"


# ── SshExecutor, end to end through the real script ───────────────────────────

def test_executor_argv_pins_the_key_and_host_keys_and_reuses_one_connection(ssh):
    """One multiplexed connection per box: a status poll every tick must not be a fresh sshd
    session each time — a previous design ran box-a's sshd into MaxStartups."""
    ex, host = ssh.executor(env={"CLUSTER_PORT": "8100", "CLUSTER_MASTER_ADDR": "192.0.2.201"})
    assert ex.status(host) is not None, ex.last_error
    assert ssh.calls() == [["-i", ssh.key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                            "-o", "StrictHostKeyChecking=yes",
                            "-o", f"UserKnownHostsFile={ssh.known_hosts}", "-o", "ConnectTimeout=5",
                            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                            "-o", "ControlMaster=auto", "-o", f"ControlPath={ssh.root}/cm-%C",
                            "-o", "ControlPersist=120", TARGET,
                            "env CLUSTER_PORT=8100 CLUSTER_MASTER_ADDR=192.0.2.201 "
                            "CLUSTER_HOST_IP=192.0.2.201 bash -s -- status"]]
    assert inspect.signature(type(ex)).parameters["control_dir"].default == "/tmp"


def test_executor_sends_the_script_on_stdin_so_nothing_is_installed_on_a_box_a(box_a, ssh):
    ex, host = ssh.executor(env={"CLUSTER_MASTER_ADDR": "192.0.2.201"})
    before = sorted(box_a.home.rglob("*"))
    assert ex.status(host) is not None, ex.last_error
    assert ssh.stdin() == TP_RANK.read_bytes(), "the LB's own copy, byte for byte, on every call"
    command = ssh.calls()[-1][-1]
    words = shlex.split(command)
    program = words.index("bash")
    assert words[0] == "env" and all(re.fullmatch(r"[A-Z_]+=.*", w) for w in words[1:program])
    assert words[program:] == ["bash", "-s", "--", "status"], \
        "no remote path anywhere: bash from the box's PATH, the script from stdin"
    assert "tp-rank" not in command and str(TP_RANK) not in command
    assert sorted(box_a.home.rglob("*")) == before, "and nothing is left behind on the box"


def test_executor_gives_each_box_its_own_interconnect_as_vllm_host_ip(box_a, ssh):
    ex, _ = ssh.executor()
    box_b = cluster_host(label="box-b-gpu", box="box-b", ssh_target="user@192.0.2.12",
                          interconnect="192.0.2.202")
    ok, msg = ex.up(box_b, rank=1, gen=3, max_num_seqs=12, max_model_len=131072)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == "env CLUSTER_HOST_IP=192.0.2.202 bash -s -- up 1 3 12 131072"
    run = [c for c in box_a.calls() if c[0] == "run"][-1]
    assert run == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=1, gen=3, seqs=12, ctx=131072, models_dir=box_a.models,
                              cache_dir=cache_of(box_a.home),
                              host_ip="192.0.2.202")
    assert "VLLM_HOST_IP=192.0.2.202" in run and ex.last_error["box-b"] == ""


def test_executor_env_values_reach_the_script_literally_through_the_remote_shell(box_a, ssh):
    """A plain key hands the command line to the box's login shell, so every value is quoted: a
    space, a quote or a `$(…)` in the configuration arrives as those characters and runs nothing —
    here in a models dir that really exists, so the script's own look into it is literal too."""
    models = box_a.place_checkpoint(box_a.root / "it's a dir; $(touch pwned) `touch pwned2`")
    served = 'Qwen/Qwen3.8 "Flash" Next'
    ex, host = ssh.executor(env={"CLUSTER_PORT": "8200", "CLUSTER_MODELS_DIRS": str(models),
                                 "CLUSTER_SERVED_MODEL_NAME": served})
    assert ex.status(host)["port"] == 8200, ex.last_error
    ok, msg = ex.up(host, rank=0, gen=7, max_num_seqs=16, max_model_len=262144)
    assert ok is True, msg
    run = [c for c in box_a.calls() if c[0] == "run"][-1]
    assert run[run.index("-v") + 1] == f"{models}:/models:ro"
    assert run[run.index("--served-model-name") + 1] == served
    assert run[run.index("--port") + 1] == "8200"
    assert not list(box_a.root.rglob("pwned*")), "a configuration value was executed"


def test_executor_refuses_an_env_key_the_remote_shell_would_parse(ssh):
    """Values can be quoted; a variable NAME cannot. A key that is not one is refused before dialing."""
    ex, host = ssh.executor(env={"CLUSTER_PORT;touch x": "8100"})
    assert ex.status(host) is None
    assert "CLUSTER_PORT;touch x" in ex.last_error["box-a"]
    assert ssh.calls() == []


def test_executor_status_parses_the_real_scripts_reply(box_a, ssh):
    box_a.seed(gen=7)
    box_a.seed("vllm", gen=None, running=False)
    ex, host = ssh.executor()
    st = ex.status(host)
    assert st == {"rank": {"present": True, "running": True, "gen": 7, "exit_code": None,
                           "started_at": STARTED, "model_choice": None, "exit_reason": ""},
                  "broker_vllm_running": False, "mem_available_gib": 111.6, "port": 8100, "boot_at": BOOTED,
                  "mem_total_gib": 116.1,
                  # L8: what this box's own ollama still holds — null here, nothing is listening on
                  # the fake box's ollama address, and a box that could not be asked has not said.
                  "ollama": None,
                  # And its GPU: null, since the fake box has no nvidia-smi either.
                  "gpu": None,
                  # And its TP link: null, the fake box has no bond0 / RDMA device.
                  "net": None}
    assert ex.last_error["box-a"] == ""


def test_executor_up_down_and_logs_round_trip_through_the_real_script(box_a, ssh):
    ex, host = ssh.executor()
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=12, max_model_len=131072)
    assert ok is True, msg
    assert ssh.calls()[-1][-1].endswith(" bash -s -- up 0 3 12 131072")
    assert [c for c in box_a.calls() if c[0] == "run"] == [
        launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=3, seqs=12, ctx=131072, models_dir=box_a.models,
                    cache_dir=cache_of(box_a.home))]
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=12, max_model_len=131072)
    assert ok is True and "already-running" in msg
    box_a.seed(gen=3, logs=["loading shard 1/20", "loading shard 2/20"])
    assert ex.logs(host, n=1) == "loading shard 2/20\n"
    assert ssh.calls()[-1][-1].endswith(" bash -s -- logs 1")
    ok, msg = ex.down(host)
    assert ok is True, msg
    assert NAME not in box_a.containers() and ex.last_error["box-a"] == ""


def test_executor_halt_stops_the_rank_through_the_real_script_and_says_what_it_did(box_a, ssh):
    ex, host = ssh.executor()
    box_a.seed(gen=3, logs=["the last thing it said"])
    ok, msg = ex.halt(host)
    assert (ok, msg) == (True, "halted"), msg
    assert ssh.calls()[-1][-1].endswith(" bash -s -- halt")
    assert box_a.containers()[NAME]["running"] is False and ex.last_error["box-a"] == ""
    assert ex.logs(host, n=5) == "the last thing it said\n", "a halted rank still answers for itself"
    assert ex.halt(host) == (True, "nothing to halt")
    box_a.set(daemon_down=True)
    ok, msg = ex.halt(host)
    assert ok is False and "Cannot connect to the Docker daemon" in msg


def test_executor_carries_the_operators_launch_settings_through_the_remote_shell(box_a, ssh):
    """What the LB forwards from its own environment arrives as tp-rank's settings: the PLE mode as
    MBX_PLE_REPLICATE and in the cache key, and an empty cache dir as no cache at all."""
    ex, host = ssh.executor(env={"CLUSTER_PLE_REPLICATE": "0", "CLUSTER_CACHE_DIR": ""})
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=12, max_model_len=131072)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == ("env CLUSTER_PLE_REPLICATE=0 CLUSTER_CACHE_DIR='' "
                                   "CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 3 12 131072")
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=3, seqs=12, ctx=131072, models_dir=box_a.models,
                                          cache_dir=None, ple="0")


def test_launch_env_is_the_one_translation_of_a_launch_profile(box_a, ssh):
    """DRY: `cluster_exec.launch_env` is the only place a ModelSpec's cluster LaunchProfile becomes tp-rank's
    settings, and every key it emits is a setting tp-rank actually has."""
    from vllm_lb.catalog import ModelCatalog, document
    from vllm_lb.cluster_exec import launch_env

    catalog = ModelCatalog.load()
    dense = launch_env(catalog.get(DENSE))
    # No CLUSTER_PLE_REPLICATE: a model with no PLE table says nothing about one, and tp-rank's default stands.
    assert dense == {"CLUSTER_IMAGE": IMAGE,
                     "CLUSTER_MODELS_DIRS": "~/models:/mnt/box-a-home/models:/srv/models",
                     "CLUSTER_MODEL_NAME": "Qwen3.8-27B-NVFP4", "CLUSTER_SERVED_MODEL_NAME": "Qwen/Qwen3.8-27B",
                     "CLUSTER_GPU_MEMORY_UTILIZATION": "0.70", "CLUSTER_KV_CACHE_MEMORY": "48000000000",
                     "CLUSTER_TOOL_CALL_PARSER": "qwen3_xml", "CLUSTER_REASONING_PARSER": "qwen3",
                     "CLUSTER_SPECULATIVE_CONFIG": SPEC, "CLUSTER_MTP_PREFIX": "mtp.",
                     "CLUSTER_KV_CACHE_DTYPE": "int4_per_token_head", "CLUSTER_ATTENTION_BACKEND": "TRITON_ATTN"}
    flash = launch_env(catalog.get(FLASH))
    assert flash["CLUSTER_MODEL_NAME"] == MODEL and flash["CLUSTER_SPECULATIVE_CONFIG"] == SPEC
    assert (flash["CLUSTER_MTP_PREFIX"], flash["CLUSTER_PLE_REPLICATE"]) == ("mtp.", "0")
    assert flash["CLUSTER_MODELS_DIRS"] == ("~/models/qwen38-flash-next-cluster:"
                                            "/mnt/box-a-home/models/qwen38-flash-next-cluster:"
                                            "/srv/models/qwen38-flash-next-cluster")
    script = TP_RANK.read_text()
    # A vLLM profile's engine flag is declared in tp-rank's VLLM_*_FLAGS arrays; every other setting by `: "${…=}"`.
    flags = set(re.findall(r"\bCLUSTER_[A-Z_]+", " ".join(re.findall(r"^VLLM_[A-Z]+_FLAGS=\(([^)]*)\)", script,
                                                                      re.M))))
    for key in set(flash) | set(dense):
        assert key in flags or re.search("^" + re.escape(f': "${{{key}') + ":?=", script, re.M), \
            f"tp-rank has no setting {key}"


def test_executor_up_sends_the_launch_profile_as_env_and_the_model_choice_as_argv(box_a, ssh):
    from vllm_lb.catalog import ModelCatalog, document

    FakeBox.place_checkpoint(box_a.home / "models", name="Qwen3.8-27B-NVFP4")
    ex, host = ssh.executor()
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=131072,
                    spec=ModelCatalog.load().get(DENSE))
    assert ok is True, msg
    command = ssh.calls()[-1][-1]
    assert command.endswith(f" bash -s -- up 0 3 8 131072 {DENSE}")
    assert "CLUSTER_MODEL_NAME=Qwen3.8-27B-NVFP4 " in command
    assert f"CLUSTER_SPECULATIVE_CONFIG={shlex.quote(SPEC)}" in command and "CLUSTER_PLE_REPLICATE" not in command
    run = last_run(box_a)
    assert run[run.index("--served-model-name") + 1] == "Qwen/Qwen3.8-27B"
    assert run[run.index("--model") + 1] == "/models/Qwen3.8-27B-NVFP4"
    assert run[run.index("--kv-cache-memory") + 1] == "48000000000"
    # This fake checkpoint has no index, so no drafter view is built and the profile's own config is the flag.
    assert run[run.index("--speculative-config") + 1] == SPEC
    assert box_a.containers()[NAME]["labels"]["lb.cluster.model_choice"] == DENSE


def test_executor_availability_asks_the_box_about_every_spec_and_brings_back_what_it_found(box_a, ssh):
    """Availability is OBSERVED per box, for every ModelSpec at once, in one call — and it is the box that
    says where it looked."""
    from vllm_lb.catalog import Availability, ModelCatalog

    box_a.set(images=[IMAGE], mounted=["/mnt/box-a-home", "/srv/hdd"])   # every share of this box's is up
    catalog = ModelCatalog.load()
    ex, host = ssh.executor()
    assert ex.availability(host, [], observed_at=1.0) == {} and ssh.calls() == [], "nothing to ask about"
    seen = ex.availability(host, [catalog.get(FLASH), catalog.get(DENSE)], observed_at=1789500000.0)
    assert set(seen) == {FLASH, DENSE}
    flash = seen[FLASH]
    assert isinstance(flash, Availability)
    assert (flash.name, flash.dir, flash.image, flash.image_present) == (MODEL, str(box_a.models), IMAGE, True)
    assert (flash.present, flash.observed_at, flash.error) == (True, 1789500000.0, "")
    dense = seen[DENSE]
    assert dense.present is False and dense.dir is None, "every folder was read: that one is missing"
    assert dense.searched == (f"{box_a.home}/models", "/mnt/box-a-home/models", "/srv/models")
    assert "Qwen3.8-27B-NVFP4/config.json" in dense.reason()
    assert ssh.calls()[-1][-1].endswith(" bash -s -- availability")
    assert ex.last_error["box-a"] == ""

    # The same box with its shares NOT mounted has not said "missing" about anything it could not look in.
    box_a.set(mounted=[])
    dense = ex.availability(host, [catalog.get(DENSE)], observed_at=1789500000.0)[DENSE]
    assert dense.present is None and dense.dir is None
    assert "/mnt/box-a-home" in dense.error and "not mounted" in dense.error, dense.error
    assert "config.json" not in dense.reason(), "never worded as the file being absent"


def test_executor_availability_is_none_when_the_box_cannot_be_asked_or_does_not_answer_one(ssh):
    from vllm_lb.catalog import ModelCatalog, document

    specs = [ModelCatalog.load().get(FLASH)]
    ex, host = ssh.executor()
    ssh.set(mode="refuse")
    assert ex.availability(host, specs) is None
    assert "Permission denied" in ex.last_error["box-a"]
    ssh.set(mode="reply", reply='{"models": "nope"}\n')
    assert ex.availability(host, specs) is None and ex.last_error["box-a"]


def test_executor_up_reports_the_remote_reason_when_the_launch_fails(box_a, ssh):
    box_a.set(run_error="docker: Error response from daemon: no GPU for you")
    ex, host = ssh.executor()
    ok, msg = ex.up(host, rank=0, gen=1, max_num_seqs=16, max_model_len=262144)
    assert ok is False and "no GPU for you" in msg
    assert "no GPU for you" in ex.last_error["box-a"]


def test_executor_validates_up_arguments_before_dialing(ssh):
    ex, host = ssh.executor()
    good = dict(rank=0, gen=1, max_num_seqs=16, max_model_len=262144)
    for bad in ({"rank": 2}, {"rank": -1}, {"gen": 0}, {"gen": 1_000_000}, {"max_num_seqs": 0},
                {"max_num_seqs": 49}, {"max_model_len": 999}, {"max_model_len": 262145},
                {"gen": True}, {"rank": "0"}):
        ok, msg = ex.up(host, **{**good, **bad})
        assert ok is False and msg, bad
        assert ex.last_error["box-a"], bad
    assert ssh.calls() == [], "an invalid launch fails here, not as an SSH round trip"


def test_executor_turns_an_ssh_failure_into_none_and_a_recorded_error(ssh):
    ex, host = ssh.executor()
    ssh.set(mode="refuse")
    assert ex.status(host) is None
    assert "Permission denied" in ex.last_error["box-a"]
    assert ex.up(host, rank=0, gen=1, max_num_seqs=16, max_model_len=262144)[0] is False
    assert ex.down(host)[0] is False
    assert ex.logs(host) == ""
    ssh.set(mode="forward")
    assert ex.status(host) is not None
    assert ex.last_error["box-a"] == "", "a success clears the box's error"


def test_executor_status_is_bounded_by_its_timeout(ssh):
    ex, host = ssh.executor(status_timeout_s=1.0)
    ssh.set(mode="hang")
    t0 = time.monotonic()
    assert ex.status(host) is None
    assert time.monotonic() - t0 < 5.0
    assert "timed out" in ex.last_error["box-a"].lower()


def test_executor_never_raises_when_ssh_or_the_script_cannot_be_had(box_a, ssh):
    from vllm_lb.cluster_exec import SshExecutor
    ex = SshExecutor(key_path="/nonexistent/key", known_hosts="/nonexistent/known_hosts",
                     script_path=str(TP_RANK), ssh_bin=str(box_a.root / "no-ssh-here"),
                     control_dir=str(box_a.root))
    host = cluster_host(label="box-b-gpu", box="box-b", ssh_target="user@192.0.2.12",
                        interconnect="192.0.2.202")
    assert ex.status(host) is None and ex.last_error["box-b"]
    assert ex.up(host, rank=1, gen=1, max_num_seqs=16, max_model_len=262144)[0] is False
    assert ex.down(host)[0] is False and ex.logs(host) == ""
    ex, host = ssh.executor(script_path=str(box_a.root / "no-such-script"))
    assert ex.status(host) is None and "no-such-script" in ex.last_error["box-a"]
    assert ssh.calls() == [], "without a script there is nothing to send, so nothing is dialed"


def test_executor_status_refuses_a_reply_that_is_not_a_status(ssh):
    ex, host = ssh.executor()
    for junk in ('{"ok": true}\n', "not json\n", "", "[1, 2]\n", '{"rank": {"present": "yes"}}\n'):
        ssh.set(mode="reply", reply=junk)
        assert ex.status(host) is None, junk
        assert ex.last_error["box-a"], junk


def test_executor_status_reads_its_line_past_login_shell_noise(box_a, ssh):
    """sshd runs the command through the login shell, and a chatty rc file may print first."""
    status_line = box_a.run("status").stdout
    ssh.set(mode="reply", reply="Welcome to box-a\n" + status_line)
    ex, host = ssh.executor()
    assert ex.status(host) == json.loads(status_line)


# ── heal: bring this box's broker worker back ─────────────────────────────────

NFS = ["/mnt/nas-a", "/mnt/nas-b", "/srv/hdd", "/mnt/box-a-home"]
UNIT = "fleet-worker.service"                                     # the deployment's own name: lb/.env CLUSTER_WORKER_UNIT
HEAL = {"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_WORKER_UNIT": UNIT}   # what the LB sends a heal
ROOT_SHELL = ["run", "--rm", "--privileged", "--pid=host", "--net=host"]
NSENTER = ["nsenter", "-t", "1", "-m", "-u", "-i", "-n", "-p", "--", "sh", "-c"]
BACKSTOP_DONE = {UNIT: {"load": "loaded", "sub": "exited"}}      # ran at boot, deployed, exited 0
BUT_HDD = [m for m in NFS if m != "/srv/hdd"]


def root_shells(box_a: FakeBox) -> list[list[str]]:
    return [c for c in box_a.calls() if c[:5] == ROOT_SHELL]


def test_heal_remounts_the_latched_mount_as_root_and_restarts_the_worker_backstop(box_a):
    """box-b, 2026-09-28: 13 s after boot /srv/hdd's automount latched `mount-start-limit-hit`, and the
    worker's backstop waited 105 checks for it while cluster generation 29 sat `reserving`. heal remounts
    what is missing — as root, through an image already on the box — and restarts the backstop."""
    box_a.set(images=["nvcr.io/nvidia/vllm:25.09", "debian:stable-slim", "alpine:latest"],
              mounted=BUT_HDD, units=BACKSTOP_DONE)
    assert reply(box_a.run("heal", config=HEAL)) == {"ok": True, "action": "healed", "remounted": ["/srv/hdd"],
                                        "unmounted": [], "worker_unit": "restarted"}
    [shell] = root_shells(box_a)
    assert shell[5] == "debian:stable-slim", "the first usable image on the box: nothing is pulled"
    assert shell[6:17] == NSENTER, "root in PID 1's namespaces: the host's systemd and mount table"
    assert "reset-failed" in shell[17] and "mount" in shell[17], shell[17]
    assert shell[18:] == ["sh", "20", "/srv/hdd"], \
        "the mount timeout, then ONLY the missing NFS mount — each its own argument; / and /boot/efi are not NFS"
    assert box_a.systemctl_calls()[-1] == ["--user", "restart", UNIT]


def test_heal_skips_commented_and_local_fstab_lines(box_a):
    box_a.set(images=["alpine:latest"], mounted=[], units=BACKSTOP_DONE)
    out = reply(box_a.run("heal", config=HEAL))
    assert out["remounted"] == NFS, "the four NFS mounts, in fstab order; not /, /boot/efi or #/mnt/retired"


def test_with_every_mount_up_heal_only_restarts_the_backstop(box_a):
    """A lane can be dead with the mounts fine — the worker container exited. The backstop's `up -d`
    brings it back; no root shell is needed, so none is opened."""
    box_a.set(images=["alpine:latest"], mounted=NFS, units=BACKSTOP_DONE)
    out = reply(box_a.run("heal", config=HEAL))
    assert out == {"ok": True, "action": "healed", "remounted": [], "unmounted": [], "worker_unit": "restarted"}
    assert box_a.calls() == [], "every mount is up: docker is not even asked for an image"


def test_heal_leaves_a_backstop_that_is_still_waiting_alone(box_a):
    """A RUNNING backstop is mid-wait; it picks the fresh mount up on its next check, and a restart would
    only interrupt its compose up."""
    box_a.set(images=["alpine:latest"], mounted=BUT_HDD, units={UNIT: {"load": "loaded", "sub": "running"}})
    out = reply(box_a.run("heal", config=HEAL))
    assert out["worker_unit"] == "waiting" and out["remounted"] == ["/srv/hdd"]
    assert not [c for c in box_a.systemctl_calls() if "restart" in c]


def test_a_mount_that_cannot_come_back_fails_heal_by_name_and_the_backstop_still_restarts(box_a):
    """The server is down: heal says which mount and in mount's own words. The backstop is restarted all the
    same — it waits for the mount itself, and never deploys onto the stub dir under it."""
    box_a.set(images=["alpine:latest"], mounted=BUT_HDD, unreachable=["/srv/hdd"], units=BACKSTOP_DONE)
    proc = box_a.run("heal", config=HEAL)
    out = reply(proc)
    assert proc.returncode == 1 and out["ok"] is False
    assert out["unmounted"] == ["/srv/hdd"] and out["remounted"] == []
    assert "still not mounted: /srv/hdd" in out["error"] and "Network is unreachable" in out["error"]
    assert out["worker_unit"] == "restarted"


def test_with_no_image_to_get_root_with_heal_says_so_and_pulls_nothing(box_a):
    box_a.set(images=["nvcr.io/nvidia/vllm:25.09"], mounted=BUT_HDD, units=BACKSTOP_DONE)
    proc = box_a.run("heal", config=HEAL)
    out = reply(proc)
    assert proc.returncode == 1 and out["unmounted"] == ["/srv/hdd"]
    assert "no alpine, busybox, debian or ubuntu image on this box" in out["error"]
    assert root_shells(box_a) == [] and out["worker_unit"] == "restarted"


def test_a_box_without_the_worker_unit_fails_heal_by_name(box_a):
    box_a.set(mounted=NFS, units={})
    proc = box_a.run("heal", config=HEAL)
    out = reply(proc)
    assert proc.returncode == 1 and out["worker_unit"] == "absent"
    assert f"{UNIT} is not a user unit here: not-found" in out["error"]


def test_a_backstop_that_will_not_restart_fails_heal_in_systemds_words(box_a):
    box_a.set(mounted=NFS, units=BACKSTOP_DONE, restart_error="Failed to connect to bus: No medium found")
    out = reply(box_a.run("heal", config=HEAL))
    assert out["ok"] is False and out["worker_unit"] == "failed"
    assert "Failed to connect to bus" in out["error"]


@pytest.mark.parametrize("config", [{"CLUSTER_WORKER_UNIT": "fleet-worker"},
                                    {"CLUSTER_WORKER_UNIT": "x.service; reboot"},
                                    {"CLUSTER_HEAL_MOUNT_TIMEOUT_S": "0"},
                                    {"CLUSTER_HEAL_MOUNT_TIMEOUT_S": "20s"}])
def test_a_bad_heal_setting_is_refused_before_anything_runs(box_a, config):
    box_a.set(images=["alpine:latest"], mounted=[], units=BACKSTOP_DONE)
    proc = box_a.run("heal", config={**HEAL, **config})
    assert proc.returncode == 2 and reply(proc)["ok"] is False
    assert box_a.calls() == [] and box_a.systemctl_calls() == []


def test_heal_takes_no_arguments(box_a):
    proc = box_a.run("heal", "now")
    assert proc.returncode == 2 and "heal takes no arguments" in reply(proc)["error"]
    assert box_a.calls() == [] and box_a.systemctl_calls() == []


def test_a_box_with_no_worker_unit_named_heals_its_mounts_and_restarts_nothing(box_a):
    """CLUSTER_WORKER_UNIT is the deployment's own (lb/.env, forwarded by the LB — TP_RANK_SETTINGS): a fleet with
    no broker worker names none, and heal remounts what is missing and says there was no backstop to restart —
    it never restarts a unit nobody named (the old default was one deployment's unit name)."""
    from vllm_lb.cluster_run import TP_RANK_SETTINGS
    assert "CLUSTER_WORKER_UNIT" in TP_RANK_SETTINGS
    box_a.set(images=["alpine:latest"], mounted=BUT_HDD, units=BACKSTOP_DONE)
    out = reply(box_a.run("heal", config={"CLUSTER_HOST_IP": "192.0.2.201"}))
    assert out == {"ok": True, "action": "healed", "remounted": ["/srv/hdd"], "unmounted": [], "worker_unit": "none"}
    assert box_a.systemctl_calls() == []


def test_the_executor_heals_over_ssh_and_says_what_it_did(box_a, ssh):
    box_a.set(images=["alpine:latest"], mounted=BUT_HDD, units=BACKSTOP_DONE)
    ex, host = ssh.executor(env={"CLUSTER_WORKER_UNIT": UNIT})
    assert ex.heal(host) == (True, "remounted /srv/hdd; worker backstop restarted")
    assert ex.last_error["box-a"] == ""
    box_a.set(mounted=NFS, units={UNIT: {"load": "loaded", "sub": "running"}})
    assert ex.heal(host) == (True, "every NFS mount was up; worker backstop waiting")
    ex, host = ssh.executor()                                   # no unit named: none to restart, said so
    assert ex.heal(host) == (True, "every NFS mount was up; no worker backstop named")


def test_a_failed_heal_reaches_the_executors_last_error(box_a, ssh):
    box_a.set(images=["alpine:latest"], mounted=BUT_HDD, unreachable=["/srv/hdd"], units=BACKSTOP_DONE)
    ex, host = ssh.executor()
    ok, why = ex.heal(host)
    assert ok is False and why.startswith("heal: exit 1: still not mounted: /srv/hdd"), why
    assert ex.last_error["box-a"] == why


# ── the GPU, on every status ───────────────────────────────────────────────────
# The gateway's saturation panel is fed from here (vllm_lb/saturation.py). Its whole value is that it
# never states a number the driver did not give, so these tests are about the blanks as much as the
# readings.

def test_status_reports_the_gpu_off_the_drivers_own_dmon_sample(box_a):
    """One `dmon -c 1` plus one `clocks.max.sm`, parsed by dmon's OWN header. `mem_duty_pct` is carried
    as measured — on a GB10 it reads 0 at `sm 96 %`, because the memory is unified with the host's, and
    calling it bandwidth would be the invention this panel exists to avoid."""
    box_a.with_gpu()
    assert reply(box_a.run("status"))["gpu"] == {
        "count": 1, "sm_pct": 96, "mem_duty_pct": 0, "pwr_w": 57, "gtemp_c": 71,
        "pclk_mhz": 2424, "pclk_max_mhz": 3003}


def test_a_dash_column_is_null_never_zero(box_a):
    """dmon prints `-` for a column this card has no value for (mtemp, mclk on a GB10). Neither is in
    the reply, and the ones that ARE must not pick up a 0 from a neighbour."""
    box_a.with_gpu(rows=[{"pwr": 13, "gtemp": 51, "sm": 0, "mem": 0, "pclk": 2405}])
    gpu = reply(box_a.run("status"))["gpu"]
    assert gpu["sm_pct"] == 0 and gpu["pwr_w"] == 13 and gpu["pclk_mhz"] == 2405
    assert "mclk_mhz" not in gpu and "mtemp_c" not in gpu


def test_the_busiest_card_is_the_one_reported_and_the_count_says_how_many(box_a):
    """As the hardware tiles do: a box with one card pinned is not half idle, and the constraint is
    the honest number."""
    box_a.with_gpu(rows=[{"pwr": 20, "gtemp": 50, "sm": 12, "mem": 0, "pclk": 2100},
                         {"pwr": 60, "gtemp": 72, "sm": 97, "mem": 0, "pclk": 2450}])
    gpu = reply(box_a.run("status"))["gpu"]
    assert gpu["count"] == 2 and gpu["sm_pct"] == 97 and gpu["pwr_w"] == 60


def test_a_box_with_no_nvidia_smi_reports_null_and_still_reports_everything_else(box_a):
    """Telemetry is display-only: a box with no driver must cost a line on a page, never the rank and
    memory observations the PLANNER gates on."""
    out = reply(box_a.run("status"))
    assert out["gpu"] is None
    assert out["mem_available_gib"] == 111.6 and out["rank"] == EMPTY_RANK


def test_a_driver_that_fails_costs_the_gpu_line_and_nothing_else(box_a):
    box_a.with_gpu(broken=True)
    box_a.seed(gen=7)
    out = reply(box_a.run("status"))
    assert out["gpu"] is None
    assert out["rank"]["running"] is True and out["mem_available_gib"] == 111.6


def test_the_operator_can_switch_the_gpu_sampling_off(box_a):
    """CLUSTER_GPU_TELEMETRY=0 — it polls two boxes over SSH on every observation, so it has an off
    switch, and nvidia-smi must not run at all when it is thrown."""
    box_a.with_gpu()
    out = reply(box_a.run("status", config={"CLUSTER_HOST_IP": "192.0.2.201",
                                            "CLUSTER_GPU_TELEMETRY": "0"}))
    assert out["gpu"] is None
    assert out["mem_available_gib"] == 111.6



def test_the_prefill_budget_reaches_docker_when_the_operator_sets_it(box_a):
    """CLUSTER_MAX_NUM_BATCHED_TOKENS is a box setting forwarded by the LB (TP_RANK_SETTINGS); it
    must land on the engine argv, and nothing else about the launch may change with it."""
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MAX_NUM_BATCHED_TOKENS": "16384"})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16,
                                          ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home), batched=16384)


@pytest.mark.parametrize("value, flag", [("2048", 2048), ("0", 0), ("", 0)], ids=repr)
def test_the_long_prefill_threshold_reaches_docker_only_when_it_is_on(box_a, value, flag):
    """CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD lands on the engine argv as ONE flag beside the budget's; 0 and empty
    (unset) are vLLM's own default, off, and add nothing — the launch from before the knob."""
    proc = box_a.run("up", "0", "7", "16", "262144",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD": value})
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=7, seqs=16,
                                          ctx=262144, models_dir=box_a.models,
                                          cache_dir=cache_of(box_a.home), threshold=flag)


def test_a_long_prefill_threshold_past_the_context_is_refused_before_docker_runs(box_a):
    """vLLM refuses a threshold past max_model_len (SchedulerConfig) only after both boxes have loaded the model."""
    proc = box_a.run("up", "0", "7", "16", "4096",
                     config={"CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD": "8192"})
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    out = reply(proc)
    assert out["ok"] is False and "CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD" in out["error"], out
    assert box_a.calls() == [], "a launch vLLM would refuse must never reach docker"


# ── the operator's engine knobs: the prefill budget and the drafter's depth ───

#: The SSH command the parent of this change (2154e3e) sent for a DEFAULT launch of Flash-Next's head — the
#: live LB's box settings (compose's CLUSTER_GPU_TELEMETRY and CLUSTER_MAX_NUM_BATCHED_TOKENS), generation 53
#: at 8 seqs and the full window — captured by running that commit's own `SshExecutor`. A launch nobody
#: changed must still be exactly this, byte for byte: tp-rank is unchanged, so its engine argv then is too.
#: THE COMMAND A DEFAULT LAUNCH OF FLASH-NEXT SENDS — the parent commit's, byte for byte, except the ONE
#: value a model's MEASURED-BEST start input now decides: the prefill budget. Flash-Next's TunedSetup says
#: 4,096 (catalog.json, 2026-10-05), so that is what a launch nobody changed carries instead of the box's
#: 8,192; `DENSE_UP_KEEPS_THE_BOX_BUDGET` below is the same proof for a model with nothing measured, where
#: the launch is still the box's own byte for byte. And the profile's PLE placement, which the parent sent as
#: tp-rank's own 1 (replicated) and the profile states as split since 2026-10-11
#: (tests/test_flash_next_ple_split.py): `ple` is that value as tp-rank takes it.
def _parent_default_flash_up(budget: int = 4096, ple: str = "1") -> str:
    return (PARENT_DEFAULT_FLASH_UP_TEMPLATE
            .replace("CLUSTER_MAX_NUM_BATCHED_TOKENS=8192", f"CLUSTER_MAX_NUM_BATCHED_TOKENS={budget}")
            .replace("CLUSTER_PLE_REPLICATE=1", f"CLUSTER_PLE_REPLICATE={ple}"))


PARENT_DEFAULT_FLASH_UP_TEMPLATE = (
    "env CLUSTER_PORT=8100 CLUSTER_MASTER_ADDR=192.0.2.201 CLUSTER_IFACE=bond0 CLUSTER_GPU_TELEMETRY=1 "
    "CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 CLUSTER_IMAGE=myllmbox/qwen38-flash-next-cluster-vllm:v5 "
    "CLUSTER_MODELS_DIRS='~/models/qwen38-flash-next-cluster:/mnt/box-a-home/models/qwen38-flash-next-cluster:"
    "/srv/models/qwen38-flash-next-cluster' CLUSTER_MODEL_NAME=Qwen3.8-Flash-Next-hibrid48 "
    "CLUSTER_SERVED_MODEL_NAME=Qwen/Qwen3.8-Flash-Next CLUSTER_GPU_MEMORY_UTILIZATION=0.70 "
    "CLUSTER_KV_CACHE_MEMORY=28000000000 CLUSTER_TOOL_CALL_PARSER=qwen3_xml CLUSTER_REASONING_PARSER=qwen3 "
    "CLUSTER_SPECULATIVE_CONFIG='{\"method\":\"mtp\",\"num_speculative_tokens\":4}' CLUSTER_MTP_PREFIX=mtp. "
    "CLUSTER_KV_CACHE_DTYPE=fp8_e4m3 "
    "CLUSTER_PLE_REPLICATE=1 CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 53 8 262144 qwen3.8-flash-next-hibrid48")
def _flash_tuned_batched() -> int:
    """Flash-Next's own measured-best prefill budget, READ FROM THE SHIPPED CATALOG rather than repeated here:
    these tests must fail when the tuned figure changes, not quietly assert a stale copy of it."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    return ModelCatalog.load().get(FLASH).profile(CLUSTER_MODE).tuned.max_num_batched_tokens


def _flash_ple() -> bool:
    """Flash-Next's own PLE placement, read from the shipped catalog for the same reason as the budget."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    return ModelCatalog.load().get(FLASH).profile(CLUSTER_MODE).ple_replicate


def _flash_kv_dtype() -> str | None:
    """Flash-Next's own KV dtype flag, read from the shipped catalog for the same reason (fp8_e4m3 since 2026-10-26)."""
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    return ModelCatalog.load().get(FLASH).profile(CLUSTER_MODE).kv_cache_dtype


FLASH_TUNED_BATCHED = _flash_tuned_batched()
FLASH_KV_DTYPE = _flash_kv_dtype()
FLASH_PLE_REPLICATE = _flash_ple()
#: …as tp-rank takes it (CLUSTER_PLE_REPLICATE, MBX_PLE_REPLICATE and the cache key's `ple<mode>`).
FLASH_PLE = "1" if FLASH_PLE_REPLICATE else "0"
PARENT_DEFAULT_FLASH_UP = _parent_default_flash_up(FLASH_TUNED_BATCHED, FLASH_PLE)
LIVE_BOX_ENV = {"CLUSTER_PORT": "8100", "CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_IFACE": "bond0",
                "CLUSTER_GPU_TELEMETRY": "1",
                "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"}


SGLANG = "qwen3.8-27b-sglang-dflash2"


def test_an_sglang_profile_launches_sglangs_own_server_on_the_same_port(box_a, ssh):
    """THE TEST THAT WAS MISSING, and the bug it would have caught. The vLLM path has 37 assertions on
    its exact argv; the SGLang grammar had none that ran it, and its first execution on real hardware
    died with `main: line 1012: seats: unbound variable` — `cmd_up`'s local is `seqs`, and the call
    site passed `$seats`. Under `set -u` that is exit 1 before docker is ever reached, so both ranks
    failed on both boxes and the LB retried every two seconds with the pair never starting.

    What this asserts is that the launch REACHES docker and is SGLang's own command, on the same
    CLUSTER_PORT vLLM's head uses — which is what keeps :8010 answering whichever engine is behind it."""
    from vllm_lb.catalog import ModelCatalog, document
    spec = ModelCatalog.from_dict(document()).get(SGLANG)
    # Its checkpoint and its drafter sit at the models root, beside each other.
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=71, max_num_seqs=16, max_model_len=262144, spec=spec)
    assert ok is True, msg

    run = last_run(box_a)
    assert "python3" in run and "sglang.launch_server" in run, run
    flags = dict(zip(run, run[1:]))
    assert flags["--model-path"] == f"/models/{spec.name}"
    assert flags["--port"] == "8100", "the same internal port vLLM's head serves, so :8010 is unchanged"
    assert flags["--tp"] == "2" and flags["--nnodes"] == "2" and flags["--node-rank"] == "0"
    # TEN GDN state slots a seat: five a running request (at four the engine silently caps the seats)
    # and five checkpoints for the prefix cache.
    assert flags["--max-mamba-cache-size"] == "160", "16 seats x 10 slots"
    assert flags["--max-running-requests"] == "16" and flags["--context-length"] == "262144"
    # 0.8, not "0.80": the catalog's number goes through json.dumps on its way to CLUSTER_ENGINE_ARGS.
    assert float(flags["--mem-fraction-static"]) == 0.80, "0.90 killed rank 1 on DFlash2's second KV pool"
    assert flags["--speculative-algorithm"] == "DFLASH"
    assert flags["--speculative-draft-model-path"] == f"/models/{spec.draft_name}"
    assert flags["--attention-backend"] == "flashinfer", "SM121; trtllm_mha is SM100-only"
    assert "--headless" not in run, "rank 0 serves the API"
    # vLLM's own flags must not leak into another engine's command line.
    for vllm_only in ("--max-num-seqs", "--gpu-memory-utilization", "--kv-cache-memory", "--model"):
        assert vllm_only not in run, f"{vllm_only} is vLLM's, not SGLang's"


def test_an_sglang_rank_one_is_headless_by_its_node_rank_not_by_a_flag(box_a, ssh):
    """vLLM takes `--headless` for the rank that serves no API; SGLang's headless node is simply the one
    whose node-rank is not 0, and passing it the flag would fail the launch."""
    from vllm_lb.catalog import ModelCatalog, document
    spec = ModelCatalog.from_dict(document()).get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=1, gen=71, max_num_seqs=16, max_model_len=262144, spec=spec)
    assert ok is True, msg
    run = last_run(box_a)
    assert "--headless" not in run
    assert dict(zip(run, run[1:]))["--node-rank"] == "1"


def test_every_start_schedules_by_the_requests_priority(box_a, ssh):
    """THE CALLER'S PRIORITY (router.PRIORITY_HEADER, lower = sooner) is read only where the engine was STARTED to:
    vLLM under --scheduling-policy priority (fcfs ignores it), SGLang under --enable-priority-scheduling — with lower
    values first, vLLM's direction, and 0 for a request naming none (SGLang's own default would put an unlabelled
    request LAST). Both engines ignore the field silently otherwise, so every start asks."""
    from vllm_lb.catalog import ModelCatalog, document
    spec = ModelCatalog.from_dict(document()).get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)
    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=74, max_num_seqs=16, max_model_len=262144, spec=spec)
    assert ok is True, msg
    run = last_run(box_a)
    assert "--enable-priority-scheduling" in run and "--schedule-low-priority-values-first" in run
    assert dict(zip(run, run[1:]))["--default-priority-value"] == "0"


def test_an_sglang_start_asks_for_the_mfu_counters_the_compute_and_bandwidth_tiles_need(box_a, ssh):
    """THE SECOND HALF OF A BLANK TILE. The gateway's COMPUTE and MEMORY BANDWIDTH tiles are drawn from
    the engine's own `estimated_*_per_gpu_total` counters, and SGLang publishes the same three vLLM does
    — but only with `--enable-mfu-metrics`, which it defaults to off and without which it does not
    register the families at all. So the probe knowing the names is not enough: the START has to ask,
    exactly as `cmd_up` asks for the vLLM path.

    One switch for both engines: `CLUSTER_MFU_METRICS=0` opts out of the flag on either."""
    from vllm_lb.catalog import ModelCatalog, document
    spec = ModelCatalog.from_dict(document()).get(SGLANG)
    models = box_a.place_checkpoint(box_a.home / "models", name=spec.name)
    box_a.place_checkpoint(models, name=spec.draft_name)

    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models)})
    ok, msg = ex.up(host, rank=0, gen=72, max_num_seqs=16, max_model_len=262144, spec=spec)
    assert ok is True, msg
    run = last_run(box_a)
    assert "--enable-mfu-metrics" in run, "the tiles have no source without it"
    # And its prerequisite: inside the engine the MFU switch is read only under `--enable-metrics`
    # (metrics_reporter.py), so one without the other publishes nothing.
    assert "--enable-metrics" in run, "--enable-mfu-metrics is gated on it inside SGLang"

    ex, host = ssh.executor(env={**LIVE_BOX_ENV, "CLUSTER_MODELS_DIRS": str(models),
                                 "CLUSTER_MFU_METRICS": "0"})
    ok, msg = ex.up(host, rank=0, gen=73, max_num_seqs=16, max_model_len=262144, spec=spec)
    assert ok is True, msg
    run = last_run(box_a)
    assert "--enable-mfu-metrics" not in run, "the same opt-out the vLLM path honours"
    assert "sglang.launch_server" in run, "and the launch is otherwise unchanged"


def _flash_and_dense():
    """Flash-Next, and the dense model with its drafter taken away: a LaunchProfile with NO speculative decoding.
    The shipped catalog has none any more — NVIDIA's 27B build carries its MTP head — and what a launch does
    for a model without a drafter still has to hold."""
    from vllm_lb.catalog import ModelCatalog, document
    doc = document()
    next(m for m in doc["models"] if m["id"] == DENSE)["launch"]["cluster"].update(speculative_config=None,
                                                                                 mtp_prefix="")
    catalog = ModelCatalog.from_dict(doc)
    return catalog.get(FLASH), catalog.get(DENSE)


@pytest.mark.parametrize("knobs", [
    {},
    {"max_num_batched_tokens": None, "num_speculative_tokens": None},
    # THE POOL KNOBS AT THEIR DEFAULTS, the pass that made them settable: null means "as the model says",
    # and an explicit 0 of disk means OFF — neither may add a byte to the command.
    {"max_num_batched_tokens": None, "num_speculative_tokens": None,
     "ple_replicate": None, "kv_cache_memory": None, "kv_offload_gib": None},
    {"ple_replicate": FLASH_PLE_REPLICATE, "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0},
    # THE LONG-PREFILL THRESHOLD AT ITS DEFAULT: null and 0 are both vLLM's own, off — not a byte more.
    {"long_prefill_token_threshold": None},
    {"long_prefill_token_threshold": 0},
    {"max_num_batched_tokens": None, "num_speculative_tokens": None, "ple_replicate": None,
     "kv_cache_memory": None, "kv_offload_gib": None, "long_prefill_token_threshold": 0},
], ids=["knobs-left-out", "knobs-null", "pool-knobs-null", "pool-knobs-at-the-models-own-values",
        "threshold-null", "threshold-0", "every-knob-at-its-default"])
def test_a_launch_nobody_changed_is_sent_byte_for_byte_as_before(box_a, ssh, knobs):
    """The operator's knobs at their defaults add NOTHING OF THE OPERATOR'S: the command is the parent commit's
    but for the model's own measured-best prefill budget (`catalog.TunedSetup`, which the default resolves to
    exactly as the pool and the PLE placement already did) and its own PLE placement (split since 2026-10-11),
    and the rank it starts has the engine argv the launch has always had (`launch_argv`, every default spelled
    out)."""
    flash, _ = _flash_and_dense()
    ex, host = ssh.executor(env=dict(LIVE_BOX_ENV))
    ok, msg = ex.up(host, rank=0, gen=53, max_num_seqs=8, max_model_len=262144, spec=flash, **knobs)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == PARENT_DEFAULT_FLASH_UP
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=53, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home, ple=FLASH_PLE),
                                          model_choice=FLASH, batched=FLASH_TUNED_BATCHED, ple=FLASH_PLE,
                                          kv_dtype=FLASH_KV_DTYPE)


def _up_from_switch(ex, host, path: Path, spec):
    """An Up exactly as the loop carries one from the switch on disk (`ClusterLoop._apply`, `cluster.Up`)."""
    from vllm_lb.cluster import DesiredStore

    d = DesiredStore(str(path)).load()
    return ex.up(host, rank=0, gen=d.generation, max_num_seqs=d.max_num_seqs, max_model_len=d.max_model_len,
                 spec=spec, max_num_batched_tokens=d.max_num_batched_tokens,
                 num_speculative_tokens=d.num_speculative_tokens, ple_replicate=d.ple_replicate,
                 kv_cache_memory=d.kv_cache_memory, kv_offload_gib=d.kv_offload_gib,
                 long_prefill_token_threshold=d.long_prefill_token_threshold)


@pytest.mark.parametrize("stored", [{}, {"kv_cache_auto": False}, {"long_prefill_token_threshold": None},
                                    {"long_prefill_token_threshold": 0}],
                         ids=["switch-from-before-auto", "switch-auto-never-chosen", "switch-threshold-null",
                              "switch-threshold-0"])
def test_a_switch_that_never_chose_auto_launches_byte_for_byte_as_before(box_a, ssh, tmp_path, stored):
    """AUTO ADDS NOTHING TO A LAUNCH THAT DID NOT CHOOSE IT. The default switch — one written before the pool could
    be `auto`, and one written since with it off — read back and carried to tp-rank as the loop carries an Up, is
    the parent's command byte for byte, and the rank it starts has the engine argv it always had."""
    flash, _ = _flash_and_dense()
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps({"mode": "cluster", "generation": 53, "max_num_seqs": 8, "max_model_len": 262144,
                                "model_choice": FLASH, **stored}))
    ex, host = ssh.executor(env=dict(LIVE_BOX_ENV))
    ok, msg = _up_from_switch(ex, host, path, flash)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == PARENT_DEFAULT_FLASH_UP
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=53, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home, ple=FLASH_PLE),
                                          model_choice=FLASH, batched=FLASH_TUNED_BATCHED, ple=FLASH_PLE,
                                          kv_dtype=FLASH_KV_DTYPE)


def test_an_untuned_model_launches_with_the_boxs_own_budget_byte_for_byte(box_a, ssh):
    """AND NOTHING IS ADDED WHERE NOTHING WAS MEASURED. The dense 27B has no TunedSetup, so a launch of it with
    every knob at its default carries the BOX's prefill budget exactly as it did before tuned defaults existed —
    the guard the test above used to be, kept for the untuned path."""
    _, dense = _flash_and_dense()
    assert dense.profile("cluster").tuned.max_num_batched_tokens is None, "the premise: no budget measured for it"
    FakeBox.place_checkpoint(box_a.home / "models", name=dense.name)
    ex, host = ssh.executor(env=dict(LIVE_BOX_ENV))
    ok, msg = ex.up(host, rank=0, gen=53, max_num_seqs=8, max_model_len=262144, spec=dense)
    assert ok is True, msg
    assert f"CLUSTER_MAX_NUM_BATCHED_TOKENS={int(LIVE_BOX_ENV['CLUSTER_MAX_NUM_BATCHED_TOKENS'])}" \
        in ssh.calls()[-1][-1]
    assert last_run(box_a)[last_run(box_a).index("--max-num-batched-tokens") + 1] \
        == LIVE_BOX_ENV["CLUSTER_MAX_NUM_BATCHED_TOKENS"]


def test_an_auto_pool_reaches_tp_rank_as_the_number_it_resolved_to_never_the_word(ssh, tmp_path):
    """A pool sized by `auto` is launched exactly as the same size typed: the switch stores the number, and the
    choice that it was sized never travels — tp-rank has no "auto", and must never be sent one."""
    flash, _ = _flash_and_dense()
    commands = []
    for extra in ({}, {"kv_cache_auto": True}):
        path = tmp_path / f"cluster-{len(commands)}.json"
        path.write_text(json.dumps({"mode": "cluster", "generation": 100, "max_num_seqs": 16,
                                    "max_model_len": 262144, "model_choice": FLASH, "max_num_batched_tokens": 4096,
                                    "ple_replicate": False, "kv_cache_memory": 36_000_000_000, **extra}))
        ex, host = ssh.executor(env=dict(LIVE_BOX_ENV))
        ok, msg = _up_from_switch(ex, host, path, flash)
        assert ok is True, msg
        commands.append(ssh.calls()[-1][-1])
    typed, sized = commands
    assert sized == typed
    assert " CLUSTER_KV_CACHE_MEMORY=36000000000 " in sized and "auto" not in sized.lower()


def test_the_operators_drafter_depth_replaces_only_the_profiles_count():
    """ONE override of a model's speculation, and a narrow one: the count. The method and every other key stay
    the catalog's; 0 sends no drafter at all; the catalog's own 4 asked for explicitly is the default's bytes."""
    from vllm_lb.cluster_exec import launch_env

    flash, dense = _flash_and_dense()
    default = launch_env(flash)
    assert launch_env(flash, num_speculative_tokens=4) == default
    two = launch_env(flash, num_speculative_tokens=2)
    assert two == {**default, "CLUSTER_SPECULATIVE_CONFIG": '{"method":"mtp","num_speculative_tokens":2}'}
    assert launch_env(flash, num_speculative_tokens=0) == {**default, "CLUSTER_SPECULATIVE_CONFIG": ""}
    assert launch_env(dense, num_speculative_tokens=0) == launch_env(dense), "off is what a model with no drafter runs"
    with pytest.raises(ValueError, match="no speculative decoding"):
        launch_env(dense, num_speculative_tokens=2)


def test_the_operators_pool_knobs_replace_only_what_they_name():
    """The PLE placement and the KV pin override the profile's and nothing else; the disk tier adds its two
    settings only when it is ON, so OFF is the launch from before it existed."""
    from vllm_lb.cluster_exec import launch_env, offload_yaml

    flash, _ = _flash_and_dense()
    default = launch_env(flash)
    assert default["CLUSTER_PLE_REPLICATE"] == "0" and default["CLUSTER_KV_CACHE_MEMORY"] == "28000000000"
    assert "CLUSTER_KV_OFFLOAD_GIB" not in default, "the disk tier is off until it is asked for"

    assert launch_env(flash, ple_replicate=False) == default, "the model's own placement, asked for explicitly"
    assert launch_env(flash, ple_replicate=True) == {**default, "CLUSTER_PLE_REPLICATE": "1"}
    assert launch_env(flash, kv_cache_memory=36_000_000_000) == {**default,
                                                                 "CLUSTER_KV_CACHE_MEMORY": "36000000000"}
    assert launch_env(flash, kv_offload_gib=0) == default, "0 is off, and off adds nothing"
    assert launch_env(flash, kv_offload_gib=None) == default

    on = launch_env(flash, kv_offload_gib=64)
    assert on == {**default, "CLUSTER_KV_OFFLOAD_GIB": "64", "CLUSTER_KV_OFFLOAD_YAML": offload_yaml(64)}
    # The one line that makes this safe on a GB10: a CPU tier would take from the very pool it caches.
    assert "local_cpu: false" in on["CLUSTER_KV_OFFLOAD_YAML"]
    assert "max_local_disk_size: 64" in on["CLUSTER_KV_OFFLOAD_YAML"]


def test_a_drafter_depth_reaches_the_engine_and_off_passes_no_speculative_config(box_a, ssh):
    flash, _ = _flash_and_dense()
    ex, host = ssh.executor()
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=262144, spec=flash, num_speculative_tokens=2)
    assert ok is True, msg
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=3, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home, ple=FLASH_PLE),
                                          model_choice=FLASH, batched=FLASH_TUNED_BATCHED, ple=FLASH_PLE,
                                          kv_dtype=FLASH_KV_DTYPE,
                                          spec='{"method":"mtp","num_speculative_tokens":2}')
    ok, msg = ex.up(host, rank=0, gen=4, max_num_seqs=8, max_model_len=262144, spec=flash, num_speculative_tokens=0)
    assert ok is True, msg
    with_drafter = launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=4, seqs=8, ctx=262144,
                               models_dir=box_a.models, cache_dir=cache_of(box_a.home, ple=FLASH_PLE),
                               model_choice=FLASH, batched=FLASH_TUNED_BATCHED, ple=FLASH_PLE,
                                          kv_dtype=FLASH_KV_DTYPE)
    assert with_drafter[-4:-2] == ["--speculative-config", SPEC]
    assert last_run(box_a) == with_drafter[:-4] + with_drafter[-2:], "speculation off is no --speculative-config at all"


def test_the_operators_prefill_budget_replaces_the_boxs_for_that_launch_only(box_a, ssh):
    """The box's CLUSTER_MAX_NUM_BATCHED_TOKENS is the DEFAULT; the operator's value replaces it in the one
    call that launches — named once on the command line, never twice — and the next default launch is the
    box's again. Nothing else on the engine argv moves."""
    ex, host = ssh.executor(env={"CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_IFACE": "bond0", "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"})
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=16384)
    assert ok is True, msg
    command = ssh.calls()[-1][-1]
    assert command == "env CLUSTER_MASTER_ADDR=192.0.2.201 CLUSTER_IFACE=bond0 CLUSTER_MAX_NUM_BATCHED_TOKENS=16384 CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 3 8 262144"
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=3, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home), batched=16384)
    ok, msg = ex.up(host, rank=0, gen=4, max_num_seqs=8, max_model_len=262144)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == "env CLUSTER_MASTER_ADDR=192.0.2.201 CLUSTER_IFACE=bond0 CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 4 8 262144"
    assert ex.env == {"CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_IFACE": "bond0", "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"}, "the box's own setting was rewritten"


def test_the_operators_long_prefill_threshold_rides_its_launch_only_when_it_is_on(box_a, ssh):
    """The threshold rides as ONE setting of the call that launches, beside the budget's, and lands as ONE flag
    on the engine argv; the next launch at 0 is the default's command again, byte for byte."""
    ex, host = ssh.executor(env={"CLUSTER_MASTER_ADDR": "192.0.2.201", "CLUSTER_IFACE": "bond0", "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192"})
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=262144, max_num_batched_tokens=8192,
                    long_prefill_token_threshold=2048)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == ("env CLUSTER_MASTER_ADDR=192.0.2.201 CLUSTER_IFACE=bond0 CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD=2048 "
                                   "CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 3 8 262144")
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=3, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home), threshold=2048)
    ok, msg = ex.up(host, rank=0, gen=4, max_num_seqs=8, max_model_len=262144, long_prefill_token_threshold=0)
    assert ok is True, msg
    assert ssh.calls()[-1][-1] == "env CLUSTER_MASTER_ADDR=192.0.2.201 CLUSTER_IFACE=bond0 CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 CLUSTER_HOST_IP=192.0.2.201 bash -s -- up 0 4 8 262144"
    assert last_run(box_a) == launch_argv(mfu_patch_dir=mfu_of(box_a), rank=0, gen=4, seqs=8, ctx=262144,
                                          models_dir=box_a.models, cache_dir=cache_of(box_a.home))


@pytest.mark.parametrize("knobs, words", [
    ({"long_prefill_token_threshold": -1}, "long_prefill_token_threshold must be an integer in 0..65536"),
    ({"long_prefill_token_threshold": 65537}, "long_prefill_token_threshold must be an integer in 0..65536"),
    ({"long_prefill_token_threshold": True}, "long_prefill_token_threshold must be an integer"),
    ({"long_prefill_token_threshold": 2048, "max_model_len": 1024}, "cannot be greater than max_model_len"),
    ({"max_num_batched_tokens": 1024}, "max_num_batched_tokens must be an integer in 2048..65536"),
    ({"max_num_batched_tokens": 65537}, "max_num_batched_tokens must be an integer in 2048..65536"),
    ({"max_num_batched_tokens": True}, "max_num_batched_tokens must be an integer"),
    ({"num_speculative_tokens": -1}, "num_speculative_tokens must be an integer in 0..8"),
    ({"num_speculative_tokens": 9}, "num_speculative_tokens must be an integer in 0..8"),
    ({"num_speculative_tokens": 2, "spec": None}, "needs a ModelChoice"),
    ({"num_speculative_tokens": 2, "spec": "dense"}, "no speculative decoding"),
], ids=repr)
def test_knobs_a_launch_cannot_take_are_refused_before_dialing(ssh, knobs, words):
    flash, dense = _flash_and_dense()
    knobs = dict(knobs)
    spec = {"dense": dense, None: None}.get(knobs.pop("spec", "flash"), flash)
    ctx = knobs.pop("max_model_len", 262144)
    ex, host = ssh.executor()
    ok, msg = ex.up(host, rank=0, gen=3, max_num_seqs=8, max_model_len=ctx, spec=spec, **knobs)
    assert ok is False and words in msg, msg
    assert ssh.calls() == [], "a launch that cannot happen dialed a GB10 box"


def test_the_resolution_follows_the_precedence_operator_then_box_or_catalog_then_tp_rank():
    """`cluster.effective` says what a launch RESOLVED to, so the resolution must be the launch's own:
    the operator's stored value, else the MODEL's measured-best (`catalog.TunedSetup`, since 2026-10-05),
    else the box's setting (prefill budget) or the catalog's LaunchProfile (speculation), else tp-rank's own
    default — which a box setting tp-rank would refuse never is."""
    from vllm_lb.cluster_exec import resolved_launch

    flash, dense = _flash_and_dense()

    def resolve(spec, env, **knobs):
        return resolved_launch(spec, env, max_num_seqs=8, max_model_len=262144, **knobs)

    assert resolve(flash, {}) == {"max_num_seqs": 8, "max_model_len": 262144,
                                  "max_num_batched_tokens": FLASH_TUNED_BATCHED,
                                  "num_speculative_tokens": 4, "ple_replicate": FLASH_PLE_REPLICATE, "kv_cache_memory": 28_000_000_000, "kv_offload_gib": 0,
                                  "long_prefill_token_threshold": 0,
                                  # None launched the original, exactly as `up` reads it; the switch resolves the default.
                                  "uncensored": False, "kv_cache_dtype": "fp8_e4m3"}
    assert resolve(flash, {}, uncensored=True)["uncensored"] is True
    assert resolve(flash, {}, long_prefill_token_threshold=2048)["long_prefill_token_threshold"] == 2048
    assert resolve(flash, {}, long_prefill_token_threshold=None)["long_prefill_token_threshold"] == 0, \
        "null is vLLM's own default: off"
    # A MEASURED BUDGET BEATS THE BOX'S for the model it was measured on, and the box's still stands for a
    # model with none measured — the dense 27B, whose profile states no tuned setup at all.
    assert resolve(flash, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "16384"})["max_num_batched_tokens"] \
        == FLASH_TUNED_BATCHED, "the model's own measurement wins over the box's one number"
    assert dense.profile("cluster").tuned.max_num_batched_tokens is None
    assert resolve(dense, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "16384"})["max_num_batched_tokens"] == 16384
    assert resolve(dense, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": ""})["max_num_batched_tokens"] is None, "no number: none"
    assert resolve(dense, {})["max_num_batched_tokens"] is None, "the fleet's budget travels in the env, or nothing does"
    for refused in ("16k", "1024", "131072", " 8192", "08192"):
        assert resolve(dense, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": refused})["max_num_batched_tokens"] is None, refused
    assert resolve(flash, {"CLUSTER_MAX_NUM_BATCHED_TOKENS": "16384"}, max_num_batched_tokens=16384)[
        "max_num_batched_tokens"] == 16384, "the operator's value wins over the box's and the model's"
    assert resolve(dense, {})["num_speculative_tokens"] == 0, "a model with no drafter"
    from vllm_lb.catalog import ModelCatalog, document
    shipped_dense = resolve(ModelCatalog.load().get(DENSE), {})
    assert (shipped_dense["num_speculative_tokens"], shipped_dense["ple_replicate"],
            shipped_dense["kv_cache_memory"]) == (4, None, 48_000_000_000), "the shipped dense model's own profile"
    assert resolve(flash, {}, num_speculative_tokens=0)["num_speculative_tokens"] == 0
    assert resolve(None, {})["num_speculative_tokens"] == 4, "no ModelChoice: tp-rank's own MTP_DRAFTER"


def test_the_mirrored_tp_rank_defaults_are_tp_ranks_own():
    """The LB can only say what a rank was launched with if it knows what tp-rank does with nothing."""
    from vllm_lb import cluster_exec

    script = TP_RANK.read_text()
    # THE PREFILL BUDGET HAS NO DEFAULT ANYWHERE (2026-10-29): the LB sends the fleet's on every call, tp-rank refuses an
    # `up` without one by name, and the LB mirrors no number for it.
    assert "CLUSTER_MAX_NUM_BATCHED_TOKENS:=" not in script
    assert not hasattr(cluster_exec, "TP_RANK_MAX_NUM_BATCHED_TOKENS")
    assert 'CLUSTER_MAX_NUM_BATCHED_TOKENS must be set' in script
    lo, hi = cluster_exec.MAX_NUM_BATCHED_TOKENS_RANGE
    assert f'int_in "$CLUSTER_MAX_NUM_BATCHED_TOKENS" {lo} {hi}' in script
    assert ': "${CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD:=0}"' in script, "off unless the LB sends it"
    lo, hi = cluster_exec.LONG_PREFILL_TOKEN_THRESHOLD_RANGE
    assert f'int_in "$CLUSTER_LONG_PREFILL_TOKEN_THRESHOLD" {lo} {hi}' in script
    drafter = re.search(r"^readonly MTP_DRAFTER='(\{.*\})'$", script, re.M).group(1)
    assert json.loads(drafter)["num_speculative_tokens"] == cluster_exec.TP_RANK_NUM_SPECULATIVE_TOKENS


def test_an_up_without_a_prefill_budget_is_refused_by_name(box_a):
    """tp-rank defaults nothing the fleet owns: with no CLUSTER_MAX_NUM_BATCHED_TOKENS sent, `up` fails naming it."""
    FakeBox.place_checkpoint(box_a.home / "models", name="Qwen3.8-Flash-Next-hibrid48")
    cfg = {**LIVE_BOX_ENV, "CLUSTER_HOST_IP": "192.0.2.201", "CLUSTER_MAX_NUM_BATCHED_TOKENS": ""}
    proc = box_a.run("up", "0", "7", "8", "262144", "qwen3.8-flash-next-hibrid48", config=cfg)
    assert proc.returncode == 1 and "CLUSTER_MAX_NUM_BATCHED_TOKENS must be set" in proc.stdout, proc.stdout
