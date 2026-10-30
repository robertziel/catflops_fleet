"""Fleet files the tests are written against (vllm_lb/fleet.py), in generic names — one per shape that runs today.

  PAIR      the box-c shape: a TP=2 pair of GB10s the LB drives over SSH, a classic server on each of the pair's
            boxes, and a box that only sends telemetry (an AMD iGPU). The env it replaces is ENV_OF_PAIR.
  ONE_HOST  the site-b shape: one box with two different cards, the second of which serves a one-host cluster the
            LB routes to but does not start.
"""
from __future__ import annotations

from pathlib import Path

PAIR = """
boxes:
  node-1:
    hostname: node-1-host
    label: node-1-gpu
    platform: linux/arm64
    ssh: operator@192.0.2.10
    cards:
      - {index: 0, type: gb10, serves: true}
  node-2:
    hostname: node-2-host
    label: node-2-gpu
    platform: linux/arm64
    ssh: operator@192.0.2.11
    cards:
      - {index: 0, type: gb10, serves: true}
  gateway:
    hostname: gateway-host
    platform: linux/amd64
    cards:
      - {index: 0, type: radeon8060s, serves: false}
clusters:
  pair:
    control: ssh
    iface: bond0
    link: roce
    url: http://192.0.2.10:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks:
      - {box: node-1, interconnect: 198.51.100.1}
      - {box: node-2, interconnect: 198.51.100.2}
backends:
  node-1: {url: http://192.0.2.10:8000}
  node-2: {url: http://192.0.2.11:8000}
"""

#: The four retired settings exactly as an LB configured for PAIR carried them before the fleet file.
ENV_OF_PAIR = {
    "LB_CLUSTER_HOSTS": "node-1-gpu=node-1=operator@192.0.2.10=198.51.100.1,"
                        "node-2-gpu=node-2=operator@192.0.2.11=198.51.100.2",
    "LB_CLUSTER_URL": "http://192.0.2.10:8100",
    "LB_VLLM_BOXES": "node-1-gpu=node-1=http://192.0.2.10:8000,node-2-gpu=node-2=http://192.0.2.11:8000",
    "LB_HW_HOSTS": "node-1-host=node-1,node-2-host=node-2,gateway-host=gateway",
    "CLUSTER_MAX_NUM_BATCHED_TOKENS": "8192",
}

ONE_HOST = """
boxes:
  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    node_limits: {memory_gib: 24, cpus: 16, pids: 4096, nice: 10}
    cards:
      - {index: 0, type: rtx3090, serves: false}
      - {index: 1, type: rtx5090, serves: true}
clusters:
  node:
    control: none
    model: qwen3.8-27b-nvidia-nvfp4@uncensored
    url: http://vllm-node:8000
    ranks:
      - {box: solo}
backends: {}
"""


#: The PARALLEL shape (worklog/parallel-clusters.md): the pair AND an independent one-host node the LB requests on a
#: third box — two clusters, two engines, two models, one LB. The node's card is one whose type already lists `local`.
#: The pair is the DEFAULT cluster: a request naming no model, or a loose name, goes to it alone.
PARALLEL = PAIR.replace("""  gateway:
    hostname: gateway-host
    platform: linux/amd64
    cards:
      - {index: 0, type: radeon8060s, serves: false}
""", """  gateway:
    hostname: gateway-host
    platform: linux/amd64
    cards:
      - {index: 0, type: radeon8060s, serves: false}
  solo:
    hostname: solo-host
    label: solo-gpu
    platform: linux/amd64
    node_limits: {memory_gib: 24, cpus: 16, pids: 4096, nice: 10}
    cards:
      - {index: 0, type: rtx5090, serves: true}
""").replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n    default: true\n").replace("backends:", f"""  gui:
    control: local
    run_at_boot: true
    model: qwen3.8-27b-nvidia-nvfp4
    url: http://198.51.100.20:8000
    launch:
      image: vllm/vllm-openai@sha256:{'1' * 64}
      max_model_len: 131072
      max_num_seqs: 8
      gpu_memory_utilization: 0.9
      reasoning_parser: qwen3
      tool_call_parser: qwen3_coder
      choices: [qwen3.8-27b-nvidia-nvfp4]
    ranks:
      - {{box: solo}}
backends:""")


#: The box-c shape since 2026-10-24: a one-host node the LB requests (control local) on node-2, which also carries
#: the pair's rank and so its ssh.
LOCAL_ON_PAIR_BOX = PAIR.replace("""    ssh: operator@192.0.2.11
""", """    ssh: operator@192.0.2.11
    node_limits: {memory_gib: 24, cpus: 8, pids: 4096, nice: 10}
    gpu_ram_max_gib: 12
""").replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n").replace("backends:", """  speech:
    control: local
    run_at_boot: true
    model: parakeet-tdt-0.6b-v3
    url: http://192.0.2.11:18000
    launch:
      image: sha256:""" + "2" * 64 + """
      max_model_len: 1
      max_num_seqs: 4
      gpu_memory_utilization: 0.08
      choices: [parakeet-tdt-0.6b-v3]
    reserve: false
    ranks:
      - {box: node-2}
backends:""")


#: The reference fleet's `checkpoints` block (lb/fleet.example.yml): where its models are staged.
REFERENCE_CHECKPOINTS = "checkpoints:" + (Path(__file__).resolve().parent.parent / "fleet.example.yml").read_text() \
    .split("\ncheckpoints:", 1)[1]


def write(directory: Path, text: str, name: str = "fleet.yml") -> str:
    """`text` as a fleet file in `directory`; its path, as LB_FLEET takes it. A fleet that names no `checkpoints` stages
    the reference fleet's (as tests/conftest.py `reference_checkpoints` reads one in-process): a file is also read by
    processes the suite starts, which that fixture does not reach."""
    path = Path(directory) / name
    if "\ncheckpoints:" not in "\n" + text:
        text = text.rstrip("\n") + "\n" + REFERENCE_CHECKPOINTS
    path.write_text(text)
    return str(path)


def one_box(url: str) -> str:
    """A fleet file naming ONE classic box (`box-a`, lane `box-a-gpu`, one GB10) serving at `url`, in a fresh
    temporary directory — what a test that boots the LB on one fake server needs; its path, as LB_FLEET takes it."""
    import tempfile
    return write(Path(tempfile.mkdtemp()), f"""
boxes:
  box-a: {{hostname: box-a-host, label: box-a-gpu, platform: linux/arm64, cards: [{{index: 0, type: gb10, serves: true}}]}}
clusters: {{}}
backends:
  box-a: {{url: "{url}"}}
""")


def gb10_pair(a: str = "box-a", b: str = "box-b"):
    """A parsed fleet of two GB10 boxes `a` and `b` under one ssh-controlled cluster, head `a` — what a test that
    builds its Config by hand (`cluster_hosts=parse_hosts(...)`) passes as `fleet=`, so the pair's cards (their
    CUDA cores, their peaks) come from where the LB takes them: the fleet's serving cards' hardware types."""
    from vllm_lb.fleet import parse_fleet
    card = "[{index: 0, type: gb10, serves: true}]"
    return parse_fleet(f"""
boxes:
  {a}: {{hostname: {a}-host, label: {a}-gpu, platform: linux/arm64, ssh: op@192.0.2.10, cards: {card}}}
  {b}: {{hostname: {b}-host, label: {b}-gpu, platform: linux/arm64, ssh: op@192.0.2.11, cards: {card}}}
clusters:
  pair:
    control: ssh
    iface: bond0
    link: roce
    url: http://192.0.2.10:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks: [{{box: {a}, interconnect: 198.51.100.1}}, {{box: {b}, interconnect: 198.51.100.2}}]
backends: {{}}
""", "fleet.yml")
