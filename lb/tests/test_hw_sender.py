"""The hardware SENDER's packaging (lb/hw_sender/): the image that runs fleet_telemetry and the compose files a server
runs it with. What the sender reads and sends is fleet_telemetry's own suite (tests/test_telemetry_*.py).

  * the image is built from lb/ and carries the fleet_telemetry package and nothing else of the tree — not the
    gateway, not a secret, not a server's .env (Dockerfile.dockerignore); stdlib only, so no install step;
  * every server-specific value comes from that server's lb/hw_sender/.env as `${VAR:?…}` (operator rule: no
    defaults); the base names no GPU kind, a gpu-*.yml names exactly one; nothing publishes a port;
  * the recorder's containers are read from two READ-ONLY mounts, never the docker socket.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from fleet_telemetry.__main__ import CGROUP_ROOT, CONTAINERS_DIR, HOST_FS
from fleet_telemetry.readers import GPU_READERS
from tests.publishable_words import offenders

LB = Path(__file__).resolve().parents[1]                 # lb/
SENDER = LB / "hw_sender"


# ── 5. the image ──────────────────────────────────────────────────────────────────────────────────

def _instructions(path: Path) -> list[str]:
    """Dockerfile instructions, continuation lines joined, comments dropped."""
    out: list[str] = []
    buf = ""
    for line in path.read_text().splitlines():
        if not buf and (not line.strip() or line.lstrip().startswith("#")):
            continue
        buf += line.rstrip("\\").strip() + " " if line.rstrip().endswith("\\") else line.strip()
        if not line.rstrip().endswith("\\"):
            out.append(" ".join(buf.split()))
            buf = ""
    return out


def test_the_image_runs_the_package_and_installs_only_ethtool_and_iproute2():
    """Stdlib only in Python; the one install is the two tools the RoCE reader runs (readers/rdma.py: ethtool -S for
    the PHY counters, iproute2's dcb for PFC / trust), and the build fails if either is not on PATH."""
    ins = _instructions(SENDER / "Dockerfile")
    assert ins[0] == "FROM python:3.12-slim", "multi-arch base"
    assert not [i for i in ins if "pip install" in i], "stdlib only: no Python package"
    assert [i for i in ins if "apt-get" in i] == [
        "RUN apt-get update && apt-get install -y --no-install-recommends ethtool iproute2 && rm -rf /var/lib/apt/lists/*"]
    assert any("ethtool --version" in i and "dcb -V" in i for i in ins), "the build proves both tools run"
    assert "COPY fleet_telemetry/ /opt/telemetry/fleet_telemetry/" in ins
    assert [i for i in ins if i.startswith("COPY")] == ["COPY fleet_telemetry/ /opt/telemetry/fleet_telemetry/"]
    assert ins[-1] == 'ENTRYPOINT ["python", "-m", "fleet_telemetry"]'


def test_the_build_context_sends_only_the_package():
    svc = _svc("docker-compose.yml")
    assert svc["build"] == {"context": "..", "dockerfile": "hw_sender/Dockerfile"}
    lines = [ln for ln in (SENDER / "Dockerfile.dockerignore").read_text().splitlines()
             if ln and not ln.startswith("#")]
    assert lines[0] == "*" and set(lines[1:3]) == {"!fleet_telemetry", "!fleet_telemetry/**"}


def test_a_gpu_kind_file_exists_for_every_reader_and_names_it():
    kinds = {p.stem.split("-", 1)[1] for p in SENDER.glob("gpu-*.yml")}
    assert kinds == set(GPU_READERS)
    for kind in kinds:
        assert _svc(f"gpu-{kind}.yml")["environment"]["HW_SENDER_GPU"] == kind


# ── 6. compose: per-server values from .env, GPU kind named, no published port ───────────────────

COMPOSE = ("docker-compose.yml", "gpu-nvidia.yml", "gpu-amd.yml", "bridged.yml", "kmsg.yml", "disks.example.yml")
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")
IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
#: Names that would make a compose file describe one installation beyond the deployment's own words
#: (tests/publishable_words.py): the LB project's default network.
FLEET_WORDS = ("lb_default",)


class _Tag:
    """A compose merge tag (`!reset`, `!override`) and what it wraps — yaml.safe_load rejects them."""

    def __init__(self, tag: str, value):
        self.tag, self.value = tag, value

    def __repr__(self) -> str:
        return f"{self.tag} {self.value!r}"


def _load(name: str) -> dict:
    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
            value = None if value in ("", "null", "~") else value
        elif isinstance(node, yaml.SequenceNode):
            value = loader.construct_sequence(node, deep=True)
        else:
            value = loader.construct_mapping(node, deep=True)
        return _Tag("!" + suffix, value)

    Loader.add_multi_constructor("!", tagged)
    return yaml.load((SENDER / name).read_text(), Loader=Loader)


def _svc(name: str) -> dict:
    return _load(name)["services"]["hw-sender"]


def _code(name: str) -> list[str]:
    """Lines that configure (a literal in a comment documents; one in code configures)."""
    return [ln for ln in (SENDER / name).read_text().splitlines() if not ln.lstrip().startswith("#")]


def _walk(node):
    if isinstance(node, _Tag):
        yield from _walk(node.value)
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_the_base_is_the_lan_broadcast_with_its_own_project():
    base = _load("docker-compose.yml")
    assert base["name"] == "vllm-hw-sender", "its own compose project, never lb's"
    assert set(base["services"]) == {"hw-sender"}
    svc = base["services"]["hw-sender"]
    assert svc["network_mode"] == "host", "a host's own broadcast reaches the LAN only from the host netns"
    assert svc["restart"] == "unless-stopped"
    assert svc["image"] == "vllm-hw-sender:latest"
    assert svc["container_name"] == "vllm-hw-sender"
    assert "hostname" not in svc, "host networking: the wire says the server's own hostname"
    assert svc["environment"] == {"HW_SENDER_PORT": "${HW_SENDER_PORT:?set in lb/hw_sender/.env}",
                                  "HW_SENDER_NETWORK": "host",
                                  "HW_SENDER_DISKS": "${HW_SENDER_DISKS:?set in lb/hw_sender/.env}",
                                  "HW_SENDER_MODELS_DIRS": "${HW_SENDER_MODELS_DIRS:?set in lb/hw_sender/.env}"}
    for key in ("devices", "deploy", "gpus", "runtime", "privileged", "cap_add", "pid"):
        assert key not in svc, f"the base grants {key}: GPU access comes only from a NAMED gpu-*.yml"


def test_the_containers_are_read_from_two_read_only_mounts_never_the_docker_socket():
    """The recorder's containers (name, image, RAM) need the host's cgroup tree and docker's containers directory,
    READ-ONLY, at the paths fleet_telemetry reads. The docker socket — even `:ro` — is the API: control of the host.
    The base mounts nothing else: no host root, no disk."""
    assert _svc("docker-compose.yml")["volumes"] == [
        f"/sys/fs/cgroup:{CGROUP_ROOT}:ro",
        f"${{HW_SENDER_DOCKER_ROOT:?set in lb/hw_sender/.env}}/containers:{CONTAINERS_DIR}:ro"]


def test_disks_binds_each_path_read_only_at_host_fs_and_grants_nothing_else():
    """Least privilege: one `<path>:HOST_FS<path>:ro` bind per configured path — the sender sees those
    directories and no other host file. The real disks.yml is a deployment's (gitignored); the example is its form."""
    svc = _svc("disks.example.yml")
    assert set(svc) == {"volumes"}
    for v in svc["volumes"]:
        host, target, mode = v.split(":")
        assert host.startswith("/") and target == HOST_FS + host and mode == "ro", v
    for name in COMPOSE:
        assert not any("docker.sock" in ln for ln in _code(name)), f"{name} mounts the docker socket"


def test_no_file_publishes_a_port():
    for name in COMPOSE:
        keys = {k for k, _ in _walk(_load(name))}
        assert not keys & {"ports", "expose"}, f"{name} publishes: the sender only SENDS"


def test_the_nvidia_override_is_every_card_as_docker_run_gpus_all():
    svc = _svc("gpu-nvidia.yml")
    assert set(svc) == {"environment", "deploy"}
    assert svc["environment"] == {"HW_SENDER_GPU": "nvidia"}
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"capabilities": ["gpu"], "count": "all"}], \
        "no `driver:` — the request `--gpus all` makes (Driver \"\", Count -1, [[gpu]])"


def test_the_amd_override_grants_nothing():
    """The amdgpu driver's sysfs is readable from any container: no ROCm tree, no device node, no host /sys."""
    assert _svc("gpu-amd.yml") == {"environment": {"HW_SENDER_GPU": "amd"}}


def test_bridged_puts_the_sender_on_the_lbs_network_under_a_named_host():
    """Same-server LB, LAN must not hear it: the broadcast stays on the LB's docker bridge."""
    doc = _load("bridged.yml")
    svc = doc["services"]["hw-sender"]
    assert isinstance(svc["network_mode"], _Tag) and svc["network_mode"].tag == "!reset", \
        "the base's host networking must be RESET, or compose refuses network_mode beside networks"
    assert svc["hostname"] == "${HW_SENDER_HOSTNAME:?set in lb/hw_sender/.env}"
    assert svc["networks"] == ["lb"]
    assert doc["networks"] == {"lb": {"name": "${HW_SENDER_LB_NETWORK:?set in lb/hw_sender/.env}",
                                      "external": True}}
    assert svc["environment"] == {"HW_SENDER_NETWORK": "bridged"}, \
        "a container's network namespace: the recorder sends no network counters (test_hw_sender_recorder.py)"
    assert set(svc) == {"network_mode", "hostname", "networks", "environment"}, "GPU access stays with the gpu-*.yml"


def test_kmsg_grants_the_kernel_log_read_only_and_nothing_else():
    """The Xid counts need /dev/kmsg (readers/kernel.py): the device with cgroup permission `r` only — the sender can
    never write into the host's log — and CAP_SYSLOG (dmesg_restrict). Opt-in: the base grants neither."""
    assert _svc("kmsg.yml") == {"devices": ["/dev/kmsg:/dev/kmsg:r"], "cap_add": ["SYSLOG"]}


def test_no_server_specific_literal_in_code():
    for name in COMPOSE:
        for ln in _code(name):
            assert not IPV4.search(ln), f"{name}: an address literal: {ln.strip()}"
            assert not re.search(r"(?<![\w.:/-])\d{4,5}(?![\w.])", ln), f"{name}: a port literal: {ln.strip()}"
            for word in FLEET_WORDS:
                assert word.lower() not in ln.lower(), f"{name}: names {word!r}: {ln.strip()}"
            assert offenders(ln) == [], f"{name}: names the deployment: {ln.strip()}"


def test_every_interpolation_refuses_when_unset_and_says_where_to_set_it():
    used = 0
    for name in COMPOSE:
        for var, rest in VAR.findall("\n".join(_code(name))):
            used += 1
            assert rest.startswith(":?"), f"{name}: ${{{var}{rest}}} must be :? (no default)"
            assert "lb/hw_sender/.env" in rest, f"{name}: ${{{var}{rest}}} must say where it is set"
    assert used, "no variable at all: the per-server values went somewhere else"


def test_every_variable_used_is_documented_and_nothing_else():
    example = (SENDER / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, re.M))
    used = {v for name in COMPOSE for v, _ in VAR.findall("\n".join(_code(name)))}
    assert not used - documented, f"used, not in .env.example: {sorted(used - documented)}"
    assert documented - used == {"COMPOSE_FILE"}, f"documented, used by nothing: {sorted(documented - used - {'COMPOSE_FILE'})}"
    combos = re.findall(r"docker-compose\.yml(?::[\w.-]+\.yml)+", example)
    assert combos, ".env.example must show the COMPOSE_FILE a server picks"
    for combo in combos:
        for part in combo.split(":"):
            # disks.yml is the deployment's own (gitignored), written from its committed disks.example.yml
            shipped = SENDER / (part.replace(".yml", ".example.yml") if part == "disks.yml" else part)
            assert shipped.is_file(), f".env.example selects {part}, which does not exist"
