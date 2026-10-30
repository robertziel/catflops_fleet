"""The sealed runner's image and its start (lb/bench_runner): the pinned Claude Code, the entrypoint, the
compose file, the Dockerfile — and that the LB's own image carries none of it.

THE PIN. Claude Code is installed from npm's per-platform tarball and checked twice at build — the
tarball's sha256 and the binary's — and the binary again at every start. The binary's hash is the one
the gateway's host sessions ran as 2.1.296 (~/.local/share/claude/versions/2.1.296, 2026-10-11), so the
sealed runner drives the cluster with byte-for-byte the same client.

THE START. The entrypoint is the only root code: it gives the state directory to `bench`, drops to it,
and refuses — naming why — when a backend variable is in its environment, when the binary is not the
pinned one, or when the seal check fails. Only then is the runner exec'd, with a command built from the
container's environment alone.
"""
from __future__ import annotations

import hashlib
import io
import pathlib
import re
import tarfile

import pytest
import yaml

from bench_runner import claude_pin, entrypoint, install_claude
from tests.publishable_words import offenders

LB = pathlib.Path(__file__).resolve().parent.parent
HERE = LB / "bench_runner"


# ── the pin ────────────────────────────────────────────────────────────────


def test_the_pin_names_one_exact_version_and_both_hashes():
    assert re.fullmatch(r"\d+\.\d+\.\d+", claude_pin.VERSION)
    assert claude_pin.VERSION == "2.1.296"
    for digest in (claude_pin.TARBALL_SHA256, claude_pin.BINARY_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert claude_pin.TARBALL_URL == ("https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/"
                                      f"claude-code-linux-x64-{claude_pin.VERSION}.tgz")
    assert claude_pin.BINARY_MEMBER == "package/claude"


def _tarball(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_the_installer_writes_the_binary_only_when_both_hashes_match(tmp_path):
    payload = b"\x7fELF pretend claude"
    tgz = _tarball("package/claude", payload)
    dest = tmp_path / "bin" / "claude"
    install_claude.verify_and_extract(tgz, hashlib.sha256(tgz).hexdigest(), "package/claude",
                                      hashlib.sha256(payload).hexdigest(), dest)
    assert dest.read_bytes() == payload and dest.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("which", ["tarball", "binary", "member"])
def test_the_installer_refuses_a_tarball_or_binary_that_is_not_the_pinned_one(tmp_path, which):
    payload = b"\x7fELF pretend claude"
    tgz = _tarball("package/claude", payload)
    good = dict(tgz_sha256=hashlib.sha256(tgz).hexdigest(), member="package/claude",
                bin_sha256=hashlib.sha256(payload).hexdigest())
    if which == "tarball":
        good["tgz_sha256"] = "0" * 64
    elif which == "binary":
        good["bin_sha256"] = "0" * 64
    else:
        good["member"] = "package/not-claude"
    dest = tmp_path / "claude"
    with pytest.raises(install_claude.PinMismatch):
        install_claude.verify_and_extract(tgz, good["tgz_sha256"], good["member"], good["bin_sha256"], dest)
    assert not dest.exists(), "nothing is installed from a tarball that failed its pin"


# ── the entrypoint ─────────────────────────────────────────────────────────

GOOD_ENV = {"BENCH_BOX": "box-under-test", "BENCH_LB_URL": "http://lb:8010",
            "BENCH_METRICS_URL": "http://engine:8000/metrics", "BENCH_EGRESS_PROXY": "http://egress:3128",
            "BENCH_NO_PROXY": "lb,engine"}


def test_the_runners_command_is_built_from_the_containers_environment_alone():
    argv = entrypoint.runner_argv(entrypoint.config(GOOD_ENV))
    assert argv == ["--box", "box-under-test", "--lb-url", "http://lb:8010",
                    "--metrics-url", "http://engine:8000/metrics",
                    "--state-dir", entrypoint.STATE_DIR, "--results-dir", entrypoint.RESULTS_DIR,
                    "--https-proxy", "http://egress:3128", "--no-proxy", "lb,engine",
                    "--seal-check", entrypoint.SEAL_CHECK]


@pytest.mark.parametrize("missing", sorted(GOOD_ENV))
def test_a_missing_value_refuses_to_start_and_names_it(missing):
    env = {k: v for k, v in GOOD_ENV.items() if k != missing}
    with pytest.raises(entrypoint.Refused) as exc:
        entrypoint.config(env)
    assert missing in str(exc.value)


def test_a_backend_variable_in_the_containers_environment_refuses_to_start():
    assert entrypoint.leaked({"PATH": "/usr/bin", **GOOD_ENV}) == []
    leaked = entrypoint.leaked({**GOOD_ENV, "ANTHROPIC_API_KEY": "sk-ant-FAKE", "CLAUDE_CODE_OAUTH_TOKEN": "x",
                                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"})
    assert leaked == ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"]


def test_the_binary_must_be_the_pinned_one(tmp_path):
    binary = tmp_path / "claude"
    binary.write_bytes(b"the pinned bytes")
    ok, why = entrypoint.binary_matches(str(binary), hashlib.sha256(b"the pinned bytes").hexdigest())
    assert ok is True
    ok, why = entrypoint.binary_matches(str(binary), "0" * 64)
    assert ok is False and "not the pinned" in why
    ok, why = entrypoint.binary_matches(str(tmp_path / "absent"), "0" * 64)
    assert ok is False


# ── the compose file and the Dockerfile ───────────────────────────────────


def compose() -> dict:
    return yaml.safe_load((HERE / "docker-compose.yml").read_text())


def _code(path: pathlib.Path) -> str:
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_the_runner_is_on_the_internal_network_and_nothing_else():
    c = compose()
    runner = c["services"]["runner"]
    assert runner["networks"] == ["sealed"], runner["networks"]
    assert c["networks"]["sealed"].get("internal") is True
    assert "network_mode" not in runner and "ports" not in runner and "extra_hosts" not in runner


def test_the_egress_is_the_only_bridge_and_is_named_lb_engine_and_egress_inside():
    c = compose()
    eg = c["services"]["egress"]
    assert set(eg["networks"]) == {"sealed", "upstream"}
    assert set(eg["networks"]["sealed"]["aliases"]) == {"lb", "engine", "egress", "egress-sealed-side"}
    assert c["networks"]["upstream"]["external"] is True
    assert "ports" not in eg, "nothing of the egress is published on the host"
    assert eg["user"] == "65534:65534" and eg["read_only"] is True and eg["cap_drop"] == ["ALL"]


def test_the_egress_listens_on_the_sealed_side_only():
    """It binds the address of a name that exists on the SEALED network alone. Compose registers a
    service's own name — and docker its container name — on every network the service joins, so `egress`
    resolved, from inside, to the upstream address: the proof run of 2026-10-11 found the proxy listening
    on the LB's network and unreachable from the runner."""
    eg = compose()["services"]["egress"]
    command = eg["command"]
    bind = command[command.index("--bind") + 1]
    assert bind == "egress-sealed-side"
    assert bind in eg["networks"]["sealed"]["aliases"]
    assert bind not in ("egress", eg["container_name"]), "a name docker registers on every network"
    assert eg["healthcheck"]["test"][-1].count(bind) == 1, "the health check dials the same side"


def test_no_credential_or_host_home_is_mounted_into_either_container():
    """The runner's one mount is the LB's swarm directory; the egress has none at all."""
    mounts = {name: [str(v) for v in service.get("volumes", [])]
              for name, service in compose()["services"].items()}
    assert mounts == {"egress": [], "runner": [
        "${BENCH_STATE_DIR:?set in lb/bench_runner/.env or by deploy/standalone/up.sh}:/data/benchq"]}
    for name, service in compose()["services"].items():
        assert not service.get("env_file"), f"{name}: no env file is handed to a sealed container"


def test_every_deployment_value_is_required_and_documented():
    text = _code(HERE / "docker-compose.yml")
    used = re.findall(r"\$\{([A-Z_][A-Z0-9_]*)([^}]*)\}", text)
    assert used, "the compose file takes its deployment values from the environment"
    for var, rest in used:
        assert rest.startswith(":?"), f"${{{var}{rest}}} must be :? (no default)"
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", (HERE / ".env.example").read_text(), re.M))
    assert {v for v, _ in used} == documented, ({v for v, _ in used}, documented)
    assert documented == {"BENCH_BOX", "BENCH_STATE_DIR", "BENCH_UPSTREAM_NETWORK", "BENCH_LB_TARGET",
                          "BENCH_ENGINE_TARGET"}


def test_a_crowded_host_names_the_sealed_subnet_itself_and_nothing_is_guessed():
    """Docker hands a new network a subnet from its default pools, and a host with ~30 networks has none
    left ("all predefined address pools have been fully subnetted" — the LB's own host, 2026-10-11). There
    the deployment states one: subnet.yml, opted into with COMPOSE_FILE, requires BENCH_SEALED_SUBNET.
    Without it the base file lets docker choose, and a host that cannot says so — never a picked range."""
    overlay = yaml.safe_load((HERE / "subnet.yml").read_text())
    config = overlay["networks"]["sealed"]["ipam"]["config"]
    assert config == [{"subnet": "${BENCH_SEALED_SUBNET:?set in lb/bench_runner/.env when subnet.yml is used}"}]
    assert "ipam" not in compose()["networks"]["sealed"], "the base file never picks a subnet"
    example = (HERE / ".env.example").read_text()
    assert re.search(r"^# ?BENCH_SEALED_SUBNET=$", example, re.M), "documented, commented: only with subnet.yml"
    assert re.search(r"^# ?COMPOSE_FILE=docker-compose\.yml:subnet\.yml$", example, re.M)


def test_the_runner_is_wired_to_the_seal_and_never_to_a_backend_variable():
    env = compose()["services"]["runner"]["environment"]
    assert env["BENCH_LB_URL"] == "http://lb:8010" and env["BENCH_METRICS_URL"] == "http://engine:8000/metrics"
    assert env["BENCH_EGRESS_PROXY"] == "http://egress:3128" and env["BENCH_NO_PROXY"] == "lb,engine"
    assert not [k for k in env if k.startswith(("ANTHROPIC", "CLAUDE"))], env


def test_the_egress_relays_its_two_fixed_ports_from_the_deployments_targets():
    command = compose()["services"]["egress"]["command"]
    joined = " ".join(command)
    assert "--bind egress" in joined and "--proxy-port 3128" in joined
    assert "8010=${BENCH_LB_TARGET:?" in joined and "8000=${BENCH_ENGINE_TARGET:?" in joined


def test_no_operator_name_or_address_is_in_the_runners_files():
    for name in ("docker-compose.yml", ".env.example", "Dockerfile", "managed-settings.json"):
        code = _code(HERE / name)
        assert re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", code) is None, name
        assert offenders(code) == [], f"{name} names the deployment (tests/publishable_words.py)"


def test_the_dockerfile_pins_its_base_by_digest_and_installs_the_pinned_claude():
    text = (HERE / "Dockerfile").read_text()
    assert re.search(r"^FROM python:3\.12-slim@sha256:[0-9a-f]{64}$", text, re.M), "base pinned by digest"
    assert "bench_runner.install_claude" in text
    assert "/etc/claude-code/managed-settings.json" in text
    assert "COPY . " not in text, "only named files enter the image"


def test_the_managed_settings_skip_the_preflight_and_hold_no_key():
    """WebFetch's preflight asks Anthropic whether a domain may be fetched; the proxy refuses that, so it
    is skipped. The auto-updater and the nonessential traffic would only ever be refused too."""
    import json
    settings = json.loads((HERE / "managed-settings.json").read_text())
    assert settings["skipWebFetchPreflight"] is True
    assert settings["env"] == {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    assert "apiKeyHelper" not in settings


def test_the_lb_image_carries_none_of_the_sealed_runner():
    pats = [ln.strip() for ln in (LB / ".dockerignore").read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    assert "bench_runner/" in pats


def test_the_runners_own_build_context_admits_only_what_it_copies():
    ignore = (HERE / "Dockerfile.dockerignore").read_text().splitlines()
    assert ignore[0].strip() == "*" or "*" in [ln.strip() for ln in ignore], "everything is excluded first"
    allowed = {ln.strip()[1:] for ln in ignore if ln.strip().startswith("!")}
    assert {"bench_queue_runner.py", "bench_session.py"} <= allowed
    assert not [a for a in allowed if a.startswith(("state", ".env", "cluster-ssh"))]


def test_the_runner_carries_the_engines_vocabularies_and_nothing_else_of_the_lb():
    """bench_session reads an engine's /metrics through vllm_lb/engine_metrics.py, the one definition the
    LB's probe reads too (2026-10-29: a private prefix list refused a live TensorFold pair). That module —
    stdlib only, no config or state — and the package marker are the only parts of the LB the sealed
    runner holds."""
    ignore = (HERE / "Dockerfile.dockerignore").read_text().splitlines()
    allowed = {ln.strip()[1:] for ln in ignore if ln.strip().startswith("!")}
    assert {a for a in allowed if a.startswith("vllm_lb")} == {"vllm_lb/__init__.py", "vllm_lb/engine_metrics.py"}
    text = (HERE / "Dockerfile").read_text()
    assert "COPY vllm_lb/__init__.py vllm_lb/engine_metrics.py /opt/bench/vllm_lb/" in text
    module = (LB / "vllm_lb" / "engine_metrics.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+vllm_lb", module, re.M), "it must import nothing of the LB's"