"""The deploy scripts and the cross-service config name no deployment (DEP-15, 17, 18, 21, 22, 27).

Stdlib only, so the harness runs in a bare python image with the whole checkout mounted:

    docker run --rm -v <checkout>:/repo -w /repo python:3.12-slim \
        python -m unittest discover -s deploy/tests -v

Shell scripts are run for real, with `env -i` and a fake `ssh` on PATH that records its argv and
changes nothing, so a test can prove which host a script would have dialled.
"""
from __future__ import annotations

import json
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
#: THE SIGNED-IN EDGE: gate/ (built against a private sign-in client) and the files that exist only to wire it —
#: the pair fleet's edge vhost, the standalone doors (compose + nginx template) and their route tests. The public
#: release ships none of them (PUBLISHING.md drops EDGE_FILES and cuts deploy/standalone/.env.example at its
#: GATE-BELOW marker), so each check of them here is conditional on gate/ being present, and a tree must carry
#: all of them or none.
EDGE_FILES = ("gate", "deploy/nginx-edge-vllm.example.conf", "deploy/standalone/docker-compose.yml",
              "deploy/standalone/nginx.conf.template", "deploy/tests/test_api_key_route.py",
              "deploy/tests/test_kill_route.py", "deploy/tests/test_no_inference_online.py")
GATE = (REPO / "gate").is_dir()
needs_gate = unittest.skipUnless(GATE, "gate/ is not in this tree (PUBLISHING.md drops the signed-in edge from the release)")
#: THE CHAT FRONT-END: chat/ is its own project (catflops_chat; operator decision, 2026-10-29) and the release ships
#: none of it — nor the files that exist only to deploy, vendor into or front it. PUBLISHING.md drops CHAT_FILES, so
#: each check of them here is conditional on chat/ being present, and a tree must carry all of them or none. Any
#: OpenAI- or Anthropic-compatible client works against the LB; the shipped docs name no chat service. An entry may be
#: a glob (`*`): the chat's vendoring script is named by its shape, so this shipped file spells no sibling project.
CHAT_FILES = ("chat", "deploy/chat-redeploy.sh", "deploy/vendor-*.sh", "deploy/code-sessions",
              "deploy/standalone/chat.override.yml", "deploy/frpc.chat.toml", "deploy/nginx-edge-chat.example.conf",
              "deploy/nginx-edge-chat-vhost.example.conf", "deploy/nginx-edge-chat-retire.example.conf",
              "deploy/tests/test_share_route.py")
CHAT = (REPO / "chat").is_dir()
needs_chat = unittest.skipUnless(CHAT, "chat/ is not in this tree (PUBLISHING.md drops the chat front-end from the "
                                       "release: it is its own project, catflops_chat)")
#: What a shipped doc may not say about a chat service: its directory and package, its container, its port, its
#: sign-in settings and what it vendors. `/v1/chat/completions` (the API) and the dashboard's own CHAT_MODEL are not it.
#: (What it vendors is a sibling project's name: the shared word list's, hunted by the tree guard below.)
CHAT_WORDS = re.compile(r"(?<![\w/])chat/|vllm_chat|vllm-chat|:8011\b|CHAT_AUTH|CHAT_PUBLIC_URL|code_collector")


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


def _code_lines(rel: str) -> list[str]:
    """A file's lines minus comments: shell `#...` and TOML/YAML `#...` lines."""
    return [ln for ln in _text(rel).splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def scratch(case: unittest.TestCase) -> Path:
    """A temp directory that lives until the test ENDS, so assertions may read what it recorded."""
    return Path(case.enterContext(tempfile.TemporaryDirectory()))


class FakeSsh:
    """A stand-in `ssh` on PATH: logs its argv one line per call and exits with `rc`."""

    def __init__(self, tmp: Path, rc: int = 0):
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.log = tmp / "ssh.log"
        script = self.bin / "ssh"
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$*\" >> '{self.log}'\n"
            f"exit {rc}\n")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


def run_script(rel: str, args=(), env=None, *, ssh: FakeSsh | None = None):
    """Run a repo script under `env -i`, with only what the test names in its environment."""
    clean = {"PATH": (f"{ssh.bin}:" if ssh else "") + "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/tester"}
    clean.update(env or {})
    return subprocess.run(["bash", str(REPO / rel), *args], env=clean, capture_output=True,
                          text=True, timeout=60)


class BenchQueueUnit(unittest.TestCase):
    """DEP-15: the systemd user unit is rendered from a template with the checkout and $HOME."""

    def test_the_unit_renders_with_this_checkout_and_this_home(self):
        r = run_script("deploy/install-bench-queue-runner.sh", ["--render"],
                       {"LB_CLUSTER_URL": "http://head.test:8100"})
        self.assertEqual(r.returncode, 0, r.stderr)
        unit = r.stdout
        self.assertIn(f"ExecStart=/usr/bin/python3 {REPO}/lb/bench_queue_runner.py", unit)
        self.assertIn("Environment=PATH=/home/tester/.local/bin:", unit)
        self.assertIn("Environment=LB_CLUSTER_URL=http://head.test:8100", unit)
        self.assertIn(f"Documentation=file://{REPO}/lb/bench_queue_runner.py", unit)
        self.assertNotIn("@", unit, "every placeholder must be substituted")

    def test_the_rendered_unit_names_no_operator_path(self):
        r = run_script("deploy/install-bench-queue-runner.sh", ["--render"],
                       {"LB_CLUSTER_URL": "http://head.test:8100"})
        self.assertNotIn("/home/user", r.stdout)

    def test_the_unit_is_not_rendered_without_the_head_engine(self):
        r = run_script("deploy/install-bench-queue-runner.sh", ["--render"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("LB_CLUSTER_URL", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_the_template_is_the_only_unit_source(self):
        self.assertFalse((DEPLOY / "vllm-bench-queue-runner.service").exists(),
                         "the operator-path unit must not be committed next to its template")
        self.assertTrue((DEPLOY / "vllm-bench-queue-runner.service.in").is_file())


class NfsHealerInstaller(unittest.TestCase):
    """DEP-17: the HARDEN script is a required variable, and boxes come from arguments only."""

    def test_it_refuses_to_run_without_HARDEN_naming_the_variable(self):
        ssh = FakeSsh(scratch(self))
        r = run_script("deploy/install-nfs-healer.sh", ["--check", "box-a"], ssh=ssh)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HARDEN", r.stderr)
        self.assertEqual(ssh.calls(), [], "nothing may be dialled before the guard passes")

    def test_it_takes_no_default_box_list(self):
        tmp = scratch(self)
        harden = tmp / "harden.sh"
        harden.write_text("#!/bin/sh\n")
        ssh = FakeSsh(tmp)
        r = run_script("deploy/install-nfs-healer.sh", ["--check"], {"HARDEN": str(harden)}, ssh=ssh)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("box", r.stderr.lower())
        self.assertEqual(ssh.calls(), [], "no box named means no box dialled, never box-a box-b")


class SeedSidecar(unittest.TestCase):
    """DEP-18: the model directory is required and the box is the argument — never an operator default or alias."""

    def _run(self, args, env, rc=0):
        ssh = FakeSsh(scratch(self), rc=rc)
        r = run_script("seed-broker-sidecar.sh", args, env, ssh=ssh)
        return r, ssh.calls()

    def test_the_model_dir_is_required(self):
        r, calls = self._run(["gpu-box"], {})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MODEL_DIR", r.stderr)
        self.assertEqual(calls, [])

    def test_the_box_is_required_and_never_defaulted(self):
        r, calls = self._run([], {"MODEL_DIR": "/models-here"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ssh-host", r.stderr)
        self.assertEqual(calls, [])

    def test_the_named_box_is_the_host_dialled(self):
        r, calls = self._run(["my-second-box"], {"MODEL_DIR": "/models-here"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(calls, "the seed must dial the named host")
        self.assertTrue(all(c.split()[1] == "my-second-box" for c in calls), calls)
        self.assertTrue(any("/models-here:/models:ro" in c for c in calls), calls)

    def test_no_operator_alias_or_mount_is_left_in_the_script(self):
        code = "\n".join(_code_lines("seed-broker-sidecar.sh"))
        self.assertNotIn("_SSH_HOST", code, "a per-box alias variable is a hardcoded fleet")
        self.assertNotIn("/srv/models", code)
        self.assertEqual(_deployment_words(code), set())


class FrpcServerAddress(unittest.TestCase):
    """DEP-22: the frps host is an environment variable in every frpc config and every frpc service."""

    CONFIGS = ("frpc/frpc.toml", "power/frpc.toml", "deploy/frpc.vllm.toml") + (
        ("deploy/frpc.chat.toml",) if CHAT else ()) + (("gate/frpc.toml",) if GATE else ())
    #: compose file -> the frpc service's container_name, so the test finds its block
    SERVICES = {"dashboard/docker-compose.yml": "vllm-frpc",
                "power/docker-compose.yml": "vllm-power-frpc",
                **({"chat/docker-compose.yml": "vllm-chat-frpc"} if CHAT else {}),
                **({"gate/docker-compose.yml": "vllm-gate-frpc"} if GATE else {})}

    def test_no_frpc_config_names_the_frps_host(self):
        for rel in self.CONFIGS:
            with self.subTest(config=rel):
                lines = _code_lines(rel)
                self.assertFalse(any("frps.example.com" in ln for ln in lines), rel)
                self.assertTrue(any('serverAddr = "{{ .Envs.FRP_SERVER }}"' in ln for ln in lines), rel)

    def test_every_frpc_service_gets_FRP_SERVER_from_the_root_env_without_failing_its_project(self):
        """chat/ and dashboard/ interpolate it (`:-`, so a stranger without frp still brings the app up);
        gate/ and power/ hand the root .env to their frpc whole, as an OPTIONAL env_file."""
        for rel, name in self.SERVICES.items():
            with self.subTest(compose=rel):
                # the service's own block runs from its container_name to the next service key
                block = _text(rel).split(f"container_name: {name}", 1)[1]
                block = re.split(r"\n  [A-Za-z0-9_-]+:\n", block, maxsplit=1)[0]
                if rel.startswith(("chat/", "dashboard/")):
                    self.assertIn("FRP_SERVER=${FRP_SERVER:-}", block, rel)
                else:
                    self.assertRegex(block, r"env_file:[^\n]*\n\s*- path: \.\./\.env[^\n]*\n\s*required: false", rel)
                self.assertNotIn("${FRP_SERVER:?", block, rel)


class ServiceComposeOrigins(unittest.TestCase):
    """DEP-21 and DEP-29: a public origin comes from the environment, never a literal, and an unset one
    REFUSES (the gate and vllm-power at boot, the chat's sign-in); the time zone is neutral."""

    @needs_gate
    def test_the_gate_origins_come_from_gate_env(self):
        text = _text("gate/docker-compose.yml")
        self.assertIn('VLLM_GATE_AUTH_ISSUER: "${VLLM_GATE_AUTH_ISSUER:-}"', text)
        self.assertIn('VLLM_GATE_ORIGIN: "${VLLM_GATE_ORIGIN:-}"', text)
        self.assertNotIn("https://auth.example.com", text)
        self.assertNotIn("https://vllm.example.com", text)

    def test_the_power_origin_and_inventory_come_from_power_env(self):
        text = _text("power/docker-compose.yml")
        self.assertIn('POWER_ORIGINS: "${POWER_ORIGINS:-}"', text)
        self.assertIn('POWER_BOXES: "${POWER_BOXES:-}"', text)
        self.assertNotIn("vllm.example.com", text)

    @needs_chat
    def test_the_chat_origin_comes_from_the_root_env_and_the_zone_is_neutral(self):
        text = _text("chat/docker-compose.yml")
        self.assertIn('CHAT_PUBLIC_URL: "${CHAT_PUBLIC_URL:-}"', text)
        self.assertIn('TZ: "${CHAT_TZ:-UTC}"', text)
        # comments may name the operator's setup; the executable lines may not
        code = "\n".join(_code_lines("chat/docker-compose.yml"))
        self.assertNotIn("chat.example.com", code)
        self.assertNotIn("Europe/Warsaw", code)

    @needs_chat
    def test_the_chats_gateway_panel_comes_from_the_root_env(self):
        """The load meter's link (CHAT_GATEWAY_URL) is each deployment's own gateway; empty = no link."""
        self.assertIn('CHAT_GATEWAY_URL: "${CHAT_GATEWAY_URL:-}"', _text("chat/docker-compose.yml"))
        code = "\n".join(_code_lines("chat/docker-compose.yml"))
        self.assertNotIn("vllm.example.com", code)
        self.assertEqual(_deployment_words(code), set())


class StageModelSearch(unittest.TestCase):
    """DEP-27: where checkpoints live comes from MODELS_DIR / MODELS_SEARCH_DIRS — deploy/models-env.sh reads the
    environment, else lb/.env — never a literal list and never a guessed folder (operator, 2026-10-09: "models dir
    also should be in .env defined, we use HDD for that"). The LB reads the same MODELS_SEARCH_DIRS."""

    #: Every script that names a model folder, the box-side launchers included since the operator decided
    #: nothing is hardcoded: launch-pair.sh sends them the list, and they refuse without one.
    SCRIPTS = ("deploy/models-env.sh", "deploy/stage-qwen38-27b-nvfp4.sh", "deploy/stage-checkpoint.sh",
               "deploy/stage-glm53-flash-nvfp4.sh", "deploy/stage-deepseek-v4-flash-nvfp4.sh",
               "deploy/sglang-27b-tp2/stage-dflash2-draft.sh", "deploy/dflash2-27b-tp2/stage-hf-model.sh",
               "deploy/dflash2-27b-tp2/rank.sh", "deploy/sglang-27b-tp2/sglang-rank")
    #: The checkout's own lb/.env must not answer for a test.
    NO_FILE = {"MODELS_ENV_FILE": "/nonexistent/lb.env", "STAGE_HEAD": "head-box", "STAGE_WORKER": "worker-box"}
    STAGE = "deploy/stage-qwen38-27b-nvfp4.sh"

    def test_no_script_carries_a_literal_model_folder(self):
        for rel in self.SCRIPTS:
            with self.subTest(script=rel):
                code = "\n".join(_code_lines(rel))
                self.assertNotIn("/mnt/box-a-models", code)
                self.assertNotIn("/srv/models", code)

    def test_check_searches_exactly_the_configured_dirs(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        run_script(self.STAGE, ["check"], {**self.NO_FILE, "MODELS_DIR": str(tmp),
                                           "MODELS_SEARCH_DIRS": "/data-one/models:/data-two/models"}, ssh=ssh)
        dialled = " ".join(ssh.calls())
        self.assertIn("/data-one/models/Qwen3.8-27B-NVFP4", dialled)
        self.assertIn("/data-two/models/Qwen3.8-27B-NVFP4", dialled)
        self.assertNotIn("/mnt/box-a-models", dialled)

    def test_a_home_relative_entry_is_left_for_the_box_to_expand(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        run_script(self.STAGE, ["check"], {**self.NO_FILE, "MODELS_DIR": str(tmp), "MODELS_SEARCH_DIRS": "~/fast"},
                   ssh=ssh)
        self.assertIn("$HOME/fast/Qwen3.8-27B-NVFP4", " ".join(ssh.calls()))

    def test_check_refuses_without_either_setting_and_dials_nothing(self):
        for missing in ("MODELS_DIR", "MODELS_SEARCH_DIRS"):
            with self.subTest(missing=missing):
                tmp = scratch(self)
                ssh = FakeSsh(tmp)
                env = {**self.NO_FILE, "MODELS_DIR": str(tmp), "MODELS_SEARCH_DIRS": "/y"}
                del env[missing]
                r = run_script(self.STAGE, ["check"], env, ssh=ssh)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(missing, r.stderr)
                self.assertEqual(ssh.calls(), [])

    def test_the_settings_come_from_lb_env_when_not_exported(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        env_file = tmp / "lb.env"
        env_file.write_text(f'OTHER=1\nMODELS_DIR="{tmp}"\nMODELS_SEARCH_DIRS=/from-the-file/models\n')
        run_script(self.STAGE, ["check"], {"MODELS_ENV_FILE": str(env_file), "STAGE_HEAD": "h", "STAGE_WORKER": "w"},
                   ssh=ssh)
        self.assertIn("/from-the-file/models/Qwen3.8-27B-NVFP4", " ".join(ssh.calls()))

    def test_the_boxes_are_required_settings_never_a_default_pair(self):
        """STAGE_HEAD (the box whose fast disk the copy lands on) and STAGE_WORKER (the box that reads its export) are
        this deployment's, from the environment; without either the script names it and dials nothing."""
        for missing in ("STAGE_HEAD", "STAGE_WORKER"):
            with self.subTest(missing=missing):
                tmp = scratch(self)
                ssh = FakeSsh(tmp)
                env = {**self.NO_FILE, "MODELS_DIR": str(tmp), "MODELS_SEARCH_DIRS": "/y"}
                del env[missing]
                r = run_script(self.STAGE, ["check"], env, ssh=ssh)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(missing, r.stderr)
                self.assertEqual(ssh.calls(), [])

    def test_the_configured_boxes_are_the_hosts_dialled(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        run_script(self.STAGE, ["check"], {**self.NO_FILE, "MODELS_DIR": str(tmp), "MODELS_SEARCH_DIRS": "/y"}, ssh=ssh)
        hosts = {w for c in ssh.calls() for w in c.split() if w in ("head-box", "worker-box")}
        self.assertEqual(hosts, {"head-box", "worker-box"})


class StageCheckpoint(unittest.TestCase):
    """ONE staging script for every checkpoint (operator 2026-10-20, GLM-5.3-Flash: "keep code DRY"): fetch into
    MODELS_DIR verified against the repo's own manifest, stage onto box-a's NVMe with direct I/O, check every root.
    WHICH checkpoint is the caller's three settings — a per-model wrapper is its repo, its pinned revision and its
    folder under the models roots (a catalog row's `path`, group folder included), and nothing else."""

    GENERIC = "deploy/stage-checkpoint.sh"
    GLM = "deploy/stage-glm53-flash-nvfp4.sh"
    ENV = {"MODELS_ENV_FILE": "/nonexistent/lb.env", "MODELS_SEARCH_DIRS": "/data-one/models:~/fast",
           "STAGE_HEAD": "head-box", "STAGE_WORKER": "worker-box"}
    PIN = {"STAGE_REPO": "org/Model", "STAGE_REV": "a" * 40, "STAGE_NAME": "group/Model-NVFP4"}

    def test_the_glm_wrapper_checks_its_grouped_folder_in_every_root(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        run_script(self.GLM, ["check"], {**self.ENV, "MODELS_DIR": str(tmp)}, ssh=ssh)
        dialled = " ".join(ssh.calls())
        self.assertIn("/data-one/models/glm53-flash-cluster/GLM-5.3-Flash-NVFP4", dialled)
        self.assertIn("$HOME/fast/glm53-flash-cluster/GLM-5.3-Flash-NVFP4", dialled)

    def test_the_generic_script_refuses_without_each_setting_and_dials_nothing(self):
        for missing in self.PIN:
            with self.subTest(missing=missing):
                tmp = scratch(self)
                ssh = FakeSsh(tmp)
                env = {**self.ENV, "MODELS_DIR": str(tmp), **self.PIN}
                del env[missing]
                r = run_script(self.GENERIC, ["check"], env, ssh=ssh)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(missing, r.stderr)
                self.assertEqual(ssh.calls(), [])

    def test_a_revision_is_a_pinned_commit_never_a_branch(self):
        tmp = scratch(self)
        ssh = FakeSsh(tmp)
        r = run_script(self.GENERIC, ["check"], {**self.ENV, "MODELS_DIR": str(tmp), **self.PIN, "STAGE_REV": "main"},
                       ssh=ssh)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("STAGE_REV", r.stderr)
        self.assertEqual(ssh.calls(), [])

    def test_each_wrapper_is_only_its_pin(self):
        for rel in (self.GLM, StageModelSearch.STAGE):
            with self.subTest(script=rel):
                code = _code_lines(rel)
                self.assertTrue(any("stage-checkpoint.sh" in ln for ln in code), rel)
                self.assertLessEqual(len(code), 6, f"{rel}: a wrapper is its repo, revision and folder: {code}")


#: SHA-256 of the identifiers that once sat in these files: the operator's mailbox, another person's
#: basic-auth login and two other people's mailboxes. Digests, so this test does not publish them itself.
REMOVED_IDENTIFIERS = {
    "3051c655be16ca4bd566d98be0d0928a791256c5acbc094f0294ace3516fb3e3",
    "f68e023c3668bae4e8bbefeef49f4f98193aebf664d4385bd446161edbac37ff",
    "417f4a1660d13c57ef0285cab6d925b8b25d6a908b81703f55d0f2b80011cc70",
    "e3f9339805a6bae924da0c4204b54deaaf89eb17ad69b3e9881df922af26f77c",
}


def matrix_tag(engine: str, build: str, accelerator: str, platform: str) -> str:
    """The image the LB's image matrix (lb/vllm_lb/library/images/<engine>.yml) names for one cell — read with the
    one-entry-a-line shape lb/tests/test_images.py holds the file to (this suite has no YAML parser)."""
    line = re.compile(rf'  - \{{build: {re.escape(build)}, accelerator: {re.escape(accelerator)}, '
                      rf'platform: {re.escape(platform)}, image: "([^"]+)", source: ')
    found = [m.group(1) for ln in (REPO / f"lb/vllm_lb/library/images/{engine}.yml").read_text().splitlines()
             if (m := line.match(ln))]
    assert len(found) == 1, (engine, build, accelerator, platform, found)
    return found[0]


def _stage(rel: str, name: str) -> str:
    """The text of one `FROM … AS <name>` stage of a Dockerfile, comments dropped, up to the next FROM."""
    lines, inside = [], False
    for ln in _code_lines(rel):
        if ln.startswith("FROM "):
            inside = ln.endswith(f" AS {name}")
        if inside:
            lines.append(ln)
    return "\n".join(lines)


class Glm53Image(unittest.TestCase):
    """The pair's GLM-5.3-Flash image (deploy/images/vllm, stage glm53; built by deploy/glm53-flash-cluster-vllm/image/
    build.sh): its base pinned by DIGEST — the one the published 2x DGX Spark runs measured on — and its tag the
    catalog's, read from the row, never a second copy."""

    DIR = "deploy/glm53-flash-cluster-vllm/image"
    PIN = "deploy/images/vllm/glm53.nvidia-sm121.env"

    def test_the_base_is_pinned_by_digest(self):
        bases = [ln for ln in _code_lines(self.PIN) if ln.startswith("BASE=")]
        self.assertEqual(len(bases), 1, bases)
        self.assertRegex(bases[0], r"^BASE=[a-z0-9./-]+@sha256:[0-9a-f]{64}$")

    def test_the_image_sets_the_gb10_environment_the_multimodal_recipe_runs_with(self):
        # MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark @aed98a1 files/Dockerfile.mm-ray: JIT for sm_121a only, the
        # nightly FlashInfer's version check off, expandable segments on the unified pool, an hour to become ready.
        env = _stage("deploy/images/vllm/Dockerfile", "glm53")
        for setting in ("TORCH_CUDA_ARCH_LIST=12.1a", "FLASHINFER_CUDA_ARCH_LIST=12.1a",
                        "FLASHINFER_DISABLE_VERSION_CHECK=1", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                        "VLLM_ENGINE_READY_TIMEOUT_S=3600"):
            self.assertIn(setting, env)

    def test_the_tag_is_the_image_matrixs(self):
        r = subprocess.run(["bash", f"{self.DIR}/build.sh", "tag"], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), matrix_tag("vllm", "glm53", "nvidia-sm121", "linux/arm64"))
        self.assertEqual(r.stdout.strip(), "myllmbox/glm53-flash-cluster-vllm:v2", "the tag the pair ran before 2026-10-30")


class DeepseekV4Image(unittest.TestCase):
    """The pair's DeepSeek-V4-Flash image (deploy/images/vllm, stage deepseek-v4-b12x, operator 2026-10-23): the B12X
    base pinned by DIGEST, entered as `vllm serve` (the base enters through nvidia_entrypoint.sh, and tp-rank passes
    vLLM's arguments as the command), its tag the catalog's — built by the ONE shared deploy/images/build.sh."""

    DIR = "deploy/deepseek-v4-flash-cluster-vllm/image"
    PIN = "deploy/images/vllm/deepseek-v4-b12x.nvidia-sm121.env"

    def test_the_base_is_pinned_by_digest_and_entered_as_vllm_serve(self):
        self.assertIn("BASE=eugr/spark-vllm-b12x@sha256:"
                      "03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595", _code_lines(self.PIN))
        lines = _stage("deploy/images/vllm/Dockerfile", "deepseek-v4-b12x").splitlines()
        self.assertIn('ENTRYPOINT ["vllm", "serve"]', lines)
        self.assertFalse([ln for ln in lines if ln.startswith("RUN ")], "no RUN: an amd64 host builds the arm64 image")

    def test_both_image_dirs_build_through_the_one_shared_script(self):
        for d, build in ((self.DIR, "deepseek-v4-b12x"), (Glm53Image.DIR, "glm53")):
            with self.subTest(dir=d):
                self.assertIn("images/build.sh", _text(f"{d}/build.sh"))
                r = subprocess.run(["bash", f"{d}/build.sh", "tag"], cwd=REPO, capture_output=True, text=True)
                want = matrix_tag("vllm", build, "nvidia-sm121", "linux/arm64")
                self.assertEqual((r.returncode, r.stdout.strip()), (0, want), r.stderr)

    def test_a_cell_the_matrix_lacks_is_refused_naming_it(self):
        r = subprocess.run(["bash", "deploy/image-tag.sh", "vllm", "glm53", "nvidia-sm120", "linux/amd64"],
                           cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nvidia-sm120", r.stderr)


class PersonalIdentifiers(unittest.TestCase):
    """DEP-34 and DEP-35: no operator mailbox, another person's login, or absolute operator path."""

    def test_the_committed_text_names_no_operator_mailbox_or_other_login(self):
        import hashlib
        files = ("chat/vllm_chat/mint_session.py", "deploy/nginx-edge-chat-vhost.example.conf",
                 "deploy/nginx-edge-chat.example.conf", "chat/tests/test_auth.py") if CHAT else ()
        self.assertEqual(bool(files), CHAT, "the files this guard reads are the chat front-end's: none without it")
        for rel in files:
            with self.subTest(file=rel):
                tokens = set(re.findall(r"[\w.+-]+@[\w.-]+|\w+", _text(rel).lower()))
                hits = {t for t in tokens if hashlib.sha256(t.encode()).hexdigest() in REMOVED_IDENTIFIERS}
                self.assertFalse(hits, f"{rel} names a removed personal identifier")

    def test_the_usage_mcp_examples_mount_from_the_checkout_not_an_absolute_path(self):
        self.assertNotIn("/home/user", _text("usage-mcp/README.md"))


if __name__ == "__main__":
    unittest.main()


@needs_chat
class ChatRedeployAuthCheck(unittest.TestCase):
    """The sanctioned chat deploy judges the auth file with the REVISION'S OWN load_config — which
    now takes the issuer (CHAT_AUTH_ISSUER) too — and checks the JWKS of the issuer that revision
    will trust, never a literal host. A 2-argument call would refuse every deploy of a revision
    whose load_config wants the issuer; a literal JWKS host would vouch for the wrong gate."""

    def test_the_issuer_comes_from_the_revisions_compose_and_reaches_load_config(self):
        text = _text("deploy/chat-redeploy.sh")
        self.assertIn("CHAT_AUTH_ISSUER", text)
        self.assertIn("inspect.signature(auth.load_config)", text)

    def test_the_jwks_checked_is_the_revisions_own_not_a_literal(self):
        code = "\n".join(_code_lines("deploy/chat-redeploy.sh"))
        self.assertNotIn("auth.example.com", code)
        self.assertIn("jwks_uri", code)


#: SHA-256 of the lowercase words that named ONE deployment in this tree before it was published: its boxes, their
#: hostnames, its edge, its sibling projects and their sessions, its disks and mount roots, a private NIC. Digests,
#: so this file does not republish them. A word is caught whole (\w+ over the lowercased text), never inside another.
#: ONE list for every guard: lb/tests/publishable_words.json (the lb tree guard and the *.node.js guards read it too),
#: so a word added for one part of the tree is hunted in all of it.
DEPLOYMENT_WORDS = frozenset(json.loads((REPO / "lb/tests/publishable_words.json").read_text(encoding="utf-8"))["digests"])
#: Upstream names that carry one of those words as a product (NVIDIA's DGX Spark and the recipes built on it): cut
#: out before the text is tokenized, so a vendor's repo name is not read as the operator's box.
_PRODUCT = "sp" + "ark"      # spelled in two halves so this guard's own source does not carry the word
VENDOR_PHRASES = re.compile(rf"dgx[ _-]{_PRODUCT}s?|eugr/{_PRODUCT}-[\w.-]+|{_PRODUCT}-vllm-[\w.-]+|{_PRODUCT}line[\w-]*",
                            re.IGNORECASE)
#: A personal sign-in site's app launcher, which the deployment's own proxy may inject but no shipped file may.
_L = "launch" + "er"          # in halves too: the guard must not carry the one word it hunts
LAUNCHER = re.compile(rf"{_L}\.js|/plugin/{_L}|data-[\w-]*{_L}|(?<![\w-]){_L}-(?:mount|div)")
#: The operator's private /30 links (RFC 1918 10.10.x.y); a shipped file names RFC 5737 TEST-NET addresses only.
PRIVATE_LINK = re.compile(r"(?<![\d.])10\.10\.\d{1,3}\.\d{1,3}(?![\d.])")
#: What the tree guard walks: every service dir this test file answers for, and the root scripts.
PUBLISHED_DIRS = ("deploy", "frpc", "power", "usage-mcp", "dashboard", "crawl")
#: The root docs ship too (README, the agent guides, NOTICE, CHANGELOG, SECURITY, the root .env template, docs/).
PUBLISHED_ROOT_FILES = ("seed-broker-sidecar.sh", "README.md", "CLAUDE.md", "AGENTS.md", "NOTICE", "CHANGELOG.md",
                        "SECURITY.md", "SETUP.md", ".env.example", ".gitignore", ".gitleaks.toml")
PUBLISHED_ROOT_DIRS = ("docs",)
#: AGENTS.md on the private branch carries a development section under this marker; the release recipe
#: (PUBLISHING.md) cuts the marker and everything after it, and so does this guard — the text ABOVE it is what ships.
PRIVATE_MARKER = "<!-- PRIVATE-BELOW"
SKIP_PARTS = {"node_modules", "__pycache__", ".pytest_cache", ".git", "state"}


#: What the release recipe drops (PUBLISHING.md): the edge and the chat front-end ship nowhere, so the tree guard
#: reads them as the release does — absent. Their own checks are the needs_gate / needs_chat tests.
RELEASE_DROPS = EDGE_FILES + CHAT_FILES


def _glob_re(rel: str) -> str:
    """A CHAT_FILES / EDGE_FILES entry as a regex: literal, but `*` is any run of path-name characters."""
    return re.escape(rel).replace(r"\*", r"[\w.-]*")


def _present(rel: str) -> bool:
    return any(REPO.glob(rel))


def _dropped(rel: Path) -> bool:
    return any(re.fullmatch(_glob_re(d), str(q)) for d in RELEASE_DROPS for q in (rel, *rel.parents))


def _published_files():
    for d in PUBLISHED_DIRS:
        for p in sorted((REPO / d).rglob("*")):
            rel = p.relative_to(REPO)
            if p.is_file() and not (SKIP_PARTS & set(rel.parts)) and not _dropped(rel):
                yield p
    for f in PUBLISHED_ROOT_FILES:
        yield REPO / f
    for d in PUBLISHED_ROOT_DIRS:
        for p in sorted((REPO / d).rglob("*")):
            if p.is_file():
                yield p


def _read(p: Path) -> str | None:
    """The file's text as it SHIPS: a private section under PRIVATE_MARKER is cut, as the release recipe cuts it."""
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None      # a binary (an image, a font) names nothing
    return text.split(PRIVATE_MARKER, 1)[0]


#: Package metadata is the one place the author's own mailbox is allowed (the authors field); its address is
#: cut out before that file is tokenized, nothing else.
METADATA_FILES = ("dashboard/package.json",)
MAILBOX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _deployment_words(text: str, *, metadata: bool = False) -> set[str]:
    import hashlib
    if metadata:
        text = MAILBOX.sub(" ", text)
    cleaned = VENDOR_PHRASES.sub(" ", text.lower())
    return {t for t in set(re.findall(r"\w+", cleaned)) if hashlib.sha256(t.encode()).hexdigest() in DEPLOYMENT_WORDS}


class PublishedTreeNamesNoDeployment(unittest.TestCase):
    """DEP-40: the published tree (deploy/, frpc/, power/, usage-mcp/, dashboard/, crawl/, the root scripts) names no
    box, host, edge, disk, sibling project or private link of one deployment, and no personal sign-in site's app
    launcher. A deployment's own values are its gitignored files — each committed `<name>.example.<ext>` has its real
    name in that directory's .gitignore, and the real file is never tracked."""

    def test_no_file_names_a_box_host_edge_or_sibling_of_one_deployment(self):
        hits = {}
        for p in _published_files():
            text = _read(p)
            if text is None:
                continue
            words = _deployment_words(text, metadata=str(p.relative_to(REPO)) in METADATA_FILES)
            if words:
                hits[str(p.relative_to(REPO))] = sorted(words)
        self.assertFalse(hits, f"a deployment's own words are committed in: {hits}")

    def test_no_file_names_a_private_link_address(self):
        hits = {str(p.relative_to(REPO)): sorted(set(PRIVATE_LINK.findall(t)))
                for p in _published_files() if (t := _read(p)) and PRIVATE_LINK.search(t)}
        self.assertFalse(hits, f"a private link address is committed in: {hits}")

    def test_no_file_mounts_a_personal_app_launcher(self):
        hits = [str(p.relative_to(REPO)) for p in _published_files() if (t := _read(p)) and LAUNCHER.search(t)]
        self.assertFalse(hits, f"an app launcher is injected by: {hits}")

    def test_the_signed_in_edge_ships_whole_or_not_at_all_and_the_recipe_drops_it(self):
        """The edge is gate/ plus the files that only wire it (EDGE_FILES): a tree with some of them would carry a
        door nothing can build. PUBLISHING.md's recipe names every one of them and the GATE-BELOW cut."""
        present = {rel for rel in EDGE_FILES if _present(rel)}
        self.assertIn(present, (set(), set(EDGE_FILES)), f"a partial edge: {sorted(present)}")
        example = (REPO / "deploy/standalone/.env.example").read_text(encoding="utf-8")
        self.assertEqual("GATE-BELOW" in example, GATE, "the door lines of the server file go with the edge")
        recipe = (REPO / "PUBLISHING.md").read_text(encoding="utf-8") if (REPO / "PUBLISHING.md").exists() else ""
        if recipe:
            for rel in EDGE_FILES:
                self.assertRegex(recipe, _glob_re(rel), f"the release recipe must drop {rel}")
            self.assertIn("GATE-BELOW/,$d", recipe, "the release recipe must cut the server file at the marker")

    def test_the_chat_front_end_ships_whole_or_not_at_all_and_the_recipe_drops_it(self):
        """The chat is chat/ plus the files that exist only for it (CHAT_FILES): a tree with some of them would carry
        a deploy script, a tunnel or a door for a service that is not there. PUBLISHING.md's recipe names every one."""
        present = {rel for rel in CHAT_FILES if _present(rel)}
        self.assertIn(present, (set(), set(CHAT_FILES)), f"a partial chat front-end: {sorted(present)}")
        recipe = (REPO / "PUBLISHING.md").read_text(encoding="utf-8") if (REPO / "PUBLISHING.md").exists() else ""
        if recipe:
            for rel in CHAT_FILES:
                self.assertRegex(recipe, _glob_re(rel), f"the release recipe must drop {rel}")
            self.assertIn("catflops_chat", recipe, "the recipe says where the chat went")

    def test_the_shipped_docs_name_no_chat_front_end(self):
        """What ships (README, CHANGELOG, the agent guides above their marker, the root .env template, SECURITY,
        NOTICE) describes no chat service: it is a separate project, and any OpenAI- or Anthropic-compatible
        client works against the LB. The API path /v1/chat/completions and the dashboard's CHAT_MODEL are not it."""
        hits = {}
        for rel in ("README.md", "CHANGELOG.md", "AGENTS.md", "CLAUDE.md", ".env.example", "SECURITY.md", "SETUP.md",
                    "NOTICE"):
            found = sorted(set(CHAT_WORDS.findall(_read(REPO / rel) or "")))
            if found:
                hits[rel] = found
        self.assertFalse(hits, f"a shipped doc names the chat front-end: {hits}")
        self.assertTrue(CHAT_WORDS.search("cd chat/ && docker compose up"))
        self.assertFalse(CHAT_WORDS.search("the OpenAI /v1/chat/completions API; the dashboard's CHAT_MODEL"))

    def test_the_shipped_docs_cite_no_path_the_release_drops_and_name_packages_that_exist(self):
        """A reader of the release has no worklog/, gate/ or PUBLISHING.md to follow, nor any file the recipe drops
        with the edge or the chat (EDGE_FILES, CHAT_FILES): the shipped docs (the root docs, docs/, every service
        README) cite none of them, and the agent guides name a package directory that exists for every
        `<service>/<package>` they list (power's is fleet_power)."""
        files = "|".join(_glob_re(rel) for rel in EDGE_FILES + CHAT_FILES if "/" in rel)
        dropped = re.compile(rf"(?<![\w/-])(worklog/|gate/|PUBLISHING\.md|{files})")
        docs = ["README.md", "CHANGELOG.md", "AGENTS.md", "CLAUDE.md", "SECURITY.md", "SETUP.md", "NOTICE"]
        docs += [str(p.relative_to(REPO)) for p in sorted(REPO.glob("docs/*.md"))]
        docs += [str(p.relative_to(REPO)) for p in sorted(REPO.glob("*/README.md"))
                 if p.parts[-2] not in ("chat", "gate", "worklog")]
        hits = {rel: sorted(set(dropped.findall(_read(REPO / rel) or ""))) for rel in docs}
        hits = {rel: found for rel, found in hits.items() if found}
        self.assertFalse(hits, f"a shipped doc cites a path the release drops: {hits}")
        self.assertTrue(dropped.search("see worklog/foo.md") and dropped.search("gate/vllm_gate/app.py"))
        self.assertTrue(dropped.search("(deploy/nginx-edge-vllm.example.conf)"))
        self.assertFalse(dropped.search("the gateway/panel; /v1/gate/x; a toll-gate/"))
        for rel in ("AGENTS.md", "CLAUDE.md"):
            for service, package in re.findall(r"`([\w-]+)/([a-z_]+)`", _read(REPO / rel) or ""):
                if (REPO / service).is_dir() and service in ("lb", "power", "usage-mcp"):
                    self.assertTrue((REPO / service / package).is_dir(), f"{rel} names {service}/{package}, which does not exist")

    def test_the_private_section_of_the_agent_guides_is_what_the_release_recipe_cuts(self):
        """AGENTS.md and CLAUDE.md may carry the private branch's notes only UNDER the marker, once each; PUBLISHING.md's
        recipe cuts each from that marker to the end (`sed '/<!-- PRIVATE-BELOW/,$d'`), and the guard reads the same
        cut."""
        recipe = (REPO / "PUBLISHING.md").read_text(encoding="utf-8") if (REPO / "PUBLISHING.md").exists() else ""
        for guide in ("AGENTS.md", "CLAUDE.md"):
            with self.subTest(guide=guide):
                raw = (REPO / guide).read_text(encoding="utf-8")
                self.assertLessEqual(raw.count(PRIVATE_MARKER), 1)
                shipped = _read(REPO / guide)
                self.assertNotIn(PRIVATE_MARKER, shipped)
                self.assertEqual(shipped, raw.split(PRIVATE_MARKER, 1)[0])
                if recipe:
                    self.assertIn(f"git show oss-prep:{guide} | sed '/<!-- PRIVATE-BELOW/,$d'", recipe,
                                  f"the release recipe must cut {guide} at the marker")

    def test_the_guard_reads_digests_and_its_patterns_match_what_they_should(self):
        self.assertEqual(_deployment_words(f"the edge, box-a and box-b over 192.0.2.2; eugr/{_PRODUCT}-vllm-docker on DGX {_PRODUCT}"), set())
        self.assertTrue(LAUNCHER.search(f'<script src="https://auth.example.com/plugin/{_L}.js" defer></script>'))
        self.assertTrue(LAUNCHER.search(f'<div data-auth-{_L}></div>'))
        self.assertFalse(LAUNCHER.search(f"the {_L} still declared failure 600 s later"))
        self.assertTrue(PRIVATE_LINK.search("ssh user@10.10." + "10.2"))
        self.assertFalse(PRIVATE_LINK.search("ssh user@192.0.2.2 or 10.0.0.1 or 110.10." + "10.2"))
        self.assertTrue(all(len(d) == 64 for d in DEPLOYMENT_WORDS))

    def _examples(self):
        for p in _published_files():
            if ".example." in p.name:
                yield p, p.with_name(p.name.replace(".example", "", 1))

    def test_every_example_file_has_its_real_name_gitignored_in_its_directory(self):
        missing = []
        for example, real in self._examples():
            ignore = example.parent / ".gitignore"
            lines = ignore.read_text().splitlines() if ignore.is_file() else []
            if real.name not in [ln.strip() for ln in lines]:
                missing.append(f"{real.relative_to(REPO)} (expected in {ignore.relative_to(REPO)})")
        self.assertFalse(missing, f"a real deployment file is not gitignored next to its example: {missing}")

    def test_no_real_deployment_file_is_tracked(self):
        import shutil
        if shutil.which("git") is None:
            self.skipTest("git is not in this image: `git ls-files` cannot be asked (run on a host with git)")
        tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n"))
        tracked_real = [str(real.relative_to(REPO)) for _, real in self._examples()
                        if str(real.relative_to(REPO)) in tracked]
        self.assertFalse(tracked_real, f"a real deployment file is tracked: {tracked_real}")
