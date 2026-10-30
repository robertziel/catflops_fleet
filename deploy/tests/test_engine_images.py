"""deploy/images — ONE Dockerfile per engine, its base chosen per accelerator by a pin file, built by ONE script.

An engine's image is the same recipe on every accelerator: an upstream image (BASE) plus this project's patches or
overlay files. What differs per (build × accelerator) is data — the base, pinned by digest, the patches, the CPU
platform — so it lives in `deploy/images/<engine>/<build>.<accelerator>.env` (KEY=VALUE, read line by line, never
sourced), never in a Dockerfile or a script. A cell nobody has pinned has no file, and the build refuses it naming the
file it looked for: never a nearest base. The tag is the LB's image matrix's (deploy/image-tag.sh), so a build lands
under exactly the name a Start launches.

The build is run for real with a fake `docker` and a fake `ssh` on PATH that record their argv and change nothing:
nothing here builds, pushes or dials anything.

    docker run --rm -v "$PWD":/repo:ro -w /repo python:3.12-slim python -m unittest discover -s deploy/tests
"""
from __future__ import annotations

import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "deploy" / "images"
BUILD = IMAGES / "build.sh"
ENGINES = ("vllm", "sglang", "tensorfold")
ACCELERATORS = {p.stem for p in (REPO / "lb/vllm_lb/hardware/accelerators").glob("*.yml")}
#: Pin keys the build reads itself; every other key is handed to `docker build` as a --build-arg.
NOT_BUILD_ARGS = ("PLATFORM", "TARGET", "BASE_ID")
PIN_LINE = re.compile(r"[A-Z][A-Z0-9_]*=\S.*")
DIGEST = re.compile(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}")
#: A remote build runs under BuildKit, which builds only the stages its --target needs (the remote dockerd may predate
#: BuildKit being its default).
REMOTE_BUILD = "DOCKER_BUILDKIT=1 docker build"

#: THE CURRENT PINS, as the per-image build scripts and Dockerfiles carried them before they were folded here.
PINS = {
    ("sglang", "fleet.4", "nvidia-sm121"): {
        "PLATFORM": "linux/arm64",
        "BASE": "lmsysorg/sglang@sha256:00205b89f74691f76a0ffbd6846376d9323971930a5d59bf63a65dadc7d67927",
        "PATCHES": "anthropic-tool-order qwen3-coder-truncated-call anthropic-thinking-display qwen4-exp-mtp-quantized "
                   "anthropic-effort-xhigh qsa-fp8-kv"},
    ("vllm", "glm53", "nvidia-sm121"): {
        "PLATFORM": "linux/arm64", "TARGET": "glm53",
        "BASE": "ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:"
                "4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6"},
    ("vllm", "deepseek-v4-b12x", "nvidia-sm121"): {
        "PLATFORM": "linux/arm64", "TARGET": "deepseek-v4-b12x",
        "BASE": "eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"},
    ("tensorfold", "native", "nvidia-sm121"): {
        "PLATFORM": "linux/arm64",
        "BASE": "tensorfold-qwen38fn:zig-db28187-ghcr",
        "BASE_ID": "sha256:5eb5d4d384773543109baa26783fc6ea38ad619dff330f01672e05c01721cfb0",
        "TF_REPO": "https://github.com/ashhart/TensorFold.git",
        "TF_REF": "db281878ddb836fd0df510d8771ecb7e0fe47d26",
        "RECIPE_REPO": "https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold.git",
        "RECIPE_REF": "be3ed857ab1055ca47b36134d22691f00f2bb94a",
        "ZIG_VERSION": "0.17.0",
        "ZIG_PLATFORM": "aarch64-linux",
        "ZIG_SHA256": "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8",
        "FLEET_PATCHES": "0013-anthropic-merge-late-system.patch 0014-work-counters.patch"},
}


def read_pin(path: Path) -> dict[str, str]:
    pin = {}
    for ln in path.read_text().splitlines():
        if ln.strip() and not ln.lstrip().startswith("#"):
            key, _, value = ln.partition("=")
            pin[key] = value
    return pin


def matrix_tag(engine: str, build: str, accelerator: str, platform: str) -> str:
    r = subprocess.run(["bash", "deploy/image-tag.sh", engine, build, accelerator, platform], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def stages(dockerfile: Path) -> dict[str, list[str]]:
    """A Dockerfile's instructions by stage name (`FROM … AS <name>`; an unnamed stage is its index), comments and
    continuation lines joined, so a test can hold one stage's exact content."""
    out: dict[str, list[str]] = {}
    current = None
    logical = ""
    for raw in dockerfile.read_text().splitlines():
        if not logical and (not raw.strip() or raw.lstrip().startswith("#")):
            continue
        logical += raw.rstrip("\\").strip() + " " if raw.endswith("\\") else raw.strip()
        if raw.endswith("\\"):
            continue
        ins, logical = logical, ""
        if ins.startswith("FROM "):
            m = re.search(r"\sAS\s+(\S+)$", ins)
            current = m.group(1) if m else str(len(out))
            out[current] = []
        if current is not None:
            out[current].append(re.sub(r"\s+", " ", ins))
    return out


class Fakes:
    """`docker` and `ssh` on PATH that log each call's argv (one arg a line, a `--` line between calls), drain a
    piped stdin (a build context, an image stream), and answer an `image inspect` with ID."""

    def __init__(self, tmp: Path, image_id: str = "sha256:" + "0" * 64):
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.log = tmp / "calls.log"
        for name in ("docker", "ssh"):
            script = self.bin / name
            script.write_text(
                "#!/bin/sh\n"
                # ONE append per call: a pipeline (`docker save | ssh … docker load`, the context streamed into a remote
                # build) runs two fakes at once, and line-by-line appends interleaved their argv in the log (a flaky
                # RemoteBuild, 2026-10-30). The record is built first and written by one printf — one write(2) under
                # O_APPEND, which does not interleave at this size.
                f"rec=$(echo '{name}'; for a in \"$@\"; do printf '%s\\n' \"$a\"; done; echo --)\n"
                f"printf '%s\\n' \"$rec\" >> '{self.log}'\n"
                "cat >/dev/null\n"
                f"case \"$*\" in *inspect*) echo '{image_id}' ;; esac\n")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        out, cur = [], []
        for ln in self.log.read_text().splitlines():
            if ln == "--":
                out.append(cur)
                cur = []
            else:
                cur.append(ln)
        return out


def remote(fakes: Fakes) -> list[tuple[str, str]]:
    """Each ssh call as (box, the command line the box's shell runs): `ssh -o … -o … <box> '<command>'`."""
    out = []
    for c in fakes.calls():
        if c[0] == "ssh":
            assert c[1:5] == ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"] and len(c) == 7, c
            out.append((c[5], c[6]))
    return out


def run(script: Path, args, fakes: Fakes | None, env=None, cwd=REPO):
    clean = {"PATH": (f"{fakes.bin}:" if fakes else "") + "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/tester"}
    clean.update(env or {})
    return subprocess.run(["bash", str(script), *args], env=clean, capture_output=True, text=True, timeout=60,
                          cwd=cwd, stdin=subprocess.DEVNULL)


def scratch(case: unittest.TestCase) -> Path:
    return Path(case.enterContext(tempfile.TemporaryDirectory()))


class EngineDockerfiles(unittest.TestCase):
    def test_one_dockerfile_per_engine_whose_base_is_passed_never_named(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                df = IMAGES / engine / "Dockerfile"
                lines = [ln for ln in df.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
                self.assertEqual(lines[0], "ARG BASE", "BASE is declared before the first FROM")
                froms = [ln for ln in lines if ln.startswith("FROM ")]
                self.assertTrue(froms)
                self.assertRegex(froms[0], r"^FROM \$\{BASE:\?[^}]+\}")
                for ln in froms:
                    self.assertRegex(ln, r"^FROM \$\{BASE(:\?[^}]+)?\}( AS \S+)?$", "a base is the pin's, never a literal")
        self.assertEqual(sorted(p.name for p in IMAGES.iterdir() if p.is_dir()), sorted(ENGINES))


class PinFiles(unittest.TestCase):
    def _pins(self):
        for engine in ENGINES:
            for p in sorted((IMAGES / engine).glob("*.env")):
                yield engine, p

    def test_every_pin_names_a_matrix_cell_and_is_plain_key_value(self):
        seen = 0
        for engine, p in self._pins():
            seen += 1
            with self.subTest(pin=str(p.relative_to(REPO))):
                build, _, accelerator = p.name[:-len(".env")].rpartition(".")
                self.assertIn(accelerator, ACCELERATORS, "<build>.<accelerator>.env names a known accelerator")
                lines = [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
                for ln in lines:
                    self.assertRegex(ln, PIN_LINE, "KEY=VALUE, a value always present")
                keys = [ln.split("=", 1)[0] for ln in lines]
                self.assertEqual(len(keys), len(set(keys)), "a key once")
                pin = read_pin(p)
                self.assertTrue({"BASE", "PLATFORM"} <= set(pin))
                if "BASE_ID" not in pin:
                    self.assertRegex(pin["BASE"], DIGEST, "a published base is pinned by digest")
                matrix_tag(engine, build, accelerator, pin["PLATFORM"])
        self.assertEqual(seen, len(PINS))

    def test_the_pins_are_the_ones_the_images_were_built_from(self):
        for (engine, build, accelerator), want in PINS.items():
            with self.subTest(cell=(engine, build, accelerator)):
                self.assertEqual(read_pin(IMAGES / engine / f"{build}.{accelerator}.env"), want)

    def test_a_vllm_pins_target_is_a_stage_of_the_engine_dockerfile(self):
        names = set(stages(IMAGES / "vllm" / "Dockerfile"))
        for p in (IMAGES / "vllm").glob("*.env"):
            self.assertIn(read_pin(p)["TARGET"], names, p.name)

    def test_each_named_patch_is_in_the_engine_dir(self):
        for p in PINS[("sglang", "fleet.4", "nvidia-sm121")]["PATCHES"].split():
            self.assertTrue((IMAGES / "sglang" / f"{p}.patch").is_file(), p)
        tf = PINS[("tensorfold", "native", "nvidia-sm121")]["FLEET_PATCHES"].split()
        self.assertEqual(tf, sorted(p.name for p in (IMAGES / "tensorfold").glob("*.patch")))


class SharedBuild(unittest.TestCase):
    """deploy/images/build.sh <engine> <build> <accelerator> <action>: the pin, the matrix's tag, one docker build."""

    def _docker_build(self, calls):
        builds = [c for c in calls if c[:2] == ["docker", "build"]]
        self.assertEqual(len(builds), 1, calls)
        return builds[0][2:]

    def test_tag_prints_the_matrix_tag(self):
        for (engine, build, accelerator), pin in PINS.items():
            with self.subTest(cell=(engine, build, accelerator)):
                r = run(BUILD, [engine, build, accelerator, "tag"], None)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout.strip(), matrix_tag(engine, build, accelerator, pin["PLATFORM"]))

    def test_build_hands_docker_the_platform_tag_target_and_every_pin_as_a_build_arg(self):
        fakes = Fakes(scratch(self))
        r = run(BUILD, ["vllm", "glm53", "nvidia-sm121", "build"], fakes)
        self.assertEqual(r.returncode, 0, r.stderr)
        tag = matrix_tag("vllm", "glm53", "nvidia-sm121", "linux/arm64")
        self.assertEqual(self._docker_build(fakes.calls()),
                         ["--platform", "linux/arm64", "-t", tag, "--target", "glm53",
                          "--build-arg", "BASE=" + PINS[("vllm", "glm53", "nvidia-sm121")]["BASE"],
                          str(IMAGES / "vllm")])

    def test_a_value_with_spaces_stays_one_build_arg(self):
        fakes = Fakes(scratch(self))
        r = run(BUILD, ["sglang", "fleet.4", "nvidia-sm121", "build"], fakes)
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = self._docker_build(fakes.calls())
        self.assertIn("PATCHES=" + PINS[("sglang", "fleet.4", "nvidia-sm121")]["PATCHES"], argv)
        self.assertNotIn("--target", argv, "a pin without TARGET builds the Dockerfile's last stage")

    def test_a_base_id_is_checked_before_the_build_and_never_passed(self):
        pin = PINS[("tensorfold", "native", "nvidia-sm121")]
        good = Fakes(scratch(self), image_id=pin["BASE_ID"])
        r = run(BUILD, ["tensorfold", "native", "nvidia-sm121", "build"], good)
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = self._docker_build(good.calls())
        self.assertFalse([a for a in argv if a.split("=")[0] in NOT_BUILD_ARGS])
        self.assertIn("ZIG_PLATFORM=aarch64-linux", argv)
        bad = Fakes(scratch(self))
        r = run(BUILD, ["tensorfold", "native", "nvidia-sm121", "build"], bad)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(pin["BASE_ID"], r.stderr)
        self.assertFalse([c for c in bad.calls() if c[:2] == ["docker", "build"]], "nothing built on the wrong base")

    def test_a_cell_with_no_pin_is_refused_naming_the_file(self):
        fakes = Fakes(scratch(self))
        for action in ("build", "tag"):
            r = run(BUILD, ["sglang", "fleet.4", "nvidia-sm120", action], fakes)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn("deploy/images/sglang/fleet.4.nvidia-sm120.env", r.stderr)
        self.assertEqual(fakes.calls(), [], "nothing run")

    def test_usage_without_the_cell_or_with_an_unknown_action(self):
        fakes = Fakes(scratch(self))
        for args in ([], ["vllm", "glm53"], ["vllm", "glm53", "nvidia-sm121", "push"]):
            r = run(BUILD, args, fakes)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn("usage", r.stderr.lower())
        self.assertEqual(fakes.calls(), [])

    def _copy_tree(self) -> Path:
        """A writable copy of what the build reads, so a test can plant a bad pin."""
        root = scratch(self)
        shutil.copytree(IMAGES, root / "deploy/images")
        shutil.copy(REPO / "deploy/image-tag.sh", root / "deploy/image-tag.sh")
        shutil.copytree(REPO / "lb/vllm_lb/library/images", root / "lb/vllm_lb/library/images")
        return root

    def test_a_malformed_pin_is_refused_naming_the_line(self):
        for bad, said in (("BASE=a\nPLATFORM=linux/arm64\nexport X=1\n", "export X=1"),
                          ("BASE=a\nPLATFORM=linux/arm64\nBASE=b\n", "BASE"),
                          ("BASE=\nPLATFORM=linux/arm64\n", "BASE"),
                          ("PLATFORM=linux/arm64\nPATCHES=x\n", "BASE"),
                          ("BASE=a\nPATCHES=x\n", "PLATFORM")):
            with self.subTest(pin=bad):
                root = self._copy_tree()
                (root / "deploy/images/sglang/fleet.4.nvidia-sm121.env").write_text(bad)
                fakes = Fakes(scratch(self))
                r = run(root / "deploy/images/build.sh", ["sglang", "fleet.4", "nvidia-sm121", "build"], fakes,
                        cwd=root)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertIn(said, r.stderr)
                self.assertEqual(fakes.calls(), [])

    def test_a_pin_platform_the_matrix_lacks_is_refused(self):
        root = self._copy_tree()
        pin = root / "deploy/images/vllm/glm53.nvidia-sm121.env"
        pin.write_text(pin.read_text().replace("PLATFORM=linux/arm64", "PLATFORM=linux/amd64"))
        fakes = Fakes(scratch(self))
        r = run(root / "deploy/images/build.sh", ["vllm", "glm53", "nvidia-sm121", "build"], fakes, cwd=root)
        self.assertEqual(r.returncode, 1)
        self.assertIn("linux/amd64", r.stderr)
        self.assertEqual(fakes.calls(), [])

    def test_ship_and_check_need_the_pair_named(self):
        fakes = Fakes(scratch(self))
        for action in ("ship", "check"):
            r = run(BUILD, ["vllm", "glm53", "nvidia-sm121", action], fakes)
            self.assertEqual(r.returncode, 2)
            self.assertIn("BOXES", r.stderr)
        self.assertFalse([c for c in fakes.calls() if c[0] == "ssh"])

    def test_ship_streams_the_built_image_to_each_box(self):
        fakes = Fakes(scratch(self))
        r = run(BUILD, ["vllm", "glm53", "nvidia-sm121", "ship"], fakes, env={"BOXES": "box-a box-b"})
        self.assertEqual(r.returncode, 0, r.stderr)
        tag = matrix_tag("vllm", "glm53", "nvidia-sm121", "linux/arm64")
        self.assertEqual([box for box, cmd in remote(fakes) if cmd == "docker load"], ["box-a", "box-b"])
        self.assertIn(["docker", "save", tag], fakes.calls())


class RemoteBuild(unittest.TestCase):
    """`remote <build-box> [<box> ...]`: the context streamed over ssh, ONE build on the first box, copied to every
    other and checked by image id — a TP pair runs the same bits on each rank (two builds of one tree need not be
    byte-identical)."""

    def test_one_build_on_the_first_box_copied_to_the_rest_and_checked(self):
        pin = PINS[("tensorfold", "native", "nvidia-sm121")]
        fakes = Fakes(scratch(self), image_id=pin["BASE_ID"])
        r = run(BUILD, ["tensorfold", "native", "nvidia-sm121", "remote", "box-a", "box-b"], fakes)
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = remote(fakes)
        self.assertFalse([c for c in fakes.calls() if c[0] == "docker"], "nothing runs on this host")
        builds = [(box, cmd) for box, cmd in calls if cmd.startswith(REMOTE_BUILD)]
        self.assertEqual(len(builds), 1, calls)
        box, cmd = builds[0]
        self.assertEqual(box, "box-a")
        self.assertTrue(cmd.startswith(REMOTE_BUILD + " -q --platform linux/arm64 -t "), cmd)
        self.assertTrue(cmd.endswith(" -"), "the context is the tar on stdin")
        self.assertIn("FLEET_PATCHES=0013-anthropic-merge-late-system.patch\\ 0014-work-counters.patch", cmd,
                      "the remote shell gets each value quoted")
        self.assertNotIn("BASE_ID", cmd)
        base_checks = [box for box, cmd in calls if cmd.endswith(" " + pin["BASE"]) and "inspect" in cmd]
        self.assertEqual(base_checks, ["box-a", "box-b"])
        self.assertEqual([box for box, cmd in calls if cmd == "docker load"], ["box-b"])

    def test_a_copy_whose_id_differs_is_refused(self):
        root = scratch(self)
        fakes = Fakes(root, image_id=PINS[("tensorfold", "native", "nvidia-sm121")]["BASE_ID"])
        # the second box answers a different id for everything after its base check
        ssh = fakes.bin / "ssh"
        ssh.write_text(ssh.read_text().replace(
            "case \"$*\" in *inspect*)",
            "case \"$*\" in *box-b*inspect*tensorfold-qwen38fn:zig-db28187-f3e1300f4f62*) echo sha256:other; exit 0 ;; esac\n"
            "case \"$*\" in *inspect*)"))
        r = run(BUILD, ["tensorfold", "native", "nvidia-sm121", "remote", "box-a", "box-b"], fakes)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("box-b", r.stderr)
        self.assertIn("sha256:other", r.stderr)

    def test_remote_names_at_least_one_box(self):
        fakes = Fakes(scratch(self))
        r = run(BUILD, ["sglang", "fleet.4", "nvidia-sm121", "remote"], fakes)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(fakes.calls(), [])


class EntryPoints(unittest.TestCase):
    """The per-image scripts operators and docs call stay, as thin pins of a cell on the shared build."""

    CELLS = {"deploy/glm53-flash-cluster-vllm/image/build.sh": ("vllm", "glm53", "nvidia-sm121"),
             "deploy/deepseek-v4-flash-cluster-vllm/image/build.sh": ("vllm", "deepseek-v4-b12x", "nvidia-sm121"),
             "deploy/sglang-27b-tp2/image/build.sh": ("sglang", "fleet.4", "nvidia-sm121"),
             "deploy/tensorfold/image/build.sh": ("tensorfold", "native", "nvidia-sm121")}

    def test_each_delegates_to_the_shared_build_and_prints_its_tag(self):
        for rel, cell in self.CELLS.items():
            with self.subTest(script=rel):
                text = (REPO / rel).read_text()
                self.assertIn("images/build.sh", text)
                self.assertNotIn("docker build", text, "one build line, in the shared script")
                self.assertNotIn("FROM", text)
                r = run(REPO / rel, ["tag"], None)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout.strip(), matrix_tag(*cell, PINS[cell]["PLATFORM"]))

    def test_the_sglang_pair_build_needs_boxes_and_builds_once_for_both(self):
        fakes = Fakes(scratch(self))
        r = run(REPO / "deploy/sglang-27b-tp2/image/build.sh", [], fakes)
        self.assertEqual(r.returncode, 2)
        self.assertIn("BOXES", r.stderr)
        self.assertEqual(fakes.calls(), [])
        r = run(REPO / "deploy/sglang-27b-tp2/image/build.sh", [], fakes, env={"BOXES": "box-a box-b"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([box for box, cmd in remote(fakes) if cmd.startswith(REMOTE_BUILD)], ["box-a"])
        self.assertEqual([box for box, cmd in remote(fakes) if cmd == "docker load"], ["box-b"])

    def test_the_tensorfold_build_takes_its_boxes_as_arguments(self):
        pin = PINS[("tensorfold", "native", "nvidia-sm121")]
        fakes = Fakes(scratch(self), image_id=pin["BASE_ID"])
        r = run(REPO / "deploy/tensorfold/image/build.sh", ["box-a", "box-b"], fakes)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([box for box, cmd in remote(fakes) if cmd.startswith(REMOTE_BUILD)], ["box-a"])

    def test_no_shared_script_is_left_beside_the_one(self):
        self.assertFalse((REPO / "deploy/cluster-image-build.sh").exists(), "deploy/images/build.sh is the one build")
        self.assertFalse((REPO / "deploy/standalone/images/vllm-fleet.Dockerfile").exists(),
                         "the vLLM node image is deploy/images/vllm/Dockerfile --target priority")


class VllmStages(unittest.TestCase):
    """The vLLM images, one stage per build over the pinned BASE — each exactly the image it was before the fold."""

    S = None

    @classmethod
    def setUpClass(cls):
        cls.S = stages(IMAGES / "vllm" / "Dockerfile")

    def test_glm53_is_the_sm121_indexer_file_and_the_recipe_environment(self):
        self.assertEqual(self.S["glm53"], [
            "FROM ${BASE:?pass--build-arg-BASE=the-vLLM-image-pinned-by-digest} AS glm53",
            "COPY glm53/sparse_attn_indexer_kpool_sm121.py "
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
            "ENV TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1 "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENGINE_READY_TIMEOUT_S=3600"])

    def test_deepseek_is_its_environment_entrypoint_quant_config_and_label(self):
        self.assertEqual(self.S["deepseek-v4-b12x"], [
            "FROM ${BASE} AS deepseek-v4-b12x",
            "ENV CUTE_DSL_ARCH=sm_121a VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_MEGA_AOT_ARTIFACT=1 "
            "VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_MOE_SKIP_PADDING=0 "
            "VLLM_ENGINE_READY_TIMEOUT_S=3600",
            'ENTRYPOINT ["vllm", "serve"]',
            "COPY deepseek-v4-b12x/quant_config.py "
            "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/quant_config.py",
            "LABEL lb.io_uring=1"])

    def test_the_pair_stages_run_nothing_so_an_amd64_host_builds_them_for_arm64(self):
        for name in ("glm53", "deepseek-v4-b12x"):
            self.assertFalse([ln for ln in self.S[name] if ln.startswith("RUN ")], name)

    def test_priority_is_the_node_image_patched_and_gated(self):
        self.assertEqual(self.S["priority"], [
            "FROM ${BASE} AS priority",
            "COPY priority/ /opt/fleet-patches/",
            "RUN cd \"$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')\" "
            "&& patch -p1 --forward --batch < /opt/fleet-patches/priority.patch",
            "RUN cd /tmp && python3 /opt/fleet-patches/check_priority.py"])
        self.assertEqual(sorted(p.name for p in (IMAGES / "vllm/priority").iterdir()),
                         ["check_priority.py", "priority.patch"])

    def test_the_build_names_its_stage_since_buildkit_builds_only_what_the_target_needs(self):
        text = BUILD.read_text()
        self.assertIn("DOCKER_BUILDKIT=1", text)


if __name__ == "__main__":
    unittest.main()
