"""deploy/standalone — one server outside box-c's edge: its own GPU node, LB, gates, nginx and chat.

Each server has different free ports, addresses and domains, so none of them is a literal here:
every value comes from that server's deploy/standalone/.env, as `${VAR:?…}` — an unset one stops
compose with its name, never a default (no fallbacks) — and every variable is documented in
.env.example. nginx's config is a template the official image renders with envsubst, filtered to
EDGE_* so nginx's own $variables survive. up.sh is the one way to bring a server up: it feeds the
same server file to every compose project, starts only named services (the chat's base file also
holds an frpc that would claim box-c's tunnel), and refuses a door without sign-in that is not on
loopback.

    docker run --rm -v "$PWD":/repo:ro -w /repo python:3.12-slim python -m unittest discover -s deploy/tests
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from test_deploy import _deployment_words          # unittest discover -s deploy/tests
except ImportError:                                      # python -m unittest deploy.tests.…
    from deploy.tests.test_deploy import _deployment_words

ROOT = Path(__file__).resolve().parents[2]
SA = ROOT / "deploy" / "standalone"
#: How a node's container attaches its card, one compose override per `gpu_runtime` a hardware type may state
#: (lb/vllm_lb/hardware/profiles.py GPU_RUNTIMES): up.sh picks the file BY THAT VALUE, never by a vendor branch.
GPU_RUNTIMES = ("nvidia", "rocm")
NODE_GPU = tuple(f"node-gpu.{rt}.yml" for rt in GPU_RUNTIMES)
#: Which server the node runs, one compose override per engine family (lb/vllm_lb/fleet.py ENGINES, the catalog row's
#: `node_engine`): up.sh picks the file BY THAT VALUE (EDGE_NODE_ENGINE), never by a branch on a model.
ENGINES = ("vllm", "faster-whisper", "nemo", "sglang")
NODE_ENGINE = tuple(f"node-engine.{e}.yml" for e in ENGINES)
#: THE SIGNED-IN EDGE (docker-compose.yml: two gates + nginx on nginx.conf.template) is built against a private
#: sign-in client, so the public release ships WITHOUT it: PUBLISHING.md's recipe drops those two files and cuts
#: .env.example at GATE_MARKER (the door lines only the edge reads). Everything of it here is conditional on the
#: file being present; the rest of the standalone fleet (node, LB, bench, chat) is the same tree either way.
EDGE = (SA / "docker-compose.yml").exists()
GATE_MARKER = "GATE-BELOW"
TEMPLATE = "nginx.conf.template"
EDGE_COMPOSE = ("docker-compose.yml",) if EDGE else ()
EDGE_TEMPLATE = (TEMPLATE,) if EDGE else ()
#: The server-file lines ONLY the edge reads: the doors' listeners, the gates' binds, origin and issuer, the per-user
#: keys, the public chat door's listener.
DOOR_VARS = {"EDGE_LOCAL_LISTEN", "EDGE_GATE_LOCAL_BIND", "EDGE_LOCAL_ORIGIN", "EDGE_PUBLIC_LISTEN",
             "EDGE_GATE_PUBLIC_BIND", "EDGE_VLLM_AUDIENCE", "EDGE_PUBLIC_PROXY", "EDGE_GATE_API_KEYS",
             "EDGE_GATE_KEY_DAYS", "EDGE_GATE_KEY_DEFAULT_DAYS", "EDGE_GATE_KEY_OWNER_DAYS", "EDGE_GATE_KEYS_DIR",
             "EDGE_PUBLIC_CHAT_LISTEN", "EDGE_PUBLIC_VLLM_ORIGIN", "EDGE_AUTH_ISSUER"}
needs_edge = unittest.skipUnless(EDGE, "the signed-in edge is not in this tree (PUBLISHING.md drops it from the release)")
#: THE CHAT FRONT-END (chat.override.yml: the chat on this server's LB) is its own project, catflops_chat, so the public
#: release ships WITHOUT it: PUBLISHING.md's recipe drops chat/ and the override, and its server-file lines sit under
#: GATE_MARKER too. up.sh then has no `chat` target, and a full bring-up is node lb bench (+edge where the edge is).
CHAT = (SA / "chat.override.yml").exists()
needs_chat = unittest.skipUnless(CHAT, "the chat front-end is not in this tree (PUBLISHING.md drops it from the "
                                       "release: it is its own project, catflops_chat)")
#: The server-file lines ONLY the chat reads: its loopback publish, its public origin and its slug at the issuer.
#: up.sh requires them only for the `chat` target (and the edge's, whose nginx proxies to the chat).
CHAT_VARS = {"EDGE_CHAT_PUBLISH", "EDGE_PUBLIC_CHAT_ORIGIN", "EDGE_CHAT_AUDIENCE"}
CHAT_COMPOSE = ("chat.override.yml",) if CHAT else ()
COMPOSE = EDGE_COMPOSE + ("node.yml", "lb.override.yml") + CHAT_COMPOSE + NODE_GPU + (
    "node-mfu.on.yml", "node-mfu.off.yml") + NODE_ENGINE
#: The compose projects up.sh needs a `.env` of before a full bring-up: the LB's, and the chat's where it is.
PROJECTS = ("lb",) + (("chat",) if CHAT else ())
#: A compose interpolation — never `$${…}`, compose's escape for a `${…}` the container's own shell expands.
VAR = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")
#: An address literal names a server — but 0.0.0.0, every interface of the container (SGLang's --host), names none.
IPV4 = re.compile(r"\b(?!0\.0\.0\.0\b)\d{1,3}(?:\.\d{1,3}){3}\b")

#: A complete, valid server file: what site-b runs, with TEST-NET addresses.
GOOD = {
    "EDGE_NODE_IMAGE": "vllm/vllm-openai@sha256:" + "0" * 64,
    "EDGE_NODE_ENGINE": "vllm",
    "EDGE_NODE_PROJECT": "vllm-standalone",
    "EDGE_NODE_NAME": "node-test",
    "EDGE_NODE_GPU": "1",
    "EDGE_NODE_GPU_RUNTIME": "nvidia",
    "EDGE_NODE_PUBLISH": "127.0.0.1:8020",
    "EDGE_NODE_NETWORK": "node-test-net",
    "EDGE_MODELS_DIR": "/srv/models",
    "EDGE_NODE_MODEL": "Some-Model-NVFP4",
    "EDGE_NODE_MAX_NUM_SEQS": "4",
    "EDGE_NODE_MFU_PATCH": "on",
    "EDGE_NODE_MEMORY": "24g",
    "EDGE_NODE_CPUS": "16",
    "EDGE_NODE_PIDS_LIMIT": "4096",
    "EDGE_NODE_NICE": "10",
    "EDGE_BOX": "testbox",
    "EDGE_LB_PUBLISH": "127.0.0.1:8010",
    "EDGE_LB_PROXY_HOSTS": "none",
    "EDGE_SEARXNG_CONTAINER": "vllm-lb-searxng",
    "EDGE_SEARXNG_PUBLISH": "127.0.0.1:8811",
    "EDGE_CHAT_PUBLISH": "127.0.0.1:8011",
    "EDGE_LOCAL_LISTEN": "127.0.0.1:8018",
    "EDGE_LOCAL_ORIGIN": "http://127.0.0.1:8018",
    "EDGE_GATE_LOCAL_BIND": "127.0.0.1:8017",
    "EDGE_GATE_PUBLIC_BIND": "127.0.0.1:8027",
    "EDGE_PUBLIC_LISTEN": "192.0.2.10:8018",
    "EDGE_PUBLIC_VLLM_ORIGIN": "https://vllm.example.test",
    "EDGE_AUTH_ISSUER": "https://auth.example.test",
    "EDGE_VLLM_AUDIENCE": "vllm-test",
    "EDGE_PUBLIC_PROXY": "192.0.2.1",
    "EDGE_GATE_API_KEYS": "1",
    "EDGE_GATE_KEY_DAYS": "7,30,90",
    "EDGE_GATE_KEY_DEFAULT_DAYS": "30",
    "EDGE_GATE_KEY_OWNER_DAYS": "30",
    "EDGE_GATE_KEYS_DIR": "/srv/vllm-gate-keys",
    "EDGE_PUBLIC_CHAT_LISTEN": "192.0.2.10:8019",
    "EDGE_PUBLIC_CHAT_ORIGIN": "https://chat.example.test",
    "EDGE_CHAT_AUDIENCE": "chat-test",
}
if not EDGE:
    GOOD = {k: v for k, v in GOOD.items() if k not in DOOR_VARS}
if not CHAT:
    GOOD = {k: v for k, v in GOOD.items() if k not in CHAT_VARS}


#: What only an LLM server's node reads (node-engine.vllm.yml and .sglang.yml, `${VAR:?}` there; the memory fraction
#: vLLM's alone): documented commented out, refused by compose, named, the moment that engine runs without them —
#: node-env renders them for its launch alone.
VLLM_ONLY = {"EDGE_NODE_SERVED_NAME": "some-model", "EDGE_NODE_MAX_MODEL_LEN": "131072", "EDGE_NODE_GPU_MEM_UTIL": "0.90"}

#: What a server file MAY leave out (.env.example names each commented out): the node's prefill chunk, absent = vLLM's
#: own default. The fleet's renderer names the same (lb/vllm_lb/fleet.py OPTIONAL_NODE_ENV).
OPTIONAL = {"EDGE_NODE_MAX_NUM_BATCHED_TOKENS", "EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_OFFLOAD_RAM_GIB",
            "EDGE_NODE_REASONING_PARSER", "EDGE_NODE_KV_CACHE_DTYPE", "EDGE_NODE_TOOL_CALL_PARSER",
            "EDGE_NODE_MAX_QUEUE", "EDGE_NODE_COMPUTE_TYPE"}
#: Host wiring of a node-only server file whose node runs on ANOTHER box's docker (up.sh, the applier): never rendered.
REMOTE_ONLY = {"EDGE_NODE_DOCKER_HOST"}
#: Host wiring only the ROCm override reads (`${VAR:?}` there): the host's NUMERIC video and render groups, which differ
#: per host (an image's /etc/group need not name `render`). Documented commented out — a server on an NVIDIA card has
#: none to give — and refused by compose, named, the moment the rocm override is used without them.
ROCM_ONLY = {"EDGE_NODE_VIDEO_GID", "EDGE_NODE_RENDER_GID"}
#: How the MFU byte-size patch is attached: one override per value node-env renders (`EDGE_NODE_MFU_PATCH`).
NODE_MFU = ("node-mfu.on.yml", "node-mfu.off.yml")

#: A NODE-ONLY server file: what a host that runs a node and nothing else of deploy/standalone gives (its container's
#: name, its loopback port, the network it shares with its LB, the checkpoints' folder, the host's ROCm groups).
NODE_HOST = {"EDGE_NODE_PROJECT": "node-test", "EDGE_NODE_NAME": "node-test", "EDGE_NODE_PUBLISH": "127.0.0.1:8020", "EDGE_NODE_NETWORK": "node-test-net",
             "EDGE_MODELS_DIR": "/srv/models", "EDGE_NODE_VIDEO_GID": "44", "EDGE_NODE_RENDER_GID": "992"}
#: The rest of the node section: what the applier renders (fleet node-env) into node.env.
APPLIED_REST = "".join(f"{k}={v}\n" for k, v in {
    "EDGE_NODE_IMAGE": "rocm/vllm@sha256:" + "0" * 64, "EDGE_NODE_ENGINE": "vllm", "EDGE_NODE_GPU": "0",
    "EDGE_NODE_GPU_RUNTIME": "rocm",
    "EDGE_NODE_MODEL": "Some-Model", "EDGE_NODE_SERVED_NAME": "some-model", "EDGE_NODE_MAX_MODEL_LEN": "32768",
    "EDGE_NODE_MAX_NUM_SEQS": "4", "EDGE_NODE_GPU_MEM_UTIL": "0.1", "EDGE_NODE_MFU_PATCH": "off",
    "EDGE_NODE_TOOL_CALL_PARSER": "hermes", "EDGE_NODE_MEMORY": "24g", "EDGE_NODE_CPUS": "16",
    "EDGE_NODE_PIDS_LIMIT": "4096", "EDGE_NODE_NICE": "10"}.items())


def _code(path: Path) -> list[str]:
    """Lines that are not comments (a literal in a comment documents; one in code configures)."""
    return [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]


def _vars(text: str) -> list[tuple[str, str]]:
    return VAR.findall(text)


def _check(env: dict[str, str], *targets: str) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
        fh.write("".join(f"{k}={v}\n" for k, v in env.items()))
    try:
        return subprocess.run(["bash", str(SA / "up.sh"), "--check", "--env", fh.name, *targets],
                              capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(fh.name)


class StandaloneLayout(unittest.TestCase):
    def test_the_server_directory_became_the_generic_one(self):
        self.assertFalse((ROOT / "deploy" / "site-b").exists(), "deploy/site-b must be moved, not copied")
        for name in COMPOSE + EDGE_TEMPLATE + ("up.sh", ".env.example"):
            self.assertTrue((SA / name).is_file(), name)

    def test_no_server_specific_literal_in_code(self):
        for name in COMPOSE + EDGE_TEMPLATE:
            for ln in _code(SA / name):
                self.assertIsNone(IPV4.search(ln), f"{name}: an address literal: {ln.strip()}")
                # A server is named by its domain, box or hostname: test_deploy's digest guard knows the words.
                self.assertEqual(_deployment_words(ln), set(), f"{name}: {ln.strip()}")


class StandaloneVariables(unittest.TestCase):
    def test_every_compose_interpolation_refuses_when_unset(self):
        for name in COMPOSE:
            text = "\n".join(_code(SA / name))
            for var, rest in _vars(text):
                if var == "HOME":
                    continue
                self.assertTrue(rest.startswith(":?"), f"{name}: ${{{var}{rest}}} must be :? (no default)")

    def test_every_variable_used_is_documented(self):
        example = (SA / ".env.example").read_text()
        documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", example, re.M))
        used = set()
        for name in COMPOSE + EDGE_TEMPLATE:
            used |= {v for v, _ in _vars("\n".join(_code(SA / name)))}
        used.discard("HOME")
        self.assertFalse(sorted(used - documented), f"used, not in .env.example: {sorted(used - documented)}")
        # Required: a line of its own (up.sh refuses the file without it). Optional: documented commented out.
        self.assertEqual(set(GOOD), set(re.findall(r"^(EDGE_[A-Z0-9_]+)=", example, re.M)),
                         "the test's GOOD file and .env.example must name the same required EDGE_ variables")
        self.assertEqual(OPTIONAL | ROCM_ONLY | REMOTE_ONLY | set(VLLM_ONLY), set(re.findall(r"^#(EDGE_[A-Z0-9_]+)=", example, re.M)),
                         "and the same optional ones")

    def test_the_door_lines_of_the_server_file_are_exactly_what_the_edge_alone_reads(self):
        """The release cuts .env.example at GATE_MARKER (PUBLISHING.md), so every line under the marker must be read by
        the edge's two files ONLY, and every line the rest of the fleet reads must sit above it — else the cut tree
        either asks for a value nothing reads or refuses to start for one that is gone."""
        example = (SA / ".env.example").read_text()
        if not EDGE:
            self.assertNotIn(GATE_MARKER, example, "the edge is gone but its marker (and the lines under it) stayed")
            self.assertFalse((DOOR_VARS | CHAT_VARS) & set(re.findall(r"^#?\s*(EDGE_[A-Z0-9_]+)=", example, re.M)))
            return
        self.assertEqual(example.count(GATE_MARKER), 1)
        above, below = example.split(GATE_MARKER, 1)
        below_names = set(re.findall(r"^#?\s*(EDGE_[A-Z0-9_]+)=", below, re.M))
        above_names = set(re.findall(r"^#?\s*(EDGE_[A-Z0-9_]+)=", above, re.M))
        # The rest of the fleet: everything the release ships — neither the edge's compose nor the chat's override.
        rest_reads = set()
        for name in COMPOSE:
            if name not in ("docker-compose.yml", *CHAT_COMPOSE):
                rest_reads |= {v for v, _ in _vars("\n".join(_code(SA / name)))}
        edge_reads = {v for v, _ in _vars("\n".join(_code(SA / "docker-compose.yml")))} | {
            v for v, _ in _vars((SA / TEMPLATE).read_text())}
        edge_reads.discard("HOME")
        chat_reads = {v for name in CHAT_COMPOSE for v, _ in _vars("\n".join(_code(SA / name)))}
        self.assertEqual(below_names, DOOR_VARS | (CHAT_VARS if CHAT else set()))
        self.assertEqual(below_names, (edge_reads | chat_reads) - rest_reads,
                         "under the marker: what the edge and the chat alone read")
        self.assertFalse(rest_reads - above_names, f"read by the rest of the fleet, documented under the marker: {rest_reads - above_names}")
        code = "\n".join(_code(SA / "up.sh"))
        self.assertIn('EDGE_FILE="$HERE/docker-compose.yml"', code)
        self.assertIn("HAS_EDGE", code)
        self.assertIn('CHAT_FILE="$HERE/chat.override.yml"', code)
        self.assertIn("HAS_CHAT", code)

    def test_the_guard_bites_a_door_line_moved_above_the_marker(self):
        """A planted violation: EDGE_GATE_KEYS_DIR documented above the marker would ship in the cut file, read by
        nothing — the set comparison names it."""
        if not EDGE:
            self.skipTest("the marker is cut from this tree")
        example = (SA / ".env.example").read_text()
        above, below = example.split(GATE_MARKER, 1)
        planted = above + "EDGE_GATE_KEYS_DIR=\n" + GATE_MARKER + below.replace("EDGE_GATE_KEYS_DIR=\n", "", 1)
        below_names = set(re.findall(r"^#?\s*(EDGE_[A-Z0-9_]+)=", planted.split(GATE_MARKER, 1)[1], re.M))
        private = DOOR_VARS | (CHAT_VARS if CHAT else set())
        self.assertNotEqual(below_names, private)
        self.assertEqual(private - below_names, {"EDGE_GATE_KEYS_DIR"})

    @needs_edge
    def test_the_template_substitutes_only_edge_variables_and_reaches_nginx(self):
        tvars = {v for v, _ in _vars((SA / TEMPLATE).read_text())}
        self.assertTrue(tvars and all(v.startswith("EDGE_") for v in tvars), sorted(tvars))
        compose = (SA / "docker-compose.yml").read_text()
        self.assertIn('NGINX_ENVSUBST_FILTER: "^EDGE_"', compose)
        self.assertIn(f"./{TEMPLATE}:/etc/nginx/templates/default.conf.template:ro", compose)
        nginx = compose.split("\n  nginx:", 1)[1]
        for v in sorted(tvars):
            self.assertIn(f"{v}: \"${{{v}:?", nginx, f"{v} is in the template but not passed to nginx")


class StandaloneDoors(unittest.TestCase):
    """What sits behind each door is in the template; up.sh refuses a server file that would put a
    door without sign-in, a gate or an internal service anywhere but loopback."""

    @needs_edge
    def test_the_template_has_the_two_doors_with_their_own_gates(self):
        t = (SA / TEMPLATE).read_text()
        local, public = t.split("listen ${EDGE_LOCAL_LISTEN};", 1)[1], t.split("listen ${EDGE_PUBLIC_LISTEN};", 1)[1]
        local = local.split("\nserver {", 1)[0]
        public = public.split("\nserver {", 1)[0]
        self.assertIn("http://${EDGE_GATE_LOCAL_BIND}/auth/check", local)
        self.assertNotIn("EDGE_GATE_PUBLIC_BIND", local)
        self.assertIn("http://${EDGE_GATE_PUBLIC_BIND}/auth/check", public)
        self.assertNotIn("EDGE_GATE_LOCAL_BIND", public)
        self.assertIn("absolute_redirect off;", public)
        self.assertNotIn("launcher", public, "a personal sign-in site's app launcher is the deployment's proxy to inject")

    def test_a_complete_loopback_file_passes(self):
        r = _check(GOOD)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_missing_variable_is_named(self):
        env = dict(GOOD)
        del env["EDGE_LB_PUBLISH"]
        r = _check(env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_LB_PUBLISH", r.stdout + r.stderr)

    def test_nothing_without_sign_in_leaves_loopback(self):
        for var in ("EDGE_LOCAL_LISTEN", "EDGE_GATE_LOCAL_BIND", "EDGE_GATE_PUBLIC_BIND", "EDGE_LB_PUBLISH",
                    "EDGE_CHAT_PUBLISH", "EDGE_NODE_PUBLISH", "EDGE_SEARXNG_PUBLISH"):
            if var not in GOOD:
                continue                                   # a door's line: not in a tree without the edge
            for bad in ("0.0.0.0:8018", "192.0.2.10:8018", "8018"):
                r = _check({**GOOD, var: bad})
                self.assertNotEqual(r.returncode, 0, f"{var}={bad} was accepted")
                self.assertIn(var, r.stdout + r.stderr)

    def test_the_public_origin_and_issuer_are_https(self):
        for var in ("EDGE_PUBLIC_VLLM_ORIGIN", "EDGE_AUTH_ISSUER", "EDGE_PUBLIC_CHAT_ORIGIN"):
            if var not in GOOD:
                continue                                   # the edge's or the chat's line: not in a tree without it
            r = _check({**GOOD, var: "http://vllm.example.test"})
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(var, r.stdout + r.stderr)

    def test_up_names_every_service_and_passes_the_server_file(self):
        code = "\n".join(_code(SA / "up.sh"))
        self.assertIn("set -euo pipefail", code)
        ups = [ln for ln in code.splitlines() if re.search(r"\bup -d\b", ln)]
        self.assertTrue(ups)
        for ln in ups:
            self.assertRegex(ln, r"up -d(?: --\S+)* [a-z]", f"a bare `up` starts every service: {ln.strip()}")
            self.assertIn("--env-file", ln, ln.strip())
        self.assertNotRegex(code, r"\bfrpc\b.*up|up.*\bfrpc\b")



class StandaloneBench(unittest.TestCase):
    """`up.sh bench` brings up the sealed swarm runner (lb/bench_runner) with every value taken from THIS
    server file — no new variable: the box, the node's network and the node's name are already in it —
    so the gateway panel's "Start test swarm" works on a standalone server. Run for real against a copy of
    the layout, with a `docker` on PATH that records what it was asked and changes nothing."""

    def _run(self, *targets: str) -> tuple[subprocess.CompletedProcess, list[dict]]:
        import json
        import shutil
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / "deploy" / "standalone").mkdir(parents=True)
        for name in ("up.sh", ".env.example", *EDGE_COMPOSE, *CHAT_COMPOSE, *NODE_GPU, *NODE_MFU, *NODE_ENGINE):
            shutil.copy(SA / name, tmp / "deploy" / "standalone" / name)   # the edge's and the chat's too, when the tree has them
        for project in (*PROJECTS, "lb/bench_runner"):
            (tmp / project).mkdir(parents=True, exist_ok=True)
        for project in PROJECTS:
            (tmp / project / ".env").write_text("")
        env_file = tmp / "server.env"
        env_file.write_text("".join(f"{k}={v}\n" for k, v in GOOD.items()))
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        calls = tmp / "docker.calls"
        fake = bin_dir / "docker"
        fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'cwd': os.getcwd(), "
                        "'env': {k: v for k, v in os.environ.items() if k.startswith('BENCH_')}}) + '\\n')\n")
        fake.chmod(0o755)
        r = subprocess.run(["bash", str(tmp / "deploy" / "standalone" / "up.sh"), "--env", str(env_file), *targets],
                           env={"PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp)},
                           capture_output=True, text=True, timeout=60)
        recorded = [json.loads(ln) for ln in calls.read_text().splitlines()] if calls.exists() else []
        return r, [c for c in recorded if "compose" in c["argv"]], tmp

    def test_bench_is_a_target_and_part_of_a_full_bring_up(self):
        code = "\n".join(_code(SA / "up.sh"))
        self.assertIn("node|lb|bench|chat|edge)", code)
        self.assertIn("TARGETS=(node lb bench); [ \"$HAS_CHAT\" = 1 ] && TARGETS+=(chat); [ \"$HAS_EDGE\" = 1 ] && TARGETS+=(edge)", code)

    def test_the_runner_takes_every_value_from_this_server_file(self):
        r, composes, tmp = self._run("bench")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(composes), 1, composes)
        call = composes[0]
        self.assertEqual(call["cwd"], str(tmp / "lb" / "bench_runner"))
        self.assertEqual(call["env"], {
            "BENCH_BOX": GOOD["EDGE_BOX"],
            "BENCH_STATE_DIR": str(tmp / "lb" / "state" / "benchq"),
            "BENCH_UPSTREAM_NETWORK": GOOD["EDGE_NODE_NETWORK"],
            "BENCH_LB_TARGET": "vllm-lb:8010",
            "BENCH_ENGINE_TARGET": f"{GOOD['EDGE_NODE_NAME']}:8000"})
        argv = call["argv"]
        self.assertEqual(argv[argv.index("up"):], ["up", "-d", "--build", "egress", "runner"])
        self.assertIn("--env-file", argv)

    def test_a_standalone_lb_never_takes_the_bench_only_network(self):
        """lb/docker-compose.bench-net.yml is for an LB whose only network is another project's (the broker's).
        Here the LB, the node and the runner's egress already share EDGE_NODE_NETWORK, so the LB's compose
        files stay the base and this server's override — and nothing names LB_BENCH_NETWORK."""
        code = "\n".join(_code(SA / "up.sh"))
        self.assertNotIn("bench-net", code)
        self.assertNotIn("LB_BENCH_NETWORK", code)
        lb_line = next(ln for ln in code.splitlines() if "lb.override.yml" in ln)
        self.assertEqual(re.findall(r"-f (\S+)", lb_line), ["docker-compose.yml", '"$HERE/lb.override.yml"'])
        for name in COMPOSE:
            self.assertNotIn("LB_BENCH_NETWORK", (SA / name).read_text())

    def test_the_sealed_runner_needs_no_new_server_variable(self):
        """It is brought up from what a server file already says, so a server that has never heard of it
        passes --check exactly as before."""
        r = _check(GOOD)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        documented = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", (SA / ".env.example").read_text(), re.M))
        self.assertFalse({v for v in documented if v.startswith("BENCH_")})


class StandaloneChat(unittest.TestCase):
    """The chat's public door is the chat itself: it signs people in (the issuer's grant on this
    server's own slug), so nginx asks no gate for it -- which is exactly why it may only ever run
    with CHAT_AUTH=required behind that door."""

    def _chat_block(self) -> str:
        t = (SA / TEMPLATE).read_text()
        self.assertIn("listen ${EDGE_PUBLIC_CHAT_LISTEN};", t)
        return t.split("listen ${EDGE_PUBLIC_CHAT_LISTEN};", 1)[1].split("\nserver {", 1)[0]

    @needs_edge
    def test_the_public_chat_door_proxies_to_the_chat_with_no_gate(self):
        block = self._chat_block()
        self.assertIn("proxy_pass http://${EDGE_CHAT_PUBLISH};", block)
        self.assertNotIn("auth_request", block)
        for directive in ("proxy_buffering off;", "client_max_body_size 4m;", "absolute_redirect off;",
                          "proxy_read_timeout 3600s;", "Strict-Transport-Security"):
            self.assertIn(directive, block)

    @needs_chat
    def test_the_chat_signs_in_on_this_servers_slug(self):
        code = "\n".join(_code(SA / "chat.override.yml"))
        self.assertIn('CHAT_AUTH: "required"', code)
        self.assertNotIn('"off"', code)
        for key, var in (("CHAT_PUBLIC_URL", "EDGE_PUBLIC_CHAT_ORIGIN"), ("CHAT_AUTH_ISSUER", "EDGE_AUTH_ISSUER"),
                         ("CHAT_AUTH_AUDIENCE", "EDGE_CHAT_AUDIENCE")):
            self.assertIn(f'{key}: "${{{var}:?', code)

    @needs_chat
    def test_the_load_meter_links_to_this_servers_public_gateway_door(self):
        """The chat's load meter is read from THIS server's LB, so it opens this server's gateway panel at
        the public door the jwt gate serves (EDGE_PUBLIC_VLLM_ORIGIN, https by up.sh) -- no new variable,
        and never the loopback door, which only a tunnel on the server reaches."""
        code = "\n".join(_code(SA / "chat.override.yml"))
        self.assertIn('CHAT_GATEWAY_URL: "${EDGE_PUBLIC_VLLM_ORIGIN:?', code)
        self.assertNotIn("EDGE_LOCAL_ORIGIN", code)



class StandaloneFleet(unittest.TestCase):
    """The LB reads what a server IS from its fleet file (lb/fleet.yml, LB_FLEET) and REFUSES to boot on
    the env settings that used to say it (vllm_lb/config.py RETIRED_KEYS): the override must set none of
    them, and up.sh must prove the new LB boots — fleet file present, config loadable by the NEW image
    with the service's own env and mounts — before it replaces the running one. A refused boot is an
    outage of the server's only gateway; the pre-flight turns it into a failed deploy instead."""

    RETIRED = re.findall(r'"(LB_[A-Z0-9_]+)"\s*:', (ROOT / "lb/vllm_lb/config.py").read_text()
                         .split("\nRETIRED_KEYS = {", 1)[1].split("\n}", 1)[0])

    def test_the_lb_override_sets_no_retired_setting(self):
        self.assertIn("LB_VLLM_BOXES", self.RETIRED)      # the list is read, not empty
        code = "\n".join(_code(SA / "lb.override.yml"))
        for key in self.RETIRED:
            self.assertNotIn(key, code, f"{key} is retired: the LB would refuse to boot")

    def test_the_lb_finds_its_catalog_folders_in_the_server_file(self):
        """The model catalog refuses to load without MODELS_SEARCH_DIRS (vllm_lb/catalog.py), and a fleet's
        one-host cluster names a catalog model, so without it the LB cannot boot. The folder is already in
        the server file (EDGE_MODELS_DIR, the node's own mount): derive it, never write it twice."""
        code = "\n".join(_code(SA / "lb.override.yml"))
        self.assertIn("MODELS_SEARCH_DIRS=${EDGE_MODELS_DIR:?", code)

    def test_the_connect_dialogs_tunnel_is_the_lbs_own_published_address(self):
        """The panel's Connect dialog offers an SSH forward to the LB (LB_CONNECT_TUNNEL, vllm_lb/connect.py): the
        address the LB is published at on this server, which the server file already says (EDGE_LB_PUBLISH, the
        `ports:` line above it). Derived, never written twice."""
        code = "\n".join(_code(SA / "lb.override.yml"))
        self.assertIn("LB_CONNECT_TUNNEL=${EDGE_LB_PUBLISH:?", code)
        self.assertIn('"${EDGE_LB_PUBLISH:?', code)

    def test_the_lb_step_proves_the_new_lb_boots_before_it_replaces_the_running_one(self):
        up = "\n".join(_code(SA / "up.sh"))
        lb = up.split("        lb)", 1)[1].split(";;", 1)[0]
        self.assertIn("fleet.yml", up)
        pre = lb.find("Config.from_env()")
        self.assertGreater(pre, -1, "the lb step must load the config with the new image first")
        self.assertIn("run --rm --no-deps", lb[:pre])
        self.assertGreater(lb.find("up -d"), pre, "`up -d` must come after the pre-flight")

    def test_up_refuses_the_lb_step_without_a_fleet_file_before_touching_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "deploy/standalone").mkdir(parents=True)
            for name in ("up.sh", ".env.example"):
                (t / "deploy/standalone" / name).write_text((SA / name).read_text())
            for project in PROJECTS:
                (t / project).mkdir()
                (t / project / ".env").write_text("X=1\n")
            env = t / "server.env"
            env.write_text("".join(f"{k}={v}\n" for k, v in GOOD.items()))
            r = subprocess.run(["bash", str(t / "deploy/standalone/up.sh"), "--env", str(env), "lb"],
                               capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fleet.yml", r.stdout + r.stderr)
        self.assertNotIn("docker: command not found", r.stdout + r.stderr)



class StandaloneHosts(unittest.TestCase):
    """THE NAMES THE LB ANSWERS TO (lb/vllm_lb/host_guard.py, the DNS-rebinding guard; LB_ALLOWED_HOSTS, required):
    on a standalone server every one of them is already in the server file, so up.sh DERIVES the value — never a
    second copy in lb/.env — from: the address the LB is published at (the gates' health probe and host-side clients
    dial it), `vllm-lb:8010` (the chat on the LB's network), `lb:8010` (the sealed runner, through its relay), the
    names nginx's two doors forward (`proxy_set_header Host $host`: the host of each door's origin, no port) where the
    edge is, and EDGE_LB_PROXY_HOSTS — what YOUR proxy presents in front of EDGE_LB_PUBLISH, or `none`."""

    def _run(self, server: dict[str, str], lb_env: str = "", *targets: str):
        import json
        import shutil
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / "deploy" / "standalone").mkdir(parents=True)
        for name in ("up.sh", ".env.example", *EDGE_COMPOSE, *CHAT_COMPOSE):
            shutil.copy(SA / name, tmp / "deploy" / "standalone" / name)
        for project in PROJECTS:
            (tmp / project).mkdir(parents=True, exist_ok=True)
            (tmp / project / ".env").write_text("")
        (tmp / "lb" / ".env").write_text(lb_env)
        (tmp / "lb" / "fleet.yml").write_text("boxes: {}\n")
        env_file = tmp / "server.env"
        env_file.write_text("".join(f"{k}={v}\n" for k, v in server.items()))
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        calls = tmp / "docker.calls"
        fake = bin_dir / "docker"
        fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], "
                        "'hosts': os.environ.get('LB_ALLOWED_HOSTS')}) + '\\n')\n")
        fake.chmod(0o755)
        r = subprocess.run(["bash", str(tmp / "deploy" / "standalone" / "up.sh"), "--env", str(env_file),
                            *(targets or ("lb",))],
                           env={"PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp)},
                           capture_output=True, text=True, timeout=60)
        recorded = [json.loads(ln) for ln in calls.read_text().splitlines()] if calls.exists() else []
        return r, [c for c in recorded if "compose" in c["argv"]]

    def _want(self, server: dict[str, str]) -> str:
        names = [server["EDGE_LB_PUBLISH"], "vllm-lb:8010", "lb:8010"]
        if EDGE:
            names += ["127.0.0.1", "vllm.example.test"]          # the doors' origins' hosts, as nginx's $host
        if server["EDGE_LB_PROXY_HOSTS"] != "none":
            names += server["EDGE_LB_PROXY_HOSTS"].split(",")
        return ",".join(names)

    def test_every_lb_compose_call_carries_the_derived_names(self):
        r, composes = self._run(GOOD)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lb_calls = [c for c in composes if "vllm-lb" in c["argv"]]
        self.assertEqual(len(lb_calls), 3, composes)                 # build, the pre-flight, up
        self.assertEqual({c["hosts"] for c in lb_calls}, {self._want(GOOD)})

    def test_the_operators_own_proxy_names_are_added(self):
        server = dict(GOOD, EDGE_LB_PROXY_HOSTS="llm.example.test,llm.example.test:8443")
        r, composes = self._run(server)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual({c["hosts"] for c in composes if "vllm-lb" in c["argv"]}, {self._want(server)})

    def test_a_second_copy_in_lb_env_is_refused_before_docker_is_touched(self):
        r, composes = self._run(GOOD, "LB_ALLOWED_HOSTS=loopback\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("LB_ALLOWED_HOSTS", r.stdout + r.stderr)
        self.assertEqual(composes, [])

    def test_a_proxy_name_that_is_not_host_and_port_is_refused_at_check(self):
        for bad in ("https://llm.example.test", "llm.example.test/x", "a b", "none,llm.example.test", "a,,b"):
            r = _check(dict(GOOD, EDGE_LB_PROXY_HOSTS=bad))
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertIn("EDGE_LB_PROXY_HOSTS", r.stdout + r.stderr, bad)

    def test_check_prints_the_names_the_lb_will_answer_to(self):
        r = _check(GOOD)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(f"LB_ALLOWED_HOSTS={self._want(GOOD)}", r.stdout)


class StandaloneNodeMetrics(unittest.TestCase):
    """The node measures its own compute and memory traffic, as the pair's engines do (lb/cluster/tp-rank):
    vLLM registers its vllm:estimated_*_per_gpu_total counters unconditionally but leaves them 0.0 unless
    started with --enable-mfu-metrics, and its MFU model DROPS a component whose quant method it has no
    byte size for — 'modelopt_mixed', what these NVFP4 checkpoints report — so tp-rank injects the measured
    0.5625 B/weight through a sitecustomize. This node gets the SAME patch, byte for byte (pinned here
    against tp-rank's heredoc, so the two can never drift)."""

    def _node(self, name: str = "node.yml") -> str:
        return (SA / name).read_text().split("\n  node:", 1)[1].split("\n  gate:", 1)[0]

    def test_the_node_is_started_with_mfu_metrics(self):
        self.assertIn("--enable-mfu-metrics", self._node("node-engine.vllm.yml"))

    def test_the_node_loads_tp_ranks_mfu_patch_byte_for_byte(self):
        tp = (ROOT / "lb/cluster/tp-rank").read_text()
        heredoc = tp.split("""cat > "$patch_dir/sitecustomize.py" <<'PATCH' || true\n""", 1)[1].split("\nPATCH\n", 1)[0]
        ours = (SA / "vllm-mfu" / "sitecustomize.py").read_text()
        self.assertEqual(ours.rstrip("\n"), heredoc.rstrip("\n"))
        # MOUNTED AND SET ONLY WHERE THE CHECKPOINT NEEDS IT (node-env's EDGE_NODE_MFU_PATCH, from the catalog row's
        # quant_method): on ROCm its sitecustomize recursed through the platform probe — 2,380 offload-arch processes,
        # OOM-killed (2026-10-17) — and a PYTHONPATH of ours drops the image's own. The base node sets neither.
        on, off = self._node("node-mfu.on.yml"), self._node("node-mfu.off.yml")
        self.assertIn("./vllm-mfu:/opt/vllm-mfu:ro", on)
        self.assertIn("PYTHONPATH: /opt/vllm-mfu", on)
        base = "\n".join(ln for ln in self._node().splitlines() if not ln.lstrip().startswith("#"))
        for word in ("vllm-mfu", "PYTHONPATH"):
            self.assertNotIn(word, base)
            self.assertNotIn(word, "\n".join(ln for ln in off.splitlines() if not ln.lstrip().startswith("#")))


def _node_argv(tc: unittest.TestCase, name: str, program: str, extra: dict[str, str]) -> list[str]:
    """Run engine override `name`'s node for real — its entrypoint and command, a `program` on PATH that prints what it
    was given — with the container's env as compose hands it (the niceness, and `extra`)."""
    import json
    node = (SA / name).read_text().split("\n  node:", 1)[1]
    entry = re.search(r"\n    entrypoint:\n((?:      - .*\n)+)", node)
    tc.assertIsNotNone(entry, "the node declares its entrypoint")
    items = [ln.strip()[2:] for ln in entry.group(1).splitlines()]
    items = [i[1:-1] if i[:1] in "'\"" and i[-1:] == i[:1] else i for i in items]
    # compose's `$$` is a literal `$` in the container; the command's ${VAR:?} are the server file's values
    items = [i.replace("$$", "$") for i in items]
    command = re.search(r"\n    command:\n((?:      (?:- .*|#.*)\n)+)", node).group(1)
    args = [a.strip()[2:].strip('"') for a in command.splitlines() if a.strip().startswith("- ")]
    args = [VAR.sub(lambda m: {**GOOD, **VLLM_ONLY}.get(m.group(1), m.group(1)), a) for a in args]
    tmp = Path(tc.enterContext(tempfile.TemporaryDirectory()))
    fake = tmp / program
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    env = {"PATH": f"{tmp}:/usr/bin:/bin", "EDGE_NODE_NICE": GOOD["EDGE_NODE_NICE"], **extra}
    r = subprocess.run([*items, *args], env=env, capture_output=True, text=True, timeout=30)
    tc.assertEqual(r.returncode, 0, r.stderr)
    return json.loads(r.stdout)


class StandaloneSglangNode(unittest.TestCase):
    """node-engine.sglang.yml: SGLang's launch_server with the node's figures; each optional flag only when its variable
    holds a value; no memory fraction (SGLang's own rule for the card — the fleet's fit is its measured memory)."""

    OPTIONAL = {"EDGE_NODE_TOOL_CALL_PARSER": ("--tool-call-parser", "qwen3_coder"),
                "EDGE_NODE_REASONING_PARSER": ("--reasoning-parser", "qwen3"),
                "EDGE_NODE_KV_CACHE_DTYPE": ("--kv-cache-dtype", "fp8_e4m3")}

    def test_the_node_runs_launch_server_with_its_figures(self):
        argv = _node_argv(self, "node-engine.sglang.yml", "python3", {})
        self.assertEqual(argv[:2], ["-m", "sglang.launch_server"])
        for flag, value in (("--model-path", f"/models/{GOOD['EDGE_NODE_MODEL']}"),
                            ("--served-model-name", VLLM_ONLY["EDGE_NODE_SERVED_NAME"]),
                            ("--context-length", VLLM_ONLY["EDGE_NODE_MAX_MODEL_LEN"]),
                            ("--max-running-requests", GOOD["EDGE_NODE_MAX_NUM_SEQS"]),
                            ("--host", "0.0.0.0"), ("--port", "8000")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--enable-metrics", argv)
        self.assertNotIn("--mem-fraction-static", argv, "SGLang sizes its own")

    def test_the_node_schedules_by_the_requests_priority(self):
        """lb/vllm_lb/router.py PRIORITY_HEADER: lower = sooner, 0 for a request naming none — SGLang's direction set
        to vLLM's. Without the flags the engine ignores the field silently, so every start asks."""
        argv = _node_argv(self, "node-engine.sglang.yml", "python3", {})
        self.assertIn("--enable-priority-scheduling", argv)
        self.assertIn("--schedule-low-priority-values-first", argv)
        self.assertEqual(argv[argv.index("--default-priority-value") + 1], "0")

    def test_each_optional_flag_only_when_set(self):
        for unset in ({}, {k: "" for k in self.OPTIONAL}):
            argv = _node_argv(self, "node-engine.sglang.yml", "python3", unset)
            for flag, _ in self.OPTIONAL.values():
                self.assertNotIn(flag, argv)
            self.assertNotIn("", argv)
        argv = _node_argv(self, "node-engine.sglang.yml", "python3", {k: v for k, (_, v) in self.OPTIONAL.items()})
        for flag, value in self.OPTIONAL.values():
            self.assertEqual((argv.count(flag), argv[argv.index(flag) + 1]), (1, value), flag)

    def test_the_optional_values_are_handed_through_never_defaulted(self):
        node = (SA / "node-engine.sglang.yml").read_text().split("\n  node:", 1)[1]
        for var in self.OPTIONAL:
            self.assertRegex(node, rf"\n      {var}:\s*\n", var)
            self.assertNotIn(f"${{{var}", node.replace(f"$${{{var}", ""), var)


class StandaloneNodePrefillChunk(unittest.TestCase):
    """vLLM's prefill chunk (--max-num-batched-tokens) is the one OPTIONAL node value: unset or empty is vLLM's own
    default, so the flag reaches vLLM only when the variable holds a number — never "" (an argument vLLM refuses) and
    never a number of ours standing in for vLLM's. A compose list cannot drop an item, so the node's entrypoint is the
    image's own `vllm serve` behind a shell that adds the flag when it is set. Run for real: the entrypoint the compose
    file declares, its command, and a `vllm` on PATH that prints what it was given."""

    VAR = "EDGE_NODE_MAX_NUM_BATCHED_TOKENS"

    def _node(self) -> str:
        return (SA / "node-engine.vllm.yml").read_text().split("\n  node:", 1)[1]

    def _argv(self, value: str | None, **others: str) -> list[str]:
        return _node_argv(self, "node-engine.vllm.yml", "vllm", {**({} if value is None else {self.VAR: value}),
                                                                 **others})

    def test_the_flag_reaches_vllm_only_when_the_chunk_is_set(self):
        for unset in (None, ""):
            argv = self._argv(unset)
            self.assertEqual(argv[0], "serve", "the image's own entrypoint: vllm serve")
            self.assertNotIn("--max-num-batched-tokens", argv, f"{self.VAR}={unset!r}: vLLM's own default")
            self.assertNotIn("", argv, "no empty argument")
            self.assertIn("--enable-mfu-metrics", argv)
        argv = self._argv("1024")
        self.assertEqual(argv[argv.index("--max-num-batched-tokens") + 1], "1024")
        self.assertEqual(argv.count("--max-num-batched-tokens"), 1)
        self.assertEqual(argv[1], f"/models/{GOOD['EDGE_NODE_MODEL']}", "the model is still the first argument")

    def test_the_node_schedules_by_the_requests_priority(self):
        """vLLM reads a request's `priority` only under --scheduling-policy priority (fcfs ignores it silently)."""
        argv = self._argv(None)
        self.assertEqual(argv[argv.index("--scheduling-policy") + 1], "priority")

    def test_the_chunk_is_handed_through_as_it_is_never_defaulted(self):
        node = self._node()
        self.assertRegex(node, rf"\n      {self.VAR}:\s*\n", "passed through as the server file / node.env has it")
        self.assertNotIn(f"${{{self.VAR}", node.replace(f"$${{{self.VAR}", ""),
                         "never a compose interpolation: no default, no :? (it is optional)")

    def test_a_server_file_may_leave_it_out_or_set_it(self):
        self.assertEqual(_check(GOOD).returncode, 0)
        r = _check({**GOOD, self.VAR: "1024"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_the_fleet_renders_exactly_the_optional_names(self):
        fleet = (ROOT / "lb/vllm_lb/fleet.py").read_text()
        m = re.search(r"^OPTIONAL_NODE_ENV = \(([^)]*)\)", fleet, re.M)
        self.assertIsNotNone(m, "fleet.py names the node-env lines that may be empty")
        self.assertEqual(set(re.findall(r'"(EDGE_[A-Z0-9_]+)"', m.group(1))), OPTIONAL)


    # The KV pool PINNED (vLLM --kv-cache-memory-bytes, engine/arg_utils.py:1352 @2eb491e) and a KV tier in the HOST's
    # RAM (--kv-offloading-size N GiB with --kv-offloading-backend native, :1405-1408): optional the same way — absent
    # or empty is vLLM's own (the pool from the memory fraction; no tier), each flag passed once and only when set.
    PIN, RAM = "EDGE_NODE_KV_CACHE_MEMORY", "EDGE_NODE_KV_OFFLOAD_RAM_GIB"

    def test_the_pinned_pool_and_the_ram_tier_reach_vllm_only_when_set(self):
        for unset in (None, ""):
            extra = {} if unset is None else {self.PIN: unset, self.RAM: unset}
            argv = self._argv(None, **extra)
            for flag in ("--kv-cache-memory-bytes", "--kv-offloading-size", "--kv-offloading-backend"):
                self.assertNotIn(flag, argv, f"{flag}: vLLM's own when unset ({unset!r})")
            self.assertNotIn("", argv)
        argv = self._argv(None, **{self.PIN: "8501810176", self.RAM: "32"})
        self.assertEqual(argv[argv.index("--kv-cache-memory-bytes") + 1], "8501810176")
        self.assertEqual(argv[argv.index("--kv-offloading-size") + 1], "32")
        self.assertEqual(argv[argv.index("--kv-offloading-backend") + 1], "native",
                         "the backend stated, never left to vLLM's default (an LMCache server would be another one)")
        for flag in ("--kv-cache-memory-bytes", "--kv-offloading-size", "--kv-offloading-backend"):
            self.assertEqual(argv.count(flag), 1, flag)
        argv = self._argv("1024", **{self.RAM: "16"})
        self.assertNotIn("--kv-cache-memory-bytes", argv, "each flag on its own variable")
        self.assertEqual(argv[argv.index("--max-num-batched-tokens") + 1], "1024")
        self.assertEqual(argv[1], f"/models/{GOOD['EDGE_NODE_MODEL']}")

    def test_the_pin_and_the_tier_are_handed_through_never_defaulted(self):
        node = self._node()
        for var in (self.PIN, self.RAM):
            self.assertRegex(node, rf"\n      {var}:\s*\n", var)
            self.assertNotIn(f"${{{var}", node.replace(f"$${{{var}", ""), var)

    # The KV cache's element type (vLLM --kv-cache-dtype; nvfp4 where the image accepts it, fleet.py KV_DTYPE): optional
    # the same way — absent or empty is vLLM's own, so a launch without it is today's argv, byte for byte.
    DTYPE = "EDGE_NODE_KV_CACHE_DTYPE"

    def test_the_kv_cache_dtype_reaches_vllm_only_when_set_and_unset_is_todays_argv(self):
        today = self._argv(None)
        self.assertNotIn("--kv-cache-dtype", today)
        self.assertEqual(self._argv(None, **{self.DTYPE: ""}), today, "empty: the same argv as unset")
        argv = self._argv(None, **{self.DTYPE: "nvfp4"})
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "nvfp4")
        self.assertEqual(argv.count("--kv-cache-dtype"), 1)
        self.assertEqual([a for a in argv if a not in ("--kv-cache-dtype", "nvfp4")], today, "nothing else moves")

    def test_the_kv_cache_dtype_is_handed_through_never_defaulted(self):
        node = self._node()
        self.assertRegex(node, rf"\n      {self.DTYPE}:\s*\n")
        self.assertNotIn(f"${{{self.DTYPE}", node.replace(f"$${{{self.DTYPE}", ""))

    def test_the_ram_tiers_dev_shm_is_the_hosts(self):
        """vLLM maps the tier in /dev/shm (v1/kv_offload/cpu/shared_offload_region.py:110) and refuses one larger than
        its free space (shm_broadcast.py:249-258): the node shares the host's IPC namespace, so it is the host's
        /dev/shm — the size the fleet's kv_offload_ram_max_gib is set under — not docker's 64 MB default."""
        self.assertIn("\n    ipc: host\n", (SA / "node.yml").read_text())

class StandaloneNodeApplier(unittest.TestCase):
    """The node the LB requests (fleet control `local`) is applied by its host: a `systemd --user` applier
    (vllm_lb/node_applier.py) renders the LB's intent into lb/state/node/node.env and runs `up.sh node` — which reads
    that file as a SECOND --env-file after the server file (later wins), so a hand-run `up.sh node` brings up what the
    panel last asked for instead of reverting it — or `up.sh node-stop`. Run for real against a copy of the layout, with
    a `docker` on PATH that records what it was asked."""

    def _run(self, *targets: str, fragment: str | None = None, server: dict | None = None,
             args: tuple = ()) -> tuple[subprocess.CompletedProcess, list[list[str]], Path]:
        import json
        import shutil
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / "deploy" / "standalone").mkdir(parents=True)
        for name in ("up.sh", ".env.example", *EDGE_COMPOSE, *CHAT_COMPOSE, *NODE_GPU, *NODE_MFU, *NODE_ENGINE):
            shutil.copy(SA / name, tmp / "deploy" / "standalone" / name)   # the edge's and the chat's too, when the tree has them
        for project in PROJECTS:
            (tmp / project).mkdir(parents=True, exist_ok=True)
            (tmp / project / ".env").write_text("")
        if fragment is not None:
            (tmp / "lb" / "state" / "node").mkdir(parents=True)
            (tmp / "lb" / "state" / "node" / "node.env").write_text(fragment)
        env_file = tmp / "server.env"
        env_file.write_text("".join(f"{k}={v}\n" for k, v in (GOOD if server is None else server).items()))
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        calls = tmp / "docker.calls"
        fake = bin_dir / "docker"
        fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                        f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                        f"open({str(tmp / 'docker.hosts')!r}, 'a').write(os.environ.get('DOCKER_HOST', '-') + '\\n')\n")
        fake.chmod(0o755)
        r = subprocess.run(["bash", str(tmp / "deploy" / "standalone" / "up.sh"), "--env", str(env_file), *args,
                            *targets],
                           env={"PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp)},
                           capture_output=True, text=True, timeout=60)
        recorded = [json.loads(ln) for ln in calls.read_text().splitlines()] if calls.exists() else []
        return r, [c for c in recorded if "compose" in c], tmp

    @staticmethod
    def _files(argv: list[str]) -> list[str]:
        return [Path(argv[i + 1]).name for i, a in enumerate(argv) if a == "-f"]

    # ── how the container attaches the card: the hardware type's `gpu_runtime`, as data ─────────────────────────

    def test_the_node_attaches_its_card_with_the_override_its_gpu_runtime_names(self):
        r, composes, _ = self._run("node")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.nvidia.yml", "node-mfu.on.yml", "node-engine.vllm.yml"])
        # the applied node.env names the runtime too (node-env renders it from the card's type): later wins
        r, composes, _ = self._run("node", fragment="EDGE_NODE_GPU_RUNTIME=rocm\nEDGE_NODE_MFU_PATCH=off\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.rocm.yml", "node-mfu.off.yml", "node-engine.vllm.yml"])
        r, composes, _ = self._run("node-stop", fragment="EDGE_NODE_GPU_RUNTIME=rocm\n")
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.rocm.yml", "node-mfu.on.yml", "node-engine.vllm.yml"])

    def test_an_mfu_patch_value_with_no_override_is_refused_before_docker_is_touched(self):
        r, composes, _ = self._run("node", server={**GOOD, "EDGE_NODE_MFU_PATCH": "maybe"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_MFU_PATCH=maybe", r.stderr)
        self.assertEqual(composes, [])

    def test_a_gpu_runtime_with_no_override_is_refused_before_docker_is_touched(self):
        r, composes, _ = self._run("node", server={**GOOD, "EDGE_NODE_GPU_RUNTIME": "cuda"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_GPU_RUNTIME=cuda", r.stderr)
        self.assertIn("node-gpu.nvidia.yml", r.stderr + "", "names the runtimes there are")
        self.assertEqual(composes, [])

    def test_the_applier_names_its_clusters_node_env_and_up_reads_that_one(self):
        """With several clusters each node dir is lb/state/clusters/<name>/node (node_control.state_path_for): the
        applier hands up.sh its cluster's node.env (`--node-env`), which is then read INSTEAD of the one-cluster path."""
        r, composes, tmp = self._run("node", fragment="EDGE_NODE_MAX_NUM_SEQS=2\n")
        mine = tmp / "lb" / "state" / "clusters" / "gui" / "node" / "node.env"
        mine.parent.mkdir(parents=True)
        mine.write_text("EDGE_NODE_GPU_RUNTIME=rocm\n")
        r, composes, tmp2 = self._run("node", args=("--node-env", str(mine)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._env_files(composes[0])[-1], str(mine))
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.rocm.yml", "node-mfu.on.yml", "node-engine.vllm.yml"])
        r, composes, _ = self._run("node", args=("--node-env", str(tmp / "nope.env")))
        self.assertNotEqual(r.returncode, 0, "a named node.env that is not there is refused, not skipped")
        self.assertIn("nope.env", r.stderr)

    @staticmethod
    def _env_files(argv: list[str]) -> list[str]:
        return [argv[i + 1] for i, a in enumerate(argv) if a == "--env-file"]

    def test_node_reads_the_applied_intent_after_the_server_file(self):
        r, composes, tmp = self._run("node", fragment="EDGE_NODE_MAX_NUM_SEQS=2\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(composes), 1, composes)
        argv = composes[0]
        self.assertEqual(self._env_files(argv), [str(tmp / "server.env"), str(tmp / "lb/state/node/node.env")])
        self.assertEqual(argv[argv.index("up"):], ["up", "-d", "node"])

    def test_without_an_applied_intent_node_is_exactly_the_server_file(self):
        r, composes, tmp = self._run("node")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._env_files(composes[0]), [str(tmp / "server.env")])

    def test_node_stop_stops_the_node_and_is_never_part_of_a_full_bring_up(self):
        r, composes, tmp = self._run("node-stop", fragment="EDGE_NODE_MAX_NUM_SEQS=2\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        argv = composes[0]
        self.assertEqual(argv[argv.index("stop"):], ["stop", "node"])
        self.assertEqual(self._env_files(argv), [str(tmp / "server.env"), str(tmp / "lb/state/node/node.env")])
        code = "\n".join(_code(SA / "up.sh"))
        self.assertIn("TARGETS=(node lb bench); [ \"$HAS_CHAT\" = 1 ] && TARGETS+=(chat); [ \"$HAS_EDGE\" = 1 ] && TARGETS+=(edge)", code)

    # ── a NODE-ONLY server file: a host that runs a node and nothing else of deploy/standalone ─────────────────

    def test_a_node_only_server_file_brings_up_its_node_with_the_applied_intent(self):
        """A box whose LB is not deploy/standalone's (box-c: its own lb/, its pair) runs only `node`/`node-stop`: its
        server file is the node's half (NODE_HOST) and the applied node.env the rest (later wins). Required is then the
        node section, read AFTER the applied file — the gate's, chat's and doors' lines are not that host's to give."""
        r, composes, _ = self._run("node", server=NODE_HOST, fragment=APPLIED_REST)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(composes), 1, composes)
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.rocm.yml", "node-mfu.off.yml", "node-engine.vllm.yml"])
        r, composes, _ = self._run("node-stop", server=NODE_HOST, fragment=APPLIED_REST)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_node_only_bring_up_hands_compose_only_files_the_node_section_can_fill(self):
        """compose interpolates EVERY service of the files it is given, the targeted one or not: a node-only server file
        handed docker-compose.yml with the gates and nginx in it stops on EDGE_LOCAL_ORIGIN before the node is touched
        (seen on the box-c kit, 2026-10-17). The node's files name nothing but the node section's values."""
        r, composes, _ = self._run("node", server=NODE_HOST, fragment=APPLIED_REST)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        used = {v for f in self._files(composes[0]) for v, _ in _vars((SA / f).read_text())}
        example = (SA / ".env.example").read_text()
        section = set(re.findall(r"(?m)^#?(EDGE_(?:NODE_[A-Z0-9_]+|MODELS_DIR))=", example))
        self.assertEqual(used - section, set(), "a value outside the node section: a node-only file cannot fill it")

    def test_a_node_only_file_without_the_applied_intent_is_refused_naming_the_line(self):
        r, composes, _ = self._run("node", server=NODE_HOST)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_IMAGE is not set", r.stderr)
        self.assertEqual(composes, [])

    def test_without_the_edge_the_edge_target_is_refused_before_anything(self):
        """A published tree has no docker-compose.yml here: `up.sh edge` says so and dials no docker; a full bring-up
        is node lb bench (and the chat where the tree has it). With the edge present the target is the two gates and
        nginx, as before."""
        r, composes, _ = self._run("edge", server=GOOD)
        self.assertEqual(composes, [] if not EDGE else composes)
        if EDGE:
            self.assertNotIn("ships no signed-in edge", r.stderr)
        else:
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("ships no signed-in edge", r.stderr)
            self.assertIn("EDGE_LB_PUBLISH", r.stderr)

    def test_without_the_chat_the_chat_target_is_refused_before_anything(self):
        """A published tree has no chat.override.yml here (the chat is its own project, catflops_chat): `up.sh chat`
        says so and dials no docker, and a full bring-up is node lb bench. With the override present the target is
        the chat and its sandbox, as before."""
        r, composes, _ = self._run("chat", server=GOOD)
        if CHAT:
            self.assertNotIn("ships no chat front-end", r.stderr)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual([c for c in composes if "up" in c], [c for c in composes if "vllm-chat" in c])
        else:
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(composes, [])
            self.assertIn("ships no chat front-end", r.stderr)
            self.assertIn("catflops_chat", r.stderr)
            self.assertIn("EDGE_LB_PUBLISH", r.stderr)

    def test_the_chat_lines_are_required_only_with_the_chat_target(self):
        """EDGE_CHAT_PUBLISH, EDGE_PUBLIC_CHAT_ORIGIN and EDGE_CHAT_AUDIENCE are the chat's: a server file without them
        brings up the LB; `up.sh chat` refuses it naming the line (where the tree has the chat at all)."""
        without = {k: v for k, v in GOOD.items() if k not in CHAT_VARS}
        r = _check(without, "lb")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = _check(without, "chat")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_CHAT_PUBLISH" if CHAT else "ships no chat front-end", r.stdout + r.stderr)
        if CHAT:
            r = _check({**without, "EDGE_CHAT_PUBLISH": "0.0.0.0:8011", "EDGE_PUBLIC_CHAT_ORIGIN": "https://c.example.test",
                        "EDGE_CHAT_AUDIENCE": "c"}, "chat")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("EDGE_CHAT_PUBLISH", r.stdout + r.stderr, "the chat's publish must stay on loopback")

    def test_a_node_only_file_brings_up_nothing_else(self):
        for target in ("lb", "bench") + (("chat",) if CHAT else ()) + (("edge",) if EDGE else ()):
            r, composes, _ = self._run(target, server=NODE_HOST, fragment=APPLIED_REST)
            self.assertNotEqual(r.returncode, 0, target)
            self.assertIn("EDGE_BOX is not set", r.stderr, target)
            self.assertEqual(composes, [], target)
        r, composes, _ = self._run("node", "lb", server=NODE_HOST, fragment=APPLIED_REST)
        self.assertNotEqual(r.returncode, 0, "node with anything else is a full file")

    def test_a_node_only_files_node_still_stays_on_loopback(self):
        r, composes, _ = self._run("node", server={**NODE_HOST, "EDGE_NODE_PUBLISH": "0.0.0.0:8020"},
                                   fragment=APPLIED_REST)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_PUBLISH=0.0.0.0:8020 must be a loopback", r.stderr)
        self.assertEqual(composes, [])

    # ── a node on ANOTHER box's docker (EDGE_NODE_DOCKER_HOST): published on that box's private link ──────────

    REMOTE = {**NODE_HOST, "EDGE_NODE_DOCKER_HOST": "ssh://operator@node-2", "EDGE_NODE_PUBLISH": "10.0.0.2:18000"}

    def test_a_node_on_another_box_runs_every_docker_call_on_that_boxs_daemon(self):
        r, composes, tmp = self._run("node", server=self.REMOTE, fragment=APPLIED_REST)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(composes), 1)
        self.assertEqual(set((tmp / "docker.hosts").read_text().split()), {"ssh://operator@node-2"})

    def test_a_node_on_another_box_is_published_on_a_private_address_never_loopback_or_everywhere(self):
        """Loopback THERE is unreachable from the LB here; 0.0.0.0 or a public address serves without sign-in."""
        for publish in ("127.0.0.1:18000", "0.0.0.0:18000", "203.0.113.5:18000"):
            r, composes, _ = self._run("node", server={**self.REMOTE, "EDGE_NODE_PUBLISH": publish},
                                       fragment=APPLIED_REST)
            self.assertNotEqual(r.returncode, 0, publish)
            self.assertIn(f"EDGE_NODE_PUBLISH={publish} must be a private address", r.stderr)
            self.assertEqual(composes, [], publish)
        for publish in ("172.16.0.2:18000", "192.168.1.2:18000"):
            r, _, _ = self._run("node", server={**self.REMOTE, "EDGE_NODE_PUBLISH": publish}, fragment=APPLIED_REST)
            self.assertEqual(r.returncode, 0, publish + r.stderr)

    def test_another_boxs_docker_is_for_a_node_only_server_file(self):
        r, composes, _ = self._run("node", "lb", server={**GOOD, "EDGE_NODE_DOCKER_HOST": "ssh://operator@node-2"},
                                   fragment=APPLIED_REST)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_DOCKER_HOST", r.stderr)
        self.assertEqual(composes, [])

    # ── the units: ONE SET PER CLUSTER, rendered from the templates by the installer ───────────────────────────

    def test_the_units_run_the_applier_for_their_cluster_on_an_intent_and_on_a_heartbeat(self):
        service = (SA / "vllm-node-applier.service.in").read_text()
        self.assertIn("Type=oneshot", service)
        self.assertIn("Environment=PYTHONPATH=@REPO@/lb", service)
        self.assertIn("ExecStart=@PY@ -m vllm_lb.node_applier --repo @REPO@ --env @ENV@ --cluster @CLUSTER@", service)
        path = (SA / "vllm-node-applier.path.in").read_text()
        self.assertIn("PathChanged=@NODE_DIR@/intent.json", path, "the cluster's own intent (the applier's --node-dir)")
        self.assertIn("Unit=vllm-node-applier-@CLUSTER@.service", path)
        timer = (SA / "vllm-node-applier.timer.in").read_text()
        self.assertIn("Unit=vllm-node-applier-@CLUSTER@.service", timer)
        period = re.search(r"^APPLIER_PERIOD_S = (\d+)$", (ROOT / "lb/vllm_lb/node_control.py").read_text(), re.M)
        self.assertIsNotNone(period)
        self.assertIn(f"OnUnitActiveSec={period.group(1)}s", timer, "the heartbeat the LB judges staleness by")
        for text in (service, path, timer):
            left = set(re.findall(r"@[A-Z_]+@", text)) - {"@PY@", "@REPO@", "@ENV@", "@CLUSTER@", "@NODE_DIR@"}
            self.assertEqual(left, set(), "every placeholder is one the installer renders")

    def test_the_installer_checks_before_it_installs_and_proves_one_pass(self):
        code = "\n".join(_code(SA / "install-node-applier.sh"))
        for needed in ("import yaml", "lb/fleet.yml", "loginctl", "chown", "--cluster", "--node-dir",
                       "-m vllm_lb.node_applier --repo", "vllm-node-applier-$CLUSTER",
                       's|@CLUSTER@|$CLUSTER|g', 's|@NODE_DIR@|$NODE_DIR|g', 's|@ENV@|$ENV_FILE|g'):
            self.assertIn(needed, code)
        self.assertLess(code.index("--node-dir"), code.index("enable --now"), "the watch is the applier's answer")
        self.assertLess(code.index("-m vllm_lb.node_applier --repo"), code.index("enable --now"),
                        "one real pass before the units exist: an applier that cannot run fails the install")

    def test_the_installer_refuses_without_a_cluster_before_anything_is_touched(self):
        r = subprocess.run(["bash", str(SA / "install-node-applier.sh")], capture_output=True, text=True, timeout=30,
                           env={"PATH": "/usr/bin:/bin", "HOME": tempfile.gettempdir()})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--cluster", r.stderr)

    # ── which server the node runs: the launch choice's engine, as data ────────────────────────────────────────

    def test_the_node_runs_the_override_its_engine_names(self):
        r, composes, _ = self._run("node", server=NODE_HOST, fragment=APPLIED_REST + "EDGE_NODE_ENGINE=faster-whisper\n")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self._files(composes[0]), ["node.yml", "node-gpu.rocm.yml", "node-mfu.off.yml",
                                                    "node-engine.faster-whisper.yml"])

    def test_an_engine_with_no_override_is_refused_before_docker_is_touched(self):
        r, composes, _ = self._run("node", server={**GOOD, "EDGE_NODE_ENGINE": "whisper.cpp"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("EDGE_NODE_ENGINE=whisper.cpp", r.stderr)
        self.assertIn("node-engine.vllm.yml", r.stderr, "names the engines there are")
        self.assertEqual(composes, [])

if __name__ == "__main__":
    unittest.main()


class StandaloneNodeLimits(unittest.TestCase):
    """The node container runs BOUNDED on a box the LB, chat and broker share (AUDIT-FINAL-5 P2-4): memory with no swap,
    CPUs, processes and its niceness — each the box's `node_limits` in the fleet file, rendered by node-env, `${VAR:?}`
    here (no default). The measured gui run's own guards: its one live failure was a process explosion only the memcg
    contained."""

    def _node(self) -> str:
        return (SA / "node.yml").read_text().split("\n  node:", 1)[1].split("\n  gate:", 1)[0]

    def test_memory_cpus_and_pids_are_the_rendered_limits_with_no_default(self):
        node = self._node()
        for key, var in (("mem_limit", "EDGE_NODE_MEMORY"), ("memswap_limit", "EDGE_NODE_MEMORY"),
                         ("cpus", "EDGE_NODE_CPUS"), ("pids_limit", "EDGE_NODE_PIDS_LIMIT")):
            self.assertRegex(node, rf'\n    {key}: "\$\{{{var}:\?[^}}]*\}}"', key)

    def test_the_engine_runs_at_the_rendered_niceness(self):
        self.assertIn('EDGE_NODE_NICE: "${EDGE_NODE_NICE:?', self._node())
        self.assertIn("exec nice -n \"$$EDGE_NODE_NICE\" vllm serve", (SA / "node-engine.vllm.yml").read_text())
        self.assertIn('nice -n "$$EDGE_NODE_NICE" python -m transcription',
                      (SA / "node-engine.faster-whisper.yml").read_text())

    def test_the_limits_are_documented_and_rendered_by_the_fleet(self):
        example = (SA / ".env.example").read_text()
        for var in ("EDGE_NODE_MEMORY", "EDGE_NODE_CPUS", "EDGE_NODE_PIDS_LIMIT", "EDGE_NODE_NICE"):
            self.assertRegex(example, rf"(?m)^{var}=$", var)
        fleet = (ROOT / "lb" / "vllm_lb" / "fleet.py").read_text()
        for var in ("EDGE_NODE_MEMORY", "EDGE_NODE_CPUS", "EDGE_NODE_PIDS_LIMIT", "EDGE_NODE_NICE"):
            self.assertIn(var, fleet)


class StandaloneNodeEngine(unittest.TestCase):
    """THE SERVER A NODE RUNS is its launch choice's engine (lb/vllm_lb/fleet.py ENGINES; node-env renders
    EDGE_NODE_ENGINE). The base node is vLLM's; node-engine.vllm.yml adds nothing, node-engine.faster-whisper.yml replaces
    its entrypoint and command with the transcription server's own (docs/design/engines.md's contract: MODEL_SIZE, DEVICE,
    MAX_ACTIVE_REQUESTS, WORKER_ID — its env, from the node's values), on the same port, mount, limits and card. Two
    nodes on one box (a chat node and a speech one) each run in a compose PROJECT of their own (EDGE_NODE_PROJECT): one
    project with one `node` service per box would recreate the other's container."""

    def _node(self, name: str) -> str:
        return (SA / name).read_text().split("\n  node:", 1)[1]

    def test_each_node_is_its_own_compose_project(self):
        self.assertRegex((SA / "node.yml").read_text(), r'(?m)^name: "\$\{EDGE_NODE_PROJECT:\?[^}]*\}"$')

    def test_the_base_runs_no_server_each_engine_runs_its_own(self):
        base = "\n".join(_code(SA / "node.yml"))
        for word in ("entrypoint:", "command:", "vllm serve", "EDGE_NODE_MAX_MODEL_LEN", "EDGE_NODE_GPU_MEM_UTIL",
                     "EDGE_NODE_SERVED_NAME", "EDGE_NODE_KV_CACHE_MEMORY"):
            self.assertNotIn(word, base, "a variable or a command only one engine reads is that engine's override's")
        vllm = "\n".join(_code(SA / "node-engine.vllm.yml"))
        for need in ("vllm serve", '"${EDGE_NODE_MAX_MODEL_LEN:?', '"${EDGE_NODE_GPU_MEM_UTIL:?',
                     '"${EDGE_NODE_SERVED_NAME:?', "EDGE_NODE_KV_CACHE_MEMORY:"):
            self.assertIn(need, vllm)
        self.assertNotIn("EDGE_NODE_MAX_QUEUE", vllm)

    def test_the_faster_whisper_override_runs_the_service_on_the_nodes_values(self):
        node = "\n".join(ln for ln in self._node("node-engine.faster-whisper.yml").splitlines()
                         if not ln.lstrip().startswith("#"))
        for need in ('MODEL_SIZE: "/models/${EDGE_NODE_MODEL:?', 'MAX_ACTIVE_REQUESTS: "${EDGE_NODE_MAX_NUM_SEQS:?',
                     'WORKER_ID: "${EDGE_NODE_NAME:?', "DEVICE: cuda", 'nice -n "$$EDGE_NODE_NICE" python -m transcription',
                     "entrypoint:", "command: []", "EDGE_NODE_MAX_QUEUE:", "EDGE_NODE_COMPUTE_TYPE:"):
            self.assertIn(need, node)
        self.assertNotIn("vllm", node)

    def _service_env(self, **env: str) -> dict:
        """Run the override's entrypoint for real, with a `python` on PATH that prints the environment it got."""
        import json
        node = self._node("node-engine.faster-whisper.yml")
        script = re.search(r"\n      - '(exec .*)'\n", node).group(1).replace("$$", "$")
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fake = tmp / "python"
        fake.write_text(f"#!{sys.executable}\nimport json, os\nprint(json.dumps(dict(os.environ)))\n")
        fake.chmod(0o755)
        r = subprocess.run(["/bin/sh", "-c", script], env={"PATH": f"{tmp}:/usr/bin:/bin", "EDGE_NODE_NICE": "10", **env},
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_the_queue_and_compute_type_reach_the_service_only_when_set(self):
        for unset in ({}, {"EDGE_NODE_MAX_QUEUE": "", "EDGE_NODE_COMPUTE_TYPE": ""}):
            got = self._service_env(**unset)
            for key in ("MAX_QUEUE_SIZE", "FAIL_FAST_WHEN_BUSY", "COMPUTE_TYPE"):
                self.assertNotIn(key, got, f"{key}: the service's / image's own when unset")
        got = self._service_env(EDGE_NODE_MAX_QUEUE="8", EDGE_NODE_COMPUTE_TYPE="int8_float16")
        self.assertEqual((got["MAX_QUEUE_SIZE"], got["FAIL_FAST_WHEN_BUSY"], got["COMPUTE_TYPE"]),
                         ("8", "false", "int8_float16"))


class StandaloneToolParser(unittest.TestCase):
    """vLLM's tool flags (--enable-auto-tool-choice --tool-call-parser) are a CHAT node's: the fleet requires the parser
    on a chat cluster and refuses it on a transcription one, and node-env renders it empty there — so the flags reach
    vLLM only when the variable holds a value, as the other optional flags do."""

    def test_the_tool_flags_reach_vllm_only_with_a_parser(self):
        run = StandaloneNodePrefillChunk("test_the_flag_reaches_vllm_only_when_the_chunk_is_set")
        run.setUp()
        try:
            for unset in (None, ""):
                extra = {} if unset is None else {"EDGE_NODE_TOOL_CALL_PARSER": unset}
                argv = run._argv(None, **extra)
                self.assertNotIn("--enable-auto-tool-choice", argv)
                self.assertNotIn("--tool-call-parser", argv)
                self.assertIn("--enable-mfu-metrics", argv, "every vLLM node measures itself (whisper too, 2026-10-23)")
            argv = run._argv(None, EDGE_NODE_TOOL_CALL_PARSER="hermes")
            self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "hermes")
            self.assertEqual(argv.count("--enable-auto-tool-choice"), 1)
        finally:
            run.doCleanups()


class StandaloneGpuRuntime(unittest.TestCase):
    """The base node service names NO way to attach a GPU; each `gpu_runtime` override does, in its runtime's own
    recipe — NVIDIA's device reservation, ROCm's device nodes (AMD ROCm docs, "Running ROCm Docker containers":
    --device /dev/kfd --device /dev/dri, seccomp unconfined) — and nothing of the other's."""

    def _node(self, name: str) -> str:
        return (SA / name).read_text().split("\n  node:", 1)[1].split("\n  gate:", 1)[0]

    def test_the_base_node_attaches_no_gpu(self):
        node = "\n".join(ln for ln in self._node("node.yml").splitlines() if not ln.lstrip().startswith("#"))
        for word in ("driver: nvidia", "/dev/kfd", "/dev/dri", "reservations"):
            self.assertNotIn(word, node)

    def test_the_nvidia_override_reserves_the_one_card_by_its_index(self):
        node = self._node("node-gpu.nvidia.yml")
        self.assertIn("driver: nvidia", node)
        self.assertIn('device_ids: ["${EDGE_NODE_GPU:?', node)
        self.assertNotIn("/dev/kfd", node)

    def test_the_rocm_override_passes_the_device_nodes_and_their_groups(self):
        node = self._node("node-gpu.rocm.yml")
        for need in ("/dev/kfd", "/dev/dri", "seccomp=unconfined", "${EDGE_NODE_GPU:?",
                     '"${EDGE_NODE_VIDEO_GID:?', '"${EDGE_NODE_RENDER_GID:?'):
            self.assertIn(need, node)
        self.assertNotIn("nvidia", node.lower())
        code = [ln.strip() for ln in node.splitlines() if not ln.lstrip().startswith("#")]
        self.assertNotIn("- video", code, "by NUMERIC gid: an image's /etc/group need not name it")
        self.assertNotIn("- render", code, "by NUMERIC gid: rocm/vllm's has no `render` (docker: Unable to find group)")

    def test_every_runtime_a_hardware_type_may_state_has_its_override(self):
        profiles = (ROOT / "lb" / "vllm_lb" / "hardware" / "facts.py").read_text()
        m = re.search(r"^GPU_RUNTIMES = \(([^)]*)\)", profiles, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(tuple(re.findall(r'"([a-z]+)"', m.group(1))), GPU_RUNTIMES)
        for name in NODE_GPU:
            self.assertTrue((SA / name).is_file(), name)
