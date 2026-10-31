"""THE SETUP WIZARD (SETUP.md, deploy/setup/) — its question library covers every config surface, and its renderer turns
an answer sheet into exactly the files the services read.

Operator, 2026-10-30: "implement agent setup wizard rules, should ask about dir where to keep models, what model should
it load etc. prepare full list to ask user before setup". A coding agent setting the fleet up COLLECTS every choice
before it writes anything, because the project's rule is no defaults and no auto-detection (AGENTS.md): every
deployment value is a deliberate answer. These tests keep that true as the code moves:

  * COVERAGE — every key a question writes exists where it lands (the matching .env.example, the fleet schema, the
    power inventory's fields); every REQUIRED key of each .env.example (an uncommented `KEY=` line) and every
    `${KEY:?}` its compose files demand is written by some question or derivation, so a new env var without a wizard
    question fails here; every key of every shipped fleet example is written; no question writes a RETIRED name.
  * LIBRARIES — a choice read from a library (hardware types, accelerators, the model library, engines, KV dtypes,
    platforms, controls, links, the node and sender compose overrides) IS that library, read here with the LB's own
    code (the vllm-lb image); in python:3.12-slim every source still reads, without PyYAML, so --check runs there.
  * RENDERING — each committed answer sheet renders, and what it renders is the shipped example: the pair sheet's
    lb/fleet.yml is lb/fleet.example.yml as data, its power/.env is power/.env.example's inventory as the power
    service parses it. A sheet with a gap, an unknown id, a bad choice or an answer to a question its `when` skips is
    refused, naming it; a secret never appears in the summary.
  * THE LB'S OWN LOADER accepts what the renderer writes (fleet.load_fleet; node-env for the standalone node). That
    loader needs PyYAML, which the vllm-lb image has and python:3.12-slim does not: those tests skip here naming the
    exact command that runs them.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "deploy" / "setup"
sys.path.insert(0, str(SETUP))
sys.path.insert(0, str(REPO / "lb"))

import miniyaml  # noqa: E402
import render  # noqa: E402

try:
    import yaml  # noqa: F401  — the vllm-lb image has it; python:3.12-slim does not
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
IN_LB_IMAGE = ('docker run --rm -v "$PWD":/repo:ro -w /repo vllm-lb:dev '
               'python -m unittest discover -s deploy/tests -p test_setup.py')
needs_lb_loader = unittest.skipUnless(HAS_YAML, "the LB's fleet loader needs PyYAML (the vllm-lb image): " + IN_LB_IMAGE)

PAIR = SETUP / "answers.example.yml"
#: The reference blueprints (lb/vllm_lb/library/blueprints): the 2× GB10 pair's and the one-RTX-5090 node's.
PAIR_BP, NODE_BP = "2x-gb10-qwen3.8-flash-next-hibrid48", "1x-rtx5090-qwen3.8-27b-nvfp4"
STANDALONE = SETUP / "answers.standalone.example.yml"
GATE_MARKER = "GATE-BELOW"


def _questions():
    return render.Library.load(SETUP / "questions.yml", REPO)


def _env_keys(path: Path, *, documented: bool) -> list[str]:
    """The keys an env example names: uncommented `KEY=` lines (required), or with `documented` also `# KEY=` ones —
    read only above the GATE-BELOW marker, the part of deploy/standalone/.env.example a published tree keeps."""
    text = path.read_text(encoding="utf-8").split(GATE_MARKER, 1)[0]
    pat = r"^(?:#\s*)?([A-Z][A-Z0-9_]*)=" if documented else r"^([A-Z][A-Z0-9_]*)="
    return re.findall(pat, text, re.M)


def _compose_required(paths) -> set[str]:
    out = set()
    for p in paths:
        text = (REPO / p).read_text(encoding="utf-8")
        out |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):\?", text))
    return out - {"VAR"}       # a compose file's own comment explaining `${VAR:?}` names no variable


def _fleet_paths(node, prefix="") -> set[str]:
    """Every key path in a fleet document, names generalised (boxes.<box>, clusters.<cluster>, backends.<box>)."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if prefix in ("boxes", "backends"):
                key = f"{prefix}.<box>"
            elif prefix == "clusters":
                key = f"{prefix}.<cluster>"
            elif prefix == "checkpoints":          # a model id has dots of its own: never split into path segments
                key = f"{prefix}.<checkpoint>"
            elif prefix.endswith(".launch.image") and isinstance(node, dict):
                key = f"{prefix}"        # an engine → image mapping: the image key itself is the leaf
                out.add(key)
                continue
            else:
                key = f"{prefix}.{k}" if prefix else str(k)
            out.add(key)
            out |= _fleet_paths(v, key)
    return out


def _render(sheet: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SETUP / "render.py"), str(sheet), "--out", str(out), *extra],
                          cwd=REPO, capture_output=True, text=True)


class MiniYaml(unittest.TestCase):
    """deploy/setup/miniyaml.py — the strict YAML subset the wizard reads without PyYAML (python:3.12-slim)."""

    def test_block_mappings_sequences_flow_and_scalars(self):
        doc = miniyaml.parse(
            "a: 1\nb: 0.98\nc: true\nd: null\ne: linux/arm64   # a comment\nf: \"x: y # z\"\ng: 'it''s'\n"
            "h: [a, \"b c\", 3]\ni: {index: 0, type: gb10, serves: true}\nj:\n  - x\n  - {k: v}\n  - k1: 1\n    k2: 2\n"
            "k:\n- one\nl: >\n  folded\n  text\nm: yes\n", "t")
        self.assertEqual(doc, {"a": 1, "b": 0.98, "c": True, "d": None, "e": "linux/arm64", "f": "x: y # z",
                               "g": "it's", "h": ["a", "b c", 3], "i": {"index": 0, "type": "gb10", "serves": True},
                               "j": ["x", {"k": "v"}, {"k1": 1, "k2": 2}], "k": ["one"], "l": "folded text",
                               "m": "yes"})

    def test_what_a_configuration_must_not_mean_is_refused_naming_the_source(self):
        for bad in ("a: 1\na: 2\n", "a:\n\t- x\n", "a: &x 1\n", "a: *x\n", "a: !!str 1\n", "---\na: 1\n",
                    "a: [1, 2\n", "a: \"open\n"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "^src: "):
                miniyaml.parse(bad, "src")

    def test_dump_round_trips_and_quotes_what_would_change_type(self):
        doc = {"boxes": {"box-a": {"hostname": "box-a-host", "cards": [{"index": 0, "type": "gb10", "serves": True}],
                                   "ssh": "operator@192.0.2.10"}},
               "s": ["true", "8010", "a: b", "", None, 0.5, "127.0.0.1:8010"], "e": {}}
        self.assertEqual(miniyaml.parse(miniyaml.dump(doc), "t"), doc)

    def test_the_shipped_fleet_examples_read_as_pyyaml_reads_them(self):
        for p in sorted((REPO / "lb").glob("fleet*.example.yml")):
            with self.subTest(p=p.name):
                doc = miniyaml.parse(p.read_text(encoding="utf-8"), p.name)
                self.assertIn("boxes", doc)
                if HAS_YAML:
                    from vllm_lb import yamlfile
                    self.assertEqual(doc, yamlfile.parse(p.read_text(encoding="utf-8"), p.name))

    @needs_lb_loader
    def test_the_wizard_files_read_as_the_strict_pyyaml_loader_reads_them(self):
        from vllm_lb import yamlfile
        for p in (SETUP / "questions.yml", PAIR, STANDALONE):
            with self.subTest(p=p.name):
                text = p.read_text(encoding="utf-8")
                self.assertEqual(miniyaml.parse(text, p.name), yamlfile.parse(text, p.name))


class QuestionLibrary(unittest.TestCase):
    """deploy/setup/questions.yml — every question well-formed, in asking order, nothing a deployment's own."""

    @classmethod
    def setUpClass(cls):
        cls.lib = _questions()

    def test_every_question_has_the_fields_an_agent_needs(self):
        for q in self.lib.questions:
            with self.subTest(q=q.id):
                for f in ("id", "section", "ask", "why", "kind", "writes", "validate"):
                    self.assertTrue(q.raw.get(f), f"{q.id} has no {f}")
                self.assertIn(q.kind, render.KINDS)
                self.assertIn(q.section, [s["id"] for s in self.lib.sections])
                if q.kind in ("choice", "multi"):
                    self.assertTrue(q.raw.get("choices") or q.raw.get("choices_from") or q.raw.get("also_from"),
                                    f"{q.id}: a choice lists them")

    def test_questions_are_grouped_in_the_sections_asking_order(self):
        order = [s["id"] for s in self.lib.sections]
        seen = [order.index(q.section) for q in self.lib.questions]
        self.assertEqual(seen, sorted(seen), "questions must appear section by section, in the sections' order")

    def test_ids_are_unique_and_every_when_names_an_earlier_question(self):
        ids = [q.id for q in self.lib.questions]
        self.assertEqual(len(ids), len(set(ids)))
        for i, q in enumerate(self.lib.questions):
            for ref in self.lib.when_refs(q):
                with self.subTest(q=q.id, ref=ref):
                    self.assertIn(ref, ids[:i], f"{q.id}'s when names {ref}, which is not asked before it")

    def test_every_secret_is_marked_and_never_given_a_probe_that_prints_it(self):
        for q in self.lib.questions:
            if q.kind == "secret":
                self.assertIs(q.raw.get("secret"), True, f"{q.id} is a secret: mark it secret: true")
            if q.raw.get("secret"):
                self.assertEqual(q.kind, "secret", q.id)

    def test_the_topics_the_operator_named_are_asked(self):
        """The models directory, which model to load, engine, context, concurrency, KV dtype, speculative decoding,
        a gated model's token, download now or staged, exposure and its sign-in, broker, power, web search,
        telemetry, usage-mcp, clients — each has a question."""
        ids = {q.id for q in self.lib.questions}
        for needed in ("shape.kind", "machines.boxes", "box.platform", "box.cards", "box.gpu_runtime", "box.docker",
                       "box.models_dir", "box.models_free_gib", "models.shared_dir", "cluster.model",
                       "cluster.launch_choices", "cluster.max_model_len", "cluster.max_num_seqs",
                       "cluster.kv_cache_dtype", "models.speculative", "models.quantisation", "models.engine",
                       "download.mode", "download.hf_token", "exposure.scope", "exposure.auth", "broker.enabled",
                       "power.enabled", "search.searxng_secret", "telemetry.enabled", "extras.usage_mcp",
                       "clients.kinds", "hwtype.new", "newmodel.hf_repo",
                       # the engines × accelerators redesign (docs/design/engines.md): where each checkpoint sits, the
                       # link a pair's ranks are joined by, a new card's accelerator, an unmeasured first Start
                       "models.staged", "checkpoint.path", "checkpoint.uncensored", "checkpoint.draft",
                       "cluster.link", "hwtype.accelerator", "accel.new", "accel.arch", "cluster.first_start",
                       # the gateway panel's default style (operator, 2026-10-31: "allow setup default in yml config")
                       "panel.style"):
            self.assertIn(needed, ids)

    def test_exposure_beyond_localhost_cannot_be_answered_without_an_auth_layer(self):
        auth = self.lib.by_id["exposure.auth"]
        self.assertNotIn("none", auth.choices(self.lib, {}))
        self.assertIn("exposure.scope", " ".join(self.lib.when_refs(auth)))

    def test_a_probe_only_proposes(self):
        """A probe is read-only: no write, install, start or pull — what it prints is a proposal the user confirms."""
        writes = re.compile(r"(?<![\w-])(rm|mv|cp|tee|dd|mkdir|chmod|chown|install|apt|apt-get|pip|up|start|pull|"
                            r"run|restart|kill|reboot|poweroff)(?![\w-])|>")
        for q in self.lib.questions:
            probe = q.raw.get("probe")
            if probe:
                with self.subTest(q=q.id):
                    cmd = re.sub(r"<[^<>]*>", "", probe.replace("2>/dev/null", "")).split("#", 1)[0]
                    self.assertFalse(writes.search(cmd), f"{q.id}'s probe changes something")


class Coverage(unittest.TestCase):
    """Every key a question writes exists where it lands; every required key is written by some question."""

    @classmethod
    def setUpClass(cls):
        cls.lib = _questions()
        cls.writes = cls.lib.all_writes()          # (file template, key) for questions and derivations

    def _written(self, file_pattern: str) -> set[str]:
        return {k.split("[")[0] for f, k in self.writes if re.fullmatch(file_pattern, f)}

    def test_every_env_file_key_written_is_documented_in_its_example(self):
        for target, spec in self.lib.files.items():
            if spec["format"] != "env":
                continue
            with self.subTest(target=target):
                documented = set(_env_keys(REPO / spec["example"], documented=True))
                if spec.get("compose_file_key"):
                    documented.add("COMPOSE_FILE")
                written = self._written(spec["pattern"])
                self.assertFalse(written - documented, f"{target}: written but not in {spec['example']}")

    def test_every_required_env_key_is_written_by_some_question(self):
        for target, spec in self.lib.files.items():
            if spec["format"] != "env":
                continue
            with self.subTest(target=target):
                required = set(_env_keys(REPO / spec["example"], documented=False))
                required |= _compose_required(c for c in spec.get("compose", []) if (REPO / c).exists())
                missing = required - self._written(spec["pattern"]) - self.lib.node_env_keys()
                self.assertFalse(missing, f"{target}: required but no question writes it: {sorted(missing)}")

    def test_no_question_writes_a_retired_name(self):
        retired = render.retired_keys(REPO)
        self.assertIn("LB_CONTEXT_LENGTH", retired)
        self.assertIn("LB_FALLBACK_URLS", retired)
        hit = sorted({k for _, k in self.writes} & retired)
        self.assertFalse(hit, f"a question writes a RETIRED name: {hit}")

    def test_every_key_of_every_shipped_fleet_example_is_written(self):
        written = self._written(r"lb/fleet\.yml")
        for p in sorted((REPO / "lb").glob("fleet*.example.yml")):
            with self.subTest(p=p.name):
                paths = _fleet_paths(miniyaml.parse(p.read_text(encoding="utf-8"), p.name))
                paths = {x for x in paths if x.count(".") >= 2}
                missing = {x for x in paths if not any(w == x or x.startswith(w + ".") for w in written)}
                self.assertFalse(missing, f"{p.name}: no question writes {sorted(missing)}")

    def test_every_fleet_key_written_is_one_the_fleet_loader_names(self):
        """Every segment of a fleet path a question writes is a key fleet.py names (a quoted word or a key constant
        in its source), so a misspelt key is caught here, before the loader refuses it at boot. The loader itself
        reads the rendered file in InLbImage."""
        names = render.fleet_keys(REPO)
        self.assertTrue({"hostname", "platform", "cards", "kv_offload_ram_max_gib", "max_queue"} <= names)
        for _, key in [w for w in self.writes if w[0] == "lb/fleet.yml"]:
            for seg in key.split("."):
                if not seg.startswith("<"):
                    with self.subTest(key=key, seg=seg):
                        self.assertIn(seg, names)

    def test_the_power_inventory_fields_are_the_services_own(self):
        src = (REPO / "power/fleet_power/boxes.py").read_text(encoding="utf-8")
        fields = set(re.findall(r'"(\w+)"', re.search(r"_BOX_FIELDS = \{(.*?)\}", src, re.S).group(1)))
        wol = set(re.findall(r'"(\w+)"', re.search(r"_WOL_FIELDS = \{(.*?)\}", src, re.S).group(1)))
        written = {k.split("].", 1)[1] for f, k in self.writes if k.startswith("POWER_BOXES[")}
        self.assertEqual({w.split(".")[0] for w in written} | {"name"}, fields)
        self.assertEqual({w.split(".", 1)[1] for w in written if w.startswith("wol.")}, wol)

    @needs_lb_loader
    def test_manual_writes_name_fields_their_schema_has(self):
        """The sub-wizards' writes (a hardware type, an accelerator, a model's library entries) are files the agent
        writes by hand with its test: each still names a field its loader takes — `arch` and `gpu_runtime` are an
        ACCELERATOR's since the engines × accelerators redesign, never a type's; a model's row is split into
        models.json (facts), recipes.json (engine settings) and measurements.json (what was measured), and where its
        checkpoint sits is the fleet file's (docs/design/engines.md)."""
        from vllm_lb import catalog
        from vllm_lb.hardware import profiles
        # The module, not the function the package re-exports under the same name.
        accelerator = importlib.import_module("vllm_lb.hardware.accelerator")
        hardware = {
            "lb/vllm_lb/hardware/types/*.yml": (set(profiles._TOP), set(profiles._FACTS)),
            "lb/vllm_lb/hardware/accelerators/*.yml": (set(accelerator._TOP), set(accelerator._FACTS)),
        }
        models_row = (catalog._SPEC_REQUIRED | catalog._SPEC_OPTIONAL) - set(catalog.RETIRED_ROW_KEYS) - {"measured_on"}
        recipe = {"model", "launch", "node_engine", "overrides"}
        profile = set(catalog._PROFILE_OPTIONAL) | {"build", "draft_checkpoint"}
        measurement = {"model", "covers", "covers_source", "memory", "engine_memory", "launch"}
        self.assertNotIn("lb/vllm_lb/catalog.json", self.lib.files, "catalog.json is retired: the library is split")
        manual = [(f, k) for f, k in self.writes if self.lib.files.get(f, {}).get("format") == "manual"]
        self.assertEqual({f for f, _ in manual}, set(hardware) | {
            "lb/vllm_lb/library/models.json", "lb/vllm_lb/library/recipes.json", "lb/vllm_lb/library/measurements.json"})
        for f, k in manual:
            with self.subTest(f=f, k=k):
                if f in hardware:
                    top, facts = hardware[f]
                    self.assertTrue(k in top or (k.startswith("facts.") and k.split(".")[1] in facts), k)
                    continue
                field, rest = re.fullmatch(r"\w+\[[^\]]+\]\.([\w-]+)(.*)", k).groups()
                if f == "lb/vllm_lb/library/models.json":
                    self.assertIn(field, models_row)
                elif f == "lb/vllm_lb/library/recipes.json":
                    self.assertIn(field, recipe)
                    if field == "launch" and rest:
                        self.assertIn(rest.split(".")[2], profile, k)
                else:
                    self.assertIn(field, measurement)


class Libraries(unittest.TestCase):
    """A choice read from a library is that library."""

    @classmethod
    def setUpClass(cls):
        cls.lib = _questions()

    @needs_lb_loader
    def test_each_library_source_is_what_the_repo_holds(self):
        """Each source against the library as the LB's OWN code reads it — the renderer reads the files without
        PyYAML (python:3.12-slim), so this is the independent read."""
        from vllm_lb import blueprints, catalog, engine_library, executors, fleet, panel_styles
        from vllm_lb.hardware import FAMILIES, accelerators, facts, types
        kv = json.loads((REPO / "lb/vllm_lb/kv_cache_dtypes.json").read_text())["layouts"]
        models = catalog.library_models()
        recipes = json.loads((REPO / "lb/vllm_lb/library/recipes.json").read_text())["recipes"]
        expect = {
            "hardware_types": sorted(types()),
            "accelerators": sorted(accelerators()),
            "blueprints": sorted(blueprints.library()),
            "library_models": [m["id"] for m in models],
            "model_refs": [r for m in models for r in ([m["id"]] + ([f"{m['id']}@uncensored"] if "uncensored" in m
                                                                     else []))],
            "uncensored_models": [m["id"] for m in models if "uncensored" in m],
            "draft_models": [r["model"] for r in recipes
                             if any(p.get("draft_checkpoint") for p in r["launch"].values())],
            "node_engines": list(catalog.NODE_ENGINES),
            "cluster_engines": list(catalog.CLUSTER_ENGINES),
            "engines": sorted(engine_library.engines()),
            "kv_dtypes": list(kv),
            "platforms": list(facts.PLATFORMS),
            "gpu_runtimes": list(facts.GPU_RUNTIMES),
            "memory_families": list(FAMILIES),
            "controls": list(fleet.CONTROLS),
            "panel_styles": list(panel_styles.keys()),
            "links": [*executors.LINKS_ACROSS_HOSTS, *executors.LINKS_ONE_HOST],
            "tasks": list(catalog.TASKS),
            "quant_methods": sorted({m["quant_method"]["value"] for m in models if m["quant_method"]["value"]}),
            "node_gpu_overrides": sorted(p.stem.split(".", 1)[1] for p in (REPO / "deploy/standalone").glob("node-gpu.*.yml")),
            "node_engine_overrides": sorted(p.stem.split(".", 1)[1] for p in (REPO / "deploy/standalone").glob("node-engine.*.yml")),
            "sender_gpu_kinds": sorted(p.stem.split("-", 1)[1] for p in (REPO / "lb/hw_sender").glob("gpu-*.yml")),
            "power_clock_caps": list(ast.literal_eval(re.search(
                r"^CLOCK_CAPS_MHZ = (\(.*?\))", (REPO / "power/fleet_power/app.py").read_text(), re.M).group(1))),
        }
        self.assertEqual(set(expect), set(render.SOURCES), "every library source is checked here")
        for name, want in expect.items():
            with self.subTest(source=name):
                self.assertEqual(render.SOURCES[name](REPO), want)

    def test_every_source_reads_without_the_lbs_code(self):
        """python:3.12-slim has no PyYAML, so the LB's library code does not import there; the renderer reads every
        library file itself (miniyaml, JSON, a module's literal constants), so --check runs anywhere."""
        for name, read in render.SOURCES.items():
            with self.subTest(source=name):
                got = read(REPO)
                self.assertIsInstance(got, list)
                self.assertTrue(got, f"{name} read nothing")

    @needs_lb_loader
    def test_the_standalone_node_overrides_are_the_engines_and_runtimes_the_lb_names(self):
        from vllm_lb import catalog
        self.assertEqual(set(render.SOURCES["node_engine_overrides"](REPO)), set(catalog.NODE_ENGINES))
        self.assertEqual(set(render.SOURCES["node_gpu_overrides"](REPO)), set(render.SOURCES["gpu_runtimes"](REPO)))

    def test_every_engine_a_model_runs_on_is_in_the_engine_library(self):
        engines = set(render.SOURCES["engines"](REPO))
        self.assertTrue(set(render.SOURCES["node_engines"](REPO)) <= engines)
        self.assertTrue(set(render.SOURCES["cluster_engines"](REPO)) <= engines)

    def test_every_choices_from_names_a_source(self):
        for q in self.lib.questions:
            for src in [q.raw.get("choices_from")] + [f.get("choices_from") for f in q.raw.get("fields", {}).values()]:
                if src:
                    self.assertIn(src, render.SOURCES, q.id)


class Rendering(unittest.TestCase):
    """deploy/setup/render.py — a valid sheet renders the shipped examples; anything else is refused by name."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _sheet(self, edit) -> Path:
        doc = miniyaml.parse(PAIR.read_text(encoding="utf-8"), PAIR.name)
        edit(doc)
        p = self.tmp / "answers.yml"
        p.write_text(miniyaml.dump(doc), encoding="utf-8")
        return p

    def test_the_pair_sheet_renders_lb_fleet_example_as_data(self):
        r = _render(PAIR, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = miniyaml.parse((self.tmp / "out/lb/fleet.yml").read_text(), "rendered")
        want = miniyaml.parse((REPO / "lb/fleet.example.yml").read_text(), "fleet.example.yml")
        self.assertEqual(got, want)

    def test_the_pair_sheet_renders_lb_env_with_the_required_keys_and_documented_ones_only(self):
        r = _render(PAIR, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        env = render.read_env(self.tmp / "out/lb/.env")
        self.assertTrue(set(_env_keys(REPO / "lb/.env.example", documented=False)) <= set(env))
        self.assertTrue(set(env) <= set(_env_keys(REPO / "lb/.env.example", documented=True)) | {"COMPOSE_FILE"})
        self.assertTrue(env["SEARXNG_SECRET"])
        self.assertIn("MODELS_SEARCH_DIRS", env)

    def test_the_pair_sheet_renders_power_env_as_the_power_service_reads_the_example(self):
        r = _render(PAIR, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        sys.path.insert(0, str(REPO / "power"))
        try:
            from fleet_power.boxes import boxes_from_json
        finally:
            sys.path.remove(str(REPO / "power"))
        got = render.read_env(self.tmp / "out/power/.env")
        want = render.read_env(REPO / "power/.env.example")
        self.assertEqual(boxes_from_json(got["POWER_BOXES"]), boxes_from_json(want["POWER_BOXES"]))
        self.assertEqual({k: got[k] for k in ("POWER_ORIGINS", "POWER_DRY_RUN")},
                         {k: want[k] for k in ("POWER_ORIGINS", "POWER_DRY_RUN")})

    def test_rendered_env_files_are_owner_only_and_the_summary_masks_every_secret(self):
        r = _render(PAIR, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.tmp / "out/lb/.env").stat().st_mode & 0o777, 0o600)
        secrets = [v for q in _questions().questions if q.kind == "secret"
                   for v in [miniyaml.parse(PAIR.read_text(), "p").get(q.id)] if isinstance(v, str)]
        self.assertTrue(secrets)
        for s in secrets:
            self.assertNotIn(s, r.stdout + r.stderr)

    def test_check_only_writes_nothing(self):
        r = _render(PAIR, self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.tmp / "out").exists())

    def test_an_existing_file_is_never_overwritten(self):
        out = self.tmp / "out"
        (out / "lb").mkdir(parents=True)
        (out / "lb/.env").write_text("SEARXNG_SECRET=mine\n")
        r = _render(PAIR, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("lb/.env", r.stderr)
        self.assertEqual((out / "lb/.env").read_text(), "SEARXNG_SECRET=mine\n")

    def test_a_missing_required_answer_is_named(self):
        r = _render(self._sheet(lambda d: d.pop("search.searxng_secret")), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("search.searxng_secret", r.stderr)
        self.assertFalse((self.tmp / "out").exists())

    def test_a_missing_per_box_answer_names_the_box(self):
        r = _render(self._sheet(lambda d: d["box.platform"].pop("box-b")), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("box.platform", r.stderr)
        self.assertIn("box-b", r.stderr)

    def test_an_unknown_id_is_refused(self):
        r = _render(self._sheet(lambda d: d.update({"lb.context_length": 131072})), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("lb.context_length", r.stderr)

    def test_a_choice_outside_its_library_is_refused_naming_the_library(self):
        def edit(d):
            d["box.cards"]["box-a"][0]["type"] = "not-a-card"
        r = _render(self._sheet(edit), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not-a-card", r.stderr)
        self.assertIn("gb10", r.stderr)

    def test_an_answer_to_a_question_its_when_skips_is_refused(self):
        r = _render(self._sheet(lambda d: d.update({"broker.checkout": "/srv/broker_parrot"})), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("broker.checkout", r.stderr)
        self.assertIn("broker.enabled", r.stderr)

    def test_a_tuning_key_must_be_documented_and_not_retired(self):
        for key, why in (("LB_CONTEXT_LENGTH", "RETIRED"), ("LB_NOT_A_KNOB", "lb/.env.example")):
            with self.subTest(key=key):
                r = _render(self._sheet(lambda d: d.update({"lb.tuning": {key: "1"}})), self.tmp / key)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(key, r.stderr)
                self.assertIn(why, r.stderr)

    def test_exposure_beyond_localhost_without_auth_is_refused(self):
        def edit(d):
            d["exposure.scope"] = "lan"
        r = _render(self._sheet(edit), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("exposure.auth", r.stderr)

    def test_each_staged_model_gets_its_checkpoint_folders_and_only_the_parts_its_row_has(self):
        """`checkpoints.<id>` (where THIS deployment staged a model) is asked per staged model: its folder always; its
        uncensored twin only where the library row has one; its drafter only where the recipe drafts with a separate
        checkpoint — and there it is required (the catalog refuses the model without it)."""
        def twin_on_a_model_without(d):
            d["checkpoint.uncensored"]["mai-ui-2b"] = "MAI-UI-2B_uncensored"
        r = _render(self._sheet(twin_on_a_model_without), self.tmp / "a", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("checkpoint.uncensored [mai-ui-2b]: answered, but not asked", r.stderr)
        r = _render(self._sheet(lambda d: d["checkpoint.draft"].pop("qwen3.8-27b-dflash2")), self.tmp / "b", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("checkpoint.draft [qwen3.8-27b-dflash2]: missing", r.stderr)
        r = _render(self._sheet(lambda d: d["checkpoint.path"].pop("mai-ui-2b")), self.tmp / "c", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("checkpoint.path [mai-ui-2b]: missing", r.stderr)

    def test_a_checkpoint_folder_is_relative_to_the_models_roots(self):
        def absolute(d):
            d["checkpoint.path"]["mai-ui-2b"] = "/srv/models/MAI-UI-2B"
        r = _render(self._sheet(absolute), self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("checkpoint.path [mai-ui-2b]", r.stderr)

    def test_a_staged_model_must_be_in_the_model_library(self):
        r = _render(self._sheet(lambda d: d["models.staged"].append("no-such-model")), self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no-such-model", r.stderr)
        self.assertIn("library_models", r.stderr)

    # ── reference blueprints: the agent's starting point, never a fleet key (operator, 2026-10-30) ─────────────────

    def test_a_cluster_is_asked_whether_to_start_from_a_reference_blueprint_which_writes_nothing(self):
        """`cluster.blueprint` chooses among the shipped reference setups (library/blueprints, read — never copied) and
        writes NO file key: it only prefills. The blueprint sub-wizard and a fleet's own `blueprints` are gone, and no
        question writes a key the fleet loader refuses as retired."""
        lib = _questions()
        q = lib.by_id["cluster.blueprint"]
        self.assertEqual((q.raw["choices_from"], q.raw["per"], q.required), ("blueprints", "cluster", False))
        self.assertNotIn("also_from", q.raw)
        self.assertEqual(q.writes(), [], "it prefills answers; the rendered fleet never names a blueprint")
        self.assertEqual(set(render.SOURCES["blueprints"](REPO)), {PAIR_BP, NODE_BP})
        self.assertFalse([x.id for x in lib.questions if x.id.startswith("blueprint") and x.id != "cluster.blueprint"])
        self.assertNotIn("blueprint", [s["id"] for s in lib.sections])
        self.assertNotIn("blueprint", lib.scopes)
        retired = render.retired_fleet_keys(REPO)
        self.assertEqual(retired, {"blueprints", "blueprint"})
        hit = [k for f, k in lib.all_writes() if f == "lb/fleet.yml" and set(k.split(".")) & retired]
        self.assertFalse(hit, f"a question writes a retired fleet key: {hit}")
        ids = [x.id for x in lib.questions]
        self.assertLess(ids.index("cluster.ranks"), ids.index("cluster.blueprint"), "it fits the ranks' cards")
        for p in [x.id for x in lib.questions if "prefill" in x.raw and x.per == "cluster"]:
            self.assertGreater(ids.index(p), ids.index("cluster.blueprint"), f"{p} is prefilled: asked after it")

    def test_every_value_of_every_reference_blueprint_is_prefilled_by_some_question(self):
        """DRY coverage: a blueprint value no question proposes would be silently lost on the way to a deployment."""
        lib = _questions()
        paths = {x.raw["prefill"] for x in lib.questions if "prefill" in x.raw}
        for bid, bp in render.read_blueprints(REPO).items():
            with self.subTest(bp=bid):
                for section in ("knobs", "fixed"):
                    for k in bp.get(section) or {}:
                        self.assertTrue(f"{section}.{k}" in paths or section in paths, f"{bid}: {section}.{k}")
                self.assertIn("ref", paths)

    def test_a_blueprint_for_other_hardware_is_refused_naming_why(self):
        for bid, why in ((NODE_BP, "it is a control-local setup; this cluster is control ssh"),
                         ("no-such-blueprint", "no-such-blueprint")):
            with self.subTest(bid=bid):
                r = _render(self._sheet(lambda d: d.__setitem__("cluster.blueprint", {"pair": bid})),
                            self.tmp / "out", "--check")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(why, r.stderr)
        sheet = self._sheet(lambda d: d["box.cards"].__setitem__("box-b", [{"index": 0, "type": "rtx5090",
                                                                             "serves": True}]))
        fits = render.fitting_blueprints(REPO, miniyaml.parse(sheet.read_text(), "s"), "pair")
        self.assertEqual(fits[PAIR_BP], "it is for nvidia-sm121; this cluster's cards are nvidia-sm120, nvidia-sm121")

    def test_prefill_lists_what_fits_and_proposes_the_pairs_answers_beside_the_sheets(self):
        sheet = self._sheet(lambda d: d.__setitem__("cluster.blueprint", {"pair": PAIR_BP}))
        r = _render(sheet, self.tmp / "unused", "--prefill")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        self.assertIn(f"pair: {PAIR_BP} — FITS", out)
        self.assertIn(f"pair: {NODE_BP} — not: it is a control-local setup", out)
        self.assertIn('cluster.first_model: {pair: "qwen3.8-flash-next-hibrid48@uncensored"}    # now '
                      '"qwen3.8-27b-nvidia-nvfp4"', out)
        self.assertIn('cluster.kv_cache_dtypes: {pair: ["fp8_e4m3"]}    # unanswered', out)
        self.assertIn('"max_num_seqs": 12, "max_model_len": 131072', out)
        self.assertNotIn("cluster.first_start:", out, "measured on 2× GB10: the estimate is not asked")
        self.assertNotIn("models.staged", out, "already staged: nothing to confirm")
        self.assertFalse((self.tmp / "unused").exists(), "--prefill writes nothing")
        lib = _questions()
        pre = render.prefill(lib, miniyaml.parse(sheet.read_text(), "s"))
        self.assertEqual(pre.answers["cluster.first_start_settings"]["pair"],
                         render.read_blueprints(REPO)[PAIR_BP]["knobs"])

    def test_prefill_proposes_a_nodes_launch_block_its_boxs_ram_tier_and_its_staged_checkpoint(self):
        doc = miniyaml.parse(STANDALONE.read_text(encoding="utf-8"), STANDALONE.name)
        doc["cluster.blueprint"] = {"node": NODE_BP}
        doc["models.staged"] = ["mai-ui-2b"]
        pre = render.prefill(_questions(), doc)
        a = pre.answers
        self.assertEqual(a["cluster.model"], {"node": "qwen3.8-27b-nvidia-nvfp4@uncensored"})
        self.assertEqual(a["cluster.launch_choices"], {"node": ["qwen3.8-27b-nvidia-nvfp4@uncensored"]})
        self.assertEqual((a["cluster.max_num_seqs"], a["cluster.max_model_len"], a["cluster.gpu_memory_utilization"]),
                         ({"node": 32}, {"node": 98304}, {"node": 0.98}))
        self.assertEqual((a["cluster.kv_cache_memory"], a["cluster.kv_offload_ram_gib"], a["cluster.kv_cache_dtype"]),
                         ({"node": 8501810176}, {"node": 32}, {"node": "fp8_e4m3"}))
        self.assertNotIn("cluster.node_max_num_batched_tokens", a, "null in the blueprint: vLLM's own, not proposed")
        self.assertNotIn("cluster.reasoning_parser", a, "the sheet already says qwen3")
        self.assertEqual(a["box.kv_offload_ram_max_gib"], {"box-s": 32}, "the RAM tier the launch needs, on its box")
        self.assertEqual(a["models.staged"], ["mai-ui-2b", "qwen3.8-27b-nvidia-nvfp4"], "added, never replaced")
        self.assertNotIn("cluster.kv_cache_dtypes", a, "a pair's question: not asked of a node")
        doc["box.kv_offload_ram_max_gib"] = {"box-s": 48}
        self.assertNotIn("box.kv_offload_ram_max_gib", render.prefill(_questions(), doc).answers, "never lowered")

    def test_the_rendered_fleet_holds_the_prefilled_settings_never_the_blueprint(self):
        sheet = self._sheet(lambda d: d.__setitem__("cluster.blueprint", {"pair": PAIR_BP}))
        answers = miniyaml.parse(sheet.read_text(), "s")
        sheet.write_text(miniyaml.dump(render.merged(answers, render.prefill(_questions(), answers).answers)))
        r = _render(sheet, self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = _render(sheet, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (self.tmp / "out/lb/fleet.yml").read_text()
        self.assertNotIn("blueprint", text)
        pair = miniyaml.parse(text, "rendered")["clusters"]["pair"]
        self.assertEqual((pair["choices"], pair["kv_cache_dtypes"]),
                         (["qwen3.8-flash-next-hibrid48@uncensored"], ["fp8_e4m3"]))

    def test_first_start_settings_name_only_the_knobs_of_a_cluster_start(self):
        self.assertEqual(render.cluster_knobs(REPO)[:2], ["max_num_seqs", "max_model_len"])
        r = _render(self._sheet(lambda d: d.__setitem__("cluster.first_start_settings",
                                                        {"pair": {"max_num_seqs": 8, "gpu_memory_utilization": 0.9}})),
                    self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cluster.first_start_settings [pair]: gpu_memory_utilization is no knob of a cluster Start",
                      r.stderr)

    def test_an_ssh_clusters_link_is_asked(self):
        r = _render(self._sheet(lambda d: d.pop("cluster.link")), self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cluster.link [pair]: missing", r.stderr)

    def test_each_cards_accelerator_is_shown_not_asked(self):
        """The accelerator is the card type's (hardware/types/<key>.yml `accelerator`): the summary shows it beside the
        card, and no answer sets it."""
        r = _render(PAIR, self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("box-a card 0: gb10 → accelerator nvidia-sm121 (sm_121, nvidia)", r.stdout)
        self.assertIn("box-c card 0: radeon8060s → accelerator amd-gfx1151 (gfx1151, rocm)", r.stdout)
        self.assertFalse([q.id for q in _questions().questions if q.id.startswith("box.accelerator")])

    def test_a_serving_cards_runtime_must_be_the_boxs_driver_stack(self):
        def edit(d):
            d["box.gpu_runtime"]["box-a"] = "rocm"
            d["box.container_toolkit"].pop("box-a")
        r = _render(self._sheet(edit), self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("box.gpu_runtime [box-a]", r.stderr)
        self.assertIn("nvidia-sm121", r.stderr)

    def test_the_first_starts_sizing_is_asked_only_where_the_model_is_unmeasured(self):
        """Unmeasured on the pair's engine × accelerator × TP, a Start is refused unless the operator chooses the
        estimate, explicitly (docs/design/engines.md) — so the sheet must say which; measured, it is not asked; any
        other refusal (no image, an engine that does not serve it) is the first model's, named."""
        unmeasured = {"refused": "not measured on nvidia-sm120 at TP=2 with vllm: …", "sizing": "unmeasured"}
        measured = {"refused": "", "sizing": "measured"}
        no_image = {"refused": "no image of vllm build fleet for nvidia-sm120 × linux/amd64", "sizing": ""}
        self.assertIn("cluster.first_start [pair]: missing", render.sizing_problem("pair", "m", unmeasured, None))
        self.assertIn("not measured on nvidia-sm120", render.sizing_problem("pair", "m", unmeasured, None))
        self.assertEqual(render.sizing_problem("pair", "m", unmeasured, "measure-first"), "")
        self.assertEqual(render.sizing_problem("pair", "m", unmeasured, "start-on-estimate"), "")
        self.assertEqual(render.sizing_problem("pair", "m", measured, None), "")
        self.assertIn("cluster.first_start [pair]: answered, but m is measured",
                      render.sizing_problem("pair", "m", measured, "start-on-estimate"))
        self.assertIn("no image of vllm build fleet", render.sizing_problem("pair", "m", no_image, None))
        self.assertIn("cluster.first_model [pair]", render.sizing_problem("pair", "m", no_image, None))

    def test_the_standalone_sheet_needs_the_lb_image_for_its_node_lines_and_says_so(self):
        if HAS_YAML:
            self.skipTest("this image has PyYAML: the node lines render (StandaloneInLbImage)")
        r = _render(STANDALONE, self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("node-env", r.stderr)
        self.assertIn("vllm-lb", r.stderr)
        r = _render(STANDALONE, self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)


class Docs(unittest.TestCase):
    """SETUP.md is the protocol the library serves: it names every section, the phases and the renderer; the README's
    coding-agent prompt and both agent guides send an agent to it."""

    def test_setup_md_names_every_section_and_phase(self):
        text = (REPO / "SETUP.md").read_text(encoding="utf-8")
        for s in _questions().sections:
            self.assertIn(f"`{s['id']}`", text, f"SETUP.md does not name section {s['id']}")
        for phase in ("Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Open items"):
            self.assertIn(phase, text)
        for needed in ("deploy/setup/questions.yml", "deploy/setup/render.py", "deploy/setup/answers.yml", "--check",
                       "--diff", "--replace"):
            self.assertIn(needed, text)

    def test_the_readme_prompt_and_the_agent_guides_point_at_the_wizard(self):
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        prompt = readme.split("## Hand it to a coding agent", 1)[1].split("\n## ", 1)[0]
        self.assertIn("SETUP.md", prompt)
        self.assertNotIn("[e.g.", prompt, "the prompt asks the user nothing itself: the wizard does")
        self.assertIn("SETUP.md", readme.split("## Contents", 1)[1].split("\n## ", 1)[0])
        for guide in ("AGENTS.md", "CLAUDE.md"):
            shipped = (REPO / guide).read_text(encoding="utf-8").split("<!-- PRIVATE-BELOW", 1)[0]
            self.assertIn("SETUP.md", shipped, f"{guide} (above its private marker) must point at the wizard")

    def test_the_rendered_files_for_other_machines_are_never_tracked(self):
        self.assertIn("/hosts/", (REPO / ".gitignore").read_text(encoding="utf-8").splitlines())


@needs_lb_loader
class InLbImage(unittest.TestCase):
    """The LB's own loader accepts what the renderer writes — run inside the vllm-lb image."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _load(self, out: Path, roots: str):
        from vllm_lb.fleet import load_fleet
        old = os.environ.get("MODELS_SEARCH_DIRS")
        os.environ["MODELS_SEARCH_DIRS"] = roots
        try:
            return load_fleet(out / "lb/fleet.yml")      # its catalog: the library resolved against THIS fleet
        finally:
            os.environ.pop("MODELS_SEARCH_DIRS") if old is None else os.environ.__setitem__("MODELS_SEARCH_DIRS", old)

    def test_the_pair_sheets_fleet_loads(self):
        r = _render(PAIR, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        fleet = self._load(self.tmp / "out", render.read_env(self.tmp / "out/lb/.env")["MODELS_SEARCH_DIRS"])
        self.assertEqual(list(fleet.clusters), ["pair"])

    def test_the_pair_sheets_first_start_resolves_to_the_image_matrixs_image(self):
        """The first Start's image is the matrix's (library/images) for its recipe × the pair's accelerator × platform,
        read by the LB's own resolver — shown, never asked."""
        r = _render(PAIR, self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("cluster pair: first Start qwen3.8-27b-nvidia-nvfp4 → vllm/fleet on nvidia-sm121 × linux/arm64 "
                      "× TP=2 → myllmbox/qwen38-flash-next-cluster-vllm:v5 (measured)", r.stdout)

    def test_choosing_the_estimate_for_a_measured_first_start_is_refused(self):
        doc = miniyaml.parse(PAIR.read_text(encoding="utf-8"), PAIR.name)
        doc["cluster.first_start"] = {"pair": "start-on-estimate"}
        sheet = self.tmp / "answers.yml"
        sheet.write_text(miniyaml.dump(doc), encoding="utf-8")
        r = _render(sheet, self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cluster.first_start [pair]: answered, but qwen3.8-27b-nvidia-nvfp4 is measured", r.stderr)

    def _prefilled(self, base: Path, cluster: str, bid: str, edit=None) -> Path:
        """`base` with `cluster.blueprint` answered and every proposal confirmed as it is — what the sheet holds after
        the user said yes to each."""
        doc = miniyaml.parse(base.read_text(encoding="utf-8"), base.name)
        doc["cluster.blueprint"] = {cluster: bid}
        doc = render.merged(doc, render.prefill(_questions(), doc).answers)
        if edit:
            edit(doc)
        sheet = self.tmp / "answers.yml"
        sheet.write_text(miniyaml.dump(doc), encoding="utf-8")
        return sheet

    def test_the_pair_blueprints_prefilled_first_start_passes_the_switchs_own_validator(self):
        r = _render(self._prefilled(PAIR, "pair", PAIR_BP), self.tmp / "out", "--check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("cluster pair: first Start qwen3.8-flash-next-hibrid48@uncensored → vllm/fleet on nvidia-sm121",
                      r.stdout)
        self.assertIn("read and accepted by the gateway's own loader", r.stdout)

    def test_a_first_start_the_switch_would_refuse_is_refused_at_check_in_its_words(self):
        sheet = self._prefilled(PAIR, "pair", PAIR_BP,
                                lambda d: d["cluster.first_start_settings"]["pair"].__setitem__("max_num_seqs", 49))
        r = _render(sheet, self.tmp / "out", "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cluster.first_start_settings [pair]: the first Start of qwen3.8-flash-next-hibrid48@uncensored "
                      "would be refused", r.stderr)

    def test_the_node_blueprints_prefilled_sheet_renders_a_fleet_the_loader_and_node_env_accept(self):
        # The image is the deployment's answer, never a blueprint's: a KV dtype is taken only on an image
        # kv_cache_dtypes.json records (the example's placeholder digest is none), so the user pins the recorded build.
        r = _render(self._prefilled(STANDALONE, "node", NODE_BP), self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no record of which KV cache dtypes", r.stderr)
        sheet = self._prefilled(STANDALONE, "node", NODE_BP, lambda d: d.__setitem__("cluster.image", {"node": (
            "sha256:7b35cbcf2a880e0e61233848f038417a04b1a94950ac691ae901948abff7f360")}))
        r = _render(sheet, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        server = next((self.tmp / "out/deploy/standalone").glob("*.env"))
        fleet = self._load(self.tmp / "out", render.read_env(server)["EDGE_MODELS_DIR"])
        node = fleet.clusters["node"]
        self.assertEqual((node.launch.max_num_seqs, node.launch.max_model_len, node.launch.kv_offload_ram_gib),
                         (32, 98304, 32))
        self.assertEqual(fleet.boxes["box-s"].kv_offload_ram_max_gib, 32)
        self.assertNotIn("blueprint", (self.tmp / "out/lb/fleet.yml").read_text())

    def test_the_blueprints_read_by_the_wizard_are_the_library_checks_own(self):
        """render.py reads library/blueprints with miniyaml (python:3.12-slim); the library check (vllm_lb.blueprints,
        PyYAML) reads the same documents — and every one passes it."""
        from vllm_lb import blueprints, yamlfile
        lib = blueprints.read_library()
        read = render.read_blueprints(REPO)
        self.assertEqual(set(read), set(lib))
        for bid, doc in read.items():
            self.assertEqual(doc, yamlfile.read(blueprints.BLUEPRINTS_DIR / f"{bid}.yml"), bid)

    def test_the_standalone_sheet_renders_a_server_file_up_sh_accepts_for_its_node(self):
        r = _render(STANDALONE, self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        server = next((self.tmp / "out/deploy/standalone").glob("*.env"))
        env = render.read_env(server)
        fleet = self._load(self.tmp / "out", env["EDGE_MODELS_DIR"])
        self.assertEqual([c.control for c in fleet.clusters.values()], ["local"])
        for k in ("EDGE_NODE_IMAGE", "EDGE_NODE_GPU_RUNTIME", "EDGE_NODE_ENGINE", "EDGE_NODE_MODEL", "EDGE_BOX"):
            self.assertIn(k, env)
        check = subprocess.run(["bash", str(REPO / "deploy/standalone/up.sh"), "--check", "--env", str(server), "node"],
                               capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
