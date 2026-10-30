#!/usr/bin/env python3
"""THE SETUP WIZARD'S RENDERER — an answer sheet validated against the question library, then written as the files the
services read (SETUP.md Phase 2 and Phase 3).

    python3 deploy/setup/render.py <answers.yml> --out <dir> --check     # validate; print the sheet (secrets masked)
                                                                         # and the files + keys it would write
    python3 deploy/setup/render.py <answers.yml> --out <dir>             # write them under <dir> (refuses to overwrite)
    python3 deploy/setup/render.py <answers.yml> --out <dir> --diff      # what would change in files that exist
    python3 deploy/setup/render.py <answers.yml> --out <dir> --replace lb/.env   # after the user said yes to that diff
    python3 deploy/setup/render.py <answers.yml> --prefill    # the reference blueprints that fit each cluster, and
                                                              # what an answered cluster.blueprint proposes

WHY a renderer: the WRITE phase must be deterministic, not hand-typed — the same sheet always gives the same files, and
a value lives in ONE answer even where two files carry it (telemetry's port is the gateway's LB_HW_PORT and every
sender's HW_SENDER_PORT). And the sheet is REFUSED, naming the question, on a required answer missing, an id the
library does not have, a choice outside its library, or an answer to a question its `when` skips — no default ever
fills a gap (AGENTS.md: no defaults, no auto-detection, no fallbacks).

Runs on the standard library (python:3.12-slim) with deploy/setup/miniyaml.py. Two steps need the gateway's own code,
which needs PyYAML (the vllm-lb image): a node's server-file lines are `python -m vllm_lb.fleet node-env`'s, and the
rendered lb/fleet.yml is read by the LB's loader (vllm_lb.fleet.load_fleet) before anything is written. Where that code
cannot be imported, the node lines are REFUSED naming the command that runs this in the vllm-lb image, and the
summary says the fleet file was not read by the loader here — never a guessed line.

What the libraries DERIVE from the answers is shown, never asked (the engines × accelerators redesign,
docs/design/engines.md): each card's accelerator (its type's), and — with the gateway's code — each ssh cluster's first
Start as its resolver reads it (engine/build, accelerator × platform × TP, the image matrix's image, measured or not).
Where that first model is unmeasured, `cluster.first_start` must say measure-first or start-on-estimate; where it is
measured, an answer is refused. The renderer reads every library file itself (miniyaml, JSON, a module's literal
constants via `ast`): the LB's library modules import PyYAML, and --check must run without it.

REFERENCE BLUEPRINTS (lb/vllm_lb/library/blueprints; operator, 2026-10-30: "Blueprints meant to be only for agent to
build their own fleet settings, not something user chooses"): `cluster.blueprint` names the proven setup a cluster's
answers START from. `--prefill` lists the ones that fit each cluster's answered control and cards and prints what the
chosen one proposes for each question carrying a `prefill` — proposals the agent confirms with the user, exactly like a
probe's. Rendering reads only a blueprint's target (does it fit the cluster?), never its values: it renders the
answers, so a fleet carries plain settings and never a blueprint id (the LB refuses one at boot).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import fnmatch
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import miniyaml  # noqa: E402

REPO = HERE.parents[1]
#: What each kind of answer is (`_check`).
KINDS = {
    "choice": "one of its choices", "multi": "a list of its choices, each once", "text": "a non-empty string",
    "path": "an absolute path or ~/…", "host": "a host name or address", "bind": "host:port", "url": "an http(s) URL",
    "port": "a port 1-65535", "secret": "a non-empty string, never printed", "number": "a number", "bool": "true or false",
    "list": "a non-empty list, each once", "records": "a list of records with the named fields",
    "map": "a non-empty mapping", "fact": "{value, source}", "image": "a digest-pinned image, or {engine: image}",
}
MASK = "********"
#: A key path's separator while it is rendered: a key's dots, never an instance's (a model id holds dots).
SEP = "\x01"
_DIGEST = re.compile(r"(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}")
IN_LB_IMAGE = ('docker run --rm -v "$PWD":/repo -w /repo vllm-lb:dev python deploy/setup/render.py <answers.yml> '
               '--out <dir>')


class Refused(ValueError):
    """The sheet or the library cannot be rendered; the message names every problem."""


# ── the libraries a choice is read from: each READ, never copied (deploy/tests/test_setup.py `Libraries`) ─────────────

def _lb(repo: Path):
    if str(repo / "lb") not in sys.path:
        sys.path.insert(0, str(repo / "lb"))


def _const(path: Path, name: str) -> list:
    """The module-level tuple `name` in `path`, read with `ast` — never imported: the LB's library modules import
    PyYAML, which python:3.12-slim lacks. Its items are literals or names of module-level string constants (catalog.py
    `NODE_ENGINES = (VLLM, "faster-whisper", "nemo", SGLANG)`); anything else is refused, naming the module, rather
    than guessed."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    names: dict[str, Any] = {}
    found = None
    for node in tree.body:
        target, value = (node.targets[0], node.value) if isinstance(node, ast.Assign) and len(node.targets) == 1 else \
            (node.target, node.value) if isinstance(node, ast.AnnAssign) and node.value is not None else (None, None)
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):      # VLLM, SGLANG = "vllm", "sglang"
            for t, v in zip(target.elts, value.elts):
                if isinstance(t, ast.Name) and isinstance(v, ast.Constant):
                    names[t.id] = v.value
        elif isinstance(target, ast.Name):
            if isinstance(value, ast.Constant):
                names[target.id] = value.value
            if target.id == name:
                found = value
    if not isinstance(found, (ast.Tuple, ast.List)):
        raise Refused(f"{path.name}: no module-level tuple {name} to read a library from")
    out = []
    for e in found.elts:
        if isinstance(e, ast.Constant):
            out.append(e.value)
        elif isinstance(e, ast.Name) and e.id in names:
            out.append(names[e.id])
        else:
            raise Refused(f"{path.name}: {name} holds {ast.unparse(e)}, which is no literal or string constant")
    return out


def _json_rows(repo: Path, name: str, key: str) -> list:
    return json.loads((repo / "lb/vllm_lb/library" / name).read_text(encoding="utf-8"))[key]


def _yml(path: Path) -> dict:
    return miniyaml.parse(path.read_text(encoding="utf-8"), str(path))


def _stems(repo: Path, rel: str, glob: str, sep: str | None = None) -> list:
    return sorted(p.stem.split(sep, 1)[1] if sep else p.stem for p in (repo / rel).glob(glob))


def card_accelerators(repo: Path) -> dict[str, dict]:
    """Card type → its ACCELERATOR (hardware/types/<key>.yml `accelerator`) and that accelerator's arch and GPU runtime
    (hardware/accelerators/<key>.yml) — what a card's kernels target is the type's, so the wizard SHOWS it beside each
    card and never asks it (docs/design/engines.md)."""
    out = {}
    for p in sorted((repo / "lb/vllm_lb/hardware/types").glob("*.yml")):
        key = _yml(p)["accelerator"]
        facts = _yml(repo / "lb/vllm_lb/hardware/accelerators" / f"{key}.yml")["facts"]
        out[p.stem] = {"accelerator": key, "arch": facts["arch"]["value"], "gpu_runtime": facts["gpu_runtime"]["value"]}
    return out


#: THE LIBRARIES A CHOICE IS READ FROM, each read from the file the LB reads — never a copied list. Since the engines ×
#: accelerators redesign (docs/design/engines.md) the models are library/models.json (their facts), what launches
#: them library/recipes.json, the engines library/engines/<engine>.yml, the kernel targets hardware/accelerators/.
SOURCES: dict[str, Callable[[Path], list]] = {
    "hardware_types": lambda r: _stems(r, "lb/vllm_lb/hardware/types", "*.yml"),
    "accelerators": lambda r: _stems(r, "lb/vllm_lb/hardware/accelerators", "*.yml"),
    # The REFERENCE BLUEPRINTS (lb/vllm_lb/blueprints.py): one <id>.yml each — what `cluster.blueprint` may start the
    # cluster's answers from; one whose target is not the cluster's is refused (`blueprint_problem`).
    "blueprints": lambda r: _stems(r, "lb/vllm_lb/library/blueprints", "*.yml"),
    "library_models": lambda r: [m["id"] for m in _json_rows(r, "models.json", "models")],
    "model_refs": lambda r: [x for m in _json_rows(r, "models.json", "models")
                             for x in [m["id"]] + ([f"{m['id']}@uncensored"] if "uncensored" in m else [])],
    # A model whose row has an uncensored twin: where it is staged, the twin's folder is asked too (checkpoint.*).
    "uncensored_models": lambda r: [m["id"] for m in _json_rows(r, "models.json", "models") if "uncensored" in m],
    # A model whose recipe drafts with a SEPARATE checkpoint (DFlash2): the deployment names the drafter's folder.
    "draft_models": lambda r: [x["model"] for x in _json_rows(r, "recipes.json", "recipes")
                               if any(p.get("draft_checkpoint") for p in x["launch"].values())],
    "quant_methods": lambda r: sorted({m["quant_method"]["value"] for m in _json_rows(r, "models.json", "models")
                                       if m["quant_method"]["value"]}),
    "node_engines": lambda r: _const(r / "lb/vllm_lb/catalog.py", "NODE_ENGINES"),
    "cluster_engines": lambda r: _const(r / "lb/vllm_lb/catalog.py", "CLUSTER_ENGINES"),
    "engines": lambda r: _stems(r, "lb/vllm_lb/library/engines", "*.yml"),
    "tasks": lambda r: _const(r / "lb/vllm_lb/catalog.py", "TASKS"),
    "kv_dtypes": lambda r: list(json.loads((r / "lb/vllm_lb/kv_cache_dtypes.json").read_text())["layouts"]),
    "platforms": lambda r: _const(r / "lb/vllm_lb/hardware/facts.py", "PLATFORMS"),
    "gpu_runtimes": lambda r: _const(r / "lb/vllm_lb/hardware/facts.py", "GPU_RUNTIMES"),
    "memory_families": lambda r: re.findall(r'"(\w+)": \w+Profile',
                                            (r / "lb/vllm_lb/hardware/profiles.py").read_text(encoding="utf-8")),
    "controls": lambda r: re.findall(r'^    "(\w+)": Control\(', (r / "lb/vllm_lb/fleet.py").read_text(encoding="utf-8"),
                                     re.M),
    # How a cluster's GPUs are joined (fleet `clusters.<name>.link`, executors.py): across hosts, then within one.
    "links": lambda r: (_const(r / "lb/vllm_lb/executors.py", "LINKS_ACROSS_HOSTS")
                        + _const(r / "lb/vllm_lb/executors.py", "LINKS_ONE_HOST")),
    "node_gpu_overrides": lambda r: _stems(r, "deploy/standalone", "node-gpu.*.yml", "."),
    "node_engine_overrides": lambda r: _stems(r, "deploy/standalone", "node-engine.*.yml", "."),
    "sender_gpu_kinds": lambda r: _stems(r, "lb/hw_sender", "gpu-*.yml", "-"),
    "power_clock_caps": lambda r: _const(r / "power/fleet_power/app.py", "CLOCK_CAPS_MHZ"),
}


def retired_keys(repo: Path) -> set[str]:
    """Every env name lb/vllm_lb/config.py RETIRED_KEYS refuses at boot — read from its source (config.py imports the
    fleet loader, which needs PyYAML)."""
    src = (repo / "lb/vllm_lb/config.py").read_text(encoding="utf-8")
    block = src[src.index("RETIRED_KEYS = {"):src.index("class ClusterConfig")]
    return set(re.findall(r'"([A-Z][A-Z0-9_]+)"', block))


def retired_fleet_keys(repo: Path) -> set[str]:
    """Every fleet key lb/vllm_lb/fleet.py RETIRED_FLEET_KEYS / RETIRED_CLUSTER_KEYS refuses at boot — read from its
    source, as `retired_keys` reads config.py's."""
    src = (repo / "lb/vllm_lb/fleet.py").read_text(encoding="utf-8")
    return set(re.findall(r'RETIRED_(?:FLEET|CLUSTER)_KEYS = \{"([a-z_]+)"', src))


def fleet_keys(repo: Path) -> set[str]:
    """The keys lb/vllm_lb/fleet.py names: every quoted lowercase word in its source (its `_keys` tuples and the key
    constants are such words) — less the RETIRED ones it names only to refuse them."""
    words = set(re.findall(r'"([a-z_][a-z0-9_]*)"', (repo / "lb/vllm_lb/fleet.py").read_text(encoding="utf-8")))
    return words - retired_fleet_keys(repo)


# ── reference blueprints: what a cluster's answers may START from (lb/vllm_lb/blueprints.py) ─────────────────────────

#: Where the shipped reference setups sit, one `<id>.yml` each — read here with miniyaml (the library check that holds
#: each to the model library, the image matrix and the measurements is lb/tests/test_blueprints.py's).
BLUEPRINTS = "lb/vllm_lb/library/blueprints"


def read_blueprints(repo: Path = REPO) -> dict[str, dict]:
    """Every reference blueprint, id → its document."""
    return {p.stem: _yml(p) for p in sorted((repo / BLUEPRINTS).glob("*.yml"))}


def cluster_knobs(repo: Path = REPO) -> list[str]:
    """The knobs of a cluster Start (blueprints.py CLUSTER_KNOBS, read with `ast`): what `cluster.first_start_settings`
    may name."""
    return _const(repo / "lb/vllm_lb/blueprints.py", "CLUSTER_KNOBS")


def _cluster_cards(answers: dict, cluster: str) -> list[str]:
    """The card TYPES a cluster serves on, from the answers: each rank's box's serving cards (`box.cards`)."""
    ranks = (answers.get("cluster.ranks") or {}).get(cluster) or []
    cards = answers.get("box.cards") or {}
    return [c.get("type") for r in ranks if isinstance(r, dict) for c in (cards.get(r.get("box")) or [])
            if isinstance(c, dict) and c.get("serves") is True]


def _rank_boxes(answers: dict, cluster: str) -> list[str]:
    return [r["box"] for r in (answers.get("cluster.ranks") or {}).get(cluster) or [] if isinstance(r, dict) and "box" in r]


def blueprint_problem(bp: dict, control: Any, card_types: list[str], accs: dict[str, dict]) -> str:
    """Why blueprint `bp` is not for a cluster of `control` serving on `card_types` — "" when it is: the same control,
    TP (one serving card a rank), and every serving card's accelerator the blueprint's."""
    t = bp["target"]
    if control != t["control"]:
        return f"it is a control-{t['control']} setup; this cluster is control {control}"
    if len(card_types) != t["tp"]:
        return f"it is for TP={t['tp']}; this cluster serves on {len(card_types)} card(s)"
    have = sorted({accs[c]["accelerator"] if c in accs else f"{c} (a new type: its own)" for c in card_types})
    if have != [t["accelerator"]]:
        return f"it is for {t['accelerator']}; this cluster's cards are {', '.join(have) or 'none'}"
    return ""


def fitting_blueprints(repo: Path, answers: dict, cluster: str) -> dict[str, str]:
    """{id: "" where it fits the cluster's answered control and cards, else why not} for every reference blueprint."""
    accs = card_accelerators(repo)
    control = (answers.get("cluster.control") or {}).get(cluster)
    cards = _cluster_cards(answers, cluster)
    return {bid: blueprint_problem(bp, control, cards, accs) for bid, bp in read_blueprints(repo).items()}


def _bp_value(bp: dict, path: str) -> Any:
    """What blueprint `bp` says for a question's `prefill` path (None: nothing — a null knob is the engine's own)."""
    if path == "ref":
        return f"{bp['model']}@uncensored" if bp["uncensored"] else bp["model"]
    if path in ("model", "estimated", "knobs"):
        return dict(bp[path]) if path == "knobs" else bp[path]
    section, _, key = path.partition(".")
    if section not in ("knobs", "fixed") or not key:
        raise Refused(f"prefill {path!r}: no such blueprint value (ref, model, estimated, knobs, knobs.<k>, fixed.<k>)")
    return (bp.get(section) or {}).get(key)


def _proposed(q: "Question", bp: dict, now: Any) -> Any:
    """The answer question `q` is proposed from blueprint `bp`, beside what the sheet answers `now` — None: none."""
    v = _bp_value(bp, q.raw["prefill"])
    values = q.raw.get("prefill_values")
    if values is not None:
        v = values.get(str(v).lower() if isinstance(v, bool) else str(v))
    if v is None:
        return None
    how = q.raw.get("prefill_as")
    if how == "list":
        return [v]
    if how == "add":
        have = list(now) if isinstance(now, list) else []
        return have if v in have else have + [v]
    if how == "at_least":
        return v if not isinstance(now, (int, float)) or isinstance(now, bool) or now < v else now
    if how is not None:
        raise Refused(f"{q.id}: prefill_as {how!r} is none of list, add, at_least")
    return v


@dataclass
class Prefill:
    """What `cluster.blueprint` proposes: `answers` — question id → value (global) or {instance: value} — and `fits`,
    per cluster, every blueprint and why it does not fit ("" where it does)."""
    answers: dict
    fits: dict[str, dict[str, str]]


def prefill(lib: "Library", answers: dict) -> Prefill:
    """THE PROPOSALS of every answered `cluster.blueprint`, from a sheet that need not be complete yet (the agent runs it
    mid-Phase 1, right after the question): for each cluster, each question carrying a `prefill` whose `when` holds for
    that cluster (a per-box one: for each of its rank boxes; a global one: once). A blueprint that does not fit its
    cluster is Refused, naming why. The proposals are shown and confirmed, never written by this function."""
    out: dict = {}
    fits: dict[str, dict[str, str]] = {}
    clusters = answers.get("clusters.names") if isinstance(answers.get("clusters.names"), list) else []
    chosen = answers.get("cluster.blueprint") or {}
    library = read_blueprints(lib.repo)
    for cluster in clusters:
        fits[cluster] = fitting_blueprints(lib.repo, answers, cluster)
        bid = chosen.get(cluster) if isinstance(chosen, dict) else None
        if bid is None:
            continue
        if bid not in library:
            raise Refused(f"cluster.blueprint [{cluster}]: no blueprint {bid!r} in {BLUEPRINTS}")
        if fits[cluster][bid]:
            raise Refused(f"cluster.blueprint [{cluster}]: {bid} does not fit {cluster}: {fits[cluster][bid]}")
        bp = library[bid]
        for q in lib.questions:
            if "prefill" not in q.raw:
                continue
            insts = [cluster] if q.per == "cluster" else _rank_boxes(answers, cluster) if q.per == "box" else [None]
            for inst in insts:
                ctx = (q.per, inst) if q.per else None
                if q.when and not q.when.eval(lambda i, c: lib.look(i, c, answers, q.id), lib, answers, ctx):
                    continue
                given = answers.get(q.id)
                now = given.get(inst) if q.per and isinstance(given, dict) else (None if q.per else given)
                v = _proposed(q, bp, (out.get(q.id, {}).get(inst) if q.per else out.get(q.id)) or now)
                if v is None or v == now:
                    continue                  # nothing proposed, or nothing to confirm: the sheet already says it
                if q.per:
                    out.setdefault(q.id, {})[inst] = v
                else:
                    out[q.id] = v
    return Prefill(out, fits)


def merged(answers: dict, proposals: dict) -> dict:
    """`answers` with `proposals` (a Prefill's answers) written over them — what the sheet holds once the user said yes
    to every proposal."""
    out = dict(answers)
    for qid, v in proposals.items():
        out[qid] = {**(out.get(qid) or {}), **v} if isinstance(v, dict) and isinstance(out.get(qid), dict) else v
    return out


def node_env_keys(repo: Path = REPO) -> set[str]:
    """The EDGE_NODE_* lines `python -m vllm_lb.fleet node-env` can render (named in fleet.py)."""
    return set(re.findall(r"EDGE_NODE_[A-Z][A-Z0-9_]*[A-Z0-9]", (repo / "lb/vllm_lb/fleet.py").read_text(encoding="utf-8")))


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of an env file (comments and blanks skipped; one pair of surrounding quotes removed)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k.strip()] = v
    return out


# ── `when`: a small expression over earlier answers ───────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"\s*(any\(|len\(|not\b|\(|\)|\[|\]|,|==|!=|>=|<=|>|<|[^\s()\[\],=!<>]+)")
_UNSET = object()


def _literal(tok: str) -> Any:
    v = miniyaml._scalar(tok)
    return tok if v is None else v


class When:
    """`id == v`, `id != v`, `id in [a, b]`, `id in @source`, `id has v`, `len(id) > n`, a bare `id`, `not`, `and`,
    `or`, parentheses and `any(<expr>)` over a scope's instances — parsed once, evaluated on the effective answers.
    Inside a `per` question the scope's own name is the instance (`checkpoint in @draft_models`)."""

    def __init__(self, text: str, where: str):
        self.text, self.where = text, where
        self.toks = [m.group(1) for m in _TOKEN.finditer(text)]
        if "".join(self.toks).replace(" ", "") != text.replace(" ", ""):
            raise Refused(f"{where}: cannot read when {text!r}")
        self.i = 0
        self.tree = self._or()
        if self.i != len(self.toks):
            raise Refused(f"{where}: unexpected {self.toks[self.i]!r} in when {text!r}")

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _take(self, want=None):
        tok = self._peek()
        if tok is None or (want is not None and tok != want):
            raise Refused(f"{self.where}: when {self.text!r} expected {want or 'more'}")
        self.i += 1
        return tok

    def _or(self):
        node = self._and()
        while self._peek() == "or":
            self._take()
            node = ("or", node, self._and())
        return node

    def _and(self):
        node = self._not()
        while self._peek() == "and":
            self._take()
            node = ("and", node, self._not())
        return node

    def _not(self):
        if self._peek() == "not":
            self._take()
            return ("not", self._not())
        return self._atom()

    def _atom(self):
        tok = self._take()
        if tok == "(":
            node = self._or()
            self._take(")")
            return node
        if tok == "any(":
            node = self._or()
            self._take(")")
            return ("any", node)
        if tok == "len(":
            ident = self._take()
            self._take(")")
            op = self._take()
            return ("len", ident, op, _literal(self._take()))
        op = self._peek()
        if op in ("==", "!=", ">", "<", ">=", "<=", "has"):
            self._take()
            return ("cmp", tok, op, _literal(self._take()))
        if op == "in" and (self.toks[self.i + 1:self.i + 2] or [""])[0].startswith("@"):
            # `<id> in @<source>`: the answer (or, for a scope's own name, the instance) is in a LIBRARY — e.g. a
            # staged model's uncensored twin is asked only where its library row has one (`checkpoint in
            # @uncensored_models`). Read from the library, never a list typed here.
            self._take()
            return ("in_source", tok, self._take()[1:])
        if op == "in":
            self._take()
            self._take("[")
            vals = []
            while self._peek() != "]":
                vals.append(_literal(self._take()))
                if self._peek() == ",":
                    self._take()
            self._take("]")
            return ("in", tok, vals)
        return ("truthy", tok)

    def refs(self) -> list[str]:
        out = []

        def walk(n):
            if n[0] in ("or", "and"):
                walk(n[1]), walk(n[2])
            elif n[0] in ("not", "any"):
                walk(n[1])
            else:
                out.append(n[1])
        walk(self.tree)
        return out

    def eval(self, look: Callable[[str, Any], Any], lib: "Library", eff: dict, inst: tuple | None) -> bool:
        def ev(n, ctx):
            kind = n[0]
            if kind == "or":
                return ev(n[1], ctx) or ev(n[2], ctx)
            if kind == "and":
                return ev(n[1], ctx) and ev(n[2], ctx)
            if kind == "not":
                return not ev(n[1], ctx)
            if kind == "any":
                scopes = {lib.by_id[r].per for r in self._sub_refs(n[1]) if r in lib.by_id and lib.by_id[r].per}
                if len(scopes) != 1:
                    raise Refused(f"{self.where}: any(...) must range over one scope, got {sorted(scopes) or 'none'}")
                scope = scopes.pop()
                return any(ev(n[1], (scope, i)) for i in lib.instances(scope, eff))
            v = look(n[1], ctx)
            if kind == "truthy":
                return v not in (_UNSET, None, False, "", [], {})
            if v is _UNSET or v is None:
                return False
            if kind == "len":
                size = len(v) if isinstance(v, (list, dict)) else 0
                return _compare(size, n[2], n[3])
            if kind == "in":
                return v in n[2]
            if kind == "in_source":
                return v in lib.source(n[2])
            op, lit = n[2], n[3]
            if op == "has":
                return isinstance(v, list) and lit in v
            return _compare(v, op, lit)
        return bool(ev(self.tree, inst))

    @staticmethod
    def _sub_refs(node) -> list[str]:
        out = []

        def walk(n):
            if n[0] in ("or", "and"):
                walk(n[1]), walk(n[2])
            elif n[0] in ("not", "any"):
                walk(n[1])
            else:
                out.append(n[1])
        walk(node)
        return out


def _compare(v, op, lit) -> bool:
    if op == "==":
        return v == lit
    if op == "!=":
        return v != lit
    try:
        return {">": v > lit, "<": v < lit, ">=": v >= lit, "<=": v <= lit}[op]
    except TypeError:
        return False


# ── the library ───────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Question:
    raw: dict
    when: When | None

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def kind(self) -> str:
        return self.raw.get("kind", "")

    @property
    def section(self) -> str:
        return self.raw.get("section", "")

    @property
    def per(self) -> str | None:
        return self.raw.get("per")

    @property
    def required(self) -> bool:
        return self.raw.get("required", True) is not False

    @property
    def secret(self) -> bool:
        return self.raw.get("secret") is True

    def writes(self) -> list[str]:
        if self.raw.get("manual"):
            return []
        return [w for w in self.raw.get("writes", []) if not w.startswith("none")]

    def choices(self, lib: "Library", eff: dict, spec: dict | None = None) -> list | None:
        spec = self.raw if spec is None else spec
        if not any(k in spec for k in ("choices", "choices_from", "also_from")):
            return None
        out = list(spec.get("choices", []))
        if spec.get("choices_from"):
            out += lib.source(spec["choices_from"])
        if spec.get("also_from"):
            extra = eff.get(spec["also_from"])
            out += list(extra) if isinstance(extra, list) else []
        return out + list(spec.get("extra_choices", []))


class Library:
    """deploy/setup/questions.yml, read and checked: sections, scopes, files, derivations, questions in asking order."""

    def __init__(self, doc: dict, repo: Path, source: str):
        self.repo, self.source_name = repo, source
        for key in ("sections", "scopes", "files", "derived", "questions"):
            if key not in doc:
                raise Refused(f"{source}: no {key}")
        self.sections = doc["sections"]
        self.scopes: dict[str, str] = doc["scopes"]
        self.files = {name: dict(spec, pattern=fnmatch.translate(name)) for name, spec in doc["files"].items()}
        self.questions = [Question(q, When(q["when"], f"{source}: {q['id']}") if q.get("when") else None)
                          for q in doc["questions"]]
        self.derived = [Question(d, When(d["when"], f"{source}: {d['id']}") if d.get("when") else None)
                        for d in doc["derived"]]
        self.by_id = {q.id: q for q in self.questions}
        if len(self.by_id) != len(self.questions):
            raise Refused(f"{source}: a question id is used twice")
        for scope, owner in self.scopes.items():
            if owner not in self.by_id or self.by_id[owner].kind not in ("list", "multi"):
                raise Refused(f"{source}: scope {scope} is named by {owner}, which is no list question")
        for q in self.questions + self.derived:
            if q.per and q.per not in self.scopes:
                raise Refused(f"{source}: {q.id} is per {q.per}, which is no scope")
            if q in self.questions and q.kind not in KINDS:
                raise Refused(f"{source}: {q.id} has kind {q.kind!r}; kinds are {', '.join(KINDS)}")
        self._sources: dict[str, list] = {}

    @classmethod
    def load(cls, path: Path, repo: Path = REPO) -> "Library":
        return cls(miniyaml.parse(Path(path).read_text(encoding="utf-8"), Path(path).name), repo, Path(path).name)

    def source(self, name: str) -> list:
        if name not in SOURCES:
            raise Refused(f"{self.source_name}: no library source {name!r}; sources are {', '.join(SOURCES)}")
        if name not in self._sources:
            self._sources[name] = SOURCES[name](self.repo)
        return self._sources[name]

    def when_refs(self, q: Question) -> list[str]:
        """The question ids a `when` reads — a scope's own name stands for the question that lists its instances."""
        return [self.scopes.get(r, r) for r in q.when.refs()] if q.when else []

    def look(self, ident: str, ctx: tuple | None, eff: dict, where: str) -> Any:
        """What `ident` is inside a `when`, with `ctx` the (scope, instance) being asked: a global answer, this
        instance's answer of a per-scope question, or — a scope's own name — the instance itself."""
        if ident in self.scopes:
            if ctx is None or ctx[0] != ident:
                raise Refused(f"{where}: when names the scope {ident} outside a question asked per {ident}")
            return ctx[1]
        q = self.by_id.get(ident)
        if q is None:
            raise Refused(f"{where}: when names {ident!r}, which is no question")
        if q.per:
            if ctx is None or ctx[0] != q.per:
                raise Refused(f"{where}: when names {ident} (per {q.per}) outside its scope: use any(...)")
            return eff.get(ident, {}).get(ctx[1], _UNSET)
        return eff.get(ident, _UNSET)

    def instances(self, scope: str, eff: dict) -> list[str]:
        v = eff.get(self.scopes[scope])
        return list(v) if isinstance(v, list) else []

    def all_writes(self) -> list[tuple[str, str]]:
        """(file with its placeholders as `*`, key) of every question's and derivation's writes, manual ones included,
        wildcard keys left out — what the coverage guard compares with the examples."""
        out = []
        for q in self.questions + self.derived:
            for w in q.raw.get("writes", []):
                if w.startswith("none"):
                    continue
                f, key = w.split(" ", 1)
                if "*" in key.split(" ")[0]:
                    continue
                out.append((re.sub(r"<[^>]+>", "*", f), key.split(" ")[0]))
        return out

    def node_env_keys(self) -> set[str]:
        return node_env_keys(self.repo)

    def secret_keys(self) -> set[str]:
        return {w.split(" ", 1)[1] for q in self.questions if q.secret for w in q.writes()}


# ── checking one answer ───────────────────────────────────────────────────────────────────────────────────────────

def _check(lib: Library, q: Question, spec: dict, v: Any, eff: dict, where: str) -> str | None:
    """None when `v` is a valid answer of `spec` (a question, or a records field), else why not. Never repeats a
    secret's value."""
    kind = spec.get("kind")
    choices = q.choices(lib, eff, spec)
    src = spec.get("choices_from") or spec.get("also_from") or "its choices"

    def bad(why):
        return f"{where}: {why}"
    if kind == "bool":
        return None if isinstance(v, bool) else bad(f"must be true or false, got {v!r}")
    if kind == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return bad(f"must be a number, got {v!r}")
        if spec.get("integer") and not isinstance(v, int):
            return bad(f"must be a whole number, got {v!r}")
        if "min" in spec and v < spec["min"] or "max" in spec and v > spec["max"]:
            return bad(f"must be within {spec.get('min', '-∞')}..{spec.get('max', '∞')}, got {v!r}")
        return None
    if kind == "port":
        return None if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65535 else bad(f"must be a port 1-65535, got {v!r}")
    if kind == "choice":
        return None if v in (choices or []) else bad(f"{v!r} is not one of {', '.join(map(str, choices or []))} ({src})")
    if kind in ("multi", "list"):
        if not isinstance(v, list) or not v:
            return bad(f"must be a non-empty list, got {v!r}")
        twice = sorted({str(x) for x in v if v.count(x) > 1})
        if twice:
            return bad(f"names {', '.join(twice)} twice")
        for x in v:
            if kind == "multi" and x not in (choices or []):
                return bad(f"{x!r} is not one of {', '.join(map(str, choices or []))} ({src})")
            if spec.get("pattern") and not (isinstance(x, str) and re.fullmatch(spec["pattern"], x)):
                return bad(f"{x!r} does not match {spec['pattern']}")
        return None
    if kind == "records":
        if not isinstance(v, list) or not v:
            return bad("must list at least one record")
        fields = spec.get("fields", {})
        for i, rec in enumerate(v):
            if not isinstance(rec, dict):
                return bad(f"[{i}] must be a record of {', '.join(fields)}")
            unknown = sorted(set(rec) - set(fields))
            if unknown:
                return bad(f"[{i}] has unknown field(s) {', '.join(unknown)}; fields are {', '.join(fields)}")
            for name, f in fields.items():
                if name not in rec:
                    if f.get("required", True) is not False:
                        return bad(f"[{i}] needs {name}")
                    continue
                why = _check(lib, q, f, rec[name], eff, f"{where}[{i}].{name}")
                if why:
                    return why
        return None
    if kind == "map":
        return None if isinstance(v, dict) and v else bad("must be a non-empty mapping")
    if kind == "fact":
        if not isinstance(v, dict) or set(v) != {"value", "source"}:
            return bad("must be {value, source}")
        return None if isinstance(v["source"], str) and v["source"].strip() else bad("source must say where the figure comes from")
    if kind == "image":
        if isinstance(v, str):
            return None if _DIGEST.fullmatch(v) else bad(f"{v!r} is not pinned (repo@sha256:<64 hex> or sha256:<64 hex>)")
        if isinstance(v, dict) and v:
            for e, ref in v.items():
                if not (isinstance(ref, str) and _DIGEST.fullmatch(ref)):
                    return bad(f"{e}: {ref!r} is not pinned")
            return None
        return bad("must be a pinned image or {engine: pinned image}")
    # string kinds
    if not isinstance(v, str) or not v.strip():
        return bad("must be a non-empty string" if kind == "secret" else f"must be a non-empty string, got {v!r}")
    if kind == "secret":
        return None
    shape = {"path": r"(/|~/?)\S*", "host": r"[A-Za-z0-9_.:\[\]-]+", "bind": r"\S+:[0-9]{1,5}",
             "url": r"https?://\S+"}.get(kind)
    for pat in (shape, spec.get("pattern")):
        if pat and not re.fullmatch(pat, v):
            return bad(f"{v!r} is not {KINDS[kind]}" + (f" matching {pat}" if pat == spec.get("pattern") else ""))
    return None


# ── the sheet ─────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Sheet:
    eff: dict                       # id → answer (global) or {instance: answer} (per scope); only asked + answered
    asked: dict                     # id → True/False (global) or {instance: bool}


def evaluate(lib: Library, answers: dict) -> Sheet:
    """The effective answers of `answers` — or Refused naming every gap, unknown id, bad answer and skipped answer."""
    if not isinstance(answers, dict):
        raise Refused("the answer sheet must be a mapping of question id → answer")
    errors: list[str] = []
    unknown = [k for k in answers if k not in lib.by_id]
    for k in unknown:
        errors.append(f"{k}: no such question in questions.yml" + (" (a derived value: it is computed, never answered)"
                                                                   if k in {d.id for d in lib.derived} else ""))
    eff: dict = {}
    asked: dict = {}

    def look(ident, ctx):
        return lib.look(ident, ctx, eff, "the sheet")

    for q in lib.questions:
        given = answers.get(q.id)
        if q.per:
            insts = lib.instances(q.per, eff)
            if given is not None and not isinstance(given, dict):
                errors.append(f"{q.id}: asked per {q.per}, so its answer is a mapping {q.per} → answer "
                              f"({', '.join(insts) or 'no instances'})")
                continue
            given = given or {}
            for extra in [k for k in given if k not in insts]:
                errors.append(f"{q.id}: {extra!r} is no {q.per} (the {q.per}s are {', '.join(insts) or 'none'})")
            vals, flags = {}, {}
            for inst in insts:
                on = q.when.eval(look, lib, eff, (q.per, inst)) if q.when else True
                flags[inst] = on
                v = given.get(inst)
                if not on:
                    if v is not None:
                        errors.append(f"{q.id} [{inst}]: answered, but not asked — when {q.when.text}")
                    continue
                if v is None:
                    if q.required:
                        errors.append(f"{q.id} [{inst}]: missing — {q.raw['ask']}")
                    continue
                why = _check(lib, q, q.raw, v, eff, f"{q.id} [{inst}]")
                if why:
                    errors.append(why)
                else:
                    vals[inst] = v
            eff[q.id], asked[q.id] = vals, flags
        else:
            on = q.when.eval(look, lib, eff, None) if q.when else True
            asked[q.id] = on
            if not on:
                if given is not None:
                    errors.append(f"{q.id}: answered, but not asked — when {q.when.text}")
                continue
            if given is None:
                if q.required:
                    errors.append(f"{q.id}: missing — {q.raw['ask']}")
                continue
            why = _check(lib, q, q.raw, given, eff, q.id)
            if why:
                errors.append(why)
            else:
                eff[q.id] = given
    # A STANDALONE SERVER is one server file: up.sh reads deploy/standalone/.env, the gateway's lines and its node's.
    if eff.get("shape.kind") == "standalone":
        for c, f in eff.get("cluster.server_file", {}).items():
            if f != ".env":
                errors.append(f"cluster.server_file [{c}]: a standalone server's node lines go in .env with the "
                              f"gateway's (up.sh's default), not {f}")
    # A SERVING CARD IS ATTACHED THROUGH ITS ACCELERATOR'S RUNTIME (hardware/accelerators/<key>.yml gpu_runtime): a box
    # whose driver stack is another cannot hand the card to a container. The accelerator is the card type's — shown,
    # never asked — so the box's answer is checked against it here, naming both.
    accs = card_accelerators(lib.repo)
    for box, cards in eff.get("box.cards", {}).items():
        rt = eff.get("box.gpu_runtime", {}).get(box)
        for c in cards:
            a = accs.get(c["type"])
            if a is not None and c["serves"] and rt is not None and rt != a["gpu_runtime"]:
                errors.append(f"box.gpu_runtime [{box}]: {rt}, but its serving card {c['index']} ({c['type']}) is "
                              f"{a['accelerator']}, attached through {a['gpu_runtime']} "
                              f"(lb/vllm_lb/hardware/accelerators/{a['accelerator']}.yml)")
    # A REFERENCE BLUEPRINT IS FOR ONE KIND OF CLUSTER (its target): one whose control, TP or accelerator is not this
    # cluster's is refused naming why — never prefilled onto cards it never ran on.
    for cluster, bid in eff.get("cluster.blueprint", {}).items():
        why = fitting_blueprints(lib.repo, eff, cluster).get(bid)
        if why:
            errors.append(f"cluster.blueprint [{cluster}]: {bid} does not fit {cluster}: {why}")
    knobs = cluster_knobs(lib.repo) if eff.get("cluster.first_start_settings") else []
    for cluster, settings in eff.get("cluster.first_start_settings", {}).items():
        unknown = [str(k) for k in settings if k not in knobs]
        if unknown:
            errors.append(f"cluster.first_start_settings [{cluster}]: {', '.join(unknown)} is no knob of a cluster "
                          f"Start ({', '.join(knobs)})")
    tuning = eff.get("lb.tuning")
    if tuning:
        documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", (lib.repo / "lb/.env.example").read_text(), re.M))
        retired = retired_keys(lib.repo)
        asked_keys = {w.split(" ", 1)[1] for q in lib.questions if q.id != "lb.tuning" for w in q.writes()
                      if w.startswith("lb/.env ")}
        for k in tuning:
            if k in retired:
                errors.append(f"lb.tuning: {k} is RETIRED (lb/vllm_lb/config.py RETIRED_KEYS): the gateway refuses to boot with it")
            elif k not in documented:
                errors.append(f"lb.tuning: {k} is not a knob lb/.env.example documents")
            elif k in asked_keys:
                errors.append(f"lb.tuning: {k} is answered by its own question — one place per value")
    if errors:
        raise Refused("\n".join(errors))
    return Sheet(eff, asked)


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────────

class Plan:
    """The files a sheet renders: env files (ordered KEY → value, JSON arrays assembled per instance), the fleet doc."""

    def __init__(self, lib: Library):
        self.lib = lib
        self.env: dict[str, dict[str, str]] = {}
        self.arrays: dict[tuple[str, str], dict[str, dict]] = {}
        self.fleet: dict | None = None
        self.origin: dict[tuple[str, str], str] = {}
        self.notes: list[str] = []
        self.errors: list[str] = []
        #: What the libraries DERIVE from the answers, shown for the user to see and never asked: each card's
        #: accelerator, each ssh cluster's first-Start resolution (image, sizing).
        self.shown: list[str] = []

    def _fleet(self) -> dict:
        if self.fleet is None:
            self.fleet = {"boxes": {}, "clusters": {}, "backends": {}}
        return self.fleet

    def put(self, file: str, key: str, value: Any, q: Question) -> None:
        values = q.raw.get("values")
        if values is not None:
            word = str(value).lower() if isinstance(value, bool) else str(value)
            if word in values:
                value = values[word]
                if value is None:
                    return
        # A SCOPE'S OWN LIST written to a container (the boxes, the clusters, the power allowlist) makes one entry per
        # name; the per-instance questions fill each entry.
        names = value if isinstance(value, list) and q.id in self.lib.scopes.values() else None
        if file == "lb/fleet.yml":
            path = key.split(SEP)
            node = self._fleet()
            if names is not None:
                for name in names:
                    node.setdefault(path[0], {}).setdefault(name, {})
                return
            for seg in path[:-1]:
                node = node.setdefault(seg, {})
            node[path[-1]] = value
            return
        m = re.fullmatch(rf"([A-Z][A-Z0-9_]*)\[([^\]]+)\]{SEP}(.+)", key)
        if m:
            arr = self.arrays.setdefault((file, m.group(1)), {})
            obj = arr.setdefault(m.group(2), {"name": m.group(2)})
            *parents, leaf = m.group(3).split(SEP)
            for p in parents:
                obj = obj.setdefault(p, {})
            obj[leaf] = value
            self.env.setdefault(file, {}).setdefault(m.group(1), "")
            return
        if names is not None:
            arr = self.arrays.setdefault((file, key), {})
            for name in names:
                arr.setdefault(name, {"name": name})
            self.env.setdefault(file, {}).setdefault(key, "")
            return
        if key.startswith("*"):
            for k, v in value.items():
                self._line(file, str(k), str(v), q.id)
            return
        if isinstance(value, list):
            value = {"colon": ":", "csv": ","}.get(q.raw.get("env_format"), " ").join(map(str, value))
        elif isinstance(value, bool):
            value = "true" if value else "false"
        self._line(file, key, "" if value is None else str(value), q.id)

    def _line(self, file: str, key: str, value: str, origin: str) -> None:
        lines = self.env.setdefault(file, {})
        if key in lines and lines[key] != value:
            self.errors.append(f"{file}: {key} would be written twice, by {self.origin[(file, key)]} and {origin} — "
                               f"one place per value")
            return
        lines[key] = value
        self.origin[(file, key)] = origin

    def text(self, file: str, sheet_name: str) -> str:
        if file == "lb/fleet.yml":
            return (f"# Rendered by deploy/setup/render.py from {sheet_name}: change the answer sheet and re-render.\n"
                    f"# What this deployment IS (lb/fleet.example.yml documents every key).\n\n"
                    + miniyaml.dump(self.fleet))
        out = [f"# Rendered by deploy/setup/render.py from {sheet_name}: change the answer sheet and re-render.",
               f"# Every name is documented in the example next to this file. Never commit it."]
        for k, v in self.env[file].items():
            arr = self.arrays.get((file, k))
            if arr is not None:
                v = "'" + json.dumps(list(arr.values()), ensure_ascii=False, separators=(",", ":")).replace("'", "\\u0027") + "'"
            elif re.search(r"[\s'\"#$\\`]", v):
                v = "'" + v.replace("'", "'\\''") + "'" if "'" not in v else json.dumps(v)
            out.append(f"{k}={v}")
        return "\n".join(out) + "\n"

    def files(self) -> list[str]:
        return (["lb/fleet.yml"] if self.fleet is not None else []) + list(self.env)


def _subst(template: str, scope: str | None, inst: str | None, sheet: Sheet, lib: Library) -> list[tuple[str, str | None]]:
    """`template` with its <placeholders> filled: <scope> → the instance, <scope.id> → that instance's answer. A
    placeholder of ANOTHER scope (a global question writing a per-instance file) expands over that scope's instances."""
    names = re.findall(r"<([^>]+)>", template)
    if not names:
        return [(template, inst)]
    other = {n.split(".")[0] for n in names} - ({scope} if scope else set())
    if other:
        (s,) = other
        return [x for i in lib.instances(s, sheet.eff) for x in _subst(template, s, i, sheet, lib)]

    def fill(m):
        n = m.group(1)
        if n == scope:
            return inst
        v = sheet.eff.get(n, {}).get(inst) if isinstance(sheet.eff.get(n), dict) else None
        if v is None:
            raise Refused(f"{template}: {n} [{inst}] is not answered")
        return str(v)
    return [(re.sub(r"<([^>]+)>", fill, template), inst)]


def _derive(rule: str, d: Question, inst: str | None, sheet: Sheet) -> str | None:
    e = sheet.eff
    if rule == "lb_compose_chain":
        parts = ["docker-compose.yml"]
        if e.get("broker.enabled") is True:
            parts.append("docker-compose.broker.yml")
        for qid, f in (("lb.bench_network", "docker-compose.bench-net.yml"), ("lb.node_network", "docker-compose.node.yml"),
                       ("lb.client_network", "docker-compose.clients.yml")):
            if e.get(qid):
                parts.append(f)
        return ":".join(parts) if len(parts) > 1 else None
    if rule == "hw_sender_compose":
        parts = ["docker-compose.yml", f"gpu-{e['sender.gpu_kind'][inst]}.yml"]
        if e["sender.bridged"].get(inst) is True:
            parts.append("bridged.yml")
        return ":".join(parts)
    if rule == "box_hostname":
        return e["box.hostname"][inst]
    if rule == "bench_compose":
        return "docker-compose.yml:subnet.yml"
    raise Refused(f"no derivation rule {rule!r}")


@contextlib.contextmanager
def _rendered_fleet(plan: Plan, roots: str | None):
    """The rendered lb/fleet.yml as a file the LB's loader reads, with MODELS_SEARCH_DIRS set to `roots` (the models
    roots a checkpoint's folder is relative to) while it is read — restored after."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fleet.yml"
        path.write_text(miniyaml.dump(plan.fleet), encoding="utf-8")
        old = os.environ.get("MODELS_SEARCH_DIRS")
        if roots:
            os.environ["MODELS_SEARCH_DIRS"] = roots
        try:
            yield path
        finally:
            if old is None:
                os.environ.pop("MODELS_SEARCH_DIRS", None)
            else:
                os.environ["MODELS_SEARCH_DIRS"] = old


def sizing_problem(cluster: str, model: str, resolution: dict, answer: str | None) -> str:
    """"" when `cluster.first_start` is answered exactly where it applies, else why not. `resolution`: the LB
    resolver's `refused` and `sizing` for the first model on the cluster's cards (resolve.Resolution). A model NOT
    MEASURED on that engine × accelerator × TP is refused at Start unless the operator explicitly chooses the
    estimate (docs/design/engines.md), so the sheet must say which — `measure-first` or `start-on-estimate`; a measured
    one is not asked. Any other refusal (no image in the matrix, an engine or accelerator that does not run it) is the
    first model's own and no answer here lifts it."""
    refused, sizing = resolution.get("refused") or "", resolution.get("sizing") or ""
    if refused and sizing != "unmeasured":
        return f"cluster.first_model [{cluster}]: {model} cannot start on the cluster's cards: {refused}"
    if sizing == "unmeasured" and answer is None:
        return (f"cluster.first_start [{cluster}]: missing — {model}: {refused}. Answer measure-first (no Start until it "
                f"is measured there) or start-on-estimate (the first Start sized by the estimate, chosen explicitly)")
    if sizing != "unmeasured" and answer is not None:
        return f"cluster.first_start [{cluster}]: answered, but {model} is measured on these cards — not asked"
    return ""


def _first_starts(plan: Plan, sheet: Sheet, path: Path, fleet) -> None:
    """Each ssh cluster's first Start, RESOLVED by the LB's own catalog against the rendered fleet: the image the
    matrix gives its recipe on the cluster's accelerator × platform × TP, and whether it is measured there — shown;
    a refusal, or a sizing answer where it does not apply, is the sheet's."""
    first = sheet.eff.get("cluster.first_model", {})
    if not first:
        return
    from vllm_lb.catalog import CLUSTER_MODE, ModelCatalog
    from vllm_lb.fleet import catalog_deployment_of, start_checker
    cat = ModelCatalog.load(deployment=catalog_deployment_of(path))
    if cat.error:
        plan.errors.append(f"the model library cannot be read for the first Start: {cat.error}")
        return
    for cluster, ref in first.items():
        model = ref.split("@", 1)[0]
        answer = sheet.eff.get("cluster.first_start", {}).get(cluster)
        spec = cat.get(model)
        profile = spec.profile(CLUSTER_MODE) if spec is not None else None
        if profile is None:
            why = cat.absent.get(model) or "its recipe (library/recipes.json) has no cluster launch"
            plan.errors.append(f"cluster.first_model [{cluster}]: {ref} cannot start on the cluster: {why}")
            continue
        res = profile.resolution
        why = sizing_problem(cluster, ref, {"refused": res.refused, "sizing": res.sizing}, answer)
        if not why and answer == "start-on-estimate":
            try:
                cat.choice(CLUSTER_MODE, model, estimated=True)
            except ValueError as exc:
                why = f"cluster.first_start [{cluster}]: {exc}"
        if why:
            plan.errors.append(why)
            continue
        # THE FIRST START, HELD TO THE SWITCH'S OWN VALIDATOR against this fleet (fleet.start_checker →
        # cluster.DesiredStore.check): its model, checkpoint and figures (`cluster.first_start_settings`, a reference
        # blueprint's or typed) — refused here in the switch's words, not at Phase 4's POST.
        settings = sheet.eff.get("cluster.first_start_settings", {}).get(cluster) or {}
        try:
            start_checker(fleet.clusters[cluster], cat).check(
                model_choice=model, uncensored=ref.endswith("@uncensored"),
                estimated=answer == "start-on-estimate", **settings)
        except (ValueError, TypeError) as exc:
            plan.errors.append(f"cluster.first_start_settings [{cluster}]: the first Start of {ref} would be refused: "
                               f"{exc}")
            continue
        plan.shown.append(f"  cluster {cluster}: first Start {ref} → {res.engine}/{res.build} on {res.target.key} → "
                          f"{res.image} ({res.sizing}{', started on the estimate' if answer == 'start-on-estimate' else ''})")


def _node_env(plan: Plan, sheet: Sheet, cluster: str, file: str, check: bool) -> None:
    """The node's lines from the LB's own `node-env` — in the vllm-lb image; refused naming it anywhere else."""
    _lb(plan.lib.repo)
    try:
        from vllm_lb.fleet import load_fleet, node_env
    except ImportError as exc:
        if check:
            plan.notes.append(f"{file}: the node's EDGE_NODE_* lines are rendered by `python -m vllm_lb.fleet node-env` "
                              f"when written (here: {exc.name} is not importable)")
            return
        plan.errors.append(f"{file}: its node lines come from `python -m vllm_lb.fleet node-env`, which needs the "
                           f"gateway's code and PyYAML ({exc.name} is not importable here): run this in the vllm-lb "
                           f"image — {IN_LB_IMAGE}")
        return
    with _rendered_fleet(plan, sheet.eff["cluster.node_models_dir"][cluster]) as path:
        try:
            lines = node_env(load_fleet(path), cluster)     # its catalog: the library resolved against THIS fleet
        except ValueError as exc:
            plan.errors.append(f"{file}: node-env refused cluster {cluster}: {exc}")
            return
    for line in lines:
        k, v = line.split("=", 1)
        plan._line(file, k, v, "node-env")


def _loader_check(plan: Plan, sheet: Sheet) -> None:
    """The rendered fleet, read by the LB's own loader where it can be imported — a refusal is the sheet's."""
    if plan.fleet is None:
        return
    _lb(plan.lib.repo)
    try:
        from vllm_lb.fleet import load_fleet
    except ImportError as exc:
        plan.notes.append(f"lb/fleet.yml was NOT read by the gateway's loader here ({exc.name} is not importable): "
                          f"Phase 4 boots it, or re-run in the vllm-lb image — {IN_LB_IMAGE}")
        if sheet.eff.get("cluster.first_model"):
            plan.notes.append("the first Start's resolution (its image from the matrix, measured or not on the "
                              "cluster's cards, so whether cluster.first_start applies) is the LB resolver's: re-run "
                              f"in the vllm-lb image — {IN_LB_IMAGE}")
        return
    roots = plan.env.get("lb/.env", {}).get("MODELS_SEARCH_DIRS") or next(
        iter(sheet.eff.get("cluster.node_models_dir", {}).values()), None)
    with _rendered_fleet(plan, roots) as path:
        try:
            fleet = load_fleet(path)     # its catalog: the library resolved against THIS fleet's cards and checkpoints
        except ValueError as exc:
            plan.errors.append(f"lb/fleet.yml: the gateway's loader refuses it: {exc}")
            return
        plan.notes.append("lb/fleet.yml: read and accepted by the gateway's own loader (vllm_lb.fleet.load_fleet)")
        _first_starts(plan, sheet, path, fleet)


def build(lib: Library, sheet: Sheet, *, check: bool) -> Plan:
    plan = Plan(lib)
    accs = card_accelerators(lib.repo)
    for box, cards in sheet.eff.get("box.cards", {}).items():
        for c in cards:
            a = accs.get(c["type"])
            plan.shown.append(f"  {box} card {c['index']}: {c['type']} → " + (
                f"accelerator {a['accelerator']} ({a['arch']}, {a['gpu_runtime']})" if a else
                "accelerator: the new type's own (hwtype.accelerator)"))
    if plan.shown:
        plan.shown.insert(0, "DERIVED FROM THE LIBRARIES (shown, never asked)")
    for q in lib.questions:
        if q.id not in sheet.eff:
            continue
        pairs = list(sheet.eff[q.id].items()) if q.per else [(None, sheet.eff[q.id])]
        for inst, value in pairs:
            for w in q.writes():
                f_t, k_t = w.split(" ", 1)
                # File and key are filled TOGETHER, so a global question writing a per-instance file (telemetry's port
                # → every sender's file) fills each file's key for the same instance. The key's own dots become SEP
                # BEFORE an instance is filled in: a model id has dots of its own (checkpoints.<checkpoint>.path).
                key_t = re.sub(r"\.(?![^<]*>)", SEP, k_t.split(" ")[0])
                for target, _ in _subst(f"{f_t}\x00{key_t}", q.per, inst, sheet, lib):
                    f, k = target.split("\x00")
                    plan.put(f, k, value, q)
    node_lines = []
    for d in lib.derived:
        insts = lib.instances(d.per, sheet.eff) if d.per else [None]
        for inst in insts:
            look_ctx = (d.per, inst) if d.per else None

            def look(ident, ctx, _d=d):
                return lib.look(ident, ctx, sheet.eff, _d.id)
            if d.when and not d.when.eval(look, lib, sheet.eff, look_ctx):
                continue
            for w in d.raw["writes"]:
                f_t, k_t = w.split(" ", 1)
                for f, i in _subst(f_t, d.per, inst, sheet, lib):
                    if d.raw["rule"] == "node_env":
                        node_lines.append((inst, f))
                        continue
                    v = _derive(d.raw["rule"], d, i, sheet)
                    if v is not None:
                        plan._line(f, k_t, v, d.id)
    for cluster, f in node_lines:
        _node_env(plan, sheet, cluster, f, check)
    _loader_check(plan, sheet)
    if plan.errors:
        raise Refused("\n".join(plan.errors))
    return plan


def _summary(lib: Library, sheet: Sheet, plan: Plan, out: Path) -> str:
    rows = ["THE ANSWER SHEET (secrets masked)"]
    for s in lib.sections:
        qs = [q for q in lib.questions if q.section == s["id"] and q.id in sheet.eff
              and (not q.per or sheet.eff[q.id])]
        if not qs:
            continue
        rows.append(f"  [{s['id']}] {s['title']}")
        for q in qs:
            pairs = sheet.eff[q.id].items() if q.per else [(None, sheet.eff[q.id])]
            for inst, v in pairs:
                shown = MASK if q.secret else json.dumps(v, ensure_ascii=False)
                rows.append(f"    {q.id}{f' [{inst}]' if inst else ''}: {shown}")
    rows += plan.shown
    rows.append(f"FILES ({out}):")
    for f in plan.files():
        keys = (", ".join(plan.env[f]) if f in plan.env else "boxes, clusters, backends")
        rows.append(f"  {f}: {keys}")
    manual = sorted({w.split(' ', 1)[0].replace(f"<{q.per}>", i)
                     for q in lib.questions if q.raw.get("manual") and q.per and q.id in sheet.eff
                     for i in sheet.eff[q.id] for w in q.raw["writes"] if not w.startswith("none")})
    for m in manual:
        rows.append(f"  {m}: BY HAND, with its test (a sub-wizard's answers; never rendered)")
    rows += [f"NOTE: {n}" for n in plan.notes]
    return "\n".join(rows)


def _masked(text: str, secret_keys: set[str]) -> list[str]:
    return [(ln.split("=", 1)[0] + "=" + MASK) if "=" in ln and ln.split("=", 1)[0] in secret_keys else ln
            for ln in text.splitlines()]


def _print_prefill(questions: Path, sheet_path: Path) -> int:
    """`--prefill`: per cluster, the reference blueprints that fit it (and why each other does not), then each answered
    `cluster.blueprint`'s proposals as answer-sheet lines (flow YAML), each beside what the sheet answers now — for the
    agent to put to the user as "proposed: … — confirm?" and write the confirmed ones into the sheet."""
    try:
        lib = Library.load(questions, REPO)
        answers = miniyaml.parse(sheet_path.read_text(encoding="utf-8"), sheet_path.name) or {}
        pre = prefill(lib, answers)
    except (Refused, ValueError) as exc:
        print(f"render: refused:\n{exc}", file=sys.stderr)
        return 2
    library = read_blueprints(REPO)
    rows = [f"REFERENCE BLUEPRINTS ({BLUEPRINTS}) — by each cluster's answered control and cards"]
    for cluster, fits in pre.fits.items():
        for bid, why in fits.items():
            rows.append(f"  {cluster}: {bid} — " + (f"FITS: {library[bid]['label']}" if not why else f"not: {why}"))
    chosen = answers.get("cluster.blueprint") or {}
    if pre.answers:
        rows.append(f"PROPOSED by {', '.join(f'{c}: {b}' for c, b in chosen.items())} — confirm or edit each with the "
                    f"user, then write it into {sheet_path.name}")
    for qid, v in pre.answers.items():
        q = lib.by_id[qid]
        pairs = v.items() if q.per else [(None, v)]
        for inst, val in pairs:
            given = answers.get(qid)
            now = given.get(inst) if q.per and isinstance(given, dict) else (None if q.per else given)
            line = (f"{qid}: {{{inst}: {json.dumps(val, ensure_ascii=False)}}}" if q.per
                    else f"{qid}: {json.dumps(val, ensure_ascii=False)}")
            said = "unanswered" if now is None else ("the same" if now == val else
                                                     f"now {json.dumps(now, ensure_ascii=False)}")
            rows.append(f"{line}    # {said}")
    for cluster, bid in chosen.items():
        bp = library.get(bid) or {}
        nulls = [k for k, x in (bp.get("knobs") or {}).items() if x is None]
        if nulls:
            rows.append(f"# {cluster}: {bid} leaves {', '.join(nulls)} null — the engine's own: leave unanswered")
        if (bp.get("target") or {}).get("control") == "local":
            rows.append(f"# {cluster}: cluster.image is this deployment's pinned build, never a blueprint's — a KV dtype "
                        f"is taken only on an image lb/vllm_lb/kv_cache_dtypes.json records (lb/vllm_lb/library/images "
                        f"names the build {bid} ran)")
        if bp.get("uncensored") and not (answers.get("checkpoint.uncensored") or {}).get(bp["model"]):
            rows.append(f"# {cluster}: {bid} runs {bp['model']}'s uncensored twin — checkpoint.uncensored "
                        f"[{bp['model']}] names its folder (this deployment's)")
    print("\n".join(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sheet", type=Path)
    ap.add_argument("--out", type=Path, help="where the files land (the checkout itself: --out .)")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.yml")
    ap.add_argument("--check", action="store_true", help="validate and print the sheet and the plan; write nothing")
    ap.add_argument("--diff", action="store_true", help="print what would change in files that exist; write nothing")
    ap.add_argument("--replace", action="append", default=[], metavar="FILE",
                    help="overwrite this existing file (only after the user approved its --diff)")
    ap.add_argument("--prefill", action="store_true",
                    help="list the reference blueprints that fit each cluster and print what each answered "
                         "cluster.blueprint proposes (a sheet still being asked is fine); write nothing")
    a = ap.parse_args(argv)
    if a.prefill:
        return _print_prefill(a.questions, a.sheet)
    if a.out is None:
        ap.error("--out is required (except with --prefill)")
    try:
        lib = Library.load(a.questions, REPO)
        sheet = evaluate(lib, miniyaml.parse(a.sheet.read_text(encoding="utf-8"), a.sheet.name))
        plan = build(lib, sheet, check=a.check)
    except (Refused, ValueError) as exc:
        print(f"render: refused:\n{exc}", file=sys.stderr)
        return 2
    print(_summary(lib, sheet, plan, a.out))
    if a.check:
        return 0
    texts = {f: plan.text(f, a.sheet.name) for f in plan.files()}
    secrets = lib.secret_keys()
    if a.diff:
        for f, new in texts.items():
            p = a.out / f
            if p.exists():
                diff = difflib.unified_diff(_masked(p.read_text(encoding="utf-8"), secrets), _masked(new, secrets),
                                            f"{f} (now)", f"{f} (rendered)", lineterm="")
                print("\n".join(diff) or f"{f}: unchanged")
        return 0
    exists = [f for f in texts if (a.out / f).exists() and f not in a.replace]
    if exists:
        print(f"render: refused: {', '.join(exists)} exist{'s' if len(exists) == 1 else ''} under {a.out} — show the "
              f"user `--diff`, and on their yes pass --replace <file> for each", file=sys.stderr)
        return 2
    for f, text in texts.items():
        p = a.out / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        p.chmod(0o644 if f == "lb/fleet.yml" else 0o600)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
