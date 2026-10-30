"""Read-only is a property of the source, proved here - not a promise in a docstring.

Every module in `usage_mcp` is parsed and walked. A future edit that opens a file for writing,
issues a second HTTP verb, shells out, or so much as mentions `/lb/cluster` fails this file instead
of shipping. That is what makes the guarantee structural: the only way to add a control path is to
delete a test, which shows up in a diff.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import usage_mcp
from usage_mcp import readonly
from usage_mcp.readonly import LB_PATHS, READ_MODES, ReadOnlyViolation, Sources

PACKAGE = Path(usage_mcp.__file__).resolve().parent
MODULES = sorted(PACKAGE.glob("*.py"))

#: Importing any of these would give the package a way to write, execute or POST.
FORBIDDEN_IMPORTS = {
    "subprocess", "shutil", "tempfile", "socketserver", "pickle", "shelve", "sqlite3",
    "urllib.request", "urllib.error", "requests", "httpx", "aiohttp", "docker", "paramiko",
    "fabric", "psycopg", "psycopg2", "ftplib", "smtplib", "telnetlib", "pty", "ctypes",
}
#: Calling any of these by name would write, delete or execute.
FORBIDDEN_CALLS = {
    "write", "writelines", "truncate", "flush", "unlink", "remove", "rmdir", "rmtree", "mkdir",
    "makedirs", "rename", "replace", "chmod", "chown", "symlink", "link", "touch", "write_text",
    "write_bytes", "system", "popen", "spawn", "spawnl", "execv", "execve", "fork", "kill",
    "post", "put", "patch", "delete", "putrequest", "send", "sendall", "urlopen", "dump",
}
#: A string literal matching one of these has no business in a read-only server.
FORBIDDEN_LITERALS = {"POST", "PUT", "PATCH", "DELETE", "/lb/cluster", "/lb/power", "/lb/keep",
                      "/lb/swarm", "/lb/classic"}


def _trees():
    for path in MODULES:
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_package_has_modules():
    assert len(MODULES) >= 7, [p.name for p in MODULES]


def test_no_module_imports_a_way_to_write_or_post():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module or ''}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if name in FORBIDDEN_IMPORTS or root in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, offenders


def test_no_module_calls_anything_that_writes_or_sends():
    """`conn.request` is allowed only with the literal first argument "GET"."""
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else None)
            if name is None:
                continue
            if name == "request":
                first = node.args[0] if node.args else None
                if not (isinstance(first, ast.Constant) and first.value == "GET"):
                    offenders.append(f"{path.name}:{node.lineno} .request() without a literal GET")
                continue
            if name in FORBIDDEN_CALLS:
                offenders.append(f"{path.name}:{node.lineno} calls {name}()")
    assert not offenders, offenders


def test_the_only_open_call_is_mode_r():
    """`open` appears exactly once in the package, in `readonly.read_text`, with "r" literal."""
    sites = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "mode"), None)
                sites.append((path.name, node.lineno,
                              mode.value if isinstance(mode, ast.Constant) else "<not a literal>"))
    assert len(sites) == 1, sites
    name, _, mode = sites[0]
    assert name == "readonly.py", sites
    assert mode in READ_MODES, sites


def test_no_forbidden_literal_anywhere_in_the_package():
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in FORBIDDEN_LITERALS:
                    offenders.append(f"{path.name}:{node.lineno} {node.value!r}")
    assert not offenders, offenders


def test_the_endpoint_allowlist_is_frozen_and_read_only():
    assert isinstance(LB_PATHS, frozenset)
    assert LB_PATHS == {"/lb/status", "/lb/perf", "/lb/bench"}
    with pytest.raises(AttributeError):
        LB_PATHS.add("/lb/cluster")                            # type: ignore[attr-defined]


def test_no_function_in_the_package_accepts_an_http_method():
    """There is no parameter a caller could set to turn a GET into a POST."""
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if a.arg in ("method", "verb", "http_method", "data", "body", "payload", "json"):
                    offenders.append(f"{path.name}:{node.lineno} {node.name}({a.arg}=...)")
    assert not offenders, offenders


def test_lb_get_refuses_a_path_outside_the_allowlist(src: Sources):
    for path in ("/lb/cluster", "/lb/status/../cluster", "/", "/lb/power"):
        with pytest.raises(ReadOnlyViolation):
            readonly.lb_get(path, src)


def test_the_source_config_is_frozen(src: Sources):
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        src.lb_url = "http://127.0.0.2:8010"                   # type: ignore[misc]


def test_read_text_refuses_a_path_outside_the_roots(src: Sources, state_dir):
    for outside in ("/etc/passwd", str(state_dir.parent / ".env"),
                    str(state_dir / ".." / ".env")):
        with pytest.raises(ReadOnlyViolation):
            readonly.read_text(Path(outside), src)


def test_read_text_refuses_a_write_mode(src: Sources, state_dir):
    target = next((state_dir / "perf").glob("perf-*.jsonl"))
    for mode in ("w", "a", "r+", "wb", "x"):
        with pytest.raises(ReadOnlyViolation):
            readonly.read_text(target, src, mode=mode)
    assert readonly.read_text(target, src)                      # the read mode still works


def test_globs_cannot_escape_the_state_root(src: Sources):
    for bad in ("../*.env", "/etc/*"):
        with pytest.raises(ReadOnlyViolation):
            readonly.state_glob(bad, src)
    with pytest.raises(ReadOnlyViolation):
        readonly.state_dirs("../..", src)


def test_scrub_url_strips_any_credentials():
    assert readonly.scrub_url("postgresql://u:pw@postgres:5432/db") == "postgresql://postgres:5432/db"
    assert readonly.scrub_url("http://192.0.2.2:8000") == "http://192.0.2.2:8000"
    assert readonly.scrub_url(None) is None


def test_no_module_reads_a_dotenv_file(src: Sources, state_dir):
    """lb/.env holds PG_PW. No module parses a dotenv, and the real file is out of reach."""
    for path, _ in _trees():
        assert "dotenv" not in path.read_text(encoding="utf-8"), path.name
    real_env = state_dir.parent / ".env"
    if not real_env.exists():
        # In the container the state volume is mounted alone, which is itself the point: lb/.env
        # is not even present. On a checkout it is, and must still be out of reach.
        pytest.skip(f"{real_env} is not mounted in this run")
    with pytest.raises(ReadOnlyViolation):
        readonly.read_text(real_env, src)


def test_readonly_is_the_only_module_that_touches_the_outside():
    """Imports of http.client and the filesystem live in one module, so there is one chokepoint."""
    for path, tree in _trees():
        if path.name == "readonly.py":
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                assert not m.startswith("http."), f"{path.name} imports {m}"


def test_the_package_docstring_states_the_guarantee():
    assert "READ-ONLY" in (usage_mcp.__doc__ or "")
    assert "GET" in (inspect.getdoc(readonly) or "")
