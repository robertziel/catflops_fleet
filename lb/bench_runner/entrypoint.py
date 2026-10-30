"""The sealed runner's start: the only code that runs as root, and only long enough to stop being root.

In order, each step refusing — with the reason, exit non-zero, nothing started — when it cannot pass:

  1. the container's configuration (BENCH_*, set by docker-compose.yml) is complete;
  2. nothing about a backend or a credential is in this environment (ANTHROPIC_*, CLAUDE_*);
  3. Claude Code at claude_pin.INSTALL_PATH is the pinned binary, by sha256;
  4. the state directory (the LB's swarm directory, mounted) is given to `bench`, and root is dropped
     for good (setgroups, setgid, setuid — and checked);
  5. the seal is proven: egress_check.py, the same check the runner repeats before every launch;
  6. the runner is exec'd with a command built from the configuration above and nothing else.

HOME is a directory of the runner's own inside the state directory: empty at first, never the operator's,
and egress_check refuses a credential the moment one appears in it.
"""
from __future__ import annotations

import hashlib
import os
import pwd
import sys

from bench_runner import claude_pin, egress_check

#: The LB's swarm directory, mounted here (docker-compose.yml): requests in, reports out.
STATE_DIR = "/data/benchq"
#: The runner's own HOME and each session's answers, inside it, so both outlive the container.
HOME_DIR = f"{STATE_DIR}/home"
RESULTS_DIR = f"{STATE_DIR}/research"
RUNNER = "/opt/bench/bench_queue_runner.py"
SEAL_CHECK = "/opt/bench/bench_runner/egress_check.py"
USER = "bench"
#: The session's PATH (bench_session passes it on claude's own command): the pinned binary first, and no
#: directory the session itself can write.
PATH = "/opt/claude/bin:/usr/local/bin:/usr/bin:/bin"
#: What the compose file sets, every one required.
REQUIRED = ("BENCH_BOX", "BENCH_LB_URL", "BENCH_METRICS_URL", "BENCH_EGRESS_PROXY", "BENCH_NO_PROXY")


class Refused(Exception):
    """The runner may not start, and this says why."""


def config(environ) -> dict[str, str]:
    missing = [k for k in REQUIRED if not (environ.get(k) or "").strip()]
    if missing:
        raise Refused(f"{', '.join(missing)} not set (docker-compose.yml sets them; .env.example documents "
                      f"the deployment's own)")
    return {k: environ[k].strip() for k in REQUIRED}


def runner_argv(cfg: dict[str, str]) -> list[str]:
    return ["--box", cfg["BENCH_BOX"], "--lb-url", cfg["BENCH_LB_URL"],
            "--metrics-url", cfg["BENCH_METRICS_URL"],
            "--state-dir", STATE_DIR, "--results-dir", RESULTS_DIR,
            "--https-proxy", cfg["BENCH_EGRESS_PROXY"], "--no-proxy", cfg["BENCH_NO_PROXY"],
            "--seal-check", SEAL_CHECK]


def leaked(environ) -> list[str]:
    return sorted(k for k in environ if k.startswith(("ANTHROPIC", "CLAUDE")))


def binary_matches(path: str, sha256: str) -> tuple[bool, str]:
    try:
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
    except OSError as exc:
        return False, f"{path} cannot be read: {exc}"
    if got != sha256:
        return False, f"{path} is not the pinned Claude Code {claude_pin.VERSION} ({got[:12]}… vs {sha256[:12]}…)"
    return True, f"Claude Code {claude_pin.VERSION}, sha256 {got[:12]}…"


def _drop_to(user: str) -> None:
    pw = pwd.getpwnam(user)
    if os.geteuid() == 0:
        os.chown(STATE_DIR, pw.pw_uid, pw.pw_gid)
        os.setgroups([])
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
    if os.geteuid() == 0 or os.getuid() == 0:
        raise Refused("still root after dropping privileges")
    if os.geteuid() != pw.pw_uid:
        raise Refused(f"running as uid {os.geteuid()}, not {user} ({pw.pw_uid})")


def main() -> int:
    def say(line: str) -> None:
        print(f"bench-runner: {line}", flush=True)

    try:
        cfg = config(os.environ)
        names = leaked(os.environ)
        if names:
            raise Refused(f"{', '.join(names)} in the container's environment — a backend variable belongs on "
                          f"claude's own command and nowhere else")
        ok, words = binary_matches(claude_pin.INSTALL_PATH, claude_pin.BINARY_SHA256)
        if not ok:
            raise Refused(words)
        say(words)
        _drop_to(USER)
        os.environ.update(HOME=HOME_DIR, USER=USER, LOGNAME=USER, PATH=PATH)
        for d in (HOME_DIR, RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
        say(f"running as {USER} (uid {os.getuid()}), HOME {HOME_DIR}; proving the seal")
        if egress_check.main([]) != 0:
            raise Refused("the seal check failed — the runner is not started, so nothing can launch claude")
    except Refused as exc:
        say(f"REFUSED: {exc}")
        return 3
    argv = [sys.executable, RUNNER, *runner_argv(cfg)]
    say(f"exec {' '.join(argv)}")
    os.execv(sys.executable, argv)
    return 0                                   # not reached


if __name__ == "__main__":
    sys.exit(main())
