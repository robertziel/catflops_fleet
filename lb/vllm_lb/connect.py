"""HOW AN API CLIENT REACHES THIS LB — the facts only the deployment can state, for the panel's Connect dialog.

Operator, 2026-10-12: "Can we include somewhere connect button, top navbar? Will show how to connect agent to this
vllm". The dialog (static/gateway.js, "connect an agent") builds every snippet from the page's own origin, the LB's
live /v1/models and what the address the page was opened at answers at /auth/session — a gate in front answers it
(`auth: "false"` without sign-in, a session or 401 with it), the LB itself 404s it. What none of that can tell is
this module's, in /lb/status `connect`:

  tunnel   LB_CONNECT_TUNNEL: the LB's address ON ITS HOST, host:port — what compose publishes it at. Anyone with
           ssh to that host reaches the LB, with no credential, through `ssh -N -L <port>:<tunnel> <host>`; that
           is the route that works whatever door is in front (the operator's own `fullclaudevllm site-b`). Blank =
           not offered. Something that is not host:port refuses the start, naming the variable.
  basic    LB_CONNECT_BASIC: the edge in front of the panel's sign-in also takes HTTP Basic on the inference API
           (the edge's vhost: `$vllm_machine_path` GET /v1/models + POST /v1/*, deploy/nginx-edge-vllm.conf). The
           LB cannot see its edge, so the deployment says it: 1/true/yes or 0/false/no/blank, anything else refuses.
  first_byte_timeout_s, stall_timeout_s
           the LB's own budgets, which a client must out-wait — fullclaudevllm's API_TIMEOUT_MS and
           CLAUDE_STREAM_IDLE_TIMEOUT_MS are these, in ms.

WHY a sign-in door needs either: the gate (a sign-in service outside this repository) reads ONLY its session cookie, nginx drops
`Authorization` before it asks the gate (the deployment's edge vhost), and the
issuer mints no token without a browser (its GET /token requires the app's Origin). No credential is ever stated
here or on the page: a credential is the client's to fill in.
"""
from __future__ import annotations

import re

#: A host: a DNS name or IPv4 address, or a bracketed IPv6 literal. No scheme, path or space — it is the remote end of
#: an `ssh -L`, not a URL.
_HOST = re.compile(r"^(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)$")
_YES = ("1", "true", "yes")
_NO = ("0", "false", "no", "")


def parse_tunnel(raw: str | None) -> str:
    """LB_CONNECT_TUNNEL → `host:port`, or "" when blank. Anything else is refused with what was given."""
    value = (raw or "").strip()
    if not value:
        return ""
    host, sep, port = value.rpartition(":")
    if not sep or not _HOST.match(host) or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"LB_CONNECT_TUNNEL={raw!r} must be host:port — the LB's address on its host, as compose "
                         f"publishes it (e.g. 127.0.0.1:<port>) — or blank for no SSH route")
    return f"{host}:{int(port)}"


def parse_flag(name: str, raw: str | None) -> bool:
    """A stated yes or no: 1/true/yes, or 0/false/no/blank. A typo is refused, never read as either."""
    value = (raw or "").strip().lower()
    if value in _YES:
        return True
    if value in _NO:
        return False
    raise ValueError(f"{name}={raw!r} must be 1/true/yes or 0/false/no (blank = no)")


def connect_view(cfg) -> dict:
    """/lb/status `connect`: the deployment's routes and the budgets a client must out-wait."""
    return {"tunnel": cfg.connect_tunnel, "basic": cfg.connect_basic,
            "first_byte_timeout_s": cfg.first_byte_timeout_s, "stall_timeout_s": cfg.stall_timeout_s}
