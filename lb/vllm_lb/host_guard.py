"""THE HOST ALLOWLIST — which names this HTTP server answers to, and which origins may change its state.

WHY (release audit, 1.0.0-beta.2): a server bound to 127.0.0.1 is NOT out of a browser's reach. A page the operator
opens on any site can DNS-REBIND its own hostname to 127.0.0.1: the browser then sends that page's requests to this
port, and — because the page's origin never changed — treats them as SAME-ORIGIN. It can read /lb/status and POST
`Content-Type: application/json` to every control route (the cluster switch, a kill, the swarm, an activation): the
content-type rule (proxy.py `_is_json_type`) stops a CROSS-origin form, not a same-origin fetch. The one thing a
rebinding page cannot forge is the name it was loaded under: the browser sends it as `Host` (and as `Origin` on a
write). So every request is checked against the names this server is really reached by, and nothing else.

THE ALLOWED SET is two parts, and neither is guessed:
  * the LISTENER's own loopback names — `127.0.0.1`, `localhost` and `[::1]` at the port the server is bound to
    (`listener_authorities`; plus the bound address itself when it is one specific address, not a wildcard). A
    healthcheck, a host-side client and a host-network container reach it this way.
  * what the DEPLOYMENT names in one REQUIRED environment variable (LB_ALLOWED_HOSTS for the gateway,
    POWER_ALLOWED_HOSTS for the power panel): the Host its reverse proxy / tunnel presents and the docker-network
    names other containers dial it by (`vllm-lb:8010`). Comma-separated `host[:port]`. There is NO default: absent,
    the server refuses to boot naming the variable. A deployment reached by nothing but loopback says so with the one
    word `loopback` — an empty value is refused too, because "nothing" must be a choice, never an omission
    (AGENTS.md: no defaults for deployment values).

THE PORT RULE. An entry with a port matches that port exactly. An entry WITHOUT one is a name behind a proxy on the
scheme's default port: it matches a Host with no port, `:80` or `:443` (nginx `proxy_set_header Host $host` sends the
name alone). A Host with no port matches an entry on :80/:443 the same way. Any other port is a different server.

THE ORIGIN RULE. A request that can change state (anything but GET/HEAD/OPTIONS) carrying an `Origin` must carry one
whose host[:port] is allowed by the same rule; `Origin: null` is refused. A request with no Origin is not a browser's
cross-site write (browsers send Origin on every non-GET), so it is judged by its Host alone. A server that already
checks Origin against an exact list of its own (the power panel: POWER_ORIGINS) uses `host_refusal` alone, so one
value never has two rules.

REFUSALS: a Host not allowed (or missing — every HTTP/1.1 request has one, a browser's always does) is 421 Misdirected
Request; a foreign Origin on a write is 403. Each body is JSON naming the value and the variable to add it to, and
each distinct Host is logged ONCE (allowed or refused), so the log says exactly which names reach this server and
which to add — bounded, so a client spraying names cannot grow it without limit. The reply never lists the allowed
names: a rebinding page can read it, and the deployment's other names are none of its business.

This file is SHARED, byte for byte, by lb/vllm_lb/host_guard.py and power/fleet_power/host_guard.py (two services, two
images, stdlib only — no package they could both import). deploy/tests/test_host_guard_copies.py fails if they
differ: change one, copy it to the other.
"""
from __future__ import annotations

import ipaddress
import re
import threading
from dataclasses import dataclass
from typing import Callable

#: The one spelling of "this server is reached by its loopback names and nothing else".
LOOPBACK = "loopback"
#: The ports a name behind a proxy is reached on when its Host carries no port (http, https).
DEFAULT_PORTS = (80, 443)
#: The loopback names every listener answers to at its own port.
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
#: Methods that change nothing: their Origin is not judged (a browser's CORS keeps a cross-origin page from READING
#: the answer; this server grants no cross-origin access).
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
#: How many distinct Host values are logged at first sight before the log stops saying new ones (and says so once).
SEEN_CAP = 256

_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]*[a-z0-9])?$")
_WILDCARDS = ("0.0.0.0", "::", "")


@dataclass(frozen=True)
class Authority:
    """A `host[:port]`: lowercased name (an IPv6 literal keeps its brackets) and its port, None when not given."""
    name: str
    port: int | None

    def __str__(self) -> str:
        return self.name if self.port is None else f"{self.name}:{self.port}"


def parse_authority(text: str) -> Authority:
    """`host`, `host:port`, `[v6]` or `[v6]:port` → Authority. ValueError on anything else — a scheme, a path, a
    user@, a wildcard, an empty name or a port outside 1..65535."""
    raw = (text or "").strip()
    if not raw or any(c in raw for c in "/@*?# \t,"):
        raise ValueError(f"{text!r} is not host[:port]")
    if raw.startswith("["):
        close = raw.find("]")
        if close < 0:
            raise ValueError(f"{text!r} is not host[:port]: an unclosed [")
        name, rest = raw[:close + 1].lower(), raw[close + 1:]
        try:
            ipaddress.IPv6Address(name[1:-1])
        except ValueError:
            raise ValueError(f"{text!r} is not host[:port]: {name} is not an IPv6 address") from None
        if rest and not rest.startswith(":"):
            raise ValueError(f"{text!r} is not host[:port]")
        port_text = rest[1:] if rest else None
    else:
        name, sep, port_text = raw.lower().partition(":")
        if not sep:
            port_text = None
        if not _NAME.match(name):
            raise ValueError(f"{text!r} is not host[:port]: {name!r} is not a host name")
    if port_text is None:
        return Authority(name, None)
    if not port_text.isdigit() or not port_text.isascii() or not 1 <= int(port_text) <= 65535:
        raise ValueError(f"{text!r} is not host[:port]: the port must be 1..65535")
    return Authority(name, int(port_text))


def parse_allowed(env_name: str, raw: str | None) -> tuple[Authority, ...]:
    """The deployment's names from `env_name`. REQUIRED: absent or blank refuses (ValueError naming it); the one word
    `loopback` is the stated choice of none beyond the listener's own loopback names (an empty tuple)."""
    hint = (f"set {env_name} to the comma-separated host[:port] names this server is reached by — the Host your "
            f"reverse proxy or tunnel presents, and the docker-network name other containers dial (e.g. "
            f"vllm-lb:8010) — or to `{LOOPBACK}` when nothing reaches it but its own loopback address")
    if raw is None:
        raise ValueError(f"{env_name} is required: {hint}")
    value = raw.strip()
    if not value:
        raise ValueError(f"{env_name} is empty: {hint} (an empty value is never read as `{LOOPBACK}`)")
    if value.lower() == LOOPBACK:
        return ()
    items = [s.strip() for s in value.split(",")]
    if any(not s for s in items):
        raise ValueError(f"{env_name}={raw!r} has an empty entry: {hint}")
    if any(s.lower() == LOOPBACK for s in items):
        raise ValueError(f"{env_name}={raw!r}: `{LOOPBACK}` stands alone — the loopback names are always allowed, "
                         f"so list only the others")
    out: list[Authority] = []
    for item in items:
        try:
            out.append(parse_authority(item))
        except ValueError as exc:
            raise ValueError(f"{env_name}: {exc} — {hint}") from None
    return tuple(dict.fromkeys(out))


def listener_authorities(listen_host: str, port: int) -> tuple[Authority, ...]:
    """The names the listener itself answers to: the loopback names at its bound port, and — bound to one specific
    address rather than a wildcard — that address too."""
    out = [Authority(name, port) for name in LOOPBACK_NAMES]
    host = (listen_host or "").strip().lower()
    if host not in _WILDCARDS:
        literal = f"[{host}]" if ":" in host and not host.startswith("[") else host
        own = Authority(literal, port)
        if own not in out:
            out.append(own)
    return tuple(out)


def matches(entry: Authority, seen: Authority) -> bool:
    """THE PORT RULE (module docstring): same name, and the same port — or one side portless and the other on a
    default port (or portless too)."""
    if entry.name != seen.name:
        return False
    if entry.port == seen.port:
        return True
    if entry.port is None:
        return seen.port in DEFAULT_PORTS
    if seen.port is None:
        return entry.port in DEFAULT_PORTS
    return False


def origin_authority(origin: str) -> Authority | None:
    """`scheme://host[:port]` → its Authority with the scheme's default port made explicit; None for `null` or
    anything that is not an http(s) origin."""
    scheme, sep, rest = (origin or "").strip().partition("://")
    scheme = scheme.lower()
    if not sep or scheme not in ("http", "https") or not rest or "/" in rest:
        return None
    try:
        a = parse_authority(rest)
    except ValueError:
        return None
    return a if a.port is not None else Authority(a.name, 80 if scheme == "http" else 443)


class HostGuard:
    """One server's allowlist: `refusal(method, host, origin)` is None for a request it may serve, else
    `(status, body)` for the reply. Thread-safe; logs each distinct Host once through `say`."""

    def __init__(self, env_name: str, allowed: tuple[Authority, ...], listen_host: str, port: int,
                 say: Callable[[str], None]) -> None:
        self.env_name = env_name
        self.allowed = tuple(dict.fromkeys(listener_authorities(listen_host, port) + tuple(allowed)))
        self._say = say
        self._seen: set[str] = set()
        self._seen_full = False
        self._lock = threading.Lock()

    def allows(self, seen: Authority) -> bool:
        return any(matches(entry, seen) for entry in self.allowed)

    def _first_sight(self, host: str, verdict: str) -> None:
        with self._lock:
            if host in self._seen:
                return
            if len(self._seen) >= SEEN_CAP:
                if self._seen_full:
                    return
                self._seen_full = True
                line = f"[host] {SEEN_CAP} distinct Host values seen: no further first sightings are logged"
            else:
                self._seen.add(host)
                line = (f"[host] first request with Host {host!r}: {verdict}"
                        + ("" if verdict == "allowed" else f" — if a proxy, tunnel or container reaches this server "
                                                            f"by that name, add it to {self.env_name}"))
        self._say(line)

    def refusal(self, method: str, host: str | None, origin: str | None) -> tuple[int, dict] | None:
        """THE WHOLE RULE: the Host first, then — for a write — the Origin."""
        return self.host_refusal(host) or self.origin_refusal(method, origin)

    def host_refusal(self, host: str | None) -> tuple[int, dict] | None:
        """421 for a Host that is not allowed (or missing); None, and its first sighting logged, for one that is."""
        shown = (host or "").strip()
        try:
            seen = parse_authority(shown) if shown else None
        except ValueError:
            seen = None
        if seen is None or not self.allows(seen):
            self._first_sight(shown or "(none)", "REFUSED (421)")
            return 421, {"error": {
                "type": "misdirected_host", "host": shown or None,
                "message": (f"this server does not answer to Host {shown!r}: " if shown else
                            "a request with no Host header is refused: ")
                + f"the names it answers to are its listener's loopback names and {self.env_name}; if a reverse "
                  f"proxy, tunnel or container reaches it by this name, add that name to {self.env_name}"}}
        self._first_sight(shown, "allowed")
        return None

    def origin_refusal(self, method: str, origin: str | None) -> tuple[int, dict] | None:
        """403 for a write whose Origin is not an allowed name; None for a read, no Origin, or an allowed one. A server
        with an exact origin list of its own (the power panel's POWER_ORIGINS) keeps that one rule instead."""
        if origin is None or method.upper() in SAFE_METHODS:
            return None
        seen_origin = origin_authority(origin)
        if seen_origin is not None and self.allows(seen_origin):
            return None
        return 403, {"error": {
            "type": "cross_origin", "origin": origin,
            "message": (f"a state change from Origin {origin!r} is refused: only a page served under a name "
                        f"in {self.env_name} (or the listener's loopback names) may make one")}}
