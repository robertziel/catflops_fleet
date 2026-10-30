"""WHAT THE SEALED RUNNER'S PROXY MAY OPEN — the whole egress policy, as pure functions.

The runner's network has no route out (`internal: true`). The egress proxy (egress.py) is its one way to
a public web page, so this file IS the seal's policy. The operator's decision, 2026-10-11:

    DENY  Anthropic: its names (*.anthropic.com, *.claude.ai, *.claude.com and the apexes) and its
          published IP ranges; every private/LAN range (10/8, 172.16/12, 192.168/16); Tailscale's CGNAT
          100.64/10; link-local and the 169.254.169.254 metadata address; loopback.
    ALLOW other public HTTPS.

WHY IT RESOLVES FOR ITSELF. The browser farm's tinyproxy filters the HOSTNAME by regex and says so in its
own config: "a public hostname that resolves to a private IP (DNS rebinding) is not caught here". The
same gap would let any name that resolves into Anthropic's range through. So a name is refused by NAME
first (never looked up), then looked up ONCE, and refused if ANY address it answers with is forbidden; the
proxy then connects to exactly the addresses checked here (`Decision.addresses`), never to a second
lookup that could answer differently.

WHY THIS IS CODE AND NOT CONFIGURATION. Every value in it is the same for every deployment, and none of
them may be loosened by one: a deny list that a server file could edit is a seal that a typo can open.
The deployment's own values (which network, which LB, which engine) are `${VAR:?}` in the compose file.

THE SOURCES:
  * Anthropic's ranges — https://platform.claude.com/docs/en/api/ip-addresses, read 2026-10-11:
    inbound IPv4 160.79.104.0/23 and IPv6 2607:6bc0::/48; outbound IPv4 160.79.104.0/21 (which contains
    the /23). The same page: "Claude Platform on AWS: the inbound endpoint
    (aws-external-anthropic.{region}.api.aws) resolves to AWS IP ranges" — refused by name.
  * RFC 1918 (10/8, 172.16/12, 192.168/16), RFC 6598 (100.64/10, Tailscale's CGNAT), RFC 3927
    (169.254/16), RFC 1122 (127/8, 0/8), RFC 5771 / 1112 (224/4, 240/4), RFC 4291 / 4193 (::1, fe80::/10,
    fc00::/7, ff00::/8). Anything else not globally routable (`ipaddress.is_global`) is refused as well,
    and an IPv6 address carrying an IPv4 one (mapped, 6to4, Teredo) is judged by the IPv4 inside it.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Callable

#: Anthropic's names: the API, the subscription, the console — the apex and every name under it.
ANTHROPIC_DOMAINS = ("anthropic.com", "claude.ai", "claude.com")
#: Claude Platform on AWS's inbound endpoint, `aws-external-anthropic.{region}.api.aws` (the page above).
ANTHROPIC_HOST_PREFIXES = ("aws-external-anthropic.",)
#: Anthropic's published ranges (the page above): outbound /21 ⊇ inbound /23, and the inbound IPv6 /48.
ANTHROPIC_NETWORKS = tuple(ipaddress.ip_network(n) for n in ("160.79.104.0/21", "2607:6bc0::/48"))
#: What is never a public web server, named so that a refusal can say which range it was.
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",       # RFC 1918: every LAN
    "100.64.0.0/10",                                        # RFC 6598: Tailscale's CGNAT
    "169.254.0.0/16",                                       # link-local, incl. 169.254.169.254 metadata
    "127.0.0.0/8", "0.0.0.0/8",                             # loopback, "this network"
    "224.0.0.0/4", "240.0.0.0/4",                           # multicast; reserved, incl. broadcast
    "::1/128", "::/128", "fe80::/10", "fc00::/7", "ff00::/8"))  # IPv6 loopback, link-local, ULA, multicast
#: Names that only exist inside a network: mDNS (RFC 6762), RFC 6761's localhost/invalid, RFC 8375's
#: home.arpa, Tor, and the suffixes LANs and docker setups give themselves — refused by name, unresolved.
LOCAL_SUFFIXES = ("local", "localhost", "internal", "lan", "home.arpa", "home", "intranet", "corp",
                  "invalid", "onion")
#: The one port a tunnel may name. Claude Code's WebFetch upgrades http:// to https:// itself, so plain
#: HTTP is never proxied — and an HTTPS tunnel to any other port is not a web page.
CONNECT_PORTS = frozenset({443})
#: A public DNS name: two or more labels of letters, digits and inner hyphens — nothing else. One label is
#: an in-network name (the LB, the engine, a box); `_`, empty labels and numeric tricks are not DNS.
_NAME = re.compile(r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
                   r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$")

Resolver = Callable[[str, int], list]


@dataclass(frozen=True)
class Decision:
    """The answer for one CONNECT: allowed or not, why, and — when allowed — the addresses checked, which
    are the only ones the proxy may connect to."""
    allowed: bool
    reason: str
    addresses: tuple[str, ...] = ()


def address_problem(address) -> str | None:
    """Why this address may not be opened, or None for a public one."""
    ip = ipaddress.ip_address(address)
    if ip.version == 6:
        inner = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if inner is not None:
            problem = address_problem(inner)
            if problem:
                return f"{ip} carries {inner}: {problem}"
    for net in ANTHROPIC_NETWORKS:
        if ip in net:
            return f"{ip} is in Anthropic's published range {net}"
    for net in PRIVATE_NETWORKS:
        if ip in net:
            return f"{ip} is in the private or reserved range {net}"
    if not ip.is_global:
        return f"{ip} is not a globally routable address"
    return None


def name_problem(host: str) -> str | None:
    """Why this (lower-case, dot-stripped) name may not even be looked up, or None."""
    for domain in ANTHROPIC_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return f"{host} is Anthropic's ({domain})"
    for prefix in ANTHROPIC_HOST_PREFIXES:
        if host.startswith(prefix):
            return f"{host} is Anthropic's (Claude Platform on AWS endpoint)"
    if not _NAME.match(host):
        return f"{host!r} is not a public DNS name (one label, or characters DNS does not use)"
    for suffix in LOCAL_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return f"{host} is a local-network name (.{suffix})"
    return None


def _literal(host: str):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def decide(host: str, port: int, resolve: Resolver) -> Decision:
    """The decision for `CONNECT host:port`. `resolve(host, port)` returns the name's addresses or raises
    OSError — `system_resolve` in the proxy, a table in the tests."""
    host = (host or "").strip().lower().rstrip(".")
    if port not in CONNECT_PORTS:
        return Decision(False, f"port {port} is not HTTPS: only 443 may be tunnelled")
    literal = _literal(host)
    if literal is not None:
        problem = address_problem(literal)
        if problem:
            return Decision(False, problem)
        return Decision(True, f"{literal} is a public address", (str(literal),))
    problem = name_problem(host)
    if problem:
        return Decision(False, problem)
    try:
        addresses = [str(ipaddress.ip_address(a)) for a in resolve(host, port)]
    except (OSError, ValueError) as exc:
        return Decision(False, f"{host} does not resolve: {exc}")
    if not addresses:
        return Decision(False, f"{host} does not resolve to any address")
    for address in addresses:
        problem = address_problem(address)
        if problem:
            return Decision(False, f"{host} resolves to {problem}")
    return Decision(True, f"{host} resolves to public {', '.join(addresses)}", tuple(addresses))


def relay_problem(address) -> str | None:
    """Why a FIXED relay may not dial this address, or None. A relay carries the runner to the LB or the
    engine — both on private networks — so its target must be private, and never Anthropic's."""
    ip = ipaddress.ip_address(address)
    for net in ANTHROPIC_NETWORKS:
        if ip in net:
            return f"{ip} is in Anthropic's published range {net}"
    if ip.is_global:
        return f"{ip} is a public address: a relay only reaches the LB and the engine on private networks"
    return None


def split_authority(target: str) -> tuple[str, int] | None:
    """`host:port` or `[v6]:port` → (host, port), or None for anything else. Strict on purpose: the port is
    a decimal number, the host has no colon unless bracketed."""
    target = (target or "").strip()
    if target.startswith("["):
        end = target.find("]")
        if end == -1 or target[end + 1:end + 2] != ":":
            return None
        host, port = target[1:end], target[end + 2:]
    else:
        host, sep, port = target.rpartition(":")
        if not sep or ":" in host:
            return None
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return host.lower(), int(port)


def system_resolve(host: str, port: int) -> list[str]:
    """Every address the system resolver answers with, in its order, each once."""
    out: list[str] = []
    for *_, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        if sockaddr[0] not in out:
            out.append(sockaddr[0])
    return out
