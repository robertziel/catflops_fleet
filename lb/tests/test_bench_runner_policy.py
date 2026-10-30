"""The sealed swarm runner's egress policy (lb/bench_runner/policy.py): what its proxy may open.

The runner's network has no route out. Its one way to a public page is the egress proxy, and this
policy is everything that proxy decides. The operator's list (2026-10-11): REFUSE Anthropic — its
domains and its published IP ranges — every private/LAN range, Tailscale's CGNAT, link-local and the
169.254 metadata address, and loopback; ALLOW other public HTTPS.

Every decision below is made against a STUB resolver, so a name's addresses are exactly what the test
says they are: the rebinding cases (a harmless-looking name that resolves somewhere forbidden) are the
ones a hostname-only filter cannot catch, and the reason this proxy resolves for itself.
"""
from __future__ import annotations

import ipaddress

import pytest

from bench_runner import policy


def resolver(table: dict[str, list[str]]):
    def resolve(host: str, port: int) -> list[str]:
        if host not in table:
            raise OSError(f"[Errno -2] Name or service not known: {host}")
        return list(table[host])
    return resolve


PUBLIC = resolver({"example.com": ["93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"]})


def test_the_published_anthropic_ranges_are_the_ones_on_anthropics_page():
    """https://platform.claude.com/docs/en/api/ip-addresses, read 2026-10-11: inbound 160.79.104.0/23 and
    2607:6bc0::/48, outbound 160.79.104.0/21 — the /21 contains the /23."""
    nets = {str(n) for n in policy.ANTHROPIC_NETWORKS}
    assert nets == {"160.79.104.0/21", "2607:6bc0::/48"}
    assert ipaddress.ip_network("160.79.104.0/23").subnet_of(ipaddress.ip_network("160.79.104.0/21"))


@pytest.mark.parametrize("host", ["api.anthropic.com", "anthropic.com", "console.anthropic.com",
                                  "statsig.anthropic.com", "API.Anthropic.COM", "api.anthropic.com.",
                                  "claude.ai", "www.claude.ai", "claude.com", "platform.claude.com",
                                  "aws-external-anthropic.us-east-1.api.aws"])
def test_every_anthropic_name_is_refused_before_it_is_even_resolved(host):
    asked: list[str] = []

    def resolve(h, p):
        asked.append(h)
        return ["93.184.215.14"]
    d = policy.decide(host, 443, resolve)
    assert d.allowed is False and "Anthropic" in d.reason, d
    assert asked == [], "an Anthropic name is refused by name; it is never looked up"


@pytest.mark.parametrize("host", ["notanthropic.com", "anthropic.com.evil.test", "claude.ai.example.org",
                                  "myclaude.com"])
def test_a_name_that_only_looks_like_anthropics_is_judged_by_its_addresses(host):
    d = policy.decide(host, 443, resolver({host: ["8.8.8.8"]}))
    assert d.allowed is True, d


@pytest.mark.parametrize("address", ["160.79.104.10", "160.79.105.255", "160.79.111.1", "2607:6bc0::10"])
def test_an_anthropic_address_is_refused_as_a_literal_and_behind_any_name(address):
    literal = policy.decide(address, 443, resolver({}))
    assert literal.allowed is False and "Anthropic" in literal.reason, literal
    rebound = policy.decide("innocent.example.net", 443, resolver({"innocent.example.net": [address]}))
    assert rebound.allowed is False and "Anthropic" in rebound.reason, rebound


@pytest.mark.parametrize("address", [
    "10.0.0.1", "192.0.2.2", "172.16.0.1", "172.25.0.3", "172.31.255.254", "192.168.0.1",
    "192.168.77.210",                                   # a home LAN address, as a shape
    "100.64.0.1", "100.96.170.20", "100.127.255.254",   # Tailscale's CGNAT (RFC 6598)
    "169.254.169.254", "169.254.0.1",                   # metadata, link-local
    "127.0.0.1", "127.8.8.8", "0.0.0.0",                # loopback, "this network"
    "224.0.0.251", "255.255.255.255",                   # multicast, broadcast
    "::1", "fe80::1", "fd12:3456::1", "fc00::1",        # IPv6 loopback, link-local, ULA
    "::ffff:10.0.0.1", "::ffff:169.254.169.254",        # IPv4-mapped IPv6 of a private address
    "2002:a00:1::1",                                    # 6to4 of 10.0.0.1
])
def test_every_private_reserved_or_metadata_address_is_refused_literal_or_resolved(address):
    literal = policy.decide(address, 443, resolver({}))
    assert literal.allowed is False, literal
    rebound = policy.decide("rebind.example.net", 443, resolver({"rebind.example.net": [address]}))
    assert rebound.allowed is False, rebound


def test_one_forbidden_address_among_public_ones_refuses_the_whole_name():
    """A name answering with a public and a private address is not half-allowed: the proxy would pick
    one, and the one it picks is not the policy's to leave to chance."""
    d = policy.decide("mixed.example.net", 443,
                      resolver({"mixed.example.net": ["8.8.8.8", "192.168.1.10"]}))
    assert d.allowed is False and "192.168.1.10" in d.reason


@pytest.mark.parametrize("host", ["localhost", "lb", "engine", "egress", "vllm-lb", "box-a",
                                  "metadata", "printer.local", "metadata.google.internal",
                                  "router.lan", "nas.home.arpa", "app.localhost", "2130706433", "0x7f000001",
                                  "under_score.example.com", "-bad.example.com", "a..b.example.com",
                                  "spa ce.example.com", ""])
def test_a_name_that_is_not_a_public_dns_name_is_refused_unresolved(host):
    """One label is an in-network name (the LB, the engine, a box) — refused without asking anyone what it
    is. So is anything that is not plain DNS: no underscores, no empty labels, no numeric tricks."""
    asked: list[str] = []

    def resolve(h, p):
        asked.append(h)
        return ["8.8.8.8"]
    d = policy.decide(host, 443, resolve)
    assert d.allowed is False, d
    assert asked == [], f"{host!r} was looked up"


def test_a_dotted_numeric_short_form_is_judged_by_where_it_really_points():
    """`127.1` is not an address to Python's ipaddress but IS 127.0.0.1 to the resolver: it is refused for
    the address it resolves to."""
    d = policy.decide("127.1", 443, resolver({"127.1": ["127.0.0.1"]}))
    assert d.allowed is False and "127.0.0.1" in d.reason


@pytest.mark.parametrize("port", [80, 22, 8010, 8443, 0, 65535])
def test_only_https_on_443_may_be_tunnelled(port):
    d = policy.decide("example.com", port, PUBLIC)
    assert d.allowed is False and "443" in d.reason


def test_a_public_name_is_allowed_with_exactly_the_addresses_that_were_checked():
    d = policy.decide("example.com", 443, PUBLIC)
    assert d.allowed is True
    assert d.addresses == ("93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"), \
        "the proxy connects to what was checked, never to a second lookup"
    assert policy.decide("8.8.8.8", 443, resolver({})).allowed is True, "a public address literal is public"


def test_a_name_that_does_not_resolve_is_refused_and_says_so():
    d = policy.decide("nowhere.example.net", 443, resolver({}))
    assert d.allowed is False and "resolve" in d.reason


@pytest.mark.parametrize("target, expected", [
    ("example.com:443", ("example.com", 443)), ("[2607:6bc0::10]:443", ("2607:6bc0::10", 443)),
    ("EXAMPLE.com:443", ("example.com", 443)), ("example.com", None), ("example.com:https", None),
    ("[::1]", None), (":443", None), ("example.com:99999", None), ("a:b:c:443", None)])
def test_a_connect_target_is_split_strictly(target, expected):
    assert policy.split_authority(target) == expected


@pytest.mark.parametrize("address, private", [("172.25.0.3", True), ("192.0.2.2", True), ("192.168.1.5", True),
                                              ("8.8.8.8", False), ("160.79.104.10", False)])
def test_a_relay_target_must_be_a_private_address_and_never_anthropics(address, private):
    """The two fixed relays carry the runner to the LB and to the engine, both on private networks. A relay
    configured towards a public address — Anthropic's above all — would be a hole the proxy never sees."""
    assert (policy.relay_problem(address) is None) is private