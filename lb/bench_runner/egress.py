"""THE SEALED RUNNER'S ONE DOOR: an HTTPS proxy that refuses Anthropic and every private address, and two
fixed relays to the LB and the engine. Stdlib asyncio; runs as `nobody` in its own container.

The runner's docker network is `internal: true`: nothing on it has a route out. This process is the only
other member of that network (compose aliases `egress`, `lb`, `engine`), and it also sits on the network
where the LB and the engine are. It listens on the SEALED side only (`--bind egress-sealed-side`, an alias
that exists on the sealed network alone, resolved to its address there), so nothing on the upstream
network can use it. Not `egress`: compose registers a service's own name on every network it joins.

  * THE PROXY (`--proxy-port`). The runner's `claude` gets HTTPS_PROXY on its own command line
    (bench_session.py), so WebFetch's page fetches come here as `CONNECT host:443`. Each one is decided
    by policy.py — refused by name, then by every address the name resolves to — and an allowed tunnel
    is dialled to exactly the addresses that decision checked. Only CONNECT is spoken: a plain-HTTP
    request (which would carry its URL in the clear for us to forward) is answered 405. Every decision
    is one log line, so `docker logs` is the record of what the swarm reached and what it was refused.
  * THE RELAYS (`--relay LISTEN=HOST:PORT`). The LB's API (claude's ANTHROPIC_BASE_URL) and the engine's
    /metrics (bench_session's guard C) are on private networks the runner cannot reach. Each relay
    carries bytes to ONE target fixed at start; its address is checked per connection (policy.
    relay_problem: private, never Anthropic's) and dialled as checked. No idle limit: a turn the LB
    holds while the cluster restarts is silent for minutes and is still a turn.

Nothing here reads or forwards a credential: there is none on either side.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import socket
import sys

from bench_runner import policy

#: The CONNECT head must arrive within this, and fit in this many bytes.
HEAD_TIMEOUT_S = 10.0
HEAD_LIMIT = 16 * 1024
#: An allowed destination must answer within this.
CONNECT_TIMEOUT_S = 10.0
#: A tunnel in which neither side has sent a byte for this long is closed. A page fetch is seconds.
TUNNEL_IDLE_S = 600.0
#: Tunnels open at once. 60 agents fetch a few pages each; past this a client is told 503, not queued.
MAX_TUNNELS = 512
_CHUNK = 64 * 1024


def _reply(writer: asyncio.StreamWriter, code: int, phrase: str, words: str) -> None:
    body = (words.strip() + "\n").encode("utf-8", "replace")
    reason = "".join(c if 32 <= ord(c) < 127 else "?" for c in words)[:300]
    writer.write((f"HTTP/1.1 {code} {phrase}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                  f"Content-Length: {len(body)}\r\nConnection: close\r\nX-Egress-Reason: {reason}\r\n\r\n"
                  ).encode("ascii") + body)


async def _close(writer: asyncio.StreamWriter) -> None:
    try:
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    writer.close()
    try:
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


async def splice(r1, w1, r2, w2, idle_s: float | None) -> None:
    """Carry bytes both ways until both directions have ended (each end's EOF is passed on as a
    half-close), or — with `idle_s` — until neither side has sent anything for that long."""
    loop = asyncio.get_running_loop()
    last = [loop.time()]

    async def pipe(src, dst):
        try:
            while True:
                data = await src.read(_CHUNK)
                if not data:
                    break
                last[0] = loop.time()
                dst.write(data)
                await dst.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                if dst.can_write_eof():
                    dst.write_eof()
            except (OSError, RuntimeError):
                pass

    tasks = [asyncio.create_task(pipe(r1, w2)), asyncio.create_task(pipe(r2, w1))]
    try:
        while not all(t.done() for t in tasks):
            await asyncio.wait(tasks, timeout=idle_s / 4 if idle_s else None)
            if idle_s and loop.time() - last[0] > idle_s:
                break
    finally:
        for t in tasks:
            t.cancel()
        for w in (w1, w2):
            w.close()


def parse_relay(spec: str) -> tuple[int, str, int]:
    """`LISTEN=HOST:PORT` → (listen, host, port). Refused (ValueError) unless both ports are ports."""
    listen, sep, target = (spec or "").partition("=")
    split = policy.split_authority(target) if sep else None
    if not listen.isdigit() or split is None or not 0 <= int(listen) < 65536:
        raise ValueError(f"a relay is LISTEN=HOST:PORT, e.g. 8010=vllm-lb:8010 — not {spec!r}")
    return int(listen), split[0], split[1]


class Egress:
    """The proxy and the relays. `decide(host, port) -> policy.Decision` is blocking (it resolves), so it
    runs in the default executor; the tests hand in a stub."""

    def __init__(self, decide=None, log=print, *, max_tunnels: int = MAX_TUNNELS,
                 tunnel_idle_s: float = TUNNEL_IDLE_S, resolve=policy.system_resolve) -> None:
        self.decide = decide or (lambda host, port: policy.decide(host, port, resolve))
        self.resolve = resolve
        self.log = log
        self.max_tunnels = max_tunnels
        self.tunnel_idle_s = tunnel_idle_s
        self.tunnels = 0
        self.ports: dict = {}
        self.servers: list = []

    def say(self, line: str) -> None:
        """One line per decision. No clock of our own: `docker logs -t` stamps every line already."""
        self.log(line)

    async def start(self, bind: str, proxy_port: int, relays: list[tuple[int, str, int]]) -> None:
        proxy = await asyncio.start_server(self._proxy, bind, proxy_port, limit=HEAD_LIMIT)
        self.servers.append(proxy)
        self.ports["proxy"] = proxy.sockets[0].getsockname()[1]
        for listen, host, port in relays:
            server = await asyncio.start_server(functools.partial(self._relay, host=host, port=port),
                                                bind, listen)
            self.servers.append(server)
            self.ports[listen] = server.sockets[0].getsockname()[1]
            self.say(f"relay {bind}:{self.ports[listen]} -> {host}:{port}")
        self.say(f"proxy {bind}:{self.ports['proxy']} (CONNECT :443 only; Anthropic and private refused)")

    # ── the proxy ──────────────────────────────────────────────────────────

    async def _proxy(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = (writer.get_extra_info("peername") or ("?",))[0]
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT_S)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError,
                ConnectionError, OSError):
            writer.close()
            return
        parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            _reply(writer, 400, "Bad Request", "not an HTTP request")
            return await _close(writer)
        method, target = parts[0], parts[1]
        if method != "CONNECT":
            self.say(f"DENY {peer} {method} {target[:120]} — only CONNECT (HTTPS) is proxied")
            _reply(writer, 405, "Method Not Allowed", "only CONNECT to :443 is proxied")
            return await _close(writer)
        split = policy.split_authority(target)
        if split is None:
            _reply(writer, 400, "Bad Request", f"not host:port: {target[:120]}")
            return await _close(writer)
        host, port = split
        loop = asyncio.get_running_loop()
        decision = await loop.run_in_executor(None, self.decide, host, port)
        if not decision.allowed:
            self.say(f"DENY {peer} CONNECT {host}:{port} — {decision.reason}")
            _reply(writer, 403, "Forbidden", f"refused by the sealed runner's egress policy: {decision.reason}")
            return await _close(writer)
        if self.tunnels >= self.max_tunnels:
            self.say(f"BUSY {peer} CONNECT {host}:{port} — {self.tunnels} tunnels open")
            _reply(writer, 503, "Service Unavailable", f"{self.tunnels} tunnels are open; try again")
            return await _close(writer)
        self.tunnels += 1
        try:
            upstream, why = None, "no address"
            for address in decision.addresses:
                try:
                    upstream = await asyncio.wait_for(asyncio.open_connection(address, port),
                                                      CONNECT_TIMEOUT_S)
                    break
                except (OSError, asyncio.TimeoutError) as exc:
                    why = f"{address}: {type(exc).__name__}: {exc}"
            if upstream is None:
                self.say(f"FAIL {peer} CONNECT {host}:{port} — {why}")
                _reply(writer, 502, "Bad Gateway", f"{host}:{port} did not answer ({why})")
                return await _close(writer)
            self.say(f"ALLOW {peer} CONNECT {host}:{port} -> {address}")
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await splice(reader, writer, upstream[0], upstream[1], self.tunnel_idle_s)
        finally:
            self.tunnels -= 1

    # ── the relays ─────────────────────────────────────────────────────────

    async def _relay(self, reader, writer, *, host: str, port: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            addresses = await loop.run_in_executor(None, self.resolve, host, port)
        except OSError as exc:
            self.say(f"FAIL relay {host}:{port} — does not resolve: {exc}")
            writer.close()
            return
        problems = [p for p in (policy.relay_problem(a) for a in addresses) if p]
        if problems or not addresses:
            self.say(f"REFUSE relay {host}:{port} — {problems[0] if problems else 'no address'}")
            writer.close()
            return
        upstream = None
        for address in addresses:
            try:
                upstream = await asyncio.wait_for(asyncio.open_connection(address, port), CONNECT_TIMEOUT_S)
                break
            except (OSError, asyncio.TimeoutError) as exc:
                self.say(f"FAIL relay {host}:{port} ({address}) — {type(exc).__name__}: {exc}")
        if upstream is None:
            writer.close()
            return
        await splice(reader, writer, upstream[0], upstream[1], None)


def bind_address(name: str) -> str:
    """The address to listen on: an address as given, or the IPv4 a name resolves to — the alias this
    container has on the sealed network ONLY (compose: `egress-sealed-side`), so the upstream network never
    sees a listener."""
    try:
        socket.inet_aton(name)
        return name
    except OSError:
        return socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bind", required=True, help="the sealed network's own name for this container, or an address")
    p.add_argument("--proxy-port", type=int, required=True)
    p.add_argument("--relay", action="append", default=[], help="LISTEN=HOST:PORT (repeatable)")
    opts = p.parse_args(argv)
    try:
        relays = [parse_relay(spec) for spec in opts.relay]
        bind = bind_address(opts.bind)
    except (ValueError, OSError) as exc:
        print(f"egress: refusing to start: {exc}", file=sys.stderr, flush=True)
        return 2

    async def serve():
        e = Egress(log=lambda line: print(line, flush=True))
        await e.start(bind, opts.proxy_port, relays)
        await asyncio.Event().wait()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
