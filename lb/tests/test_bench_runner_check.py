"""The seal proof (lb/bench_runner/egress_check.py): the check the sealed runner passes before it starts,
and again before EVERY session it launches.

It must FAIL — and so stop claude from ever starting — when any of these is true:
  * a credential is within reach: an OAuth or API-key file in HOME, a key helper in a settings file, or a
    backend/cloud credential variable in the environment;
  * Anthropic (or anything outside) is reachable directly: the network is not the sealed one;
  * the egress proxy lets Anthropic, a private/LAN/metadata/loopback address or an in-network name
    through: anything but a 403 is a failure, including the proxy not answering at all;
  * the LB does not answer through its relay (a seal with no LB behind it is not a runner).

Here every outside thing is a local stand-in — a fake proxy answering what the test says, a fake LB —
and the direct dial is injected, because this suite's own container DOES have a route out.
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench_runner import egress_check as check


class FakeProxy:
    """Answers every CONNECT with `status` (403 unless told otherwise), or with `per_target[target]`."""

    def __init__(self, status: int = 403, per_target: dict | None = None) -> None:
        self.status, self.per_target, self.asked = status, per_target or {}, []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                head = b""
                while b"\r\n\r\n" not in head:
                    data = conn.recv(4096)
                    if not data:
                        break
                    head += data
                target = head.split(b" ", 2)[1].decode() if head.count(b" ") >= 2 else "?"
                self.asked.append(target)
                code = self.per_target.get(target, self.status)
                conn.sendall(f"HTTP/1.1 {code} X\r\nX-Egress-Reason: test says {code}\r\n"
                             f"Content-Length: 0\r\n\r\n".encode())

    def close(self) -> None:
        self.sock.close()


class FakeLb:
    def __init__(self, status: int = 200) -> None:
        code = status

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                body = b'{"ok": true}' if code == 200 else b"no"
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def refused(addr, timeout=None):
    raise OSError(101, "Network is unreachable")


def connected(addr, timeout=None):
    class _S:
        def close(self):
            pass
    return _S()


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    h.mkdir()
    return h


def run(proxy, lb, home, *, dial=refused, environ=None, settings=()):
    return check.run(proxy_url=proxy.url, lb_url=lb.url, home=str(home), dial=dial,
                     environ={} if environ is None else environ, settings_files=settings, timeout=2.0)


def test_a_sealed_runner_passes_every_check(home):
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        results = run(proxy, lb, home)
        assert results and all(r.ok for r in results), [r for r in results if not r.ok]
        kinds = {r.kind for r in results}
        assert kinds == {"credentials", "direct", "proxy", "lb"}
        targets = " ".join(r.target for r in results if r.kind == "proxy")
        for must in ("api.anthropic.com:443", "claude.ai:443", "160.79.104.10:443", "192.168.0.1:443",
                     "100.64.0.1:443", "169.254.169.254:443", "127.0.0.1:443", "localhost:443"):
            assert must in targets, f"the proxy is never asked about {must}"
        assert set(proxy.asked) >= {"api.anthropic.com:443", "192.168.0.1:443"}
        direct = " ".join(r.target for r in results if r.kind == "direct")
        assert "api.anthropic.com:443" in direct and "160.79.104.10:443" in direct
    finally:
        proxy.close()
        lb.close()


def test_anthropic_reachable_directly_fails_the_seal(home):
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        results = run(proxy, lb, home, dial=connected)
        bad = [r for r in results if not r.ok]
        assert bad and all(r.kind == "direct" for r in bad)
        assert any("api.anthropic.com" in r.target for r in bad)
    finally:
        proxy.close()
        lb.close()


@pytest.mark.parametrize("target, status", [("api.anthropic.com:443", 200), ("192.168.0.1:443", 200),
                                            ("169.254.169.254:443", 502), ("claude.ai:443", 407)])
def test_anything_but_a_403_from_the_proxy_fails_the_seal(home, target, status):
    """A 200 is the hole itself; a 502 means the proxy TRIED to dial what policy must refuse unasked."""
    proxy, lb = FakeProxy(403, {target: status}), FakeLb(200)
    try:
        bad = [r for r in run(proxy, lb, home) if not r.ok]
        assert [r.target for r in bad] == [target], bad
    finally:
        proxy.close()
        lb.close()


def test_a_proxy_that_does_not_answer_fails_the_seal(home):
    lb = FakeLb(200)
    proxy = FakeProxy(403)
    proxy.close()
    try:
        bad = [r for r in run(proxy, lb, home) if not r.ok]
        assert bad and all(r.kind == "proxy" for r in bad)
    finally:
        lb.close()


def test_an_lb_that_does_not_answer_fails_the_check(home):
    proxy, lb = FakeProxy(403), FakeLb(500)
    try:
        bad = [r for r in run(proxy, lb, home) if not r.ok]
        assert [r.kind for r in bad] == ["lb"]
    finally:
        proxy.close()
        lb.close()


@pytest.mark.parametrize("relative, body", [
    (".claude/.credentials.json", '{"claudeAiOauth": {"accessToken": "x"}}'),
    (".claude.json", json.dumps({"oauthAccount": {"emailAddress": "someone@example.test"}})),
    (".claude.json", json.dumps({"primaryApiKey": "sk-ant-FAKE"})),
    (".aws/credentials", "[default]\naws_access_key_id = FAKE\n"),
    (".config/gcloud/application_default_credentials.json", "{}"),
])
def test_a_credential_file_in_home_fails_the_seal(home, relative, body):
    path = home / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        bad = [r for r in run(proxy, lb, home) if not r.ok]
        assert [r.kind for r in bad] == ["credentials"], bad
        assert "sk-ant" not in bad[0].said, "a credential is named, never printed"
    finally:
        proxy.close()
        lb.close()


def test_a_harmless_claude_json_is_not_a_credential(home):
    (home / ".claude.json").write_text(json.dumps({"numStartups": 3, "projects": {}}))
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        assert all(r.ok for r in run(proxy, lb, home))
    finally:
        proxy.close()
        lb.close()


@pytest.mark.parametrize("name", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                                  "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "AWS_ACCESS_KEY_ID",
                                  "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                                  "AWS_BEARER_TOKEN_BEDROCK", "GOOGLE_APPLICATION_CREDENTIALS"])
def test_a_credential_variable_in_the_runners_environment_fails_the_seal(home, name):
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        bad = [r for r in run(proxy, lb, home, environ={name: "FAKE-VALUE"}) if not r.ok]
        assert [r.kind for r in bad] == ["credentials"] and name in bad[0].said
        assert "FAKE-VALUE" not in bad[0].said
    finally:
        proxy.close()
        lb.close()


def test_a_key_helper_in_any_settings_file_fails_the_seal(home, tmp_path):
    settings = tmp_path / "managed-settings.json"
    settings.write_text(json.dumps({"apiKeyHelper": "/bin/print-a-key"}))
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        bad = [r for r in run(proxy, lb, home, settings=[str(settings)]) if not r.ok]
        assert [r.kind for r in bad] == ["credentials"] and "apiKeyHelper" in bad[0].said
    finally:
        proxy.close()
        lb.close()


def test_main_prints_one_line_per_check_and_exits_by_the_verdict(home, monkeypatch, capsys):
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        monkeypatch.setattr(check, "default_dial", refused)
        monkeypatch.setenv("HOME", str(home))
        code = check.main(["--proxy", proxy.url, "--lb", lb.url, "--no-settings"])
        out = capsys.readouterr().out
        assert code == 0 and "SEALED" in out and "NOT SEALED" not in out
        assert out.count("PASS") >= 20
        monkeypatch.setattr(check, "default_dial", connected)
        code = check.main(["--proxy", proxy.url, "--lb", lb.url, "--no-settings"])
        out = capsys.readouterr().out
        assert code == 1 and "NOT SEALED" in out and "FAIL" in out
    finally:
        proxy.close()
        lb.close()


def test_main_takes_its_addresses_from_the_runners_environment(home, monkeypatch, capsys):
    """The runner calls the check with no arguments (`--seal-check <path>`): the proxy and the LB come from
    the same variables the entrypoint builds the runner's own command from — one definition each."""
    proxy, lb = FakeProxy(403), FakeLb(200)
    try:
        monkeypatch.setattr(check, "default_dial", refused)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("BENCH_EGRESS_PROXY", proxy.url)
        monkeypatch.setenv("BENCH_LB_URL", lb.url)
        assert check.main(["--no-settings"]) == 0
        monkeypatch.delenv("BENCH_EGRESS_PROXY")
        assert check.main(["--no-settings"]) == 2, "no proxy configured is a refusal, never a pass"
        assert "BENCH_EGRESS_PROXY" in capsys.readouterr().err
    finally:
        proxy.close()
        lb.close()