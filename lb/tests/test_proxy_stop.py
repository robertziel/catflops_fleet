"""Proxy.stop() is safe when two threads call it at once.

drain() calls stop() from its own thread while a SIGTERM path or a test's teardown may call it too.
stop() used to check `self._server is not None` and then use `self._server` three times: a second
caller finishing in between left the first one calling `server_close()` on None — the
AttributeError at proxy.py `stop` that surfaced as a drain-teardown "flake" in
test_request_queue. The server is now taken out under a lock, and only its taker shuts it down.
"""
from __future__ import annotations

import threading

from vllm_lb.config import Config
from vllm_lb.hub import Hub
from vllm_lb.proxy import Proxy


def _started():
    cfg = Config(listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    return p


def test_a_second_caller_finishing_in_between_does_not_crash_the_first():
    p = _started()
    server = p._server
    real_shutdown = server.shutdown

    def shutdown_then_a_concurrent_stop_finishes():
        real_shutdown()
        p._server = None          # what a second stop() finishing at this instant leaves behind

    server.shutdown = shutdown_then_a_concurrent_stop_finishes
    p.stop()                       # raised AttributeError: 'NoneType' object has no attribute 'server_close'
    assert p._server is None


def test_many_concurrent_stops_close_the_server_once_and_raise_nothing():
    p = _started()
    closes, errors = [], []
    real_close = p._server.server_close

    def counted_close():
        closes.append(1)
        real_close()

    p._server.server_close = counted_close
    go = threading.Event()

    def stopper():
        go.wait()
        try:
            p.stop()
        except Exception as e:     # noqa: BLE001 - the point is that nothing escapes
            errors.append(e)

    threads = [threading.Thread(target=stopper) for _ in range(8)]
    for t in threads:
        t.start()
    go.set()
    for t in threads:
        t.join(10)
    assert errors == [] and len(closes) == 1 and p._server is None


def test_stop_before_start_and_twice_is_a_no_op():
    cfg = Config(listen_host="127.0.0.1", listen_port=0)
    p = Proxy(cfg, Hub(cfg))
    p.stop()
    p.start()
    p.stop()
    p.stop()
    assert p._server is None
