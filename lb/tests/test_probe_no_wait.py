"""A probe never holds the other backends' probes back. At one probe a second, a tick that waited for
its slowest probe — a box powered off answers nothing until /health's 1.5 s and /v1/models' 3 s run out — made every
OTHER backend's next probe arrive 4-5 s late: past the rate meters' gap rule (3 probe intervals), so every live
server's tok/s went blank while one box was off. A tick waits for its probes at most one tick; a probe still out
lands when it answers, and its backend is not probed again until it has."""
from __future__ import annotations

import threading
import time

from vllm_lb.config import Config, VllmBox
from vllm_lb.control import Controller
from vllm_lb.hub import Hub

FAST, SLOW = "http://192.0.2.1:8000", "http://192.0.2.2:8000"
CFG = Config(vllm_boxes=(VllmBox("fast-gpu", "fast", FAST), VllmBox("slow-gpu", "slow", SLOW)),
             probe_interval_s=0, tick_s=0.2)


class Prober:
    """FAST answers at once; SLOW answers when `release` is set — a box that is off, until its timeouts run out."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[str] = []

    def __call__(self, b, cfg, now):
        self.calls.append(b.url)
        if b.url == SLOW:
            self.release.wait(10)
        return {"live": True, "served_models": ("m",), "waiting": 0, "running": 0, "last_error": "",
                "last_probe_at": now}


def test_a_slow_probe_does_not_hold_the_tick_or_the_other_probes():
    hub, prober = Hub(CFG), Prober()
    ctl = Controller(CFG, hub, None, prober=prober)
    try:
        t0 = time.monotonic()
        ctl.tick()
        assert time.monotonic() - t0 < 1.0, "the tick waited for the slow probe"
        assert hub.backend_for(FAST).live and not hub.backend_for(SLOW).live
        ctl.tick()
        ctl.tick()
        assert prober.calls.count(FAST) == 3
        assert prober.calls.count(SLOW) == 1, "a backend whose probe is still out is not probed again"
        prober.release.set()
        deadline = time.monotonic() + 5
        while not hub.backend_for(SLOW).live and time.monotonic() < deadline:
            time.sleep(0.02)
        assert hub.backend_for(SLOW).live, "the slow probe's answer lands when it comes"
        ctl.tick()
        assert prober.calls.count(SLOW) == 2
    finally:
        prober.release.set()
        ctl.close()


def test_the_tick_period_is_tick_s_not_tick_s_plus_the_ticks_own_work():
    """`run_forever` sleeps what is LEFT of `tick_s` after a tick, so a 1 s tick is a 1 s period — it slept a whole
    `tick_s` after every tick, and a tick that waited its full `tick_s` for a probe made a 2 s period."""
    cfg = Config(vllm_boxes=(), probe_interval_s=1.0, tick_s=0.3)
    starts: list[float] = []

    class Slow(Controller):
        def tick(self) -> None:
            starts.append(time.monotonic())
            time.sleep(0.2)                         # two thirds of a tick's work

    ctl, stop = Slow(cfg, Hub(cfg), None, prober=Prober()), threading.Event()
    loop = threading.Thread(target=ctl.run_forever, args=(stop,), daemon=True)
    loop.start()
    time.sleep(2.0)
    stop.set()
    loop.join(2)
    ctl.close()
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(gaps) >= 4 and max(gaps) < 0.4, gaps


def test_a_period_of_zero_or_below_is_refused_at_load_naming_its_key():
    """A probe interval or tick of 0 is not "as fast as possible": the page would divide by it (graph columns) and
    the loop would spin. Refused at boot with the key's name."""
    import pytest
    for key in ("LB_PROBE_INTERVAL_S", "LB_TICK_S"):
        for bad in ("0", "-1"):
            with pytest.raises(ValueError, match=key):
                Config.from_env({"LB_ALLOWED_HOSTS": "loopback", key: bad})
