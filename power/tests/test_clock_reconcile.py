"""The configured clock: every GPU box is kept at `default_gpu_clock_mhz` (box-a: 2200 in the suite
inventory) or released when it has none (box-b), and gets it again each time it comes back up — a
clock lock dies with a reboot. A manual click holds until the box is next seen down."""
from __future__ import annotations

from tests.fakes import FakeProbe, FakeRunner, audit_lines, make_app, post

LIMIT, RELEASE = "/keys/fleet-clock-limit", "/keys/fleet-clock-release"


def keys_by_host(runner):
    return [(a[-2].split("@")[1], a[a.index("-i") + 1]) for a in runner.calls]


def test_first_pass_applies_the_cap_and_releases_the_box_without_one(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    app.reconcile_clocks()
    assert sorted(keys_by_host(runner)) == [("192.0.2.2", LIMIT), ("198.51.100.2", RELEASE)]
    assert app._clock_state(app.boxes["box-a"])["applied_mhz"] == 2200
    assert app._clock_state(app.boxes["box-a"])["applied_by"] == "config"
    assert app._clock_state(app.boxes["box-b"])["applied"] == "released"
    rec = {(l["box"], l["action"], l["result"]) for l in audit_lines(tmp_path) if l["event"] == "reconcile"}
    assert rec == {("box-a", "clock_limit_2200", "sent"), ("box-b", "clock_release", "sent")}


def test_a_box_that_stays_up_is_not_touched_again(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    app.reconcile_clocks()
    app.reconcile_clocks()
    assert len(runner.calls) == 2


def test_the_cap_is_reapplied_after_a_reboot(tmp_path):
    runner, probe = FakeRunner(), FakeProbe()
    app = make_app(tmp_path, runner=runner, probe=probe)
    app.reconcile_clocks()
    probe.up["box-a"] = False
    app.reconcile_clocks()
    assert len(runner.calls) == 2                    # nothing sent to a box that is down
    probe.up["box-a"] = True
    app.reconcile_clocks()
    assert keys_by_host(runner)[-1] == ("192.0.2.2", LIMIT) and len(runner.calls) == 3


def test_a_failed_apply_is_retried_every_pass_and_audited(tmp_path):
    runner = FakeRunner((9, "", "NVIDIA-SMI has failed: driver not loaded"))
    app = make_app(tmp_path, runner=runner)
    app.reconcile_clocks()
    app.reconcile_clocks()
    assert [h for h, _ in keys_by_host(runner)].count("192.0.2.2") == 2
    assert app._clock_state(app.boxes["box-a"])["applied"] is None
    assert ("box-a", "failed") in {(l["box"], l["result"]) for l in audit_lines(tmp_path)
                                   if l["event"] == "reconcile"}
    runner.result = (0, "", "")
    app.reconcile_clocks()
    assert app._clock_state(app.boxes["box-a"])["applied_mhz"] == 2200


def test_a_manual_click_holds_until_the_box_next_goes_down(tmp_path):
    runner, probe = FakeRunner(), FakeProbe()
    app = make_app(tmp_path, runner=runner, probe=probe)
    app.reconcile_clocks()
    token = post(app, "/power/api/confirm", {"box": "box-a", "action": "clock_release"})[1]["token"]
    assert post(app, "/power/api/execute", {"token": token})[0] == 200
    app.reconcile_clocks()
    assert keys_by_host(runner)[-1] == ("192.0.2.2", RELEASE) and len(runner.calls) == 3
    probe.up["box-a"] = False
    app.reconcile_clocks()
    probe.up["box-a"] = True
    app.reconcile_clocks()
    assert keys_by_host(runner)[-1] == ("192.0.2.2", LIMIT)     # back up: the config wins again


def test_dry_run_reconciles_with_the_dry_run_key_only(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, dry_run=True)
    app.reconcile_clocks()
    assert {k for _, k in keys_by_host(runner)} == {"/keys/fleet-clock-dryrun"}
    assert app._clock_state(app.boxes["box-a"])["applied"] is None
