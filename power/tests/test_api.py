"""The power API's rules. Each test names the rule it pins."""
from __future__ import annotations

import json

import pytest

from fleet_power.app import HOLD_MS, HOLD_S
from tests.fakes import (GATE, ORIGIN, USER, FakeClock, FakeProbe, FakeRunner, FakeWol,
                         audit_lines, get, make_app, post)


def confirm(app, box="box-b", action="poweroff", **hdrs):
    return post(app, "/power/api/confirm", {"box": box, "action": action}, **hdrs)


def execute(app, token, held_ms=HOLD_MS, wait=HOLD_S, **extra):
    """What an honest client does: hold for 5 s (the fake clock moves on by `wait`), then send the
    token with the hold it measured."""
    app.clock.t += wait
    body = {"token": token, "held_ms": held_ms}
    body.update(extra)
    return post(app, "/power/api/execute", body)


# ── the door: only nginx's authenticated, same-origin JSON reaches an action ────────────────

def test_request_without_the_nginx_gate_secret_is_refused(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, X_Power_Gate=None)[0] == 403
    assert confirm(app, X_Power_Gate="wrong")[0] == 403
    assert get(app, "/power/api/state", X_Power_Gate=None)[0] == 403


def test_request_without_an_authenticated_user_is_refused(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, X_Power_User=None)[0] == 403
    assert confirm(app, X_Power_User="")[0] == 403


def test_an_empty_gate_secret_setting_refuses_everything(tmp_path):
    app = make_app(tmp_path)
    app.settings = app.settings.__class__(**{**app.settings.__dict__, "gate_secret": ""})
    assert confirm(app, X_Power_Gate="")[0] == 403


def test_post_without_json_content_type_is_415(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, Content_Type="text/plain")[0] == 415
    assert confirm(app, Content_Type=None)[0] == 415
    assert confirm(app, Content_Type="application/x-www-form-urlencoded")[0] == 415
    assert confirm(app, Content_Type="application/json; charset=utf-8")[0] == 200


def test_post_from_another_origin_is_403(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, Origin="https://evil.example")[0] == 403
    assert confirm(app, Origin=None)[0] == 403
    assert confirm(app, Origin="null")[0] == 403
    assert confirm(app, Sec_Fetch_Site="cross-site")[0] == 403
    assert confirm(app, Sec_Fetch_Site="same-origin")[0] == 200


def test_state_changes_are_post_only(tmp_path):
    app = make_app(tmp_path)
    assert get(app, "/power/api/confirm")[0] == 405
    assert get(app, "/power/api/execute")[0] == 405
    assert post(app, "/power/api/state", {})[0] == 405


def test_oversized_or_non_object_body_is_refused(tmp_path):
    app = make_app(tmp_path)
    assert post(app, "/power/api/confirm", b"x" * 5000)[0] == 413
    assert post(app, "/power/api/confirm", b"[1,2]")[0] == 400
    assert post(app, "/power/api/confirm", b"not json")[0] == 400


# ── the allowlist ───────────────────────────────────────────────────────────────────────────

def test_unknown_box_is_404(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, box="edge")[0] == 404
    assert confirm(app, box="../box-a")[0] == 404
    assert confirm(app, box="")[0] == 404


def test_unknown_action_is_404(tmp_path):
    app = make_app(tmp_path)
    assert confirm(app, action="reboot")[0] == 404
    assert confirm(app, action="rm -rf /")[0] == 404


def test_box_c_has_no_power_on(tmp_path):
    """box-c hosts this service: when it is off nothing here can wake it (its smart plug does)."""
    app = make_app(tmp_path)
    status, body = confirm(app, box="box-c", action="poweron")
    assert status == 404
    assert "smart plug" in body["error"]["message"]


def test_unknown_path_is_404(tmp_path):
    app = make_app(tmp_path)
    assert get(app, "/power/api/nope")[0] == 404
    assert get(app, "/power/../etc/passwd")[0] == 404


# ── the confirmation step ───────────────────────────────────────────────────────────────────

def test_confirm_then_execute_after_a_5s_hold_powers_off(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    status, body = confirm(app)
    assert status == 200 and body["box"] == "box-b"
    assert body["needs_hold"] is True and body["hold_ms"] == 5000 == HOLD_MS
    assert body["token"] and body["expires_in"] > 0
    assert runner.calls == []                         # asking for a token does nothing
    status, out = execute(app, body["token"])
    assert status == 200, out
    assert out["result"] == "sent"
    assert len(runner.calls) == 1


def test_the_box_name_no_longer_has_to_be_typed(tmp_path):
    """The hold replaced the typed name: `typed` is neither needed nor looked at."""
    app = make_app(tmp_path)
    assert execute(app, confirm(app)[1]["token"], typed="not-the-box")[0] == 200


@pytest.mark.parametrize("held", [None, 0, 4999, 4999.9, -5000, "5000", True, [5000], {"ms": 5000},
                                  float("nan"), float("inf"), float("-inf")])
def test_execute_needs_a_reported_hold_of_5_seconds(tmp_path, held):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    body = {} if held is None else {"held_ms": held}
    app.clock.t += 60                                 # the token is old enough: only held_ms fails
    status, out = post(app, "/power/api/execute", {"token": token, **body})
    assert status == 400 and out["error"]["type"] == "hold_too_short", out
    assert out["hold_ms"] == HOLD_MS
    assert runner.calls == []


def test_a_hold_reported_as_a_huge_number_is_just_a_long_hold(tmp_path):
    app = make_app(tmp_path)
    token = confirm(app)[1]["token"]
    app.clock.t += HOLD_S
    assert post(app, "/power/api/execute", b'{"token": "%s", "held_ms": %s}'
                % (token.encode(), b"9" * 400))[0] == 200


def test_the_server_enforces_the_wait_a_token_younger_than_the_hold_is_refused(tmp_path):
    """A script can claim any `held_ms`; it cannot make the token older. The token is issued before
    the hold starts, so an honest 5 s hold always clears this."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    status, out = execute(app, token, wait=0)                    # claims 5000 ms, sent at once
    assert status == 400 and out["error"]["type"] == "too_soon", out
    assert "0.0 s old" in out["error"]["message"]
    status, out = execute(app, token, wait=4.99)                 # still short of 5 s
    assert status == 400 and out["error"]["type"] == "too_soon"
    assert "4.9 s old" in out["error"]["message"]                # never rounded up to "5.0"
    assert runner.calls == []
    assert execute(app, token, wait=0.01)[0] == 200              # 5.00 s: the same token works
    assert len(runner.calls) == 1


def test_a_refused_hold_does_not_burn_the_token(tmp_path):
    """Releasing early (or a client racing ahead) must not cost the operator a new confirm."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    assert execute(app, token, held_ms=1200)[0] == 400
    assert execute(app, token, held_ms=HOLD_MS)[0] == 200
    assert len(runner.calls) == 1


def test_a_power_on_needs_the_hold_too(tmp_path):
    wol = FakeWol()
    app = make_app(tmp_path, wol=wol, probe=FakeProbe({"box-a": True, "box-b": False, "box-c": True}))
    token = confirm(app, action="poweron")[1]["token"]
    assert execute(app, token, held_ms=10)[0] == 400
    assert execute(app, confirm(app, action="poweron")[1]["token"], wait=1)[0] == 400
    assert wol.sent == []
    assert execute(app, token)[0] == 200
    assert wol.sent == ["box-b"]


def test_a_token_is_single_use(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    assert execute(app, token)[0] == 200
    assert execute(app, token)[0] == 409
    assert len(runner.calls) == 1


def test_a_token_expires(tmp_path):
    clock = FakeClock()
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, clock=clock)
    token = confirm(app)[1]["token"]
    clock.t += 121
    assert execute(app, token)[0] == 409
    assert runner.calls == []


def test_a_made_up_token_is_refused(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    assert execute(app, "A" * 32)[0] == 409
    assert execute(app, None)[0] == 400
    assert runner.calls == []


def test_a_token_is_bound_to_the_user_who_asked_for_it(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    app.clock.t += HOLD_S
    status, _ = post(app, "/power/api/execute", {"token": token, "held_ms": HOLD_MS},
                     X_Power_User="someone-else")
    assert status == 409
    assert runner.calls == []


# ── the cluster ─────────────────────────────────────────────────────────────────────────────

RUNNING = {"ok": True, "state": "running", "boxes": ["box-a", "box-b"]}


def test_powering_off_a_gpu_box_under_a_running_cluster_needs_the_explicit_ack(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, cluster=RUNNING)
    status, body = confirm(app)
    assert status == 200 and body["cluster"]["state"] == "running" and body["needs_cluster_ack"]
    status, out = execute(app, body["token"])
    assert status == 409 and out["error"]["type"] == "cluster_running"
    assert runner.calls == []
    assert execute(app, body["token"], ack_cluster=True)[0] == 200
    assert len(runner.calls) == 1


def test_an_unknown_cluster_state_counts_as_running(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, cluster={"ok": False, "state": None, "boxes": []})
    token = confirm(app)[1]["token"]
    assert execute(app, token)[0] == 409
    assert runner.calls == []


def test_box_c_poweroff_needs_no_cluster_ack(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, cluster=RUNNING)
    body = confirm(app, box="box-c")[1]
    assert body["needs_cluster_ack"] is False
    assert execute(app, body["token"])[0] == 200


# ── what actually runs ──────────────────────────────────────────────────────────────────────

def test_poweroff_is_ssh_with_the_dedicated_key_and_pinned_host_keys(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    execute(app, confirm(app)[1]["token"])
    argv = runner.calls[0]
    assert argv[0] == "ssh"
    joined = " ".join(argv)
    assert "-i /keys/fleet-power " in joined
    for opt in ("IdentitiesOnly=yes", "BatchMode=yes", "StrictHostKeyChecking=yes",
                "UserKnownHostsFile=/keys/known_hosts", "IdentityAgent=none",
                "ForwardAgent=no", "ClearAllForwardings=yes"):
        assert opt in argv, opt
    assert argv[-2] == "fleet-power@198.51.100.2"
    # The target runs its forced command whatever we send; we send nothing a shell could use.
    assert argv[-1] == "poweroff"


def test_each_box_is_reached_on_its_own_address(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    for box in ("box-a", "box-b", "box-c"):
        execute(app, confirm(app, box=box)[1]["token"])
        app.clock.t += 3600          # clear the per-box rate limit
    targets = [c[-2] for c in runner.calls]
    assert targets == ["fleet-power@192.0.2.2", "fleet-power@198.51.100.2", "fleet-power@127.0.0.1"]


def test_dry_run_uses_the_dry_run_key(tmp_path):
    runner = FakeRunner((0, "DRY-RUN fleet-power: would run /usr/bin/systemctl poweroff", ""))
    app = make_app(tmp_path, runner=runner, dry_run=True)
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 200 and out["dry_run"] is True
    assert "-i /keys/fleet-power-dryrun " in " ".join(runner.calls[0]) + " "
    assert "/keys/fleet-power " not in " ".join(runner.calls[0]) + " "
    assert "DRY-RUN" in out["output"]


def test_a_dropped_connection_while_the_box_goes_down_counts_as_sent(tmp_path):
    runner = FakeRunner((255, "", "Connection to 198.51.100.2 closed by remote host."))
    app = make_app(tmp_path, runner=runner)
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 200 and out["result"] == "sent"


@pytest.mark.parametrize("err", [
    "fleet-power@198.51.100.2: Permission denied (publickey).",
    "Host key verification failed.",
    "ssh: connect to host 198.51.100.2 port 22: Connection timed out",
    "ssh: connect to host 198.51.100.2 port 22: No route to host",
])
def test_an_ssh_failure_is_reported_as_failed(tmp_path, err):
    app = make_app(tmp_path, runner=FakeRunner((255, "", err)))
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 502 and out["result"] == "failed"
    assert err in out["output"]


def test_a_refused_sudo_is_reported_as_failed(tmp_path):
    app = make_app(tmp_path, runner=FakeRunner((1, "", "sudo: a password is required")))
    status, out = execute(app, confirm(app)[1]["token"])
    assert status == 502 and out["result"] == "failed"


def test_power_on_sends_wake_on_lan_and_runs_no_ssh(tmp_path):
    runner, wol = FakeRunner(), FakeWol()
    app = make_app(tmp_path, runner=runner, wol=wol,
                   probe=FakeProbe({"box-a": True, "box-b": False, "box-c": True}))
    body = confirm(app, action="poweron")[1]
    status, out = execute(app, body["token"])
    assert status == 200 and out["result"] == "sent"
    assert wol.sent == ["box-b"] and runner.calls == []


def test_dry_run_power_on_checks_the_route_but_sends_no_packet(tmp_path):
    """A DRY RUN must not wake a box that is really off."""
    wol = FakeWol()
    app = make_app(tmp_path, wol=wol, dry_run=True,
                   probe=FakeProbe({"box-a": True, "box-b": False, "box-c": True}))
    status, out = execute(app, confirm(app, action="poweron")[1]["token"])
    assert status == 200 and out["dry_run"] is True
    assert wol.sent == [] and wol.checked_only == ["box-b"]


def test_power_on_with_no_link_to_the_box_is_a_clear_failure(tmp_path):
    from fleet_power.wol import WolError
    app = make_app(tmp_path, wol=FakeWol(WolError("no route to 198.51.100.3 via eth-box-b")),
                   probe=FakeProbe({"box-b": False, "box-c": True}))
    status, out = execute(app, confirm(app, action="poweron")[1]["token"])
    assert status == 502 and "eth-box-b" in out["output"]


# ── rate limit ──────────────────────────────────────────────────────────────────────────────

def test_one_action_per_box_per_minute(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    assert execute(app, confirm(app)[1]["token"])[0] == 200
    token = confirm(app)[1]["token"]
    assert execute(app, token)[0] == 429
    assert len(runner.calls) == 1
    app.clock.t += 61
    assert execute(app, confirm(app)[1]["token"])[0] == 200


def test_confirm_requests_are_rate_limited(tmp_path):
    app = make_app(tmp_path)
    codes = [confirm(app)[0] for _ in range(12)]
    assert codes[:10] == [200] * 10
    assert codes[10:] == [429, 429]


# ── audit log ───────────────────────────────────────────────────────────────────────────────

def test_every_action_is_audited_with_who_when_what_and_result(tmp_path):
    runner = FakeRunner((0, "ok", ""))
    app = make_app(tmp_path, runner=runner)
    token = confirm(app)[1]["token"]
    execute(app, token)
    lines = audit_lines(tmp_path)
    assert [l["event"] for l in lines] == ["confirm", "execute"]
    last = lines[-1]
    assert last["user"] == USER and last["ip"] == "203.0.113.7"
    assert last["box"] == "box-b" and last["action"] == "poweroff"
    assert last["result"] == "sent" and last["dry_run"] is False
    assert last["ts"].endswith("Z")
    assert token not in json.dumps(lines)          # a live token never lands in a log


def test_refusals_are_audited_too(tmp_path):
    app = make_app(tmp_path)
    confirm(app, Origin="https://evil.example")
    confirm(app, box="edge")
    token = confirm(app)[1]["token"]
    execute(app, token, wait=0)                  # at once: too soon (and the hold is fine)
    execute(app, token, held_ms=300)             # 5 s later: old enough, but let go too early
    results = [(l["event"], l["result"]) for l in audit_lines(tmp_path)]
    assert ("confirm", "refused:origin") in results
    assert ("confirm", "refused:unknown_box") in results
    assert ("execute", "refused:hold_too_short") in results
    assert ("execute", "refused:too_soon") in results


def test_audit_survives_a_hostile_user_name(tmp_path):
    app = make_app(tmp_path)
    confirm(app, X_Power_User='x"\n{"event":"forged"}')
    lines = audit_lines(tmp_path)
    assert len(lines) == 1 and lines[0]["event"] == "confirm"


# ── state and the button ────────────────────────────────────────────────────────────────────

def test_state_lists_the_three_boxes_and_what_each_can_do(tmp_path):
    app = make_app(tmp_path, probe=FakeProbe({"box-a": True, "box-b": False, "box-c": True}))
    status, h, out = get(app, "/power/api/state")
    assert status == 200
    s = json.loads(out)
    boxes = s["boxes"]
    assert set(boxes) == {"box-a", "box-b", "box-c"}
    assert boxes["box-a"]["phase"] == "on" and boxes["box-b"]["phase"] == "off"
    assert boxes["box-b"]["can_power_on"] is True
    assert boxes["box-c"]["can_power_on"] is False
    assert "smart plug" in boxes["box-c"]["power_on_note"]
    assert s["dry_run"] is False
    assert h["Cache-Control"] == "no-store"


def test_state_tracks_an_in_flight_poweroff(tmp_path):
    probe = FakeProbe()
    app = make_app(tmp_path, probe=probe)
    execute(app, confirm(app)[1]["token"])
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-b"]["phase"] == "powering_off"
    probe.up["box-b"] = False
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-b"]["phase"] == "off"


def test_state_tracks_an_in_flight_wake(tmp_path):
    probe = FakeProbe({"box-a": True, "box-b": False, "box-c": True})
    app = make_app(tmp_path, probe=probe)
    execute(app, confirm(app, action="poweron")[1]["token"])
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-b"]["phase"] == "waking"
    app.clock.t += 601
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-b"]["phase"] == "wake_timeout"
    probe.up["box-b"] = True
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-b"]["phase"] == "on"


def test_button_js_is_served_as_javascript(tmp_path):
    app = make_app(tmp_path)
    status, h, out = get(app, "/power/button.js")
    assert status == 200
    assert h["Content-Type"].startswith("application/javascript")
    assert h["X-Content-Type-Options"] == "nosniff"
    assert b"/power/api/state" in out


def test_every_response_carries_the_hardening_headers(tmp_path):
    app = make_app(tmp_path)
    for status, h, _ in (get(app, "/power/api/state"), get(app, "/power/nope"),
                         get(app, "/power/api/state", X_Power_Gate=None)):
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert h["Cache-Control"] == "no-store"


# ── the GPU clock cap (what used to be nvidia-clock-lock.service on every boot) ──────────────

def clock_confirm(app, box="box-b", action="clock_limit_2200", **hdrs):
    return post(app, "/power/api/confirm", {"box": box, "action": action}, **hdrs)


def test_each_clock_action_is_its_own_forced_command_key(tmp_path):
    """The MHz live in the key's forced command on the box, so the action IS the key: a client can
    neither choose a frequency nor reach the other action's key."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    for action, key, sent in (("clock_limit_2200", "/keys/fleet-clock-limit", "clock-limit-2200"),
                              ("clock_limit_2300", "/keys/fleet-clock-limit-2300", "clock-limit-2300"),
                              ("clock_release", "/keys/fleet-clock-release", "clock-release")):
        token = clock_confirm(app, action=action)[1]["token"]
        assert post(app, "/power/api/execute", {"token": token})[0] == 200, action
        argv = runner.calls[-1]
        assert f"-i {key} " in " ".join(argv) + " ", argv
        assert argv[-2] == "fleet-power@198.51.100.2"
        assert argv[-1] == sent
        app.clock.t += 61
    assert len(runner.calls) == 3
    keys = [a[a.index("-i") + 1] for a in runner.calls]
    assert len(set(keys)) == 3, keys          # one cap cannot reach the other cap's key


def test_a_clock_change_needs_no_hold_but_still_needs_its_token(tmp_path):
    """A clock change is sent the moment it is clicked: no hold, no minimum token age."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    status, body = clock_confirm(app)
    assert status == 200 and body["needs_hold"] is False and body["hold_ms"] is None
    assert body["kind"] == "clock" and "2200" in body["title"]
    assert post(app, "/power/api/execute", {"token": "A" * 32})[0] == 409
    assert runner.calls == []
    assert post(app, "/power/api/execute", {"token": body["token"]})[0] == 200
    assert post(app, "/power/api/execute", {"token": body["token"]})[0] == 409   # single use
    assert len(runner.calls) == 1


def test_a_poweroff_still_needs_the_hold(tmp_path):
    """No hold for the clock must not mean no hold for the actions you cannot take back."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    token = confirm(app, action="poweroff")[1]["token"]
    app.clock.t += 60
    assert post(app, "/power/api/execute", {"token": token})[0] == 400
    assert runner.calls == []


def test_a_clock_action_on_a_box_with_no_gpu_is_404(tmp_path):
    app = make_app(tmp_path)
    status, body = clock_confirm(app, box="box-c")
    assert status == 404 and body["error"]["type"] == "no_gpu_clock"
    assert ("confirm", "refused:no_gpu_clock") in [(l["event"], l["result"]) for l in audit_lines(tmp_path)]


def test_a_clock_change_needs_no_cluster_ack_because_it_drops_nothing(tmp_path):
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner, cluster=RUNNING)
    body = clock_confirm(app)[1]
    assert body["needs_cluster_ack"] is False
    assert post(app, "/power/api/execute", {"token": body["token"]})[0] == 200
    assert len(runner.calls) == 1


def test_dry_run_uses_the_clock_dry_run_key(tmp_path):
    runner = FakeRunner((0, "DRY-RUN: (root) NOPASSWD: /usr/bin/nvidia-smi -lgc 300,2200", ""))
    app = make_app(tmp_path, runner=runner, dry_run=True)
    status, out = execute(app, clock_confirm(app)[1]["token"])
    assert status == 200 and out["dry_run"] is True
    joined = " ".join(runner.calls[0]) + " "
    assert "-i /keys/fleet-clock-dryrun " in joined
    assert "/keys/fleet-clock-limit " not in joined and "/keys/fleet-clock-release " not in joined
    assert "/keys/fleet-clock-limit-2300 " not in joined


def test_a_dropped_connection_on_a_clock_change_is_a_failure_not_a_success(tmp_path):
    """The box is not going down, so ssh dropping means the cap did NOT land."""
    app = make_app(tmp_path, runner=FakeRunner((255, "", "Connection to 198.51.100.2 closed by remote host.")))
    status, out = execute(app, clock_confirm(app)[1]["token"])
    assert status == 502 and out["result"] == "failed"


def test_a_refused_sudo_on_a_clock_change_is_reported(tmp_path):
    app = make_app(tmp_path, runner=FakeRunner((1, "", "sudo: a password is required")))
    status, out = execute(app, clock_confirm(app)[1]["token"])
    assert status == 502 and out["result"] == "failed" and "password" in out["output"]


def test_state_reports_the_cap_this_panel_last_applied(tmp_path):
    app = make_app(tmp_path)
    s = json.loads(get(app, "/power/api/state")[2])
    assert s["boxes"]["box-c"]["clock"] is None            # no GPU to cap
    c = s["boxes"]["box-b"]["clock"]
    assert c["applied"] is None and c["applied_mhz"] is None
    assert c["limits_mhz"] == [2200, 2300] and c["min_mhz"] == 300
    assert set(c["actions"]) == {"clock_limit_2200", "clock_limit_2300", "clock_release"}
    assert "no lock state" in c["note"]                      # never claimed as a read-back

    execute(app, clock_confirm(app)[1]["token"])
    c = json.loads(get(app, "/power/api/state")[2])["boxes"]["box-b"]["clock"]
    assert c["applied"] == "limited" and c["applied_mhz"] == 2200
    assert c["applied_by"] == USER and c["applied_at"] == app.clock.t
    app.clock.t += 11
    execute(app, clock_confirm(app, action="clock_limit_2300")[1]["token"])
    c = json.loads(get(app, "/power/api/state")[2])["boxes"]["box-b"]["clock"]
    assert c["applied"] == "limited" and c["applied_mhz"] == 2300   # which cap, not just "capped"
    app.clock.t += 11
    execute(app, clock_confirm(app, action="clock_release")[1]["token"])
    c = json.loads(get(app, "/power/api/state")[2])["boxes"]["box-b"]["clock"]
    assert c["applied"] == "released" and c["applied_mhz"] is None
    assert json.loads(get(app, "/power/api/state")[2])["boxes"]["box-a"]["clock"]["applied"] is None


def test_a_dry_run_records_no_applied_cap(tmp_path):
    app = make_app(tmp_path, dry_run=True)
    execute(app, clock_confirm(app)[1]["token"])
    assert json.loads(get(app, "/power/api/state")[2])["boxes"]["box-b"]["clock"]["applied"] is None


def test_a_clock_click_is_rate_limited_on_its_own_bucket(tmp_path):
    """Ten seconds stops a double click; it must not block, or be blocked by, a power action."""
    runner = FakeRunner()
    app = make_app(tmp_path, runner=runner)
    assert execute(app, clock_confirm(app)[1]["token"])[0] == 200
    assert execute(app, clock_confirm(app, action="clock_release")[1]["token"])[0] == 429
    assert execute(app, confirm(app)[1]["token"])[0] == 200          # the poweroff bucket is free
    app.clock.t += 11
    assert execute(app, clock_confirm(app)[1]["token"])[0] == 200    # the clock bucket reopens
    assert execute(app, confirm(app)[1]["token"])[0] == 429          # the power one has not
    assert len(runner.calls) == 3


def test_clock_actions_are_audited_like_any_other(tmp_path):
    app = make_app(tmp_path, runner=FakeRunner((0, "GPU clocks set to '(gpuClkMin 300, gpuClkMax 2200)'", "")))
    execute(app, clock_confirm(app)[1]["token"])
    last = audit_lines(tmp_path)[-1]
    assert last["action"] == "clock_limit_2200" and last["box"] == "box-b"
    assert last["result"] == "sent" and last["user"] == USER and "gpuClkMax 2200" in last["output"]


def test_every_capped_frequency_matches_a_forced_command_installed_on_the_box(tmp_path):
    """One value per cap, two files: every number the service offers must be one sudo and sshd
    actually allow — and each cap must have its OWN key, or a click would set the wrong clock."""
    import pathlib

    from fleet_power.app import ACTION_SPECS, CLOCK_CAPS_MHZ, CLOCK_MIN_MHZ

    setup = (pathlib.Path(__file__).resolve().parents[1] / "install" / "host-setup.sh").read_text()
    assert CLOCK_CAPS_MHZ == (2200, 2300)
    assert f"CLOCK_MIN={CLOCK_MIN_MHZ}" in setup
    sudoers = setup.split("NOPASSWD:")[1].split('"')[0]
    for n, mhz in enumerate(CLOCK_CAPS_MHZ, start=1):
        assert f"CLOCK_MAX{n}={mhz}" in setup
        assert f'command="sudo -n /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$CLOCK_MAX{n}"' in setup
        assert f'"$CLOCK_MAX{n}"' in setup                 # and in the sudoers loop below
    assert 'command="sudo -n /usr/bin/nvidia-smi -rgc"' in setup
    assert 'command="sudo -n -l /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$CLOCK_MAX1"' in setup
    # one argv per cap, the comma inside it escaped for sudoers (which separates on commas)
    assert r'/usr/bin/nvidia-smi -lgc $CLOCK_MIN\\,$m' in setup
    assert '/usr/bin/nvidia-smi -rgc' in sudoers and "*" not in sudoers

    # one key per cap, installed under its own name
    keys = {s.key_attr for s in ACTION_SPECS.values() if s.limit_mhz is not None}
    assert len(keys) == len(CLOCK_CAPS_MHZ)
    installed = [l.rsplit("$", 1)[1] for l in setup.splitlines()
                 if l.startswith("restrict") and "-n /usr/bin/nvidia-smi -lgc" in l]
    assert len(installed) == len(CLOCK_CAPS_MHZ)                 # one line per cap
    assert len(set(installed)) == len(installed), installed      # each with its own key variable
    pubs = [l for l in setup.splitlines() if l.startswith("CLOCK_LIMIT")]
    assert len({l.split("=", 1)[1] for l in pubs}) == len(pubs), pubs   # and its own public key


def test_nothing_sets_a_clock_at_boot(tmp_path):
    """The whole point: the cap is a click, never a unit. The removal script must not install one."""
    import pathlib

    def code(text):                 # what the shell runs, without the comments explaining it
        return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))

    d = pathlib.Path(__file__).resolve().parents[1] / "install"
    remove = code((d / "remove-clock-lock.sh").read_text())
    assert "systemctl disable" in remove and "rm -f" in remove
    assert "systemctl stop" not in remove          # ExecStop would -rgc under a running job
    for f in sorted(d.glob("*.sh")):
        body = code(f.read_text())
        assert "systemctl enable" not in body, f   # no install script may create a boot job
        assert "ExecStart" not in body and "[Unit]" not in body, f
