#!/bin/sh
# RED proof per rule: break ONE guard in a scratch copy, run its tests, expect them to FAIL.
# Run: docker run --rm -v "$PWD":/src:ro vllm-power:test sh /src/tests/mutations.sh
set -u
fail=0
m() {  # m <name> <file> <python-literal-old> <python-literal-new> <pytest -k expr>
  rm -rf /tmp/m && cp -r /src /tmp/m && cd /tmp/m
  python3 - "$2" "$3" "$4" <<'PY' || { echo "MUTATION DID NOT APPLY: $1"; exit 1; }
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
assert sys.argv[2] in s, "old text not found"
p.write_text(s.replace(sys.argv[2], sys.argv[3], 1))
PY
  if python3 -m pytest -q -p no:cacheprovider -k "$5" >/tmp/out 2>&1; then
    echo "SURVIVED (bad): $1"; fail=1
  else
    echo "KILLED: $1 -> $(grep -cE '^FAILED' /tmp/out) failing: $(grep -E '^FAILED' /tmp/out | sed 's/ - .*//; s/.*:://' | tr '\n' ' ')"
  fi
  cd /
}
m origin        fleet_power/app.py 'if not origin or origin not in self.settings.origins:' 'if False:' 'origin'
m content_type  fleet_power/app.py 'if ctype != "application/json":' 'if False:' '415'
m gate          fleet_power/app.py 'if not secret or not hmac.compare_digest(gate.encode(), secret.encode()):' 'if False:' 'gate'
m user          fleet_power/app.py 'if not ctx["user"]:' 'if False:' 'authenticated_user'
m allowlist     fleet_power/app.py 'box = self.boxes.get(name) if isinstance(name, str) else None' 'box = self.boxes.get(name, self.boxes["box-a"]) if isinstance(name, str) else self.boxes["box-a"]' 'unknown_box'
m action_list   fleet_power/app.py 'if action not in ACTIONS:' 'if False:' 'unknown_action'
m hold_ms       fleet_power/app.py 'if not _held_long_enough(data.get("held_ms")):' 'if False:' 'hold'
m hold_nan      fleet_power/app.py 'if isinstance(held_ms, float) and not math.isfinite(held_ms):' 'if False:' 'reported_hold'
m hold_age      fleet_power/app.py 'if age < HOLD_S:' 'if False:' 'enforces_the_wait'
m hold_issued   fleet_power/app.py 'age = now - t["issued"]' 'age = HOLD_S' 'enforces_the_wait'
m hold_burn     fleet_power/app.py '        if spec.needs_hold:
' '        if spec.needs_hold:
            self.tokens.pop(token, None)
' 'burn'
m hold_poweron  fleet_power/app.py 'ActionSpec("power", True, "key_path", "poweroff", "Power on", "")' 'ActionSpec("power", False, "key_path", "poweroff", "Power on", "")' 'power_on_needs_the_hold'
m single_use    fleet_power/app.py 'self.tokens.pop(token, None)' 'pass' 'single_use'
m expiry        fleet_power/app.py 't["exp"] <= now or ' '' 'expires'
m token_user    fleet_power/app.py ' or t["user"] != ctx["user"]' '' 'bound_to_the_user'
m cluster_ack   fleet_power/app.py 'and data.get("ack_cluster") is not True:' 'and False:' 'cluster'
m rate_box      fleet_power/app.py 'if last is not None and now - last < window:' 'if False:' 'per_box'
m rate_confirm  fleet_power/app.py 'if not allowed:' 'if False:' 'confirm_requests'
m audit         fleet_power/app.py 'f.write(line + "\n")' 'pass' 'audit'
m box_c_on     fleet_power/app.py 'if action == "poweron" and box.wol is None:' 'if False:' 'box_c_has_no'
m dry_key       fleet_power/app.py 'key = self.settings.dry_key_path if self.settings.dry_run else self.settings.key_path' 'key = self.settings.key_path' 'dry_run'
m ssh_failure   fleet_power/app.py 'if rc == 255 and not any(s in output for s in _SSH_NEVER_GOT_IN):' 'if rc == 255:' 'ssh_failure'
m wol_route     fleet_power/wol.py 'if via != w.iface:' 'if False:' 'cable_route'
m dry_wol      fleet_power/app.py 'self.wol(box, send=not self.settings.dry_run)' 'self.wol(box)' 'dry_run_power_on'
m dry_wol_send fleet_power/wol.py '    if not send:' '    if False:' 'send_false'
# the GPU clock cap: the action is the key, and none of it may soften a power action
m clock_key    fleet_power/app.py 'getattr(self.settings, spec.key_attr)' 'self.settings.clock_limit_key_path' 'own_forced_command_key'
m clock_sent   fleet_power/app.py 'ACTION_SPECS[action].sent' '"poweroff"' 'own_forced_command_key'
m clock_dry    fleet_power/app.py 'key = (self.settings.clock_dry_key_path if self.settings.dry_run' 'key = (getattr(self.settings, spec.key_attr) if self.settings.dry_run' 'clock_dry_run_key'
m clock_gpu    fleet_power/app.py 'if spec.kind == "clock" and not box.gpu_clock:' 'if False:' 'no_gpu'
m clock_hold   fleet_power/app.py 'if spec.needs_hold:' 'if True:' 'no_hold'
m clock_255    fleet_power/app.py 'landed = rc == 0' 'landed = rc in (0, 255)' 'dropped_connection_on_a_clock'
m clock_applied fleet_power/app.py '"limited" if spec.limit_mhz else "released"' '"limited"' 'last_applied'
m clock_which  fleet_power/app.py '"mhz": spec.limit_mhz, "at": now' '"mhz": CLOCK_CAPS_MHZ[0], "at": now' 'last_applied'
m clock_mhz    fleet_power/app.py 'CLOCK_CAPS_MHZ = (2200, 2300)' 'CLOCK_CAPS_MHZ = (2200, 2400)' 'matches_a_forced_command'
m clock_caps   fleet_power/app.py 'CLOCK_CAPS_MHZ = (2200, 2300)' 'CLOCK_CAPS_MHZ = (2200,)' 'clock'
m clock_cap_key fleet_power/app.py 'return "clock_limit_key_path" if mhz == CLOCK_CAPS_MHZ[0] else f"clock_limit_{mhz}_key_path"' 'return "clock_limit_key_path"' 'own_forced_command_key'
# the configured clock: refused when unknown, reapplied after a reboot, released when absent
m clock_cfg_cap fleet_power/boxes.py 'mhz not in CLOCK_CAPS_MHZ' 'False' 'unknown_default_gpu_clock'
m clock_reboot  fleet_power/app.py 'self.clock_settled[box.name] = False
                    continue' 'continue' 'reapplied_after_a_reboot'
m clock_release fleet_power/app.py 'action = clock_action(mhz) if mhz else "clock_release"' 'action = clock_action(mhz or 2200)' 'releases'
m clock_retry   fleet_power/app.py 'self.clock_settled[box.name] = ok' 'self.clock_settled[box.name] = True' 'retried'
exit $fail
