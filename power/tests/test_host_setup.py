"""install/host-setup.sh installs what the OPERATOR's key directory holds, and nothing it carries itself.

A stranger who runs this on their own box must not grant the author's keys poweroff or clock control, so
the script ships no public key and no source address. Its keys come from FLEET_POWER_KEYS, and a missing
one is refused BEFORE anything on the system changes. FLEET_POWER_PRINT=1 prints exactly what would be
installed (authorized_keys and the sudoers file) and touches nothing — it needs no root, so the rendering
is testable here.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "install" / "host-setup.sh"

# Fake keys: well-formed `ssh-ed25519` lines with an obviously fake body. Test data only.
KEY_FILES = {
    "fleet-power.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPOWERTESTKEYONLYFORTHESUITE0000000000 power@test",
    "fleet-power-dryrun.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDRYRUNTESTKEYONLYFORTHESUITE000000000 dryrun@test",
    "fleet-clock-limit.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICLOCKLIMITTESTKEYSUITE0000000000 limit@test",
    "fleet-clock-limit-2300.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICLOCK2300TESTKEYSUITE00000000 limit2@test",
    "fleet-clock-release.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICLOCKRELEASETESTKEYSUITE000000 release@test",
    "fleet-clock-dryrun.pub": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICLOCKDRYRUNTESTKEYSUITE0000000 cdry@test",
}


def make_keys(d: pathlib.Path, *, sources=("box-a", "box-b")) -> pathlib.Path:
    for name, body in KEY_FILES.items():
        (d / name).write_text(body + "\n")
    addresses = {"box-a": "192.0.2.1", "box-b": "198.51.100.1", "box-c": "203.0.113.1"}
    for box in sources:
        (d / f"from-{box}").write_text(addresses[box] + "\n")
    return d


def run(args, env=None):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    return subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True,
                          env={**base, **(env or {})}, timeout=60)


def test_the_script_carries_no_public_key_and_no_operator_address():
    text = SCRIPT.read_text()
    assert "ssh-ed25519 AAAA" not in text
    assert "10.10." not in text
    assert "@box-c" not in text


def test_without_a_key_directory_it_refuses_and_names_the_variable():
    r = run(["box-a", "--gpu"])
    assert r.returncode != 0
    assert "FLEET_POWER_KEYS" in r.stderr


def test_a_missing_key_is_named_and_refused(tmp_path):
    make_keys(tmp_path)
    (tmp_path / "fleet-clock-release.pub").unlink()
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode != 0
    assert "fleet-clock-release.pub" in r.stderr


def test_a_key_that_is_not_a_public_key_is_refused(tmp_path):
    make_keys(tmp_path)
    (tmp_path / "fleet-power.pub").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode != 0
    assert "fleet-power.pub" in r.stderr and "public key" in r.stderr


def test_a_gpu_box_renders_six_restricted_keys_from_its_own_source(tmp_path):
    make_keys(tmp_path)
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode == 0, r.stderr
    auth = [ln for ln in r.stdout.splitlines() if ln.startswith("restrict,")]
    assert len(auth) == 6, auth
    assert all('from="192.0.2.1"' in ln for ln in auth)
    assert 'command="sudo -n /usr/bin/systemctl poweroff" ' + KEY_FILES["fleet-power.pub"] in r.stdout
    assert 'command="sudo -n -l /usr/bin/systemctl poweroff" ' + KEY_FILES["fleet-power-dryrun.pub"] in r.stdout
    assert 'command="sudo -n /usr/bin/nvidia-smi -lgc 300,2200" ' + KEY_FILES["fleet-clock-limit.pub"] in r.stdout
    assert 'command="sudo -n /usr/bin/nvidia-smi -lgc 300,2300" ' + KEY_FILES["fleet-clock-limit-2300.pub"] in r.stdout
    assert 'command="sudo -n /usr/bin/nvidia-smi -rgc" ' + KEY_FILES["fleet-clock-release.pub"] in r.stdout
    assert 'command="sudo -n -l /usr/bin/nvidia-smi -lgc 300,2200" ' + KEY_FILES["fleet-clock-dryrun.pub"] in r.stdout


def test_the_second_box_is_restricted_to_its_own_cable(tmp_path):
    make_keys(tmp_path)
    r = run(["box-b", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode == 0, r.stderr
    assert all('from="198.51.100.1"' in ln for ln in r.stdout.splitlines() if ln.startswith("restrict,"))


def test_the_sudoers_grant_names_only_the_three_clock_argvs_on_a_gpu_box(tmp_path):
    make_keys(tmp_path)
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    sudoers = [ln for ln in r.stdout.splitlines() if ln.startswith("fleet-power ALL=")]
    assert len(sudoers) == 1
    assert sudoers[0].count("nvidia-smi") == 3 and "-rgc" in sudoers[0]
    assert "NOPASSWD: /usr/bin/systemctl poweroff" in sudoers[0]


def test_box_c_has_no_gpu_keys_and_only_its_loopback_source(tmp_path):
    make_keys(tmp_path, sources=())                 # box-c's source is loopback: no from- file is read
    r = run(["box-c", "--local"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode == 0, r.stderr
    auth = [ln for ln in r.stdout.splitlines() if ln.startswith("restrict,")]
    assert len(auth) == 2 and all('from="127.0.0.1,::1"' in ln for ln in auth)
    assert "nvidia-smi" not in r.stdout


@pytest.mark.parametrize("args", [[], ["--gpu"], ["box-a", "--rack"], ["rack 9"], ["box-a", "gpu"]])
def test_a_missing_box_or_an_unknown_flag_is_a_usage_error(args):
    """Any name is a box (the inventory's, so its from-<box> file is read); what it IS comes from the flags, never
    from a list of one deployment's machines in the script."""
    r = run(args, {"FLEET_POWER_KEYS": "/nonexistent"})
    assert r.returncode == 2 and "usage" in r.stderr


def test_a_box_of_any_name_without_a_gpu_reads_its_own_cable_and_installs_two_keys(tmp_path):
    make_keys(tmp_path, sources=())
    (tmp_path / "from-rack9").write_text("203.0.113.9\n")
    r = run(["rack9"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode == 0, r.stderr
    auth = [ln for ln in r.stdout.splitlines() if ln.startswith("restrict,")]
    assert len(auth) == 2 and all('from="203.0.113.9"' in ln for ln in auth)
    assert "nvidia-smi" not in r.stdout


def test_the_source_may_be_a_cidr_and_nothing_else_reaches_the_from_option(tmp_path):
    make_keys(tmp_path)
    (tmp_path / "from-box-a").write_text("10.231.99.0/24\n")
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode == 0, r.stderr
    assert 'from="10.231.99.0/24"' in r.stdout


@pytest.mark.parametrize("hostile", ['192.0.2.1",command="id', "192.0.2.1,198.51.100.1", "example.test"])
def test_a_source_that_is_not_an_address_is_refused_before_it_is_rendered(tmp_path, hostile):
    make_keys(tmp_path)
    (tmp_path / "from-box-a").write_text(hostile + "\n")
    r = run(["box-a", "--gpu"], {"FLEET_POWER_KEYS": str(tmp_path), "FLEET_POWER_PRINT": "1"})
    assert r.returncode != 0 and "from-box-a" in r.stderr
    assert 'command="id' not in r.stdout
