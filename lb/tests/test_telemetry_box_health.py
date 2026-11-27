"""fleet_telemetry's BOX-HEALTH readers: the RoCE ports, the kernel's counters, the disks and the
models roots' folder sizes — each against what its source states, each absent with a reason when it cannot be read.

The rules pinned:
  * RDMA (readers/rdma.py): one row per port under /sys/class/infiniband, its state / rate / MTU and the error counters
    sysfs states, CUMULATIVE as read (the gateway derives the rates). A counter the device does not expose is absent
    with why — never 0. No /sys/class/infiniband → no ports, with why. Counters sysfs never states (rx_discards_phy,
    PFC, trust) are named once, with where they live.
  * the kernel (readers/kernel.py): OOM kills from /proc/vmstat `oom_kill`; NVRM Xid records off /dev/kmsg, counted and
    the last one kept. /dev/kmsg missing or refused → the Xid figures absent with why, the OOM count still read. A
    ring buffer that no longer holds the boot's first record makes the count a count SINCE its first record — said.
  * disks (readers/disk.py): statvfs over EXACTLY the paths the sender's config names, as `df` counts them; a path that
    cannot be statted is that path's why.
  * models roots (readers/models.py): each top-level folder's on-disk size (`du -s`: blocks, a hard link once, a
    symlink not followed), on a slow cadence of its own, with when it was measured and how long it took.
"""
from __future__ import annotations

import json
import os

import pytest

from fleet_telemetry.readers.base import Binds
from fleet_telemetry.readers.disk import Disks
from fleet_telemetry.readers.kernel import KernelLog, parse_kmsg
from fleet_telemetry.readers.models import ModelDirs
from fleet_telemetry.readers.rdma import PHY_COUNTERS, RdmaPorts
from tests.telemetry_fixtures import (DCB_APP_DSCP, DCB_APPTRUST, DCB_PFC, ETHTOOL_S_MLX5, KMSG_RECORDS, RDMA_TWO_PORTS,
                                      VMSTAT, plant)


# ── RDMA ports ────────────────────────────────────────────────────────────────

class Tools:
    """ethtool / dcb as the sender runs them: argv → stdout, or the tool's refusal (RuntimeError with its words,
    FileNotFoundError when the binary is not in the image). Records every call."""

    def __init__(self, answers=None, clock=None):
        self.calls = []
        self.answers = answers if answers is not None else {
            ("ethtool", "-S", "enp1s0f0np0"): ETHTOOL_S_MLX5,
            ("ethtool", "-S", "enp2s0np0"): RuntimeError("exit 96: Cannot get stats strings information: "
                                                        "No such device"),
            ("dcb", "pfc", "show", "dev", "enp1s0f0np0", "prio-pfc"): DCB_PFC,
            ("dcb", "apptrust", "show", "dev", "enp1s0f0np0", "order"): RuntimeError(
                "exit 1: Attribute read: Operation not supported"),
            ("dcb", "-N", "app", "show", "dev", "enp1s0f0np0", "dscp-prio"): DCB_APP_DSCP,
        }

    def __call__(self, argv):
        self.calls.append(tuple(argv))
        a = self.answers.get(tuple(argv), RuntimeError("exit 1: Attribute read: No such device"))
        if isinstance(a, Exception):
            raise a
        return a


def ports(tmp_path, tools=None, clock=lambda: 100.0):
    return RdmaPorts(root=str(plant(tmp_path, RDMA_TWO_PORTS)), run=tools or Tools(), clock=clock)


def test_each_rdma_port_is_its_state_rate_mtu_and_cumulative_counters(tmp_path):
    a, b = ports(tmp_path).read()
    assert (a.dev, a.port, a.state, a.phys_state, a.rate_gbps, a.link_layer) == (
        "mlx5_0", 1, "ACTIVE", "LinkUp", 200.0, "Ethernet")
    assert (a.netdev, a.mtu) == ("enp1s0f0np0", 9000)
    assert a.counters["out_of_buffer"] == 1234 and a.counters["local_ack_timeout_err"] == 7
    assert a.counters["port_rcv_errors"] == 0, "a counter at 0 is a reading"
    assert set(a.why) == {"trust"}, a.why
    assert (b.dev, b.state, b.rate_gbps) == ("mlx5_1", "DOWN", 40.0)


def test_the_phy_counters_are_read_from_ethtool_of_the_ports_netdev_cumulative(tmp_path):
    a, _ = ports(tmp_path).read()
    assert a.counters["rx_discards_phy"] == 4242 and a.counters["tx_discards_phy"] == 0
    assert a.counters["rx_crc_errors_phy"] == 3 and a.counters["rx_pause_ctrl_phy"] == 17
    assert set(PHY_COUNTERS) <= set(a.counters), "every PHY counter this mlx5e netdev states"


def test_a_phy_counter_the_netdev_does_not_state_is_absent_naming_ethtool(tmp_path):
    t = Tools()
    t.answers[("ethtool", "-S", "enp1s0f0np0")] = "NIC statistics:\n     rx_packets: 10\n     rx_discards_phy: 1\n"
    a, _ = ports(tmp_path, t).read()
    assert a.counters["rx_discards_phy"] == 1
    assert "tx_discards_phy" not in a.counters and "ethtool -S enp1s0f0np0" in a.why["tx_discards_phy"]


def test_a_refused_ethtool_leaves_the_phy_counters_absent_with_its_words_never_zero(tmp_path):
    t = Tools()
    t.answers[("ethtool", "-S", "enp1s0f0np0")] = RuntimeError("exit 94: no stats available")
    a, _ = ports(tmp_path, t).read()
    assert not set(PHY_COUNTERS) & set(a.counters)
    assert a.why["ethtool"] == "ethtool -S: exit 94: no stats available", "one reason for every PHY counter"


def test_a_netdev_in_another_namespace_is_not_asked_and_its_figures_say_so(tmp_path):
    t = Tools()
    _, b = ports(tmp_path, t).read()
    assert not set(PHY_COUNTERS) & set(b.counters) and b.pfc_prios is None
    assert "namespace" in b.why["ethtool"] and "namespace" in b.why["qos"]
    assert not [c for c in t.calls if "enp2s0np0" in c], "no tool is run for a netdev this namespace lacks"


def test_a_missing_tool_is_the_why(tmp_path):
    t = Tools(answers={})
    t.answers[("ethtool", "-S", "enp1s0f0np0")] = FileNotFoundError(2, "No such file or directory", "ethtool")
    t.answers[("dcb", "pfc", "show", "dev", "enp1s0f0np0", "prio-pfc")] = FileNotFoundError(2, "No such file", "dcb")
    a, _ = ports(tmp_path, t).read()
    assert "rx_discards_phy" not in a.counters and "not in the sender image" in a.why["ethtool"]
    assert a.pfc_prios is None and "dcb is not in the sender image" in a.why["pfc"]


def test_one_ethtool_call_per_netdev_per_sample(tmp_path):
    t = Tools()
    p = ports(tmp_path, t)
    p.read()
    p.read()
    eth = [c for c in t.calls if c[0] == "ethtool"]
    assert eth == [("ethtool", "-S", "enp1s0f0np0")] * 2, "one call per netdev per sample"


def test_pfc_and_trust_from_dcb_on_their_own_slow_cadence_with_their_age(tmp_path):
    t, now = Tools(), [100.0]
    p = ports(tmp_path, t, clock=lambda: now[0])
    a, _ = p.read()
    assert a.pfc_prios == [3] and a.dscp_map == 4 and a.qos_at == 100.0
    assert a.trust is None and "Operation not supported" in a.why["trust"], "the tool's words, never a guess"
    dcb_calls = len([c for c in t.calls if c[0] == "dcb"])
    now[0] = 130.0
    a, _ = p.read()
    assert len([c for c in t.calls if c[0] == "dcb"]) == dcb_calls, "a QoS setting is re-read every QOS_EVERY_S"
    assert a.pfc_prios == [3] and a.qos_at == 100.0, "the last reading with its time"
    now[0] = 100.0 + RdmaPorts.QOS_EVERY_S
    p.read()
    assert len([c for c in t.calls if c[0] == "dcb"]) > dcb_calls


def test_apptrust_order_when_the_driver_states_it(tmp_path):
    t = Tools()
    t.answers[("dcb", "apptrust", "show", "dev", "enp1s0f0np0", "order")] = DCB_APPTRUST
    a, _ = ports(tmp_path, t).read()
    assert a.trust == ["dscp", "pcp"] and "trust" not in a.why


def test_pfc_all_off_is_an_empty_list_not_absent(tmp_path):
    t = Tools()
    t.answers[("dcb", "pfc", "show", "dev", "enp1s0f0np0", "prio-pfc")] = (
        "prio-pfc 0:off 1:off 2:off 3:off 4:off 5:off 6:off 7:off\n")
    a, _ = ports(tmp_path, t).read()
    assert a.pfc_prios == [] and "pfc" not in a.why


def test_a_counter_the_device_does_not_expose_is_absent_with_why_never_zero(tmp_path):
    _, b = ports(tmp_path).read()
    assert "out_of_buffer" not in b.counters
    assert "hw_counters/" in b.why["hw_counters"], "one reason for the whole missing directory"


def test_an_mtu_whose_netdev_is_not_in_this_namespace_is_absent_with_why(tmp_path):
    _, b = ports(tmp_path).read()
    assert b.netdev == "enp2s0np0" and b.mtu is None and "enp2s0np0" in b.why["mtu"]


def test_no_infiniband_class_is_a_failure_that_says_so(tmp_path):
    with pytest.raises(OSError, match="/sys/class/infiniband"):
        RdmaPorts(root=str(tmp_path), run=Tools()).read()


# ── the kernel ────────────────────────────────────────────────────────────────

class FakeKmsg:
    def __init__(self, chunks, lost=0):
        self.chunks, self.lost = list(chunks), lost

    def drain(self):
        out, self.chunks = self.chunks, []
        lost, self.lost = self.lost, 0
        return out, lost


def test_kmsg_records_parse_to_seq_time_and_message():
    recs = parse_kmsg(KMSG_RECORDS)
    assert recs[0] == (0, 0, "Linux version 6.8.0")
    assert recs[2][0] == 1802 and recs[2][2].startswith("NVRM: Xid (PCI:0000:01:00): 79")
    assert all(not m.startswith(" ") for _s, _t, m in recs), "a continuation line is not a record"


def test_xids_are_counted_off_the_kernel_log_and_ooms_off_vmstat(tmp_path):
    root = plant(tmp_path, {"proc/vmstat": VMSTAT, "proc/uptime": "1000.00 7000.00"})
    k = KernelLog(root=str(root), kmsg=lambda: FakeKmsg(KMSG_RECORDS), clock=lambda: 50_000.0)
    r = k.read()
    assert (r.oom_kills_total, r.xid_total) == (3, 2)
    assert r.last_xid == {"code": 13, "at": pytest.approx(50_000.0 - 1000.0 + 905.5)}
    assert r.xid_complete is True and r.xid_since == pytest.approx(49_000.0), "seq 0 read: counted from boot"
    assert r.why is None


def test_new_xids_add_to_the_count_on_the_next_read(tmp_path):
    root = plant(tmp_path, {"proc/vmstat": VMSTAT, "proc/uptime": "1000.00 7000.00"})
    fake = FakeKmsg(KMSG_RECORDS)
    k = KernelLog(root=str(root), kmsg=lambda: fake, clock=lambda: 50_000.0)
    k.read()
    fake.chunks = ["6,4000,990000000,-;NVRM: Xid (PCI:0000:01:00): 31, pid=9, Ch 00000008\n"]
    r = k.read()
    assert r.xid_total == 3 and r.last_xid["code"] == 31


def test_a_ring_buffer_past_its_first_record_counts_since_its_first(tmp_path):
    root = plant(tmp_path, {"proc/vmstat": VMSTAT, "proc/uptime": "1000.00 7000.00"})
    k = KernelLog(root=str(root), kmsg=lambda: FakeKmsg(KMSG_RECORDS[1:]), clock=lambda: 50_000.0)
    r = k.read()
    assert r.xid_complete is False and r.xid_since == pytest.approx(49_000.0 + 12.0)


def test_lost_records_make_the_count_incomplete(tmp_path):
    root = plant(tmp_path, {"proc/vmstat": VMSTAT, "proc/uptime": "1000.00 7000.00"})
    k = KernelLog(root=str(root), kmsg=lambda: FakeKmsg(KMSG_RECORDS, lost=1), clock=lambda: 50_000.0)
    assert k.read().xid_complete is False


@pytest.mark.parametrize("exc,said", [(FileNotFoundError(2, "No such file"), "kmsg.yml"),
                                      (PermissionError(1, "Operation not permitted"), "CAP_SYSLOG")])
def test_an_unreadable_kernel_log_is_the_xid_figures_why_and_the_oom_count_still_reads(tmp_path, exc, said):
    root = plant(tmp_path, {"proc/vmstat": VMSTAT, "proc/uptime": "1000.00 7000.00"})

    def refuse():
        raise exc
    r = KernelLog(root=str(root), kmsg=refuse, clock=lambda: 1.0).read()
    assert r.xid_total is None and r.last_xid is None and said in r.why
    assert r.oom_kills_total == 3


def test_a_kernel_without_oom_kill_in_vmstat_is_absent_with_why(tmp_path):
    root = plant(tmp_path, {"proc/vmstat": "nr_free_pages 1\n", "proc/uptime": "1.0 1.0"})
    r = KernelLog(root=str(root), kmsg=lambda: FakeKmsg(KMSG_RECORDS), clock=lambda: 50_000.0).read()
    assert r.oom_kills_total is None and "oom_kill" in r.why and r.xid_total == 2


# ── disks ─────────────────────────────────────────────────────────────────────

def binds(tmp_path, *paths) -> Binds:
    """The sender's view: `tmp_path/fs` is the bind prefix (HOST_FS), and a mountinfo lists a bind per path given."""
    fs = tmp_path / "fs"
    lines = [f"{30 + i} 1 8:1 {p.replace(' ', chr(92) + '040')} {os.path.normpath(str(fs) + p).replace(' ', chr(92) + '040')} ro,relatime - ext4 "
             f"/dev/sda1 ro" for i, p in enumerate(paths)]
    (tmp_path / "mountinfo").write_text("\n".join(["1 0 0:1 / / rw - overlay overlay rw", *lines]) + "\n")
    return Binds(str(fs), str(tmp_path / "mountinfo"))


def test_a_bind_is_found_in_mountinfo_spaces_unescaped(tmp_path):
    b = binds(tmp_path, "/", "/mnt/my models")
    assert b.target("/") == str(tmp_path / "fs") and b.missing("/") is None
    assert b.missing("/mnt/my models") is None
    why = b.missing("/data")
    assert f"- /data:{tmp_path / 'fs'}/data:ro" in why and "disks.yml" in why


def test_an_unreadable_mountinfo_is_every_paths_why(tmp_path):
    b = Binds(str(tmp_path / "fs"), str(tmp_path / "nope"))
    assert "mountinfo" in b.missing("/")

def test_each_named_path_is_statvfs_as_df_counts_it(tmp_path):
    (tmp_path / "fs" / "models").mkdir(parents=True)
    st = os.statvfs(tmp_path / "fs" / "models")
    (d,) = Disks(["/models"], binds(tmp_path, "/models")).read()
    assert d.path == "/models" and d.why is None
    assert d.total_bytes == st.f_blocks * st.f_frsize
    # the test host's disk is live: free / used move between the two reads, so to within 1 GiB
    assert d.free_bytes == pytest.approx(st.f_bavail * st.f_frsize, abs=2 ** 30), "free to a user, as df's Avail"
    assert d.used_bytes == pytest.approx((st.f_blocks - st.f_bfree) * st.f_frsize, abs=2 ** 30), "used as df's Used"


def test_a_path_that_cannot_be_statted_is_its_own_why(tmp_path):
    (tmp_path / "fs" / "ok").mkdir(parents=True)
    ok, gone = Disks(["/ok", "/gone"], binds(tmp_path, "/ok", "/gone")).read()
    assert ok.why is None
    assert gone.total_bytes is None and gone.used_bytes is None and "/gone" in gone.why


def test_a_configured_path_with_no_bind_is_absent_naming_the_missing_bind_never_the_containers_disk(tmp_path):
    (tmp_path / "fs" / "data").mkdir(parents=True)          # the directory exists in the container, unbound
    (d,) = Disks(["/data"], binds(tmp_path)).read()
    assert (d.total_bytes, d.used_bytes, d.free_bytes) == (None, None, None)
    assert "not bind-mounted" in d.why and f"- /data:{tmp_path / 'fs'}/data:ro" in d.why


# ── models roots ──────────────────────────────────────────────────────────────

def _blocks(p) -> int:
    return os.lstat(p).st_blocks * 512


def test_each_top_level_folder_is_its_on_disk_size_measured_with_its_time(tmp_path):
    m = tmp_path / "fs" / "mnt" / "models"
    (m / "Qwen-A").mkdir(parents=True)
    (m / "Qwen-A" / "model.safetensors").write_bytes(b"x" * 100_000)
    (m / "hub" / "models--x" / "blobs").mkdir(parents=True)
    (m / "hub" / "models--x" / "snapshots" / "abc").mkdir(parents=True)
    blob = m / "hub" / "models--x" / "blobs" / "b1"
    blob.write_bytes(b"y" * 50_000)
    (m / "hub" / "models--x" / "snapshots" / "abc" / "model.safetensors").symlink_to(blob)
    os.link(blob, m / "hub" / "models--x" / "blobs" / "b1-hardlink")
    (m / "loose.bin").write_bytes(b"z")
    clock = iter([100.0, 103.5])
    (r,) = ModelDirs(["/mnt/models"], binds(tmp_path, "/mnt/models"), clock=lambda: next(clock)).walk()
    assert r.root == "/mnt/models" and r.why is None
    assert r.dirs["Qwen-A"] == _blocks(m / "Qwen-A" / "model.safetensors")
    assert r.dirs["hub"] == _blocks(blob), "the symlink is not followed and the hard link counts once"
    assert "loose.bin" not in r.dirs, "a file at the top is no folder"
    assert (r.measured_at, r.took_s) == (103.5, 3.5)


def test_a_models_root_that_is_not_there_is_its_why(tmp_path):
    (r,) = ModelDirs(["/nowhere"], binds(tmp_path, "/nowhere"), clock=lambda: 1.0).walk()
    assert r.dirs == {} and "/nowhere" in r.why


def test_a_models_root_with_no_bind_is_absent_naming_the_missing_bind(tmp_path):
    (tmp_path / "fs" / "m" / "x").mkdir(parents=True)
    (r,) = ModelDirs(["/m"], binds(tmp_path), clock=lambda: 1.0).walk()
    assert r.dirs == {} and r.measured_at is None and "not bind-mounted" in r.why


def test_before_the_first_walk_ends_the_root_says_it_is_measuring(tmp_path):
    md = ModelDirs(["/mnt/models"], binds(tmp_path, "/mnt/models"), clock=lambda: 10.0, start=False)
    (r,) = md.read()
    assert r.dirs == {} and r.measured_at is None and "first walk" in r.why


def test_the_largest_folders_are_sent_and_the_rest_counted(tmp_path):
    m = tmp_path / "fs" / "m"
    for i in range(ModelDirs.SENT + 5):
        (m / f"d{i:03d}").mkdir(parents=True)
        (m / f"d{i:03d}" / "f").write_bytes(b"x" * (4096 * (i + 1)))
    (r,) = ModelDirs(["/m"], binds(tmp_path, "/m"), clock=lambda: 1.0).walk()
    assert len(r.dirs) == ModelDirs.SENT and "d000" not in r.dirs
    assert r.more_count == 5 and r.more_bytes > 0


# ── on the wire (sampler.py) ──────────────────────────────────────────────────

def _wire(tmp_path, **readers):
    from fleet_telemetry.sampler import envelope
    from tests.test_telemetry_sample import apu_sampler
    return json.loads(envelope(apu_sampler(tmp_path, **readers).sample(), host="h", now=0.0))


def test_a_box_without_rdma_says_why_and_every_other_block_still_flows(tmp_path):
    w = _wire(tmp_path)
    assert w["rdma"] == {"ports": [], "why": "/sys/class/infiniband failed: no /sys/class/infiniband"}
    assert w["kernel"]["xid_total"] == 0 and w["disks"][0]["used_bytes"] == 40
    assert w["models"] == [] and "HW_SENDER_MODELS_DIRS=none" in w["models_why"]


def test_a_failing_kernel_reader_is_the_kernel_blocks_why(tmp_path):
    from tests.test_telemetry_sample import Fixed
    w = _wire(tmp_path, kernel=Fixed("/proc/vmstat", OSError("gone")))
    assert w["kernel"]["xid_total"] is None and "gone" in w["kernel"]["why"]


def test_every_disk_unreadable_is_disks_why(tmp_path):
    from fleet_telemetry.model import DiskUsage
    from tests.test_telemetry_sample import Fixed
    w = _wire(tmp_path, disks=Fixed("statvfs", [DiskUsage("/m", None, None, None, why="statvfs /m: gone")]))
    assert "statvfs /m: gone" in w["disks_why"] and w["disks"][0]["why"] == "statvfs /m: gone"


def test_the_health_blocks_add_under_4_kb_for_four_ports_and_a_full_models_root(tmp_path):
    from fleet_telemetry.model import ModelsRoot
    from tests.test_telemetry_sample import Fixed
    root = plant(tmp_path / "ib", RDMA_TWO_PORTS)
    ports = RdmaPorts(root=str(root), run=Tools(), clock=lambda: 1.0).read() * 2
    dirs = {("x" * 44) + f"{i:04d}": 10 ** 12 + i for i in range(ModelDirs.SENT)}
    w = _wire(tmp_path, rdma=Fixed("rdma", ports),
              models=Fixed("walk", [ModelsRoot("/mnt/models", dirs, 1.0, 2.0, 3, 4)]))
    assert len(w["rdma"]["ports"]) == 4 and len(w["models"][0]["dirs"]) == ModelDirs.SENT
    size = len(json.dumps({k: w[k] for k in ("rdma", "kernel", "disks", "disks_why", "models", "models_why")}, separators=(",", ":")))  # as on the wire
    assert size < 4096, size
