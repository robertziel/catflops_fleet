"""PROMETHEUS `GET /metrics` ON THE GATEWAY (vllm_lb/prom.py).

Pinned here:
  * the exposition is rendered from the SAME status document the panel reads (proxy.status()) — one source;
  * an absent figure (None, a stale box's withdrawn live numbers, a key a sender does not send) is an OMITTED series,
    never a 0;
  * every box heard carries `collector_last_success_timestamp_seconds` (now − its sample age), stale or not;
  * the RoCE ports, kernel counters and disks of the stream-A contract are exported when present;
  * label values are escaped; every family has HELP and TYPE once;
  * the route is OPT-IN (LB_METRICS): off it is a 404 naming the switch; and it is the admin's — a member is refused.
"""
from __future__ import annotations

import http.client

import pytest

from vllm_lb import roles
from vllm_lb.config import Config, VllmBox
from vllm_lb.hub import Hub
from vllm_lb.prom import render
from vllm_lb.proxy import Proxy

NOW = 1_795_176_000.0


def status(**over):
    s = {
        "now": NOW,
        "backends": [
            {"kind": "vllm", "url": "http://10.0.0.1:8000", "box": "box-a", "cluster": None, "live": True,
             "waiting": 2, "running": 5, "inflight": 5, "depth_known": True, "tokens_per_s": 41.5,
             "gen_tokens_total": 1000, "prompt_tokens_total": 5000},
            {"kind": "cluster", "url": "http://10.0.0.2:8100", "box": "box-b", "cluster": "pair", "live": False,
             "waiting": 0, "running": 0, "inflight": 0, "depth_known": False, "tokens_per_s": None,
             "gen_tokens_total": None, "prompt_tokens_total": None},
        ],
        "clusters": {"pair": {"name": "pair", "state": "starting"}},
        "hardware": {
            "box-a": {"age_s": 2.0, "stale": False, "cpu_percent": 12.5, "ram_percent": 40.0, "gpu_pct": 77,
                      "mem_used_mb": 1024, "mem_total_mb": 2048,
                      "gpus": [{"id": 0, "use_pct": 77, "vram_used_mb": 1024, "vram_total_mb": 2048}],
                      "recorder": {"figures": {"gpu_temp_c": 61, "gpu_power_w": 150.5, "net_in_mbps": 8.0,
                                               "net_out_mbps": None}},
                      "rdma": {"ports": [{"dev": "mlx5_0", "port": 1, "state": "ACTIVE", "rate_gbps": 200.0,
                                          "mtu": 4096, "counters": {"out_of_buffer": 3, "port_rcv_errors": 0},
                                          "rates": {"out_of_buffer": 0.5}, "why": {}}], "why": None},
                      "kernel": {"xid_total": 1, "oom_kills_total": None, "last_xid": None, "why": None},
                      "disks": [{"path": "/models", "total_bytes": 100, "used_bytes": 60, "free_bytes": 40}]},
            "box-b": {"age_s": 120.0, "stale": True, "cpu_percent": None, "ram_percent": None, "gpu_pct": None,
                      "mem_used_mb": None, "mem_total_mb": None, "gpus": None, "recorder": {"figures": {}}},
        },
    }
    s.update(over)
    return s


def lines(text):
    return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]


def series(text, name):
    return [ln for ln in lines(text) if ln.split("{")[0].split(" ")[0] == name]


def test_backends_live_depth_and_rates():
    t = render(status())
    # a backend in no cluster has NO cluster label: in Prometheus an empty value IS an absent label, written out it is noise
    assert 'fleet_backend_up{url="http://10.0.0.1:8000",box="box-a",kind="vllm"} 1' in t
    assert 'fleet_backend_up{url="http://10.0.0.2:8100",box="box-b",kind="cluster",cluster="pair"} 0' in t
    assert len(series(t, "fleet_backend_requests_waiting")) == 1, "a depth nobody read is omitted, not 0"
    assert len(series(t, "fleet_backend_tokens_per_second")) == 1
    assert 'fleet_backend_generated_tokens_total{url="http://10.0.0.1:8000"' in t


def test_no_series_carries_an_empty_label():
    assert not [ln for ln in lines(render(status())) if '=""' in ln]


def test_the_cluster_state_is_a_labelled_gauge():
    t = render(status())
    assert 'fleet_cluster_state{cluster="pair",state="starting"} 1' in t


def test_box_figures_and_an_absent_one_is_omitted():
    t = render(status())
    assert 'fleet_box_cpu_percent{box="box-a"} 12.5' in t
    assert 'fleet_box_gpu_temperature_celsius{box="box-a"} 61' in t
    assert 'fleet_box_gpu_power_watts{box="box-a"} 150.5' in t
    assert 'fleet_box_network_receive_bytes_per_second{box="box-a"} 1000000' in t
    assert not series(t, "fleet_box_network_transmit_bytes_per_second"), "None is omitted"
    assert 'fleet_box_gpu_utilization_percent{box="box-a",card="0"} 77' in t
    assert 'fleet_box_gpu_memory_used_bytes{box="box-a"} 1073741824' in t
    assert not [ln for ln in lines(t) if ln.startswith("fleet_box_") and 'box="box-b"' in ln
                and "sample_age" not in ln], "a stale box's withdrawn numbers are omitted"


def test_every_box_heard_has_its_last_success_timestamp():
    t = render(status())
    assert f'collector_last_success_timestamp_seconds{{box="box-a"}} {int(NOW - 2.0)}' in t
    assert f'collector_last_success_timestamp_seconds{{box="box-b"}} {int(NOW - 120.0)}' in t


def test_rdma_kernel_and_disks_when_present():
    t = render(status())
    assert 'fleet_box_rdma_port_active{box="box-a",dev="mlx5_0",port="1"} 1' in t
    assert 'fleet_box_rdma_port_rate_gbps{box="box-a",dev="mlx5_0",port="1"} 200' in t
    assert 'fleet_box_rdma_counter_total{box="box-a",dev="mlx5_0",port="1",counter="out_of_buffer"} 3' in t
    assert 'fleet_box_gpu_xid_errors_total{box="box-a"} 1' in t
    assert not series(t, "fleet_box_oom_kills_total"), "an unreadable counter is omitted"
    assert 'fleet_box_disk_free_bytes{box="box-a",path="/models"} 40' in t


def test_a_box_without_the_new_keys_renders():
    s = status()
    for k in ("rdma", "kernel", "disks"):
        s["hardware"]["box-a"].pop(k)
    t = render(s)
    assert not series(t, "fleet_box_rdma_port_active") and not series(t, "fleet_box_disk_free_bytes")


def test_label_values_are_escaped_and_each_family_has_help_and_type_once():
    s = status()
    s["clusters"] = {'we"ird\\': {"state": "off\nline"}}
    t = render(s)
    assert 'fleet_cluster_state{cluster="we\\"ird\\\\",state="off\\nline"} 1' in t
    helps = [ln for ln in t.splitlines() if ln.startswith("# HELP ")]
    assert len(helps) == len(set(helps))
    for h in helps:
        name = h.split()[2]
        assert f"# TYPE {name} " in t


def test_no_nan_or_boolean_is_exported():
    s = status()
    s["hardware"]["box-a"]["cpu_percent"] = float("nan")
    s["hardware"]["box-a"]["ram_percent"] = True
    t = render(s)
    assert not series(t, "fleet_box_cpu_percent") and not series(t, "fleet_box_ram_percent")


# ── the route ───────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[False, True], ids=["off", "on"])
def proxy(request):
    cfg = Config(vllm_boxes=(VllmBox("box-a-gpu", "box-a", "http://127.0.0.1:9"),), listen_host="127.0.0.1",
                 listen_port=0, metrics_enabled=request.param)
    p = Proxy(cfg, Hub(cfg))
    p.start()
    yield p
    p.stop()


def _get(port, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    return r.status, r.getheader("Content-Type") or "", r.read().decode()


def test_the_route_is_opt_in_and_the_admins(proxy):
    status_, ctype, body = _get(proxy.port, "/metrics")
    if not proxy.cfg.metrics_enabled:
        assert status_ == 404 and "LB_METRICS" in body
        return
    assert status_ == 200 and ctype.startswith("text/plain")
    assert "fleet_backend_up{" in body
    status_, _, _ = _get(proxy.port, "/metrics", {roles.ROLE_HEADER: roles.MEMBER, roles.MODELS_HEADER: "x"})
    assert status_ == 403


def test_the_switch_is_read_strictly():
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_METRICS": "1"}).metrics_enabled is True
    assert Config.from_env({"LB_ALLOWED_HOSTS": "loopback"}).metrics_enabled is False
    with pytest.raises(ValueError, match="LB_METRICS"):
        Config.from_env({"LB_ALLOWED_HOSTS": "loopback", "LB_METRICS": "maybe"})
