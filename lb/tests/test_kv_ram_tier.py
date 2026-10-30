"""A KV TIER IN THE HOST'S RAM (vLLM's native OffloadingConnector, `--kv-offloading-size N --kv-offloading-backend
native`), read off the engine's own /metrics and published in /lb/status beside the KV pool — and a node's perf record
says what it runs, the tier included.

WHAT vLLM 2eb491e EXPOSES for the tier (vllm/vllm-openai@sha256:514cd427…, read in the image 2026-10-12):

  * its SIZE, in `vllm:cache_config_info{kv_offloading_size="32.0",kv_offloading_backend="native"}` — GiB, the whole
    buffer summed over TP ranks (config/cache.py:266-270), turned into the connector's `cpu_bytes_to_use` only for the
    native backend (config/vllm.py:1106-1123; lmcache keeps its storage in another process, :1124-1127). A pair with no
    tier says `kv_offloading_size="None"` beside the default backend word "native" (the TP=2 head's capture);
  * TRANSFER COUNTERS: `vllm:kv_offload_load_bytes_total` / `_load_time_total` (RAM → GPU, a reload) and
    `_store_bytes_total` / `_store_time_total` (GPU → RAM), offloading/metrics.py:25-30,65-80. The time is each
    transfer's CUDA-event elapsed time (cpu/gpu_worker.py:708-712), so bytes over time is the copy's own speed. These
    flat counters are created at their first increment (metrics.py:410-423); the deprecated
    `vllm:kv_offload_total_bytes_total{transfer_type=…}` twins are the same counter — the only spelling vLLM v0.31
    has, read into the same keys;
  * NO OCCUPANCY. `vllm:kv_offload_cpu_cache_usage_perc` is (allocated − free − evictable) / chunks
    (cpu/manager.py:452-459): the chunks PINNED by transfers in flight, with every stored, evictable block left out —
    "0.0 = idle" while the tier holds 32 GiB of prefixes. So the fill is DERIVED (2026-10-12, operator: "show how
    much data is filled"): an LRU tier frees a chunk only to make room, so it holds min(size, bytes stored).

The fixtures are two real reads of site-b's node 5 s apart (2026-10-11 23:52:52Z and 23:52:57Z), unedited.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from tests.test_cluster_local import RAMMED, RELAUNCH, Local, post, served  # noqa: F401  (served: fixture)
from tests.test_cluster_one_host import HOST, LIVE, NODE, scrape
from vllm_lb import perflog
from vllm_lb.config import Config
from vllm_lb.hub import HISTORY_RINGS, Hub
from vllm_lb.probe import parse_metrics
from vllm_lb.proxy import Proxy
from vllm_lb.rates import FIRST_SAMPLE, NOT_EXPOSED, SCRAPE_FAILED, WHY
from vllm_lb.throughput import OCCUPANCY_DERIVED

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "throughput"
TIER_1 = (FIXTURES / "vllm-2eb491e-ram-tier-1.metrics").read_text()
TIER_2 = (FIXTURES / "vllm-2eb491e-ram-tier-2.metrics").read_text()
#: The TP=2 pair's head: `kv_offloading_size="None"`, `kv_offloading_backend="native"` — no tier.
PAIR_1 = (FIXTURES / "vllm-v0.29.0-cluster-head-1.metrics").read_text()
PAIR_2 = (FIXTURES / "vllm-v0.29.0-cluster-head-2.metrics").read_text()

HEAD = NODE
T0 = 1_791_762_772.0                        # 2026-10-11 23:52:52Z, the first read
#: The counters as the two reads print them.
LOAD = (137_937_223_680, 145_027_104_768)
STORE = (56_078_303_232, 56_488_951_808)
LOAD_S, STORE_S = 10.976185746192936, 4.348286755442622


def cfg(**over) -> Config:
    """A node this LB only observes (any server's block is built the same way), probed every 5 s."""
    return Config(**{"cluster_name": "node", "cluster_hosts": (HOST,), "cluster_url": HEAD, "cluster_control": "none",
                     "listen_host": "127.0.0.1", "listen_port": 0, "probe_interval_s": 5, **over})


def feed(hub: Hub, text: str | None, at: float) -> None:
    """One probe result as probe_backend returns it: the parsed scrape (None = /metrics did not answer)."""
    got = parse_metrics(text) if text is not None else None
    hub.update_backend(HEAD, live=True, last_probe_at=at, mono_at=at, depth_known=got is not None,
                       waiting=(got or {}).get("waiting") or 0, running=(got or {}).get("running") or 0,
                       served_models=("Qwen/Qwen3.8-27B",), max_model_len=131072, scrape=got)


def backend(hub: Hub, config: Config) -> dict:
    return next(b for b in Proxy(config, hub).status()["backends"] if b["url"] == HEAD)


def whys(node) -> set:
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("why") and value is not None:
                found.add(value)
            found |= whys(value)
    return found


# ── the scrape ─────────────────────────────────────────────────────────────────────────────────


def test_the_scrape_reads_the_tier_size_and_the_four_transfer_counters_off_a_real_capture():
    got = parse_metrics(TIER_2)
    assert got["kv_offload_ram_gib"] == 32.0
    assert got["kv_offload_backend"] == "native"
    assert (got["kv_offload_load_bytes_total"], got["kv_offload_store_bytes_total"]) == (LOAD[1], STORE[1])
    assert got["kv_offload_load_seconds_total"] == pytest.approx(LOAD_S)
    assert got["kv_offload_store_seconds_total"] == pytest.approx(STORE_S)


def test_a_pair_without_a_tier_reads_none_of_it_although_it_names_the_default_backend():
    got = parse_metrics(PAIR_1)
    assert got["kv_offload_backend"] == "native", "the default word is printed whether a tier runs or not"
    for key in ("kv_offload_ram_gib", "kv_offload_load_bytes_total", "kv_offload_store_bytes_total",
                "kv_offload_load_seconds_total", "kv_offload_store_seconds_total"):
        assert got[key] is None, key


# ── /lb/status: the block, its rates and their past ───────────────────────────────────────────


def test_status_publishes_the_tier_its_reload_and_store_rates_and_the_copy_speed():
    config = cfg()
    hub = Hub(config)
    feed(hub, TIER_1, T0)
    feed(hub, TIER_2, T0 + 5)
    b = backend(hub, config)
    off = b["throughput"]["kv_offload"]
    assert off["tier_gib"] == 32.0, "the engine's own figure, not a constant"
    assert off["reload"]["bytes_s"] == pytest.approx((LOAD[1] - LOAD[0]) / 5)          # 1.418 GB/s
    assert off["store"]["bytes_s"] == pytest.approx((STORE[1] - STORE[0]) / 5)         # 0.082 GB/s
    assert (off["reload"]["window_s"], off["reload"]["why"]) == (5, None)
    assert (off["store"]["window_s"], off["store"]["why"]) == (5, None)
    # The COPY's own speed: every byte moved over every second of transfer the engine timed, since it started.
    assert off["reload"]["copy_gbps"] == pytest.approx(LOAD[1] / LOAD_S / 1e9)       # 13.21
    assert off["store"]["copy_gbps"] == pytest.approx(STORE[1] / STORE_S / 1e9)      # 12.99
    assert off["reload"]["copy_why"] is None and off["store"]["copy_why"] is None
    # OCCUPANCY IS DERIVED, never vLLM's pinned fraction (0.0 in these very reads): the tier had stored 56.5 GB, more
    # than its 32 GiB, so it is full (test_the_tier_is_as_full_as_it_has_stored below).
    assert (off["occupancy_pct"], off["used_gib"], off["occupancy_why"]) == (100.0, 32.0, None)
    assert whys(off) <= WHY
    # THE PAST, one sample per probe on the KV tile's clock and retention.
    h = b["history"]
    assert len(h["kv_reload_bytes_s"]) == len(h["kv_store_bytes_s"]) == len(h["at"]) == len(h["kv_pct"]) == 2
    assert h["kv_reload_bytes_s"][0] is None and h["kv_store_bytes_s"][0] is None, "a first sample is a gap"
    assert h["kv_reload_bytes_s"][1] == pytest.approx(off["reload"]["bytes_s"])
    assert h["kv_store_bytes_s"][1] == pytest.approx(off["store"]["bytes_s"])


def with_stored(text: str, stored: float | None) -> str:
    """A real read with its GPU → RAM byte counter set to `stored` — or its line removed (None)."""
    pattern = r"^vllm:kv_offload_store_bytes_total\{[^}]*\} \S+$"
    assert re.search(pattern, text, re.M), "the fixture carries the counter"
    if stored is None:                      # no counter at all: both spellings of it gone (the v0.31 one too)
        twin = r'^vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="GPU_to_CPU"[^}]*\} \S+$'
        return re.sub(twin + r"\n?", "", re.sub(pattern + r"\n?", "", text, flags=re.M), flags=re.M)
    return re.sub(pattern, lambda m: m.group(0).rsplit(" ", 1)[0] + f" {stored!r}", text, flags=re.M)


def test_the_tier_is_as_full_as_it_has_stored_up_to_its_size():
    """THE TIER'S FILL IS DERIVED from the engine's own counter: the tier is an LRU cache that frees a chunk only to
    make room for another (cpu/manager.py prepare_store :311-334, the evicted chunk reused at once), on a failed
    store (complete_store :384-388) or a cache reset (:429-441) — so what it holds is min(its size, the bytes stored
    since the engine started). 56.5 GB stored into 32 GiB: full, and every probe says so on the KV tile's clock."""
    config = cfg()
    hub = Hub(config)
    feed(hub, TIER_1, T0)
    feed(hub, TIER_2, T0 + 5)
    b = backend(hub, config)
    off = b["throughput"]["kv_offload"]
    assert (off["used_gib"], off["occupancy_pct"], off["occupancy_why"]) == (32.0, 100.0, None)
    assert off["occupancy_derived"] == OCCUPANCY_DERIVED
    assert OCCUPANCY_DERIVED.startswith("derived") and "upper bound" in OCCUPANCY_DERIVED
    assert "kv_ram_pct" in HISTORY_RINGS
    assert b["history"]["kv_ram_pct"] == [100.0, 100.0], "a full tier reads full from the first read: no rate needed"


def test_a_tier_still_filling_reads_the_bytes_it_stored():
    config = cfg()
    hub = Hub(config)
    feed(hub, with_stored(TIER_1, 4 * 2**30), T0)
    feed(hub, with_stored(TIER_2, 8 * 2**30), T0 + 5)
    b = backend(hub, config)
    off = b["throughput"]["kv_offload"]
    assert (off["used_gib"], off["occupancy_pct"]) == (8.0, 25.0)
    assert b["history"]["kv_ram_pct"] == [12.5, 25.0]


def test_a_tier_whose_store_counter_is_not_exposed_says_so():
    """No counter, no fill: vLLM creates it at the first store (metrics.py:410-423), so before one — or on a build
    without it — the fill is not known, never 0."""
    config = cfg()
    hub = Hub(config)
    feed(hub, with_stored(TIER_1, None), T0)
    b = backend(hub, config)
    off = b["throughput"]["kv_offload"]
    assert (off["used_gib"], off["occupancy_pct"], off["occupancy_why"]) == (None, None, NOT_EXPOSED)
    assert b["history"]["kv_ram_pct"] == [None]


def test_a_server_without_the_tier_has_no_fill_ring_samples():
    config = cfg()
    hub = Hub(config)
    feed(hub, PAIR_1, T0)
    assert backend(hub, config)["history"]["kv_ram_pct"] == [None]


def test_the_first_read_has_the_tier_but_no_rate_yet():
    config = cfg()
    hub = Hub(config)
    feed(hub, TIER_1, T0)
    off = backend(hub, config)["throughput"]["kv_offload"]
    assert off["tier_gib"] == 32.0
    assert (off["reload"]["bytes_s"], off["reload"]["why"]) == (None, FIRST_SAMPLE)
    assert off["reload"]["copy_gbps"] == pytest.approx(LOAD[0] / 10.40546222019196 / 1e9)


def test_a_server_without_the_tier_has_no_offload_block_at_all():
    config = cfg()
    hub = Hub(config)
    feed(hub, PAIR_1, T0)
    feed(hub, PAIR_2, T0 + 5)
    b = backend(hub, config)
    assert b["throughput"]["kv_offload"] is None, "no block — not zeros"
    assert b["history"]["kv_reload_bytes_s"] == [None, None] and b["history"]["kv_store_bytes_s"] == [None, None]


def test_the_tier_size_is_whatever_the_engine_says_and_lmcache_is_not_this_tier():
    config = cfg()
    hub = Hub(config)
    feed(hub, TIER_1.replace('kv_offloading_size="32.0"', 'kv_offloading_size="16.0"'), T0)
    assert backend(hub, config)["throughput"]["kv_offload"]["tier_gib"] == 16.0
    other = Hub(config)
    feed(other, TIER_1.replace('kv_offloading_backend="native"', 'kv_offloading_backend="lmcache"'), T0)
    assert backend(other, config)["throughput"]["kv_offload"] is None, "lmcache's storage is another process's"


def test_a_scrape_that_fails_keeps_the_tier_and_says_why_its_rates_are_missing():
    config = cfg()
    hub = Hub(config)
    feed(hub, TIER_1, T0)
    feed(hub, TIER_2, T0 + 5)
    feed(hub, None, T0 + 10)
    b = backend(hub, config)
    off = b["throughput"]["kv_offload"]
    assert off["tier_gib"] == 32.0
    assert (off["reload"]["bytes_s"], off["reload"]["why"]) == (None, SCRAPE_FAILED)
    assert b["history"]["kv_reload_bytes_s"][-1] is None


def test_the_rings_keep_the_kv_tiles_retention():
    config = cfg(stat_history=3)
    hub = Hub(config)
    for i, text in enumerate((TIER_1, TIER_2, TIER_2, TIER_2, TIER_2)):
        feed(hub, text, T0 + 5 * i)
    h = backend(hub, config)["history"]
    assert "kv_reload_bytes_s" in HISTORY_RINGS and "kv_store_bytes_s" in HISTORY_RINGS
    assert len(h["kv_reload_bytes_s"]) == len(h["kv_pct"]) == 3
    assert h["kv_reload_bytes_s"][-1] == 0.0, "a counter that did not move is a measured 0"


# ── a LOCAL node's perf record: what its host applied, and what the node says ─────────────────


#: The node's /metrics as the probe reads it while it runs the RELAUNCH: its pool, fraction and tier in its own words.
TIERED = {**LIVE, "scrape": scrape(kv_pool_bytes=8_501_810_176, gpu_memory_utilization=0.95,
                                   kv_offload_ram_gib=32.0, kv_offload_backend="native")}


def applied_view(tmp_path, served, launch: dict) -> dict:
    node = Local(tmp_path, TIERED, fleet=RAMMED)
    node.tick()
    status, out = post(served(node), {"enabled": True, **launch})
    assert status == 200, out
    node.tick()
    node.beat(applied=node.applied())
    return node.tick()


def test_a_local_nodes_perf_record_carries_its_applied_launch(tmp_path, served):
    view = applied_view(tmp_path, served, {**RELAUNCH, "max_num_batched_tokens": 2048})
    config, source = perflog.effective_config(view)
    assert source == perflog.EFFECTIVE
    assert config["kv_offload_ram_gib"] == 32, "the tier, which was recorded as kv_offload_gib=None"
    assert config["gpu_memory_utilization"] == 0.95
    assert config["kv_cache_memory"] == 8_501_810_176
    assert config["max_num_batched_tokens"] == 2048, "the applied intent's: the node publishes none"
    assert (config["max_num_seqs"], config["max_model_len"]) == (16, 131072)
    assert view["declared"]["checks"]["kv_offload_ram_gib"] == {"fleet": 32, "node": 32.0, "verdict": "agrees"}


def test_a_tier_change_is_another_configuration_in_the_perf_log(tmp_path, served):
    view = applied_view(tmp_path, served, RELAUNCH)
    acc = perflog.WindowAccumulator(60.0)
    s = {"running": 1, "waiting": 0, "gen_tokens_total": 100, "prompt_tokens_total": 100, "kv_cache_usage": 0.5}
    acc.observe(at=0.0, mono=0.0, live=True, scrape=s, cluster=view)
    acc.observe(at=5.0, mono=5.0, live=True, scrape={**s, "gen_tokens_total": 200}, cluster=view)
    untiered = {**view, "effective": {**view["effective"], "kv_offload_ram_gib": None}}
    closed = acc.observe(at=10.0, mono=10.0, live=True, scrape={**s, "gen_tokens_total": 300}, cluster=untiered)
    assert [w["config"]["kv_offload_ram_gib"] for w in closed] == [32]


def test_the_pairs_record_gains_no_key_it_cannot_state(tmp_path):
    pair = {"configured": True, "effective": {k: None for k in perflog.CONFIG_KEYS}}
    config, _ = perflog.effective_config(pair)
    assert "kv_offload_ram_gib" not in config and "gpu_memory_utilization" not in config


#: vLLM v0.31.1rc1 (the 5090 node, 2026-10-26) before its tier's first transfer: only the `transfer_type` families
#: exist (the flat counters are created at the first one). Their lines as the node exposed them, made non-zero here.
V031 = """\
# TYPE vllm:kv_offload_total_bytes_total counter
vllm:kv_offload_total_bytes_total{engine="0",model_name="Qwen/Qwen3.8-27B",transfer_type="CPU_to_GPU"} 2147483648.0
vllm:kv_offload_total_bytes_total{engine="0",model_name="Qwen/Qwen3.8-27B",transfer_type="GPU_to_CPU"} 4294967296.0
# TYPE vllm:kv_offload_total_time_total counter
vllm:kv_offload_total_time_total{engine="0",model_name="Qwen/Qwen3.8-27B",transfer_type="CPU_to_GPU"} 0.25
vllm:kv_offload_total_time_total{engine="0",model_name="Qwen/Qwen3.8-27B",transfer_type="GPU_to_CPU"} 0.5
"""


def test_a_v031_engine_states_its_transfers_as_one_family_split_by_transfer_type():
    """Operator 2026-10-26 at site-b's KV · RAM tile: "no data in site-b?" — until its tier's first transfer vLLM
    v0.31.1rc1 exposes only the `transfer_type` family (CPU_to_GPU = a reload, GPU_to_CPU = a store), so every figure
    said "not exposed by this server" beside a 32 GiB tier that was simply empty. Both names are the one counter."""
    got = parse_metrics(V031)
    assert (got["kv_offload_load_bytes_total"], got["kv_offload_store_bytes_total"]) == (2 ** 31, 2 ** 32)
    assert (got["kv_offload_load_seconds_total"], got["kv_offload_store_seconds_total"]) == (0.25, 0.5)
    zero = parse_metrics(V031.replace("2147483648.0", "0.0").replace("4294967296.0", "0.0"))
    assert zero["kv_offload_store_bytes_total"] == 0, "a tier that stored nothing yet is EMPTY, not unknown"
