"""The TP link rate: tp-rank reports monotonic RDMA byte counters, the loop turns two readings into GB/s."""
from types import SimpleNamespace

from vllm_lb.cluster_loop import ClusterLoop


def loop():
    o = ClusterLoop.__new__(ClusterLoop)        # only `_net_rate` and its memory are under test
    o._net_last, o._net_prev = {}, {}
    return o


def reading(t, tx, rx, lanes=256):
    return {"tx_bytes": tx, "rx_bytes": rx, "devices": 4, "pcie_gt_lanes": lanes, "t": t}


def test_the_first_reading_is_not_a_rate():
    assert loop()._net_rate("box-a", reading(100.0, 10**12, 10**12)) is None


def test_two_readings_are_gb_per_second_against_the_pcie_ceiling():
    o = loop()
    o._net_rate("box-a", reading(100.0, 10**12, 10**12))
    r = o._net_rate("box-a", reading(105.0, 10**12 + 3_300_000_000, 10**12 + 3_200_000_000))
    # 2 PCIe Gen5 x4 links: 256 GT/s·lanes × 128/130 ÷ 8 = 31.5 GB/s a direction, not the 50 GB/s ports
    assert r == {"tx_gbps": 0.66, "rx_gbps": 0.64, "link_gbps": 31.5, "devices": 4, "window_s": 5.0}


def test_a_counter_that_went_backwards_is_a_reset_not_a_negative_rate():
    o = loop()
    o._net_rate("box-a", reading(100.0, 10**12, 10**12))
    assert o._net_rate("box-a", reading(105.0, 5, 5)) is None
    # …and the reset reading is the new baseline
    assert o._net_rate("box-a", reading(110.0, 5 + 10**9, 5 + 10**9))["tx_gbps"] == 0.2


def test_a_box_with_no_rdma_reports_none_never_zero():
    o = loop()
    assert o._net_rate("box-b", None) is None
    assert o._net_rate("box-b", None) is None


def test_a_gap_drops_the_baseline():
    o = loop()
    o._net_rate("box-a", reading(100.0, 10**12, 10**12))
    o._net_rate("box-a", None)                   # the box did not answer this tick
    assert o._net_rate("box-a", reading(110.0, 10**12 + 10**9, 10**12)) is None


def test_boxes_are_independent():
    o = loop()
    o._net_rate("box-a", reading(100.0, 0, 0))
    o._net_rate("box-b", reading(100.0, 0, 0))
    assert o._net_rate("box-a", reading(101.0, 10**9, 0))["tx_gbps"] == 1.0
    assert o._net_rate("box-b", reading(101.0, 2 * 10**9, 0))["tx_gbps"] == 2.0


def test_the_same_reading_seen_twice_in_a_tick_keeps_its_rate_not_null():
    """The view and the planner both observe in one tick; the second must not publish a gap."""
    o = loop()
    o._net_rate("box-a", reading(100.0, 0, 0))
    first = o._net_rate("box-a", reading(105.0, 5 * 10**9, 5 * 10**9))
    again = o._net_rate("box-a", reading(105.0, 5 * 10**9, 5 * 10**9))
    assert first is not None and again == first
