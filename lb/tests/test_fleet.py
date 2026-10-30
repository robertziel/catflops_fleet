"""THE FLEET FILE (vllm_lb/fleet.py): what one deployment IS — boxes, every card in each, which cards serve, its
clusters and its plain classic servers — in YAML, validated at start, REFUSED loudly on anything missing, unknown or
inconsistent. No defaults, no auto-detection (AGENTS.md: a deployment is configuration, never code).

Two shapes are pinned, in generic names, because two fleets run it today:
  * PAIR — the box-c shape: a TP=2 pair of GB10s the LB drives over SSH, a classic server on each of the pair's
    boxes, and a box that only sends telemetry (an AMD iGPU);
  * ONE_HOST — the site-b shape: one box with two different cards, the second of which serves a one-host cluster
    the LB routes to but does not start.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fleet_files import LOCAL_ON_PAIR_BOX, ONE_HOST, PAIR
from vllm_lb.fleet import CONTROLS, Fleet, load_fleet, parse_fleet
from vllm_lb.hardware import profile

LB = Path(__file__).resolve().parent.parent


def _fleet(text: str) -> Fleet:
    return parse_fleet(text, "fleet.yml")


# ── the two shapes ──────────────────────────────────────────────────────────────────────────────────

def test_the_pair_shape_reads_as_boxes_cards_a_cluster_and_backends():
    f = _fleet(PAIR)
    assert list(f.boxes) == ["node-1", "node-2", "gateway"]
    n1 = f.boxes["node-1"]
    assert (n1.hostname, n1.label, n1.ssh) == ("node-1-host", "node-1-gpu", "operator@192.0.2.10")
    assert [(c.box, c.index, c.type.key, c.serves) for c in n1.cards] == [("node-1", 0, "gb10", True)]
    pair = f.clusters["pair"]
    assert (pair.control, pair.url, pair.tp) == ("ssh", "http://192.0.2.10:8100", 2)
    assert [(r.box, r.interconnect) for r in pair.ranks] == [("node-1", "198.51.100.1"), ("node-2", "198.51.100.2")]
    assert [(c.box, c.index) for c in pair.serving_cards] == [("node-1", 0), ("node-2", 0)], "head first"
    assert {label: b.url for label, b in f.backends.items()} == {"node-1": "http://192.0.2.10:8000",
                                                                 "node-2": "http://192.0.2.11:8000"}
    assert f.hw_hosts == {"node-1-host": "node-1", "node-2-host": "node-2", "gateway-host": "gateway"}


def test_the_one_host_shape_serves_on_the_card_it_names_never_its_neighbour():
    f = _fleet(ONE_HOST)
    solo = f.boxes["solo"]
    assert [(c.index, c.type.key, c.serves) for c in solo.cards] == [(0, "rtx3090", False), (1, "rtx5090", True)]
    assert [(c.index, c.type.key) for c in solo.serving_cards] == [(1, "rtx5090")]
    node = f.clusters["node"]
    assert (node.control, node.tp, node.ranks[0].interconnect) == ("none", 1, None)
    assert [c.type.key for c in node.serving_cards] == ["rtx5090"], "the 5090 it runs on, never the 3090"
    assert f.backends == {}


def test_an_ssh_cluster_states_the_fraction_of_each_boxs_ram_its_rank_may_take():
    """`ram_usable_fraction` (operator, 2026-10-26: "0.98 of ram can be used, even 0.99, make it settable"): what a
    start may take of each rank box's MemTotal. The rest — MemTotal x (1 - it) — is the ONE reserve the start gate,
    the KV auto pool and tp-rank's memory guard all keep. Stated, never defaulted."""
    assert _fleet(PAIR).clusters["pair"].ram_usable_fraction == 0.98
    with pytest.raises(ValueError, match="ram_usable_fraction"):
        _fleet(PAIR.replace("    ram_usable_fraction: 0.98\n", ""))


@pytest.mark.parametrize("bad", ["0", "1", "1.2", "-0.5", "x", "true", "null"])
def test_a_ram_usable_fraction_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError, match="ram_usable_fraction"):
        _fleet(PAIR.replace("ram_usable_fraction: 0.98", f"ram_usable_fraction: {bad}"))


def test_a_cluster_the_lb_does_not_launch_over_ssh_takes_no_ram_usable_fraction():
    """Only tp-rank's ranks are gated and guarded by it: on a node or a routed cluster it would be a setting nothing
    reads, refused as the unknown key it is there."""
    with pytest.raises(ValueError, match="ram_usable_fraction"):
        _fleet(ONE_HOST.replace("    control: none\n", "    control: none\n    ram_usable_fraction: 0.98\n"))


def test_an_ssh_cluster_states_the_prefill_budget_its_ranks_launch_with():
    """`max_num_batched_tokens` (operator, 2026-10-29: the compose default CLUSTER_MAX_NUM_BATCHED_TOKENS=8192 was an env
    number standing in for the deployment's own): the prefill budget a launch uses when neither the operator's knob
    nor the model's own (tuned / engine_args) says — vLLM's --max-num-batched-tokens, SGLang's chunk. The fleet
    states it per cluster, in tp-rank's own range; nothing defaults it."""
    assert _fleet(PAIR).clusters["pair"].max_num_batched_tokens == 8192
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _fleet(PAIR.replace("    max_num_batched_tokens: 8192\n", ""))
    for bad in ("1024", "65537", "x", "4096.5", "true", "null", "'8192'"):
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            _fleet(PAIR.replace("max_num_batched_tokens: 8192", f"max_num_batched_tokens: {bad}"))


def test_a_cluster_the_lb_does_not_launch_over_ssh_takes_no_prefill_budget():
    """A node's chunk is its launch block's own (`max_num_batched_tokens` there, optional); on the cluster it would be
    a second definition, refused as the unknown key it is there."""
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _fleet(ONE_HOST.replace("    control: none\n", "    control: none\n    max_num_batched_tokens: 8192\n"))


def test_an_ssh_cluster_names_the_interface_its_ranks_talk_over():
    """`iface` (open-source rule, AGENTS.md): the NIC NCCL and Gloo bind on each rank box (NCCL_SOCKET_IFNAME). It is
    this deployment's link — a bond, an RDMA port, eth0 — so the fleet states it, tp-rank never defaults it."""
    assert _fleet(PAIR).clusters["pair"].iface == "bond0"
    with pytest.raises(ValueError, match="iface"):
        _fleet(PAIR.replace("    iface: bond0\n", ""))


@pytest.mark.parametrize("bad", ["''", "' '", "0", "true", "null", "[bond0]"])
def test_an_interface_that_is_not_a_name_is_refused(bad):
    with pytest.raises(ValueError, match="iface"):
        _fleet(PAIR.replace("iface: bond0", f"iface: {bad}"))


def test_a_cluster_the_lb_does_not_launch_over_ssh_takes_no_interface():
    """Only tp-rank's ranks bind NCCL on it: elsewhere it is a setting nothing reads, refused as an unknown key."""
    with pytest.raises(ValueError, match="iface"):
        _fleet(ONE_HOST.replace("    control: none\n", "    control: none\n    iface: bond0\n"))


def test_an_ssh_cluster_may_list_the_rows_its_panel_offers():
    """`choices` on the pair (operator, 2026-10-26: "remove original censored from our vllm setup"): the catalog rows
    THIS deployment offers, `<id>` or `<id>@uncensored` — the catalog stays the shared library. Absent: every row."""
    assert _fleet(PAIR).clusters["pair"].offers is None
    f = _fleet(PAIR.replace("    ram_usable_fraction: 0.98\n", "    ram_usable_fraction: 0.98\n"
                            "    choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, glm-5.3-flash-nvfp4]\n", 1))
    assert f.clusters["pair"].offers == ("qwen3.8-27b-nvidia-nvfp4@uncensored", "glm-5.3-flash-nvfp4")


@pytest.mark.parametrize("bad, words", [
    ("[no-such-model]", "no-such-model"),
    ("[glm-5.3-flash-nvfp4@uncensored]", "uncensored"),                  # a twin the catalog does not have
    ("[glm-5.3-flash-nvfp4, glm-5.3-flash-nvfp4]", "twice"),
    ("[]", "choices"),
])
def test_a_pairs_choices_name_catalog_rows_once(bad, words):
    with pytest.raises(ValueError, match=words):
        _fleet(PAIR.replace("    ram_usable_fraction: 0.98\n", f"    ram_usable_fraction: 0.98\n    choices: {bad}\n", 1))


FP8_UP = "    kv_cache_dtypes: [fp8, fp8_e4m3, fp8_e5m2, nvfp4, fp4_mx_block16, int4_per_token_head]\n"
OFFERS = "    choices: [qwen3.8-27b-nvidia-nvfp4@uncensored, glm-5.3-flash-nvfp4]\n"


def _pair_with(*lines: str) -> str:
    return PAIR.replace("    ram_usable_fraction: 0.98\n", "    ram_usable_fraction: 0.98\n" + "".join(lines), 1)


def test_an_ssh_cluster_may_list_the_kv_dtypes_it_runs():
    """`kv_cache_dtypes` (operator, 2026-10-26: "FP8 is better than FP16, skip FP16 everywhere, FP8 is default unless
    can run nvfp4"): the KV element types THIS deployment runs. Absent: every one the images take."""
    assert _fleet(PAIR).clusters["pair"].kv_dtypes is None
    f = _fleet(_pair_with(OFFERS, FP8_UP))
    assert f.clusters["pair"].kv_dtypes == ("fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4", "fp4_mx_block16", "int4_per_token_head")


def test_an_offered_row_whose_own_kv_dtype_is_not_listed_refuses_the_fleet():
    """The 27B DFlash2 vLLM row launches vLLM's own `auto` (bf16 here): offered under an FP8-up list, every Start of it
    would be refused — so the file is, naming the row and its dtype."""
    with pytest.raises(ValueError, match="qwen3.8-27b-dflash2.*auto"):
        _fleet(_pair_with("    choices: [qwen3.8-27b-dflash2]\n", FP8_UP))
    # No `choices`: every row is offered, that one too — and the first the catalog lists with a 16-bit pool is named
    # (TensorFold 0.6.6's MLX row, bf16 KV only).
    with pytest.raises(ValueError, match="(qwen3.8-27b-dflash2.*auto|tensorfold-mlx-mtp runs KV bfloat16)"):
        _fleet(_pair_with(FP8_UP))


@pytest.mark.parametrize("bad, words", [("[fp9]", "fp9"), ("[]", "kv_cache_dtypes"), ("[fp8, fp8]", "twice")])
def test_a_kv_dtype_list_names_known_dtypes_once(bad, words):
    with pytest.raises(ValueError, match=words):
        _fleet(_pair_with(OFFERS, f"    kv_cache_dtypes: {bad}\n"))


def test_tp_is_derived_from_the_serving_cards_never_typed_twice():
    """One place per value: a cluster's tensor-parallel size IS the cards serving under its ranks, so a `tp:` beside
    them would be a second definition — refused as the unknown key it is."""
    with pytest.raises(ValueError, match="tp"):
        _fleet(PAIR.replace("    control: ssh\n", "    control: ssh\n    tp: 2\n"))


def test_the_committed_examples_are_valid_fleets():
    for name in ("fleet.example.yml", "fleet.one-host.example.yml", "fleet.parallel.example.yml"):
        f = load_fleet(LB / name)
        assert f.boxes and f.clusters, name


# ── refusals: each names what is wrong ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, says", [
    (PAIR + "extras: {}\n", "extras"),                                                     # unknown section
    (PAIR.replace("backends:\n  node-1: {url: http://192.0.2.10:8000}\n  node-2: {url: http://192.0.2.11:8000}\n",
                  ""), "backends"),                                                        # a section missing
    (PAIR.replace("    hostname: node-1-host\n", ""), "hostname"),                          # a box key missing
    (PAIR.replace("    label: node-1-gpu\n", "    label: node-1-gpu\n    colour: red\n"), "colour"),
    (PAIR.replace("type: radeon8060s", "type: radeon9999"), "radeon9999"),                  # an unknown type
    (PAIR.replace("{index: 0, type: radeon8060s, serves: false}",
                  "{index: 0, type: radeon8060s, serves: no}"), "serves"),                  # only true/false are booleans
    (PAIR.replace("{index: 0, type: radeon8060s, serves: false}",
                  "{index: 0, type: radeon8060s}"), "serves"),                              # never a default
    (ONE_HOST.replace("{index: 1, type: rtx5090", "{index: 0, type: rtx5090"), "index"),     # an index twice
    (ONE_HOST.replace("{index: 1, type: rtx5090", "{index: -1, type: rtx5090"), "index"),
    (ONE_HOST.replace("{index: 0, type: rtx3090", "{index: 0, type: gb10"), "memory"),       # unified beside discrete
    (PAIR.replace("hostname: node-2-host", "hostname: node-1-host"), "node-1-host"),         # a hostname twice
    (PAIR.replace("label: node-2-gpu", "label: node-1-gpu"), "node-1-gpu"),                  # a label twice
    (PAIR.replace("control: ssh", "control: kubernetes"), "kubernetes"),                     # an unknown control
    (PAIR.replace("    url: http://192.0.2.10:8100\n", ""), "url"),
    (PAIR.replace("url: http://192.0.2.10:8100", "url: 192.0.2.10:8100"), "url"),            # not an http(s) URL
    (PAIR.replace("{box: node-2, interconnect: 198.51.100.2}", "{box: node-9, interconnect: 198.51.100.2}"),
     "node-9"),                                                                            # a rank on no box
    (PAIR.replace("      - {box: node-2, interconnect: 198.51.100.2}\n", ""), "2 ranks"),    # tp-rank runs a pair
    (PAIR.replace("{box: node-2, interconnect: 198.51.100.2}", "{box: node-2}"), "interconnect"),
    (PAIR.replace("    ssh: operator@192.0.2.11\n", ""), "ssh"),                             # ssh cluster, no ssh
    (ONE_HOST.replace("{box: solo}", "{box: solo, interconnect: 198.51.100.1}"), "interconnect"),
    (PAIR.replace("{index: 0, type: gb10, serves: true}\n  node-2",
                  "{index: 0, type: gb10, serves: false}\n  node-2"), "node-1"),           # a rank on no serving card
    (PAIR.replace("type: gb10, serves: true}\n  gateway", "type: rtx5090, serves: true}\n  gateway"),
     "rtx5090"),                                                                           # tp-rank drives GB10s only
    (PAIR.replace("{index: 0, type: radeon8060s, serves: false}",
                  "{index: 0, type: radeon8060s, serves: true}"), "gateway"),               # serves, but nothing runs
    (PAIR.replace("  node-2: {url: http://192.0.2.11:8000}\n", "  gateway: {url: http://192.0.2.12:8000}\n"),
     "gateway"),                                                                           # a backend on no serving card
    (PAIR.replace("  node-2: {url: http://192.0.2.11:8000}\n", "  node-7: {url: http://192.0.2.11:8000}\n"),
     "node-7"),                                                                            # a backend on no box
    (PAIR.replace("    label: node-2-gpu\n", ""), "label"),                                 # a serving box needs a label
    (PAIR + "  node-3: {url: http://192.0.2.13:8000}\nboxes: {}\n", "boxes"),               # a YAML key twice
    ("", "mapping"),
    ("- a\n- list\n", "mapping"),
])
def test_a_broken_fleet_is_refused_naming_the_value(text, says):
    with pytest.raises(ValueError) as e:
        _fleet(text)
    assert says in str(e.value) and "fleet.yml" in str(e.value)


def test_two_clusters_on_the_same_boxes_are_alternatives_the_file_says_which_boots():
    """Several clusters are a catalog (worklog/parallel-clusters.md); on the same boxes they are alternatives."""
    two = PAIR.replace("    control: ssh\n", "    control: ssh\n    run_at_boot: true\n    default: true\n").replace("backends:", """  again:
    control: ssh
    iface: bond0
    link: roce
    run_at_boot: false
    url: http://192.0.2.12:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks:
      - {box: node-1, interconnect: 198.51.100.1}
      - {box: node-2, interconnect: 198.51.100.2}
backends:""")
    assert [c.run_at_boot for c in _fleet(two).clusters.values()] == [True, False]
    with pytest.raises(ValueError, match="run_at_boot"):
        _fleet(two.replace("    run_at_boot: false\n", ""))


def test_a_missing_file_is_refused_naming_it(tmp_path):
    with pytest.raises(ValueError, match="nope.yml"):
        load_fleet(tmp_path / "nope.yml")
    with pytest.raises(ValueError, match="not a file"):
        load_fleet(tmp_path)


def test_every_control_method_says_what_it_runs():
    assert set(CONTROLS) == {"ssh", "local", "none"}
    assert CONTROLS["ssh"].ranks == 2 and CONTROLS["ssh"].cards_per_rank == 1
    assert CONTROLS["local"].ranks == 1 and CONTROLS["local"].cards_per_rank == 1
    assert CONTROLS["none"].ranks is None


def test_every_type_names_only_control_methods_that_exist():
    from vllm_lb.hardware import types
    for key, kind in types().items():
        assert set(kind.controls) <= set(CONTROLS), (key, kind.controls)


def test_a_card_reads_as_its_box_index_type_and_whether_it_serves():
    card = _fleet(ONE_HOST).boxes["solo"].cards[1]
    assert card.type is profile("rtx5090")
    assert card.view() == {"box": "solo", "index": 1, "type": "rtx5090", "name": "NVIDIA GeForce RTX 5090",
                           "serves": True}


# ── a local node beside the pair, on one of its boxes ───────────────────────────────────────────────

def test_a_local_node_may_share_a_box_whose_ssh_the_pair_uses():
    f = _fleet(LOCAL_ON_PAIR_BOX)
    assert f.clusters["speech"].control == "local" and f.boxes["node-2"].ssh == "operator@192.0.2.11"


def test_an_ssh_no_cluster_uses_is_still_refused_on_a_local_nodes_box():
    lonely = LOCAL_ON_PAIR_BOX.replace("""  pair:
    control: ssh
    run_at_boot: true
    iface: bond0
    link: roce
    url: http://192.0.2.10:8100
    ram_usable_fraction: 0.98
    max_num_batched_tokens: 8192
    ranks:
      - {box: node-1, interconnect: 198.51.100.1}
      - {box: node-2, interconnect: 198.51.100.2}
""", "")
    with pytest.raises(ValueError, match="node-2.*nothing would use it"):
        _fleet(lonely)


def test_a_local_node_that_says_reserve_false_holds_no_box_through_the_broker():
    """node-2 has a classic server (backends): its lane is the broker's, the node shares the card beside it (allowed by
    broker_parrot's GPU guard) — so it takes no reservation. The pair still does."""
    f = _fleet(LOCAL_ON_PAIR_BOX)
    assert (f.clusters["speech"].reserves, f.clusters["pair"].reserves) == (False, True)
    with pytest.raises(ValueError, match="reserve"):
        _fleet(LOCAL_ON_PAIR_BOX.replace("    reserve: false\n", "    reserve: true\n"))
    with pytest.raises(ValueError, match="node-2"):            # reserving, it would share node-2's lane name
        from vllm_lb.config import topology_of
        topology_of(_fleet(LOCAL_ON_PAIR_BOX.replace("    reserve: false\n", "")))
    from vllm_lb.config import topology_of
    assert [c.name for c in topology_of(f)["clusters"]] == ["pair", "speech"]
