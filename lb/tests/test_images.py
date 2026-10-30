"""THE IMAGE MATRIX AND THE RESOLVER (vllm_lb/images.py, vllm_lb/resolve.py; library/images/<engine>.yml).

The regression the operator asked for (2026-10-30): every catalog row, resolved against the REFERENCE fleet (the
committed lb/fleet.example.yml — a TP=2 pair of GB10s, a classic server on each), must resolve to EXACTLY the image it
pinned before the image left the catalog. The live pair launches byte for byte what it launched before.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vllm_lb.catalog import CLASSIC, CLUSTER_MODE, ModelCatalog, document
from vllm_lb.fleet import catalog_deployment_of, catalog_targets_of
from vllm_lb.hardware import accelerator
from vllm_lb.images import ENTRY_KEYS, IMAGES_DIR, ImageMissing, image_for, images, load_images
from vllm_lb.resolve import Target, resolve

LB = Path(__file__).resolve().parent.parent
REFERENCE = LB / "fleet.example.yml"

#: What every row pinned in lb/vllm_lb/catalog.json at a576efce (`launch.<mode>.image`), typed from that file.
PINNED_BEFORE = {
    ("qwen3.8-flash-next-hibrid48", CLUSTER_MODE): "myllmbox/qwen38-flash-next-cluster-vllm:v5",
    ("qwen3.8-flash-next-sglang", CLUSTER_MODE): "myllmbox/sglang-27b-tp2:708f51e-fleet.4",
    ("qwen3.8-flash-next-tensorfold-nvfp4", CLUSTER_MODE): "tensorfold-qwen38fn:zig-db28187-f3e1300f4f62-fleet.2",
    ("qwen3.8-flash-next-tensorfold-mlx-mtp", CLUSTER_MODE): "tensorfold-py066:cb2ebf0",
    ("qwen3.8-27b-nvidia-nvfp4", CLUSTER_MODE): "myllmbox/qwen38-flash-next-cluster-vllm:v5",
    ("qwen3.8-27b-nvfp4", CLASSIC): "vllm/vllm-openai:v0.23.0",
    ("qwen3.8-27b-dflash2", CLUSTER_MODE): "myllmbox/qwen38-flash-next-cluster-vllm:v5",
    ("qwen3.8-27b-sglang-dflash2", CLUSTER_MODE): "myllmbox/sglang-27b-tp2:708f51e-fleet.2",
    ("glm-5.3-flash-nvfp4", CLUSTER_MODE): "myllmbox/glm53-flash-cluster-vllm:v2",
    ("deepseek-v4-flash-nvfp4", CLUSTER_MODE): "myllmbox/deepseek-v4-flash-cluster-vllm:v1",
}


# ── the regression: the live pair does not change ───────────────────────────────────────────────────

def test_the_reference_fleet_is_a_tp2_gb10_pair_and_a_classic_server_a_box():
    t = catalog_targets_of(REFERENCE)
    assert t[CLUSTER_MODE].key == "nvidia-sm121 × linux/arm64 × TP=2"
    assert t[CLASSIC].key == "nvidia-sm121 × linux/arm64 × TP=1"


def test_every_catalog_row_resolves_on_the_reference_fleet_to_exactly_the_image_it_pinned():
    catalog = ModelCatalog.load(deployment=catalog_deployment_of(REFERENCE))
    assert not catalog.error, catalog.error
    got = {(s.id, mode): p.image for s in catalog.specs for mode, p in s.launch.items()}
    assert got == PINNED_BEFORE
    for s in catalog.specs:
        for mode, p in s.launch.items():
            assert p.refused == "", (s.id, mode, p.refused)
            assert p.resolution.target.accelerator.key == "nvidia-sm121", (s.id, mode)


def test_the_default_catalog_resolves_against_the_fleet_lb_fleet_names():
    """conftest points LB_FLEET at the reference fleet, as compose points it at lb/fleet.yml."""
    catalog = ModelCatalog.load()
    assert catalog.get("glm-5.3-flash-nvfp4").profile(CLUSTER_MODE).image == "myllmbox/glm53-flash-cluster-vllm:v2"


# ── another accelerator: refused, naming the missing combination ────────────────────────────────────

ONE_5090_PAIR = """
boxes:
  a: {hostname: a-host, label: a-gpu, platform: linux/amd64, ssh: op@192.0.2.10, cards: [{index: 0, type: rtx5090, serves: true}]}
  b: {hostname: b-host, label: b-gpu, platform: linux/amd64, ssh: op@192.0.2.11, cards: [{index: 0, type: rtx5090, serves: true}]}
clusters:
  pair:
    control: ssh
    iface: eth0
    link: roce
    url: http://192.0.2.10:8100
    ram_usable_fraction: 0.9
    max_num_batched_tokens: 8192
    ranks: [{box: a, interconnect: 198.51.100.1}, {box: b, interconnect: 198.51.100.2}]
backends: {}
"""


def test_a_pair_of_other_cards_resolves_no_image_and_says_which_combination_is_missing(tmp_path):
    f = tmp_path / "fleet.yml"
    f.write_text(ONE_5090_PAIR)
    catalog = ModelCatalog.from_dict(document(catalog_deployment_of(REFERENCE)), targets=catalog_targets_of(f))
    p = catalog.get("qwen3.8-27b-dflash2").profile(CLUSTER_MODE)
    assert p.image is None
    assert "fleet" in p.refused and "nvidia-sm120" in p.refused and "linux/amd64" in p.refused, p.refused
    with pytest.raises(ValueError) as e:
        catalog.choice(CLUSTER_MODE, "qwen3.8-27b-dflash2")
    assert "nvidia-sm120" in str(e.value)


def test_a_quant_format_the_accelerator_has_no_kernel_for_is_refused_before_any_image(tmp_path):
    f = tmp_path / "fleet.yml"
    f.write_text(ONE_5090_PAIR)
    catalog = ModelCatalog.from_dict(document(catalog_deployment_of(REFERENCE)), targets=catalog_targets_of(f))
    p = catalog.get("glm-5.3-flash-nvfp4").profile(CLUSTER_MODE)
    assert "compressed-tensors" in p.refused and "nvidia-sm120" in p.refused, p.refused


def test_a_fleet_with_no_ssh_cluster_refuses_every_cluster_profile_naming_why(tmp_path):
    f = tmp_path / "fleet.yml"
    f.write_text("boxes:\n  solo: {hostname: s, platform: linux/amd64, cards: [{index: 0, type: rtx5090, "
                 "serves: true}]}\nclusters: {}\nbackends: {}\n")
    catalog = ModelCatalog.from_dict(document(catalog_deployment_of(REFERENCE)), targets=catalog_targets_of(f))
    assert not catalog.error
    for s in catalog.for_mode(CLUSTER_MODE):
        assert "fleet.yml names no cluster" in s.profile(CLUSTER_MODE).refused, s.id


def test_no_fleet_file_refuses_every_profile_naming_lb_fleet():
    from vllm_lb.catalog import deployment_targets
    catalog = ModelCatalog.from_dict(document(catalog_deployment_of(REFERENCE)), targets=deployment_targets(env={}))
    assert not catalog.error, catalog.error
    for s in catalog.specs:
        for p in s.launch.values():
            assert "LB_FLEET" in p.refused


# ── the matrix ──────────────────────────────────────────────────────────────────────────────────────

def test_each_engine_with_a_cluster_profile_has_a_matrix():
    assert {"vllm", "sglang", "tensorfold"} <= set(images())


def test_every_entry_is_one_line_in_the_key_order_the_deploy_scripts_grep():
    """deploy/images/build.sh reads a tag with grep (the host has no YAML parser)."""
    line = re.compile(r'  - \{build: (\S+), accelerator: (\S+), platform: (\S+), image: "([^"]+)", source: "[^"]+"\}')
    for f in IMAGES_DIR.glob("*.yml"):
        entries = [ln for ln in f.read_text().splitlines() if ln.startswith("  - ")]
        assert entries, f.name
        for ln in entries:
            assert line.fullmatch(ln), (f.name, ln)
    assert ENTRY_KEYS == ("build", "accelerator", "platform", "image", "source")


def test_a_missing_combination_names_it_and_what_the_build_has():
    with pytest.raises(ImageMissing) as e:
        image_for("vllm", "glm53", "nvidia-sm120", "linux/amd64")
    said = str(e.value)
    assert "glm53" in said and "nvidia-sm120 × linux/amd64" in said and "nvidia-sm121 × linux/arm64" in said


def test_an_unknown_build_is_refused_naming_the_builds():
    with pytest.raises(ImageMissing) as e:
        image_for("sglang", "latest", "nvidia-sm121", "linux/arm64")
    assert "fleet.2" in str(e.value) and "fleet.4" in str(e.value)


GOOD = (IMAGES_DIR / "sglang.yml").read_text()


@pytest.mark.parametrize("broken, says", [
    (GOOD.replace("accelerator: nvidia-sm121", "accelerator: nvidia-sm999", 1), "nvidia-sm999"),
    (GOOD.replace("platform: linux/arm64", "platform: linux/amd64", 1), "never comes in"),
    (GOOD.replace("build: fleet.2,", "build: fleet.9,", 1), "fleet.9"),
    (GOOD.replace("build: fleet.4,", "build: fleet.2,", 1), "twice"),
    (GOOD + "extra: 1\n", "extra"),
])
def test_a_broken_matrix_is_refused_naming_file_and_value(tmp_path, broken, says):
    assert broken != GOOD
    (tmp_path / "sglang.yml").write_text(broken)
    with pytest.raises(ValueError) as e:
        load_images(tmp_path)
    assert "sglang.yml" in str(e.value) and says in str(e.value)


# ── the resolver's order ────────────────────────────────────────────────────────────────────────────

GB10_PAIR = Target(accelerator("nvidia-sm121"), "linux/arm64", 2, source="test")


def test_the_engine_check_comes_before_the_image():
    r = resolve(engine="sglang", build="fleet.4", target=GB10_PAIR, fmt="hf",
                architectures=("DeepseekV4ForCausalLM",), quant_format="modelopt_mixed")
    assert r.image is None and "DeepseekV4ForCausalLM" in r.refused


def test_a_kv_dtype_the_image_does_not_run_on_the_arch_is_refused():
    """vLLM's fleet image takes nvfp4 KV only on sm_10x (kv_cache_dtypes.json): not on a GB10's sm_121."""
    r = resolve(engine="vllm", build="fleet", target=GB10_PAIR, fmt="hf",
                architectures=("Qwen3_5ForConditionalGeneration",), quant_format="modelopt_mixed",
                kv_cache_dtype="nvfp4")
    assert r.image is None and "nvfp4" in r.refused and "sm_121" in r.refused, r.refused


def test_a_resolution_names_its_image_and_the_entry_source():
    r = resolve(engine="vllm", build="glm53", target=GB10_PAIR, fmt="hf",
                architectures=("Glm5NextForConditionalGeneration",), quant_format="compressed-tensors")
    assert (r.image, r.refused) == ("myllmbox/glm53-flash-cluster-vllm:v2", "")
    v = r.view()
    assert v["target"]["accelerator"] == "nvidia-sm121" and v["image_source"]
