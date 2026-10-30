"""THE KV CACHE'S ELEMENT TYPE of a one-host node (control local) — vLLM's --kv-cache-dtype, nvfp4 among them (operator,
2026-10-22: "implement KV nvfp4 to our models … add to each engine").

`kv_cache_dtype` is an OPTIONAL launch key an intent moves (and a launch block may name as its boot default): absent or
null is vLLM's own ("auto", the model's dtype) — node-env renders the line EMPTY and the node's entrypoint passes no
flag, so a launch without it is today's launch. A value is checked against what the node's IMAGE accepts — its own
vllm/config/cache.py CacheDType, read out of the image into lb/vllm_lb/kv_cache_dtypes.json, keyed by the
image ref the launch runs. An image the file does not state, or states no list for, takes no value: refused, naming it.
"""
from __future__ import annotations


import pytest

from tests.test_fleet_local import DIGEST, LOCAL, _fleet
from vllm_lb import fleet as fleet_mod
from vllm_lb import kv_dtype
from vllm_lb.fleet import KV_DTYPE, MOVABLE, OPTIONAL_NODE_ENV, node_env, validate_launch

LIB = kv_dtype.library()
#: The two local nodes' images, as their fleets pin them (the gui node's rocm build; the 5090 node's vllm-openai).
ROCM = "rocm/vllm@sha256:394194d36edcf9b36bcb563e143b21b80e64e7d04f33a447b448c0c0c00c04a8"
CUDA = "vllm/vllm-openai@sha256:514cd42702d29a7b8efe20b0a916523ac473ecebdec8d53b1a0d0f5439a48634"


def _node(image: str):
    return _fleet(LOCAL.replace(DIGEST, image)).clusters["node"]


def test_both_node_images_are_in_the_one_library_by_their_pinned_refs():
    for ref in (ROCM, CUDA):
        assert fleet_mod._DIGEST.fullmatch(ref) and ref in LIB.images, ref


def test_what_the_images_say_nvfp4_on_the_cuda_build_and_none_on_the_rocm_build():
    """Read out of each image (cache.py CacheDType), 2026-10-22 — never from memory."""
    assert {"nvfp4", "fp8_e4m3"} <= set(LIB.images[CUDA].accepted)
    assert not [v for v in LIB.images[ROCM].accepted if "fp4" in v] and "nvfp4" in LIB.images[ROCM].refused


def test_it_is_a_key_an_intent_moves_and_an_optional_node_line():
    assert KV_DTYPE == "kv_cache_dtype" and KV_DTYPE in MOVABLE
    assert "EDGE_NODE_KV_CACHE_DTYPE" in OPTIONAL_NODE_ENV


def test_absent_or_null_is_vllms_own_and_the_line_is_empty():
    node = _node(CUDA)
    assert validate_launch(node, {}).kv_cache_dtype is None
    assert validate_launch(node, {KV_DTYPE: None}).kv_cache_dtype is None
    lines = node_env(_fleet(LOCAL.replace(DIGEST, CUDA)), "node", {"max_num_seqs": 2})
    assert "EDGE_NODE_KV_CACHE_DTYPE=" in lines


def test_absent_on_an_image_the_file_does_not_state_is_still_todays_launch():
    """No list is needed to launch WITHOUT a dtype: the placeholder image the tests pin is in no file."""
    assert DIGEST not in LIB.images
    assert validate_launch(_fleet(LOCAL).clusters["node"], {"max_num_seqs": 2}).kv_cache_dtype is None


def test_nvfp4_on_the_cuda_build_reaches_node_env():
    launch = validate_launch(_node(CUDA), {KV_DTYPE: "nvfp4"})
    assert launch.kv_cache_dtype == "nvfp4"
    assert "EDGE_NODE_KV_CACHE_DTYPE=nvfp4" in node_env(_fleet(LOCAL.replace(DIGEST, CUDA)), "node", {KV_DTYPE: "nvfp4"})


def test_nvfp4_on_the_rocm_build_is_refused_naming_what_it_accepts():
    with pytest.raises(ValueError) as e:
        validate_launch(_node(ROCM), {KV_DTYPE: "nvfp4"})
    said = str(e.value)
    assert KV_DTYPE in said and "'nvfp4'" in said and ROCM in said
    assert LIB.images[ROCM].refused["nvfp4"] in said, "the image's own reason, named"
    assert validate_launch(_node(ROCM), {KV_DTYPE: "fp8_e4m3"}).kv_cache_dtype == "fp8_e4m3"


@pytest.mark.parametrize("value", ["", "NVFP4", 4, True, ["nvfp4"], "nvfp4 --x"])
def test_anything_but_one_of_the_images_values_is_refused(value):
    with pytest.raises(ValueError, match=KV_DTYPE):
        validate_launch(_node(CUDA), {KV_DTYPE: value})


def test_a_value_for_an_image_the_file_does_not_state_is_refused_never_guessed():
    with pytest.raises(ValueError) as e:
        validate_launch(_fleet(LOCAL).clusters["node"], {KV_DTYPE: "fp8_e4m3"})
    assert DIGEST in str(e.value) and kv_dtype.LIBRARY_PATH.name in str(e.value)


def test_a_launch_block_may_name_it_as_the_boot_default_checked_the_same_way():
    block = LOCAL.replace(DIGEST, CUDA).replace("      tool_call_parser: qwen3_coder\n",
                                                "      tool_call_parser: qwen3_coder\n      kv_cache_dtype: nvfp4\n")
    node = _fleet(block).clusters["node"]
    assert node.launch.kv_cache_dtype == "nvfp4"
    assert validate_launch(node, {}).kv_cache_dtype == "nvfp4", "the boot default carries into an intent"
    assert validate_launch(node, {KV_DTYPE: None}).kv_cache_dtype is None, "null gives it back to vLLM"
    with pytest.raises(ValueError, match="nvfp4"):
        _fleet(block.replace(CUDA, ROCM))
