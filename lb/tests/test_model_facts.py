"""THE MODEL AXIS IS READ, NEVER TYPED: `python -m vllm_lb.model_facts <checkpoint folder>` reads a staged checkpoint's
config.json into the facts library/models.json states (format, architectures, quant format, the layer shapes the
estimate needs), each {value, source} naming the file and the key it came from; `--check <id>` compares them with the
library's row and fails naming every difference (docs/design/engines.md)."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vllm_lb.model_facts import facts_of


def _folder(tmp_path, config: dict | None, *files: str):
    d = tmp_path / "ckpt"
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    for f in files:
        (d / f).write_bytes(b"")
    return d


HYBRID = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
          "quantization_config": {"quant_method": "modelopt_mixed"},
          "text_config": {"num_hidden_layers": 48, "num_key_value_heads": 2, "head_dim": 256,
                          "max_position_embeddings": 262144,
                          "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12}}


def test_a_hugging_face_folder_reads_its_architectures_quant_and_shapes(tmp_path):
    f = facts_of(_folder(tmp_path, HYBRID, "model.safetensors"))
    assert f["format"]["value"] == "hf"
    assert f["architectures"]["value"] == ["Qwen4ExpForConditionalGeneration"]
    assert (f["quant_format"]["value"], f["quant_method"]["value"]) == ("modelopt_mixed", "modelopt_mixed")
    shape = {k: v["value"] for k, v in f["shape"].items()}
    assert shape == {"num_hidden_layers": 48, "full_attention_layers": 12, "num_key_value_heads": 2, "head_dim": 256,
                     "max_position_embeddings": 262144, "kv_lora_rank": None}
    for fact in (f["architectures"], f["quant_format"], f["shape"]["head_dim"]):
        assert "config.json" in fact["source"] and "read 20" in fact["source"], fact


def test_mlx_and_unquantised_and_ctranslate2_and_nemo(tmp_path):
    mlx = facts_of(_folder(tmp_path, {"architectures": ["X"], "quantization": {"bits": 4, "group_size": 32}}))
    assert (mlx["quant_format"]["value"], mlx["quant_method"]["value"]) == ("mlx", None)
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps({"architectures": ["WhisperForConditionalGeneration"]}))
    plain = facts_of(tmp_path / "ckpt")
    assert (plain["quant_format"]["value"], plain["quant_method"]["value"]) == ("none", None)
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps({"alignment_heads": [[2, 4]]}))
    (tmp_path / "ckpt" / "model.bin").write_bytes(b"")
    ct2 = facts_of(tmp_path / "ckpt")
    assert (ct2["format"]["value"], ct2["architectures"]["value"]) == ("ctranslate2", [])


def test_a_nemo_archive_is_its_own_format(tmp_path):
    f = facts_of(_folder(tmp_path, None, "parakeet.nemo"))
    assert (f["format"]["value"], f["architectures"]["value"], f["quant_format"]["value"]) == ("nemo", [], "none")


def test_a_folder_with_no_checkpoint_is_refused_naming_it(tmp_path):
    with pytest.raises(ValueError) as e:
        facts_of(_folder(tmp_path, None))
    assert "ckpt" in str(e.value) and "config.json" in str(e.value)


def test_mla_attention_states_its_latent_rank(tmp_path):
    f = facts_of(_folder(tmp_path, {"architectures": ["DeepseekV4ForCausalLM"], "num_hidden_layers": 43,
                                    "kv_lora_rank": 512, "num_key_value_heads": 1, "head_dim": 512}))
    assert f["shape"]["kv_lora_rank"]["value"] == 512


def test_the_cli_prints_json_and_check_names_each_difference(tmp_path):
    d = _folder(tmp_path, {"architectures": ["Glm5NextForConditionalGeneration"],
                           "quantization_config": {"quant_method": "compressed-tensors"}})
    ok = subprocess.run([sys.executable, "-m", "vllm_lb.model_facts", str(d)], capture_output=True, text=True,
                        timeout=60)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["architectures"]["value"] == ["Glm5NextForConditionalGeneration"]
    same = subprocess.run([sys.executable, "-m", "vllm_lb.model_facts", str(d), "--check", "glm-5.3-flash-nvfp4"],
                          capture_output=True, text=True, timeout=60)
    assert same.returncode == 0, same.stdout + same.stderr
    differs = subprocess.run([sys.executable, "-m", "vllm_lb.model_facts", str(d), "--check", "mai-ui-2b"],
                             capture_output=True, text=True, timeout=60)
    assert differs.returncode == 1 and "architectures" in differs.stderr and "quant_format" in differs.stderr
