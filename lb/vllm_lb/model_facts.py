"""WHAT A CHECKPOINT IS, READ FROM ITS OWN FILES — the model axis of the engine redesign (docs/design/engines.md).

    python -m vllm_lb.model_facts <checkpoint folder>               # the facts, as library/models.json states them
    python -m vllm_lb.model_facts <checkpoint folder> --check <id>  # …compared with the library's row for <id>

A model's facts are never TYPED from memory or a model card: they are read at STAGE TIME from the folder a box holds,
and each carries its source — the file, the key, the date it was read. What it reads:

  format          `hf` (a config.json naming `architectures`), `ctranslate2` (model.bin beside a config.json that names
                  none — faster-whisper's conversion), `nemo` (a .nemo archive)
  architectures   config.json `architectures` ([] for a format that has none: the engine is checked by format)
  quant_method    config.json `quantization_config.quant_method` (None: unquantised, or MLX's own)
  quant_format    what the engine and the accelerator are checked against: the quant method, `mlx` for MLX's own
                  `quantization` block, `none` for unquantised weights
  shape           the layer shapes the memory ESTIMATE is computed from (memory_estimate.estimated_footprint):
                  num_hidden_layers, full_attention_layers (a hybrid's `layer_types` counted; every layer otherwise),
                  num_key_value_heads, head_dim (hidden_size / num_attention_heads where not stated),
                  max_position_embeddings, kv_lora_rank (MLA attention; None where the model has none). A multimodal
                  config's `text_config` is the language model's and is read first.

`--check <id>` exits 1 naming every fact that differs from library/models.json's row (the shape is not compared:
the library states only what a row needs), 0 when they agree — how an INFERRED fact (two TensorFold rows' architectures,
Open items) is confirmed on a box that holds the checkpoint. Read-only: it writes nothing.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

USAGE = "usage: python -m vllm_lb.model_facts <checkpoint folder> [--check <model id>]"


def _fact(value, source: str) -> dict:
    return {"value": value, "source": source}


def facts_of(folder: str | Path, *, today: str | None = None) -> dict:
    """The facts of the checkpoint in `folder` — or ValueError naming the folder and what it lacks."""
    d = Path(folder)
    day = today or _dt.date.today().isoformat()
    where = f"read {day} from {d.name}"
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else None
    nemo = sorted(d.glob("*.nemo")) if d.is_dir() else []
    if cfg is None and nemo:
        return {"format": _fact("nemo", f"{where}: {nemo[0].name}, a NeMo archive"),
                "architectures": _fact([], f"{where}: a .nemo archive names a NeMo model class, not an architecture"),
                "quant_method": _fact(None, f"{where}: a .nemo archive carries no quantization_config"),
                "quant_format": _fact("none", f"{where}: unquantised (no quantization_config)"),
                "shape": {}}
    if not isinstance(cfg, dict):
        raise ValueError(f"{d}: no config.json (and no .nemo archive) — not a checkpoint folder this tool reads")
    arch = cfg.get("architectures")
    if arch is None:
        if not (d / "model.bin").is_file():
            raise ValueError(f"{d}/config.json names no `architectures` and there is no model.bin beside it: neither "
                             f"a Hugging Face checkpoint nor a CTranslate2 conversion")
        return {"format": _fact("ctranslate2", f"{where}: model.bin beside a config.json with no architectures"),
                "architectures": _fact([], f"{where}/config.json: no `architectures` (a CTranslate2 conversion)"),
                "quant_method": _fact(None, f"{where}/config.json: no quantization_config"),
                "quant_format": _fact("none", f"{where}/config.json: no quantization_config"),
                "shape": {}}
    src = f"{where}/config.json"
    q = cfg.get("quantization_config") or {}
    method = q.get("quant_method") if isinstance(q, dict) else None
    if method is not None:
        qf = _fact(method, f"{src} (quantization_config.quant_method)")
    elif isinstance(cfg.get("quantization"), dict):
        qf = _fact("mlx", f"{src} (MLX's own `quantization` block, no quantization_config)")
    else:
        qf = _fact("none", f"{src}: no quantization_config — unquantised")
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    tsrc = f"{src} ({'text_config.' if text is not cfg else ''}"

    def get(key):
        return text.get(key, cfg.get(key))

    layers = get("num_hidden_layers")
    kinds = get("layer_types")
    full = (sum(1 for k in kinds if k == "full_attention") if isinstance(kinds, list) else layers)
    head_dim = get("head_dim")
    if head_dim is None and get("hidden_size") and get("num_attention_heads"):
        head_dim = get("hidden_size") // get("num_attention_heads")
    shape = {"num_hidden_layers": _fact(layers, f"{tsrc}num_hidden_layers)"),
             "full_attention_layers": _fact(full, f"{tsrc}{'layer_types' if isinstance(kinds, list) else 'num_hidden_layers'})"),
             "num_key_value_heads": _fact(get("num_key_value_heads"), f"{tsrc}num_key_value_heads)"),
             "head_dim": _fact(head_dim, f"{tsrc}head_dim, else hidden_size / num_attention_heads)"),
             "max_position_embeddings": _fact(get("max_position_embeddings"), f"{tsrc}max_position_embeddings)"),
             "kv_lora_rank": _fact(get("kv_lora_rank"), f"{tsrc}kv_lora_rank; None = no MLA)")}
    return {"format": _fact("hf", f"{src}: names its architectures"),
            "architectures": _fact(list(arch), f"{src} (architectures)"),
            "quant_method": _fact(method, f"{src} (quantization_config.quant_method)"),
            "quant_format": qf, "shape": shape}


#: The facts `--check` compares with the library's row.
CHECKED = ("format", "architectures", "quant_format")


def check(facts: dict, model_id: str) -> list[str]:
    """Every difference between `facts` and library/models.json's row `model_id`, in words; [] when they agree."""
    from vllm_lb.catalog import library_models
    row = next((m for m in library_models() if m.get("id") == model_id), None)
    if row is None:
        return [f"library/models.json has no model {model_id!r}"]
    return [f"{k}: the folder says {facts[k]['value']!r}, library/models.json says {row[k]['value']!r} "
            f"({row[k]['source']})" for k in CHECKED if facts[k]["value"] != row[k]["value"]]


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 3) or (len(argv) == 3 and argv[1] != "--check"):
        print(USAGE, file=sys.stderr)
        return 64
    try:
        facts = facts_of(argv[0])
    except (OSError, ValueError) as exc:
        print(f"model_facts: {exc}", file=sys.stderr)
        return 78
    print(json.dumps(facts, indent=2))
    if len(argv) == 3:
        differs = check(facts, argv[2])
        for line in differs:
            print(f"model_facts: {argv[2]}: {line}", file=sys.stderr)
        return 1 if differs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
