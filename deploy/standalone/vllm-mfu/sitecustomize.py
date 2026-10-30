try:
    from vllm.v1.metrics import perf as _perf
    _perf._QUANT_WEIGHT_BYTE_SIZE.setdefault("modelopt_mixed", 0.5625)
    _perf._QUANT_WEIGHT_BYTE_SIZE.setdefault("deepseek_v4_fp8", 0.5625)
except Exception:
    pass
