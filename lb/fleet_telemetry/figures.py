"""The figures and the wire's field names — ONE definition, read by the sender and by the gateway's recorder.

WHY HERE. The sender writes a figure's reason under its key (`deep.why.gpu_temp_c`) and the gateway's recorder
(vllm_lb/recorder.py) draws the figure under the same key, label and unit. Two copies of that table drift the day a
figure is added; so the recorder imports this module (the LB image carries the package; it is stdlib only).
"""
from __future__ import annotations

#: (key, label, unit, fmt) — the recorder's figures, in the panel's order. `fmt` is how the panel writes the number
#: (recorder.js `recFmt`, the one formatter): "int" whole (°C, rpm, MHz); "rate" with the precision its size needs
#: (≥10 whole, 1–10 one decimal, <1 two — a 0.14 Mb/s link is not "0"); "pct" one decimal; "flags" a decoded mask
#: (the throttle's reasons, not its 0-100 index). A label is at most 10 characters: it must fit a phone tile whole.
FIGURES_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("gpu_temp_c", "GPU TEMP", "°C", "int"),
    ("board_temp_c", "BOARD MAX", "°C", "int"),
    ("cpu_temp_c", "CPU TEMP", "°C", "int"),
    ("fan_rpm", "FAN SPEED", "rpm", "int"),
    ("gpu_power_w", "GPU POWER", "W", "rate"),
    ("throttle", "THROTTLE", "", "flags"),
    ("gpu_mhz", "GPU CLOCK", "MHz", "int"),
    ("cpu_mhz", "CPU CLOCK", "MHz", "int"),
    ("cpu_pct", "CPU UTIL", "%", "pct"),
    ("net_in_mbps", "NET IN", "Mb/s", "rate"),
    ("net_out_mbps", "NET OUT", "Mb/s", "rate"),
)
#: The instantaneous GPU draw is recorded (the power peak) but is not a tile of its own.
GPU_POWER_INST_W = "gpu_power_inst_w"
KEYS: tuple[str, ...] = tuple(k for k, *_ in FIGURES_TABLE) + (GPU_POWER_INST_W,)

#: How many containers a datagram carries, largest RAM first; `containers_total` counts them all.
SENT_CONTAINERS = 16

# ── the wire: top-level fields the gateway reads ─────────────────────────────────────────────────────────────────
HOST, SAMPLED_AT = "host", "sampled_at"
CPU_PERCENT, CPU_PER_CORE = "cpu_percent", "cpu_per_core"
RAM_PERCENT, RAM_USED_MB, RAM_TOTAL_MB, SWAP_USED_MB = "ram_percent", "ram_used_mb", "ram_total_mb", "swap_used_mb"
GPUS = "gpus"                                   # [{id, use_pct, vram_used_mb, vram_total_mb}] — the tiles' card rows
DEEP = "deep"
CONTAINERS, CONTAINERS_TOTAL, CONTAINERS_WHY = "containers", "containers_total", "containers_why"

# ── the `deep` block (the recorder's) ────────────────────────────────────────────────────────────────────────────
DEEP_GPU = "gpu"                                # one entry per card:
CARD_TEMP_C, CARD_POWER_W, CARD_POWER_INST_W = "temp_c", "power_w", "power_inst_w"
CARD_SM_MHZ, CARD_THROTTLE_HEX, CARD_THROTTLE_INDEX = "sm_mhz", "throttle_hex", "throttle_index"
CARD_POWER_LABEL, CARD_FAN_PCT = "power_label", "fan_pct"
#: A driver's power-sensor label → what the reading covers, as the panel names it ("GPU POWER (package)"): amdgpu's
#: `PPT` on an APU is the whole package, CPU and GPU — not the GPU's own draw. A label not here names nothing more.
POWER_SCOPE = {"PPT": "package"}
DEEP_THROTTLE_INDEX = "throttle_index"          # the worst card's
DEEP_ZONES, ZONE_TYPE, ZONE_MILLI_C = "tz", "type", "milli_c"
DEEP_CPU, CPU_TEMP_C, CPU_FREQ_MHZ = "cpu", "temp_c", "freq_mhz"
DEEP_FAN_RPM_MAX = "fan_rpm_max"
NET_RX_BYTES, NET_TX_BYTES, NET_SPEED_MBPS = "net_rx_bytes", "net_tx_bytes", "net_speed_mbps"
DEEP_WHY = "why"                                # {figure key: reason} for what this box cannot read

# ── containers ───────────────────────────────────────────────────────────────────────────────────────────────────
C_NAME, C_IMAGE, C_RAM_MB = "name", "image", "ram_mb"
TOTAL_COUNT, TOTAL_RAM_MB = "count", "ram_mb"
