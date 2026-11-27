# vllm-hw-sender: THIS server's hardware telemetry — the fleet_telemetry package (lb/fleet_telemetry), run as
# `python -m fleet_telemetry`. Every second it samples CPU, RAM, GPUs, temperatures, power, clocks, fans, network
# counters and containers, and broadcasts one JSON datagram (UDP HW_SENDER_PORT) the gateway draws its hardware tiles
# and per-box FLIGHT RECORDER from (vllm_lb/hw.py, vllm_lb/recorder.py).
#
# Built from lb/ (docker-compose.yml: context `..`) so the image carries the package the gateway also imports its
# figure table from — one definition. Dockerfile.dockerignore next to this file sends only lb/fleet_telemetry.
# STDLIB ONLY in Python: no pip install, no compiler, so it builds unchanged on x86_64 and aarch64.
FROM python:3.12-slim
# TWO DEBIAN TOOLS for the RoCE ports (fleet_telemetry/readers/rdma.py): `ethtool -S <netdev>` for mlx5e's
# PHY counters (rx_discards_phy, …), which no sysfs file states, and iproute2's `dcb` for the PFC priorities, apptrust
# order and DSCP map — the kernel DCB state mlnx_qos reads, without MLNX_OFED. Both are read-only here (stats and
# netlink GETs). A box without RDMA never runs them.
RUN apt-get update && apt-get install -y --no-install-recommends ethtool iproute2 && rm -rf /var/lib/apt/lists/*
COPY fleet_telemetry/ /opt/telemetry/fleet_telemetry/
WORKDIR /opt/telemetry
ENV PYTHONPATH=/opt/telemetry PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Fail the BUILD if the package does not import (a syntax error would otherwise show up only as a restarting box).
RUN python -c "import fleet_telemetry.__main__, fleet_telemetry.readers" && ethtool --version && dcb -V
ENTRYPOINT ["python", "-m", "fleet_telemetry"]
