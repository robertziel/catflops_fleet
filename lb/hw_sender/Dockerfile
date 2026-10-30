# vllm-hw-sender: THIS server's hardware telemetry — the fleet_telemetry package (lb/fleet_telemetry), run as
# `python -m fleet_telemetry`. Every 5 s it samples CPU, RAM, GPUs, temperatures, power, clocks, fans, network
# counters and containers, and broadcasts one JSON datagram (UDP HW_SENDER_PORT) the gateway draws its hardware tiles
# and per-box FLIGHT RECORDER from (vllm_lb/hw.py, vllm_lb/recorder.py).
#
# Built from lb/ (docker-compose.yml: context `..`) so the image carries the package the gateway also imports its
# figure table from — one definition. Dockerfile.dockerignore next to this file sends only lb/fleet_telemetry.
# STDLIB ONLY: no pip install, no compiler, so it builds unchanged on x86_64 and aarch64.
FROM python:3.12-slim
COPY fleet_telemetry/ /opt/telemetry/fleet_telemetry/
WORKDIR /opt/telemetry
ENV PYTHONPATH=/opt/telemetry PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Fail the BUILD if the package does not import (a syntax error would otherwise show up only as a restarting box).
RUN python -c "import fleet_telemetry.__main__, fleet_telemetry.readers"
ENTRYPOINT ["python", "-m", "fleet_telemetry"]
