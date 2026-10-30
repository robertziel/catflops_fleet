# vllm-power — the fleet's power buttons (poweroff over a forced-command ssh key, Wake-on-LAN).
# Stdlib Python at runtime; openssh-client is the one system package. pytest is here so
# `docker run --rm vllm-power:latest pytest` is the hermetic full-suite run (repo convention).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pytest
RUN useradd --uid 1000 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin power
COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER 1000:1000
CMD ["python", "-m", "fleet_power"]
