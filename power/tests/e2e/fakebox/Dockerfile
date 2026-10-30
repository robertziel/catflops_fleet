# Build from power/:  docker build -t fleet-power-fakebox:test -f tests/e2e/fakebox/Dockerfile .
# A stand-in for box-a / box-b / box-c in the dry-run end-to-end test: Ubuntu 24.04 (the boxes'
# release), OpenSSH, sudo, and the REAL install/host-setup.sh, applied at START (start.sh) from the
# PUBLIC keys the harness mounts at /fp-keys. No key is baked into the image.
# /usr/bin/systemctl and /usr/bin/nvidia-smi are stubs that only record their arguments, so even the
# real power and clock keys power nothing off and set no clock here.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server sudo \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd
RUN printf '#!/bin/sh\necho "FAKE-SYSTEMCTL(uid=$(id -u)) $*" | tee -a /tmp/systemctl.log\n' > /usr/bin/systemctl \
    && chmod 755 /usr/bin/systemctl
# Same for nvidia-smi: the clock keys' forced commands run this stub, so no clock is ever set here.
RUN printf '#!/bin/sh\necho "FAKE-NVIDIA-SMI(uid=$(id -u)) $*" | tee -a /tmp/nvidia-smi.log\n' > /usr/bin/nvidia-smi \
    && chmod 755 /usr/bin/nvidia-smi
COPY install/host-setup.sh /host-setup.sh
COPY tests/e2e/fakebox/start.sh /start.sh
CMD ["sh", "/start.sh"]
