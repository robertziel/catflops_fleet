#!/bin/sh
# The fake box applies the REAL host-setup.sh from the public keys the harness mounts at /fp-keys
# (its from-box-b names the test network), then runs sshd in the foreground.
set -eu
FLEET_POWER_KEYS=/fp-keys sh /host-setup.sh box-b --gpu
exec /usr/sbin/sshd -D -e
