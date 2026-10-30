#!/bin/sh
# Undo host-setup.sh on this box:  sudo sh host-uninstall.sh
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
rm -f /etc/sudoers.d/fleet-power
visudo -c >/dev/null
if id fleet-power >/dev/null 2>&1; then
  pkill -u fleet-power 2>/dev/null || true
  userdel --remove fleet-power
fi
echo "fleet-power removed."
