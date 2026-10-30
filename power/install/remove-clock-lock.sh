#!/bin/sh
# Take the boot-time GPU clock cap off this box.  Run as root on box-a / box-b:
#
#     sudo sh remove-clock-lock.sh
#
# The cap (300-2200 MHz) used to be `nvidia-clock-lock.service`, applied on every boot. It now
# lives behind a click on the gateway panel instead (see ../README.md, actions clock_limit /
# clock_release), so nothing sets a clock at boot any more.
#
# What it does NOT do: `systemctl stop`. That unit's ExecStop runs `nvidia-smi -rgc`, which would
# change the clocks in force under whatever the box is running right now. Removing the unit leaves
# the live clocks exactly as they are — the next boot simply has no cap.
set -eu
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
U=nvidia-clock-lock.service
clk() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true; }

echo "clocks before: $(clk)$(clk)$(clk)"
if [ -e "/etc/systemd/system/$U" ]; then
  systemctl disable "$U" 2>&1 || true          # drops the .wants symlink; does not stop the unit
  rm -f "/etc/systemd/system/$U"
  echo "removed /etc/systemd/system/$U"
else
  echo "/etc/systemd/system/$U was already absent"
fi
rm -f "/etc/systemd/system/multi-user.target.wants/$U"      # in case disable could not run
systemctl daemon-reload
echo "is-enabled: $(systemctl is-enabled "$U" 2>&1 || true)"
echo "unit file:  $( [ -e "/etc/systemd/system/$U" ] && echo PRESENT || echo gone )"
echo "clocks after:  $(clk)$(clk)$(clk)"
