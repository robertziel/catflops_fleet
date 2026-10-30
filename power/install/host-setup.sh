#!/bin/sh
# fleet-power host setup — run ONCE on each box, by its operator, with sudo:
#
#     sudo FLEET_POWER_KEYS=~/.config/fleet-power/keys sh host-setup.sh <box> --gpu     # on a GPU box
#     sudo FLEET_POWER_KEYS=~/.config/fleet-power/keys sh host-setup.sh <box> --local   # on the box that runs vllm-power
#
# <box> is the name the service's inventory (POWER_BOXES) gives the machine: its from-<box> file is read.
# --gpu installs the four clock keys too; --local pins the keys to loopback (the service dials itself) and
# reads no from- file. Neither is assumed from the name: a box list in this script would be one deployment's.
#
# THIS SCRIPT CARRIES NO KEY AND NO ADDRESS. Everything that makes a box trust a key lives in the
# directory FLEET_POWER_KEYS names. The person running it supplies that directory; it is never shipped.
# It is the directory `ssh-keygen -f` wrote the power keys into (their *.pub files are read here):
#   fleet-power.pub, fleet-power-dryrun.pub                    the poweroff key and its dry-run check (every box)
#   fleet-clock-limit.pub, fleet-clock-limit-2300.pub,         the four GPU-clock keys (--gpu)
#   fleet-clock-release.pub, fleet-clock-dryrun.pub
#   from-<box>                                                 the address the keys are pinned to (from=): the
#                                                              HOST-side end of the direct cable to <box>, one
#                                                              IPv4 address or CIDR. Not read with --local: that
#                                                              box's service runs on itself (loopback).
# A missing or non-key file is refused BEFORE anything on the system changes.
#
# What it does, and nothing else:
#   1. a system user `fleet-power`: no password (field "*"), home /var/lib/fleet-power,
#      shell /bin/sh (sshd runs a forced command through the login shell);
#   2. its ~/.ssh/authorized_keys with exactly TWO keys (four more on a GPU box), each `restrict`
#      (no pty, no forwarding of any kind, no ~/.ssh/rc) and `from=` the source above:
#        - the power key   → command="sudo -n /usr/bin/systemctl poweroff"
#        - the dry-run key → command="sudo -n -l /usr/bin/systemctl poweroff"  (only ASKS sudo
#          whether it would be allowed; powers nothing off)
#      Whatever command a client sends is ignored: sshd runs the forced one.
#   3. on a box with a GPU (--gpu), FOUR more keys for the panel's clock control — one key per
#      cap, so the MHz live in the forced command and no number a client sends can reach root:
#        - the 2200 key    → command="sudo -n /usr/bin/nvidia-smi -lgc 300,2200"
#        - the 2300 key    → command="sudo -n /usr/bin/nvidia-smi -lgc 300,2300"
#        - the release key → command="sudo -n /usr/bin/nvidia-smi -rgc"
#        - the dry-run key → command="sudo -n -l /usr/bin/nvidia-smi -lgc 300,2200"  (only ASKS)
#   4. /etc/sudoers.d/fleet-power: `fleet-power` may run exactly `/usr/bin/systemctl poweroff`
#      as root, with no password — plus, on a GPU box, exactly `/usr/bin/nvidia-smi -lgc 300,2200`,
#      `/usr/bin/nvidia-smi -lgc 300,2300` and `/usr/bin/nvidia-smi -rgc`. No wildcard, no other
#      argument. Checked with `visudo -cf` before it is installed.
#
# FLEET_POWER_PRINT=1 renders steps 2 and 4 to stdout and changes nothing (no root needed): read it
# before you install. Undo: host-uninstall.sh.
set -eu

usage() { echo "usage: sudo FLEET_POWER_KEYS=<dir> sh $0 <box> [--gpu] [--local]" >&2; exit 2; }
BOX="${1:-}"
case "$BOX" in ""|-*|*[!A-Za-z0-9_.-]*) usage ;; esac
shift
GPU=no; LOCAL=no
for flag in "$@"; do
  case "$flag" in
    --gpu) GPU=yes ;;
    --local) LOCAL=yes ;;
    *) usage ;;
  esac
done

KEYS="${FLEET_POWER_KEYS:-}"
[ -n "$KEYS" ] || { echo "set FLEET_POWER_KEYS to the directory holding this box's public keys (fleet-power.pub, fleet-power-dryrun.pub, ...)" >&2; exit 2; }
[ -d "$KEYS" ] || { echo "FLEET_POWER_KEYS=$KEYS is not a directory" >&2; exit 2; }

# read_pub NAME: the one-line public key in $KEYS/NAME, or refuse naming the file. (A subshell: `return`.)
read_pub() {
  f="$KEYS/$1"
  [ -f "$f" ] || { echo "missing $f (FLEET_POWER_KEYS): a public key file is required" >&2; return 2; }
  line=$(sed -n '1p' "$f" | tr -d '\r')
  case "$line" in
    ssh-*\ *|ecdsa-*\ *) printf '%s\n' "$line" ;;
    *) echo "$f does not hold a public key: one line starting with ssh-…" >&2; return 2 ;;
  esac
}

# read_from BOX: the source address for BOX, from $KEYS/from-BOX. Digits and dots only, so nothing that is
# not an address can reach the from="…" option.
read_from() {
  f="$KEYS/from-$1"
  [ -f "$f" ] || { echo "missing $f (FLEET_POWER_KEYS): the source address for $1" >&2; return 2; }
  addr=$(sed -n '1p' "$f" | tr -d ' \t\r')
  case "$addr" in
    ""|*[!0-9./]*) echo "$f must hold one IPv4 address or CIDR (digits, dots and a slash)" >&2; return 2 ;;
  esac
  printf '%s\n' "$addr"
}

if [ "$LOCAL" = yes ]; then
  FROM="127.0.0.1,::1"                                  # the service runs on this box itself
else
  FROM=$(read_from "$BOX") || exit 2
fi

POWER_KEY=$(read_pub fleet-power.pub) || exit 2
DRY_KEY=$(read_pub fleet-power-dryrun.pub) || exit 2
if [ "$GPU" = yes ]; then
  CLOCK_LIMIT_KEY=$(read_pub fleet-clock-limit.pub) || exit 2
  CLOCK_LIMIT_2300_KEY=$(read_pub fleet-clock-limit-2300.pub) || exit 2
  CLOCK_RELEASE_KEY=$(read_pub fleet-clock-release.pub) || exit 2
  CLOCK_DRY_KEY=$(read_pub fleet-clock-dryrun.pub) || exit 2
fi

# == CLOCK_MIN_MHZ / CLOCK_CAPS_MHZ in fleet_power/app.py, one CLOCK_MAXn per cap the panel offers
CLOCK_MIN=300; CLOCK_MAX1=2200; CLOCK_MAX2=2300
HOME_DIR=/var/lib/fleet-power

AUTH="$(mktemp)"; SUDO="$(mktemp)"
trap 'rm -f "$AUTH" "$SUDO"' EXIT

# 2. the keys, rendered (not yet installed)
cat > "$AUTH" <<EOF
restrict,from="$FROM",command="sudo -n /usr/bin/systemctl poweroff" $POWER_KEY
restrict,from="$FROM",command="sudo -n -l /usr/bin/systemctl poweroff" $DRY_KEY
EOF
if [ "$GPU" = yes ]; then
  cat >> "$AUTH" <<EOF
restrict,from="$FROM",command="sudo -n /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$CLOCK_MAX1" $CLOCK_LIMIT_KEY
restrict,from="$FROM",command="sudo -n /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$CLOCK_MAX2" $CLOCK_LIMIT_2300_KEY
restrict,from="$FROM",command="sudo -n /usr/bin/nvidia-smi -rgc" $CLOCK_RELEASE_KEY
restrict,from="$FROM",command="sudo -n -l /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$CLOCK_MAX1" $CLOCK_DRY_KEY
EOF
fi

# 4. the privileged commands — exactly these argv, nothing else
if [ "$GPU" = yes ]; then
  # A literal comma inside an argument is escaped for sudoers, which uses commas as separators.
  CMDS="/usr/bin/systemctl poweroff"
  for m in "$CLOCK_MAX1" "$CLOCK_MAX2"; do      # one argv per cap the panel offers, no wildcard
    CMDS="$CMDS, /usr/bin/nvidia-smi -lgc $CLOCK_MIN\\,$m"
  done
  printf '%s\n' "fleet-power ALL=(root) NOPASSWD: $CMDS, /usr/bin/nvidia-smi -rgc" > "$SUDO"
else
  echo 'fleet-power ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff' > "$SUDO"
fi

if [ "${FLEET_POWER_PRINT:-}" = 1 ]; then
  echo "== $HOME_DIR/.ssh/authorized_keys (from=$FROM)"; cat "$AUTH"
  echo "== /etc/sudoers.d/fleet-power"; cat "$SUDO"
  exit 0
fi

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
visudo -cf "$SUDO" >/dev/null

# 1. the user
if ! id fleet-power >/dev/null 2>&1; then
  useradd --system --home-dir "$HOME_DIR" --create-home --shell /bin/sh \
          --comment "fleet power button (poweroff only)" fleet-power
fi
usermod -p '*' fleet-power          # no password at all; only the keys below log in

# 2. install the keys
install -d -m 700 -o fleet-power -g fleet-power "$HOME_DIR/.ssh"
install -m 600 -o fleet-power -g fleet-power "$AUTH" "$HOME_DIR/.ssh/authorized_keys"

# 4. install the sudoers grant
install -m 440 -o root -g root "$SUDO" /etc/sudoers.d/fleet-power
visudo -c >/dev/null

echo "fleet-power installed on $BOX (from=$FROM)."
echo "check: sudo -u fleet-power sudo -n -l /usr/bin/systemctl poweroff   # prints the command"
echo "check: sudo -u fleet-power sudo -n -l /usr/bin/systemctl reboot     # must be refused"
if [ "$GPU" = yes ]; then
  for m in "$CLOCK_MAX1" "$CLOCK_MAX2"; do
    echo "check: sudo -u fleet-power sudo -n -l /usr/bin/nvidia-smi -lgc $CLOCK_MIN,$m   # prints it"
  done
  echo "check: sudo -u fleet-power sudo -n -l /usr/bin/nvidia-smi -lgc 300,3000        # must be refused"
fi
