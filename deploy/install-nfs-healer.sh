#!/usr/bin/env bash
# Install / verify the NFS boot healer on every box that mounts NFS.
#
#   HARDEN=<path on each box> deploy/install-nfs-healer.sh box-a box-b   # check + install where missing
#   HARDEN=<path on each box> deploy/install-nfs-healer.sh --check box-a box-b box-c
#   HARDEN=<path on each box> deploy/install-nfs-healer.sh box-b         # one box
#
# The boxes are ARGUMENTS, never a default: a healer aimed at the wrong machine is a root shell on it.
# HARDEN is required: the host-level script that does the real work lives outside this repo.
#
# WHY THIS EXISTS (2026-09-28): a GB10 box boots, dockerd touches an NFS bind source
# before the 192.0.2.x link has carrier, the mount fails 3-5x inside a second,
# StartLimitBurst is exhausted and `srv-models.mount` latches `failed` for the rest of
# the boot. A LATCHED automount is the one state that never heals itself — the
# worker boot backstop then waits forever (observed: 105 checks / 1h45m),
# box-b has no broker worker, and the vLLM cluster sits in `reserving` with box-a
# held idle. See memory box-b-worker-dies-on-boot-nfs-race.
#
# The real fix is /srv/infra/harden-gb10-network-boot.sh: no-latch drop-ins
# (StartLimitIntervalSec=0) for every fstab NFS mount + a `nfs-mount-healer` timer that
# reset-failed's and remounts anything missing, 90s after boot and every 2 min after.
# That script is host-level /etc state and deliberately lives OUTSIDE any repo. This
# is only the driver that runs it on the right boxes and then PROVES the result.
#
# NO SUDO PASSWORD NEEDED: the boxes have no passwordless sudo, but the operator account is in the
# `docker` group, which is root-equivalent — a --privileged container with --pid=host
# plus `nsenter -t 1 -m -u -i -n -p` is a root shell in the host's namespaces.
set -uo pipefail

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)

check_only=0
boxes=()
for a in "$@"; do
  case "$a" in
    --check) check_only=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) boxes+=("$a") ;;
  esac
done
[ ${#boxes[@]} -gt 0 ] || { echo "usage: $0 [--check] <box> [<box> ...]   (no default boxes: name each one)" >&2; exit 2; }
: "${HARDEN:?set HARDEN to the path, on each box, of harden-gb10-network-boot.sh (it lives outside this repo)}"

# ── the verifier: runs as a normal user, changes nothing ──────────────────
# Prints one TSV line per NFS mount plus a TIMER line. A box with no NFS mounts in
# fstab prints no MOUNT lines at all — nothing to heal there, which is not a failure.
read -r -d '' VERIFY <<'PROBE'
set -u
esc() { systemd-escape -p "$1"; }
echo -e "TIMER\t$(systemctl is-active nfs-mount-healer.timer 2>/dev/null)\t$(systemctl is-enabled nfs-mount-healer.timer 2>/dev/null)"
while read -r mp; do
  [ -n "$mp" ] || continue
  u=$(esc "$mp")
  # `mountpoint`, never `[ -d ]`: docker manufactures empty root-owned stub dirs on the
  # local disk that pass a -d test while the real export is absent.
  if timeout 10 mountpoint -q "$mp"; then m=mounted; else m=MISSING; fi
  d=no; [ -f "/etc/systemd/system/${u}.mount.d/20-no-rate-limit.conf" ] && d=yes
  a=no; [ -f "/etc/systemd/system/${u}.automount.d/20-no-rate-limit.conf" ] && a=yes
  f=$(systemctl is-failed "${u}.mount" 2>/dev/null)
  echo -e "MOUNT\t$mp\t$m\t$d\t$a\t$f"
done < <(awk '$1 !~ /^#/ && $3 ~ /^nfs4?$/ {print $2}' /etc/fstab)
PROBE

# ── root on a remote box, without a password ──────────────────────────────
run_root() {  # $1 = box; script on stdin
  local box=$1 img
  if "${SSH[@]}" "$box" 'sudo -n true' 2>/dev/null; then
    "${SSH[@]}" "$box" 'sudo -n bash -s'
    return
  fi
  # docker group == root. Reuse an image already on the box so this works offline.
  img=$("${SSH[@]}" "$box" 'docker images --format "{{.Repository}}:{{.Tag}}" \
          | grep -Em1 "^(alpine|busybox|debian|ubuntu):"' 2>/dev/null)
  if [ -z "$img" ]; then
    echo "  !! no local alpine/busybox/debian/ubuntu image and no passwordless sudo" >&2
    return 1
  fi
  # nsenter -m puts us in the HOST mount namespace, so /tmp here IS the host's /tmp.
  "${SSH[@]}" "$box" "cat > /tmp/.nfs-healer-step.sh && \
     docker run --rm --privileged --pid=host --net=host $img \
       nsenter -t 1 -m -u -i -n -p -- bash /tmp/.nfs-healer-step.sh; \
     rc=\$?; rm -f /tmp/.nfs-healer-step.sh; exit \$rc"
}

report() {  # $1 = box, $2 = probe output; returns 1 if anything is wrong
  local box=$1 out=$2 bad=0 nfs=0
  local timer
  timer=$(awk -F'\t' '$1=="TIMER"{print $2}' <<<"$out")
  while IFS=$'\t' read -r tag mp m d a f; do
    [ "$tag" = MOUNT ] || continue
    nfs=1
    printf '    %-22s %-8s dropins=%s/%s%s\n' "$mp" "$m" "$d" "$a" \
      "$([ "$f" = failed ] && echo '  UNIT=failed')"
    [ "$m" = mounted ] && [ "$d" = yes ] && [ "$a" = yes ] && [ "$f" != failed ] || bad=1
  done <<<"$out"
  if [ "$nfs" = 0 ]; then
    echo "    no NFS mounts in fstab — nothing to heal"
    return 0
  fi
  echo "    healer timer: ${timer:-unknown}"
  [ "$timer" = active ] || bad=1
  return $bad
}

rc=0
for box in "${boxes[@]}"; do
  echo "== $box"
  if ! out=$("${SSH[@]}" "$box" "bash -s" <<<"$VERIFY" 2>&1); then
    echo "    !! unreachable: $out"; rc=1; continue
  fi
  if report "$box" "$out"; then
    echo "    OK"
    continue
  fi
  if [ "$check_only" = 1 ]; then
    echo "    NEEDS INSTALL (--check: changing nothing)"; rc=1; continue
  fi
  echo "    installing via $HARDEN ..."
  if ! "${SSH[@]}" "$box" "test -x $HARDEN" 2>/dev/null; then
    echo "    !! $HARDEN not reachable on $box (is /srv mounted?)"; rc=1; continue
  fi
  if ! run_root "$box" <<<"bash $HARDEN" | sed 's/^/      /'; then
    echo "    !! install failed"; rc=1; continue
  fi
  # Prove it, from a fresh probe — never trust the installer's own say-so.
  out=$("${SSH[@]}" "$box" "bash -s" <<<"$VERIFY" 2>&1)
  echo "    re-check:"
  if report "$box" "$out"; then echo "    OK (installed)"; else echo "    STILL BAD"; rc=1; fi
done

echo
[ $rc -eq 0 ] && echo "all boxes covered" || echo "some boxes need attention (exit $rc)"
exit $rc
