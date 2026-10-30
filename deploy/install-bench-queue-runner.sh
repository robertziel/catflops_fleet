#!/usr/bin/env bash
# The swarm runner (lb/bench_queue_runner.py) as a systemd --user unit, rendered from its template.
#
#   LB_CLUSTER_URL=http://<head>:8100 deploy/install-bench-queue-runner.sh      install or update, start
#   LB_CLUSTER_URL=http://<head>:8100 deploy/install-bench-queue-runner.sh --render   print it, change nothing
#   deploy/install-bench-queue-runner.sh --uninstall                           stop, disable, remove
#
# WHY A TEMPLATE: the unit needs this checkout's path, the operator's HOME (where `claude` lives) and the
# head engine's address. None of them belongs in a committed file, so they are substituted here.
#
# LB_CLUSTER_URL is required. It is the head rank's vLLM API (the same value as the LB's own), and the
# runner's session reads that head's /metrics for its guard C. There is no default: a guessed head is
# a session that measures the wrong machine.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT=vllm-bench-queue-runner.service
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TEMPLATE="$REPO/deploy/$UNIT.in"
PY=/usr/bin/python3                      # the system python: no conda/venv on a unit's PATH

# A value spliced into sed's replacement text: escape the delimiter and the replacement's own specials.
sed_escape() { printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g'; }

render() {
    : "${LB_CLUSTER_URL:?set LB_CLUSTER_URL to the head rank vLLM API, e.g. LB_CLUSTER_URL=http://<head>:8100}"
    [[ -f "$TEMPLATE" ]] || { echo "no template at $TEMPLATE" >&2; exit 1; }
    sed -e "s|@PY@|$(sed_escape "$PY")|g" \
        -e "s|@REPO@|$(sed_escape "$REPO")|g" \
        -e "s|@HOME@|$(sed_escape "$HOME")|g" \
        -e "s|@CLUSTER_URL@|$(sed_escape "$LB_CLUSTER_URL")|g" "$TEMPLATE"
}

case "${1:-}" in
    --render)
        render
        exit 0
        ;;
    --uninstall)
        systemctl --user disable --now "$UNIT" 2>/dev/null || true
        rm -f "$UNIT_DIR/$UNIT"
        systemctl --user daemon-reload
        echo "removed $UNIT"
        exit 0
        ;;
    "") ;;
    *) echo "usage: $0 [--render|--uninstall]   (LB_CLUSTER_URL required to install)" >&2; exit 2 ;;
esac

mkdir -p "$UNIT_DIR"
render > "$UNIT_DIR/.$UNIT.tmp"          # a render that fails leaves the installed unit untouched
mv "$UNIT_DIR/.$UNIT.tmp" "$UNIT_DIR/$UNIT"
systemctl --user daemon-reload
systemctl --user enable "$UNIT" >/dev/null 2>&1
systemctl --user restart "$UNIT"
systemctl --user is-active "$UNIT"
echo "installed $UNIT_DIR/$UNIT"
echo "to keep it running with nobody logged in: sudo loginctl enable-linger $(id -un)"
