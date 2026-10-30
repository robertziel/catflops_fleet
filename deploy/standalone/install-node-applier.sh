#!/usr/bin/env bash
# deploy/standalone/install-node-applier.sh -- the node applier (vllm_lb/node_applier.py) for ONE fleet cluster as
# systemd --user units, on the host of a node the LB requests (fleet control `local`: the LB requests, this host
# applies). One set per cluster, rendered from the templates beside this script:
#   vllm-node-applier-<cluster>.service   one pass: --cluster <cluster> --env <its server file>
#   vllm-node-applier-<cluster>.path      runs it when the LB writes that cluster's intent (<its node dir>/intent.json)
#   vllm-node-applier-<cluster>.timer     the heartbeat
#
#   deploy/standalone/install-node-applier.sh --cluster NAME [--env FILE]   install or update (idempotent), start, verify
#   deploy/standalone/install-node-applier.sh --cluster NAME --uninstall    stop + disable + remove that cluster's units
#
# --env is the node's server file (default deploy/standalone/.env): a host that runs nothing else of deploy/standalone
# gives only the node's section (up.sh, "node-only"). The node dir is the APPLIER's answer (`--node-dir`: the LB's own
# rule, lb/state/node for a fleet of one, lb/state/clusters/<name>/node for several), never re-derived here.
# Run it as the user who runs this host's compose projects (in the docker group), from this checkout. It refuses,
# naming the fix, before anything is installed: no cluster named, no server file or fleet file, no PyYAML for the
# host's python (the applier validates with the fleet's own loader), a cluster that is not a node this host applies,
# user units that would stop at logout (no linger), or a node dir this user cannot write (lb/state is created by
# Docker as root on the LB's first start). The pre-cluster units (vllm-node-applier.{service,path,timer}, no cluster
# in the name) are removed on install: two appliers for one node would race.
# shortcut: every node's compose project is deploy/standalone's (service `node`), so two local nodes on ONE host would
# replace each other's container — one local node per host until up.sh names a project per node.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
KINDS=(service path timer)
LEGACY=(vllm-node-applier.service vllm-node-applier.path vllm-node-applier.timer)
PY=/usr/bin/python3                      # the system python: no conda/venv on a unit's PATH
CLUSTER=""
ENV_FILE="$REPO/deploy/standalone/.env"
UNINSTALL=0

die() { echo "install-node-applier: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cluster) [ $# -ge 2 ] || die "--cluster needs a name"; CLUSTER="$2"; shift ;;
        --env) [ $# -ge 2 ] || die "--env needs a file"; ENV_FILE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift ;;
        --uninstall) UNINSTALL=1 ;;
        *) die "unknown argument: $1 (--cluster NAME [--env FILE] [--uninstall])" ;;
    esac
    shift
done
[ -n "$CLUSTER" ] || die "name the fleet cluster this host applies: --cluster NAME (lb/fleet.yml, control local)"
[[ "$CLUSTER" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || die "--cluster $CLUSTER is not a fleet cluster name"
UNITS=(); for k in "${KINDS[@]}"; do UNITS+=("vllm-node-applier-$CLUSTER.$k"); done

if [ "$UNINSTALL" = 1 ]; then
    systemctl --user disable --now "vllm-node-applier-$CLUSTER.path" "vllm-node-applier-$CLUSTER.timer" 2>/dev/null || true
    for u in "${UNITS[@]}"; do rm -f "$UNIT_DIR/$u"; done
    systemctl --user daemon-reload
    echo "removed ${UNITS[*]} (its node dir under lb/state stays)"
    exit 0
fi

[[ -f "$ENV_FILE" ]] || die "no server file at $ENV_FILE"
[[ -f "$REPO/lb/fleet.yml" ]] || die "no fleet file at $REPO/lb/fleet.yml"
"$PY" -c 'import yaml' 2>/dev/null || die "$PY cannot import yaml: install python3-yaml (the fleet loader reads YAML)"
command -v docker >/dev/null || die "no docker CLI on PATH"
loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -qx 'Linger=yes' \
    || die "user units stop when $(id -un) logs out: run  loginctl enable-linger $(id -un)  first"

# WHERE this cluster's intent lives: the applier says (and refuses a cluster that is not a node this host applies).
NODE_DIR="$(PYTHONPATH="$REPO/lb" "$PY" -m vllm_lb.node_applier --repo "$REPO" --env "$ENV_FILE" --cluster "$CLUSTER" --node-dir)" \
    || die "the applier refused cluster $CLUSTER (above)"
mkdir -p "$NODE_DIR" 2>/dev/null || true
if [[ ! -d "$NODE_DIR" || ! -w "$NODE_DIR" ]]; then
    echo "$NODE_DIR is not writable by $(id -un) (lb/state was created by Docker as root)." >&2
    rel="${NODE_DIR#"$REPO/lb/state/"}"
    die "fix: docker run --rm -v \"$REPO/lb/state\":/d alpine sh -c 'mkdir -p /d/$rel && chown $(id -u):$(id -g) /d/$rel'"
fi

# One real pass BEFORE the units exist: an applier that cannot run fails here, loudly, and writes the first heartbeat.
PYTHONPATH="$REPO/lb" "$PY" -m vllm_lb.node_applier --repo "$REPO" --env "$ENV_FILE" --cluster "$CLUSTER"

for u in "${LEGACY[@]}"; do
    [[ -f "$UNIT_DIR/$u" ]] || continue
    systemctl --user disable --now "$u" 2>/dev/null || true
    rm -f "$UNIT_DIR/$u"
    echo "removed the pre-cluster unit $u"
done
mkdir -p "$UNIT_DIR"
for k in "${KINDS[@]}"; do
    u="vllm-node-applier-$CLUSTER.$k"
    sed -e "s|@PY@|$PY|g" -e "s|@REPO@|$REPO|g" -e "s|@ENV@|$ENV_FILE|g" -e "s|@CLUSTER@|$CLUSTER|g" \
        -e "s|@NODE_DIR@|$NODE_DIR|g" "$REPO/deploy/standalone/vllm-node-applier.$k.in" > "$UNIT_DIR/.$u.tmp"
    mv "$UNIT_DIR/.$u.tmp" "$UNIT_DIR/$u"
done
systemctl --user daemon-reload
systemctl --user enable --now "vllm-node-applier-$CLUSTER.path" "vllm-node-applier-$CLUSTER.timer"
systemctl --user is-active "vllm-node-applier-$CLUSTER.path" "vllm-node-applier-$CLUSTER.timer"
echo "report: $NODE_DIR/report.json"
