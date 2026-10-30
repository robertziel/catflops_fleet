#!/usr/bin/env bash
# deploy/standalone/up.sh -- check, or bring up, ONE standalone server from its deploy/standalone/.env.
#
#   up.sh --check [--env FILE]          validate the server file; touch no container
#   up.sh [--env FILE] [TARGET ...]     TARGET: node lb bench chat edge   (none = all, in that order; `chat` only
#                                       where chat.override.yml is in the tree, `edge` only where docker-compose.yml,
#                                       the signed-in edge, is)
#   up.sh [--env FILE] node-stop        stop the node (never part of a full bring-up)
#   A server file for node / node-stop alone may carry only the node's section (EDGE_NODE_*, EDGE_MODELS_DIR): a host
#   whose LB is not this directory's (its own lb/) runs a node and nothing else from here.
#   --node-env FILE                     the applied node.env to read after the server file (the applier passes its
#                                       cluster's: lb/state/clusters/<name>/node/node.env when the fleet names several);
#                                       left out, the one-cluster lb/state/node/node.env when it exists
#
# The node's GPU is attached by node-gpu.<EDGE_NODE_GPU_RUNTIME>.yml (nvidia: a device reservation; rocm: /dev/kfd +
# /dev/dri) -- the serving card's hardware type states the runtime and node-env renders it, so the file is chosen
# by DATA, never by a branch on a vendor here. The server it runs is node-engine.<EDGE_NODE_ENGINE>.yml (vllm,
# faster-whisper): the launch choice's catalog `node_engine`, rendered the same way.
#
# The node the LB requests (fleet control `local`): its host's applier (vllm_lb/node_applier.py, installed by
# install-node-applier.sh) renders the LB's last intent into lb/state/node/node.env and runs `node` / `node-stop`.
# Both read that file as a SECOND --env-file, after the server file (later wins), so a hand-run `up.sh node` brings
# up what the panel last asked for instead of reverting it; without the file they are the server file alone.
#
# `bench` is the gateway's sealed swarm runner (lb/bench_runner): the "Start test swarm" button's other
# half. It needs no variable of its own — its box, network and engine are this server file's.
#
# Why one script instead of `docker compose up` in four directories:
#   - every compose project gets the SAME server file (--env-file), so a value two of them share --
#     the LB's address is published by lb/ and dialled by both gates and nginx -- is one line, never
#     two that can disagree. lb/.env still carries the LB's own settings.
#   - only NAMED services are started: a project's base file may define a tunnel client that would claim a
#     shared proxy on the frps, and a bare `up` would start it.
#   - before any container is touched, a server file is refused when a value is missing (named) or
#     when anything that serves without sign-in -- the loopback door, either gate, the LB, the node,
#     searxng -- is not on loopback. The public door is the only listener that may leave the box, and it
#     asks the jwt gate.
#
# A chat front-end is a separate project (catflops_chat); where its override (chat.override.yml) is in the tree the
# `chat` target brings it up on this server's LB, and its server-file lines (EDGE_CHAT_*) are required only then. Any
# OpenAI- or Anthropic-compatible client works against the LB (EDGE_LB_PUBLISH) without it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$HERE/.env"
NODE_ENV=""
CHECK=0
TARGETS=()

die() { echo "up.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1 ;;
        --env) [ $# -ge 2 ] || die "--env needs a file"; ENV_FILE="$2"; shift ;;
        --node-env) [ $# -ge 2 ] || die "--node-env needs a file"; NODE_ENV="$2"; shift ;;
        node-stop|node|lb|bench|chat|edge) TARGETS+=("$1") ;;
        *) die "unknown argument: $1 (targets: node lb bench chat edge, or node-stop)" ;;
    esac
    shift
done
# THE SIGNED-IN EDGE (docker-compose.yml: two gates + nginx) is built against a private sign-in client, so a published
# tree ships without it (PUBLISHING.md cuts its files and the door lines of .env.example): the `edge` target then does
# not exist, and the proxy in front of the LB is the operator's own. THE CHAT (chat.override.yml, on chat/) is its own
# project, catflops_chat, and leaves the published tree the same way: no `chat` target without the override.
EDGE_FILE="$HERE/docker-compose.yml"
if [ -f "$EDGE_FILE" ]; then HAS_EDGE=1; else HAS_EDGE=0; fi
CHAT_FILE="$HERE/chat.override.yml"
if [ -f "$CHAT_FILE" ]; then HAS_CHAT=1; else HAS_CHAT=0; fi
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=(node lb bench); [ "$HAS_CHAT" = 1 ] && TARGETS+=(chat); [ "$HAS_EDGE" = 1 ] && TARGETS+=(edge); fi
if [ "$HAS_EDGE" = 0 ] && printf '%s\n' "${TARGETS[@]}" | grep -qx edge; then
    die "edge: this tree ships no signed-in edge ($EDGE_FILE is not here) -- put your own authenticating proxy in front of the LB (EDGE_LB_PUBLISH)"
fi
if [ "$HAS_CHAT" = 0 ] && printf '%s\n' "${TARGETS[@]}" | grep -qx chat; then
    die "chat: this tree ships no chat front-end ($CHAT_FILE is not here) -- a chat is a separate project, catflops_chat; any OpenAI- or Anthropic-compatible client works against the LB (EDGE_LB_PUBLISH)"
fi
# The chat's lines (its loopback publish, public origin and slug) are read by the chat -- and, with the edge, by nginx's
# public chat door -- so they are required only for those targets: a server file without them still brings up the LB.
CHAT_VARS=(EDGE_CHAT_PUBLISH EDGE_PUBLIC_CHAT_ORIGIN EDGE_CHAT_AUDIENCE)
RUNS_CHAT=0
printf '%s\n' "${TARGETS[@]}" | grep -qx 'chat\|edge' && RUNS_CHAT=1
[ -f "$ENV_FILE" ] || die "no server file at $ENV_FILE -- copy $HERE/.env.example to it and fill in every line"

# The server file is READ, never sourced: KEY=VALUE lines only, nothing in it is executed.
declare -A V=()
read_env() {
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        [[ "$line" == *=* ]] || die "not KEY=VALUE in $1: $line"
        V["${line%%=*}"]="${line#*=}"
    done < "$1"
}
read_env "$ENV_FILE"

# The node's env: the server file, then — when the applier has applied an intent — its rendering (later wins): the
# one it names (--node-env, its cluster's), else the one-cluster path when it exists. A named file that is not there
# is refused: the applier asked for exactly that intent.
APPLIED=()
if [ -n "$NODE_ENV" ]; then
    [ -f "$NODE_ENV" ] || die "--node-env $NODE_ENV is not a file"
    APPLIED=(--env-file "$NODE_ENV")
elif [ -f "$REPO/lb/state/node/node.env" ]; then
    APPLIED=(--env-file "$REPO/lb/state/node/node.env")
fi
[ ${#APPLIED[@]} -gt 0 ] && read_env "${APPLIED[1]}"
# A NODE-ONLY bring-up (every target node / node-stop): a host that runs a node and nothing else of deploy/standalone
# (its LB is its own lb/, e.g. box-c's) gives the node's section alone — the gate's, the chat's and the doors' lines are not
# its to give. Required is then that section (EDGE_NODE_* and EDGE_MODELS_DIR), read AFTER the applied node.env (the
# applier renders the rest from the fleet). Any other target needs the whole file.
NODE_ONLY=1
for t in "${TARGETS[@]}"; do case "$t" in node|node-stop) ;; *) NODE_ONLY=0 ;; esac; done

# Required = the EDGE_ names .env.example documents (all, or the node section): one list, the same one the tests read.
if [ "$NODE_ONLY" = 1 ]; then
    mapfile -t REQUIRED < <(sed -n 's/^\(EDGE_\(NODE_[A-Z0-9_]*\|MODELS_DIR\)\)=.*/\1/p' "$HERE/.env.example")
    LOOPBACK=(EDGE_NODE_PUBLISH)
else
    mapfile -t REQUIRED < <(sed -n 's/^\(EDGE_[A-Z0-9_]*\)=.*/\1/p' "$HERE/.env.example")
    if [ "$RUNS_CHAT" = 0 ]; then
        mapfile -t REQUIRED < <(printf '%s\n' "${REQUIRED[@]}" | grep -vx "$(printf '%s\|' "${CHAT_VARS[@]}")__none__")
    fi
    LOOPBACK=(EDGE_LB_PUBLISH EDGE_NODE_PUBLISH EDGE_SEARXNG_PUBLISH)
    [ "$RUNS_CHAT" = 1 ] && LOOPBACK+=(EDGE_CHAT_PUBLISH)
    [ "$HAS_EDGE" = 1 ] && LOOPBACK+=(EDGE_LOCAL_LISTEN EDGE_GATE_LOCAL_BIND EDGE_GATE_PUBLIC_BIND)
fi
[ ${#REQUIRED[@]} -gt 0 ] || die "$HERE/.env.example names no EDGE_ variable"
for k in "${REQUIRED[@]}"; do
    [ -n "${V[$k]:-}" ] || die "$k is not set in $ENV_FILE${APPLIED[1]:+ or ${APPLIED[1]}}"
done

loopback() {
    case "$1" in
        127.*:[0-9]*|localhost:[0-9]*|'[::1]':[0-9]*) return 0 ;;
        *) return 1 ;;
    esac
}
# A NODE ON ANOTHER BOX (EDGE_NODE_DOCKER_HOST, e.g. ssh://user@host — a node-only file): every docker call goes to
# that box's daemon, and the node is published on its private link to this box — loopback THERE is out of the LB's
# reach, and anything wider serves without sign-in.
private() {
    case "$1" in
        10.*:[0-9]*|192.168.*:[0-9]*|172.1[6-9].*:[0-9]*|172.2[0-9].*:[0-9]*|172.3[01].*:[0-9]*) return 0 ;;
        *) return 1 ;;
    esac
}
if [ -n "${V[EDGE_NODE_DOCKER_HOST]:-}" ]; then
    [ "$NODE_ONLY" = 1 ] || die "EDGE_NODE_DOCKER_HOST names another box's docker: only a node-only server file (targets node / node-stop) may"
    export DOCKER_HOST="${V[EDGE_NODE_DOCKER_HOST]}"
    LOOPBACK=()
    private "${V[EDGE_NODE_PUBLISH]}" || die "EDGE_NODE_PUBLISH=${V[EDGE_NODE_PUBLISH]} must be a private address (10/8, 172.16/12, 192.168/16) on $DOCKER_HOST's link to this box: loopback there is out of reach, anything wider serves without sign-in"
fi
for k in "${LOOPBACK[@]}"; do
    loopback "${V[$k]}" || die "$k=${V[$k]} must be a loopback host:port (127.x, localhost or [::1]): it serves without sign-in, or only to this box"
done
if [ "$NODE_ONLY" = 0 ]; then
    [ "$HAS_EDGE" = 0 ] || case "${V[EDGE_LOCAL_ORIGIN]}" in
        http://127.*|http://localhost:*|'http://[::1]':*) ;;
        *) die "EDGE_LOCAL_ORIGIN=${V[EDGE_LOCAL_ORIGIN]} must be an http loopback origin (what the browser sends through the tunnel)" ;;
    esac
    # The public origins the gates and the chat trust: https, where the tree has the edge (its lines are documented
    # then) or the chat runs; a published tree has neither, and its own proxy's origin is its own.
    ORIGINS=()
    [ "$HAS_EDGE" = 1 ] && ORIGINS+=(EDGE_PUBLIC_VLLM_ORIGIN EDGE_AUTH_ISSUER)
    [ "$RUNS_CHAT" = 1 ] && ORIGINS+=(EDGE_PUBLIC_VLLM_ORIGIN EDGE_AUTH_ISSUER EDGE_PUBLIC_CHAT_ORIGIN)
    for k in "${ORIGINS[@]}"; do
        case "${V[$k]:-}" in
            https://*) ;;
            *) die "$k=${V[$k]:-} must be an https origin" ;;
        esac
    done
fi
# THE NAMES THIS SERVER'S LB ANSWERS TO (lb/vllm_lb/host_guard.py, the DNS-rebinding guard): LB_ALLOWED_HOSTS is
# REQUIRED by the LB, and every name in it is already in this server file -- so it is DERIVED here and handed to the lb
# step's compose calls, never written a second time in lb/.env (refused there below):
#   EDGE_LB_PUBLISH   the address the LB is published at: the gates' health probe and every host-side client dial it
#   vllm-lb:8010      the LB's name on its networks: the chat (CHAT_UPSTREAM) and any client container
#   lb:8010           the sealed swarm runner, which calls http://lb:8010 through its egress relay (lb/bench_runner)
#   the doors' hosts  with the edge, nginx forwards each door's Host as `$host` -- the name alone, no port: the host of
#                     EDGE_LOCAL_ORIGIN (the loopback door) and of EDGE_PUBLIC_VLLM_ORIGIN (the public one)
#   EDGE_LB_PROXY_HOSTS  what YOUR proxy presents in front of EDGE_LB_PUBLISH (a tree without the edge), or `none`
authority() {
    [[ "$1" =~ ^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9]([A-Za-z0-9_.-]*[A-Za-z0-9])?)(:[0-9]{1,5})?$ ]]
}
origin_host() {
    local o="${1#*://}"
    o="${o%%/*}"
    case "$o" in
        '['*) echo "${o%%]*}]" ;;
        *) echo "${o%%:*}" ;;
    esac
}
LB_ALLOWED=""
if [ "$NODE_ONLY" = 0 ]; then
    LB_HOSTS=("${V[EDGE_LB_PUBLISH]}" vllm-lb:8010 lb:8010)
    [ "$HAS_EDGE" = 1 ] && LB_HOSTS+=("$(origin_host "${V[EDGE_LOCAL_ORIGIN]}")" "$(origin_host "${V[EDGE_PUBLIC_VLLM_ORIGIN]}")")
    PROXY_HOSTS="${V[EDGE_LB_PROXY_HOSTS]}"
    if [ "$PROXY_HOSTS" != none ]; then
        case "$PROXY_HOSTS" in
            ,*|*,|*,,*) die "EDGE_LB_PROXY_HOSTS=$PROXY_HOSTS has an empty entry: comma-separated host[:port], or none" ;;
        esac
        IFS=, read -ra EXTRA <<< "$PROXY_HOSTS"
        for h in "${EXTRA[@]}"; do
            [ "$h" != none ] || die "EDGE_LB_PROXY_HOSTS=$PROXY_HOSTS: none stands alone"
            authority "$h" || die "EDGE_LB_PROXY_HOSTS=$PROXY_HOSTS: '$h' is not host[:port] (no scheme, no path) -- the Host your proxy presents to the LB, or none"
        done
        LB_HOSTS+=("${EXTRA[@]}")
    fi
    LB_ALLOWED="$(IFS=,; echo "${LB_HOSTS[*]}")"
    echo "up.sh: the LB answers to LB_ALLOWED_HOSTS=$LB_ALLOWED"
fi
echo "up.sh: $ENV_FILE is complete; everything without sign-in is on loopback"
[ "$CHECK" = 1 ] && exit 0

PROJECTS=(lb); [ "$RUNS_CHAT" = 1 ] && [ "$HAS_CHAT" = 1 ] && PROJECTS+=(chat)
[ "$NODE_ONLY" = 1 ] || for project in "${PROJECTS[@]}"; do
    [ -f "$REPO/$project/.env" ] || die "$REPO/$project/.env is missing (its own settings: see $project/.env.example)"
done
# ONE place per value: the LB's names are derived from this server file above, so a line of its own in lb/.env would be
# a second definition that silently loses to it -- refused instead.
if [ "$NODE_ONLY" = 0 ] && grep -qE '^[[:space:]]*(export[[:space:]]+)?LB_ALLOWED_HOSTS=' "$REPO/lb/.env"; then
    die "$REPO/lb/.env sets LB_ALLOWED_HOSTS: on a standalone server up.sh derives it from $ENV_FILE (EDGE_LB_PUBLISH, EDGE_LB_PROXY_HOSTS, the doors) -- remove the line"
fi
# The LB reads what this server IS from its fleet file: without it the new LB cannot boot, so refuse
# before any container is touched.
if printf '%s\n' "${TARGETS[@]}" | grep -qx lb; then
    [ -f "$REPO/lb/fleet.yml" ] || die "$REPO/lb/fleet.yml is missing: the LB reads this server's box, cards and cluster from it (copy lb/fleet.one-host.example.yml and fill it in)"
fi
docker network inspect "${V[EDGE_NODE_NETWORK]}" >/dev/null 2>&1 \
    || docker network create "${V[EDGE_NODE_NETWORK]}" >/dev/null

# HOW THE NODE'S CONTAINER ATTACHES ITS CARD: the override named by its runtime (the applied value wins, as compose's).
GPU_FILE="$HERE/node-gpu.${V[EDGE_NODE_GPU_RUNTIME]}.yml"
# THE MFU BYTE-SIZE PATCH: the override named by its value (node-env renders on/off from the checkpoint's quant method).
MFU_FILE="$HERE/node-mfu.${V[EDGE_NODE_MFU_PATCH]}.yml"
# THE SERVER IT RUNS: the override named by its engine (node-env renders it from the launch choice's catalog row).
ENGINE_FILE="$HERE/node-engine.${V[EDGE_NODE_ENGINE]}.yml"
if printf '%s\n' "${TARGETS[@]}" | grep -qx 'node\|node-stop'; then
    [ -f "$ENGINE_FILE" ] || die "EDGE_NODE_ENGINE=${V[EDGE_NODE_ENGINE]} names no override: there is $(cd "$HERE" && ls node-engine.*.yml | tr '\n' ' ')"
    [ -f "$GPU_FILE" ] || die "EDGE_NODE_GPU_RUNTIME=${V[EDGE_NODE_GPU_RUNTIME]} names no override: there is $(cd "$HERE" && ls node-gpu.*.yml | tr '\n' ' ')"
    [ -f "$MFU_FILE" ] || die "EDGE_NODE_MFU_PATCH=${V[EDGE_NODE_MFU_PATCH]} names no override: there is $(cd "$HERE" && ls node-mfu.*.yml | tr '\n' ' ')"
fi

for t in "${TARGETS[@]}"; do
    echo "up.sh: -- $t --"
    case "$t" in
        node)
            docker compose --env-file "$ENV_FILE" "${APPLIED[@]}" -f "$HERE/node.yml" -f "$GPU_FILE" -f "$MFU_FILE" -f "$ENGINE_FILE" up -d node ;;
        node-stop)
            docker compose --env-file "$ENV_FILE" "${APPLIED[@]}" -f "$HERE/node.yml" -f "$GPU_FILE" -f "$MFU_FILE" -f "$ENGINE_FILE" stop node ;;
        lb)
            # Prove the NEW LB boots before the running one is replaced: build it, then load its config
            # (fleet file, retired settings refused) with the service's own env and mounts in a throwaway
            # container. A refused boot is then a failed deploy, not an outage of this server's gateway.
            # LB_ALLOWED_HOSTS: derived from this server file above (the shell's value wins compose's interpolation).
            (cd "$REPO/lb" && export LB_ALLOWED_HOSTS="$LB_ALLOWED" \
                && docker compose --env-file .env --env-file "$ENV_FILE" -f docker-compose.yml -f "$HERE/lb.override.yml" build vllm-lb \
                && docker compose --env-file .env --env-file "$ENV_FILE" -f docker-compose.yml -f "$HERE/lb.override.yml" run --rm --no-deps --entrypoint python vllm-lb -c "from vllm_lb.config import Config; Config.from_env(); print('up.sh: the new LB loads its config')" \
                && docker compose --env-file .env --env-file "$ENV_FILE" -f docker-compose.yml -f "$HERE/lb.override.yml" up -d vllm-lb searxng) ;;
        bench)
            # Every value from THIS server file, mapped once, here: the panel names EDGE_BOX; the egress
            # reaches the LB (container vllm-lb, :8010) and the node (:8000) on EDGE_NODE_NETWORK, which
            # both join; the swarm directory is the one the LB mounts (lb/state -> /data, /data/benchq).
            (cd "$REPO/lb/bench_runner" && \
                BENCH_BOX="${V[EDGE_BOX]}" \
                BENCH_STATE_DIR="$REPO/lb/state/benchq" \
                BENCH_UPSTREAM_NETWORK="${V[EDGE_NODE_NETWORK]}" \
                BENCH_LB_TARGET="vllm-lb:8010" \
                BENCH_ENGINE_TARGET="${V[EDGE_NODE_NAME]}:8000" \
                docker compose --env-file "$ENV_FILE" up -d --build egress runner) ;;
        chat)
            (cd "$REPO/chat" && docker compose --env-file .env --env-file "$ENV_FILE" -f docker-compose.yml -f "$HERE/chat.override.yml" up -d --build vllm-chat vllm-chat-sandbox) ;;
        edge)
            # The public gate's API key store is a bind mount: docker would create a missing folder as root, where
            # the gate (uid 1000) cannot keep it. Say so, with the command, before a container is touched.
            [ -d "${V[EDGE_GATE_KEYS_DIR]}" ] || die "EDGE_GATE_KEYS_DIR=${V[EDGE_GATE_KEYS_DIR]} is not a directory: create it for the gate's uid -- install -d -m 700 -o 1000 -g 1000 ${V[EDGE_GATE_KEYS_DIR]}"
            docker compose --env-file "$ENV_FILE" -f "$HERE/docker-compose.yml" up -d gate gate-public
            # nginx renders its template only at start: recreate it so a changed address takes effect.
            docker compose --env-file "$ENV_FILE" -f "$HERE/docker-compose.yml" up -d --force-recreate nginx ;;
    esac
done
