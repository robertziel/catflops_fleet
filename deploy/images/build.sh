#!/usr/bin/env bash
# Build an engine's image for one cell of the image matrix — THE one build script for every engine:
#
#   deploy/images/build.sh <engine> <build> <accelerator> build            # here: docker build --platform <PLATFORM>
#   deploy/images/build.sh <engine> <build> <accelerator> remote <box> …   # on the first box, copied to the rest by id
#   deploy/images/build.sh <engine> <build> <accelerator> ship             # this host's build to BOXES (save | ssh load)
#   deploy/images/build.sh <engine> <build> <accelerator> check            # which of BOXES holds the tag, and its id
#   deploy/images/build.sh <engine> <build> <accelerator> tag              # the tag, as the image matrix names it
#
# ONE DOCKERFILE PER ENGINE (deploy/images/<engine>/Dockerfile) takes its base as `ARG BASE`; what differs per
# (build × accelerator) — the base pinned by digest, the patches or overlay, the CPU platform — is DATA in the PIN FILE
# deploy/images/<engine>/<build>.<accelerator>.env: KEY=VALUE, one a line, read here line by line and never sourced (a
# pin is not code). PLATFORM (--platform), TARGET (--target: the Dockerfile stage of this build, where the engine has one
# per build) and BASE_ID (the id a locally loaded BASE must have, checked on every box before anything is built) are
# this script's; every other key is handed to docker as a --build-arg, in file order. BASE and PLATFORM are required.
#
# NO FALLBACKS: a cell with no pin file has no known base and is refused naming the file it looked for — never the
# nearest accelerator's base. The tag is the LB's image matrix's (lb/vllm_lb/library/images/<engine>.yml, read by
# deploy/image-tag.sh with the pin's PLATFORM), so a build lands under exactly the name a Start launches, and a cell
# the matrix lacks is refused before anything runs.
#
# WHERE IT BUILDS: an image whose Dockerfile only COPYs and sets ENV over its base (the vLLM pair stages) builds `here`
# for any platform — no instruction runs. One that RUNs (patches applied, a build gate, a compile) builds `remote`, on a
# box of its platform: the context is streamed over ssh (nothing is staged on the boxes), built ONCE on the first box
# and copied to every other box by `docker save | docker load`, each copy's id checked — a TP pair runs the same bits on
# each rank, and two builds of one tree are not byte-identical. Boxes are ssh aliases, never defaulted.
set -euo pipefail
usage() { echo "usage: $0 <engine> <build> <accelerator> build|remote <box> [<box> ...]|ship|check|tag" >&2; sed -n '2,8p' "$0" >&2; exit 2; }
(( $# >= 4 )) || usage
engine=$1 build=$2 accelerator=$3 action=$4
shift 4
case $action in build|remote|ship|check|tag) ;; *) usage ;; esac
if [[ $action == remote ]]; then (( $# >= 1 )) || usage; fi

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
dir="$here/$engine"
pin_rel="deploy/images/$engine/$build.$accelerator.env"
die() { echo "$*" >&2; exit 1; }
[[ -f $dir/Dockerfile ]] || die "no image recipe for engine $engine (deploy/images/$engine/Dockerfile)"
[[ -f $repo/$pin_rel ]] || die "no pin for $engine build $build on $accelerator: $pin_rel is missing — its base is not known here; add the file (BASE pinned by digest, PLATFORM, the build's patches)"

# THE PIN: plain KEY=VALUE, a value always present, each key once. Anything else (an `export`, a quote-wrapped shell
# line, an empty value) is refused naming the line: a pin that does not read as data is not guessed at.
declare -A pin=()
keys=()
n=0
while IFS= read -r line || [[ -n $line ]]; do
  n=$((n + 1))
  [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
  [[ $line =~ ^([A-Z][A-Z0-9_]*)=([^[:space:]].*)$ ]] || die "$pin_rel line $n is not KEY=VALUE with a value: $line"
  key=${BASH_REMATCH[1]}
  [[ -z ${pin[$key]+set} ]] || die "$pin_rel gives $key twice (line $n)"
  pin[$key]=${BASH_REMATCH[2]}
  keys+=("$key")
done <"$repo/$pin_rel"
for key in BASE PLATFORM; do
  [[ -n ${pin[$key]+set} ]] || die "$pin_rel has no $key"
done

TAG=$("$repo/deploy/image-tag.sh" "$engine" "$build" "$accelerator" "${pin[PLATFORM]}")

# docker build's arguments after `build`, the context last but not included: the pin's platform and target, and
# every other key as a build-arg.
args=(--platform "${pin[PLATFORM]}" -t "$TAG")
[[ -z ${pin[TARGET]+set} ]] || args+=(--target "${pin[TARGET]}")
for key in "${keys[@]}"; do
  case $key in PLATFORM|TARGET|BASE_ID) ;; *) args+=(--build-arg "$key=${pin[$key]}") ;; esac
done

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8)
# q <argv…>: argv as one line for a box's shell, each word quoted for it (a build-arg with spaces stays one word).
q() {
  local cmd
  printf -v cmd '%q ' "$@"
  printf '%s' "${cmd% }"
}
on() {  # on <box> <argv…>: run argv on the box
  local box=$1
  shift
  "${SSH[@]}" "$box" "$(q "$@")"
}
# base_ok <where> <the id it has>: a locally loaded base is held to the pin's BASE_ID before anything is built on it.
base_ok() {
  [[ $2 == "${pin[BASE_ID]}" ]] || die "$1: ${pin[BASE]} is $2, expected ${pin[BASE_ID]} ($pin_rel BASE_ID)"
}
boxes() {  # the pair, as ssh names it: never a default (ship/check dial nothing without it)
  [[ -n ${BOXES:-} ]] || { echo "BOXES is not set: the pair ssh hosts, space-separated" >&2; exit 2; }
  read -ra BOXES <<<"$BOXES"
}

case $action in
  tag) echo "$TAG" ;;
  build)
    [[ -z ${pin[BASE_ID]+set} ]] || base_ok here "$(docker image inspect -f '{{.Id}}' "${pin[BASE]}")"
    DOCKER_BUILDKIT=1 docker build "${args[@]}" "$dir"
    docker image inspect -f '{{.Id}} {{.Architecture}}' "$TAG" ;;
  remote)
    if [[ -n ${pin[BASE_ID]+set} ]]; then
      for box in "$@"; do base_ok "$box" "$(on "$box" docker image inspect -f '{{.Id}}' "${pin[BASE]}")"; done
    fi
    builder=$1
    shift
    echo "== $builder: building $TAG" >&2
    # BuildKit forced on: it builds only the stages a --target needs, whatever the box's docker defaults to.
    tar -C "$dir" -c --exclude='*.env' . | "${SSH[@]}" "$builder" "DOCKER_BUILDKIT=1 $(q docker build -q "${args[@]}" -)"
    built=$(on "$builder" docker image inspect -f '{{.Id}}' "$TAG")
    for box in "$@"; do
      echo "== $box: copying $TAG from $builder" >&2
      on "$builder" docker save "$TAG" | on "$box" docker load >/dev/null
      got=$(on "$box" docker image inspect -f '{{.Id}}' "$TAG")
      [[ $got == "$built" ]] || die "$box: $TAG is $got, $builder built $built"
    done
    echo "$TAG = $built on $builder $*" >&2 ;;
  ship)
    boxes
    docker image inspect "$TAG" >/dev/null || die "no $TAG here: build first"
    for box in "${BOXES[@]}"; do
      echo "== $box: $TAG" >&2
      docker save "$TAG" | on "$box" docker load
    done ;;
  check)
    boxes
    echo "here $(docker image inspect -f '{{.Id}}' "$TAG" 2>/dev/null || echo absent)"
    for box in "${BOXES[@]}"; do
      echo "$box $(on "$box" docker image inspect -f '{{.Id}}' "$TAG" 2>/dev/null || echo absent)"
    done ;;
esac
