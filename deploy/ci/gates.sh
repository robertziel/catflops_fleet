#!/usr/bin/env bash
# The CI gates: every suite CLAUDE.md lists, each in its own throwaway container, runnable anywhere Docker is.
#
#   deploy/ci/gates.sh <gate>      gate = lb | power | usage-mcp | deploy | browser
#   deploy/ci/gates.sh all         every gate, in order; exits non-zero when any one fails
#
# WHY no bind mount: a CI agent that drives the host's Docker daemon through its socket (a Jenkins container
# with /var/run/docker.sock) sees a workspace path the daemon cannot see, so `-v "$PWD":/app` would mount an
# empty directory. Each suite gets its files as a tar on stdin instead — the files git tracks plus new ones it
# does not ignore, so a developer's gitignored .env or fleet.yml never leaves the checkout. A service suite runs
# in that service's image (its toolchain) with the service directory unpacked over /app, which is what the
# documented `-v "$PWD":/app` run sees. Nothing here writes the checkout.
#
# CI_CPUS caps each test container (a CI box usually serves something else too); unset = no cap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${CI_TAG:-ci-$$}"
CPUS=()
[[ -n "${CI_CPUS:-}" ]] && CPUS=(--cpus "${CI_CPUS}")

die() { echo "gates.sh: $*" >&2; exit 2; }

# <path>: a tar of the path's files, relative to it — tracked plus untracked-but-not-ignored.
pack() {
  git -C "${ROOT}/$1" ls-files -z --cached --others --exclude-standard \
    | tar -C "${ROOT}/$1" -c --null -T -
}

# <service dir> <image name>: build the service image, run its pytest over the unpacked directory, remove the image.
service_gate() {
  local dir="$1" image="$2:${TAG}" rc=0
  docker build -q -t "${image}" "${ROOT}/${dir}" >/dev/null
  pack "${dir}" | docker run --rm -i "${CPUS[@]}" --user 0 -w /app "${image}" \
    sh -c 'tar -x && python -m pytest -p no:cacheprovider -rfE' || rc=$?
  docker rmi -f "${image}" >/dev/null || true
  return "${rc}"
}

# <image> <command>: the whole checkout unpacked to /repo, then the command.
tar_gate() {
  local image="$1" cmd="$2"
  pack . | docker run --rm -i "${CPUS[@]}" "${image}" sh -c "mkdir -p /repo && cd /repo && tar -x && ${cmd}"
}

gate() {
  case "$1" in
    lb)        service_gate lb vllm-lb-ci ;;
    power)     service_gate power vllm-power-ci ;;
    usage-mcp) service_gate usage-mcp vllm-usage-mcp-ci ;;
    deploy)    tar_gate python:3.12-slim "python -m unittest discover -s deploy/tests" ;;
    browser)   tar_gate node:22-slim 'cd lb && fail=0; n=0; for f in tests/*.node.js; do n=$((n+1)); node "$f" >/tmp/out 2>&1 || { fail=$((fail+1)); echo "FAIL $f"; cat /tmp/out; }; done; echo "$((n-fail))/$n browser files passed"; [ "$fail" -eq 0 ]' ;;
    *)         die "unknown gate '$1' (lb | power | usage-mcp | deploy | browser | all)" ;;
  esac
}

[[ $# -eq 1 ]] || die "usage: gates.sh <lb | power | usage-mcp | deploy | browser | all>"
if [[ "$1" == all ]]; then
  failed=()
  for g in lb power usage-mcp deploy browser; do
    echo "== ${g}"
    gate "${g}" || failed+=("${g}")
  done
  [[ ${#failed[@]} -eq 0 ]] || { echo "failed: ${failed[*]}" >&2; exit 1; }
else
  gate "$1"
fi
