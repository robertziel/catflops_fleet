#!/usr/bin/env bash
# THE UX AUDIT: Playwright (pinned 1.65.0, in its own image) against the running demo (up.sh), joined to
# demo-lb's network namespace, reading the harness's inputs from the demo's state volume.
#
#   lb/tests/demo/up.sh && lb/tests/demo/audit.sh <screenshot dir> ; lb/tests/demo/down.sh
#
# Same DEMO_PORT / DEMO_ID as up.sh (demo-env.sh). The playwright package is installed once into the instance's labelled
# volume fleet-demo-pw-<id> (down.sh removes it with the rest).
# Exits with audit.mjs's status: 1 when any row fails.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:?usage: audit.sh <screenshot dir>}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
img=mcr.microsoft.com/playwright:v1.65.0-noble
. "$here/demo-env.sh"
docker volume create --label "$label" --label "$ilabel" "$pwvol" >/dev/null
docker run --rm --label "$label" --label "$ilabel" -v "$pwvol":/pw -w /pw "$img" \
  sh -c '[ -d node_modules/playwright ] || { npm init -y >/dev/null && npm install --no-audit --no-fund playwright@1.65.0 >/dev/null; }'
docker run --rm --label "$label" --label "$ilabel" --network "container:$lbname" -v "$pwvol":/pw:ro \
  -v "$vol":/data:ro -v "$here":/demo:ro -v "$out":/out "$img" node /demo/audit.mjs /out
