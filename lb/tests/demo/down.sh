#!/usr/bin/env bash
# Stop and remove everything lb/tests/demo/up.sh started for ONE instance (the same DEMO_PORT / DEMO_ID as up.sh:
# demo-env.sh): its containers, its state volume and its playwright volume — every resource labelled with its id, and
# nothing of another instance. Prints what it removed; exits non-zero if anything of this instance is still there.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/demo-env.sh"
ids="$(docker ps -aq --filter "label=$ilabel")"
if [ -n "$ids" ]; then docker rm -f $ids >/dev/null; echo "$id: removed containers: $(echo $ids | wc -w)"; fi
for v in $(docker volume ls -q --filter "label=$ilabel"); do docker volume rm "$v" >/dev/null && echo "$id: removed volume $v"; done
left="$(docker ps -aq --filter "label=$ilabel")$(docker volume ls -q --filter "label=$ilabel")"
[ -z "$left" ] || { echo "$id: still present: $left" >&2; exit 1; }
