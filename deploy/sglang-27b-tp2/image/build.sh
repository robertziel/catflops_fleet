#!/usr/bin/env bash
# The cluster's SGLang image: the image matrix cell sglang × fleet.4 × nvidia-sm121 (the newest build, for the pair's
# GB10s), on the ONE build (deploy/images/build.sh; the recipe is deploy/images/sglang/Dockerfile, its base and patches
# fleet.4.nvidia-sm121.env).
#   build.sh         # on the pair (BOXES="<host> <host>", ssh aliases; no default): built once on the first box — the
#                    # Dockerfile's test step fails the build if a patched adapter regresses — and copied to the other
#   build.sh tag     # the tag, as the image matrix names it
set -euo pipefail
cell=(sglang fleet.4 nvidia-sm121)
case ${1:-} in
  tag) exec "$(dirname "$0")/../../images/build.sh" "${cell[@]}" tag ;;
  "")
    [[ -n ${BOXES:-} ]] || { echo "BOXES is not set: the pair ssh hosts, space-separated" >&2; exit 2; }
    read -ra boxes <<<"$BOXES"
    exec "$(dirname "$0")/../../images/build.sh" "${cell[@]}" remote "${boxes[@]}" ;;
  *) sed -n '2,7p' "$0" >&2; exit 2 ;;
esac
