#!/usr/bin/env bash
# THE TAG AN IMAGE IS BUILT UNDER — read from the LB's image matrix, the one place it is written:
#
#   deploy/image-tag.sh <engine> <build> <accelerator> <platform>
#
# lb/vllm_lb/library/images/<engine>.yml holds ONE ENTRY PER LINE in a fixed key order (build, accelerator, platform,
# image, source; lb/tests/test_images.py holds every entry to it), so this host — which has no YAML parser — reads the
# tag with a fixed-string grep. Exactly one line must match: none is the missing combination, named; two cannot happen
# (the LB refuses a matrix stating a combination twice). The LB resolves a cluster's image from the same line, so a
# build tagged here is exactly what a Start launches.
set -euo pipefail
(( $# == 4 )) || { echo "usage: $0 <engine> <build> <accelerator> <platform>" >&2; exit 2; }
engine=$1 build=$2 accelerator=$3 platform=$4
matrix="$(cd "$(dirname "$0")/.." && pwd)/lb/vllm_lb/library/images/$engine.yml"
[[ -f $matrix ]] || { echo "no image matrix for engine $engine ($matrix)" >&2; exit 1; }
lines=$(grep -F "  - {build: $build, accelerator: $accelerator, platform: $platform, image: \"" "$matrix" || true)
count=$(printf '%s' "$lines" | grep -c . || true)
if [[ $count != 1 ]]; then
    echo "no image of $engine build $build for $accelerator × $platform in $matrix (found $count): add the entry" >&2
    exit 1
fi
sed -E 's/.*, image: "([^"]+)", source: .*/\1/' <<<"$lines"
