#!/usr/bin/env bash
# The fleet's TensorFold image: the image matrix cell tensorfold × native × nvidia-sm121, on the ONE build
# (deploy/images/build.sh; the recipe is deploy/images/tensorfold/Dockerfile, every pin native.nvidia-sm121.env).
#   ./build.sh <build-box> [<box> ...]   # built ONCE on the first box named and copied to the others by id (a TP=2
#                                        # pair runs the same bits on each rank: two Zig builds of one tree are not
#                                        # byte-identical). Each box must already hold BASE with exactly BASE_ID; the
#                                        # Dockerfile's gate (TensorFold's server unit tests) fails a patch that regresses
#   ./build.sh tag                       # the tag, as the image matrix names it
set -euo pipefail
(( $# > 0 )) || { echo "usage: $0 <box> [<box> ...] | tag" >&2; exit 2; }
[[ $* != tag ]] || exec "$(dirname "$0")/../../images/build.sh" tensorfold native nvidia-sm121 tag
exec "$(dirname "$0")/../../images/build.sh" tensorfold native nvidia-sm121 remote "$@"
