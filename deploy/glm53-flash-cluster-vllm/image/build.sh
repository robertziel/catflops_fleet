#!/usr/bin/env bash
# The pair's GLM-5.3-Flash vLLM image: the image matrix cell vllm × glm53 × nvidia-sm121, on the ONE build
# (deploy/images/build.sh; the recipe is deploy/images/vllm/Dockerfile stage glm53, its pin glm53.nvidia-sm121.env).
#   build.sh build | ship | check | tag     (ship/check: BOXES="<host> <host>", ssh aliases)
exec "$(dirname "$0")/../../images/build.sh" vllm glm53 nvidia-sm121 "$@"
