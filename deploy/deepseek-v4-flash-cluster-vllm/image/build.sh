#!/usr/bin/env bash
# The pair's DeepSeek-V4-Flash vLLM image: the image matrix cell vllm × deepseek-v4-b12x × nvidia-sm121, on the ONE
# build (deploy/images/build.sh; the recipe is deploy/images/vllm/Dockerfile stage deepseek-v4-b12x, its pin
# deepseek-v4-b12x.nvidia-sm121.env).
#   build.sh build | ship | check | tag     (ship/check: BOXES="<host> <host>", ssh aliases)
exec "$(dirname "$0")/../../images/build.sh" vllm deepseek-v4-b12x nvidia-sm121 "$@"
