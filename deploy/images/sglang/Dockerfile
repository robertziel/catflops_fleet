# The fleet's SGLang image: upstream, pinned by digest (BASE), plus the fixes this fleet needs (PATCHES) — both the
# accelerator's, in <build>.<accelerator>.env beside this file, never defaulted here. One Dockerfile for every
# accelerator: a cell with no pin file has no known base, and deploy/images/build.sh refuses it naming the file.
#   deploy/images/build.sh sglang <build> <accelerator> remote <build-box> [<box> ...]
# builds it ON a box of the pin's platform (the RUN steps below run there) and copies it to the rest by id: a TP=2 pair
# needs the same image on each rank (deploy/sglang-27b-tp2/image/build.sh pins the pair's cell, on BOXES). A one-host
# node (deploy/standalone/node-engine.sglang.yml) on the box that runs it, with an upstream image for its platform and
# the patches that apply there — the build gate below fails a patch that does not:
#   docker build -t <tag> --build-arg BASE=lmsysorg/sglang@sha256:<digest> --build-arg PATCHES="<patch> ..." deploy/images/sglang
# The built image's id (docker image inspect -f '{{.Id}}') is what the fleet's launch block names for engine sglang.
ARG BASE
FROM ${BASE:?pass--build-arg-BASE=an-upstream-SGLang-image-pinned-by-digest}

# Applied in this order (each patch is made against the tree the ones before it leave):
#   anthropic-tool-order        /v1/messages dropped a tool call's closing "}" when one decode step
#                               ended the call and began the text after it
#   qwen3-coder-truncated-call  a string argument streams while it is written (no minutes of
#                               silence), and a call cut by max_tokens is never sent as a whole `{}`
#   anthropic-thinking-display  Claude Code's `thinking.display` values ("updates" by default,
#                               "highlights") were a 400 before the adapter ran
#   qwen4-exp-mtp-quantized     _mtp_quant_config disabled quantization for every modelopt_mixed
#                               checkpoint, assuming the drafter ships BF16. Flash-Next's does not -
#                               its own config names mtp.*.experts as NVFP4 - so the drafter was built
#                               at full width against a TP shard and no speculative start could
#                               initialise (2560 vs 1280, generations 149/160/163)
#   anthropic-effort-xhigh      the /v1/messages adapter collapsed output_config.effort "xhigh" to
#                               the OpenAI word "max"; qwen3.8's chat template accepts
#                               low/medium/xhigh and raises on "max", so every high-effort turn 500'd
#                               at template render (both client "xhigh" and "max"-after-the-LB's-
#                               rewrite land here). The OpenAI Literal now carries xhigh, so the word
#                               passes through verbatim (2026-10-08)
#   qsa-fp8-kv                  Flash-Next's SM121 QSA decode kernel and its chunk-prefill kernel read
#                               BF16 K/V only, so --kv-cache-dtype fp8_e4m3 died "unsupported SM121 QSA
#                               call: expected BF16" (generation 153). The rows gathered from an FP8 pool
#                               (<= 2,055 a request) are widened to BF16, the layer's KV scale multiplied
#                               back; a BF16 pool is untouched (2026-10-26, -fleet.4)
ARG PATCHES
COPY *.patch fleet_harness.py test_*.py /opt/fleet-patches/
RUN : "${PATCHES:?pass --build-arg PATCHES=the patches to apply, in order}" && cd /sgl-workspace/sglang && \
    for p in $PATCHES; do git apply --verbose "/opt/fleet-patches/$p.patch" || exit 1; done

# Build gate: the APPLIED fixes' own tests (test_<patch, - as _>.py, where one exists) and upstream's suites for every
# file they touch.
RUN cd /tmp && tests="" && for p in $PATCHES; do t="/opt/fleet-patches/test_$(echo "$p" | tr - _).py"; \
        if [ -f "$t" ]; then tests="$tests $t"; fi; done && \
    python3 -m pytest -q -p no:cacheprovider $tests \
        /sgl-workspace/sglang/test/registered/unit/entrypoints/anthropic/test_serving.py \
        /sgl-workspace/sglang/test/registered/unit/entrypoints/openai/test_serving_chat.py \
        /sgl-workspace/sglang/test/registered/unit/function_call/test_function_call_parser.py
