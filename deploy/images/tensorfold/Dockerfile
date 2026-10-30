# The fleet's TensorFold image: the mia-ai recipe's image (BASE: TensorFold zig-flashnext + the recipe's patch set,
# built and published by MiaAI-Lab), with tensorfold-native rebuilt from the SAME tree plus the fleet's patches.
# Everything else in BASE (the Triton kernel set, the Python tree the vision helper runs, the CUDA/torch userland) is
# kept as published. Every pin (the base and its id, the tree, the Zig) is the accelerator's, in
# <build>.<accelerator>.env beside this file — none is defaulted here. Build with deploy/images/build.sh tensorfold
# <build> <accelerator> remote <build-box> [<box> ...] (deploy/tensorfold/image/build.sh pins the pair's cell).
#
# Applied after the recipe's patches (each made against the tree the ones before it leave):
#   0013-anthropic-merge-late-system  /v1/messages answered every Claude Code turn 400 "the model's chat template does
#                                     not support mid-conversation system messages": Claude Code sends its reminders
#                                     as a trailing `role: system` message and Qwen3.8's template renders no later
#                                     system message. On such a template every system text now joins the leading
#                                     system message ("\n", inline texts stripped), as SGLang's and vLLM's Anthropic
#                                     adapters do (`_merge_inline_system`); a template that renders late system text
#                                     keeps it in place as before (2026-10-29)
#   0014-work-counters                /metrics published no pass, computed-prompt or busy-time figure, so the panel's
#                                     COMPUTE and PREFILL tiles had nothing to draw from (operator 2026-10-29: "Explain
#                                     why other stats are still empty"). The lane core's backend is wrapped in a counter
#                                     (core/lanes/counting.zig: decode forwards, rows verified, the last forward's
#                                     streams and context, seconds inside the backend); the Flash-Next engine counts its
#                                     committed prompt chunks and their rows (cuda_engine `notePromptChunk`); /metrics
#                                     renders them (decode_passes_total, prompt_passes_total, prompt_tokens_computed_total,
#                                     prompt_tokens_cached_total, verify_rows_total, decode_streams, decode_context_tokens,
#                                     verify_width, busy_seconds_total, request_prefill_computed_tokens_total)
ARG BASE
FROM ${BASE:?pass--build-arg-BASE=the-mia-ai-image} AS build
# Zig's tarball for the build platform (ZIG_PLATFORM: aarch64-linux on an arm64 host), held to its checksum.
ARG ZIG_VERSION
ARG ZIG_PLATFORM
ARG ZIG_SHA256
RUN : "${ZIG_VERSION:?}" "${ZIG_PLATFORM:?}" "${ZIG_SHA256:?}" && \
    curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-${ZIG_PLATFORM}-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
ARG TF_REPO
ARG TF_REF
ARG RECIPE_REPO
ARG RECIPE_REF
RUN : "${TF_REPO:?}" "${TF_REF:?}" "${RECIPE_REPO:?}" "${RECIPE_REF:?}" && \
    git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "${TF_REF}" && \
    git init -q /recipe && cd /recipe && git remote add origin "${RECIPE_REPO}" && \
    git fetch -q origin && git checkout -q --detach "${RECIPE_REF}" && \
    cd /tensorfold && for p in /recipe/patches/*.patch; do git apply "$p" || exit 1; done
COPY *.patch /opt/fleet-patches/
RUN cd /tensorfold && for p in /opt/fleet-patches/*.patch; do git apply --verbose "$p" || exit 1; done
# Build gate: TensorFold's own server unit tests (the fleet patches' among them), as zig/tests/server/zig_test.sh runs them,
# and the lane core's (0014's counter among them).
RUN cd /tensorfold && ZIG=/opt/zig/zig bash zig/tests/server/zig_test.sh && \
    /opt/zig/zig test -lc zig/src/core/lanes/lanes.zig
# The recipe's own build line, into a fresh prefix.
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tf-fleet \
      -j"$(nproc)" fatbins install native && \
    test -x /opt/tf-fleet/native/bin/tensorfold-native

FROM ${BASE}
COPY --from=build /opt/tf-fleet/native /opt/tensorfold/native
COPY --from=build /opt/tf-fleet/fatbin /opt/tensorfold/fatbin
COPY --from=build /opt/fleet-patches /opt/fleet-patches
ARG FLEET_PATCHES
LABEL tf.fleet.patches=${FLEET_PATCHES}
