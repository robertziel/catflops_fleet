# syntax=docker/dockerfile:1
# deploy/standalone/images/faster-whisper-rocm.Dockerfile -- the faster-whisper node's image (EDGE_NODE_ENGINE=
# faster-whisper, node-engine.faster-whisper.yml) on an AMD ROCm card: Vexa's transcription worker
# (core/meetings/services/transcription) with CTranslate2's upstream ROCm wheel in place of the CPU wheel from PyPI.
# A copy of the Vexa deployment's recipe (2026-10-23, Vexa a01db95ec), kept here so this repo can build every image its
# fleet names.
#
# BUILD CONTEXT: a Vexa checkout's ROOT (the COPY lines below are paths inside it) -- the deployment's, given on the
# command line, never named here. BASE: the ROCm dev image for the card's ROCm (built 2026-10-23 on
# rocm/dev-ubuntu-24.04:7.2-complete for a gfx1151 host on ROCm 7.2) -- pin it by digest.
#   docker build -f deploy/standalone/images/faster-whisper-rocm.Dockerfile \
#       --build-arg BASE=rocm/dev-ubuntu-24.04@sha256:<digest> -t vexa-stt-rocm:<tag> "$VEXA_CHECKOUT"
# The built image's id (docker image inspect -f '{{.Id}}') is what the fleet's launch block names for engine
# faster-whisper.
# CT2_CUDA_ALLOCATOR: CTranslate2's default allocator on ROCm (hipMallocAsync) yields garbage transcripts from
# a freshly started process on gfx1151 (measured 2026-10-23: 0-2/6 correct); cub_caching gives 8/8 + 4/4 concurrent.
ARG BASE
FROM ${BASE:?pass--build-arg-BASE=the-ROCm-dev-image-pinned-by-digest}
ARG CT2_VERSION=4.8.3
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH \
    DEVICE=cuda \
    COMPUTE_TYPE=float16 \
    CT2_CUDA_ALLOCATOR=cub_caching

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libsndfile1 ffmpeg unzip python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages uv==0.9.22
RUN uv python install 3.12

WORKDIR /app
COPY core/meetings/services/transcription/pyproject.toml core/meetings/services/transcription/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

RUN curl -fsSL -o /tmp/ct2.zip \
        "https://github.com/OpenNMT/CTranslate2/releases/download/v${CT2_VERSION}/rocm-python-wheels-Linux.zip" \
    && mkdir /tmp/ct2 && unzip -q /tmp/ct2.zip -d /tmp/ct2 \
    && find /tmp/ct2 -name "*.whl" \
    && whl=$(find /tmp/ct2 -name "ctranslate2-${CT2_VERSION}-cp312-cp312-*x86_64*.whl" | head -1) \
    && test -n "$whl" \
    && uv pip install --python /opt/venv/bin/python --no-deps --reinstall "$whl" \
    && rm -rf /tmp/ct2 /tmp/ct2.zip

COPY core/meetings/services/transcription/src ./src
RUN mkdir -p /app/models

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["python", "-m", "transcription"]
