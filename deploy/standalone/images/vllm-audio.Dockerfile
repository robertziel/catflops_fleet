# deploy/standalone/images/vllm-audio.Dockerfile -- a vLLM image that can serve SPEECH-TO-TEXT (POST
# /v1/audio/transcriptions): the fleet's pinned vLLM build plus the audio decoders vLLM imports lazily.
#
# WHY: vLLM's OpenAI server decodes an uploaded file with librosa/soundfile (and PyAV for compressed formats); the
# stock image ships none of them, so every upload is answered 400 "Invalid or unsupported audio file" / "Please install
# vllm[audio]" (measured 2026-10-23 on rocm/vllm 0.19.1+rocm7.13.0rc2). They must be importable when the server STARTS:
# installed into a running container they are never picked up. Versions pinned: what pip resolved on 2026-10-23 for
# this base, so a rebuild is the same image.
#
# BASE is the vLLM image this one extends, pinned by digest -- the deployment's choice (its card's build), passed at
# build time, never named here:
#   docker build -f deploy/standalone/images/vllm-audio.Dockerfile --build-arg BASE=<repo>@sha256:<digest> \
#       -t vllm-audio:<tag> deploy/standalone/images
# The built image's id (docker image inspect -f '{{.Id}}') is what the fleet's launch block names for engine vllm.
ARG BASE
FROM ${BASE:?pass--build-arg-BASE=the-vLLM-image-pinned-by-digest}
RUN pip install --no-cache-dir librosa==1.0.0 soundfile==0.14.0 av==19.0.2 \
    && python3 -c "import librosa, soundfile, av"
