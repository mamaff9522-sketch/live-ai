# Live AI — one container: llama.cpp (llama-server, CPU) + the Live AI web app.
# Works on any Linux machine with Docker, x86-64 or ARM64 (e.g. Oracle Cloud Always Free Ampere A1).
# The model (Qwen3.5-2B Q4_K_M, ~1.3 GB) is downloaded on first start and checked by SHA-256.

ARG LLAMA_TAG=master

FROM debian:bookworm-slim AS build
ARG LLAMA_TAG
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /src
# Portable build: every CPU variant is built and llama-server picks the best one for the machine it
# runs on at start-up (x86-64: SSE4 → AVX2 → AVX-512; ARM64: e.g. Oracle Ampere A1 with dot-product).
# (ARM64 uses one armv8.2 build: Debian 12's compiler cannot build the newest armv9 variants.)
RUN if [ "$(uname -m)" = "x86_64" ]; then CPU="-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON"; \
    else CPU="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; fi \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF $CPU \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF \
    && cmake --build /src/build -j"$(nproc)" \
    && mkdir -p /out && cp -a /src/build/bin/llama-server /src/build/bin/*.so* /out/ && /out/llama-server --version

FROM debian:bookworm-slim AS whisper
ARG WHISPER_TAG=master
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${WHISPER_TAG}" https://github.com/ggml-org/whisper.cpp /src
RUN if [ "$(uname -m)" = "x86_64" ]; then CPU="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"; \
    else CPU="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; fi \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF $CPU \
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
    && cmake --build /src/build --target whisper-cli -j"$(nproc)" && cp /src/build/bin/whisper-cli /whisper-cli

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates ffmpeg fonts-tlwg-loma-otf \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
COPY --from=build /out/ /opt/llama/
COPY --from=whisper /whisper-cli /opt/whisper/whisper-cli
ENV LD_LIBRARY_PATH=/opt/llama \
    LIVEAI_LLAMA_SERVER=/opt/llama/llama-server \
    LIVEAI_WHISPER_BIN=/opt/whisper/whisper-cli \
    LIVEAI_DATA_DIR=/data \
    LIVEAI_PORT=7860 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=user live_ai/ live_ai/
COPY --chown=user web/ web/
RUN mkdir -p /data && chown user /data
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/health',timeout=8)"
CMD ["python", "-m", "live_ai"]
