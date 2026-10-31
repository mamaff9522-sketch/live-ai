# Live AI — one container: llama.cpp (llama-server, CPU) + the Live AI web app.
# Works on a Hugging Face Space (Docker, free "CPU basic": 2 vCPU, 16 GB) or any Linux VPS.
# The model (Qwen3.5-2B Q4_K_M, ~1.3 GB) is downloaded on first start and checked by SHA-256.

ARG LLAMA_TAG=master

FROM debian:bookworm-slim AS build
ARG LLAMA_TAG
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /src
# Portable build for any x86-64 server CPU with AVX2 (all current cloud CPUs, incl. Hugging Face Spaces)
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
      -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
    && cmake --build /src/build --target llama-server -j"$(nproc)" \
    && mkdir -p /out && cp -a /src/build/bin/llama-server /src/build/bin/*.so* /out/ && /out/llama-server --version

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates ffmpeg fonts-tlwg-loma-otf \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
COPY --from=build /out/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama \
    LIVEAI_LLAMA_SERVER=/opt/llama/llama-server \
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
