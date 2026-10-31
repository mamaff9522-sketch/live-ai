# Live AI — one container: llama.cpp (llama-server, CPU) + the Live AI web app.
# Works on a Hugging Face Space (Docker, free "CPU basic": 2 vCPU, 16 GB) or any Linux VPS.
# The model (Qwen3.5-2B Q4_K_M, ~1.3 GB) is downloaded on first start and checked by SHA-256.

ARG LLAMA_TAG=master

FROM debian:bookworm-slim AS build
ARG LLAMA_TAG
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /src
# GGML_NATIVE=OFF + all CPU variants: the binary picks the best instructions of whatever CPU it lands on
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
    && cmake --build /src/build --target llama-server -j"$(nproc)" \
    && mkdir -p /out && cp /src/build/bin/llama-server /out/ && cp /src/build/bin/*.so /out/ 2>/dev/null || true

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
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
