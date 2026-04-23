FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch/torchvision FIRST — иначе anomalib тянет CUDA-версию (~5–8 ГБ)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchvision==0.19.1+cpu

COPY pyproject.toml ./
COPY argus ./argus
COPY config.yaml ./config.yaml

RUN pip install --no-cache-dir .

RUN mkdir -p /data /models && chmod 777 /data /models

EXPOSE 8000

ENTRYPOINT ["argus"]
CMD ["--help"]
