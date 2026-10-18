FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -U pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# COMET-Kiwi and LaBSE checkpoints are pulled from the HF Hub on first use and cached
# under HF_HOME (mount a volume there to avoid re-downloading per container). No
# entrypoint is fixed -- run `python3 main.py ...` or `python3 generate_translations.py ...`.
