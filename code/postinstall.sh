#!/usr/bin/env bash
set -euo pipefail

# Keep the PyTorch packages on matching CUDA 12.4 versions.
python -m pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# SeaQuery inference, training and analysis dependencies.
python -m pip install --no-cache-dir \
    setuptools==75.6.0 \
    PyYAML==6.0.2 \
    numpy==2.2.6 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    Pillow==11.0.0 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    accelerate==1.1.1 \
    peft==0.13.2 \
    huggingface-hub==0.26.2 \
    safetensors==0.4.5 \
    einops==0.8.0 \
    timm==1.0.12 \
    sentencepiece==0.2.0 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    scikit-learn==1.5.2 \
    bert-score==0.3.13

python -m pip check