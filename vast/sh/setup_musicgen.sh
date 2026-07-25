#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/setup_musicgen.sh  --  Run this ON a fresh Vast.ai instance
# =============================================================================
# Scope:
#   Setup only the MusicGen/AudioCraft model environment.
#
# What it does:
#   1. Installs system dependencies needed by AudioCraft/PyAV/soundfile.
#   2. Creates a clean Python venv.
#   3. Installs AudioCraft.
#   4. Reinstalls a modern CUDA PyTorch build for newer GPUs.
#   5. Pins NumPy 1.26.x for PyTorch/AudioCraft compatibility.
#   6. Installs xformers without letting it downgrade torch.
#   7. Verifies imports and optionally preloads a MusicGen checkpoint.
#
# Usage:
#   cd /workspace/musicgen
#   bash vast/sh/setup_musicgen.sh
#
# Optional overrides:
#   WORK=/workspace
#   VENV_DIR=/workspace/.venv-musicgen
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   HF_HOME=/workspace/hf_cache
#   MUSICGEN_MODEL=facebook/musicgen-melody
#   PRELOAD_MODEL=true
# =============================================================================

# ============================ CONFIG =========================================
WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-musicgen}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
SCIPY_VERSION="${SCIPY_VERSION:-1.12.0}"
AUDIOCRAFT_VERSION="${AUDIOCRAFT_VERSION:-1.3.0}"

HF_HOME="${HF_HOME:-$WORK/hf_cache}"
MUSICGEN_MODEL="${MUSICGEN_MODEL:-facebook/musicgen-melody-large}"
PRELOAD_MODEL="${PRELOAD_MODEL:-true}"
# =============================================================================

log() {
    printf '%s\n' "$*"
}

error_exit() {
    printf '[error] %s\n' "$*" >&2
    exit 1
}

trap 'error_exit "Setup failed at line $LINENO."' ERR


log "==> [1/6] Runtime paths"
log "    WORK:            $WORK"
log "    PROJECT_DIR:     $PROJECT_DIR"
log "    VENV_DIR:        $VENV_DIR"
log "    TORCH_INDEX_URL: $TORCH_INDEX_URL"
log "    NUMPY_VERSION:   $NUMPY_VERSION"
log "    SCIPY_VERSION:   $SCIPY_VERSION"
log "    HF_HOME:         $HF_HOME"
log "    MUSICGEN_MODEL:  $MUSICGEN_MODEL"
log "    PRELOAD_MODEL:   $PRELOAD_MODEL"

mkdir -p "$WORK" "$HF_HOME"


log "==> [2/6] Install system dependencies"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libsndfile1 \
    libswresample-dev \
    libswscale-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    wget


log "==> [3/6] Create Python virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel


log "==> [4/6] Install MusicGen / AudioCraft Python stack"
# Install modern torch first. AudioCraft 1.3.0 metadata pins torch 2.1.0, but
# that wheel is not available for newer Python versions and is too old for some
# fresh Vast/Blackwell instances.
python -m pip install --upgrade --force-reinstall \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"

# Install AudioCraft without dependency resolution so it cannot force torch 2.1.
python -m pip install --no-deps "audiocraft==$AUDIOCRAFT_VERSION"

# Runtime deps needed for MusicGen inference. Keep torch-family packages managed
# above; do not let old AudioCraft metadata downgrade them.
python -m pip install --upgrade \
    av==11.0.0 \
    accelerate \
    demucs \
    einops \
    encodec \
    flashy \
    huggingface_hub \
    hydra-colorlog \
    hydra-core \
    julius \
    librosa \
    num2words \
    omegaconf \
    safetensors \
    "scipy==$SCIPY_VERSION" \
    sentencepiece \
    soundfile \
    spacy \
    submitit \
    torchmetrics \
    tqdm \
    transformers \
    treetable

# AudioCraft imports xformers directly. Do not let xformers downgrade torch.
python -m pip install --upgrade --no-deps xformers

# Pin NumPy/SciPy after all dependency installs, because some packages may
# upgrade them. Torch/AudioCraft can report "Numpy is not available" with
# NumPy 2.x, while newer SciPy builds may require NumPy 2.x.
python -m pip install --force-reinstall \
    "numpy==$NUMPY_VERSION" \
    "scipy==$SCIPY_VERSION"


log "==> [5/6] Persist Hugging Face cache environment"
grep -qxF "export HF_HOME=$HF_HOME" ~/.bashrc 2>/dev/null || \
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

export HF_HOME


log "==> [6/6] Verify MusicGen installation"
python - <<'PY'
import os

import numpy as np
import torch
from transformers.utils import is_torch_available

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers sees torch:", is_torch_available())

import xformers
print("xformers:", xformers.__version__)

from audiocraft.models import MusicGen
print("audiocraft MusicGen import ok")

if os.environ.get("PRELOAD_MODEL", "true").lower() == "true":
    model_name = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-melody")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"preloading MusicGen model: {model_name} on {device}")
    MusicGen.get_pretrained(model_name, device=device)
    print("MusicGen model preload ok")
PY


log ""
log "[done] MusicGen setup completed."
log ""
log "Activate env:"
log "  source $VENV_DIR/bin/activate"
log ""
log "Run a quick import check:"
log "  python - <<'PY'"
log "  from audiocraft.models import MusicGen"
log "  print('MusicGen ok')"
log "  PY"