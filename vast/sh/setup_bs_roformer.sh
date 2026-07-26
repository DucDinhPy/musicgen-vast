#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/setup_bs_roformer.sh  --  Run this ON a fresh Vast.ai instance
# =============================================================================
# Scope:
#   Setup only BS-RoFormer source separation.
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/setup_bs_roformer.sh
#
# Optional overrides:
#   WORK=/workspace
#   VENV_DIR=/workspace/.venv-bs-roformer
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   BS_ROFORMER_MODEL=roformer-model-bs-roformer-sw-by-jarredou
#   PRELOAD_MODEL=true
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-bs-roformer}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
BS_ROFORMER_MODEL="${BS_ROFORMER_MODEL:-roformer-model-bs-roformer-sw-by-jarredou}"
PRELOAD_MODEL="${PRELOAD_MODEL:-true}"

log() {
    printf '%s\n' "$*"
}

error_exit() {
    printf '[error] %s\n' "$*" >&2
    exit 1
}

trap 'error_exit "Setup failed at line $LINENO."' ERR

log "==> [1/5] Runtime paths"
log "    WORK:              $WORK"
log "    PROJECT_DIR:       $PROJECT_DIR"
log "    VENV_DIR:          $VENV_DIR"
log "    TORCH_INDEX_URL:   $TORCH_INDEX_URL"
log "    BS_ROFORMER_MODEL: $BS_ROFORMER_MODEL"
log "    PRELOAD_MODEL:     $PRELOAD_MODEL"

mkdir -p "$WORK"

log "==> [2/5] Install system dependencies"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    python3 \
    python3-pip \
    python3-venv \
    wget

log "==> [3/5] Create Python virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

log "==> [4/5] Install BS-RoFormer stack"
python -m pip install --upgrade --force-reinstall \
    torch torchaudio \
    --index-url "$TORCH_INDEX_URL"

python -m pip install --force-reinstall "numpy==$NUMPY_VERSION"
python -m pip install --upgrade \
    bs-roformer-infer \
    soundfile \
    tqdm

log "==> [5/5] Verify installation"
python - <<'PY'
import os
import shutil

import numpy as np
import torch

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

print("bs-roformer-infer:", shutil.which("bs-roformer-infer"))
print("bs-roformer-download:", shutil.which("bs-roformer-download"))

from bs_roformer import DEFAULT_MODEL
print("default model:", DEFAULT_MODEL)

if os.environ.get("PRELOAD_MODEL", "true").lower() == "true":
    import subprocess

    model = os.environ.get("BS_ROFORMER_MODEL", DEFAULT_MODEL)
    print(f"pre-downloading model: {model}")
    subprocess.run(["bs-roformer-download", "--model", model], check=True)
    print("BS-RoFormer model download ok")
PY

log ""
log "[done] BS-RoFormer setup completed."
log ""
log "Activate env:"
log "  source $VENV_DIR/bin/activate"
log ""
log "Run a quick check:"
log "  python $PROJECT_DIR/vast/scripts/bs_roformer.py check"
