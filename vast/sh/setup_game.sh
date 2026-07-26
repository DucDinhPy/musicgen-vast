#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/setup_game.sh  --  Setup OpenVPI GAME on a fresh Vast.ai instance
# =============================================================================
# Scope:
#   Setup only OpenVPI GAME for vocal melody extraction.
#
# What it does:
#   1. Installs system dependencies.
#   2. Clones or updates https://github.com/openvpi/GAME.
#   3. Creates a clean Python venv.
#   4. Installs PyTorch and GAME requirements.
#   5. Creates a model folder for your GAME checkpoint.
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/setup_game.sh
#
# Optional overrides:
#   WORK=/workspace
#   VENV_DIR=/workspace/.venv-game
#   GAME_DIR=/workspace/GAME
#   MODELS_DIR=/workspace/models/game
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
#   MODEL_URL=https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-large.zip
# =============================================================================

WORK="${WORK:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-game}"
GAME_DIR="${GAME_DIR:-$WORK/GAME}"
GAME_REPO="${GAME_REPO:-https://github.com/openvpi/GAME.git}"
MODELS_DIR="${MODELS_DIR:-$WORK/models/game}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
MODEL_URL="${MODEL_URL:-}"
MODEL_NAME="${MODEL_NAME:-game.pt}"

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
log "    VENV_DIR:        $VENV_DIR"
log "    GAME_DIR:        $GAME_DIR"
log "    MODELS_DIR:      $MODELS_DIR"
log "    TORCH_INDEX_URL: $TORCH_INDEX_URL"
log "    MODEL_URL:       ${MODEL_URL:-'(none)'}"

mkdir -p "$WORK" "$MODELS_DIR"


log "==> [2/6] Install system dependencies"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    wget


log "==> [3/6] Clone or update OpenVPI GAME"
if [ ! -d "$GAME_DIR" ]; then
    git clone "$GAME_REPO" "$GAME_DIR"
elif [ -d "$GAME_DIR/.git" ]; then
    git -C "$GAME_DIR" pull --ff-only
else
    error_exit "$GAME_DIR exists but is not a git repository."
fi


log "==> [4/6] Create Python virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel


log "==> [5/6] Install PyTorch and GAME requirements"
python -m pip install --upgrade --force-reinstall \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"

python -m pip install --upgrade -r "$GAME_DIR/requirements.txt"


log "==> [6/6] Optional model download and verification"
if [ -n "$MODEL_URL" ]; then
    if [[ "$MODEL_URL" == *.zip ]]; then
        ARCHIVE_PATH="$MODELS_DIR/$(basename "$MODEL_URL")"
        curl -L "$MODEL_URL" -o "$ARCHIVE_PATH"
        unzip -o "$ARCHIVE_PATH" -d "$MODELS_DIR"

        MODEL_FILE="$(
            python - "$MODELS_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
models = sorted(root.rglob("*.pt"))
if not models:
    raise SystemExit(f"No .pt model found under {root}")
print(models[0])
PY
        )"
        ln -sfn "$MODEL_FILE" "$MODELS_DIR/game.pt"
        log "Downloaded archive: $ARCHIVE_PATH"
        log "Selected model:     $MODEL_FILE"
        log "Symlink:            $MODELS_DIR/game.pt"
    else
        curl -L "$MODEL_URL" -o "$MODELS_DIR/$MODEL_NAME"
        ln -sfn "$MODELS_DIR/$MODEL_NAME" "$MODELS_DIR/game.pt"
        log "Downloaded model to: $MODELS_DIR/$MODEL_NAME"
        log "Symlink:             $MODELS_DIR/game.pt"
    fi
else
    log "No MODEL_URL provided."
    log "Download a GAME .pt checkpoint manually and place it under:"
    log "    $MODELS_DIR"
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY

python "$GAME_DIR/infer.py" extract --help >/dev/null

log ""
log "Done."
log "Activate with:"
log "    source $VENV_DIR/bin/activate"
log ""
log "Check paths with:"
log "    python /workspace/musicgen-vast/vast/scripts/game_melody.py check \\"
log "      --game-dir $GAME_DIR \\"
log "      --model-path $MODELS_DIR/$MODEL_NAME"
