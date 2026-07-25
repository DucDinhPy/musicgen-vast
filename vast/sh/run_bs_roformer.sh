#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_bs_roformer.sh  --  Run BS-RoFormer separation on Vast.ai
# =============================================================================
# This shell script is a thin runner around:
#   vast/scripts/bs_roformer.py separate
#
# Usage:
#   cd /workspace/musicgen
#   bash vast/sh/run_bs_roformer.sh \
#     --input-dir /workspace/data/raw_audio \
#     --output-dir /workspace/outputs/stems \
#     --device cuda \
#     --convert-to-wav
#
# Common options forwarded to Python:
#   --input-dir PATH
#   --output-dir PATH
#   --device auto|cpu|cuda|cuda:0
#   --model roformer-model-bs-roformer-sw-by-jarredou
#   --models-dir PATH
#   --convert-to-wav
#   --extensions .wav,.mp3,.flac,.m4a,.ogg
#   --limit N
#   --start-index N
#   --dry-run
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-bs-roformer}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[error] Missing venv: $VENV_DIR"
    echo "        Run: bash vast/sh/setup_bs_roformer.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python "$PROJECT_DIR/vast/scripts/bs_roformer.py" separate "$@"
