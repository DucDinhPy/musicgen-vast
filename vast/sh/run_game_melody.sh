#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_game_melody.sh  --  Run OpenVPI GAME melody extraction on Vast.ai
# =============================================================================
# This shell script is a thin runner around:
#   vast/scripts/game_melody.py extract
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/run_game_melody.sh --limit 5
#
# Default paths:
#   STEMS_DIR=/workspace/musicgen-vast/data/stems/stems_set_single
#   OUTPUT_DIR=/workspace/musicgen-vast/data/melody/game
#   GAME_DIR=/workspace/GAME
#   MODEL_PATH=/workspace/models/game/game.pt
#
# Common forwarded options:
#   --limit N
#   --start-index N
#   --dry-run
#   --overwrite
#   --keep-going
#   --game-extra-args "..."
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-game}"

STEMS_DIR="${STEMS_DIR:-$PROJECT_DIR/data/stems/stems_set_single}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/melody/game}"
GAME_DIR="${GAME_DIR:-$WORK/GAME}"
MODEL_PATH="${MODEL_PATH:-$WORK/models/game/game.pt}"
INPUT_STEM="${INPUT_STEM:-vocals}"
OUTPUT_FORMATS="${OUTPUT_FORMATS:-mid,txt,csv}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[error] Missing venv: $VENV_DIR"
    echo "        Run: bash vast/sh/setup_game.sh"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[error] Missing GAME model: $MODEL_PATH"
    echo "        Download a GAME .pt checkpoint and set MODEL_PATH if needed."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python "$PROJECT_DIR/vast/scripts/game_melody.py" extract \
    --stems-dir "$STEMS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --game-dir "$GAME_DIR" \
    --model-path "$MODEL_PATH" \
    --input-stem "$INPUT_STEM" \
    --output-formats "$OUTPUT_FORMATS" \
    "$@"
