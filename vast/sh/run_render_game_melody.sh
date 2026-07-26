#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_render_game_melody.sh  --  Render GAME MIDI to piano WAV
# =============================================================================
# This shell script is a thin runner around:
#   vast/scripts/render_game_melody.py
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/run_render_game_melody.sh --limit 5
#
# Default paths:
#   INPUT_DIR=/workspace/musicgen-vast/data/melody/game
#   OUTPUT_DIR=/workspace/musicgen-vast/data/melody/rendered
#
# Common forwarded options:
#   --limit N
#   --start-index N
#   --dry-run
#   --overwrite
#   --keep-raw
#   --keep-going
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/melody/game}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/melody/rendered}"

if ! command -v fluidsynth >/dev/null 2>&1; then
    echo "[error] Missing fluidsynth."
    echo "        Install with: apt-get install -y fluidsynth fluid-soundfont-gm"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[error] Missing ffmpeg."
    echo "        Install with: apt-get install -y ffmpeg"
    exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/vast/scripts/render_game_melody.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
