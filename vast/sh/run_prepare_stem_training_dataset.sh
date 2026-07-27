#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_prepare_stem_training_dataset.sh
# =============================================================================
# Prepare chunked MusicGen training data:
#
#   melody_piano.wav + prompt -> target stem chunk
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/run_prepare_stem_training_dataset.sh --limit 5 --dry-run
#   bash vast/sh/run_prepare_stem_training_dataset.sh --limit 5
#
# Default paths:
#   MELODY_DIR=/workspace/musicgen-vast/data/melody/rendered
#   STEMS_DIR=/workspace/musicgen-vast/data/stems/stems_set_single
#   OUTPUT_DIR=/workspace/musicgen-vast/data/datasets/stem_training
#
# Common forwarded options:
#   --target-stems drums,bass,other
#   --chunk-seconds 30
#   --hop-seconds 30
#   --limit N
#   --dry-run
#   --overwrite
#   --keep-going
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MELODY_DIR="${MELODY_DIR:-$PROJECT_DIR/data/melody/rendered}"
STEMS_DIR="${STEMS_DIR:-$PROJECT_DIR/data/stems/stems_set_single}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/datasets/stem_training}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[error] Missing ffmpeg."
    echo "        Install with: apt-get install -y ffmpeg"
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "[error] Missing ffprobe."
    echo "        Install with: apt-get install -y ffmpeg"
    exit 1
fi

"$PYTHON_BIN" "$PROJECT_DIR/vast/scripts/prepare_stem_training_dataset.py" \
    --melody-dir "$MELODY_DIR" \
    --stems-dir "$STEMS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
