#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_train_musicgen_melody_smoke.sh
# =============================================================================
# Smoke-train paired MusicGen Melody:
#
#   melody_piano.wav + prompt -> instrumental.wav
#
# Usage:
#   cd /workspace/musicgen-vast
#   bash vast/sh/run_train_musicgen_melody_smoke.sh
#
# Optional overrides:
#   VENV_DIR=/workspace/.venv-musicgen
#   MODEL=facebook/musicgen-melody-large
#   MAX_STEPS=100
#   TRAINABLE=output_linears
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-musicgen}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/datasets/instrumental_training}"
TRAIN_METADATA="${TRAIN_METADATA:-$DATASET_ROOT/splits/metadata_smoke_train.jsonl}"
VALID_METADATA="${VALID_METADATA:-$DATASET_ROOT/splits/metadata_smoke_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/musicgen_smoke}"

MODEL="${MODEL:-facebook/musicgen-melody-large}"
DEVICE="${DEVICE:-cuda}"
TRAINABLE="${TRAINABLE:-output_linears}"
MAX_STEPS="${MAX_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-1e-5}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[error] Missing venv: $VENV_DIR"
    echo "        Run: bash vast/sh/setup_musicgen.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python "$PROJECT_DIR/vast/scripts/train_musicgen_melody_paired.py" \
    --train-metadata "$TRAIN_METADATA" \
    --valid-metadata "$VALID_METADATA" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --trainable "$TRAINABLE" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --amp \
    "$@"
