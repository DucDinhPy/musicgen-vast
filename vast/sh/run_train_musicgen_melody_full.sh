#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# vast/sh/run_train_musicgen_melody_full.sh
# =============================================================================
# Full paired fine-tune for RTX 8000-class GPUs.
#
# Objective:
#   melody_piano.wav + prompt -> instrumental.wav
#
# Default config is conservative for ~45GB VRAM:
#   model: facebook/musicgen-melody-large
#   trainable: last_layers
#   last_n_layers: 2
#   batch_size: 1
#   grad_accum_steps: 8
# =============================================================================

WORK="${WORK:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$WORK/.venv-musicgen}"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/datasets/instrumental_training}"
TRAIN_METADATA="${TRAIN_METADATA:-$DATASET_ROOT/splits/metadata_train.jsonl}"
VALID_METADATA="${VALID_METADATA:-$DATASET_ROOT/splits/metadata_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/musicgen_full}"

MODEL="${MODEL:-facebook/musicgen-melody-large}"
DEVICE="${DEVICE:-cuda}"
TRAINABLE="${TRAINABLE:-last_layers}"
LAST_N_LAYERS="${LAST_N_LAYERS:-2}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-6}"
SAVE_EVERY="${SAVE_EVERY:-250}"
VALID_EVERY="${VALID_EVERY:-100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[error] Missing venv: $VENV_DIR"
    echo "        Run: bash vast/sh/setup_musicgen.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

EXTRA_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python "$PROJECT_DIR/vast/scripts/train_musicgen_melody_paired.py" \
    --train-metadata "$TRAIN_METADATA" \
    --valid-metadata "$VALID_METADATA" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --trainable "$TRAINABLE" \
    --last-n-layers "$LAST_N_LAYERS" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --valid-every "$VALID_EVERY" \
    --save-every "$SAVE_EVERY" \
    --amp \
    "${EXTRA_ARGS[@]}" \
    "$@"
