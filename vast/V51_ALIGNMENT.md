# MusicGen V5.1 vocal/beat alignment

V5.1 keeps the clean instrumental as `target_audio`. It adds four timing
features from the source vocal and upgrades the original V5 rhythm conditioner
to a roughly 10-second, per-EnCodec-codebook conditioner.

The old V5 checkpoint is used as initialization. Its LM weights, temporal
front end, output bias, and global step are retained; the new context blocks
start as identity layers.

## 1. Prepare V5.1 metadata

```bash
cd /workspace/musicgen-vast

PY=/workspace/.venv-musicgen/bin/python
DATASET_ROOT=/workspace/musicgen-vast/vast_data/data/data/melody_instrumental_v1_base
STEMS_ROOT=/workspace/musicgen-vast/vast_data/data/data/stems/stems_instrumental_resurrection
V5_META="$DATASET_ROOT/metadata_instrumental_v5_beatthis_target.jsonl"
V51_META="$DATASET_ROOT/metadata_instrumental_v51_alignment.jsonl"

"$PY" vast/scripts/prepare_musicgen_v51_alignment_dataset.py \
  --metadata "$V5_META" \
  --dataset-root "$DATASET_ROOT" \
  --output "$V51_META" \
  --condition-root "$DATASET_ROOT/v51_vocal_timing" \
  --timing-audio-field source_vocal \
  --stems-root "$STEMS_ROOT" \
  --vocal-stem-name vocal \
  --chunk-hop-seconds 30 \
  --feature-rate 50 \
  --overwrite \
  --keep-going
```

Check that all rows were written before training:

```bash
wc -l "$V5_META" "$V51_META"
grep -c '"status": "error"' "$V51_META.report.jsonl"
```

If the original stem file is named `vocals.wav`, use
`--vocal-stem-name vocals`. The resolver also checks both common names.

## 2. Create leakage-safe splits

```bash
SPLIT_DIR="$DATASET_ROOT/splits_v51_alignment"

"$PY" vast/scripts/split_musicgen_v5_dataset.py \
  --input "$V51_META" \
  --output-dir "$SPLIT_DIR" \
  --train-ratio 0.95 \
  --seed 1337 \
  --smoke-train-rows 8 \
  --smoke-valid-rows 4 \
  --overwrite
```

## 3. Two-step preflight upgrade

Use the same `--init-from`, model, trainable mode, last-layer count, and hidden
dimension as the V5 checkpoint being upgraded.

```bash
OLD_V5=/workspace/musicgen-vast/vast_data/checkpoints/musicgen_v5_pretrained_l20_full_r1/checkpoint_last.pt
PREFLIGHT=/workspace/musicgen-vast/vast_data/checkpoints/musicgen_v51_alignment_preflight
mkdir -p "$PREFLIGHT"

"$PY" vast/scripts/train_musicgen_v5_beatthis.py \
  --train-metadata "$SPLIT_DIR/metadata_smoke_train.jsonl" \
  --valid-metadata "$SPLIT_DIR/metadata_smoke_valid.jsonl" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$PREFLIGHT" \
  --init-from pretrained \
  --resume-v5-checkpoint "$OLD_V5" \
  --conditioner-architecture alignment_v1 \
  --model facebook/musicgen-melody-large \
  --trainable last_layers \
  --last-n-layers 20 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --epochs 1 \
  --max-steps 2 \
  --lr 2e-7 \
  --rhythm-lr 3e-5 \
  --rhythm-hidden-dim 256 \
  --rhythm-dropout 0.1 \
  --num-workers 2 \
  --log-every 1 \
  --valid-every 1 \
  --max-valid-batches 1 \
  --save-every 2 \
  --seed 1337 \
  --amp
```

The log must contain:

```text
Upgraded legacy V5 conditioner to V5.1
Conditioner architecture: alignment_v1
```

## 4. Continue for 4,000 additional steps

```bash
OUT_DIR=/workspace/musicgen-vast/vast_data/checkpoints/musicgen_v51_alignment_r1
mkdir -p "$OUT_DIR"

"$PY" vast/scripts/train_musicgen_v5_beatthis.py \
  --train-metadata "$SPLIT_DIR/metadata_train.jsonl" \
  --valid-metadata "$SPLIT_DIR/metadata_valid.jsonl" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUT_DIR" \
  --init-from pretrained \
  --resume-v5-checkpoint "$OLD_V5" \
  --conditioner-architecture alignment_v1 \
  --model facebook/musicgen-melody-large \
  --trainable last_layers \
  --last-n-layers 20 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --epochs 6 \
  --max-steps 4000 \
  --lr 2e-7 \
  --rhythm-lr 3e-5 \
  --rhythm-hidden-dim 256 \
  --rhythm-dropout 0.1 \
  --num-workers 4 \
  --log-every 10 \
  --valid-every 200 \
  --max-valid-batches 10 \
  --save-every 1000 \
  --seed 1337 \
  --amp 2>&1 | tee "$OUT_DIR/train.log"
```

`--max-steps 4000` means 4,000 additional micro-batch steps in this trainer;
the saved global step continues from the old checkpoint.

## 5. Generate and measure alignment

The inference pipeline now uses a dedicated HyperACE V2 vocals checkpoint by
default. Start with report-only mode so audio is not automatically warped:

```bash
"$PY" vast/scripts/generate_musicgen_v5_beatthis_pipeline.py \
  --input-audio "$INPUT" \
  --checkpoint "$OUT_DIR/checkpoint_last.pt" \
  --output-dir "$OUT" \
  --beat-alignment report
```

Read `beat_alignment_report.json`:

- `constant_offset`: rerun in a new output folder with `--beat-alignment shift`.
- `tempo_drift`: test `--beat-alignment affine`; this requires `librosa`.
- `window_boundary_jump`: do not warp or train more yet; inspect the 18-second
  long-generation window handling.
- `local_rhythm_mismatch`: the new vocal timing conditioner needs more
  training or stronger conditioning.

The uncorrected generated file is retained under `work/generated_background_raw.wav`
when `shift` or `affine` correction is requested.
