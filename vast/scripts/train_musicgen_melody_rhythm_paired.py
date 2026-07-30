#!/usr/bin/env python3
"""Experimental V4 trainer: melody + rhythm grid -> instrumental.

This intentionally leaves the stable V1 trainer untouched. It reuses MusicGen
Melody conditioning and adds a small rhythm-conditioned logit bias trained from
per-chunk `.npz` files produced by `prepare_rhythm_conditions.py`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_musicgen_melody_paired import (
    _cast_trainable_params,
    _freeze,
    _load_wav,
    _masked_cross_entropy,
    _pad_or_trim,
    _read_jsonl,
    _resolve_path,
    _set_trainable,
)


class RhythmLogitBias(nn.Module):
    """Project rhythm features to a vocabulary bias for each generated frame."""

    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, rhythm: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise RuntimeError(f"Expected logits [B,K,T,V], got {tuple(logits.shape)}")
        time_steps = logits.shape[2]
        rhythm = _fit_rhythm_length(rhythm, time_steps)
        bias = self.net(rhythm).unsqueeze(1)
        return logits + bias


class RhythmPairDataset(Dataset):
    def __init__(self, rows: list[dict], dataset_root: Path):
        self.rows = rows
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if "rhythm_condition" not in row:
            raise KeyError("Metadata row has no rhythm_condition. Run prepare_rhythm_conditions.py first.")
        return {
            "input_audio": _resolve_path(row["input_audio"], self.dataset_root),
            "target_audio": _resolve_path(row["target_audio"], self.dataset_root),
            "rhythm_condition": _resolve_path(row["rhythm_condition"], self.dataset_root),
            "text": row["text"],
            "track_id": row.get("track_id", ""),
            "chunk_index": row.get("chunk_index", -1),
        }


def train(args: argparse.Namespace) -> None:
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    model.compression_model.eval()
    model.lm.train()

    _freeze(model.compression_model)
    trainable_names = _set_trainable(model.lm, args.trainable, args.last_n_layers)
    if args.trainable_dtype == "float32":
        _cast_trainable_params(model.lm, torch.float32)

    vocab_size = _infer_vocab_size(model)
    rhythm_bias = RhythmLogitBias(
        input_dim=args.rhythm_input_dim,
        hidden_dim=args.rhythm_hidden_dim,
        vocab_size=vocab_size,
    ).to(device)

    if args.resume_checkpoint is not None:
        _load_rhythm_checkpoint(model.lm, rhythm_bias, args.resume_checkpoint, device)

    trainable_params = [p for p in model.lm.parameters() if p.requires_grad]
    trainable_params += list(rhythm_bias.parameters())
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected.")

    print(f"Model sample rate:      {model.sample_rate}")
    print(f"Model channels:         {model.audio_channels}")
    print(f"LM trainable mode:      {args.trainable}")
    print(f"LM trainable tensors:   {len(trainable_names)}")
    print(f"Rhythm input dim:       {args.rhythm_input_dim}")
    print(f"Rhythm hidden dim:      {args.rhythm_hidden_dim}")
    print(f"Vocab size:             {vocab_size}")
    print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}")

    train_rows = _read_jsonl(args.train_metadata)
    valid_rows = _read_jsonl(args.valid_metadata) if args.valid_metadata else []
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_valid_rows is not None:
        valid_rows = valid_rows[: args.max_valid_rows]

    train_loader = _loader(train_rows, args, model, convert_audio, shuffle=True)
    valid_loader = _loader(valid_rows, args, model, convert_audio, shuffle=False) if valid_rows else None

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    use_grad_scaler = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            global_step += 1
            loss = _training_step(model, rhythm_bias, batch, device, scaler, args.amp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")
            loss_for_backward = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if global_step % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(f"[train] epoch={epoch + 1} step={global_step} loss={loss.item():.4f} elapsed={elapsed:.1f}s")

            if valid_loader is not None and global_step % args.valid_every == 0:
                valid_loss = _evaluate(model, rhythm_bias, valid_loader, device, args.amp, args.max_valid_batches)
                print(f"[valid] step={global_step} loss={valid_loss:.4f}")

            if global_step % args.save_every == 0:
                _save_checkpoint(
                    args.output_dir / f"checkpoint_step_{global_step}.pt",
                    model=model,
                    rhythm_bias=rhythm_bias,
                    model_name=args.model,
                    trainable_names=trainable_names,
                    global_step=global_step,
                    args=args,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    _save_checkpoint(
        args.output_dir / "checkpoint_last.pt",
        model=model,
        rhythm_bias=rhythm_bias,
        model_name=args.model,
        trainable_names=trainable_names,
        global_step=global_step,
        args=args,
    )
    print("Done.")
    print(f"Last checkpoint: {args.output_dir / 'checkpoint_last.pt'}")


def _loader(rows, args, model, convert_audio, shuffle: bool) -> DataLoader:
    return DataLoader(
        RhythmPairDataset(rows, args.dataset_root),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda batch: _collate(
            batch=batch,
            sample_rate=model.sample_rate,
            channels=model.audio_channels,
            audio_duration=args.audio_duration,
            rhythm_input_dim=args.rhythm_input_dim,
            feature_rate=args.rhythm_feature_rate,
            convert_audio_fn=convert_audio,
        ),
    )


def _collate(
    batch: list[dict],
    sample_rate: int,
    channels: int,
    audio_duration: float,
    rhythm_input_dim: int,
    feature_rate: float,
    convert_audio_fn,
) -> dict:
    import numpy as np
    import soundfile as sf

    melodies = []
    targets = []
    rhythms = []
    texts = []
    meta = []
    target_frames = int(audio_duration * sample_rate)
    rhythm_frames = int(audio_duration * feature_rate)

    for item in batch:
        melody, melody_sr = _load_wav(item["input_audio"], sf)
        target, target_sr = _load_wav(item["target_audio"], sf)
        melody = convert_audio_fn(melody, melody_sr, sample_rate, channels)
        target = convert_audio_fn(target, target_sr, sample_rate, channels)
        melodies.append(_pad_or_trim(melody, target_frames))
        targets.append(_pad_or_trim(target, target_frames))

        features = np.load(item["rhythm_condition"])["features"].astype("float32")
        features = _pad_or_trim_features(features, rhythm_frames, rhythm_input_dim)
        rhythms.append(torch.from_numpy(features))
        texts.append(item["text"])
        meta.append({"track_id": item["track_id"], "chunk_index": item["chunk_index"]})

    return {
        "melodies": torch.stack(melodies, dim=0),
        "targets": torch.stack(targets, dim=0),
        "rhythms": torch.stack(rhythms, dim=0),
        "texts": texts,
        "meta": meta,
    }


def _training_step(model, rhythm_bias, batch: dict, device: torch.device, scaler, amp: bool) -> torch.Tensor:
    targets = batch["targets"].to(device)
    rhythms = batch["rhythms"].to(device)
    melodies = [wav.to(device) for wav in batch["melodies"]]
    texts = batch["texts"]

    with torch.no_grad():
        codes, scale = model.compression_model.encode(targets)
        if scale is not None:
            raise RuntimeError("Expected MusicGen compression scale to be None.")
        attributes, _ = model._prepare_tokens_and_attributes(
            descriptions=texts,
            prompt=None,
            melody_wavs=melodies,
        )

    autocast_enabled = amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        output = model.lm.compute_predictions(codes, attributes)
        logits = rhythm_bias(rhythms, output.logits)
        loss = _masked_cross_entropy(logits, codes, output.mask)
    return loss


@torch.no_grad()
def _evaluate(model, rhythm_bias, loader, device: torch.device, amp: bool, max_batches: int) -> float:
    was_training = model.lm.training
    model.lm.eval()
    rhythm_bias.eval()
    losses = []
    for index, batch in enumerate(loader, start=1):
        loss = _training_step(model, rhythm_bias, batch, device, scaler=None, amp=amp)
        losses.append(float(loss.item()))
        if index >= max_batches:
            break
    if was_training:
        model.lm.train()
        rhythm_bias.train()
    return sum(losses) / max(1, len(losses))


def _pad_or_trim_features(features, target_frames: int, feature_dim: int):
    import numpy as np

    if features.ndim != 2:
        raise RuntimeError(f"Expected rhythm features [T,F], got {features.shape}")
    if features.shape[1] < feature_dim:
        pad = np.zeros((features.shape[0], feature_dim - features.shape[1]), dtype=features.dtype)
        features = np.concatenate([features, pad], axis=1)
    elif features.shape[1] > feature_dim:
        features = features[:, :feature_dim]

    if features.shape[0] == target_frames:
        return features
    if features.shape[0] > target_frames:
        return features[:target_frames]
    pad = np.zeros((target_frames - features.shape[0], features.shape[1]), dtype=features.dtype)
    return np.concatenate([features, pad], axis=0)


def _fit_rhythm_length(rhythm: torch.Tensor, time_steps: int) -> torch.Tensor:
    if rhythm.shape[1] == time_steps:
        return rhythm
    rhythm = rhythm.transpose(1, 2)
    rhythm = F.interpolate(rhythm, size=time_steps, mode="linear", align_corners=False)
    return rhythm.transpose(1, 2)


def _infer_vocab_size(model) -> int:
    if hasattr(model.lm, "card"):
        return int(model.lm.card)
    if hasattr(model.lm, "linears") and len(model.lm.linears) > 0:
        return int(model.lm.linears[0].out_features)
    raise RuntimeError("Could not infer LM vocabulary size.")


def _load_rhythm_checkpoint(lm, rhythm_bias, checkpoint: Path, device: torch.device) -> None:
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)

    trainable = state.get("trainable")
    if isinstance(trainable, dict):
        missing, unexpected = lm.load_state_dict(trainable, strict=False)
        print(f"Loaded LM partial checkpoint: {checkpoint}")
        print(f"LM partial keys: {len(trainable)} missing={len(missing)} unexpected={len(unexpected)}")

    rhythm_state = state.get("rhythm_conditioner")
    if isinstance(rhythm_state, dict):
        rhythm_bias.load_state_dict(rhythm_state, strict=True)
        print("Loaded rhythm conditioner state.")
    else:
        print("No rhythm conditioner state found; initializing rhythm conditioner from scratch.")


def _save_checkpoint(path: Path, model, rhythm_bias, model_name: str, trainable_names: list[str], global_step: int, args) -> None:
    state = model.lm.state_dict()
    trainable = {
        name: state[name].detach().cpu()
        for name in trainable_names
        if name in state
    }
    torch.save(
        {
            "model_name": model_name,
            "global_step": global_step,
            "trainable": trainable,
            "trainable_names": trainable_names,
            "rhythm_conditioner": rhythm_bias.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"[save] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MusicGen Melody with V4 rhythm conditions.")
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--valid-metadata", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trainable", choices=["output_linears", "last_layers", "linears", "all"], default="last_layers")
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument("--trainable-dtype", choices=["float32", "keep"], default="float32")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-duration", type=float, default=30.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--valid-every", type=int, default=25)
    parser.add_argument("--max-valid-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--rhythm-input-dim", type=int, default=6)
    parser.add_argument("--rhythm-hidden-dim", type=int, default=512)
    parser.add_argument("--rhythm-feature-rate", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    train(args)


if __name__ == "__main__":
    main()
