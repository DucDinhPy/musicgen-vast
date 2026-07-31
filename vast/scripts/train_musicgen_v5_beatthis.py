#!/usr/bin/env python3
"""Train independent MusicGen V5 from V1 plus Beat This! event conditions.

Objective:

    melody_piano + text/BPM + Beat This events -> instrumental

The MusicGen LM is initialized from a V1 partial checkpoint. V5 converts raw
Schema-2 beat/downbeat NPZ events into a deterministic temporal feature grid
and trains a new residual logit conditioner. It does not import or load V4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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


V5_CHECKPOINT_KIND = "musicgen_v5_beatthis"
V5_CONDITION_SCHEMA = 2
V5_DETECTOR = "beat_this"
V5_TEMPO_ESTIMATOR = "filtered_interval_mean_v1"
RHYTHM_FEATURE_NAMES = (
    "beat_pulse",
    "downbeat_pulse",
    "beat_phase_sin",
    "beat_phase_cos",
    "bar_phase_sin",
    "bar_phase_cos",
    "bpm_normalized",
    "tempo_confidence",
    "downbeat_reliable",
)


class BeatThisLogitConditioner(nn.Module):
    """Convert temporal Beat This features into a residual vocabulary bias."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or vocab_size <= 0:
            raise ValueError("Conditioner dimensions must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        # Start exactly at the V1 logits. The output projection learns first,
        # then propagates useful gradients into the temporal encoder.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rhythm: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if rhythm.ndim != 3:
            raise RuntimeError(
                f"Expected rhythm [B,T,F], got {tuple(rhythm.shape)}"
            )
        if rhythm.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"Expected {self.input_dim} rhythm features, got "
                f"{rhythm.shape[-1]}"
            )
        if logits.ndim != 4:
            raise RuntimeError(
                f"Expected logits [B,K,T,V], got {tuple(logits.shape)}"
            )
        if logits.shape[-1] != self.vocab_size:
            raise RuntimeError(
                f"Expected vocabulary {self.vocab_size}, got {logits.shape[-1]}"
            )
        rhythm = _fit_feature_length(rhythm, logits.shape[2])
        hidden = self.temporal(rhythm.transpose(1, 2)).transpose(1, 2)
        bias = self.output(hidden).unsqueeze(1)
        return logits + bias


class V5PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], dataset_root: Path):
        if not rows:
            raise RuntimeError("V5 dataset has no rows.")
        self.rows = rows
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        required = {
            "input_audio",
            "target_audio",
            "text",
            "rhythm_condition",
            "bpm",
            "v5_schema_version",
            "rhythm_detector",
            "rhythm_analysis_audio",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise KeyError(f"V5 row {index} is missing fields: {missing}")
        if int(row["v5_schema_version"]) != V5_CONDITION_SCHEMA:
            raise ValueError(
                f"V5 row {index} schema is {row['v5_schema_version']}, "
                f"expected {V5_CONDITION_SCHEMA}."
            )
        if str(row["rhythm_detector"]) != V5_DETECTOR:
            raise ValueError(
                f"V5 row {index} detector is not {V5_DETECTOR}."
            )
        if str(row["rhythm_analysis_audio"]) != "target_audio":
            raise ValueError(
                f"V5 row {index} was not analyzed from target_audio."
            )
        return {
            "input_audio": _resolve_path(
                row["input_audio"], self.dataset_root
            ),
            "target_audio": _resolve_path(
                row["target_audio"], self.dataset_root
            ),
            "rhythm_condition": _resolve_path(
                row["rhythm_condition"], self.dataset_root
            ),
            "text": str(row["text"]),
            "bpm": float(row["bpm"]),
            "track_id": str(row.get("track_id", "")),
            "chunk_index": int(row.get("chunk_index", -1)),
        }


def train(args: argparse.Namespace) -> None:
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    model.compression_model.eval()
    model.lm.train()
    _freeze(model.compression_model)

    trainable_names = _set_trainable(
        model.lm, args.trainable, args.last_n_layers
    )
    if args.trainable_dtype == "float32":
        _cast_trainable_params(model.lm, torch.float32)

    if args.init_from == "v1":
        print(f"Loading V1 base checkpoint: {args.base_v1_checkpoint}")
        _load_v1_base_checkpoint(
            lm=model.lm,
            checkpoint=args.base_v1_checkpoint,
            expected_model=args.model,
        )
    else:
        print(
            "Initializing V5 directly from the original pretrained "
            f"checkpoint: {args.model}"
        )

    vocab_size = _infer_vocab_size(model)
    conditioner = BeatThisLogitConditioner(
        input_dim=len(RHYTHM_FEATURE_NAMES),
        hidden_dim=args.rhythm_hidden_dim,
        vocab_size=vocab_size,
        dropout=args.rhythm_dropout,
    ).to(device)

    resumed_step = 0
    if args.resume_v5_checkpoint is not None:
        resumed_step = _load_v5_checkpoint(
            lm=model.lm,
            conditioner=conditioner,
            checkpoint=args.resume_v5_checkpoint,
            expected_initialization=args.init_from,
        )

    lm_trainable = [
        parameter for parameter in model.lm.parameters()
        if parameter.requires_grad
    ]
    conditioner_trainable = list(conditioner.parameters())
    trainable_params = lm_trainable + conditioner_trainable
    if not trainable_params:
        raise RuntimeError("No V5 trainable parameters selected.")

    train_rows = _read_jsonl(args.train_metadata)
    valid_rows = (
        _read_jsonl(args.valid_metadata) if args.valid_metadata else []
    )
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_valid_rows is not None:
        valid_rows = valid_rows[: args.max_valid_rows]
    _assert_disjoint_tracks(train_rows, valid_rows)

    print(f"Model sample rate:       {model.sample_rate}")
    print(f"Model channels:          {model.audio_channels}")
    print(f"LM trainable mode:       {args.trainable}")
    print(f"LM trainable tensors:    {len(trainable_names)}")
    print(f"LM trainable params:     {sum(p.numel() for p in lm_trainable):,}")
    print(f"LM learning rate:        {args.lr}")
    print(f"Rhythm features:         {len(RHYTHM_FEATURE_NAMES)}")
    print(f"Rhythm hidden dim:       {args.rhythm_hidden_dim}")
    print(f"Rhythm feature rate:     {args.rhythm_feature_rate}")
    print(
        f"Rhythm trainable params: "
        f"{sum(p.numel() for p in conditioner_trainable):,}"
    )
    print(f"Rhythm learning rate:    {args.rhythm_lr}")
    print(f"Train rows:              {len(train_rows)}")
    print(f"Valid rows:              {len(valid_rows)}")
    print(f"Resumed V5 step:         {resumed_step}")

    train_loader = _loader(
        rows=train_rows,
        args=args,
        model=model,
        convert_audio_fn=convert_audio,
        shuffle=True,
    )
    valid_loader = (
        _loader(
            rows=valid_rows,
            args=args,
            model=model,
            convert_audio_fn=convert_audio,
            shuffle=False,
        )
        if valid_rows else None
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": lm_trainable, "lr": args.lr},
            {"params": conditioner_trainable, "lr": args.rhythm_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    use_grad_scaler = (
        args.amp
        and device.type == "cuda"
        and all(parameter.dtype != torch.float16 for parameter in trainable_params)
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    optimizer.zero_grad(set_to_none=True)

    global_step = resumed_step
    steps_this_run = 0
    start_time = time.time()
    stop = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            global_step += 1
            steps_this_run += 1
            loss = _training_step(
                model=model,
                conditioner=conditioner,
                batch=batch,
                device=device,
                amp=args.amp,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {global_step}: {loss.item()}"
                )
            loss_for_backward = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if steps_this_run % args.grad_accum_steps == 0:
                _optimizer_step(
                    optimizer=optimizer,
                    scaler=scaler,
                    trainable_params=trainable_params,
                    grad_clip=args.grad_clip,
                )

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[train] epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f} elapsed={elapsed:.1f}s"
                )

            if (
                valid_loader is not None
                and global_step % args.valid_every == 0
            ):
                valid_loss = _evaluate(
                    model=model,
                    conditioner=conditioner,
                    loader=valid_loader,
                    device=device,
                    amp=args.amp,
                    max_batches=args.max_valid_batches,
                )
                print(f"[valid] step={global_step} loss={valid_loss:.4f}")

            if global_step % args.save_every == 0:
                _save_checkpoint(
                    path=args.output_dir / f"checkpoint_step_{global_step}.pt",
                    model=model,
                    conditioner=conditioner,
                    model_name=args.model,
                    trainable_names=trainable_names,
                    global_step=global_step,
                    args=args,
                )

            if args.max_steps is not None and steps_this_run >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if steps_this_run % args.grad_accum_steps:
        _optimizer_step(
            optimizer=optimizer,
            scaler=scaler,
            trainable_params=trainable_params,
            grad_clip=args.grad_clip,
        )

    _save_checkpoint(
        path=args.output_dir / "checkpoint_last.pt",
        model=model,
        conditioner=conditioner,
        model_name=args.model,
        trainable_names=trainable_names,
        global_step=global_step,
        args=args,
    )
    print("Done.")
    print(f"Last checkpoint: {args.output_dir / 'checkpoint_last.pt'}")


def _loader(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model: Any,
    convert_audio_fn: Any,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        V5PairDataset(rows, args.dataset_root),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda batch: _collate(
            batch=batch,
            sample_rate=model.sample_rate,
            channels=model.audio_channels,
            audio_duration=args.audio_duration,
            feature_rate=args.rhythm_feature_rate,
            convert_audio_fn=convert_audio_fn,
        ),
    )


def _collate(
    batch: list[dict[str, Any]],
    sample_rate: int,
    channels: int,
    audio_duration: float,
    feature_rate: float,
    convert_audio_fn: Any,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    target_frames = int(round(audio_duration * sample_rate))
    melodies = []
    targets = []
    rhythms = []
    texts = []
    metadata = []
    for item in batch:
        melody, melody_sr = _load_wav(item["input_audio"], sf)
        target, target_sr = _load_wav(item["target_audio"], sf)
        melody = convert_audio_fn(
            melody, melody_sr, sample_rate, channels
        )
        target = convert_audio_fn(
            target, target_sr, sample_rate, channels
        )
        melodies.append(_pad_or_trim(melody, target_frames))
        targets.append(_pad_or_trim(target, target_frames))

        with np.load(item["rhythm_condition"], allow_pickle=False) as data:
            condition = _load_condition_arrays(
                data=data,
                condition_path=item["rhythm_condition"],
                metadata_bpm=item["bpm"],
            )
        rhythms.append(
            torch.from_numpy(
                _events_to_features(
                    condition=condition,
                    duration=audio_duration,
                    feature_rate=feature_rate,
                )
            )
        )
        texts.append(item["text"])
        metadata.append(
            {
                "track_id": item["track_id"],
                "chunk_index": item["chunk_index"],
                "bpm": item["bpm"],
                "downbeat_reliable": condition["downbeat_reliable"],
            }
        )

    return {
        "melodies": torch.stack(melodies, dim=0),
        "targets": torch.stack(targets, dim=0),
        "rhythms": torch.stack(rhythms, dim=0),
        "texts": texts,
        "meta": metadata,
    }


def _load_condition_arrays(
    data: Any,
    condition_path: Path,
    metadata_bpm: float,
) -> dict[str, Any]:
    import numpy as np

    required = {
        "schema_version",
        "detector",
        "tempo_estimator",
        "analysis_source",
        "duration",
        "beats",
        "downbeats",
        "bpm",
        "tempo_relative_mad",
        "estimated_beats_per_bar",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(
            f"Condition {condition_path} is missing keys: {missing}"
        )
    schema = int(data["schema_version"][0])
    detector = str(data["detector"][0])
    estimator = str(data["tempo_estimator"][0])
    analysis_source = str(data["analysis_source"][0])
    if schema != V5_CONDITION_SCHEMA:
        raise RuntimeError(
            f"Condition {condition_path} schema {schema}, expected "
            f"{V5_CONDITION_SCHEMA}."
        )
    if detector != V5_DETECTOR:
        raise RuntimeError(
            f"Condition {condition_path} detector {detector!r}, expected "
            f"{V5_DETECTOR!r}."
        )
    if estimator != V5_TEMPO_ESTIMATOR:
        raise RuntimeError(
            f"Condition {condition_path} estimator {estimator!r}, expected "
            f"{V5_TEMPO_ESTIMATOR!r}."
        )
    if analysis_source != "target_audio":
        raise RuntimeError(
            f"Condition {condition_path} source is {analysis_source!r}; "
            "V5 training requires target_audio/instrumental."
        )

    bpm = float(data["bpm"][0])
    if abs(bpm - metadata_bpm) > 0.05:
        raise RuntimeError(
            f"Condition/metadata BPM mismatch for {condition_path}: "
            f"{bpm:.4f} != {metadata_bpm:.4f}"
        )
    beats = _clean_events(data["beats"], "beats", condition_path, np)
    downbeats = _clean_events(
        data["downbeats"], "downbeats", condition_path, np
    )
    beats_per_bar = int(data["estimated_beats_per_bar"][0])
    ratio = len(downbeats) / max(1, len(beats))
    downbeat_reliable = (
        beats_per_bar == 4 and 0.18 <= ratio <= 0.35
    )
    return {
        "duration": float(data["duration"][0]),
        "beats": beats,
        "downbeats": downbeats,
        "bpm": bpm,
        "tempo_relative_mad": float(data["tempo_relative_mad"][0]),
        "estimated_beats_per_bar": beats_per_bar,
        "downbeat_reliable": downbeat_reliable,
    }


def _clean_events(
    values: Any,
    label: str,
    condition_path: Path,
    np_module: Any,
) -> Any:
    events = np_module.asarray(values, dtype=np_module.float32).reshape(-1)
    events = events[np_module.isfinite(events)]
    events = np_module.unique(events[events >= 0.0])
    if label == "beats" and len(events) < 4:
        raise RuntimeError(
            f"Condition {condition_path} has fewer than four beats."
        )
    return events


def _events_to_features(
    condition: dict[str, Any],
    duration: float,
    feature_rate: float,
) -> Any:
    import numpy as np

    if duration <= 0 or feature_rate <= 0:
        raise ValueError("Duration and feature rate must be positive.")
    frame_count = int(round(duration * feature_rate))
    times = np.arange(frame_count, dtype=np.float32) / feature_rate
    features = np.zeros(
        (frame_count, len(RHYTHM_FEATURE_NAMES)), dtype=np.float32
    )
    beats = condition["beats"]
    downbeats = condition["downbeats"]
    _write_event_pulses(features[:, 0], beats, feature_rate)

    downbeat_reliable = bool(condition["downbeat_reliable"])
    if downbeat_reliable:
        _write_event_pulses(features[:, 1], downbeats, feature_rate)

    beat_phase = _event_phase(times, beats, np)
    features[:, 2] = np.sin(2.0 * np.pi * beat_phase)
    features[:, 3] = np.cos(2.0 * np.pi * beat_phase)

    if downbeat_reliable:
        bar_phase = _event_phase(times, downbeats, np)
        features[:, 4] = np.sin(2.0 * np.pi * bar_phase)
        features[:, 5] = np.cos(2.0 * np.pi * bar_phase)

    bpm = float(condition["bpm"])
    features[:, 6] = np.clip((bpm - 135.0) / 15.0, -1.0, 1.0)
    relative_mad = max(0.0, float(condition["tempo_relative_mad"]))
    features[:, 7] = 1.0 - np.clip(relative_mad / 0.1, 0.0, 1.0)
    features[:, 8] = 1.0 if downbeat_reliable else 0.0
    return features


def _write_event_pulses(
    destination: Any,
    events: Any,
    feature_rate: float,
) -> None:
    frame_count = len(destination)
    for event in events:
        index = int(round(float(event) * feature_rate))
        if 0 <= index < frame_count:
            destination[index] = 1.0
            if index > 0:
                destination[index - 1] = max(destination[index - 1], 0.5)
            if index + 1 < frame_count:
                destination[index + 1] = max(destination[index + 1], 0.5)


def _event_phase(times: Any, events: Any, np_module: Any) -> Any:
    phase = np_module.zeros_like(times, dtype=np_module.float32)
    if len(events) < 2:
        return phase
    previous_indices = np_module.searchsorted(events, times, side="right") - 1
    valid = (previous_indices >= 0) & (previous_indices < len(events) - 1)
    indices = previous_indices[valid]
    starts = events[indices]
    intervals = events[indices + 1] - starts
    safe = intervals > 1e-6
    values = np_module.zeros_like(starts, dtype=np_module.float32)
    values[safe] = (times[valid][safe] - starts[safe]) / intervals[safe]
    phase[valid] = np_module.clip(values, 0.0, 1.0)
    return phase


def _fit_feature_length(
    rhythm: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    if rhythm.shape[1] == time_steps:
        return rhythm
    return F.interpolate(
        rhythm.transpose(1, 2),
        size=time_steps,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def _training_step(
    model: Any,
    conditioner: BeatThisLogitConditioner,
    batch: dict[str, Any],
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    targets = batch["targets"].to(device)
    melodies = [audio.to(device) for audio in batch["melodies"]]
    rhythms = batch["rhythms"].to(device)
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
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        output = model.lm.compute_predictions(codes, attributes)
        logits = conditioner(rhythms, output.logits)
        return _masked_cross_entropy(logits, codes, output.mask)


@torch.no_grad()
def _evaluate(
    model: Any,
    conditioner: BeatThisLogitConditioner,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> float:
    lm_was_training = model.lm.training
    conditioner_was_training = conditioner.training
    model.lm.eval()
    conditioner.eval()
    losses = []
    for batch_index, batch in enumerate(loader, start=1):
        loss = _training_step(
            model=model,
            conditioner=conditioner,
            batch=batch,
            device=device,
            amp=amp,
        )
        losses.append(float(loss.item()))
        if batch_index >= max_batches:
            break
    if lm_was_training:
        model.lm.train()
    if conditioner_was_training:
        conditioner.train()
    return sum(losses) / max(1, len(losses))


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    trainable_params: list[torch.Tensor],
    grad_clip: float,
) -> None:
    if grad_clip > 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _infer_vocab_size(model: Any) -> int:
    if hasattr(model.lm, "card"):
        return int(model.lm.card)
    if hasattr(model.lm, "linears") and len(model.lm.linears) > 0:
        return int(model.lm.linears[0].out_features)
    raise RuntimeError("Could not infer MusicGen LM vocabulary size.")


def _assert_disjoint_tracks(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
) -> None:
    if not valid_rows:
        return
    train_ids = {str(row.get("track_id", "")) for row in train_rows}
    valid_ids = {str(row.get("track_id", "")) for row in valid_rows}
    if "" in train_ids or "" in valid_ids:
        raise ValueError("Every train/valid row must have a track_id.")
    leaked = sorted(train_ids.intersection(valid_ids))
    if leaked:
        raise RuntimeError(
            f"Train/valid track leakage detected: {leaked[:10]}"
        )


def _load_v5_checkpoint(
    lm: nn.Module,
    conditioner: BeatThisLogitConditioner,
    checkpoint: Path,
    expected_initialization: str,
) -> int:
    try:
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    if state.get("checkpoint_kind") != V5_CHECKPOINT_KIND:
        raise RuntimeError(f"Not a V5 Beat This checkpoint: {checkpoint}")
    checkpoint_initialization = state.get("initialization", "v1")
    if checkpoint_initialization != expected_initialization:
        raise RuntimeError(
            f"V5 checkpoint initialization {checkpoint_initialization!r} "
            f"does not match requested {expected_initialization!r}."
        )
    trainable = state.get("trainable")
    rhythm_state = state.get("rhythm_conditioner")
    if not isinstance(trainable, dict) or not isinstance(rhythm_state, dict):
        raise RuntimeError(f"Incomplete V5 checkpoint: {checkpoint}")
    missing, unexpected = lm.load_state_dict(trainable, strict=False)
    conditioner.load_state_dict(rhythm_state, strict=True)
    global_step = int(state.get("global_step", 0))
    print(f"Resumed V5 checkpoint: {checkpoint}")
    print(
        f"V5 LM keys: {len(trainable)} missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )
    print(f"V5 resumed step: {global_step}")
    return global_step


def _load_v1_base_checkpoint(
    lm: nn.Module,
    checkpoint: Path,
    expected_model: str,
) -> None:
    try:
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid V1 checkpoint object: {checkpoint}")
    if "rhythm_conditioner" in state or state.get("checkpoint_kind"):
        raise RuntimeError(
            f"Base checkpoint is not a clean V1 checkpoint: {checkpoint}"
        )
    checkpoint_model = state.get("model_name")
    if checkpoint_model and checkpoint_model != expected_model:
        raise RuntimeError(
            f"V1 checkpoint model {checkpoint_model!r} does not match "
            f"{expected_model!r}."
        )
    trainable = state.get("trainable")
    if not isinstance(trainable, dict) or not trainable:
        raise RuntimeError(
            f"V1 checkpoint has no trainable LM state: {checkpoint}"
        )
    missing, unexpected = lm.load_state_dict(trainable, strict=False)
    print(f"Loaded clean V1 checkpoint: {checkpoint}")
    print(
        f"V1 LM keys: {len(trainable)} missing={len(missing)} "
        f"unexpected={len(unexpected)}"
    )


def _save_checkpoint(
    path: Path,
    model: Any,
    conditioner: BeatThisLogitConditioner,
    model_name: str,
    trainable_names: list[str],
    global_step: int,
    args: argparse.Namespace,
) -> None:
    lm_state = model.lm.state_dict()
    trainable = {
        name: lm_state[name].detach().cpu()
        for name in trainable_names
        if name in lm_state
    }
    torch.save(
        {
            "checkpoint_kind": V5_CHECKPOINT_KIND,
            "initialization": args.init_from,
            "condition_schema": V5_CONDITION_SCHEMA,
            "detector": V5_DETECTOR,
            "tempo_estimator": V5_TEMPO_ESTIMATOR,
            "rhythm_feature_names": RHYTHM_FEATURE_NAMES,
            "model_name": model_name,
            "global_step": global_step,
            "trainable": trainable,
            "trainable_names": trainable_names,
            "rhythm_conditioner": conditioner.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"[save] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent MusicGen V5 from a V1 checkpoint and Beat This "
            "Schema-2 instrumental conditions."
        )
    )
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--valid-metadata", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init-from",
        choices=["v1", "pretrained"],
        default="v1",
        help=(
            "Initialize from the project V1 checkpoint or directly from the "
            "original pretrained MusicGen model. Default: v1."
        ),
    )
    parser.add_argument("--base-v1-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-v5-checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trainable",
        choices=["output_linears", "last_layers", "linears", "all"],
        default="last_layers",
    )
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument(
        "--trainable-dtype",
        choices=["float32", "keep"],
        default="float32",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Optimizer/micro-batch steps to run in this invocation.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-duration", type=float, default=30.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=8e-7)
    parser.add_argument(
        "--rhythm-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the new Beat This conditioner.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--valid-every", type=int, default=25)
    parser.add_argument("--max-valid-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--rhythm-hidden-dim", type=int, default=256)
    parser.add_argument("--rhythm-dropout", type=float, default=0.1)
    parser.add_argument("--rhythm-feature-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.train_metadata, "train metadata"),
        (args.dataset_root, "dataset root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.init_from == "v1":
        if args.base_v1_checkpoint is None:
            raise ValueError(
                "--base-v1-checkpoint is required with --init-from v1."
            )
        if not args.base_v1_checkpoint.is_file():
            raise FileNotFoundError(
                f"V1 checkpoint does not exist: {args.base_v1_checkpoint}"
            )
    elif args.base_v1_checkpoint is not None:
        raise ValueError(
            "Do not pass --base-v1-checkpoint with --init-from pretrained."
        )
    if args.valid_metadata is not None and not args.valid_metadata.is_file():
        raise FileNotFoundError(
            f"valid metadata does not exist: {args.valid_metadata}"
        )
    if args.resume_v5_checkpoint is not None and not args.resume_v5_checkpoint.is_file():
        raise FileNotFoundError(
            f"V5 checkpoint does not exist: {args.resume_v5_checkpoint}"
        )
    if args.audio_duration <= 0 or args.rhythm_feature_rate <= 0:
        raise ValueError("Audio duration and rhythm feature rate must be > 0.")
    if args.grad_accum_steps <= 0 or args.batch_size <= 0:
        raise ValueError("Batch size and gradient accumulation must be > 0.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0 when provided.")
    if args.lr <= 0 or args.rhythm_lr <= 0:
        raise ValueError("LM and rhythm learning rates must be > 0.")
    if not 0.0 <= args.rhythm_dropout < 1.0:
        raise ValueError("--rhythm-dropout must be in [0, 1).")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )
    train(args)


if __name__ == "__main__":
    main()
