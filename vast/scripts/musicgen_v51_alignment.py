#!/usr/bin/env python3
"""Shared V5.1 vocal/melody timing feature extraction helpers.

The features intentionally describe *when* the sung melody is active and
changes.  They do not replace MusicGen's chroma melody conditioner and they do
not turn the full mix into a training target.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


V51_SCHEMA_VERSION = 1
V51_CHECKPOINT_KIND = "musicgen_v51_alignment"
V51_CONDITIONER_ARCHITECTURE = "alignment_v1"
VOCAL_TIMING_FEATURE_NAMES = (
    "vocal_activity",
    "vocal_onset_strength",
    "vocal_onset_pulse",
    "vocal_phrase_onset_pulse",
)


def extract_vocal_timing_features(
    audio_path: Path,
    duration: float,
    feature_rate: float = 50.0,
    start_seconds: float = 0.0,
) -> Any:
    """Return deterministic frame-aligned timing features as float32.

    This works with either a separated vocal or a monophonic rendered melody.
    The latter is the default for the existing V5 dataset because every row
    already has ``input_audio`` and its note timing follows the vocal.
    """
    import numpy as np
    import soundfile as sf

    if duration <= 0.0 or feature_rate <= 0.0:
        raise ValueError("duration and feature_rate must be positive")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Timing audio does not exist: {audio_path}")

    if start_seconds < 0.0:
        raise ValueError("start_seconds must be non-negative")
    with sf.SoundFile(str(audio_path), mode="r") as handle:
        sample_rate = int(handle.samplerate)
        start_frame = int(round(start_seconds * sample_rate))
        handle.seek(min(start_frame, len(handle)))
        read_frames = int(math.ceil(duration * sample_rate))
        audio = handle.read(
            frames=read_frames, dtype="float32", always_2d=True
        )
    if sample_rate <= 0:
        raise RuntimeError(f"Invalid sample rate for {audio_path}: {sample_rate}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    frame_count = max(1, int(round(duration * feature_rate)))

    # A 40 ms centered RMS window is stable for both singing and rendered
    # piano.  Cumulative sums keep extraction cheap for the full dataset.
    half_window = max(1, int(round(sample_rate * 0.020)))
    centers = np.round(
        np.arange(frame_count, dtype=np.float64) * sample_rate / feature_rate
    ).astype(np.int64)
    starts = np.clip(centers - half_window, 0, len(mono))
    ends = np.clip(centers + half_window, 0, len(mono))
    squared = np.square(mono.astype(np.float64, copy=False))
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(squared, dtype=np.float64)]
    )
    counts = np.maximum(1, ends - starts)
    rms = np.sqrt((cumulative[ends] - cumulative[starts]) / counts)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-7))

    finite = rms_db[np.isfinite(rms_db)]
    if finite.size == 0:
        activity = np.zeros(frame_count, dtype=np.float32)
    else:
        noise_floor = float(np.percentile(finite, 20.0))
        active_level = float(np.percentile(finite, 95.0))
        span = max(12.0, active_level - noise_floor)
        activity = np.clip(
            (rms_db - (noise_floor + 3.0)) / span, 0.0, 1.0
        ).astype(np.float32)

    # Smooth one frame to suppress sample-level and separator artifacts.
    if frame_count >= 3:
        activity = np.convolve(
            activity, np.asarray([0.2, 0.6, 0.2], dtype=np.float32), mode="same"
        ).astype(np.float32)

    positive_delta = np.maximum(
        0.0, np.diff(activity, prepend=activity[:1])
    ).astype(np.float32)
    scale = float(np.percentile(positive_delta, 95.0))
    if scale > 1e-5:
        onset_strength = np.clip(positive_delta / scale, 0.0, 1.0)
    else:
        onset_strength = np.zeros_like(positive_delta)

    onset_pulse = _local_peak_pulses(
        onset_strength, threshold=0.30, activity=activity, np_module=np
    )

    # Phrase starts are activity transitions following at least 160 ms of
    # near-silence.  This exposes vocal entries/breaks separately from every
    # note onset so the conditioner can learn fills and transitions.
    phrase_pulse = np.zeros(frame_count, dtype=np.float32)
    silence_frames = max(1, int(round(0.16 * feature_rate)))
    active_mask = activity >= 0.22
    for index in np.flatnonzero(onset_pulse > 0.0):
        left = max(0, int(index) - silence_frames)
        if index == 0 or not np.any(active_mask[left:index]):
            phrase_pulse[index] = 1.0
            if index > 0:
                phrase_pulse[index - 1] = 0.5
            if index + 1 < frame_count:
                phrase_pulse[index + 1] = 0.5

    features = np.stack(
        [activity, onset_strength, onset_pulse, phrase_pulse], axis=1
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"Non-finite timing features extracted from {audio_path}")
    return features


def load_vocal_timing_condition(
    path: Path,
    expected_rate: float,
    expected_duration: float,
) -> Any:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "feature_names",
            "feature_rate",
            "duration",
            "features",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"V5.1 condition {path} is missing: {missing}")
        schema = int(data["schema_version"][0])
        names = tuple(str(value) for value in data["feature_names"].tolist())
        rate = float(data["feature_rate"][0])
        duration = float(data["duration"][0])
        features = np.asarray(data["features"], dtype=np.float32)

    if schema != V51_SCHEMA_VERSION:
        raise RuntimeError(
            f"V5.1 condition {path} schema {schema}, expected {V51_SCHEMA_VERSION}"
        )
    if names != VOCAL_TIMING_FEATURE_NAMES:
        raise RuntimeError(f"V5.1 feature mismatch in {path}: {names}")
    if abs(rate - expected_rate) > 1e-4:
        raise RuntimeError(f"V5.1 feature rate mismatch in {path}: {rate}")
    if features.ndim != 2 or features.shape[1] != len(VOCAL_TIMING_FEATURE_NAMES):
        raise RuntimeError(f"Invalid V5.1 feature shape in {path}: {features.shape}")
    stored_frames = max(1, int(round(duration * expected_rate)))
    if features.shape[0] != stored_frames:
        raise RuntimeError(
            f"V5.1 stored frame count mismatch in {path}: "
            f"{features.shape[0]} != {stored_frames}"
        )
    expected_frames = max(1, int(round(expected_duration * expected_rate)))
    if features.shape[0] == expected_frames:
        return features
    if features.shape[0] > expected_frames:
        return features[:expected_frames]
    padding = np.zeros(
        (expected_frames - features.shape[0], features.shape[1]),
        dtype=np.float32,
    )
    return np.concatenate([features, padding], axis=0)


def save_vocal_timing_condition(
    path: Path,
    features: Any,
    feature_rate: float,
    duration: float,
    source_audio: Path,
    source_field: str,
    source_start_seconds: float = 0.0,
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    stat = source_audio.stat()
    np.savez_compressed(
        path,
        schema_version=np.asarray([V51_SCHEMA_VERSION], dtype=np.int16),
        feature_names=np.asarray(VOCAL_TIMING_FEATURE_NAMES),
        feature_rate=np.asarray([feature_rate], dtype=np.float32),
        duration=np.asarray([duration], dtype=np.float32),
        source_audio=np.asarray([str(source_audio.resolve())]),
        source_field=np.asarray([source_field]),
        source_start_seconds=np.asarray(
            [source_start_seconds], dtype=np.float32
        ),
        source_size=np.asarray([stat.st_size], dtype=np.int64),
        source_mtime_ns=np.asarray([stat.st_mtime_ns], dtype=np.int64),
        features=np.asarray(features, dtype=np.float32),
    )


def _local_peak_pulses(
    strength: Any,
    threshold: float,
    activity: Any,
    np_module: Any,
) -> Any:
    pulses = np_module.zeros(len(strength), dtype=np_module.float32)
    if len(strength) == 0:
        return pulses
    left = np_module.concatenate([strength[:1], strength[:-1]])
    right = np_module.concatenate([strength[1:], strength[-1:]])
    peaks = (
        (strength >= threshold)
        & (strength >= left)
        & (strength >= right)
        & (activity >= 0.08)
    )
    indices = np_module.flatnonzero(peaks)
    # Keep the strongest event when peaks are closer than 60 ms (3 frames at
    # the canonical 50 Hz feature rate).
    kept: list[int] = []
    for raw_index in indices:
        index = int(raw_index)
        if kept and index - kept[-1] <= 2:
            if strength[index] > strength[kept[-1]]:
                kept[-1] = index
        else:
            kept.append(index)
    for index in kept:
        pulses[index] = 1.0
        if index > 0:
            pulses[index - 1] = max(float(pulses[index - 1]), 0.5)
        if index + 1 < len(pulses):
            pulses[index + 1] = max(float(pulses[index + 1]), 0.5)
    return pulses
