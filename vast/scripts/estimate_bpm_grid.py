#!/usr/bin/env python3
"""Estimate BPM and first beat/downbeat for bar-aligned dataset preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_bpm_grid(
    audio_dir: Path,
    output: Path,
    audio_name: str,
    target_bpm: float,
    min_confidence: float,
    limit: int | None,
    overwrite: bool,
) -> None:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder does not exist: {audio_dir}")

    audio_files = sorted(audio_dir.rglob(audio_name))
    if limit is not None:
        audio_files = audio_files[:limit]
    if not audio_files:
        raise RuntimeError(f"No {audio_name} files found under: {audio_dir}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Audio dir:      {audio_dir.resolve()}")
    print(f"Audio name:     {audio_name}")
    print(f"Files:          {len(audio_files)}")
    print(f"Target BPM:     {target_bpm}")
    print(f"Min confidence: {min_confidence}")
    print(f"Output:         {output.resolve()}")
    print("")

    with output.open("w", encoding="utf-8") as handle:
        for index, audio_path in enumerate(audio_files, start=1):
            row = _estimate_one(audio_path, audio_dir, target_bpm, min_confidence)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(audio_files)}] {row['track_id']} "
                f"bpm={row['bpm']:.2f} first={row['first_downbeat']:.3f}s "
                f"confidence={row['confidence']:.2f} review={row['needs_review']}"
            )


def _estimate_one(audio_path: Path, audio_dir: Path, target_bpm: float, min_confidence: float) -> dict:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: librosa/numpy. Install with `python -m pip install librosa numpy`."
        ) from exc

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    first_downbeat = float(beat_times[0]) if len(beat_times) else 0.0
    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_count = int(len(beat_times))

    # A simple confidence score: enough beats + tempo close to target BPM family.
    tempo_ratio = min(tempo, target_bpm) / max(tempo, target_bpm) if tempo > 0 else 0.0
    beat_density = min(1.0, beat_count / max(1.0, duration / 2.0))
    confidence = float(max(0.0, min(1.0, 0.65 * tempo_ratio + 0.35 * beat_density)))

    rel_track = audio_path.parent.relative_to(audio_dir).as_posix()
    return {
        "track_id": rel_track,
        "audio": str(audio_path),
        "bpm": tempo,
        "target_bpm": target_bpm,
        "first_downbeat": first_downbeat,
        "duration": duration,
        "beat_count": beat_count,
        "confidence": confidence,
        "needs_review": confidence < min_confidence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate BPM/downbeat report for audio folders.")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-name", default="instrumental.wav")
    parser.add_argument("--target-bpm", type=float, default=128.0)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    estimate_bpm_grid(
        audio_dir=args.audio_dir,
        output=args.output,
        audio_name=args.audio_name,
        target_bpm=args.target_bpm,
        min_confidence=args.min_confidence,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
