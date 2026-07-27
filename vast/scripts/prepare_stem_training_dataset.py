#!/usr/bin/env python3
"""
Prepare chunked stem-specific MusicGen training data.

Input:

    data/melody/rendered/set_01/song_id/melody_piano.wav
    data/stems/stems_set_single/set_01/song_id/drums.wav
    data/stems/stems_set_single/set_01/song_id/bass.wav
    data/stems/stems_set_single/set_01/song_id/other.wav

Output:

    data/datasets/stem_training/
      audio/
        melody/set_01/song_id/chunk_0000.wav
        drums/set_01/song_id/chunk_0000.wav
        bass/set_01/song_id/chunk_0000.wav
      metadata_all.jsonl
      metadata_drums.jsonl
      metadata_bass.jsonl

Each metadata row follows:

    input_audio + text -> target_audio
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


DEFAULT_PROMPTS = {
    "instrumental": (
        "generate a full clean energetic vinahouse instrumental backing track, "
        "punchy drums, rolling bass, bright synth chords, build ups and drops, "
        "no lead vocal"
    ),
    "drums": (
        "generate energetic vinahouse drums, punchy four-on-the-floor kick, "
        "snare claps, percussion groove, no bass, no vocals"
    ),
    "bass": (
        "generate vinahouse rolling bassline, club low end, follow the melody "
        "harmony, no drums, no vocals"
    ),
    "other": (
        "generate vinahouse synth chords, stabs, risers, effects, supporting "
        "harmony, no drums, no bass, no vocals"
    ),
    "piano": (
        "generate vinahouse piano chords and rhythmic harmonic accompaniment, "
        "no drums, no bass, no vocals"
    ),
    "guitar": (
        "generate vinahouse guitar accompaniment, rhythmic plucks and fills, "
        "no drums, no bass, no vocals"
    ),
}


def prepare_dataset(
    melody_dir: Path,
    stems_dir: Path,
    output_dir: Path,
    target_stems: list[str],
    chunk_seconds: float,
    hop_seconds: float,
    min_chunk_seconds: float,
    sample_rate: int,
    min_target_rms_db: float,
    relative_to: Path | None,
    start_index: int,
    limit: int | None,
    overwrite: bool,
    keep_going: bool,
    dry_run: bool,
) -> None:
    _validate_dir(melody_dir, "melody")
    _validate_dir(stems_dir, "stems")

    melody_files = sorted(melody_dir.rglob("melody_piano.wav"))
    melody_files = melody_files[start_index:]
    if limit is not None:
        melody_files = melody_files[:limit]
    if not melody_files:
        raise RuntimeError(f"No melody_piano.wav files found under: {melody_dir}")

    metadata_all_path = output_dir / "metadata_all.jsonl"
    report_path = output_dir / "prepare_stem_training_report.jsonl"
    metadata_by_stem = {
        stem: output_dir / f"metadata_{stem}.jsonl"
        for stem in target_stems
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Melody dir:        {melody_dir.resolve()}")
    print(f"Stems dir:         {stems_dir.resolve()}")
    print(f"Output dir:        {output_dir.resolve()}")
    print(f"Target stems:      {', '.join(target_stems)}")
    print(f"Chunk seconds:     {chunk_seconds}")
    print(f"Hop seconds:       {hop_seconds}")
    print(f"Min chunk seconds: {min_chunk_seconds}")
    print(f"Sample rate:       {sample_rate}")
    print(f"Min target RMS:    {min_target_rms_db:.1f} dBFS")
    print(f"Relative to:       {relative_to.resolve() if relative_to else output_dir.resolve()}")
    print(f"Start index:       {start_index}")
    print(f"Limit:             {limit if limit is not None else '(none)'}")
    print(f"Overwrite:         {overwrite}")
    print(f"Dry run:           {dry_run}")
    print("")

    ok_count = 0
    skip_count = 0
    error_count = 0

    metadata_all = None if dry_run else metadata_all_path.open("w", encoding="utf-8")
    stem_files = {
        stem: None if dry_run else path.open("w", encoding="utf-8")
        for stem, path in metadata_by_stem.items()
    }

    try:
        with report_path.open("w", encoding="utf-8") as report:
            for track_index, melody_path in enumerate(melody_files, start=1):
                rel_track = melody_path.parent.relative_to(melody_dir)
                track_id = rel_track.as_posix()
                melody_duration = _ffprobe_duration(melody_path)

                for stem in target_stems:
                    target_stem_path = stems_dir / rel_track / f"{stem}.wav"
                    if not target_stem_path.exists():
                        skip_count += 1
                        row = {
                            "track_id": track_id,
                            "target_stem": stem,
                            "status": "missing_target",
                            "melody": str(melody_path),
                            "target": str(target_stem_path),
                        }
                        report.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"[skip] Missing {stem}: {rel_track}")
                        continue

                    try:
                        target_duration = _ffprobe_duration(target_stem_path)
                        usable_duration = min(melody_duration, target_duration)
                        chunk_plan = _chunk_plan(
                            usable_duration=usable_duration,
                            chunk_seconds=chunk_seconds,
                            hop_seconds=hop_seconds,
                            min_chunk_seconds=min_chunk_seconds,
                        )

                        for chunk_index, start, duration in chunk_plan:
                            melody_chunk = (
                                output_dir
                                / "audio"
                                / "melody"
                                / rel_track
                                / f"chunk_{chunk_index:04d}.wav"
                            )
                            target_chunk = (
                                output_dir
                                / "audio"
                                / stem
                                / rel_track
                                / f"chunk_{chunk_index:04d}.wav"
                            )

                            base_row = {
                                "track_id": track_id,
                                "chunk_index": chunk_index,
                                "start": start,
                                "duration": duration,
                                "target_stem": stem,
                                "source_melody": str(melody_path),
                                "source_target": str(target_stem_path),
                                "input_audio": _display_path(melody_chunk, relative_to or output_dir),
                                "target_audio": _display_path(target_chunk, relative_to or output_dir),
                                "text": _prompt_for_stem(stem),
                            }

                            if dry_run:
                                ok_count += 1
                                row = dict(base_row)
                                row["status"] = "planned"
                                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                                continue

                            _write_chunk(
                                source=melody_path,
                                target=melody_chunk,
                                start=start,
                                duration=duration,
                                sample_rate=sample_rate,
                                overwrite=overwrite,
                            )
                            _write_chunk(
                                source=target_stem_path,
                                target=target_chunk,
                                start=start,
                                duration=duration,
                                sample_rate=sample_rate,
                                overwrite=overwrite,
                            )

                            stats = _audio_stats(target_chunk)
                            if stats["rms_db"] < min_target_rms_db:
                                skip_count += 1
                                row = dict(base_row)
                                row["status"] = "silent_target"
                                row.update(stats)
                                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                                print(
                                    f"[skip] Silent {stem} chunk {chunk_index:04d}: "
                                    f"{rel_track} ({stats['rms_db']:.1f} dBFS)"
                                )
                                continue

                            ok_count += 1
                            row = dict(base_row)
                            row["status"] = "ok"
                            row.update(stats)
                            report.write(json.dumps(row, ensure_ascii=False) + "\n")
                            metadata_row = {
                                "input_audio": row["input_audio"],
                                "target_audio": row["target_audio"],
                                "text": row["text"],
                                "target_stem": stem,
                                "track_id": track_id,
                                "chunk_index": chunk_index,
                                "duration": duration,
                            }
                            metadata_all.write(json.dumps(metadata_row, ensure_ascii=False) + "\n")
                            stem_files[stem].write(json.dumps(metadata_row, ensure_ascii=False) + "\n")

                    except Exception as exc:
                        error_count += 1
                        row = {
                            "track_id": track_id,
                            "target_stem": stem,
                            "status": "error",
                            "error": str(exc),
                        }
                        report.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"[error] {stem}: {rel_track}: {exc}")
                        if not keep_going:
                            raise
                        continue

                print(f"[track] {track_index}/{len(melody_files)} {rel_track}")
    finally:
        if metadata_all is not None:
            metadata_all.close()
        for handle in stem_files.values():
            if handle is not None:
                handle.close()

    print("")
    print("Done.")
    print(f"OK/planned rows: {ok_count}")
    print(f"Skipped rows:    {skip_count}")
    print(f"Errors:          {error_count}")
    print(f"Metadata all:    {metadata_all_path.resolve()}")
    for stem, path in metadata_by_stem.items():
        print(f"Metadata {stem}: {path.resolve()}")
    print(f"Report:          {report_path.resolve()}")


def _validate_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path must be a folder: {path}")


def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _chunk_plan(
    usable_duration: float,
    chunk_seconds: float,
    hop_seconds: float,
    min_chunk_seconds: float,
) -> list[tuple[int, float, float]]:
    chunks: list[tuple[int, float, float]] = []
    if usable_duration < min_chunk_seconds:
        return chunks

    chunk_index = 0
    start = 0.0
    while start < usable_duration:
        remaining = usable_duration - start
        duration = min(chunk_seconds, remaining)
        if duration < min_chunk_seconds:
            break
        chunks.append((chunk_index, start, duration))
        chunk_index += 1
        start += hop_seconds
    return chunks


def _write_chunk(
    source: Path,
    target: Path,
    start: float,
    duration: float,
    sample_rate: int,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _audio_stats(path: Path) -> dict[str, float]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: numpy/soundfile. Install with "
            "`python -m pip install numpy soundfile`."
        ) from exc

    audio, _ = sf.read(str(path), always_2d=False)
    if audio.size == 0:
        return {"rms_db": -math.inf, "peak_db": -math.inf}

    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return {
        "rms_db": _to_db(rms),
        "peak_db": _to_db(peak),
    }


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _display_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def _prompt_for_stem(stem: str) -> str:
    return DEFAULT_PROMPTS.get(
        stem,
        f"generate vinahouse {stem} accompaniment, no vocals",
    )


def _parse_stems(value: str) -> list[str]:
    stems = [part.strip() for part in value.split(",") if part.strip()]
    if not stems:
        raise argparse.ArgumentTypeError("At least one target stem is required.")
    return stems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare chunked melody-to-stem MusicGen training manifests."
    )
    parser.add_argument(
        "--melody-dir",
        type=Path,
        required=True,
        help="Root folder containing rendered melody_piano.wav files.",
    )
    parser.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Root folder containing organized target stems.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--target-stems",
        type=_parse_stems,
        default=_parse_stems("drums,bass,other"),
        help="Comma-separated target stems. Default: drums,bass,other.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk length in seconds. Default: 30.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=30.0,
        help="Hop length in seconds. Default: 30, no overlap.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=15.0,
        help="Skip final chunks shorter than this. Default: 15.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Output WAV sample rate. Default: 32000.",
    )
    parser.add_argument(
        "--min-target-rms-db",
        type=float,
        default=-55.0,
        help="Skip target chunks quieter than this RMS dBFS. Default: -55.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Store metadata paths relative to this folder. Default: output dir.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many discovered tracks. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many melody tracks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chunks.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after an error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rows without writing audio chunks or metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(
        melody_dir=args.melody_dir,
        stems_dir=args.stems_dir,
        output_dir=args.output_dir,
        target_stems=args.target_stems,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        sample_rate=args.sample_rate,
        min_target_rms_db=args.min_target_rms_db,
        relative_to=args.relative_to,
        start_index=args.start_index,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
