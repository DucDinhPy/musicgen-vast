#!/usr/bin/env python3
"""Prepare bar-aligned MusicGen training chunks from BPM-normalized audio."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track at 128 bpm, "
    "punchy drums, rolling bass, bright synth chords, build ups and drops, no lead vocal"
)


def prepare_bar_aligned_dataset(
    melody_root: Path,
    instrumental_root: Path,
    output_dir: Path,
    bpm: float,
    bars_per_chunk: int,
    hop_bars: int,
    sample_rate: int,
    melody_name: str,
    instrumental_name: str,
    prompt: str,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not melody_root.exists():
        raise FileNotFoundError(f"Melody root does not exist: {melody_root}")
    if not instrumental_root.exists():
        raise FileNotFoundError(f"Instrumental root does not exist: {instrumental_root}")

    melody_files = sorted(melody_root.rglob(melody_name))
    if limit is not None:
        melody_files = melody_files[:limit]
    if not melody_files:
        raise RuntimeError(f"No {melody_name} files found under: {melody_root}")

    chunk_seconds = bars_per_chunk * 4.0 * 60.0 / bpm
    hop_seconds = hop_bars * 4.0 * 60.0 / bpm
    metadata_path = output_dir / "metadata_instrumental.jsonl"
    report_path = output_dir / "prepare_bar_aligned_report.jsonl"

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Melody root:       {melody_root.resolve()}")
    print(f"Instrumental root: {instrumental_root.resolve()}")
    print(f"Output dir:        {output_dir.resolve()}")
    print(f"BPM:               {bpm}")
    print(f"Bars per chunk:    {bars_per_chunk}")
    print(f"Hop bars:          {hop_bars}")
    print(f"Chunk seconds:     {chunk_seconds:.3f}")
    print(f"Hop seconds:       {hop_seconds:.3f}")
    print(f"Sample rate:       {sample_rate}")
    print(f"Dry run:           {dry_run}")
    print("")

    ok = 0
    skipped = 0
    missing = 0

    metadata_file = None if dry_run else metadata_path.open("w", encoding="utf-8")
    try:
        with report_path.open("w", encoding="utf-8") as report_file:
            for track_index, melody_path in enumerate(melody_files, start=1):
                rel_track = melody_path.parent.relative_to(melody_root)
                instrumental_path = instrumental_root / rel_track / instrumental_name
                if not instrumental_path.exists():
                    missing += 1
                    row = {
                        "track_id": rel_track.as_posix(),
                        "status": "missing_instrumental",
                        "melody": str(melody_path),
                        "instrumental": str(instrumental_path),
                    }
                    report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[missing] {rel_track}")
                    continue

                duration = min(_duration(melody_path), _duration(instrumental_path))
                starts = _chunk_starts(duration, chunk_seconds, hop_seconds)
                for chunk_index, start in enumerate(starts):
                    melody_chunk = output_dir / "audio" / "melody" / rel_track / f"chunk_{chunk_index:04d}.wav"
                    target_chunk = output_dir / "audio" / "instrumental" / rel_track / f"chunk_{chunk_index:04d}.wav"
                    row = {
                        "track_id": rel_track.as_posix(),
                        "chunk_index": chunk_index,
                        "start": start,
                        "duration": chunk_seconds,
                        "bars_per_chunk": bars_per_chunk,
                        "bpm": bpm,
                        "input_audio": _relative(melody_chunk, output_dir),
                        "target_audio": _relative(target_chunk, output_dir),
                        "text": prompt,
                        "status": "planned" if dry_run else "ok",
                    }

                    if dry_run:
                        ok += 1
                        report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue

                    _write_chunk(melody_path, melody_chunk, start, chunk_seconds, sample_rate, overwrite)
                    _write_chunk(instrumental_path, target_chunk, start, chunk_seconds, sample_rate, overwrite)
                    metadata_file.write(json.dumps({k: row[k] for k in (
                        "input_audio", "target_audio", "text", "track_id", "chunk_index", "duration", "bpm", "bars_per_chunk"
                    )}, ensure_ascii=False) + "\n")
                    report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ok += 1

                if not starts:
                    skipped += 1
                    print(f"[skip] too short: {rel_track}")
                else:
                    print(f"[track] {track_index}/{len(melody_files)} {rel_track} chunks={len(starts)}")
    finally:
        if metadata_file is not None:
            metadata_file.close()

    print("")
    print("Done.")
    print(f"OK/planned rows: {ok}")
    print(f"Missing tracks:   {missing}")
    print(f"Skipped tracks:   {skipped}")
    print(f"Metadata:         {metadata_path.resolve()}")
    print(f"Report:           {report_path.resolve()}")


def _chunk_starts(duration: float, chunk_seconds: float, hop_seconds: float) -> list[float]:
    starts = []
    start = 0.0
    while start + chunk_seconds <= duration + 1e-6:
        starts.append(start)
        start += hop_seconds
    return starts


def _duration(path: Path) -> float:
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


def _write_chunk(source: Path, target: Path, start: float, duration: float, sample_rate: int, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(target),
        ],
        check=True,
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare bar-aligned paired MusicGen dataset.")
    parser.add_argument("--melody-root", type=Path, required=True)
    parser.add_argument("--instrumental-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=128.0)
    parser.add_argument("--bars-per-chunk", type=int, default=16)
    parser.add_argument("--hop-bars", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--melody-name", default="melody_piano.wav")
    parser.add_argument("--instrumental-name", default="instrumental.wav")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_bar_aligned_dataset(
        melody_root=args.melody_root,
        instrumental_root=args.instrumental_root,
        output_dir=args.output_dir,
        bpm=args.bpm,
        bars_per_chunk=args.bars_per_chunk,
        hop_bars=args.hop_bars,
        sample_rate=args.sample_rate,
        melody_name=args.melody_name,
        instrumental_name=args.instrumental_name,
        prompt=args.prompt,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
