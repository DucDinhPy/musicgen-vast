#!/usr/bin/env python3
"""Prepare preserve-timing vocal -> instrumental MusicGen training chunks.

This is the v3 dataset path for "vocal input -> background that follows vocal":

    vocal.wav + prompt with BPM/section labels -> instrumental.wav

Unlike the v2 bar-aligned dataset, this script does not warp audio to 128 BPM.
It keeps the source timing and cuts input/target chunks at identical timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy four-on-the-floor kick, rolling bass, bright synth chords, "
    "club build ups and drops, no lead vocal"
)


def prepare_dataset(
    stems_dir: Path,
    output_dir: Path,
    input_stems: list[str],
    target_stems: list[str],
    chunk_seconds: float,
    hop_seconds: float,
    min_chunk_seconds: float,
    sample_rate: int,
    prompt: str,
    min_target_rms_db: float,
    bpm_source: str,
    min_bpm_confidence: float,
    relative_to: Path | None,
    start_index: int,
    limit: int | None,
    overwrite: bool,
    keep_going: bool,
    dry_run: bool,
) -> None:
    if not stems_dir.exists():
        raise FileNotFoundError(f"Stems folder does not exist: {stems_dir}")
    if not stems_dir.is_dir():
        raise NotADirectoryError(f"Stems path must be a folder: {stems_dir}")

    track_dirs = _discover_track_dirs(stems_dir, input_stems, target_stems)
    track_dirs = track_dirs[start_index:]
    if limit is not None:
        track_dirs = track_dirs[:limit]
    if not track_dirs:
        raise RuntimeError(
            f"No track folders with input stems {input_stems} and target stems {target_stems} under: {stems_dir}"
        )

    metadata_path = output_dir / "metadata_vocal_instrumental.jsonl"
    report_path = output_dir / "prepare_vocal_preserve_report.jsonl"
    relative_root = relative_to or output_dir

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stems dir:         {stems_dir.resolve()}")
    print(f"Output dir:        {output_dir.resolve()}")
    print(f"Input stems:       {', '.join(input_stems)}")
    print(f"Target stems:      {', '.join(target_stems)}")
    print(f"Chunk seconds:     {chunk_seconds}")
    print(f"Hop seconds:       {hop_seconds}")
    print(f"Min chunk seconds: {min_chunk_seconds}")
    print(f"Sample rate:       {sample_rate}")
    print(f"BPM source:        {bpm_source}")
    print(f"Min BPM conf:      {min_bpm_confidence}")
    print(f"Relative to:       {relative_root.resolve()}")
    print(f"Tracks:            {len(track_dirs)}")
    print(f"Dry run:           {dry_run}")
    print("")

    ok_count = 0
    skipped_count = 0
    error_count = 0
    metadata_file = None if dry_run else metadata_path.open("w", encoding="utf-8")

    try:
        with report_path.open("w", encoding="utf-8") as report_file:
            for track_index, track_dir in enumerate(track_dirs, start=1):
                try:
                    input_path = _find_stem(track_dir, input_stems)
                    target_path = _find_stem(track_dir, target_stems)
                    if input_path is None or target_path is None:
                        skipped_count += 1
                        row = _track_report_row(stems_dir, track_dir, "missing_stem")
                        row["input"] = str(input_path) if input_path else None
                        row["target"] = str(target_path) if target_path else None
                        report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"[skip] missing stem: {track_dir}")
                        continue

                    input_duration = _duration(input_path)
                    target_duration = _duration(target_path)
                    usable_duration = min(input_duration, target_duration)
                    chunk_plan = _chunk_plan(usable_duration, chunk_seconds, hop_seconds, min_chunk_seconds)
                    if not chunk_plan:
                        skipped_count += 1
                        row = _track_report_row(stems_dir, track_dir, "too_short")
                        row.update({"input_duration": input_duration, "target_duration": target_duration})
                        report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"[skip] too short: {track_dir}")
                        continue

                    bpm_audio = target_path if bpm_source == "target" else input_path
                    bpm_row = _estimate_bpm(bpm_audio, min_bpm_confidence)
                    rel_track = track_dir.resolve().relative_to(stems_dir.resolve())
                    track_id = rel_track.as_posix()

                    for chunk_index, start, duration in chunk_plan:
                        input_chunk = output_dir / "audio" / "vocal" / rel_track / f"chunk_{chunk_index:04d}.wav"
                        target_chunk = output_dir / "audio" / "instrumental" / rel_track / f"chunk_{chunk_index:04d}.wav"

                        base_row = {
                            "track_id": track_id,
                            "chunk_index": chunk_index,
                            "start": start,
                            "duration": duration,
                            "bpm": bpm_row["bpm"],
                            "bpm_confidence": bpm_row["confidence"],
                            "bpm_needs_review": bpm_row["needs_review"],
                            "input_stem": input_path.stem,
                            "target_stem": target_path.stem,
                            "source_input": str(input_path),
                            "source_target": str(target_path),
                            "input_audio": _display_path(input_chunk, relative_root),
                            "target_audio": _display_path(target_chunk, relative_root),
                        }

                        if dry_run:
                            ok_count += 1
                            row = dict(base_row)
                            row["status"] = "planned"
                            report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            continue

                        _write_chunk(input_path, input_chunk, start, duration, sample_rate, overwrite)
                        _write_chunk(target_path, target_chunk, start, duration, sample_rate, overwrite)

                        target_stats = _audio_stats(target_chunk)
                        input_stats = _audio_stats(input_chunk)
                        if target_stats["rms_db"] < min_target_rms_db:
                            skipped_count += 1
                            row = dict(base_row)
                            row["status"] = "silent_target"
                            row["target_stats"] = target_stats
                            row["input_stats"] = input_stats
                            report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            print(
                                f"[skip] silent target {track_id} chunk={chunk_index} "
                                f"rms={target_stats['rms_db']:.1f} dBFS"
                            )
                            continue

                        active = _detect_active_intervals(input_chunk)
                        sections = _build_sections(active, duration)
                        structure_text = _structure_text(sections)
                        text = _build_prompt(prompt, float(bpm_row["bpm"]), structure_text)

                        row = dict(base_row)
                        row.update(
                            {
                                "text": text,
                                "structure_text": structure_text,
                                "sections": sections,
                                "vocal_active": active,
                                "input_stats": input_stats,
                                "target_stats": target_stats,
                                "status": "ok",
                            }
                        )
                        report_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                        metadata_row = {
                            "input_audio": row["input_audio"],
                            "target_audio": row["target_audio"],
                            "text": text,
                            "track_id": track_id,
                            "chunk_index": chunk_index,
                            "start": start,
                            "duration": duration,
                            "bpm": row["bpm"],
                            "bpm_confidence": row["bpm_confidence"],
                            "sections": sections,
                            "vocal_active": active,
                        }
                        metadata_file.write(json.dumps(metadata_row, ensure_ascii=False) + "\n")
                        ok_count += 1

                    print(f"[track] {track_index}/{len(track_dirs)} {track_id} chunks={len(chunk_plan)}")
                except Exception as exc:
                    error_count += 1
                    row = _track_report_row(stems_dir, track_dir, "error")
                    row["error"] = str(exc)
                    report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[error] {track_dir}: {exc}")
                    if not keep_going:
                        raise
    finally:
        if metadata_file is not None:
            metadata_file.close()

    print("")
    print("Done.")
    print(f"OK/planned rows: {ok_count}")
    print(f"Skipped rows:    {skipped_count}")
    print(f"Errors:          {error_count}")
    print(f"Metadata:        {metadata_path.resolve()}")
    print(f"Report:          {report_path.resolve()}")


def _discover_track_dirs(stems_dir: Path, input_stems: list[str], target_stems: list[str]) -> list[Path]:
    candidates: set[Path] = set()
    for stem in input_stems:
        for path in stems_dir.rglob(f"{stem}.wav"):
            if path.is_file():
                candidates.add(path.parent)
    return sorted(
        track_dir for track_dir in candidates
        if _find_stem(track_dir, target_stems) is not None
    )


def _find_stem(track_dir: Path, stems: list[str]) -> Path | None:
    for stem in stems:
        candidate = track_dir / f"{stem}.wav"
        if candidate.exists():
            return candidate
    return None


def _track_report_row(stems_dir: Path, track_dir: Path, status: str) -> dict:
    try:
        track_id = track_dir.resolve().relative_to(stems_dir.resolve()).as_posix()
    except ValueError:
        track_id = track_dir.name
    return {"track_id": track_id, "track_dir": str(track_dir), "status": status}


def _estimate_bpm(audio_path: Path, min_confidence: float) -> dict:
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
    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_count = int(len(beat_times))
    beat_density = min(1.0, beat_count / max(1.0, duration / 2.0))
    confidence = float(max(0.0, min(1.0, beat_density)))
    return {
        "bpm": tempo,
        "first_beat": float(beat_times[0]) if len(beat_times) else 0.0,
        "duration": duration,
        "beat_count": beat_count,
        "confidence": confidence,
        "needs_review": confidence < min_confidence,
    }


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
        chunks.append((chunk_index, round(start, 3), round(duration, 3)))
        chunk_index += 1
        start += hop_seconds
    return chunks


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
    )


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


def _audio_stats(path: Path) -> dict[str, float]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy/soundfile.") from exc

    audio, _ = sf.read(str(path), always_2d=False)
    if audio.size == 0:
        return {"rms_db": -math.inf, "peak_db": -math.inf}
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    return {"rms_db": _to_db(rms), "peak_db": _to_db(peak)}


def _detect_active_intervals(
    audio_path: Path,
    frame_seconds: float = 0.25,
    threshold_db: float = -45.0,
    min_active_seconds: float = 1.0,
    merge_gap_seconds: float = 1.0,
) -> list[list[float]]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy/soundfile.") from exc

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_size = max(1, int(frame_seconds * sr))
    active_frames = []
    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if _to_db(rms) >= threshold_db:
            active_frames.append((start / sr, min(len(audio), start + frame_size) / sr))

    intervals = _merge_intervals(active_frames, merge_gap_seconds)
    return [
        [round(start, 3), round(end, 3)]
        for start, end in intervals
        if end - start >= min_active_seconds
    ]


def _merge_intervals(intervals: list[tuple[float, float]], merge_gap_seconds: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_sections(active: list[list[float]], duration: float) -> list[list]:
    sections: list[list] = []
    cursor = 0.0
    if not active:
        return [["instrumental_intro", 0.0, round(duration, 3)]]

    for start, end in active:
        if start > cursor + 0.25:
            label = "intro" if not sections else "break_or_fill"
            sections.append([label, round(cursor, 3), round(start, 3)])
        sections.append(["vocal_backing", round(start, 3), round(end, 3)])
        cursor = end

    if cursor < duration - 0.25:
        sections.append(["transition_or_outro", round(cursor, 3), round(duration, 3)])
    return sections


def _structure_text(sections: list[list]) -> str:
    parts = [f"{label} {start:.1f}-{end:.1f}s" for label, start, end in sections]
    return "structure: " + ", ".join(parts)


def _build_prompt(base_prompt: str, bpm: float, structure_text: str) -> str:
    return f"{base_prompt}, {bpm:.1f} bpm, {structure_text}"


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _display_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def _parse_stems(value: str) -> list[str]:
    stems = [part.strip() for part in value.split(",") if part.strip()]
    if not stems:
        raise argparse.ArgumentTypeError("At least one stem name is required.")
    return stems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare v3 preserve-timing vocal -> instrumental MusicGen chunks."
    )
    parser.add_argument("--stems-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-stems", type=_parse_stems, default=_parse_stems("vocal,vocals"))
    parser.add_argument("--target-stems", type=_parse_stems, default=_parse_stems("instrumental"))
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--hop-seconds", type=float, default=30.0)
    parser.add_argument("--min-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--min-target-rms-db", type=float, default=-65.0)
    parser.add_argument("--bpm-source", choices=("target", "input"), default="target")
    parser.add_argument("--min-bpm-confidence", type=float, default=0.35)
    parser.add_argument("--relative-to", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(
        stems_dir=args.stems_dir,
        output_dir=args.output_dir,
        input_stems=args.input_stems,
        target_stems=args.target_stems,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        sample_rate=args.sample_rate,
        prompt=args.prompt,
        min_target_rms_db=args.min_target_rms_db,
        bpm_source=args.bpm_source,
        min_bpm_confidence=args.min_bpm_confidence,
        relative_to=args.relative_to,
        start_index=args.start_index,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
