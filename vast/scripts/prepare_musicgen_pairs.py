#!/usr/bin/env python3
"""
Build a MusicGen-style paired dataset manifest from separated stems.

The first useful training target for this project is:

    input_audio + text_prompt -> target_audio

For BS-RoFormer outputs, a practical v1 pairing is:

    input_audio  = vocals.wav
    target_audio = instrumental.wav

This lets the model learn "given the song melody/vocal contour, generate a
Vinahouse backing track". Later, `vocals.wav` can be replaced with a clean
piano melody or another melody stem.

Example:

    cd /workspace/musicgen-vast
    python vast/scripts/prepare_musicgen_pairs.py \\
      --stems-dir /workspace/musicgen-vast/data/stems/stems_set_single \\
      --output /workspace/musicgen-vast/data/datasets/accompaniment_pairs/metadata.jsonl \\
      --relative-to /workspace/musicgen-vast \\
      --prompt "add clean energetic vinahouse instrumental backing, punchy kick, rolling bass, bright synth chords, no vocals"

Dry-run first:

    python vast/scripts/prepare_musicgen_pairs.py \\
      --stems-dir /workspace/musicgen-vast/data/stems/stems_set_single \\
      --output /workspace/musicgen-vast/data/datasets/accompaniment_pairs/metadata.jsonl \\
      --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PROMPT = (
    "add clean energetic vinahouse instrumental backing, punchy kick, "
    "rolling bass, bright synth chords, no vocals"
)


def prepare_pairs(
    stems_dir: Path,
    output: Path,
    input_stem: str,
    target_stem: str,
    prompt: str,
    relative_to: Path | None,
    min_duration: float,
    max_duration: float | None,
    duration_tolerance: float,
    dry_run: bool,
) -> None:
    if not stems_dir.exists():
        raise FileNotFoundError(f"Stems folder does not exist: {stems_dir}")
    if not stems_dir.is_dir():
        raise NotADirectoryError(f"Stems path must be a folder: {stems_dir}")

    input_name = f"{input_stem}.wav"
    target_name = f"{target_stem}.wav"
    input_files = sorted(stems_dir.rglob(input_name))

    if not input_files:
        raise RuntimeError(f"No {input_name} files found under: {stems_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(output.suffix + ".report.jsonl")

    print(f"Stems dir:          {stems_dir.resolve()}")
    print(f"Output:             {output.resolve()}")
    print(f"Report:             {report_path.resolve()}")
    print(f"Input stem:         {input_stem}")
    print(f"Target stem:        {target_stem}")
    print(f"Min duration:       {min_duration:.2f}s")
    print(f"Max duration:       {max_duration if max_duration is not None else '(none)'}")
    print(f"Duration tolerance: {duration_tolerance:.2f}s")
    print(f"Dry run:            {dry_run}")
    print("")

    ok_count = 0
    skipped_count = 0

    metadata_file = None if dry_run else output.open("w", encoding="utf-8")
    try:
        with report_path.open("w", encoding="utf-8") as report_file:
            for input_path in input_files:
                track_dir = input_path.parent
                target_path = track_dir / target_name

                row = {
                    "track_dir": _display_path(track_dir, relative_to),
                    "input_audio": _display_path(input_path, relative_to),
                    "target_audio": _display_path(target_path, relative_to),
                    "text": prompt,
                    "input_stem": input_stem,
                    "target_stem": target_stem,
                }

                status, error = _validate_pair(
                    input_path=input_path,
                    target_path=target_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    duration_tolerance=duration_tolerance,
                    row=row,
                )
                row["status"] = status
                if error:
                    row["error"] = error

                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                if status != "ok":
                    skipped_count += 1
                    print(f"[skip] {track_dir.name}: {error}")
                    continue

                ok_count += 1
                if metadata_file is not None:
                    metadata = {
                        "input_audio": row["input_audio"],
                        "target_audio": row["target_audio"],
                        "text": prompt,
                        "duration": row["duration"],
                        "track_id": _track_id(track_dir, stems_dir),
                    }
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                print(f"[ok] {track_dir.name}: {row['duration']:.2f}s")
    finally:
        if metadata_file is not None:
            metadata_file.close()

    print("")
    print("Done.")
    print(f"Pairs written/planned: {ok_count}")
    print(f"Skipped:               {skipped_count}")
    if dry_run:
        print("Dry run only: metadata file was not written.")


def _validate_pair(
    input_path: Path,
    target_path: Path,
    min_duration: float,
    max_duration: float | None,
    duration_tolerance: float,
    row: dict[str, object],
) -> tuple[str, str | None]:
    if not target_path.exists():
        return "missing_target", f"Missing target stem: {target_path.name}"

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: soundfile. Install it with `python -m pip install soundfile`."
        ) from exc

    try:
        input_info = sf.info(str(input_path))
        target_info = sf.info(str(target_path))
    except RuntimeError as exc:
        return "audio_error", str(exc)

    input_duration = input_info.frames / input_info.samplerate
    target_duration = target_info.frames / target_info.samplerate
    duration = min(input_duration, target_duration)

    row["duration"] = duration
    row["input_duration"] = input_duration
    row["target_duration"] = target_duration
    row["sample_rate"] = target_info.samplerate

    if duration < min_duration:
        return "too_short", f"Duration {duration:.2f}s is shorter than {min_duration:.2f}s"
    if max_duration is not None and duration > max_duration:
        return "too_long", f"Duration {duration:.2f}s is longer than {max_duration:.2f}s"
    if abs(input_duration - target_duration) > duration_tolerance:
        return (
            "duration_mismatch",
            f"Input {input_duration:.2f}s vs target {target_duration:.2f}s",
        )

    return "ok", None


def _display_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def _track_id(track_dir: Path, stems_dir: Path) -> str:
    try:
        return "/".join(track_dir.resolve().relative_to(stems_dir.resolve()).parts)
    except ValueError:
        return track_dir.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create metadata.jsonl pairs from organized BS-RoFormer stems."
    )
    parser.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Root folder containing organized stem folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write metadata.jsonl.",
    )
    parser.add_argument(
        "--input-stem",
        default="vocals",
        help="Input condition stem name without .wav. Default: vocals.",
    )
    parser.add_argument(
        "--target-stem",
        default="instrumental",
        help="Target output stem name without .wav. Default: instrumental.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt stored for every pair.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Store audio paths relative to this folder.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Skip pairs shorter than this many seconds. Default: 5.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Skip pairs longer than this many seconds. Default: no limit.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=1.0,
        help="Skip if input/target durations differ by more than this. Default: 1s.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and write only the report; do not write metadata output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_pairs(
        stems_dir=args.stems_dir,
        output=args.output,
        input_stem=args.input_stem,
        target_stem=args.target_stem,
        prompt=args.prompt,
        relative_to=args.relative_to,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        duration_tolerance=args.duration_tolerance,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
