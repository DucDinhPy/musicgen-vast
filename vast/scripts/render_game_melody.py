#!/usr/bin/env python3
"""
Render OpenVPI GAME melody MIDI files into piano WAV files.

This is the next step after `game_melody.py extract`.

Input layout:

    data/melody/game/
      set_01/
        song_id/
          melody.mid

Output layout:

    data/melody/rendered/
      set_01/
        song_id/
          melody_piano.wav

Example:

    cd /workspace/musicgen-vast
    python vast/scripts/render_game_melody.py \\
      --input-dir /workspace/musicgen-vast/data/melody/game \\
      --output-dir /workspace/musicgen-vast/data/melody/rendered \\
      --limit 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
)


def render_melodies(
    input_dir: Path,
    output_dir: Path,
    soundfont: Path | None,
    midi_name: str,
    output_name: str,
    raw_sample_rate: int,
    sample_rate: int,
    loudnorm: str,
    start_index: int,
    limit: int | None,
    overwrite: bool,
    keep_raw: bool,
    keep_going: bool,
    dry_run: bool,
    report: Path | None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a folder: {input_dir}")

    soundfont = soundfont or _discover_soundfont()
    if not soundfont.exists():
        raise FileNotFoundError(f"SoundFont does not exist: {soundfont}")

    midi_files = sorted(input_dir.rglob(midi_name))
    midi_files = midi_files[start_index:]
    if limit is not None:
        midi_files = midi_files[:limit]
    if not midi_files:
        raise RuntimeError(f"No {midi_name} files found under: {input_dir}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path = report or (output_dir / "render_game_melody_report.jsonl")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input dir:       {input_dir.resolve()}")
    print(f"Output dir:      {output_dir.resolve()}")
    print(f"SoundFont:       {soundfont.resolve()}")
    print(f"MIDI files:      {len(midi_files)}")
    print(f"Raw sample rate: {raw_sample_rate}")
    print(f"Sample rate:     {sample_rate}")
    print(f"Loudnorm:        {loudnorm}")
    print(f"Start index:     {start_index}")
    print(f"Limit:           {limit if limit is not None else '(none)'}")
    print(f"Overwrite:       {overwrite}")
    print(f"Keep raw:        {keep_raw}")
    print(f"Dry run:         {dry_run}")
    print(f"Report:          {report_path.resolve()}")
    print("")

    ok_count = 0
    skip_count = 0
    error_count = 0

    with report_path.open("w", encoding="utf-8") as report_file:
        for index, midi_path in enumerate(midi_files, start=1):
            target_dir = output_dir / midi_path.parent.relative_to(input_dir)
            output_wav = target_dir / output_name
            raw_wav = target_dir / output_name.replace(".wav", "_raw.wav")

            row = {
                "midi": str(midi_path),
                "output_wav": str(output_wav),
                "raw_wav": str(raw_wav),
                "status": "planned" if dry_run else "ok",
            }

            if output_wav.exists() and not overwrite:
                skip_count += 1
                row["status"] = "skipped_exists"
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[skip] {index}/{len(midi_files)} exists: {output_wav}")
                continue

            if dry_run:
                ok_count += 1
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[plan] {index}/{len(midi_files)} {midi_path} -> {output_wav}")
                continue

            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                _render_one(
                    midi_path=midi_path,
                    raw_wav=raw_wav,
                    output_wav=output_wav,
                    soundfont=soundfont,
                    raw_sample_rate=raw_sample_rate,
                    sample_rate=sample_rate,
                    loudnorm=loudnorm,
                    overwrite=overwrite,
                )
                if not keep_raw and raw_wav.exists():
                    raw_wav.unlink()
            except subprocess.CalledProcessError as exc:
                error_count += 1
                row["status"] = "error"
                row["error"] = f"Command failed with exit code {exc.returncode}"
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[error] {index}/{len(midi_files)} {midi_path}: {row['error']}")
                if not keep_going:
                    raise
                continue

            ok_count += 1
            report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ok] {index}/{len(midi_files)} {midi_path.name} -> {output_wav}")

    print("")
    print("Done.")
    print(f"OK/planned: {ok_count}")
    print(f"Skipped:    {skip_count}")
    print(f"Errors:     {error_count}")


def _render_one(
    midi_path: Path,
    raw_wav: Path,
    output_wav: Path,
    soundfont: Path,
    raw_sample_rate: int,
    sample_rate: int,
    loudnorm: str,
    overwrite: bool,
) -> None:
    if overwrite:
        for path in (raw_wav, output_wav):
            if path.exists():
                path.unlink()

    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            str(soundfont),
            str(midi_path),
            "-F",
            str(raw_wav),
            "-r",
            str(raw_sample_rate),
        ],
        check=True,
    )

    audio_filter = f"loudnorm={loudnorm}" if loudnorm else "anull"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-af",
            audio_filter,
            str(output_wav),
        ],
        check=True,
    )


def _discover_soundfont() -> Path:
    for candidate in DEFAULT_SOUNDFONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    search_roots = [Path("/usr/share/sounds"), Path("/usr/share/soundfonts")]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob("*.sf2"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No .sf2 SoundFont found. Install one with: "
        "apt-get install -y fluidsynth fluid-soundfont-gm"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render GAME melody.mid files to piano WAV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing GAME melody.mid files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root for rendered melody_piano.wav files.",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="Path to .sf2 SoundFont. Default: auto-detect FluidR3_GM.",
    )
    parser.add_argument(
        "--midi-name",
        default="melody.mid",
        help="MIDI filename to search for. Default: melody.mid.",
    )
    parser.add_argument(
        "--output-name",
        default="melody_piano.wav",
        help="Rendered WAV filename. Default: melody_piano.wav.",
    )
    parser.add_argument(
        "--raw-sample-rate",
        type=int,
        default=44100,
        help="FluidSynth render sample rate. Default: 44100.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Final WAV sample rate. Default: 32000 for MusicGen.",
    )
    parser.add_argument(
        "--loudnorm",
        default="I=-20:TP=-2:LRA=11",
        help="ffmpeg loudnorm options. Use empty string to disable.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many discovered MIDI files. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most this many MIDI files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rendered WAV files.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep intermediate *_raw.wav files.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a render failure.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without rendering audio.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSONL path. Default: OUTPUT_DIR/render_game_melody_report.jsonl.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render_melodies(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        soundfont=args.soundfont,
        midi_name=args.midi_name,
        output_name=args.output_name,
        raw_sample_rate=args.raw_sample_rate,
        sample_rate=args.sample_rate,
        loudnorm=args.loudnorm,
        start_index=args.start_index,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_raw=args.keep_raw,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
        report=args.report,
    )


if __name__ == "__main__":
    main()
