#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif"}


def prune_silent_audio(
    input_dir: Path,
    extensions: set[str],
    rms_threshold_db: float,
    peak_threshold_db: float,
    move_dir: Path | None,
    delete: bool,
    dry_run: bool,
    report_path: Path,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a folder: {input_dir}")

    audio_files = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not audio_files:
        raise RuntimeError(f"No audio files found in: {input_dir}")

    if delete and move_dir is not None:
        raise RuntimeError("Use either --delete or --move-dir, not both.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if move_dir is not None:
        move_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:             {input_dir.resolve()}")
    print(f"Files:             {len(audio_files)}")
    print(f"RMS threshold:     {rms_threshold_db:.1f} dBFS")
    print(f"Peak threshold:    {peak_threshold_db:.1f} dBFS")
    print(f"Move dir:          {move_dir.resolve() if move_dir else '(none)'}")
    print(f"Delete:            {delete}")
    print(f"Dry run:           {dry_run}")
    print(f"Report:            {report_path.resolve()}")

    silent_count = 0
    kept_count = 0

    with report_path.open("w", encoding="utf-8") as report:
        for index, path in enumerate(audio_files, start=1):
            stats = _audio_stats(path)
            is_silent = (
                stats["rms_db"] <= rms_threshold_db
                and stats["peak_db"] <= peak_threshold_db
            )

            action = "keep"
            if is_silent:
                silent_count += 1
                action = _handle_silent_file(
                    path=path,
                    input_dir=input_dir,
                    move_dir=move_dir,
                    delete=delete,
                    dry_run=dry_run,
                )
            else:
                kept_count += 1

            row = {
                "index": index,
                "path": str(path),
                "duration_sec": stats["duration_sec"],
                "sample_rate": stats["sample_rate"],
                "channels": stats["channels"],
                "rms_db": stats["rms_db"],
                "peak_db": stats["peak_db"],
                "is_silent": is_silent,
                "action": action,
            }
            report.write(json.dumps(row, ensure_ascii=False) + "\n")

            status = "SILENT" if is_silent else "KEEP"
            print(
                f"[{index}/{len(audio_files)}] {status:6} "
                f"rms={stats['rms_db']:7.2f}dB "
                f"peak={stats['peak_db']:7.2f}dB "
                f"{path}"
            )

    print("")
    print("Done.")
    print(f"Silent candidates: {silent_count}")
    print(f"Kept:              {kept_count}")
    if dry_run:
        print("Dry run only. Re-run with --move-dir or --delete to apply changes.")


def _audio_stats(path: Path, blocksize: int = 262_144) -> dict:
    total_square = 0.0
    total_samples = 0
    peak = 0.0

    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        channels = audio.channels
        frames = len(audio)

        for block in audio.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            abs_block = np.abs(block)
            peak = max(peak, float(abs_block.max(initial=0.0)))
            total_square += float(np.square(block).sum())
            total_samples += int(block.size)

    if total_samples == 0:
        rms = 0.0
    else:
        rms = math.sqrt(total_square / total_samples)

    return {
        "duration_sec": frames / sample_rate if sample_rate else 0.0,
        "sample_rate": sample_rate,
        "channels": channels,
        "rms_db": _linear_to_db(rms),
        "peak_db": _linear_to_db(peak),
    }


def _handle_silent_file(
    path: Path,
    input_dir: Path,
    move_dir: Path | None,
    delete: bool,
    dry_run: bool,
) -> str:
    if dry_run:
        return "dry-run"

    if delete:
        path.unlink()
        return "deleted"

    if move_dir is not None:
        relative = path.relative_to(input_dir)
        target = move_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        return f"moved:{target}"

    return "candidate"


def _linear_to_db(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value)


def _parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.add(item)
    return extensions or set(DEFAULT_EXTENSIONS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and remove/move silent or near-silent audio stem files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing stem audio files.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated audio extensions to scan.",
    )
    parser.add_argument(
        "--rms-threshold-db",
        type=float,
        default=-60.0,
        help="Files with RMS <= this dBFS and peak <= peak threshold are silent.",
    )
    parser.add_argument(
        "--peak-threshold-db",
        type=float,
        default=-45.0,
        help="Files with peak <= this dBFS and RMS <= RMS threshold are silent.",
    )
    parser.add_argument(
        "--move-dir",
        type=Path,
        default=None,
        help="Move silent files here instead of deleting them.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete silent files. Dangerous; prefer --move-dir first.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply move/delete. Without this flag, the script is dry-run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("silent_audio_report.jsonl"),
        help="JSONL report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prune_silent_audio(
        input_dir=args.input_dir,
        extensions=_parse_extensions(args.extensions),
        rms_threshold_db=args.rms_threshold_db,
        peak_threshold_db=args.peak_threshold_db,
        move_dir=args.move_dir,
        delete=args.delete,
        dry_run=not args.apply,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
