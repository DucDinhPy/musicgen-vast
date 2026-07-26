#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


STEM_NAMES = {
    "vocals",
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
    "instrumental",
}


def organize_flat_stems(
    input_dir: Path,
    output_dir: Path,
    move: bool,
    overwrite: bool,
    dry_run: bool,
    report: Path,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a folder: {input_dir}")

    files = [
        path
        for path in sorted(input_dir.glob("*.wav"))
        if path.is_file()
    ]
    if not files:
        raise RuntimeError(f"No flat WAV files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:     {input_dir.resolve()}")
    print(f"Output:    {output_dir.resolve()}")
    print(f"Files:     {len(files)}")
    print(f"Move:      {move}")
    print(f"Overwrite: {overwrite}")
    print(f"Dry run:   {dry_run}")
    print(f"Report:    {report.resolve()}")

    ok_count = 0
    skip_count = 0
    error_count = 0

    with report.open("w", encoding="utf-8") as report_file:
        for source in files:
            try:
                target = _target_path(source, output_dir)
            except ValueError as exc:
                error_count += 1
                row = {
                    "source": str(source),
                    "target": None,
                    "status": "error",
                    "error": str(exc),
                }
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[error] {source.name}: {exc}")
                continue

            row = {
                "source": str(source),
                "target": str(target),
                "status": "planned" if dry_run else "ok",
                "mode": "move" if move else "copy",
            }

            if target.exists() and not overwrite:
                skip_count += 1
                row["status"] = "skipped_exists"
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[skip] Exists: {target}")
                continue

            if dry_run:
                ok_count += 1
                report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[plan] {source.name} -> {target}")
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and overwrite:
                target.unlink()

            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)

            ok_count += 1
            report_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ok] {source.name} -> {target}")

    print("")
    print("Done.")
    print(f"OK/planned: {ok_count}")
    print(f"Skipped:    {skip_count}")
    print(f"Errors:     {error_count}")


def _target_path(source: Path, output_dir: Path) -> Path:
    song_id, stem_name = _split_stem_name(source.stem)
    song_base, chunk_id = _split_chunk(song_id)

    if chunk_id is not None:
        return output_dir / song_base / f"chunk_{chunk_id:04d}" / f"{stem_name}.wav"
    return output_dir / song_id / f"{stem_name}.wav"


def _split_stem_name(stem: str) -> tuple[str, str]:
    for stem_name in sorted(STEM_NAMES, key=len, reverse=True):
        suffix = f"_{stem_name}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], stem_name
    raise ValueError(f"Cannot detect stem suffix from filename: {stem}")


def _split_chunk(song_id: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<song>.+)_chunk_(?P<chunk>\d+)$", song_id)
    if not match:
        return song_id, None
    return match.group("song"), int(match.group("chunk"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Organize old flat BS-RoFormer stem outputs into per-song folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Flat folder containing files like song_bass.wav, song_drums.wav.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Organized output folder.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying. Default is copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing organized files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy/move files. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("organize_flat_stems_report.jsonl"),
        help="JSONL report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    organize_flat_stems(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        move=args.move,
        overwrite=args.overwrite,
        dry_run=not args.apply,
        report=args.report,
    )


if __name__ == "__main__":
    main()
