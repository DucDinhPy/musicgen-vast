#!/usr/bin/env python3
"""Quantize GAME melody MIDI files to a fixed BPM grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def quantize_game_midi(
    report: Path,
    midi_root: Path,
    output_root: Path,
    midi_name: str,
    target_bpm: float | None,
    grid_division: int,
    min_note_seconds: float,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pretty_midi. Install with `python -m pip install pretty_midi`.") from exc

    rows = _read_jsonl(report)
    if limit is not None:
        rows = rows[:limit]
    if not midi_root.exists():
        raise FileNotFoundError(f"MIDI root does not exist: {midi_root}")
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    print(f"Report:        {report.resolve()}")
    print(f"MIDI root:     {midi_root.resolve()}")
    print(f"Output root:   {output_root.resolve()}")
    print(f"MIDI name:     {midi_name}")
    print(f"Grid division: {grid_division}")
    print(f"Dry run:       {dry_run}")
    print("")

    ok = 0
    skipped = 0
    missing = 0

    for index, row in enumerate(rows, start=1):
        track_id = row["track_id"]
        src = midi_root / track_id / midi_name
        dst = output_root / track_id / midi_name

        if not src.exists():
            missing += 1
            print(f"[missing] {track_id}: {src}")
            continue
        if dst.exists() and not overwrite:
            skipped += 1
            print(f"[skip] exists: {dst}")
            continue

        bpm = float(target_bpm if target_bpm is not None else row["target_bpm"])
        downbeat = max(0.0, float(row.get("first_downbeat", 0.0)))

        if dry_run:
            ok += 1
            print(f"[plan] {track_id}: bpm={bpm:.2f}, downbeat={downbeat:.3f}")
            continue

        midi = pretty_midi.PrettyMIDI(str(src))
        _quantize_midi(
            midi=midi,
            target_bpm=bpm,
            downbeat=downbeat,
            grid_division=grid_division,
            min_note_seconds=min_note_seconds,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(dst))
        ok += 1
        print(f"[ok] {index}/{len(rows)} {track_id} -> {dst}")

    print("")
    print("Done.")
    print(f"OK/planned: {ok}")
    print(f"Skipped:    {skipped}")
    print(f"Missing:    {missing}")


def _quantize_midi(
    midi,
    target_bpm: float,
    downbeat: float,
    grid_division: int,
    min_note_seconds: float,
) -> None:
    beat_seconds = 60.0 / target_bpm
    grid_seconds = beat_seconds / grid_division

    for instrument in midi.instruments:
        new_notes = []
        for note in instrument.notes:
            start = _quantize_time(note.start - downbeat, grid_seconds)
            end = _quantize_time(note.end - downbeat, grid_seconds)
            start = max(0.0, start)
            end = max(start + min_note_seconds, end)
            note.start = start
            note.end = end
            new_notes.append(note)
        instrument.notes = new_notes


def _quantize_time(value: float, grid_seconds: float) -> float:
    return round(value / grid_seconds) * grid_seconds


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize GAME MIDI files to a fixed BPM grid.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--midi-name", default="melody.mid")
    parser.add_argument("--target-bpm", type=float, default=None)
    parser.add_argument("--grid-division", type=int, default=4, help="4 means 16th-note grid in 4/4.")
    parser.add_argument("--min-note-seconds", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    quantize_game_midi(
        report=args.report,
        midi_root=args.midi_root,
        output_root=args.output_root,
        midi_name=args.midi_name,
        target_bpm=args.target_bpm,
        grid_division=args.grid_division,
        min_note_seconds=args.min_note_seconds,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
