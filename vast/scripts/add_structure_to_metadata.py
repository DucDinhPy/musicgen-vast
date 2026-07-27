#!/usr/bin/env python3
"""Add simple structure labels to MusicGen metadata based on melody activity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def add_structure_to_metadata(
    input_metadata: Path,
    output_metadata: Path,
    dataset_root: Path,
    frame_seconds: float,
    threshold_db: float,
    min_active_seconds: float,
    merge_gap_seconds: float,
    overwrite_text: bool,
    limit: int | None,
) -> None:
    rows = _read_jsonl(input_metadata)
    if limit is not None:
        rows = rows[:limit]

    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:              {input_metadata.resolve()}")
    print(f"Output:             {output_metadata.resolve()}")
    print(f"Dataset root:       {dataset_root.resolve()}")
    print(f"Rows:               {len(rows)}")
    print(f"Frame seconds:      {frame_seconds}")
    print(f"Threshold dBFS:     {threshold_db}")
    print(f"Min active seconds: {min_active_seconds}")
    print(f"Merge gap seconds:  {merge_gap_seconds}")
    print(f"Overwrite text:     {overwrite_text}")
    print("")

    with output_metadata.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            melody_path = _resolve_path(row["input_audio"], dataset_root)
            duration = float(row.get("duration", 30.0))
            active = _detect_active_intervals(
                melody_path=melody_path,
                frame_seconds=frame_seconds,
                threshold_db=threshold_db,
                min_active_seconds=min_active_seconds,
                merge_gap_seconds=merge_gap_seconds,
            )
            sections = _build_sections(active, duration)
            structure_text = _structure_text(sections)

            row["melody_active"] = active
            row["sections"] = sections
            row["structure_text"] = structure_text

            base_text = row.get("text", "")
            if overwrite_text:
                row["text"] = structure_text
            else:
                row["text"] = f"{base_text}, {structure_text}" if base_text else structure_text

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row.get('track_id', '')} chunk={row.get('chunk_index', '?')} {structure_text}")


def _detect_active_intervals(
    melody_path: Path,
    frame_seconds: float,
    threshold_db: float,
    min_active_seconds: float,
    merge_gap_seconds: float,
) -> list[list[float]]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy/soundfile.") from exc

    audio, sr = sf.read(str(melody_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_size = max(1, int(frame_seconds * sr))

    active_frames = []
    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        db = _to_db(rms)
        if db >= threshold_db:
            active_frames.append((start / sr, min(len(audio), start + frame_size) / sr))

    intervals = _merge_intervals(active_frames, merge_gap_seconds)
    intervals = [
        [round(start, 3), round(end, 3)]
        for start, end in intervals
        if end - start >= min_active_seconds
    ]
    return intervals


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

    for index, (start, end) in enumerate(active):
        if start > cursor + 0.25:
            label = "intro" if not sections else "break_or_fill"
            sections.append([label, round(cursor, 3), round(start, 3)])
        sections.append(["melody_backing", round(start, 3), round(end, 3)])
        cursor = end

    if cursor < duration - 0.25:
        sections.append(["transition_or_outro", round(cursor, 3), round(duration, 3)])

    return sections


def _structure_text(sections: list[list]) -> str:
    parts = [
        f"{label} {start:.1f}-{end:.1f}s"
        for label, start, end in sections
    ]
    return "structure: " + ", ".join(parts)


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add structure labels to MusicGen metadata.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--threshold-db", type=float, default=-45.0)
    parser.add_argument("--min-active-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite-text", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    add_structure_to_metadata(
        input_metadata=args.input,
        output_metadata=args.output,
        dataset_root=args.dataset_root,
        frame_seconds=args.frame_seconds,
        threshold_db=args.threshold_db,
        min_active_seconds=args.min_active_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        overwrite_text=args.overwrite_text,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
