#!/usr/bin/env python3
"""Warp audio files to target BPM and align first downbeat to t=0."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def warp_audio_to_bpm(
    report: Path,
    source_root: Path,
    output_root: Path,
    audio_name: str,
    target_bpm: float | None,
    sample_rate: int,
    channels: int,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    rows = _read_jsonl(report)
    if limit is not None:
        rows = rows[:limit]
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    print(f"Report:      {report.resolve()}")
    print(f"Source root: {source_root.resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Audio name:  {audio_name}")
    print(f"Target BPM:  {target_bpm if target_bpm is not None else 'from report'}")
    print(f"Sample rate: {sample_rate}")
    print(f"Channels:    {channels}")
    print(f"Dry run:     {dry_run}")
    print("")

    ok = 0
    skipped = 0
    missing = 0

    for index, row in enumerate(rows, start=1):
        track_id = row["track_id"]
        src = _resolve_source(source_root, track_id, audio_name)
        dst = output_root / track_id / audio_name

        if not src.exists():
            missing += 1
            print(f"[missing] {track_id}: {src}")
            continue
        if dst.exists() and not overwrite:
            skipped += 1
            print(f"[skip] exists: {dst}")
            continue

        src_bpm = float(row["bpm"])
        dst_bpm = float(target_bpm if target_bpm is not None else row["target_bpm"])
        first_downbeat = max(0.0, float(row.get("first_downbeat", 0.0)))
        atempo = dst_bpm / src_bpm
        filters = _atempo_filters(atempo)

        if dry_run:
            ok += 1
            print(f"[plan] {track_id}: bpm {src_bpm:.2f}->{dst_bpm:.2f}, trim {first_downbeat:.3f}s")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{first_downbeat:.6f}",
            "-i",
            str(src),
            "-af",
            filters,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        ok += 1
        print(f"[ok] {index}/{len(rows)} {track_id} -> {dst}")

    print("")
    print("Done.")
    print(f"OK/planned: {ok}")
    print(f"Skipped:    {skipped}")
    print(f"Missing:    {missing}")


def _atempo_filters(factor: float) -> str:
    if factor <= 0:
        raise ValueError(f"Invalid atempo factor: {factor}")
    factors = []
    while factor < 0.5:
        factors.append(0.5)
        factor /= 0.5
    while factor > 2.0:
        factors.append(2.0)
        factor /= 2.0
    factors.append(factor)
    return ",".join(f"atempo={value:.8f}" for value in factors)


def _resolve_source(source_root: Path, track_id: str, audio_name: str) -> Path:
    direct = source_root / track_id / audio_name
    if direct.exists():
        return direct

    rel = Path(track_id)
    if len(rel.parts) == 2 and rel.parts[0] == "pre_audio_single":
        fallback = source_root / rel.parts[1] / audio_name
        if fallback.exists():
            return fallback

    return direct


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warp audio to target BPM using a BPM grid report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audio-name", default="instrumental.wav")
    parser.add_argument("--target-bpm", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    warp_audio_to_bpm(
        report=args.report,
        source_root=args.source_root,
        output_root=args.output_root,
        audio_name=args.audio_name,
        target_bpm=args.target_bpm,
        sample_rate=args.sample_rate,
        channels=args.channels,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
