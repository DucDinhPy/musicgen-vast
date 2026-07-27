#!/usr/bin/env python3
"""Smoke-check MusicGen melody-to-audio metadata rows."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def check_dataset(
    metadata: Path,
    dataset_root: Path,
    limit: int | None,
    random_sample: bool,
    seed: int,
    expected_sample_rate: int,
    expected_duration: float,
    duration_tolerance: float,
    min_rms_db: float,
) -> None:
    rows = _read_jsonl(metadata)
    if not rows:
        raise RuntimeError(f"No rows found in: {metadata}")

    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    print(f"Metadata:             {metadata.resolve()}")
    print(f"Dataset root:         {dataset_root.resolve()}")
    print(f"Rows checked:         {len(rows)}")
    print(f"Expected sample rate: {expected_sample_rate}")
    print(f"Expected duration:    {expected_duration:.2f}s")
    print(f"Duration tolerance:   {duration_tolerance:.2f}s")
    print(f"Min RMS warning:      {min_rms_db:.1f} dBFS")
    print("")

    ok_count = 0
    warning_count = 0
    error_count = 0

    for index, row in enumerate(rows, start=1):
        try:
            input_audio = _resolve_audio_path(row["input_audio"], dataset_root)
            target_audio = _resolve_audio_path(row["target_audio"], dataset_root)
            _check_exists(input_audio, "input_audio")
            _check_exists(target_audio, "target_audio")

            input_stats = _audio_stats(input_audio)
            target_stats = _audio_stats(target_audio)

            row_warnings = []
            for label, stats in (
                ("input_audio", input_stats),
                ("target_audio", target_stats),
            ):
                if stats["sample_rate"] != expected_sample_rate:
                    row_warnings.append(
                        f"{label} sample_rate={stats['sample_rate']}"
                    )
                if abs(stats["duration"] - expected_duration) > duration_tolerance:
                    row_warnings.append(
                        f"{label} duration={stats['duration']:.2f}s"
                    )
                if stats["rms_db"] < min_rms_db:
                    row_warnings.append(
                        f"{label} quiet rms={stats['rms_db']:.1f}dB"
                    )

            if row_warnings:
                warning_count += 1
                print(
                    f"[warn] {index}: {row.get('track_id', '(no track_id)')} "
                    f"chunk={row.get('chunk_index', '?')} | "
                    + "; ".join(row_warnings)
                )
            else:
                ok_count += 1
                print(
                    f"[ok] {index}: {row.get('track_id', '(no track_id)')} "
                    f"chunk={row.get('chunk_index', '?')}"
                )

        except Exception as exc:
            error_count += 1
            print(f"[error] {index}: {exc}")

    print("")
    print("Done.")
    print(f"OK:       {ok_count}")
    print(f"Warnings: {warning_count}")
    print(f"Errors:   {error_count}")

    if error_count:
        raise SystemExit(1)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def _resolve_audio_path(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _audio_stats(path: Path) -> dict[str, float]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: numpy/soundfile. Install with "
            "`python -m pip install numpy soundfile`."
        ) from exc

    info = sf.info(str(path))
    audio, _ = sf.read(str(path), always_2d=False)
    if audio.size == 0:
        rms_db = -math.inf
        peak_db = -math.inf
    else:
        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        rms_db = _to_db(rms)
        peak_db = _to_db(peak)

    return {
        "sample_rate": float(info.samplerate),
        "duration": float(info.frames / info.samplerate),
        "rms_db": rms_db,
        "peak_db": peak_db,
    }


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check MusicGen metadata and audio chunks."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Metadata JSONL file to check.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder used to resolve relative audio paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to check. Default: 10.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Shuffle rows before applying --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed. Default: 1337.",
    )
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=32000,
        help="Expected sample rate. Default: 32000.",
    )
    parser.add_argument(
        "--expected-duration",
        type=float,
        default=30.0,
        help="Expected chunk duration. Default: 30s.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=0.25,
        help="Allowed duration deviation in seconds. Default: 0.25.",
    )
    parser.add_argument(
        "--min-rms-db",
        type=float,
        default=-65.0,
        help="Warn if input/target RMS is quieter than this. Default: -65.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    check_dataset(
        metadata=args.metadata,
        dataset_root=args.dataset_root,
        limit=args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
        expected_sample_rate=args.expected_sample_rate,
        expected_duration=args.expected_duration,
        duration_tolerance=args.duration_tolerance,
        min_rms_db=args.min_rms_db,
    )


if __name__ == "__main__":
    main()
