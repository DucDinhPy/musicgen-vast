#!/usr/bin/env python3
"""Add frame-aligned vocal/melody timing conditions to a V5 manifest.

The existing V5 Beat This condition remains unchanged and ``target_audio``
remains the clean instrumental.  By default, timing is extracted from the
rendered melody in ``input_audio`` so this script works with the dataset that
already exists.  If a future manifest contains separated vocal chunks, pass
``--timing-audio-field vocal_audio``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from musicgen_v51_alignment import (
    V51_SCHEMA_VERSION,
    VOCAL_TIMING_FEATURE_NAMES,
    extract_vocal_timing_features,
    load_vocal_timing_condition,
    save_vocal_timing_condition,
)
from train_musicgen_melody_paired import _read_jsonl, _resolve_path
from train_musicgen_v5_beatthis import (
    V5_CONDITION_SCHEMA,
    V5_DETECTOR,
)


def prepare(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.metadata)
    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("No V5 rows selected.")

    output = args.output.resolve()
    condition_root = args.condition_root.resolve()
    dataset_root = args.dataset_root.resolve()
    report = output.with_suffix(output.suffix + ".report.jsonl")
    print(f"V5 metadata:       {args.metadata.resolve()}")
    print(f"Dataset root:      {dataset_root}")
    print(f"V5.1 metadata:     {output}")
    print(f"Condition root:    {condition_root}")
    print(f"Timing audio field:{args.timing_audio_field}")
    if args.stems_root is not None:
        print(f"Stems root:        {args.stems_root.resolve()}")
    print(f"Feature rate:      {args.feature_rate}")
    print(f"Rows selected:     {len(selected)} / {len(rows)}")
    print(f"Features:          {', '.join(VOCAL_TIMING_FEATURE_NAMES)}")
    print(f"Overwrite:         {args.overwrite}")
    print("")

    output.parent.mkdir(parents=True, exist_ok=True)
    condition_root.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(output.name + ".tmp")
    report_tmp = report.with_name(report.name + ".tmp")
    for path in (output_tmp, report_tmp):
        if path.exists():
            path.unlink()

    written = 0
    errors = 0
    try:
        with (
            output_tmp.open("w", encoding="utf-8") as output_handle,
            report_tmp.open("w", encoding="utf-8") as report_handle,
        ):
            for local_index, row in enumerate(selected):
                source_index = args.start_index + local_index
                try:
                    updated, condition_path = _prepare_row(
                        row=row,
                        row_index=source_index,
                        dataset_root=dataset_root,
                        condition_root=condition_root,
                        timing_audio_field=args.timing_audio_field,
                        stems_root=args.stems_root,
                        vocal_stem_name=args.vocal_stem_name,
                        chunk_hop_seconds=args.chunk_hop_seconds,
                        feature_rate=args.feature_rate,
                        overwrite=args.overwrite,
                    )
                    output_handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
                    report_handle.write(
                        json.dumps(
                            {
                                "source_row_index": source_index,
                                "track_id": updated["track_id"],
                                "chunk_index": updated["chunk_index"],
                                "status": "ok",
                                "timing_audio": updated["v51_timing_audio"],
                                "condition": updated["v51_timing_condition"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                    print(
                        f"[{local_index + 1}/{len(selected)}] "
                        f"{updated['track_id']} chunk={updated['chunk_index']} "
                        f"condition={condition_path.name}"
                    )
                except Exception as exc:
                    errors += 1
                    report_handle.write(
                        json.dumps(
                            {
                                "source_row_index": source_index,
                                "track_id": row.get("track_id"),
                                "chunk_index": row.get("chunk_index"),
                                "status": "error",
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    print(f"[error] row={source_index}: {exc}")
                    if not args.keep_going:
                        raise

        if written == 0:
            raise RuntimeError("No V5.1 rows were written.")
        os.replace(output_tmp, output)
        os.replace(report_tmp, report)
    except Exception:
        for path in (output_tmp, report_tmp):
            if path.exists():
                path.unlink()
        raise

    print("")
    print("Done.")
    print(f"Rows written: {written}")
    print(f"Errors:       {errors}")
    print(f"Metadata:     {output}")
    print(f"Report:       {report}")


def _prepare_row(
    row: dict[str, Any],
    row_index: int,
    dataset_root: Path,
    condition_root: Path,
    timing_audio_field: str,
    stems_root: Path | None,
    vocal_stem_name: str,
    chunk_hop_seconds: float,
    feature_rate: float,
    overwrite: bool,
) -> tuple[dict[str, Any], Path]:
    required = {
        "input_audio",
        "target_audio",
        "rhythm_condition",
        "track_id",
        "chunk_index",
        "duration",
        "v5_schema_version",
        "rhythm_detector",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise KeyError(f"V5 row {row_index} is missing fields: {missing}")
    if int(row["v5_schema_version"]) != V5_CONDITION_SCHEMA:
        raise ValueError(f"Row {row_index} is not V5 schema {V5_CONDITION_SCHEMA}")
    if str(row["rhythm_detector"]) != V5_DETECTOR:
        raise ValueError(f"Row {row_index} is not a Beat This V5 row")
    if timing_audio_field not in row and timing_audio_field != "source_vocal":
        raise KeyError(
            f"Row {row_index} has no {timing_audio_field!r}; use "
            "--timing-audio-field input_audio for the current dataset"
        )

    if timing_audio_field == "source_vocal":
        if stems_root is None:
            raise ValueError(
                "--stems-root is required with --timing-audio-field source_vocal"
            )
        source_audio = _resolve_source_vocal(
            stems_root=stems_root,
            track_id=str(row["track_id"]),
            vocal_stem_name=vocal_stem_name,
        )
        source_start_seconds = float(
            row.get("start", int(row["chunk_index"]) * chunk_hop_seconds)
        )
    else:
        source_audio = _resolve_path(row[timing_audio_field], dataset_root).resolve()
        source_start_seconds = 0.0
    if not source_audio.is_file():
        raise FileNotFoundError(f"Timing audio does not exist: {source_audio}")
    duration = float(row["duration"])
    track_id = str(row["track_id"])
    chunk_index = int(row["chunk_index"])
    condition_path = _condition_path(condition_root, track_id, chunk_index)

    if condition_path.exists() and not overwrite:
        load_vocal_timing_condition(
            condition_path,
            expected_rate=feature_rate,
            expected_duration=duration,
        )
        _validate_cached_source(
            condition_path,
            source_audio=source_audio,
            source_field=timing_audio_field,
            source_start_seconds=source_start_seconds,
        )
    else:
        features = extract_vocal_timing_features(
            source_audio,
            duration=duration,
            feature_rate=feature_rate,
            start_seconds=source_start_seconds,
        )
        save_vocal_timing_condition(
            condition_path,
            features=features,
            feature_rate=feature_rate,
            duration=duration,
            source_audio=source_audio,
            source_field=timing_audio_field,
            source_start_seconds=source_start_seconds,
        )

    updated = dict(row)
    updated["v51_schema_version"] = V51_SCHEMA_VERSION
    updated["v51_timing_audio_field"] = timing_audio_field
    updated["v51_timing_audio"] = _display_path(source_audio, dataset_root)
    updated["v51_timing_audio_start"] = float(source_start_seconds)
    updated["v51_timing_condition"] = _display_path(condition_path, dataset_root)
    updated["v51_timing_feature_rate"] = float(feature_rate)
    updated["v51_timing_feature_names"] = list(VOCAL_TIMING_FEATURE_NAMES)
    return updated, condition_path


def _validate_cached_source(
    condition_path: Path,
    source_audio: Path,
    source_field: str,
    source_start_seconds: float,
) -> None:
    import numpy as np

    with np.load(condition_path, allow_pickle=False) as data:
        cached_audio = str(data["source_audio"][0]) if "source_audio" in data else ""
        cached_field = str(data["source_field"][0]) if "source_field" in data else ""
        cached_start = (
            float(data["source_start_seconds"][0])
            if "source_start_seconds" in data
            else 0.0
        )
    if Path(cached_audio).resolve() != source_audio.resolve():
        raise RuntimeError(
            f"Cached V5.1 condition uses another audio file: {condition_path}. "
            "Re-run with --overwrite."
        )
    if cached_field != source_field or abs(cached_start - source_start_seconds) > 0.001:
        raise RuntimeError(
            f"Cached V5.1 source field/start changed: {condition_path}. "
            "Re-run with --overwrite."
        )


def _resolve_source_vocal(
    stems_root: Path,
    track_id: str,
    vocal_stem_name: str,
) -> Path:
    relative = Path(track_id.replace("\\", "/"))
    candidates = [
        stems_root / relative / f"{vocal_stem_name}.wav",
        stems_root / relative / "vocals.wav",
        stems_root / relative / "vocal.wav",
    ]
    if len(relative.parts) == 1:
        candidates.extend(
            [
                stems_root / "pre_audio_single" / relative / f"{vocal_stem_name}.wav",
                stems_root / "pre_audio_single" / relative / "vocals.wav",
                stems_root / "pre_audio_single" / relative / "vocal.wav",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find source vocal. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def _condition_path(root: Path, track_id: str, chunk_index: int) -> Path:
    safe_parts = []
    for part in Path(track_id.replace("\\", "/")).parts:
        if part in ("", ".", ".."):
            continue
        safe_parts.append("".join(c if c.isalnum() or c in "-_." else "_" for c in part))
    if not safe_parts:
        safe_parts = ["unknown_track"]
    path = root.joinpath(*safe_parts, f"chunk_{chunk_index:04d}.npz")
    resolved_parent = path.parent.resolve()
    try:
        resolved_parent.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe track_id: {track_id!r}") from exc
    return path


def _display_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MusicGen V5.1 vocal/melody timing conditions."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument(
        "--timing-audio-field",
        choices=["input_audio", "vocal_audio", "source_vocal"],
        default="input_audio",
    )
    parser.add_argument(
        "--stems-root",
        type=Path,
        default=None,
        help="Original stems root, required for source_vocal.",
    )
    parser.add_argument("--vocal-stem-name", default="vocal")
    parser.add_argument("--chunk-hop-seconds", type=float, default=30.0)
    parser.add_argument("--feature-rate", type=float, default=50.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path, label in (
        (args.metadata, "metadata"),
        (args.dataset_root, "dataset root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.feature_rate <= 0.0:
        raise ValueError("--feature-rate must be positive")
    if args.chunk_hop_seconds <= 0.0:
        raise ValueError("--chunk-hop-seconds must be positive")
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Invalid --start-index/--limit")
    prepare(args)


if __name__ == "__main__":
    main()
