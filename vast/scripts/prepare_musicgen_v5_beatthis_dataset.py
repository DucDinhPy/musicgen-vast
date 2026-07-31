#!/usr/bin/env python3
"""Prepare an independent V5 Beat This! dataset from V1 paired metadata.

V5 starts from the stable V1 objective and does not depend on V4 code or
metadata:

    melody_piano.wav + text -> instrumental.wav

This script validates that the input is clean V1 metadata, analyzes the
conditioning audio with Beat This!, writes raw beat/downbeat events to one NPZ
file per row, and creates canonical rhythm fields plus namespaced ``v5_*``
fields in a new JSONL manifest. The V5 text prompt is built from the original
V1 prompt and the BPM estimated by Beat This!.

By default, Beat This! analyzes ``target_audio`` (the clean instrumental) so
the supervised V5 rhythm labels come from the complete backing track rather
than the sparse melody piano. Use ``--analysis-audio input`` only for an
explicit inference-side experiment.

The condition file intentionally stores events rather than a V4-style feature
grid. A future V5 trainer can choose its own event encoding without depending
on the V4 rhythm conditioner.

Vast.ai example:

    source /workspace/.venv-musicgen/bin/activate
    python -m pip install "beat-this==1.1.0"

    DATASET_ROOT=/workspace/musicgen-vast/vast_data/data/data/melody_instrumental_v1_base

    python vast/scripts/prepare_musicgen_v5_beatthis_dataset.py \
      --metadata "$DATASET_ROOT/metadata_instrumental.jsonl" \
      --dataset-root "$DATASET_ROOT" \
      --output "$DATASET_ROOT/metadata_instrumental_v5_beatthis.jsonl" \
      --condition-root "$DATASET_ROOT/v5_beatthis" \
      --analysis-audio target \
      --device cuda \
      --overwrite

Beat This! model weights are downloaded automatically on first use.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 2
DETECTOR_NAME = "beat_this"
DEFAULT_CHECKPOINT = "final0"
TEMPO_ESTIMATOR = "filtered_interval_mean_v1"
LEGACY_RHYTHM_FIELDS = {
    "bpm",
    "bpm_raw",
    "bpm_folded",
    "bpm_in_range",
    "first_beat",
    "first_downbeat",
    "beat_count",
    "downbeat_count",
    "estimated_beats_per_bar",
    "tempo_relative_mad",
    "tempo_estimator",
    "rhythm_condition",
    "rhythm_feature_rate",
    "beats_per_bar",
    "sections",
    "structure_text",
    "rhythm_detector",
    "rhythm_detector_checkpoint",
    "rhythm_analysis_audio",
}


def prepare_dataset(
    metadata: Path,
    dataset_root: Path,
    output: Path,
    condition_root: Path,
    analysis_audio: str,
    checkpoint: str,
    device: str,
    bpm_min: float | None,
    bpm_max: float | None,
    min_beats: int,
    min_downbeats: int,
    start_index: int,
    limit: int | None,
    overwrite: bool,
    keep_going: bool,
    dry_run: bool,
    tracker_factory: Callable[[str, str], Any] | None = None,
) -> None:
    """Analyze V1 rows and write a new V5 metadata manifest."""
    metadata = metadata.resolve()
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    condition_root = condition_root.resolve()

    _validate_arguments(
        metadata=metadata,
        dataset_root=dataset_root,
        output=output,
        condition_root=condition_root,
        analysis_audio=analysis_audio,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        min_beats=min_beats,
        min_downbeats=min_downbeats,
        start_index=start_index,
        limit=limit,
    )

    all_rows = _read_jsonl(metadata)
    selected = all_rows[start_index:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("No metadata rows selected.")

    report_path = output.with_suffix(output.suffix + ".report.jsonl")

    print(f"V1 metadata:       {metadata}")
    print(f"Dataset root:      {dataset_root}")
    print(f"V5 metadata:       {output}")
    print(f"Condition root:    {condition_root}")
    print(f"Report:            {report_path}")
    print(f"Rows selected:     {len(selected)} / {len(all_rows)}")
    print(f"Analysis audio:    {analysis_audio}_audio")
    print(f"Beat This model:   {checkpoint}")
    print(f"Device:            {device}")
    print(f"BPM label range:   {_range_text(bpm_min, bpm_max)}")
    print(f"Minimum beats:     {min_beats}")
    print(f"Minimum downbeats: {min_downbeats}")
    print(f"Overwrite:         {overwrite}")
    print(f"Dry run:           {dry_run}")
    print("")

    if dry_run:
        _print_dry_run(
            rows=selected,
            dataset_root=dataset_root,
            condition_root=condition_root,
            analysis_audio=analysis_audio,
            start_index=start_index,
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    condition_root.mkdir(parents=True, exist_ok=True)

    output_tmp = output.with_name(output.name + ".tmp")
    report_tmp = report_path.with_name(report_path.name + ".tmp")
    for temp_path in (output_tmp, report_tmp):
        if temp_path.exists():
            temp_path.unlink()

    tracker = None
    audio_cache: dict[Path, dict[str, Any]] = {}
    ok_count = 0
    error_count = 0

    try:
        with (
            output_tmp.open("w", encoding="utf-8") as output_handle,
            report_tmp.open("w", encoding="utf-8") as report_handle,
        ):
            for selected_index, row in enumerate(selected):
                source_index = start_index + selected_index
                try:
                    prepared = _prepare_row(
                        row=row,
                        source_index=source_index,
                        dataset_root=dataset_root,
                        condition_root=condition_root,
                        analysis_audio=analysis_audio,
                        checkpoint=checkpoint,
                        device=device,
                        bpm_min=bpm_min,
                        bpm_max=bpm_max,
                        min_beats=min_beats,
                        min_downbeats=min_downbeats,
                        overwrite=overwrite,
                        tracker=tracker,
                        tracker_factory=tracker_factory,
                        audio_cache=audio_cache,
                    )
                    tracker = prepared["tracker"]
                    updated = prepared["row"]
                    summary = prepared["summary"]

                    output_handle.write(
                        json.dumps(updated, ensure_ascii=False) + "\n"
                    )
                    report_handle.write(
                        json.dumps(
                            {
                                "source_row_index": source_index,
                                "track_id": updated["track_id"],
                                "chunk_index": updated["chunk_index"],
                                "analysis_audio": updated["v5_analysis_audio"],
                                "condition": updated["v5_condition"],
                                "status": "ok",
                                **_report_summary(summary),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_handle.flush()
                    report_handle.flush()
                    ok_count += 1
                    print(
                        f"[{selected_index + 1}/{len(selected)}] "
                        f"{updated['track_id']} chunk={updated['chunk_index']} "
                        f"bpm={updated['v5_bpm']:.2f} "
                        f"beats={updated['v5_beat_count']} "
                        f"downbeats={updated['v5_downbeat_count']}"
                    )
                except Exception as exc:
                    error_count += 1
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
                    report_handle.flush()
                    print(
                        f"[error] row={source_index} "
                        f"track={row.get('track_id', '(missing)')}: {exc}"
                    )
                    if not keep_going:
                        raise

        if ok_count == 0:
            if output_tmp.exists():
                output_tmp.unlink()
            _replace_file(report_tmp, report_path)
            raise RuntimeError(
                f"No V5 rows were written. Review the report: {report_path}"
            )

        _replace_file(output_tmp, output)
        _replace_file(report_tmp, report_path)
    except Exception:
        for temp_path in (output_tmp, report_tmp):
            if temp_path.exists():
                temp_path.unlink()
        raise

    print("")
    print("Done.")
    print(f"Rows written: {ok_count}")
    print(f"Errors:       {error_count}")
    print(f"Metadata:     {output}")
    print(f"Report:       {report_path}")
    if error_count:
        print("Some rows failed. Review the report before training V5.")


def _prepare_row(
    row: dict[str, Any],
    source_index: int,
    dataset_root: Path,
    condition_root: Path,
    analysis_audio: str,
    checkpoint: str,
    device: str,
    bpm_min: float | None,
    bpm_max: float | None,
    min_beats: int,
    min_downbeats: int,
    overwrite: bool,
    tracker: Any,
    tracker_factory: Callable[[str, str], Any] | None,
    audio_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    _validate_v1_row(row, source_index)

    input_path = _resolve_path(
        str(row["input_audio"]), dataset_root
    ).resolve()
    target_path = _resolve_path(
        str(row["target_audio"]), dataset_root
    ).resolve()
    _validate_audio_path(input_path, "input_audio")
    _validate_audio_path(target_path, "target_audio")

    audio_key = f"{analysis_audio}_audio"
    audio_path = input_path if analysis_audio == "input" else target_path

    track_id = str(
        row.get("track_id")
        or _fallback_track_id(str(row["input_audio"]), source_index)
    )
    chunk_index = int(row.get("chunk_index", source_index))
    condition_path = _condition_path(
        condition_root=condition_root,
        track_id=track_id,
        chunk_index=chunk_index,
    )

    summary: dict[str, Any]
    if condition_path.exists() and not overwrite:
        cached_summary = _load_condition(condition_path)
        _validate_cached_condition(
            summary=cached_summary,
            audio_path=audio_path,
            analysis_audio=analysis_audio,
            checkpoint=checkpoint,
            allow_schema_v1=True,
        )
        if cached_summary["schema_version"] == 1:
            # Schema 1 already contains the raw Beat This! events. Recompute
            # tempo metadata with the less quantization-sensitive schema 2
            # estimator and replace the cache without another model call.
            summary = _summarize_events(
                beats=cached_summary["beats"],
                downbeats=cached_summary["downbeats"],
                duration=float(cached_summary["duration"]),
                bpm_min=bpm_min,
                bpm_max=bpm_max,
            )
            _validate_summary(
                summary=summary,
                min_beats=min_beats,
                min_downbeats=min_downbeats,
                audio_path=audio_path,
            )
            _save_condition(
                path=condition_path,
                summary=summary,
                audio_path=audio_path,
                analysis_audio=analysis_audio,
                checkpoint=checkpoint,
            )
        else:
            summary = cached_summary
    elif audio_path in audio_cache:
        summary = dict(audio_cache[audio_path])
        _save_condition(
            path=condition_path,
            summary=summary,
            audio_path=audio_path,
            analysis_audio=analysis_audio,
            checkpoint=checkpoint,
        )
    else:
        if tracker is None:
            factory = tracker_factory or _create_tracker
            tracker = factory(checkpoint, device)

        beats, downbeats = tracker(str(audio_path))
        duration = float(row.get("duration") or _audio_duration(audio_path))
        summary = _summarize_events(
            beats=beats,
            downbeats=downbeats,
            duration=duration,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
        )
        _validate_summary(
            summary=summary,
            min_beats=min_beats,
            min_downbeats=min_downbeats,
            audio_path=audio_path,
        )
        audio_cache[audio_path] = dict(summary)
        _save_condition(
            path=condition_path,
            summary=summary,
            audio_path=audio_path,
            analysis_audio=analysis_audio,
            checkpoint=checkpoint,
        )

    _validate_summary(
        summary=summary,
        min_beats=min_beats,
        min_downbeats=min_downbeats,
        audio_path=audio_path,
    )

    updated = dict(row)
    original_text = str(row["text"]).strip()
    v5_text = _build_v5_prompt(original_text, float(summary["bpm"]))

    updated["track_id"] = track_id
    updated["chunk_index"] = chunk_index
    updated["text_v1"] = original_text
    updated["text"] = v5_text

    # Canonical V5 training fields. The input validator rejects pre-existing
    # rhythm fields, so these values can only originate from Beat This!.
    updated["rhythm_detector"] = DETECTOR_NAME
    updated["rhythm_detector_checkpoint"] = checkpoint
    updated["rhythm_analysis_audio"] = audio_key
    updated["rhythm_condition"] = _display_path(
        condition_path, dataset_root
    )
    updated["bpm"] = float(summary["bpm"])
    updated["bpm_raw"] = float(summary["bpm_raw"])
    updated["bpm_folded"] = bool(summary["bpm_folded"])
    updated["bpm_in_range"] = bool(summary["bpm_in_range"])
    updated["first_beat"] = float(summary["first_beat"])
    updated["first_downbeat"] = _finite_float_or_none(
        summary["first_downbeat"]
    )
    updated["beat_count"] = int(len(summary["beats"]))
    updated["downbeat_count"] = int(len(summary["downbeats"]))
    updated["estimated_beats_per_bar"] = int(
        summary["estimated_beats_per_bar"]
    )
    updated["tempo_relative_mad"] = float(summary["tempo_relative_mad"])
    updated["tempo_estimator"] = TEMPO_ESTIMATOR

    # Namespaced aliases make the V5 trainer contract explicit and prevent
    # accidental coupling to any older experimental trainer.
    updated["v5_schema_version"] = SCHEMA_VERSION
    updated["v5_detector"] = DETECTOR_NAME
    updated["v5_detector_checkpoint"] = checkpoint
    updated["v5_analysis_source"] = audio_key
    updated["v5_analysis_audio"] = _display_path(audio_path, dataset_root)
    updated["v5_condition"] = _display_path(condition_path, dataset_root)
    updated["v5_bpm"] = float(summary["bpm"])
    updated["v5_bpm_raw"] = float(summary["bpm_raw"])
    updated["v5_bpm_folded"] = bool(summary["bpm_folded"])
    updated["v5_bpm_in_range"] = bool(summary["bpm_in_range"])
    updated["v5_first_beat"] = float(summary["first_beat"])
    updated["v5_first_downbeat"] = _finite_float_or_none(
        summary["first_downbeat"]
    )
    updated["v5_beat_count"] = int(len(summary["beats"]))
    updated["v5_downbeat_count"] = int(len(summary["downbeats"]))
    updated["v5_estimated_beats_per_bar"] = int(
        summary["estimated_beats_per_bar"]
    )
    updated["v5_tempo_relative_mad"] = float(summary["tempo_relative_mad"])
    updated["v5_tempo_estimator"] = TEMPO_ESTIMATOR

    return {"row": updated, "summary": summary, "tracker": tracker}


def _create_tracker(checkpoint: str, device: str) -> Any:
    try:
        import torch
        from beat_this.inference import File2Beats
    except ImportError as exc:
        raise RuntimeError(
            "Missing Beat This! dependencies. Activate the MusicGen venv and run: "
            'python -m pip install "beat-this==1.1.0"'
        ) from exc

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({resolved_device}) but torch CUDA is unavailable."
        )

    print(
        f"Loading Beat This! checkpoint={checkpoint} device={resolved_device} "
        "(first use may download model weights)"
    )
    return File2Beats(
        checkpoint_path=checkpoint,
        device=resolved_device,
        dbn=False,
    )


def _summarize_events(
    beats: Any,
    downbeats: Any,
    duration: float,
    bpm_min: float | None,
    bpm_max: float | None,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy.") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration: {duration}")

    beat_array = _clean_events(beats, duration, "beats", np)
    downbeat_array = _clean_events(downbeats, duration, "downbeats", np)
    if len(beat_array) < 2:
        raise RuntimeError("Beat This! returned fewer than two valid beats.")

    intervals = np.diff(beat_array)
    valid_intervals = intervals[
        np.isfinite(intervals) & (intervals >= 0.15) & (intervals <= 2.0)
    ]
    if len(valid_intervals) == 0:
        raise RuntimeError("No valid beat intervals in the 30-400 BPM range.")

    filtered_intervals = _filter_interval_outliers(valid_intervals, np)
    # Beat This! emits events on a 50 FPS grid. A median interval therefore
    # snaps a true ~140 BPM pulse to either 0.42 s (142.86 BPM) or 0.44 s
    # (136.36 BPM), depending on which quantized interval occurs once more in
    # a chunk. The mean after robust outlier rejection recovers the underlying
    # interval while still discarding missed/doubled beat gaps.
    beat_interval = float(np.mean(filtered_intervals))
    bpm_raw = 60.0 / beat_interval
    bpm, bpm_folded, bpm_in_range = _fold_bpm_to_range(
        bpm=bpm_raw,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
    )

    interval_mad = float(
        np.median(np.abs(filtered_intervals - beat_interval))
    )
    tempo_relative_mad = interval_mad / beat_interval
    estimated_beats_per_bar = _estimate_beats_per_bar(
        downbeats=downbeat_array,
        beat_interval=beat_interval,
        np_module=np,
    )

    return {
        "duration": float(duration),
        "beats": beat_array.astype(np.float32),
        "downbeats": downbeat_array.astype(np.float32),
        "bpm": float(bpm),
        "bpm_raw": float(bpm_raw),
        "bpm_folded": bool(bpm_folded),
        "bpm_in_range": bool(bpm_in_range),
        # Retain the schema-1 key for compatibility. In schema 2 it contains
        # the robust interval estimate rather than the literal median.
        "median_beat_interval": beat_interval,
        "tempo_estimator": TEMPO_ESTIMATOR,
        "tempo_relative_mad": float(tempo_relative_mad),
        "first_beat": float(beat_array[0]),
        "first_downbeat": (
            float(downbeat_array[0]) if len(downbeat_array) else math.nan
        ),
        "estimated_beats_per_bar": int(estimated_beats_per_bar),
    }


def _clean_events(
    values: Any,
    duration: float,
    label: str,
    np_module: Any,
) -> Any:
    array = np_module.asarray(values, dtype=np_module.float64).reshape(-1)
    array = array[np_module.isfinite(array)]
    array = array[(array >= 0.0) & (array <= duration + 0.1)]
    array = np_module.unique(array)
    array.sort()
    if label == "beats" and len(array) == 0:
        raise RuntimeError("Beat This! returned no valid beats.")
    return array


def _filter_interval_outliers(intervals: Any, np_module: Any) -> Any:
    if len(intervals) < 4:
        return intervals
    median = float(np_module.median(intervals))
    absolute_deviation = np_module.abs(intervals - median)
    mad = float(np_module.median(absolute_deviation))
    tolerance = max(3.0 * mad, 0.03)
    filtered = intervals[absolute_deviation <= tolerance]
    return filtered if len(filtered) >= 2 else intervals


def _fold_bpm_to_range(
    bpm: float,
    bpm_min: float | None,
    bpm_max: float | None,
) -> tuple[float, bool, bool]:
    """Fold by powers of two only when an octave-equivalent is in range."""
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"Invalid BPM: {bpm}")
    if bpm_min is None or bpm_max is None:
        return bpm, False, True

    candidates = [
        bpm * (2.0**octave)
        for octave in range(-4, 5)
        if bpm_min <= bpm * (2.0**octave) <= bpm_max
    ]
    if not candidates:
        return bpm, False, False

    center = math.sqrt(bpm_min * bpm_max)
    selected = min(candidates, key=lambda value: abs(math.log(value / center)))
    return selected, not math.isclose(selected, bpm), True


def _estimate_beats_per_bar(
    downbeats: Any,
    beat_interval: float,
    np_module: Any,
) -> int:
    if len(downbeats) < 2 or beat_interval <= 0:
        return 0
    ratios = np_module.diff(downbeats) / beat_interval
    ratios = ratios[np_module.isfinite(ratios)]
    ratios = ratios[(ratios >= 1.0) & (ratios <= 16.0)]
    if len(ratios) == 0:
        return 0
    return int(round(float(np_module.median(ratios))))


def _validate_summary(
    summary: dict[str, Any],
    min_beats: int,
    min_downbeats: int,
    audio_path: Path,
) -> None:
    beat_count = len(summary["beats"])
    downbeat_count = len(summary["downbeats"])
    if beat_count < min_beats:
        raise RuntimeError(
            f"Too few beats for {audio_path}: {beat_count} < {min_beats}"
        )
    if downbeat_count < min_downbeats:
        raise RuntimeError(
            f"Too few downbeats for {audio_path}: "
            f"{downbeat_count} < {min_downbeats}"
        )


def _save_condition(
    path: Path,
    summary: dict[str, Any],
    audio_path: Path,
    analysis_audio: str,
    checkpoint: str,
) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int32),
            detector=np.asarray([DETECTOR_NAME]),
            detector_checkpoint=np.asarray([checkpoint]),
            tempo_estimator=np.asarray([TEMPO_ESTIMATOR]),
            analysis_source=np.asarray([f"{analysis_audio}_audio"]),
            analysis_audio=np.asarray([str(audio_path)]),
            analysis_audio_size=np.asarray(
                [audio_path.stat().st_size], dtype=np.int64
            ),
            analysis_audio_mtime_ns=np.asarray(
                [audio_path.stat().st_mtime_ns], dtype=np.int64
            ),
            duration=np.asarray([summary["duration"]], dtype=np.float32),
            beats=summary["beats"],
            downbeats=summary["downbeats"],
            bpm=np.asarray([summary["bpm"]], dtype=np.float32),
            bpm_raw=np.asarray([summary["bpm_raw"]], dtype=np.float32),
            bpm_folded=np.asarray([summary["bpm_folded"]], dtype=np.bool_),
            bpm_in_range=np.asarray(
                [summary["bpm_in_range"]], dtype=np.bool_
            ),
            median_beat_interval=np.asarray(
                [summary["median_beat_interval"]], dtype=np.float32
            ),
            tempo_relative_mad=np.asarray(
                [summary["tempo_relative_mad"]], dtype=np.float32
            ),
            first_beat=np.asarray([summary["first_beat"]], dtype=np.float32),
            first_downbeat=np.asarray(
                [summary["first_downbeat"]], dtype=np.float32
            ),
            estimated_beats_per_bar=np.asarray(
                [summary["estimated_beats_per_bar"]], dtype=np.int32
            ),
        )
    _replace_file(temp_path, path)


def _load_condition(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: numpy.") from exc

    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "detector",
            "detector_checkpoint",
            "analysis_source",
            "analysis_audio",
            "analysis_audio_size",
            "analysis_audio_mtime_ns",
            "duration",
            "beats",
            "downbeats",
            "bpm",
            "bpm_raw",
            "bpm_folded",
            "bpm_in_range",
            "median_beat_interval",
            "tempo_relative_mad",
            "first_beat",
            "first_downbeat",
            "estimated_beats_per_bar",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                f"Invalid V5 condition {path}; missing keys: {missing}"
            )
        schema_version = int(data["schema_version"][0])
        tempo_estimator = (
            str(data["tempo_estimator"][0])
            if "tempo_estimator" in data.files
            else "interval_median_v1"
        )
        return {
            "schema_version": schema_version,
            "tempo_estimator": tempo_estimator,
            "detector": str(data["detector"][0]),
            "detector_checkpoint": str(data["detector_checkpoint"][0]),
            "analysis_source": str(data["analysis_source"][0]),
            "analysis_audio": str(data["analysis_audio"][0]),
            "analysis_audio_size": int(data["analysis_audio_size"][0]),
            "analysis_audio_mtime_ns": int(
                data["analysis_audio_mtime_ns"][0]
            ),
            "duration": float(data["duration"][0]),
            "beats": data["beats"].astype(np.float32),
            "downbeats": data["downbeats"].astype(np.float32),
            "bpm": float(data["bpm"][0]),
            "bpm_raw": float(data["bpm_raw"][0]),
            "bpm_folded": bool(data["bpm_folded"][0]),
            "bpm_in_range": bool(data["bpm_in_range"][0]),
            "median_beat_interval": float(
                data["median_beat_interval"][0]
            ),
            "tempo_relative_mad": float(data["tempo_relative_mad"][0]),
            "first_beat": float(data["first_beat"][0]),
            "first_downbeat": float(data["first_downbeat"][0]),
            "estimated_beats_per_bar": int(
                data["estimated_beats_per_bar"][0]
            ),
        }


def _validate_cached_condition(
    summary: dict[str, Any],
    audio_path: Path,
    analysis_audio: str,
    checkpoint: str,
    allow_schema_v1: bool = False,
) -> None:
    accepted_schemas = {SCHEMA_VERSION}
    if allow_schema_v1:
        accepted_schemas.add(1)
    if summary["schema_version"] not in accepted_schemas:
        raise RuntimeError(
            "Cached condition schema mismatch. Re-run with --overwrite: "
            f"{summary['schema_version']} not in {sorted(accepted_schemas)}"
        )
    if (
        summary["schema_version"] == SCHEMA_VERSION
        and summary.get("tempo_estimator") != TEMPO_ESTIMATOR
    ):
        raise RuntimeError(
            "Cached tempo estimator differs from the requested schema. "
            "Re-run with --overwrite."
        )
    if summary["detector"] != DETECTOR_NAME:
        raise RuntimeError(
            f"Cached condition was produced by {summary['detector']!r}, "
            f"expected {DETECTOR_NAME!r}. Re-run with --overwrite."
        )
    if summary["detector_checkpoint"] != checkpoint:
        raise RuntimeError(
            "Cached Beat This! checkpoint differs from the requested model. "
            "Re-run with --overwrite."
        )
    expected_source = f"{analysis_audio}_audio"
    if summary["analysis_source"] != expected_source:
        raise RuntimeError(
            "Cached analysis source differs from the requested source. "
            "Re-run with --overwrite."
        )
    if Path(summary["analysis_audio"]).resolve() != audio_path:
        raise RuntimeError(
            "Cached condition points to a different audio file. "
            "Re-run with --overwrite."
        )
    audio_stat = audio_path.stat()
    if (
        summary["analysis_audio_size"] != audio_stat.st_size
        or summary["analysis_audio_mtime_ns"] != audio_stat.st_mtime_ns
    ):
        raise RuntimeError(
            "Analysis audio changed after the cached condition was created. "
            "Re-run with --overwrite."
        )


def _validate_arguments(
    metadata: Path,
    dataset_root: Path,
    output: Path,
    condition_root: Path,
    analysis_audio: str,
    bpm_min: float | None,
    bpm_max: float | None,
    min_beats: int,
    min_downbeats: int,
    start_index: int,
    limit: int | None,
) -> None:
    if not metadata.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist or is not a folder: {dataset_root}"
        )
    if output == metadata:
        raise ValueError(
            "V5 output must differ from V1 metadata; V1 is never overwritten."
        )
    if _is_relative_to(output, condition_root):
        raise ValueError("Metadata output cannot be inside the condition root.")
    if analysis_audio not in {"input", "target"}:
        raise ValueError(f"Unsupported analysis audio: {analysis_audio}")
    if (bpm_min is None) != (bpm_max is None):
        raise ValueError("--bpm-min and --bpm-max must be set together.")
    if bpm_min is not None and (bpm_min <= 0 or bpm_max <= bpm_min):
        raise ValueError("BPM range must satisfy 0 < bpm_min < bpm_max.")
    if min_beats < 2:
        raise ValueError("--min-beats must be at least 2.")
    if min_downbeats < 0:
        raise ValueError("--min-downbeats must be >= 0.")
    if start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0.")


def _validate_v1_row(row: dict[str, Any], row_index: int) -> None:
    for key in ("input_audio", "target_audio", "text"):
        if key not in row:
            raise KeyError(f"V1 row {row_index} is missing {key!r}.")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError(f"V1 row {row_index} has an empty text prompt.")

    legacy_fields = sorted(LEGACY_RHYTHM_FIELDS.intersection(row))
    v5_fields = sorted(key for key in row if key.startswith("v5_"))
    if legacy_fields or v5_fields:
        contaminated = legacy_fields + v5_fields
        raise ValueError(
            f"Row {row_index} is not clean V1 metadata; found existing rhythm "
            f"fields: {contaminated}. Use the original V1 "
            "metadata_instrumental.jsonl, not a V4/Librosa/V5 manifest."
        )

    normalized_text = row["text"].lower()
    if " bpm" in normalized_text or "structure:" in normalized_text:
        raise ValueError(
            f"Row {row_index} prompt already contains BPM/structure labels. "
            "Use the original clean V1 metadata so V5 labels come only from "
            "Beat This!."
        )


def _validate_audio_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _condition_path(
    condition_root: Path,
    track_id: str,
    chunk_index: int,
) -> Path:
    safe_track = _safe_track_path(track_id)
    path = condition_root / safe_track / f"chunk_{chunk_index:04d}.npz"
    resolved_parent = path.parent.resolve()
    if not _is_relative_to(resolved_parent, condition_root):
        raise ValueError(f"Unsafe track_id escapes condition root: {track_id!r}")
    return path


def _safe_track_path(track_id: str) -> Path:
    raw = Path(track_id.replace("\\", "/"))
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"Unsafe track_id: {track_id!r}")

    safe_parts = []
    for part in raw.parts:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._")
        if not safe:
            raise ValueError(f"Unsafe empty track_id component: {track_id!r}")
        safe_parts.append(safe)
    return Path(*safe_parts)


def _fallback_track_id(input_audio: str, row_index: int) -> str:
    parent = Path(input_audio).parent
    if parent.name:
        return parent.as_posix().lstrip("/")
    return f"row_{row_index:06d}"


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Metadata row has no duration and soundfile is unavailable."
        ) from exc
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}: {path}"
                )
            rows.append(row)
    return rows


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _report_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tempo_estimator": TEMPO_ESTIMATOR,
        "bpm": float(summary["bpm"]),
        "bpm_raw": float(summary["bpm_raw"]),
        "bpm_folded": bool(summary["bpm_folded"]),
        "bpm_in_range": bool(summary["bpm_in_range"]),
        "beat_count": int(len(summary["beats"])),
        "downbeat_count": int(len(summary["downbeats"])),
        "first_beat": float(summary["first_beat"]),
        "first_downbeat": _finite_float_or_none(summary["first_downbeat"]),
        "estimated_beats_per_bar": int(
            summary["estimated_beats_per_bar"]
        ),
        "tempo_relative_mad": float(summary["tempo_relative_mad"]),
    }


def _range_text(
    bpm_min: float | None,
    bpm_max: float | None,
) -> str:
    if bpm_min is None or bpm_max is None:
        return "(disabled)"
    return f"{bpm_min:.1f}-{bpm_max:.1f}"


def _finite_float_or_none(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _build_v5_prompt(v1_text: str, bpm: float) -> str:
    return f"{v1_text.rstrip(' ,')}, {bpm:.1f} bpm"


def _print_dry_run(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    condition_root: Path,
    analysis_audio: str,
    start_index: int,
) -> None:
    audio_key = f"{analysis_audio}_audio"
    for selected_index, row in enumerate(rows):
        source_index = start_index + selected_index
        _validate_v1_row(row, source_index)
        track_id = str(
            row.get("track_id")
            or _fallback_track_id(str(row["input_audio"]), source_index)
        )
        chunk_index = int(row.get("chunk_index", source_index))
        input_path = _resolve_path(str(row["input_audio"]), dataset_root)
        target_path = _resolve_path(str(row["target_audio"]), dataset_root)
        _validate_audio_path(input_path, "input_audio")
        _validate_audio_path(target_path, "target_audio")
        audio_path = input_path if analysis_audio == "input" else target_path
        condition_path = _condition_path(
            condition_root, track_id, chunk_index
        )
        print(
            f"[plan] row={source_index} track={track_id} "
            f"chunk={chunk_index} audio={audio_path} "
            f"condition={condition_path}"
        )
    print("")
    print("Dry run complete. No files were written and Beat This! was not loaded.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare independent V5 Beat This! event conditions from V1 "
            "MusicGen paired metadata."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="V1 metadata JSONL.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root used to resolve V1 relative audio paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New V5 metadata JSONL. Must not equal --metadata.",
    )
    parser.add_argument(
        "--condition-root",
        type=Path,
        required=True,
        help="Folder for V5 Beat This! NPZ event files.",
    )
    parser.add_argument(
        "--analysis-audio",
        choices=("input", "target"),
        default="target",
        help=(
            "Audio used for Beat This! analysis. Default: target, so V5 "
            "training labels are extracted from the full instrumental."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Beat This! checkpoint name. Default: final0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Beat This! device: auto, cpu, cuda, cuda:0. Default: auto.",
    )
    parser.add_argument(
        "--bpm-min",
        type=float,
        default=120.0,
        help="Minimum octave-folded BPM label. Default: 120.",
    )
    parser.add_argument(
        "--bpm-max",
        type=float,
        default=150.0,
        help="Maximum octave-folded BPM label. Default: 150.",
    )
    parser.add_argument(
        "--no-bpm-fold",
        action="store_true",
        help="Keep raw BPM without attempting power-of-two folding.",
    )
    parser.add_argument(
        "--min-beats",
        type=int,
        default=4,
        help="Reject rows with fewer valid beats. Default: 4.",
    )
    parser.add_argument(
        "--min-downbeats",
        type=int,
        default=1,
        help="Reject rows with fewer downbeats. Use 0 to keep them. Default: 1.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N V1 rows. Default: 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run Beat This! and replace existing V5 conditions.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failed rows in the report and continue.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print paths without loading Beat This! or writing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bpm_min = None if args.no_bpm_fold else args.bpm_min
    bpm_max = None if args.no_bpm_fold else args.bpm_max
    prepare_dataset(
        metadata=args.metadata,
        dataset_root=args.dataset_root,
        output=args.output,
        condition_root=args.condition_root,
        analysis_audio=args.analysis_audio,
        checkpoint=args.checkpoint,
        device=args.device,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        min_beats=args.min_beats,
        min_downbeats=args.min_downbeats,
        start_index=args.start_index,
        limit=args.limit,
        overwrite=args.overwrite,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
