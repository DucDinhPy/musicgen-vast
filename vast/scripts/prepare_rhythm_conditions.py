#!/usr/bin/env python3
"""Create V4 rhythm-grid condition files for paired MusicGen metadata.

Input metadata is the regular paired format:

    input_audio + text -> target_audio

This script estimates BPM/beat/downbeat from each target audio chunk and writes:

    rhythm/<track_id>/chunk_0000.npz

Each `.npz` contains a fixed-rate feature grid aligned to MusicGen time:

    features[:, 0] = beat pulse
    features[:, 1] = downbeat pulse
    features[:, 2] = sin(beat phase)
    features[:, 3] = cos(beat phase)
    features[:, 4] = sin(bar phase)
    features[:, 5] = cos(bar phase)

The output metadata keeps the original paths and adds rhythm_condition/bpm fields.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FEATURE_RATE = 50.0


def prepare_rhythm_conditions(
    metadata: Path,
    dataset_root: Path,
    output: Path,
    condition_root: Path,
    feature_rate: float,
    beats_per_bar: int,
    bpm_min: float | None,
    bpm_max: float | None,
    pulse_width_seconds: float,
    add_section_labels: bool,
    section_frame_seconds: float,
    section_threshold_db: float,
    section_min_active_seconds: float,
    section_merge_gap_seconds: float,
    limit: int | None,
    overwrite: bool,
) -> None:
    rows = _read_jsonl(metadata)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No rows found in: {metadata}")

    output.parent.mkdir(parents=True, exist_ok=True)
    condition_root.mkdir(parents=True, exist_ok=True)

    print(f"Metadata:       {metadata.resolve()}")
    print(f"Dataset root:   {dataset_root.resolve()}")
    print(f"Output:         {output.resolve()}")
    print(f"Condition root: {condition_root.resolve()}")
    print(f"Rows:           {len(rows)}")
    print(f"Feature rate:   {feature_rate}")
    print(f"Beats/bar:      {beats_per_bar}")
    print("")

    with output.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            target_audio = _resolve_path(row["target_audio"], dataset_root)
            duration = float(row.get("duration") or _duration(target_audio))
            track_id = str(row.get("track_id") or Path(row["target_audio"]).parent.as_posix())
            chunk_index = int(row.get("chunk_index", index - 1))
            condition_path = condition_root / track_id / f"chunk_{chunk_index:04d}.npz"

            if condition_path.exists() and not overwrite:
                rhythm = _load_npz_summary(condition_path)
            else:
                rhythm = _estimate_rhythm(
                    audio_path=target_audio,
                    duration=duration,
                    feature_rate=feature_rate,
                    beats_per_bar=beats_per_bar,
                    bpm_min=bpm_min,
                    bpm_max=bpm_max,
                    pulse_width_seconds=pulse_width_seconds,
                )
                condition_path.parent.mkdir(parents=True, exist_ok=True)
                _save_npz(condition_path, rhythm)

            updated = dict(row)
            updated["rhythm_condition"] = _display_path(condition_path, dataset_root)
            updated["bpm"] = rhythm["bpm"]
            updated["bpm_raw"] = rhythm["bpm_raw"]
            updated["first_beat"] = rhythm["first_beat"]
            updated["first_downbeat"] = rhythm["first_downbeat"]
            updated["beat_count"] = len(rhythm["beats"])
            updated["downbeat_count"] = len(rhythm["downbeats"])
            updated["rhythm_feature_rate"] = feature_rate
            updated["beats_per_bar"] = beats_per_bar
            structure_text = ""
            if add_section_labels:
                input_audio = _resolve_path(row["input_audio"], dataset_root)
                sections = _build_section_labels(
                    input_audio=input_audio,
                    duration=duration,
                    chunk_index=chunk_index,
                    frame_seconds=section_frame_seconds,
                    threshold_db=section_threshold_db,
                    min_active_seconds=section_min_active_seconds,
                    merge_gap_seconds=section_merge_gap_seconds,
                )
                structure_text = _structure_text(sections)
                updated["sections"] = sections
                updated["structure_text"] = structure_text
            updated["text"] = _append_prompt_labels(
                text=str(row.get("text", "")),
                bpm=float(rhythm["bpm"]),
                structure_text=structure_text,
            )
            out.write(json.dumps(updated, ensure_ascii=False) + "\n")

            print(
                f"[{index}/{len(rows)}] {track_id} chunk={chunk_index} "
                f"bpm={float(rhythm['bpm']):.2f} beats={len(rhythm['beats'])} "
                f"downbeats={len(rhythm['downbeats'])}"
            )

    print("")
    print("Done.")
    print(f"Wrote: {output.resolve()}")


def _estimate_rhythm(
    audio_path: Path,
    duration: float,
    feature_rate: float,
    beats_per_bar: int,
    bpm_min: float | None,
    bpm_max: float | None,
    pulse_width_seconds: float,
) -> dict:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: librosa/numpy.") from exc

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    bpm_raw = float(np.asarray(tempo).reshape(-1)[0])
    bpm = _fold_bpm_to_range(bpm_raw, bpm_min, bpm_max)
    beats = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beats) < 2 and bpm > 0:
        beat_s = 60.0 / bpm
        beats = np.arange(0.0, duration + 1e-6, beat_s)

    first_beat = float(beats[0]) if len(beats) else 0.0
    downbeats = beats[::beats_per_bar] if len(beats) else np.asarray([0.0])
    first_downbeat = float(downbeats[0]) if len(downbeats) else first_beat

    frame_count = max(1, int(math.ceil(duration * feature_rate)))
    times = np.arange(frame_count, dtype=np.float32) / feature_rate
    features = np.zeros((frame_count, 6), dtype=np.float32)

    pulse_radius = max(1, int(round(pulse_width_seconds * feature_rate)))
    _write_pulses(features[:, 0], beats, feature_rate, pulse_radius)
    _write_pulses(features[:, 1], downbeats, feature_rate, pulse_radius)

    beat_period = 60.0 / bpm if bpm > 0 else max(duration, 1.0)
    bar_period = beat_period * beats_per_bar
    beat_phase = ((times - first_beat) / beat_period) % 1.0
    bar_phase = ((times - first_downbeat) / bar_period) % 1.0
    features[:, 2] = np.sin(2.0 * np.pi * beat_phase)
    features[:, 3] = np.cos(2.0 * np.pi * beat_phase)
    features[:, 4] = np.sin(2.0 * np.pi * bar_phase)
    features[:, 5] = np.cos(2.0 * np.pi * bar_phase)

    return {
        "features": features,
        "times": times,
        "bpm": float(bpm),
        "bpm_raw": float(bpm_raw),
        "beats": np.asarray(beats, dtype=np.float32),
        "downbeats": np.asarray(downbeats, dtype=np.float32),
        "first_beat": first_beat,
        "first_downbeat": first_downbeat,
    }


def _write_pulses(column, events, feature_rate: float, pulse_radius: int) -> None:
    for event in events:
        center = int(round(float(event) * feature_rate))
        start = max(0, center - pulse_radius)
        end = min(len(column), center + pulse_radius + 1)
        column[start:end] = 1.0


def _fold_bpm_to_range(bpm: float, bpm_min: float | None, bpm_max: float | None) -> float:
    if bpm <= 0 or bpm_min is None or bpm_max is None:
        return bpm
    folded = bpm
    while folded < bpm_min:
        folded *= 2.0
    while folded > bpm_max:
        folded /= 2.0
    return folded


def _save_npz(path: Path, rhythm: dict) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        features=rhythm["features"],
        times=rhythm["times"],
        beats=rhythm["beats"],
        downbeats=rhythm["downbeats"],
        bpm=np.asarray([rhythm["bpm"]], dtype=np.float32),
        bpm_raw=np.asarray([rhythm["bpm_raw"]], dtype=np.float32),
        first_beat=np.asarray([rhythm["first_beat"]], dtype=np.float32),
        first_downbeat=np.asarray([rhythm["first_downbeat"]], dtype=np.float32),
    )


def _load_npz_summary(path: Path) -> dict:
    import numpy as np

    data = np.load(path)
    return {
        "bpm": float(data["bpm"][0]),
        "bpm_raw": float(data["bpm_raw"][0]),
        "first_beat": float(data["first_beat"][0]),
        "first_downbeat": float(data["first_downbeat"][0]),
        "beats": data["beats"],
        "downbeats": data["downbeats"],
    }


def _build_section_labels(
    input_audio: Path,
    duration: float,
    chunk_index: int,
    frame_seconds: float,
    threshold_db: float,
    min_active_seconds: float,
    merge_gap_seconds: float,
) -> list[list]:
    active = _detect_active_intervals(
        audio_path=input_audio,
        frame_seconds=frame_seconds,
        threshold_db=threshold_db,
        min_active_seconds=min_active_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )
    if not active:
        label = "intro_no_melody" if chunk_index == 0 else "break_or_drop_no_melody"
        return [[label, 0.0, round(duration, 3)]]

    sections: list[list] = []
    cursor = 0.0
    melody_seen = False
    for start, end in active:
        start = float(start)
        end = min(float(end), duration)
        if start > cursor + 0.25:
            if not melody_seen and chunk_index == 0:
                label = "intro_no_melody"
            else:
                label = "break_or_drop_no_melody"
            sections.append([label, round(cursor, 3), round(start, 3)])

        melody_label = "main_melody" if not melody_seen else "melody_backing"
        sections.append([melody_label, round(start, 3), round(end, 3)])
        melody_seen = True
        cursor = end

    if cursor < duration - 0.25:
        sections.append(["outro_no_melody", round(cursor, 3), round(duration, 3)])
    return sections


def _detect_active_intervals(
    audio_path: Path,
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

    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_size = max(1, int(frame_seconds * sr))

    active_frames = []
    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if _to_db(rms) >= threshold_db:
            active_frames.append((start / sr, min(len(audio), start + frame_size) / sr))

    intervals = _merge_intervals(active_frames, merge_gap_seconds)
    return [
        [round(start, 3), round(end, 3)]
        for start, end in intervals
        if end - start >= min_active_seconds
    ]


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


def _structure_text(sections: list[list]) -> str:
    parts = [f"{label} {start:.1f}-{end:.1f}s" for label, start, end in sections]
    return "structure: " + ", ".join(parts)


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _append_prompt_labels(text: str, bpm: float, structure_text: str) -> str:
    output = _append_bpm_to_text(text, bpm)
    if structure_text:
        output = f"{output}, {structure_text}" if output else structure_text
    return output


def _append_bpm_to_text(text: str, bpm: float) -> str:
    bpm_text = f"{bpm:.1f} bpm"
    if " bpm" in text.lower():
        return text
    return f"{text}, {bpm_text}" if text else bpm_text


def _duration(path: Path) -> float:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare V4 rhythm condition files.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--feature-rate", type=float, default=FEATURE_RATE)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=150.0)
    parser.add_argument("--pulse-width-seconds", type=float, default=0.03)
    parser.add_argument("--no-section-labels", dest="add_section_labels", action="store_false")
    parser.set_defaults(add_section_labels=True)
    parser.add_argument("--section-frame-seconds", type=float, default=0.25)
    parser.add_argument("--section-threshold-db", type=float, default=-45.0)
    parser.add_argument("--section-min-active-seconds", type=float, default=1.0)
    parser.add_argument("--section-merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_rhythm_conditions(
        metadata=args.metadata,
        dataset_root=args.dataset_root,
        output=args.output,
        condition_root=args.condition_root,
        feature_rate=args.feature_rate,
        beats_per_bar=args.beats_per_bar,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        pulse_width_seconds=args.pulse_width_seconds,
        add_section_labels=args.add_section_labels,
        section_frame_seconds=args.section_frame_seconds,
        section_threshold_db=args.section_threshold_db,
        section_min_active_seconds=args.section_min_active_seconds,
        section_merge_gap_seconds=args.section_merge_gap_seconds,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
