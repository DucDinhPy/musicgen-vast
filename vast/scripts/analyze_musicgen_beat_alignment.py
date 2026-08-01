#!/usr/bin/env python3
"""Measure and optionally correct beat offset/drift in generated backing audio."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from prepare_musicgen_v5_beatthis_dataset import (
    DEFAULT_CHECKPOINT,
    _create_tracker,
)


def analyze_alignment(
    reference_audio: Path,
    generated_audio: Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    extend_stride: float = 18.0,
    correction: str = "none",
    corrected_output: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    tracker = _create_tracker(checkpoint, device)
    print("Detecting reference beats with Beat This!...")
    reference_beats, reference_downbeats = tracker(str(reference_audio))
    print("Detecting generated beats with Beat This!...")
    generated_beats, generated_downbeats = tracker(str(generated_audio))
    reference_beats = _events(reference_beats, np)
    generated_beats = _events(generated_beats, np)
    reference_downbeats = _events(reference_downbeats, np)
    generated_downbeats = _events(generated_downbeats, np)

    beat_fit = _fit_event_grids(reference_beats, generated_beats, np)
    downbeat_fit = (
        _fit_event_grids(reference_downbeats, generated_downbeats, np)
        if len(reference_downbeats) >= 3 and len(generated_downbeats) >= 3
        else None
    )
    segment_errors = _segment_errors(
        beat_fit, extend_stride=extend_stride, np_module=np
    )
    classification = _classify(beat_fit, segment_errors)
    report: dict[str, Any] = {
        "reference_audio": str(reference_audio.resolve()),
        "generated_audio": str(generated_audio.resolve()),
        "reference_beat_count": int(len(reference_beats)),
        "generated_beat_count": int(len(generated_beats)),
        "reference_downbeat_count": int(len(reference_downbeats)),
        "generated_downbeat_count": int(len(generated_downbeats)),
        "beat_fit": _json_fit(beat_fit),
        "downbeat_fit": _json_fit(downbeat_fit) if downbeat_fit else None,
        "extend_stride_seconds": float(extend_stride),
        "stride_segments": segment_errors,
        "classification": classification,
        "correction": correction,
        "corrected_output": None,
    }

    if correction != "none":
        if corrected_output is None:
            raise ValueError("corrected_output is required when correction is enabled")
        _correct_audio_alignment(
            generated_audio=generated_audio,
            reference_audio=reference_audio,
            output=corrected_output,
            fit=beat_fit,
            mode=correction,
        )
        report["corrected_output"] = str(corrected_output.resolve())

    return report


def print_report(report: dict[str, Any]) -> None:
    fit = report["beat_fit"]
    print("")
    print("===== BEAT ALIGNMENT =====")
    print(
        f"Beats reference/generated: "
        f"{report['reference_beat_count']}/{report['generated_beat_count']}"
    )
    print(f"Matched beats:             {fit['matched_events']}")
    print(f"Initial offset:            {fit['offset_seconds']:+.4f}s")
    print(f"Generated/reference tempo: {fit['tempo_ratio']:.7f}")
    print(f"Drift per minute:          {fit['drift_seconds_per_minute']:+.4f}s")
    print(f"Median residual:           {fit['median_abs_residual_seconds']:.4f}s")
    print(f"P95 residual:              {fit['p95_abs_residual_seconds']:.4f}s")
    print(f"Diagnosis:                 {report['classification']}")
    if report.get("corrected_output"):
        print(f"Corrected audio:           {report['corrected_output']}")


def _fit_event_grids(reference: Any, generated: Any, np_module: Any) -> dict[str, Any]:
    if len(reference) < 3 or len(generated) < 3:
        raise RuntimeError("Need at least three events in both beat grids")
    best = None
    max_shift = min(16, max(len(reference), len(generated)) - 2)
    for index_shift in range(-max_shift, max_shift + 1):
        reference_start = max(0, -index_shift)
        generated_start = max(0, index_shift)
        count = min(
            len(reference) - reference_start,
            len(generated) - generated_start,
        )
        if count < 3:
            continue
        ref = reference[reference_start : reference_start + count]
        gen = generated[generated_start : generated_start + count]
        slope, intercept = np_module.polyfit(ref, gen, 1)
        if not 0.80 <= slope <= 1.20:
            continue
        residuals = gen - (intercept + slope * ref)
        median_abs = float(np_module.median(np_module.abs(residuals)))
        # Prefer a stable fit, then the phase requiring the smallest shift at
        # time zero.  This disambiguates equivalent whole-beat index offsets.
        score = median_abs + min(abs(float(intercept)), 1.0) * 1e-3
        candidate = (score, abs(float(intercept)), -count, index_shift, ref, gen)
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not fit reference and generated beat grids")

    _, _, _, index_shift, ref, gen = best
    # Remove gross outliers and refit once.
    slope, intercept = np_module.polyfit(ref, gen, 1)
    residuals = gen - (intercept + slope * ref)
    center = float(np_module.median(residuals))
    mad = float(np_module.median(np_module.abs(residuals - center)))
    limit = max(0.08, 4.0 * 1.4826 * mad)
    keep = np_module.abs(residuals - center) <= limit
    if int(np_module.sum(keep)) >= 3:
        ref = ref[keep]
        gen = gen[keep]
        slope, intercept = np_module.polyfit(ref, gen, 1)
        residuals = gen - (intercept + slope * ref)

    return {
        "reference": ref,
        "generated": gen,
        "residuals": residuals,
        "index_shift": int(index_shift),
        "slope": float(slope),
        "intercept": float(intercept),
        "median_abs": float(np_module.median(np_module.abs(residuals))),
        "p95_abs": float(np_module.percentile(np_module.abs(residuals), 95.0)),
    }


def _segment_errors(fit: dict[str, Any], extend_stride: float, np_module: Any) -> list[dict[str, Any]]:
    if extend_stride <= 0.0:
        return []
    reference = fit["reference"]
    generated = fit["generated"]
    raw_error = generated - reference
    segments = []
    max_time = float(reference[-1]) if len(reference) else 0.0
    start = 0.0
    while start <= max_time:
        end = start + extend_stride
        mask = (reference >= start) & (reference < end)
        if np_module.any(mask):
            values = raw_error[mask]
            segments.append(
                {
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "matched_beats": int(np_module.sum(mask)),
                    "median_error_seconds": round(
                        float(np_module.median(values)), 6
                    ),
                }
            )
        start = end
    return segments


def _classify(fit: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    offset = abs(float(fit["intercept"]))
    drift_per_minute = abs((float(fit["slope"]) - 1.0) * 60.0)
    p95 = float(fit["p95_abs"])
    jumps = []
    for previous, current in zip(segments, segments[1:]):
        jumps.append(
            abs(
                float(current["median_error_seconds"])
                - float(previous["median_error_seconds"])
            )
        )
    if jumps and max(jumps) >= 0.10:
        return "window_boundary_jump"
    if drift_per_minute >= 0.08:
        return "tempo_drift"
    if offset >= 0.06 and p95 <= 0.10:
        return "constant_offset"
    if p95 >= 0.12:
        return "local_rhythm_mismatch"
    return "aligned"


def _correct_audio_alignment(
    generated_audio: Path,
    reference_audio: Path,
    output: Path,
    fit: dict[str, Any],
    mode: str,
) -> None:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(
        str(generated_audio), dtype="float32", always_2d=True
    )
    reference_info = sf.info(str(reference_audio))
    target_frames = int(round(reference_info.duration * sample_rate))
    slope = float(fit["slope"])
    intercept = float(fit["intercept"])

    if mode == "affine":
        if not 0.90 <= slope <= 1.10:
            raise RuntimeError(
                f"Refusing unsafe affine correction with tempo ratio {slope:.4f}"
            )
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError(
                "Affine correction requires librosa in the MusicGen environment"
            ) from exc
        channels_first = audio.T
        stretched = librosa.effects.time_stretch(channels_first, rate=slope)
        audio = np.asarray(stretched, dtype=np.float32).T
        shift_seconds = -intercept / slope
    elif mode == "shift":
        shift_seconds = -intercept
    else:
        raise ValueError(f"Unsupported correction mode: {mode}")

    shift_frames = int(round(shift_seconds * sample_rate))
    if shift_frames > 0:
        audio = np.concatenate(
            [np.zeros((shift_frames, audio.shape[1]), dtype=np.float32), audio],
            axis=0,
        )
    elif shift_frames < 0:
        audio = audio[min(len(audio), -shift_frames) :]
    if len(audio) < target_frames:
        audio = np.concatenate(
            [
                audio,
                np.zeros((target_frames - len(audio), audio.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
    else:
        audio = audio[:target_frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, sample_rate, subtype="PCM_16")


def _events(values: Any, np_module: Any) -> Any:
    events = np_module.asarray(values, dtype=np_module.float64).reshape(-1)
    events = events[np_module.isfinite(events)]
    return np_module.unique(events[events >= 0.0])


def _json_fit(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "matched_events": int(len(fit["reference"])),
        "index_shift": int(fit["index_shift"]),
        "offset_seconds": round(float(fit["intercept"]), 8),
        "tempo_ratio": round(float(fit["slope"]), 10),
        "drift_seconds_per_minute": round(
            (float(fit["slope"]) - 1.0) * 60.0, 8
        ),
        "median_abs_residual_seconds": round(float(fit["median_abs"]), 8),
        "p95_abs_residual_seconds": round(float(fit["p95_abs"]), 8),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure generated/reference Beat This alignment."
    )
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--generated-audio", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extend-stride", type=float, default=18.0)
    parser.add_argument(
        "--correction", choices=["none", "shift", "affine"], default="none"
    )
    parser.add_argument("--corrected-output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in (args.reference_audio, args.generated_audio):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.correction != "none" and args.corrected_output is None:
        raise ValueError("--corrected-output is required with --correction")
    report = analyze_alignment(
        reference_audio=args.reference_audio,
        generated_audio=args.generated_audio,
        checkpoint=args.checkpoint,
        device=args.device,
        extend_stride=args.extend_stride,
        correction=args.correction,
        corrected_output=args.corrected_output,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(report)
    print(f"Report:                    {args.output_report.resolve()}")


if __name__ == "__main__":
    main()

