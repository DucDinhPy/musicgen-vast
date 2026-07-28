#!/usr/bin/env python3
"""Bar-aligned inference pipeline for a paired MusicGen Melody checkpoint.

This script mirrors the v2 training data shape:

    melody audio -> 128 BPM aligned melody chunks + structured prompt -> background audio

It is intended for melody-like audio. If your source is a raw vocal recording,
extract/render a clean melody_piano.wav first for the closest train/test match.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
from pathlib import Path

import torch


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy drums, rolling bass, bright synth chords, build ups and drops, no lead vocal"
)

DEFAULT_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
)


def infer(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "work"
    chunks_dir = work_dir / "melody_chunks"
    generated_dir = args.output_dir / "generated_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    melody_input = _prepare_melody_input(args, work_dir)

    bpm_row = _estimate_or_override_bpm(
        melody_input,
        target_bpm=args.target_bpm,
        source_bpm=args.source_bpm,
        first_downbeat=args.first_downbeat,
        min_confidence=args.min_confidence,
    )
    _write_json(args.output_dir / "bpm_report.json", bpm_row)

    aligned_melody = work_dir / "aligned_128" / "melody_piano.wav"
    if args.tempo_mode == "preserve":
        _normalize_audio(
            source=melody_input,
            target=aligned_melody,
            sample_rate=args.sample_rate,
            channels=1,
            overwrite=args.overwrite,
        )
    elif args.tempo_mode == "warp_to_target":
        _warp_audio(
            source=melody_input,
            target=aligned_melody,
            source_bpm=float(bpm_row["bpm"]),
            target_bpm=args.target_bpm,
            first_downbeat=float(bpm_row["first_downbeat"]),
            sample_rate=args.sample_rate,
            channels=1,
            max_warp_factor=args.max_warp_factor,
            allow_large_warp=args.allow_large_warp,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported tempo mode: {args.tempo_mode}")

    conditioning_audio = aligned_melody
    arrangement_sections = None
    if args.pre_beats > 0 or args.post_beats > 0:
        conditioning_audio = work_dir / "arranged_condition.wav"
        arrangement_bpm = _resolve_arrangement_bpm(float(bpm_row["bpm"]), args)
        conditioning_audio, arrangement_sections = _apply_context_beats(
            source=aligned_melody,
            target=conditioning_audio,
            bpm=arrangement_bpm,
            first_downbeat=float(bpm_row["first_downbeat"]) if args.tempo_mode == "preserve" else 0.0,
            pre_beats=args.pre_beats,
            post_beats=args.post_beats,
            sf_module=sf,
            np_module=np,
            subtype=args.subtype,
            overwrite=args.overwrite,
        )
        _write_json(
            args.output_dir / "arrangement_report.json",
            {
                "source_audio": str(aligned_melody),
                "conditioning_audio": str(conditioning_audio),
                "source_bpm": float(bpm_row["bpm"]),
                "arrangement_bpm": arrangement_bpm,
                "first_downbeat": float(bpm_row["first_downbeat"]),
                "pre_beats": args.pre_beats,
                "post_beats": args.post_beats,
                "sections": arrangement_sections,
            },
        )

    chunk_seconds = args.bars_per_chunk * 4.0 * 60.0 / args.target_bpm
    hop_seconds = args.hop_bars * 4.0 * 60.0 / args.target_bpm
    starts = _chunk_starts(
        duration=_duration(conditioning_audio),
        chunk_seconds=chunk_seconds,
        hop_seconds=hop_seconds,
        include_partial_final=args.include_partial_final,
        min_final_seconds=args.min_final_seconds,
    )
    if not starts:
        raise RuntimeError(
            f"Aligned melody is too short for {args.bars_per_chunk} bars "
            f"({chunk_seconds:.2f}s). Use --include-partial-final or lower --bars-per-chunk."
        )

    print(f"Input audio:       {args.input_audio.resolve()}")
    print(f"Input kind:        {args.input_kind}")
    print(f"Tempo mode:        {args.tempo_mode}")
    print(f"Melody input:      {melody_input.resolve()}")
    print(f"Checkpoint:        {args.checkpoint.resolve() if args.checkpoint else 'base model only'}")
    print(f"Detected BPM:      {float(bpm_row['bpm']):.2f}")
    print(f"First downbeat:    {float(bpm_row['first_downbeat']):.3f}s")
    print(f"Confidence:        {float(bpm_row['confidence']):.2f}")
    print(f"Aligned melody:    {aligned_melody.resolve()}")
    print(f"Condition audio:   {conditioning_audio.resolve()}")
    print(f"Chunk seconds:     {chunk_seconds:.3f}")
    print(f"Chunks:            {len(starts)}")
    print(f"Output dir:        {args.output_dir.resolve()}")
    print("")

    print(f"Loading model: {args.model}")
    device = torch.device(args.device)
    model = MusicGen.get_pretrained(args.model, device=str(device))
    if args.checkpoint:
        _load_partial_checkpoint(model, args.checkpoint, device)

    generated_paths: list[Path] = []
    metadata_rows = []

    for chunk_index, start in enumerate(starts):
        duration = min(chunk_seconds, _duration(conditioning_audio) - start)
        if duration <= 0:
            continue

        melody_chunk = chunks_dir / f"chunk_{chunk_index:04d}.wav"
        output_chunk = generated_dir / f"chunk_{chunk_index:04d}.wav"
        _write_chunk(conditioning_audio, melody_chunk, start, duration, args.sample_rate, args.overwrite)

        if arrangement_sections:
            sections = _sections_for_chunk(arrangement_sections, start, duration)
            active = [
                [section[1], section[2]]
                for section in sections
                if section[0] == "main_vocal_melody"
            ]
        else:
            active = _detect_active_intervals(
                melody_path=melody_chunk,
                frame_seconds=args.frame_seconds,
                threshold_db=args.threshold_db,
                min_active_seconds=args.min_active_seconds,
                merge_gap_seconds=args.merge_gap_seconds,
            )
            sections = _build_sections(active, duration)
        structure_text = _structure_text(sections)
        bpm_label = _resolve_bpm_label(
            row_bpm=float(bpm_row["bpm"]),
            target_bpm=args.target_bpm,
            mode=args.prompt_bpm_mode,
            label_min=args.bpm_label_min,
            label_max=args.bpm_label_max,
        )
        prompt = _join_prompt(args.prompt, bpm_label, structure_text, args.disable_structure_prompt)

        model.set_generation_params(
            duration=duration,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            cfg_coef=args.cfg_coef,
        )

        melody, sr = _load_wav(melody_chunk, sf)
        melody = convert_audio(melody, sr, model.sample_rate, model.audio_channels)

        print(f"[{chunk_index + 1}/{len(starts)}] start={start:.3f}s duration={duration:.3f}s")
        print(f"  {structure_text}")
        with torch.no_grad():
            wav = model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody,
                melody_sample_rate=model.sample_rate,
                progress=True,
            )

        audio = wav[0].detach().cpu().float().numpy().T
        sf.write(str(output_chunk), audio, model.sample_rate, subtype=args.subtype)
        generated_paths.append(output_chunk)
        metadata_rows.append(
            {
                "chunk_index": chunk_index,
                "start": start,
                "duration": duration,
                "bpm": args.target_bpm,
                "bars_per_chunk": args.bars_per_chunk,
                "input_audio": str(melody_chunk),
                "generated_audio": str(output_chunk),
                "text": prompt,
                "melody_active": active,
                "sections": sections,
                "structure_text": structure_text,
            }
        )

    _write_jsonl(args.output_dir / "inference_metadata.jsonl", metadata_rows)
    _concat_audio(generated_paths, args.output, sf, np, subtype=args.subtype)

    if args.keep_aligned_melody:
        final_melody = args.output_dir / "aligned_melody_128.wav"
        _copy_audio(conditioning_audio, final_melody)
        print(f"Aligned melody copy: {final_melody}")

    print("")
    print("Done.")
    print(f"Generated chunks: {len(generated_paths)}")
    print(f"Final output:     {args.output.resolve()}")


def _estimate_or_override_bpm(
    audio_path: Path,
    target_bpm: float,
    source_bpm: float | None,
    first_downbeat: float | None,
    min_confidence: float,
) -> dict:
    if source_bpm is not None:
        return {
            "track_id": audio_path.stem,
            "audio": str(audio_path),
            "bpm": float(source_bpm),
            "target_bpm": target_bpm,
            "first_downbeat": float(first_downbeat or 0.0),
            "duration": _duration(audio_path),
            "beat_count": None,
            "confidence": 1.0,
            "needs_review": False,
            "source": "manual",
        }

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: librosa/numpy. Install in the MusicGen venv with "
            "`python -m pip install librosa numpy`."
        ) from exc

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    detected_downbeat = float(beat_times[0]) if len(beat_times) else 0.0
    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_count = int(len(beat_times))

    tempo_ratio = min(tempo, target_bpm) / max(tempo, target_bpm) if tempo > 0 else 0.0
    beat_density = min(1.0, beat_count / max(1.0, duration / 2.0))
    confidence = float(max(0.0, min(1.0, 0.65 * tempo_ratio + 0.35 * beat_density)))

    return {
        "track_id": audio_path.stem,
        "audio": str(audio_path),
        "bpm": tempo,
        "target_bpm": target_bpm,
        "first_downbeat": float(first_downbeat if first_downbeat is not None else detected_downbeat),
        "duration": duration,
        "beat_count": beat_count,
        "confidence": confidence,
        "needs_review": confidence < min_confidence,
        "source": "librosa",
    }


def _resolve_arrangement_bpm(row_bpm: float, args: argparse.Namespace) -> float:
    if args.arrangement_bpm is not None:
        return float(args.arrangement_bpm)
    if args.bpm_label_min is not None and args.bpm_label_max is not None:
        return _fold_bpm_to_range(row_bpm, args.bpm_label_min, args.bpm_label_max)
    return row_bpm


def _apply_context_beats(
    source: Path,
    target: Path,
    bpm: float,
    first_downbeat: float,
    pre_beats: float,
    post_beats: float,
    sf_module,
    np_module,
    subtype: str,
    overwrite: bool,
) -> tuple[Path, list[list]]:
    if target.exists() and not overwrite:
        duration = _duration(target)
        pre_seconds = pre_beats * 60.0 / bpm
        post_seconds = post_beats * 60.0 / bpm
        return target, _arrangement_sections(duration, pre_seconds, post_seconds)

    audio, sr = sf_module.read(str(source), dtype="float32", always_2d=True)
    first_downbeat = max(0.0, min(float(first_downbeat), len(audio) / sr))
    trim_start = int(round(first_downbeat * sr))
    melody_audio = audio[trim_start:]

    pre_seconds = pre_beats * 60.0 / bpm
    post_seconds = post_beats * 60.0 / bpm
    pre = np_module.zeros((int(round(pre_seconds * sr)), audio.shape[1]), dtype=audio.dtype)
    post = np_module.zeros((int(round(post_seconds * sr)), audio.shape[1]), dtype=audio.dtype)
    arranged = np_module.concatenate([pre, melody_audio, post], axis=0)

    target.parent.mkdir(parents=True, exist_ok=True)
    sf_module.write(str(target), arranged, sr, subtype=subtype)
    sections = [
        ["intro_no_melody", 0.0, round(pre_seconds, 3)],
        ["main_vocal_melody", round(pre_seconds, 3), round(pre_seconds + len(melody_audio) / sr, 3)],
        [
            "outro_no_melody",
            round(pre_seconds + len(melody_audio) / sr, 3),
            round(pre_seconds + len(melody_audio) / sr + post_seconds, 3),
        ],
    ]
    return target, sections


def _arrangement_sections(duration: float, pre_seconds: float, post_seconds: float) -> list[list]:
    main_start = pre_seconds
    main_end = max(main_start, duration - post_seconds)
    return [
        ["intro_no_melody", 0.0, round(main_start, 3)],
        ["main_vocal_melody", round(main_start, 3), round(main_end, 3)],
        ["outro_no_melody", round(main_end, 3), round(duration, 3)],
    ]


def _sections_for_chunk(sections: list[list], chunk_start: float, chunk_duration: float) -> list[list]:
    chunk_end = chunk_start + chunk_duration
    local = []
    for label, start, end in sections:
        overlap_start = max(float(start), chunk_start)
        overlap_end = min(float(end), chunk_end)
        if overlap_end <= overlap_start:
            continue
        local.append([
            label,
            round(overlap_start - chunk_start, 3),
            round(overlap_end - chunk_start, 3),
        ])
    return local


def _prepare_melody_input(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.input_kind in {"direct_audio", "piano_melody"}:
        return args.input_audio
    if args.input_kind != "vocal":
        raise ValueError(f"Unsupported input kind: {args.input_kind}")

    game_dir: Path = args.game_dir
    game_model_path: Path = args.game_model_path
    game_python: Path = args.game_python
    infer_script = game_dir / "infer.py"

    if not infer_script.exists():
        raise FileNotFoundError(f"GAME infer.py not found: {infer_script}")
    if not game_model_path.exists():
        raise FileNotFoundError(f"GAME model not found: {game_model_path}")
    if not game_python.exists():
        raise FileNotFoundError(f"GAME Python not found: {game_python}")

    game_work = work_dir / "game"
    game_work.mkdir(parents=True, exist_ok=True)
    staged_vocal = game_work / "source_vocal.wav"
    midi_path = staged_vocal.with_suffix(".mid")
    rendered_piano = game_work / "melody_piano.wav"

    if rendered_piano.exists() and not args.overwrite:
        print(f"[skip] vocal->piano exists: {rendered_piano}")
        return rendered_piano

    shutil.copy2(args.input_audio, staged_vocal)
    _run_game_extract(
        game_python=game_python,
        infer_script=infer_script,
        model_path=game_model_path,
        vocal_path=staged_vocal,
        game_extra_args=args.game_extra_args,
        overwrite=args.overwrite,
    )
    _render_midi_to_piano(
        midi_path=midi_path,
        output_wav=rendered_piano,
        soundfont=args.soundfont,
        raw_sample_rate=args.render_raw_sample_rate,
        sample_rate=args.sample_rate,
        loudnorm=args.render_loudnorm,
        keep_raw=args.keep_render_raw,
        overwrite=args.overwrite,
    )
    return rendered_piano


def _run_game_extract(
    game_python: Path,
    infer_script: Path,
    model_path: Path,
    vocal_path: Path,
    game_extra_args: str,
    overwrite: bool,
) -> None:
    expected = [vocal_path.with_suffix(suffix) for suffix in (".mid", ".txt", ".csv")]
    if overwrite:
        for path in expected:
            if path.exists():
                path.unlink()

    cmd = [
        str(game_python),
        str(infer_script),
        "extract",
        str(vocal_path),
        "-m",
        str(model_path.resolve()),
        "--output-formats",
        "mid,txt,csv",
    ]
    if game_extra_args:
        cmd.extend(shlex.split(game_extra_args))

    print("Running GAME melody extraction...")
    subprocess.run(cmd, cwd=str(infer_script.parent), check=True)
    if not vocal_path.with_suffix(".mid").exists():
        raise RuntimeError(f"GAME did not write expected MIDI: {vocal_path.with_suffix('.mid')}")


def _render_midi_to_piano(
    midi_path: Path,
    output_wav: Path,
    soundfont: Path | None,
    raw_sample_rate: int,
    sample_rate: int,
    loudnorm: str,
    keep_raw: bool,
    overwrite: bool,
) -> None:
    if output_wav.exists() and not overwrite:
        return
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    soundfont = soundfont or _discover_soundfont()
    if not soundfont.exists():
        raise FileNotFoundError(f"SoundFont does not exist: {soundfont}")

    raw_wav = output_wav.with_name(output_wav.stem + "_raw.wav")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (raw_wav, output_wav):
            if path.exists():
                path.unlink()

    print("Rendering GAME MIDI to piano WAV...")
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            str(soundfont),
            str(midi_path),
            "-F",
            str(raw_wav),
            "-r",
            str(raw_sample_rate),
        ],
        check=True,
    )

    audio_filter = f"loudnorm={loudnorm}" if loudnorm else "anull"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-af",
            audio_filter,
            str(output_wav),
        ],
        check=True,
    )
    if raw_wav.exists() and not keep_raw:
        raw_wav.unlink()


def _discover_soundfont() -> Path:
    for candidate in DEFAULT_SOUNDFONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    for root in (Path("/usr/share/sounds"), Path("/usr/share/soundfonts")):
        if not root.exists():
            continue
        matches = sorted(root.rglob("*.sf2"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No .sf2 SoundFont found. Install one with: "
        "apt-get install -y fluidsynth fluid-soundfont-gm"
    )


def _warp_audio(
    source: Path,
    target: Path,
    source_bpm: float,
    target_bpm: float,
    first_downbeat: float,
    sample_rate: int,
    channels: int,
    max_warp_factor: float,
    allow_large_warp: bool,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    warp_factor = target_bpm / source_bpm
    if not allow_large_warp and (warp_factor > max_warp_factor or warp_factor < 1.0 / max_warp_factor):
        raise RuntimeError(
            f"Refusing large tempo warp: source_bpm={source_bpm:.2f}, "
            f"target_bpm={target_bpm:.2f}, factor={warp_factor:.3f}. "
            "Use --tempo-mode preserve for vocal-lock inference, or pass "
            "--allow-large-warp if you intentionally want this."
        )
    filters = _atempo_filters(warp_factor)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, first_downbeat):.6f}",
            "-i",
            str(source),
            "-af",
            filters,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(target),
        ],
        check=True,
    )


def _normalize_audio(
    source: Path,
    target: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> None:
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(target),
        ],
        check=True,
    )


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


def _chunk_starts(
    duration: float,
    chunk_seconds: float,
    hop_seconds: float,
    include_partial_final: bool,
    min_final_seconds: float,
) -> list[float]:
    starts = []
    start = 0.0
    while start + chunk_seconds <= duration + 1e-6:
        starts.append(start)
        start += hop_seconds

    if include_partial_final:
        final_start = starts[-1] + hop_seconds if starts else 0.0
        if duration - final_start >= min_final_seconds:
            starts.append(final_start)

    return starts


def _write_chunk(source: Path, target: Path, start: float, duration: float, sample_rate: int, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(target),
        ],
        check=True,
    )


def _detect_active_intervals(
    melody_path: Path,
    frame_seconds: float,
    threshold_db: float,
    min_active_seconds: float,
    merge_gap_seconds: float,
) -> list[list[float]]:
    import numpy as np
    import soundfile as sf

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


def _build_sections(active: list[list[float]], duration: float) -> list[list]:
    sections: list[list] = []
    cursor = 0.0

    if not active:
        return [["instrumental_intro", 0.0, round(duration, 3)]]

    for _index, (start, end) in enumerate(active):
        if start > cursor + 0.25:
            label = "intro" if not sections else "break_or_fill"
            sections.append([label, round(cursor, 3), round(start, 3)])
        sections.append(["melody_backing", round(start, 3), round(end, 3)])
        cursor = end

    if cursor < duration - 0.25:
        sections.append(["transition_or_outro", round(cursor, 3), round(duration, 3)])

    return sections


def _structure_text(sections: list[list]) -> str:
    parts = [f"{label} {start:.1f}-{end:.1f}s" for label, start, end in sections]
    return "structure: " + ", ".join(parts)


def _resolve_bpm_label(
    row_bpm: float,
    target_bpm: float,
    mode: str,
    label_min: float | None,
    label_max: float | None,
) -> str | None:
    if mode == "none":
        return None
    if mode == "target":
        bpm = target_bpm
    elif mode == "auto":
        bpm = row_bpm
    else:
        raise ValueError(f"Unsupported prompt BPM mode: {mode}")

    if label_min is not None and label_max is not None:
        bpm = _fold_bpm_to_range(bpm, label_min, label_max)
    return f"{bpm:.1f} bpm"


def _fold_bpm_to_range(bpm: float, label_min: float, label_max: float) -> float:
    if bpm <= 0:
        return bpm
    folded = bpm
    while folded < label_min:
        folded *= 2.0
    while folded > label_max:
        folded /= 2.0
    return folded


def _join_prompt(
    base_prompt: str,
    bpm_label: str | None,
    structure_text: str,
    disable_structure_prompt: bool,
) -> str:
    parts = [base_prompt] if base_prompt else []
    if bpm_label:
        parts.append(bpm_label)
    if disable_structure_prompt:
        return ", ".join(parts)
    parts.append(structure_text)
    return ", ".join(parts)


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _load_partial_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)
    trainable = state.get("trainable")
    if not isinstance(trainable, dict):
        raise RuntimeError(f"Checkpoint has no trainable state: {checkpoint}")
    missing, unexpected = model.lm.load_state_dict(trainable, strict=False)
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Partial keys:      {len(trainable)}")
    print(f"Missing keys:      {len(missing)}")
    print(f"Unexpected keys:   {len(unexpected)}")


def _load_wav(path: Path, sf_module) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf_module.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(audio).transpose(0, 1).contiguous()
    return tensor, int(sample_rate)


def _concat_audio(paths: list[Path], output: Path, sf_module, np_module, subtype: str) -> None:
    if not paths:
        raise RuntimeError("No generated chunks to concatenate.")

    arrays = []
    sample_rate = None
    for path in paths:
        audio, sr = sf_module.read(str(path), dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise RuntimeError(f"Sample rate mismatch: {path} has {sr}, expected {sample_rate}")
        arrays.append(audio)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf_module.write(str(output), np_module.concatenate(arrays, axis=0), sample_rate, subtype=subtype)


def _copy_audio(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _duration(path: Path) -> float:
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


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one melody audio file like the v2 bar-aligned dataset and generate backing audio."
    )
    parser.add_argument("--input-audio", type=Path, required=True, help="Input audio.")
    parser.add_argument(
        "--input-kind",
        choices=("direct_audio", "piano_melody", "vocal"),
        default="direct_audio",
        help=(
            "direct_audio uses the input as-is, piano_melody is an alias for direct_audio, "
            "vocal runs GAME->piano first."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fine-tuned checkpoint_step_*.pt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for chunks/reports.")
    parser.add_argument("--output", type=Path, required=True, help="Final generated WAV.")
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-bpm-mode",
        choices=("auto", "target", "none"),
        default="auto",
        help="Append BPM label to prompt from detected BPM, target BPM, or not at all. Default: auto.",
    )
    parser.add_argument(
        "--bpm-label-min",
        type=float,
        default=None,
        help="Optional min BPM for folding labels, e.g. 120 for Vinahouse.",
    )
    parser.add_argument(
        "--bpm-label-max",
        type=float,
        default=None,
        help="Optional max BPM for folding labels, e.g. 150 for Vinahouse.",
    )
    parser.add_argument("--target-bpm", type=float, default=128.0)
    parser.add_argument(
        "--pre-beats",
        type=float,
        default=0.0,
        help="Add this many silent beats before the detected/overridden first downbeat.",
    )
    parser.add_argument(
        "--post-beats",
        type=float,
        default=0.0,
        help="Add this many silent beats after the trimmed input audio.",
    )
    parser.add_argument(
        "--arrangement-bpm",
        type=float,
        default=None,
        help="Manual BPM used to convert pre/post beats to seconds. Default: detected/folded BPM.",
    )
    parser.add_argument(
        "--tempo-mode",
        choices=("preserve", "warp_to_target"),
        default="preserve",
        help=(
            "preserve keeps input timing/duration for vocal-lock inference. "
            "warp_to_target trims first downbeat and time-stretches to target BPM."
        ),
    )
    parser.add_argument(
        "--max-warp-factor",
        type=float,
        default=1.12,
        help="Safety limit for warp_to_target. Default allows about +/-12%% tempo change.",
    )
    parser.add_argument(
        "--allow-large-warp",
        action="store_true",
        help="Allow aggressive tempo warps. Not recommended for raw vocal.",
    )
    parser.add_argument("--source-bpm", type=float, default=None, help="Manual source BPM override.")
    parser.add_argument("--first-downbeat", type=float, default=None, help="Manual first downbeat seconds override.")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--game-dir", type=Path, default=Path("/workspace/GAME"))
    parser.add_argument("--game-model-path", type=Path, default=Path("/workspace/models/game/game.pt"))
    parser.add_argument("--game-python", type=Path, default=Path("/workspace/.venv-game/bin/python"))
    parser.add_argument("--game-extra-args", default="")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--render-raw-sample-rate", type=int, default=44100)
    parser.add_argument("--render-loudnorm", default="I=-20:TP=-2:LRA=11")
    parser.add_argument("--keep-render-raw", action="store_true")
    parser.add_argument("--bars-per-chunk", type=int, default=16)
    parser.add_argument("--hop-bars", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--include-partial-final", action="store_true", default=True)
    parser.add_argument("--no-include-partial-final", dest="include_partial_final", action="store_false")
    parser.add_argument("--min-final-seconds", type=float, default=8.0)
    parser.add_argument("--frame-seconds", type=float, default=0.25)
    parser.add_argument("--threshold-db", type=float, default=-45.0)
    parser.add_argument("--min-active-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--disable-structure-prompt", action="store_true")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--subtype", default="PCM_16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-aligned-melody", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    infer(args)


if __name__ == "__main__":
    main()
