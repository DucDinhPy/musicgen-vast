#!/usr/bin/env python3
"""End-to-end interactive inference pipeline for MusicGen V5 Beat This.

Pipeline:

    original song
    -> BS-RoFormer HyperACE V2 -> vocals + reference instrumental
    -> OpenVPI GAME on vocals -> MIDI -> rendered piano melody
    -> Beat This! on reference instrumental -> Schema-2 rhythm features
    -> interactive review/override of generation settings
    -> MusicGen V5 -> generated background instrumental
    -> mix generated background with the separated vocal

The preprocessing intentionally reuses the V5 dataset/trainer helpers so BPM,
event cleanup, downbeat quality gating, and the nine 50 Hz rhythm features are
identical to training.

The V5 residual logit conditioner is applied inside ``LMModel.forward`` during
autoregressive sampling.  It is aligned through AudioCraft's codebook pattern,
including delayed codebooks and MusicGen's overlapping windows for generation
longer than 30 seconds.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

from prepare_musicgen_v5_beatthis_dataset import (
    DEFAULT_CHECKPOINT as DEFAULT_BEATTHIS_CHECKPOINT,
    _create_tracker,
    _load_condition,
    _save_condition,
    _summarize_events,
    _validate_cached_condition,
    _validate_summary,
)
from train_musicgen_v5_beatthis import (
    RHYTHM_FEATURE_NAMES,
    V5_CHECKPOINT_KIND,
    V5_CONDITION_SCHEMA,
    V5_DETECTOR,
    V5_TEMPO_ESTIMATOR,
    BeatThisLogitConditioner,
    _events_to_features,
    _fit_feature_length,
    _infer_vocab_size,
    _load_condition_arrays,
    _load_v1_base_checkpoint,
)


DEFAULT_SEPARATOR_MODEL = (
    "roformer-model-bs-roformer-hyperace-v2-instrumental-by-pcunwa"
)
DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy drums, rolling bass, bright synth chords, build ups and drops, "
    "no lead vocal"
)
DEFAULT_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def run_pipeline(args: argparse.Namespace) -> None:
    _validate_cli(args)
    source_audio = args.input_audio.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            Path("/workspace/musicgen-vast/vast_data/outputs/v5_generate")
            / _safe_name(source_audio.stem)
        )
    )
    work_dir = output_dir / "work"
    separator_input_dir = work_dir / "separator_input"
    stems_dir = work_dir / "stems"
    game_dir = work_dir / "game"
    condition_path = work_dir / "beatthis_instrumental_schema2.npz"
    generated_path = output_dir / f"{_safe_name(source_audio.stem)}_background_instrumental.wav"
    mixed_path = output_dir / f"{_safe_name(source_audio.stem)}_remix_with_vocal.wav"

    _guard_output_files((generated_path, mixed_path), args.overwrite)
    _validate_or_write_source_manifest(
        source_audio=source_audio,
        manifest_path=work_dir / "source.json",
        overwrite=args.reprocess,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_duration = _audio_duration(source_audio, args.ffprobe)
    staged_source = separator_input_dir / "source.wav"
    if args.reprocess or not staged_source.exists():
        separator_input_dir.mkdir(parents=True, exist_ok=True)
        _convert_source_to_wav(source_audio, staged_source, args.ffmpeg)

    vocals_path, reference_instrumental_path = _separate_hyperace(
        args=args,
        staged_source=staged_source,
        separator_input_dir=separator_input_dir,
        stems_dir=stems_dir,
    )
    melody_path = _prepare_game_melody(
        args=args,
        vocals_path=vocals_path,
        output_dir=game_dir,
        target_duration=source_duration,
    )
    beat_summary = _prepare_beatthis_condition(
        args=args,
        instrumental_path=reference_instrumental_path,
        condition_path=condition_path,
    )

    detected_config = _initial_generation_config(
        args=args,
        source_duration=source_duration,
        beat_summary=beat_summary,
    )
    _print_processing_summary(
        args=args,
        source_audio=source_audio,
        vocals_path=vocals_path,
        reference_instrumental_path=reference_instrumental_path,
        melody_path=melody_path,
        condition_path=condition_path,
        beat_summary=beat_summary,
        generated_path=generated_path,
        mixed_path=mixed_path,
        config=detected_config,
    )
    config = _review_config(detected_config, assume_yes=args.yes)
    if config is None:
        print("Cancelled. Preprocessed files were kept in:", work_dir)
        return

    _validate_generation_config(config, args.bpm_min, args.bpm_max)
    if float(config["duration"]) > source_duration + 0.05:
        print(
            "[warn] Generation duration exceeds the source song. Melody and "
            "Beat This events after the source ends contain no new information."
        )
    config["prompt"] = _append_bpm_to_prompt(
        str(config["base_prompt"]), float(config["bpm"])
    )
    _print_final_config(config)
    _write_json(
        output_dir / "generation_config.json",
        {
            "input_audio": str(source_audio),
            "checkpoint": str(args.checkpoint.resolve()),
            "separator_model": args.bs_model,
            "beat_this_checkpoint": args.beatthis_checkpoint,
            "vocals": str(vocals_path),
            "reference_instrumental": str(reference_instrumental_path),
            "melody_piano": str(melody_path),
            "beatthis_condition": str(condition_path),
            "detected": _json_safe(beat_summary),
            "generation": _json_safe(config),
            "outputs": {
                "background_instrumental": str(generated_path),
                "background_instrumental_plus_vocal": str(mixed_path),
            },
        },
    )

    if args.dry_run:
        print("Dry run complete. Preprocessing/config review ran; generation was skipped.")
        return

    condition = _condition_for_features(
        condition_path=condition_path,
        detected_bpm=float(beat_summary["bpm"]),
        selected_bpm=float(config["bpm"]),
    )
    _release_cuda()
    _generate_v5(
        args=args,
        melody_path=melody_path,
        condition=condition,
        config=config,
        output_path=generated_path,
    )
    _mix_vocal_and_background(
        ffmpeg=args.ffmpeg,
        background_path=generated_path,
        vocal_path=vocals_path,
        output_path=mixed_path,
        background_gain_db=float(config["background_gain_db"]),
        vocal_gain_db=float(config["vocal_gain_db"]),
        sample_rate=args.mix_sample_rate,
    )

    print("")
    print("===== DONE =====")
    print(f"Background instrumental:       {generated_path}")
    print(f"Background instrumental+vocal: {mixed_path}")
    print(f"Config:                        {output_dir / 'generation_config.json'}")


def _validate_cli(args: argparse.Namespace) -> None:
    if not args.input_audio.exists() or not args.input_audio.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {args.input_audio}")
    if args.input_audio.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported input audio extension: {args.input_audio.suffix}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"V5 checkpoint does not exist: {args.checkpoint}")
    if args.game_model_path is None or not args.game_model_path.exists():
        raise FileNotFoundError(f"GAME model does not exist: {args.game_model_path}")
    if not (args.game_dir / "infer.py").exists():
        raise FileNotFoundError(f"GAME infer.py does not exist: {args.game_dir / 'infer.py'}")
    if not args.game_python.exists():
        raise FileNotFoundError(f"GAME Python does not exist: {args.game_python}")
    if not args.bs_python.exists():
        raise FileNotFoundError(f"BS-RoFormer Python does not exist: {args.bs_python}")
    _require_executable(args.ffmpeg)
    _require_executable(args.ffprobe)
    _require_executable("fluidsynth")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but CUDA is unavailable.")
    if args.bpm_min <= 0 or args.bpm_max <= args.bpm_min:
        raise ValueError("BPM range must satisfy 0 < bpm-min < bpm-max.")


def _guard_output_files(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output already exists. Use --overwrite or another --output-dir:\n"
            + joined
        )


def _validate_or_write_source_manifest(
    source_audio: Path,
    manifest_path: Path,
    overwrite: bool,
) -> None:
    stat = source_audio.stat()
    signature = {
        "path": str(source_audio),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if manifest_path.exists() and not overwrite:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached != signature:
            raise RuntimeError(
                "This output directory contains preprocessing for another or "
                "modified input. Use another --output-dir or --overwrite."
            )
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, signature)


def _convert_source_to_wav(source: Path, output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Converting source to separator WAV...")
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output),
        ]
    )


def _separate_hyperace(
    args: argparse.Namespace,
    staged_source: Path,
    separator_input_dir: Path,
    stems_dir: Path,
) -> tuple[Path, Path]:
    vocal = _find_stem(stems_dir, "vocals")
    instrumental = _find_stem(stems_dir, "instrumental")
    if not args.reprocess and (vocal is not None or instrumental is not None):
        reused = vocal or instrumental
        print("Reusing available HyperACE stem:", reused)
    else:
        wrapper = Path(__file__).resolve().parent / "bs_roformer.py"
        command = [
            str(args.bs_python),
            str(wrapper),
            "separate",
            "--input-dir",
            str(separator_input_dir),
            "--output-dir",
            str(stems_dir),
            "--model",
            args.bs_model,
            "--device",
            args.bs_device,
            "--convert-to-wav",
            "--output-layout",
            "organized",
            "--limit",
            "1",
        ]
        if args.bs_models_dir is not None:
            command.extend(["--models-dir", str(args.bs_models_dir)])
        if args.reprocess:
            command.append("--overwrite-output")

        env = dict(os.environ)
        env["PATH"] = (
            str(args.bs_python.parent) + os.pathsep + env.get("PATH", "")
        )
        print(f"Running BS-RoFormer HyperACE V2 ({args.bs_model})...")
        _run(command, env=env)
        vocal = _find_stem(stems_dir, "vocals")
        instrumental = _find_stem(stems_dir, "instrumental")

    if vocal is None and instrumental is None:
        files = "\n".join(str(path) for path in sorted(stems_dir.rglob("*.wav")))
        raise RuntimeError(
            "HyperACE did not produce a recognizable vocal or instrumental stem. "
            f"Files found:\n{files}"
        )

    # Binary target models normally emit both the target and its residual.
    # Keep a deterministic fallback for package versions that only save one.
    fallback_dir = stems_dir / "resolved"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    if instrumental is None:
        instrumental = fallback_dir / "instrumental.wav"
        _write_residual(staged_source, vocal, instrumental)
    if vocal is None:
        vocal = fallback_dir / "vocals.wav"
        _write_residual(staged_source, instrumental, vocal)
    return vocal.resolve(), instrumental.resolve()


def _find_stem(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    aliases = {
        "instrumental": (
            "instrumental",
            "instrument",
            "accompaniment",
            "no_vocals",
        ),
        "vocals": ("vocals", "vocal"),
    }.get(name, (name,))
    exact = sorted(
        path
        for alias in aliases
        for path in root.rglob(f"{alias}.wav")
        if path.is_file()
    )
    if exact:
        return exact[0]
    fuzzy = sorted(
        path
        for path in root.rglob("*.wav")
        if path.is_file()
        and any(alias in path.stem.lower() for alias in aliases)
    )
    return fuzzy[0] if len(fuzzy) == 1 else None


def _write_residual(source: Path, known: Path | None, output: Path) -> None:
    if known is None:
        raise RuntimeError(f"Cannot reconstruct residual stem for {output}")
    import numpy as np
    import soundfile as sf

    source_audio, source_sr = sf.read(str(source), dtype="float32", always_2d=True)
    known_audio, known_sr = sf.read(str(known), dtype="float32", always_2d=True)
    if source_sr != known_sr:
        raise RuntimeError(
            f"Cannot subtract stems with different sample rates: {source_sr} != {known_sr}"
        )
    if known_audio.shape[1] == 1 and source_audio.shape[1] == 2:
        known_audio = np.repeat(known_audio, 2, axis=1)
    if source_audio.shape[1] != known_audio.shape[1]:
        raise RuntimeError("Cannot subtract stems with incompatible channel counts.")
    frames = min(len(source_audio), len(known_audio))
    residual = np.clip(source_audio[:frames] - known_audio[:frames], -1.0, 1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), residual, source_sr, subtype="PCM_16")


def _prepare_game_melody(
    args: argparse.Namespace,
    vocals_path: Path,
    output_dir: Path,
    target_duration: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_vocal = output_dir / "source_vocal.wav"
    midi_path = staged_vocal.with_suffix(".mid")
    melody_path = output_dir / "melody_piano.wav"
    if melody_path.exists() and not args.reprocess:
        print("Reusing GAME piano melody:", melody_path)
        return melody_path

    shutil.copy2(vocals_path, staged_vocal)
    if args.reprocess:
        for suffix in (".mid", ".txt", ".csv"):
            path = staged_vocal.with_suffix(suffix)
            if path.exists():
                path.unlink()

    command = [
        str(args.game_python),
        str(args.game_dir / "infer.py"),
        "extract",
        str(staged_vocal),
        "-m",
        str(args.game_model_path.resolve()),
        "--output-formats",
        "mid,txt,csv",
    ]
    if args.game_extra_args:
        command.extend(shlex.split(args.game_extra_args))
    print("Running OpenVPI GAME melody extraction on vocals...")
    _run(command, cwd=args.game_dir)
    if not midi_path.exists():
        raise RuntimeError(f"GAME did not write expected MIDI: {midi_path}")

    soundfont = args.soundfont or _discover_soundfont()
    raw_path = output_dir / "melody_piano_raw.wav"
    print("Rendering GAME MIDI with piano SoundFont...")
    _run(
        [
            "fluidsynth",
            "-ni",
            str(soundfont),
            str(midi_path),
            "-F",
            str(raw_path),
            "-r",
            str(args.render_raw_sample_rate),
        ]
    )
    filters = []
    if args.render_loudnorm:
        filters.append(f"loudnorm={args.render_loudnorm}")
    # Training pads/trims each rendered melody chunk to the target duration.
    # Do the equivalent here so long generation never wraps a short MIDI file.
    filters.extend(["apad", f"atrim=duration={target_duration:.6f}"])
    _run(
        [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-ac",
            "1",
            "-ar",
            str(args.sample_rate),
            "-af",
            ",".join(filters),
            str(melody_path),
        ]
    )
    if raw_path.exists() and not args.keep_render_raw:
        raw_path.unlink()
    return melody_path


def _discover_soundfont() -> Path:
    for candidate in DEFAULT_SOUNDFONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    for root in (Path("/usr/share/sounds"), Path("/usr/share/soundfonts")):
        if root.exists():
            matches = sorted(root.rglob("*.sf2"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "No .sf2 SoundFont found. Install: apt-get install -y fluidsynth "
        "fluid-soundfont-gm"
    )


def _prepare_beatthis_condition(
    args: argparse.Namespace,
    instrumental_path: Path,
    condition_path: Path,
) -> dict[str, Any]:
    if condition_path.exists() and not args.reprocess:
        summary = _load_condition(condition_path)
        _validate_cached_condition(
            summary=summary,
            audio_path=instrumental_path.resolve(),
            analysis_audio="target",
            checkpoint=args.beatthis_checkpoint,
        )
        _validate_summary(summary, 4, 1, instrumental_path)
        print("Reusing Beat This! Schema-2 condition:", condition_path)
        return summary

    tracker = _create_tracker(args.beatthis_checkpoint, args.beatthis_device)
    beats, downbeats = tracker(str(instrumental_path))
    duration = _audio_duration(instrumental_path, args.ffprobe)
    summary = _summarize_events(
        beats=beats,
        downbeats=downbeats,
        duration=duration,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
    )
    _validate_summary(summary, 4, 1, instrumental_path)
    _save_condition(
        path=condition_path,
        summary=summary,
        audio_path=instrumental_path.resolve(),
        analysis_audio="target",
        checkpoint=args.beatthis_checkpoint,
    )
    del tracker
    _release_cuda()
    return summary


def _initial_generation_config(
    args: argparse.Namespace,
    source_duration: float,
    beat_summary: dict[str, Any],
) -> dict[str, Any]:
    bpm = float(args.bpm) if args.bpm is not None else float(beat_summary["bpm"])
    return {
        "base_prompt": args.prompt,
        "bpm": bpm,
        # Duration is deliberately locked to the original song. It is not an
        # editable generation setting because the final background and remix
        # must remain sample-aligned with the source vocal.
        "duration": source_duration,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "cfg_coef": args.cfg_coef,
        "extend_stride": args.extend_stride,
        "seed": args.seed,
        "background_gain_db": args.background_gain_db,
        "vocal_gain_db": args.vocal_gain_db,
    }


def _print_processing_summary(
    args: argparse.Namespace,
    source_audio: Path,
    vocals_path: Path,
    reference_instrumental_path: Path,
    melody_path: Path,
    condition_path: Path,
    beat_summary: dict[str, Any],
    generated_path: Path,
    mixed_path: Path,
    config: dict[str, Any],
) -> None:
    first_downbeat = beat_summary.get("first_downbeat")
    first_downbeat_text = (
        f"{float(first_downbeat):.3f}s"
        if first_downbeat is not None and math.isfinite(float(first_downbeat))
        else "none"
    )
    reliable = _downbeat_reliable(beat_summary)
    print("")
    print("===== V5 PREPROCESSING (MATCHES TRAINING) =====")
    print(f"source audio:          {source_audio}")
    print(f"separator:             BS-RoFormer HyperACE V2")
    print(f"separator model:       {args.bs_model}")
    print(f"vocals:                {vocals_path}")
    print(f"reference instrumental:{reference_instrumental_path}")
    print("melody process:        vocals -> GAME -> MIDI -> piano mono 32 kHz")
    print(f"melody piano:          {melody_path}")
    print("rhythm process:       instrumental -> Beat This -> Schema 2 -> 9 features")
    print(f"Beat This checkpoint:  {args.beatthis_checkpoint}")
    print(f"condition file:        {condition_path}")
    print(f"detected BPM raw:      {float(beat_summary['bpm_raw']):.3f}")
    print(f"BPM label:             {float(beat_summary['bpm']):.3f}")
    print(f"beats/downbeats:       {len(beat_summary['beats'])}/{len(beat_summary['downbeats'])}")
    print(f"first downbeat:        {first_downbeat_text}")
    print(f"estimated meter:       {int(beat_summary['estimated_beats_per_bar'])}/4")
    print(f"downbeat reliable:     {reliable}")
    print(f"tempo relative MAD:    {float(beat_summary['tempo_relative_mad']):.6f}")
    print("")
    print("===== DETECTED GENERATION CONFIG =====")
    _print_config_values(config)
    print(f"checkpoint:            {args.checkpoint}")
    print(f"background output:     {generated_path}")
    print(f"mixed output:          {mixed_path}")


def _review_config(
    config: dict[str, Any],
    assume_yes: bool,
) -> dict[str, Any] | None:
    if assume_yes:
        return dict(config)
    print("")
    answer = input(
        "Use this config? [Y]es / [E]dit values / [N]o: "
    ).strip().lower()
    if answer in {"n", "no"}:
        return None
    if answer not in {"e", "edit"}:
        return dict(config)

    edited = dict(config)
    print("Leave a field blank to keep its current value.")
    edited["base_prompt"] = _input_text("Prompt", str(edited["base_prompt"]))
    edited["bpm"] = _input_float("BPM", float(edited["bpm"]), minimum=1.0)
    edited["top_k"] = _input_int("Top-k", int(edited["top_k"]), minimum=0)
    edited["top_p"] = _input_float(
        "Top-p", float(edited["top_p"]), minimum=0.0, maximum=1.0
    )
    edited["temperature"] = _input_float(
        "Temperature", float(edited["temperature"]), minimum=0.000001
    )
    edited["cfg_coef"] = _input_float(
        "CFG coefficient", float(edited["cfg_coef"]), minimum=0.0
    )
    edited["extend_stride"] = _input_float(
        "Long-generation stride seconds",
        float(edited["extend_stride"]),
        minimum=0.1,
        maximum=29.999,
    )
    edited["seed"] = _input_int("Seed", int(edited["seed"]), minimum=0)
    edited["background_gain_db"] = _input_float(
        "Background gain dB", float(edited["background_gain_db"])
    )
    edited["vocal_gain_db"] = _input_float(
        "Vocal gain dB", float(edited["vocal_gain_db"])
    )
    return edited


def _validate_generation_config(
    config: dict[str, Any],
    bpm_min: float,
    bpm_max: float,
) -> None:
    if not str(config["base_prompt"]).strip():
        raise ValueError("Prompt cannot be empty.")
    if float(config["bpm"]) <= 0:
        raise ValueError("BPM must be positive.")
    if not bpm_min <= float(config["bpm"]) <= bpm_max:
        print(
            f"[warn] BPM {float(config['bpm']):.3f} is outside the V5 training "
            f"label range {bpm_min:.1f}-{bpm_max:.1f}."
        )
    if float(config["duration"]) <= 0:
        raise ValueError("Duration must be positive.")
    if int(config["top_k"]) < 0:
        raise ValueError("Top-k must be non-negative.")
    if not 0.0 <= float(config["top_p"]) <= 1.0:
        raise ValueError("Top-p must be between 0 and 1.")
    if float(config["temperature"]) <= 0:
        raise ValueError("Temperature must be positive.")
    if float(config["cfg_coef"]) < 0:
        raise ValueError("CFG coefficient must be non-negative.")
    if not 0.0 < float(config["extend_stride"]) < 30.0:
        raise ValueError("Extend stride must be greater than 0 and below 30 seconds.")
    if int(config["seed"]) < 0:
        raise ValueError("Seed must be non-negative.")


def _input_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _input_float(
    label: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = input(f"{label} [{default:g}]: ").strip()
    result = default if not value else float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _input_int(label: str, default: int, minimum: int | None = None) -> int:
    value = input(f"{label} [{default}]: ").strip()
    result = default if not value else int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _print_final_config(config: dict[str, Any]) -> None:
    print("")
    print("===== FINAL GENERATION CONFIG =====")
    _print_config_values(config)
    print(f"final prompt:          {config['prompt']}")


def _print_config_values(config: dict[str, Any]) -> None:
    print(f"base prompt:           {config['base_prompt']}")
    print(f"BPM:                   {float(config['bpm']):.3f}")
    print(f"duration:              {float(config['duration']):.3f}s")
    print(f"top-k/top-p:           {int(config['top_k'])}/{float(config['top_p']):g}")
    print(f"temperature:           {float(config['temperature']):g}")
    print(f"CFG coefficient:       {float(config['cfg_coef']):g}")
    print(f"extend stride:         {float(config['extend_stride']):g}s")
    print(f"seed:                  {int(config['seed'])}")
    print(f"background/vocal gain: {float(config['background_gain_db']):g}/{float(config['vocal_gain_db']):g} dB")


def _append_bpm_to_prompt(prompt: str, bpm: float) -> str:
    cleaned = re.sub(
        r"(?:,\s*)?\b\d+(?:\.\d+)?\s*bpm\b",
        "",
        prompt,
        flags=re.IGNORECASE,
    ).strip(" ,")
    return f"{cleaned}, {bpm:.1f} bpm" if cleaned else f"{bpm:.1f} bpm"


def _condition_for_features(
    condition_path: Path,
    detected_bpm: float,
    selected_bpm: float,
) -> dict[str, Any]:
    import numpy as np

    with np.load(condition_path, allow_pickle=False) as data:
        condition = _load_condition_arrays(
            data=data,
            condition_path=condition_path,
            metadata_bpm=detected_bpm,
        )
    if not math.isclose(detected_bpm, selected_bpm, abs_tol=0.05):
        print(
            "[warn] Manual BPM override changes the prompt and normalized BPM "
            "feature; Beat This event timestamps remain unchanged."
        )
    condition["bpm"] = selected_bpm
    return condition


def _generate_v5(
    args: argparse.Namespace,
    melody_path: Path,
    condition: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen
    import soundfile as sf

    device = torch.device(args.device)
    checkpoint_state = _read_v5_checkpoint(args.checkpoint)
    checkpoint_model = str(checkpoint_state.get("model_name") or args.model)
    if args.model != checkpoint_model:
        raise RuntimeError(
            f"Requested model {args.model!r} does not match checkpoint model "
            f"{checkpoint_model!r}."
        )

    print(f"Loading base model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    initialization = str(checkpoint_state.get("initialization", "v1"))
    if initialization == "v1":
        if args.base_v1_checkpoint is None:
            raise RuntimeError(
                "This V5 checkpoint was initialized from V1. Pass "
                "--base-v1-checkpoint so frozen V1 weights are restored."
            )
        _load_v1_base_checkpoint(
            lm=model.lm,
            checkpoint=args.base_v1_checkpoint,
            expected_model=args.model,
        )
    elif initialization == "pretrained":
        if args.base_v1_checkpoint is not None:
            raise RuntimeError(
                "Checkpoint initialization is pretrained; do not pass "
                "--base-v1-checkpoint."
            )
    else:
        raise RuntimeError(f"Unsupported V5 initialization: {initialization!r}")

    hidden_dim = _checkpoint_hidden_dim(checkpoint_state)
    vocab_size = _infer_vocab_size(model)
    conditioner = BeatThisLogitConditioner(
        input_dim=len(RHYTHM_FEATURE_NAMES),
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dropout=0.0,
    ).to(device)
    missing, unexpected = model.lm.load_state_dict(
        checkpoint_state["trainable"], strict=False
    )
    conditioner.load_state_dict(
        checkpoint_state["rhythm_conditioner"], strict=True
    )
    model.lm.eval()
    model.compression_model.eval()
    conditioner.eval()
    print(f"Loaded V5 checkpoint: {args.checkpoint}")
    print(f"Initialization:       {initialization}")
    print(f"Global step:          {int(checkpoint_state.get('global_step', 0))}")
    print(f"V5 LM keys:           {len(checkpoint_state['trainable'])}")
    print(f"Missing/unexpected:   {len(missing)}/{len(unexpected)}")
    print(f"Rhythm hidden dim:    {hidden_dim}")

    melody, melody_sr = _load_wav(melody_path, sf)
    melody = convert_audio(
        melody, melody_sr, model.sample_rate, model.audio_channels
    )
    duration = float(config["duration"])
    feature_rate = float(
        checkpoint_state.get("args", {}).get("rhythm_feature_rate", 50.0)
    )
    features = _events_to_features(
        condition=condition,
        duration=duration,
        feature_rate=feature_rate,
    )
    rhythm = torch.from_numpy(features).unsqueeze(0).to(device)
    total_frames = int(duration * model.frame_rate)
    rhythm = _fit_feature_length(rhythm, total_frames)
    rhythm_bias = _precompute_rhythm_bias(
        conditioner=conditioner,
        rhythm=rhythm,
        vocab_size=vocab_size,
        device=device,
        amp=args.amp,
    )

    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
    model.set_generation_params(
        duration=duration,
        top_k=int(config["top_k"]),
        top_p=float(config["top_p"]),
        temperature=float(config["temperature"]),
        cfg_coef=float(config["cfg_coef"]),
        two_step_cfg=False,
        extend_stride=float(config["extend_stride"]),
    )

    print("Generating V5 background instrumental...")
    print(f"Prompt:       {config['prompt']}")
    print(f"Duration:     {duration:.3f}s")
    print(f"Rhythm frames:{total_frames}")
    print(f"Output:       {output_path}")
    stride_frames = int(model.frame_rate * float(config["extend_stride"]))
    with torch.no_grad(), _patched_v5_autoregressive_conditioning(
        lm=model.lm,
        rhythm_bias=rhythm_bias,
        extend_stride_frames=stride_frames,
    ):
        wav = model.generate_with_chroma(
            descriptions=[str(config["prompt"])],
            melody_wavs=melody,
            melody_sample_rate=model.sample_rate,
            progress=True,
        )

    audio = wav[0].detach().cpu().float().numpy().T
    # AudioCraft generates discrete frames at 50 Hz, so decoding may differ
    # from the requested duration by a few milliseconds. Pad/trim the decoded
    # waveform to the exact requested sample count. Pipeline duration is
    # locked to the original input song duration.
    target_samples = int(round(duration * model.sample_rate))
    audio = _pad_or_trim_numpy_audio(audio, target_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, model.sample_rate, subtype=args.subtype)
    print(f"Wrote generated background: {output_path}")


def _pad_or_trim_numpy_audio(audio: Any, target_samples: int) -> Any:
    import numpy as np

    if target_samples <= 0:
        raise ValueError("Target audio sample count must be positive.")
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[0] >= target_samples:
        return audio[:target_samples]
    padding = np.zeros(
        (target_samples - audio.shape[0], audio.shape[1]),
        dtype=audio.dtype,
    )
    return np.concatenate([audio, padding], axis=0)


def _read_v5_checkpoint(path: Path) -> dict[str, Any]:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid checkpoint object: {path}")
    if state.get("checkpoint_kind") != V5_CHECKPOINT_KIND:
        raise RuntimeError(f"Not a V5 Beat This checkpoint: {path}")
    if int(state.get("condition_schema", -1)) != V5_CONDITION_SCHEMA:
        raise RuntimeError("V5 checkpoint condition schema mismatch.")
    if state.get("detector") != V5_DETECTOR:
        raise RuntimeError("V5 checkpoint detector mismatch.")
    if state.get("tempo_estimator") != V5_TEMPO_ESTIMATOR:
        raise RuntimeError("V5 checkpoint tempo estimator mismatch.")
    names = tuple(state.get("rhythm_feature_names", ()))
    if names != tuple(RHYTHM_FEATURE_NAMES):
        raise RuntimeError(
            f"V5 rhythm feature mismatch: {names} != {RHYTHM_FEATURE_NAMES}"
        )
    if not isinstance(state.get("trainable"), dict):
        raise RuntimeError("V5 checkpoint has no LM trainable state.")
    if not isinstance(state.get("rhythm_conditioner"), dict):
        raise RuntimeError("V5 checkpoint has no rhythm conditioner state.")
    return state


def _checkpoint_hidden_dim(state: dict[str, Any]) -> int:
    args = state.get("args")
    if isinstance(args, dict) and args.get("rhythm_hidden_dim"):
        return int(args["rhythm_hidden_dim"])
    weight = state["rhythm_conditioner"].get("temporal.0.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        raise RuntimeError("Cannot infer rhythm hidden dimension from checkpoint.")
    return int(weight.shape[0])


def _precompute_rhythm_bias(
    conditioner: BeatThisLogitConditioner,
    rhythm: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    zeros = torch.zeros(
        (rhythm.shape[0], 1, rhythm.shape[1], vocab_size),
        dtype=rhythm.dtype,
        device=device,
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        bias = conditioner(rhythm, zeros)
    return bias[:, 0].detach()


@contextlib.contextmanager
def _patched_v5_autoregressive_conditioning(
    lm: Any,
    rhythm_bias: torch.Tensor,
    extend_stride_frames: int,
) -> Iterator[None]:
    """Apply dense-code-time V5 biases to pattern-time autoregressive logits."""
    original_forward = lm.forward
    original_sample = lm._sample_next_token
    original_generate = lm.generate
    segment_offset = 0
    state: dict[str, Any] = {
        "pattern": None,
        "next_step": None,
        "active_step": None,
        "segment_offset": 0,
    }

    def patched_generate(
        prompt: torch.Tensor | None = None,
        conditions: list[Any] | None = None,
        num_samples: int | None = None,
        max_gen_len: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        nonlocal segment_offset
        pattern = lm.pattern_provider.get_pattern(max_gen_len)
        prompt_frames = 0 if prompt is None else int(prompt.shape[-1])
        start_step = pattern.get_first_step_with_timesteps(prompt_frames)
        if start_step is None:
            raise RuntimeError(
                f"No pattern step found for prompt timestep {prompt_frames}."
            )
        state.update(
            pattern=pattern,
            next_step=start_step,
            active_step=None,
            segment_offset=segment_offset,
        )
        result = original_generate(
            prompt=prompt,
            conditions=[] if conditions is None else conditions,
            num_samples=num_samples,
            max_gen_len=max_gen_len,
            **kwargs,
        )
        segment_offset += extend_stride_frames
        return result

    def patched_sample(sequence: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        step = state.get("next_step")
        if step is None:
            raise RuntimeError("V5 generation pattern was not initialized.")
        state["active_step"] = int(step)
        try:
            return original_sample(sequence, *args, **kwargs)
        finally:
            state["next_step"] = int(step) + 1
            state["active_step"] = None

    def patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        logits = original_forward(*args, **kwargs)
        active_step = state.get("active_step")
        pattern = state.get("pattern")
        if active_step is None or pattern is None:
            return logits
        if active_step >= len(pattern.layout):
            raise RuntimeError(
                f"Pattern step {active_step} exceeds layout {len(pattern.layout)}."
            )
        logits = logits.clone()
        for coord in pattern.layout[active_step]:
            global_t = int(state["segment_offset"]) + int(coord.t)
            if global_t < 0 or global_t >= rhythm_bias.shape[1]:
                continue
            bias = rhythm_bias[:, global_t, :].to(
                device=logits.device, dtype=logits.dtype
            )
            if bias.shape[0] == 1 and logits.shape[0] != 1:
                bias = bias.expand(logits.shape[0], -1)
            elif bias.shape[0] != logits.shape[0]:
                repeats = math.ceil(logits.shape[0] / bias.shape[0])
                bias = bias.repeat(repeats, 1)[: logits.shape[0]]
            logits[:, int(coord.q), -1, :] += bias
        return logits

    lm.forward = patched_forward
    lm._sample_next_token = patched_sample
    lm.generate = patched_generate
    try:
        yield
    finally:
        lm.forward = original_forward
        lm._sample_next_token = original_sample
        lm.generate = original_generate


def _mix_vocal_and_background(
    ffmpeg: str,
    background_path: Path,
    vocal_path: Path,
    output_path: Path,
    background_gain_db: float,
    vocal_gain_db: float,
    sample_rate: int,
) -> None:
    print("Mixing separated vocal with generated background...")
    filter_graph = (
        f"[0:a]volume={background_gain_db}dB[bg];"
        f"[1:a]volume={vocal_gain_db}dB[voc];"
        "[bg][voc]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.95[out]"
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(background_path),
            "-i",
            str(vocal_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            str(output_path),
        ]
    )


def _downbeat_reliable(summary: dict[str, Any]) -> bool:
    ratio = len(summary["downbeats"]) / max(1, len(summary["beats"]))
    return (
        int(summary["estimated_beats_per_bar"]) == 4
        and 0.18 <= ratio <= 0.35
    )


def _load_wav(path: Path, sf_module: Any) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf_module.read(
        str(path), dtype="float32", always_2d=True
    )
    return torch.from_numpy(audio).transpose(0, 1).contiguous(), int(sample_rate)


def _audio_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    result = subprocess.run(
        [
            ffprobe,
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
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Invalid audio duration for {path}: {duration}")
    return duration


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe[:100] or "audio"


def _require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"Missing executable: {name}")
    return resolved


def _run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("==>", " ".join(shlex.quote(str(part)) for part in command))
    subprocess.run(
        [str(part) for part in command],
        cwd=None if cwd is None else str(cwd),
        env=env,
        check=True,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Separate an original song, reproduce V5 training conditions, "
            "generate an instrumental, and remix it with the vocal."
        )
    )
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-v1-checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--bs-python",
        type=Path,
        default=Path("/workspace/.venv-bs-roformer/bin/python"),
    )
    parser.add_argument("--bs-model", default=DEFAULT_SEPARATOR_MODEL)
    parser.add_argument("--bs-device", default="cuda")
    parser.add_argument("--bs-models-dir", type=Path, default=None)

    parser.add_argument("--game-dir", type=Path, default=Path("/workspace/GAME"))
    parser.add_argument(
        "--game-model-path",
        type=Path,
        default=Path("/workspace/models/game/game.pt"),
    )
    parser.add_argument(
        "--game-python",
        type=Path,
        default=Path("/workspace/.venv-game/bin/python"),
    )
    parser.add_argument("--game-extra-args", default="")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--render-raw-sample-rate", type=int, default=44100)
    parser.add_argument("--render-loudnorm", default="I=-20:TP=-2:LRA=11")
    parser.add_argument("--keep-render-raw", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=32000)

    parser.add_argument(
        "--beatthis-checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT
    )
    parser.add_argument("--beatthis-device", default="cuda")
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=150.0)
    parser.add_argument("--bpm", type=float, default=None)

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--extend-stride", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--background-gain-db", type=float, default=-3.0)
    parser.add_argument("--vocal-gain-db", type=float, default=0.0)
    parser.add_argument("--mix-sample-rate", type=int, default=32000)
    parser.add_argument("--subtype", default="PCM_16")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help=(
            "Re-run HyperACE, GAME, and Beat This instead of reusing matching "
            "cached preprocessing."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept detected/default generation config without prompting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preprocessing and config review, but skip MusicGen/mixing.",
    )
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
