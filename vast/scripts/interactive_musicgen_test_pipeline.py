#!/usr/bin/env python3
"""Interactive test pipeline for MusicGen melody/rhythm checkpoints.

Pipeline:

    vocal or melody_piano
    -> optional GAME vocal-to-MIDI-to-piano
    -> BPM/downbeat/section detection
    -> user review/override
    -> MusicGen generation
"""
from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy four-on-the-floor kick, rolling bass, bright synth chords, "
    "club build ups and drops, no lead vocal"
)

DEFAULT_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
)


def run_pipeline(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    melody_audio = _prepare_input(args, work_dir)
    detection = _detect(melody_audio, args)
    reviewed = _review_detection(detection, args)
    prompt = _build_prompt(args.prompt, reviewed["bpm"], reviewed["structure_text"])

    print("")
    print("===== FINAL GENERATION CONFIG =====")
    print(f"generator:       {args.generator}")
    print(f"melody_audio:    {melody_audio}")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"bpm:             {reviewed['bpm']:.2f}")
    print(f"first_downbeat:  {reviewed['first_downbeat']:.3f}s")
    print(f"prompt:          {prompt}")
    print(f"output:          {args.output}")
    print("")

    if not args.yes:
        confirm = input("Run generation with this config? [Y/n]: ").strip().lower()
        if confirm in {"n", "no"}:
            print("Cancelled.")
            return

    _run_generation(
        generator=args.generator,
        melody_audio=melody_audio,
        checkpoint=args.checkpoint,
        model=args.model,
        duration=args.duration,
        bpm=reviewed["bpm"],
        first_downbeat=reviewed["first_downbeat"],
        prompt=prompt,
        output=args.output,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )


def _prepare_input(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.input_type == "melody_piano":
        return args.input_audio

    game_work = work_dir / "game"
    game_work.mkdir(parents=True, exist_ok=True)
    staged_vocal = game_work / "source_vocal.wav"
    rendered_piano = game_work / "melody_piano.wav"
    if rendered_piano.exists() and not args.overwrite:
        return rendered_piano

    shutil.copy2(args.input_audio, staged_vocal)
    _run_game_extract(
        game_python=args.game_python,
        infer_script=args.game_dir / "infer.py",
        model_path=args.game_model_path,
        vocal_path=staged_vocal,
        game_extra_args=args.game_extra_args,
        overwrite=args.overwrite,
    )
    _render_midi_to_piano(
        midi_path=staged_vocal.with_suffix(".mid"),
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
    if not infer_script.exists():
        raise FileNotFoundError(f"GAME infer.py not found: {infer_script}")
    if not model_path.exists():
        raise FileNotFoundError(f"GAME model not found: {model_path}")
    if not game_python.exists():
        raise FileNotFoundError(f"GAME Python not found: {game_python}")

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


def _detect(melody_audio: Path, args: argparse.Namespace) -> dict:
    bpm_row = _estimate_or_override_bpm(
        melody_audio,
        target_bpm=args.target_bpm,
        source_bpm=args.bpm,
        first_downbeat=args.first_downbeat,
        min_confidence=args.min_confidence,
    )
    bpm = float(bpm_row["bpm"])
    if args.bpm_label_min is not None and args.bpm_label_max is not None:
        bpm = _fold_bpm_to_range(bpm, args.bpm_label_min, args.bpm_label_max)

    duration = _duration(melody_audio)
    active = _detect_active_intervals(
        melody_path=melody_audio,
        frame_seconds=args.section_frame_seconds,
        threshold_db=args.section_threshold_db,
        min_active_seconds=args.section_min_active_seconds,
        merge_gap_seconds=args.section_merge_gap_seconds,
    )
    sections = _build_sections(active, duration)
    structure_text = _structure_text(sections)

    return {
        "bpm_raw": float(bpm_row["bpm"]),
        "bpm": bpm,
        "first_downbeat": float(bpm_row["first_downbeat"]),
        "confidence": float(bpm_row["confidence"]),
        "duration": duration,
        "active": active,
        "sections": sections,
        "structure_text": structure_text,
    }


def _estimate_or_override_bpm(
    audio_path: Path,
    target_bpm: float,
    source_bpm: float | None,
    first_downbeat: float | None,
    min_confidence: float,
) -> dict:
    if source_bpm is not None:
        return {
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
        raise RuntimeError("Missing dependency: librosa/numpy. Install them or pass --bpm manually.") from exc

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    duration = float(librosa.get_duration(y=y, sr=sr))
    beat_count = int(len(beat_times))
    detected_downbeat = float(beat_times[0]) if len(beat_times) else 0.0

    tempo_ratio = min(tempo, target_bpm) / max(tempo, target_bpm) if tempo > 0 else 0.0
    beat_density = min(1.0, beat_count / max(1.0, duration / 2.0))
    confidence = float(max(0.0, min(1.0, 0.65 * tempo_ratio + 0.35 * beat_density)))

    return {
        "bpm": tempo,
        "target_bpm": target_bpm,
        "first_downbeat": float(first_downbeat if first_downbeat is not None else detected_downbeat),
        "duration": duration,
        "beat_count": beat_count,
        "confidence": confidence,
        "needs_review": confidence < min_confidence,
        "source": "librosa",
    }


def _fold_bpm_to_range(bpm: float, label_min: float, label_max: float) -> float:
    if bpm <= 0:
        return bpm
    folded = bpm
    while folded < label_min:
        folded *= 2.0
    while folded > label_max:
        folded /= 2.0
    return folded


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
    for index, (start, end) in enumerate(active):
        if start > cursor + 0.25:
            label = "intro_no_melody" if index == 0 else "break_or_drop_no_melody"
            sections.append([label, round(cursor, 3), round(start, 3)])
        sections.append(["main_melody" if index == 0 else "melody_backing", round(start, 3), round(end, 3)])
        cursor = end
    if cursor < duration - 0.25:
        sections.append(["outro_no_melody", round(cursor, 3), round(duration, 3)])
    return sections


def _structure_text(sections: list[list]) -> str:
    parts = [f"{label} {start:.1f}-{end:.1f}s" for label, start, end in sections]
    return "structure: " + ", ".join(parts)


def _to_db(value: float) -> float:
    if value <= 0:
        return -math.inf
    return 20.0 * math.log10(value)


def _review_detection(detection: dict, args: argparse.Namespace) -> dict:
    print("")
    print("===== DETECTION REVIEW =====")
    print(f"raw_bpm:         {detection['bpm_raw']:.2f}")
    print(f"prompt_bpm:      {detection['bpm']:.2f}")
    print(f"first_downbeat:  {detection['first_downbeat']:.3f}s")
    print(f"confidence:      {detection['confidence']:.2f}")
    print(f"duration:        {detection['duration']:.3f}s")
    print(f"active:          {detection['active']}")
    print(f"sections:        {detection['sections']}")
    print(f"structure_text:  {detection['structure_text']}")

    if args.yes:
        return detection

    answer = input("Accept this detection? [Y/n]: ").strip().lower()
    if answer not in {"n", "no"}:
        return detection

    bpm = _input_float("BPM", detection["bpm"])
    first_downbeat = _input_float("First downbeat seconds", detection["first_downbeat"])
    structure_text = input(
        "Structure text override (blank keeps detected): "
    ).strip() or detection["structure_text"]

    updated = dict(detection)
    updated["bpm"] = bpm
    updated["first_downbeat"] = first_downbeat
    updated["structure_text"] = structure_text
    return updated


def _input_float(label: str, default: float) -> float:
    value = input(f"{label} [{default:.3f}]: ").strip()
    return default if not value else float(value)


def _build_prompt(base_prompt: str, bpm: float, structure_text: str) -> str:
    parts = [base_prompt, f"{bpm:.1f} bpm"]
    if structure_text:
        parts.append(structure_text)
    return ", ".join(part for part in parts if part)


def _run_generation(
    generator: str,
    melody_audio: Path,
    checkpoint: Path,
    model: str,
    duration: str,
    bpm: float,
    first_downbeat: float,
    prompt: str,
    output: Path,
    top_k: int,
    top_p: float,
    temperature: float,
    cfg_coef: float,
) -> None:
    script_dir = Path(__file__).resolve().parent
    if generator == "rhythm":
        script = script_dir / "generate_musicgen_melody_rhythm_paired.py"
        cmd = [
            sys.executable,
            str(script),
            "--melody-audio",
            str(melody_audio),
            "--checkpoint",
            str(checkpoint),
            "--model",
            model,
            "--duration",
            duration,
            "--bpm",
            f"{bpm:.6f}",
            "--first-downbeat",
            f"{first_downbeat:.6f}",
            "--prompt",
            prompt,
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--temperature",
            str(temperature),
            "--cfg-coef",
            str(cfg_coef),
            "--output",
            str(output),
        ]
    elif generator == "base":
        script = script_dir / "generate_musicgen_melody_paired.py"
        cmd = [
            sys.executable,
            str(script),
            "--melody-audio",
            str(melody_audio),
            "--checkpoint",
            str(checkpoint),
            "--model",
            model,
            "--duration",
            duration,
            "--prompt",
            prompt,
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--temperature",
            str(temperature),
            "--cfg-coef",
            str(cfg_coef),
            "--output",
            str(output),
        ]
    else:
        raise ValueError(f"Unsupported generator: {generator}")

    print("Running:")
    print(" ".join(_quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def _quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return repr(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive input test pipeline for MusicGen.")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--input-type", choices=("melody_piano", "vocal"), default="melody_piano")
    parser.add_argument("--generator", choices=("base", "rhythm"), default="rhythm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--duration", default="auto")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target-bpm", type=float, default=128.0)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--first-downbeat", type=float, default=None)
    parser.add_argument("--bpm-label-min", type=float, default=120.0)
    parser.add_argument("--bpm-label-max", type=float, default=150.0)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--section-frame-seconds", type=float, default=0.25)
    parser.add_argument("--section-threshold-db", type=float, default=-45.0)
    parser.add_argument("--section-min-active-seconds", type=float, default=1.0)
    parser.add_argument("--section-merge-gap-seconds", type=float, default=1.0)
    parser.add_argument("--game-dir", type=Path, default=Path("/workspace/GAME"))
    parser.add_argument("--game-model-path", type=Path, default=Path("/workspace/models/game/game.pt"))
    parser.add_argument("--game-python", type=Path, default=Path("/workspace/.venv-game/bin/python"))
    parser.add_argument("--game-extra-args", default="")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--render-raw-sample-rate", type=int, default=44100)
    parser.add_argument("--render-loudnorm", default="I=-20:TP=-2:LRA=11")
    parser.add_argument("--keep-render-raw", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Accept detection and generate without prompting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = args.output.parent
    run_pipeline(args)


if __name__ == "__main__":
    main()
