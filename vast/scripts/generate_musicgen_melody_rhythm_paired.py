#!/usr/bin/env python3
"""Generate with an experimental V4 melody + rhythm-conditioned checkpoint."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
from pathlib import Path

import torch

from train_musicgen_melody_rhythm_paired import RhythmLogitBias, _infer_vocab_size, _load_rhythm_checkpoint


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy drums, rolling bass, bright synth chords, build ups and drops, no lead vocal"
)


def generate(args: argparse.Namespace) -> None:
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models import MusicGen
    import soundfile as sf

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    melody_path, prompt = _resolve_inputs(args)
    print(f"Loading model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    vocab_size = _infer_vocab_size(model)
    rhythm_bias = RhythmLogitBias(args.rhythm_input_dim, args.rhythm_hidden_dim, vocab_size).to(device)
    _load_rhythm_checkpoint(model.lm, rhythm_bias, args.checkpoint, device)
    rhythm_bias.eval()

    melody, sample_rate = _load_wav(melody_path, sf)
    melody = convert_audio(melody, sample_rate, model.sample_rate, model.audio_channels)
    duration = melody.shape[-1] / model.sample_rate if args.duration == "auto" else float(args.duration)
    bpm = _resolve_bpm(args, melody_path)
    prompt = _append_bpm_to_text(prompt, bpm)
    rhythm = _build_rhythm_features(
        duration=duration,
        bpm=bpm,
        first_downbeat=args.first_downbeat,
        feature_rate=args.rhythm_feature_rate,
        input_dim=args.rhythm_input_dim,
        beats_per_bar=args.beats_per_bar,
    ).unsqueeze(0).to(device)

    model.set_generation_params(
        duration=duration,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )

    print(f"Melody:      {melody_path}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Prompt:      {prompt}")
    print(f"BPM:         {bpm:.2f}")
    print(f"Downbeat:    {args.first_downbeat:.3f}s")
    print(f"Duration:    {duration:.2f}s")
    print(f"Output:      {args.output}")

    with torch.no_grad(), _patched_compute_predictions(model.lm, rhythm_bias, rhythm):
        wav = model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=melody,
            melody_sample_rate=model.sample_rate,
            progress=True,
        )

    audio = wav[0].detach().cpu().float().numpy().T
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, model.sample_rate, subtype=args.subtype)
    _write_json(args.output.with_suffix(".rhythm.json"), {
        "bpm": bpm,
        "first_downbeat": args.first_downbeat,
        "duration": duration,
        "feature_rate": args.rhythm_feature_rate,
        "beats_per_bar": args.beats_per_bar,
        "prompt": prompt,
    })
    print(f"Wrote: {args.output}")


@contextlib.contextmanager
def _patched_compute_predictions(lm, rhythm_bias: RhythmLogitBias, rhythm: torch.Tensor):
    original = lm.compute_predictions

    def patched(codes, attributes, *args, **kwargs):
        output = original(codes, attributes, *args, **kwargs)
        logits = rhythm_bias(rhythm, output.logits)
        return _replace_logits(output, logits)

    lm.compute_predictions = patched
    try:
        yield
    finally:
        lm.compute_predictions = original


def _replace_logits(output, logits: torch.Tensor):
    try:
        output.logits = logits
        return output
    except Exception:
        pass
    if hasattr(output, "_replace"):
        return output._replace(logits=logits)
    if dataclasses.is_dataclass(output):
        return dataclasses.replace(output, logits=logits)
    raise RuntimeError("Could not replace logits on LM output object.")


def _build_rhythm_features(
    duration: float,
    bpm: float,
    first_downbeat: float,
    feature_rate: float,
    input_dim: int,
    beats_per_bar: int,
) -> torch.Tensor:
    import numpy as np

    frame_count = max(1, int(math.ceil(duration * feature_rate)))
    times = np.arange(frame_count, dtype=np.float32) / feature_rate
    features = np.zeros((frame_count, input_dim), dtype=np.float32)
    beat_period = 60.0 / bpm
    bar_period = beat_period * beats_per_bar
    beats = np.arange(first_downbeat, duration + 1e-6, beat_period)
    downbeats = np.arange(first_downbeat, duration + 1e-6, bar_period)
    _write_pulses(features[:, 0], beats, feature_rate)
    if input_dim > 1:
        _write_pulses(features[:, 1], downbeats, feature_rate)
    if input_dim > 3:
        beat_phase = ((times - first_downbeat) / beat_period) % 1.0
        features[:, 2] = np.sin(2.0 * np.pi * beat_phase)
        features[:, 3] = np.cos(2.0 * np.pi * beat_phase)
    if input_dim > 5:
        bar_phase = ((times - first_downbeat) / bar_period) % 1.0
        features[:, 4] = np.sin(2.0 * np.pi * bar_phase)
        features[:, 5] = np.cos(2.0 * np.pi * bar_phase)
    return torch.from_numpy(features)


def _write_pulses(column, events, feature_rate: float) -> None:
    radius = max(1, int(round(0.03 * feature_rate)))
    for event in events:
        center = int(round(float(event) * feature_rate))
        start = max(0, center - radius)
        end = min(len(column), center + radius + 1)
        column[start:end] = 1.0


def _resolve_bpm(args: argparse.Namespace, melody_path: Path) -> float:
    if args.bpm is not None:
        return args.bpm
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing dependency: librosa/numpy, or pass --bpm manually.") from exc
    y, sr = librosa.load(str(melody_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    return _fold_bpm_to_range(bpm, args.bpm_min, args.bpm_max)


def _fold_bpm_to_range(bpm: float, bpm_min: float, bpm_max: float) -> float:
    if bpm <= 0:
        return bpm
    folded = bpm
    while folded < bpm_min:
        folded *= 2.0
    while folded > bpm_max:
        folded /= 2.0
    return folded


def _append_bpm_to_text(text: str, bpm: float) -> str:
    if " bpm" in text.lower():
        return text
    return f"{text}, {bpm:.1f} bpm" if text else f"{bpm:.1f} bpm"


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, str]:
    if args.metadata and args.melody_audio:
        raise ValueError("Use either --metadata or --melody-audio, not both.")
    if args.metadata:
        row = _read_metadata_row(args.metadata, args.row_index)
        dataset_root = args.dataset_root or args.metadata.parent
        melody = _resolve_path(row["input_audio"], dataset_root)
        prompt = args.prompt or row.get("text") or DEFAULT_PROMPT
        return melody, prompt
    if not args.melody_audio:
        raise ValueError("Either --metadata or --melody-audio is required.")
    return args.melody_audio, args.prompt or DEFAULT_PROMPT


def _read_metadata_row(path: Path, row_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"Row index {row_index} not found in {path}")


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_wav(path: Path, sf_module) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf_module.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(audio).transpose(0, 1).contiguous()
    return tensor, int(sample_rate)


def _write_json(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate with V4 rhythm-conditioned MusicGen Melody.")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--melody-audio", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="facebook/musicgen-melody-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", default="auto")
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--bpm-min", type=float, default=120.0)
    parser.add_argument("--bpm-max", type=float, default=150.0)
    parser.add_argument("--first-downbeat", type=float, default=0.0)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--rhythm-input-dim", type=int, default=6)
    parser.add_argument("--rhythm-hidden-dim", type=int, default=512)
    parser.add_argument("--rhythm-feature-rate", type=float, default=50.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtype", default="PCM_16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate(args)


if __name__ == "__main__":
    main()
