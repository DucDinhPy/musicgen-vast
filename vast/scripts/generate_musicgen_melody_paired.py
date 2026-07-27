#!/usr/bin/env python3
"""Generate a test instrumental from a paired MusicGen Melody checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_PROMPT = (
    "generate a full clean energetic vinahouse instrumental backing track, "
    "punchy drums, rolling bass, bright synth chords, build ups and drops, "
    "no lead vocal"
)


def generate(args: argparse.Namespace) -> None:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio_utils import convert_audio
    import soundfile as sf

    device = torch.device(args.device)
    melody_path, prompt = _resolve_inputs(args)

    print(f"Loading model: {args.model}")
    model = MusicGen.get_pretrained(args.model, device=str(device))
    if args.checkpoint:
        _load_partial_checkpoint(model, args.checkpoint, device)

    melody, sample_rate = _load_wav(melody_path, sf)
    melody = convert_audio(melody, sample_rate, model.sample_rate, model.audio_channels)
    duration = _resolve_duration(args.duration, melody, model.sample_rate)

    model.set_generation_params(
        duration=duration,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
    )

    print(f"Melody:     {melody_path}")
    print(f"Prompt:     {prompt}")
    print(f"Duration:   {duration:.2f}")
    print(f"Output:     {args.output}")

    with torch.no_grad():
        wav = model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=melody,
            melody_sample_rate=model.sample_rate,
            progress=True,
        )

    audio = wav[0].detach().cpu().float().numpy().T
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, model.sample_rate, subtype=args.subtype)
    print(f"Wrote: {args.output}")


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


def _resolve_duration(value: str, melody: torch.Tensor, sample_rate: int) -> float:
    if value.lower() == "auto":
        return melody.shape[-1] / sample_rate
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a test WAV from a paired MusicGen Melody checkpoint."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSONL. Uses input_audio/text from selected row.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root used to resolve relative metadata paths.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Metadata row index to generate from. Default: 0.",
    )
    parser.add_argument(
        "--melody-audio",
        type=Path,
        default=None,
        help="Direct melody_piano.wav path.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt override.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Partial checkpoint from train_musicgen_melody_paired.py.",
    )
    parser.add_argument(
        "--model",
        default="facebook/musicgen-melody-large",
        help="Base MusicGen model. Default: facebook/musicgen-melody-large.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device. Default: cuda.",
    )
    parser.add_argument(
        "--duration",
        default="30",
        help="Generation duration in seconds, or 'auto' to match melody length. Default: 30.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
        help="Sampling top-k. Default: 250.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Sampling top-p. Default: 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default: 1.",
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=3.0,
        help="Classifier-free guidance coefficient. Default: 3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output WAV path.",
    )
    parser.add_argument(
        "--subtype",
        default="PCM_16",
        help="soundfile WAV subtype. Default: PCM_16.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    generate(args)


if __name__ == "__main__":
    main()
