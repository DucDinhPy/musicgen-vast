from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def generate_musicgen_melody(
    melody_path: Path | None,
    output_path: Path,
    prompt: str,
    model_name: str,
    duration: float | None,
    device: str,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
) -> None:
    try:
        from audiocraft.models import MusicGen
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu audiocraft. Cài trong env riêng: python -m pip install audiocraft"
        ) from exc

    if melody_path is not None and not melody_path.exists():
        raise FileNotFoundError(f"Không tìm thấy melody audio: {melody_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    melody = None
    sample_rate = None
    if melody_path is not None:
        melody_np, sample_rate = sf.read(str(melody_path), dtype="float32", always_2d=True)
        # soundfile returns samples x channels; AudioCraft expects channels x samples.
        melody = torch.from_numpy(melody_np.T.copy())
        if duration is None:
            duration = min(30.0, melody.shape[-1] / sample_rate)
    elif duration is None:
        duration = 15.0

    print(f"Model:       {model_name}")
    print(f"Device:      {device}")
    print(f"Prompt:      {prompt}")
    print(f"Melody:      {melody_path.resolve() if melody_path else '(text-only)'}")
    print(f"Duration:    {duration:.2f}s")
    print(f"Output:      {output_path.resolve()}")

    model = MusicGen.get_pretrained(model_name, device=device)
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
    )

    with torch.no_grad():
        if melody is None:
            wav = model.generate(
                descriptions=[prompt],
                progress=True,
            )
        else:
            wav = model.generate_with_chroma(
                descriptions=[prompt],
                melody_wavs=melody.unsqueeze(0),
                melody_sample_rate=sample_rate,
                progress=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = wav[0].detach().float().cpu().numpy()
    print(f"Generated shape: {generated.shape}")
    print(f"Generated min:   {float(generated.min()):.6f}")
    print(f"Generated max:   {float(generated.max()):.6f}")
    print(f"Generated peak:  {float(np.max(np.abs(generated))):.6f}")

    if generated.ndim == 2:
        if generated.shape[0] in {1, 2}:
            generated = generated.T
        elif generated.shape[1] in {1, 2}:
            pass
        else:
            raise RuntimeError(f"Unexpected generated audio shape: {generated.shape}")

    # FLOAT avoids PCM_16 clipping while debugging model quality.
    sf.write(str(output_path), generated, model.sample_rate, subtype="FLOAT")
    print(f"Wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate music with MusicGen Melody from melody audio + prompt."
    )
    parser.add_argument(
        "--melody",
        type=Path,
        required=False,
        help="Input melody audio, e.g. piano melody WAV/MP3.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Ignore --melody and generate from text only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output WAV path.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the desired output music.",
    )
    parser.add_argument(
        "--model",
        default="facebook/musicgen-melody",
        help="MusicGen model name. Use facebook/musicgen-melody for melody conditioning.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Generation duration in seconds. Default: min(30s, melody duration).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    generate_musicgen_melody(
        melody_path=None if args.text_only else args.melody,
        output_path=args.output,
        prompt=args.prompt,
        model_name=args.model,
        duration=args.duration,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_coef=args.cfg_coef,
    )


if __name__ == "__main__":
    main()
