#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_MODEL = "roformer-model-bs-roformer-sw-by-jarredou"
DEFAULT_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def cmd_check(_: argparse.Namespace) -> None:
    print("python cli: ok")
    print("bs-roformer-infer:", shutil.which("bs-roformer-infer"))
    print("bs-roformer-download:", shutil.which("bs-roformer-download"))
    print("ffmpeg:", shutil.which("ffmpeg"))

    import numpy as np
    import torch

    print("numpy:", np.__version__)
    print("torch:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    try:
        from bs_roformer import DEFAULT_MODEL as package_default_model

        print("bs_roformer import ok")
        print("package default model:", package_default_model)
    except Exception as exc:
        print("bs_roformer import failed:", repr(exc))
        raise


def cmd_download(args: argparse.Namespace) -> None:
    downloader = _require_executable("bs-roformer-download")

    command = [downloader, "--model", args.model]
    if args.output_dir:
        command.extend(["--output-dir", str(args.output_dir)])

    print("==>", " ".join(command))
    subprocess.run(command, check=True)


def cmd_separate(args: argparse.Namespace) -> None:
    infer = _require_executable("bs-roformer-infer")

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = _parse_extensions(args.extensions)
    audio_files = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if args.start_index > 0:
        audio_files = audio_files[args.start_index :]
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        raise RuntimeError(f"No audio files found in: {input_dir}")

    print(f"Input:          {input_dir.resolve()}")
    print(f"Output:         {output_dir.resolve()}")
    print(f"Files:          {len(audio_files)}")
    print(f"Model:          {args.model}")
    print(f"Device:         {args.device}")
    print(f"Convert to WAV: {args.convert_to_wav}")
    print(f"Extensions:     {', '.join(sorted(extensions))}")

    if args.dry_run:
        for path in audio_files:
            print(path)
        return

    needs_wav_stage = args.convert_to_wav or any(
        path.suffix.lower() != ".wav" for path in audio_files
    )

    if needs_wav_stage:
        ffmpeg = _require_executable("ffmpeg")
        with tempfile.TemporaryDirectory(prefix="bs_roformer_wav_") as tmp:
            wav_dir = Path(tmp)
            _stage_wavs(ffmpeg, audio_files, wav_dir)
            _run_infer(
                infer=infer,
                input_dir=wav_dir,
                output_dir=output_dir,
                model=args.model,
                device=args.device,
                models_dir=args.models_dir,
            )
    else:
        # bs-roformer-infer only processes top-level WAV files, so create a
        # staging folder if inputs were found recursively.
        with tempfile.TemporaryDirectory(prefix="bs_roformer_wav_") as tmp:
            wav_dir = Path(tmp)
            for index, src in enumerate(audio_files, start=1):
                dst = wav_dir / _safe_wav_name(index, src)
                shutil.copy2(src, dst)
            _run_infer(
                infer=infer,
                input_dir=wav_dir,
                output_dir=output_dir,
                model=args.model,
                device=args.device,
                models_dir=args.models_dir,
            )


def _run_infer(
    infer: str,
    input_dir: Path,
    output_dir: Path,
    model: str,
    device: str,
    models_dir: Path | None,
) -> None:
    command = [
        infer,
        "--input_folder",
        str(input_dir.resolve()),
        "--store_dir",
        str(output_dir.resolve()),
        "--model",
        model,
    ]

    if device:
        command.extend(["--device", device])
    if models_dir is not None:
        command.extend(["--models_dir", str(models_dir)])

    print("==>", " ".join(command))
    subprocess.run(command, check=True)


def _stage_wavs(ffmpeg: str, audio_files: list[Path], wav_dir: Path) -> None:
    print(f"Staging WAV files in: {wav_dir}")
    for index, src in enumerate(audio_files, start=1):
        dst = wav_dir / _safe_wav_name(index, src)
        if src.suffix.lower() == ".wav":
            shutil.copy2(src, dst)
            continue

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-ar",
                "44100",
                "-ac",
                "2",
                str(dst),
            ],
            check=True,
        )


def _safe_wav_name(index: int, src: Path) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in src.stem
    ).strip("._-")
    if not safe:
        safe = "audio"
    return f"{index:05d}_{safe[:80]}.wav"


def _parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.add(item)
    return extensions or set(DEFAULT_EXTENSIONS)


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Missing executable: {name}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BS-RoFormer helper CLI for setup/check/download/separation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Verify BS-RoFormer environment.")
    check.set_defaults(func=cmd_check)

    download = subparsers.add_parser("download", help="Download a BS-RoFormer model.")
    download.add_argument("--model", default=DEFAULT_MODEL)
    download.add_argument("--output-dir", type=Path, default=None)
    download.set_defaults(func=cmd_download)

    separate = subparsers.add_parser("separate", help="Separate audio folder.")
    separate.add_argument("--input-dir", type=Path, required=True)
    separate.add_argument("--output-dir", type=Path, required=True)
    separate.add_argument("--model", default=DEFAULT_MODEL)
    separate.add_argument("--device", default="auto")
    separate.add_argument("--models-dir", type=Path, default=None)
    separate.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated extensions to include.",
    )
    separate.add_argument(
        "--convert-to-wav",
        action="store_true",
        help="Force ffmpeg conversion/copy into a temporary WAV folder first.",
    )
    separate.add_argument("--limit", type=int, default=None)
    separate.add_argument("--start-index", type=int, default=0)
    separate.add_argument("--dry-run", action="store_true")
    separate.set_defaults(func=cmd_separate)

    return parser


def main() -> None:
    # Avoid inherited env surprises from other model stacks.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
