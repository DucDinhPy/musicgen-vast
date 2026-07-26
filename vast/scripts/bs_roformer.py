#!/usr/bin/env python3
"""
BS-RoFormer helper CLI for Vast.ai.

This script wraps the `bs-roformer-infer` and `bs-roformer-download` command
line tools with a friendlier workflow for this project.

What it is for
--------------
Use this script when you want to:

1. Check whether the BS-RoFormer environment is installed correctly.
2. Download/cache a BS-RoFormer model.
3. Separate a folder of audio files into stems.
4. Automatically stage MP3/FLAC/M4A/OGG files as temporary WAV files before
   running BS-RoFormer.
5. Organize separated stems into a clean folder layout.

Recommended setup
-----------------
On a fresh Vast.ai instance, run:

    cd /workspace/musicgen-vast
    bash vast/sh/setup_bs_roformer.sh
    source /workspace/.venv-bs-roformer/bin/activate

Quick checks
------------
Verify the environment:

    python vast/scripts/bs_roformer.py check

Download the default 6-stem model:

    python vast/scripts/bs_roformer.py download \
      --model roformer-model-bs-roformer-sw-by-jarredou

Separate a folder
-----------------
Basic CUDA run:

    python vast/scripts/bs_roformer.py separate \
      --input-dir /workspace/musicgen-vast/data/preprocess/pre_audio_single \
      --output-dir /workspace/musicgen-vast/outputs/stems \
      --device cuda \
      --convert-to-wav

Default organized output layout:

    /workspace/outputs/stems/
      pre_audio_set/
        vinahouse_2PILLZ_1/
          chunk_0000/
            vocals.wav
            drums.wav
            bass.wav
            guitar.wav
            piano.wav
            other.wav
            instrumental.wav
      pre_audio_single/
        vinahouse_01_anh_vui/
          vocals.wav
          drums.wav
          bass.wav
          ...

Use `--output-layout flat` to keep the raw `bs-roformer-infer` filenames.

Dry-run the file selection:

    python vast/scripts/bs_roformer.py separate \\
      --input-dir /workspace/data/raw_audio \\
      --output-dir /workspace/outputs/stems \\
      --dry-run

Process only the first 10 files:

    python vast/scripts/bs_roformer.py separate \\
      --input-dir /workspace/data/raw_audio \\
      --output-dir /workspace/outputs/stems \\
      --device cuda \\
      --limit 10

Important notes
---------------
- `bs-roformer-infer` only consumes WAV files from a folder. This wrapper can
  convert/copy supported audio files into a temporary WAV staging folder.
- Output stems are written by `bs-roformer-infer` into `--output-dir`.
- By default, this wrapper moves flat stem files into a clearer
  `source_group/song_id/chunk_xxxx/stem.wav` layout.
- The default model is BS-RoFormer-SW by jarredou, a common 6-stem separation
  checkpoint: vocals, drums, bass, guitar, piano, other.
- Use `--device cuda` on a GPU instance. Use `--device cpu` only for debugging.
"""
from __future__ import annotations

import argparse
import textwrap
import os
import re
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
            staged = _stage_wavs(ffmpeg, audio_files, wav_dir)
            _run_infer(
                infer=infer,
                input_dir=wav_dir,
                output_dir=output_dir,
                model=args.model,
                device=args.device,
                models_dir=args.models_dir,
            )
            if args.output_layout == "organized":
                _organize_outputs(
                    flat_output_dir=output_dir,
                    staged_to_source=staged,
                    input_root=input_dir,
                    overwrite=args.overwrite_output,
                )
    else:
        # bs-roformer-infer only processes top-level WAV files, so create a
        # staging folder if inputs were found recursively.
        with tempfile.TemporaryDirectory(prefix="bs_roformer_wav_") as tmp:
            wav_dir = Path(tmp)
            staged = {}
            for index, src in enumerate(audio_files, start=1):
                dst = wav_dir / _safe_wav_name(index, src)
                shutil.copy2(src, dst)
                staged[dst.stem] = src
            _run_infer(
                infer=infer,
                input_dir=wav_dir,
                output_dir=output_dir,
                model=args.model,
                device=args.device,
                models_dir=args.models_dir,
            )
            if args.output_layout == "organized":
                _organize_outputs(
                    flat_output_dir=output_dir,
                    staged_to_source=staged,
                    input_root=input_dir,
                    overwrite=args.overwrite_output,
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


def _stage_wavs(
    ffmpeg: str,
    audio_files: list[Path],
    wav_dir: Path,
) -> dict[str, Path]:
    print(f"Staging WAV files in: {wav_dir}")
    staged = {}
    for index, src in enumerate(audio_files, start=1):
        dst = wav_dir / _safe_wav_name(index, src)
        staged[dst.stem] = src
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
    return staged


def _organize_outputs(
    flat_output_dir: Path,
    staged_to_source: dict[str, Path],
    input_root: Path,
    overwrite: bool,
) -> None:
    print("Organizing BS-RoFormer outputs...")

    moved = 0
    for staged_stem, source_path in staged_to_source.items():
        stem_files = sorted(flat_output_dir.glob(f"{staged_stem}_*.wav"))
        if not stem_files:
            print(f"[warn] No output stems found for staged file: {staged_stem}")
            continue

        target_dir = _organized_target_dir(
            output_root=flat_output_dir,
            input_root=input_root,
            source_path=source_path,
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        for stem_file in stem_files:
            stem_name = stem_file.name[len(staged_stem) + 1 :]
            target_path = target_dir / stem_name

            if target_path.exists():
                if overwrite:
                    target_path.unlink()
                else:
                    print(f"[skip] Exists: {target_path}")
                    continue

            stem_file.replace(target_path)
            moved += 1

    print(f"Organized stem files: {moved}")


def _organized_target_dir(
    output_root: Path,
    input_root: Path,
    source_path: Path,
) -> Path:
    try:
        relative = source_path.relative_to(input_root)
    except ValueError:
        relative = Path(source_path.name)

    if len(relative.parts) > 1:
        source_group = relative.parts[0]
    else:
        source_group = input_root.name

    song_id, chunk_id = _split_song_and_chunk(source_path.stem)
    if chunk_id is not None:
        return output_root / source_group / song_id / f"chunk_{chunk_id:04d}"
    return output_root / source_group / song_id


def _split_song_and_chunk(stem: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<song>.+)_chunk_(?P<chunk>\d+)$", stem)
    if not match:
        return stem, None
    return match.group("song"), int(match.group("chunk"))


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
        description="BS-RoFormer helper CLI for setup/check/download/separation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""
            Examples:
              python vast/scripts/bs_roformer.py check

              python vast/scripts/bs_roformer.py download \\
                --model {DEFAULT_MODEL}

              python vast/scripts/bs_roformer.py separate \\
                --input-dir /workspace/data/raw_audio \\
                --output-dir /workspace/outputs/stems \\
                --device cuda \\
                --convert-to-wav

              python vast/scripts/bs_roformer.py separate \\
                --input-dir /workspace/data/raw_audio \\
                --output-dir /workspace/outputs/stems \\
                --dry-run

            Setup:
              bash vast/sh/setup_bs_roformer.sh
              source /workspace/.venv-bs-roformer/bin/activate
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Verify BS-RoFormer environment.",
        description="Print installed executables, torch/CUDA status, and bs_roformer import status.",
    )
    check.set_defaults(func=cmd_check)

    download = subparsers.add_parser(
        "download",
        help="Download a BS-RoFormer model.",
        description="Download/cache a model with bs-roformer-download.",
    )
    download.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model slug to download. Default: {DEFAULT_MODEL}",
    )
    download.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom model output/cache directory.",
    )
    download.set_defaults(func=cmd_download)

    separate = subparsers.add_parser(
        "separate",
        help="Separate an audio folder into stems.",
        description=(
            "Find audio files recursively, optionally convert/copy them into a "
            "temporary WAV staging folder, then call bs-roformer-infer."
        ),
    )
    separate.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing source audio files.",
    )
    separate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where separated stems will be written.",
    )
    separate.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"BS-RoFormer model slug. Default: {DEFAULT_MODEL}",
    )
    separate.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0. Default: auto.",
    )
    separate.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Optional directory containing downloaded model assets.",
    )
    separate.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated extensions to include. Default: common audio formats.",
    )
    separate.add_argument(
        "--convert-to-wav",
        action="store_true",
        help="Force ffmpeg conversion/copy into a temporary WAV folder first.",
    )
    separate.add_argument(
        "--output-layout",
        choices=["organized", "flat"],
        default="organized",
        help=(
            "organized writes stems as source_group/song_id[/chunk_0000]/stem.wav; "
            "flat keeps bs-roformer-infer's default flat filenames."
        ),
    )
    separate.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite organized output files if they already exist.",
    )
    separate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files after filtering and start-index.",
    )
    separate.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N selected files. Useful for resuming batches.",
    )
    separate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files and exit without running separation.",
    )
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
