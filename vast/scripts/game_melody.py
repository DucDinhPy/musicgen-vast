#!/usr/bin/env python3
"""
Extract melody MIDI from organized vocal stems with OpenVPI GAME.

This wrapper is intentionally project-focused:

1. Scan an organized BS-RoFormer stem folder for `vocals.wav`.
2. Run OpenVPI GAME `infer.py extract` for each vocal file.
3. Store clean outputs under a parallel melody folder.

Example:

    cd /workspace/musicgen-vast
    source /workspace/.venv-game/bin/activate

    python vast/scripts/game_melody.py extract \\
      --stems-dir /workspace/musicgen-vast/data/stems/stems_set_single \\
      --output-dir /workspace/musicgen-vast/data/melody/game \\
      --game-dir /workspace/GAME \\
      --model-path /workspace/models/game/game.pt \\
      --limit 5

Output layout:

    data/melody/game/
      set_04/
        set04_track01_song_name/
          melody.mid
          melody.txt
          melody.csv
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_FORMATS = "mid,txt,csv"


def cmd_check(args: argparse.Namespace) -> None:
    game_dir = args.game_dir
    infer_script = game_dir / "infer.py"
    resolved_model_path = args.model_path.resolve() if args.model_path else None

    print("python:", sys.executable)
    print("game_dir:", game_dir)
    print("infer.py:", infer_script)
    print("infer.py exists:", infer_script.exists())
    print("model_path:", args.model_path)
    print("model exists:", args.model_path.exists() if args.model_path else False)
    print("resolved model_path:", resolved_model_path)


def cmd_extract(args: argparse.Namespace) -> None:
    stems_dir: Path = args.stems_dir
    output_dir: Path = args.output_dir
    game_dir: Path = args.game_dir
    model_path: Path = args.model_path
    infer_script = game_dir / "infer.py"
    formats = _parse_formats(args.output_formats)
    report_path = args.report or (output_dir / "game_melody_report.jsonl")

    _validate_inputs(
        stems_dir=stems_dir,
        output_dir=output_dir,
        game_dir=game_dir,
        infer_script=infer_script,
        model_path=model_path,
        dry_run=args.dry_run,
    )
    model_path = model_path.resolve()

    vocal_files = _discover_vocals(stems_dir, args.input_stem)
    vocal_files = vocal_files[args.start_index :]
    if args.limit is not None:
        vocal_files = vocal_files[: args.limit]

    if not vocal_files:
        raise RuntimeError(
            f"No {args.input_stem}.wav files found under: {stems_dir}"
        )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Stems dir:       {stems_dir.resolve()}")
    print(f"Output dir:      {output_dir.resolve()}")
    print(f"GAME dir:        {game_dir.resolve()}")
    print(f"Model path:      {model_path.resolve()}")
    print(f"Input stem:      {args.input_stem}")
    print(f"Output formats:  {','.join(formats)}")
    print(f"Start index:     {args.start_index}")
    print(f"Limit:           {args.limit if args.limit is not None else '(none)'}")
    print(f"Batch mode:      {not args.single_process}")
    print(f"Dry run:         {args.dry_run}")
    print(f"Report:          {report_path.resolve()}")
    print("")

    ok_count = 0
    skip_count = 0
    error_count = 0

    with report_path.open("w", encoding="utf-8") as report:
        pending: list[dict[str, object]] = []

        for index, vocal_path in enumerate(vocal_files, start=1):
            target_dir = _target_dir(vocal_path, stems_dir, output_dir)
            target_files = {
                fmt: target_dir / f"{args.output_basename}.{fmt}"
                for fmt in formats
            }

            row = {
                "source_vocal": str(vocal_path),
                "target_dir": str(target_dir),
                "formats": formats,
                "status": "planned" if args.dry_run else "ok",
            }

            if _all_outputs_exist(target_files) and not args.overwrite:
                skip_count += 1
                row["status"] = "skipped_exists"
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[skip] {index}/{len(vocal_files)} exists: {target_dir}")
                continue

            if args.dry_run:
                ok_count += 1
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[plan] {index}/{len(vocal_files)} {vocal_path} -> {target_dir}")
                continue

            pending.append(
                {
                    "index": index,
                    "vocal_path": vocal_path,
                    "target_dir": target_dir,
                    "target_files": target_files,
                    "row": row,
                }
            )

        if args.single_process:
            for item in pending:
                try:
                    written = _extract_one(
                        vocal_path=item["vocal_path"],  # type: ignore[arg-type]
                        target_dir=item["target_dir"],  # type: ignore[arg-type]
                        target_files=item["target_files"],  # type: ignore[arg-type]
                        infer_script=infer_script,
                        model_path=model_path,
                        output_formats=",".join(formats),
                        output_basename=args.output_basename,
                        python_bin=args.python_bin,
                        game_extra_args=args.game_extra_args,
                        overwrite=args.overwrite,
                    )
                except subprocess.CalledProcessError as exc:
                    error_count += 1
                    row = item["row"]  # type: ignore[assignment]
                    row["status"] = "error"
                    row["error"] = f"GAME failed with exit code {exc.returncode}"
                    report.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[error] {item['index']}/{len(vocal_files)} {item['vocal_path']}: {row['error']}")
                    if not args.keep_going:
                        raise
                    continue

                ok_count += 1
                row = item["row"]  # type: ignore[assignment]
                row["written"] = [str(path) for path in written]
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[ok] {item['index']}/{len(vocal_files)} {Path(item['vocal_path']).name} -> {item['target_dir']}")
        elif pending:
            try:
                _extract_batch(
                    items=pending,
                    infer_script=infer_script,
                    model_path=model_path,
                    output_formats=",".join(formats),
                    output_basename=args.output_basename,
                    python_bin=args.python_bin,
                    game_extra_args=args.game_extra_args,
                    overwrite=args.overwrite,
                )
            except subprocess.CalledProcessError as exc:
                error_count += len(pending)
                for item in pending:
                    row = item["row"]  # type: ignore[assignment]
                    row["status"] = "error"
                    row["error"] = f"GAME batch failed with exit code {exc.returncode}"
                    report.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[error] GAME batch failed with exit code {exc.returncode}")
                if not args.keep_going:
                    raise
            else:
                for item in pending:
                    ok_count += 1
                    row = item["row"]  # type: ignore[assignment]
                    row["written"] = [str(path) for path in item["written"]]  # type: ignore[index]
                    report.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[ok] {item['index']}/{len(vocal_files)} {Path(item['vocal_path']).name} -> {item['target_dir']}")

    print("")
    print("Done.")
    print(f"OK/planned: {ok_count}")
    print(f"Skipped:    {skip_count}")
    print(f"Errors:     {error_count}")


def _validate_inputs(
    stems_dir: Path,
    output_dir: Path,
    game_dir: Path,
    infer_script: Path,
    model_path: Path,
    dry_run: bool,
) -> None:
    if not stems_dir.exists():
        raise FileNotFoundError(f"Stems folder does not exist: {stems_dir}")
    if not stems_dir.is_dir():
        raise NotADirectoryError(f"Stems path must be a folder: {stems_dir}")
    if not game_dir.exists():
        raise FileNotFoundError(f"GAME folder does not exist: {game_dir}")
    if not infer_script.exists():
        raise FileNotFoundError(f"GAME infer.py not found: {infer_script}")
    if not model_path.exists():
        raise FileNotFoundError(f"GAME model does not exist: {model_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path must be a folder: {output_dir}")
    if dry_run:
        return


def _discover_vocals(stems_dir: Path, input_stem: str) -> list[Path]:
    return sorted(
        path
        for path in stems_dir.rglob(f"{input_stem}.wav")
        if path.is_file()
    )


def _target_dir(vocal_path: Path, stems_dir: Path, output_dir: Path) -> Path:
    relative_track_dir = vocal_path.parent.resolve().relative_to(stems_dir.resolve())
    return output_dir / relative_track_dir


def _all_outputs_exist(target_files: dict[str, Path]) -> bool:
    return all(path.exists() for path in target_files.values())


def _extract_one(
    vocal_path: Path,
    target_dir: Path,
    target_files: dict[str, Path],
    infer_script: Path,
    model_path: Path,
    output_formats: str,
    output_basename: str,
    python_bin: str,
    game_extra_args: str,
    overwrite: bool,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for target_file in target_files.values():
            if target_file.exists():
                target_file.unlink()

    with tempfile.TemporaryDirectory(prefix="game_melody_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_audio = temp_dir / f"{output_basename}.wav"
        shutil.copy2(vocal_path, staged_audio)

        cmd = [
            python_bin,
            str(infer_script),
            "extract",
            str(staged_audio),
            "-m",
            str(model_path),
            "--output-formats",
            output_formats,
        ]
        if game_extra_args:
            cmd.extend(shlex.split(game_extra_args))

        subprocess.run(cmd, cwd=str(infer_script.parent), check=True)

        written: list[Path] = []
        for fmt, target_file in target_files.items():
            source_file = staged_audio.with_suffix(f".{fmt}")
            if not source_file.exists():
                raise RuntimeError(
                    f"GAME did not write expected {fmt} output: {source_file}"
                )
            shutil.copy2(source_file, target_file)
            written.append(target_file)

    return written


def _extract_batch(
    items: list[dict[str, object]],
    infer_script: Path,
    model_path: Path,
    output_formats: str,
    output_basename: str,
    python_bin: str,
    game_extra_args: str,
    overwrite: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="game_melody_batch_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for batch_index, item in enumerate(items, start=1):
            staged_audio = temp_dir / f"{batch_index:05d}_{output_basename}.wav"
            shutil.copy2(item["vocal_path"], staged_audio)  # type: ignore[arg-type]
            item["staged_audio"] = staged_audio

            target_dir = item["target_dir"]  # type: ignore[assignment]
            target_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                target_files = item["target_files"]  # type: ignore[assignment]
                for target_file in target_files.values():
                    if target_file.exists():
                        target_file.unlink()

        cmd = [
            python_bin,
            str(infer_script),
            "extract",
            str(temp_dir),
            "-m",
            str(model_path),
            "--glob",
            "*.wav",
            "--output-formats",
            output_formats,
        ]
        if game_extra_args:
            cmd.extend(shlex.split(game_extra_args))

        subprocess.run(cmd, cwd=str(infer_script.parent), check=True)

        for item in items:
            staged_audio = item["staged_audio"]  # type: ignore[assignment]
            target_files = item["target_files"]  # type: ignore[assignment]
            written: list[Path] = []

            for fmt, target_file in target_files.items():
                source_file = staged_audio.with_suffix(f".{fmt}")
                if not source_file.exists():
                    raise RuntimeError(
                        f"GAME did not write expected {fmt} output: {source_file}"
                    )
                shutil.copy2(source_file, target_file)
                written.append(target_file)

            item["written"] = written


def _parse_formats(value: str) -> list[str]:
    formats = [part.strip().lower().lstrip(".") for part in value.split(",")]
    formats = [part for part in formats if part]
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenVPI GAME helpers for melody extraction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Print GAME paths and availability.")
    check.add_argument(
        "--game-dir",
        type=Path,
        default=Path("/workspace/GAME"),
        help="Path to cloned OpenVPI GAME repository.",
    )
    check.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/game/game.pt"),
        help="Path to GAME checkpoint.",
    )
    check.set_defaults(func=cmd_check)

    extract = subparsers.add_parser(
        "extract",
        help="Extract melody files from vocals.wav stems.",
    )
    extract.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Organized stem root containing vocals.wav files.",
    )
    extract.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root for melody files.",
    )
    extract.add_argument(
        "--game-dir",
        type=Path,
        default=Path("/workspace/GAME"),
        help="Path to cloned OpenVPI GAME repository.",
    )
    extract.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to GAME checkpoint .pt file.",
    )
    extract.add_argument(
        "--input-stem",
        default="vocals",
        help="Stem filename without .wav to use as GAME input. Default: vocals.",
    )
    extract.add_argument(
        "--output-basename",
        default="melody",
        help="Output basename inside each target folder. Default: melody.",
    )
    extract.add_argument(
        "--output-formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated GAME output formats. Default: mid,txt,csv.",
    )
    extract.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run GAME infer.py. Default: current Python.",
    )
    extract.add_argument(
        "--game-extra-args",
        default="",
        help="Extra quoted arguments forwarded to GAME infer.py extract.",
    )
    extract.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSONL path. Default: OUTPUT_DIR/game_melody_report.jsonl.",
    )
    extract.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many discovered vocal files. Default: 0.",
    )
    extract.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many vocal files.",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing melody outputs.",
    )
    extract.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a GAME failure.",
    )
    extract.add_argument(
        "--single-process",
        action="store_true",
        help="Run one GAME process per vocal file. Slower, but useful for debugging.",
    )
    extract.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without running GAME.",
    )
    extract.set_defaults(func=cmd_extract)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
