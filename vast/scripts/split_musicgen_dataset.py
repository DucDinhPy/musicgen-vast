#!/usr/bin/env python3
"""Split a MusicGen JSONL metadata file into train/valid/smoke manifests."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_dataset(
    input_metadata: Path,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    smoke_train_rows: int,
    smoke_valid_rows: int,
) -> None:
    if not input_metadata.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {input_metadata}")

    rows = _read_jsonl(input_metadata)
    if not rows:
        raise RuntimeError(f"No rows found in: {input_metadata}")

    rng = random.Random(seed)
    rng.shuffle(rows)

    train_count = int(len(rows) * train_ratio)
    train_rows = rows[:train_count]
    valid_rows = rows[train_count:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "metadata_train.jsonl"
    valid_path = output_dir / "metadata_valid.jsonl"
    smoke_train_path = output_dir / "metadata_smoke_train.jsonl"
    smoke_valid_path = output_dir / "metadata_smoke_valid.jsonl"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    _write_jsonl(smoke_train_path, train_rows[:smoke_train_rows])
    _write_jsonl(smoke_valid_path, valid_rows[:smoke_valid_rows])

    print("Input:       ", input_metadata.resolve())
    print("Output dir:  ", output_dir.resolve())
    print("Rows:        ", len(rows))
    print("Train ratio: ", train_ratio)
    print("Seed:        ", seed)
    print("")
    print(f"Train:       {len(train_rows)} -> {train_path}")
    print(f"Valid:       {len(valid_rows)} -> {valid_path}")
    print(f"Smoke train: {min(len(train_rows), smoke_train_rows)} -> {smoke_train_path}")
    print(f"Smoke valid: {min(len(valid_rows), smoke_valid_rows)} -> {smoke_valid_path}")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split MusicGen metadata JSONL into train/valid/smoke files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input metadata JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to write split metadata files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Train split ratio. Default: 0.95.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Shuffle seed. Default: 1337.",
    )
    parser.add_argument(
        "--smoke-train-rows",
        type=int,
        default=300,
        help="Rows for smoke train subset. Default: 300.",
    )
    parser.add_argument(
        "--smoke-valid-rows",
        type=int,
        default=30,
        help="Rows for smoke valid subset. Default: 30.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_dataset(
        input_metadata=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        smoke_train_rows=args.smoke_train_rows,
        smoke_valid_rows=args.smoke_valid_rows,
    )


if __name__ == "__main__":
    main()
