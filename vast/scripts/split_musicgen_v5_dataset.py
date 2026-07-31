#!/usr/bin/env python3
"""Create leakage-safe V5 MusicGen train/valid/smoke manifests.

All chunks with the same ``track_id`` are kept in one split. This is required
for the V1/V5 dataset because neighbouring 30-second chunks from a song share
melody, production, tempo and timbre; a row-level shuffle would leak the same
song into validation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def split_dataset(
    input_metadata: Path,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    group_field: str,
    valid_track_count: int | None,
    smoke_train_rows: int,
    smoke_valid_rows: int,
    overwrite: bool,
) -> None:
    if not input_metadata.is_file():
        raise FileNotFoundError(
            f"Metadata file does not exist: {input_metadata}"
        )
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if smoke_train_rows < 0 or smoke_valid_rows < 0:
        raise ValueError("Smoke row counts must be >= 0.")

    rows = _read_jsonl(input_metadata)
    if not rows:
        raise RuntimeError(f"No rows found in: {input_metadata}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        value = row.get(group_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Row {row_index} has no non-empty {group_field!r}."
            )
        groups[value.strip()].append(row)

    group_count = len(groups)
    if group_count < 2:
        raise RuntimeError("At least two track groups are required to split.")

    if valid_track_count is None:
        valid_track_count = max(1, round(group_count * (1.0 - train_ratio)))
    if not 1 <= valid_track_count < group_count:
        raise ValueError(
            "--valid-track-count must be at least 1 and smaller than the "
            f"number of groups ({group_count})."
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    target_valid_rows = round(len(rows) * (1.0 - train_ratio))
    valid_ids = _closest_group_subset(
        group_items=group_items,
        group_count=valid_track_count,
        target_rows=target_valid_rows,
    )
    train_ids = set(groups).difference(valid_ids)

    train_rows = [row for group_id in train_ids for row in groups[group_id]]
    valid_rows = [row for group_id in valid_ids for row in groups[group_id]]
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    _assert_no_group_leakage(
        train_rows=train_rows,
        valid_rows=valid_rows,
        group_field=group_field,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / "metadata_train.jsonl",
        "valid": output_dir / "metadata_valid.jsonl",
        "smoke_train": output_dir / "metadata_smoke_train.jsonl",
        "smoke_valid": output_dir / "metadata_smoke_valid.jsonl",
        "report": output_dir / "split_report.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Split outputs already exist: {names}. Use --overwrite to replace."
        )

    smoke_train = train_rows[:smoke_train_rows]
    smoke_valid = valid_rows[:smoke_valid_rows]
    _write_jsonl_atomic(outputs["train"], train_rows)
    _write_jsonl_atomic(outputs["valid"], valid_rows)
    _write_jsonl_atomic(outputs["smoke_train"], smoke_train)
    _write_jsonl_atomic(outputs["smoke_valid"], smoke_valid)

    report = {
        "input_metadata": str(input_metadata.resolve()),
        "group_field": group_field,
        "seed": seed,
        "requested_train_ratio": train_ratio,
        "actual_train_ratio": len(train_rows) / len(rows),
        "total_rows": len(rows),
        "total_tracks": group_count,
        "train_rows": len(train_rows),
        "train_tracks": len(train_ids),
        "valid_rows": len(valid_rows),
        "valid_tracks": len(valid_ids),
        "smoke_train_rows": len(smoke_train),
        "smoke_valid_rows": len(smoke_valid),
        "valid_track_ids": sorted(valid_ids),
        "leaked_track_ids": [],
    }
    _write_json_atomic(outputs["report"], report)

    print(f"Input:             {input_metadata.resolve()}")
    print(f"Output dir:        {output_dir.resolve()}")
    print(f"Group field:       {group_field}")
    print(f"Seed:              {seed}")
    print(f"Rows:              {len(rows)}")
    print(f"Tracks:            {group_count}")
    print(f"Train:             {len(train_rows)} rows / {len(train_ids)} tracks")
    print(f"Valid:             {len(valid_rows)} rows / {len(valid_ids)} tracks")
    print(f"Actual train ratio:{len(train_rows) / len(rows):10.4f}")
    print(f"Smoke train:       {len(smoke_train)} rows")
    print(f"Smoke valid:       {len(smoke_valid)} rows")
    print("Track leakage:     0")
    print(f"Report:            {outputs['report']}")


def _closest_group_subset(
    group_items: list[tuple[str, list[dict[str, Any]]]],
    group_count: int,
    target_rows: int,
) -> set[str]:
    """Choose exactly N groups with a row count closest to the target."""
    states: list[dict[int, tuple[int, ...]]] = [
        {} for _ in range(group_count + 1)
    ]
    states[0][0] = ()
    for item_index, (_, item_rows) in enumerate(group_items):
        row_count = len(item_rows)
        upper = min(group_count, item_index + 1)
        for selected_count in range(upper, 0, -1):
            previous = states[selected_count - 1]
            current = states[selected_count]
            for previous_rows, indices in list(previous.items()):
                total_rows = previous_rows + row_count
                current.setdefault(total_rows, indices + (item_index,))

    candidates = states[group_count]
    if not candidates:
        raise RuntimeError("Could not construct the requested validation groups.")
    selected_rows = min(candidates, key=lambda value: abs(value - target_rows))
    return {group_items[index][0] for index in candidates[selected_rows]}


def _assert_no_group_leakage(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    group_field: str,
) -> None:
    train_ids = {str(row[group_field]) for row in train_rows}
    valid_ids = {str(row[group_field]) for row in valid_rows}
    leaked = sorted(train_ids.intersection(valid_ids))
    if leaked:
        raise RuntimeError(f"Group leakage detected: {leaked}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}: {path}"
                )
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split V5 MusicGen metadata by track_id without train/valid leakage."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--group-field", default="track_id")
    parser.add_argument(
        "--valid-track-count",
        type=int,
        default=None,
        help="Exact number of validation tracks. Default: ratio-based.",
    )
    parser.add_argument("--smoke-train-rows", type=int, default=300)
    parser.add_argument("--smoke-valid-rows", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_dataset(
        input_metadata=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        group_field=args.group_field,
        valid_track_count=args.valid_track_count,
        smoke_train_rows=args.smoke_train_rows,
        smoke_valid_rows=args.smoke_valid_rows,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
