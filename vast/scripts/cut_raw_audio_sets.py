#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class Track:
    start: str
    title: str


SOURCE_SET_TITLES = [
    "1 NHẠC REMIX TIKTOK TRIỆU VIEW - BXH Nhạc Trẻ Remix Hay Nhất Hiện Nay - Top 20 Nhạc Hot TikTok 2026",
    "2 NHẠC REMIX TIKTOK TRIỆU VIEW - BXH Nhạc Trẻ Remix Hay Nhất Hiện Nay - Top 20 Nhạc TikTok Hay 2026",
    "3 NHẠC REMIX TIKTOK TRIỆU VIEW - BXH Nhạc Trẻ Remix Hay Nhất Hiện Nay Top 20 Nhạc TikTok Hay 2026",
    "4 TOP 30 NHẠC REMIX TIKTOK TRIỆU VIEW 2024 Vở Kịch Của Em Thu Cuối Lao Tâm Khổ Tứ Nguyệt Hồng Phai",
]


TRACKLISTS: list[list[Track]] = [
    [
        Track("00:00:00", "Sau Này Em Cưới Ai Rồi - Kiều Chi x Orinn Mix"),
        Track("00:05:11", "Thương Anh Ai Thương Em - Gia Hân x Orinn Mix"),
        Track("00:09:49", "Anh Đã Lừa Dối Em Rồi - Quang Kiệt x Orinn Mix"),
        Track("00:13:54", "Em Thua Cô Ta - Min Quỳnh Anh x Orinn Mix"),
        Track("00:18:23", "Ngày Hai Ta Sát Vai - Lê Thu Thảo x Orinn Mix"),
        Track("00:21:39", "Mở Lối Cho Em 2 - Lương Quý Tuấn, An Clock x Orinn Mix"),
        Track("00:25:19", "Lệ Lưu Ly - Vũ Phụng Tiên, DT Tập Rap x Orinn Mix"),
        Track("00:27:50", "Yêu 3 Năm Dại 1 Giờ - Chu Thúy Quỳnh x Orinn Mix"),
        Track("00:31:59", "Có Công Mài Sắc - Ngô Lan Hương x Orinn Mix"),
        Track("00:36:29", "Ngày Em Cưới - Nguyễn Vĩ x Orinn Mix"),
        Track("00:40:40", "Một Tình Yêu Hai Thử Thách - Luân Ken x Orinn Mix"),
        Track("00:44:42", "Cảm Ơn Vì Tất Cả - Anh Quân Idol x Orinn Mix"),
        Track("00:49:00", "Giá Như Anh Là Người Vô Tâm - Gia Hân Cover x Orinn Mix"),
        Track("00:52:36", "Một Bước Yêu Vạn Dặm Đau - Mr Siro x Orinn Mix"),
        Track("00:57:53", "Nợ Nhau Một Lời - Phúc Chinh x Orinn Mix"),
        Track("01:01:44", "Xin Một Đêm Yêu Em - Lan Vy Cover x Orinn Mix"),
    ],
    [
        Track("00:00", "Sau Này Em Cưới Ai Rồi"),
        Track("05:27", "Anh Đã Lừa Dối Em Rồi"),
        Track("09:35", "Lỡ Một Lời Thương"),
        Track("13:14", "Yêu Thật Khó Xoá Thật Đau"),
        Track("17:11", "Sơn Thuỷ Trùng Mây"),
        Track("21:25", "Chấp Niệm Trong Em"),
        Track("24:37", "Một Tình Yêu Hai Thử Thách"),
        Track("30:04", "Vở Kịch Của Em"),
        Track("34:28", "Câu Hứa Chưa Vẹn Tròn"),
        Track("38:45", "Anh Vội Quên"),
        Track("42:00", "Có Mình Và Ta"),
        Track("47:38", "Nắng Dưới Chân Mây"),
        Track("51:59", "Hẹn Hò Nhưng Không Yêu"),
        Track("55:12", "Một Lần Yêu"),
        Track("58:14", "Thương Lấy Phận Mình"),
        Track("01:02:54", "Mở Lòng Vì Ai"),
    ],
    [
        Track("00:00", "Anh Đã Lừa Dối Em Rồi"),
        Track("04:37", "Chữ Vấn Chữ Vương"),
        Track("09:18", "Kẻ Say Tình 2"),
        Track("13:08", "Sau Này Em Cưới Ai Rồi"),
        Track("17:48", "Hạt Mưa Vương Vấn"),
        Track("23:30", "Mashup Mở Lòng Vì Ai x Em Thua Cô Ta"),
        Track("28:59", "Thương Lấy Phận Mình"),
        Track("33:06", "Chấp Niệm Trong Em"),
        Track("36:16", "Thôi Nín Đi Em"),
        Track("41:26", "Em Ơi Anh Phải Làm Sao"),
        Track("45:25", "Vầng Trăng Khóc"),
        Track("49:59", "Khó Gần Dễ Xa"),
        Track("54:14", "Khoảng Cách"),
        Track("58:03", "Trú Mưa"),
        Track("01:02:13", "Không Cảm Xúc"),
    ],
    [
        Track("00:00", "Vở Kịch Của Em"),
        Track("03:52", "Thu Cuối"),
        Track("08:18", "Lao Tâm Khổ Tứ"),
        Track("12:02", "Nguyệt Hồng Phai"),
        Track("15:25", "Không Bằng"),
        Track("19:16", "Khi Yêu Nào Đâu Ai Muốn"),
        Track("23:26", "Cắt Đôi Nỗi Sầu"),
        Track("26:35", "Lệ Lưu Ly"),
        Track("29:38", "Vừa Hận Vừa Yêu"),
        Track("32:16", "Thương Thầm"),
        Track("36:06", "Tuyệt Sắc"),
        Track("39:56", "Em Mây"),
        Track("43:37", "Hoa Cỏ Lau"),
        Track("47:44", "Cẩm Tú Cầu"),
        Track("52:04", "Có Lẽ Bên Nhau Là Sai"),
        Track("55:23", "Đông Mang"),
        Track("58:45", "Ngoại Trừ Anh x Vết Thương Chưa Lành"),
        Track("01:01:58", "Người Tính Duyên Trời"),
        Track("01:06:42", "Tát Nhật Lãng Rực Rỡ"),
        Track("01:10:04", "Gió"),
        Track("01:13:31", "Mây"),
        Track("01:18:13", "Buồn Không Thể Buông"),
        Track("01:21:23", "Khoan Thai"),
        Track("01:25:17", "Unknown Track 24"),
        Track("01:28:14", "Em Vội Quên"),
        Track("01:31:08", "Unknown Track 26"),
        Track("01:34:13", "Không Cảm Xúc"),
        Track("01:37:14", "Hy Vọng Quá Hoá Đau Lòng"),
        Track("01:40:54", "Hoa Cưới"),
        Track("01:44:47", "Một Đời Tương Tư"),
        Track("01:48:24", "Khuất Lối"),
        Track("01:51:44", "Tòng Phu"),
    ],
]


def cut_all_sets(
    input_dir: Path,
    output_dir: Path,
    input_files: list[Path] | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Missing ffmpeg. Install it before running this script.")

    sources = input_files or _discover_audio_files_by_expected_names(input_dir)
    if len(sources) != len(TRACKLISTS):
        raise RuntimeError(
            f"Expected {len(TRACKLISTS)} source audio files, got {len(sources)}. "
            "Make sure filenames match the expected set titles, or use "
            "--input-file four times to force the mapping."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cut_manifest.jsonl"

    print(f"Input dir:  {input_dir.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Sources:    {len(sources)}")
    print(f"Dry run:    {dry_run}")
    print("")
    for index, source in enumerate(sources, start=1):
        print(f"set_{index:02d}: {source}")

    if dry_run:
        print("")
        print("Dry run only. No files will be written.")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for set_index, (source, tracks) in enumerate(zip(sources, TRACKLISTS), start=1):
            set_dir = output_dir / f"set_{set_index:02d}"
            set_dir.mkdir(parents=True, exist_ok=True)

            for track_index, track in enumerate(tracks, start=1):
                next_start = tracks[track_index].start if track_index < len(tracks) else None
                output_name = _output_name(set_index, track_index, track.title)
                output_path = set_dir / output_name

                row = {
                    "set_index": set_index,
                    "track_index": track_index,
                    "title": track.title,
                    "start": track.start,
                    "end": next_start,
                    "source": str(source),
                    "output": str(output_path),
                }

                if dry_run:
                    print(json.dumps(row, ensure_ascii=False))
                    continue

                if output_path.exists() and not overwrite:
                    print(f"[skip] Exists: {output_path}")
                    manifest.write(json.dumps({**row, "status": "skipped"}, ensure_ascii=False) + "\n")
                    continue

                _cut_audio(
                    ffmpeg=ffmpeg,
                    source=source,
                    output=output_path,
                    start=track.start,
                    end=next_start,
                )
                manifest.write(json.dumps({**row, "status": "ok"}, ensure_ascii=False) + "\n")
                manifest.flush()
                print(f"Wrote: {output_path}")

    print("")
    print(f"Done. Manifest: {manifest_path}")


def _discover_audio_files_by_expected_names(input_dir: Path) -> list[Path]:
    audio_files = _discover_audio_files(input_dir)
    matched: list[Path] = []
    used: set[Path] = set()

    for set_index, expected_title in enumerate(SOURCE_SET_TITLES, start=1):
        expected_slug = _slugify(expected_title)
        candidates = [
            path
            for path in audio_files
            if path not in used and _filename_matches_expected(path, expected_slug)
        ]

        if len(candidates) != 1:
            print("")
            print(f"[error] Could not uniquely match source set {set_index:02d}")
            print(f"Expected title: {expected_title}")
            print(f"Expected slug:  {expected_slug}")
            print("Candidates:")
            for path in candidates:
                print(f"  - {path.name}")
            print("")
            print("Available audio files:")
            for path in audio_files:
                print(f"  - {path.name}")
            raise RuntimeError(
                "Filename matching failed. Rename the source file to match the "
                "tracklist title, or pass --input-file four times in the desired order."
            )

        matched.append(candidates[0])
        used.add(candidates[0])

    return matched


def _discover_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input must be a folder: {input_dir}")

    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def _filename_matches_expected(path: Path, expected_slug: str) -> bool:
    file_slug = _slugify(path.stem)
    return file_slug == expected_slug or expected_slug in file_slug


def _cut_audio(
    ffmpeg: str,
    source: Path,
    output: Path,
    start: str,
    end: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        start,
        "-i",
        str(source),
    ]

    if end is not None:
        command.extend(["-to", end])

    command.extend(
        [
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )

    subprocess.run(command, check=True)


def _output_name(set_index: int, track_index: int, title: str) -> str:
    slug = _slugify(title)
    return f"set{set_index:02d}_track{track_index:02d}_{slug}.wav"


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    ascii_text = ascii_text.strip("_")
    return ascii_text or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut four long raw audio sets into single-track WAV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/musicgen-vast/data/raw/raw_audio_set_single"),
        help="Folder containing the four long source audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/musicgen-vast/data/preprocess/pre_audio_set_single"),
        help="Folder where cut tracks will be written.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        action="append",
        default=None,
        help=(
            "Explicit source file mapping. Pass exactly four times to override "
            "filename-based matching."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cut files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned cuts without writing files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cut_all_sets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_files=args.input_file,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
