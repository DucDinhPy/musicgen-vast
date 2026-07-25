#!/usr/bin/env python3
from __future__ import annotations

"""
CLI commands and arguments

download:
  --work
  --mode {public,account}
  --local-dir
  --url
  --remote-dir
  --email
  --password

upload:
  --work
  --local-dir
  --destination-dir
  --email
  --password

Examples:
  python mega_cli.py download --mode public --url "<MEGA_URL>" --local-dir /workspace/data

  python mega_cli.py download --mode account --remote-dir /Root/data --local-dir /workspace/data

  python mega_cli.py upload --local-dir /workspace/data --destination-dir /Root/data

Prefer setting credentials through environment variables:
  export MEGA_EMAIL="user@example.com"
  export MEGA_PASSWORD="password"
"""

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VAST_DIR = SCRIPT_DIR.parent
SH_DIR = VAST_DIR / "sh"
DOWNLOAD_SCRIPT = SH_DIR / "download_mega.sh"
UPLOAD_SCRIPT = SH_DIR / "upload_mega.sh"


def run_download(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    _set_if_not_none(env, "WORK", args.work)
    _set_if_not_none(env, "DOWNLOAD_MODE", args.mode)
    _set_if_not_none(env, "DOWNLOAD_DATA_DIR", args.local_dir)
    _set_if_not_none(env, "MEGA_DATA_URL", args.url)
    _set_if_not_none(env, "MEGA_DOWNLOAD_REMOTE_DIR", args.remote_dir)
    _set_if_not_none(env, "MEGA_EMAIL", args.email)
    _set_if_not_none(env, "MEGA_PASSWORD", args.password)

    _run_shell(DOWNLOAD_SCRIPT, env)


def run_upload(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    _set_if_not_none(env, "WORK", args.work)
    _set_if_not_none(env, "UPLOAD_DATA_DIR", args.local_dir)
    _set_if_not_none(env, "DESTINATION_DIR", args.destination_dir)
    _set_if_not_none(env, "MEGA_EMAIL", args.email)
    _set_if_not_none(env, "MEGA_PASSWORD", args.password)

    _run_shell(UPLOAD_SCRIPT, env)


def _set_if_not_none(
    env: dict[str, str],
    key: str,
    value: str | Path | None,
) -> None:
    if value is not None:
        env[key] = str(value)


def _run_shell(script: Path, env: dict[str, str]) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing shell script: {script}")

    print(f"==> Running: {script}")
    subprocess.run(["bash", str(script)], check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper CLI dùng lại vast/sh/download_mega.sh và "
            "vast/sh/upload_mega.sh."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download",
        help="Download data từ MEGA về máy local/Vast.",
    )
    download.add_argument(
        "--work",
        default=None,
        help="Override WORK. Default trong shell script là /workspace.",
    )
    download.add_argument(
        "--mode",
        choices=["public", "account"],
        default=None,
        help="public dùng MEGA_DATA_URL, account dùng MEGA account remote dir.",
    )
    download.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Override DOWNLOAD_DATA_DIR.",
    )
    download.add_argument(
        "--url",
        default=None,
        help="Public MEGA folder/file URL. Set MEGA_DATA_URL.",
    )
    download.add_argument(
        "--remote-dir",
        default=None,
        help="MEGA remote dir for account mode. Set MEGA_DOWNLOAD_REMOTE_DIR.",
    )
    download.add_argument(
        "--email",
        default=None,
        help="MEGA email. Prefer env MEGA_EMAIL to avoid shell history.",
    )
    download.add_argument(
        "--password",
        default=None,
        help="MEGA password. Prefer env MEGA_PASSWORD to avoid shell history.",
    )
    download.set_defaults(func=run_download)

    upload = subparsers.add_parser(
        "upload",
        help="Upload folder local/Vast lên MEGA.",
    )
    upload.add_argument(
        "--work",
        default=None,
        help="Override WORK. Default trong shell script là /workspace.",
    )
    upload.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Override UPLOAD_DATA_DIR.",
    )
    upload.add_argument(
        "--destination-dir",
        default=None,
        help="MEGA destination dir. Set DESTINATION_DIR.",
    )
    upload.add_argument(
        "--email",
        default=None,
        help="MEGA email. Prefer env MEGA_EMAIL to avoid shell history.",
    )
    upload.add_argument(
        "--password",
        default=None,
        help="MEGA password. Prefer env MEGA_PASSWORD to avoid shell history.",
    )
    upload.set_defaults(func=run_upload)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
