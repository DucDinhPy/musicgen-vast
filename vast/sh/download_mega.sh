#!/usr/bin/env bash

set -euo pipefail

# ============================ CONFIG =========================================
WORK="${WORK:-/workspace}"
DOWNLOAD_DATA_DIR="${DOWNLOAD_DATA_DIR:-$WORK/datasets/vinahouse/mega_audio}"


DOWNLOAD_MODE="${DOWNLOAD_MODE:-public}"

# Public download configuration.
MEGA_DATA_URL="${MEGA_DATA_URL:-}"

# Account download configuration.
MEGA_EMAIL="${MEGA_EMAIL:-}"
MEGA_PASSWORD="${MEGA_PASSWORD:-}"
MEGA_DOWNLOAD_REMOTE_DIR="${MEGA_DOWNLOAD_REMOTE_DIR:-/Root/vinahouse_audio}"
# =============================================================================

echo "==> [1/3] Validate config"

case "$DOWNLOAD_MODE" in
    public
        if [ -z "$MEGA_DATA_URL" ]; then
            echo "[error] DOWNLOAD_MODE=public requires MEGA_DATA_URL."
            exit 1
        fi
        ;;

    account
        if [ -z "$MEGA_EMAIL" ] || [ -z "$MEGA_PASSWORD" ]; then
            echo "[error] DOWNLOAD_MODE=account requires:"
            echo "        MEGA_EMAIL and MEGA_PASSWORD"
            exit 1
        fi
        ;;

    *
        echo "[error] Invalid DOWNLOAD_MODE: $DOWNLOAD_MODE"
        echo "        Valid values: public or account"
        exit 1
        ;;
esac

mkdir -p "$DOWNLOAD_DATA_DIR"

echo "==> [2/3] Install required MEGA tools"

apt-get update -qq
apt-get install -y -qq ca-certificates wget

if [ "$DOWNLOAD_MODE" = "public" ]; then
    if ! command -v mega-get >/dev/null 2>&1; then
        if ! apt-get install -y -qq megacmd; then
            . /etc/os-release

            MEGACMD_UBUNTU_VERSION="${MEGACMD_UBUNTU_VERSION:-$VERSION_ID}"
            MEGACMD_DEB_URL="${MEGACMD_DEB_URL:-https://mega.nz/linux/repo/xUbuntu_${MEGACMD_UBUNTU_VERSION}/amd64/megacmd-xUbuntu_${MEGACMD_UBUNTU_VERSION}_amd64.deb}"

            echo "Downloading: $MEGACMD_DEB_URL"

            wget -O /tmp/megacmd.deb "$MEGACMD_DEB_URL"
            apt-get install -y /tmp/megacmd.deb
        fi
    fi
else
    if ! command -v megacopy >/dev/null 2>&1; then
        apt-get install -y -qq megatools
    fi
fi

echo "==> [3/3] Download dataset"
echo "    Download mode: $DOWNLOAD_MODE"
echo "    Destination:   $DOWNLOAD_DATA_DIR"

if [ "$DOWNLOAD_MODE" = "public" ]; then
    mega-get "$MEGA_DATA_URL" "$DOWNLOAD_DATA_DIR"
else
    megacopy \
        --download \
        --local "$DOWNLOAD_DATA_DIR" \
        --remote "$MEGA_DOWNLOAD_REMOTE_DIR" \
        --username "$MEGA_EMAIL" \
        --password "$MEGA_PASSWORD"
fi

echo ""
echo "[done] Download completed."
echo "Local folder: $DOWNLOAD_DATA_DIR"