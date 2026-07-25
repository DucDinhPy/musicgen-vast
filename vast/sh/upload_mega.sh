#!/usr/bin/env bash

set -Eeuo pipefail

# ============================ CONFIG =========================================
WORK="${WORK:-/workspace}"

# Thư mục local trên Vast.ai cần upload.
UPLOAD_DATA_DIR="${UPLOAD_DATA_DIR:-$WORK/datasets/vinahouse/mega_audio}"

# Thư mục đích trong tài khoản MEGA.
# Ví dụ: /Root/vinahouse_midi_pipeline
DESTINATION_DIR="${DESTINATION_DIR:-/Root/vinahouse_midi_pipeline}"

# Nên export từ terminal thay vì ghi trực tiếp mật khẩu vào file.
MEGA_EMAIL="${MEGA_EMAIL:-}"
MEGA_PASSWORD="${MEGA_PASSWORD:-}"
# =============================================================================

log() {
    printf '%s\n' "$*"
}

error_exit() {
    printf '[error] %s\n' "$*" >&2
    exit 1
}

cleanup() {
    # Không để phiên đăng nhập MEGA tồn tại sau khi script kết thúc.
    mega-logout >/dev/null 2>&1 || true
}

trap cleanup EXIT
trap 'error_exit "Upload failed at line $LINENO."' ERR


log "==> [1/4] Validate configuration"

[[ -n "$MEGA_EMAIL" ]] ||
    error_exit "MEGA_EMAIL is required."

[[ -n "$MEGA_PASSWORD" ]] ||
    error_exit "MEGA_PASSWORD is required."

[[ -n "$DESTINATION_DIR" ]] ||
    error_exit "DESTINATION_DIR must not be empty."

[[ -d "$UPLOAD_DATA_DIR" ]] ||
    error_exit "Upload directory does not exist: $UPLOAD_DATA_DIR"

log "    Upload source: $UPLOAD_DATA_DIR"
log "    Destination:   $DESTINATION_DIR"


log "==> [2/4] Install required tools"

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg


if ! command -v mega-put >/dev/null 2>&1; then
    log "    MEGAcmd not found. Installing..."

    # Thử package đã có sẵn trong repository của image trước.
    if ! apt-get install -y -qq megacmd; then
        log "    Installing MEGAcmd from the official MEGA repository..."

        curl -fsSL \
            "https://mega.nz/linux/repo/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb" \
            -o /tmp/megacmd.deb

        apt-get install -y /tmp/megacmd.deb
        rm -f /tmp/megacmd.deb
    fi
fi

command -v mega-login >/dev/null 2>&1 ||
    error_exit "mega-login was not installed correctly."

command -v mega-put >/dev/null 2>&1 ||
    error_exit "mega-put was not installed correctly."


log "==> [3/4] Login and prepare destination"

# Xóa phiên đăng nhập cũ nếu có.
mega-logout >/dev/null 2>&1 || true

mega-login "$MEGA_EMAIL" "$MEGA_PASSWORD"

# Kiểm tra tài khoản đã đăng nhập.
mega-whoami

# Tạo thư mục đích nếu chưa tồn tại.
mega-mkdir -p "$DESTINATION_DIR" >/dev/null 2>&1 || true


log "==> [4/4] Upload dataset"

mega-put \
    -- \
    "$UPLOAD_DATA_DIR" \
    "$DESTINATION_DIR"


log ""
log "[done] Upload completed."
log "Local folder: $UPLOAD_DATA_DIR"
log "MEGA folder:  $DESTINATION_DIR"