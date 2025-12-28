#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/mnt/personal_drive/jwst/logs"
KEEP_DAYS="${1:-14}"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "Missing log dir: $LOG_DIR"
  exit 1
fi

find "$LOG_DIR" -type f -name 'jwst_fits_*.log' -mtime +"$KEEP_DAYS" -print -delete
echo "Rotated logs older than ${KEEP_DAYS} days in $LOG_DIR"
