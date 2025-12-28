#!/usr/bin/env bash
set -euo pipefail

FITS_DIR="/mnt/personal_drive/jwst/fits"
LOG_DIR="/mnt/personal_drive/jwst/logs"

if [[ ! -d "$FITS_DIR" ]]; then
  echo "Missing FITS dir: $FITS_DIR"
  exit 1
fi

echo "FITS directory: $FITS_DIR"
echo "Total FITS files: $(find "$FITS_DIR" -type f \\( -name '*.fits' -o -name '*.fits.gz' \\) | wc -l)"
echo "Total size: $(du -sh "$FITS_DIR" | awk '{print $1}')"

latest_log="$(ls -t "$LOG_DIR"/jwst_fits_*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "$latest_log" ]]; then
  echo "Latest log: $latest_log"
  echo "Last 20 lines:"
  tail -n 20 "$latest_log"
else
  echo "No logs found in $LOG_DIR"
fi
