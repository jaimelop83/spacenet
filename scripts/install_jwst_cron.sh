#!/usr/bin/env bash
set -euo pipefail

CRON_LINE="0 21 * * * /home/jaimelop/spacenet/scripts/download_jwst_fits.py --out-dir /mnt/personal_drive/jwst/fits --max-files 500 --instruments NIRCAM MIRI >> /mnt/personal_drive/jwst/logs/jwst_fits_\\$(date +\\%F).log 2>&1"
CRON_TAG="# jwst_fits_download"

mkdir -p /mnt/personal_drive/jwst/logs

existing="$(crontab -l 2>/dev/null || true)"
if echo "$existing" | grep -q "$CRON_TAG"; then
  echo "Cron entry already exists."
  exit 0
fi

{
  echo "$existing"
  echo "$CRON_LINE $CRON_TAG"
} | crontab -

echo "Installed cron job:"
echo "$CRON_LINE $CRON_TAG"
