#!/usr/bin/env bash
set -euo pipefail

CRON_LINE="0 21 * * * /home/jaimelop/spacenet/scripts/download_jwst_fits.py --out-dir /mnt/personal_drive/jwst/fits --max-files 500 --instruments NIRCAM MIRI >> /mnt/personal_drive/jwst/logs/jwst_fits_\\$(date +\\%F).log 2>&1"
ROTATE_LINE="30 21 * * * /home/jaimelop/spacenet/scripts/rotate_jwst_logs.sh 14 >> /mnt/personal_drive/jwst/logs/jwst_rotate_\\$(date +\\%F).log 2>&1"
OOD_LINE="0 1 * * * /home/jaimelop/spacenet/scripts/run_ood_ddp_fits.sh /mnt/personal_drive/jwst/fits --metric max_softmax --auto-threshold-tpr 0.95 --out-csv /mnt/personal_drive/jwst/logs/ood_scores_fits_\\$(date +\\%F).csv --batch-size 256 --num-workers 8 >> /mnt/personal_drive/jwst/logs/jwst_ood_\\$(date +\\%F).log 2>&1"
CRON_TAG="# jwst_fits_download"
ROTATE_TAG="# jwst_fits_rotate"
OOD_TAG="# jwst_fits_ood"

mkdir -p /mnt/personal_drive/jwst/logs

existing="$(crontab -l 2>/dev/null || true)"
if echo "$existing" | grep -q "$CRON_TAG"; then
  echo "Download cron entry already exists."
else
  existing="${existing}\n${CRON_LINE} ${CRON_TAG}"
fi

if echo "$existing" | grep -q "$ROTATE_TAG"; then
  echo "Rotate cron entry already exists."
else
  existing="${existing}\n${ROTATE_LINE} ${ROTATE_TAG}"
fi

if echo "$existing" | grep -q "$OOD_TAG"; then
  echo "OOD cron entry already exists."
else
  existing="${existing}\n${OOD_LINE} ${OOD_TAG}"
fi


printf "%b\n" "$existing" | crontab -

echo "Installed/verified cron jobs."
