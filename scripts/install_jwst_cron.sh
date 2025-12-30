#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/jaimelop/anaconda3/bin/python}"
CRON_LINE='0 21 * * * '"${PYTHON_BIN}"' /home/jaimelop/spacenet/scripts/download_jwst_fits.py --out-dir /mnt/personal_drive/jwst/fits --max-files 500 --instruments NIRCAM MIRI >> /mnt/personal_drive/jwst/logs/jwst_fits_$(date +\%F).log 2>&1'
ROTATE_LINE='30 21 * * * /home/jaimelop/spacenet/scripts/rotate_jwst_logs.sh 14 >> /mnt/personal_drive/jwst/logs/jwst_rotate_$(date +\%F).log 2>&1'
CONVERT_LINE='0 23 * * * '"${PYTHON_BIN}"' /home/jaimelop/spacenet/scripts/convert_fits_to_png.py --in-root /mnt/personal_drive/jwst/fits --out-root /mnt/personal_drive/jwst/fits_png --overwrite >> /mnt/personal_drive/jwst/logs/jwst_fits_png_$(date +\%F).log 2>&1'
OOD_LINE='0 1 * * * /home/jaimelop/spacenet/scripts/run_ood_ddp.sh /home/jaimelop/spacenet/img /mnt/personal_drive/jwst/fits_png --ood-flat --metric max_softmax --auto-threshold-tpr 0.95 --out-csv /mnt/personal_drive/jwst/logs/ood_scores_fits_$(date +\%F).csv --batch-size 256 --num-workers 8 >> /mnt/personal_drive/jwst/logs/jwst_ood_$(date +\%F).log 2>&1'
CRON_TAG="# jwst_fits_download"
ROTATE_TAG="# jwst_fits_rotate"
CONVERT_TAG="# jwst_fits_png"
OOD_TAG="# jwst_fits_ood"

mkdir -p /mnt/personal_drive/jwst/logs

existing="$(crontab -l 2>/dev/null || true)"
existing="$(printf "%s\n" "$existing" | grep -v "$CRON_TAG" | grep -v "$ROTATE_TAG" | grep -v "$OOD_TAG" || true)"
existing="${existing}\n${CRON_LINE} ${CRON_TAG}"
existing="${existing}\n${ROTATE_LINE} ${ROTATE_TAG}"
existing="${existing}\n${CONVERT_LINE} ${CONVERT_TAG}"
existing="${existing}\n${OOD_LINE} ${OOD_TAG}"

printf "%b\n" "$existing" | crontab -

echo "Installed/updated cron jobs."
