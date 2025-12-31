#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="${LOG_DIR:-/mnt/personal_drive/jwst/logs}"
REPORTS_DIR="${REPORTS_DIR:-$REPO_ROOT/reports}"
PUBLIC_REPORTS_DIR="${PUBLIC_REPORTS_DIR:-/mnt/personal_drive/reports}"

latest_csv="$(ls -t "$LOG_DIR"/ood_scores_fits_*.csv 2>/dev/null | head -n 1 || true)"
if [[ -z "$latest_csv" ]]; then
  echo "No FITS OOD CSV found in $LOG_DIR."
  exit 0
fi

python "$REPO_ROOT/scripts/plot_ood_hist.py" \
  --csv-path "$latest_csv" \
  --out-path "$REPORTS_DIR/fits_ood_hist.png" \
  --per-class --per-class-dir "$REPORTS_DIR/fits_ood_hist_per_class"

python "$REPO_ROOT/scripts/make_ood_examples.py" \
  --csv-path "$latest_csv" \
  --out-dir "$REPORTS_DIR/fits_ood_examples" \
  --top-n 60 --copy --html-prefix fits_ood_examples

rsync -a --delete "$REPORTS_DIR"/ "$PUBLIC_REPORTS_DIR"/ || {
  echo "Warning: rsync to $PUBLIC_REPORTS_DIR failed (permissions?)."
}
