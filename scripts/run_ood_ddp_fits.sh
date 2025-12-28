#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <fits_root> [extra args]"
  exit 1
fi

FITS_ROOT="$1"
shift 1

CHECKPOINT="$(ls -t ./checkpoints/spacenet_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "No spacenet_* checkpoints found in ./checkpoints"
  exit 1
fi

exec torchrun --nproc_per_node=2 evaluate_ood.py \
  --id-root ./img \
  --ood-root "$FITS_ROOT" \
  --ood-flat \
  --checkpoint "$CHECKPOINT" \
  "$@"
