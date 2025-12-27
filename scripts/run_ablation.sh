#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./img}"
ENCODER="${2:-}"
EPOCHS="${3:-30}"
BATCH_SIZE="${4:-64}"

COMMON_ARGS=(--data-root "$DATA_ROOT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --amp)
if [[ -n "$ENCODER" ]]; then
  COMMON_ARGS+=(--pretrained-encoder "$ENCODER")
fi

for AUG in light default heavy; do
  torchrun --nproc_per_node=2 train.py \
    "${COMMON_ARGS[@]}" \
    --aug-strength "$AUG" \
    --csv-log "./logs/train_${AUG}.csv" \
    --log-dir "./runs/${AUG}"
done
