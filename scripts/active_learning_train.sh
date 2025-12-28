#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./img_active}"
CHECKPOINT="$(ls -t ./checkpoints/spacenet_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "No spacenet_* checkpoints found in ./checkpoints"
  exit 1
fi

exec torchrun --nproc_per_node=2 train.py \
  --data-root "$DATA_ROOT" \
  --epochs 30 \
  --batch-size 64 \
  --amp \
  --pretrained-encoder "$CHECKPOINT"
