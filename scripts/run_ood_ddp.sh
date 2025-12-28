#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <id_root> <ood_root> [extra args]"
  exit 1
fi

ID_ROOT="$1"
OOD_ROOT="$2"
shift 2

CHECKPOINT="$(ls -t ./checkpoints/spacenet_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "No spacenet_* checkpoints found in ./checkpoints"
  exit 1
fi

exec torchrun --nproc_per_node=2 evaluate_ood.py \
  --id-root "$ID_ROOT" \
  --ood-root "$OOD_ROOT" \
  --checkpoint "$CHECKPOINT" \
  "$@"
