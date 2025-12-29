#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <id_root> <ood_root> [extra args]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ID_ROOT="$1"
OOD_ROOT="$2"
shift 2

if [[ "$ID_ROOT" != /* ]]; then
  ID_ROOT="$REPO_ROOT/$ID_ROOT"
fi
if [[ "$OOD_ROOT" != /* ]]; then
  OOD_ROOT="$REPO_ROOT/$OOD_ROOT"
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-/home/jaimelop/anaconda3/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  TORCHRUN_BIN="torchrun"
fi

CHECKPOINT="$(ls -t "$REPO_ROOT"/checkpoints/spacenet_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "No spacenet_* checkpoints found in $REPO_ROOT/checkpoints"
  exit 1
fi

exec "$TORCHRUN_BIN" --nproc_per_node=2 "$REPO_ROOT/evaluate_ood.py" \
  --id-root "$ID_ROOT" \
  --ood-root "$OOD_ROOT" \
  --checkpoint "$CHECKPOINT" \
  "$@"
