#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <fits_root> [extra args]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FITS_ROOT="$1"
shift 1

if [[ "$FITS_ROOT" != /* ]]; then
  FITS_ROOT="$REPO_ROOT/$FITS_ROOT"
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
  --id-root "$REPO_ROOT/img" \
  --ood-root "$FITS_ROOT" \
  --ood-flat \
  --checkpoint "$CHECKPOINT" \
  "$@"
