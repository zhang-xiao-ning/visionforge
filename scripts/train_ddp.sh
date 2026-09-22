#!/usr/bin/env bash
# Launch DDP training with torchrun.
#
# Usage:
#   bash scripts/train_ddp.sh --experiment mlp --epochs 5
#   NPROC=2 bash scripts/train_ddp.sh --experiment vit --epochs 10
#
# Env vars:
#   NPROC       number of processes (default: 1)
#   MASTER_PORT rendezvous port (default: 29500)
set -euo pipefail

NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Launching DDP: nproc_per_node=${NPROC}, master_port=${MASTER_PORT}"
exec uv run torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    src/main.py "$@"