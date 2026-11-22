#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CAPX_ROOT/outputs/reproduction/multitask-numba-cache}"
exec "$CAPX_ROOT/.venv/bin/python" -B -m capx.multitask.cli "$@"
