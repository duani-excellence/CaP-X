#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GRASPNET_SERVICE_URL=http://127.0.0.1:8215
exec "$CAPX_ROOT/.venv/bin/python" -B -m capx.training.teacher run "$@"
