#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -B -m capx.experiments.unified "$@"
