#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
exec .venv/bin/python scripts/local/run.py "$@"
