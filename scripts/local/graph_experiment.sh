#!/usr/bin/env bash
# Thin entry point. Services, model settings and budgets belong to --config.
set -euo pipefail
case "${1:---help}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/local/graph_experiment.sh COMMAND [arguments]

Commands:
  pair                 Run one real DeepSeek paired development experiment.
  prepare-development  Create a resumable development campaign and scene allocation.
  prepare-snapshots    Prepare replay-verified fixed scene snapshots without model calls.
  export-initials      Freeze all normal initial outcomes for paired graph ablations.
  freeze               Freeze a formal campaign after development acceptance.
  run                  Start pending campaign units (paid model calls).
  resume               Resume completed-unit checkpoints without duplicate requests.
  request-stop         Finish the current pair, then pause without losing progress.
  status               Read campaign progress and budget usage.
  analyze              Compute paired, scene-clustered results.
  contrast             Compare full-G1 to ablated-G1 on the fixed shared subset.

Use COMMAND --help for the existing runner's full options.
Always choose a new output directory. Only resume reuses a campaign directory.
HELP
    exit 0 ;;
  pair) CAPX_GRAPH_MODULE=capx.experiments.graph_v2 ;;
  prepare-development|prepare-snapshots|export-initials|freeze|run|resume|request-stop|status|analyze|contrast) CAPX_GRAPH_MODULE=capx.experiments.graph_campaign ;;
  *) echo "Unknown command: $1" >&2; exit 2 ;;
esac
CAPX_GRAPH_COMMAND="$1"
shift
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/capx-graph-numba-$UID}"
if [[ "$CAPX_GRAPH_COMMAND" == pair ]]; then
  exec "$CAPX_ROOT/.venv/bin/python" -B -m "$CAPX_GRAPH_MODULE" "$@"
else
  exec "$CAPX_ROOT/.venv/bin/python" -B -m "$CAPX_GRAPH_MODULE" "$CAPX_GRAPH_COMMAND" "$@"
fi
