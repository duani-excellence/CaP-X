#!/usr/bin/env bash
# coder-status: whether CaP-X is running, which ports are up, GPU usage, log tail.
set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"

if is_running; then
  log "运行中 (tmux session '${TMUX_SESSION}', pid $(main_pid || echo '?'))"
else
  log "未运行"
fi

for entry in "${CAPX_WEB_UI_PORT}:web-ui" "${INTERNAL_PORTS[@]}"; do
  port="${entry%%:*}"
  name="${entry#*:}"
  if port_listening "${port}"; then
    printf '  %-9s :%-5s up\n' "${name}" "${port}"
  else
    printf '  %-9s :%-5s down\n' "${name}" "${port}"
  fi
done

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
print_access_info

if [[ -e "${LOG_DIR}/latest.log" ]]; then
  log "tail ${LOG_DIR}/latest.log"
  tail -n "${TAIL_LINES:-20}" "${LOG_DIR}/latest.log"
fi
