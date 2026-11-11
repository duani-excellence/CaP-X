#!/usr/bin/env bash
# coder-stop: stop the CaP-X Web UI started by coder-start (tmux session) and
# its API servers (SAM3 / GraspNet / PyRoKi), then free the GPU.
#
# Order: SIGINT to launch.py (its `finally` terminates the API servers) ->
# SIGTERM / SIGKILL to the whole process tree -> kill the tmux session ->
# sweep leftover venv processes and verify the ports are free.
set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"

GRACE_SECONDS="${GRACE_SECONDS:-30}"

usage() {
  cat <<EOF
用法: coder-stop [-f|--force] [-h|--help]
  -f, --force   跳过优雅退出，直接强制结束所有相关进程
EOF
}

FORCE=0
case "${1:-}" in
  -f|--force) FORCE=1 ;;
  -h|--help)  usage; exit 0 ;;
  "") ;;
  *) die "unknown option: $1 (see coder-stop -h)" ;;
esac

descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    echo "${child}"
    descendants "${child}"
  done
}

alive_any() {
  local pid
  for pid in "$@"; do
    kill -0 "${pid}" 2>/dev/null && return 0
  done
  return 1
}

wait_gone() {
  local seconds="$1"; shift
  for ((i = 0; i < seconds; i++)); do
    alive_any "$@" || return 0
    sleep 1
  done
  return 1
}

venv_pids() {
  pgrep -f "^${VENV_PY} " 2>/dev/null || true
}

if ! is_running && [[ -z "$(venv_pids)" ]]; then
  log "服务未在运行"
  rm -f "${PID_FILE}" "${STATE_FILE}"
  exit 0
fi

PID="$(main_pid || true)"
if [[ -n "${PID}" ]]; then
  # Collect the tree first: children get re-parented once launch.py exits.
  mapfile -t TREE < <(echo "${PID}"; descendants "${PID}")

  if (( ! FORCE )); then
    log "stopping launch.py (pid ${PID}) gracefully, up to ${GRACE_SECONDS}s..."
    kill -INT "${PID}" 2>/dev/null || true
    wait_gone "${GRACE_SECONDS}" "${TREE[@]}" || warn "graceful shutdown timed out"
  fi

  if alive_any "${TREE[@]}"; then
    log "terminating process tree (${#TREE[@]} processes)"
    kill -TERM "${TREE[@]}" 2>/dev/null || true
    wait_gone 5 "${TREE[@]}" || kill -KILL "${TREE[@]}" 2>/dev/null || true
  fi
elif session_running && (( ! FORCE )); then
  # No PID file (e.g. started by hand inside the session): emulate Ctrl-C.
  log "no pid file; sending Ctrl-C to tmux session '${TMUX_SESSION}'"
  tmux send-keys -t "=${TMUX_SESSION}" C-c 2>/dev/null || true
  for ((i = 0; i < GRACE_SECONDS; i++)); do
    session_running || break
    sleep 1
  done
fi

if session_running; then
  log "closing tmux session '${TMUX_SESSION}'"
  tmux kill-session -t "=${TMUX_SESSION}" 2>/dev/null || true
fi

LEFTOVER="$(venv_pids)"
if [[ -n "${LEFTOVER}" ]]; then
  log "killing leftover processes: $(echo ${LEFTOVER})"
  # shellcheck disable=SC2086
  kill -TERM ${LEFTOVER} 2>/dev/null || true
  sleep 3
  LEFTOVER="$(venv_pids)"
  # shellcheck disable=SC2086
  [[ -n "${LEFTOVER}" ]] && kill -KILL ${LEFTOVER} 2>/dev/null || true
fi

rm -f "${PID_FILE}" "${STATE_FILE}"

busy=()
for entry in "${CAPX_WEB_UI_PORT}:web-ui" "${INTERNAL_PORTS[@]}"; do
  port_listening "${entry%%:*}" && busy+=("${entry}")
done
if (( ${#busy[@]} )); then
  warn "ports still in use: ${busy[*]} (held by a process outside ${VENV_PY}?); try: coder-stop --force"
  exit 1
fi

log "已停止 (GPU 显存已释放)；重新启动请执行 coder-start"
