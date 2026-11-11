#!/usr/bin/env bash
# Shared settings for the server-side scripts (setup_env / start / stop / status).
# Sourced, not executed.

CLOUD_SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
APP_DIR="$(cd "${CLOUD_SCRIPT_DIR}/../.." && pwd)"
CLOUD_ENV_FILE="${CLOUD_ENV_FILE:-${CLOUD_SCRIPT_DIR}/cloud.env}"

if [[ -f "${CLOUD_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CLOUD_ENV_FILE}"
  set +a
fi

TMUX_SESSION="${TMUX_SESSION:-coder}"
LOG_DIR="${LOG_DIR:-${APP_DIR}/logs}"
STATE_FILE="${LOG_DIR}/.coder.state"
PID_FILE="${LOG_DIR}/.coder.pid"
VENV_PY="${APP_DIR}/.venv/bin/python"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.uv-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-${APP_DIR}/models/sam3.pt}"
export CAPX_SAM3_CHECKPOINT="${CAPX_SAM3_CHECKPOINT:-${SAM3_CHECKPOINT_PATH}}"
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-${APP_DIR}/robot_descriptions}"
export CAPX_PANDA_URDF_PATH="${CAPX_PANDA_URDF_PATH:-${ROBOT_DESCRIPTIONS_CACHE}/example-robot-data/robots/panda_description/urdf/panda.urdf}"

CAPX_WEB_UI_PORT="${CAPX_WEB_UI_PORT:-6006}"

# Internal API servers; ports are fixed in capx/integrations/* clients.
INTERNAL_PORTS=(8114:sam3 8115:graspnet 8116:pyroki)

if [[ -t 1 ]]; then
  C_RED=$'\033[1;31m'
  C_RESET=$'\033[0m'
else
  C_RED=""
  C_RESET=""
fi

log() { printf '[coder] %s\n' "$*"; }
warn() { printf '[coder] WARNING: %s\n' "$*" >&2; }
die() { printf '[coder] ERROR: %s\n' "$*" >&2; exit 1; }

# AutoDL "学术加速" proxy for GitHub / HuggingFace. Only enable inside a
# subshell: it slows down the domestic PyPI mirror and the DeepSeek API.
with_turbo() {
  (
    if [[ -f /etc/network_turbo ]]; then
      # shellcheck disable=SC1091
      source /etc/network_turbo >/dev/null
    fi
    "$@"
  )
}

port_listening() {
  timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

session_running() {
  tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null
}

main_pid() {
  local pid
  [[ -f "${PID_FILE}" ]] || return 1
  pid="$(<"${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  # A stale PID file (reboot / migrated instance) may point at an unrelated process.
  [[ "$(tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null | head -n 1)" == "${VENV_PY}" ]] || return 1
  echo "${pid}"
}

is_running() {
  session_running || main_pid >/dev/null
}

# Port / public URL of the running instance (coder-start may override cloud.env).
if [[ -f "${STATE_FILE}" ]] && is_running; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
fi

print_access_info() {
  echo "  访问方式:"
  if [[ -n "${CAPX_PUBLIC_URL:-}" ]]; then
    echo "    公网访问:  ${CAPX_PUBLIC_URL}"
  else
    echo "    公网访问:  AutoDL 不支持，请使用下面的 SSH 隧道"
  fi
  echo "    SSH 隧道:  在自己电脑上另开一个终端窗口执行:"
  echo "               ssh -N -L ${CAPX_WEB_UI_PORT}:127.0.0.1:${CAPX_WEB_UI_PORT} ${CAPX_SSH_ENDPOINT:--p <端口> root@<地址>}"
  if [[ -n "${CAPX_SSH_PASSWORD:-}" ]]; then
    echo "               密码: ${CAPX_SSH_PASSWORD}  (输入时不显示字符)"
  fi
  echo "               回车后无输出、光标停住即为成功，窗口保持打开"
  echo "               然后浏览器打开 http://localhost:${CAPX_WEB_UI_PORT}"
}
