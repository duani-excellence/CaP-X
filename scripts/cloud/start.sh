#!/usr/bin/env bash
# coder-start: start the CaP-X Web UI (and its SAM3 / GraspNet / PyRoKi servers)
# in a detached tmux session; keeps running after SSH logout.
# Priority: command-line options > scripts/cloud/cloud.env > defaults.
set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "${APP_DIR}"

usage() {
  cat <<EOF
用法: coder-start [选项]

  -k, --api-key KEY       LLM API key (DeepSeek)
  -p, --port PORT         Web UI 端口 (当前: ${CAPX_WEB_UI_PORT})
  -m, --model NAME        模型名 (当前: ${CAPX_MODEL:-deepseek-v4-pro})
  -u, --server-url URL    OpenAI 兼容接口 (当前: ${CAPX_SERVER_URL:-https://api.deepseek.com/v1/chat/completions})
  -c, --config PATH       任务配置 yaml (当前: ${CAPX_CONFIG_PATH:-env_configs/cube_stack/franka_robosuite_cube_stack.yaml})
  -t, --temperature T     采样温度
      --public-url URL    公网访问地址 (仅用于提示显示)
      --save              把本次传入的参数写入 cloud.env，下次直接 coder-start 即可
      --fg                前台运行 (调试用，Ctrl-C 退出)
  -h, --help              显示帮助

示例:
  coder-start
  coder-start -k sk-xxxx --save
  coder-start -p 6008 -m deepseek-v4-flash
EOF
}

SAVE=0
FOREGROUND=0
declare -A CLI=()
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "option $1 requires a value (see coder-start -h)"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--api-key)     need_value "$@"; CLI[CAPX_API_KEY]="$2"; shift 2 ;;
    -p|--port)        need_value "$@"; CLI[CAPX_WEB_UI_PORT]="$2"; shift 2 ;;
    -m|--model)       need_value "$@"; CLI[CAPX_MODEL]="$2"; shift 2 ;;
    -u|--server-url)  need_value "$@"; CLI[CAPX_SERVER_URL]="$2"; shift 2 ;;
    -c|--config)      need_value "$@"; CLI[CAPX_CONFIG_PATH]="$2"; shift 2 ;;
    -t|--temperature) need_value "$@"; CLI[CAPX_TEMPERATURE]="$2"; shift 2 ;;
    --public-url)     need_value "$@"; CLI[CAPX_PUBLIC_URL]="$2"; shift 2 ;;
    --save)           SAVE=1; shift ;;
    --fg)             FOREGROUND=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *)                die "unknown option: $1 (see coder-start -h)" ;;
  esac
done

if [[ -n "${CLI[CAPX_WEB_UI_PORT]:-}" ]]; then
  new_port="${CLI[CAPX_WEB_UI_PORT]}"
  [[ "${new_port}" =~ ^[0-9]+$ ]] && (( new_port > 0 && new_port < 65536 )) \
    || die "invalid port: ${new_port}"
  for entry in "${INTERNAL_PORTS[@]}" 8888:jupyter 6007:tensorboard 6008:safelog; do
    [[ "${new_port}" == "${entry%%:*}" ]] && die "port ${new_port} is reserved for ${entry#*:}"
  done
fi

save_to_env_file() {
  local key value tmp
  touch "${CLOUD_ENV_FILE}"
  chmod 600 "${CLOUD_ENV_FILE}"
  for key in "${!CLI[@]}"; do
    value="$(printf '%q' "${CLI[${key}]}")"
    tmp="$(mktemp)"
    grep -v "^${key}=" "${CLOUD_ENV_FILE}" > "${tmp}" || true
    printf '%s=%s\n' "${key}" "${value}" >> "${tmp}"
    cat "${tmp}" > "${CLOUD_ENV_FILE}"
    rm -f "${tmp}"
  done
  log "saved ${!CLI[*]} to ${CLOUD_ENV_FILE}"
}

if is_running; then
  if (( ${#CLI[@]} )); then
    warn "已在运行中，本次参数未生效；如需按新参数重启: coder-stop && coder-start ..."
  elif port_listening "${CAPX_WEB_UI_PORT}"; then
    log "已在运行中 (tmux session '${TMUX_SESSION}')，无需重复启动"
  else
    log "正在启动中，请稍候；查看进度: tail -f ${LOG_DIR}/latest.log"
  fi
  (( SAVE )) && save_to_env_file
  print_access_info
  exit 0
fi

for key in "${!CLI[@]}"; do
  printf -v "${key}" '%s' "${CLI[${key}]}"
done

CONFIG_PATH="${CAPX_CONFIG_PATH:-env_configs/cube_stack/franka_robosuite_cube_stack.yaml}"
SERVER_URL="${CAPX_SERVER_URL:-https://api.deepseek.com/v1/chat/completions}"
MODEL="${CAPX_MODEL:-deepseek-v4-pro}"
API_KEY="${CAPX_API_KEY:-${DEEPSEEK_API_KEY:-}}"
TEMPERATURE="${CAPX_TEMPERATURE:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"

[[ -x "${VENV_PY}" ]] || die "venv missing: run scripts/cloud/setup_env.sh first"
[[ -f "${CONFIG_PATH}" ]] || die "config not found: ${CONFIG_PATH}"
[[ -f web-ui/dist/index.html ]] || die "web-ui/dist missing: re-run sync_to_cloud.sh"
# launch.py rebuilds a stale dist with node, which the server does not have.
if [[ -n "$(find web-ui/src web-ui/package.json -newer web-ui/dist/index.html -type f -print -quit 2>/dev/null)" ]]; then
  die "web-ui/dist is older than web-ui/src: rebuild locally (scripts/cloud/sync_to_cloud.sh without SKIP_WEB_BUILD)"
fi
[[ -n "${API_KEY}" && "${API_KEY}" != "sk-your-deepseek-key" ]] \
  || die "API key not set: coder-start -k <key> [--save], or edit ${CLOUD_ENV_FILE}"
[[ -f "${SAM3_CHECKPOINT_PATH}" ]] || warn "SAM3 checkpoint missing: ${SAM3_CHECKPOINT_PATH}"
[[ -f "${CAPX_PANDA_URDF_PATH}" ]] || warn "panda URDF missing: ${CAPX_PANDA_URDF_PATH} (will git-clone at runtime)"

# Client rewrites openrouter/* models to a localhost proxy; call the API directly.
MODEL="${MODEL#openrouter/}"

port_listening "${CAPX_WEB_UI_PORT}" \
  && die "port ${CAPX_WEB_UI_PORT} is used by another process; pick another with -p, or run coder-stop"
for entry in "${INTERNAL_PORTS[@]}"; do
  port_listening "${entry%%:*}" \
    && die "internal port ${entry%%:*} (${entry#*:}) still in use by a leftover process; run coder-stop first"
done

(( SAVE )) && save_to_env_file

CMD=(
  "${VENV_PY}" capx/envs/launch.py
  --config-path "${CONFIG_PATH}"
  --server-url "${SERVER_URL}"
  --model "${MODEL}"
  --api-key "${API_KEY}"
  --web-ui True
  --web-ui-port "${CAPX_WEB_UI_PORT}"
)
[[ -n "${TEMPERATURE}" ]] && CMD+=(--temperature "${TEMPERATURE}")

log "model=${MODEL} server_url=${SERVER_URL}"
log "config=${CONFIG_PATH} port=${CAPX_WEB_UI_PORT}"

# The DeepSeek API is reached directly; AutoDL turbo proxy must not leak in.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if (( FOREGROUND )); then
  exec "${CMD[@]}"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/coder-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"
rm -f "${PID_FILE}"

# A running tmux server does not pass this shell's environment to new sessions,
# so the runner exports everything explicitly. `exec` keeps the PID in PID_FILE
# equal to launch.py; `env --default-signal=INT` makes sure coder-stop's SIGINT
# reaches Python as KeyboardInterrupt even if an ancestor ignored it; `tee -i`
# ignores SIGINT so the cleanup output is not cut off.
RUNNER="${LOG_DIR}/.coder-run.sh"
{
  echo '#!/usr/bin/env bash'
  for var in PATH LD_LIBRARY_PATH CUDA_HOME MUJOCO_GL PYTHONUNBUFFERED \
             SAM3_CHECKPOINT_PATH CAPX_SAM3_CHECKPOINT ROBOT_DESCRIPTIONS_CACHE CAPX_PANDA_URDF_PATH; do
    printf 'export %s=%q\n' "${var}" "${!var}"
  done
  echo 'unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY'
  printf 'cd %q\n' "${APP_DIR}"
  printf 'echo $$ > %q\n' "${PID_FILE}"
  printf 'exec > >(exec tee -ia %q) 2>&1\n' "${LOG_FILE}"
  printf 'exec env --default-signal=INT'
  printf ' %q' "${CMD[@]}"
  echo
} > "${RUNNER}"
chmod 700 "${RUNNER}"

{
  printf 'CAPX_WEB_UI_PORT=%q\n' "${CAPX_WEB_UI_PORT}"
  printf 'CAPX_PUBLIC_URL=%q\n' "${CAPX_PUBLIC_URL:-}"
  printf 'CODER_LOG_FILE=%q\n' "${LOG_FILE}"
} > "${STATE_FILE}"

tmux new-session -d -s "${TMUX_SESSION}" "bash $(printf '%q' "${RUNNER}")"
log "started tmux session '${TMUX_SESSION}', log: ${LOG_FILE}"
log "实时输出: tmux attach -t ${TMUX_SESSION}  (退出查看按 Ctrl-b 再按 d，不要按 Ctrl-C)"

log "waiting for Web UI on :${CAPX_WEB_UI_PORT} (SAM3 / GraspNet / PyRoKi load first, up to ${STARTUP_TIMEOUT}s)..."
for ((i = 0; i < STARTUP_TIMEOUT; i += 3)); do
  if port_listening "${CAPX_WEB_UI_PORT}"; then
    log "${C_RED}启动成功${C_RESET}。关闭 SSH 不影响服务；停止请执行 coder-stop"
    print_access_info
    exit 0
  fi
  if ! is_running; then
    tail -n 40 "${LOG_FILE}" >&2 || true
    die "process exited during startup, see ${LOG_FILE}"
  fi
  sleep 3
done
warn "Web UI not listening after ${STARTUP_TIMEOUT}s; still starting? check: tail -f ${LOG_DIR}/latest.log"
print_access_info
