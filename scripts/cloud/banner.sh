#!/usr/bin/env bash
# Login banner shown to whoever SSHs into the server (hooked into ~/.bashrc).
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"

if is_running && port_listening "${CAPX_WEB_UI_PORT}"; then
  state="运行中"
elif is_running; then
  state="启动中"
else
  state="未启动"
fi

cat <<EOF

=================== Efort_AICoder 交互界面 ===================
  当前状态:  ${state}

  一键启动:  coder-start      (首次加载模型约 1-3 分钟，关闭 SSH 不影响服务)
             coder-start -k <API key> -p <端口> --save   (可选参数，详见 coder-start -h)
  一键停止:  coder-stop       (停止所有服务并释放 GPU)
  查看状态:  coder-status
  使用说明:  cat ${APP_DIR}/README.AICODER.md
EOF
print_access_info
echo "================================================================"
echo
