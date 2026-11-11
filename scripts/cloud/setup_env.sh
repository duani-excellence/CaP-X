#!/usr/bin/env bash
# One-time (idempotent) environment setup on the AutoDL server.
# Steps: apt libs -> uv -> Python 3.10
# venv -> uv sync (robosuite + contactgraspnet) -> robot URDF -> self-check.
#
# Run on the server after scripts/cloud/sync_to_cloud.sh:
#   bash /root/autodl-tmp/capx/scripts/cloud/setup_env.sh
#
# Env overrides:
#   PYPI_MIRROR=https://mirrors.ustc.edu.cn/pypi     (default: tuna)
#   SKIP_APT=1 / SKIP_SYNC=1 / SKIP_CHECK=1
#   TORCH_CUDA_ARCH_LIST=8.9                         (RTX 4080 = sm_89)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "${APP_DIR}"

# Aliyun is ~150 KB/s over HTTP/1.1 (pip / uv) from AutoDL; tuna is ~12 MB/s.
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn}"
PYTHON_MIRROR="${PYTHON_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}"
UV_VERSION="${UV_VERSION:-0.11.32}"
GIT_MIRRORS="${GIT_MIRRORS:-/root/autodl-tmp/git-mirrors}"
EXTRAS=(--extra robosuite --extra contactgraspnet)

export UV_DEFAULT_INDEX="${PYPI_MIRROR}/simple"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_NO_GITHUB_FAST_PATH=1
export UV_COMPILE_BYTECODE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-32}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${LOG_DIR}"

# ----- 1) System libraries (EGL for MuJoCo offscreen, ninja for CUDA ext) -----
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  missing=()
  for pkg in libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ninja-build; do
    dpkg -s "${pkg}" >/dev/null 2>&1 || missing+=("${pkg}")
  done
  if (( ${#missing[@]} )); then
    log "apt install: ${missing[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  else
    log "system libraries already installed"
  fi
fi

[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found under CUDA_HOME=${CUDA_HOME}"
log "nvcc: $("${CUDA_HOME}/bin/nvcc" --version | tail -1)"

# ----- 2) uv -----
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" || "$("${UV_BIN}" --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  log "installing uv==${UV_VERSION} via ${PYPI_MIRROR}"
  /root/miniconda3/bin/pip install -q --disable-pip-version-check -i "${PYPI_MIRROR}/simple" "uv==${UV_VERSION}"
  UV_BIN=/root/miniconda3/bin/uv
fi
log "uv: $("${UV_BIN}" --version)"

# ----- 3) Python 3.10 -----
if ! "${UV_BIN}" python find 3.10 >/dev/null 2>&1; then
  log "installing Python 3.10 (mirror: ${PYTHON_MIRROR})"
  UV_PYTHON_INSTALL_MIRROR="${PYTHON_MIRROR}" "${UV_BIN}" python install 3.10 \
    || { warn "mirror failed, retrying from GitHub via AutoDL turbo"; with_turbo "${UV_BIN}" python install 3.10; }
fi

# ----- 4) Dependencies -----
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  # Path deps of extras we do not install (curobo / libero / verl) are not synced.
  bash "${CLOUD_SCRIPT_DIR}/stub_optional_path_deps.sh"

  # uv.lock pins files.pythonhosted.org (~20 KB/s from AutoDL). Mirrors keep the
  # same /packages/<hash>/ layout, so file hashes in the lock stay valid.
  sed -i \
    -e "s#https://files.pythonhosted.org/packages/#${PYPI_MIRROR}/packages/#g" \
    -e "s#https://pypi.org/simple#${PYPI_MIRROR}/simple#g" \
    uv.lock
  log "uv.lock rewritten to ${PYPI_MIRROR}"

  # Git deps come from local mirrors uploaded by sync_to_cloud.sh (GitHub is too slow).
  if [[ -f "${GIT_MIRRORS}/MAP" ]]; then
    # git refuses repos owned by another uid ("dubious ownership").
    chown -R "$(id -u):$(id -g)" "${GIT_MIRRORS}"
    while read -r url repo; do
      [[ -d "${GIT_MIRRORS}/${repo}" ]] || continue
      git config --global --unset-all "url.file://${GIT_MIRRORS}/${repo}.insteadof" 2>/dev/null || true
      git config --global --add "url.file://${GIT_MIRRORS}/${repo}.insteadOf" "${url}"
      log "git: ${url} -> ${GIT_MIRRORS}/${repo}"
    done < "${GIT_MIRRORS}/MAP"
  fi

  [[ -x "${VENV_PY}" ]] || "${UV_BIN}" venv -p 3.10 .venv
  log "uv sync --frozen ${EXTRAS[*]} (torch + CUDA wheels, several GB)..."
  "${UV_BIN}" sync --frozen "${EXTRAS[@]}" 2>&1 | tee "${LOG_DIR}/setup_uv_sync.log"
  "${UV_BIN}" pip install --python "${VENV_PY}" "robosuite-models==1.0.0"
fi

# ----- 5) Robot URDF (normally rsynced from the local cache) -----
if [[ ! -f "${CAPX_PANDA_URDF_PATH}" ]]; then
  log "panda URDF missing, cloning example-robot-data via turbo..."
  mkdir -p "${ROBOT_DESCRIPTIONS_CACHE}"
  with_turbo "${VENV_PY}" -c "from robot_descriptions import panda_description; print(panda_description.URDF_PATH)"
fi
[[ -f "${CAPX_PANDA_URDF_PATH}" ]] || die "panda URDF still missing: ${CAPX_PANDA_URDF_PATH}"
log "panda URDF: ${CAPX_PANDA_URDF_PATH}"

# ----- 6) Self-check -----
if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  [[ -f "${SAM3_CHECKPOINT_PATH}" ]] || die "SAM3 checkpoint missing: ${SAM3_CHECKPOINT_PATH}"
  [[ -f web-ui/dist/index.html ]] || die "web-ui/dist missing; run sync_to_cloud.sh (it builds the frontend locally)"
  "${VENV_PY}" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see CUDA"
print(f"[check] torch {torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")

import mujoco
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
ctx.free()
print(f"[check] mujoco {mujoco.__version__} EGL offscreen OK")

import robosuite, robosuite_models, sam3, contact_graspnet_pytorch  # noqa: F401
print("[check] robosuite / robosuite_models / sam3 / contact_graspnet_pytorch import OK")
EOF
  log "SAM3 checkpoint: ${SAM3_CHECKPOINT_PATH} ($(du -h "${SAM3_CHECKPOINT_PATH}" | awk '{print $1}'))"
fi

# ----- 7) One-click commands + login banner -----
bash "${CLOUD_SCRIPT_DIR}/install_commands.sh"

log "setup done. Next: coder-start -k <DeepSeek key> --save   (or edit ${CLOUD_ENV_FILE})"
