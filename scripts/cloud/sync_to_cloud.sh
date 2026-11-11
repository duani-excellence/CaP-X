#!/usr/bin/env bash
# Run LOCALLY: build the frontend and rsync the repo, SAM3 weights and the Panda
# URDF cache to the AutoDL server. Incremental — re-run after code changes.
#
# Usage:
#   ./scripts/cloud/sync_to_cloud.sh
#   CLOUD_HOST=root@connect.nmb2.seetacloud.com CLOUD_PORT=25446 ./scripts/cloud/sync_to_cloud.sh
#   SKIP_WEB_BUILD=1 SKIP_MODELS=1 ./scripts/cloud/sync_to_cloud.sh   # code only
#
# Tip: run `ssh-copy-id -p <port> <host>` once to avoid password prompts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CLOUD_HOST="${CLOUD_HOST:-root@connect.nmb2.seetacloud.com}"
CLOUD_PORT="${CLOUD_PORT:-25446}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/capx}"
REMOTE_GIT_MIRRORS="${REMOTE_GIT_MIRRORS:-/root/autodl-tmp/git-mirrors}"
SAM3_SRC="${SAM3_SRC:-${HOME}/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt}"
URDF_CACHE_SRC="${URDF_CACHE_SRC:-${HOME}/.cache/robot_descriptions/example-robot-data}"

log() { printf '[sync] %s\n' "$*"; }
die() { printf '[sync] ERROR: %s\n' "$*" >&2; exit 1; }

# One multiplexed connection: the password (if any) is asked only once.
SSH_CTL="/tmp/capx-ssh-${USER:-me}-%r@%h:%p"
SSH=(ssh -p "${CLOUD_PORT}" -o ControlMaster=auto -o ControlPath="${SSH_CTL}" -o ControlPersist=10m)
RSYNC=(rsync -az --no-owner --no-group --info=progress2 --human-readable -e "${SSH[*]}")

# ----- 1) Frontend (built here so the server needs no node) -----
if [[ "${SKIP_WEB_BUILD:-0}" != "1" ]]; then
  log "building web-ui/dist"
  (cd web-ui && { [[ -d node_modules ]] || npm ci; } && npm run build)
fi
[[ -f web-ui/dist/index.html ]] || die "web-ui/dist/index.html missing"

"${SSH[@]}" "${CLOUD_HOST}" "mkdir -p '${REMOTE_DIR}/models' '${REMOTE_DIR}/robot_descriptions'"

# ----- 2) Code -----
# Unused third_party trees are stubbed on the server (scripts/cloud/stub_optional_path_deps.sh);
# excluded paths are also protected from --delete, so the stubs survive re-syncs.
log "syncing code -> ${CLOUD_HOST}:${REMOTE_DIR}"
"${RSYNC[@]}" --delete \
  --exclude='.git' \
  --exclude='.venv' --exclude='.venv-*' \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
  --exclude='.cursor' --exclude='agent-transcripts' \
  --exclude='/outputs' --exclude='/logs' --exclude='/models' --exclude='/robot_descriptions' \
  --exclude='/*.jpg' --exclude='/*.png' --exclude='*.mp4' \
  --exclude='/.env' --exclude='/.openrouterkey' --exclude='*.tar.gz' \
  --exclude='/web-ui/node_modules' \
  --exclude='/capx/third_party/b1k' \
  --exclude='/capx/third_party/LIBERO-PRO' \
  --exclude='/capx/third_party/libero_dependencies' \
  --exclude='/capx/third_party/verl' \
  --exclude='/capx/third_party/curobo' \
  --exclude='/scripts/cloud/cloud.env' \
  ./ "${CLOUD_HOST}:${REMOTE_DIR}/"

# ----- 3) Secrets (only if present locally; never deleted remotely) -----
if [[ -f "${SCRIPT_DIR}/cloud.env" ]]; then
  log "uploading scripts/cloud/cloud.env"
  "${RSYNC[@]}" --chmod=F600 "${SCRIPT_DIR}/cloud.env" "${CLOUD_HOST}:${REMOTE_DIR}/scripts/cloud/cloud.env"
fi

# ----- 4) Models / URDF cache (resumable, skipped when unchanged) -----
if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
  [[ -f "${SAM3_SRC}" ]] || die "SAM3 checkpoint not found: ${SAM3_SRC} (set SAM3_SRC=...)"
  log "syncing SAM3 checkpoint ($(du -h "${SAM3_SRC}" | awk '{print $1}'))"
  "${RSYNC[@]}" --partial "${SAM3_SRC}" "${CLOUD_HOST}:${REMOTE_DIR}/models/sam3.pt"

  if [[ -d "${URDF_CACHE_SRC}" ]]; then
    log "syncing robot_descriptions/example-robot-data"
    "${RSYNC[@]}" "${URDF_CACHE_SRC}" "${CLOUD_HOST}:${REMOTE_DIR}/robot_descriptions/"
  else
    log "local URDF cache not found (${URDF_CACHE_SRC}); setup_env.sh will clone it"
  fi

  # Git deps in uv.lock (pyroki, jaxls) clone at ~25 KB/s from AutoDL and the TLS
  # connection often drops. Build bare mirrors (HEAD = locked commit) from the
  # local uv git cache; setup_env.sh redirects the GitHub URLs via insteadOf.
  LOCAL_UV_GIT_DB="${LOCAL_UV_GIT_DB:-${HOME}/.cache/uv/git-v0/db}"
  MIRROR_STAGE="$(mktemp -d)"
  trap 'rm -rf "${MIRROR_STAGE}"' EXIT
  : > "${MIRROR_STAGE}/MAP"
  while IFS='#' read -r url sha; do
    name="$(basename "${url}" .git)"
    src=""
    for db in "${LOCAL_UV_GIT_DB}"/*; do
      git -C "${db}" cat-file -e "${sha}^{commit}" 2>/dev/null && { src="${db}"; break; }
    done
    if [[ -z "${src}" ]]; then
      log "WARNING: ${name}@${sha:0:7} not in local uv cache; server will clone from GitHub"
      continue
    fi
    git init -q --bare "${MIRROR_STAGE}/${name}.git"
    git -C "${src}" push -q "${MIRROR_STAGE}/${name}.git" "${sha}:refs/heads/main" "${sha}:refs/commit/${sha}"
    git -C "${MIRROR_STAGE}/${name}.git" symbolic-ref HEAD refs/heads/main
    printf '%s %s.git\n' "${url}" "${name}" >> "${MIRROR_STAGE}/MAP"
    log "git mirror: ${name}@${sha:0:7}"
  done < <(grep -oE 'source = \{ git = "[^"]+"' uv.lock | sed -E 's/.*git = "([^"]+)"/\1/' | sort -u)
  if [[ -s "${MIRROR_STAGE}/MAP" ]]; then
    "${RSYNC[@]}" --delete "${MIRROR_STAGE}/" "${CLOUD_HOST}:${REMOTE_GIT_MIRRORS}/"
  fi
fi

log "done. On the server:"
log "  bash ${REMOTE_DIR}/scripts/cloud/setup_env.sh   # first time / after dependency changes"
log "  coder-start / coder-stop / coder-status"
