#!/usr/bin/env bash
# Install the one-click commands coder-start / coder-stop / coder-status into
# /usr/local/bin and show a usage banner on interactive SSH login.
# Called by setup_env.sh; safe to re-run.
set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"

BIN_DIR="${BIN_DIR:-/usr/local/bin}"
BASHRC="${BASHRC:-/root/.bashrc}"
MARK_BEGIN="# >>> coder banner >>>"
MARK_END="# <<< coder banner <<<"

for name in start stop status; do
  target="${BIN_DIR}/coder-${name}"
  printf '#!/usr/bin/env bash\nexec bash %q "$@"\n' "${CLOUD_SCRIPT_DIR}/${name}.sh" > "${target}"
  chmod 755 "${target}"
  log "installed ${target}"
done

if [[ -f "${BASHRC}" ]]; then
  sed -i "/^${MARK_BEGIN}\$/,/^${MARK_END}\$/d" "${BASHRC}"
fi
cat >> "${BASHRC}" <<EOF
${MARK_BEGIN}
if [[ \$- == *i* && -z "\${TMUX:-}" && -f $(printf '%q' "${CLOUD_SCRIPT_DIR}/banner.sh") ]]; then
  bash $(printf '%q' "${CLOUD_SCRIPT_DIR}/banner.sh")
fi
${MARK_END}
EOF
log "login banner hooked into ${BASHRC}"
