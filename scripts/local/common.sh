#!/usr/bin/env bash
set -euo pipefail
CAPX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CAPX_ROOT
cd "$CAPX_ROOT"
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi
export PYTHONPATH="$CAPX_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export SAM3_CHECKPOINT_PATH="$CAPX_ROOT/models/sam3/sam3.pt"
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-/home/hanyu/.cache/robot_descriptions}"
export CAPX_PANDA_URDF_PATH="${CAPX_PANDA_URDF_PATH:-$ROBOT_DESCRIPTIONS_CACHE/example-robot-data/robots/panda_description/urdf/panda.urdf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=4
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export OLLAMA_MODELS="$CAPX_ROOT/models/ollama"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_PARALLEL=1
mkdir -p logs outputs/reproduction
