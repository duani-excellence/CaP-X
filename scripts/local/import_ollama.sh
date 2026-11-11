#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
model="$CAPX_ROOT/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
test -s "$model"
cat > "$CAPX_ROOT/models/qwen2.5-coder-7b/Modelfile" <<EOF
FROM $model
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
EOF
exec "$CAPX_ROOT/runtime/ollama/bin/ollama" create qwen2.5-coder:7b \
  -f "$CAPX_ROOT/models/qwen2.5-coder-7b/Modelfile"
