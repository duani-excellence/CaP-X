#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
backend="${1:-ollama}"
case "$backend" in
  ollama)
    exec "$CAPX_ROOT/runtime/ollama/bin/ollama" serve
    ;;
  vllm)
    exec "$CAPX_ROOT/.venv-vllm/bin/vllm" serve \
      "$CAPX_ROOT/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
      --tokenizer "$CAPX_ROOT/models/qwen2.5-coder-7b/tokenizer" \
      --hf-config-path "$CAPX_ROOT/models/qwen2.5-coder-7b/tokenizer" \
      --served-model-name qwen2.5-coder:7b --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.32 \
      --kv-cache-memory-bytes 805306368 --max-num-batched-tokens 512 \
      --enforce-eager
    ;;
  openrouter)
    if [[ ! -f .openrouterkey ]] || ! grep -qE '^[[:space:]]*[^#[:space:]]' .openrouterkey; then
      echo '缺少 OpenRouter API key：请写入项目根目录 .openrouterkey（每行一个）。' >&2
      exit 2
    fi
    exec .venv/bin/python -m capx.serving.openrouter_server \
      --key-file .openrouterkey --host 127.0.0.1 --port 8110
    ;;
  *) echo '用法：scripts/local/start_llm.sh ollama|vllm|openrouter' >&2; exit 2 ;;
esac
