# CaP-X 常用指令速查

图驱动执行反馈的条件图、局部恢复与当前复跑命令见 [Graph V22 说明](docs/feedback/graph-v22.md)，验收范围见 [V22 开发报告](docs/reports/graph-feedback-v22-development-2026-11-22.md)。原路径和历史结果保留在 [Graph V21 说明](docs/feedback/graph-v2.md#v21-动作准入与有界恢复)、[V21.3 验收报告](docs/reports/graph-feedback-v21-online-resume-2026-11-22.md)和 [早期 V21 报告](docs/reports/graph-feedback-v21-2026-11-22.md)。

Stack 的 DeepSeek + Vision 分层反馈与 A–E 配对实验见 [分层反馈 README](docs/feedback/README.md)。

当前多任务推理、Teacher 采集、SFT 导出与 Web 使用请先看 [多任务 README](docs/multitask/README.md)。现有主环境已配置，本轮无需按下方历史安装步骤重新安装。

面向本地快速上手：安装、LLM、感知服务、Demo / Web UI、排障。更完整的说明见 `docs/`。

---

## 0. 环境要求

- Python **3.10**
- CUDA GPU（Robosuite demo 推荐 ≥16GB 显存；SAM3 + GraspNet + 大模型同卡时建议 ≥24GB）
- 包管理：[uv](https://docs.astral.sh/uv/)

---

## 1. 安装

```bash
# 克隆（含 submodule）
git clone --recurse-submodules https://github.com/capgym/cap-x && cd cap-x
# 若已克隆但未拉 submodule：
git submodule update --init --recursive

# 安装 uv（若尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 venv + 基础依赖
uv python install 3.10 && uv venv -p 3.10
uv sync

# Robosuite 任务（叠方块等）必装
uv sync --extra robosuite

# 感知相关（非 privileged 配置通常需要）
uv sync --extra contactgraspnet
```

激活环境：

```bash
source .venv/bin/activate
```

可选 extras：

```bash
uv sync --extra curobo    # GPU 运动规划
uv sync --extra verl      # RL / GRPO 训练
```

> Robosuite 与 LIBERO 依赖冲突，需分环境；LIBERO / BEHAVIOR 见官方 README。

---

## 2. LLM 后端

评估通过 OpenAI 兼容的 chat completions 接口调模型。

### 方案 A：Ollama（本地，推荐先跑通）

```bash
# 安装并拉取模型（示例）
ollama pull qwen2.5-coder:7b
# 更大更强但更吃显存：
# ollama pull qwen3-coder:30b

# 确认服务（默认 11434）
curl -s http://127.0.0.1:11434/api/tags | head
```

启动 demo 时把 endpoint 指到 Ollama：

```bash
--server-url http://127.0.0.1:11434/v1/chat/completions \
--model qwen2.5-coder:7b
```

### 方案 B：DeepSeek 官方 API

与项目根目录 `.env` Mode B 相同：

```bash
# 推荐: set -a && source .env && set +a
export CAPX_SERVER_URL=https://api.deepseek.com/v1/chat/completions
export CAPX_MODEL=deepseek-v4-pro
export CAPX_API_KEY=sk-your-deepseek-key
```

启动时传：

```bash
--server-url "$CAPX_SERVER_URL" \
--model "$CAPX_MODEL" \
--api-key "$CAPX_API_KEY"
```

模型名用官方 id（`deepseek-v4-pro`），不要加 `deepseek/` 或 `openrouter/` 前缀。

### 方案 C：OpenRouter 代理

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py \
  --key-file .openrouterkey --port 8110
```

默认 URL：`http://127.0.0.1:8110/chat/completions`。

---

## 3. 感知服务（SAM3 / GraspNet）

多数 YAML 里的 `api_servers` 会在 `launch.py` 启动时**自动拉起**；也可预启动复用：

```bash
# 预览 GPU 分配
uv run --no-sync --active capx/serving/launch_servers.py --profile default --dry-run

# 实际启动（SAM3 + GraspNet + PyRoKi）
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

| Profile   | 内容 |
|-----------|------|
| `default` | SAM3 + GraspNet + PyRoKi |
| `full`    | 再加 OWL-ViT、SAM2 等 |
| `minimal` | 仅 PyRoKi（privileged / oracle） |

**SAM3 权重：** HuggingFace 需申请权限；也可用本机缓存。本仓库支持：

```bash
# 环境变量指定本地 checkpoint（优先于 HF 下载）
export SAM3_CHECKPOINT_PATH=~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
# 或
export CAPX_SAM3_CHECKPOINT=/path/to/sam3.pt
```

默认还会尝试 ModelScope / HF 本地缓存路径。

**显存提示：** 同卡同时跑「大 LLM + SAM3 + GraspNet」易 OOM。30B 级模型常占 ~20GB，再开感知服务容易在 `http://127.0.0.1:8115/plan` 报 500。可换 7B、或用 privileged 配置、或分卡。

---

## 4. Demo 运行（Robosuite 叠方块 cube_stack）

在项目根目录、已 `source .venv/bin/activate` 下执行。`--no-sync` 避免每次重新解析依赖。

### 4.1 交互 Web UI（推荐入门）

先 `cp .env.example .env`，编辑 LLM（Ollama 或 DeepSeek），再：

```bash
set -a && source .env && set +a
uv run --no-sync --active capx/envs/launch.py \
  --config-path "$CAPX_CONFIG_PATH" \
  --server-url "$CAPX_SERVER_URL" \
  --model "$CAPX_MODEL" \
  --api-key "$CAPX_API_KEY" \
  --web-ui True
```

`.env` 默认可切 DeepSeek；切 Ollama 时用 `http://127.0.0.1:11434/v1/chat/completions` 与 `qwen2.5-coder:7b`（见 `.env.example`）。

浏览器打开：**http://localhost:8200**（不要用 `0.0.0.0`）。

- CLI 带了 `--config-path` 时，页面会 **auto_start**，无需再点 Start。
- 前端若改了 `web-ui/src`，`launch.py` 会在 `dist/` 过期时自动 `npm run build`。

多轮 + Web UI：

```bash
set -a && source .env && set +a
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml \
  --server-url "$CAPX_SERVER_URL" \
  --model "$CAPX_MODEL" \
  --api-key "${CAPX_API_KEY:-}" \
  --web-ui True
```

### 4.2 无头 CLI（批量评估）

```bash
# 单轮
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml \
  --server-url http://127.0.0.1:11434/v1/chat/completions \
  --model qwen2.5-coder:7b

# 多轮 + 视觉差分（VDM）
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
  --server-url http://127.0.0.1:11434/v1/chat/completions \
  --model qwen2.5-coder:7b
```

### 4.3 Privileged（跳过 SAM3 / GraspNet，省显存、验流程）

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
  --server-url http://127.0.0.1:11434/v1/chat/completions \
  --model qwen2.5-coder:7b \
  --web-ui True
```

### 4.4 常用 CLI 参数

| 参数 | 含义 |
|------|------|
| `--config-path` | YAML 任务配置 |
| `--server-url` | LLM chat completions URL |
| `--model` | 模型名 |
| `--temperature` | 采样温度（默认 1.0） |
| `--web-ui True` | 开交互页面（8200） |
| `--web-ui-port` | 改 Web UI 端口 |
| `--total-trials` | 覆盖 YAML 里的 trial 数 |
| `--debug True` | 打印更全的模型响应 |

---

## 5. 常用配置速查

| 场景 | 配置路径 |
|------|----------|
| 单轮叠方块（默认） | `env_configs/cube_stack/franka_robosuite_cube_stack.yaml` |
| 多轮叠方块 | `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml` |
| 单轮擦拭 | `env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml` |
| 多轮擦拭 | `env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn.yaml` |
| Privileged 擦拭 | `env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml` |

结果默认在 `outputs/` 下（代码、视频、日志）。

---

## 6. 回归 / 冒烟

```bash
./scripts/regression_test.sh quick    # ~10 trial 冒烟
./scripts/regression_test.sh test1    # 完整单轮
```

---

## 7. 前端开发（可选）

生产模式：`launch.py --web-ui True` 已托管 `web-ui/dist`。

热更新开发：

```bash
# 终端 1：后端
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --web-ui True

# 终端 2：Vite（代理 /api、/ws → 8200）
cd web-ui && npm install && npm run dev
# 打开 http://localhost:5173
```

---

## 8. 排障速查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| SAM3 HuggingFace 401 | 未授权 / 无本地权重 | 申请 HF 权限，或设 `SAM3_CHECKPOINT_PATH` / ModelScope 缓存 |
| `8115/plan` 500 / GraspNet RuntimeError | 多为 **CUDA OOM** | 换小模型、停 Ollama 大模型、privileged、或分 GPU |
| `green_extent` / `NoneType` 下标 | LLM 生成代码漏 `return_bbox_extent=True` | 换更强模型，或多轮 Pause 反馈 |
| Web UI 打不开 / 502 | 用了 `0.0.0.0:8200` | 改用 `http://localhost:8200` |
| `uv sync` 长时间无输出 | 在下大包（PyTorch 等） | 正常，等即可 |

查 GPU：

```bash
nvidia-smi
```

停 Ollama 某模型释放显存：

```bash
ollama stop qwen3-coder:30b
```

---

## 9. 最小可复制 Demo（复制即用）

```bash
cd /path/to/cap-x
source .venv/bin/activate

# 终端：确保 Ollama 已拉取 qwen2.5-coder:7b

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml \
  --server-url http://127.0.0.1:11434/v1/chat/completions \
  --model qwen2.5-coder:7b \
  --web-ui True

# 浏览器: http://localhost:8200
```

显存紧张时，把 `franka_robosuite_spill_wipe.yaml` 换成 `franka_robosuite_spill_wipe_privileged.yaml`。
