# Stack 分层反馈实验

沿用现有 P0 Stack 环境、Bubblewrap RPC、Vision API、DeepSeek Provider、多轮提示词和物理评分。默认推理入口不启用新监控；这个实验入口显式选择 A–E。仅支持模拟 Stack 的 Structured Vision，DeepSeek 接收文本/API 文档，不直接接收图像。

| 组 | 行为 |
|---|---|
| A | 原程序整块执行，无新监控 |
| B | +L0：几何有效性、另一种 RGB-D 表面估计的一致性检查；无效或显著不一致最多重新感知一次 |
| C | +L1：TCP 目标/实际位姿、关节与夹爪编码器；控制不收敛最多重试一次 |
| D | +L2：抓取/抬升/空夹监测，释放前 RGB-D 验证；最多两次有限局部对齐；仍不可靠则中断 |
| E | +L3：相同底层监控；结构化可观测反馈进入现有多轮生成，最多两个 DeepSeek 恢复代码块 |

每个 Seed 只生成一次真实 DeepSeek 初始程序，然后 A–E 配对执行同一份程序、Prompt/API 文档、初始 Snapshot 和成功判据。E 的恢复请求另计。这个设计估计的是固定初始程序条件下反馈机制的影响，不是五组独立采样生成代码的排行榜。

图驱动的 G0/G1 配对入口、离线机制验收和历史初始程序复跑见 [图驱动执行反馈说明](GRAPH.md)。

## 运行

使用正式项目现有 `.venv`。密钥仍由 `scripts/local/common.sh` 从本地环境文件加载，不放入命令行。下列服务地址是本轮探测确认仍在工作的存量服务；代码本身不写死这些地址，也不会启动或停止服务。18116 的 ASPIRE IK 接口虽兼容，但算法与原 CaP-X 不同，曾导致旧 Oracle 验收失败，因此不能直接用它复现本轮正式结果。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SAM3_SERVICE_URL=http://127.0.0.1:18114
export GRASPNET_SERVICE_URL=http://127.0.0.1:8215
# 使用已验收的 CaP-X 原求解器 CPU 实例；端口由 OS 分配。
export PYROKI_SERVICE_URL="$(.venv/bin/python -B -c 'import json; print(json.load(open("outputs/reproduction/layered-feedback-20261119/ik-runtime.json"))["url"])')"
```

本机驱动/NVML 版本暂不匹配。此次仿真实际采用 Mesa 软件渲染生成新 RGB-D，视觉模型通过已加载的服务在线推理，未使用冻结感知替代：

```bash
export JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MUJOCO_EGL_DEVICE_ID=1
export NUMBA_CACHE_DIR="$(mktemp -d /tmp/capx-feedback-numba.XXXXXX)"
```

历史场景 Smoke Test（会产生 API 费用）：

```bash
.venv/bin/python -B -m capx.experiments.layered_feedback \
  --output "outputs/reproduction/feedback-smoke-$(date +%Y%m%d_%H%M%S)" \
  --snapshot-manifest outputs/model_comparison/pilot_20261112_193853/manifest/scenes.json \
  --seeds 0 21 --max-requests 6 --cost-limit-usd 3
```

本轮五个独立评估场景的完整消融复跑（使用已验收的 Snapshot，每组先恢复）：

```bash
.venv/bin/python -B -m capx.experiments.layered_feedback \
  --output "outputs/reproduction/feedback-paired-$(date +%Y%m%d_%H%M%S)" \
  --snapshot-manifest outputs/reproduction/layered-feedback-20261119/scene-bank/paired-scenes.json \
  --seeds 1009 2027 4093 6151 8089 \
  --groups A B C D E --max-requests 15 --cost-limit-usd 5 \
  --max-tokens 32768 --reasoning-effort low --max-blocks 3 \
  --api-budget 64 --max-local 2 --wall-seconds 3600
```

另选未见场景时，省略 `--snapshot-manifest` 并提供新的 Seed，入口会通过既有流程生成、静置和验证快照；不要把复跑这五个 Seed 算成新增独立样本。

选组使用 `--groups A` 或 `--groups D E`。所有局部恢复同样扣除 episode 的 API 调用预算。每块原有 90 秒执行期限不变；LLM 单请求默认 400 秒，全局另有时间和请求/成本上限。不会自动重试 LLM。出现基础设施故障即停止批次。已有输出目录会被拒绝，不会覆盖。失败批次需保留并新建目录，不自动重复消费。

帮助、单测、离线汇总：

```bash
.venv/bin/python -B -m capx.experiments.layered_feedback --help
.venv/bin/python -B -m pytest -p no:cacheprovider tests/feedback -q
.venv/bin/python -B -m capx.experiments.feedback_analysis \
  --run "$RUN" --output "$RUN/analysis.json"
```

## 反馈边界与日志

`capx/execution/feedback.py` 不导入仿真器/评分器，只接收白名单 RGB、Depth、相机标定和机器人 TCP/关节/夹爪编码器。接触传感器当前不可用，明确记录为 unavailable。物体模拟真值、接触真值只写入独立的 `feedback_evaluation` 和原物理 Trace，供离线标注，不进入反馈消息。

释放前使用 Stack 红/绿颜色的可见点云表面，独立于 SAM3/OBB 算法，但依然共享相机标定、深度噪声和遮挡。它不是独立传感器，也不证明稳定接触。缺乏可见表面为 UNCERTAIN；PASS 只是允许执行的局部条件。阈值为通用的米制工程容差，不含 Seed 特例或位置补偿。颜色依赖和遮挡是当前适用范围限制。

`ExecutionInterrupted` 在父进程 RPC 边界终止本块，生成程序的 `try/except` 无法绕过。环境物理状态保留，下一块新建隔离 actor，模型被告知变量可能丢失。超时/服务错误仍按原 P0 重建逻辑处理。`FINISH` 只记录声明，绝不设置物理成功。

- `experiment.json`：预算、服务配置、Git HEAD 和关键源码 hash。
- `initial/`、`llm/`：实际初始代码来源、完整输入、content、usage、HTTP 状态和响应 ID；无鉴权头。
- `episodes/*/tools/`：SAM3/GraspNet/IK 实际输入输出；大数组放同名压缩 NPZ。
- `observations/`：用于监控的可观测数据。
- `trace/*/trace.jsonl`：原 API 前后物理状态、动作、检测、恢复及独立物理评价。
- `evaluation-geometry.json`：仅供离线评价的几何元数据。
- MP4 与 `.timestamps.json`：按真实仿真时间对齐的视频。
- `summary.json`、`analysis.json`：物理结果、技能指标、检测误报/漏报及调用成本。

成本上限按不含缓存优惠的峰时价格保守预留；有 usage 才能估算实际 Token 费用。超时且无 usage 的请求费用未知，不能记作免费。共享初始生成的账单只支付一次，但单组独立部署成本应包含一次初始生成，加该组自己的恢复请求。

`UNCERTAIN` 单独统计，不自动算检测正确。离线释放风险标签是几何诊断标签，不替代 P0 稳定物理评分。小样本结果不能证明总体成功率或可靠性。

## 原 IK CPU 实例

本轮通过 `scripts/local/start_feedback_ik.py` 调用项目原有求解器，不修改求解算法。这个独立服务不使用 GPU，默认向 OS 申请空闲端口，运行信息写入指定 JSON。若当前实例已退出，可在单独终端启动一个新实例（不会替换别的服务）：

```bash
source scripts/local/common.sh
IK_RUNTIME="outputs/reproduction/feedback-ik-$(date +%Y%m%d_%H%M%S).json"
IK_WARMUP="$(.venv/bin/python -B -c 'import json; print(json.load(open("outputs/reproduction/layered-feedback-20261119/ik-runtime.json"))["warmup_source"])')"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B scripts/local/start_feedback_ik.py \
  --runtime-file "$IK_RUNTIME" --warmup-requests "$IK_WARMUP"
```

随后将实验终端的 `PYROKI_SERVICE_URL` 设置为新 runtime JSON 中的 `url`。预热只求解保存的 IK 请求，不执行动作。该脚本的 `--port` 可显式指定；默认 0 不占用已监听端口。

## 本轮真实验收

正式报告：[分层反馈消融与逐例诊断](../reports/capx-layered-feedback-ablation-2026-11-19.md)。5 个独立场景 A/B 为 2/5，C/D 为 3/5，E 为 5/5；历史调试 0/21 的 E 都未恢复。L1 有直接收益，L2 既有改善也有退化，L3 用 3 次额外请求恢复了 2 个 episode。小样本不能证明总体可靠性。

原有 P0/Provider 68 项和新增反馈 28 项均通过。种子、共享初始程序、来源 hash、完整工具输入输出、视频、私有物理评价、Token 与费用分别保存于 `outputs/reproduction/layered-feedback-20261119/`。其中 `cf-seed1009-*` 为强制释放的诊断反事实，不能作为部署策略或并入 A–E 统计。
