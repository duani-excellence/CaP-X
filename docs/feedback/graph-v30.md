# V30：阶段恢复与外部方法适配

正式项目仍为 `/home/hanyu/dqb/cap-x-20261106`。本版沿用 V29 Runner、父进程 Broker、Sandbox、状态记忆和原生物理评分。所有新增技能均属于共同底座，G0/G1 及 LOGIV 适配使用相同能力。

## 当前实现与边界

- `skill_recovery_version=30`：空手接近失败后的受限撤退、排除失败候选、重新抓取；当前表面配准提出附着假设，再用实际小幅运动与夹爪稳定性验证。失败回滚假设，不能释放。
- 每个对象、每种恢复阶段最多两次；原 episode/API/Skill/LLM 预算继续有效。不会重放已完成调用。
- V29 条件回归现在可发现新技能的补证需求，生成“物理恢复 → 重新取得工具证据 → 继续”片段。预测效果不是动作许可，派发仍重新检查。
- 原深度 GraspNet 无候选时，使用已有服务的场景＋对象点云请求作为有限 Fallback，要求 Seed 回显。候选仍走原姿态投影、碰撞和 IK 检查；未宣称任意 6D 抓取能力。
- cuRobo 实际规划探测失败（CUDA 804）。本版没有把 PyRoKi 冒充碰撞规划器，也没有修改驱动或重启服务。MTC 只借鉴分阶段候选/Fallback 设计，没有运行 ROS/MTC 引擎。

外部方法比较是后续主实验，G0/G1 是内部消融。本版可运行的是 **LOGIV 官方核心的共同 Skill 适配**；CaP-Agent0、ASPIRE 原版尚未完成公平端到端复现。不能把普通多轮 DeepSeek 标成 CaP-Agent0，也不能把复用 ASPIRE 感知协议标成 ASPIRE 完整算法。

## 小规模真实 DeepSeek 配对

先复用已有服务配置和 IK runtime，不启动新模型服务。以下路径指向本次保留的固定上游源码，换机器时可克隆同一 revision；不要复用旧输出目录。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LOGIV_SRC="$PWD/outputs/model_comparison/graph_v30_20261124_185019/upstream/LOGIV-official"
IK_RUNTIME="$PWD/outputs/model_comparison/graph_v28_20261124_155408/ik-runtime.json"
RUN="$PWD/outputs/model_comparison/v30_external_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -B -m capx.experiments.contract_models \
  --config env_configs/graph_feedback/development-v30.json \
  --ik-runtime "$IK_RUNTIME" --external-logiv "$LOGIV_SRC" \
  --seed 0 --v29-skills-control \
  --ablations no_causal no_local_repair no_cost --output "$RUN"
.venv/bin/python -B scripts/local/analyze_contract_models.py "$RUN" \
  --output "$RUN/analysis.json"
```

此命令会收费，串行运行 LOGIV 适配、G0、G1、三个消融和旧 Skill 对照。每组共享本次新生成的 DeepSeek 初始程序及 Snapshot。实际恢复能力/验证器相同，策略与消融开关不同；未满足条件时允许有界失败。默认最多 24 个新请求、估算 USD 5、每个 episode 1200 秒、24 个 Skill、256 次底层 API；输出含每个请求的实际 usage 和估算费用，不含密钥。

`V29-Skills-G1` 表示当前共同调度器＋原 V29 Skill，不是整个历史 V29 checkout。LOGIV 适配保留官方 `symbolic/graph/repair/execution/verification`，用公共 Skill 验证器、有限能力搜索、符号验证替换原 GPT-4o/VLA、FastDownward 和 VAL；每次运行校验官方源码与 Git revision。训练成本为零不代表原论文的训练/技能发现成本为零。

LOGIV 的物理谓词与瞬态工具条件分开映射；瞬态条件仍由同一 Broker 检查。恢复库特有的物理条件通过官方 `tracked_atoms` 更新，避免恢复规划读取旧阶段状态。不能把缺少当前证据转换成闭世界 FALSE 来方便规划。

当前验收：Graph/P0 合计 1239 项通过；最终 Seed 0 配对三组均为 1/4 子目标、完整失败。详细失败、消融和成本见报告；不要直接扩大为正式评测。

## 历史程序＋实时 Vision（不新增 LLM）

```bash
.venv/bin/python -B scripts/local/replay_skill_recovery.py \
  --config env_configs/graph_feedback/development-v30.json \
  --reference outputs/model_comparison/graph_v29_20261124_173441/final-normal0 \
  --ik-runtime "$IK_RUNTIME" --group G1 \
  --output "$PWD/outputs/model_comparison/v30_replay_$(date +%Y%m%d_%H%M%S)"
```

可添加 `--external-logiv "$LOGIV_SRC"` 检查外部执行器适配。它是历史程序重放，不计入新的 DeepSeek 端到端样本。

## 公开 LIBERO-Recovery 场景与协议

官方源码 revision：`832d54dfc43f0c9e4240bb8d92f5fda7001bb093`。本次只下载固定抽样的两个 LIBERO-Long 场景（BDDL、物理状态），没有下载模型。保存于本次输出的 `recovery-scenes/`；选择清单在环境验收前写入。

```bash
ART="$PWD/outputs/model_comparison/graph_v30_20261124_185019"
# 使用已有隔离 LIBERO 环境；不在主 .venv 安装 LIBERO 依赖。
LIBERO_PY=/home/hanyu/dqb/ASPIRE/.venv-libero/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export MUJOCO_EGL_DEVICE_ID=1  # 本机软件 EGL 的实测编号，其他机器需检查
"$LIBERO_PY" -B scripts/local/libero_recovery_gate.py \
  --manifest "$ART/recovery-scenes/manifest.json" \
  --protocol "$ART/upstream/recovery-protocol/examples/LIBERO/eval_files/eval_libero_custom_scene.py" \
  --libero-config /home/hanyu/dqb/ASPIRE/.libero \
  --output "$PWD/outputs/model_comparison/libero_recovery_gate_$(date +%Y%m%d_%H%M%S)"
```

这里只验收状态恢复、官方扰动/夹爪初始化以及正常 Long-Swap 环境重置，**不执行策略、不报告恢复成功率**。本机状态格式与官方扰动函数存在经算术反例确认的索引偏移，详见报告；未悄悄修正官方协议。原版协议、修正版协议以及 LIBERO-PRO 正常长程结果必须分别记录，不能混合统计。

结果、失败原因和是否可扩大实验见 [V30 开发报告](../reports/graph-feedback-v30-development-2026-11-24.md)。
