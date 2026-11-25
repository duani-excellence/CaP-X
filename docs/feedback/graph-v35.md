# V35：预规划与决策状态（开发版本）

沿用正式工程、当前分支、统一 Runner、Sandbox、Broker 和原生评分。没有新的 Agent 或独立机器人执行循环。

本版本已接入候选预搜索、IK 残差查询、精确缓存、抓取后附着重验证及目标相关状态编码。**尚未打通倾斜持物轨迹的执行许可，不是四物体 PickPlace 已成功的版本。** PyRoKi 轨迹候选不等于碰撞自由证明；cuRobo 未安装且当前 CUDA 不可用。完整运动底座统一仍未完成。

## 当前可用命令

在工程根目录加载现有配置；不要覆盖 `.env.local` / `.env.api.local`。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
V35_RUN="$PWD/outputs/model_comparison/v35_diagnosis_$(date +%Y%m%d_%H%M%S)"
IK_RUNTIME="$(.venv/bin/python -c 'import json; print(json.load(open("env_configs/inference/local-v31.json"))["ik_runtime"])')"
HISTORY="$PWD/outputs/model_comparison/graph_v34_20261125_153341/open-g0-protocol-final/seed-0-repeat-0/G0"

# 冻结公开几何 + 当前 IK 服务，无 LLM、无机器人运动。
.venv/bin/python -B scripts/local/diagnose_preplanning.py \
  --episode "$HISTORY" --object Can --target 3 \
  --ik-runtime "$IK_RUNTIME" --output "$V35_RUN/placement"

# 扩大姿态诊断，并实际调用已有 PyRoKi trajopt，仍不派发运动。
JAX_PLATFORMS=cpu .venv/bin/python -B scripts/local/diagnose_pose_search.py \
  --diagnosis "$V35_RUN/placement/diagnosis.json" \
  --ik-runtime "$IK_RUNTIME" --output "$V35_RUN/pose-search" \
  --tilts 0 20 40 --yaw-step 45 --max-solver-calls 512 --mature

# 原公开 RGB-D 检查机器人与持物的扫掠采样。
.venv/bin/python -B scripts/local/check_public_trajectory.py \
  --diagnosis "$V35_RUN/placement/diagnosis.json" \
  --pose-result "$V35_RUN/pose-search/results.json" \
  --output "$V35_RUN/path-check.json"
```

这些命令的姿态搜索是明确标记的诊断，不能把 40° 倾斜 IK 候选传给原直立释放合同。无可行候选时不会发起成熟轨迹探针。历史帧不冒充当前在线视觉。

真实 Vision 历史程序重放，零新增 LLM：

```bash
scripts/local/infer.sh replay \
  --harness open --skill v32 --force-mode on --methods G0 \
  --config env_configs/graph_feedback/development-v35.json \
  --decision-encoding encoded --reference "$(dirname "$HISTORY")" \
  --snapshot-group G0 --recovery-requests 0 \
  --output "$V35_RUN/live-replay"
```

`development-v35.json` 使用当前实测 Mesa EGL 设备 0；V34 配置设备 1 保留。不同主机先核对设备；这不是物理算法参数。

未来状态编码对照的同一入口（当前仅验证参数，不建议解除 dry-run）：

```bash
for ENCODING in legacy encoded; do
  scripts/local/infer.sh replay \
    --harness open --skill v32 --force-mode on --methods G0 \
    --config env_configs/graph_feedback/development-v35.json \
    --decision-encoding "$ENCODING" --reference "$(dirname "$HISTORY")" \
    --snapshot-group G0 --recovery-requests 2 --max-requests 2 \
    --output "$V35_RUN/context-$ENCODING" --dry-run
done
```

移除 `--dry-run` 才会产生真实 DeepSeek 恢复费用。两组共享初始历史程序、Snapshot、API 文档、技能、感知与预算，仅反馈表示不同。本轮由于安全路径关未通过，**没有执行该付费对照**。

## 新工具与返回语义

- `preview_skill(skill, params={...})`：真实候选/IK 搜索，可采集公开观测，不执行任何物理操作。返回紧凑结果、候选状态、条件性后续放置可行性及查询 ID。
- `planning_query(id=None)`：查询完整搜索、IK 位置/旋转残差、参考配置、迭代、关节限制和失败类型。搜索 ID 不能传给 `run_plan` 或当观测 ID。
- `decision_state()`：目标相关物体估计、名义物体/TCP 变换、带来源的空间关系、未满足条件、已测候选、剩余搜索范围及可选工具。
- `plan_skill()`：V35 下会实际预搜索 acquire/deposit，再登记可执行合同。`run_plan()` 仍通过原父进程检查，物理变化会使旧计划失效；释放仍由最终当前观测决定。

`NO_SOLUTION_FOUND` 只代表有限求解未成功，不代表物理不可达。`KINEMATIC_CANDIDATE` 也不是运动许可。未来放置检查基于明确标记的假设附着；抓取后必须重新注册、重新搜索。其他恢复 Skill 仍可能只返回 `CAPABILITY_CONTRACT_ONLY`，不伪造其已经做完轨迹搜索。

默认只返回摘要，详情保留在查询/日志中，避免打印完整候选表耗尽 Sandbox stdout 预算。缓存命中单独记本次耗时与原始计算耗时。目标变化、参考配置变化、场景/附着变化不复用不适用的缓存。
