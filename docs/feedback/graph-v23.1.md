# Graph V23.1：未来证据与有界补证

这是开发版本，尚未通过真实遮挡预测释放及 Restack 完整闭环验收。保留 V23 配置与历史失败。默认实验配置：`env_configs/graph_feedback/development-v23.1.json`。不启动正式批量实验。

## 运行

以下命令在正式仓库执行，所有输出必须使用新目录。已有配置负责服务地址、模型预算、软件 EGL 和 CPU IK；不需要重启服务或升级依赖。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN="$PWD/outputs/model_comparison/v231_replay_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -B scripts/local/replay_graph_initial.py \
  --config env_configs/graph_feedback/development-v23.1.json \
  --source-unit outputs/model_comparison/graph_v22_20261122_180352/online-v22-2/normal-restack \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/restack-1482414119/prepared-snapshot.json \
  --bind-current-contract --output "$RUN"
```

这是**历史真实 DeepSeek 程序 + 当前真实 Vision + 有界本地恢复**，新 LLM 请求为零；不属于新模型端到端成功。超出本地恢复能力时，NoModel 阻止新请求并退出。日志中的历史 token/生成时间不应再次算入本轮费用与墙钟。以 `total_seconds` 和新的请求计数记账。

正常 Stack 复跑用同一入口，替换两项参数：

```bash
--source-unit outputs/model_comparison/graph_v22_20261122_180352/online-v22-1/normal-stack
--snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-1257104511/prepared-snapshot.json
```

冻结观测误差分析与传播重放：

```bash
.venv/bin/python -B scripts/local/diagnose_evidence.py \
  --frozen-rows outputs/model_comparison/graph_v23_20261122_214021/frozen-restack-memory-final/rows.json \
  --profile env_configs/graph_feedback/calibration/memory-development-v23.json \
  --table-episode outputs/model_comparison/graph_v23_20261122_214021/replay-restack-2/episodes/G1 \
  --output "$RUN-frozen-diagnosis"
.venv/bin/python -B scripts/local/replay_state_memory.py \
  --episode outputs/model_comparison/graph_v22_20261122_180352/online-v22-2/normal-restack/episodes/G1 \
  --profile env_configs/graph_feedback/calibration/memory-development-v23.json \
  --output "$RUN-frozen-propagation"
.venv/bin/python -B scripts/local/analyze_relative_memory.py \
  --profile env_configs/graph_feedback/calibration/memory-development-v23.json \
  --output "$RUN-relative-diagnostic"
```

这些入口没有模型客户端，不执行机器人动作。最后一项只分析同帧相对误差，不导出更窄的在线许可包络。

```bash
.venv/bin/python -B -m pytest -q tests/graph_v2/test_v231_evidence.py \
  tests/graph_v2/test_v23_goals_memory.py \
  tests/graph_v2/test_release_permission_edges.py \
  tests/graph_v2/test_inflight_invalidation.py
```

## 执行机制

`budgets.evidence_planning=true` 启用下降前检查。父进程将 `evidence_plan` 连接到既有 `skill_intent`、当前观察、原计划和可传播锚点。预测未来状态不改变记忆年龄、已走路径或任何许可；零未来延迟只用于必要可行性筛查，实际执行仍检查真实时间。

持物锚点不足时，在高位中断并请求 `prepare_evidence()`：只有已验证抓持、公开安全间隙和剩余预算都满足，才允许一次既有 wrist-yaw 补证。目标是获得可识别完整面，不接受同视角的新 ID 或预测框作为新信息。然后使用新来源、新计划；精确控制器跳过已经收敛的运动段。

桌面 Skill 保持当前实例掩膜、机器人几何和 footprint 清障契约；缺失清障不能用单纯 `observe_surfaces()` 替代。条件图追踪 `clearance:object:table`，运动使其过期。G0/G1 使用相同感知、姿态传播、主动观测、安全阈值和预算。

## 状态与误差

V23.1 保留原锚点经验包络和增长率；刚体平移/年龄增长只进入位姿误差，不再重复进入内禀边长。倾斜对几何范围的影响单独用角点位移上界计入。每次预测输出组成项、锚点引用、标定版本、年龄、路径长度及条件假设。

没有自动抵消“同一相机”的全部误差。现有标定把位姿、附着与标定误差混在锚点包络中，不能伪造三者独立数字。相对几何离线结果说明保守性的来源，但尚不足以批准动态相对状态模型。

预测许可还检查估计方法的标定范围：完整面包络不能借给宽松顶面/颜色质心。目标 API 与验证器优先使用同一当前完整面结果；另一侧缺少合格标定证据时保持 UNKNOWN。预测形状的历史 `current_*` 来源名称也不能使它被当成当前实例形状。当前语义像素必须重新通过原面拟合才能补全当前掩膜，且不自动获得历史预测锚点资格。

原直接观测路径、释放 XY ≤8 mm、间隙 [-4,16] mm、最后 capture 后准入、Sandbox、独立物理评分均保留。预测证据明确标记非当前视觉检测。滑移、抓持丢失、标定改变、矛盾、可能接触、过期和运动越界仍阻断。

完整结果及限制见 [V23.1 开发报告](../reports/graph-feedback-v23.1-development-2026-11-23.md)。
