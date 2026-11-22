# Graph V22 开发原型（当前修订 V22.2）

在现有父进程 broker、Sandbox、GroundedMonitor、GroundedRecovery 和 graph_v2 Runner 上增量实现。启用 `budgets.task_conditions=true`；未启用时保留 V21 阶段执行路径。G0/G1 共用该条件验证与动作准入、公开感知、技能、物理评分和单 episode 预算。G1 通过 `selective_recovery` 使用条件回归；G0 保留保守后缀恢复和跨阶段续接。

## 图如何控制执行

- `TaskConditions` 把 TRUE/FALSE/UNKNOWN 条件、任务子目标、预期效果放入已有 `DependencyGraph`。数据边失效传播到计划；`attestation` 边仅保存历史验证来源，过期的位置不撤销“曾经验证”的事实。持续条件通过动作影响和当前公开观测重新评估。
- 已返回调用、已验证物理条件、完整任务候选完成分别记录。历史调用不转回待执行；返回的 place/open 不直接证明放置成功。
- Restack 图保存绿色桌面放置及红色最终堆叠两个阶段。完整候选要求红在绿上且已安全撤退；仍由独立原物理评分决定 reward。
- `GraphExecutive` 是有界、确定性的条件回归调度器：从未满足目标和当前持物/释放约束选择已有合法配方，不按 DAG 拓扑顺序执行机器人动作，不恢复 Python 指令指针。
- 已定位且当前对齐有效：G1 可直接请求“释放＋新撤退”，不重新执行放置。释放前最后一次 capture 后仍必须通过原准入；图中的 TRUE 本身不是执行许可。
- 已完成放置效果缺乏证据：只补当前表面观测；明确反例才重建场景。抓持不明用 verify_hold；已释放只做新撤退；未知物理效果继续终止隔离。
- 每次恢复记录 failed_conditions、affected_nodes、preserved_nodes、repair_goals、selected_actions、reused_sources、新计划、实际调用及 verification_results。选中节点不计作调用收益。
- V22.2 在完整配对结束后修正桌面后置条件的原始/精化中心混用，保留原几何阈值与释放前清障。实际放置目标绑定技能/来源，不用最终任务目标代替执行历史；条件测量指标保留在图中。

## 运行

所有命令在正式项目根目录执行。配置使用当前已核验服务，脚本自动加载本地密钥；不需要把 key 写入命令。

```bash
cd /home/hanyu/dqb/cap-x-20261106
# 少量真实 DeepSeek + Vision 配对；创建新输出目录，会付费。
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v22.json \
  --task stack --seed 1257104511 --groups G0 G1 \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-1257104511/prepared-snapshot.json \
  --output "outputs/model_comparison/graph_v22_stack_$(date +%Y%m%d_%H%M%S)"

# 已有真实 DeepSeek 程序＋实时 Vision 诊断；绝不新调用 LLM。
source scripts/local/common.sh
.venv/bin/python -B scripts/local/replay_graph_initial.py \
  --config env_configs/graph_feedback/development-v22.json \
  --source-unit outputs/model_comparison/graph_v21_online_20261122_154720/normal-restack \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/restack-677061216/prepared-snapshot.json \
  --bind-current-contract \
  --output "outputs/reproduction/graph_v22_restack_replay_$(date +%Y%m%d_%H%M%S)"
```

`--bind-current-contract` 仅允许诊断性契约迁移，分别保存原/现 Prompt 和 API 文档 hash；历史代码/响应/原请求凭据保持原样，不宣称该代码是新 Prompt 生成的。历史重放与新 DeepSeek 结果必须分开统计。

修订后的固定 Restack 开发回归命令如下。它重新调用 DeepSeek 生成初始程序和必要恢复；与旧版本结果分开记录，不能合并统计为独立场景：

```bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v22.json \
  --task restack --seed 1482414119 --groups G0 G1 \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/restack-1482414119/prepared-snapshot.json \
  --output "outputs/model_comparison/graph_v22_restack_$(date +%Y%m%d_%H%M%S)"
```

当前开发配置每个独立 G0/G1 pair 上限为 32 请求、估计 $5、4 小时；每个 episode 受 16 blocks、192 实际 API、24 局部动作、8 视角、6 恢复配方及连续 2 次无进展限制。32 次覆盖两组各 16 blocks、共享一次初始生成的最坏请求数 31。多次独立 pair 各有自己的账本，不能把单 pair 上限当作多个命令的总预算。不要直接据此发起新的 360/1800 episode 批次。

在零新 LLM 重放命令后增加 `--pause-before-release`，可注入一次“放置已返回、释放前暂停”的软件边界中断，用于验证局部续接。该开关只存在于诊断入口：不改变共享历史程序，不移动物体，不绕过释放验证。其结果必须标注为软件机制压力，不能当作自然故障或新模型生成成功。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/graph_v2 tests/graph_feedback
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/feedback
```

两组目录中存在同名 test_analysis.py，所以原分层反馈目录单独运行。真实渲染的六项测试沿用 `CAPX_DEPTH_RUNTIME_CONFIG/SNAPSHOT` 开关；P0 真实视觉测试沿用 `CAPX_P0_VISION=1` 和 `CAPX_TEST_OUTPUT`，必须指向新产物目录。

## 边界

只在 API/技能边界干预，无运动中取消。视觉证据可能有共同系统偏差；条件 TRUE 不是接触传感或原物理成功。公开几何不足保持 UNKNOWN，有界停止优于盲目复现旧动作。完整图保留父进程及归档；LLM 仅接收有界诊断、关键当前条件、未完成目标和预算。裁剪不改变执行许可。

实测结果、失败诊断与版本差异见 [V22 开发报告](../reports/graph-feedback-v22-development-2026-11-22.md)。仅以小规模固定开发结果判断当前覆盖范围，开发机制通过不等于正式任务效果提升。
