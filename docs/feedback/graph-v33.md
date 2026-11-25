# V33 Persistent Agent Harness

V33 在 `capx.experiments.contract_models.run_episode` 内接入持久化服务，沿用 V31 CLI、Vision Skills、父进程 Broker、Sandbox、V29 因果修复和 V32 F/T；没有另建机器人执行循环。当前只接入 PickPlace 的 G0/G1，其他 Runner 明确拒绝 persistent 组合。默认仍为 legacy。

## 实际调用链

```text
infer.sh → unified.RunSpec → contract_models.run_episode
  → TaskSupervisor（目标、剩余预算、进展）
  → AgentController（现有 PersistentModelBudget / DeepSeek）
  → 新版本 Python → Sandbox RPC → NativeBoundary._invoke
  → ExecutionMonitor → ContractBroker / 公共恢复工具
  → 既有 Skill → 父进程 trusted_motion → 原控制/碰撞/力觉检查
  → 公共观测 / 条件更新 → RecoveryOrchestrator
  → R1 现有确定性修复 或 R2 新 DeepSeek 程序 或 R4 挂起
```

图中仍保存目标、条件与因果计划。`graph_query` 向模型提供状态、依赖、证据、失败查询；开放操作可产生新的 API 图节点。图建议不能代替动作准入。

| 模块 | 文件 | 职责 |
|---|---|---|
| Supervisor / Controller / Monitor / Orchestrator | `capx/execution/persistent_harness.py` | 在现有循环内管理生命周期、错误升级、程序、预算与进展 |
| 不可变事件与 Checkpoint | `capx/execution/agent_events.py` | JSONL hash 链、动作状态、独立程序版本与不可变现场 |
| Open Python / Graph Skill | `capx/execution/open_recovery.py` | 可公开查询和有限观测运动，不提供 env、原始释放或任意轨迹 |
| Runner | `capx/experiments/contract_models.py` | opt-in 持续执行、R1/R2 升级、独立原生评分 |
| 统一入口 | `capx/experiments/unified.py` | 明确开启持久化与真实恢复请求；历史重放默认不付费 |
| 工作台 | `capx/web/experiments.py`、`RecoveryTimeline.tsx` | 真实事件分页、程序版本、恢复与挂起时间线 |

## 许可、恢复和持久性边界

- `no_legal_recovery_capability` 是 R1 没找到方案。存在预算且物理效果可核对时升级 R2；两组同样允许生成新程序。
- R0 只对公开只读 API 的连接/超时错误做最多一次额外重试；已有工具客户端重试策略保留。高层 Skill、运动和模型请求不能隐式重试。
- R1 复用当前因果/条件修复。R2 可调用 Graph Skill 再分析和组合新 Python 操作；当前没有单独新建第二个 R3 Agent。
- UNKNOWN/隔离、硬安全与环境故障进入 R4 或终止。没有信息增益连续两次、API/Skill/时间或恢复请求耗尽均有界退出。新 source ID 不算进展。
- `execute_skill` 仍经过合同、已完成调用检查、资源和物理安全门。Agent 恢复可以超出旧的图工作清单，但不能超出父进程能力与契约。
- DONE 调用 ID 不允许再次派发；高层完成 Skill 不可直接重放。失败 Skill 如需重试须 `repair_skill` 产生新 ID，并重新核验当前条件。不同参数的合法新操作不会被视为历史调用重放。
- 物理原语出错后：已知同步安全边界停止标记 FAILED；未分类派发错误标记 UNKNOWN 并阻断后续运动。子进程退出不代表物理世界回滚。
- Checkpoint 保存程序、公开现场、条件、预算、完成/未知动作和事件位置。进程内重建 actor 可继续同一仿真。**本版不自动恢复崩溃后的仿真/真实机器人进程**；重读日志将未完成 IN_FLIGHT 归为 UNKNOWN，需要人工/专门状态核对，不能用历史 Snapshot 回滚续跑。
- 最终成功仅由原生物理评分决定。打印 FINISH、工具返回、程序正常结束及安全退出均不是任务成功。

## 已开放的 Python 恢复接口

除 `execute_skill`、`repair_skill`、`refresh_evidence`、`execution_state`，增加：

- `capture_observation()`：新公开 RGB-D、机器人状态、启用时 F/T、证据文件及图像身份；不开放文件系统。
- `inspect_region('source'|'destination')`：当前公开几何或 UNKNOWN，不把旧边界当当前检测。
- `verify_conditions(keys)`：共享补证预算内的当前核验。
- `graph_query('state'|'failures')`、`graph_query('evidence'|'dependencies', key)`。
- `move_tcp(position, quaternion_wxyz, operation_id)`：仅支持既有垂直抓取走廊中的小范围观测运动。必须已验证持物、当前邻物/容器可见碰撞扫掠与 IK 通过，随后保持 TCP 和 F/T 检查。相同 XY/朝向、每次向上 1–20 mm、总高度不超原抓取点 0.20 m；还受旧走廊适用范围约束。

这不是任意 Cartesian 运动规划器。没有主动相机执行器、任意横向持物避障、任意持物临时放置或历史容器边界自动融合许可。需要这些能力时应报告缺口，不能假装 API 已存在。可见点云扫掠也不是全机器人关节空间/未知障碍安全证明。

## 命令

从项目根目录执行；服务地址由项目配置和 `--ik-runtime` 提供。

```bash
cd /home/hanyu/dqb/cap-x-20261106

# 原有模式，不改旧实验语义
scripts/local/infer.sh run --methods G0 G1 --skill v32 --force-mode on --seeds 0

# 新生成初始程序 + 持久化恢复；最多两次恢复、总计三次请求
scripts/local/infer.sh run --methods G1 --harness persistent \
  --skill v32 --force-mode on --seeds 0 --recovery-requests 2 --max-requests 3

# V32 原始初始程序 + 原始 Snapshot + 真实 Vision + 最多两次新恢复请求
# 每次选择一个全新的输出目录，不能在已完成物理 episode 目录内覆盖/续跑。
scripts/local/infer.sh replay --methods G1 --harness persistent \
  --skill v32 --force-mode on --seeds 0 \
  --reference outputs/model_comparison/graph_v32_20261124_221234/online-fourway-final/seed-0-repeat-0 \
  --snapshot-group D-FT-G1 --recovery-requests 2

# 同一命令 --recovery-requests 0 是零新增 LLM 的机制/历史初始程序重放。
# --dry-run 校验配置，不发请求、不创建仿真。
scripts/local/infer.sh serve --host 127.0.0.1 --port 8200
```

两组均接受 `--harness persistent`。配对可用 `--methods G0 G1`；`--max-requests` 是整个作业共享上限，本轮最多两次实际请求，因此本轮真实在线验收只运行 G1，不构成 G0/G1 效果比较。

## 归档

每个 episode：原有 `result.json`、`trace.jsonl`、`contract-graph.json`、`graph-events.jsonl`、视频、RGB-D、F/T 和 request receipt 原样保留；新增 `harness/events.jsonl`、`program-NNN.json`、`checkpoint-NNNNNN.json`、`latest.json`。

事件包括 TaskStarted、PlanGenerated、ActionStarted/Result、ObservationCaptured、ConditionChanged、FailureDetected、RecoveryProposed/Rejected/Executed、VerificationResult、CheckpointSaved、TaskCompleted/Suspended/Terminated。事件中的独立最终评分不送入 Agent；在线 Recovery Context 只来自公开状态，私有物理 Trace 只给离线诊断。

HTTP 工具实际调用和 LLM 请求成本仍以原账本为准。历史初始请求仅用于来源和归因，不计为本轮新调用；不可把 `llm_requests`（含历史来源）直接当本轮支出。

## 架构参考（没有安装或宣称复现这些系统）

采用 [OpenHands 的执行与事件职责分离](https://github.com/OpenHands/docs/blob/main/openhands/usage/architecture/backend.mdx)、[SWE-agent 的可检查 trajectory](https://swe-agent.com/latest/usage/inspector/)、[LangGraph 的持久化思想](https://docs.langchain.com/oss/python/langgraph/persistence)和 [Nav2 的有界 RecoveryNode](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/bt_plugins/controls/RecoveryNode/)。软件步骤重放不能直接用于机器人动作：本实现对物理副作用采用不可重放/UNKNOWN 隔离，未实现分布式 exactly-once 控制。

## 零新增模型的公开运动诊断

下面命令明确使用手写诊断片段，只测试受限观测运动和重新验证，不是新的 DeepSeek 成功试验。它仍复用 `run_episode`、原始程序/Snapshot、同一 Sandbox 与父进程许可，不能读仿真真值。

```bash
source scripts/local/common.sh
.venv/bin/python -B scripts/local/harness_probe.py \
  --reference outputs/model_comparison/graph_v32_20261124_221234/online-fourway-final/seed-0-repeat-0 \
  --snapshot-group D-FT-G1 \
  --ik-runtime outputs/model_comparison/graph_v33_20261125_000920/ik-runtime.json \
  --output "outputs/model_comparison/public_probe_$(date +%Y%m%d_%H%M%S)"
```

V33 在线验收发现的反馈矛盾已经修复：Agent 恢复时 `recovery_scope.enforced=false`，旧工作清单明确标为建议；父进程实时条件和许可仍强制执行。恢复提示词显式声明新的 API 版本覆盖初始程序的封闭工具列表。真实模型回执来自修复前构建，修复后只完成机制与零新增模型的真实物理验收；不能据此声称模型已学会有效恢复。

进展判断使用当前条件与量化的实际机器人位姿/编码器；仅增加运动计数、生成新代码、新来源 ID 或重复等价观测不算进展。未知物理效果仍隔离。所有历史构建与失败均保留。
