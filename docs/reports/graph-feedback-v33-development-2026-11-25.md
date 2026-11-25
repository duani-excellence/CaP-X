# CaP-X V33：Persistent Agent Harness 开发与真实恢复验收

日期：2026-11-25。正式工程 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，HEAD `9c7813f9cf6bb98cf4dd633b310fcc4d59233c44`。未提交或推送；原未提交工作保留。

**结论：已打通确定性恢复失败 → 真实 DeepSeek → 新程序 → 同一物理 episode 继续执行的链路。两次真实模型恢复均执行，但没有产生新的物理恢复，四物体任务仍为 2/4、reward 0。在线结果暴露了旧图工作清单误导模型的问题，现已修复并完成本地回归；修复后没有第三次付费请求。不能宣称模型已经恢复 Cereal 或完成 Can，也没有 G1 优于 G0 的证据。**

所有新产物在 [V33 归档](../../outputs/model_comparison/graph_v33_20261125_000920)，机器可读汇总 [final-audit-v33-delivery.json](../../outputs/model_comparison/graph_v33_20261125_000920/final-audit-v33-delivery.json)。架构、接口边界和命令见 [graph-v33.md](../feedback/graph-v33.md)。

## 1. 原退出链与修改

原链为 `unified → contract_models.run_episode → Sandbox → NativeBoundary → ModelContractRuntime → ContractBroker → Skill`。在 `prepare_recovery()` 返回 false 时，Runner 立即 break，即使还有 API、时间和模型预算。V32 Seed 0 正是在 `target_geometry:2=UNKNOWN / missing_bounded_wall_pair` 后以 `no_legal_recovery_capability` 结束。

本版在同一个 Runner 内加入 Supervisor、Controller、Monitor、Recovery Orchestrator 与 append-only Event Store：

1. 每次父进程 RPC 与物理原语都有程序位置、输入输出和 PENDING/IN_FLIGHT/DONE/FAILED/UNKNOWN 状态。
2. 保留原 R1 确定性修复；没有方案时，可恢复错误升级 R2。UNKNOWN 物理效果、硬安全与环境故障不盲目重试。
3. R2 使用现有 DeepSeek Provider、请求账本与 Sandbox，新 actor 保持同一物理世界。图查询可辅助重新分析，没有增加另一个 Agent 或 Runner。
4. 新公开工具允许当前观测、区域补证、条件核验、图查询和受限 TCP 观测运动。没有开放任意运动、原始释放、env 或私有状态。
5. 两次等价无进展、请求/动作/时间预算耗尽时保存现场并挂起。最终进展判定使用条件和实际机器人测量；新 source ID、运动计数或新代码本身不算进展。
6. 旧工作清单在 Agent 恢复时是建议；当前合同、资源锁、父进程动态许可与安全检查仍强制生效。G0/G1 共用这些能力。

Checkpoint 记录公开现场、程序和事件位置，不恢复 MuJoCo 历史状态。本版支持进程内 actor 续接与持久化审计；**没有实现崩溃后自动续接真实物理进程**。重读日志会把未完成 IN_FLIGHT 标成 UNKNOWN，需要显式状态核对。严重状态无法核实时保持 R4。

## 2. 实验来源与版本

使用原 V32 最终 `online-fourway-final/seed-0-repeat-0/initial.json` 和 `D-FT-G1/snapshot.json`，从初始状态重新执行，没有注入中途状态。所有完整试验的 Snapshot hash 为：

`642d1d5a14b89950ac7941358992a46846002a685018a1f9e56a180af9333184`

初始程序 hash 为：

`abc26d225a4f2ad62a6740cd1d9c6a46838de669518ea16cd8efd105bf0a73ae`

运行记录：

- `zero-llm-gate`：恢复请求预算为 0，复现原阻塞并验证事件/现场保存。
- `online-recovery`：工程预检发现位姿数组还含夹爪分量，在新增模型请求前中止。完整失败记录保留，不计作完成试验。
- `online-recovery-final`：修正四元数切片后，完成唯一一次真实 DeepSeek 恢复 episode，新增请求 2 次。此构建仍有旧 `recovery_scope.enforced` 反馈矛盾。
- 随后修正公开状态与恢复提示词：工作清单改为建议，新的恢复 API 版本明确覆盖初始封闭工具列表。通过回归，不追加付费请求。
- `controlled-view-probe`、`controlled-view-probe-final`：手写诊断片段、零新增 LLM、真实 Vision/F/T。最终运行退出码 0；前一次实体试验和档案已保存，但打印 numpy 摘要报错，已修复并保留原记录。这两次都不能算新的 DeepSeek 结果。
- 最后还补齐开放观测运动后的抓持失效反例：派发前撤销旧抓持证据，只有当前核验通过才恢复 TRUE；若运动后反证/缺证，进入待确认阶段。此失败分支通过单元测试，真实诊断中没有触发滑移。

每次运行都有独立源码归档和源码核对；修复前后的结果没有合并成同一个算法版本。

## 3. 真实模型结果

| 记录 | 原生进度 / reward | API / capture | Skill | SAM3 / GraspNet / IK | 本轮新 LLM | 本轮墙钟秒 |
|---|---|---:|---:|---:|---:|---:|
| V32 历史 G1+F/T | 2/4 / 0 | 57 / 74 | 8 | 228 / 7 / 99 | 0 | 95.71 |
| V33 零 LLM 回归 | 2/4 / 0 | 57 / 74 | 8 | 228 / 7 / 99 | 0 | 96.91 |
| V33 真实恢复 | 2/4 / 0 | 61 / 76 | 8 | 228 / 7 / 99 | **2** | **254.09** |
| 受控观测运动最终诊断，非模型 | 2/4 / 0 | 63 / 81 | 8 | 238 / 7 / 102 | 0 | 106.52 |

GraspNet 的 7 次包含 5 次 `plan` 和 2 次 `plan_point_clouds`。API 总数包含模型可调用工具和内部原语；机器人运动/夹爪命令在真实模型恢复前后相同：23 次 goto、6 次 open、3 次 close，另有 16 次 bounded wait。恢复程序没有新增释放或搬运。新增两次 capture 和区域核验复用了精确缓存，没有增加 SAM3 HTTP 请求。

### 两次恢复程序实际做了什么

[第一份恢复程序](../../outputs/model_comparison/graph_v33_20261125_000920/online-recovery-final/seed-0-repeat-0/G1/actor-07-program.json)：

- 查询 `execution_state()`。
- `inspect_region('destination')`。
- `verify_conditions(['target_geometry:2'])`。
- 结果仍为 UNKNOWN；没有动作派发。

[第二份恢复程序](../../outputs/model_comparison/graph_v33_20261125_000920/online-recovery-final/seed-0-repeat-0/G1/actor-08-program.json)：查询状态并打印无法继续，理由包含“enforced=true 且清单为空”。没有尝试 `move_tcp`。

这个理由揭示了真实工程缺陷：Python 派发已经允许受检查的开放恢复，但公开状态仍声称图清单强制限制，初始 API 文档也保留了旧规则。不能把这次失败全部归因于模型能力。现在状态、文档版本和实际准入语义一致；单元测试同时验证“清单是建议”与“缺少当前条件仍不派发”。该提示修复的真实模型效果尚未验证。

两轮都没有改变可执行条件，最终 `no_information_progress`，保存 TaskSuspended。Milk、Bread 保留，Cereal 持有，Can 未执行。没有把安全退出算成任务完成。

### Token、费用与回执

| 恢复请求 | HTTP / finish | 输入 | 输出（含 reasoning） | reasoning | 秒 | USD 保守估算 |
|---|---|---:|---:|---:|---:|---:|
| recovery-7 | 200 / stop | 19,805 | 7,133 | 6,919 | 93.66 | 0.05438928 |
| recovery-8 | 200 / stop | 20,078 | 4,612 | 4,433 | 64.45 | 0.04476648 |
| 合计 | 各 1 次尝试 | **39,883** | **11,745** | **11,352** | **158.11** | **0.09915576** |

模型请求与响应均为 `deepseek-v4-pro`，max_tokens 32768、reasoning_effort low；没有程序截断或把 reasoning 当控制代码。服务没有提供 request_id；response_id 分别为 `95355a03-449e-435e-9d37-e276fc3d7b8e`、`4615a0cc-69ed-451d-a335-ac964bc2708d`。费用来自现有账本的保守估算，不是服务商实际账单。

本轮真实错误反馈 2 次、生成 2 份恢复程序、实际执行 2 份；新增请求严格为 2，未自动重试。历史初始程序的旧请求没有计入本轮支出。

## 4. 新运动能力的独立物理诊断

为区分“模型没有尝试”与“接口不能执行”，最终手写诊断只使用公开 API：读取当前 TCP，沿已验证抓取走廊向上请求 10 mm，再检查目标区域和持物。

实际测得 TCP 位移约 `[-0.028, -0.148, +9.505] mm`。当前抓持验证 PASS，已完成两个目标保留。目标区域依然是 `missing_bounded_wall_pair / UNKNOWN`，因此没有释放，也没有新增完成目标。图使旧工具几何失效，保留已完成效果，补证仍不足时按预算挂起。

这是**公开运动接口的真实可执行证据**，不是模型创造有效恢复策略，也不是 Cereal 物理恢复成功。仅有这一次垂直变化，不能证明所有可行视角都失败；模型声称“无法观察”的说法也不能当作几何证明。当前缺少已验收的任意横向持物避障、相机视角控制，以及足以授权释放的容器历史几何融合。

## 5. 测试与平台

- 图＋平台全量回归：**1230 passed / 9 skipped / 0 failed**，见 `graph-platform.xml/log`。这是较早构建；9 项为 6 个显式渲染夹具、3 个官方 LOGIV checkout 夹具未配置，不是通过。
- 最终代码针对性回归：**108 passed / 0 failed**，见 `delivery-regression.xml/log`。包含持久化、崩溃 UNKNOWN、不可重放、G0/G1 等价升级、API 数据维度、无信息进展、开放工作范围仍受当前合同约束、观测运动后的抓持失效，以及已有 V27/V30/V32 与平台回归。不同测试运行有重叠，不能相加当成独立测试数量。
- 原 P0：**68/68 passed**，含现有在线视觉与配对 Oracle 回归，见 `p0.xml`、`p0-regression.log`、`p0-artifacts/`。
- 前端 `npm --prefix web-ui run build` 通过；真实 `/api/experiments/events` 返回 200 并读取实际事件。事件 hash 链核对通过。
- 工作台新增持久化模式、明确的恢复请求上限、分页恢复时间线和 checkpoint 引用；原视频、图和力觉页面保留。
- **浏览器内验收未完成**：浏览器工具安全策略拒绝本地页面访问；没有尝试其他浏览器通道绕过，也没有把后端接口检查冒充浏览器验收。

## 6. 运行与保存情况

复用 SAM3 18114 和 GraspNet 8215，OpenAPI 200。NVML 仍是 driver/library mismatch；没有修驱动或重启其他 GPU 服务。重新启动自己的动态端口 CPU IK 服务，沿用旧 solver；继续使用 Mesa/EGL。没有下载模型、改依赖、训练或操作真实机器人。

`pre-v33-work.tar.gz` 保存开发前工作；`preservation-audit.json` 中先前 56 个工作文件没有丢失，49 个字节未变、7 个扩展修改。API 配置仍为 0600。旧 V32 实验未写入；新结果独立存档。

新增/修改文件：

- `capx/execution/agent_events.py`、`persistent_harness.py`、`open_recovery.py`。
- `capx/experiments/contract_models.py`、`native_tasks.py`、`unified.py`。
- `capx/web/experiments.py`；`web-ui/src/components/ExperimentWorkbench.tsx`、`RecoveryTimeline.tsx`。
- `env_configs/graph_feedback/development-v33.json`、`env_configs/inference/local-v31.json`（当前 IK runtime 引用）。
- `scripts/local/harness_probe.py`。
- `tests/graph_v2/test_v33_harness.py`、`tests/platform/test_v33_events.py`。
- `README.md`、`docs/feedback/graph-v33.md` 和本报告。

## 7. 验收判断

| 层级 | 结论 |
|---|---|
| Harness 持续恢复 | **已实测**：旧恢复器无方案后升级真实模型，同一物理 episode 执行新程序，最后有界挂起 |
| 恢复程序执行 | **2/2 执行通过**；均为补证/查询，不能据此声称物理恢复 |
| 新公开运动能力 | **受控真实诊断通过**：新 Python 操作可派发受限运动并验证抓持；非模型策略 |
| Cereal 物理恢复 | **未通过**：边界证据仍 UNKNOWN，未放置 |
| 完整四物体任务 | **未通过：2/4，reward 0**；Can 未执行 |
| G1 相对 G0 收益 | **未评估**：共同机制有回归，只有一个 G1 真实恢复 episode，不能推导优越性 |
| 扩大实验 | **NO-GO**：先验证反馈修复后的真实模型决策，再补足公开可验证的主动观测能力 |

下一次最有价值的小实验是使用同一 Snapshot、最终反馈契约和相同预算重新请求模型，观察它是否实际使用已验收的开放工具；不应直接扩大场景数。若仍缺容器边界，应针对公开可观测性和安全视角能力定位，不能降低释放阈值或让模型直接读仿真真值。
