# V36 reviewed repair：闭环修复与真实开发配对

正式目录：`/home/hanyu/dqb/cap-x-20261106`。分支 `feature/graph-complete-20261120`，基线 HEAD `8052fa7ed3d2d74955a82454d7242a41dc0c9307`。本轮未提交/推送；保留原 V34–V36 未提交工作。交付的是增量工作树，不能仅凭 HEAD 重建。

任务依据为用户 ZIP 的 `CODEX_TASK.txt`、`SOURCE_AUDIT.txt` 和四个反例。ZIP 没有 ACCEPTANCE.txt，用户明确允许忽略。原审查源码归档 SHA256 `9c77f4dacc1f882bf16dcef58909888d716a91c7449249fccf3ae1c6f122cddf` 未变。

本轮所有新产物：[`v36_review_repair_20261125_200443`](../../outputs/model_comparison/v36_review_repair_20261125_200443/)。各阶段有独立源码字节归档和 `source-postcheck.json`；不同阶段不能混为同一版本结果。命令见 [使用说明](../feedback/graph-v36-review.md)。

## 已获得的证据与边界

新 DeepSeek 配对已实际请求模型、运行真实 SAM3/GraspNet、公开 RGB-D、编码器和 MuJoCo 腕部 F/T。G0/G1 共享初始程序和 Snapshot，原生物理评分不变。**两组均未完成四物体任务，reward 均为 0。** G0 有一条明确的 Bread 受控失败 → 模型提出撤退/重抓 → 新附着探测通过 → 成功放置的真实恢复链。G1 的绕行和续接完成了另一个子目标，但不能据单场景声称图提高成功率。

新轨迹后端仍未获得实际空爪/持物成功验收，不能作为可用的成熟 Skill 替代品。动态物理组合的注册、合同推导、规划搜索和父进程执行已通过明确标记的符号夹具；本次真实模型没有注册这种组合，因此不是其物理恢复验收。

## 逐项修复

| 审查项 | 原问题及修改 | 验证与当前范围 |
|---|---|---|
| 1 补证后进展 | `next()` 在必要 `prepare_recovery()` 和条件协调后只计算一次进展；硬安全、未知效果、真实预算仍优先 | 原第四次补证反例先失败后通过；在线所有恢复经过此入口 |
| 2 旧条件恢复/负前提 | ever-TRUE 仅作历史统计；有界的语义重新建立计数与当前义务分开；TRUE/UNKNOWN 抖动不能无限续命；仅候选 ID 变化不算搜索收益 | 旧目标恢复、FALSE 前提、来源换名、噪声反复切换、候选换名测试；当前是保守的条件级有限额度，尚非完整噪声模型 |
| 3 验证协调 | 父进程 `reconcile_validation` 将明确 effects 同步到 Skill、ContractGraph、Belief；held UNKNOWN 保留对象身份并进入 closed_unverified；不污染无关条件 | 观测、预规划和 Skill stop 共用入口；在线 Bread/Cereal UNKNOWN 身份保留。现场 Broker 准入没有取消 |
| 4 真实派发/部分运动 | Native 控制步登记 started/returned；规划前错误 NOT_DISPATCHED；受控部分执行经公开编码器核对为 RECONCILED；未知错误保留隔离。trajectory finally 推进运动 epoch、撤销证据且保留已执行前缀；空爪脱离按当前证据更新阶段 | 部分轨迹、派发前异常、真正未确认派发、嵌套包装回执测试。新在线发现高层 wrapper 日志标签不一致，配对结束后修复；不改原记录 |
| 5 两层数据来源 | 公共 `result_ref`/`dependency_refs` 接到真实估计；包装/投影用 data 边，调用历史用 attestation；组合返回继续携带来源协议 | 原估计→两层投影→待执行计划一并失效，已完成调用仍 DONE。任意 Python 内部计算不伪称完整因果追踪 |
| 6 动态能力成熟度 | 上下文指纹排除记录 ID、目录、时间等；复用不同测量也不自动 REUSABLE；保留限定范围 VERIFIED 和案例 | ID 改名不增加独立案例；类型检查、投影忠实性与物理验证分开；提示词同步此语义 |
| 7 open 因果断路 | open 建立活动 CausalPlan；可静态确定的无条件字面量调用提前编译，其他 Python 分支保持条件性；真实选择时增量登记；`plan_skill` 登记条件性 future_candidate | open 分支可执行，实际选中调用进入因果图；no_causal 软件反例确有不同边。未来候选不是动作许可或强制序列 |
| 8 可调用→可规划 | 只对最多四个静态绑定可信子 Skill 的直线组合推导合同、资源和破坏效果；加入现有技能库/producer 搜索；复杂分支保留 CALLABLE_ONLY；每个子动作继续走原 Broker | 原搜索深度限制下无方案→合格组合注册→可检索修复→执行子动作→保留无关完成目标；是符号夹具，不冒充模型物理成功 |
| 9 程序反馈/真实 API | 初始/恢复复用 runtime API 文档和动态绑定签名；恢复包包含当前异常、失败行邻域、已完成不可重放调用；不再以旧 last_failure 替代本轮异常；常见 Python 错误归 PROGRAM_ERROR，跳过无意义物理 prepare | 通过真实业务类回归；完整单一类型 schema 尚未覆盖所有旧 API，仍是部分完成。所有重试受原请求/时间/费用预算 |
| 10 轨迹能力 | PyRoKi CPU TrajOpt 实际收到当前公开 RGB-D 障碍；同相机机器人表面剔除、保守占据体素、来源和标定版本记录；旧实体体积包含使用闭合网格，禁止自动补洞伪造实体 | 实测 2537 个障碍，优化结果起点误差和终点误差超限，0 派发。持物-机器人接触排除合同仍未完成，后端不验收。869 旧拒绝点未证明为误报 |
| 11 联合抓放选择 | 最多三种通过基础检查的抓法，每种最多八个未来放置候选；稳定 candidate_id 可传入预览/计划/执行，执行复核相同候选；抓后重验真实公开附着 | 有限候选回归及真实执行使用。未来假设不授权释放；完整 6D 抓法仍 DIAGNOSTIC_ONLY，未擅自启用 |
| 12 实验有效性 | 统一父进程 Skill 完成钩子覆盖 execute_skill、plan/run 和组合；未触发标 scenario_not_activated；每 episode 动态能力库隔离；保留实际因果事件 | 本轮在线为 normal，无故障激活；不能把配置开关当实际消融收益。外部 LOGIV/B0 路径保持，ASPIRE/CaP-Agent0 未完成适配不记零分 |

## 真实 DeepSeek 配对（修复在线版）

目录 `live-pair/seed-0-repeat-0`。Snapshot hash：`642d1d5a14b89950ac7941358992a46846002a685018a1f9e56a180af9333184`。初始程序仅生成一次；恢复程序由各组自己的真实状态包生成。模型 `deepseek-v4-pro`、输出上限 32768、相同现有推理配置，HTTP 全部 200，finish_reason 全部 stop；没有推理耗尽导致的截断，也没有自动重试。

| 指标 | G0 | G1 |
|---|---:|---:|
| 完整任务成功 | 0/1 | 0/1 |
| 原生子目标 | 2/4：Milk、Bread | 2/4：Milk、Cereal |
| reward | 0 | 0 |
| 机器人及 Agent API 计数 | 93 | 76 |
| 底层物理 API（含 wait） | 53 | 48 |
| Capture | 95 | 86 |
| SAM3 / GraspNet / IK 实际 HTTP | 313 / 11 / 485 | 276 / 10 / 377 |
| 新恢复 LLM 请求 | 3 | 2 |
| episode 墙钟（不含共同初始生成） | 483.34 s | 402.01 s |
| 终止原因 | recovery_request_budget | unknown_physical_effect（内部 AssertionError） |

共同初始生成另耗 86.85 s。两组相加加初始请求约 972.19 s；这是工作阶段耗时，外层初始化/导出开销应看总运行日志。单场景、不同恢复代码、不同已完成对象和 G1 工程错误是重要混杂，不能把较少 API 或较短时间直接归因于图更高效。

### 恢复证据

**共同开头：** 初始 DeepSeek 程序完成抓取，但 Milk 放置局部对齐报告无进展。Harness 没有把程序返回当成功，现有确定性续接完成 Milk 放置。下一次 Bread 接近触发原 F/T 保护（约 450.61 N、43.34 Nm），动作受控停止，仍保留已完成 Milk。

**G0 Bread：** actor-03（真实 DeepSeek）规划并执行 retreat_empty 后重新抓取；当前附着 registration UNKNOWN，未释放。actor-04（真实 DeepSeek）选择 confirm_attachment；通过新表面测量、夹爪收敛和实际受限共动验证，00064 观测记录 `new_current_surface_fit_and_measured_rigid_comotion`。随后放置 Bread，独立物理评分升至 2/4。Cereal 后续力矩保护约 4.26 Nm，附着仍 UNKNOWN，最后一个模型方案缺少前提，耗尽 3 次恢复请求退出。这不是 4/4，也不是力阈值单独授权。

**G1：** actor-03（真实 DeepSeek）先撤退，再做独立 Cereal 子目标；受控力矩停止后，图/父进程选出已有可执行续接，actor-04 完成 Cereal。剩余目标不含 Milk/Cereal，没有重放它们。actor-05 尝试 Bread 并在附着注册 UNKNOWN 停止。actor-06 模型调用 confirm_attachment 时父进程出现 AssertionError；原记录保持隔离退出。该软件错误的最终定位及重放见下方补验，不归因于物理不可达。

贡献区分：真实派发协调、可信验证、候选选择、Skill 能力是两组共同底座；撤退/重新尝试和换独立对象有模型选择贡献；G1 存在实际条件回归与选择性续接事件，但“跳过 Bread”本身来自生成代码，不能全部算作图推导。实际因果修复决策记录 G0=2、G1=7，确定性续接 G0=3、G1=4；这些事件数仅说明机制参与，不是收益指标。

轨迹、视频、程序与当前公开状态分别见各组 `trace.jsonl`、`video.mp4`、`actor-*-program.json`、`actor-*-public.json` 和 `harness/events.jsonl`。Trace 中的独立物理真值仅用于评分/本报告，未输入在线模型上下文。

## 请求和成本

共 **6 次新增真实请求**：1 次共享初始、3 次 G0 恢复、2 次 G1 恢复。输入 **67200** Token，输出 **40984** Token，其中 reasoning **35951**（已包含在输出，不重复相加）；LLM 请求墙钟累计 **629.48 s**。项目既有费率口径的估算上界 **$0.25100064**，不是服务商账单；共享初始只计一次。原配置本轮上限 7 次/$3，没有自动扩大。

脱敏回执在 `live-pair/requests/` 和 `request-ledger.json`，统计在 `analysis.json`。请求 ID header 未返回，但 provider response_id 均保留；未导出 Key、Authorization 或私有环境变量。

## 零新增 LLM 与轨迹后端

- `historical-pair`：修复早期版本、旧真实模型程序、真实 Vision/F/T，两组均完成 Milk 1/4 后在 Bread 力保护受阻；0 新请求。候选搜索当时尚未进一步限界，不能与最终配对直接比较成本。全部失败保留。
- `trajectory-zero-llm`：受控空爪上移 1 cm，不是模型输出。真实 RGB-D 给出 2537 个占据体素障碍，PyRoKi 求解返回的起点关节偏差 0.005686 rad 超过原 0.002 rad 限制；终点误差约 17.73 mm / 1.37°，未派发。不能把算法返回轨迹当作物理可执行。
- `old-sweep-classification.json`：重读旧冻结 RGB-D、标定和机器人网格；旧 869 个拒绝点中 **0** 个被现有严格闭合原实体证明为内部点，869 仍缺自由空间证据。未将 open mesh 自动补洞，也没有统一忽略 UNKNOWN。当前证据不支持“869 全为射线误报”。
- 持物轨迹仍明确缺少非抓持链的附件碰撞合同。真实空爪轨迹正例、持物运输正例及其碰撞/滑移反例未通过；本轮在线采用原成熟 Skill/受限运动能力，不暗换新后端。

## 软件回归与来源保护

四个原探针先原样复现缺陷（它们退出成功表示发现缺陷，不是验收通过）。新增对应回归先有 12 失败；后续分阶段修复日志都保留。在线前相关回归 **263 passed / 0 skipped**，含 P0 reward/Sandbox、统一入口、Native B0 和官方 LOGIV bridge 检查，不宣称全项目测试通过。

配对期间源码 postcheck 无变化。配对结束后的两层投影、ID 进展和包装回执新增反例再次先失败；修复后的本轮局部集合 46 passed。最终回归和最终重放结果在下方补验部分更新，不与在线冻结版混用。

本轮未升级依赖、未下载模型、未重启其他服务、未训练、未接触真实机器人、未更改原生物理评分。API 配置仍 0600。历史原归档 hash 不变；本轮代码清单与哈希差异保存在新增目录，不以 Git HEAD 掩盖未提交状态。

## 正式评测判断

**NO-GO。** 当前有真实的受控停止、模型干预、技能验证、继续放置证据；还没有完整 4/4 或足以估计图增益的多场景对照。新轨迹后端、真实动态物理组合、完整统一 API schema 仍未验收。应先解决当前具体软件错误和持物证据/可执行运动缺口，保留本轮失败，再做少量同能力开发复验；不启动正式大批次。


## 配对结束后的补验（单独版本）

`g1-diagnostic-replay`：原始 Snapshot + 此次 G1 的初始/actor-03/actor-06 真实模型程序，0 新增 LLM。真实 Vision/F/T 完成 **3/4（Milk、Bread、Cereal）**，reward 0；95 API、110 Capture、141.90 s，未隔离，因无剩余模型请求退出。保留该重放所有请求工具和物理记录，不以它替换前述在线 G1 的 2/4。

这次没有再发生 AssertionError，**根因尚未确定，不能声称已修复**。两次轨迹的第一次显著控制目标差异发生在第 42 个底层物理调用，TCP 目标相差约 0.157 mm，sim_time 相差 0.05 s。相同 Snapshot 不保证在线感知/控制全过程逐位一致；重放成功不能直接归因于回执日志修正。新父进程异常日志会保存异常类型和调用栈位置（无局部变量/密钥），用于后续精确定位。

剩余 Can 的失败不是当前容器缺测：三种合格抓法各测试八个直立放置候选，均报 `ik_unconverged_before_motion`；位置残差约 46.7–62.4 mm、姿态残差约 5.18–7.02°，远超原 IK 门限。这只能证明该有限集合无解，不证明任务物理不可达。详见 `can-planning-diagnosis.json`。

发现一个真实接口缺口：联合搜索失败细节只在 Trace，`planning_query` 只能返回泛化的拒绝原因。补上可检索的 joint_search、future_search、误差和适用范围；新增 `future_positions=1..12`，默认仍 2，最多三抓法 × 四 yaw × 十二位置。Agent 可以有界扩大搜索，但不能改变释放姿态能力或安全条件。新增反例先失败后通过；最终相关回归 **271 passed / 0 skipped / 0 failed**，两个既有 PyRoKi/JAXLS 弃用警告；JUnit 和完整日志 `final-regression.xml/.log`。

最终版将 `enabled_mechanisms` 与 `actual_active_mechanisms` 分开：后者记录实际选中调用、未来候选、因果修复决策、确定性续接、条件回归、补证及组合执行的次数。次数不等于效率或物理成功。旧实验原字段保持原样，不追改历史记录。


### 有界新模型续接：G1，非新初始配对

`g1-bounded-continuation` 使用同一 Snapshot、归档初始程序和两个归档恢复程序，再付费生成 **1 个**新的恢复程序。它不是另一个全新初始生成，不与上表合并成功率。完成 3/4，103 API、115 Capture、238.88 s，reward 0，`recovery_request_budget` 退出，无未知物理调用。

新模型主动查询规划诊断并设 `future_positions=12`。实际测试三抓法 × 48 放置候选 = **144** 个候选，全为 `ik_unconverged_before_motion`，最小位置残差 **22.83 mm**、最小姿态残差 **2.72°**，不能放宽原门限执行。它提出的观测运动也被原扫掠检查拒绝：437 个点、272 个有自由空间支持、覆盖率 0.6224，reason=`swept_volume_occupied_or_unobserved`。没有派发该运动；后续重复预览命中缓存、新求解数 0、无新增子目标。模型代码内还存在未执行分支的参数传递问题（deposit 的 params 被写成第二个位置参数），不能把未执行分支当成通过验收。

完整扩展搜索保存在 `can-expanded-search.json`。未知是否存在其他直立目标或更完整 6D/轨迹能力可解，不写成“物理不可达”。本轮停止继续付费是已用完预设的小实验请求上限，不是把 3/4 当成完成。

本轮总计 **7 次**新请求，输入 **78665**、输出 **46761** Token（reasoning **40742** 已含在输出），费率估算上界 **$0.28901136**；总模型请求耗时约 **717.50 s**。没有重试、换模型、追加预算。`live-pair` 与这一次续接账本分别保存。

在同等已完成 Milk、其余三目标未完成且已经撤退的共同阶段，两组顶层 actor API 都为 9 次，执行时间 G0 47.73 s、G1 47.46 s（不含 LLM）；没有观察到此阶段的图成本收益。后续完成对象不同，必须分路径分析，不能仅用 API 总数宣称图更优。


### 最终 G0 真实 Vision 回归与交付清单

`g0-final-replay`：沿用已记录的 G0 模型初始/恢复程序，0 新 LLM，原生 2/4（Milk、Bread）、reward 0，93 API / 95 Capture / 134.86 s；未隔离，恢复请求预算结束。Bread 的实际附着补证和放置未被最终修改破坏。所有运行的 `source-postcheck` 均为 changed=[]。

本轮相对开始工作树修改的已有文件：

- `README.md`
- `capx/execution/contract_recovery.py`
- `capx/execution/open_recovery.py`
- `capx/execution/persistent_harness.py`
- `capx/execution/preplanning.py`
- `capx/execution/semantic_capabilities.py`
- `capx/execution/trajectory_execution.py`
- `capx/execution/vision_skills.py`
- `capx/experiments/contract_models.py`
- `capx/experiments/native_tasks.py`
- `capx/serving/launch_pyroki_server.py`
- `scripts/local/harness_probe.py`
- `tests/graph_v2/test_v36_semantic.py`

新增文件：

- `docs/feedback/graph-v36-review.md`
- `docs/reports/graph-feedback-v36-reviewed-repair-2026-11-25.md`
- `env_configs/graph_feedback/development-v36-review.json`
- `tests/graph_v2/test_v36_review_repair.py`

这些是相对本轮起点的清单，不把此前 V34–V36 未提交文件冒认成本轮新增。没有删除原文件。全部原始观测、工具输出与失败保留在实验目录；审计包另做脱敏副本，不改原产物。


当前全部未提交源码及精选脱敏审计记录归档：[capx-v36-reviewed-repair-20261125.tar.gz](../../outputs/audit/capx-v36-reviewed-repair-20261125.tar.gz)。原始 RGB-D/点云与大工具 payload 留在完整实验目录；归档的 record-selection.json 明示选择范围，manifest.json 保留源/导出 SHA256。导出文本路径脱敏可能改变原哈希链文本，应按 source_sha256 对照原记录，不能冒充重算后的事件链。
