# CaP-X V36 开发与实际验收

正式工程 `/home/hanyu/dqb/cap-x-20261106`；分支 `feature/graph-complete-20261120`；起始 HEAD `8052fa7ed3d2d74955a82454d7242a41dc0c9307`。本轮保留 V34/V35 未提交修改，没有提交或推送 Git，没有操作真实机器人、改评分或降低释放/力觉安全阈值。

**结论：部分实现与真实验证完成，完整 V36 验收未通过。** 固定 Seed 0 的新 DeepSeek + 在线 Vision + F/T 运行完成 **2/4（Milk、Cereal）**，原生完整任务成功为 false、reward=0。动态能力确由模型生成、注册和执行；最终摘要核验只证明忠实提取公开工具结果。新共享轨迹底座仍缺真实获准执行正例，Bread 稳定附着及 Can 安全放置未解决。扩大正式评测：**NO-GO**。

全部本轮记录位于 [graph_v36_20261125_175831](../../outputs/model_comparison/graph_v36_20261125_175831)。下文路径简写均相对此目录。运行命令见 [V36 使用说明](../feedback/graph-v36.md)。

## 1. 实际实现与调用链

没有新增平行 Agent 或仿真 Runner。调用链是：

`统一 infer.sh → contract_models → PersistentHarness → 初始/恢复 DeepSeek → 原 Sandbox → 原父进程 Broker → Open API / Skill → 原安全检查与独立物理评分`。

新增能力注册链为：

`capability_tools → 模型提出合同与函数 → register_capability → 实际函数绑定/Schema/AST/副作用校验 → run_capability → 嵌套的同类 Sandbox → 原 Broker → 实际调用和结果节点 → 独立输出/物理条件核验`。

新增或主要修改：

| 文件 | 本轮实际作用 |
|---|---|
| `capx/execution/semantic_capabilities.py` | 持久化、带实现版本的开放能力注册；真实 API 绑定；同一 Broker 执行；拒绝错误合同；成熟度与来源核验 |
| `capx/execution/decision_state.py` | 目标、公开空间关系、名义附着变换、F/T、搜索范围与残差摘要；区分 IK、路径和释放许可 |
| `capx/execution/open_recovery.py` | 初始/恢复共用工具文档；注册工具；可选真实轨迹预规划/消费；明确返回状态 |
| `capx/execution/persistent_harness.py` | 接入注册器与原预算；把轨迹调用纳入接口返回/新鲜观测/验证生命周期 |
| `capx/execution/trajectory_execution.py` | 共享轨迹适配、完整 waypoint 执行、起点/新鲜编码器/收敛/预算检查；并非仅执行终点 |
| `capx/integrations/motion/public_planning.py`、`collision_audit.py`、`pyroki_snippets/_trajopt.py` | 复用已有 PyRoKi TrajOpt；绑定实测起点与非控制夹爪关节；固定刚体组件碰撞对；FCL 网格窄相检查 |
| `capx/execution/preplanning.py` | 精确 IK 缓存增加求解器参数和标定版本，避免不一致条件命中 |
| `capx/experiments/native_vision_skills.py`、`native_tasks.py`、`contract_models.py` | 两组共享可选轨迹底座、可信父进程派发和配置；原生评分不变 |
| `scripts/local/decision_semantic_gate.py` | 冻结公开状态上的真实 DeepSeek、实际 IK 与零新增 LLM 程序重放；无仿真/物理派发 |
| `scripts/local/diagnose_self_collision.py` | 公开 URDF 几何的球模型/网格碰撞诊断 |
| `env_configs/graph_feedback/development-v36.json`、`tests/graph_v2/test_v36_*.py` | 可选开关、版本来源与机制回归 |

`v36-changed-existing-source.json` 对照的是本轮开始时的实际工作区字节，不是 HEAD。源码压缩包包含当前所有未提交 V34/V35/V36 文件；没有把旧工作归为本轮新开发。

## 2. 17 项问题追踪

“机制通过”不等于“物理通过”；以下保留未解决项。

| # | 源码审计结论与本轮处理 | 验收状态/阻塞 |
|---|---|---|
| 1 初始/恢复契约 | 统一 `runtime_api_doc`。真实运行发现 `attachment` 字符串与结构体同名，已统一名称类型，详情移至 `attachment_estimate` | 机制通过；类型修复后尚无新付费端到端复验 |
| 2 名义规划 | 现有 `preview_skill` 实际搜索并保留结果；共享后端 `plan_motion` 实际产生轨迹。兼容 `plan_skill` 返回 `READY_FOR_LIVE_VERIFICATION`，仍允许执行时补证 | 部分；不能把合同 READY 当完整搜索/路径许可 |
| 3 GraspNet 6D | 现有原始候选可诊断，但成熟 acquire 仍采用受限抓取转换；完整 6D 手眼/TCP/夹持适用性未物理验收 | 未解决；未用未标定候选直接派发 |
| 4 IK 诊断 | 实际位置/旋转残差、候选姿态、求解范围可查询；DSE 区分 bounded failure 与不可达 | 机制与真实模型查询通过；错误字段使用仍暴露 AEI 缺口 |
| 5 成熟 Skill 终点运动 | 新增可选全轨迹消费，默认旧路径保留 | 软件通过；新路径未获物理许可，不能宣称已替换全部运动 |
| 6 `/plan` 线性插值 | 没把旧服务接口当碰撞规划；新适配直接调用已有 CPU TrajOpt | 部分；新路径目前传空障碍列表，环境靠独立公开扫掠拒绝，仍非完整环境感知规划 |
| 7 起点/碰撞/持物 | 起点和非控制关节固定；修正固定组件碰撞对，增加成熟 FCL 窄相 | 实际规划/自碰撞检查通过；环境缺测、持物—机器人合同仍阻塞 |
| 8 运动准入不一致 | `pyroki_public` 可同时挂到成熟 Skill 与开放 API | 机制接通；尚无两类路径都实际执行成功的物理正例 |
| 9 单帧自由空间 | 原公开 RGB-D 扫掠和 UNKNOWN 策略保留 | 未解决；1 cm 空手轨迹仍有 869 个新增采样点无可见自由空间证明 |
| 10 抓放联合规划 | 保留 V35 后续放置预览和当前附着后的重验证 | 部分；未形成获准执行的 6D 抓取—持物轨迹—放置联合解 |
| 11 重复搜索 | 精确缓存条件增加求解器配置；输出唯一搜索/实际候选 | 机制通过；真实编码组仍提出 48 个历史同姿态候选，策略层未避免重搜 |
| 12 恢复原语缺口 | 受限撤退、观察机会、重新验证/规划复用已有接口 | 未解决安全暂放/通用重抓/持物脱困；不能用注册创造底层能力 |
| 13 DSE 效果 | 已做真实 DeepSeek Legacy/Encoded 对照 | 没有证明收益；均无可行新放置，Encoded 发生 Sandbox 错误且候选全部重复 |
| 14 固定合同/语义图 | 模型生成全新组合函数与合同，动态注册持久化，记录实际工具依赖 | 部分真实通过；跨新物理 episode/任务的可靠复用尚未验证 |
| 15 返回与效果混淆 | 保留原生命周期；轨迹检查真实关节；语义返回类型不再直接给 VERIFIED | 机制通过；组合语义仅有提取核验，无新的已验证物理能力 |
| 16 放置质量 | 保留目标偏差、边界余量等独立公开指标 | Milk/Cereal 有实际观测；不改 reward，未验证后续可操作性或重新摆放质量 |
| 17 过早退出/无进展 | 原有有界 Harness 保留，多轮新模型恢复确实触发 | 未闭合；最后两次因类型错误无物理进展并耗尽 3 次恢复额度，修复有测试但无新端到端确认 |

## 3. Gate A：物理底座与零 LLM 诊断

### A1 Can

复用 V35 `can-zero-llm/diagnosis.json`、真实公开附着和目标容器证据。Gate B 两组各实际搜索 48 个候选、384 次 bounded IK 调用，均为 `NO_SOLUTION_FOUND`。V35 的倾斜终点 IK 正例仍仅是终点可达；未因此授权持物轨迹或释放。

现有限制是候选姿态、实际附着与路径覆盖共同造成的能力缺口，不能从有限搜索失败断言物理不可达。没有取得 Can 新物理放置成功。

### A2 Bread

`bread-history-public-diagnostics.json` 为公开证据；`bread-history-OFFLINE-PRIVATE-evaluation.json` 严格标为离线独立评价，不输入 Agent。

- 接近位置残差 3.564 mm；一次抬升控制残差 5.87 mm，settle 后 2.287 mm，通过原控制阈值。
- 公开附着注册创新量约 5.128 mm / 17.621°，拟合残差约 2.797 mm，仍不足以确认刚性附着。
- 独立离线真值显示该历史抬升中物体升高约 173.718 mm，同时转动约 12.651°。两个手指有接触，不代表附着变换不变；承载变化同样不能证明无滑移。
- 本轮真实运行再次在 `unconstrained_or_conflicting_registration` 阻断。未降低抓持、力觉或释放阈值；安全重抓能力仍不足。

### A3 Cereal

手写公开 API 诊断 `gate-A3-cereal` 在抓取阶段即被原腕部力保护阻断，没有到达容器几何验收。阻断力矩约 `[4.111,-0.849,-1.214] Nm`，模长超过原 4 Nm 界限。该失败记录保留。

后续真实模型运行中，模型选择不同剩余顺序，经现有确定性续接完成 Cereal 放置，独立评分确认。因此本轮有一个**实际 Cereal 抓放结果**，但不能解释为先前那段手写轨迹已修复，也不能宣称所有缺测容器都可恢复。

### A4/A5 完整轨迹与共享运动底座

| 记录 | 实际检查 | 物理派发 |
|---|---|---|
| `gate-A4-trajectory` | 已有 CPU TrajOpt 对空手上移 1 cm 生成路径；球模型自碰撞最小距离约 -58.218 mm，拒绝 | 0 |
| `self-collision-diagnostic.json` | link6/hand 球包络重叠，但配置网格相距约 29.92 mm；link7/hand 属同一固定装配；原夹爪起点不一致 | 诊断，不派发 |
| `gate-A4-trajectory-refined` | 依据 URDF 固定连接通用排除、实测夹爪固定、FCL 窄相后自碰撞通过；起点误差 1.905e-7 rad；末端残差约 0.0282 mm / 0.0246° | 环境扫掠 869 个未知点，拒绝；0 |

没有以“起始就碰撞”为由屏蔽任意碰撞对；没有把优化器软代价当证明。FCL 只针对配置模型的采样网格检查，不构成连续碰撞证明。持物体与非抓取机器人链接的碰撞合同仍缺失。

`execute_waypoints` 已在机制测试验证逐 waypoint 派发、起点过期拒绝、UNKNOWN 不派发、编码器未更新/未收敛拒绝。**A4/A5 真实完整轨迹执行验收未通过。** 本轮 2/4 运行使用保留的旧 Skill 底座，不能归功于新轨迹后端。

依赖处理：只将 `python-fcl==0.7.0.10` 安装到本轮输出下 `optional-cpu-deps`，没有升级主 `.venv`。复用 [PyRoKi 官方实现](https://github.com/chungmin99/pyroki) 和 [python-fcl 官方实现](https://github.com/BerkeleyAutomation/python-fcl)，没有自研全局 IK/规划器。

## 4. Gate B：真实模型决策表示对照

同一 Can 冻结公开失败状态、`deepseek-v4-pro`、同一工具权限、各 2 次新请求。工具执行是真实 IK；没有物理派发。

| 指标 | Legacy | Encoded |
|---|---:|---:|
| 输入状态字符 | 12,261 | 13,124 |
| 新 DeepSeek 请求 | 2 | 2 |
| 实际 API 调用 | 5 | 3 |
| 唯一新搜索/候选 | 1 / 48 | 1 / 48 |
| IK 调用/精确缓存命中 | 384 / 0 | 384 / 0 |
| 相对历史姿态：新 / 相同 | 20 / 28 | 0 / 48 |
| 程序状态 | ok、ok | policy_violation、ok |
| 新可行路径/物理恢复 | 0 / 未测试 | 0 / 未测试 |
| 模型请求耗时 | 207.17 s | 174.24 s |
| 配置费用上界估计 | US$0.057293 | US$0.049804 |

Legacy 提议的 preferred XY 在容器外，但父进程只在真实容器合法范围搜索。其口头描述不等于实际测试了该坐标。Encoded 首次使用禁止的 `__name__`，第二次修正；随后重复历史的 48 个姿态，不能把调用数减少归因于更好搜索。

历史 IK 缓存没有跨改变后的实测关节参考配置复用，这是正确的条件隔离，并非已证明的缓存失效缺陷。两种 Prompt 来自相同公开来源、可查询同样详情，但不是严格逐事实等量序列化；不作图结构或编码单因素因果结论。**该单例没有支持 DSE 提高规划或物理成功率。** 详见 `gate-B-analysis.json`，完整 messages 与 receipt 在 `model-ledger/requests`。

## 5. Gate C：真实模型动态注册

本轮 4 次真实 DeepSeek 请求：

1. 提议错误 verification_refs 类型，父进程返回 REJECTED。
2. 函数含默认参数，与声明签名不符，返回 REJECTED。
3. 模型生成新 `planner_failure_digest_v1`，组合 `decision_state` 与 `planning_query`，在 Bubblewrap 实际注册/执行。但读取了不存在的 `tested_summary`，错误报告候选数 0。旧版类型检查曾标 VERIFIED；该标签过强，已修复。
4. 收到真实审计反馈后，模型生成 v2，使用实际 `tested`，提出嵌套输出投影；当时注册器仅支持顶层字段，因此拒绝。后续通用修复支持嵌套字段，**未经修改的原模型程序**在零新增 LLM 重放中成功。

最终 `gate-C-delivery-replay` 中，父进程逐字段对照两次实际 API 返回：候选 48、首个位置残差 0.0610041 m、旋转残差 6.55977°，目标列表和可用绑定均一致。状态为限定范围的 `VERIFIED / audited_output_projections`，`physical_verified=false`。仅证明忠实提取，不证明诊断完整或策略有效。

最终成熟度规则：类型正确仅为 EXECUTED；全部输出字段通过独立来源核验才可得到上述 VERIFIED；实际物理效果必须用原父进程新鲜条件验证。更名同一冻结重放目录不构成跨情境复用，不能升级 REUSABLE。旧错误标签/拒绝保留在原事件中，当前查询不再信任类型级 VERIFIED。

实际 DATA 边包括工具返回 ID → 使用该 ID 的调用、验证通过的输出字段 → 来源结果；Semantic 关系仍为模型假设。未知 Python 计算依赖明确标记不完整，没有伪造完整数据流或物理因果边。模型不能写可信条件、扩大权限或绕过动作准入。

| Gate | 证据与限制 |
|---|---|
| C1/C2 | 真实模型产生此前未注册的组合能力，实际运行；可靠输出核验发生在修复后的原程序重放 |
| C3 | 真实错误合同拒绝与修订；虚假工具/错误效果/权限扩张等额外反例为本地测试，未冒充模型实验 |
| C4 | 最终重放实际生成调用/结果/投影来源边并核验输出；不是物理恢复 |
| C5 | 实现版本变化 DEPRECATED、过期证据拒绝有机制测试；旧注册实现在当前工具文档变化后失效 |
| C6 | 持久化检索与不同工具结果的复用有机制测试；未完成新物理 episode/任务的可靠复用 |
| C7 | 新模型实际组合观察、撤退与计划改变后续顺序，但没有验证“新注册恢复能力解决物理异常”；仍未通过 |

## 6. 新模型真实 PickPlace：2/4 的来源

`live-open-g0` 使用原 Seed 0 Snapshot，hash `642d1d5a14b89950ac7941358992a46846002a685018a1f9e56a180af9333184`。一条新初始生成和三条新恢复生成；`max_tokens=32768`、reasoning effort low、每次最多一次请求；所有响应 finish_reason=stop，没有输出截断。

| Actor | 来源 | 实际结果 |
|---|---|---|
| 00 | 新 DeepSeek 初始 | `set.add` 不在当前 Sandbox 允许方法内，执行前拒绝，0 物理调用 |
| 01–04 | 原有确定性续接 | Milk 抓取；首次放置对齐无进展；局部续接完成 Milk；Bread 接近被 F/T 阻断 |
| 05 | 新 DeepSeek 恢复 | 调用受限撤退、读取缺失区域与观察机会；自主改为 Cereal → Can → Bread；Cereal 阶段被力保护中断，但持物证据已验证 |
| 06 | 原有确定性续接 | 完成 Cereal 放置 |
| 07 | 原有确定性续接 | Bread 抓后附着无法确认，停在 closed_unverified |
| 08、09 | 新 DeepSeek 恢复 | 都把 `execution_state().attachment` 字符串当结构体，`.get()` 报错，各只有 1 次查询，无物理修复 |

最终原生 `objects_in_bins=[1,0,1,0]`，**完整任务 0/1 成功，子目标 2/4**；退出原因 `recovery_request_budget`，不是模型 FINISH 或受控退出成功。不能将确定性完成动作全部归因于新模型；也不能用这个单例与旧 1/4 宣称统计提升。

类型混淆随后通过共同契约修复及测试，没有修改已生成程序或原始结果。`set.add` 的 Sandbox 限制未为此放开。最终类型修复尚缺新的付费恢复验证。

独立放置质量来自公开几何估计：Milk 边界余量约 60.386 mm、目标 XY 误差 0.408 mm；Cereal 余量约 38.509 mm、误差 0.529 mm。不是独立真值精度结论，也不证明姿态或后续可操作性。日志 `travel_m=0` 是最终质量计算中的相对量，**不是实际搬运路程**。

实际费用/调用：

- 68 个父进程机器人/API 计数、9 次 Skill、77 capture；真实端到端 747.55 s，其中模型请求合计约 625.32 s。
- SAM3 HTTP 276 次；GraspNet `/plan_point_clouds` 3 次；旧 PyRoKi `/plan` 14 次、`/ik` 129 次。旧 `/plan` 不计作新完整 TrajOpt 验收。
- SAM3 精确重复 0；语义缓存复用 277（跨 capture 269）、几何复用 42、计划复用 1。抓持表面验证 158 次、合同验证 86 次；仍有明显持续验证成本。
- 4 次模型请求：输入 33,316、输出 37,528（含 reasoning 32,753）；配置费用上界约 US$0.192588。

视频：[live G0](../../outputs/model_comparison/graph_v36_20261125_175831/live-open-g0/seed-0-repeat-0/G0/video.mp4)。程序、Trace、图、Belief、F/T、视频时间戳均在同一 episode 目录。`live-perception-audit.json` 对照真实请求像素核验调用归因，`live-placement-quality.json` 保存质量事件。

## 7. 回归、运行环境与版本边界

最终针对性回归 **230 passed / 0 skipped / 0 failed**，包含 V36 注册/投影/真实 Bubblewrap/过期版本与证据/嵌套调用依赖、DSE 类型、轨迹逐点派发，以及 V35/V34/V33、相关 Skill/IK 与 P0 Sandbox/Reward/Provider/physics 子集。另有 2 条上游弃用警告。日志 `delivery-baseline-tests.log`、JUnit `delivery-baseline-tests.xml`（含后补的 Baseline 和官方 LOGIV 核心桥接测试）。早期 195 项结果另行保留。**没有声称重跑全套千余项 Graph 测试或完整 P0 视觉套件。** 初次失败测试和中间验收保留。

固定 V34 历史程序加归档恢复程序，真实 Vision 零新 LLM 重放：`fixed-regression` 为 1/4、36 API、41 capture、77.94 s；接口/数据边修复后的 `final-source-replay` 仍为 1/4、36/41、77.64 s。两次源码运行中 hash 检查均无变化。它们不是新的 DeepSeek 端到端样本。最终成熟度调整仅改语义登记/检索规则，并另做冻结工具重放与回归，不把此前物理结果视作该新增规则的独立物理验证。

本轮新付费运行也是运行中源码 hash 无变化。冻结 Gate B/C 的最早源码副本是在请求后补存的 `gate-source`，因此不声称它具有完整请求前冻结证明；实际 Prompt、响应、候选和调用日志保留。后续源码改动与原运行分开归档。

服务 SAM3、Seed GraspNet、CPU IK 在预检时 OpenAPI 可用，地址从现有配置读取；没有重启其他服务。主机 `nvidia-smi` 报 NVML 驱动/库不匹配，未修驱动。现有 Vision 服务与 CPU IK 可运行；渲染沿用现有 EGL/Mesa 配置。真实硬件品牌/型号仍未确认，本轮没有连接硬件。

`.env.local`、`.env.api.local` 保留，权限均为 0600。完整源码包排除它们、模型、虚拟环境与本轮可选依赖安装。

## 8. 成本、总体边界与下一步

V36 核心验收共 **12 次新 DeepSeek 请求，全部 HTTP 200、无自动重试**。输入 89,694、输出 65,146 Token，其中 reasoning 55,626 已包含在输出中。基于项目现有输入/输出每百万 US$1.32/3.96 的保守配置，估计 **US$0.376374**；不是服务商账单，也没有将推理 Token 重复相加。分组和 response ID 见 `request-accounting.json`，没有保存鉴权头或密钥。

本轮真实主试验只验证 G0 开放共同底座；没有可用于归因 G1 图收益的新模型 G0/G1 配对。动态能力、状态编码与可选运动底座均不按 G0/G1 独占分派。不能把新增工具能力或额外输入信息算作图收益。

下一步优先顺序：

1. 用已统一的工具返回类型复验 Bread 恢复，先减少接口错误耗尽预算；同时对当前 Python 允许子集做生成前明确约束，不改写失败模型输出。
2. 针对真实 Bread 附着变化补齐经验证的安全重抓/暂放能力。载荷和视觉必须共同提供证据，不能让任一单独替代刚性关系验证。
3. 完善公开环境多帧自由空间及持物—机器人碰撞模型，取得真正完整路径执行正例后再切换共享后端。不要靠清空未知体素或略去手臂体积获得许可。
4. 对真实 6D GraspNet 变换与实际附着标定，再做抓取—搬运—放置联合搜索；Can 终点 IK 正例不能越过路径许可。
5. 完成 C6 新上下文能力检索复用和小规模等信息模型对照，再决定是否扩大评测。

目前证据支持的是：开放注册/来源核验可运行；新模型能尝试改变合法操作顺序；部分真实任务可继续推进。**尚不支持通用物理恢复闭合、完整 4/4、成熟共享轨迹底座已通过或图方法优越。**


## 9. 后续 Baseline 补验与最终接口修复

根据用户追加要求，复用了 V29–V31 的统一入口与外部适配，没有重做实验 Runner。详细版本区分、真实代码/感知/物理证据及复跑命令见 [Baseline 兼容性报告](v36-baseline-compatibility-2026-11-25.md)。

- 新增 `native_b0.py` 最小 PickPlace 环境适配，直接调用原 `graph_v2` B0 完整程序＋程序级反馈。统一入口保留 B0 独立程序、LOGIV 官方策略、G0/G1 原策略，禁止静默将 V36 open 配置套给 legacy/LOGIV。
- B0 同一 Seed 0 初始物理状态，新 DeepSeek 初始＋恢复共 **2 请求**；第一条因 Sandbox 语法子集拒绝，第二条真实执行 43 API、四轮抓放，原生 **0/4**。实际目标箱定位落在 Milk 上，低置信度结果未被原 API 拒绝；Milk/Can 有实际抬升，Bread 未抬起。不是新增 Graph 阻断导致该次失败。
- 当前源码上 G0/G1/LOGIV 适配复用 V30c 同一历史程序、同一 Snapshot，真实 Vision 零新 LLM 重放均为 **1/4**；API/capture 分别 **45/62、38/49、24/36**。三组恢复行为不同，分别阶段预算耗尽、夹爪不收敛、官方 UNRESOLVED，不能因更早退出或调用少称为优势。
- 上述 B0 不是官方逐字复现；LOGIV 是官方核心＋共同 Skill 适配；ASPIRE 与 CaP-Agent0 完整方法仍未接入，未计为失败参赛方法。

新增预算单独记录：B0 输入 3493、输出 15486（含 reasoning 13129），配置费用估计上界 US$0.06593532。**连同核心验收，本轮合计 14 请求、输入 93187、输出 80632、reasoning 68755（已含于输出），费用估计上界 US$0.44230956。** 不是账单，未修改既有 12 请求账本。见 `all-request-accounting.json`。

真实父进程能力注册验收暴露并修复：初次 capture 前 DSE 读取不存在的 scene.obs；Prompt 与运行策略开关不一致导致 register_capability 未注册；日志 kind 参数冲突。每项均有先失败后通过的回归。最终 `harness-registration-native-policy` 实际两次公开查询、嵌套 Sandbox、Broker 与输出投影核验通过，无物理动作、无新 LLM。它是手写接口探针，不替代 Gate C 的新模型结果，也不证明物理恢复。

Seed 7 另做固定历史程序真实 Vision/F/T 重放：G0/G1 均 0/4，API 都为 1，capture 2/5，8.74/9.90 秒，`Missing precondition evidence` 后因零新 LLM 预算退出。首次配置将 episode_seconds 误设为 300，与冻结 1200 不一致，被严格 Snapshot gate 拒绝；失败记录 `seed7-historical-pair` 保留，修正配置后另存 `seed7-historical-pair-budget-matched`。不把这些重放当作新模型泛化或图收益。

三组兼容性重放和 B0 新模型运行均通过运行中源码 hash 检查。之后只新增统一入口对不兼容配置的提前拒绝及文档；最终 230 项相关测试通过。中间一次测试与实际实验争抢原全局锁失败，待实验结束重跑通过，未改测试或锁机制。没有新启正式批次。

完整源码与脱敏记录导出至 `outputs/audit/capx-v36-20261125-final-source-and-records.tar.gz`，包含当前所有未提交源码；凭据、虚拟环境、模型和隔离可选依赖目录排除。打包清单保留原文件 hash 与脱敏后的 hash；原实验文件不改写。**最终仍为 NO-GO：未通过完整 4/4、共享全轨迹物理执行正例、稳定 Bread 恢复，以及公平的外部方法正式评测。**
