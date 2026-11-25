# CaP-X V30 开发与外部方法接入报告

## 结论与交付范围

本轮实现了可组合的阶段恢复，以及 LOGIV 官方核心的共同 Skill 适配，并运行真实 DeepSeek＋Vision 开发实验。已有自然 TCP 失败后的**实际撤退、换候选、重新抓取抬升**证据；尚未把该失败转化为完整 PickPlace 成功。不能把局部动作成功写成整项恢复成功。

外部方法比较作为后续主实验，G0/G1 及图机制消融为辅助。当前只有 LOGIV 适配具有模型配对运行结果；CaP-Agent0、ASPIRE 原版和 LIBERO-Recovery 上的方法对比尚未完成。**不具备扩大正式评测的条件。**

正式目录 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，基准 HEAD `9c7813f9cf6bb98cf4dd633b310fcc4d59233c44`。保留原 V29 未提交修改，未提交或推送本轮代码。已有修改备份、差异和新结果均在：

`outputs/model_comparison/graph_v30_20261124_185019/`

该目录包含源码冻结字节、Snapshot、完整日志、原始 HTTP 计数、模型 request/response/receipt、物理 Trace、视频、测试 XML 和所有失败。旧输出目录没有用于本轮写入。`preservation-audit.json` 区分继承且未变的文件与在已有修改上扩展的文件。

## 共同技能与图修复

| 原有缺口 | 本轮实现 | 实际限制 |
|---|---|---|
| 接近 TCP 不收敛后退出 | 保存接近阶段；确认空手及当前可见撤退走廊；退到可达的原接近位置；排除失败姿态后重抓 | 只覆盖已观测的竖直接近范围，不是任意路径避障 |
| 配准变化使旧附着关系失效 | 当前表面拟合只产生新假设；等待夹爪收敛；通过小幅实际运动、相对位姿与夹爪变化验证；失败回滚 | 真正的滑动或旋转变化仍阻断；本轮没有取得真实附着重估成功 |
| 新技能的必要证据不在原 intent 中 | 条件回归发现技能库的前置条件，先请求必要证据 | 不把 UNKNOWN 改成 TRUE；无能力/无进展有界退出 |
| 撤退后才能重新取得抓取计划 | 规划可以包含物理修复前缀及显式后续补证义务，恢复时使用新 ID | 条件性计划不是许可；Broker 仍检查全部实际前置条件 |
| 深度 GraspNet 无候选 | 复用既有对象＋场景点云接口，增加有限多视角 Fallback 和 Seed 回显校验 | 实际触发时仍返回 0 候选；没有证明它改善成功率 |

代码主要位于 `capx/execution/skill_recovery.py`、`vision_skills.py`、`causal_plan.py`、`capx/experiments/native_vision_skills.py`、`contract_models.py` 和 `capx/integrations/vision/graspnet.py`。原 V29 Skill 可通过 `--v29-skills-control` 对照；该对照使用当前共同调度器，不代表整个历史 V29 系统。

新恢复最多每对象、每阶段两次，并计入已有调用与时间预算。机器人状态、当前 RGB-D/语义点云和执行阶段为在线依据；物理真值只用于独立最终评价。新代码未增加任务名、对象名、颜色、Seed 或故障标签分派动作的规则。原评分、释放阈值、Sandbox 和父进程准入未改。

GraspNet 候选仍经过原有 CGN→Panda 轴约定、直立姿态投影、宽度、碰撞和 IK 筛选。当前控制底座没有直接消费任意候选平移/6D 姿态；不是已经接入了通用多姿态抓取器。新增阶段能力仍是开发能力，不能等同于上游论文已验证的完整 Skill。

## 上游复用与外部方法

| 来源 | 固定 revision / 实际工作 | 状态 |
|---|---|---|
| [LOGIV](https://github.com/Enqiao-Lu/LOGIV) | `0ea7a7ff765016a33409c54572d733bed4503d4d`；直接导入未修改的 symbolic、graph、repair、execution、verification | 共同 Skill 适配已运行；不是原论文复现 |
| [CaP-Agent0 / CaP-X](https://github.com/capgym/cap-x) | `acac62fcb1b6f9e02975004a2c729a65e11bffed`；读取官方配置和图像差分、技能库、集成机制 | 未完成端到端方法接入；没有将 B0/G0 重命名为 Agent0 |
| [ASPIRE](https://github.com/NVlabs/ASPIRE) | `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`；核对原始 graspnet/curobo client，复用点云协议及既有隔离 LIBERO 环境 | 原版 coding-agent 技能发现/评测流程未复现；不能将协议复用算成完整 ASPIRE |
| [cuRobo](https://github.com/NVlabs/curobo) | `163e335cbfc559b6cec48dbcb9d82e4375ea2f70`；检查接口，真实发送公共点云＋七关节规划请求 | HTTP 500 / CUDA Error 804；没有派发规划轨迹 |
| [MoveIt Task Constructor](https://github.com/moveit/moveit_task_constructor) | `8762faeab116c4807a097fb481faaf7960edb3ad`；读取 container/Fallback/test 实现 | 借鉴阶段、候选与 Fallback 思路；没有安装或运行 ROS/MTC |

`upstream-lock.json` 保存来源和本地状态。ASPIRE 本地已有其他工作留下的未提交修改，未触碰它们；本次参考的两个 provider client 与其 HEAD 字节一致。CaP-Agent0、LIBERO-Recovery 是部分 checkout，缺省文件在 Git 状态中表现为删除，不是我们修改了上游实现。

LOGIV 适配 `capx/experiments/external_logiv.py` 保留官方执行/验证/局部修复算法，用共同的公开 Skill 验证器代替原感知后端，用有限技能搜索、符号验证代替 FastDownward/VAL。DeepSeek 初始程序、Snapshot、底层动作与独立评分与其他组共享。它不复现原 VLA/WAM、GPT-4o、原仿真任务或原训练成本；报告名为 `LOGIV-official-core-common-skills-adapter`。

接口调试发现并保留两类失败：最初 Observation 字段名错误；其后把瞬态工具产物作为持续物理事实导致握持后 UNRESOLVED。后者通过明确区分工具条件与物理谓词修正，工具条件仍在共同 Broker 中逐次验证，未删除安全检查。不同适配版本结果分开存放，不能拿早期接口失败充当方法劣势。

CaP-Agent0 原配置依赖图像差分、技能合成/技能库以及并行多模型集成。将其改成单个纯文本 DeepSeek 重试循环会改变方法；还需接入其合法图像后端、计量技能准备与集成成本，并对齐 Skill/Broker 边界。ASPIRE 原版技能发现由 coding-agent 工作流组织，也不能通过执行一份上游专家程序假装完成新的模型实验。两项属于未完成工作，不是零分参赛方法。

## 公开恢复协议与长程环境

使用 [LIBERO-Recovery](https://github.com/liulin815/LIBERO-Recovery) revision `832d54dfc43f0c9e4240bb8d92f5fda7001bb093`。公开 ModelScope 索引实际有 4,803 行，其中 LIBERO-Long (`libero_10`) 726 行。按固定随机种子在排序后的 Long 条目中先抽取两个，选择后才测试；只下载 BDDL 与状态文件，约 56 KB。数据 revision、索引 SHA 和每个文件 SHA 见 `recovery-scenes/manifest.json`。

两个场景分别为“双罐放入篮子”和“黑碗放入底层抽屉并关上”。使用现有 ASPIRE `.venv-libero`，不修改主环境依赖。两场景各一次开夹爪、一次闭夹爪，共四次原始状态恢复＋官方扰动＋20 步夹爪初始化通过；`libero_10_swap` 正常任务 0 环境重置通过。**没有执行策略、没有方法成功率。** 官方默认 30 场景、每场景 5 closed＋5 open，本轮不是完整协议评测。

协议审计在所用 Robosuite/LIBERO 版本上确认索引错位：`MjSimState.flatten()` 是 `[time, qpos, qvel]`，官方 `perturb_object_xy()` 却直接使用 qpos 地址索引 flat state，没有加时间前缀。执行未修改的官方函数和状态类的反例：预期 XY 增量 `[-0.0169477, 0.0111968]`，实际 XY 为 `[0.0111968, 0]`，前一 qpos 被加上 `-0.0169477`。原始数组未改变。

证据为 `recovery-state-layout-audit.json`，可用 `scripts/local/audit_recovery_state_layout.py` 复跑。这是确定的本机协议兼容性问题，不推断所有上游版本都相同；没有静默修改官方协议。本轮不以这种恢复初始化计算方法排行榜。正常 LIBERO-PRO/Long、官方原协议、经确认的修正版协议需要分别报告，先锁定版本与语义，再接策略。

## 实验与成本

最终适配版本 `online-v30c-normal0` 使用一个新的真实 DeepSeek 初始程序，三组共享代码 hash、Snapshot hash、Skill、感知、独立评分及总执行上限。模型为 `deepseek-v4-pro`，输出预算 32768，reasoning effort 为 low。没有用 Oracle 冒充模型；常规恢复由确定性控制器处理，未追加恢复 LLM 请求。

| 方法（Seed 0） | 完整成功 | 完成子目标 | API / capture | SAM3 / GraspNet / IK HTTP | 含初始生成端到端秒 |
|---|---:|---:|---:|---:|---:|
| LOGIV 官方核心适配 | 否 | 1/4 | 24 / 36 | 136 / 2 / 27 | 52.83 |
| 新共同 Skill＋强 G0 | 否 | 1/4 | 45 / 62 | 211 / 6 / 78 | 87.17 |
| 新共同 Skill＋G1 | 否 | 1/4 | 38 / 49 | 171 / 4 / 50 | 70.29 |

LOGIV 完成首物体抓放后，在下一次接近 TCP 不收敛处停止。此时公共验证器无法确定部分效果（例如物体是否不在目标中，区域墙面证据不足）；官方执行器对 UNRESOLVED 直接退出。新增恢复库的状态已通过官方 `tracked_atoms` 更新；机制测试证明它可在条件已解析时搜索撤退并继续，但本次真实场景没有进入那个成功分支。它耗时较低的主要原因是没有继续执行物理恢复，不能据此认定效率更好。

G0/G1 都从同一自然接近失败中执行了 `retreat_empty`，重新计算候选、抓取并抬升。G1 通过条件回归组合 `retreat_empty → acquire`，继续保留 deposit 及其他对象的未执行计划；后续为 held UNKNOWN 选择 `confirm_attachment`。G0 也拥有相同能力。两组均未新增已完成目标，未重做已完成的首个物体。

G0 进入了第二次附着探测，G1 在第一次夹爪收敛不足后有界退出。因此 45→38 API、211→171 SAM3 的差异不能全部解释成图复用：一部分来自不同的恢复尝试/退出时点。当前没有完整任务恢复率提升证据。

### 固定开发集与内部消融

`online-v30b-normal0/7/21` 共 13 episode、3 次新初始生成。G0/G1 公共实现与最终版本相同；这一版 LOGIV 尚未补齐恢复库专有条件的 tracked-state 映射，因此外部方法结果保留在其版本中，不合并进最终版本成绩。

| 场景 | G0 完成目标 | G1 完成目标 | 主要阻塞 |
|---|---:|---:|---|
| 0 | 1/4 | 1/4 | 接近恢复后附着验证不通过 |
| 7 | 0/4 | 0/4 | 邻物/实例几何缺失，无法建立抓取计划 |
| 21 | 0/4 | 0/4 | 容器/区域几何前置条件不足 |

两组完整成功均为 0/3；这是开发集描述，场景曾用于开发，不能计算成未见测试性能。V30b LOGIV 也为 0/3；Seed 7 的瞬态工具条件持续不足导致其多次重规划，直到原执行预算结束（24 次 API、73 capture）。此结果同时受适配后瞬态证据表达影响，不可直接外推原论文。

Seed 0 的辅助消融采用同一初始程序和 Snapshot：

| 组 | 子目标 | API / capture | SAM3 / GraspNet / IK | 端到端秒 | 实际恢复 |
|---|---:|---:|---:|---:|---|
| G1 | 1/4 | 38 / 49 | 171 / 4 / 50 | 71.69 | 撤退、重抓；附着验证退出 |
| G1-no-causal | 1/4 | 38 / 49 | 171 / 3 / 44 | 70.00 | 同样的物理恢复 |
| G1-no-local-repair | 1/4 | 24 / 28 | 136 / 2 / 27 | 52.21 | 原接近失败处退出 |
| G1-no-cost | 1/4 | 38 / 53 | 171 / 3 / 44 | 70.58 | 同样的物理恢复 |
| V29-Skills-G1 | 1/4 | 24 / 31 | 136 / 2 / 27 | 53.09 | 缺少新物理恢复能力 |

局部修复确实改变了实际物理动作；但自动因果构图没有提升该顺序任务的结果，还多出 1 次 GraspNet 和 6 次 IK。代价感知合并补证将 capture 从 53 减到 49，却没有降低 SAM3，且完整 G1 的 GraspNet/IK 更多。计划补证 scope 和实际工具请求均有记录，不能把 capture 减少泛化为工具总成本下降。当前不支持图机制整体优越的结论。

### 附着拒绝的离线反例

`online-v30b-normal0/G0` 的第二次附着探测，实际 TCP 位移约 8.75 mm，公开估计相对位置差 1.40 mm，旋转不一致 **13.30°**，因而拒绝。离线读取相同 `goto_pose` 的前后物理 Trace，物体世界坐标旋转仅 **0.030°**，向上位移 **8.715 mm**；详见 `attachment-offline-diagnostic.json`。

这强烈提示姿态配准/旋转可观测性可能造成误阻断，不能直接称为真实滑移。世界旋转不等于 TCP 相对滑移，现有该事件未保存完整同步 TCP 前后变换，因此仍不能仅凭上述数值最终归因。真值分析没有进入在线反馈，没有调整阈值来刷通过。当前阻塞不仅是“缺少运动技能”，还包括无法可靠验证已有抓持。

### 全部开发开销与版本隔离

本轮共 7 个在线开发批次、**30 个 episode 记录、7 次真实 DeepSeek 请求**；包含早期接口错误和不同适配版本，不能视作 30 个独立实验样本。最新版本只有上面的 3 个配对 episode。早期异常导致 Snapshot 字段缺失的那一条保留为工程失败，不纳入有效配对。

独立请求合计输入 **16,331 tokens**、输出 **4,029 tokens**（其中 reasoning **3,464**）；按现有项目记账配置估算上界 **USD 0.037512**，不是服务商账单，缓存计价可能降低实际费用。每组报告可归属一次共享初始请求，但总费用按请求路径去重，只计 7 次；额外恢复 LLM 请求为 0。本轮未做技能学习/模型训练，新增训练成本为 0，上游已有技能开发成本未测量。

`experiment-accounting.json` 保存逐批统计与逐请求 receipt。历史程序实时 Vision 重放、P0 Oracle、纯机制 Mock、公开场景初始化和真实 DeepSeek episode 分别标记，没有混算。每个完成批次的 `source-postcheck.json` 均无源码漂移。

## 回归、保护与未解决项

首轮全量 Graph/P0 为 1236 passed / 1 failed / 0 skipped，唯一失败是测试输出父目录不存在；没有失败的控制/评分断言。更正输出路径后，真实渲染、V29 条件规划及新恢复/外部接口的 42 项补验通过。P0 首轮 68/68 通过。初期红测试和接口失败也全部保留，最终统计见下方补验记录。

最终代码完整复跑：**1239 passed / 0 failed / 0 skipped**，255.02 秒，其中 P0 **68/68**、Graph 1171 项。日志 `full-final-tests.log`，JUnit `full-final-tests.xml`，真实视觉/Oracle/Sandbox 产物在 `final-acceptance/`。这些回归证明软件路径与原验收没有破坏，不能代替上面的真实模型任务成绩。

新增测试覆盖：失败接近不派发 close；附着假设不能作为许可；实际探测失败回滚且不释放；每阶段尝试有界；新技能库补证；撤退后重建工具证据；旧视觉服务 Seed 不回显时拒绝；官方 LOGIV 在多对象瞬态证据失效时仍能推进；官方修复器获得当前恢复条件后能从合成接近失败中组合撤退并完成后续技能。最后一项是 Mock 机制正例，不是物理恢复成功。

本轮修改/新增文件：

- 执行与服务：`capx/execution/skill_recovery.py`、`vision_skills.py`、`causal_plan.py`；`capx/integrations/vision/graspnet.py`。
- 复用的实验入口：`capx/experiments/contract_models.py`、`native_vision_skills.py`；新增 `external_logiv.py`；`env_configs/graph_feedback/development-v30.json`。
- 工具：`scripts/local/analyze_contract_models.py`、`replay_skill_recovery.py`、`fetch_recovery_scenes.py`、`libero_recovery_gate.py`、`audit_recovery_state_layout.py`。
- 测试：`tests/graph_v2/test_v30_skill_recovery.py`、`test_v30_external_bridge.py`。
- 文档：根 README、`docs/feedback/graph-v30.md`、本报告。

`graph_executive.py`、`skill_contracts.py` 等原有 V29 未提交修改仍在，但本轮没有在它们上再次改动。`git diff --check` 通过；新产物中扫描了 6,050 个文本文件，没有发现已加载的三项密钥值泄漏。

`.env.local` 和 `.env.api.local` 仍为 0600 且被 Git 忽略；没有改密钥配置、依赖版本、训练或模型文件。NVIDIA NVML 仍报驱动/库不匹配，现有 SAM3/GraspNet 服务可用；软件 EGL＋已有 CPU PyRoKi 完成仿真。没有停止其他服务或修复系统驱动。

尚未解决：

1. 自然失败恢复尚未转为完整任务成功；视觉相对旋转不一致、夹爪未收敛和区域几何缺失依然阻断。不能把视觉拒绝直接解释成真实滑移。
2. cuRobo 不可运行，当前能力只覆盖受限接近走廊；没有成熟全臂碰撞规划、通用侧抓、重抓前临时放置等完整能力。
3. LOGIV 适配仍有原版后端替换和细粒度证据语义差异；不能将小样本开发结果外推为论文优劣。
4. CaP-Agent0、ASPIRE 原版/完整适配尚无公平端到端成绩，技能学习成本也未重测。
5. LIBERO-Recovery 只完成数据/环境/协议诊断，尚未接入本轮方法；协议索引问题需先版本化解决。正常 Long 也只有重置验收。

下一步应优先解决附着验证的可观测性问题，并恢复可用的碰撞规划后端处理独立的接近失败；继续完成外部方法的公共状态适配与公开恢复协议确认。保持外部方法为主实验，不以更多自造故障替代公开恢复场景。现有证据支持继续小规模能力开发，**不支持扩大正式评测或声称图系统优于外部方法**。

复跑命令见 [V30 使用说明](../feedback/graph-v30.md)。
