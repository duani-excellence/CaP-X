# CaP-X Stack 分层执行反馈与真实 Vision 消融实验

日期：2026-11-19。正式目录：`/home/hanyu/dqb/cap-x-20261106`。分支 `feature/layered-feedback-20261119`，基于 HEAD `823b878e2dc010e3ec0f3683c9f846ffb97fa17c`。本轮修改未提交；已有 Teacher、多任务、Web 等未提交工作保留，没有把它们当成本轮新增。

**最小系统已实现并实际运行。5 个独立评估场景中，A/B 为 2/5，C/D 为 3/5，E 为 5/5。L1 有一例直接收益；L2 一例改善、一例退化，净收益为零；L3 在两例中用 3 次真实重新规划恢复成功。历史调试 Seed 0/21 的 E 仍均失败。该结果支持继续研究，不支持宣称高可靠性或总体 100% 成功率。**

## 1. 实现与复用边界

复用现有 `pilot.environment/prepare/load_scene`、`AuditedStackEnv.step`、Bubblewrap RPC、Franka Vision API、DeepSeek Provider、原多轮决策提示词及解析器、独立 `PhysicalCriteria`、请求审计与视频时间戳。已阅读原 Vision Oracle 诊断；它的几何偏差与撤退问题作为诊断依据，没有把历史 Oracle 程序当成本轮模型输出。

新增监控挂在父进程 API broker，程序仍整块执行，按实际 API/技能边界检查，不按 Python 行数切分。`ExecutionInterrupted` 在父进程终止 actor，模型的 `try/except` 无法绕过；物理场景保留，恢复块使用新 actor，明确告知变量可能丢失。超时/服务异常沿用 P0 重建路径。未安装监控时保留原执行模式。

| 层 | 当前实现 | 局部恢复与限制 |
|---|---|---|
| L0 | SAM3/GraspNet/定位输入输出留档；有限数值、四元数、尺寸检查；SAM3/OBB 与新 RGB-D 颜色表面的 XY 一致性 | 无效或显著分歧最多重感知一次。复用同一估计器不构成独立确认；完整尺寸偏差仍会漏检 |
| L1 | `goto_pose` 目标/实际 TCP、关节、夹爪；位置 25 mm、姿态 20° 容差 | 控制重试一次，仍超差则中断；真实接触传感器不可用，标记 unavailable |
| L2 | 空夹/持物丢失线索、视觉抬升检查、Place 检查、释放前守卫 | 可见 XY ≤8 mm，间隙 −4～16 mm 才允许释放；最多 2 次有限局部对齐，缺证据 UNCERTAIN 并阻止释放 |
| L3 | 最近检查、异常类型、公共机器人状态、RGB-D 表面证据、恢复历史及剩余预算进入既有多轮上下文 | DeepSeek 决定结束或产生恢复代码；最多 2 次额外请求。FINISH 只记录声明，不能设置 reward |

所有层输出 PASS/FAIL/UNCERTAIN、指标、原因、建议动作。局部动作也扣除 episode API 预算。TCP 使用机器人基座米制坐标、WXYZ；在线接口只接收 RGB、Depth、标定及机器人编码器状态。仿真物体位姿、接触真值、成功标志仅进入独立 `feedback_evaluation`/原物理 Trace，未进入模型提示词。

第二种表面估计是不同算法，不是独立传感器：仍共享相机、标定、遮挡，且当前依赖 Stack 红/绿颜色。没有实现通用物体重建、完整技能前置条件、API 内部连续碰撞监测或释放后的安全撤退控制；本轮结果不能外推到真实机器人。

## 2. 实际环境与测试

- Python 3.12.13；PyTorch 2.8.0；MuJoCo 3.5.0；Robosuite 1.5.1；JAX/JAXlib 0.4.29。没有升级依赖、下载模型或训练。
- NVIDIA 内核 580.173.02 与 NVML 580.178 不匹配，`nvidia-smi` 返回错误，无法可信读取显存。没有修驱动或重启已有 GPU 服务。
- 原 8114/8116 离线。复用已加载 SAM3 18114、支持 Seed 的 GraspNet 8215，实际推理可用。用 Mesa 软件渲染每一步的新 RGB-D，不是冻结输出替代在线感知。
- ASPIRE 18116 的 API 兼容，但 IK 算法不同；首次回归 91 通过、1 失败，原 Seed 21 privileged Oracle 未成功。保留该失败记录，没有修改测试或评分。
- 启动独立的**原 CaP-X IK 求解器 CPU 实例**，OS 动态分配端口，配置见 [ik-runtime.json](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/ik-runtime.json)。相同历史 Snapshot/Oracle 恢复成功，随后原 P0 60 + Provider Mock 8 全通过。原 SAM3/GraspNet/ASPIRE 进程 PID 保持不变。
- 新反馈测试 **28/28 通过，0 skipped、0 failed**：包括隐藏字段过滤、失败后 actor 中断不可捕获、同一环境恢复、空夹、释放阻断、有界修正、总 API 预算、reset 清理、解析与 reasoning 分离、离线标签计算。相关测试先失败后实现的日志保留。
- 完整回归那次为 **92/92**（68 原有 + 当时 24 个新增）；新增测试最终独立重跑为 28/28。不能把两次有重叠的测试数直接相加成 120 个独立测试。
- 真实 P0 视觉 Seed 回显/视频、Oracle 配对、Snapshot、Reward Worker Mock、Sandbox 均经过回归。另有 3 个冻结 RGB-D 检查的离线重复验收，单独标记，不计入在线成功率。

JUnit：[完整回归](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/regression-original-ik.xml)，[最终反馈测试](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/feedback-final-28.xml)。配对及保密检查：[delivery-audit.json](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/delivery-audit.json)。35/35 episode 的同 Seed Snapshot、初始 Prompt、API 文档、初始代码 hash 一致；4 个核心运行文件在实验期间未变化。扫描 1,142 个 JSON/JSONL/日志文件，未发现配置密钥、Authorization 鉴权头或选定评价私有字段进入模型 prompt；该扫描同时由白名单边界测试支撑。两份环境文件权限仍为 0600。

## 3. 试验设计和真实结果

历史调试 Seed：0、21，复用旧 Pilot Snapshot。独立评估 Seed：1009、2027、4093、6151、8089，与检索到的既有场景清单无重合；通过现有流程静置、有效性检查及状态 hash 校验。这里的“未见”指本项目先前场景清单，不代表模型预训练分布未知。

每个 Seed 只请求一次真实 DeepSeek 初始程序，A–E 共享该程序，避免随机代码生成差异混入反馈消融。A 是既有 P0 Pilot 的单程序基线，不代表所有历史 Web 多轮模式。E 额外调用属于 L3 处理；没有“相同重试次数但仅给通用错误文本”的对照，因此尚不能把 E 的收益完全归因于结构化反馈文本本身。

统一配置：`deepseek-v4-pro`，`reasoning_effort=low`，输出总预算 32,768 token（含 reasoning），单请求 400 s，不自动重试；每代码块原限时 90 s；episode 最多 64 次 API（含局部恢复）；最多 3 块，L2 每次释放最多修正两次。本轮批次总请求上限 21、保守费用上限 $5、墙钟上限 3,600 s。实际批次 2666.9 s，35 个 episode 完成，0 基础设施错误。

模型能力/格式按官方文档及实际响应核实：Structured Vision 通过 API 返回文本几何，不是模型直接看图；reasoning 与 content 分开保存，只解析 content。本批次 14 次请求全部 HTTP 200、一次尝试、模型名匹配、`finish_reason=stop`，没有截断程序。响应 ID、Token 与耗时都留档；缺少 header request ID 时保留 response ID，不伪造请求 ID。[DeepSeek thinking 文档](https://api-docs.deepseek.com/guides/thinking_mode/)

| Seed | A 原始 | B +L0 | C +L1 | D +L2 | E +L3 |
|---|---|---|---|---|---|
| 1009 | 失败 | 失败 | 失败 | 成功 | 成功 |
| 2027 | 成功 | 成功 | 成功 | 失败 | 成功 |
| 4093 | 失败 | 失败 | 失败 | 失败 | 成功 |
| 6151 | 成功 | 成功 | 成功 | 成功 | 成功 |
| 8089 | 失败 | 失败 | 成功 | 成功 | 成功 |

| 组 | 最终物理成功 | 控制/感知 API 总数 | 工具 HTTP 总数 | 额外 LLM 请求 | 平均执行秒数 | 平均独立运行估计秒数 |
|---|---|---|---|---|---|---|
| A | 2/5 (40%) | 56 | 51 | 0 | 21.1 | 53.7 |
| B | 2/5 (40%) | 56 | 51 | 0 | 21.4 | 53.9 |
| C | 3/5 (60%) | 57 | 52 | 0 | 21.8 | 54.3 |
| D | 3/5 (60%) | 62 | 59 | 0 | 23.0 | 55.5 |
| E | 5/5 (100%) | 131 | 124 | 3 | 36.3 | 207.8 |

API 总数包括模型调用的控制/定位函数及自动恢复；工具 HTTP 为 SAM3/GraspNet/IK 实际请求，不等同于 API 函数数量。执行秒数不含等待 LLM；“独立运行估计”将共享初始代码的生成时间加回各组，包含环境/视频开销，不是隔离硬件基准。

历史调试另计：Seed 0 A/B 成功、C/D/E 失败；Seed 21 全组失败。因此全部 7 场景的 E 是 **5/7**，不能写成所有场景全成功。旧 ASPIRE IK 的不完整 smoke-v1/v2 不混入以上效果统计。

## 4. 技能、检测及恢复指标

| 组 | 至少一次稳定抓持 | 抬升动作成功/尝试 | 释放后曾稳定/释放次数 | P0 累计 slipped episode |
|---|---|---|---|---|
| A | 5/5 | 6/6 | 2/5 | 0 |
| B | 5/5 | 6/6 | 2/5 | 0 |
| C | 5/5 | 6/6 | 3/5 | 0 |
| D | 5/5 | 6/6 | 4/4 | 0 |
| E | 5/5 | 11/11 | 6/6 | 1 |

“稳定抓持”沿用 P0：抬升超过 50 mm、双侧抓持持续 0.5 s。抬升动作以请求上移 >50 mm、物体实际上移 >35 mm 且有接触统计；包括恢复程序再次抬升。“释放后曾稳定”沿用原物理稳定标准，但不保证后续撤退不会破坏，因此 D 为 4/4 曾稳定，最终只有 3/5。闭爪瞬间接触不足不直接算整个 Grasp 失败。E 的累计 slipped 标记来自恢复过程中，最终仍成功；不等于新增一个最终失败。

| 组 | 检查 | TP | TN | FP | FN | UNCERTAIN | 已决标签准确率 |
|---|---|---|---|---|---|---|---|
| B | get_object_pose | 0 | 0 | 0 | 10 | 0 | 0.0% |
| C | tcp_tracking | 1 | 21 | 0 | 0 | 0 | 100.0% |
| D | pre_release | 7 | 0 | 2 | 4 | 0 | 53.8% |
| E | get_object_pose | 0 | 3 | 0 | 32 | 6 | 8.6% |
| E | tcp_tracking | 4 | 40 | 0 | 0 | 0 | 100.0% |
| E | pre_release | 13 | 0 | 2 | 6 | 0 | 61.9% |
| E | place | 3 | 0 | 5 | 0 | 0 | 37.5% |

标签是**离线诊断**，需按以下含义理解：

- L0 真值异常：中心 XY >25 mm 或任一完整尺寸误差 >20 mm。B 的 10 次均因未覆盖的几何偏差漏检；其 PASS 只证明有限几何/XY 一致，不能作为完整位姿可信证据。
- L1 标签为真实 TCP 超出同一位置/姿态容差，因此 100% 主要证明观测/坐标转换与监控一致，不证明“所有危险动作均能检测”。Seed 0 的中断政策仍伤害了成功率。
- pre-release 标签为 XY >10 mm、AABB 下表面间隙 >20 mm 或 <−6 mm、或无抓持接触的**几何风险代理**。这些不是“释放一定失败”的真实标签；例如 Seed 1009/4093 成功释放也可能被这个代理标为风险。不能把表中 FN 直接宣称真实危险漏放率。当前只对 Seed 1009 补做了轴向因果对照。
- place 标签用检查时刻原始物理成功，最终结果仍用未修改的 P0 稳定判据。E 有 5 个误报，显示可见面验证会拒绝实际已堆叠的场景。UNCERTAIN 单列；sample_grasp 候选没有可靠独立成功标签，未伪造准确率。

| 组 | 局部恢复类型 | 尝试数 | 下一次对应检查 PASS |
|---|---|---|---|
| C | control_retry | 1 | 1 |
| D | align | 8 | 4 |
| D | control_retry | 1 | 1 |
| E | align | 13 | 6 |
| E | control_retry | 4 | 4 |
| E | reperceive | 2 | 0 |

检查清除并非任务恢复。D 的 8 次对齐有 4 次让下一检查 PASS，但相对 C 只有一例新成功、同时一例退化。E 的 LLM 重新规划 3 次，2 次对应代码块立即物理成功；2 个触发重新规划的 episode 最终均恢复。前一次未成功的重新规划也计入调用和耗时，没有只记录有效请求。

## 5. 逐例因果证据

**Seed 8089：L1 控制重试直接有益。** 原程序抬升目标处 TCP 误差 33.04 mm；重复一次同目标后降至 18.91 mm，姿态仍约 16.1°、在原容差内。A/B 最终 XY 46.80 mm，失败；C 最终 22.72 mm，按原稳定物理标准成功。C 比 A 多 1 次 API，执行时间约 21.72 →22.75 s，无新增 LLM。D 进一步两次对齐把最终偏差降至 7.72 mm，但 C 已成功，不能再计一个成功率收益。

**Seed 1009：L2 修正有效，单纯降低高度不足。** 原模型真实抓持抬升，但开夹后侧向滑移。原释放前真值 XY 11.03 mm，而可见面估计 29.37 mm；L2 修正后估计 3.93 mm、真值反而为 18.82 mm，最终仍稳定成功。不能解释成“定位误差被准确消除”。追加两次零 LLM、真实在线视觉反事实，复制同一模型代码并校验到首次干预前 API/物理前缀完全一致。只在诊断进程中保留一次修正的 XY 或 Z 分量，然后强制释放；这是因果探测，不是可部署策略，也不混入 A–E。

| 干预 | 最终结果 | 释放前真值 XY mm | 释放前 AABB 间隙 mm | 最终 XY mm |
|---|---|---|---|---|
| 原始 | 失败 | 11.03 | 64.75 | 55.30 |
| 只修正 TCP Z | 失败 | 10.01 | 14.83 | 56.07 |
| 只修正 TCP XY | 成功 | 17.72 | 54.71 | 8.53 |
| L2 联合 XYZ | 成功 | 18.82 | 14.18 | 11.06 |

原始/只 Z 在开夹窗口沿世界 X 侧移约 +39.8/+41.1 mm；联合修正/只 XY 为 +7.7/+12.3 mm。只 XY 虽仍有 54.71 mm 间隙，却成功；只 Z 间隙降至 14.83 mm 仍失败。证据支持本例**水平调整改变了释放接触轨迹**，不支持“只要降高度就恢复”。可见面偏差、抓持姿态、开夹与支撑接触各自贡献尚未完全分离，不能据此添加固定补偿。D 比 A 多 1 次 API、约 +1.94 s，无额外 LLM。[反事实详细记录](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/case1009-axis-counterfactuals.json)

**Seed 2027：L2 局部成功后撤退碰倒，L3 才恢复。** A–C 原程序成功；D 两次对齐后开夹，在仿真 15.70–16.10 s 已满足原稳定标准，红绿 XY 约 3.27 mm。下一段原撤退轨迹中出现 `cubeA_g0` 与 `gripper0_right_finger2_collision` 接触，红块落回桌面，最终 XY 90.91 mm。这是局部修正改变后续轨迹条件的具体反例。E 首次重新规划重感知、重抓并估计持物偏移，仍被释放检查阻断；第二次保留持物、抬高后反复感知/对齐、释放并撤退，最终成功。3 块执行合计约 73.31 s，64 次 API（正好触及上限），2 次模型恢复请求约 417.36 s。不能把这一收益归给单独 L2。[稳定后碰倒的接触时间线](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/case2027-retreat-damage.json)

**Seed 4093：局部恢复不够，L3 在保持抓持的条件下恢复。** 原程序释放失败；D 两次修正后仍有 35.59 mm 的可见面偏差而中断（此时真值 XY 已为 9.83 mm，但 AABB 重叠约 9.39 mm）。E 将公共 TCP、可见表面和恢复历史交给 DeepSeek，一次真实重新规划保持夹持、抬高、估计持物偏移、在线复测，再配合一次 L2 对齐释放，最终 XY 22.15 mm、物理成功。模型代码中的常量来自当次公共反馈，不是仿真真值或按 Seed 写死的补偿。E 共 29 次 API、执行约 40.35 s；相对 D 多 1 次 LLM 请求和 18 次 API。

**Seed 6151：所有组成功。** D/E 仍执行局部对齐，未提高此例最终成功率，应作为额外开销而不是“成功恢复”计数。

**调试失败：** Seed 0 原始程序成功，但 L1 因约 29.7° 姿态偏差重复后仍未清除而停止，E 两轮也未恢复；偏差确实存在，中断策略过强。Seed 21 的空夹/持物缺失未恢复；“empty_jaws_after_close”只证明当前没有可靠持物，不应自动断言此前已稳定抓住后滑落。新场景中全组都至少完成一次稳定抓持，仍需更多早期抓取失败场景检验检测覆盖率。

![离线物理轨迹：有益修正与撤退退化](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/case-comparison.png)

视频：[1009 原始失败](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/paired-original-ik-v1/episodes/A-seed-1009/video_A-seed-1009.mp4)；[1009 L2 恢复](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/paired-original-ik-v1/episodes/D-seed-1009/video_D-seed-1009.mp4)；[2027 局部成功后碰倒](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/paired-original-ik-v1/episodes/D-seed-2027/video_D-seed-2027.mp4)；[2027 L3 恢复](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/paired-original-ik-v1/episodes/E-seed-2027/video_E-seed-2027.mp4)；[4093 L3 恢复](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/paired-original-ik-v1/episodes/E-seed-4093/video_E-seed-4093.mp4)。每个视频旁均保留仿真时间戳。

## 6. 调用与成本

五场景实际只支付 **8 次请求**：5 次共享初始生成 +3 次 E 恢复。输入 13,982 token，输出 55,577 token，其中 reasoning 51,750（已包含在输出中，不能重复计费）。按缓存实际命中和峰/谷价格估计 **$0.1154–$0.2307**。独立运行 A/B/C/D 任一组的 5 场景模型费用约 $0.0220–$0.0440；E 约 $0.1154–$0.2307，平均墙钟估计从 A 的 53.7 s 增至 E 的 207.8 s。

包含历史调试与废弃 IK smoke 的本轮总请求 **20 次**，其中 19 次有 usage：输入 36,802、输出 135,922（reasoning 127,760），已知部分约 **$0.2848–$0.5695**。另 1 次早期请求在旧 205 s 总期限超时，无 usage，费用未知；保守预留 $0.15005，不能记为免费。正式 35 episode 批次 14 次全部成功，无自动重试。

计价采用官方每百万 token：缓存输入 $0.044、未缓存输入 $1.32、输出 $3.96；谷价为半价，因此报告区间而非假装是账单。实际应以账户账单为准。[官方价格](https://api-docs.deepseek.com/quick_start/pricing/)；[逐请求 usage、HTTP、响应 ID、耗时和费用估算](/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119/costs-final.json)。

## 7. 交付与下一步

运行命令和当前可用服务配置见 [简明 README](/home/hanyu/dqb/cap-x-20261106/docs/feedback/README.md)；入口 `python -B -m capx.experiments.layered_feedback --help` 已实测。结果目录 `/home/hanyu/dqb/cap-x-20261106/outputs/reproduction/layered-feedback-20261119`；正式 35 episode 在 `paired-original-ik-v1`，两个反事实单独保存。旧实验未覆盖；输出目录和 JSON 使用排他创建；核心源码副本及 hash 保存在 `source-freeze` 与实验配置中。

本轮修改：`capx/envs/tasks/audited_stack.py`（可选监控/预算/reset）、`capx/execution/sandbox.py`（可恢复中断）、`capx/integrations/vision/sam3.py`、`capx/integrations/motion/pyroki.py`（保留旧默认并支持环境变量覆盖）、`capx/llm/client.py`（可选 DeepSeek effort，默认请求不变）。这 5 个原文件合计 +31/−3 行。新增 `capx/execution/feedback.py`、`capx/experiments/layered_feedback.py`、`capx/experiments/feedback_analysis.py`、`scripts/local/start_feedback_ik.py`、4 个 `tests/feedback` 文件、README/本报告；根 README 加入入口链接。未修改物理成功定义或扩大 Sandbox 可访问范围。

可运行研究原型已经成立；距离高可靠性闭环还缺：

1. **释放—撤退作为整体技能验证**，局部改位后重算后续轨迹；当前只能在 API 边界阻断，不能打断一次 goto 内部的接触过程。
2. **完整几何与不确定性估计**：多视角/校准、持物相对 TCP 变换、遮挡处理；同一有偏相机的两种算法不能充当独立确认。
3. **更合理的异常分级**：区分可容忍偏差与必须中断，处理 Seed 0 这类政策误拦；需要可部署触觉/力反馈才能可靠确认抓持与接触。
4. **独立验证局部控制器**：本轮对齐清除率有限，存在过修正与接触动力学耦合；不应凭 5 个场景修改阈值刷分。
5. **等重新规划预算的反馈对照及更大预注册场景集**，重复模型生成、更多早期抓取/遮挡/碰撞失败，报告置信区间。E 的 5/5 是小样本观察，不是总体可靠性结论。
6. **运行保障**：修复系统驱动/NVML 后再验收服务重启能力；当前成功依赖已加载视觉服务和 CPU 原 IK，不能保证重启后一键恢复 GPU 栈。

本轮已停止新增模型请求，没有批量 Teacher 采集或训练。保留独立 CPU IK 实例及 runtime JSON 供复跑，未停止其他项目服务。
