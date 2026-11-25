# CaP-X V32：原生腕部力觉、附着恢复与四组真实模型验收

日期：2026-11-24。工程 `/home/hanyu/dqb/cap-x-20261106`，分支 `feature/graph-complete-20261120`，基准 HEAD `9c7813f9cf6bb98cf4dd633b310fcc4d59233c44`。本轮没有提交或推送 Git；保留 V29–V31 未提交工作。

**结论：已接入经典 Panda/MuJoCo 原生六维力配置，并完成真实 DeepSeek + Vision 四组小规模实验。Seed 0 的 F/T 两组都恢复了 Bread 抓持并放置成功，任务进度达到 2/4；完整 PickPlace 仍为 0/8，不能宣称完成长程任务或高可靠闭环。** Seed 7 四组都被共同的邻物几何缺失阻断。本轮不扩大正式评测。

所有产物位于 [V32 归档目录](../../outputs/model_comparison/graph_v32_20261124_221234)。以下路径相对此目录。机器可读核对见 [final-audit.json](../../outputs/model_comparison/graph_v32_20261124_221234/final-audit.json)，复跑说明见 [graph-v32.md](../feedback/graph-v32.md)。

## 1. 范围、硬件与环境

Thor 地址连接超时，主机别名未解析，未读取远程 SDK。用户随后选择“用经典配置”，本轮使用当前 Panda 模型已有的 `force_ee / torque_ee`。BOX v4.0 是用户提供的采集板线索，不能据此确定腕部传感器本体型号。没有猜测型号、SN、标定数据，也没有连接真实机器人。详见 [硬件审计](v32-hardware-ft-audit-2026-11-24.md)。

现有 SAM3、ContactGraspNet 服务继续复用；最终只读检查二者 OpenAPI 和 Web 均为 HTTP 200。NVML 仍报告 `Driver/library version mismatch`（library 580.178）；未更新驱动或重启 GPU 服务。IK 使用原有 PyRoKi 服务脚本与 CPU 后端，动态端口记录在 `ik-runtime-v2.json`；当前本机配置引用该 runtime。使用 Mesa/EGL 渲染，无依赖升级、模型下载或训练。

`pre-v32-work.tar.gz` 与 `before-hashes.json` 保存本轮前的未提交工作；`preservation-audit.json` 记录保留/扩展情况。API 配置文件仍为 0600。旧实验未覆盖；所有诊断中间失败保留。

## 2. 原生 F/T 实现与标定边界

使用 `gripper0_right_ft_frame` 上的原生六轴传感器，读取原生 sensordata、机器人编码器/FK 与机器人自身惯性参数。没有用物体接触真值、物体质量或物体位姿合成在线力觉。

实测发现控制步返回的 sensordata 与当前状态不同步。现在通过独立 scratch `MjData` 的 `mj_copyData + mj_forward` 重算当前传感器；标定验收确认读取不改变原 Snapshot、控制或 RNG。反作用力符号按 `external = -(raw - bias) - tool_gravity` 处理；wrench 转到基座坐标、TCP 原点及 TCP 坐标系，包含平移原点产生的力矩叉乘项。

工具质量 0.52 kg 来自机器人模型的夹爪子树；TCP 偏移复用现有接口标定。偏置、噪声显式为零，属于理想仿真假设。输出包含 sim timestamp、sequence、坐标系、单位、有效期限、原始/补偿/滤波值、TCP、夹爪开度及关节速度。约 20 Hz 控制边界采样；80 ms 滤波供显示，保护使用未滤波值。

`sensor-gate-final` 重跑了空载、1 N 已知载荷及 15 个独立轴向载荷检查：三个力轴各 0.02/0.05/0.2 N，三个力矩轴各 0.002/0.02 Nm。1 N 检查误差 3.23e-6 N；六轴检查均通过 5 mN / 2 mNm 开发误差包络。Snapshot 不变。误差配置及校准来源 hash 保存在 `env_configs/graph_feedback/calibration/wrist-native-v32.json`。

**这些结果仅覆盖理想原生传感器、所测姿态和准静态载荷，不是完整工作空间标定或真实硬件精度证明。** 没有完整动力学补偿；速度/加速度不满足限制时不估计准静态承载。未知负载重力保留；确认后的公开承载估计可从接触候选中扣除竖直负载，未知物体质心和负载力矩仍不假定已知。

通用 `FTAdapter` / `ConfiguredHardwareFT` 接口保留设备身份、驱动版本、解码、单位、时钟映射、sensor-to-TCP、偏置与工具重力参数入口。缺少核实配置时拒绝输出；它尚不是已验证的 BOX 驱动。

## 3. 力觉怎样影响执行

复用父进程 Broker、原 Skill、ContractGraph、Sandbox 和物理评分。四组都使用 V32 Skill；关闭 F/T 时公开观测不含 `wrist_ft`，不调用力觉验证。离线传感器档案供操作者对照，未向 Vision-only Agent 提供。

| 能力 | 实际作用 | 限制 |
|---|---|---|
| `force_guarded_motion` | 每个 20 Hz 控制边界检查 40 N / 4 Nm，超限固定当前关节目标并中断 Skill，进入已有安全撤退/候选切换 | 边界保护会先观察到峰值；不是积分子步取消或工业阻抗控制 |
| `detect_contact` | 对当前公开 wrench、已确认承载生成接触候选证据 | 不推断接触物体身份，未取代碰撞/释放检查 |
| `verify_grasp_force` | 当前短窗口承载下界、力波动、夹爪收敛等支持额外有限等待 | `hold_permission`、`release_permission` 始终为 false |
| `monitor_slip` | 承载区间上界显著下降并伴夹爪变化/视觉冲突时中断；无充分证据为 UNKNOWN | 没有验证总体滑移检测准确率；不能把稳定负载等同于无滑移 |
| `recover_attachment` | 复用 `confirm_attachment`：有限等待、现有安全共动探测、实际视觉验证、验证后续接 | 未实现任意持物丢失后的可靠安全重抓 |

力觉承载区间包括开发测量包络和当前窗口波动，不能只围绕中值加一个理想误差。夹爪仍使用原 0.5 mm 收敛阈值；等待最多 20×0.25 s，无进展/无载荷/失效/预算耗尽退出。随后仍要求原 4 mm / 3° 共动条件和当前视觉检查通过。

图以版本化 observation/condition 索引力觉窗口、来源序号和有效期。新窗口替代旧窗口；过期节点传播失效，不能继续当作当前许可。最终 C/D 的 Seed 0 各有 28 次 `wrist_evidence_expired` 事件。独立视觉持物证书不会仅因一个力觉窗口过期被错误改写；需要力觉时仍必须取得当前窗口。

## 4. Bread 诊断与实际恢复证据

零新增 LLM 的历史程序重放用于定位问题，未计入真实模型结果：

| 归档 | 结果 | 发现 |
|---|---|---|
| `replay-ft-v1` | 0/4，系统退出 | 原 IK 进程失效；恢复已有 CPU 服务后继续 |
| `replay-ft-v2` | 1/4 | 初版未经标定的 0.15 N 承载门槛不适合该轻物体；没有直接调低后宣称成功，先做独立已知载荷标定 |
| `replay-ft-v3` | 1/4 | 标定后的力觉允许夹爪收敛，但合成棱柱参考给出 13.104° 共动差；离线真实物体旋转只有 0.0467° |
| `replay-ft-v4` | 2/4，整任务失败 | 当前真实可见点云作为配准参考，原共动阈值通过；Bread 放置成功，之后目标区域缺证 |

共同视觉修复：当前新鲜、非退化可见点云转换到已有物体记忆坐标系，作为跨运动配准参考。原完整形状边界/碰撞检查不变。它首先是待验证的附着假设，不会立刻授权持物；共动验证失败回滚。该能力同时提供给 A/B/C/D，没有 Bread 名称或 Seed 特判。

最终真实 DeepSeek 的 D 组 Seed 0 轨迹：

1. Milk 放置完成，原生评分进度 1/4。
2. Bread 第一次接近在 sim 22.15 s 出现约 158.08 N / 15.35 Nm 峰值，原生 F/T 触发边界停止。离线 trace 有夹爪手部与未命名 geom 的接触，尚未据此确定具体障碍物身份。安全撤退与候选切换沿用原技能。
3. 后续抓取发生视觉附着不一致；图保留 Milk 完成状态，选择附着验证。
4. 六次 0.25 s 有界等待后，夹爪开度 37.443 mm，最后变化 0.029 mm，短窗口漂移 0.051 mm；承载 0.0404 N，区间 [0.0232, 0.0576] N。力觉支持继续验证，未直接给出抓持/释放许可。
5. 8.50 mm 共动探测测得相对误差 0.421 mm / 1.463°，满足原 4 mm / 3° 条件。离线物体真实旋转 0.0467°、竖直位移 8.467 mm，仅用于此报告的独立检查。
6. Bread 放置与撤退通过，原生进度 2/4；Cereal 被抓起，但目标区域返回 `missing_bounded_wall_pair`。补证未恢复合法几何，系统在 `held_verified` 有界退出，没有释放。

这是实际公共 Vision + F/T 驱动的恢复和新子目标完成证据，尚不是完整任务成功。A/Vision-G0 也借助共同视觉修复完成过 Bread 共动验证，但随后因分割实例歧义退出；因此不能声称只有力觉才能恢复附着。B/Vision-G1 则在原有较短等待内夹爪未收敛。共同模板修复、力觉条件等待及提前保护的独立贡献还需要更细机制对照。

## 5. 最终冻结版本：真实 DeepSeek 四组结果

主结果使用 `online-fourway-final`：固定开发 Seed 0、7，各四组，共 8 个实际物理 episode。每个 Seed 只生成一次初始程序并共享给四组，Snapshot、模型 hash、初始代码 hash 已逐组断言一致。全部新请求 HTTP 200、`deepseek-v4-pro`、单次尝试、finish=stop，输出上限 32768、reasoning_effort=low。初始代码是真实模型生成的合法 `execute_skill` 序列；不是 Oracle。G0 的一次恢复代码也由新 DeepSeek 请求产生。

使用当前配置合法顺序 Milk → Bread → Cereal → Can，未擅自交换原目标区域绑定。未到达 Can，不能称为四物体覆盖验收。预算相同：24 个总请求、$5 上限、总墙钟 7200 s、12 个代码块、每块 240 s；各组沿用共同 API/Skill/恢复上限。

| Seed | 组 | 原生进度 | 完整成功 | API / capture | SAM3 / GraspNet / IK | 归因 LLM / Token | 执行 / E2E 秒 |
|---|---|---|---|---|---|---|---|
| 0 | A Vision G0 | 1/4 | 否 | 46 / 63 | 217 / 6 / 88 | 1 / 3025 | 83.64 / 94.49 |
| 0 | B Vision G1 | 1/4 | 否 | 38 / 49 | 171 / 4 / 50 | 1 / 3025 | 64.19 / 75.04 |
| 0 | C F/T G0 | 2/4 | 否 | 58 / 78 | 244 / 11 / 136 | 2 / 12893 | 130.70 / 141.55 |
| 0 | D F/T G1 | 2/4 | 否 | 57 / 74 | 228 / 7 / 99 | 1 / 3025 | 95.71 / 106.56 |
| 7 | A Vision G0 | 0/4 | 否 | 1 / 3 | 18 / 4 / 0 | 1 / 2892 | 8.36 / 17.83 |
| 7 | B Vision G1 | 0/4 | 否 | 1 / 4 | 18 / 4 / 0 | 1 / 2892 | 8.20 / 17.67 |
| 7 | C F/T G0 | 0/4 | 否 | 1 / 3 | 18 / 4 / 0 | 1 / 2892 | 8.12 / 17.59 |
| 7 | D F/T G1 | 0/4 | 否 | 1 / 4 | 18 / 4 / 0 | 1 / 2892 | 8.24 / 17.71 |

执行时间含恢复 LLM，不含共享初始生成；E2E 对每组都加回同一初始生成时间，避免只让 A 承担生成延迟。表中 LLM/Token 为方法成本归因，同一初始请求不可四次计费。API 数包含高层 Skill/证据请求及低层调用，不等于独立物理动作数。

终止原因：A0 `unknown_effect_or_timeout`（实际为 `ambiguous disconnected instance components`，未知效果隔离）；B0 `no_legal_recovery_capability`（夹爪等待未收敛）；C0 `model_stopped_without_admissible_continuation`；D0 `no_legal_recovery_capability`（目标几何缺证）。Seed 7 首次抓取前缺少邻物几何，`grasp_ready` 保持 UNKNOWN；四组没有派发低层运动，也没有力觉能够补足的交互信息。

各组完整成功率都是 0/2；合计 0/8。它们只有两个开发场景，组间共享场景，不是八个独立场景。没有总体成功率提升、检测准确率、误报/漏报率或统计显著性结论。

### 同等进展下的 G0/G1 成本与决策

**基线实现复核补充：当前 G0 并非“无图恢复”。** `ModelContractRuntime.compile_program()` 给两组创建 `CausalPlan`，G0 关闭 automatic 因果支持构图、扩大受影响范围；`_prepare_local()` 在 G0 保守补证后仍调用与 G1 相同的 `GraphExecutive.repair_causal_plan()`，并提交修复片段和确定性续接。因此本轮主要比较共享修复引擎上的依赖/证据范围策略，不能用来估计“整套图恢复相对于无图恢复”的因果收益。公平共享底层 Skill 不等于必须共享图上的修复规划器；后续主对照需要把这两层明确分开。

Seed 0 两组实际选择了相同的 `retreat_empty → acquire → confirm_attachment → deposit` 恢复链。视频看起来高度相同与此一致：相同初始程序、相同 Snapshot、相同合法技能和相同修复搜索均限制了动作差异。已确认实际派发的恢复包括超载后撤退、重新抓取以及约 8.5 mm 共动探测；不是只有日志标签。最终 Cereal 容器缺证只有补观测和退出，没有可执行的物理恢复。前面的局部恢复不能被描述为最终失败已恢复。

Seed 0 C/D 都完成 Milk 和 Bread、持有 Cereal 后阻塞，可比较这一进度下的成本。G1 保留有效目标/持物证据，主要补 `target_geometry:2`；G0 保守刷新更大剩余条件集合，额外调用 DeepSeek，生成一次 `refresh_evidence` 后仍无合法续接。

G1 少 16 次 SAM3、4 次 GraspNet、37 次 IK、4 次 capture、1 次恢复 LLM；执行时间少 34.98 s（约 26.8%）。实际 `goto_pose` 都是 23 次，open/close 都是 6/3 次；wait 为 G0 17、G1 16。**没有证据表明图减少了大批必要物理动作。** 缩减主要来自证据范围及避免无可行续接的模型请求。

| Seed 0 工具耗时 | C F/T G0 | D F/T G1 |
|---|---:|---:|
| SAM3 HTTP 累计 | 35.99 s | 33.17 s |
| GraspNet HTTP 累计 | 8.21 s | 6.40 s |
| IK HTTP 累计 | 0.253 s | 0.178 s |
| Capture/归档累计 | 8.114 s | 8.015 s |
| 新恢复 LLM | 23.45 s | 0 s |

工具累计时间是请求/归档仪表读数，不能与总墙钟重复相加。两组成功的恢复阶段记录均为 2/2（接近撤退、附着验证），不包括最终不可恢复的缺证退出。任务级恢复成功率不能以这个阶段比值替代。Seed 7 G1 比 G0 多一次 capture 仍无进展，说明图并非每个场景都降低成本。

### 实际费用与版本隔离

本轮先完成 `online-fourway` 的 8 个新模型 episode，再把力觉窗口波动纳入区间、过期事件接入图、滑移判定改为区间上界后，冻结并重跑上述最终 8 个。两批各自 `source-postcheck.json` 均为 `changed: []`，源码/配置/原始响应保留；不混合计算主结果。

| 批次 | 实际独立请求 | Input | Output（含 reasoning） | 其中 reasoning | 配置费率上界估算 | 全批墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 首批在线 | 3 | 13033 | 6498 | 6220 | $0.04293564 | 491.31 s |
| 最终在线 | 3 | 13103 | 2682 | 2408 | $0.02791668 | 430.25 s |
| 合计 | 6 | 26136 | 9180 | 8628 | $0.07085232 | 921.56 s |

合计 16 个真实 episode、6 次唯一新 DeepSeek 请求、35316 Token。首批 Seed 0 进度也是 A/B 1/4、C/D 2/4，Seed 7 都为 0/4。零 LLM 重放和离线标定不计入此表。费用来自已有预算账本的 `cost_upper_usd`，不是服务商结算单；没有将 cache 折扣或汇率假定为已验证金额。未记录密钥或鉴权头。

## 6. 软件、前端与实际验收

- Graph 回归首跑 1182 passed / 6 opt-in skipped；指定真实 Snapshot 后六项渲染测试单独 6/6 通过。按测试标识合并为 **1188 个不同测试通过**，没有未补验项。
- P0 **68/68**，启用真实 Vision 集成；JUnit、完整日志及重放视频位于 `p0-*`。
- 聚焦回归 **61/61**，包含 Graph/V32/V30 已计入部分和平台 33 项；不要与 Graph 总数重复相加。Web 独立 10/10 是平台子集。
- 新增 F/T 测试覆盖坐标/重力变换、传感器关闭不公开、非有限/过期/动态观测、窗口误差、无进展退出、真实旋转反例、非退化可见参考、力觉证据过期不误伤独立视觉证书、未配置硬件拒绝读取。
- 原生标定真实环境验收通过；前端 `tsc + vite build` 成功；`git diff --check` 通过。最终统一命令 dry-run 正常解析四组、V32 Skill 和本机 IK runtime。
- 工作台已启动，真实浏览器验证 force-mode、配置校验、历史视频、1038 个传感器采样、曲线播放/暂停、时间轴和 ±2 s 窗口。截图 `force-workbench.png` 展示的是明确标识的 `replay-ft-v4` 历史重放，不冒充新模型运行。新模型实验从 CLI 启动，未额外点击前端产生付费请求。

前端现可查看六轴未滤波补偿曲线、夹爪/TCP、标定与当前样本、相关恢复事件；已有真实图、日志和模型程序仍可查看。Vision-only 图表标为离线诊断。视频播放时钟与 sim timestamp 不直接等同，沿用视频 sidecar；力觉时间轴目前不驱动视频自动跳转。

![V32 力觉工作台历史回放](../../outputs/model_comparison/graph_v32_20261124_221234/force-workbench.png)

## 7. 修改清单

| 文件 | 本轮作用 |
|---|---|
| `capx/integrations/sensors/__init__.py`, `wrist_ft.py` | 原生传感器、坐标/重力、窗口证据、可插拔硬件接口 |
| `capx/execution/force_skills.py` | 共同力觉验证、边界运动、有限附着恢复、观测表面参考 |
| `capx/execution/skill_recovery.py`, `vision_skills.py` | 复用/抽出原等待与附着验证 hook，保留当前可见表面 |
| `capx/experiments/native_tasks.py` | 父进程传感器、逐控制边界保护、独立归档 |
| `capx/experiments/native_vision_skills.py`, `contract_models.py` | V32 共同 Skill、公开数据连接、图引用、策略与 Snapshot 分离 |
| `capx/experiments/unified.py`, `artifacts.py` | force-mode、四组共享初始程序、结果/成本字段 |
| `capx/web/experiments.py` | 有界只读力觉历史接口，处理实时半行与不同事件格式 |
| `web-ui/src/components/ForcePanel.tsx`, `ExperimentWorkbench.tsx`, `experiments.css` | 力觉配置、曲线、时间轴与历史联查 |
| `env_configs/graph_feedback/development-v32.json`, `calibration/wrist-native-v32.json` | 冻结开发配置、校准来源及包络 |
| `env_configs/inference/local-v31.json` | 本机现有 CPU IK runtime 引用 |
| `scripts/local/force_sensor_gate.py`, `analyze_force_trials.py` | 离线标定、独立物理诊断、曲线/成本分析 |
| `tests/graph_v2/test_v32_force.py`, `tests/platform/test_web.py` | 机制与 Web 回归 |
| `README.md`, `docs/feedback/graph-v32.md`, 本报告及硬件审计 | 使用说明、结果和能力边界 |

没有修改原生任务评分、释放阈值、Sandbox、通用图的任务选择算法或添加任务/颜色/Seed 专用恢复分支。

## 8. 剩余问题与下一步判定

1. **4/4 未完成。** Seed 0 的最早未解决阻塞是持物后的目标容器边界缺证；需要有目标的合法观测机会或经验证的几何传播。F/T 不能补出缺失的容器形状，不能仅凭承载正常释放。
2. Seed 7 邻物几何缺失使碰撞筛选不可用；现有能力无法在安全前提下完成补证。这与附着力觉问题分开处理。
3. 仍没有实际滑落后重新抓取、可靠临时放置、力控插入或顺应控制验收。监测到风险和有界退出不是这些能力已经解决。
4. 力觉保护采样率低，峰值可能在一个控制边界前出现。准静态包络只在已测试条件获得证据；硬件的实际型号/轴向/采样率/标定/同步全部待核实。
5. 图在同等 2/4 进度下减少了一次无效恢复 LLM 和部分感知/IK，但没有提高本轮完整任务成功率。没有足够样本证明协同优势、滑移检测准确率或安全误阻断率。

**判断：仿真 F/T 原型与 Bread 局部恢复可继续有限开发；完整长程闭环和扩大正式评测均 NO-GO。** 下一步优先补目标区域观测/几何能力，保留本轮冻结版本做回归。不能靠增加 Seed 搜索、延长无进展循环或放宽释放阈值把 2/4 改写为成功。
