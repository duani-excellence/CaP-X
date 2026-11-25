# CaP-X V31 统一推理与执行图验收

日期：2026-11-24。正式目录：`/home/hanyu/dqb/cap-x-20261106`。分支：`feature/graph-complete-20261120`；HEAD：`9c7813f9cf6bb98cf4dd633b310fcc4d59233c44`。未提交、未 push。

## 结论与范围

已实现并运行当前可用任务的统一 CLI / Web 后端、真实图事件记录与回放、多方法对比及后台停止。正式页面：`http://127.0.0.1:8200/#experiments`。原交互推理、聊天、Viser、任务配置和多任务面板保留。

**可用于已支持组合的开发诊断。不是所有方法、Provider、状态模式已统一完成，也不是新的算法效果评测结论。** PickPlace：G0/G1/LOGIV 适配、v29/v30 Skills；Stack/Lift/Restack：原 B0/G0/G1 grounded 路径。实际本轮运行 PickPlace、Stack；Lift/Restack 的路由和机制测试不冒充新在线验收。

原工作备份：`outputs/model_comparison/graph_v31_20261124_201722/pre-v31-work.tar.gz`。此前 25 个未提交文件中 23 个逐字节保留；只在原 README 与 skill_contracts 上增量加入说明/只读遥测，未丢弃 V29/V30 工作。其他 V31 改动见下表。

## 实现

| 文件 | 作用 |
|---|---|
| `capx/experiments/unified.py`、`scripts/local/infer.sh` | 严格 RunSpec；同一配置校验；复用 Contract / grounded Runner；单方法、共享初始程序配对、多 Seed/repeat 批次、零新 LLM 重放；互斥、预算、源码归档、边界停止 |
| `capx/execution/graph_journal.py`、`dependency_graph.py` | 可选 ContextVar 增量图事件日志；不改变图节点或许可；I/O 失败单独记录 |
| `capx/execution/skill_contracts.py` | 父进程准入后、调用 Skill 前记录真实 dispatch；只给观察器，不创建新执行权限 |
| `capx/experiments/artifacts.py` | 回放原图事件；读取真实 Trace、程序、评分和 HTTP；展开原计划内部显式因果/资源边；记录来源；密钥脱敏与路径边界 |
| `capx/web/experiments.py`、`server.py`、`multitask.py` | 后台子进程调用同一 CLI；状态、日志、完整配对后停止与有界强制终止；原交互/多任务互斥；只读产物接口 |
| `web-ui/src/App.tsx`、`components/ExperimentWorkbench.tsx`、`experiments.css`、`web-ui/index.html` | 保留交互页，新增配置、作业、比较、图分层/类型/版本/时间轴、节点边详情、公开图像、视频、程序、Trace；按需加载、取消轮询 |
| `env_configs/inference/local-v31.json` | 当前已存在 IK runtime / 官方 LOGIV checkout 的本机配置，避免在代码中猜端口 |
| `env_configs/graph_feedback/development-v31-grounded.json` | 原 V23.1 策略与阈值，当前服务与源文件来源；不是替换旧冻结记录 |
| `tests/platform/` | 31 项配置、配对、重放、文件边界、统计、停止、失败遥测等测试 |

没有训练、模型下载、依赖升级、真实机器人操作；独立物理评分、Sandbox、Broker、安全阈值和 G0/G1 策略未更改。新调度仅选择现有执行器。图只读展示不回流为 Agent 输入；包含隐藏物理状态的 Trace 明确标成用户离线评估资料。

## 真实模型验收

共 **6 个真实 DeepSeek＋Vision episode**：CLI PickPlace 2、Web PickPlace 2、Stack B0/G1 2。均使用 Seed 0 开发场景，不能作为独立新场景总体统计。PickPlace 的每组共享 Snapshot 与真实初始程序；Stack B0/G1 使用同一 Snapshot，各用自身合法 API / 提示词生成程序。

| 入口 / 任务 | 方法 | Reward | 物理进展 | API / capture | SAM3 / GraspNet / IK HTTP | 归因 LLM 请求 | 端到端秒 |
|---|---|---:|---|---|---|---:|---:|
| CLI / PickPlace | G0 | 0.0 | 1/4 | 45 / 62 | 211 / 6 / 78 | 1 | 90.64 |
| CLI / PickPlace | G1 | 0.0 | 1/4 | 38 / 49 | 171 / 4 / 50 | 1 | 72.95 |
| Web / PickPlace | G0 | 0.0 | 1/4 | 45 / 62 | 211 / 6 / 78 | 1 | 89.04 |
| Web / PickPlace | G1 | 0.0 | 1/4 | 38 / 49 | 171 / 4 / 50 | 1 | 71.43 |
| CLI / Stack | B0 | 1.0 | 完整成功 | 11 / 未记录 | 3 / 1 / 6 | 2 | 62.39 |
| CLI / Stack | G1 | 1.0 | 完整成功 | 12 / 60 | 3 / 1 / 8 | 1 | 115.09 |

PickPlace 四次都为 1/4、最终失败。G0/G1 都发生实际安全撤退和重抓/抬升，仍无法可靠完成附着恢复：G0 `Stage recovery budget exhausted`，G1 `Jaw did not settle within attachment recovery budget`，最终 `no_legal_recovery_capability`。图和平台没有把阶段恢复、受控退出或 FINISH 当作完整任务成功。

Stack B0：第一次生成程序使用不允许的 `.copy` 属性，Sandbox 在 0 次 API 时拒绝；一次真实模型重写后成功。G1：初始程序在 9 次 API 后因 `future_evidence_not_ready` 中断；现有 `bounded_skill_recipe / replace_placement` 补表面证据、重算放置、释放、撤退，最终物理成功。G1 没有额外恢复 LLM 请求。**这是已有恢复能力经统一入口的真实复验，不是 V31 新增算法收益。**

另外完成 **3 个历史程序＋真实 Vision episode**，新增 LLM 为 0：LOGIV 适配 1、G1 重放 2。其中 LOGIV 24 API / 36 capture、1/4，因 TCP 未收敛而停止；G1 38 / 49、1/4。所有 replay 使用明确历史标记，不并入新模型样本。

唯一新请求 **5** 次，均 HTTP 200；输入 **10018**、输出 **8928**、其中 reasoning **7634** Token。按原账本费率估算 **USD 0.04857864**，不是提供商账单。方法表含共享初始请求归因，不能逐行相加当作实际付费总额。

## Web / 图验收

- 浏览器实际操作了配置校验、启动、配对后停止、选择历史、方法勾选对比、图层/节点、时间轴起点与实时恢复；实际视频返回 206，公开 RGB 帧返回 200。浏览器无新增控制台错误。
- Web 作业 `35b0857411b842b495231ffc8f17cf7d` 配置 Seed 0、7；在 Seed 0 执行期间请求停止。两个方法完整保存后状态 `stopped / operator_stop`；没有 Seed 7 目录，只有 1 次新模型请求。强制终止另外用真实自有进程组验证，避免为了测试按钮增加付费请求。
- 首组 G0/G1 的 281 / 256 个图节点经事件重建后，state、valid、data、invalidated_by 与最终归档零差异。最终遥测重放另包含 6 个真实 dispatch 记录；running 只由该记录产生，返回后变为 returned/failed，不把 pending 猜成运行中。
- L1 展示目标、Skill、因果计划、资源/冲突关系；L2 是程序、调用、计算/验证；L3 是实际观测、估计、条件证据与状态记忆。原始点云/图像仍在独立归档，节点只引用已有来源。
- 因果计划内部的边仅在其真实 action 节点之间展开，并带 `source_record`、revision、`predicted_only`。预测因果支持不伪装为已验证物理效果。
- B0 实测图为空，页面展示其真实程序和恢复 Trace。LOGIV 页面明确为共享 Skill 契约记录，官方调度/搜索事件单独查看，不冒充其原生图。
- 比较表包含真实 HTTP 次数/累计时长、图失效/patch、实际语义/几何/计划复用。CLI PickPlace G0 复用事件 212/13/1，G1 125/6/0；这些是实际记录的复用，**保留节点数不算收益**。不同任务进度与模型生成时间必须分开分析，不能由这组小样本推出图优越性。

图采用轮询增量记录（约 2.5 秒），并提供层、边类型、关键词、证据版本和因果计划筛选；单视图最多 180 个匹配节点。旧日志没有的活动状态显示未知，不补造历史。当前没有点云 3D 播放器或任意时间点视频/图的逐帧强同步；视频自带原仿真时间戳文件，Trace 保留对应 capture 来源。

## 测试与修复过程

最终 **1270 个不同测试通过**：Graph 1171、P0 68、平台 31；失败 0，待补跳过 0。主运行 1259 passed / 9 skipped；6 项渲染与3项官方LOGIV按原要求单独补齐；后续平台统计/进程测试共31项重跑通过。JUnit、完整日志和归并清单保存。

保留的开发问题与处理：

1. 测试先于新模块添加，首轮因模块缺失失败，原日志保留。
2. 前端首轮编译发现 JSX 和 nullable 类型错误，修正后 `tsc + vite build` 通过。
3. 旧 grounded 配置检测到 GraspNet 客户端源码变化，启动拒绝、0 新请求；重新只读采集当前服务来源并生成 V31 开发配置后实测成功，没有关闭来源检查。
4. 首组平台 checkpoint 将 NumPy 进展序列变成字符串；改为数值数组后补测试。原初轮 checkpoint 不覆盖，其 episode 原始 result 一直为正确数值数组；后续 Web checkpoint 已验证。
5. 单元测试最初预期目录写入错误为 IsADirectoryError，实际独占创建返回 FileExistsError；修正平台相关断言，拒绝许可及遥测错误隔离断言保持。
6. 渲染补验首次指向不存在的 Snapshot 文件，6 项 fixture error 保留；改用实际场景 manifest 的 `scene/snapshots/seed-0.json` 后6项通过，未改测试。

`pre-v31-work.tar.gz`、`preservation-audit.json`、`test-acceptance-summary-final.json`、`actual-request-accounting.json`、`final-episode-accounting.json` 是主要汇总证据。扫描 3487 个文本产物，对已加载密钥值匹配为 0；未输出密钥。

## 限制与下一步

- 统一图 Provider 仍只支持 DeepSeek V4 Pro。OpenAI/本地模型的原交互功能保留，但尚无等价图预算/成本 adapter；Privileged 未混进 Vision 图试验。能力列表与错误信息明确拒绝，不能把可配置字段当作已接入。
- CaP-Agent0、ASPIRE 完整策略、LIBERO-Recover 在线策略未接入；cuRobo 仍受已知 CUDA804 阻塞。PickPlace 没有合法 B0 raw-API adapter。LOGIV 只支持当前 Contract/PickPlace 自然场景，不伪造受控扰动等价性。
- NVIDIA/NVML 仍不可读，长驻 SAM3/GraspNet 实际推理可用；复用 CPU IK / Mesa，不修驱动、不重启视觉服务。新服务 inventory 如实保留“当前源码未必等于长期进程已加载字节”的限制，因此不宣称正式冻结验收通过。
- `wall_seconds` 在请求与完整配对边界约束；物理执行继续沿用 episode/block 超时。立即终止会留下不完整配对，明确排除为任务成功。
- Web 重启只认本进程拥有的子进程；未知旧 PID 不自动接管/杀死。没有跨任意旧 CLI/服务器的全局作业调度。
- 原 grounded 的高级 `initial-manifest` 复现仍走原入口；统一 replay 当前只接 Contract 归档。不支持任意中途 Snapshot 注入。

**下一步可以在当前能力矩阵内继续小规模开发调试；没有依据扩大正式算法评测。** 新入口和图界面使真实失败可定位，但没有消除 PickPlace 的附着/几何瓶颈，也没有新增图算法收益证据。

使用命令、完整参数、单方法/配对/批量/重放/停止示例见 [graph-v31.md](../feedback/graph-v31.md)。

证据根目录：`/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v31_20261124_201722`。实际浏览器截图：`/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/graph_v31_20261124_201722/v31-workbench-final.png`。
