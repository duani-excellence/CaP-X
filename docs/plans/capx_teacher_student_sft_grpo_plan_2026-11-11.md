# CaP-X Teacher–Student + SFT + GRPO 落地方案

审计日期：2026-11-11（UTC+08:00）  
项目：/home/hanyu/dqb/cap-x-20261106  
性质：只读技术审计与下一阶段实施设计。本轮未修改现有源码、配置、环境或训练数据，未下载模型、发起 Teacher 推理、启动仿真/训练或操作真实机器人。唯一新增文件为本报告。

**建议：先修复可复现性、观测边界和奖励语义，建立结构化感知接口；以 Qwen2.5-Coder-1.5B-Instruct 的文本 LoRA SFT 跑通闭环，再做单卡、单环境、G=4 的 Terminal GRPO。Qwen3.5-4B 保留为第二阶段模型升级；完整原生 VLM 和 Physical Critic 暂缓。当前 GRPO 判定 NO-GO。**

全文状态约定：

- **[V] 已实现并有验证证据**：注明历史运行、此次只读检查或纯 Mock；三者不相互替代。
- **[S] 源码存在、尚未实际验证**：读到了实现，但没有本机相应端到端证据。
- **[N] 需要新增开发/修复**：包括以下所有拟议训练配置、采集协议和验收门槛。
- 外部资料事实标注来源；显存、工时和成功率门槛是方案估计，不是测量结果。

## 1. 当前工程真实完成情况与实验核验

### 1.1 找到的全部已完成试验

只统计项目 outputs 下独立 trial 目录中的 summary.txt，避免把上层 summaries.txt 和复制到聊天工作区的文件重复计数。共 **15 个已完成 trial**，全部 sandbox_rc=0、每个 1 个代码块；只有 4 个 task_completed=True。以下成功率只是历史样本描述，未做配对 seed、不能当泛化评估。

| 模式 | 完成次数 | 任务成功 | reward（各次） | 判定 |
|---|---:|---:|---|---|
| Privileged + Oracle | 2 | 2/2 | 1、1 | [V] 参考程序可完成堆叠 |
| Vision + Oracle | 3 | 1/3 | 1、0.009400387、0.001647630 | [V] 曾成功，也有两次失败；视觉执行链路不稳定 |
| Privileged + DeepSeek V4 Pro | 1 | 1/1 | 1 | [V] 不仅代码执行成功，任务也完成 |
| Vision + DeepSeek V4 Pro | 3 | 0/3 | 0.000629783、0.002839944、0.000253461 | [V] 都未完成堆叠 |
| Privileged + Qwen2.5-Coder-7B / vLLM | 1 | 0/1 | 0.030577640 | [V] 程序可运行，任务失败 |
| Vision + Qwen2.5-Coder-7B / Ollama 或 vLLM | 5 | 0/5 | 0.002154925、0.041251250、0.001092095、0.004112046、0.048779148 | [V] 任务均失败 |

**修正背景表述**：“Vision Oracle 失败”描述了部分记录，完整事实是 1 成功、2 失败。不能用最早那次成功证明链路可靠，也不能忽略这次成功而认定完全不可用。目录名含 ollama 的 Oracle 试验仍执行参考代码，不代表调用了 Ollama。

证据目录（每个目录内的 trial_*/summary.txt、all_responses.json、程序和录像可交叉核查）：

- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/oracle/oracle-first
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/oracle/oracle-ollama-20261110-162921
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/oracle/vision-oracle-first
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/oracle/vision-oracle-ollama-20261110-162940
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/oracle/vision-oracle-ollama-20261110-164702
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/deepseek-v4-pro/deepseek-privileged-control
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/deepseek-v4-pro/deepseek-vision-first
- /home/hanyu/dqb/cap-x-20261106/outputs/franka_robosuite_cube_stack_main_use_server/deepseek-v4-pro/20261110_181446
- /home/hanyu/dqb/cap-x-20261106/outputs/franka_robosuite_cube_stack_main_use_server/deepseek-v4-pro/20261110_182032
- /home/hanyu/dqb/cap-x-20261106/outputs/reproduction/qwen2.5-coder:7b/{vllm-privileged-first,vllm-vision,ollama-vision}
- /home/hanyu/dqb/cap-x-20261106/outputs/franka_robosuite_cube_stack_main_use_server/qwen2.5-coder:7b/{20261110_155752,20261110_171903,20261110_171921}

另外复查了已有 DeepSeek Vision 抽帧：中段夹爪移至绿色方块上方时红块仍在桌面，末帧两块仍分离；Privileged 成功末帧为红块堆在绿块上且夹爪已退出。这支持“发生空载搬运/未堆叠”的现象判断，但稀疏抽帧不足以区分最初空抓与短暂抓住后滑落，仍需第 5 节的接触与相对位姿 trace。

### 1.2 时间、请求数与缺失指标

| 代表试验 | 代码生成秒 | trial 内总时间秒 | runner 总时间秒 | 纯执行时间 | 已保存模型响应 |
|---|---:|---:|---:|---|---:|
| Privileged Oracle 首次 | 不适用 | 4.70 | 9.16 | 无独立计时 | 0 次模型请求 |
| Vision Oracle 首次 | 不适用 | 7.51 | 11.12 | 无独立计时 | 0 次模型请求 |
| DeepSeek Privileged | 33.84 | 35.44 | 36.05 | 无独立计时 | 1 |
| DeepSeek Vision CLI | 73.77 | 81.14 | 90.78 | 无独立计时 | 1 |
| DeepSeek Vision Web 两次 | 无可可靠关联的逐 trial 完整时间字段 | 缺失 | 缺失 | 缺失 | 每 trial 1 |

证据：[Privileged 日志](/home/hanyu/dqb/cap-x-20261106/logs/deepseek-privileged-control.log)、[Vision 日志](/home/hanyu/dqb/cap-x-20261106/logs/deepseek-vision-first.log)、[Oracle 日志](/home/hanyu/dqb/cap-x-20261106/logs/oracle-first.log)、[Vision Oracle 日志](/home/hanyu/dqb/cap-x-20261106/logs/vision-oracle-first.log)。

trial 总时间减生成时间约为 1.60 / 7.37 秒，但包含其它处理，**不得改名为代码执行耗时**。Web 日志有多个 streaming 计时条目，已有两份完成产物不能与所有请求一一对应；不要据此补造精确请求数。当前没有完整 HTTP 尝试计数、usage、缓存命中、reasoning token 和账单字段；“保存 1 个响应”不一定等于“网络只发 1 次请求”。

**新增统一指标**：t_codegen（含重试另列）、t_exec（仅程序执行）、t_env_init、t_perception、t_eval、t_total；logical_request_count、http_attempt_count、code_block_count 分开；sandbox_rc、raw_reward、task_completed、physical_success 分开。请求失败、超时、空输出也必须进入分母。

### 1.3 已完成与尚未完成的边界

[V] 官方 DeepSeek 文本调用、SAM3/ContactGraspNet/PyRoKi 服务、Robosuite 堆叠、CLI/Web 生成与执行、视频和部分日志均已有运行证据。Qwen 7B GGUF 的本地推理已运行。

[S] 仓库包含 VeRL、GRPO、SFT、多轮数据处理、LoRA、权重同步和 checkpoint 代码，但未发现本机完成训练的产物。docs/rl-training.md 的论文成功率、脚本注释“约 13 steps 达 95%”不能作为本机结果。

[N] 可靠视觉采集、真实 seed 配对、可部署观测边界、干净 SFT 数据、单 Reward Worker 本机仿真验收、Student 训练及部署闭环均未完成。

## 2. 可复用模块及实际边界

| 模块 | 当前作用 | 状态与复用方式 |
|---|---|---|
| [launch.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/launch.py)、[runner.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/runner.py)、[trial.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/trial.py) | 配置、服务、试验、生成代码、执行、产物 | [V] CLI 主流程复用；补种子、计时和完整事件记录 |
| [tasks/base.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/base.py:153) | Python 持久命名空间、reset、step、reward/info | [V] 执行可用；不是隔离沙箱，必须限制暴露对象 |
| [franka_pick_place.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/franka/franka_pick_place.py) | 堆叠任务说明与 Oracle 程序 | [V] 可作诊断控制组；不是直接可训练的完整数据集 |
| [robosuite_cubes.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cubes.py) | Stack、摄像头、物理状态、成功判断 | [V] 仿真已跑；[N] 修复 seed、四元数转换、状态快照 |
| [control.py](/home/hanyu/dqb/cap-x-20261106/capx/integrations/franka/control.py) | 视觉定位、抓取、IK、夹爪 API | [V] 已执行；精度和可靠率未验收 |
| [control_privileged.py](/home/hanyu/dqb/cap-x-20261106/capx/integrations/franka/control_privileged.py) | 真值定位和固定向下抓取 | [V] 诊断与独立 privileged baseline；不冒充 Vision |
| [llm/client.py](/home/hanyu/dqb/cap-x-20261106/capx/llm/client.py:185) | HTTP/streaming 模型调用 | [V] DeepSeek 文本可用；[N] usage、模型版本、有限重试和能力检测 |
| [prepare_verl_dataset.py](/home/hanyu/dqb/cap-x-20261106/capx/cli/prepare_verl_dataset.py) | prompt parquet、data_source、reward_model、extra_info | [S] 可复用格式；不是 SFT 采集器 |
| [capx_franka_reward.py](/home/hanyu/dqb/cap-x-20261106/verl_agent_reward/capx_franka_reward.py) | 创建环境、执行整段程序、返回分数 | [S] 外壳可复用；奖励与错误处理必须改 |
| [vendored VeRL](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl) | GRPO/FSDP/vLLM、SFT、LoRA、checkpoint | [S] 版本文件 0.5.0.dev；不是已装好的训练环境 |
| [scripts/local/run.py](/home/hanyu/dqb/cap-x-20261106/scripts/local/run.py) | Oracle/vision/privileged/web 入口 | [V] 适合人工 smoke；不支持 seed 清单，不能充当正式配对评估器 |

## 3. 当前阻塞、根因证据与优先级

### P0：必须先修复

1. **默认环境注册映射不存在。** [tasks 注册](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/__init__.py:36)把 franka_pick_place_code_env 指向 franka_cubes_low_level；[低层注册](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/__init__.py)只有 franka_robosuite_cubes_low_level 等名称。默认训练路径在实例化阶段就不成立。已有 franka_robosuite_pick_place_code_env 可复用，但它使用 Privileged API，且没有显式 privileged=True，继承默认 False；还需统一语义和关闭不必要渲染。

2. **奖励错误已用纯 Mock 确认。** [compute_score](/home/hanyu/dqb/cap-x-20261106/verl_agent_reward/capx_franka_reward.py:56)对 sandbox_rc=0 使用 max(reward,0.1)，won=score>0。此次仅 AST 提取函数并注入假环境，输入 pass、reward=0、task_completed=False，得到 score=0.1、won=True。没有启动仿真或写文件。这证明奖励语义问题，不证明 Reward Worker 可用。

3. **seed 目前不代表相同初始场景。** [数据生成](/home/hanyu/dqb/cap-x-20261106/capx/cli/prepare_verl_dataset.py:41)计算 env_seed 后仅写 extra_info，没有 env.reset(seed=env_seed)。[Stack reset](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cubes.py:119)只重建包装层 self._rng，实际 placement_initializer 绑定的是 robosuite_env.rng；构造 Stack 时也没有传该 seed。[Robosuite base](/home/hanyu/dqb/cap-x-20261106/capx/third_party/robosuite/robosuite/environments/base.py:141)另建 rng。修复必须覆盖对象采样器、机器人初始噪声、感知/抓取采样 RNG，记录初始 qpos/qvel/物体状态摘要验证；只改 metadata 不够。

4. **Student 可以绕过视觉、甚至修改被评分环境。** [执行 globals](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/base.py:153)暴露 env、APIS、obs、INPUTS；[get_observation](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cubes.py:199)含 cube_poses 和 Robosuite 原始真值。即使 prompt 没写真值，生成程序仍能访问它。必须使用进程隔离的 API 代理，只返回允许观测；评分器独立持有 simulator，不能让程序修改成功标志。AST allowlist 是辅助，不能独立构成隔离边界。

5. **默认并发远超单卡条件。** [训练脚本](/home/hanyu/dqb/cap-x-20261106/scripts/train_franka_grpo.sh)为 7B、batch=256、G=15、micro=8、vLLM 0.75 显存、agent workers=25；而 [PrimeRewardManager](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/workers/reward_manager/prime.py)另硬编码 ProcessPoolExecutor 64 个进程。改 agent.num_workers=1 并不会把这 64 个 reward 进程改成 1。第一轮应改用已有 naive manager 顺序评估，异步奖励关闭，并给单独仿真进程提供硬超时。

### P1：会污染训练或造成错误解释

- 环境构造在 compute_score 的 try 之外，注册/依赖/初始化错误不会返回该函数宣称的统一 schema；上层 Prime 又可能吞成 0 分，掩盖系统故障。
- _time_limit 用 SIGALRM，但程序 exec 捕获 BaseException，可能把超时吞为普通执行错误；Python signal 不能可靠终止卡死的本地扩展。需父进程 deadline + 子进程终止/重建，而非仅靠 signal。
- 环境跨样本缓存，API 对象也复用；reset 清 exec globals，却没有明确清 control.cfg 的 IK warm start、视觉缓存与 API 内部状态。样本顺序可能改变结果；需 episode reset hook。
- 程序执行后才检查 max_steps，不是每个动作强制预算；循环可超量运动。需逐动作预算和墙钟硬截止。
- _extract_code 从第一个 python fence 到最后一个 fence 截取，多代码块可能混入解释或 Markdown。统一解析器，仅接受协议允许的代码块；格式错误不得悄悄执行自然语言。
- terminated 直接等于 reward==1.0，而 info.task_completed 独立计算。应由统一物理成功/时间预算确定，保存原始字段便于追溯。
- Prime 的正常评分路径只保留 score，丢弃 task_completed/error 等细节；已有 naive manager 支持 dict 指标，优先复用。
- 训练脚本仅检查 DATA_ROOT 目录存在，不核验 parquet 完整性/hash；空目录也跳过生成。脚本强设 osmesa，已有成功路径用 egl；不要未经验证切换渲染栈。
- prepare_verl_dataset 的 oracle_pool 读取了但没有用于实际样本构建；ground_truth.program 放 Oracle，reward 函数实际未使用。该路径不是 Teacher distillation。
- get_object_pose/sample_grasp_pose 无条件写项目根 depth_image.jpg，既覆盖诊断图又导致并发竞争。需 episode 独立产物路径并默认关闭调试写入。
- 模型 HTTP 对部分状态码无限重试，统计和试验时间不可控；增加明确次数、总 deadline、429 处理和终止原因。
- 现有证据没有完整抓取接触、滑落时刻、末端误差序列；“视觉失败主要由某个坐标偏移引起”仍是待测假设。

## 4. Teacher–Student 架构与实际视觉输入

### 4.1 当前 DeepSeek 有没有看到图像？

**没有当前实测向 DeepSeek V4 Pro 发送 RGB 的证据；源码配置和已保存完整 CLI prompt 均支持“只发送文本”。**

[V] DeepSeek Vision CLI 的 all_responses.json 内 initial_prompt 为 system 字符串 + user 的单个 type=text 项；没有 image_url 或 data:image。初始文本为任务说明和 API 文档，约 3,286 字符的 user 内容。生成程序在本地执行时才调用 get_object_pose/sample_grasp_pose；这些感知结果成为 Python 数值或 stdout，单次整段程序模式不会自动再回送给 LLM。

[V] 本地 run.py 明确设置 use_visual_feedback/use_img_differencing/use_video_differencing=False。DeepSeek 的通用 HTTP 分支发送 model、messages、temperature、max_tokens；已运行记录证明该文本请求被接受，但不是图像兼容性证明。

[S] [trial.py:220](/home/hanyu/dqb/cap-x-20261106/capx/envs/trial.py:220)有 PNG/base64、image_url、多相机和图像/视频差分实现。直接 VF 的主门控是 use_visual_feedback 且模型在 VLM_MODELS；DeepSeek V4 Pro 不在该视觉模型列表。差分还可能调用另一个 VLM，将描述作为文本交给 coder。不能靠把模型名塞进列表赋予 API 图像能力。

截至本次审计，DeepSeek 官方模型表明确 deepseek-v4-pro 对应 DeepSeek-V4-Pro-0813、**不支持 Vision**；deepseek-flash 是另一模型，不能暗中替换 Teacher。官方 Chat Completions 文本接口可继续使用。[DeepSeek 模型能力表](https://api-docs.deepseek.com/quick_start/pricing/)

| 模式 | 感知由谁完成 | coder 接收内容 | 当前状态 |
|---|---|---|---|
| Privileged | 仿真真值 API | 文本任务/API；程序调用真值 API | 已运行；诊断基线 |
| Standard Vision | SAM3 + RGB-D + ContactGraspNet | 当前首轮仅文本任务/API；程序读感知返回值 | 已运行；不稳定 |
| VF | 外部视觉 API，且 coder 可直接看图 | RGB/base64 + 文本/历史 | 源码存在；DeepSeek Pro 不适配 |
| 图像/视频差分 | 独立 VLM 描述场景变化 | coder 接收差分文本，可能另有 VF | 源码存在；本地入口关闭 |
| 拟议结构化反馈 | 外部感知 + API receipt | 可部署的坐标、置信度、夹爪/运动状态、历史 | 需要开发；首选训练接口 |

DeepSeek 默认 thinking=enabled/high；当前专用 thinking/effort 未显式配置，temperature 不应被当作稳定控制其思考采样的依据。返回解析目前读取 reasoning 而非 DeepSeek reasoning_content，也没有保存 usage；应补适配，但不把隐藏推理过程作为 SFT 必需字段。[官方 thinking 文档](https://api-docs.deepseek.com/guides/thinking_mode/)

### 4.2 目标架构

~~~mermaid
flowchart LR
  T["冻结 Teacher：DeepSeek 官方文本 API"] --> C["生成 Python / 离线失败分析"]
  O["任务 + API 文档 + 可部署观测 + 历史"] --> T
  O --> S["Student 文本 LM：仅 LoRA 更新"]
  S --> C
  C --> X["受限执行进程：只暴露 API 代理"]
  X --> P["冻结 SAM3 / RGB-D / ContactGraspNet"]
  X --> M["冻结 PyRoKi / Franka Controller"]
  P --> R["可部署结构化 receipt"]
  M --> E["MuJoCo / Robosuite"]
  E --> R
  R --> O
  E --> G["隔离的真值日志 / 成功判定 / Reward"]
  G --> D["通过重放的 SFT 样本"]
  D --> S
  G --> U["GRPO 更新"]
  U --> S
  G -. "后续训练专用" .-> V["Physical Critic"]
~~~

Teacher 参数冻结，不通过推理 API 训练 DeepSeek。SAM3、ContactGraspNet、IK、控制器、仿真器均冻结；SFT 与 E3/E4 只更新 Student LoRA。Critic 如启用，独立参数、独立真值输入、部署时移除。Teacher 分析若看过 privileged state，样本必须标注，不能把其含真值描述的分析塞进 Vision Student prompt。

## 5. 第一阶段：视觉与执行链路排障

### 5.1 最小配对实验

[N] 先 5 个固定初始状态做 smoke，每个状态做 A/B/C/D；通过后扩到至少 20 个独立初始状态。A/B 采用同一 Oracle；C/D 控制相同任务说明、同一 Teacher 设置，记录实际 API 文档差异。固定 scene_id 和状态快照比仅写 seed 更强。初始场景应尚未完成任务，排除天然已堆叠状态，防止 pass 通过终态检查。每次重放都重新初始化机器人、物体、API 缓存与 RNG。

1. A：Privileged + Oracle，验证控制与任务可解性。
2. B：Vision + Oracle，隔离生成策略差异。
3. C：Privileged + DeepSeek，验证 coder 与 API 语义。
4. D：Vision + DeepSeek，验证完整系统。

**额外必要对照**：在同一 snapshot 重放同一份程序，逐次只替换一项感知结果（位置、抓取姿态、尺寸等）。四组总成功率只能定位到大模块，不能单独证明 ContactGraspNet 或 TCP 的因果责任。真值替换仅用于诊断，不能混入 Vision 训练/评估。

### 5.2 源码提示的重点检查

| 阶段 | 记录与对照 | 特别关注 |
|---|---|---|
| 分割 | RGB、mask、目标实例、置信度；仿真 instance mask 仅给评估器 | 错物体、遮挡、桌面混入、mask/depth 分辨率 |
| 深度与定位 | 原始深度、metric depth、内外参、有效点数、OBB、目标真值 | 部分可见点云 OBB 中心偏移和尺寸缩小/轴交换 |
| 姿态与尺寸 | 四元数顺序、参考系、OBB 轴、世界竖直范围 | OBB.extent[2] 不必等于世界竖直高度；不能一律除二就当堆叠高度 |
| 抓取 | 全候选、score、被选 index、相机/基座/TCP 三个 pose | 当前直接 argmax，置信度不是物理成功率 |
| 坐标变换 | T_base_camera、T_camera_grasp、T_grasp_TCP、往返残差 | control.py 中 *_world 实际来自相机到 robot base，不应只看变量名 |
| TCP | 抓取平移 +0.12 m；goto_pose 中 TCP offset [0,0,-0.107] | 两项在不同坐标链，不能简单相减认定 13 mm bug，需标定目标语义 |
| 进近 | 目标和真实路径、接近方向、碰撞 | 文档称 Z 轴接近，实现在旋转后的工具局部 -Z 方向进近；倾斜抓取会横移 |
| IK/控制 | 目标末端 pose、FK pose、q_target/q_actual、位置/角度残差、达到 max_steps | move_to_joints_blocking 使用关节误差阈值 0.02、最多 100 步；“Motion complete”不等于末端到位 |
| 闭爪/抬升 | 指间距、接触对、物体随动、物体-夹爪相对变换 | 区分闭爪完成、空抓、碰撞推走、抓住后滑落 |
| 放置 | release 前后位置/速度、支撑接触、退出后稳定性 | 只靠接近目标中心不能证明堆叠 |

已发现一个可静态确认的坐标格式问题：[Stack sensor](/home/hanyu/dqb/cap-x-20261106/capx/third_party/robosuite/robosuite/environments/manipulation/stack.py:449)返回 cubeA_quat/cubeB_quat 为 XYZW，而 [包装层](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cubes.py:163)直接作为 SE3.wxyz_xyz 拼接。应转换后再使用，并验证真值记录。Privileged 抓取硬编码向下姿态，所以已有成功不排除此 bug；Vision 抓取来自另一条链，也不能据此把所有视觉失败归因于此问题。

### 5.3 事件记录和物理成功定义

[N] 每次 API 调用记录 episode_id、request_id、action_id、时刻、输入、返回/错误、耗时；分别保存：

- **actor_observation**：部署可获得的 RGB-D 派生信息、关节/夹爪状态、API receipt、置信度和坐标约定。
- **evaluator_state**：qpos/qvel、物体 pose、接触、真值子目标、reward、task_completed；存独立 sidecar，禁止训练 prompt 自动拼接。
- 每个 API 前后及闭爪/抬升期间保存状态；动作执行过程按固定采样间隔记录，避免错过短暂滑落；视频帧与 sim timestamp 对齐。

抓取成功建议定义为目标接触与物体-夹爪相对位姿持续稳定；Lift 要求目标离开初始支撑面至少 5 cm 并保持 0.5 秒；Placement 要求释放、退出后仍满足任务几何/支撑关系至少 0.5 秒。阈值是拟议协议，需按方块尺寸/仿真频率标定；同时保留原始 Robosuite task_completed 以防新旧指标混淆。

分类：segmentation_miss、wrong_instance、pose_error、grasp_pose_error、frame_or_tcp_error、ik_unreachable、motion_not_converged、empty_grasp、slip_after_lift、placement_error、syntax/API_error、timeout、infrastructure_error、unknown。记录 primary_failure + secondary_failures，证据不足用 unknown。

### 5.4 继续 Vision 还是先用结构化基线？

**保留 Vision 作诊断；暂停用当前 Vision 大量生成训练正样本。** 第一目标是统一可部署结构化 API 和 receipt。现有 Vision 已有坐标返回，但缺置信度/状态确认/清晰 frame 合约，不能称为稳定接口。

若视觉链路暂未通过，允许先做明确标注的 Privileged 训练闭环，证明 SFT/GRPO 软件栈；它的成绩只能算 privileged 成绩。Vision Student 的监督数据必须在实际 Vision 接口重放成功。Cube Lift 可作为较简单的阶段任务，但要单独报告，不能代替 Cube Stack 的验收。

## 6. Teacher → Student 数据管线与 SFT 数据集

### 6.1 采集流程

~~~mermaid
flowchart TD
  A["冻结 task / API / seed 与初始状态 manifest"] --> B["Teacher / Oracle / 人工修正候选"]
  B --> C["协议解析 + Python AST + API 边界检查"]
  C --> D["目标部署接口中的实际仿真重放"]
  D --> E{"物理成功且日志完整？"}
  E -- "否" --> F["失败库：错误分类 / 诊断 / 后续修正"]
  E -- "是" --> G["来源标签 + 状态/代码/模型版本哈希"]
  G --> H["按场景家族分组划分 Train / Val / Test"]
  H --> I["去重与模板限权 / 生成 assistant loss mask"]
  I --> J["LoRA SFT"]
  J --> K["Adapter 重载与独立 seed 仿真验收"]
  F -. "修正后重新执行" .-> C
~~~

[N] 第一批目标为 **100–300 条通过目标接口重放的成功 episode**，至少 50 个训练场景，另设 10 个验证场景和 20 个锁定测试场景。数量只是启动规模，不能靠复制同一 Oracle 程序凑数。先采 20–30 次 pilot 估计成功率，再决定 Teacher 请求预算；成功率低先排障，不靠大量请求硬刷。

来源标签分成两个独立维度，避免“Teacher”和“Vision”互斥混淆：

- generator：deepseek_teacher / oracle / human_corrected / student。
- observation_regime：privileged / standard_vision / structured_vision / native_vlm。
- 另存 teacher_privileged_assistance、repair_parent_id、replay_regime、teacher_model_version。经人工修正的程序不能再标为 Teacher 原始成功。

**Privileged → Vision 的转换门槛**：程序只使用部署 API；移除 env/obs 真值访问、物体硬编码真值、隐藏状态推导的文本；以 Vision 实际观测重建 prompt，并在 Vision 重新执行成功。接口同名不是等价证据。无法转换的成功轨迹留在独立 privileged 数据集。

### 6.2 每条 episode 的建议 schema

以下为 [N] 数据设计，不是当前已有 JSON 格式。缺失字段用 null + missing_reason，不补造：

~~~json
{
  "schema_version": "capx-trajectory-v1",
  "episode_id": "unique-id",
  "task_id": "cube_stack",
  "instruction": "Place the red cube on the green cube.",
  "provenance": {
    "generator": "deepseek_teacher",
    "observation_regime": "structured_vision",
    "replay_regime": "structured_vision",
    "teacher_privileged_assistance": false,
    "repair_parent_id": null
  },
  "scene": {
    "seed": 101,
    "scene_family_id": "layout-family-id",
    "initial_state_hash": "sha256",
    "config_path": "record-the-absolute-config-path",
    "config_sha256": "sha256"
  },
  "interface": {
    "api_version": "version",
    "api_docs": "exact docs presented to the model",
    "prompt_version": "version",
    "frame": "robot_base",
    "quaternion_order": "wxyz",
    "length_unit": "m"
  },
  "model": {
    "provider": "deepseek",
    "requested_model": "deepseek-v4-pro",
    "response_model": null,
    "version_evidence": "recorded response or dated provider documentation",
    "generation_config": {}
  },
  "messages": [
    {"role": "system", "content": "exact system prompt"},
    {"role": "user", "content": "task, docs and allowed observation"},
    {"role": "assistant", "content": "exact generated program"}
  ],
  "requests": [],
  "api_events": [],
  "programs": [],
  "outcome": {
    "sandbox_rc": 0,
    "raw_reward": 1.0,
    "task_completed": true,
    "physical_success": true,
    "subgoals": {"grasp": true, "lift": true, "placement": true},
    "failure_type": null
  },
  "metrics": {
    "codegen_seconds": null,
    "execution_seconds": null,
    "logical_requests": null,
    "http_attempts": null,
    "input_tokens": null,
    "output_tokens": null,
    "reasoning_tokens": null,
    "api_cost_usd": null
  },
  "artifacts": {
    "video": "episode-relative-video",
    "evaluator_state_sidecar": "not-part-of-policy-input"
  },
  "revisions": {
    "source_digest": "sha256",
    "student_base_revision": null,
    "perception_checkpoint_hashes": {},
    "runtime_manifest": {}
  }
}
~~~

requests 内逐次保存实际输入 messages、响应、finish_reason、模型返回 ID、usage、时间和重试；api_events 保存前后 actor observation 与 evaluator sidecar 的索引。不要把原始 API key、Authorization header、环境变量加入 manifest。

### 6.3 清洗、划分与多轮序列化

1. 验证 JSON/schema、代码可解析、API 允许范围、完整终止、日志/录像对应；不允许截断的半段 Python。
2. 成功正样本要求 sandbox_rc=0、task_completed=True、独立稳定放置检查通过。系统错误、空抓、伪 FINISH 全部进失败库。
3. 对 raw code、规范化 AST、prompt、初始状态分别 hash。相同场景的重试、修正、所有轮次属于同组；同组不得跨 split。近似布局和相同派生 scene_family 也分组。
4. 同一通用 Oracle 程序跨多个场景成功可以作为重放证据，但不能无限重复权重；按程序模板家族限权，保留真正不同的观测、恢复策略和任务变化。完全相同 prompt+answer 只留一份或降权。
5. 先冻结场景级 train/val/test manifest，再选训练成功轨迹；不要按成功结果挑测试场景。原有 15 次无固定初态记录的产物只作诊断种子资料，不充当锁定测试集。
6. 第一版 SFT 训练单轮完整程序；加入多轮时使用与部署一致的 messages：system → user(task/docs/obs) → assistant(code) → user(API receipt/错误) → assistant(next code)。当前不是原生 function/tool-call 协议，勿任意改成不匹配的 tool 角色。
7. assistant 正确动作及 EOS 参与 loss；system、user、receipt、padding 不参与。失败尝试可留在历史中作为纠错上下文，但将失败 assistant turn 的 loss_mask=0，仅监督验证后的修正；整段失败 episode 不作正样本。
8. 上下文超长按请求边界切窗，重复必要 task/API 和可部署状态摘要；保留依赖变量/辅助函数定义。不能从 Python 中间截断或把未来成功状态放入过去输入。
9. 离线 Teacher 修复可看到 evaluator_state，但学生输入只用当时可获得信息；凡修复答案依赖学生不可知的信息，丢弃或重新通过在线观测产生等价答案。
10. 成功不足：先修链路；先收 Cube Lift、固定难度布局与 Oracle 重放；人工纠正程序后重放；Teacher 生成少量策略变体。未经执行的合成代码只能做静态 API 练习，单独标签、限权，不能计算为操作成功数据。

### 6.4 RL parquet 与 SFT 数据不同

现有 prepare_verl_dataset 生成的是 prompt-only RL 行：

~~~text
data_source: 已注册环境字符串
prompt: [{role, content: string}, ...]
ability: agent
reward_model: {style: sim_code, ground_truth: {program: ...}}
extra_info: {split, index, seed, ...}
~~~

VeRL custom reward 根据 data_source 在 CaP-X 创建环境；不必修改默认数学题 reward 注册表。ground_truth 的 Oracle 不自动变为训练目标，也不得进入 Student prompt。修复后扩充 scene/config/API/observation manifest 关联；必须用本模型 tokenizer 预检长度。

SFT 另建 messages parquet/JSONL，不用 prompt-only RL parquet 冒充监督数据。现有脚本将多模态 content 列表仅保留文本，会静默丢图；因此它只适合文本路线，原生 VLM 数据需另设计。

## 7. 模型候选、许可证与兼容性

外部能力在 2026-11-11 查询；本机适配结论以当前安装源码为准。

| 候选 | 架构/规模 | 许可证、图像 | 权重预算 | 当前项目兼容性与选择 |
|---|---|---|---|---|
| **Qwen3.5-4B** | dense 混合线性注意力/全注意力，32 个文本层 + 24 层视觉塔；不是 MoE | Apache-2.0；原生图像 | 发布索引张量总字节 9,319,737,856，约 8.68 GiB；以完整 checkpoint 预算，不只按名字 4B 算 8 GB | 当前 Transformers 4.57.6 与 vLLM 0.11.0 无 qwen3_5/Qwen3_5ForConditionalGeneration 注册；不能直接运行。第二阶段优先评估 |
| **Qwen2.5-Coder-1.5B-Instruct** | dense Qwen2，1.54B；全部参数而非 MoE 激活子集 | Apache-2.0；纯文本 | BF16 约 3.08 GB / 2.87 GiB | 当前两库有 Qwen2 架构；VeRL 有 Qwen2 LoRA 路径。**首个闭环推荐**，仍需单卡 forward/backward/sync 实测 |
| Qwen2.5-Coder-3B-Instruct | dense Qwen2，3.09B | **qwen-research，不是 Apache-2.0**；纯文本 | BF16 约 6.18 GB / 5.76 GiB | 更大容量的研究备选；按许可证适用范围决定，不默认可商用分发 |
| 已有 Qwen2.5-Coder-7B Q4_K_M GGUF | 7B 级文本量化推理模型 | 现有推理用 | GGUF 约 4.68 GB | [V] 可做历史 zero-shot 对照；不是 Transformers/PEFT 的可训练 BF16 base，不直接拿 GGUF 做本方案 SFT |

来源：[Qwen3.5 模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)、[配置](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)、[权重索引](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/model.safetensors.index.json)、[Coder 1.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)、[Coder 3B 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)。

Qwen3.5 发布权重若全部按 BF16 张量计，约 4.660B 个存储参数；加载时 tied tensors/忽略模块会影响实际参数统计，未下载就不声称 measured numel。dense 模型也可能有不用于纯文本路径的视觉塔，不能把“全部参数”和“文本路径活跃参数”混为一谈。

官方 vLLM recipe 列 Qwen3.5-4B 从 0.17.0+ 支持；模型卡给出 language-model-only 文本推理方式。但**推理支持不等于 vendored VeRL 的 LoRA 梯度、混合 attention kernel、position_ids、weight sync、sleep/wake 和 checkpoint 都支持**。模型 config 中 transformers_version=4.57.0.dev0 也不是“当前 4.57.6 能用”的证据，实际注册表缺失已核查。[vLLM 官方模型列表](https://recipes.vllm.ai/Qwen)、[v0.17 支持说明](https://docs.vllm.ai/en/v0.17.0/models/supported_models/)

所有候选都有聊天生成接口或能力基础，但 CaP-X 多轮恢复能力均未在本机验证。对比必须用同一 receipt 协议、相同上下文/请求预算，而不是以通用 benchmark 代替机器人评估。

**CaPtain 可借鉴流程，不能照搬资源规模。** 作者使用 Qwen3.5-4B、assistant-only SFT、物理 critic 请求级信用，并在训练/部署保持相同 harness；其 SFT 是冻结视觉塔后的语言模型全参训练，RL 使用多 H100 节点，不是本机 24 GB LoRA 的现成配置。论文附录报告 RL 双 8-GPU 节点布局，足以说明这里应缩减任务、上下文和并发。[CaPtain 论文](https://arxiv.org/html/2610.40799v1)、[作者仓库](https://github.com/waltstephen/CaPtain)

## 8. Student SFT 实施方案与初始超参数

### 8.1 框架选择

**首选 [N] Transformers + PEFT 的小型单卡训练入口**：容易明确 labels=-100、冻结参数、数据版本和 Adapter 导出；避免为了单卡 SFT 先承担 Ray/FSDP 环境集成。主 .venv 当前没有 peft、accelerate、datasets、pyarrow；因此不是“今天已有命令直接可跑”，要新建隔离训练环境并锁版本，不能改现有仿真环境。

**可复用 [S] VeRL SFT**：[fsdp_sft_trainer.py](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/trainer/fsdp_sft_trainer.py)、[MultiTurnSFTDataset](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/utils/dataset/multiturn_sft_dataset.py)、[配置](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/trainer/config/sft_trainer.yaml)已支持 LoRA、多轮 messages、assistant mask 和逐消息 loss_mask。可作为不另写 trainer 的替代入口，但现有训练器硬编码 flash_attention_2，需先验证本机 kernel；它并不因为数据集类存在就自动适配 CaP-X 的观察边界。

### 8.2 初始参数

| 项目 | 1.5B 首轮建议 | 备注 |
|---|---|---|
| 基座 | Qwen2.5-Coder-1.5B-Instruct，固定 revision | 需下载原始 safetensors/tokenizer；本轮未下载 |
| 精度 | BF16 base + BF16 计算 | PyTorch CUDA 12.8；首轮不做 QLoRA |
| LoRA | r=16，alpha=32，dropout=0 | 便于与当前 VeRL LoRA dropout 默认路径保持一致 |
| 目标模块 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj | Qwen2 语言层；冻结 embedding/lm_head 与所有外部机器人模块 |
| 序列长度 | 总长 4096，单轮 response 目标 ≤1024 | 根据真实 token 分布调整，不能直接套模型最大长度 |
| micro batch | 1 | gradient accumulation=16，effective batch=16 |
| 优化器 | AdamW，lr=1e-4，betas=(0.9,0.95)，weight_decay=0.01 | 初始值；验证过拟合后可降 lr=5e-5 |
| 调度 | cosine，warmup_ratio=0.05 | gradient clipping=1.0 |
| epoch | 2，最多 3 | 小数据优先 validation 和仿真效果，非追求训练 loss 最低 |
| 显存控制 | gradient checkpointing，use_cache=False | 单卡先 SDPA；FlashAttention 需单独验证 |
| loss | 仅合格 assistant 程序/EOS | system、task/API、观测、错误历史、padding=-100 |
| 随机性 | seed=42；保存数据/基座/API/代码 digest | 训练种子与仿真 seed 分开 |
| 保存 | 每 epoch + 验证最佳；保留 adapter、tokenizer、optimizer/RNG | checkpoint 可恢复，Adapter 可独立加载 |

Qwen2 的模块名可在实际下载的 model.named_modules() 再确认；报告不把它们套到 Qwen3.5 的 Gated-DeltaNet 投影层。Qwen3.5 后续只训练语言侧适配器并冻结视觉塔，需另外列模块匹配和 LoRA/vLLM 支持结果。

### 8.3 长代码、多轮和存储

- 数据预检逐条 tokenize 并统计 p50/p95/max、assistant token 数和截断率；禁止静默右截断。4096 放不下时先减少重复 API 文档或按请求切窗；保留完整当前程序及依赖。扩大至 8192 需要重新做显存门槛。
- 多轮只监督正确 assistant 请求；训练所见 receipt 与部署实际返回完全一致。跨请求 Python 变量的状态要能从保留历史恢复，不把不可见 namespace 当作模型知道的信息。
- 初版不 packing，避免不同 episode 串扰；以后 packing 要有隔离 attention/边界与 mask 验证。
- 保存 PEFT adapter_config.json、adapter_model.safetensors、tokenizer/chat_template、base revision、训练配置和数据 manifest。模型目录完整性与 hash 是验收项。
- 部署最简单方式：在 CPU 将 SFT Adapter 合并到原始 BF16 base，得到独立 merged-sft checkpoint；vLLM 从该目录服务，CaP-X 继续使用 OpenAI-compatible chat/completions。
- 首轮 GRPO 从 merged-sft 启动**新的 RL LoRA**。本地 VeRL 的 LoRA reference 通过 disable_adapter() 得到 base：这样 reference 恰好是冻结 SFT 模型。若直接在未合并 base 上叠 SFT/RL Adapter，会把 reference 错设为原始基座，必须避免。
- Adapter 直接用 vLLM LoRA 服务也可作为后续优化；首轮用合并权重减少加载/版本歧义。导出与原 Adapter 在固定输入上的 greedy 输出/概率要核对，不把“文件存在”算部署成功。

## 9. GRPO 链路审计与训练流程

### 9.1 当前链路逐段判定

| 环节 | 当前实现 | 判定 |
|---|---|---|
| data_source | parquet 字段 → custom compute_score → CaP-X get_config/get_exec_env | [S] 路由存在；默认低层名字错误 |
| 环境实例化 | 按 source 缓存 env；privileged 时可关 render | [S] 构造异常在 try 外；cfg.privileged 不一致 |
| rollout | vendored VeRL sync vLLM 生成 response tokens | [S] 普通本地推理已运行，不等于 VeRL rollout 已验收 |
| 程序执行 | 解码文本 → 抽 code fence → env.step | [S] 只评整段程序；不是每轮执行再生成的多轮 RL |
| reward | scalar/dict → token 最后一个有效位置 | [S] 形状接口存在；当前分数/“won”语义错误 |
| group advantage | uid 分组，rollout.n 重复，组内均值/标准差 | [S] 核心算法存在；初始状态目前不一致 |
| LoRA actor | fsdp_workers 中 LoraConfig/get_peft_model | [S] lora_rank/alpha/target_modules 配置可用；原训练脚本未启用 LoRA |
| 同步 | fsdp_vllm + TensorLoRARequest/add_lora；支持 sleep/wake | [S] 与 vLLM 0.11 的实际训练闭环未验证 |
| reference | LoRA 时 actor disable_adapter 复用 base | [S] 可省一份 reference，前提 base 是 merged SFT |
| checkpoint | FSDP manager + actor/lora_adapter 导出 | [S] Adapter 保存异常可能仅记录日志；必须实际重载验证 |
| 评估 | trainer test_freq 与外部 CaP-X runner | [S] 原脚本 test_freq=-1、val_before_train=False，不能自动给出可靠提升结论 |

源码证据：[reward loader](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/trainer/ppo/reward.py)、[naive manager](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/workers/reward_manager/naive.py)、[GRPO 算法](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/trainer/ppo/core_algos.py:265)、[trainer](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/trainer/ppo/ray_trainer.py)、[LoRA/导出](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/workers/fsdp_workers.py:1038)、[同步](/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl/verl/workers/sharding_manager/fsdp_vllm.py)。

**单 Reward Worker 结论：目前不能宣称通过，默认配置静态审计即 NO-GO。** 本轮按只读约束，没有实例化会创建服务/渲染/调试文件的真实环境。下一阶段第一项动态验收必须是单 worker 对 Oracle、pass、语法错、无限循环、假 FINISH 的结果验证；不是直接启动原训练脚本。

### 9.2 最小 on-policy 训练流程

~~~mermaid
flowchart TD
  A["冻结 SFT merged base + 可训练 RL LoRA θ_k"] --> B["同一 prompt / scene snapshot / group uid"]
  B --> C["vLLM 当前策略采样 G=4 段完整 Python"]
  C --> D["逐个恢复同一初始状态，单仿真环境执行"]
  D --> E["独立物理评分：r1...rG + 完整事件"]
  E --> F["按 uid 计算组相对 advantage"]
  F --> G["重算 old/ref log-prob，response token mask"]
  G --> H["一次 PPO/GRPO 更新，仅 LoRA"]
  H --> I["同步 adapter 到 rollout，校验版本与输出"]
  I --> J["checkpoint 重载 / 独立 seed 评估"]
  I --> B
~~~

G=4 是每个 prompt 的候选数，不是同时开 4 个仿真器。选定同一初始 snapshot 和 perception RNG，四次顺序执行；场景 hash 不同则该组作废并报告 infrastructure_error。

对组内 reward，A_i=(r_i-mean(r))/(std(r)+1e-6)，将序列级 A_i 仅广播到有效 response tokens；采用当前源码的样本标准差约定。所有候选全失败或全成功时 advantage 为 0，记录 mixed_group_rate，不能靠“代码不报错”奖励制造假学习信号。先使 SFT 在训练场景有非零成功，必要时用较简单任务课程；G 不足以弥补系统性空抓。

### 9.3 分阶段奖励

**E3 Terminal baseline [N]：**

~~~text
valid = parsed_program AND allowed_API_only AND no_runtime_error
        AND within_time/action_budget AND no_evaluator_tampering
r_terminal = 1.0 if valid AND independently_verified_final_success else 0.0
won = independently_verified_final_success AND valid
~~~

这里 valid 是 1 分的必要条件，不给格式正确、pass、观察、FINISH 声明额外分。最终成功依据隔离仿真状态和释放后稳定检查；原始任务成功与新 physical_success 同时输出。超时和依赖异常分别统计，后者应中止训练排障，不能作为普通失败大量灌入。

**E4 Shaped [N]：** 在 E3 成立后加入一次性、单调阶梯；候选初值 grasp=0.10、lift=0.15、transport=0.15，失败最高 0.40，完整成功固定 1.00。抓取、抬升、搬运必须由物理状态确认，transport 要在持物时进入目标区域；不奖励重复开闭爪、反复抬放或观察。失去物体/无效终止如何结算要在版本化 reward spec 固定；非法代码和评分篡改强制 0。E4 结果仍以 TSR 评估，不能用更高 shaped reward 宣称任务提升。

**E5 Physical Critic [N，后续]：** 单独小型 MLP 读取规范化物体 pose/速度、夹爪/接触、子目标等 privileged state，不读 policy hidden states 或 model 文字。目标为未来成功率/return，验证校准与 held-out 泛化。请求前后价值差在一致时间条件下提供信用；只观察的请求不应因墙钟推进得正分。需要多轮 rollout、request→token span、每请求 log-prob/行为策略版本和轨迹 return，目前整程序 reward 接口不能靠增加一个开关实现。

**n-step TD [N，后续对照]：** 请求为步，A_t=Σ(j=0..n-1)γ^j r_(t+j)+γ^n V(s_(t+n))-V(s_t)，terminal 时 bootstrap=0；可先 n=3、γ=0.99 作对照。不能把 clipped/time-limit truncation 一概当终止，也不能把每个 token 当物理一步。E5 与 n-step 单独实验，不和 E3 同时上线。

### 9.4 第一轮 GRPO 超参数

| 参数 | 初值 |
|---|---|
| 模型 | merged SFT 1.5B base + 新 LoRA r16/alpha32 |
| GPU / env / reward 并发 | 1 / 1 / 1 |
| train_batch_size / ppo_mini_batch_size | 1 prompt / 1 prompt；源码乘 G 后为 4 trajectories |
| rollout.n | 4 |
| micro batch | actor/logprob/ref 均 1 |
| 上下文 | prompt≤2048、response≤1024、总≤3072；超长报错 |
| rollout 采样 | temperature=0.7、top_p=0.95；eval greedy、n=1 |
| lr / ppo_epochs / clip | 1e-5 / 1 / 0.2 |
| KL | use_kl_loss=True，coef=0.01，low_var_kl；不另加 reward KL |
| entropy | 0 起步 |
| engine | sync vLLM、TP=1、enforce_eager=True、max_num_seqs=1 |
| memory | BF16 actor/base，param/optimizer offload，free_cache_engine=True |
| worker | naive reward manager、launch_reward_fn_async=False |
| smoke / pilot | 2 updates / 20–50 updates；先核对数值再谈学习提升 |
| 保存/验证 | save_freq=5、test_freq=5、val_before_train=True；console 日志 |
| 停机条件 | NaN/Inf、同步失配、场景 hash 不同、系统错误、显存余量不足、奖励作弊 |

每 prompt 四个候选顺序生成/执行也能构成 G=4；不能把 max_num_seqs 当 group size。单 prompt 更新方差大，是工程基线；通过后再将 prompt batch 增到 2–4。这些不是已测得的最佳参数。

## 10. 实测硬件、软件与单卡资源预算

### 10.1 本次只读实测

| 项目 | 当前值 |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 5090 D v2 |
| 显存 | 总 24,455 MiB ≈23.88 GiB；已用 16,394 MiB；可用 7,572 MiB（瞬时值） |
| Driver / Torch CUDA | 580.173.02 / 12.8 |
| GPU 进程 | 两个项目 Python 子进程分别 6,886、7,536 MiB；远程桌面进程约 652 MiB，其余为显示/上下文开销 |
| CPU | Intel Core i9-14900K，24 核、32 逻辑 CPU |
| RAM | 62 GiB；available 约 27 GiB |
| swap | 2 GiB，总使用约 1.6 GiB；不能把 swap 当可用训练内存 |
| 磁盘 | /dev/nvme0n1p2，1.9T，总可用 475G |
| 主环境 | Python 3.12.13，torch 2.8.0+cu128，numpy 1.26.4，Transformers 4.57.6 |
| 主仿真依赖 | MuJoCo 3.5.0、Robosuite 1.5.1、JAX 0.4.29、Ray 2.48.0 |
| 推理环境 | .venv-vllm：vLLM 0.11.0、torch 2.8.0、Transformers 4.57.6、numpy 2.2.6、Ray 2.60.0 |
| 训练依赖 | 两环境均未装 VeRL、PEFT、Accelerate、Datasets、PyArrow；.venv-rl 不存在 |
| VeRL 源码 | vendored version 文件为 0.5.0.dev；存在源码不代表已安装或和所有依赖匹配 |

GPU 计数与 CUDA 版本另外用主环境 torch.cuda.device_count()=1 验证。两项目 GPU 子进程由现有 Web 服务启动，其日志显示 SAM3/ContactGraspNet 服务；没有将每个 PID 的显存精确归属到某个模型，也没有将缓存后的驻留量误称为权重大小。本轮未停止这些服务。

项目 models 约 12G，主 .venv 约 10G，.venv-vllm 约 8.8G。共享 Hugging Face cache 约 115G，含 Qwen3-VL-8B、Cosmos、Wan、pi05 等其它项目模型；不能随意删除，也不能认为它们适合本任务。未见本方案的 Qwen3.5-4B/1.5B/3B 原始训练权重。

已有可复用权重：

- SAM3：/home/hanyu/dqb/cap-x-20261106/models/sam3/sam3.pt。
- Qwen2.5-Coder-7B GGUF：/home/hanyu/dqb/cap-x-20261106/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q4_k_m.gguf。
- ContactGraspNet：/home/hanyu/dqb/cap-x-20261106/capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt。
- 模型缓存是本地可读证据；本轮未重新校验大型权重全量 SHA、未加载模型。

**依赖风险**：项目 extra verl 钉 torch 2.8/vLLM 0.11 和特定 FlashAttention wheel；vendored VeRL requirements 写 numpy<2，而独立 vLLM 环境实际 numpy=2.2.6。主环境又有 numpy=1.26.4 override。因此不能把两个虚拟环境目录简单拼在 PYTHONPATH 上。训练侧与仿真侧优先分进程/RPC，统一 JSON 协议，独立锁版本。

此外，VeRL actor 和旧 SFT trainer 硬编码 flash_attention_2。本机普通 vLLM 推理成功不证明该训练 kernel 在 Blackwell 上前后向可用。要做一个 batch 的 forward/backward 和数值检查；失败再新增可配置 SDPA 路径，不能靠设置 use_remove_padding=False 就声称绕开了硬编码。

### 10.2 估算方法与范围

以下是**规划估计 GiB**，未运行训练 profiler。假设 BF16、LoRA r16、micro=1、总长 3072–4096、梯度检查点、无训练时录像、关闭无关模型，至少留 2–3 GiB 余量。Qwen3.5 按完整发布权重保守估算；图片 token 的额外激活不在纯文本预算内。

| 模型 | BF16 权重 | SFT LoRA 峰值估计 | 全参 AdamW 仅模型状态估计 | GRPO actor 阶段估计 | 独立 vLLM rollout 估计 |
|---|---:|---:|---:|---:|---:|
| Coder 1.5B | 2.87 | 6–10 | 17–23，再加激活等 →约 23–32+ | 7–12 | 4–6 |
| Coder 3B | 5.76 | 10–16 | 35–46，再加激活 | 11–18 | 7–10 |
| Qwen3.5-4B 完整包 | 8.68 | 14–21（文本，视觉塔冻结） | 52–69，再加激活；若只全参语言侧可降低，但仍需重算 | 16–24+，且适配未通过 | 10–14；纯语言加载可能更低，未测 |
| 7B 级 BF16 | 约 14 | 20–30+ | 超单卡范围 | 非首轮预算 | 约 16–20+ |

全参状态以参数量 ×12–16 bytes 估算（权重、梯度、Adam moments，是否有 FP32 master copy 导致差异），不是任何实现的恒定精确值。LoRA 也保留 base 权重和完整前后向激活，并非只有 adapter 占显存。Qwen2 1.5B 按上述七类线性层/r16，Adapter 为千万参数量级，优化器状态远小于激活；实际 numel 由加载后统计确认。

KV cache 可按 2×层数×KV heads×head_dim×序列长度×bytes×并发估算，再加 runtime buffer；1.5B 的 28 层、2 KV heads、head_dim=128、4096 tokens、BF16 单序列约 112 MiB。vLLM 的总显存还含模型、算子工作区和分配器预留，不能只用 KV 估算。

### 10.3 感知、仿真和组合预算

- [V] 当前两个视觉服务进程合计约 **14.08 GiB** 驻留。这已经高于“只按 SAM3 权重 3.45 GB + GraspNet 权重 26 MB”所得预算，必须以实际服务峰值为准。
- [估计] 单 Robosuite EGL 环境额外预留 0.5–2 GiB GPU、1–3 GiB RAM；无渲染 privileged 环境 GPU 可明显更小，均未单独实测峰值。
- [估计] 显示和其它进程留 1–2 GiB，再留训练余量 2–3 GiB。
- 因而即使 1.5B，**视觉服务 + vLLM + actor 同驻也可能超过 24 GB**；当前 7.4 GiB 空闲不能直接启动训练。
- CPU offload 会占 RAM 并增加 PCIe 传输时间，当前 available 27 GiB、swap 已高使用，不能无限 offload。建议单阶段 host available≥16 GiB，1.5B 工作预算 8–16 GiB RAM；3B/4B 需重新测量并清理任务自有驻留，不动其它项目文件。

### 10.4 可行调度

1. **SFT 阶段**：完成采集后释放本任务视觉/LLM 服务；只保留训练进程。建议 1.5B LoRA 峰值≤20 GiB、空闲≥3 GiB，无 swap 持续增长。
2. **最小 GRPO 工程验收**：可先单独 privileged、无渲染、无 SAM3/GraspNet；actor/vLLM 顺序调度，policy offload、vLLM sleep/wake，单 reward worker。只报告 privileged 结果。
3. **Vision GRPO**：同一 θ_k 下先生成 G 段代码，卸载/睡眠 rollout 并 offload actor；再加载视觉服务顺序执行全部候选；释放视觉服务后训练一步，再同步 θ_(k+1)。这种整轮屏障保持 on-policy。服务切换成本高，所以先测 2 轮，不能承诺吞吐。
4. 上述完整的视觉服务调度器 **[N] 尚不存在**。当前 VeRL 有 actor offload 和 vLLM sleep，但不会自动释放外部 SAM3/GraspNet 进程；free_cache_engine 不能完成整个调度。
5. 如果仍 OOM，优先缩短实际序列/减 batch/选择 1.5B；可研究 CPU 感知或先单独做 SFT 与在线评估。不为凑进显存改成长期离线数据 PPO，却继续称 on-policy GRPO。
6. 初始观测可缓存用于重放/诊断，但动作后的图像不能用动作前缓存替代。训练后评价必须真实调用当前视觉链路。

## 11. 需要新增或修改的文件

本节全部为 [N] **计划**，本轮没有实施。路径统一位于 /home/hanyu/dqb/cap-x-20261106；表内链接指向现有源码，新增路径列为完整路径。

| 文件 | 当前作用 | 为什么改 / 拟改内容 | 复用 |
|---|---|---|---|
| [capx/envs/tasks/__init__.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/__init__.py) | env/config 注册 | 修默认映射；显式 privileged；为标准视觉使用独立 source，避免标签伪装 | 注册机制全复用 |
| [capx/envs/simulators/robosuite_cubes.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cubes.py) | Stack 包装 | 正确重置 RNG/采样器；snapshot；XYZW→WXYZ；真值观察与 actor 观察分离 | 仿真/成功判断复用 |
| [capx/envs/simulators/robosuite_cube_lift.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_cube_lift.py) | Lift 包装 | 若用课程任务，同步 seed 与观测修复 | 任务复用 |
| [capx/envs/simulators/robosuite_base.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/simulators/robosuite_base.py) | 公共运动/相机 | 记录 FK/末端误差、动作预算、接触/帧同步、稳定成功检查支撑 | 控制实现不重写 |
| [capx/envs/tasks/base.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/base.py) | Python 执行 | 移除直接真值/环境对象暴露；进程化 API 代理；硬超时；API reset hook；统一 termination | prompt/env.step 外层复用 |
| [capx/integrations/franka/control.py](/home/hanyu/dqb/cap-x-20261106/capx/integrations/franka/control.py) | Vision API | receipt、frame/TCP 合约、置信度、世界竖直范围、抓取验证；调试图不覆盖根文件 | SAM3/GraspNet/IK 调用复用 |
| [capx/integrations/franka/control_privileged.py](/home/hanyu/dqb/cap-x-20261106/capx/integrations/franka/control_privileged.py) | 真值 API | 同协议回执与 episode reset，保持 privileged 标签 | 参考控制复用 |
| [capx/llm/client.py](/home/hanyu/dqb/cap-x-20261106/capx/llm/client.py) | API 客户端 | request/usage/version/finish_reason、有限重试、DeepSeek 正确字段、模型能力验证 | HTTP/streaming 复用 |
| [capx/envs/trial.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/trial.py) 与 [runner.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/runner.py) | 运行与产物 | scene manifest、API 前后 trace、纯执行计时、完整请求计数；unique 输出目录 | 主循环/录像复用 |
| [capx/envs/launch.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/launch.py) 与 [scripts/local/run.py](/home/hanyu/dqb/cap-x-20261106/scripts/local/run.py) | 入口参数 | seed/scene manifest、student model/endpoint 参数、校验模型能力；现有模型名硬编码不适合多 Student | 入口复用 |
| [capx/cli/prepare_verl_dataset.py](/home/hanyu/dqb/cap-x-20261106/capx/cli/prepare_verl_dataset.py) | RL parquet | 真正 reset、场景 hash、分组划分、长度/重复校验；读取采集 manifest 的纯数据路径，避免训练环境 import 仿真 | parquet schema 复用 |
| [verl_agent_reward/capx_franka_reward.py](/home/hanyu/dqb/cap-x-20261106/verl_agent_reward/capx_franka_reward.py) | reward hook | terminal/shaped 分开；正确 won；构造异常捕获；调用隔离仿真 worker；统一 metrics | compute_score 签名复用 |
| [scripts/train_franka_grpo.sh](/home/hanyu/dqb/cap-x-20261106/scripts/train_franka_grpo.sh) | 训练启动 | 单卡 LoRA 参数、naive/同步 reward、正确 dataset 验证、validation、console、精度/显存、可恢复输出 | main_ppo 调用复用 |
| [capx/envs/tasks/franka/franka_pick_place.py](/home/hanyu/dqb/cap-x-20261106/capx/envs/tasks/franka/franka_pick_place.py) | 任务 prompt 与 Oracle | 根据确认后的 API 合约修正 extent 竖直高度和进近方向说明；同步 Oracle 与 SFT 文档 | 任务结构复用 |
| /home/hanyu/dqb/cap-x-20261106/capx/training/trajectory_schema.py | 无 | episode/API/request schema、mask、provenance、哈希 | 新增 |
| /home/hanyu/dqb/cap-x-20261106/capx/training/sim_worker.py | 无 | .venv 仿真进程服务，只接受允许程序/API；父进程超时与重建 | 新增，内部复用 env |
| /home/hanyu/dqb/cap-x-20261106/capx/training/phase_scheduler.py | 无 | Vision GRPO 阶段屏障、服务生命周期、版本一致性和资源门槛 | 新增；复用 VeRL offload |
| /home/hanyu/dqb/cap-x-20261106/scripts/training/audit_rollouts.py | 无 | A/B/C/D 配对、逐项真值替换、统一评估汇总 | 新增，复用 runner |
| /home/hanyu/dqb/cap-x-20261106/scripts/training/build_sft_dataset.py | 无 | 成功筛选、来源转换、去重、scene split、assistant masks | 新增 |
| /home/hanyu/dqb/cap-x-20261106/scripts/training/train_sft_lora.py | 无 | Transformers/PEFT 单卡训练、恢复/导出 | 新增；也可不用它而走已有 VeRL SFT |
| /home/hanyu/dqb/cap-x-20261106/scripts/training/export_adapter.py | 无 | CPU Adapter 合并、hash、重载对齐 | 新增，复用 PEFT merge_and_unload |
| /home/hanyu/dqb/cap-x-20261106/training/pyproject.toml 与 /home/hanyu/dqb/cap-x-20261106/training/uv.lock | 无 | 锁训练依赖，避开主项目 numpy/extra 混合；先测 torch/attention/vLLM/LoRA 再锁 | 新增，不改主 lock |
| /home/hanyu/dqb/cap-x-20261106/training/configs/{audit,sft,grpo_terminal,grpo_shaped}.yaml | 无 | 固定输入/输出/模型 revision/资源上限；显式区分观察来源 | 新增 |
| /home/hanyu/dqb/cap-x-20261106/tests/training/ | 无 | seed/状态重放、观察泄漏、no-op reward、timeout、mask、Adapter 同步与保存恢复测试 | 新增必要行为测试 |
| [docs/rl-training.md](/home/hanyu/dqb/cap-x-20261106/docs/rl-training.md) | 旧训练说明 | 新验收结果、正确端点/环境、单卡命令；论文指标与本机指标分开 | 说明结构复用 |

只有 attention kernel 或 vLLM 内部同步确实不兼容时，才修改 vendored VeRL 的 fsdp_workers.py / fsdp_sft_trainer.py / fsdp_vllm.py；单卡 SDPA 支持需明确配置和数值测试，不先大范围升级第三方源码。Qwen3.5 另建隔离适配分支，当前能工作的仿真环境不随其升级。

## 12. 具体执行命令与前置门槛

**本节为下一阶段 runbook，本轮没有执行训练/下载/采集命令。** 分类为：

- **现有接口**：模块/参数已从源码核对，但仍可能因缺环境或 P0 问题不能运行。
- **修复后运行**：只有完成前述修复、依赖锁和单 worker 验收才可使用。
- **拟议新接口**：尚未实现，下面明确给出开发后的入口约定，不能现在复制执行。

### 12.1 只读检查（当前可执行）

~~~bash
cd /home/hanyu/dqb/cap-x-20261106
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv
free -h
df -h /home/hanyu/dqb/cap-x-20261106
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
cat capx/third_party/verl/verl/version/version
~~~

主环境无 VeRL/PEFT，不能把 import 失败说成训练脚本自身 bug。本轮采用 importlib.metadata 读取 dist-info，没有运行安装或同步命令。

### 12.2 复用现有四组入口（会产生新试验，仅用于后续非配对 smoke）

~~~bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/run.sh oracle --trials 1 --output-dir outputs/plan-smoke/A-v1
bash scripts/local/run.sh vision-oracle --trials 1 --output-dir outputs/plan-smoke/B-v1
bash scripts/local/run.sh privileged --backend cloud --trials 1 --output-dir outputs/plan-smoke/C-v1
bash scripts/local/run.sh vision --backend cloud --trials 1 --output-dir outputs/plan-smoke/D-v1
~~~

上述参数存在；DeepSeek key 从 .env.local 读取，不写在命令行。运行前确认输出目录未存在。**入口目前没有 seed 参数，因此这四条不能代替第 5 节的配对实验。** 目前也不要为了收数据重复跑；先做 P0/trace 修复。

正式配对采集/评价采用拟议接口（**[N] 需要开发**；--config 是拟定义的单参数入口，不是现有功能）：

~~~bash
.venv/bin/python scripts/training/audit_rollouts.py --config training/configs/audit.yaml
~~~

audit.yaml 必须含 A/B/C/D 组、冻结 seed/snapshot manifest、模式/模型、单 worker、动作和时间预算、唯一输出目录；先 5 seeds，后 20 seeds。同一入口增加配置中的 operation=collect/evaluate/replay，不能通过猜测现有 launch 的 --seed 或 --manifest 参数来运行。

### 12.3 隔离训练环境与模型获取

**[N] 前置开发工作**：生成并验证 training/pyproject.toml 与 training/uv.lock，锁 Python 3.12、torch/Transformers/PEFT/Accelerate、vendored VeRL、vLLM、Ray、attention kernel。优先沿用已运行的 torch2.8/vLLM0.11/Qwen2；具体 PEFT/attention wheel 组合必须先在隔离环境测试。仿真执行通过 .venv 子进程，不把两个 site-packages 混用。

在锁文件已存在并验收后，使用已有 uv CLI 的真实命令：

~~~bash
/home/hanyu/.local/bin/uv sync --project /home/hanyu/dqb/cap-x-20261106/training --frozen
~~~

该命令将创建 training/.venv；**今天没有该 manifest/lock，所以不能运行**。不照抄主项目 uv sync --extra verl 去改变已可用主环境。

模型下载使用训练环境的 huggingface_hub 官方 Python API；以下仅在训练 lock 包含该库、准备好网络和磁盘后运行，首次取 main 后必须记录实际 commit，以后固定该 commit：

~~~bash
training/.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
revision = HfApi().model_info(model_id).sha
print("PINNED_MODEL_REVISION", revision)
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir="/home/hanyu/dqb/cap-x-20261106/models/qwen2.5-coder-1.5b-instruct",
)
PY
~~~

国内镜像可在后续网络配置阶段选择，但仍校验 repo/revision/hash，不通过镜像名字推断模型相同。公开模型通常不需额外 key；遇到访问控制再按实际错误处理。本轮没有需要用户补充的 DeepSeek key。

### 12.4 构建 SFT 与训练

拟议新增入口（**[N] 不存在，开发后执行**）：

~~~bash
training/.venv/bin/python scripts/training/build_sft_dataset.py --config training/configs/sft.yaml
training/.venv/bin/python scripts/training/train_sft_lora.py --config training/configs/sft.yaml
training/.venv/bin/python scripts/training/export_adapter.py --config training/configs/sft.yaml
~~~

sft.yaml 固定第 6/8 节的来源过滤、数据路径、模型 revision、mask、LoRA/优化器参数与独立输出目录。export 动作只合并指定 Adapter，不覆盖 base。

若选择复用已有 VeRL SFT，以下模块与参数**已在 vendored 源码找到**；前提是训练环境/attention 验收通过、messages parquet 已由采集器生成：

~~~bash
PYTHONPATH=/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl \
training/.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/home/hanyu/dqb/cap-x-20261106/data/sft-v1/train.parquet \
  data.val_files=/home/hanyu/dqb/cap-x-20261106/data/sft-v1/val.parquet \
  data.multiturn.enable=True \
  data.multiturn.messages_key=messages \
  data.max_length=4096 \
  data.truncation=error \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  model.partial_pretrain=/home/hanyu/dqb/cap-x-20261106/models/qwen2.5-coder-1.5b-instruct \
  model.fsdp_config.model_dtype=bf16 \
  model.enable_gradient_checkpointing=True \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  'model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  optim.lr=1e-4 \
  optim.warmup_steps_ratio=0.05 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=2 \
  trainer.logger='[console]' \
  trainer.default_local_dir=/home/hanyu/dqb/cap-x-20261106/checkpoints/sft-v1 \
  trainer.resume_mode=disable
~~~

这是替代训练路径，不要在同一实验中把两种 trainer 的结果混成一组。数据字段/mask、模型精度及保存格式须分别检查。

### 12.5 单 Reward Worker 验收

现有 compute_score 签名可直接调用，以下是**修复后、服务/隔离执行已就绪才执行**的单样本检查；当前默认映射/奖励/seed 未修时不能通过：

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
from capx.envs.tasks import get_config, get_exec_env
from verl_agent_reward.capx_franka_reward import compute_score

source = "franka_robosuite_pick_place_code_env"
cfg = get_config(source)
assert cfg.privileged is True, "先修复 registry 的 privileged 语义"
oracle = get_exec_env(source).oracle_code
assert isinstance(oracle, str) and oracle.strip()
for name, program in [("oracle", oracle), ("no_op", "pass"), ("bad_syntax", "if :")]:
    result = compute_score(source, program, {}, {"seed": 101})
    print(name, result)
PY
~~~

先确认该任务类确实以类属性提供 oracle_code（当前源码如此）。这只验证直接调用；还要在实际 Ray/naive manager 路径用同样三例、跨 seed 重复，并验证超时、假 FINISH、隐藏状态访问被拒绝。无限循环测试必须在完成硬超时子进程后才加，不能直接放进当前 exec 主进程。

返回至少要包括 score、won、sandbox_rc、task_completed、physical_success、seed/state_hash、elapsed、error_type；Oracle 应为真成功，pass 和语法错应为 0 且 won=False。所有组内 state_hash 必须相同。

### 12.6 RL parquet 与最小 GRPO

当前 prepare_verl_dataset 的以下参数存在；**先修复 seed/注册/manifest 和依赖隔离**，再执行。推荐训练环境通过新 sim_worker 获取 prompt 或读取冻结采集 manifest，避免导入整套仿真依赖；这是第 11 节明确的改造内容。

~~~bash
training/.venv/bin/python -m capx.cli.prepare_verl_dataset \
  --output-dir data/grpo-v1 \
  --train-size 32 \
  --val-size 20 \
  --data-source franka_robosuite_pick_place_code_env \
  --seed 1000 \
  --num-workers 1
~~~

此 source 是 **Privileged 工程基线**。不能用它的成功率声称 Vision GRPO。现有脚本产物名为 test.parquet，实际在此作为 validation；真正 held-out test 另外冻结，不用于 checkpoint 选择。

以下使用已核对的现有 Hydra 参数，**仅供修复/安装/单 worker/Adapter 同步门槛通过后的 2-step smoke**。merged-sft-v1 为第 12.4 步产物；当前并不存在。训练环境须可导入项目（由新训练 manifest 配置 editable 包/路径），以及修复后的 reward RPC。

~~~bash
MUJOCO_GL=egl \
PYTHONPATH=/home/hanyu/dqb/cap-x-20261106:/home/hanyu/dqb/cap-x-20261106/capx/third_party/verl \
training/.venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=/home/hanyu/dqb/cap-x-20261106/data/grpo-v1/train.parquet \
  data.val_files=/home/hanyu/dqb/cap-x-20261106/data/grpo-v1/test.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=/home/hanyu/dqb/cap-x-20261106/checkpoints/merged-sft-v1 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=naive \
  reward_model.launch_reward_fn_async=False \
  custom_reward_function.path=/home/hanyu/dqb/cap-x-20261106/verl_agent_reward/capx_franka_reward.py \
  custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.logger='[console]' \
  trainer.total_training_steps=2 \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.default_local_dir=/home/hanyu/dqb/cap-x-20261106/checkpoints/grpo-terminal-smoke-v1
~~~

0.30 是 vLLM 分配比例起点，不是精确显存硬上限；加载/峰值要实测。不调用原 shell 脚本，避免其硬编码大 batch、Prime 并发与 osmesa 覆盖。正式 pilot 改 total_training_steps=20–50、save/test_freq=5；其它变量保持一致。Shaped reward 用独立版本配置和输出目录，不能在原 terminal run 过程中悄悄换奖励。

### 12.7 Adapter 部署与评估

VeRL LoRA checkpoint 预计含 global_step_N/actor/lora_adapter；检查 adapter_config.json 和 adapter_model.safetensors 的实际存在、非空与重载结果，不只看“Saved”日志。FSDP 完整合并工具已有：

~~~bash
training/.venv/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /home/hanyu/dqb/cap-x-20261106/checkpoints/grpo-terminal-smoke-v1/global_step_2/actor \
  --target_dir /home/hanyu/dqb/cap-x-20261106/checkpoints/grpo-terminal-smoke-v1/export
~~~

**注意**：LoRA 分支可能导出 Adapter 而非完整 base，不能未经目录检查就把 export 当完整模型服务。首选通过拟议 export_adapter.py 将 RL Adapter 合并到它对应的 merged-sft base；保存两层 lineage，导出完整 HF 模型到 checkpoints/merged-grpo-v1。

完成导出、停止任务自有训练进程、显存满足门槛后，以下 vLLM 参数均来自现有 0.11 接口：

~~~bash
.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/hanyu/dqb/cap-x-20261106/checkpoints/merged-grpo-v1 \
  --served-model-name capx-student \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.25 \
  --enforce-eager
~~~

先做单次现有入口验证，再用开发后的配对评价器跑锁定 test manifest：

~~~bash
.venv/bin/python -m capx.envs.launch \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --server-url http://127.0.0.1:8000/v1/chat/completions \
  --model capx-student \
  --max-tokens 1024 \
  --temperature 0 \
  --total-trials 1 \
  --num-workers 1 \
  --output-dir outputs/student-grpo-smoke-v1
~~~

运行时需已有本地 common.sh 对环境变量的配置或等价配置；所有新目录必须不覆盖已有产物。这里只引用现有参数，不添加不存在的 --seed；正式 20-scene 评估依赖新 audit_rollouts.py。Qwen3.5 适配、Physical Critic 与 n-step TD 现在没有可执行项目命令，不给出伪装成现成功能的一键启动。

## 13. 实验矩阵、指标定义与 GO / NO-GO

### 13.1 实验矩阵

每个模型都在同一任务/API 版本、同一锁定 test scene manifest 上评价。Privileged 与 structured/standard Vision 分开成表；终端奖励和 shaped reward 的训练分数不能跨实验直接比较。

| 实验 | 模型/训练 | 控制项 | 进入条件与验收 |
|---|---|---|---|
| E0 | DeepSeek V4 Pro，Zero-shot Teacher | 固定 provider/model 版本记录、thinking/输出预算、同一 observation | 完成 A/B 控制组；5-seed smoke 后至少20场景；得到真实 Teacher 上限与成本 |
| E1 | Coder 1.5B，Zero-shot | 与 E2 同 tokenizer/chat template/API、相同测试 seed | 完整加载/推理成功、记录所有失败；低 TSR 本身不阻止 SFT |
| E2 | 同 Student，SFT LoRA | 与 E1 唯一核心差异为 Adapter | 见下方 G3：可重载、代码成功率和物理成功率提升；验证集选择 checkpoint |
| E3 | SFT + Terminal GRPO | 保留 E2 base、相同数据/采样/训练预算 | G4/G5 通过；先2步工程 smoke，再20–50步 pilot |
| E4 | SFT + Shaped GRPO | 与 E3 相同初始化、seed、group、计算预算，仅改 reward spec | E3稳定后；TSR改善且无循环动作/观察作弊 |
| E5 | SFT + Physical Critic | 单独多轮系统、独立 critic | 延后；E3/E4闭环可信，长程信用分配确实成为瓶颈后才进入 |

Qwen3.5-4B 作为后续同矩阵的模型替换组；文本-only 与 native VLM 单独报告。7B GGUF 历史结果可作背景，不拿不同量化/接口/seed 的旧记录直接作为 E1。

### 13.2 统一指标

| 指标 | 定义与要求 |
|---|---|
| Task Success Rate | 独立物理终态成功 episode / 全部评价 episode；另列原始 task_completed |
| Code Execution Success Rate | 无解析/运行/预算错误的 episode 比例；逐代码块成功率另列 |
| Grasp Success Rate | 满足接触/随动定义的 episode / 全部 episode；另列成功抓取次数/实际抓取尝试 |
| Lift Success Rate | 满足高度与稳定时间的 episode / 全部 episode；不能仅检查末端高度 |
| Placement Success Rate | 释放并退出后目标稳定成立 / 全部 episode；另列条件成功率 |
| 代码块/请求数 | 全 episode 均值及成功 episode 均值；HTTP retry 单列 |
| 耗时 | codegen、exec、perception、总时间，报告均值/中位数/p95；失败按实际耗时记录 |
| Token | 输入、输出、reasoning、缓存命中、总 token；未知记 N/A，非 0 |
| 成本 | API 返回 usage × 当时费率；训练/本地推理同时报 GPU 小时与墙钟时间 |
| 泛化 | 固定 held-out seed/布局家族；按位置、旋转、遮挡、光照等切片 |
| GRPO 健康度 | mixed_group_rate、adv均值/方差、KL、clipfrac、loss、grad_norm、OOM/系统错误率、策略版本一致性 |

模型说 FINISH 不算物理成功；没说 FINISH 但任务物理成功，仍记录 physical_success=True，并另记 agent_finish=False。如果以后使用“严格自报告完成”指标，要保留 host success 和 strict success 两列，不能偷偷改变 TSR 定义。

系统错误单独报出。正式评价遇到系统故障应修复后整批按原 manifest 重跑，保留失败原记录，不能只删除失败样本提高分数。每轮评价固定允许的请求/动作预算；多轮模型不能无限请求而与单轮模型直接比均值。

至少 20 个独立测试场景仅是早期验收：1 个场景就是 5 个百分点。报告成功数/分母和 Wilson 95%区间；E2/E3/E4用配对场景差值/配对 bootstrap 作探索性比较。20场景上多成功2次不足以宣称普遍统计显著，正式结论应扩到50–100场景，并重复训练种子。

### 13.3 分层门槛

以下阈值为工程验收建议，实施前冻结，不是当前已经达到的成绩。

| Gate | GO | NO-GO / 处理 |
|---|---|---|
| G0 可复现与观测边界 | 相同 snapshot 3次 reset 的初态一致；同程序可重放；actor 无真值/评分对象访问；错误/超时可回收 | 初态不一致、真值泄漏、卡死不可终止 → 不采训练数据 |
| G1 控制与感知 | A 在20场景≥19成功；B在20场景≥18成功；逐 API 误差/接触日志完整 | A差先修控制；A好B差先修感知/变换；当前B=1/3，无资格放量 |
| G2 数据 | ≥100条重放成功episode；≥50训练场景；来源/模型/配置/seed齐全；scene split无交叉；mask抽查全部正确 | 成功不足、同程序复制、隐藏状态污染 → 回采/修复 |
| G3 SFT | 8条样本可过拟合工程验证；Adapter 重载一致；独立test≥20场景，代码执行成功≥90%；TSR初步目标≥40%且优于E1 | 训练loss降但TSR不升 → 检查数据/接口/感知，不直接上RL |
| G4 Reward Worker | Oracle物理成功得1；pass/语法错/假FINISH得0；timeout/非法访问正确处理；同组初态一致；主进程和Ray路径都通过 | 任一错误 → 不启动GRPO；本轮目前NO-GO |
| G5 GRPO工程 | 2步无OOM/NaN；LoRA梯度非零且参数更新；base/感知不变；rollout版本匹配；checkpoint重载；主机不持续swap | “只有日志在跑”不算GO；sync/adapter缺失立即停 |
| G6 GRPO学习 | pilot有足够非零adv组，mixed_group_rate建议≥20%；validation不退化；test至少20，探索性目标比E2多≥2成功且成本受控 | 全组0分/全组1分先改课程/数据，不堆轮数；TSR退化回到E2 |
| G7 Shaped | 与E3等预算，TSR至少不退，动作/请求没有异常膨胀；失败轨迹无重复领取子目标奖励 | 只涨shaped reward不涨TSR → reward设计失败 |
| G8 VLM/Critic | 有明确文本瓶颈/长程信用问题；兼容性、数据量、显存独立验收 | 当前没有必要直接进入 |

允许在 G1 的 Vision 部分未通过时做单独的 Privileged G2–G5 工程验证，但任何输出标题、模型卡和指标都必须带 privileged；不得把这条旁路当作 Vision 已过门槛。

### 13.4 时间与 API 预算

已有 DeepSeek CLI 两例生成耗时约34–74秒，完整 runner约36–91秒；不能据此承诺稳定吞吐。先20–30次 pilot 测 p95、失败率和真正 usage。采集 200 次串行 Teacher 请求，按每次60–90秒粗估约3.3–5小时，服务启动/重放/人工检查额外计；如果只有10%成功，就必须回排障而不是默认发数千次。

费率随时间变化：审计时官方 Pro 每百万 token 峰时未缓存输入$1.32、缓存输入$0.044、输出$3.96，非峰时减半。未来采集应把费率快照和时间一起保存。[官方费率](https://api-docs.deepseek.com/quick_start/pricing/)

示例预算而非已发生费用：200次请求、每次2,000未缓存输入+4,000总输出（包括计费推理部分），峰时约$3.70；若每次实际输出接近16,384 token上限，费用显著更高。成本必须按实际 usage 结算，不能只对最终 Python 文本 tokenize 后当总账单。第一批建议明确请求数上限和费用上限，由采集器到限停止。

SFT pilot 规划预留0.5–2 GPU小时（含载入、验证和工程排错），不是吞吐承诺；GRPO前10个组测得实际每组时间后，按总组数外推，不能套原脚本1小时29分的注释。Vision 分阶段加载模型的开销可能超过程序执行时间，必须计入。

## 14. 实施顺序、工作量与最终建议

### 14.1 实施顺序

以下按单个熟悉 Python/仿真工程的开发者估计，1人日约一个工作日；涉及视觉根因和训练 kernel 的时间不确定。

| 顺序 | 工作包 | 预计投入 | 交付与停止条件 |
|---|---|---:|---|
| 1 | P0：注册/seed/观察边界/奖励/超时；建立trace | 2–3人日 | 单环境可重放、no-op不领奖、真值不可越界 |
| 2 | A/B/C/D + 感知/坐标/抓取消融 | 1–3人日 | 5-seed smoke、20-scene基线、主失败证据；不通过不放量 |
| 3 | 数据schema、来源过滤、split、成功采集 | 1–2人日工程 + 采集墙钟 | ≥100条成功重放数据与固定test manifest |
| 4 | 1.5B LoRA SFT、导出、E1/E2 | 1–2人日 + GPU运行 | 可重载模型、真实机器人仿真指标对比 |
| 5 | VeRL隔离环境、单reward worker、同步/checkpoint smoke | 1–3人日 | G4/G5通过；若kernel不兼容可增加1–3人日 |
| 6 | Terminal GRPO pilot，E3评价 | 1–2人日 + rollout墙钟 | 有效组/任务提升证据；没有则回数据/SFT |
| 7 | Shaped reward、Vision阶段调度、E4 | 2–4人日 | 等预算提升且无作弊；可按瓶颈拆开做 |
| 8 | Qwen3.5文本/VLM、Physical Critic、多轮RL | 单独立项，至少数人日到数周 | 不纳入第一周承诺，先明确必要性与资源 |

这些工作有依赖，总体不保证一周完成全部 SFT+GRPO。第一周优先完成可信的数据和一个小模型闭环，而不是强行产生一条 RL 曲线。

### 14.2 第一周实际完成的三个任务

1. **修复可复现性与物理评价，做出20场景诊断报告。** 完成 seed/snapshot、正确四元数/坐标合约核查、API 前后 trace、隐藏状态隔离、terminal reward 和硬超时；先跑5个seed，再A/B/C/D扩至20个，明确空抓/滑落/放置失败的主因。若视觉未通过，明确标注 NO-GO，不继续盲采。
2. **产出可重放、无泄漏的第一版 SFT 数据集。** 建立provenance与Train/Val/Test场景隔离；目标100–300条成功episode，既保存失败库又拒绝失败正样本。Vision不足就先产单独privileged工程集并继续补视觉数据，不能混标。
3. **完成1.5B LoRA SFT → 导出 → CaP-X评估闭环，并验收单Reward Worker。** 先8样本过拟合检查mask/梯度，再E1/E2至少20独立场景比较；保存/重载Adapter和merged模型。GRPO仅在G4/G5通过后做2步工程smoke，正式训练移到下一阶段。

若第一个任务的视觉排障超过预计，第三项只在清楚标注的privileged基线上验证软件链路；不能为了“按时完成”降低物理成功标准。

### 14.3 当前24 GB的最短路径

**可靠结构化感知与执行反馈 → 少量通过重放的Teacher/Oracle数据 → Apache-2.0的Coder 1.5B LoRA SFT → frozen merged-SFT base +新RL LoRA → 单环境G=4 Terminal GRPO。**

先使用文本代码模型和现有外部视觉组件，训练时分阶段释放显存。规模升级的顺序是：数据质量与成功率 → 1.5B闭环 → 必要时更大文本模型 → Qwen3.5-4B适配 → 多轮/VLM/Critic。3B候选的研究许可证与Qwen3.5的兼容性成本分别考虑，不因为参数名接近就互换。

### 14.4 现在不值得做

- 直接运行原7B、256 batch、G15、Prime64进程训练脚本。
- 在视觉Oracle不稳定、seed不真实、奖励把pass标为won时采大数据或长跑RL。
- 把所有失败程序作为SFT正样本，或把privileged真值偷偷混进Vision输入。
- 用现有GGUF直接做PEFT训练，或把另一个项目缓存的大模型当本任务训练基座。
- 24GB上首轮全参4B/7B、66k长上下文、多仿真并发，或照搬CaPtain的多H100训练布局。
- 为Qwen3.5一次性升级主仿真环境；原生VLM、Physical Critic、n-step TD、复杂reward同时上线。
- 只看sandbox_rc=0、loss下降、shaped reward上涨就宣布机器人能力提升。
- 现在投入真实机器人接入；当前应先完成可验证的仿真闭环。

**本轮最终状态**：只读审计完成；历史推理链路已有真实运行证据，视觉稳定性和训练闭环尚未验收。SFT/GRPO仍是下一阶段开发任务，未安装依赖、未下载Student、未启动训练。报告中的命令均区分现有接口、修复后的调用和未实现的新入口。
