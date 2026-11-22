# CaP-X 主项目目录迁移验收报告

审计日期：2026-11-11（Asia/Ulaanbaatar）。
范围：只读核对迁移、运行现有测试、新增本报告和独立审计结果；未修改业务源码、依赖、服务配置，未下载模型、调用大模型 API、启动训练或操作真实机器人。

## 1. 总结与 GO 判断

- **正式且唯一开发入口：`/home/hanyu/dqb/cap-x-20261106`。**
- **P0 源码完整迁移：通过。** 对照提交和原工作区核对实现，14 个关键文件与 P0 提交字节一致；531 个可比较已跟踪文件中 530 个与原工作区相同，仅 `.gitignore` 存在预期迁移差异。
- **API 配置保留：通过配置层核验。** 两份私有配置存在且权限 0600；本轮未验证密钥有效性、模型授权、余额或真实响应。
- **本轮原 P0 集合：60 项收集，58 通过、2 跳过、0 失败。** 历史 60/60 不能代替本轮结果。另 8 项 API Mock 测试全部通过。
- 历史 Oracle Snapshot 在正式目录成功重放，初始和最终 hash 与历史完全相同。
- **主目录继续开发：GO。完整视觉 A/B/C/D 配对实验：当前 NO-GO。** 需后续单独处理 Seed 感知服务运行版本和 Web 兼容性，再补齐两个视觉集成测试。本轮不执行这些修改。

## 2. 路径、Git 和重复目录

桌面聊天初始 cwd 是 `/home/hanyu/Documents/Codex/2026-11-10/cap-x-20261106-tar-gz`；这只是审计工具及结果存放目录。对项目的 Git、测试及运行时命令均显式指定正式项目 workdir，不在旧工作区执行被测代码。

| 项目 | 核验结果 |
|---|---|
| 正式目录 realpath | /home/hanyu/dqb/cap-x-20261106 |
| Git 根目录 | /home/hanyu/dqb/cap-x-20261106 |
| 当前分支 | p0-repro-foundation |
| HEAD | 363559a3d3a36ecea9872741dfc08593a457ce65 |
| P0 基准提交 | a7a0931af644c242a535b992425e83bd5829f5ee |
| 原独立工作区 HEAD | b1d11f10d17bad43e6cd2361bc2e982cb6b8d2a6 |
| 已跟踪文件未提交修改/暂存修改 | 无 |
| 本轮新增项目文件 | 本报告；测试产物写入独立审计目录 |

审计前和测试后 git status 相同：

```text
?? .env.example
?? .env.local.example
?? .gitmodules
?? .rayignore
?? README.AICODER.md
?? README.LOG.md
?? README.md
?? SUBMODULES.txt
?? web-ui/
```

上述未跟踪文件迁移后实际存在，但尚未纳入版本控制，尤其 `web-ui/`。不能声称仅凭 HEAD 就能重建完整 Web 项目。本轮未替用户 add/commit。

原独立工作区仍在：
`/home/hanyu/Documents/Codex/2026-11-10/cap-x-20261106-tar-gz/work/capx-p0`。
扫描 /home/hanyu/dqb 和 /home/hanyu/Documents/Codex 的项目代码目录，发现正式项目与旧工作区这两份。HEAD 不同，但可比较业务代码没有分叉；旧工作区应作为历史参考，停止作为开发入口。未删除或改名任何目录。

正式项目中仅发现根目录 .git，未发现嵌套业务 Git 仓库。扫描排除依赖虚拟环境、模型、node_modules、输出及运行缓存；该结论不代表全磁盘扫描。正式项目 .venv、models、capx/third_party 为实际目录；旧工作区通过符号链接复用资源。主项目迁移提交删除了 Git 中记录的 third_party 绝对符号链接，并忽略实际 vendor 目录及 runtime；并非删除本机 vendor 实体。

对 capx（排除 third_party）、scripts、tests、env_configs、pyproject.toml 搜索旧工作区绝对路径，未发现硬编码引用。历史结果内 trace_path 仍记载旧路径，这是不可变历史元数据；审计重放通过已迁移目录定位原 trace，未篡改历史路径。用户此前提出的 capx/training、tests/training 等理想目录重构不属于本次迁移验收，也未执行。

## 3. P0 关键实现逐项核验

以下依据实现阅读、与 P0 Git 对象比对和本轮测试；并非仅核对文件存在。

| 范围 | 实现核验与本轮结论 |
|---|---|
| Reward 独立物理评价 | verl_agent_reward/capx_franka_reward.py 根据可信父进程 physical_success、执行状态和有效运动调用判断 won；空程序、pass、语法错误、虚假 FINISH 均无正奖励。system_error/timeout 与任务失败分开，系统故障不作为有效训练样本并清理环境。Mock 和真实 Worker 负例通过。 |
| Seed | capx/envs/reproducibility.py 重建环境 RNG，绑定 placement sampler RNG，独立抓取 RNG；相关 Robosuite 初始化路径使用重置后的 RNG，覆盖布局、机器人初始状态、抓取 seed 序列。重复 Seed 和历史重放通过。 |
| Snapshot / 初始 hash | 保存 physics、ctrl、qacc_warmstart、model hash、RNG、抓取 RNG、关节和夹爪状态，精确浮点序列哈希；恢复时验证版本、物理 hash、模型和初始重置结果。v1 仅支持 episode 初始状态，不宣称支持中途控制器快照。 |
| Bubblewrap | capx/execution/sandbox.py 和 actor.py 使用独立进程、namespace 隔离、清空环境、移除 capabilities、有限只读挂载；JSON RPC 不暴露父进程对象。AST 白名单与 OS 隔离双层限制。实际 bwrap 测试通过。 |
| 隐藏状态限制 | audited_stack.py 输出观测白名单，Student 不获得 object-state、cube_poses、隐藏 segmentation、env/obs/全局对象；禁止绑定 privileged API。父进程持有环境与评价器。12 组隐藏状态/宿主访问拒绝用例通过。 |
| 超时和重建 | 子进程超时后 kill/wait，下一次新 PID、旧变量不可见；父进程 API deadline 测试通过；环境故障标记后 reset 重建。当前 deadline 使用主线程信号机制，这是 Web 线程池不兼容的直接原因。 |
| API 物理 Trace | audited_stack.py 和 physical_evaluation.py 在调用前后记录可信物理状态：物体、EE、TCP、夹爪、接触、相对位姿；逐 episode 唯一目录保存 trace/requests。测试验证记录与计数。 |
| 请求和视频 | RequestAudit 记录请求载荷、状态、耗时及实际 HTTP/LLM 次数；不记录 Authorization。info 返回代码块/API/请求数量和仿真帧时间。根目录 depth_image.jpg 不再作为视觉接口固定写出位置。记录单测通过；两项完整视觉/视频集成测试本轮跳过。 |
| 四元数和坐标 | geometry.py 明确 XYZW→WXYZ；视觉相机候选经过相机到基座变换；控制接口和 IK 的 WXYZ 顺序一致。TCP 使用 right_hand +0.107m，EE/夹爪参考点与 TCP 分开；候选过滤有限值、旋转正交性和 det=1，按有效分数选取。 |
| TCP/进近边界 | 控制 TCP 偏移、候选抓取偏移和进近方向均在现有实现中保留；候选 +0.12m 与基座 +Z 进近属于当前约定。单测证明数学一致性，不足以证明所有视角和物体的实际抓取鲁棒性；本轮未调参。 |
| 抓取/滑落/稳定放置 | 独立评价持续抓取接触与抬升、闭爪时接触丢失或跌落、释放后稳定窗口；放置要求底层任务成功、非持物、夹爪打开、线/角速度低于阈值，连续保持约 0.5 秒。Oracle 成功且未滑落；无需 FINISH 自报。 |
| Seed 回显 GraspNet | 服务源码在受保护推理区设置/恢复 RNG，响应回显 seed；客户端校验回显，不兼容服务报错。源码完整迁移，但当前 8115 服务运行接口仍无 seed，部署运行状态未同步，见第 5 节。 |

14 个关键文件的逐项字节比对结果保存在 inventory.json。P0 以后变更是 DeepSeek/OpenAI provider 配置、client、local runner/common.sh、8 项 provider 测试和迁移忽略规则；未发现上述 P0 核心实现被回退。

## 4. API 配置、CLI 和 Web

- .env.local 与 .env.api.local 均存在、0600，git check-ignore 确认被忽略，未被跟踪。本轮前后内容哈希未变，报告不披露密钥或私有配置哈希。
- .env.local 保留旧 CAPX_CLOUD 配置；.env.api.local 中 DeepSeek/OpenAI Key 与模型配置均有非空值。非空只证明填写，不能证明有效或不是占位内容。
- scripts/local/common.sh 先加载 .env.local，再加载 .env.api.local，后者覆盖同名变量。这是两份配置并存时的实际优先级。
- CLI provider 支持 deepseek、openai（chatgpt 别名）及原 cloud、openrouter、ollama、vllm；不同官方 provider 使用各自配置；缺必要字段会报错，Oracle 模式不需要云密钥。8 项 Mock 测试全部通过。
- 本地 qwen2.5-coder-7b、Ollama 模型目录、SAM3 权重目录、ContactGraspNet vendor 与服务代码保留。配置/文件存在不代表服务当前运行或模型输出质量已验证。
- Web 前端具备模型下拉和自定义模型、server_url 输入；但后端 server.py 的 Key fallback 使用请求值/default_api_key/通用环境变量，未证明更换 provider 时会自动选择匹配的密钥。**Web 多 provider 切换不能按 CLI 验收结果直接判定通过。**
- Web async_trial_runner.py:141 使用 ThreadPoolExecutor 运行环境，而 sandbox.py:64 明确要求主线程 deadline。**当前 Web 路径不能作为通过 P0 的执行入口。** 本轮为静态确认，未发起 Web 试验。
- 请求审计代码已迁移，Oracle 实测 requests 日志有本地 IK 请求、LLM 请求 0；云 API 的真实请求日志与响应本轮未生成。Web 跨线程 RequestAudit context 是否完整传播未专项验证。
- 本轮未调用 DeepSeek/OpenAI/Ollama/vLLM 推理接口；只读查询的 HTTP 接口是现有本地服务 /openapi.json。

## 5. 运行环境与现有服务

| 项目 | 本轮结果 |
|---|---|
| Python | 正式目录 .venv，3.12.13 |
| PyTorch / CUDA | 2.8.0 / 12.8，CUDA available=True |
| NumPy / MuJoCo / Robosuite | 1.26.4 / 3.5.0 / 1.5.1 |
| Transformers | 4.57.6 |
| vLLM / VeRL | 主 .venv 无对应发行包；独立 .venv-vllm 中 vLLM 0.11.0，vendor/verl 保留 |
| SAM3 / PyRoKi | 发行元数据 0.1.0 / 0.0.0，已有服务可读 schema |
| ContactGraspNet | vendor 目录保留，8115 服务在线但 schema 缺 seed |
| GPU | NVIDIA GeForce RTX 5090 D v2；驱动 580.173.02；总显存 24455 MiB |
| 显存采样 | 已用 16350 MiB，可用 7616 MiB；瞬时值，会随进程变化 |

主环境上述核心版本与迁移保留的 runtime-manifest.json 相符；pyproject.toml 和 uv.lock 本轮未改变。没有升级 Transformers、vLLM、VeRL 或其他包。

现有进程均来自正式项目 cwd：
- PID 184215：Web/local runner，8200（另有 8080）。
- PID 184236：8114，SAM3 接口。
- PID 184237：8115，ContactGraspNet 接口。
- PID 184238：8116，PyRoKi IK 接口。

**关键环境差异：** 8115 的 PlanRequest、PlanResponse 实际 OpenAPI 属性没有 seed；P0 视觉验收所需 8215 不监听。磁盘源码已经更新并不等于已运行进程加载了更新。本轮没有重启、停止或启动这些服务。8000、11434、8110 本轮未发现监听，因此不宣称本地 LLM 服务可用。全部运行时证据见 runtime.json / processes.json。

## 6. 测试结果及历史重放

使用正式目录 .venv，PYTHONPATH 指向正式目录，MUJOCO_GL=egl，禁用 pyc 和 pytest 缓存，NUMBA_CACHE_DIR、pytest 临时目录及 CAPX_TEST_OUTPUT 均指向新的独立审计目录。

```bash
.venv/bin/python -B -m pytest tests/p0 --ignore=tests/p0/test_api_providers.py -q -p no:cacheprovider
.venv/bin/python -B -m pytest tests/p0/test_api_providers.py -q -p no:cacheprovider
```

实际命令额外指定独立 --basetemp 和 --junitxml。未设置 CAPX_P0_VISION，因为它要求现存 P0 服务；不是忽略已发现的失败。

| 测试 | 本轮结果 |
|---|---|
| 原 P0 60 项 | 58 passed / 2 skipped / 0 failed，19.74 秒 |
| Provider Mock | 8 passed / 0 skipped / 0 failed |
| test_seeded_visual_service_and_video | 跳过：P0 8215 服务未运行 |
| test_paired_oracle_physical_accounting | 跳过：同上 |
| 隐藏状态访问拒绝 | 12 个参数化用例全部通过 |
| Sandbox OS 边界 | 关闭 AST guard 后仍无法看到宿主项目路径、密钥环境或访问宿主 IK 网络；通过 |
| 超时、PID 消亡、环境重建 | 对应 Sandbox / audited env 用例通过 |
| Reward Mock | 非成功不得正奖励、独立成功、系统错误区分、超时清理等通过 |
| 真实 Reward Worker 单样本 | score=1、won=True、physical_success=True、status=success；6 motion / 10 API / 1 code block / 6 HTTP / 0 LLM |
| Reward 真实负例 | pass、空代码、语法错误、FINISH、打印 FINISH、env 访问、无限循环均 score=0、won=False；无限循环标记 timeout |
| 本轮相同 Seed/Snapshot 两次 Oracle | 初始 hash、最终 hash 一致，两次 reward=1 |
| 迁移保留的历史 Snapshot 独立重放 | 直接读取历史 trace 的 snapshot，初始和最终 hash 均与历史记录完全相同，reward=1，grasp_success=True，slipped=False |

历史 Snapshot 来源：
`outputs/codex-migration-20261111-231653/p0-20261111/complete-acceptance/replay/e4324cf78154474389d40ac6203af153/trace.jsonl`。

Seed=0：
- 初始 hash：`565d18a70b061e9f127ace98262f8f294b45b1237d4a776c9554b3f8ff9eda6d`
- 最终 hash：`11be38d7ce0df1f50c66df31bf68df03df472e47b253f49df7402569481c7bf6`

测试有 2 条已有依赖 DeprecationWarning（Cost.create_factory），不是失败。本轮没有为清除警告改依赖。

## 7. 产物与配置保护

对审计开始时记录的文件逐个 SHA-256 比较：
- 531 个已跟踪文件：0 改变、0 丢失。
- 2 个私有 API 配置文件：0 改变、0 丢失。
- outputs/runs 内 369 个已有实验产物：0 改变、0 丢失。
- pyproject.toml、uv.lock：0 改变、0 丢失。

该核验覆盖上述明确集合，不声称已对全部模型权重和虚拟环境文件做前后全量哈希。新增测试产物与旧实验目录分离；本报告是本轮唯一计划新增的正式项目文件。

## 8. 尚未完成或不能宣称通过的部分

1. **P0 源码未发现漏迁移；P0 GraspNet 运行服务尚未同步。** 后续需授权服务维护窗口，采用现有环境启动 Seed 回显版本并补跑两项视觉测试。
2. Web 执行线程模型与 P0 主线程信号 deadline 冲突。后续最小方案是让 Web 使用独立、主线程运行的 Worker，保持现有 API；这是架构接口修复，需单独任务实施。
3. Web provider/credential 配对与跨线程请求审计尚未验收；不能把 CLI 的 8 项 Mock 结果外推为 Web 真调用成功。
4. .env Key 的有效性和具体 DeepSeek/OpenAI 模型服务可用性未验证，符合本轮禁止真实 API 调用要求。
5. web-ui、README 等未跟踪，vendor/模型/环境不在 Git 可重建范围；当前本机完整性通过，不等于仅 checkout HEAD 即可异机复现。
6. Snapshot v1 的初始状态限制、TCP/抓取偏移的场景泛化、底层不可中断 native 阻塞的外部 watchdog 能力没有扩大验证。
7. 旧工作区和历史绝对 trace_path 仍存在。开发统一使用正式目录，旧记录作为只读历史保留。

**最终决定：允许在主目录继续开发；不批准当前运行状态直接进入完整 A/B/C/D 配对实验。** 先补齐上述运行服务与入口验收，再将本轮 58+2skip 更新为当前环境真实 60/60。

## 9. 审计证据

所有新证据位于：
`/home/hanyu/Documents/Codex/2026-11-10/cap-x-20261106-tar-gz/outputs/migration-audit-1794410547306`

- inventory.json：Git、目录、关键源码逐项比对、脱敏配置存在性。
- runtime.json、processes.json：版本、GPU、端口、服务 schema 和进程入口。
- p0-tests.log、p0-tests.xml：原 60 项完整结果。
- api-mock-tests.log、api-mock-tests.xml：8 项 provider Mock。
- acceptance/：本轮 Worker、负例、双次 Snapshot 重放及 trace/request 日志。
- historical-replay.json、historical-replay.log、historical-replay/：历史 Snapshot 验证。
- preservation-and-tests.json：原产物、源码、配置、锁文件前后核验。
