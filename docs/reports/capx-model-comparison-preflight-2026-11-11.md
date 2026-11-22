# CaP-X 模型对比实验前最终预检（2026-11-11）

## 1. 最终判断

| 项目 | 结论 | 边界 |
|---|---|---|
| P0 测试 | GO | 原 P0 60/60，全目录含 Provider 共 68/68；零跳过、零失败 |
| DeepSeek API | GO | deepseek-v4-pro 真实文本请求一次，HTTP 200 |
| GPT API | GO | gpt-6-astra 真实文本请求一次，HTTP 200 |
| CLI 模型切换 | GO（切换接口） | 四组 CLI dry-run + 两 Provider 真实客户端调用；不等于模型机器人 Trial 已验证 |
| 五场景正式对比 | NO-GO | Snapshot 导入、跨模式完整 Prompt/API 文档、预算、初始静止条件尚未闭环 |

本轮未执行正式模型对比、30 Trial、批量 Teacher、训练、真实机器人操作。真实 API 恰好两次，每家一次，无重试。运行了用户允许的 P0 Oracle 集成测试。

## 2. 工作区和保护

唯一项目：/home/hanyu/dqb/cap-x-20261106。
分支 p0-repro-foundation；HEAD 363559a3d3a36ecea9872741dfc08593a457ce65。
已有未跟踪文件：.env.example、.env.local.example、.gitmodules、.rayignore、README.AICODER.md、README.LOG.md、README.md、SUBMODULES.txt、web-ui/、docs/reports/。
未切换分支或提交。531 个 tracked 文件、2 个 API 配置、369 个旧产物前后哈希无改变。配置仍 0600。未改源码、原测试、依赖、Reward 或 Sandbox，未下载模型。

所有本轮脚本、测试产物与草案均在正式目录的新输出目录：
/home/hanyu/dqb/cap-x-20261106/outputs/model_comparison/preflight-20261111-1794411344735
没有使用旧工作区开发。本报告写入 docs/reports/capx-model-comparison-preflight-2026-11-11.md。

## 3. 服务同步

先阅读 P0_RUNTIME、迁移报告、服务 main 和两个视觉测试。没有 scripts/local/start_graspnet.sh；依据运行说明的实际模块入口启动。

8115 旧 PID 184237，父进程 184215，cwd 是正式项目，Python multiprocessing spawn 启动。两次 TCP 检查均无连接，ep_poll 等待；Web /api/sessions 仅 complete，会话 /api/active-session 为空。旧 GraspNet 占 7536 MiB，GPU 总 24455 MiB，空闲 7608 MiB。

确认安全维护条件后，仅 SIGTERM 184237，确认端口退出后启动新 8215，未同时加载两份模型。
新 PID 264142：
.venv/bin/python -B -m capx.serving.launch_contact_graspnet_server --port 8215
环境使用 PYTHONPATH=$PWD、PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1、PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
复用已有权重，启动环境移除了 API Key/token/secret 项。8215 保持运行。

SAM3 8114 PID 184236、PyRoKi 8116 PID 184238 保持运行，测试后 schema GET 均 200；其他 GPU 进程 awesun PID 233426 未停止。Web 未停止。
旧 8115 不再监听，后续客户端需要 GRASPNET_SERVICE_URL=http://127.0.0.1:8215。
原 vision YAML 的 api_servers 仍声明 8115，后续必须显式复用现有服务，避免启动管理器重复加载。本轮未改原 YAML。

## 4. Seed 与视觉测试

8215 PlanRequest、PlanResponse schema 均有 seed。两次环境 Seed=21 产生相同抓取请求 Seed，客户端实际校验回显通过；详见 seed-request-summary.json。
两次均得到 35 候选，候选与分数联合 hash：
784196f6b9e42552ed9c3ac95e1da6eacd582037a0433ac79df117d2605e3d27
两次初始 hash：
b1f00899857567f412695c5bdb596bb0ff9a8f5b795e5ac49f0d8dce885d363e

test_seeded_visual_service_and_video 通过：候选/分数、stdout 一致，帧与仿真时间检查通过。
test_paired_oracle_physical_accounting 通过：Privileged reward=1，Structured Vision reward=0，均 execution_status=ok。视觉组没有稳定放置成功，测试通过意味着正确记录成功/失败，不能外推视觉任务成功率达标。

旧无 seed 响应通过当前客户端 Mock 验证 fail closed，抛出 ConnectionError；未重新启动旧模型。真实回显与重复抓取由 8215 测试验证。

## 5. P0 结果

正式 .venv，CAPX_P0_VISION=1、MUJOCO_GL=egl，禁用 pyc/pytest 缓存，Numba/临时文件与 CAPX_TEST_OUTPUT 指向新目录。
运行 python -B -m pytest tests/p0 -q -p no:cacheprovider，并指定独立 --basetemp、--junitxml。
**68 passed，0 skipped，0 failed，31.59 秒**：
原 P0 60 passed；Provider Mock 8 passed。8 条已有弃用警告，未改依赖。
保留 p0-full.xml/log、acceptance/vision/video.mp4、paired-oracle/video_A.mp4、video_B.mp4 及全部时间戳、trace/requests 日志。

## 6. API Smoke

沿用 common.sh 先 .env.local 后 .env.api.local。使用 provider_config 与 query_model，没有重新实现 HTTP 客户端；一次性验证脚本对 audited_post 加限次、错误立即停止保护，attempt.lock 防误重跑。原客户端不修改，debug 关闭。

| 字段 | DeepSeek | OpenAI |
|---|---|---|
| requested / response model | deepseek-v4-pro / deepseek-v4-pro | gpt-6-astra / gpt-6-astra |
| endpoint | https://api.deepseek.com/v1/chat/completions | https://api.openai.com/v1/chat/completions |
| HTTP | 200 | 200 |
| header request ID | 所检查 x-request-id/x-ds-request-id 未返回 | req_72fda9c907c44920ac5f9a18b701b233 |
| response ID | 15d5a868-564c-4321-873c-9783b96ea1b9 | chatcmpl-Ej4fyhikZQHrxisycMzwmszBBQ375 |
| finish_reason | stop | stop |
| input tokens | 93 | 14 |
| output tokens（含 reasoning） | 23 | 8 |
| reasoning tokens | 15 | 0 |
| latency | 1.358s | 3.006s |
| retries | 0 | 0 |
| 实际及解析 content | CAPX_DEEPSEEK_OK | CAPX_OPENAI_OK |

分别发送 Return exactly: CAPX_DEEPSEEK_OK / CAPX_OPENAI_OK，预算 64。
DeepSeek reasoning_content 未混入解析的 message.content，不作为 Python 执行。
OpenAI 使用 Chat Completions messages 和 max_completion_tokens=64，不发 temperature/max_tokens。当前客户端移除 reasoning_effort，故此次仅验证默认 effort，不能声称显式设置生效。官方 Astra 支持的 effort 不含 none，本轮未改参数重试。
API 请求成功不等于机器人任务成功。没有图像输入。

## 7. 成本

约 15:37 UTC，DeepSeek off-peak，无缓存。
- DeepSeek：93 input×0.66 USD/M +23 output×1.98 USD/M = 0.00010692 USD。
- GPT：14 input×10 USD/M +8 output×50 USD/M = 0.00054000 USD。
- 合计 0.00064692 USD。Reasoning 已包含 output，没有重复计算。估算按公开标准价格，实际以账单为准。

[DeepSeek 价格](https://api-docs.deepseek.com/quick_start/pricing/)
[GPT-6 Astra 模型/价格](https://developers.openai.com/api/docs/models/gpt-6-astra)
[OpenAI 推理参数](https://developers.openai.com/api/docs/guides/reasoning)

## 8. Provider 统一性

同一 scripts/local/run.py 通过 --backend 切换；四组 dry-run 拦截 launch.main，不启动 Trial/API。模型配置正确。
同一观测模式内换模型不改 config_path、Prompt、API 文档；两模型使用相同内容/代码解析器、Sandbox、Reward 和 Snapshot 类。

**跨 Privileged/Structured Vision：任务文本相同，但 API 文档和完整 Prompt 不同**，hash 证据位于 prepared-config-v2/prompt-comparison.json。不能宣称四组严格一致。
请求日志通过 episode UUID 目录关联，payload 有 model 和 URL；生产日志没有显式 provider/trial_id，需正式 manifest 映射。本轮 smoke.json 明确记录两者。
Key 留在后端配置/内存，未传前端；请求日志不保存鉴权头，新增文本扫描无现有凭据值。未测试 Web，已知线程池与主线程 deadline 不兼容。

原 query_model 对部分 404/5xx 无限重试，Smoke 限次保护不代表正式 CLI 已具备请求预算。
run_single_trial 直接 seed=trial reset，尚无指定 Snapshot 导入。不能把 manifest 要求写成已经实现。
CLI 切换 GO 只覆盖路由和真实 Provider 客户端，不覆盖完整配对 Trial。

## 9. 五场景配置

prepared-config-v2/ 含 E1 DeepSeek+Privileged、E2 GPT+Privileged、E3 DeepSeek+Structured Vision、E4 GPT+Structured Vision。
Seed 0、21 复用历史 Snapshot，1、2、3 从现有初始 reset 生成。五份在两个模式均 restore 且 state_hash 相同，无机器人动作或模型 Trial。
预算草案统一 64 API/block、256 API/episode、90秒/block、10 code blocks、4096 completion tokens；每次 restore/hash 核对；原 Reward。
关闭原始图像、visual feedback、image/video differencing 和 ensemble。Structured Vision 是视觉服务返回的结构化位姿/抓取数据，不是 DeepSeek 原生看图。

所有组 execute=false、status=PREPARED_NOT_EXECUTABLE。CLI 尚不消费这些 JSON；10 code block 严格预算与原 <=10 循环边界需明确。
场景均有限值、物体分离、非已成功状态、Snapshot 可恢复；但初始线速度约0.04467m/s，未通过额外采用的0.02m/s静止阈值。scene-validation.json 保留 valid_initial_physics=false。
首次准备脚本因此失败，完整日志保留；第二次只继续记录诊断并生成不可执行草案，没有放宽标记或改原测试。不能将五场景稳定初态称为已验收；新增三场景视觉有效性/可完成性尚未验证。
两种模式完整 Prompt/API 文档分别保存，未隐藏差异。

## 10. GPU 与后续条件

无 OOM。测试后 SAM3 6886MiB、新 GraspNet 6908MiB、其他 awesun 652MiB；SAM3/awesun 与维护前相同。旧 GraspNet 的7536MiB释放。完整前后 nvidia-smi 与进程 CSV 保存。

正式实验最小前置工作：
1. CLI/worker 接入 Snapshot manifest，固定 reset→restore→核验→模型请求。
2. 统一公共 API 文档/完整 Prompt，明确跨观测模式的允许差异。
3. 明确代码块、请求/重试、token 和总超时预算。
4. 明确初始静止验收条件；若改 settle，单独修复并重建/复验 Snapshot，不改历史 hash。
5. 明确复用 8114/8116/8215 的启动配置及 trial/provider/Snapshot 映射。
6. 验证新增三场景视觉有效性，并保留视觉 Oracle reward=0 的基线。

涉及入口/配置契约，已报告而未擅自重构。Web 不在本轮 GO 范围。

## 11. 证据

service.json、8215-schema.json、seed-request-summary.json、graspnet-8215.log；
p0-full.xml/log、acceptance/；
deepseek-smoke.json、openai-smoke.json、各 provider requests.jsonl；
cost-estimate.json、cli-routing-dry-run.json；
prepared-config-v2/、prepare-scenes.log、prepare-scenes-v2.log；
gpu-before/after.txt、gpu-processes-before/after.csv；
preservation-results.json、secret-scan.json、git-before/after.txt。

preservation-private.json 只存校验摘要、0600，不包含 Key，不作为公开附件。未保存鉴权请求头或私有环境变量。
