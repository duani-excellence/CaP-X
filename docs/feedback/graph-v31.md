# V31 统一推理与执行图工作台

在正式工程根目录运行。`infer.sh` 加载现有 `.env.local` / `.env.api.local`，密钥不放入命令行、配置 JSON 或浏览器。它调用已有 `contract_models.run_episode` 或 `graph_v2.run_pair`，没有新建机器人 Agent / Runner。

## 最短命令

```bash
cd /home/hanyu/dqb/cap-x-20261106
scripts/local/infer.sh run --methods G0 G1 --task pick_place --seeds 0
```

Web：

```bash
scripts/local/infer.sh serve --host 127.0.0.1 --port 8200
```

打开 `http://127.0.0.1:8200/#experiments`。本轮先用 8201 验收，再核实 8200 无活动任务并保存会话清单后优雅更新；临时 8201 已关闭。页面的“交互推理”返回原来的聊天、任务设置、机器人 Viser 和多任务入口。新工作台的相机画面来自实际公开 RGB-D capture；执行结束后显示归档视频。

当前服务 runtime / LOGIV 官方源码位置在 `env_configs/inference/local-v31.json`。它是本机开发配置，不是可移植保证。视觉服务迁移时修改此文件或显式传 `--ik-runtime` / `--external-logiv`；不自动猜端口、不自动启动重复 GPU 模型。

## 可执行组合

| 任务 | 方法 | Skill | 观测 / 模型 | 实际执行器 |
|---|---|---|---|---|
| PickPlace | G0 / G1 / LOGIV | v30 / v29 | Vision / DeepSeek V4 Pro | Contract Runner |
| Stack / Lift / Restack | B0 / G0 / G1 | v23 | Vision / DeepSeek V4 Pro | 原 grounded Graph Runner |

LOGIV 是官方核心 + 共同 Skill 适配版，不能称为原论文完整复现。B0 是原 CaP-X 程序级反馈，不能称为 CaP-Agent0。ASPIRE 完整方法、CaP-Agent0、LIBERO-Recover 在线策略、cuRobo 规划尚不可用，界面明确禁用并列出原因。

现有图实验账本只支持 `deepseek/deepseek-v4-pro`。统一图入口中的 OpenAI、本地模型、Privileged 暂不开放；原交互前端 / Teacher 的已有能力保持原样。本轮没有把未适配的 Provider 或真值模式伪装成支持。模型名称错误会拒绝，不自动切换。

## 校验、单次、配对与批量

```bash
# 无模型 / 无仿真：打印完整解析配置
scripts/local/infer.sh run --methods G0 G1 --seeds 0 --dry-run
scripts/local/infer.sh capabilities
scripts/local/infer.sh run --help

# 单方法
scripts/local/infer.sh run --methods G1 --task pick_place --seeds 0

# 外部方法 + 内部对照；共享真实初始程序与初始状态
scripts/local/infer.sh run --methods LOGIV G0 G1 --seeds 0 --max-requests 6 --cost-limit-usd 1

# 固定开发场景，2 次独立初始生成；不会在本轮自动运行此批次
scripts/local/infer.sh run --methods G0 G1 --seeds 0 7 21 --repeats 2 --max-requests 24 --cost-limit-usd 5

# B0 使用自己的合法 API / 提示词；G0/G1 共享初始程序
scripts/local/infer.sh run --task stack --methods B0 G0 G1 --seeds 0

# 受控证据过期；同一消融只应用 G1，G0 仍是完整强基线
scripts/local/infer.sh run --methods G0 G1 --scenario evidence_expiry --ablation no_local_repair --seeds 0
```

`normal` 保留自然失败；PickPlace 另支持已有 `evidence_expiry` 和 `effect_displacement`。后者由独立评估器施加真实物理扰动，标签不传给 Agent。Grounded 任务支持已有 `support_refresh / missing_baseline / target_shift`；B0、LOGIV 缺少等价注入接口时明确拒绝。公开恢复场景只已有环境初始化 Gate，尚无统一在线策略接口。

配置文件采用 `RunSpec` JSON：

```json
{
  "task": "pick_place", "methods": ["G0", "G1"], "seeds": [0],
  "repeats": 1, "skill": "v30", "scenario": "normal",
  "max_requests": 6, "cost_limit_usd": 1.0, "wall_seconds": 1800,
  "max_tokens": 32768, "reasoning_effort": "low"
}
```

```bash
scripts/local/infer.sh run --spec my-experiment.json --output outputs/inference/my-new-run
```

显式 CLI 参数覆盖 `--spec`。不认识的字段会拒绝；输出目录已存在时拒绝覆盖。每个 Seed × repeat 是一个配对单元；每个单元独立生成初始代码，单元内共享代码。方法顺序由 `--methods` 顺序决定，不宣称已做顺序平衡。

## 历史程序重放（零新模型请求）

`--reference` 指向已有 Contract 配对目录（有 `initial.json`、`config.json`、`G1/snapshot.json`），不会把重放记成新 DeepSeek 实验。

```bash
scripts/local/infer.sh replay --methods G1 --seeds 0 \
  --reference outputs/model_comparison/graph_v30_20261124_185019/online-v30c-normal0 \
  --snapshot-group G1 --output outputs/inference/replay-v30c-new
```

所有需要新模型恢复的分支都会明确停止；不偷偷发新请求。Snapshot 沿用原 Runner 的精确重放契约：Contract 环境按原 Seed 和 spec 生成初始状态并逐项核对 Snapshot，不支持任意中途状态注入。Grounded 的 `--initial-manifest` 源码/Prompt 校验机制仍通过其原入口使用，统一 replay 尚未替代它。

## 参数速查

- 身份：`--task`、`--environment`、`--methods ...`、`--seeds ...`、`--repeats`、`--snapshot`。
- 方法：`--skill auto|v30|v29|v23`、`--planner pyroki`、`--scenario`、`--ablation`。
- 消融：`none / no_selective / no_future / no_memory / no_graph_repair / no_cost / no_causal / no_local_repair / no_resume_validation`。沿用原实现，一次选择一个；不能混用不等价 Runner 的开关。
- 模型：`--provider`、`--model`、`--max-tokens`、`--reasoning-effort`、`--request-timeout`。当前可执行 Provider 见支持表，不接受密钥参数。
- 预算：`--max-requests`、`--cost-limit-usd`、`--wall-seconds`、`--max-blocks`、`--block-seconds`、`--api-budget`、`--episode-seconds`、`--skill-budget`。
- 配置与产物：`--spec`（统一配置）、`--config`（原 Runner 配置）、`--ik-runtime`、`--external-logiv`、`--output`、`--log-level`。
- 重放：`--reference`、`--snapshot-group`。

沿用原请求预算：无自动重试；不确定的已发请求不能自动重发。总 wall 预算在请求与配对单元边界检查；当前物理 episode 仍由原 episode/block deadline 约束，不能把它当成整批严格杀进程截止时间。预算与实际请求写入 `request-ledger.json`。

## 停止与产物

```bash
scripts/local/infer.sh stop outputs/inference/my-new-run
scripts/local/infer.sh inspect outputs/inference/my-new-run/seed-0-repeat-0/G1
```

正常停止等待完整配对边界。Web 另有立即终止：SIGINT 后有界升级 SIGKILL，标记 `operator_terminated_incomplete_pair`，不计任务成功。不会停止视觉服务。统一进程有互斥锁；服务器重启后不猜测旧 PID 归属、不自动杀不认识的进程。

`spec.json` / `config.json` 是本次解析后的参数，`status.json` 为进程进度，`summary.json` 为结果，`source-archive.json` 和 `source-postcheck.json` 记录源码。每个配对目录保留原 Runner 的 Snapshot、程序、请求、Trace、视频、工具记录和物理评分。

## 图与结果的解读

- 图节点/边来自原 `DependencyGraph` 增量事件；时间轴严格重建实际记录，未记录的状态不补造。
- 原因果计划内部显式 `edges` 映射至同一 plan 的真实 action 节点，详情保留 `source_record` 和计划 revision。不是根据执行顺序猜因果边。
- pending 不等于正在运动；done 是历史调用事实，不等于物理成功。没有 graph 的 B0 只显示真实代码/Trace。LOGIV 中的共享契约图与官方方法事件分别展示，不伪称 CaP-X 图为 LOGIV 原始图。
- 最近 180 个匹配节点上屏；可筛选层、版本、关键词和历史时间点。详情保留来源、参数、验证与失效原因；大型点云保存在原观察产物中，未塞进 Prompt。
- 结果表对比物理 reward、子目标、API/capture、实际 HTTP 工具次数、恢复、失效/patch、LLM、Token/费用及时间。只有 Snapshot 和初始程序 hash 都一致才标明严格共享配对。
- 各方法成本含共享初始请求的归因，不能直接相加作为实际付款；批次唯一请求和估算费用以账本为准。历史缺失的统计显示未知，不推算成零。
- 完整 Trace 含独立评估器物理状态，属于用户离线诊断，不是在线 Agent 工具。Web 文件读取限定在 outputs；不提供任意文件浏览或环境变量导出。

这是调试平台，不是图收益结论。算法、安全阈值与物理评分未改变。

Grounded V31 配置基于原 V23.1 策略重新采集当前服务与源码身份（`development-v31-grounded.json`）；旧配置会因 V30 客户端源码变化正确拒绝。新记录仍保留长驻服务源码可能与已加载字节不一致的限制，不宣称满足正式冻结的加载字节证明。
