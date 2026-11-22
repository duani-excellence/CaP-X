# CaP-X：推理、DeepSeek Teacher 采集与 Qwen SFT 导出

正式目录 `/home/hanyu/dqb/cap-x-20261106`。本管线只准备数据，不启动训练。
复用 `capx.experiments.pilot` 的 Snapshot、Prompt、单程序执行、物理评价和记录；
不改变 P0 奖励或 Sandbox。Teacher 固定 `deepseek-v4-pro`，不自动换模型。

## 按合格数据目标自动采集（大规模入口）

新增模式在同一个脚本中，用 `--target-success` 开启。目标是**每个 target mode 经物理成功、
SFT 校验、去重和模板配额筛选后的样本数**，包含 train 与 validation，不是请求数。
以下为实际可执行命令；会付费，本轮没有执行这些大规模采集。

### 100 条 Vision 合格示范

```bash
bash /home/hanyu/dqb/cap-x-20261106/scripts/local/generate_teacher_data.sh \
  --target-success 100 --target-modes vision --modes vision --no-distill-vision \
  --start-seed 0 --max-scenes 1000 --max-requests 1000 --max-cost-usd 100 \
  --output /home/hanyu/dqb/cap-x-20261106/data/teacher_runs/vision100
```

### 500 条 Vision 合格示范

```bash
bash /home/hanyu/dqb/cap-x-20261106/scripts/local/generate_teacher_data.sh \
  --target-success 500 --target-modes vision --modes vision --no-distill-vision \
  --start-seed 0 --max-scenes 5000 --max-requests 5000 --max-cost-usd 500 \
  --output /home/hanyu/dqb/cap-x-20261106/data/teacher_runs/vision500
```

### 1,000 条 Vision 合格示范

```bash
bash /home/hanyu/dqb/cap-x-20261106/scripts/local/generate_teacher_data.sh \
  --target-success 1000 --target-modes vision --modes vision --no-distill-vision \
  --start-seed 0 --max-scenes 10000 --max-requests 10000 --max-cost-usd 1000 \
  --output /home/hanyu/dqb/cap-x-20261106/data/teacher_runs/vision1000
```

命令持续采集，直到达到合格样本目标，或触及累计请求、费用、场景、时间、磁盘上限。
每个 episode 都保存检查点，不存在 batch-size 或每 20 条退出的逻辑。
中断或时间/磁盘限制解除后，重复同一命令并加 `--resume`；保持原输出目录。
可调整 `--max-hours`、`--min-free-gb`；目标、模式、累计预算、Seed 范围和生成配置必须保持一致。
默认每次最多调度 4 小时，剩余磁盘低于 5 GiB 时不再调度；正在执行的 episode 受 360 秒 worker 超时约束。
长时间采集可设 `--max-hours 24`。资源上限内不保证达到目标。
旧版因 batch_complete 停止的检查点可兼容恢复，首次恢复会备份旧检查点；已有数据与预算累计保留。

### Privileged 和蒸馏

将上述模式参数替换为 `--modes privileged vision --target-modes privileged vision`，
并移除 `--no-distill-vision`，即可要求**两种模式各达到指定目标**，并保留跨模式重放。
只采 Privileged Teacher、供 Vision Student 使用可设
`--modes privileged --target-modes vision`（默认启用蒸馏）。
Privileged 原始数据不会直接计入 Vision；必须成功通过同 Snapshot 下 Vision API 重放和导出校验。
蒸馏重放不额外请求模型。同一 mode+snapshot 最多接纳一条，原生 Vision 与重放不能重复计数。
Vision 是部署可用的结构化视觉工具接口，不是模型原生图像输入。

### 预算、恢复与失败

- Seed 从 `--start-seed` 开始顺序生成，最多 `--max-scenes` 个，每个新场景先准备并验证 Snapshot。
- `--max-requests` 在新管线中是保守的 episode 请求槽上限，跨恢复累计；Oracle 拒绝也占一个槽，实际 HTTP 请求可能更少。
- 默认 `--retries 1`：仅明确的 task_failure / invalid_program 允许同场景再请求一次；重试保存独立目录并占预算。成功、Oracle 拒绝和多样性筛选拒绝不重试。
- API/系统错误、超时或中断后不能确定请求结果时停止自动续采，需要人工核查；不为凑数据自动重发可能已扣费的请求。
- 每个 episode 默认预留 `--request-cost-cap 1` 美元；剩余累计核算预算不足此额则停止。完成后按已有收据核算，未知结果保留预留额。价格来自现有保守核算器，不是供应商账单保证。
- 检查点通过原子写入和 fsync 保存；进程锁防止同一个目录并行采集。完成但尚未写入检查点的结果可恢复，无须重新请求。
- 完整配置和调度模块摘要绑定检查点；除了此次已审计的旧版调度迁移，代码或配置改变会拒绝续采。旧小规模目录不是 campaign，不支持直接 `--resume` 转换。
- `state.json` 是当前汇总；`attempts/` 保存每次请求、代码、执行反馈、物理 Trace 和视频。
  正常停止时在 `exports/时间戳/` 生成新的两模式 SFT 视图、筛选后 records 和统计，不覆盖早先导出。
- `physical_success` 为物理成功记录数量；`sft_eligible` 为校验通过但尚未去重的数量；
  `accepted` 才是目标计数。另有原生/蒸馏接纳数、各来源物理成功数、模板数及拒绝原因。
- `--mock` 可无模型费用测试上述机制，但 Mock 永不计入目标；会运行到资源上限。

### 多样性边界（采集前必读）

当前执行器仍只有一个方块任务族；更换 Seed 增加布局变化，不能宣称得到不同任务能力。
本管线按完整输入/输出、mode+snapshot 去重，并将代码 AST 中常量、局部变量名、注释和
文档字符串归一化。默认 `--max-per-template 20` 限制每种模式每个模板最多接纳 20 条。
因此目标 100 / 500 / 1,000 至少需要 5 / 25 / 50 个代码模板；如果现有 Teacher 一直返回相似代码，
最终会因资源上限退出而达不到目标。这是明确的采集结果，不应取消限制后声称多样性已解决。
AST 配额只是启发式，不能证明行为多样性，也不防止所有语义近似代码。
后续需要扩展经过独立物理判据验证的任务族并人工审核数据；本次不修改任务、Reward 或 Sandbox。

## 一条命令（推荐）

从任意目录运行，无需手动 source：

```bash
bash /home/hanyu/dqb/cap-x-20261106/scripts/local/generate_teacher_data.sh
```

自动加载当前配置，准备 Seed 0 Snapshot，执行 Privileged 和 Vision Teacher 采集，
将成功的 Privileged 代码在 Vision 中重放，再分别导出两种 SFT 数据。
默认最多 2 次 DeepSeek 请求、不重试，核算预算 1 美元、生成上限 32768 Token；
核算金额是估算值，不是账单保证。需要现有 8114/8215/8116 服务在线。
产物写入新的 `data/teacher_runs/teacher_时间戳/`，不会覆盖旧数据。
子目录为 `scenes/`、`raw/`、`sft/vision/`、`sft/privileged/`；根目录有汇总。
失败样本仍保留但不导出；系统错误中止命令，检查 `raw/aborted.json` 和 worker 日志。

不付费验证：命令末尾加 `--mock`，仍运行真实仿真，但 Mock 标签不会进入 SFT。
查看选项加 `--help`；指定输出目录加 `--output /新的/目录`。
增加场景时须同时调整请求预算，例如 `--seeds 0 21 --max-requests 4`。
单场景不能保证划分出验证集；空导出表示没有合格示范，不代表训练数据已就绪。

以下是可选的分阶段入口。

## 环境准备

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GRASPNET_SERVICE_URL=http://127.0.0.1:8215
STAMP=$(date +%Y%m%d_%H%M%S)
export NUMBA_CACHE_DIR="$PWD/outputs/teacher/cache_$STAMP"
```

使用当前 `.venv` 和 `.env.api.local`；不安装或升级依赖。
Privileged 需要已在线的 PyRoKi 8116；Vision/跨模式重放还需要 SAM3 8114、支持 Seed 的 GraspNet 8215。
命令只检查这些服务，不加载第二份 GPU 模型。Key 不进入命令参数、数据集或前端。

## 当前推理入口

Web 已运行时直接访问 <http://127.0.0.1:8200/>，选择模型与观测模式，手动开始。
只有在 8200 空闲时启动：

```bash
.venv/bin/python -B scripts/local/run.py web --backend deepseek --max-tokens 32768
```

CLI 单场景推理可使用下面的 `prepare` + `collect --modes privileged --max-requests 1`；
它同时保存可审计的 Teacher 数据。将模式换成 `vision` 即使用视觉工具。
这是本轮完成实测的串行进程入口；不要用旧的 Vision 启动配置重复加载 8115 服务。
原配对实验仍用 `scripts/local/run.py --snapshot-manifest ... --pilot-phase prepare/oracle/models`，
与本采集管线分别保存产物。OpenAI 网络连通问题尚未解决，本采集器仅使用 DeepSeek。

## 第一批小规模采集（最多两次付费模型请求）

```bash
SCENES="$PWD/data/manifests/teacher_$STAMP"
RAW="$PWD/data/raw/teacher_$STAMP"
SFT="$PWD/data/processed/teacher_$STAMP"

# 无模型 API：生成、静置并验证 seed 0 的新 Snapshot
.venv/bin/python -B -m capx.training.teacher prepare \
  --output "$SCENES" --seeds 0

# 最多两次 DeepSeek 请求：Privileged 一次、Vision 一次；不自动重试
# 额外将 Privileged 成功代码在同一 Snapshot 的 Vision 环境重放，不增加模型请求
.venv/bin/python -B -m capx.training.teacher collect \
  --manifest "$SCENES/manifest/scenes.json" \
  --output "$RAW" --modes privileged vision \
  --max-requests 2 --max-cost-usd 1 --max-tokens 32768 \
  --worker-timeout 360 --distill-vision

# 只导出 Vision Student 能使用的示范
.venv/bin/python -B -m capx.training.teacher export \
  --runs "$RAW" --target vision --output "${SFT}_vision"

# 真值模式 Student 的数据单独导出
.venv/bin/python -B -m capx.training.teacher export \
  --runs "$RAW" --target privileged --output "${SFT}_privileged"
```

每个模式/场景先执行固定 Oracle，基线失败则跳过该场景的模型调用（记录仍保留）。
这是保守的数据质量门槛，不代表 Oracle 失败的场景绝对不可解，统计时应报告该筛选偏差。
每次 Teacher 调用只有一个完整代码块，最多 64 次 API 调用、90 秒执行，无恢复追问。
每个任务有独立进程及总超时；系统/API 错误终止后续采集。Ctrl+C 会终止当前 worker。
`--max-requests` 是所有 Seed × 模式的请求上限；总组合数超限会在调用前拒绝。
预算使用现有保守费率及预留逻辑，**不是供应商实时账单**；中断或无 usage 时费用可能未知。

输出目录必须不存在，拒绝覆盖；当前不支持在原目录续跑，失败后使用新的 STAMP。
要扩大到两个 Seed，使用 `--seeds 0 21`，并相应调整请求上限；本轮未执行批量采集。
已有 pilot Snapshot 可直接用其 `manifest/scenes.json`，并加 `--seeds 0` 选择子集；会校验并复制到新目录。

## 不付费的本地通路检查

把上述 collect 命令加上 `--mock`，并改为全新的输出目录：

```bash
.venv/bin/python -B -m capx.training.teacher collect \
  --manifest "$SCENES/manifest/scenes.json" \
  --output "$PWD/outputs/teacher/mock_$STAMP" \
  --modes privileged vision --max-requests 2 --distill-vision --mock
```

Mock 用固定 Oracle 代码替代大模型，但执行真实仿真和视觉服务。
所有 Mock 记录均标记为 `mock`，默认导出严格排除，不能冒充 DeepSeek 示范。

## 复用已有 DeepSeek 实验结果

```bash
INDEX="$PWD/data/raw/import_pilot_$STAMP"
.venv/bin/python -B -m capx.training.teacher import-pilot \
  --source "$PWD/outputs/model_comparison/pilot_20261112_200508" --output "$INDEX"
.venv/bin/python -B -m capx.training.teacher export \
  --runs "$INDEX" --target privileged --output "$PWD/data/processed/import_pilot_$STAMP"
```

导入只读引用原始实验产物，不改写或复制其视频；需保留原目录。
`export --runs RUN1 RUN2 ...` 可合并多批数据，并去除重复输入/代码样本。
历史 Privileged 数据不会直接进入 Vision 导出。

## 数据含义与格式

|来源|可训练的 Student|条件|
|---|---|---|
|Privileged Teacher|使用相同真值 API 的代码 Student|独立物理成功|
|Vision Teacher|使用同一视觉工具 API 的代码 Student|独立物理成功|
|Privileged → Vision 重放|Vision 工具型 Student|同 Snapshot、原程序不变、Vision 重放成功、源 Teacher 合格|

这里的 Vision 是 Structured Vision 工具接口，不是模型直接接收图片。
跨模式导出使用重放时构建的 Vision Prompt/API 文档/公开机器人状态，绝不复制 Teacher 真值。
它提供的是经过视觉工具验证的程序标签，不是完整 DAgger、在线 Student 纠错或图像 SFT。
仅通过重放仍可能存在场景过拟合，需要更多独立场景验证。

原始目录保存 `collection.json`、`records.jsonl`、Snapshot、每任务 `job.json/worker.log`，以及复用 pilot 的
`prompt.json`、`api-docs.json`、`actor-observation.json`、`model-response.json`、`generated.py`、
`http-attempt-*.json`、`result.json`、Trace、请求记录、抓取候选、视频及时间戳。
物理真值和模型 reasoning 可在原始审计产物中保留，**不进入 SFT messages**。

导出为 `train.jsonl`、`validation.jsonl`：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"任务、API 文档和公开观测"},{"role":"assistant","content":"完整 Python 程序"}]}
```

可供后续 Qwen tokenizer/chat template 使用；训练时只对 assistant 代码计算监督损失。
本轮不绑定训练框架、不添加特殊 Token、不启动 SFT/GRPO。
`index.jsonl` 单独保存来源、模式和 Snapshot hash；`rejections.jsonl` 保存剔除原因。
筛选要求：物理成功、reward=1、sandbox_rc=0、execution_status=ok、有动作、未截断；
生成必须 `finish_reason=stop`，代码须与模型原始输出一致，Prompt hash 须一致。
系统错误、空程序、pass、假 FINISH、Mock、Oracle、跨模式未验证的数据不能进入成功示范集。

默认按 Snapshot hash 确定性划分验证集（20%），相同场景始终在同一 split。
少量场景可能导致验证集为空，这是正常但不足以评估训练；不要把同一场景随机拆到训练与验证集。

## 本轮验证

产物：`outputs/reproduction/teacher-pipeline-check-20261112/`。
已真实执行 prepare、Mock collect（两模式与 Vision 重放）、import-pilot、export。
已将历史 5 条 DeepSeek Privileged 成功示范导出；Vision 导出正确拒绝这 5 条直接真值样本。
未进行任何新的真实大模型调用。真实 API 的限额/错误处理由本地 Mock 测试验证；首批付费采集仍需运行上方命令。

最终自动化回归：35 passed（数据管线 17 项 + 共享 pilot/预算 18 项）。
真实 Mock 仿真：Privileged、Vision、Privileged → Vision 重放共 3 条成功记录，审计到的 LLM HTTP 请求数为 0。
历史导出：Privileged 5 条训练样本、0 条验证样本；直接 Vision 导出拒绝全部 5 条，Mock 导出拒绝全部 3 条。
