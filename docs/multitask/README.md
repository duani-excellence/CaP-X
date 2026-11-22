# 多任务 Privileged 推理与 Teacher 采集

正式入口：`/home/hanyu/dqb/cap-x-20261106`。使用现有 `.venv` 和本地 API 配置；无需安装依赖或下载模型。默认 Teacher 是 DeepSeek V4 Pro，输出上限 32768 tokens。需要现有 PyRoKi 服务正常运行。

## 已实测范围

| 任务 ID | 具身任务 | 状态及物理验收 |
|---|---|---|
| stack | 方块堆叠 | 已实测可采集，保留原 P0 路径；本轮 Seed 21 Oracle、Seed 0 Mock 采集回归 |
| lift | 方块抬升 | 已实测可采集，Seed 0、21；抬升、实际抓持并稳定 |
| restack | 重新堆叠 | 已实测可采集，Seed 0、21；反向堆叠、释放并稳定 |
| nut_assembly | 螺母装配 | 已实测可采集，Seed 0、21；正确插柱、释放并稳定 |
| handover | 双臂交接 | 已实测可采集，Seed 0、21；接收手抓柄、交出手释放、高度及稳定 |
| wipe | 污渍擦拭 | 已接入但未验收；Seed 0 Oracle 未清除全部污渍，禁止付费采集 |
| two_arm_lift | 双臂抬举 | 已接入但未验收；Seed 21 成功，Seed 0 未抓稳双把手，禁止付费采集 |

“已实测可采集”是有限场景下 Oracle、Snapshot、隔离与物理评分闭环通过，不是任意场景成功率保证，也不是付费 Teacher 质量已验证。每个采集场景仍先运行本任务 Oracle；失败场景不会请求 Teacher。

LIBERO 已检查打开微波炉、抓取汤罐适配，当前依赖缺少 `robosuite.environments.manipulation.single_arm_env`，需隔离旧 Robosuite 依赖栈后另行验收；不能直接把 `done` 当物理成功。R1Pro/BEHAVIOR 缺少 OmniGibson/BDDL，本轮不新搭重型平台。它们不计入已通过任务数。

## 当前可执行命令

所有命令在正式项目目录运行。输出路径必须是新目录；续采显式加 `--resume`，其他参数与原作业一致。以下时间戳目录避免覆盖。

```bash
cd /home/hanyu/dqb/cap-x-20261106
bash scripts/local/multitask_teacher.sh list
# 免费 Oracle：实际仿真，不调用大模型。也可指定未验收的任务用于诊断。
bash scripts/local/multitask_teacher.sh oracle --tasks stack lift restack nut_assembly handover --seed 0 --output "outputs/reproduction/multitask_oracle_$(date +%Y%m%d_%H%M%S)"
# 单任务/多任务模型推理：每任务最多一次付费请求；先通过该场景 Oracle。
bash scripts/local/multitask_teacher.sh infer --tasks lift --output "outputs/model_comparison/lift_$(date +%Y%m%d_%H%M%S)"
bash scripts/local/multitask_teacher.sh infer --tasks lift nut_assembly --output "outputs/model_comparison/multitask_$(date +%Y%m%d_%H%M%S)"
```

第一批小规模多任务 Teacher 采集，一条命令：

```bash
bash scripts/local/multitask_teacher.sh collect --targets stack=2 lift=2 restack=2 nut_assembly=2 handover=2 --max-requests-per-task 10 --max-cost-usd-per-task 2 --max-scenes 30 --max-hours 1 --output "outputs/teacher/multitask_$(date +%Y%m%d_%H%M%S)"
```

这是每任务目标 2 条经过成功、完整响应、去重及模板配额过滤的合格样本，不保证预算内达到目标。上例整体最多 50 个请求槽、10 美元记账预算，串行运行，每任务最多 1 小时（单次正在执行的 worker 另有超时）。请求槽包含保守预留，因此可能先于真实 HTTP 次数耗尽。费用使用现有项目估算，不等于供应商结算账单。`--mock` 只用固定 Oracle 程序测试全流程，不调用模型，也绝不导出为 Teacher 样本。

续采同一作业（把路径换成上次真实目录；不要再用时间戳）：

```bash
bash scripts/local/multitask_teacher.sh collect --targets stack=2 lift=2 restack=2 nut_assembly=2 handover=2 --max-requests-per-task 10 --max-cost-usd-per-task 2 --max-scenes 30 --max-hours 1 --output outputs/teacher/你的作业目录 --resume
```

预算累计，不在续采时归零。代码/关键配置变化会拒绝续采；不确定是否已付费的中断请求不会静默重放。各任务独立检查点、统计、日志、场景和请求收据；一个任务失败不阻断其他任务。

## SFT 导出与观测边界

采集结束自动导出，也可重新生成一个新的导出目录：

```bash
bash scripts/local/multitask_teacher.sh export --run outputs/teacher/你的作业目录 --output "data/processed/multitask_$(date +%Y%m%d_%H%M%S)"
```

- 每任务 `state.json` 区分 physical_success、sft_eligible、accepted；不能把 Oracle 或 Mock 成功算作 Teacher 数据。
- 原始输入、代码、模型响应、API 前后物理 trace、视频/时间戳、结果和 snapshot 留在各任务产物目录。
- `train.jsonl` / `validation.jsonl` 为 messages 格式；`index.jsonl` 保留 task、source、mode、snapshot_hash、split、sample_digest 和 artifacts。
- 合并数据采用等任务配额和交错顺序，避免 Stack 占据大部分数据；任一选择任务没有合格数据时，平衡合并导出为空，各任务自己的导出仍保留全部样本。
- 按任务+Snapshot 分组划分场景，并复用现有去重/模板数量限制；不同 Seed 或指令改写不算新任务。小验证集不构成泛化基准。
- 本轮新增任务只有 **Privileged**。此类程序可能依赖真实物体状态，仅适用于同 API 的 Privileged Student；不能直接宣称可训练部署 Vision Student。
- 原有 Stack 的 Vision 及 Privileged→Vision 隔离重放蒸馏入口不变，见 `docs/teacher-data/`。未把它们未经验证地套到新增任务。

## Web

打开 `http://127.0.0.1:8200/`，展开“多任务推理与 Teacher 采集 · Privileged”：勾选任务、设置各任务合格目标、每任务请求和美元预算，可运行免费 Oracle、单次推理、采集、停止及续采。未验收/阻塞任务禁用。

顶部原有模型/观测选择仍服务于 Stack 交互推理；多任务面板明确使用 DeepSeek + Privileged，不读取顶部选项。多任务作业使用同一 CLI 独立进程，页面展示统计和产物路径，暂不接入原有 Viser 实时画面。服务器重启后的历史作业显示“状态请核查”，需核对进程再续采。
