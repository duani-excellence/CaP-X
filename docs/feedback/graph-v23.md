# Graph V23：通用目标与有界状态记忆

在现有 GroundedMonitor、GraphExecutive、父进程 broker、Sandbox 和 Graph V2 Runner 上扩展。当前是开发原型；**Restack 遮挡释放尚未验收通过，不应启动正式大批次或把本版称为可靠遮挡操作系统**。详见 [开发报告](../reports/graph-feedback-v23-development-2026-11-22.md)。

## 开关与结构

使用 `env_configs/graph_feedback/development-v23.json`。`budgets.task_conditions=true` 启用目标条件；`generic_goals=true` 给模型公开结构化目标与声明接口；`memory_profile` 启用共同的有界记忆。移除 `memory_profile` 即保留原直接观测路径。G0/G1 使用完全相同的误差模型、视觉服务、技能、阈值和预算。旧实验配置、历史源文件归档和 B0 路径保留。

`task_definition.py` 是基准任务绑定适配器。可向 Monitor 传 `task_definition`，或在运行配置 `budgets.task_definition` 提供结构化对象、目标和依赖。执行器不读取任务名称决定恢复。当前技能的物理适用范围是同一 Panda、公开 RGB-D 定位的直立刚性方块；颜色识别仍由原感知适配器承担。任意新形状、同色物体区分和新物理评分不在此版支持范围。

```json
{
  "name": "custom_name",
  "objects": {
    "payload": {"sensor_key": "red", "perception_name": "red cube"},
    "pedestal": {"sensor_key": "green", "perception_name": "green cube"}
  },
  "subgoals": [
    {"condition": "placed:pedestal:table", "object": "pedestal", "destination": "table", "requires": []},
    {"condition": "placed:payload:pedestal", "object": "payload", "destination": "pedestal", "requires": ["placed:pedestal:table"]}
  ],
  "goals": ["placed:payload:pedestal", "safe_retreat"]
}
```

该示例是现有物体的身份别名；改变真实任务目标必须有匹配的独立环境/评分适配，不能只改在线目标后仍宣称原任务已完成。本轮没有修改任何物理评分。

父进程在执行程序前建立 `task_goal → task_subgoal → skill_intent`。每个意图带对象、目标、顺序约束和技能契约。初始 Agent 仍输出完整 Python，可通过 `declare_intent(rows)` 提供自己的意图列表；父进程校验对象、技能、目标和前向依赖，声明不能授权运动。实际 plan、调用和返回凭据与意图关联，返回与物理效果验证分开。G1 回归未满足目标：证据未知先补证，明确不成立才修复，已完成动作不重放。

## 记忆的含义

`state_memory.py` 索引已有 NPZ 观测文件、几何、相机标定、物体—TCP 变换和公开机器人运动。大型 RGB-D 不进入图或提示词。当前直接估计、传播预测、观测矛盾和未知有不同状态。

只有现有完整面估计器能够初始化可预测锚点；部分颜色质心、旧位姿或一次抓爪闭合不够。附着预测要求已验证抓持、同一 grasp epoch、当前非空夹爪、有效标定与直立约束。静态支撑需要积极的非接触几何证据，可能接触会失效且不能靠随后抬升复活。当前深度必须支持视线被遮挡；深度缺失或视野外不算预期遮挡。

误差按移动路径、旋转、经过时间累积。反证、滑移、抓持丢失、可能接触、标定变化或未知动作效果撤销相关证据。所有释放区间必须完整包含于原要求：XY ≤ 8 mm，间隙在 [-4, 16] mm。最终派发前再次核验，旧 PASS 永远不是当前许可。直接观测路径与 `check_surface()` 保留；没有把 `allow_retained=False` 改成 True。

本版标定是开发历史样本的经验包络加余量，增长率属于明确的保守工程假设，不是概率保证。独立验证不足、任意未观测外力不在该模型覆盖范围。严禁把 `validated` 字段解释成已通过真实机器人安全认证。

## 实际复跑命令

以下重放不构造 LLM 客户端，使用历史真实 DeepSeek 程序、实时 SAM3/GraspNet/仿真和原评分。输出目录必须是新的。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/local/replay_graph_initial.py \
  --config env_configs/graph_feedback/development-v23.json \
  --source-unit outputs/model_comparison/graph_v21_online_20261122_154720/normal-stack \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/stack-953265258/prepared-snapshot.json \
  --bind-current-contract --pause-before-release \
  --output "outputs/reproduction/v23_pause_$(date +%Y%m%d_%H%M%S)"
```

`--pause-before-release` 仅注入一次软件边界中断，不是物理故障。`--bind-current-contract` 保存旧/新文档和提示词 hash，不声称旧代码由新提示词生成。重放原始 result 中的历史 initial latency/token 不是新成本；比较实际重放墙钟用 `total_seconds`，`diagnostic-summary.json` 明确新增请求和成本为零。

复现此前 Restack 阻塞，将 source-unit 换成 `outputs/model_comparison/graph_v22_20261122_180352/online-v22-2/normal-restack`，Snapshot 换成 `restack-1482414119/prepared-snapshot.json`，去掉暂停选项，使用新的输出目录。当前会在有限恢复后退出，不能把 NoModel 的 `budget_exceeded` 解释为供应商余额不足。

冻结传感器诊断（完全离线，不是实时 Vision）：

```bash
.venv/bin/python -B scripts/local/replay_state_memory.py \
  --episode outputs/model_comparison/graph_v22_20261122_180352/online-v22-2/normal-restack/episodes/G1 \
  --profile env_configs/graph_feedback/calibration/memory-development-v23.json \
  --output "outputs/reproduction/v23_frozen_$(date +%Y%m%d_%H%M%S)"
```

`calibrate_state_memory.py --episodes EPISODE_DIR ... --output NEW_DIR` 可重新生成只含误差界、来源 hash 和模型假设的标定产物。该脚本是离线评价工具，会读取归档真值；在线 Monitor 不导入它，也不读取其中的物体位姿。

待零新增 LLM 验收通过后，原真实配对入口仍可使用（会付费；本轮未执行）：

```bash
bash scripts/local/graph_experiment.sh pair \
  --config env_configs/graph_feedback/development-v23.json \
  --task restack --seed 1482414119 --groups G0 G1 \
  --snapshot outputs/model_comparison/graph_final_v20_20261122/development-snapshots/scenes/restack-1482414119/prepared-snapshot.json \
  --output "outputs/model_comparison/v23_online_$(date +%Y%m%d_%H%M%S)"
```

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/graph_v2/test_v23_goals_memory.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/graph_v2 tests/graph_feedback
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/feedback
```

真实渲染六项测试与 P0 视觉测试沿用原显式环境开关；必须使用新的测试产物目录。所有干预仍在 API 边界，未增加运动中取消接口。
