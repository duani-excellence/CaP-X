# Graph V24：通用契约与原生任务验收入口

正式工程仍为 `/home/hanyu/dqb/cap-x-20261106`。保留 V23.1 原路径、未提交修改和历史失败。

**目前是任务基准与契约执行的开发版本，不是已通过的非方块 Vision Agent。**
原生四物体 PickPlace 已在连续物理 episode 中完成；该证据来自 Privileged Oracle。
真实视觉已探测，但容器/装配特征和动作安全证据仍有缺口，因此不开放未经验收的物理技能，不启动新模型配对。

## 接入原则

- `GraphExecutive.next_contract` 只处理参数化目标、条件、技能意图、前置依赖和资源；不读取任务名、颜色、Seed 或故障标签。
- `ContractGraph` 复用 `DependencyGraph`。未来意图在动作前登记；当前许可绑定条件版本，已完成调用不会转回待执行；逻辑汇合条件随其依赖失效。
- `ContractBroker` 在最后一次补证之后申请动作许可。actor 只能指定声明的 intent ID；可信父进程提供技能实现和验证器。异常物理效果未知时隔离，缺少合格能力时拒绝执行。
- 普通物理后置条件不因旧定位来源更新就自动撤销；新的反证必须更新该条件。聚合目标则通过 `effect_dependencies` 明确订阅当前子目标。
- 局部修复从未满足目标回归，分配新 intent ID；相同条件状态及仅改变来源 ID 不算进展。有界修复不恢复 Python 指令指针。
- 环境定义负责对象角色与原生评分；技能实现负责可复用的物理操作。Oracle 程序独立保存，不能伪装成图算法或 DeepSeek 输出。

新环境共用 `native_tasks.py` 的 Robosuite 边界。新增任务不需要修改通用图核心。
但新增**物理能力**仍需独立实现/验收：区域内放置、插入、双臂交接不能借用方块堆叠的释放验证器。
当前 V23 的方块状态记忆误差包络也不能套用到包装盒、螺母或铰链物体。

## 命令

以下命令无 LLM 请求。配置文件提供服务地址和软件渲染设置；不要修改现有依赖或启动另一份模型服务。

```bash
cd /home/hanyu/dqb/cap-x-20261106
source scripts/local/common.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN="$PWD/outputs/model_comparison/v24_$(date +%Y%m%d_%H%M%S)"

# Gate 1：同一 Seed 两次全新环境重建、Snapshot 对比、原生评分及首帧。
.venv/bin/python -B scripts/local/native_task_gate.py \
  --config env_configs/graph_feedback/development-v24.json \
  --tasks pick_place nut_single nut_dual handover door peg_in_hole tool_hang \
  --gate environment --seed 0 --output "$RUN/environment"

# Gate 2：四物体连续搬运参考策略。原生完整成功要求四个目标都满足。
.venv/bin/python -B scripts/local/native_task_gate.py \
  --config env_configs/graph_feedback/development-v24.json --tasks pick_place \
  --gate oracle --seed 0 \
  --program env_configs/human_oracle_code/robosuite/native_bin_transfer.py \
  --output "$RUN/pickplace-oracle"

# 同一参数化插入参考程序，分别执行单螺母和双螺母。
.venv/bin/python -B scripts/local/native_task_gate.py \
  --config env_configs/graph_feedback/development-v24.json --tasks nut_single nut_dual \
  --gate oracle --seed 0 \
  --program env_configs/human_oracle_code/robosuite/native_insertion.py \
  --output "$RUN/insertion-oracle"

# Gate 3 诊断：真实 SAM3、RGB-D 和 GraspNet；不执行未验收动作。
.venv/bin/python -B scripts/local/native_task_gate.py \
  --config env_configs/graph_feedback/development-v24.json \
  --tasks pick_place nut_single nut_dual door --gate vision-probe \
  --output "$RUN/vision-tools"

# 实际视觉证据进入通用图。当前会在缺少几何/技能许可时有界退出。
# 这是 Graph 补证/阻断诊断，不能作为完整 Gate 4 物理任务成功。
.venv/bin/python -B scripts/local/native_task_gate.py \
  --config env_configs/graph_feedback/development-v24.json \
  --tasks pick_place nut_single nut_dual --gate vision-graph \
  --output "$RUN/vision-contracts"

# 通用图驱动的物理参考配对：明确是 Oracle，既不是 Vision 也不是 DeepSeek。
for GROUP in G0 G1; do
  .venv/bin/python -B scripts/local/native_task_gate.py \
    --config env_configs/graph_feedback/development-v24.json --tasks pick_place \
    --gate oracle-graph --group "$GROUP" --seed 0 \
    --program env_configs/human_oracle_code/robosuite/native_bin_transfer.py \
    --output "$RUN/reference-$GROUP"
done
```

`--seed 21` 是另一固定开发场景。配置中另列出的 heldout Seeds 未运行，不按成功重新抽取。
输出目录必须不存在；失败结果也保留。此验收 CLI 不支持直接续接中途物理状态，不能以重新 reset 冒充续接同一个 episode。
完整模型 B0/G0/G1 入口仍为原 Graph 实验框架；本版新增任务尚未通过其付费准入，不能仅把任务名加进旧命令启动模型批次。

## 产物及边界

每次运行保存配置、源码归档/变化核对、Snapshot、原生物理状态与子目标进展。
操作轨迹保存 RPC 参数/返回/错误、动作前后私有物理状态、IK 收敛证据及时间对齐视频。
视觉探测保存 RGB-D、相机标定、分割掩码、局部点云、候选分数与实际 HTTP 请求。
图运行保存未来意图、条件版本、资源许可、完成调用、新修复 ID 和决策。

`oracle_*` API 仅存在于显式 Oracle actor。Vision actor 只有当前已开放的感知/机器人观测接口，不能通过 raw `goto_pose` 或 `open_gripper` 绕过资格检查。
`FINISH`、验证器返回和脚本正常结束均不等于原生物理成功；原有 Sandbox、物理评分与原任务代码不变。

机制测试：

```bash
.venv/bin/python -B -m pytest -q \
  tests/graph_v2/test_v24_contracts.py \
  tests/graph_v2/test_v24_boundary.py \
  tests/graph_v2/test_v24_perception.py -p no:cacheprovider
```

任务状态、失败根因和实际比较见 [V24 开发报告](../reports/graph-feedback-v24-development-2026-11-23.md)。
